"""Structured solver result containers."""

# pylint: disable=duplicate-code,too-many-instance-attributes,too-many-lines

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_hex
import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure

from morana._arrays import (
    readonly_float_array,
    require_finite_nonnegative_array,
)
from morana._plotting import (
    EXCLUDED_FLUX_COLOR,
    FLUX_COLORMAP,
    add_planar_matplotlib_cells,
    add_planar_plotly_cells,
    add_planar_plotly_hover_targets,
    configure_hex_axes,
    configure_hex_figure,
)
from morana._validation import (
    require_integer,
    require_nonempty_string,
    require_nonnegative_integer,
)
from morana.configuration import ProblemConfigurationSnapshot
from morana.material_mesh import MaterialMesh
from morana.materials import CrossSections
from morana.hex_planar_mesh import OpenMCIndex
from morana.normalization import (
    FissionSourceNormalization,
    PowerNormalization,
)
from morana.operators import (
    _check_normalization_consistency,
    extract_cross_section_data,
)
from morana.execution_reports import (
    KeffSolveReport,
    LinearSolveReport,
)
from morana.solve_settings import FixedSourceSettings, KeffSettings

_PLOT_VALUE_EQUALITY_RELATIVE_TOLERANCE = 1.0e-12
_PLOT_VALUE_EQUALITY_ABSOLUTE_TOLERANCE = 1.0e-14
_PLOT_ZERO_VALUE_RELATIVE_TOLERANCE = 0.0
_PLOT_ZERO_VALUE_ABSOLUTE_TOLERANCE = 1.0e-14
_BALANCE_CONSISTENCY_RELATIVE_TOLERANCE = 1.0e-12
_BALANCE_CONSISTENCY_ABSOLUTE_TOLERANCE = 0.0
_FIXED_SOURCE_BALANCE_TERMS = frozenset(
    {
        "source",
        "boundary_source",
        "scattering_coupling",
        "fission_production",
        "fission_emission",
        "removal",
        "absorption",
        "radial_leakage",
        "axial_leakage",
        "net_scattering",
        "residual",
    }
)
_KEFF_BALANCE_TERMS = frozenset(
    {
        "fission_production",
        "keff_source",
        "scattering_coupling",
        "removal",
        "absorption",
        "radial_leakage",
        "axial_leakage",
        "net_scattering",
        "residual",
    }
)
_LOSS_FRACTION_TERMS = ("absorption", "radial_leakage", "axial_leakage")
_SOURCE_NORMALIZED_TERMS = (
    "absorption",
    "radial_leakage",
    "axial_leakage",
    "net_scattering",
)


@dataclass(frozen=True, init=False)
class FixedSourceBalance:
    """Store one immutable fixed-source neutron-balance record.

    Instances are retained by solver-produced results or restored from checked
    result archives. Direct construction is not supported.

    Attributes
    ----------
    by_group
        Finite group-resolved vectors in ``n / s`` for every fixed-source
        balance term: ``source``, ``boundary_source``,
        ``scattering_coupling``, ``fission_production``,
        ``fission_emission``, ``removal``, ``absorption``,
        ``radial_leakage``, ``axial_leakage``, ``net_scattering``, and
        ``residual``.
    by_layer_group
        Bottom-to-top finite group-resolved vectors in ``n / s`` for the same
        terms. Each layer tuple contracts to its corresponding ``by_group``
        vector.

    """

    by_group: Mapping[str, np.ndarray]
    by_layer_group: Mapping[str, tuple[np.ndarray, ...]]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject direct construction; balances belong to completed results."""
        _ = args, kwargs
        raise TypeError(
            "FixedSourceBalance instances are retained by completed results or "
            "Result.load_from_disk()"
        )

    @classmethod
    def _from_validated(
        cls,
        *,
        by_group: Mapping[str, np.ndarray],
        by_layer_group: Mapping[str, tuple[np.ndarray, ...]],
    ) -> "FixedSourceBalance":
        """Build one checked balance from internal solver or archive data."""
        by_group = _owned_balance_vectors(by_group, "by_group")
        by_layer_group = _owned_layer_balance_vectors(by_layer_group, by_group)
        _require_balance_terms(by_group, _FIXED_SOURCE_BALANCE_TERMS, "fixed-source")
        instance = object.__new__(cls)
        object.__setattr__(instance, "by_group", MappingProxyType(by_group))
        object.__setattr__(instance, "by_layer_group", MappingProxyType(by_layer_group))
        return instance

    @property
    def scalar(self) -> Mapping[str, float]:
        """Return immutable scalar totals contracted from group vectors."""
        return _scalar_balance(self.by_group)

    def __getitem__(self, name: str) -> float:
        """Return one scalar balance term by its documented name."""
        return self.scalar[name]

    def items(self):
        """Return immutable scalar balance term pairs."""
        return self.scalar.items()

    def keys(self):
        """Return immutable scalar balance term names."""
        return self.scalar.keys()

    @property
    def loss_fractions(self) -> Mapping[str, float] | None:
        """Return loss fractions, or ``None`` when total loss is zero."""
        return _loss_fractions_or_none(self.scalar)

    @property
    def source_normalized(self) -> Mapping[str, float] | None:
        """Return source-normalized terms, or ``None`` when source is zero."""
        scalar = self.scalar
        driving_source = sum(
            scalar[name] for name in ("source", "boundary_source", "fission_emission")
        )
        if driving_source == 0.0:
            return None
        return _source_normalized(scalar, driving_source)


@dataclass(frozen=True, init=False)
class KeffBalance:
    """Store one immutable criticality neutron-balance record.

    Instances are retained by solver-produced criticality results or restored
    from checked result archives. Direct construction is not supported.

    Attributes
    ----------
    by_group
        Finite group-resolved vectors in ``n / s`` for every criticality
        balance term: ``fission_production``, ``keff_source``,
        ``scattering_coupling``, ``removal``, ``absorption``,
        ``radial_leakage``, ``axial_leakage``, ``net_scattering``, and
        ``residual``.
    by_layer_group
        Bottom-to-top finite group-resolved vectors in ``n / s`` for the same
        terms. Each layer tuple contracts to its corresponding ``by_group``
        vector.

    """

    by_group: Mapping[str, np.ndarray]
    by_layer_group: Mapping[str, tuple[np.ndarray, ...]]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject direct construction; balances belong to completed results."""
        _ = args, kwargs
        raise TypeError(
            "KeffBalance instances are retained by completed results or "
            "Result.load_from_disk()"
        )

    @classmethod
    def _from_validated(
        cls,
        *,
        by_group: Mapping[str, np.ndarray],
        by_layer_group: Mapping[str, tuple[np.ndarray, ...]],
    ) -> "KeffBalance":
        """Build one checked balance from internal solver or archive data."""
        by_group = _owned_balance_vectors(by_group, "by_group")
        by_layer_group = _owned_layer_balance_vectors(by_layer_group, by_group)
        _require_balance_terms(by_group, _KEFF_BALANCE_TERMS, "criticality")
        scalar = _scalar_balance(by_group)
        if _total_loss(scalar) <= 0.0 or scalar["keff_source"] <= 0.0:
            raise ValueError(
                "criticality balance requires positive loss and fission-source terms"
            )
        instance = object.__new__(cls)
        object.__setattr__(instance, "by_group", MappingProxyType(by_group))
        object.__setattr__(instance, "by_layer_group", MappingProxyType(by_layer_group))
        return instance

    @property
    def scalar(self) -> Mapping[str, float]:
        """Return immutable scalar totals contracted from group vectors."""
        return _scalar_balance(self.by_group)

    def __getitem__(self, name: str) -> float:
        """Return one scalar balance term by its documented name."""
        return self.scalar[name]

    def items(self):
        """Return immutable scalar balance term pairs."""
        return self.scalar.items()

    def keys(self):
        """Return immutable scalar balance term names."""
        return self.scalar.keys()

    @property
    def loss_fractions(self) -> Mapping[str, float]:
        """Return immutable fractions of the positive total loss."""
        scalar = self.scalar
        return _source_normalized(scalar, _total_loss(scalar), _LOSS_FRACTION_TERMS)

    @property
    def source_normalized(self) -> Mapping[str, float]:
        """Return immutable terms normalized by the positive fission source."""
        scalar = self.scalar
        return _source_normalized(scalar, scalar["keff_source"])


@dataclass(frozen=True, init=False)
class CellInspection:
    """Store one read-only active-cell inspection value.

    Instances are returned by ``Result.cell_at``. Direct construction is not
    supported.

    Attributes
    ----------
    material_key
        Material-layout key assigned to the selected cell.
    cross_sections
        Immutable macroscopic cross sections assigned to ``material_key``.
    flux
        Read-only one-dimensional cell-average scalar-flux array in
        ``n / cm^2 / s``, ordered fast to thermal.
    """

    material_key: str
    cross_sections: CrossSections
    flux: np.ndarray

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject direct construction; inspect a completed result instead."""
        _ = args, kwargs
        raise TypeError("CellInspection instances are returned by Result.cell_at()")

    @classmethod
    def _from_trusted(
        cls,
        *,
        material_key: str,
        cross_sections: CrossSections,
        flux: np.ndarray,
    ) -> "CellInspection":
        """Build one inspection value from completed result-owned data."""
        instance = object.__new__(cls)
        object.__setattr__(instance, "material_key", material_key)
        object.__setattr__(instance, "cross_sections", cross_sections)
        object.__setattr__(instance, "flux", flux)
        return instance


@dataclass(frozen=True, init=False)
class Result:
    """Store solution arrays and immutable run provenance.

    Instances are returned by the supported solve functions or restored with
    ``load_from_disk``. Direct construction is not supported.

    Attributes
    ----------
    flux
        Bottom-to-top read-only cell-average scalar-flux arrays in
        ``n / cm^2 / s``. Each layer has shape
        ``(groups, active_cells_in_layer)``.
    balance
        Immutable mode-specific balance record containing group, layer-group,
        scalar, and derived-ratio diagnostics.
    configuration_snapshot
        Immutable complete, non-aliasing configuration snapshot associated
        with the completed solve. Its public configuration values are freshly
        reconstructed on access. Call ``to_configuration()`` to create an
        independent mutable problem definition. A snapshot is required for
        plotting and VTM export.
    solve_settings
        Immutable numerical settings captured for the completed solve.
    normalization
        Immutable fission-source-rate or recoverable-power normalization for a
        criticality result, otherwise ``None``.
    execution_report
        Immutable typed solver diagnostics and convergence information.
    keff
        Final multiplication-factor estimate for a criticality report,
        otherwise ``None``.
    groups
        Shared number of energy groups, ordered fast to thermal.
    n_axial_layers
        Number of bottom-to-top stored flux layers.

    Notes
    -----
    ``Result`` is immutable. Flux arrays and balance diagnostics are owned
    immutable values, preserving a complete internally consistent record for
    plotting, export, and provenance.

    The active-cell axis is compact and layer-local. Use ``flux_layer()`` to
    select one layer, and use the configuration snapshot or inspection methods
    to map values back to full planar positions; excluded positions have no
    stored flux value.
    """

    flux: tuple[np.ndarray, ...]
    execution_report: LinearSolveReport | KeffSolveReport
    balance: FixedSourceBalance | KeffBalance
    configuration_snapshot: ProblemConfigurationSnapshot
    solve_settings: FixedSourceSettings | KeffSettings
    normalization: FissionSourceNormalization | PowerNormalization | None = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject direct construction; use a solve function or archive loader."""
        _ = args, kwargs
        raise TypeError(
            "Result instances are returned by solve functions or "
            "Result.load_from_disk()"
        )

    @classmethod
    def _from_validated(
        cls,
        *,
        flux: tuple[np.ndarray, ...],
        execution_report: LinearSolveReport | KeffSolveReport,
        balance: FixedSourceBalance | KeffBalance,
        configuration_snapshot: ProblemConfigurationSnapshot,
        solve_settings: FixedSourceSettings | KeffSettings,
        normalization: FissionSourceNormalization | PowerNormalization | None = None,
    ) -> "Result":
        """Build a checked result from internal solver or archive data."""
        _check_result_provenance(
            solve_settings, normalization, execution_report, balance
        )
        flux_layers = _owned_flux_layers(flux)
        snapshot = configuration_snapshot
        material_mesh = snapshot.material_mesh
        if material_mesh.n_axial_layers != len(flux_layers):
            raise ValueError(
                "result flux layers must match configuration snapshot layers"
            )
        for axial_index, flux_layer in enumerate(flux_layers):
            active_cells = material_mesh.n_active_cells(axial_index)
            if flux_layer.shape[1] != active_cells:
                raise ValueError(
                    "result flux active-cell count must match configuration "
                    "snapshot layer "
                    f"{axial_index}"
                )
        cross_sections = extract_cross_section_data(snapshot)
        if flux_layers[0].shape[0] != cross_sections.groups:
            raise ValueError(
                "result flux groups must match configuration cross sections"
            )
        result = object.__new__(cls)
        object.__setattr__(result, "solve_settings", solve_settings)
        object.__setattr__(result, "normalization", normalization)
        object.__setattr__(result, "execution_report", execution_report)
        object.__setattr__(result, "balance", balance)
        object.__setattr__(result, "flux", tuple(flux_layers))
        object.__setattr__(result, "configuration_snapshot", snapshot)
        if normalization is not None:
            _check_normalization_consistency(
                snapshot,
                cross_sections,
                flux_layers,
                normalization,
            )
        groups = flux_layers[0].shape[0]
        if any(
            len(layers) != len(flux_layers)
            or any(values.shape != (groups,) for values in layers)
            for layers in balance.by_layer_group.values()
        ):
            raise ValueError(
                "result balance layers must match result flux layers and groups"
            )
        keff = (
            execution_report.final_outer_iteration.keff
            if isinstance(execution_report, KeffSolveReport)
            else None
        )
        _check_balance_equation(balance, keff)
        return result

    @property
    def keff(self) -> float | None:
        """Return the final multiplication-factor estimate, when applicable."""
        if isinstance(self.execution_report, KeffSolveReport):
            return self.execution_report.final_outer_iteration.keff
        return None

    @property
    def solve_mode(self) -> str:
        """Return the solve-mode identifier derived from typed provenance."""
        if isinstance(self.execution_report, LinearSolveReport):
            return "fixed_source"
        return "keff"

    @property
    def groups(self) -> int:
        """Return the shared fast-to-thermal energy-group count."""
        return int(self.flux[0].shape[0])

    @property
    def n_axial_layers(self) -> int:
        """Return the number of stored axial flux layers."""
        return len(self.flux)

    def flux_layer(self, axial_index: int) -> np.ndarray:
        """Return the owned read-only group-major flux for one axial layer.

        Parameters
        ----------
        axial_index
            Nonnegative bottom-to-top axial-layer index.

        Returns
        -------
        numpy.ndarray
            Array shaped ``(groups, active_cells_in_layer)``.

        Raises
        ------
        TypeError
            If ``axial_index`` is not an integer or is a boolean.
        ValueError
            If ``axial_index`` is negative or outside the stored layers.
        """
        axial_index = require_nonnegative_integer("axial_index", axial_index)
        if axial_index >= self.n_axial_layers:
            raise ValueError("result axial_index is outside the stored flux layers")
        return self.flux[axial_index]

    def cell_at(self, axial_index: int, openmc_index: OpenMCIndex) -> CellInspection:
        """Return the material data and all-group flux at one active cell.

        Parameters
        ----------
        axial_index
            Nonnegative bottom-to-top axial-layer index.
        openmc_index
            Planar OpenMC-style position in the complete material mesh.

        Returns
        -------
        CellInspection
            The selected material key, its macroscopic cross sections, and a
            read-only one-dimensional flux array in fast-to-thermal order.

        Raises
        ------
        TypeError
            If ``axial_index`` is not an integer or ``openmc_index`` is not
            an ``OpenMCIndex``.
        IndexError
            If ``axial_index`` is outside the material-slice stack.
        KeyError
            If ``openmc_index`` is outside the complete planar mesh.
        ValueError
            If the selected position is excluded, or its material has no
            cross sections.
        """
        material_mesh = self.configuration_snapshot.material_mesh
        material_key = material_mesh.key_at(axial_index, openmc_index)
        active_id = material_mesh.active_id_at(axial_index, openmc_index)
        if active_id is None:
            raise ValueError("selected cell is excluded and has no result data")
        cross_sections = self.configuration_snapshot.materials[material_key].xs
        if cross_sections is None:
            raise ValueError("selected cell material has no cross sections")
        return CellInspection._from_trusted(  # pylint: disable=protected-access
            material_key=material_key,
            cross_sections=cross_sections,
            flux=self.flux_layer(axial_index)[:, active_id],
        )

    def plot_matplotlib(
        self,
        group: int,
        axial_index: int,
        ax: Axes | None = None,
    ) -> Axes:
        """Plot one energy group on one axial slice using Matplotlib.

        Parameters
        ----------
        group
            Zero-based fast-to-thermal group index.
        axial_index
            Nonnegative bottom-to-top layer index.
        ax
            Optional axes to populate. A new figure and axes are created when
            omitted.

        Returns
        -------
        matplotlib.axes.Axes
            Populated axes with a scalar-flux colorbar. Excluded positions are
            rendered in the fixed excluded-flux color.

        Raises
        ------
        TypeError
            If ``group`` is not an integer or is a boolean, or if ``ax`` is
            neither an ``Axes`` nor ``None``.
        ValueError
            If the selected group or layer is invalid or the layer has no
            active flux values.
        """
        material_mesh = self.configuration_snapshot.material_mesh
        values = self._full_flux_layer(group, axial_index, material_mesh)
        if ax is None:
            _, ax = plt.subplots()
        elif not isinstance(ax, Axes):
            raise TypeError("ax must be an Axes or None")
        finite = values[np.isfinite(values)]
        norm = _flux_normalize(finite)
        colormap = plt.get_cmap(FLUX_COLORMAP)
        add_planar_matplotlib_cells(
            material_mesh.mesh,
            ax,
            facecolor_for=lambda planar_id: (
                colormap(norm(values[planar_id]))
                if np.isfinite(values[planar_id])
                else EXCLUDED_FLUX_COLOR
            ),
        )
        configure_hex_axes(ax)
        ax.set_title(f"Flux group {group}, axial slice {axial_index}")
        colorbar = ax.figure.colorbar(
            ScalarMappable(norm=norm, cmap=colormap),
            ax=ax,
        )
        colorbar.set_label("scalar flux [n cm⁻² s⁻¹]")
        return ax

    def plot_plotly(self, group: int, axial_index: int) -> Figure:
        """Return a Plotly plot of one energy group on one axial slice.

        Polygon hover text includes full-lattice material-position details
        and the selected scalar flux. Excluded positions use the fixed
        excluded-flux color and report ``nan`` flux.

        Raises
        ------
        TypeError
            If ``group`` is not an integer or is a boolean.
        ValueError
            If the selected group or layer is invalid or the layer has no
            active flux values.
        """
        material_mesh = self.configuration_snapshot.material_mesh
        values = self._full_flux_layer(group, axial_index, material_mesh)
        finite = values[np.isfinite(values)]
        norm = _flux_normalize(finite)
        colormap = plt.get_cmap(FLUX_COLORMAP)
        figure = go.Figure()
        layer = material_mesh.layers[axial_index]
        add_planar_plotly_cells(
            material_mesh.mesh,
            figure,
            fillcolor_for=lambda planar_id: (
                to_hex(colormap(norm(values[planar_id])))
                if np.isfinite(values[planar_id])
                else EXCLUDED_FLUX_COLOR
            ),
        )
        add_planar_plotly_hover_targets(
            material_mesh.mesh,
            figure,
            hover_text_for=lambda planar_id: self._flux_hover_label(
                material_mesh,
                layer,
                planar_id,
                axial_index=axial_index,
                group=group,
                value=values[planar_id],
            ),
        )
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "color": [norm.vmin],
                    "cmin": norm.vmin,
                    "cmax": norm.vmax,
                    "colorscale": FLUX_COLORMAP,
                    "showscale": True,
                    "colorbar": {"title": "scalar flux [n cm⁻² s⁻¹]"},
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )
        configure_hex_figure(
            figure,
            material_mesh.mesh,
            f"Flux group {group}, axial slice {axial_index}",
        )
        return figure

    def export_vtm(self, path: str | Path) -> None:
        """Export reconstructed material layout and flux data as VTM blocks.

        The VTM references active and excluded VTU leaves. It includes one
        full-lattice ``flux_gN`` cell array per group in fast-to-thermal order;
        excluded-cell flux entries are ``NaN``. Parent directories are created
        when needed.

        The VTM leaf directory contains ``material_keys.json``, which maps each
        exported ``material_key_id`` to its material key.

        Raises
        ------
        ValueError
            If ``path`` does not have a ``.vtm`` suffix.
        IsADirectoryError
            If ``path`` identifies an existing directory.
        OSError
            If an output directory cannot be created or an output file cannot
            be written.
        """
        path = Path(path)
        material_mesh = self.configuration_snapshot.material_mesh
        full_flux = {}
        for group in range(self.groups):
            full_flux[f"flux_g{group}"] = tuple(
                tuple(self._full_flux_layer(group, axial_index, material_mesh))
                for axial_index in range(self.n_axial_layers)
            )
        material_mesh._export_vtm_with_cell_data(  # pylint: disable=protected-access
            path,
            extra_cell_data=full_flux,
        )

    def save_to_disk(self, path: str | Path) -> None:
        """Save this result as a versioned non-pickle ``.morana-result`` archive.

        The archive is a standard DEFLATE-compressed ZIP file containing an
        explicit JSON manifest and named ``.npy`` payloads. It records the
        result, detailed balances, solve settings, normalization, and complete
        configuration provenance. Array payloads are written without pickle and
        are checksummed. Parent directories are created and the completed
        archive replaces ``path`` atomically.

        Parameters
        ----------
        path
            Destination archive path. The suffix is conventionally
            ``.morana-result`` but is not required.

        Raises
        ------
        ValueError
            If the result cannot be represented by the archive schema.
        OSError
            If the destination cannot be created or replaced.
        """
        # pylint: disable-next=import-outside-toplevel,cyclic-import
        from morana._result_archive import save_result

        save_result(self, path)

    @classmethod
    def load_from_disk(cls, path: str | Path) -> "Result":
        """Load a checked result from a versioned non-pickle archive.

        The loader accepts only supported archive schema versions, validates
        the ZIP member inventory and payload SHA-256 hashes, loads all NumPy
        payloads with ``allow_pickle=False``, and reconstructs a checked
        completed result from its internal archive representation.

        Parameters
        ----------
        path
            Source ``.morana-result`` archive path.
            The suffix is conventional and is neither required nor added.

        Returns
        -------
        Result
            New immutable result with independently owned arrays and
            provenance.

        Raises
        ------
        ValueError
            If the archive is malformed, unsupported, inconsistent, or fails
            checked completed-result reconstruction.
        OSError
            If the archive cannot be read.
        """
        # pylint: disable-next=import-outside-toplevel,cyclic-import
        from morana._result_archive import load_result

        return load_result(path)

    def _full_flux_layer(
        self,
        group: int,
        axial_index: int,
        material_mesh: MaterialMesh,
    ) -> np.ndarray:
        """Map compact active flux onto full planar positions with NaN exclusions."""
        group = require_integer("group", group)
        compact = self.flux_layer(axial_index)
        if group < 0 or group >= compact.shape[0]:
            raise ValueError(f"group index {group} is out of range")
        values = np.full(material_mesh.mesh.n_cells, np.nan)
        for active_id, openmc_index in enumerate(
            material_mesh.active_indices(axial_index)
        ):
            planar_id = material_mesh.mesh.planar_id_at(openmc_index)
            if planar_id is None:
                raise ValueError("snapshot active position is missing from mesh")
            values[planar_id] = compact[group, active_id]
        return values

    @staticmethod
    def _flux_hover_label(
        material_mesh: MaterialMesh,
        layer: Mapping[OpenMCIndex, str],
        planar_id: int,
        *,
        axial_index: int,
        group: int,
        value: float,
    ) -> str:
        """Return one full-lattice flux hover label."""
        openmc_index = material_mesh.mesh.openmc_indices[planar_id]
        material_label = (
            material_mesh._material_hover_label(  # pylint: disable=protected-access
                planar_id,
                openmc_index,
                layer[openmc_index],
                axial_index,
                line_break="<br>",
            )
        )
        flux_label = f"{value:.8g}" if np.isfinite(value) else "nan"
        return f"{material_label}<br>flux group {group}: {flux_label} n cm⁻² s⁻¹"


def _flux_normalize(finite_values: np.ndarray) -> Normalize:
    """Return a meaningful color normalization for finite flux values."""
    if finite_values.size == 0:
        raise ValueError("selected result layer has no active flux values")
    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))
    if np.isclose(
        minimum,
        maximum,
        rtol=_PLOT_VALUE_EQUALITY_RELATIVE_TOLERANCE,
        atol=_PLOT_VALUE_EQUALITY_ABSOLUTE_TOLERANCE,
    ):
        mean = float(np.mean(finite_values))
        if np.isclose(
            mean,
            0.0,
            rtol=_PLOT_ZERO_VALUE_RELATIVE_TOLERANCE,
            atol=_PLOT_ZERO_VALUE_ABSOLUTE_TOLERANCE,
        ):
            minimum = 0.0
            maximum = 1.0
        else:
            delta = 0.1 * abs(mean)
            minimum = mean - delta
            maximum = mean + delta
    return Normalize(vmin=minimum, vmax=maximum)


# pylint: disable=too-many-branches
def _check_result_provenance(
    solve_settings: FixedSourceSettings | KeffSettings,
    normalization: FissionSourceNormalization | PowerNormalization | None,
    execution_report: LinearSolveReport | KeffSolveReport,
    balance: FixedSourceBalance | KeffBalance,
) -> None:
    """Require internal solver or archive provenance to be mode-consistent."""
    if isinstance(solve_settings, FixedSourceSettings):
        if normalization is not None:
            raise ValueError("fixed-source result does not accept normalization")
        if not isinstance(execution_report, LinearSolveReport):
            raise ValueError(
                "fixed-source result execution_report must be LinearSolveReport"
            )
        if execution_report.linear_solve != solve_settings.linear_solve:
            raise ValueError("fixed-source execution report must match solve_settings")
        if not isinstance(balance, FixedSourceBalance):
            raise ValueError("fixed-source result balance must be FixedSourceBalance")
        return
    if normalization is None:
        raise ValueError("k-effective result normalization is required")
    if not isinstance(execution_report, KeffSolveReport):
        raise ValueError("k-effective result execution_report must be KeffSolveReport")
    if not isinstance(balance, KeffBalance):
        raise ValueError("k-effective result balance must be KeffBalance")
    if execution_report.eigenvalue_iteration != solve_settings.eigenvalue_iteration:
        raise ValueError("k-effective execution report must match solve_settings")
    if any(
        report.linear_solve.linear_solve != solve_settings.inner_linear_solve
        for report in execution_report.outer_iterations
    ):
        raise ValueError("k-effective execution reports must match solve_settings")
    final = execution_report.final_outer_iteration
    if (
        execution_report.iterations > solve_settings.max_outer_iterations
        or final.keff_change > solve_settings.keff_change_tolerance
        or final.flux_change > solve_settings.flux_change_tolerance
        or (
            final.keff_relative_residual
            > solve_settings.keff_relative_residual_tolerance
        )
    ):
        raise ValueError(
            "k-effective execution report must satisfy convergence controls"
        )


def _owned_flux_layers(layers: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    """Return checked owned read-only group-major flux layers."""
    if not layers:
        raise ValueError("result.flux must be a non-empty tuple of layer arrays")
    owned = []
    group_count = None
    for layer in layers:
        flux = readonly_float_array(layer)
        if flux.ndim != 2:
            raise ValueError(
                "each result flux layer must have shape "
                "(groups, active_cells_in_layer)"
            )
        if flux.shape[0] == 0:
            raise ValueError("result flux layers require at least one group")
        require_finite_nonnegative_array("result flux", flux)
        if group_count is None:
            group_count = flux.shape[0]
        elif flux.shape[0] != group_count:
            raise ValueError("all result flux layers must have the same group count")
        owned.append(flux)
    return tuple(owned)


def _require_balance_terms(
    values: Mapping[str, np.ndarray], expected_terms: frozenset[str], mode: str
) -> None:
    """Require exactly the documented balance terms for one solve mode."""
    if set(values) != expected_terms:
        raise ValueError(f"{mode} balance terms do not match the required schema")


def _scalar_balance(values: Mapping[str, np.ndarray]) -> Mapping[str, float]:
    """Return immutable scalar totals contracted from group-resolved terms."""
    return MappingProxyType(
        {name: float(np.sum(vector)) for name, vector in values.items()}
    )


def _total_loss(scalar: Mapping[str, float]) -> float:
    """Return the scalar total over the three physical loss terms."""
    return sum(scalar[name] for name in _LOSS_FRACTION_TERMS)


def _loss_fractions_or_none(scalar: Mapping[str, float]) -> Mapping[str, float] | None:
    """Return loss fractions when their scalar denominator is nonzero."""
    total_loss = _total_loss(scalar)
    if total_loss == 0.0:
        return None
    return _source_normalized(scalar, total_loss, _LOSS_FRACTION_TERMS)


def _source_normalized(
    scalar: Mapping[str, float],
    denominator: float,
    terms: tuple[str, ...] = _SOURCE_NORMALIZED_TERMS,
) -> Mapping[str, float]:
    """Return immutable named scalar terms divided by one checked denominator."""
    return MappingProxyType({name: scalar[name] / denominator for name in terms})


def _check_balance_equation(
    balance: FixedSourceBalance | KeffBalance,
    keff: float | None,
) -> None:
    """Require balance vectors to satisfy their mode-specific group equation."""
    by_group = balance.by_group
    if isinstance(balance, FixedSourceBalance):
        equation_terms = (
            "source",
            "boundary_source",
            "fission_emission",
            "net_scattering",
            "absorption",
            "radial_leakage",
            "axial_leakage",
        )
        expected_residual = (
            by_group["source"]
            + by_group["boundary_source"]
            + by_group["fission_emission"]
            + by_group["net_scattering"]
            - by_group["absorption"]
            - by_group["radial_leakage"]
            - by_group["axial_leakage"]
        )
    else:
        equation_terms = (
            "keff_source",
            "net_scattering",
            "absorption",
            "radial_leakage",
            "axial_leakage",
        )
        expected_residual = (
            by_group["keff_source"]
            + by_group["net_scattering"]
            - by_group["absorption"]
            - by_group["radial_leakage"]
            - by_group["axial_leakage"]
        )
        if keff is None:
            raise ValueError("criticality balance requires a multiplication factor")
        scalar = balance.scalar
        expected_source = scalar["fission_production"] / keff
        if not np.isclose(
            scalar["keff_source"],
            expected_source,
            rtol=_BALANCE_CONSISTENCY_RELATIVE_TOLERANCE,
            atol=_BALANCE_CONSISTENCY_ABSOLUTE_TOLERANCE,
        ):
            raise ValueError(
                "criticality keff_source must equal fission_production divided by keff"
            )
    equation_scale = sum(np.abs(by_group[name]) for name in equation_terms)
    equation_error = np.abs(by_group["residual"] - expected_residual)
    tolerance = (
        _BALANCE_CONSISTENCY_RELATIVE_TOLERANCE * equation_scale
        + _BALANCE_CONSISTENCY_ABSOLUTE_TOLERANCE
    )
    if np.any(equation_error > tolerance):
        raise ValueError(
            "result balance residual does not satisfy its balance equation"
        )


def _owned_balance_vectors(
    values: Mapping[str, np.ndarray], context: str
) -> dict[str, np.ndarray]:
    """Return owned finite one-dimensional group balance vectors."""
    if not isinstance(values, Mapping) or not values:
        if not isinstance(values, Mapping):
            raise TypeError(f"balance {context} must be a mapping")
        raise ValueError(f"balance {context} must be a nonempty mapping")
    owned = {}
    for name, value in values.items():
        require_nonempty_string(f"balance {context} key", name)
        vector = readonly_float_array(value)
        if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError(
                f"balance {context} values must be finite nonempty vectors"
            )
        owned[name] = vector
    return owned


def _owned_layer_balance_vectors(
    values: Mapping[str, tuple[np.ndarray, ...]], by_group: Mapping[str, np.ndarray]
) -> dict[str, tuple[np.ndarray, ...]]:
    """Return owned layer vectors consistent with group-resolved totals."""
    if not isinstance(values, Mapping):
        raise TypeError("balance by_layer_group must be a mapping")
    if set(values) != set(by_group):
        raise ValueError("balance by_layer_group keys must match by_group keys")
    owned = {}
    for name, layers in values.items():
        if not isinstance(layers, tuple) or not layers:
            if not isinstance(layers, tuple):
                raise TypeError("balance by_layer_group values must be tuples")
            raise ValueError("balance by_layer_group values must be tuples")
        layer_values = tuple(readonly_float_array(layer) for layer in layers)
        if any(
            layer.ndim != 1
            or layer.shape != by_group[name].shape
            or not np.all(np.isfinite(layer))
            for layer in layer_values
        ):
            raise ValueError("balance layer vectors must match finite group vectors")
        if not np.allclose(
            np.sum(layer_values, axis=0),
            by_group[name],
            rtol=_BALANCE_CONSISTENCY_RELATIVE_TOLERANCE,
            atol=_BALANCE_CONSISTENCY_ABSOLUTE_TOLERANCE,
        ):
            raise ValueError("balance layer vectors must contract to group vectors")
        owned[name] = layer_values
    return owned
