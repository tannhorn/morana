"""Finite-volume operator assembly helpers."""

# Public NumPy-style operator docstrings make this focused assembly module
# exceed Pylint's generic physical-line heuristic.
# pylint: disable=too-many-lines

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from morana._arrays import (
    freeze_owned_float_array,
    readonly_float_array,
    require_finite_array,
    require_finite_nonnegative_array,
)
from morana.boundary import (
    BoundaryConditionSet,
    _ResolvedBoundaryCondition,
    _resolve_boundary_condition,
)
from morana.configuration import ProblemConfiguration, ProblemConfigurationSnapshot
from morana._validation import require_integer, require_nonnegative_integer
from morana.material_mesh import (
    DOMAIN_FACE_KIND_INTERNAL,
    EXPOSED_DOMAIN_FACE_KINDS,
    DomainFace,
    MaterialMesh,
)
from morana.materials import check_material_name
from morana.normalization import (
    FissionSourceNormalization,
    PowerNormalization,
)
from morana.sources import CellSource

__all__ = [
    "CrossSectionData",
    "CrossSectionLayerData",
    "assemble_boundary_rhs",
    "assemble_fission_matrix",
    "assemble_loss_matrix",
    "assemble_source_rhs",
    "extract_cross_section_data",
]


_JOULES_PER_ELECTRON_VOLT = 1.602176634e-19
_FISSION_FUNCTIONAL_RELATIVE_TOLERANCE = 1.0e-12
_FISSION_FUNCTIONAL_ABSOLUTE_TOLERANCE = 0.0
_RESULT_NORMALIZATION_RELATIVE_TOLERANCE = 1.0e-10
_RESULT_NORMALIZATION_ABSOLUTE_TOLERANCE = 0.0


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class CrossSectionLayerData:
    """Store compact group-indexed cross sections for one axial layer.

    Parameters
    ----------
    axial_index
        Zero-based axial-layer index.
    diffusion
        Diffusion coefficients shaped ``(groups, active_cells)``.
    sigma_a
        Absorption cross sections shaped ``(groups, active_cells)``.
    fission_transfer
        Fission-neutron transfer cross sections shaped
        ``(groups, groups, active_cells)`` and event-oriented as
        ``fission_transfer[g_from, g_to, active_id]``.
    sigma_s
        Complete ordinary scattering-event matrices shaped
        ``(groups, groups, active_cells)`` and indexed from-group, to-group.
    multiplicity_matrix
        Resolved scattering-neutron emission multiplicities shaped
        ``(groups, groups, active_cells)`` and indexed from-group, to-group.
    material_by_active_id
        Compact active-ID-to-material mapping for this layer.

    Attributes
    ----------
    axial_index
        Zero-based axial-layer index associated with every stored array.
    diffusion, sigma_a, sigma_s, multiplicity_matrix, fission_transfer
        Owned read-only arrays using the documented group-major conventions.
    sigma_r
        Fresh read-only conventional removal array shaped
        ``(groups, active_cells)``.
    scattering_coupling
        Fresh read-only signed scattering-neutron coupling shaped
        ``(groups, groups, active_cells)``.
    fission_production
        Fresh read-only incident-group fission-neutron production shaped
        ``(groups, active_cells)`` and derived by summing the outgoing-group
        axis of ``fission_transfer``.
    material_by_active_id
        Copied slice-local compact material provenance with contiguous IDs.

    Raises
    ------
    TypeError
        If ``axial_index`` is not an integer, ``material_by_active_id`` is not
        a mapping, its IDs are not integers, or its material names are not
        strings.
    ValueError
        If no energy group is supplied, array shapes, finite nonnegative values,
        or compact material IDs are invalid.

    Notes
    -----
    ``sigma_r`` and ``scattering_coupling`` are derived from ordinary event
    data and resolved multiplicities. Array values are copied so later changes
    to constructor inputs cannot change an assembled operator. The stored
    arrays and copied ``material_by_active_id`` mapping are
    immutable. Construct a replacement value when changing compact
    cross-section data.
    """

    axial_index: int
    diffusion: np.ndarray
    sigma_a: np.ndarray
    sigma_s: np.ndarray
    multiplicity_matrix: np.ndarray
    fission_transfer: np.ndarray
    material_by_active_id: Mapping[int, str]

    @property
    def groups(self) -> int:
        """Return the number of energy groups."""
        return int(self.diffusion.shape[0])

    @property
    def active_cells(self) -> int:
        """Return the number of active cells."""
        return int(self.diffusion.shape[1])

    @property
    def sigma_r(self) -> np.ndarray:
        """Return fresh read-only conventional absorption-plus-outscatter removal."""
        outscatter = self.sigma_s.copy()
        outscatter[np.diag_indices(self.groups)] = 0.0
        sigma_r = self.sigma_a + np.sum(outscatter, axis=1)
        require_finite_array("sigma_r", sigma_r)
        return freeze_owned_float_array(sigma_r)

    @property
    def scattering_coupling(self) -> np.ndarray:
        """Return fresh read-only signed scattering-neutron coupling."""
        scattering_coupling = self.multiplicity_matrix * self.sigma_s
        diagonal = np.diag_indices(self.groups)
        scattering_coupling[diagonal] -= self.sigma_s[diagonal]
        require_finite_array("scattering_coupling", scattering_coupling)
        return freeze_owned_float_array(scattering_coupling)

    @property
    def fission_production(self) -> np.ndarray:
        """Return fresh read-only fission-neutron production by incident group."""
        return freeze_owned_float_array(np.sum(self.fission_transfer, axis=1))

    def __post_init__(self) -> None:
        """Check compact physics invariants and store owned arrays."""
        axial_index = require_nonnegative_integer("axial_index", self.axial_index)
        diffusion = readonly_float_array(self.diffusion)
        sigma_a = readonly_float_array(self.sigma_a)
        sigma_s = readonly_float_array(self.sigma_s)
        multiplicity_matrix = readonly_float_array(self.multiplicity_matrix)
        arrays_2d = {
            "diffusion": diffusion,
            "sigma_a": sigma_a,
        }
        if any(array.ndim != 2 for array in arrays_2d.values()):
            raise ValueError(
                "cross-section data must have shape (groups, active_cells)"
            )
        if any(array.shape != diffusion.shape for array in arrays_2d.values()):
            raise ValueError(
                "cross-section arrays must have matching (groups, active_cells) shapes"
            )
        groups, active_cells = diffusion.shape
        if groups == 0:
            raise ValueError("cross-section data must contain at least one group")
        expected_scatter_shape = (groups, groups, active_cells)
        if sigma_s.shape != expected_scatter_shape:
            raise ValueError(
                "sigma_s must have shape "
                f"{expected_scatter_shape}, got {sigma_s.shape}"
            )
        if multiplicity_matrix.shape != expected_scatter_shape:
            raise ValueError(
                "multiplicity_matrix must have shape "
                f"{expected_scatter_shape}, got {multiplicity_matrix.shape}"
            )
        fission_transfer = readonly_float_array(self.fission_transfer)
        if fission_transfer.shape != expected_scatter_shape:
            raise ValueError(
                "fission_transfer must have shape "
                f"{expected_scatter_shape}, got {fission_transfer.shape}"
            )
        for name, array in arrays_2d.items():
            require_finite_nonnegative_array(name, array)
        require_finite_nonnegative_array("sigma_s", sigma_s)
        require_finite_nonnegative_array("multiplicity_matrix", multiplicity_matrix)
        require_finite_nonnegative_array("fission_transfer", fission_transfer)

        material_by_active_id = _checked_material_by_active_id(
            self.material_by_active_id, active_cells
        )
        object.__setattr__(self, "axial_index", axial_index)
        object.__setattr__(self, "diffusion", diffusion)
        object.__setattr__(self, "sigma_a", sigma_a)
        object.__setattr__(self, "sigma_s", sigma_s)
        object.__setattr__(self, "multiplicity_matrix", multiplicity_matrix)
        object.__setattr__(self, "fission_transfer", fission_transfer)
        object.__setattr__(
            self, "material_by_active_id", MappingProxyType(material_by_active_id)
        )

    @classmethod
    def _from_owned_validated(
        cls,
        *,
        axial_index: int,
        diffusion: np.ndarray,
        sigma_a: np.ndarray,
        sigma_s: np.ndarray,
        multiplicity_matrix: np.ndarray,
        fission_transfer: np.ndarray,
        material_by_active_id: dict[int, str],
    ) -> "CrossSectionLayerData":
        """Create one layer from owned arrays checked by extraction.

        This private construction path avoids sending arrays freshly assembled
        by `extract_cross_section_data` back through the public raw-input
        constructor.  Its caller must retain sole ownership and have already
        established every compact-physics invariant.
        """
        layer = object.__new__(cls)
        for values in (
            diffusion,
            sigma_a,
            sigma_s,
            multiplicity_matrix,
            fission_transfer,
        ):
            values.setflags(write=False)
        object.__setattr__(layer, "axial_index", axial_index)
        object.__setattr__(layer, "diffusion", diffusion)
        object.__setattr__(layer, "sigma_a", sigma_a)
        object.__setattr__(layer, "sigma_s", sigma_s)
        object.__setattr__(layer, "multiplicity_matrix", multiplicity_matrix)
        object.__setattr__(layer, "fission_transfer", fission_transfer)
        object.__setattr__(
            layer, "material_by_active_id", MappingProxyType(material_by_active_id)
        )
        return layer


# pylint: enable=too-many-instance-attributes


def _checked_material_by_active_id(
    values: Mapping[int, str], active_cells: int
) -> dict[int, str]:
    """Return copied compact material provenance with contiguous active IDs."""
    if not isinstance(values, Mapping):
        raise TypeError("material_by_active_id must be a mapping")
    material_by_active_id = {
        require_nonnegative_integer("material_by_active_id key", active_id): material
        for active_id, material in values.items()
    }
    for material in material_by_active_id.values():
        check_material_name(material)
    if tuple(sorted(material_by_active_id)) != tuple(range(active_cells)):
        raise ValueError("material_by_active_id must match compact active cells")
    return material_by_active_id


@dataclass(frozen=True)
class CrossSectionData:
    """Store compact cross sections for all axial material layers.

    Parameters
    ----------
    layers
        Bottom-to-top iterable of compact layer data using a common group count.

    Attributes
    ----------
    layers
        Bottom-to-top tuple of compact cross-section layers.
    groups
        Shared positive number of energy groups.
    n_axial_layers
        Number of stored axial layers.

    Raises
    ------
    TypeError
        If ``layers`` is not iterable or contains values other than
        ``CrossSectionLayerData``.
    ValueError
        If no layer is supplied, the layers contain no active cells, or the
        supplied layers do not use a common group count.

    Notes
    -----
    This container preserves slice-local active-cell numbering. Global
    node-major packing is created only by the assembly helpers. Layers and
    their ``CrossSectionLayerData`` entries are immutable checked input
    snapshots for one assembly scope.
    """

    layers: tuple[CrossSectionLayerData, ...]

    def __post_init__(self) -> None:
        """Store layers as an immutable tuple and check group consistency."""
        try:
            layers = tuple(self.layers)
        except TypeError as exc:
            raise TypeError("layers must be iterable") from exc
        if any(not isinstance(layer, CrossSectionLayerData) for layer in layers):
            raise TypeError("layers must contain only CrossSectionLayerData")
        if not layers:
            raise ValueError("cross-section data must contain at least one layer")
        if sum(layer.active_cells for layer in layers) == 0:
            raise ValueError("cross-section data must contain at least one active cell")
        group_counts = {layer.groups for layer in layers}
        if len(group_counts) > 1:
            raise ValueError(
                "all cross-section data layers must use the same group count"
            )
        object.__setattr__(self, "layers", layers)

    @property
    def groups(self) -> int:
        """Return the number of energy groups."""
        return self.layers[0].groups

    @property
    def n_axial_layers(self) -> int:
        """Return the number of axial material layers."""
        return len(self.layers)

    def layer(self, axial_index: int) -> CrossSectionLayerData:
        """Return cross-section data for one axial layer.

        Parameters
        ----------
        axial_index
            Zero-based axial-layer index.

        Returns
        -------
        CrossSectionLayerData
            Compact data for the selected layer.

        Raises
        ------
        TypeError
            If ``axial_index`` is not an integer.
        IndexError
            If ``axial_index`` does not select a stored layer.
        """
        axial_index = require_integer("axial_index", axial_index)
        if axial_index < 0 or axial_index >= len(self.layers):
            raise IndexError("axial_index is outside cross-section data")
        return self.layers[axial_index]


@dataclass(frozen=True)
class _FiniteVolumeLayout:
    """Own node-major, group-fastest finite-volume degree-of-freedom layout."""

    groups: int
    active_cells_by_layer: tuple[int, ...]
    node_offsets: tuple[int, ...]

    @classmethod
    def from_active_cells(
        cls,
        *,
        groups: int,
        active_cells_by_layer: tuple[int, ...],
    ) -> "_FiniteVolumeLayout":
        """Construct one layout with offsets owned by its active-cell counts."""
        offsets = []
        offset = 0
        for active_cells in active_cells_by_layer:
            offsets.append(offset)
            offset += active_cells
        return cls(
            groups=groups,
            active_cells_by_layer=active_cells_by_layer,
            node_offsets=tuple(offsets),
        )

    @property
    def size(self) -> int:
        """Return the global number of degrees of freedom."""
        return self.groups * sum(self.active_cells_by_layer)

    def index(self, axial_index: int, active_id: int, group: int) -> int:
        """Return the packed index for one validated group-major value."""
        return (self.node_offsets[axial_index] + active_id) * self.groups + group

    def pack(self, layers: tuple[np.ndarray, ...]) -> np.ndarray:
        """Pack public ``(groups, active_cells)`` arrays into one vector."""
        if len(layers) != len(self.active_cells_by_layer):
            raise ValueError("layer count does not match finite-volume layout")
        packed = np.empty(self.size, dtype=float)
        for axial_index, values in enumerate(layers):
            expected_shape = (self.groups, self.active_cells_by_layer[axial_index])
            if values.shape != expected_shape:
                raise ValueError(
                    f"layer {axial_index} must have shape {expected_shape}, "
                    f"got {values.shape}"
                )
            for active_id in range(expected_shape[1]):
                for group in range(self.groups):
                    packed[self.index(axial_index, active_id, group)] = values[
                        group, active_id
                    ]
        return packed

    def unpack(self, packed: np.ndarray) -> tuple[np.ndarray, ...]:
        """Unpack a node-major vector into public group-major layer arrays."""
        if packed.shape != (self.size,):
            raise ValueError(f"packed vector must have shape ({self.size},)")
        layers = []
        for axial_index, active_cells in enumerate(self.active_cells_by_layer):
            values = np.empty((self.groups, active_cells), dtype=float)
            for active_id in range(active_cells):
                for group in range(self.groups):
                    values[group, active_id] = packed[
                        self.index(axial_index, active_id, group)
                    ]
            layers.append(values)
        return tuple(layers)


@dataclass(frozen=True)
class _InternalInterface:
    """One canonically ordered active-cell interface with shared geometry."""

    primary_axial_index: int
    primary_active_id: int
    secondary_axial_index: int
    secondary_active_id: int
    radial: bool
    face_area: float
    center_to_face: float


@dataclass(frozen=True)
class _ExposedFace:
    """One exposed face with resolved boundary physics and geometry."""

    topology: DomainFace
    face_area: float
    center_to_face: float
    boundary: _ResolvedBoundaryCondition


@dataclass(frozen=True)
class _FiniteVolumeAssemblyContext:
    """One checked layout plus unique interfaces and exposed faces per solve."""

    layout: _FiniteVolumeLayout
    cross_sections: CrossSectionData
    material_mesh: MaterialMesh
    internal_interfaces: tuple[_InternalInterface, ...]
    exposed_faces: tuple[_ExposedFace, ...]


# pylint: disable=too-many-branches,too-many-statements
def extract_cross_section_data(
    configuration: ProblemConfiguration | ProblemConfigurationSnapshot,
) -> CrossSectionData:
    """Extract compact group-indexed data for every axial material layer.

    Parameters
    ----------
    configuration
        Mutable problem configuration or immutable snapshot containing cross
        sections for every active material. Boundary coverage and a source are
        not required. A mutable configuration is captured at entry.

    Returns
    -------
    CrossSectionData
        Owned compact arrays in bottom-to-top, slice-local active-ID order.
        Complete ordinary ``sigma_s`` and resolved multiplicity arrays;
        removal and coupling are derived, and nonfissionable positions receive
        zero fission-transfer blocks. Fissionable positions retain the
        canonical event-oriented transfer data regardless of whether their
        material supplied separable or general transfer input.

    Raises
    ------
    TypeError
        If ``configuration`` is neither ``ProblemConfiguration`` nor
        ``ProblemConfigurationSnapshot``.
    ValueError
        If no material-mesh cell is active, active materials are missing or
        lack cross sections, or active materials use inconsistent energy-group
        counts.

    Notes
    -----
    Extraction is independent of source and boundary data. The returned object
    is suitable for public assembly with the same selected snapshot. Pass a
    snapshot explicitly when several operator calls must use one stable
    problem definition.
    """
    return _extract_cross_section_data(_configuration_snapshot(configuration))


# pylint: disable=too-many-branches,too-many-statements
def _extract_cross_section_data(
    configuration: ProblemConfigurationSnapshot,
) -> CrossSectionData:
    """Extract compact cross sections from one immutable problem definition."""
    material_mesh = configuration.material_mesh
    materials = configuration.materials
    layer_columns: list[
        tuple[
            int,
            dict[int, str],
            list[np.ndarray],
            list[np.ndarray],
            list[np.ndarray],
            list[np.ndarray],
            list[np.ndarray],
        ]
    ] = []
    group_count: int | None = None
    for axial_index in range(material_mesh.n_axial_layers):
        material_by_active_id = material_mesh.material_by_active_id(axial_index)
        diffusion_columns: list[np.ndarray] = []
        sigma_a_columns: list[np.ndarray] = []
        sigma_s_columns: list[np.ndarray] = []
        multiplicity_columns: list[np.ndarray] = []
        fission_transfer_columns: list[np.ndarray] = []
        for active_id in sorted(material_by_active_id):
            material_name = material_by_active_id[active_id]
            try:
                material = materials[material_name]
            except KeyError as exc:
                raise ValueError(
                    f"unknown material in material_mesh: {material_name!r}"
                ) from exc
            if material.xs is None:
                raise ValueError(
                    f"material {material_name!r} is missing cross sections"
                )
            if group_count is None:
                group_count = material.xs.groups
            if material.xs.groups != group_count:
                raise ValueError(
                    "active materials must use the same energy group count"
                )
            diffusion_columns.append(material.xs.D)
            sigma_a_columns.append(material.xs.sigma_a)
            event_scattering = material.xs.sigma_s
            sigma_s_columns.append(event_scattering)
            if material.xs.multiplicity_matrix is None:
                multiplicity_columns.append(
                    np.ones((material.xs.groups, material.xs.groups), dtype=float)
                )
            else:
                multiplicity_columns.append(material.xs.multiplicity_matrix)
            fission = material.xs.fission
            if fission is None:
                fission_transfer_columns.append(
                    np.zeros((material.xs.groups, material.xs.groups), dtype=float)
                )
            else:
                fission_transfer_columns.append(fission.fission_transfer)
        layer_columns.append(
            (
                axial_index,
                material_by_active_id,
                diffusion_columns,
                sigma_a_columns,
                sigma_s_columns,
                multiplicity_columns,
                fission_transfer_columns,
            )
        )

    if group_count is None:
        raise ValueError("cross-section data must contain at least one active cell")

    layers = []
    for (
        axial_index,
        material_by_active_id,
        diffusion_columns,
        sigma_a_columns,
        sigma_s_columns,
        multiplicity_columns,
        fission_transfer_columns,
    ) in layer_columns:
        if not diffusion_columns:
            diffusion = np.empty((group_count, 0))
            sigma_a = np.empty((group_count, 0))
            sigma_s = np.empty((group_count, group_count, 0))
            multiplicity_matrix = np.empty((group_count, group_count, 0))
            fission_transfer = np.empty((group_count, group_count, 0))
        else:
            diffusion = np.column_stack(diffusion_columns)
            sigma_a = np.column_stack(sigma_a_columns)
            sigma_s = np.stack(sigma_s_columns, axis=2)
            multiplicity_matrix = np.stack(multiplicity_columns, axis=2)
            fission_transfer = np.stack(fission_transfer_columns, axis=2)
        # The extraction loop owns these arrays and has derived them solely
        # from immutable checked material data.
        # pylint: disable=protected-access
        layers.append(
            CrossSectionLayerData._from_owned_validated(
                axial_index=axial_index,
                diffusion=diffusion,
                sigma_a=sigma_a,
                sigma_s=sigma_s,
                multiplicity_matrix=multiplicity_matrix,
                fission_transfer=fission_transfer,
                material_by_active_id=material_by_active_id,
            )
        )
        # pylint: enable=protected-access

    return CrossSectionData(layers=tuple(layers))


# pylint: enable=too-many-branches,too-many-statements


def assemble_fission_matrix(
    configuration: ProblemConfiguration | ProblemConfigurationSnapshot,
    cross_sections: CrossSectionData,
) -> csr_matrix:
    """Assemble global volume-integrated fission emission.

    Parameters
    ----------
    configuration
        Mutable problem configuration or immutable snapshot owning the material
        layout. A mutable configuration is captured at entry.
    cross_sections
        Complete layered compact data corresponding to ``configuration``.

    Returns
    -------
    scipy.sparse.csr_matrix
        Fresh mutable node-major, group-fastest fission-emission matrix. For
        cell ``n``, it maps source group ``g_from`` to destination group
        ``g_to`` with ``fission_transfer[g_from, g_to, n] * volume[n]``.

    Raises
    ------
    TypeError
        If ``configuration`` is neither ``ProblemConfiguration`` nor
        ``ProblemConfigurationSnapshot``, or ``cross_sections`` is not a
        ``CrossSectionData``.
    ValueError
        If compact data are not complete or do not match the configuration
        layout, or volume-integrated fission entries are nonfinite or negative.

    Notes
    -----
    This independent assembly operation does not require a fixed source or
    boundary coverage. It returns fission emission only; the solver forms the
    fixed-source operator as loss minus this matrix. Its column sums are
    checked against the volume-integrated incident-group production functional,
    which verifies transfer orientation and volume integration for either
    public fission representation.
    """
    snapshot = _configuration_snapshot(configuration)
    layout = _checked_layout(snapshot, cross_sections)
    return _assemble_fission_matrix(snapshot, cross_sections, layout)


def _assemble_fission_matrix(
    configuration: ProblemConfigurationSnapshot,
    cross_sections: CrossSectionData,
    layout: _FiniteVolumeLayout,
) -> csr_matrix:
    """Assemble fission emission using an already-checked DOF layout."""
    material_mesh = configuration.material_mesh
    entries: list[tuple[int, int, float]] = []
    with np.errstate(over="ignore", invalid="ignore"):
        for layer in cross_sections.layers:
            volume = material_mesh.cell_volume(layer.axial_index)
            for active_id in range(layer.active_cells):
                for group_to in range(layout.groups):
                    for group_from in range(layout.groups):
                        _append_matrix_entry(
                            entries,
                            layout.index(layer.axial_index, active_id, group_to),
                            layout.index(layer.axial_index, active_id, group_from),
                            layer.fission_transfer[group_from, group_to, active_id]
                            * volume,
                        )
    matrix = _entries_matrix(entries, layout.size)
    require_finite_nonnegative_array("global fission matrix entries", matrix.data)
    production_functional = _fission_production_functional(
        configuration, cross_sections, layout
    )
    fission_emission = np.asarray(matrix.sum(axis=0)).ravel()
    if not np.allclose(
        fission_emission,
        production_functional,
        rtol=_FISSION_FUNCTIONAL_RELATIVE_TOLERANCE,
        atol=_FISSION_FUNCTIONAL_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError(
            "fission emission does not preserve the fission-production functional"
        )
    return matrix


def _fission_production_functional(
    configuration: ProblemConfigurationSnapshot,
    cross_sections: CrossSectionData,
    layout: _FiniteVolumeLayout,
) -> np.ndarray:
    """Return fission-production coefficients using a checked DOF layout."""
    material_mesh = configuration.material_mesh
    functional = np.zeros(layout.size, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        for layer in cross_sections.layers:
            volume = material_mesh.cell_volume(layer.axial_index)
            production = layer.fission_production
            for active_id in range(layer.active_cells):
                for group in range(layout.groups):
                    functional[layout.index(layer.axial_index, active_id, group)] = (
                        production[group, active_id] * volume
                    )
    require_finite_nonnegative_array("fission production values", functional)
    return functional


def _fission_power_functional(
    configuration: ProblemConfigurationSnapshot,
    cross_sections: CrossSectionData,
    layout: _FiniteVolumeLayout,
) -> np.ndarray:
    """Return fission-power coefficients using a checked DOF layout."""
    _require_power_normalization_availability(configuration, cross_sections)
    return _assemble_fission_power_functional(configuration, cross_sections, layout)


def _assemble_fission_power_functional(
    configuration: ProblemConfigurationSnapshot,
    cross_sections: CrossSectionData,
    layout: _FiniteVolumeLayout,
) -> np.ndarray:
    """Assemble fission-power coefficients after availability checking."""
    material_mesh = configuration.material_mesh
    materials = configuration.materials
    functional = np.zeros(layout.size, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        for layer in cross_sections.layers:
            volume = material_mesh.cell_volume(layer.axial_index)
            for active_id, material_name in layer.material_by_active_id.items():
                material = materials[material_name]
                fission = material.xs.fission
                kappa_sigma_f = None if fission is None else fission.kappa_sigma_f
                if kappa_sigma_f is None:
                    continue
                for group in range(layout.groups):
                    functional[layout.index(layer.axial_index, active_id, group)] = (
                        kappa_sigma_f[group] * _JOULES_PER_ELECTRON_VOLT * volume
                    )
    require_finite_nonnegative_array("fission power values", functional)
    return functional


def _require_power_normalization_availability(
    configuration: ProblemConfigurationSnapshot,
    cross_sections: CrossSectionData,
) -> None:
    """Require recoverable-energy data after compact layout validation."""
    materials = configuration.materials
    missing = []
    for layer in cross_sections.layers:
        production = layer.fission_production
        for active_id, material_name in layer.material_by_active_id.items():
            if not np.any(production[:, active_id] > 0.0):
                continue
            material = materials[material_name]
            if material.xs is None or material.xs.fission is None:
                missing.append(f"{material_name!r} at axial layer {layer.axial_index}")
                continue
            if material.xs.fission.kappa_sigma_f is None:
                missing.append(f"{material_name!r} at axial layer {layer.axial_index}")
    if missing:
        raise ValueError(
            "power normalization requires kappa_sigma_f for every active "
            "fissionable material; missing " + ", ".join(missing)
        )


def _check_normalization_consistency(
    configuration: ProblemConfigurationSnapshot,
    cross_sections: CrossSectionData,
    flux_layers: tuple[np.ndarray, ...],
    normalization: FissionSourceNormalization | PowerNormalization,
    *,
    relative_tolerance: float = _RESULT_NORMALIZATION_RELATIVE_TOLERANCE,
) -> None:
    """Require flux and compact cross sections to reproduce the target."""
    layout = _checked_layout(configuration, cross_sections)
    if isinstance(normalization, FissionSourceNormalization):
        functional = _fission_production_functional(
            configuration, cross_sections, layout
        )
        target = normalization.rate
        label = "fission-source rate"
    else:
        functional = _fission_power_functional(configuration, cross_sections, layout)
        target = normalization.power
        label = "power"
    response = float(np.dot(functional, layout.pack(flux_layers)))
    if not np.isfinite(response) or not np.isclose(
        response,
        target,
        rtol=relative_tolerance,
        atol=_RESULT_NORMALIZATION_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError(
            f"{label} normalization is inconsistent with the result flux and "
            "configuration snapshot"
        )


# pylint: disable=too-many-nested-blocks
def assemble_loss_matrix(
    configuration: ProblemConfiguration | ProblemConfigurationSnapshot,
    cross_sections: CrossSectionData,
) -> csr_matrix:
    """Assemble diffusion loss, removal, and scattering coupling.

    Parameters
    ----------
    configuration
        Mutable problem configuration or immutable snapshot with complete
        exposed-face boundary coverage. A mutable configuration is captured at
        entry.
    cross_sections
        Compact data extracted from the same configuration.

    Returns
    -------
    scipy.sparse.csr_matrix
        Fresh mutable node-major, group-fastest global loss matrix. It combines
        removal, scattering coupling, radial/axial leakage, and resolved
        boundary response.

    Raises
    ------
    TypeError
        If ``configuration`` is neither ``ProblemConfiguration`` nor
        ``ProblemConfigurationSnapshot``, or ``cross_sections`` is not a
        ``CrossSectionData``.
    ValueError
        If compact data are not complete or do not match the configuration,
        diffusion coefficients are invalid, exposed faces lack complete
        group-compatible boundary coverage, or assembled entries fail their
        finite/sign checks.

    Notes
    -----
    In the packed system, diagonal loss entries are positive and transfer or
    neighbor couplings are nonpositive. Inhomogeneous boundary data add both
    a response contribution here and a separate contribution from
    ``assemble_boundary_rhs()``. No volumetric source is required.
    """
    context = _finite_volume_assembly_context(
        _configuration_snapshot(configuration), cross_sections
    )
    return _assemble_loss_matrix(context)


def _assemble_loss_matrix(context: _FiniteVolumeAssemblyContext) -> csr_matrix:
    """Assemble loss using already-checked finite-volume assembly data."""
    layout = context.layout
    cross_sections = context.cross_sections
    material_mesh = context.material_mesh
    entries: list[tuple[int, int, float]] = []
    for layer in cross_sections.layers:
        axial_index = layer.axial_index
        cell_volume = material_mesh.cell_volume(axial_index)
        sigma_r = layer.sigma_r
        scattering_coupling = layer.scattering_coupling
        for group in range(layout.groups):
            for active_id in range(layer.active_cells):
                row = layout.index(axial_index, active_id, group)
                _append_matrix_entry(
                    entries, row, row, sigma_r[group, active_id] * cell_volume
                )
            for active_id in range(layer.active_cells):
                for group_to in range(layout.groups):
                    _append_matrix_entry(
                        entries,
                        layout.index(axial_index, active_id, group_to),
                        layout.index(axial_index, active_id, group),
                        -scattering_coupling[group, group_to, active_id] * cell_volume,
                    )
    for interface in context.internal_interfaces:
        for group in range(layout.groups):
            conductance = _internal_interface_conductance(
                context, interface, group=group
            )
            primary = layout.index(
                interface.primary_axial_index, interface.primary_active_id, group
            )
            secondary = layout.index(
                interface.secondary_axial_index,
                interface.secondary_active_id,
                group,
            )
            _append_matrix_entry(entries, primary, primary, conductance)
            _append_matrix_entry(entries, primary, secondary, -conductance)
            _append_matrix_entry(entries, secondary, secondary, conductance)
            _append_matrix_entry(entries, secondary, primary, -conductance)
    for face in context.exposed_faces:
        layer = cross_sections.layer(face.topology.axial_index)
        for group in range(layout.groups):
            conductance = _exposed_face_conductance(face, layer, group)
            _append_matrix_entry(
                entries,
                layout.index(face.topology.axial_index, face.topology.active_id, group),
                layout.index(face.topology.axial_index, face.topology.active_id, group),
                conductance,
            )
    return _loss_entries_matrix(entries, layout.size)


# pylint: enable=too-many-nested-blocks


def assemble_source_rhs(
    configuration: ProblemConfiguration | ProblemConfigurationSnapshot,
    cross_sections: CrossSectionData,
) -> np.ndarray:
    """Assemble the volume-integrated volumetric source right-hand side.

    Parameters
    ----------
    configuration
        Mutable problem configuration or immutable snapshot. Boundary coverage
        is not required by this independent assembly operation. A mutable
        configuration is captured at entry.
    cross_sections
        Complete layered compact data corresponding to ``configuration``.

    Returns
    -------
    numpy.ndarray
        Fresh mutable node-major, group-fastest global source vector. Each
        volumetric source value is multiplied by its selected cell volume. If
        no volumetric source is configured, the vector is zero.

    Raises
    ------
    TypeError
        If ``configuration`` is neither ``ProblemConfiguration`` nor
        ``ProblemConfigurationSnapshot``, or ``cross_sections`` is not a
        ``CrossSectionData``.
    ValueError
        If compact data are incomplete or do not match the configuration
        layout; a cell source has the wrong layer count; or evaluated source
        values have an invalid shape, are nonfinite, negative, or overflow
        after volume integration.

    Notes
    -----
    Boundary coverage is not required. This function assembles only the
    volumetric fixed-source term; add ``assemble_boundary_rhs()`` separately
    when an inhomogeneous resolved boundary is present.
    """
    snapshot = _configuration_snapshot(configuration)
    layout = _checked_layout(snapshot, cross_sections)
    return _assemble_source_rhs(snapshot, cross_sections, layout)


def _assemble_source_rhs(
    configuration: ProblemConfigurationSnapshot,
    cross_sections: CrossSectionData,
    layout: _FiniteVolumeLayout,
) -> np.ndarray:
    """Assemble an optional volumetric source using an already-checked layout."""
    source = configuration.source
    material_mesh = configuration.material_mesh
    if source is None:
        return np.zeros(layout.size, dtype=float)
    if (
        isinstance(source, CellSource)
        and len(source.layers) != cross_sections.n_axial_layers
    ):
        raise ValueError(
            "cell source layer count must match the material-mesh axial stack"
        )
    values_by_layer = []
    for layer in cross_sections.layers:
        values = np.asarray(
            source.values(layer.axial_index, layer.material_by_active_id),
            dtype=float,
        )
        expected_shape = (layout.groups, layer.active_cells)
        if values.shape != expected_shape:
            raise ValueError(
                f"source values must have shape {expected_shape}, got {values.shape}"
            )
        require_finite_nonnegative_array("source values", values)
        with np.errstate(over="ignore", invalid="ignore"):
            integrated_values = np.array(
                values * material_mesh.cell_volume(layer.axial_index),
                dtype=float,
                copy=True,
            )
        require_finite_nonnegative_array("source RHS values", integrated_values)
        values_by_layer.append(integrated_values)
    return layout.pack(tuple(values_by_layer))


def assemble_boundary_rhs(
    configuration: ProblemConfiguration | ProblemConfigurationSnapshot,
    cross_sections: CrossSectionData,
) -> np.ndarray:
    """Assemble prescribed-flux and incoming-current boundary sources.

    Parameters
    ----------
    configuration
        Mutable problem configuration or immutable snapshot with complete
        exposed-face boundary coverage. A mutable configuration is captured at
        entry.
    cross_sections
        Complete layered compact data corresponding to ``configuration``.

    Returns
    -------
    numpy.ndarray
        Fresh mutable node-major, group-fastest global boundary-source vector.
        It contains only additive prescribed-flux and imposed-current boundary
        terms, integrated over their selected exposed faces.

    Raises
    ------
    TypeError
        If ``configuration`` is neither ``ProblemConfiguration`` nor
        ``ProblemConfigurationSnapshot``, or ``cross_sections`` is not a
        ``CrossSectionData``.
    ValueError
        If compact data are incomplete or do not match the configuration;
        exposed faces lack complete, group-compatible boundary coverage; or
        integrated boundary entries are nonfinite or negative.

    Notes
    -----
    A fixed source is not required. Homogeneous conditions produce no additive
    entry; their response remains in ``assemble_loss_matrix()``.
    """
    context = _finite_volume_assembly_context(
        _configuration_snapshot(configuration), cross_sections
    )
    return _assemble_boundary_rhs(context)


def _assemble_boundary_rhs(context: _FiniteVolumeAssemblyContext) -> np.ndarray:
    """Assemble boundary source using already-resolved exposed-face data."""
    layout = context.layout
    cross_sections = context.cross_sections
    rhs = np.zeros(layout.size, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        for face in context.exposed_faces:
            layer = cross_sections.layer(face.topology.axial_index)
            for group in range(layout.groups):
                diffusion = layer.diffusion[group, face.topology.active_id]
                index = layout.index(
                    face.topology.axial_index, face.topology.active_id, group
                )
                rhs[index] += _boundary_rhs_contribution(
                    face.boundary,
                    diffusion,
                    face.face_area,
                    face.center_to_face,
                    group,
                )
    require_finite_nonnegative_array("boundary RHS values", rhs)
    return rhs


def _checked_layout(
    configuration: ProblemConfigurationSnapshot,
    cross_sections: CrossSectionData,
) -> _FiniteVolumeLayout:
    """Check complete compact data and construct its sole DOF layout."""
    if not isinstance(cross_sections, CrossSectionData):
        raise TypeError("global assembly requires complete CrossSectionData")
    material_mesh = configuration.material_mesh
    if cross_sections.n_axial_layers != material_mesh.n_axial_layers:
        raise ValueError("cross-section layer count does not match configuration")
    if cross_sections.groups <= 0:
        raise ValueError("cross-section data must contain at least one group")
    for axial_index, layer in enumerate(cross_sections.layers):
        if layer.axial_index != axial_index:
            raise ValueError("cross-section layer indices must be contiguous")
        material_by_active_id = material_mesh.material_by_active_id(axial_index)
        if layer.material_by_active_id != material_by_active_id:
            raise ValueError(
                "cross-section data does not match the selected material-mesh layer"
            )
    active_cells_by_layer = tuple(layer.active_cells for layer in cross_sections.layers)
    return _FiniteVolumeLayout.from_active_cells(
        groups=cross_sections.groups,
        active_cells_by_layer=active_cells_by_layer,
    )


def _configuration_snapshot(
    configuration: ProblemConfiguration | ProblemConfigurationSnapshot,
) -> ProblemConfigurationSnapshot:
    """Capture one mutable operator input or retain an immutable input."""
    if isinstance(configuration, ProblemConfigurationSnapshot):
        return configuration
    if isinstance(configuration, ProblemConfiguration):
        return configuration.snapshot()
    raise TypeError(
        "configuration must be a ProblemConfiguration or "
        "ProblemConfigurationSnapshot"
    )


def _finite_volume_assembly_context(
    configuration: ProblemConfigurationSnapshot,
    cross_sections: CrossSectionData,
) -> _FiniteVolumeAssemblyContext:
    """Construct one short-lived checked context for related assembly work."""
    layout = _checked_layout(configuration, cross_sections)
    return _finite_volume_assembly_context_from_layout(
        configuration, cross_sections, layout
    )


def _finite_volume_assembly_context_from_layout(
    configuration: ProblemConfigurationSnapshot,
    cross_sections: CrossSectionData,
    layout: _FiniteVolumeLayout,
) -> _FiniteVolumeAssemblyContext:
    """Construct one finite-volume context using an already-checked layout."""
    material_mesh = configuration.material_mesh
    internal_interfaces, exposed_faces = _resolve_finite_volume_interfaces(
        material_mesh, configuration.boundary, layout
    )
    return _FiniteVolumeAssemblyContext(
        layout=layout,
        cross_sections=cross_sections,
        material_mesh=material_mesh,
        internal_interfaces=internal_interfaces,
        exposed_faces=exposed_faces,
    )


def _resolve_finite_volume_interfaces(
    material_mesh: MaterialMesh,
    boundary: BoundaryConditionSet,
    layout: _FiniteVolumeLayout,
) -> tuple[tuple[_InternalInterface, ...], tuple[_ExposedFace, ...]]:
    """Resolve unique interfaces and exposed faces for one assembly scope."""
    interfaces: list[_InternalInterface] = []
    exposed_faces: list[_ExposedFace] = []
    boundary_cache: dict[
        tuple[str, str, str | None, str | None], _ResolvedBoundaryCondition
    ] = {}
    geometry_cache: dict[tuple[int, str], tuple[float, float]] = {}
    directions = material_mesh.face_direction_labels
    for axial_index, active_cells in enumerate(layout.active_cells_by_layer):
        for active_id in range(active_cells):
            for direction in directions:
                face = material_mesh.face(axial_index, active_id, direction)
                geometry_key = (axial_index, direction)
                geometry = geometry_cache.get(geometry_key)
                if geometry is None:
                    geometry = _face_geometry(material_mesh, axial_index, direction)
                    geometry_cache[geometry_key] = geometry
                face_area, center_to_face = geometry
                if face.kind == DOMAIN_FACE_KIND_INTERNAL:
                    _append_internal_interface(
                        interfaces,
                        face,
                        face_area,
                        center_to_face,
                        material_mesh,
                    )
                    continue
                if face.kind in EXPOSED_DOMAIN_FACE_KINDS:
                    boundary_key = (
                        face.kind,
                        face.direction,
                        face.neighbor_key,
                        face.neighbor_key_kind,
                    )
                    resolved_boundary = boundary_cache.get(boundary_key)
                    if resolved_boundary is None:
                        resolved_boundary = _resolve_boundary_condition(
                            boundary.resolve(face), layout.groups
                        )
                        boundary_cache[boundary_key] = resolved_boundary
                    exposed_faces.append(
                        _ExposedFace(
                            topology=face,
                            face_area=face_area,
                            center_to_face=center_to_face,
                            boundary=resolved_boundary,
                        )
                    )
                    continue
                raise ValueError(f"unknown domain-face kind: {face.kind!r}")
    return tuple(interfaces), tuple(exposed_faces)


def _append_internal_interface(
    interfaces: list[_InternalInterface],
    face: DomainFace,
    face_area: float,
    center_to_face: float,
    material_mesh: MaterialMesh,
) -> None:
    """Append one canonical interface from one directed topology query."""
    if face.neighbor_axial_index is None or face.neighbor_active_id is None:
        raise ValueError("internal face is missing an active neighbor")
    primary = (face.axial_index, face.active_id)
    secondary = (face.neighbor_axial_index, face.neighbor_active_id)
    if primary >= secondary:
        return
    interfaces.append(
        _InternalInterface(
            primary_axial_index=face.axial_index,
            primary_active_id=face.active_id,
            secondary_axial_index=face.neighbor_axial_index,
            secondary_active_id=face.neighbor_active_id,
            radial=face.direction in material_mesh.mesh.direction_labels,
            face_area=face_area,
            center_to_face=center_to_face,
        )
    )


def _append_matrix_entry(
    entries: list[tuple[int, int, float]],
    row: int,
    column: int,
    value: float,
) -> None:
    """Append one sparse matrix contribution."""
    if value != 0.0:
        entries.append((row, column, value))


def _entries_matrix(entries: list[tuple[int, int, float]], size: int) -> csr_matrix:
    """Return a square sparse matrix from accumulated contributions."""
    return _entries_coo_matrix(entries, size).tocsr()


def _loss_entries_matrix(
    entries: list[tuple[int, int, float]], size: int
) -> csr_matrix:
    """Check accumulated loss entries, then return their CSR representation."""
    matrix = _entries_coo_matrix(entries, size)
    with np.errstate(over="ignore", invalid="ignore"):
        matrix.sum_duplicates()
    _check_global_loss_matrix(matrix)
    return matrix.tocsr()


def _entries_coo_matrix(entries: list[tuple[int, int, float]], size: int) -> coo_matrix:
    """Return a square COO matrix from accumulated contributions."""
    if not entries:
        return coo_matrix((size, size), dtype=float)
    rows, columns, values = zip(*entries)
    return coo_matrix((values, (rows, columns)), shape=(size, size))


def _check_global_loss_matrix(matrix: coo_matrix) -> None:
    """Require COO loss/transfer signs before CSR conversion."""
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("global loss matrix must be square")
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("global loss matrix entries must be finite")
    diagonal_mask = matrix.row == matrix.col
    diagonal = np.bincount(
        matrix.row[diagonal_mask],
        weights=matrix.data[diagonal_mask],
        minlength=matrix.shape[0],
    )
    if np.any(diagonal <= 0.0):
        raise ValueError("global loss matrix diagonal entries must be positive")
    if np.any(matrix.data[~diagonal_mask] > 0.0):
        raise ValueError("global loss matrix off-diagonal entries must be non-positive")


def _weighted_harmonic_mean(
    left_diffusion: float,
    right_diffusion: float,
    left_distance: float,
    right_distance: float,
) -> float:
    """Return the distance-weighted harmonic interface diffusion coefficient."""
    if left_diffusion == 0.0 or right_diffusion == 0.0:
        return 0.0
    return (left_distance + right_distance) / (
        left_distance / left_diffusion + right_distance / right_diffusion
    )


def _radial_conductance(
    left_diffusion: float,
    right_diffusion: float,
    face_area: float,
    center_distance: float,
) -> float:
    """Return the harmonic-interface radial diffusion conductance."""
    half_distance = center_distance / 2.0
    return (
        _weighted_harmonic_mean(
            left_diffusion,
            right_diffusion,
            half_distance,
            half_distance,
        )
        * face_area
        / center_distance
    )


def _axial_conductance(
    lower_diffusion: float,
    upper_diffusion: float,
    face_area: float,
    lower_height: float,
    upper_height: float,
) -> float:
    """Return the unequal-half-cell axial diffusion conductance."""
    lower_half_height = lower_height / 2.0
    upper_half_height = upper_height / 2.0
    return (
        _weighted_harmonic_mean(
            lower_diffusion,
            upper_diffusion,
            lower_half_height,
            upper_half_height,
        )
        * face_area
        / (lower_half_height + upper_half_height)
    )


def _internal_interface_conductance(
    context: _FiniteVolumeAssemblyContext,
    interface: _InternalInterface,
    *,
    group: int,
) -> float:
    """Return the conductance across one canonical internal interface."""
    material_mesh = context.material_mesh
    primary_layer = context.cross_sections.layer(interface.primary_axial_index)
    secondary_layer = context.cross_sections.layer(interface.secondary_axial_index)
    if interface.radial:
        return _radial_conductance(
            primary_layer.diffusion[group, interface.primary_active_id],
            secondary_layer.diffusion[group, interface.secondary_active_id],
            interface.face_area,
            2.0 * interface.center_to_face,
        )
    return _axial_conductance(
        primary_layer.diffusion[group, interface.primary_active_id],
        secondary_layer.diffusion[group, interface.secondary_active_id],
        interface.face_area,
        material_mesh.layer_height(interface.primary_axial_index),
        material_mesh.layer_height(interface.secondary_axial_index),
    )


def _exposed_face_conductance(
    face: _ExposedFace,
    layer: CrossSectionLayerData,
    group: int,
) -> float:
    """Return the loss conductance for one resolved exposed face."""
    conductance, _ = _exposed_boundary_contribution(
        boundary=face.boundary,
        diffusion=layer.diffusion[group, face.topology.active_id],
        face_area=face.face_area,
        center_to_face=face.center_to_face,
        group=group,
    )
    return conductance


def _face_geometry(
    material_mesh: MaterialMesh,
    axial_index: int,
    direction: str,
) -> tuple[float, float]:
    """Return the area and cell-center distance for one canonical face label."""
    if direction in material_mesh.mesh.direction_labels:
        return (
            material_mesh.radial_face_area(axial_index),
            material_mesh.mesh.center_to_face,
        )
    return (
        material_mesh.axial_face_area(axial_index),
        material_mesh.layer_height(axial_index) / 2.0,
    )


def _boundary_rhs_contribution(
    boundary: _ResolvedBoundaryCondition,
    diffusion: float,
    face_area: float,
    center_to_face: float,
    group: int,
) -> float:
    """Return one integrated prescribed boundary source contribution."""
    if boundary.kind == "dirichlet":
        return diffusion * face_area / center_to_face * boundary.flux[group]
    _, source = _exposed_boundary_contribution(
        boundary=boundary,
        diffusion=diffusion,
        face_area=face_area,
        center_to_face=center_to_face,
        group=group,
    )
    return source


def _exposed_boundary_contribution(
    boundary: _ResolvedBoundaryCondition,
    diffusion: float,
    face_area: float,
    center_to_face: float,
    *,
    group: int,
) -> tuple[float, float]:
    """Return affine Robin loss and source contributions for one exposed face.

    Dirichlet conditions remain a separate prescribed-flux path. Every other
    condition is normalized to ``J_out = alpha * phi_face - s``. Eliminating
    the face flux yields a diagonal loss conductance and an incoming boundary
    source, both integrated over the radial face area.
    """
    if boundary.kind == "dirichlet":
        return diffusion * face_area / center_to_face, 0.0
    if boundary.alpha == 0.0:
        return 0.0, face_area * boundary.source[group]
    denominator = diffusion + boundary.alpha * center_to_face
    return (
        face_area * boundary.alpha * diffusion / denominator,
        face_area * diffusion * boundary.source[group] / denominator,
    )
