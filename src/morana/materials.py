"""Material identities and macroscopic cross-section containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from morana._arrays import (
    float_array,
    require_finite_nonnegative_array,
    require_nonzero_array,
)
from morana._openmc_mgxs import read_openmc_runtime_mgxs
from morana._validation import (
    require_finite_positive_real,
    require_human_readable_identifier,
    require_optional_string,
)

INACTIVE_EXCLUDED_KEY = "0"
INACTIVE_EXCLUDED_KIND = "inactive"
INACTIVE_EXCLUDED_REGION_COLOR = "white"
DEFAULT_EXCLUDED_REGION_COLOR = "#d9d9d9"


@dataclass(frozen=True)
class ExcludedRegion:
    """Describe a non-solved material-mesh region.

    Parameters
    ----------
    kind
        Human-readable public region-kind identifier used by boundary selectors
        and provenance. It must contain a non-whitespace character and only
        printable characters. Its spelling is preserved, it is not restricted
        to a closed vocabulary, and kind-based selection uses exact equality.
    color
        Optional nonempty plotting-color string. Its syntax is not validated at
        construction. A custom region without a color resolves to ``"#d9d9d9"`` for
        material-layout plots.

    Raises
    ------
    ValueError
        If ``kind`` is empty, whitespace-only, or contains a non-printable
        character.
    TypeError
        If ``kind`` is not a string, or ``color`` is neither a string nor
        ``None``.

    Notes
    -----
    ``ExcludedRegion`` is an immutable region description, not an excluded-position
    identity. Register it as a value in ``MaterialMesh.stack``'s
    ``excluded_regions`` mapping; that mapping's string key is the
    excluded-region identity. Select one such identity with
    ``on_excluded(key=...)`` or every identity sharing this ``kind`` with
    ``on_excluded(kind=...)``.

    Excluded regions live in ``MaterialMesh`` layouts but are not active
    materials. They receive no unknown, source, cross sections, or neutron
    balance contribution. The built-in key ``"0"`` is always the reserved
    ``"inactive"`` region with plotting color ``"white"``; it cannot be
    redefined.
    """

    kind: str
    color: str | None = None

    def __post_init__(self) -> None:
        """Check the excluded-region description."""
        require_human_readable_identifier("excluded-region kind", self.kind)
        require_optional_string("excluded-region color", self.color)


BUILTIN_EXCLUDED_REGIONS = {
    INACTIVE_EXCLUDED_KEY: ExcludedRegion(
        kind=INACTIVE_EXCLUDED_KIND, color=INACTIVE_EXCLUDED_REGION_COLOR
    ),
}


@dataclass(frozen=True)
class SeparableFission:
    """Store compact separable fission-neutron production data.

    Parameters
    ----------
    nu_sigma_f
        One-dimensional incident-group fission-neutron production cross
        section in ``1 / cm``. It must be finite, nonnegative, and contain at
        least one positive value.
    chi
        One-dimensional outgoing-group fission spectrum. It must have the
        same group count as ``nu_sigma_f``, be finite and nonnegative, have a
        positive sum, and be within ``chi_normalization_tolerance`` of one.
        Accepted spectra are normalized before storage.
    chi_normalization_tolerance
        Finite positive accepted deviation of the input fission-spectrum sum
        from one. Defaults to ``1.0e-8``.

    Attributes
    ----------
    groups
        Number of energy groups.
    fission_transfer
        Read-only derived fission-neutron transfer array with event-oriented
        indexing ``[g_from, g_to]``.
    fission_production
        Read-only derived total neutron production by incident group.

    Notes
    -----
    ``SeparableFission`` owns immutable numerical inputs. Its derived transfer
    is ``nu_sigma_f[g_from] * chi[g_to]``.
    """

    nu_sigma_f: np.ndarray | list[float]
    chi: np.ndarray | list[float]
    chi_normalization_tolerance: float = field(default=1.0e-8, kw_only=True)
    groups: int = field(default=0, init=False)
    fission_transfer: np.ndarray = field(init=False, repr=False)
    fission_production: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and normalize compact fission-neutron production data."""
        production = float_array("nu_sigma_f", self.nu_sigma_f)
        spectrum = float_array("chi", self.chi)
        if production.ndim != 1 or spectrum.ndim != 1:
            raise ValueError("nu_sigma_f and chi must be one-dimensional arrays")
        if production.shape != spectrum.shape:
            raise ValueError("nu_sigma_f and chi must have the same group count")
        if production.size == 0:
            raise ValueError("at least one energy group is required")
        require_finite_nonnegative_array("nu_sigma_f", production)
        require_finite_nonnegative_array("chi", spectrum)
        require_nonzero_array("nu_sigma_f", production)

        tolerance = require_finite_positive_real(
            "chi_normalization_tolerance", self.chi_normalization_tolerance
        )
        spectrum_sum = float(np.sum(spectrum))
        if spectrum_sum <= 0.0:
            raise ValueError("chi must have a positive sum")
        if abs(spectrum_sum - 1.0) > tolerance:
            raise ValueError(
                "chi sum must be within chi_normalization_tolerance of one"
            )
        spectrum /= spectrum_sum
        transfer = production[:, np.newaxis] * spectrum[np.newaxis, :]
        derived_production = np.sum(transfer, axis=1)

        for array in (production, spectrum, transfer, derived_production):
            array.setflags(write=False)
        object.__setattr__(self, "nu_sigma_f", production)
        object.__setattr__(self, "chi", spectrum)
        object.__setattr__(self, "chi_normalization_tolerance", tolerance)
        object.__setattr__(self, "groups", int(production.size))
        object.__setattr__(self, "fission_transfer", transfer)
        object.__setattr__(self, "fission_production", derived_production)


@dataclass(frozen=True)
class FissionTransfer:
    """Store general fission-neutron transfer data.

    Parameters
    ----------
    fission_transfer
        Square fission-neutron transfer array in ``1 / cm`` with
        event-oriented indexing ``fission_transfer[g_from, g_to]``. It must
        be finite, nonnegative, and have nonzero total neutron production.

    Attributes
    ----------
    groups
        Number of energy groups.
    fission_production
        Read-only derived total neutron production by incident group.
    """

    fission_transfer: np.ndarray | list[list[float]]
    groups: int = field(default=0, init=False)
    fission_production: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate general fission-neutron transfer data."""
        transfer = float_array("fission_transfer", self.fission_transfer)
        if transfer.ndim != 2 or transfer.shape[0] != transfer.shape[1]:
            raise ValueError("fission_transfer must be a square two-dimensional array")
        if transfer.shape[0] == 0:
            raise ValueError("at least one energy group is required")
        require_finite_nonnegative_array("fission_transfer", transfer)
        require_nonzero_array("fission_transfer", transfer)
        with np.errstate(over="ignore"):
            production = np.sum(transfer, axis=1)
        require_finite_nonnegative_array("fission_production", production)

        transfer.setflags(write=False)
        production.setflags(write=False)
        object.__setattr__(self, "fission_transfer", transfer)
        object.__setattr__(self, "groups", int(transfer.shape[0]))
        object.__setattr__(self, "fission_production", production)


@dataclass(frozen=True)
class FissionData:
    """Bundle fission physics for one multigroup material.

    Parameters
    ----------
    neutron_production
        Exactly one checked neutron-production representation:
        ``SeparableFission`` or ``FissionTransfer``.
    kappa_sigma_f
        Optional incident-group recoverable fission-energy production cross
        section in ``eV / cm``. When supplied, it must be finite,
        nonnegative, and match the neutron-production group count.

    Attributes
    ----------
    groups
        Number of energy groups.
    fission_transfer
        Canonical read-only event-oriented transfer array with indexing
        ``[g_from, g_to]``.
    fission_production
        Canonical read-only total neutron production by incident group.

    Notes
    -----
    ``FissionData`` retains the caller-selected neutron-production value. Use
    ``SeparableFission`` when a common outgoing spectrum is appropriate; use
    ``FissionTransfer`` for general incident-to-outgoing group production.
    The derived ``fission_transfer`` and ``fission_production`` properties give
    both forms one common operator-facing representation.
    """

    neutron_production: SeparableFission | FissionTransfer
    kappa_sigma_f: np.ndarray | list[float] | None = field(default=None, kw_only=True)
    groups: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Validate the selected production representation and energy data."""
        production = self.neutron_production
        if not isinstance(production, (SeparableFission, FissionTransfer)):
            raise TypeError(
                "neutron_production must be a SeparableFission or FissionTransfer"
            )

        energy_production = None
        if self.kappa_sigma_f is not None:
            energy_production = _required_cross_section_array(
                "kappa_sigma_f", self.kappa_sigma_f, (production.groups,)
            )
            require_finite_nonnegative_array("kappa_sigma_f", energy_production)
            energy_production.setflags(write=False)

        object.__setattr__(self, "kappa_sigma_f", energy_production)
        object.__setattr__(self, "groups", production.groups)

    @property
    def fission_transfer(self) -> np.ndarray:
        """Return the selected canonical event-oriented transfer array."""
        return self.neutron_production.fission_transfer

    @property
    def fission_production(self) -> np.ndarray:
        """Return the selected canonical incident-group production array."""
        return self.neutron_production.fission_production


@dataclass(frozen=True)
class CrossSections:
    """Store group-major macroscopic cross sections.

    Parameters
    ----------
    D
        Nonempty one-dimensional diffusion-coefficient array in cm. Its length
        defines the number of energy groups.
    sigma_a
        One-dimensional absorption macroscopic cross-section array in
        ``1 / cm`` with the same length as ``D``.
    sigma_s
        Complete P0 scattering-transfer matrix in ``1 / cm`` with shape
        ``(groups, groups)`` and convention ``sigma_s[g_from, g_to]``.
    multiplicity_matrix
        Optional scattering-neutron emission multiplicity with shape
        ``(groups, groups)`` and the same incoming-to-outgoing convention as
        ``sigma_s``. Values must be finite and nonnegative. ``None`` retains
        the compact unit-multiplicity representation: every scattering event
        emits one neutron.
    fission
        Optional checked fission-physics bundle. ``None`` is the sole
        representation for a nonfissile material. A supplied bundle must have
        the same group count as the other cross sections.

    Attributes
    ----------
    groups
        Number of energy groups derived from the length of ``D``.

    Raises
    ------
    TypeError
        If a numerical field contains a non-real or Boolean value.
    ValueError
        If shapes are inconsistent, no groups are supplied, a numeric value is
        nonfinite or negative, or the supplied fission bundle has an
        inconsistent group count. ``FissionData`` and its selected
        neutron-production representation validate their own fission-specific
        inputs.

    Notes
    -----
    Public group ordering is fast-to-thermal. Inputs are converted to NumPy
    arrays immediately and copied so the container owns its numerical data.
    Stored numerical arrays are read-only. ``CrossSections`` is immutable after
    construction. Construct a replacement value when changing group data.
    """

    D: np.ndarray | list[float]
    sigma_a: np.ndarray | list[float]
    sigma_s: np.ndarray | list[list[float]]
    fission: FissionData | None
    multiplicity_matrix: np.ndarray | list[list[float]] | None = field(
        default=None, kw_only=True
    )
    groups: int = field(default=0, init=False)

    @classmethod
    def from_openmc_mgxs_hdf5(
        cls,
        path: str | Path,
        dataset: str,
        temperature: float,
        *,
        diffusion: str,
    ) -> CrossSections:
        """Import selected macroscopic data from an OpenMC runtime-MGXS file.

        Parameters
        ----------
        path
            OpenMC runtime-library HDF5 file emitted by
            ``openmc.MGXSLibrary.export_to_hdf5(...)``. Morana accepts
            ``filetype="mgxs"`` format version 1.0, reads it directly, and
            does not import OpenMC.
        dataset
            Exact name of one direct-child macroscopic record in the runtime
            library. Nuclide-like records carrying ``atomic_weight_ratio`` are
            rejected.
        temperature
            Exact stored physical temperature in K, apart from a small
            floating-point representation tolerance.
        diffusion
            Required diffusion convention. ``"total"`` uses the runtime
            record's ``total`` vector directly in ``D = 1 / (3 total)``;
            OpenMC may have populated that field from either ``TotalXS`` or
            ``TransportXS``. ``"p1-outscatter"`` requires an uncorrected
            ``TotalXS`` in ``total`` and derives
            ``D_g = 1 / (3 (Sigma_t,g - sum_h Sigma_s1,g->h))``.

        Returns
        -------
        CrossSections
            Immutable selected material data. The imported value retains no
            source-file, energy-grid, or selection provenance.

        Raises
        ------
        TypeError
            If ``path``, ``dataset``, ``temperature``, or ``diffusion`` has an
            unsupported type.
        ValueError
            If the file is unreadable or is not a supported runtime-MGXS
            library; the material or temperature selection is invalid; the
            diffusion convention is unsupported; or required record data are
            missing, inconsistent, or outside their accepted ranges.

        Notes
        -----
        The conversion supports nonfissionable records and fissionable records
        with either separable vector ``nu-fission`` plus ``chi`` or general
        transfer matrix ``nu-fission`` production data. It requires
        finite nonnegative P0 scalar-flux isotropic scattering, and finite
        signed higher Legendre moments, in the runtime writer's
        ``[G][G'][Order]`` layout and expands its compact
        bands to Morana's complete incoming-to-outgoing
        ``sigma_s[g_from, g_to]`` matrix. A separate valid
        ``multiplicity_matrix`` is retained; its absence uses the compact
        unit-multiplicity representation. A present valid ``kappa-fission``
        vector is retained as
        ``FissionData.kappa_sigma_f`` in eV / cm.

        The importer warns when discarding higher scattering moments, known
        optional data outside Morana's steady diffusion scope, or unrecognized
        selected-temperature fields. See the OpenMC MGXS import guide for the
        data, unit, warning, and rejection contracts.

        The runtime format does not identify whether ``total`` came from
        OpenMC ``TotalXS`` or ``TransportXS``. Selecting ``"p1-outscatter"``
        for a ``TransportXS`` value would apply a second transport correction;
        the caller must avoid this invalid combination because Morana cannot
        detect it from the file.
        """
        imported = read_openmc_runtime_mgxs(path, dataset, temperature, diffusion)
        fission = None
        fission_import = imported.fission
        if fission_import is not None and fission_import.nu_sigma_f is not None:
            fission = FissionData(
                neutron_production=SeparableFission(
                    fission_import.nu_sigma_f, fission_import.chi
                ),
                kappa_sigma_f=fission_import.kappa_sigma_f,
            )
        elif fission_import is not None and fission_import.fission_transfer is not None:
            fission = FissionData(
                neutron_production=FissionTransfer(fission_import.fission_transfer),
                kappa_sigma_f=fission_import.kappa_sigma_f,
            )
        return cls(
            D=imported.coefficients,
            sigma_a=imported.absorption,
            sigma_s=imported.scatter,
            fission=fission,
            multiplicity_matrix=imported.multiplicity,
        )

    def __post_init__(
        self,
    ) -> None:
        """Convert arrays and check physical group data."""
        diffusion = float_array("D", self.D)
        absorption = float_array("sigma_a", self.sigma_a)
        if diffusion.ndim != 1 or absorption.ndim != 1:
            raise ValueError("D and sigma_a must be one-dimensional arrays")
        if diffusion.shape != absorption.shape:
            raise ValueError("D and sigma_a must have the same group count")
        if diffusion.size == 0:
            raise ValueError("at least one energy group is required")

        groups = diffusion.size
        scatter = _required_cross_section_array(
            "sigma_s", self.sigma_s, (groups, groups)
        )
        multiplicity = None
        if self.multiplicity_matrix is not None:
            multiplicity = _required_cross_section_array(
                "multiplicity_matrix", self.multiplicity_matrix, (groups, groups)
            )
        for name, array in (
            ("D", diffusion),
            ("sigma_a", absorption),
            ("sigma_s", scatter),
        ):
            require_finite_nonnegative_array(name, array)
        if multiplicity is not None:
            require_finite_nonnegative_array("multiplicity_matrix", multiplicity)
        if self.fission is not None:
            if not isinstance(self.fission, FissionData):
                raise TypeError("fission must be a FissionData or None")
            if self.fission.groups != groups:
                raise ValueError("fission must have the same group count as D")

        for array in (diffusion, absorption, scatter):
            array.setflags(write=False)
        if multiplicity is not None:
            multiplicity.setflags(write=False)

        object.__setattr__(self, "D", diffusion)
        object.__setattr__(self, "sigma_a", absorption)
        object.__setattr__(self, "sigma_s", scatter)
        object.__setattr__(self, "multiplicity_matrix", multiplicity)
        object.__setattr__(self, "groups", int(groups))


@dataclass(frozen=True)
class Material:
    """Represent a user-visible material identity.

    Parameters
    ----------
    name
        Public material name used by material meshes and source definitions.
        The built-in excluded-region key ``"0"`` is reserved and cannot be a
        material name. A ``ProblemConfiguration`` also rejects a material name
        that collides with any custom excluded-region key in its material mesh;
        its ``materials`` mapping key must exactly equal this value.
    xs
        Optional cross sections associated with this material. Cross sections
        may be omitted for layout-only or plotting use, but every active
        material requires them when a solver constructs its operators.
    color
        Optional nonempty plotting-color string. Its syntax is passed through
        without validation. Material-layout plots use it when this material is
        supplied in their ``materials`` mapping; otherwise they use their
        active-material palette.

    Raises
    ------
    ValueError
        If ``name`` is empty, whitespace-only, contains a non-printable
        character, or is the built-in excluded-region key ``"0"``.
    TypeError
        If ``name`` is not a string, ``xs`` is neither ``CrossSections`` nor
        ``None``, or ``color`` is neither a string nor ``None``.
    Notes
    -----
    ``Material`` is immutable. Construct and validate a replacement material
    before placing it in a ``ProblemConfiguration``.
    """

    name: str
    xs: CrossSections | None = None
    color: str | None = None

    def __post_init__(self) -> None:
        """Check material identity and optional display fields."""
        check_material_name(self.name)
        if self.xs is not None and not isinstance(self.xs, CrossSections):
            raise TypeError("xs must be a CrossSections or None")
        require_optional_string("material color", self.color)


def _required_cross_section_array(
    name: str,
    value: np.ndarray | list[float] | list[list[float]],
    shape: tuple[int, ...],
) -> np.ndarray:
    """Convert one required array field and check its exact shape."""
    array = float_array(name, value)
    if array.shape != shape:
        raise ValueError(f"expected shape {shape}, got {array.shape}")
    return array


def check_material_name(material: str) -> None:
    """Check a public material name."""
    require_human_readable_identifier("material name", material)
    if is_excluded_material_key(material):
        raise ValueError(
            f"material name {material!r} is an excluded-region key for "
            f"{excluded_key_kind(material)!r} excluded positions"
        )


def normalize_excluded_regions(
    excluded_regions: Mapping[str, ExcludedRegion] | None = None,
) -> dict[str, ExcludedRegion]:
    """Return built-in plus user-defined excluded regions keyed by identifier."""
    normalized = dict(BUILTIN_EXCLUDED_REGIONS)
    if excluded_regions is None:
        return normalized
    if not isinstance(excluded_regions, Mapping):
        raise TypeError("excluded_regions must be a mapping or None")
    for key, region in dict(excluded_regions).items():
        require_human_readable_identifier("material key", key)
        if key in BUILTIN_EXCLUDED_REGIONS:
            if region == BUILTIN_EXCLUDED_REGIONS[key]:
                continue
            raise ValueError(f"cannot redefine built-in excluded-region key {key!r}")
        if key in normalized:
            raise ValueError(
                "duplicate excluded-region key after normalization: " f"{key!r}"
            )
        if not isinstance(region, ExcludedRegion):
            raise TypeError("excluded-region values must be ExcludedRegion objects")
        normalized[key] = region
    return normalized


def excluded_key_kind(
    value: str,
    excluded_regions: Mapping[str, ExcludedRegion] | None = None,
) -> str | None:
    """Return the excluded-region kind for a material key, if any.

    Raises
    ------
    TypeError
        If ``value`` is not a string, ``excluded_regions`` is neither a
        mapping nor ``None``, or its selected value is not an
        ``ExcludedRegion``.
    ValueError
        If ``value`` is empty, whitespace-only, or contains a non-printable
        character.
    """
    regions = _excluded_region_catalog(excluded_regions)
    require_human_readable_identifier("material key", value)
    region = regions.get(value)
    if region is None:
        return None
    if not isinstance(region, ExcludedRegion):
        raise TypeError("excluded-region values must be ExcludedRegion objects")
    return region.kind


def excluded_key_color(
    value: str,
    excluded_regions: Mapping[str, ExcludedRegion] | None = None,
) -> str | None:
    """Return the plotting color for an excluded material key, if any.

    Raises
    ------
    TypeError
        If ``value`` is not a string, ``excluded_regions`` is neither a
        mapping nor ``None``, or its selected value is not an
        ``ExcludedRegion``.
    ValueError
        If ``value`` is empty, whitespace-only, or contains a non-printable
        character.
    """
    regions = _excluded_region_catalog(excluded_regions)
    require_human_readable_identifier("material key", value)
    region = regions.get(value)
    if region is None:
        return None
    if not isinstance(region, ExcludedRegion):
        raise TypeError("excluded-region values must be ExcludedRegion objects")
    return region.color or DEFAULT_EXCLUDED_REGION_COLOR


def is_excluded_material_key(
    value: str,
    excluded_regions: Mapping[str, ExcludedRegion] | None = None,
) -> bool:
    """Return whether a material key represents an excluded region."""
    return excluded_key_kind(value, excluded_regions) is not None


def is_active_material_key(
    value: str,
    excluded_regions: Mapping[str, ExcludedRegion] | None = None,
) -> bool:
    """Return whether a material key represents an active material."""
    return not is_excluded_material_key(value, excluded_regions)


def _excluded_region_catalog(
    excluded_regions: Mapping[str, ExcludedRegion] | None,
) -> Mapping[str, ExcludedRegion]:
    """Return one mapping-shaped excluded-region catalog for a key lookup."""
    if excluded_regions is None:
        return BUILTIN_EXCLUDED_REGIONS
    if not isinstance(excluded_regions, Mapping):
        raise TypeError("excluded_regions must be a mapping or None")
    return excluded_regions
