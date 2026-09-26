"""Frozen synthetic many-group workload for finite-volume performance studies.

This module owns the scientific workload definition only.  It deliberately
does not measure time or memory and does not import external data.  The
cross sections are deterministic synthetic values intended to exercise
Morana's many-group transfer assembly; they are not evaluated or
condensed nuclear data.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import ceil
from types import MappingProxyType

import numpy as np

from morana import (
    BoundaryCondition,
    BoundaryConditionSet,
    CrossSections,
    DirectLinearSolveSettings,
    FissionData,
    HexPlanarMesh,
    KeffSettings,
    Material,
    MaterialMesh,
    MaterialSlice,
    PowerIterationSettings,
    ProblemConfiguration,
    SeparableFission,
)

GROUP_COUNTS = (6, 18, 36, 72)
SCALING_AXIAL_LAYER_COUNTS = (6, 12, 24)
SMOKE_AXIAL_LAYER_COUNT = 2
NUM_PLANAR_RINGS = 5
LATTICE_PITCH_CM = 11.0 * 2.54
ACTIVE_HEIGHT_CM = 72.0 * 2.54
REFERENCE_MATERIAL = "reference"
HIGH_LEAKAGE_MATERIAL = "high_leakage"
HIGH_REACTIVITY_MATERIAL = "high_reactivity"
FUEL_REMOVAL_MATERIAL = "fuel_removal"
LOCALIZED_ABSORBER_MATERIAL = "localized_absorber"
SYNTHETIC_MATERIAL_NAMES = (
    REFERENCE_MATERIAL,
    HIGH_LEAKAGE_MATERIAL,
    HIGH_REACTIVITY_MATERIAL,
    FUEL_REMOVAL_MATERIAL,
    LOCALIZED_ABSORBER_MATERIAL,
)
STRUCTURED_PLACEMENT = "structured"
PERMUTED_PLACEMENT = "permuted"
PLACEMENTS = (STRUCTURED_PLACEMENT, PERMUTED_PLACEMENT)
PERMUTATION_ALGORITHM = "sha256-global-inventory-permutation-v1"
PERMUTATION_SEED = 0

CALIBRATION_SCALAR = 0.6004909682078956


@dataclass(frozen=True)
class _ScatteringKernel:
    """Smooth compact-transfer controls for one synthetic material role."""

    diagonal_weight: float
    downscatter_profile: tuple[float, float, float]
    upscatter_profile: tuple[float, float, float]
    directional_fractions: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class _FissionEmission:
    """Fast-emission shape controls for one synthetic material role."""

    center: float
    width: float


_REFERENCE_TRANSFER = _ScatteringKernel(
    diagonal_weight=1.0,
    downscatter_profile=(0.25, 0.08, 1.0),
    upscatter_profile=(0.05, 0.04, 0.05),
)
_BROAD_TRANSFER = _ScatteringKernel(
    diagonal_weight=4.0,
    downscatter_profile=(0.42, 0.12, 0.8),
    upscatter_profile=(0.25, 0.10, 0.35),
    directional_fractions=(0.30, 0.50, 0.20),
)
_WEAK_TRANSFER = _ScatteringKernel(
    diagonal_weight=10.0,
    downscatter_profile=(0.10, 0.05, 1.0),
    upscatter_profile=(0.04, 0.02, 0.015),
    directional_fractions=(0.875, 0.11, 0.015),
)

FROZEN_ARRAY_DIGESTS = MappingProxyType(
    {
        6: MappingProxyType(
            {
                "diffusion": (
                    "7e554d14b1faaee30233700a07665a145e8e65bc68fe543a71ad3b105c00fee0"
                ),
                "absorption": (
                    "9fb78dbe7839cbeece42a93b87d4e7a69f5d95a528003aefc6f5f090f7c8c991"
                ),
                "scattering": (
                    "7539dfb1eb81de713b0f83c7365d58b7c8eeb228a94f6187f191a16ccb5ccdd8"
                ),
                "fission_transfer": (
                    "297a6de80876c6d63994e95d17aca61895a249a05c5d91dca9e1a8967566bf25"
                ),
            }
        ),
        18: MappingProxyType(
            {
                "diffusion": (
                    "03cd7322947ee093b65c7f0c06cec388fb46c0c8ec1607f8c926c91980ac0129"
                ),
                "absorption": (
                    "970a42cb87f564280af24af881cb793b1190aa8a6d7e7ff710cb4588c1732e84"
                ),
                "scattering": (
                    "7f52b991317c172e476fc595a594af7cf2a9883caef299a13c3efc91508b20f3"
                ),
                "fission_transfer": (
                    "ceed187dc50d7e5dd856d3b0ec4ea4705fe40ff8ace1000c13965e91ebcf454f"
                ),
            }
        ),
        36: MappingProxyType(
            {
                "diffusion": (
                    "f490872b54e1c6709ca45db0f0241b257a9c994f62da2fc5e3e1e71282e38712"
                ),
                "absorption": (
                    "f0d624e4d6974dabda5a7b527a8fcd23c152707e74ea297bd8d89764026a6330"
                ),
                "scattering": (
                    "b913fe19f7da17754c9607725992fe7672c55b1d6eeda9e3e770b5f1ecb7f8c7"
                ),
                "fission_transfer": (
                    "9d97fd8cded237faf101e0218d141e753ab6faf085f2581a13448a40152781a3"
                ),
            }
        ),
        72: MappingProxyType(
            {
                "diffusion": (
                    "cb339ceaf3ec529aab2a4d6292cdc909d215a2778f68d8943c120bfabddfef08"
                ),
                "absorption": (
                    "8be238b1be457ca54d196e98fa37038bf3cac88dff873bae473819dbf435e1f7"
                ),
                "scattering": (
                    "c387d1268f152f66afd63bfc5e789982acadf704e95939fbca0024863da201fb"
                ),
                "fission_transfer": (
                    "c3bb5ae87ed506783e8085ad8ef7b77443504a97d834129174c6b8447a08f3ff"
                ),
            }
        ),
    }
)

FROZEN_MATERIAL_FAMILY_DIGESTS = MappingProxyType(
    {
        6: "bc189ad5594e0e45c003f38256279c8c281d60f429d7225ea449a8c72a510b5d",
        18: "be4c4b8ed013661b96a60074f95c5a1e4f2c8b136b3211f6644a4ee3db9269f5",
        36: "ebb905e41caadc5dae0c460670fe67217bf93405b1b9b4dd674292766a6d99e2",
        72: "e934ee202f1c83e848c3a905bd6b1162cae47ce8909e6a37cb8dba076ce9adfd",
    }
)
FROZEN_PLACEMENT_DIGESTS = MappingProxyType(
    {
        2: "0004af9104a8f039ab239d49391ff0b88ec6365ecbd5fc0cea3bc388b7767aa1",
        6: "701d6291526904e5cd709fde0e281769af84dd4cfd4cf869079610be13a10773",
        12: "e34fdaa24ec9821b938b03a25e121fb1d791e5242526327cb4f4bd6a79eccd55",
        24: "b209b42df18f86bae8f55bf7ccdcc1720c8ac2ff70c943f6745fc1465e0369bd",
    }
)
FROZEN_PERMUTED_PLACEMENT_DIGESTS = MappingProxyType(
    {
        2: "1ab5fda069fc8f8944f5a1af90b882a58c580e0970581b292e69b8ac16ca4886",
        6: "da07441dff019d1c9e7e34f05aab50cd687fd94049d6e286c499e6ba7233da9b",
        12: "fab3f781ae3f880ec9cdc3e5e891fcb3ea5709836280194becf5f5cae9d664d8",
        24: "dd314a4f62b2e2134b7fccc240c3768300e3920e1e16f2bd37652afb9d988894",
    }
)


def _require_group_count(groups: int) -> int:
    """Require one group count in the frozen study matrix."""
    if isinstance(groups, bool) or not isinstance(groups, int):
        raise TypeError("groups must be an integer")
    if groups not in GROUP_COUNTS:
        raise ValueError(f"groups must be one of {GROUP_COUNTS}")
    return groups


def _require_axial_layers(axial_layers: int) -> int:
    """Require one scaling-sweep or smoke axial-layer count."""
    if isinstance(axial_layers, bool) or not isinstance(axial_layers, int):
        raise TypeError("axial_layers must be an integer")
    supported = (SMOKE_AXIAL_LAYER_COUNT, *SCALING_AXIAL_LAYER_COUNTS)
    if axial_layers not in supported:
        raise ValueError(f"axial_layers must be one of {supported}")
    return axial_layers


def _require_placement(placement: str) -> str:
    """Require one maintained placement family."""
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}")
    return placement


def _material_modifiers(group_positions: np.ndarray, material_name: str) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    _ScatteringKernel,
    _FissionEmission,
]:
    """Return smooth synthetic material and transfer modifiers."""
    groups = len(group_positions)
    resonance_low = np.exp(-(((group_positions - 0.42) / 0.07) ** 2))
    resonance_high = np.exp(-(((group_positions - 0.63) / 0.08) ** 2))
    epithermal = np.exp(-(((group_positions - 0.43) / 0.29) ** 2))
    thermal = group_positions**1.4
    absorption_modifier = np.ones(groups, dtype=np.float64)
    diffusion_modifier = np.ones(groups, dtype=np.float64)
    scattering_row_modifier = np.ones(groups, dtype=np.float64)
    fission_modifier = np.ones(groups, dtype=np.float64)
    absorption_increment = np.zeros(groups, dtype=np.float64)
    scattering_kernel = _REFERENCE_TRANSFER
    fission_emission = _FissionEmission(center=0.12, width=0.14)
    if material_name == HIGH_LEAKAGE_MATERIAL:
        diffusion_modifier += 0.30 + 0.15 * (1.0 - group_positions)
        fission_emission = _FissionEmission(center=0.19, width=0.12)
    elif material_name == HIGH_REACTIVITY_MATERIAL:
        absorption_modifier += 0.11 + 0.09 * epithermal + 0.05 * thermal
        fission_modifier += 0.20 + 0.38 * epithermal + 0.14 * thermal
        diffusion_modifier -= 0.20 - 0.08 * group_positions
        scattering_row_modifier += 0.15 + 0.07 * thermal
        scattering_kernel = _BROAD_TRANSFER
        fission_emission = _FissionEmission(center=0.16, width=0.16)
    elif material_name == FUEL_REMOVAL_MATERIAL:
        absorption_modifier -= 0.05 + 0.11 * (1.0 - thermal)
        fission_modifier -= 0.22 + 0.20 * epithermal + 0.12 * (1.0 - thermal)
        diffusion_modifier += 0.16 + 0.22 * (1.0 - group_positions)
        scattering_row_modifier -= 0.22 + 0.08 * (1.0 - thermal)
        scattering_kernel = _WEAK_TRANSFER
        fission_emission = _FissionEmission(center=0.075, width=0.09)
    elif material_name == LOCALIZED_ABSORBER_MATERIAL:
        absorption_modifier -= 0.03 + 0.08 * (1.0 - thermal)
        absorption_increment = 0.008 * resonance_low + 0.005 * resonance_high
        fission_modifier -= 0.10 + 0.08 * epithermal + 0.05 * thermal
        diffusion_modifier -= 0.12 * resonance_low + 0.08 * resonance_high
        scattering_row_modifier += 0.003 * resonance_low
        fission_emission = _FissionEmission(center=0.11, width=0.11)
    return (
        absorption_modifier,
        absorption_increment,
        diffusion_modifier,
        scattering_row_modifier,
        fission_modifier,
        scattering_kernel,
        fission_emission,
    )


def _scattering_row(
    groups: int, incident_group: int, kernel: _ScatteringKernel
) -> np.ndarray:
    """Construct one normalized smooth transfer row for a synthetic role."""
    row = np.zeros(groups, dtype=np.float64)
    row[incident_group] = kernel.diagonal_weight
    downscatter_width, downscatter_scale, downscatter_weight = (
        kernel.downscatter_profile
    )
    downscatter_indices = tuple(
        outgoing_group
        for offset in range(1, ceil(groups * downscatter_width) + 1)
        if (outgoing_group := incident_group + offset) < groups
    )
    for outgoing_group in downscatter_indices:
        row[outgoing_group] = downscatter_weight * np.exp(
            -(outgoing_group - incident_group) / max(1.0, groups * downscatter_scale)
        )
    upscatter_width, upscatter_scale, upscatter_weight = kernel.upscatter_profile
    upscatter_indices = tuple(
        outgoing_group
        for offset in range(1, ceil(groups * upscatter_width) + 1)
        if (outgoing_group := incident_group - offset) >= 0
    )
    for outgoing_group in upscatter_indices:
        row[outgoing_group] = upscatter_weight * np.exp(
            -(incident_group - outgoing_group) / max(1.0, groups * upscatter_scale)
        )
    if kernel.directional_fractions is None:
        return row / row.sum()
    diagonal_fraction, downscatter_fraction, upscatter_fraction = (
        kernel.directional_fractions
    )
    row[incident_group] = diagonal_fraction
    if downscatter_indices:
        downscatter_values = row[list(downscatter_indices)]
        row[list(downscatter_indices)] = (
            downscatter_fraction * downscatter_values / downscatter_values.sum()
        )
    else:
        row[incident_group] += downscatter_fraction
    if upscatter_indices:
        upscatter_values = row[list(upscatter_indices)]
        row[list(upscatter_indices)] = (
            upscatter_fraction * upscatter_values / upscatter_values.sum()
        )
    else:
        row[incident_group] += upscatter_fraction
    return row


def _synthetic_arrays(groups: int, material_name: str) -> dict[str, np.ndarray]:
    """Construct one role-based synthetic material without external data.

    The normalized coordinate runs fast to thermal.  The material labels name
    workload roles, not substances: the formulas are deliberately synthetic
    smooth perturbations selected to exercise heterogeneous loss, transfer,
    and fission blocks.  They are not evaluated or condensed cross sections.
    """
    groups = _require_group_count(groups)
    if material_name not in SYNTHETIC_MATERIAL_NAMES:
        raise ValueError(f"material_name must be one of {SYNTHETIC_MATERIAL_NAMES}")
    group_positions = (np.arange(groups, dtype=np.float64) + 0.5) / groups

    diffusion = 1.4 - 0.6 * group_positions
    absorption = 5.0e-4 + 9.5e-3 * group_positions**3

    (
        absorption_modifier,
        absorption_increment,
        diffusion_modifier,
        scattering_row_modifier,
        fission_modifier,
        scattering_kernel,
        fission_emission,
    ) = _material_modifiers(group_positions, material_name)

    diffusion *= diffusion_modifier
    absorption = absorption * absorption_modifier + absorption_increment

    scattering = np.zeros((groups, groups), dtype=np.float64)
    for incident_group, position in enumerate(group_positions):
        scattering[incident_group] = (
            _scattering_row(groups, incident_group, scattering_kernel)
            * (0.08 + 0.04 * position)
            * scattering_row_modifier[incident_group]
        )

    fission_production = 2.0e-4 + 1.98e-2 * group_positions**3
    fission_production *= CALIBRATION_SCALAR * 5.15e-3 / fission_production.mean()
    fission_production *= fission_modifier
    emission_groups = ceil(0.30 * groups)
    outgoing_positions = group_positions[:emission_groups]
    emission_weights = np.exp(
        -(
            ((outgoing_positions - fission_emission.center) / fission_emission.width)
            ** 2
        )
    )
    emission = np.zeros(groups, dtype=np.float64)
    emission[:emission_groups] = emission_weights / emission_weights.sum()

    return {
        "diffusion": diffusion,
        "absorption": absorption,
        "scattering": scattering,
        "fission_production": fission_production,
        "fission_emission": emission,
    }


def build_cross_sections(groups: int, material_name: str) -> CrossSections:
    """Return one checked, role-named synthetic cross-section set.

    Parameters are deliberately limited to the frozen study group counts and
    material roles.  This is a performance-workload generator, not a material
    model or a pathway for importing transport data.
    """
    arrays = _synthetic_arrays(groups, material_name)
    return CrossSections(
        D=arrays["diffusion"],
        sigma_a=arrays["absorption"],
        sigma_s=arrays["scattering"],
        multiplicity_matrix=None,
        fission=FissionData(
            neutron_production=SeparableFission(
                arrays["fission_production"], arrays["fission_emission"]
            )
        ),
    )


def _layer_roles(layer_index: int) -> tuple[str, ...]:
    """Return the outer-to-inner synthetic role pattern for one axial layer."""
    patterns = (
        (
            HIGH_LEAKAGE_MATERIAL,
            REFERENCE_MATERIAL,
            HIGH_REACTIVITY_MATERIAL,
            REFERENCE_MATERIAL,
            LOCALIZED_ABSORBER_MATERIAL,
        ),
        (
            HIGH_REACTIVITY_MATERIAL,
            REFERENCE_MATERIAL,
            HIGH_LEAKAGE_MATERIAL,
            FUEL_REMOVAL_MATERIAL,
            REFERENCE_MATERIAL,
        ),
    )
    return patterns[layer_index % len(patterns)]


def _permutation_key(axial_index: int, ring: int, position: int) -> bytes:
    """Return the versioned seed-0 ordering key for one spatial cell."""
    identity = (
        f"{PERMUTATION_ALGORITHM}\0{PERMUTATION_SEED}\0{axial_index}\0"
        f"{ring}\0{position}"
    )
    return sha256(identity.encode("ascii")).digest()


def _permuted_layers(baseline: ProblemConfiguration) -> tuple[MaterialSlice, ...]:
    """Permute the complete structured inventory over all spatial cells."""
    mesh = baseline.mesh
    material_mesh = baseline.material_mesh
    positions = tuple(
        (axial_index, openmc_index)
        for axial_index in range(material_mesh.n_axial_layers)
        for openmc_index in mesh.openmc_indices
    )
    roles = sorted(
        material_mesh.key_at(axial_index, openmc_index)
        for axial_index, openmc_index in positions
    )
    ordered_positions = sorted(
        positions,
        key=lambda item: (
            _permutation_key(item[0], item[1].ring, item[1].position),
            item[0],
            item[1].ring,
            item[1].position,
        ),
    )
    assignments = dict(zip(ordered_positions, roles, strict=True))
    return tuple(
        MaterialSlice(
            mesh,
            {
                openmc_index: assignments[(axial_index, openmc_index)]
                for openmc_index in mesh.openmc_indices
            },
            material_mesh.axial_layer_heights[axial_index],
        )
        for axial_index in range(material_mesh.n_axial_layers)
    )


def build_configuration(
    groups: int,
    axial_layers: int,
    *,
    placement: str,
) -> ProblemConfiguration:
    """Build one heterogeneous, role-named synthetic criticality problem."""
    groups = _require_group_count(groups)
    axial_layers = _require_axial_layers(axial_layers)
    placement = _require_placement(placement)
    mesh = HexPlanarMesh(num_rings=NUM_PLANAR_RINGS, pitch=LATTICE_PITCH_CM)
    materials = {
        material_name: Material(
            material_name, xs=build_cross_sections(groups, material_name)
        )
        for material_name in SYNTHETIC_MATERIAL_NAMES
    }
    layers = tuple(
        MaterialSlice.from_openmc_rings(
            mesh,
            [
                [material_name] * max(6 * (mesh.num_rings - 1 - ring), 1)
                for ring, material_name in enumerate(_layer_roles(layer_index))
            ],
            height=ACTIVE_HEIGHT_CM / axial_layers,
        )
        for layer_index in range(axial_layers)
    )
    structured = ProblemConfiguration(
        mesh=mesh,
        materials=materials,
        material_mesh=MaterialMesh.stack(layers),
        boundary=BoundaryConditionSet(
            BoundaryCondition.vacuum().on_radial(),
            BoundaryCondition.vacuum().on_bottom(),
            BoundaryCondition.vacuum().on_top(),
        ),
        name=f"synthetic_heterogeneous_many_group_g{groups}_z{axial_layers}",
    )
    if placement == STRUCTURED_PLACEMENT:
        return structured
    return ProblemConfiguration(
        mesh=mesh,
        materials=materials,
        material_mesh=MaterialMesh.stack(_permuted_layers(structured)),
        boundary=structured.boundary,
        name=f"synthetic_permuted_many_group_g{groups}_z{axial_layers}_seed0",
    )


def solve_settings() -> KeffSettings:
    """Return the immutable numerical settings shared by every workload."""
    return KeffSettings(
        inner_linear_solve=DirectLinearSolveSettings(
            relative_residual_tolerance=1.0e-10
        ),
        max_outer_iterations=500,
        keff_change_tolerance=1.0e-10,
        flux_change_tolerance=1.0e-10,
        keff_relative_residual_tolerance=1.0e-10,
        flux_nonnegativity_tolerance=1.0e-12,
        eigenvalue_iteration=PowerIterationSettings(),
    )


def scaling_workloads() -> tuple[tuple[int, int], ...]:
    """Return the complete group-major Cartesian scaling matrix."""
    return tuple(
        (groups, axial_layers)
        for groups in GROUP_COUNTS
        for axial_layers in SCALING_AXIAL_LAYER_COUNTS
    )


def array_digest(array: np.ndarray) -> str:
    """Return a shape-, type-, and byte-sensitive SHA-256 array digest."""
    contiguous = np.ascontiguousarray(array, dtype="<f8")
    digest = sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(size) for size in contiguous.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def generated_array_digests(groups: int, material_name: str) -> dict[str, str]:
    """Return deterministic digests for one checked synthetic material."""
    cross_sections = build_cross_sections(groups, material_name)
    if cross_sections.fission is None:  # pragma: no cover - construction invariant
        raise RuntimeError("synthetic cross sections unexpectedly lack fission data")
    return {
        "diffusion": array_digest(cross_sections.D),
        "absorption": array_digest(cross_sections.sigma_a),
        "scattering": array_digest(cross_sections.sigma_s),
        "fission_transfer": array_digest(cross_sections.fission.fission_transfer),
    }


def generated_material_family_digest(groups: int) -> str:
    """Return one deterministic digest covering every synthetic material array."""
    digest = sha256()
    for material_name in SYNTHETIC_MATERIAL_NAMES:
        digest.update(material_name.encode("ascii"))
        for component, value in generated_array_digests(groups, material_name).items():
            digest.update(component.encode("ascii"))
            digest.update(value.encode("ascii"))
    return digest.hexdigest()


def generated_placement_digest(axial_layers: int, placement: str) -> str:
    """Return one deterministic digest of the role placement at every cell."""
    configuration = build_configuration(6, axial_layers, placement=placement)
    digest = sha256()
    for axial_index in range(axial_layers):
        for openmc_index in configuration.mesh.openmc_indices:
            digest.update(
                configuration.material_mesh.key_at(axial_index, openmc_index).encode(
                    "ascii"
                )
            )
            digest.update(b"\0")
    return digest.hexdigest()


def placement_record(placement: str) -> dict[str, object]:
    """Return concise reproducibility provenance for one placement family."""
    placement = _require_placement(placement)
    if placement == STRUCTURED_PLACEMENT:
        return {"kind": placement, "algorithm": None, "seed": None}
    return {
        "kind": placement,
        "algorithm": PERMUTATION_ALGORITHM,
        "seed": PERMUTATION_SEED,
    }
