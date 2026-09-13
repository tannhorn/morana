"""Frozen synthetic many-group workload for finite-volume performance studies.

This module owns the scientific workload definition only.  It deliberately
does not measure time or memory and does not import external data.  The
cross sections are deterministic synthetic values intended to exercise
Morana's many-group transfer assembly; they are not evaluated nuclear data or
a reactor-design model.
"""

from __future__ import annotations

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
    FissionTransfer,
    HexPlanarMesh,
    KeffSettings,
    Material,
    MaterialMesh,
    MaterialSlice,
    PowerIterationSettings,
    ProblemConfiguration,
)

GROUP_COUNTS = (6, 18, 36, 72)
BASELINE_AXIAL_LAYER_COUNTS = (5, 10, 20)
SMOKE_AXIAL_LAYER_COUNT = 2
NUM_PLANAR_RINGS = 5
LATTICE_PITCH_CM = 11.0 * 2.54
ACTIVE_HEIGHT_CM = 72.0 * 2.54
_MATERIAL_NAME = "synthetic_many_group_material"

CALIBRATION_SCALAR = 0.6020060777418164

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
                    "103bf9fef1a8cbd0706c1a0b3e5303d8deedb059a1b6346fb5e9da003ba88aad"
                ),
                "fission_transfer": (
                    "b34b4e218f1f41b8d5f5bb3f240a65d55b6ce4cabe6074c3388f21a755f3b467"
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
                    "65d86a34ec58f5100fc1c8aa773ab0a23e8ad7a58054cb65a7c812cec9bec95e"
                ),
                "fission_transfer": (
                    "5b0ecc2b0141db1f3320360e5bc35e1f6dd2388a2e578a37427e079ca632539d"
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
                    "fd78b58d32fdb54a00904d540343b7457566dff6d75080c4585ba528c16531dd"
                ),
                "fission_transfer": (
                    "a5b9fc2e6badfd5a5588aaeb7a5221c3ddb22f0d509b4e7c88006f9806787794"
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
                    "e7dfd3b8d53cb97af456b38c43ae41275cbb77cde72d2ec9c3f322202ae03df2"
                ),
                "fission_transfer": (
                    "e2db2b66fcfba820a85875b37dd6720a9cba6a0016bd89cf41bd308368ab7246"
                ),
            }
        ),
    }
)


def _require_group_count(groups: int) -> int:
    """Require one group count in the frozen benchmark matrix."""
    if isinstance(groups, bool) or not isinstance(groups, int):
        raise TypeError("groups must be an integer")
    if groups not in GROUP_COUNTS:
        raise ValueError(f"groups must be one of {GROUP_COUNTS}")
    return groups


def _require_axial_layers(axial_layers: int) -> int:
    """Require one baseline or smoke axial-layer count."""
    if isinstance(axial_layers, bool) or not isinstance(axial_layers, int):
        raise TypeError("axial_layers must be an integer")
    supported = (SMOKE_AXIAL_LAYER_COUNT, *BASELINE_AXIAL_LAYER_COUNTS)
    if axial_layers not in supported:
        raise ValueError(f"axial_layers must be one of {supported}")
    return axial_layers


def _synthetic_arrays(
    groups: int,
    *,
    fission_scale: float,
) -> dict[str, np.ndarray]:
    """Construct the analytic synthetic arrays for one group count."""
    groups = _require_group_count(groups)
    group_positions = (np.arange(groups, dtype=np.float64) + 0.5) / groups

    diffusion = 1.4 - 0.6 * group_positions
    absorption = 5.0e-4 + 9.5e-3 * group_positions**3

    scattering = np.zeros((groups, groups), dtype=np.float64)
    downscatter_width = ceil(groups / 4)
    upscatter_width = ceil(groups / 20)
    downscatter_scale = max(1.0, 0.08 * groups)
    upscatter_scale = max(1.0, 0.04 * groups)
    for incident_group, position in enumerate(group_positions):
        scattering[incident_group, incident_group] = 1.0
        for offset in range(1, downscatter_width + 1):
            outgoing_group = incident_group + offset
            if outgoing_group < groups:
                scattering[incident_group, outgoing_group] = np.exp(
                    -offset / downscatter_scale
                )
        for offset in range(1, upscatter_width + 1):
            outgoing_group = incident_group - offset
            if outgoing_group >= 0:
                scattering[incident_group, outgoing_group] = 0.05 * np.exp(
                    -offset / upscatter_scale
                )
        scattering[incident_group] *= (0.08 + 0.04 * position) / scattering[
            incident_group
        ].sum()

    fission_production = 2.0e-4 + 1.98e-2 * group_positions**3
    fission_production *= 5.15e-3 / fission_production.mean()
    emission_groups = ceil(0.30 * groups)
    fission_transfer = np.zeros((groups, groups), dtype=np.float64)
    outgoing_positions = group_positions[:emission_groups]
    for incident_group, position in enumerate(group_positions):
        weights = np.exp(
            -(((outgoing_positions - (0.08 + 0.08 * position)) / 0.14) ** 2)
        )
        fission_transfer[incident_group, :emission_groups] = (
            fission_scale * fission_production[incident_group] * weights / weights.sum()
        )

    return {
        "diffusion": diffusion,
        "absorption": absorption,
        "scattering": scattering,
        "fission_transfer": fission_transfer,
    }


def build_cross_sections(groups: int) -> CrossSections:
    """Return checked synthetic cross sections for a frozen group count."""
    arrays = _synthetic_arrays(groups, fission_scale=CALIBRATION_SCALAR)
    return CrossSections(
        D=arrays["diffusion"],
        sigma_a=arrays["absorption"],
        sigma_s=arrays["scattering"],
        multiplicity_matrix=None,
        fission=FissionData(
            neutron_production=FissionTransfer(arrays["fission_transfer"])
        ),
    )


def build_configuration(groups: int, axial_layers: int) -> ProblemConfiguration:
    """Build one homogeneous 61-cell layered synthetic criticality problem."""
    groups = _require_group_count(groups)
    axial_layers = _require_axial_layers(axial_layers)
    mesh = HexPlanarMesh(num_rings=NUM_PLANAR_RINGS, pitch=LATTICE_PITCH_CM)
    material = Material(_MATERIAL_NAME, xs=build_cross_sections(groups))
    layer = MaterialSlice(
        mesh,
        {index: _MATERIAL_NAME for index in mesh.openmc_indices},
        ACTIVE_HEIGHT_CM / axial_layers,
    )
    return ProblemConfiguration(
        mesh=mesh,
        materials={_MATERIAL_NAME: material},
        material_mesh=MaterialMesh.stack(layer.extrude(count=axial_layers)),
        boundary=BoundaryConditionSet(
            BoundaryCondition.vacuum().on_radial(),
            BoundaryCondition.vacuum().on_bottom(),
            BoundaryCondition.vacuum().on_top(),
        ),
        name=f"synthetic_many_group_g{groups}_z{axial_layers}",
    )


def solve_settings() -> KeffSettings:
    """Return the immutable numerical settings shared by every workload."""
    return KeffSettings(
        inner_linear_solve=DirectLinearSolveSettings(
            relative_residual_tolerance=1.0e-10
        ),
        max_outer_iterations=200,
        keff_change_tolerance=1.0e-10,
        flux_change_tolerance=1.0e-10,
        keff_relative_residual_tolerance=1.0e-10,
        flux_nonnegativity_tolerance=1.0e-12,
        eigenvalue_iteration=PowerIterationSettings(),
    )


def baseline_workloads() -> tuple[tuple[int, int], ...]:
    """Return the complete group-major Cartesian baseline matrix."""
    return tuple(
        (groups, axial_layers)
        for groups in GROUP_COUNTS
        for axial_layers in BASELINE_AXIAL_LAYER_COUNTS
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


def generated_array_digests(groups: int) -> dict[str, str]:
    """Return deterministic digests for the final checked workload arrays."""
    cross_sections = build_cross_sections(groups)
    if cross_sections.fission is None:  # pragma: no cover - construction invariant
        raise RuntimeError("synthetic cross sections unexpectedly lack fission data")
    return {
        "diffusion": array_digest(cross_sections.D),
        "absorption": array_digest(cross_sections.sigma_a),
        "scattering": array_digest(cross_sections.sigma_s),
        "fission_transfer": array_digest(cross_sections.fission.fission_transfer),
    }
