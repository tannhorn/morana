"""Tests for the frozen synthetic many-group performance workload."""

from __future__ import annotations

from collections import Counter
from math import ceil, exp

import numpy as np
import pytest

from studies.finite_volume_performance import workload
from morana import (
    DirectLinearSolveSettings,
    FissionSourceNormalization,
    PowerIterationSettings,
    SeparableFission,
)
from morana.solvers.finite_volume import solve_keff


def test_workload_matrix_has_expected_dimensions() -> None:
    """The group-major Cartesian matrix retains all twelve scaling cases."""
    assert workload.scaling_workloads() == (
        (6, 6),
        (6, 12),
        (6, 24),
        (18, 6),
        (18, 12),
        (18, 24),
        (36, 6),
        (36, 12),
        (36, 24),
        (72, 6),
        (72, 12),
        (72, 24),
    )


def test_frozen_calibration_and_digest_coverage() -> None:
    """The calibration scalar, material arrays, and placement remain frozen."""
    assert workload.CALIBRATION_SCALAR == 0.6004909682078956
    assert set(workload.FROZEN_ARRAY_DIGESTS) == set(workload.GROUP_COUNTS)
    assert set(workload.FROZEN_MATERIAL_FAMILY_DIGESTS) == set(workload.GROUP_COUNTS)
    assert set(workload.FROZEN_PLACEMENT_DIGESTS) == {2, 6, 12, 24}
    assert set(workload.FROZEN_PERMUTED_PLACEMENT_DIGESTS) == {2, 6, 12, 24}


@pytest.mark.parametrize("groups", workload.GROUP_COUNTS)
def test_analytic_vectors_and_integrated_strengths(groups: int) -> None:
    """Every group count follows the frozen analytic shape and strength rules."""
    cross_sections = workload.build_cross_sections(groups, workload.REFERENCE_MATERIAL)
    positions = (np.arange(groups, dtype=np.float64) + 0.5) / groups

    np.testing.assert_array_equal(
        cross_sections.D,
        1.4 - 0.6 * positions,
    )
    np.testing.assert_array_equal(
        cross_sections.sigma_a,
        5.0e-4 + 9.5e-3 * positions**3,
    )
    np.testing.assert_allclose(
        cross_sections.sigma_s.sum(axis=1),
        0.08 + 0.04 * positions,
        rtol=2.0e-16,
        atol=2.0e-17,
    )
    assert cross_sections.multiplicity_matrix is None
    assert cross_sections.fission is not None
    assert isinstance(cross_sections.fission.neutron_production, SeparableFission)
    np.testing.assert_allclose(
        cross_sections.fission.fission_production.mean(),
        workload.CALIBRATION_SCALAR * 5.15e-3,
        rtol=5.0e-16,
    )


@pytest.mark.parametrize("groups", workload.GROUP_COUNTS)
def test_role_named_synthetic_materials_have_bounded_spectral_contrasts(
    groups: int,
) -> None:
    """Role labels select deterministic synthetic, rather than physical, data."""
    reference = workload.build_cross_sections(groups, workload.REFERENCE_MATERIAL)
    leakage = workload.build_cross_sections(groups, workload.HIGH_LEAKAGE_MATERIAL)
    high = workload.build_cross_sections(groups, workload.HIGH_REACTIVITY_MATERIAL)
    removal = workload.build_cross_sections(groups, workload.FUEL_REMOVAL_MATERIAL)
    absorber = workload.build_cross_sections(
        groups, workload.LOCALIZED_ABSORBER_MATERIAL
    )

    assert reference.fission is not None
    assert leakage.fission is not None
    assert high.fission is not None
    assert removal.fission is not None
    assert absorber.fission is not None
    np.testing.assert_allclose(
        leakage.fission.fission_production,
        reference.fission.fission_production,
        rtol=5.0e-16,
        atol=0.0,
    )
    assert np.all(
        high.fission.fission_production > reference.fission.fission_production
    )
    assert np.all(
        removal.fission.fission_production < reference.fission.fission_production
    )
    assert np.all(
        absorber.fission.fission_production < reference.fission.fission_production
    )
    assert absorber.sigma_a[np.argmax(absorber.sigma_a / reference.sigma_a)] > (
        reference.sigma_a[np.argmax(absorber.sigma_a / reference.sigma_a)]
    )
    assert np.all(high.D < reference.D)
    assert np.all(removal.D > reference.D)
    assert np.all(leakage.D > removal.D)
    assert np.min(absorber.D / reference.D) < 0.89
    fission_spectra = []
    for cross_sections in (reference, leakage, high, removal, absorber):
        assert cross_sections.fission is not None
        fission_spectra.append(cross_sections.fission.neutron_production.chi)
    assert len({workload.array_digest(spectrum) for spectrum in fission_spectra}) == 5


@pytest.mark.parametrize("groups", workload.GROUP_COUNTS)
def test_transfer_roles_span_weak_and_broad_synthetic_coupling(groups: int) -> None:
    """Fuel removal and high reactivity retain distinct transfer topologies."""
    reference = workload.build_cross_sections(
        groups, workload.REFERENCE_MATERIAL
    ).sigma_s
    broad = workload.build_cross_sections(
        groups, workload.HIGH_REACTIVITY_MATERIAL
    ).sigma_s
    weak = workload.build_cross_sections(groups, workload.FUEL_REMOVAL_MATERIAL).sigma_s

    def direction_shares(matrix: np.ndarray) -> tuple[float, float, float]:
        total = float(matrix.sum())
        return (
            float(np.triu(matrix, 1).sum() / total),
            float(np.trace(matrix) / total),
            float(np.tril(matrix, -1).sum() / total),
        )

    _, _, reference_up = direction_shares(reference)
    broad_down, broad_diagonal, broad_up = direction_shares(broad)
    weak_down, weak_diagonal, weak_up = direction_shares(weak)

    assert broad_down > 0.39
    assert broad_down > weak_down
    assert broad_diagonal < 0.43
    assert broad_up > 10.0 * reference_up
    assert weak_down < 0.11
    assert weak_diagonal > 0.87
    assert weak_up < 0.016


def test_scattering_support_weights_and_density_at_72_groups() -> None:
    """The 72-group scattering matrix has the exact specified sparse support."""
    groups = 72
    matrix = workload.build_cross_sections(groups, workload.REFERENCE_MATERIAL).sigma_s
    assert np.count_nonzero(matrix) == 1_475
    density = np.count_nonzero(matrix) / matrix.size
    assert density == pytest.approx(0.284_529_320_987_654_3)

    incident_group = 30
    downscatter_scale = max(1.0, 0.08 * groups)
    upscatter_scale = max(1.0, 0.04 * groups)
    assert matrix[incident_group, incident_group + 2] / matrix[
        incident_group, incident_group
    ] == pytest.approx(exp(-2 / downscatter_scale))
    assert matrix[incident_group, incident_group - 2] / matrix[
        incident_group, incident_group
    ] == pytest.approx(0.05 * exp(-2 / upscatter_scale))
    assert matrix[incident_group, incident_group + ceil(groups / 4) + 1] == 0.0
    assert matrix[incident_group, incident_group - ceil(groups / 20) - 1] == 0.0


def test_fission_transfer_support_shape_and_density_at_72_groups() -> None:
    """Each synthetic role retains a broad, normalized fast emission band."""
    groups = 72
    cross_sections = workload.build_cross_sections(groups, workload.REFERENCE_MATERIAL)
    assert cross_sections.fission is not None
    matrix = cross_sections.fission.fission_transfer
    emission_groups = ceil(0.30 * groups)
    assert np.count_nonzero(matrix) == 1_584
    density = np.count_nonzero(matrix) / matrix.size
    assert density == pytest.approx(0.305_555_555_555_555_6)
    assert np.all(matrix[:, :emission_groups] > 0.0)
    assert np.all(matrix[:, emission_groups:] == 0.0)
    first_spectrum = matrix[0] / matrix[0].sum()
    last_spectrum = matrix[-1] / matrix[-1].sum()
    np.testing.assert_allclose(first_spectrum, last_spectrum, rtol=5.0e-16)


@pytest.mark.parametrize("groups", workload.GROUP_COUNTS)
def test_frozen_array_digests(groups: int) -> None:
    """Reference arrays and the full role-named family remain frozen."""
    assert (
        workload.generated_array_digests(groups, workload.REFERENCE_MATERIAL)
        == workload.FROZEN_ARRAY_DIGESTS[groups]
    )
    assert (
        workload.generated_material_family_digest(groups)
        == workload.FROZEN_MATERIAL_FAMILY_DIGESTS[groups]
    )


@pytest.mark.parametrize("axial_layers", (2, 6, 12, 24))
def test_configuration_is_complete_heterogeneous_geometry(axial_layers: int) -> None:
    """The builder creates a complete, role-named vacuum mini-core."""
    configuration = workload.build_configuration(
        6, axial_layers, placement=workload.STRUCTURED_PLACEMENT
    )
    assert configuration.mesh.num_rings == 5
    assert configuration.mesh.n_cells == 61
    assert configuration.mesh.pitch == pytest.approx(27.94)
    assert configuration.material_mesh.n_axial_layers == axial_layers
    assert configuration.material_mesh.z_max == pytest.approx(182.88)
    assert all(
        configuration.material_mesh.n_active_cells(index) == 61
        for index in range(axial_layers)
    )
    assert set(configuration.materials) == set(workload.SYNTHETIC_MATERIAL_NAMES)
    assert (
        workload.generated_placement_digest(axial_layers, workload.STRUCTURED_PLACEMENT)
        == workload.FROZEN_PLACEMENT_DIGESTS[axial_layers]
    )
    assert configuration.source is None
    configuration.check_boundary_coverage()
    configuration.check_no_unused_materials()


@pytest.mark.parametrize("axial_layers", (2, 6, 12, 24))
def test_canonical_permutation_is_frozen_and_preserves_inventory(
    axial_layers: int,
) -> None:
    """The maintained permutation changes adjacency without changing roles."""
    structured = workload.build_configuration(
        6, axial_layers, placement=workload.STRUCTURED_PLACEMENT
    )
    permuted = workload.build_configuration(
        6, axial_layers, placement=workload.PERMUTED_PLACEMENT
    )

    def roles(configuration):
        return tuple(
            configuration.material_mesh.key_at(layer, index)
            for layer in range(axial_layers)
            for index in configuration.mesh.openmc_indices
        )

    def adjacency(configuration, directions):
        pairs = []
        material_mesh = configuration.material_mesh
        for layer in range(axial_layers):
            for active_id in range(material_mesh.n_active_cells(layer)):
                here = material_mesh.material_by_active_id(layer)[active_id]
                for direction in directions:
                    face = material_mesh.face(layer, active_id, direction)
                    if face.kind == "internal":
                        neighbor = material_mesh.material_by_active_id(
                            face.neighbor_axial_index
                        )[face.neighbor_active_id]
                        pairs.append(tuple(sorted((here, neighbor))))
        return Counter(pairs)

    assert Counter(roles(permuted)) == Counter(roles(structured))
    assert roles(permuted) != roles(structured)
    assert adjacency(permuted, ("x+", "u+", "v+")) != adjacency(
        structured, ("x+", "u+", "v+")
    )
    assert adjacency(permuted, ("top",)) != adjacency(structured, ("top",))
    assert (
        workload.generated_placement_digest(axial_layers, workload.PERMUTED_PLACEMENT)
        == workload.FROZEN_PERMUTED_PLACEMENT_DIGESTS[axial_layers]
    )


def test_workload_settings_are_exact_direct_power_controls() -> None:
    """The builder returns the fixed direct-power settings without fallback."""
    settings = workload.solve_settings()
    assert settings.max_outer_iterations == 500
    assert settings.keff_change_tolerance == 1.0e-10
    assert settings.flux_change_tolerance == 1.0e-10
    assert settings.keff_relative_residual_tolerance == 1.0e-10
    assert settings.flux_nonnegativity_tolerance == 1.0e-12
    assert settings.inner_linear_solve == DirectLinearSolveSettings(
        relative_residual_tolerance=1.0e-10
    )
    assert settings.eigenvalue_iteration == PowerIterationSettings()


def test_smoke_workload_solves_with_frozen_numerical_controls() -> None:
    """The inexpensive workload remains a numerically checked solve path."""
    result = solve_keff(
        workload.build_configuration(
            6,
            workload.SMOKE_AXIAL_LAYER_COUNT,
            placement=workload.STRUCTURED_PLACEMENT,
        ),
        FissionSourceNormalization(rate=1.0),
        workload.solve_settings(),
    )
    assert result.keff == pytest.approx(1.039_943_512_290_362, rel=1.0e-12)
    assert (
        result.execution_report.final_outer_iteration.keff_relative_residual
        <= result.solve_settings.keff_relative_residual_tolerance
    )
    assert abs(result.balance["residual"]) <= 1.0e-12


@pytest.mark.parametrize(
    ("builder", "value", "error"),
    (
        (
            lambda value: workload.build_cross_sections(
                value, workload.REFERENCE_MATERIAL
            ),
            True,
            TypeError,
        ),
        (
            lambda value: workload.build_cross_sections(
                value, workload.REFERENCE_MATERIAL
            ),
            12,
            ValueError,
        ),
        (
            lambda value: workload.build_configuration(
                6, value, placement=workload.STRUCTURED_PLACEMENT
            ),
            False,
            TypeError,
        ),
        (
            lambda value: workload.build_configuration(
                6, value, placement=workload.STRUCTURED_PLACEMENT
            ),
            3,
            ValueError,
        ),
    ),
)
def test_workload_rejects_unfrozen_dimensions(builder, value, error) -> None:
    """Only dimensions with a frozen role and digest can enter the workload."""
    with pytest.raises(error):
        builder(value)
