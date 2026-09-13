"""Tests for the frozen synthetic many-group performance workload."""

from __future__ import annotations

from math import ceil, exp

import numpy as np
import pytest

from examples.many_group_performance import workload
from morana import (
    DirectLinearSolveSettings,
    FissionSourceNormalization,
    FissionTransfer,
    PowerIterationSettings,
)
from morana.solvers.finite_volume import solve_keff


def test_workload_matrix_has_expected_dimensions() -> None:
    """The group-major Cartesian matrix retains all twelve baseline cases."""
    assert workload.baseline_workloads() == (
        (6, 5),
        (6, 10),
        (6, 20),
        (18, 5),
        (18, 10),
        (18, 20),
        (36, 5),
        (36, 10),
        (36, 20),
        (72, 5),
        (72, 10),
        (72, 20),
    )


def test_frozen_calibration_and_digest_coverage() -> None:
    """The calibration scalar and digest coverage remain frozen."""
    assert workload.CALIBRATION_SCALAR == 0.6020060777418164
    assert set(workload.FROZEN_ARRAY_DIGESTS) == set(workload.GROUP_COUNTS)


@pytest.mark.parametrize("groups", workload.GROUP_COUNTS)
def test_analytic_vectors_and_integrated_strengths(groups: int) -> None:
    """Every group count follows the frozen analytic shape and strength rules."""
    cross_sections = workload.build_cross_sections(groups)
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
    assert isinstance(cross_sections.fission.neutron_production, FissionTransfer)
    np.testing.assert_allclose(
        cross_sections.fission.fission_production.mean(),
        workload.CALIBRATION_SCALAR * 5.15e-3,
        rtol=5.0e-16,
    )


def test_scattering_support_weights_and_density_at_72_groups() -> None:
    """The 72-group scattering matrix has the exact specified sparse support."""
    groups = 72
    matrix = workload.build_cross_sections(groups).sigma_s
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
    """The general transfer retains incident-dependent broad fast emission."""
    groups = 72
    cross_sections = workload.build_cross_sections(groups)
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
    assert not np.array_equal(first_spectrum, last_spectrum)


@pytest.mark.parametrize("groups", workload.GROUP_COUNTS)
def test_frozen_array_digests(groups: int) -> None:
    """Every final generated array matches its frozen digest."""
    assert (
        workload.generated_array_digests(groups)
        == workload.FROZEN_ARRAY_DIGESTS[groups]
    )


@pytest.mark.parametrize("axial_layers", (2, 5, 10, 20))
def test_configuration_is_complete_homogeneous_geometry(axial_layers: int) -> None:
    """The builder creates the specified complete vacuum mini-core."""
    configuration = workload.build_configuration(6, axial_layers)
    assert configuration.mesh.num_rings == 5
    assert configuration.mesh.n_cells == 61
    assert configuration.mesh.pitch == pytest.approx(27.94)
    assert configuration.material_mesh.n_axial_layers == axial_layers
    assert configuration.material_mesh.z_max == pytest.approx(182.88)
    assert all(
        configuration.material_mesh.n_active_cells(index) == 61
        for index in range(axial_layers)
    )
    assert len(configuration.materials) == 1
    assert configuration.source is None
    configuration.check_boundary_coverage()
    configuration.check_no_unused_materials()


def test_workload_settings_are_exact_direct_power_controls() -> None:
    """The builder returns the fixed direct-power settings without fallback."""
    settings = workload.solve_settings()
    assert settings.max_outer_iterations == 200
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
        workload.build_configuration(6, workload.SMOKE_AXIAL_LAYER_COUNT),
        FissionSourceNormalization(rate=1.0),
        workload.solve_settings(),
    )
    assert result.keff == pytest.approx(1.001_363_867_359_448_3, rel=1.0e-12)
    assert result.execution_report.iterations == 66
    assert (
        result.execution_report.final_outer_iteration.keff_relative_residual
        <= result.solve_settings.keff_relative_residual_tolerance
    )
    assert abs(result.balance["residual"]) <= 1.0e-12


@pytest.mark.parametrize(
    ("builder", "value", "error"),
    (
        (workload.build_cross_sections, True, TypeError),
        (workload.build_cross_sections, 12, ValueError),
        (lambda value: workload.build_configuration(6, value), False, TypeError),
        (lambda value: workload.build_configuration(6, value), 3, ValueError),
    ),
)
def test_workload_rejects_unfrozen_dimensions(builder, value, error) -> None:
    """Only dimensions with a frozen role and digest can enter the workload."""
    with pytest.raises(error):
        builder(value)
