"""Tests for finite-volume diffusion solve settings and execution."""

from dataclasses import FrozenInstanceError
from fractions import Fraction

import numpy as np
import pytest

from morana import (
    FissionData,
    FissionTransfer,
    SeparableFission,
    BoundaryCondition,
    BoundaryConditionSet,
    CrossSections,
    DirectLinearSolveSettings,
    ExcludedRegion,
    Material,
    MaterialMesh,
    MaterialSlice,
    MaterialSource,
    HexPlanarMesh,
    FixedSourceSettings,
    FissionSourceNormalization,
    KeffSettings,
    KeffSolveReport,
    KeffOuterIterationReport,
    LinearSolveReport,
    GmresLinearSolveSettings,
    IluPreconditioner,
    JacobiPreconditioner,
    NoPreconditioner,
    PowerIterationSettings,
    PowerNormalization,
    ProblemConfiguration,
    ProblemConfigurationSnapshot,
    UniformSource,
    WielandtShiftSettings,
)
from morana.solvers import finite_volume as finite_volume_module
from morana.solvers.finite_volume import solve_fixed_source, solve_keff

from morana.operators import (
    assemble_boundary_rhs,
    assemble_fission_matrix,
    assemble_loss_matrix,
    assemble_source_rhs,
    extract_cross_section_data,
)
from tests.morana.test_configuration import make_configuration


def test_fixed_source_settings_default_to_typed_direct_execution() -> None:
    """Fixed-source settings should default to the typed direct policy."""
    assert isinstance(FixedSourceSettings().linear_solve, DirectLinearSolveSettings)


@pytest.mark.parametrize("rate", [0.0, -1.0, np.inf, np.nan])
def test_fission_source_normalization_requires_positive_finite_rate(
    rate: float,
) -> None:
    """A criticality normalization target must be finite and positive."""
    with pytest.raises(ValueError, match="rate"):
        FissionSourceNormalization(rate=rate)


def test_fission_source_normalization_rejects_nonreal_rate() -> None:
    """A criticality normalization target must be a real scalar."""
    with pytest.raises(TypeError, match="rate"):
        FissionSourceNormalization(rate=True)


def test_fission_source_normalization_normalizes_accepted_real_rate() -> None:
    """A checked fission-source target should retain one built-in float."""
    normalization = FissionSourceNormalization(rate=Fraction(1, 10))

    assert isinstance(normalization.rate, float)


def test_fission_source_normalization_rejects_unrepresentable_real_rate() -> None:
    """A target must remain finite and positive in its stored float form."""
    with pytest.raises(ValueError, match="rate"):
        FissionSourceNormalization(rate=Fraction(1, 10**1000))


@pytest.mark.parametrize("power", [0.0, -1.0, np.inf, np.nan])
def test_power_normalization_requires_positive_finite_power(power: float) -> None:
    """A power-normalization target must be finite and positive."""
    with pytest.raises(ValueError, match="power"):
        PowerNormalization(power=power)


def test_power_normalization_rejects_nonreal_power() -> None:
    """A power-normalization target must be a real scalar."""
    with pytest.raises(TypeError, match="power"):
        PowerNormalization(power=True)


def test_power_normalization_normalizes_accepted_real_power() -> None:
    """A checked power target should retain one built-in float."""
    normalization = PowerNormalization(power=np.float64(1.0))

    assert isinstance(normalization.power, float)


def test_power_normalization_rejects_unrepresentable_real_power() -> None:
    """A target must remain finite and positive in its stored float form."""
    with pytest.raises(ValueError, match="power"):
        PowerNormalization(power=Fraction(1, 10**1000))


@pytest.mark.parametrize("tolerance", [0.0, -1.0e-8, np.inf, np.nan])
def test_direct_linear_solve_settings_check_tolerance(tolerance: float) -> None:
    """The direct residual tolerance must be finite and positive."""
    with pytest.raises(ValueError, match="relative_residual_tolerance"):
        DirectLinearSolveSettings(relative_residual_tolerance=tolerance)


def test_direct_linear_solve_settings_reject_nonreal_tolerance() -> None:
    """The direct residual tolerance must be a real scalar."""
    with pytest.raises(TypeError, match="relative_residual_tolerance"):
        DirectLinearSolveSettings(relative_residual_tolerance=True)


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("keff_change_tolerance", 0.0),
        ("keff_relative_residual_tolerance", np.nan),
        ("flux_change_tolerance", -1.0e-8),
    ],
)
def test_keff_settings_check_eigenvalue_controls(
    setting: str,
    value: float,
) -> None:
    """Configured eigenvalue controls must be finite and positive."""
    with pytest.raises(ValueError, match=setting):
        KeffSettings(**{setting: value})


def test_keff_settings_store_eigenvalue_controls() -> None:
    """Eigenvalue convergence controls belong to settings."""
    settings = KeffSettings(
        keff_change_tolerance=1.0e-9,
        flux_change_tolerance=1.0e-8,
        keff_relative_residual_tolerance=1.0e-7,
    )

    assert settings.keff_change_tolerance == 1.0e-9
    assert settings.flux_change_tolerance == 1.0e-8
    assert settings.keff_relative_residual_tolerance == 1.0e-7
    assert settings.eigenvalue_iteration == PowerIterationSettings()


@pytest.mark.parametrize("shift", [-1.0e-8, np.inf, np.nan])
def test_wielandt_shift_requires_a_finite_nonnegative_inverse_keff(
    shift: float,
) -> None:
    """A fixed shift must be numerical and applicable to the inner operator."""
    with pytest.raises(ValueError, match="shift_inverse_keff"):
        WielandtShiftSettings(shift)


def test_wielandt_shift_rejects_nonreal_inverse_keff() -> None:
    """A fixed shift must be a real scalar."""
    with pytest.raises(TypeError, match="shift_inverse_keff"):
        WielandtShiftSettings(True)


def test_keff_settings_require_one_typed_eigenvalue_iteration_policy() -> None:
    """Criticality settings should reject an untyped eigenvalue policy."""
    with pytest.raises(TypeError, match="eigenvalue_iteration"):
        KeffSettings(eigenvalue_iteration="wielandt")  # type: ignore[arg-type]


def test_gmres_settings_own_one_typed_preconditioner() -> None:
    """GMRES policy should retain its typed execution configuration."""
    settings = GmresLinearSolveSettings(
        max_krylov_iterations=25,
        restart=10,
        preconditioner=IluPreconditioner(drop_tolerance=1.0e-5, fill_factor=4.0),
    )

    assert settings.preconditioner == IluPreconditioner(1.0e-5, 4.0)
    assert GmresLinearSolveSettings(
        preconditioner=NoPreconditioner()
    ).preconditioner == (NoPreconditioner())
    assert GmresLinearSolveSettings(
        preconditioner=JacobiPreconditioner()
    ).preconditioner == (JacobiPreconditioner())


@pytest.mark.parametrize(
    ("constructor", "match"),
    [
        (
            lambda: GmresLinearSolveSettings(preconditioner="jacobi"),
            "preconditioner",
        ),
        (lambda: FixedSourceSettings(linear_solve="direct"), "linear_solve"),
        (lambda: KeffSettings(inner_linear_solve="direct"), "inner_linear_solve"),
        (
            lambda: KeffSettings(eigenvalue_iteration="power"),
            "eigenvalue_iteration",
        ),
    ],
)
def test_solve_settings_reject_unsupported_policy_types(
    constructor: object,
    match: str,
) -> None:
    """Policy slots should identify unsupported runtime types as type errors."""
    with pytest.raises(TypeError, match=match):
        constructor()  # type: ignore[operator]


def test_solve_settings_normalize_accepted_numeric_scalars() -> None:
    """Settings should retain built-in scalar types after accepting real inputs."""
    direct = DirectLinearSolveSettings(Fraction(1, 10))
    fixed_source = FixedSourceSettings(flux_nonnegativity_tolerance=Fraction(1, 10))
    settings = KeffSettings(
        inner_linear_solve=GmresLinearSolveSettings(
            relative_residual_tolerance=Fraction(1, 10),
            max_krylov_iterations=np.int64(25),
            restart=np.int64(10),
            preconditioner=IluPreconditioner(
                drop_tolerance=Fraction(1, 1000),
                fill_factor=np.float64(4.0),
            ),
        ),
        max_outer_iterations=np.int64(20),
        keff_change_tolerance=Fraction(1, 10_000),
        flux_change_tolerance=Fraction(1, 10_000),
        keff_relative_residual_tolerance=Fraction(1, 10_000),
        flux_nonnegativity_tolerance=Fraction(1, 1_000_000),
        eigenvalue_iteration=WielandtShiftSettings(Fraction(4, 5)),
    )

    assert isinstance(direct.relative_residual_tolerance, float)
    assert isinstance(fixed_source.flux_nonnegativity_tolerance, float)
    linear_solve = settings.inner_linear_solve
    assert isinstance(linear_solve.relative_residual_tolerance, float)
    assert isinstance(linear_solve.max_krylov_iterations, int)
    assert isinstance(linear_solve.restart, int)
    assert isinstance(linear_solve.preconditioner.drop_tolerance, float)
    assert isinstance(linear_solve.preconditioner.fill_factor, float)
    assert isinstance(settings.max_outer_iterations, int)
    assert isinstance(settings.keff_change_tolerance, float)
    assert isinstance(settings.flux_change_tolerance, float)
    assert isinstance(settings.keff_relative_residual_tolerance, float)
    assert isinstance(settings.flux_nonnegativity_tolerance, float)
    assert isinstance(settings.eigenvalue_iteration.shift_inverse_keff, float)


@pytest.mark.parametrize("value", [Fraction(1, 10**1000), Fraction(10**1000)])
def test_solve_settings_reject_real_values_outside_float_range(value: Fraction) -> None:
    """Settings should reject real values that cannot retain a finite float."""
    with pytest.raises(ValueError, match="relative_residual_tolerance"):
        DirectLinearSolveSettings(value)


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        (GmresLinearSolveSettings, "max_krylov_iterations"),
        (
            lambda: GmresLinearSolveSettings(restart=3, max_krylov_iterations=2),
            "restart",
        ),
        (lambda: IluPreconditioner(drop_tolerance=-1.0), "drop_tolerance"),
        (lambda: IluPreconditioner(fill_factor=0.0), "fill_factor"),
    ],
)
def test_iterative_policy_checks_its_controls(settings: object, match: str) -> None:
    """Iterative policy construction should reject invalid numerical controls."""
    if settings is GmresLinearSolveSettings:
        with pytest.raises(ValueError, match=match):
            GmresLinearSolveSettings(max_krylov_iterations=0)
    else:
        with pytest.raises(ValueError, match=match):
            settings()  # type: ignore[operator]


@pytest.mark.parametrize(
    "tolerance",
    [-1.0e-12, np.inf, np.nan],
)
def test_fixed_source_settings_check_flux_nonnegativity_tolerance(
    tolerance: float,
) -> None:
    """The negative-roundoff threshold must be finite and nonnegative."""
    with pytest.raises(ValueError, match="flux_nonnegativity_tolerance"):
        FixedSourceSettings(flux_nonnegativity_tolerance=tolerance)


def test_fixed_source_settings_reject_nonreal_flux_nonnegativity_tolerance() -> None:
    """The negative-roundoff threshold must be a real scalar."""
    with pytest.raises(TypeError, match="flux_nonnegativity_tolerance"):
        FixedSourceSettings(flux_nonnegativity_tolerance=True)


def test_fixed_source_settings_allow_zero_flux_nonnegativity_tolerance() -> None:
    """Zero should disable acceptance of negative roundoff."""
    settings = FixedSourceSettings(flux_nonnegativity_tolerance=0.0)

    assert settings.flux_nonnegativity_tolerance == 0.0


@pytest.mark.parametrize("max_outer_iterations", [True, 1.5])
def test_keff_settings_reject_noninteger_max_outer_iterations(
    max_outer_iterations: object,
) -> None:
    """Iteration limits should reject noninteger types."""
    with pytest.raises(TypeError, match="max_outer_iterations must be an integer"):
        KeffSettings(max_outer_iterations=max_outer_iterations)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_outer_iterations", [0, -1])
def test_keff_settings_reject_nonpositive_max_outer_iterations(
    max_outer_iterations: int,
) -> None:
    """Iteration limits should remain strictly positive."""
    with pytest.raises(
        ValueError, match="max_outer_iterations must be a positive integer"
    ):
        KeffSettings(max_outer_iterations=max_outer_iterations)  # type: ignore[arg-type]


def test_solve_settings_are_immutable() -> None:
    """Checked numerical settings should remain construction-time data."""
    settings = FixedSourceSettings()

    with pytest.raises(FrozenInstanceError):
        settings.linear_solve = DirectLinearSolveSettings(1.0e-8)


@pytest.mark.parametrize("configuration", [None, object(), "configuration"])
def test_fixed_source_requires_problem_definition(
    configuration: object,
) -> None:
    """A fixed-source solve must require a supported problem definition."""
    with pytest.raises(
        TypeError,
        match="ProblemConfiguration or ProblemConfigurationSnapshot",
    ):
        solve_fixed_source(configuration)  # type: ignore[arg-type]


@pytest.mark.parametrize("settings", [KeffSettings(), object(), "direct"])
def test_fixed_source_solve_rejects_unsupported_settings(
    settings: object,
) -> None:
    """A fixed-source call must require its own settings type or ``None``."""
    with pytest.raises(TypeError, match="FixedSourceSettings"):
        solve_fixed_source(make_configuration(), settings)  # type: ignore[arg-type]


def test_direct_solve_matches_reflected_one_cell_analytic_flux() -> None:
    """A reflected homogeneous cell should satisfy phi = Q / sigma_a."""
    configuration = make_configuration()

    result = solve_fixed_source(configuration)

    np.testing.assert_allclose(result.flux_layer(0), [[50.0]])
    assert result.n_axial_layers == 1
    assert result.keff is None
    assert result.execution_report.iterations == 1
    assert result.execution_report.true_relative_residual < 1.0e-12
    assert result.balance["source"] == pytest.approx(
        configuration.material_mesh.cell_volume(0)
    )
    assert result.balance["absorption"] == pytest.approx(result.balance["source"])
    assert result.balance["radial_leakage"] == pytest.approx(0.0, abs=1.0e-12)
    assert result.balance["residual"] == pytest.approx(0.0, abs=1.0e-12)
    assert result.solve_mode == "fixed_source"
    assert result.solve_settings == FixedSourceSettings()
    assert isinstance(result.execution_report, LinearSolveReport)
    assert result.execution_report.iterations == 1
    assert result.execution_report.strategy == "direct"
    assert result.execution_report.preconditioner is None
    assert result.groups == 1
    np.testing.assert_allclose(
        result.balance.by_group["source"], [result.balance.scalar["source"]]
    )
    assert result.configuration_snapshot.boundary.assignments[0].condition.kind == (
        "reflective"
    )


@pytest.mark.parametrize(
    "preconditioner",
    [NoPreconditioner(), JacobiPreconditioner(), IluPreconditioner()],
    ids=["none", "jacobi", "ilu"],
)
def test_gmres_fixed_source_matches_direct_analytic_flux(
    preconditioner: NoPreconditioner | JacobiPreconditioner | IluPreconditioner,
) -> None:
    """Each supported GMRES preconditioner should solve the analytic cell case."""
    linear_solve = GmresLinearSolveSettings(
        relative_residual_tolerance=1.0e-12,
        max_krylov_iterations=10,
        restart=5,
        preconditioner=preconditioner,
    )

    result = solve_fixed_source(
        make_configuration(), FixedSourceSettings(linear_solve=linear_solve)
    )

    np.testing.assert_allclose(result.flux_layer(0), [[50.0]])
    assert result.execution_report.linear_solve is linear_solve
    assert result.execution_report.strategy == "gmres"
    assert result.execution_report.preconditioner is preconditioner
    assert 1 <= result.execution_report.iterations <= linear_solve.max_krylov_iterations
    assert (
        result.execution_report.true_relative_residual
        <= linear_solve.relative_residual_tolerance
    )


def test_gmres_rejects_backend_nonconvergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nonzero SciPy GMRES status must not produce a result."""

    def unfinished_gmres(*_args, **_kwargs):
        """Imitate an exhausted GMRES iteration budget."""
        return np.ones(1), 1

    monkeypatch.setattr(finite_volume_module, "gmres", unfinished_gmres)
    settings = FixedSourceSettings(
        linear_solve=GmresLinearSolveSettings(max_krylov_iterations=1, restart=1)
    )

    with pytest.raises(ValueError, match="GMRES did not converge"):
        solve_fixed_source(make_configuration(), settings)


def test_gmres_checks_its_true_residual_after_backend_convergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A false successful backend status must still fail Morana's residual check."""

    def inaccurate_gmres(*_args, **_kwargs):
        """Imitate a backend that incorrectly reports convergence."""
        return np.ones(1), 0

    monkeypatch.setattr(finite_volume_module, "gmres", inaccurate_gmres)
    settings = FixedSourceSettings(
        linear_solve=GmresLinearSolveSettings(relative_residual_tolerance=1.0e-12)
    )

    with pytest.raises(ValueError, match="linear residual exceeds tolerance"):
        solve_fixed_source(make_configuration(), settings)


def test_uniform_reflected_domain_has_uniform_analytic_flux() -> None:
    """Internal diffusion should cancel with a reflected inactive center."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    medium = Material(
        "medium",
        xs=CrossSections(
            D=[1.2],
            sigma_a=[0.02],
            sigma_s=np.zeros((len([1.2]), len([1.2]))),
            fission=None,
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=MaterialMesh.stack(
            (
                MaterialSlice.from_openmc_rings(
                    mesh, [["medium"] * 6, ["0"]], height=1.0
                ),
            ),
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
        source=UniformSource([2.0]),
    )

    result = solve_fixed_source(configuration)

    np.testing.assert_allclose(result.flux_layer(0), np.full((1, 6), 100.0))
    assert result.balance["radial_leakage"] == pytest.approx(0.0, abs=1.0e-10)


def test_vacuum_boundary_reduces_one_cell_flux() -> None:
    """Exposed vacuum leakage should reduce flux below the reflected solution."""
    configuration = make_configuration()
    configuration.set_boundary(
        BoundaryConditionSet(BoundaryCondition.vacuum().globally())
    )

    result = solve_fixed_source(configuration)

    assert 0.0 < result.flux_layer(0)[0, 0] < 50.0
    assert result.balance["radial_leakage"] > 0.0
    assert result.balance["axial_leakage"] > 0.0
    assert result.balance["residual"] == pytest.approx(0.0, abs=1.0e-12)


def test_incoming_current_solution_and_balance_match_analytic_case() -> None:
    """An imposed incoming current should be a boundary source in the balance."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    diffusion = 1.2
    sigma_a = 0.02
    incoming_current = 3.0
    medium = Material(
        "medium",
        xs=CrossSections(
            D=[diffusion],
            sigma_a=[sigma_a],
            sigma_s=np.zeros((len([diffusion]), len([diffusion]))),
            fission=None,
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),),
        ),
        boundary=BoundaryConditionSet(
            BoundaryCondition.incoming_current([incoming_current]).globally(),
            BoundaryCondition.reflective().on_bottom(),
            BoundaryCondition.reflective().on_top(),
        ),
    )
    area = configuration.material_mesh.radial_face_area(0)
    alpha = 0.5
    affine_source = 2.0 * incoming_current
    denominator = diffusion + alpha * mesh.center_to_face
    conductance = area * alpha * diffusion / denominator
    face_source = area * diffusion * affine_source / denominator
    boundary_source = 6.0 * face_source
    expected_flux = boundary_source / (
        sigma_a * configuration.material_mesh.cell_volume(0) + 6.0 * conductance
    )
    expected_absorption = (
        sigma_a * configuration.material_mesh.cell_volume(0) * expected_flux
    )

    result = solve_fixed_source(configuration)

    assert result.flux_layer(0)[0, 0] == pytest.approx(expected_flux)
    assert result.balance["source"] == 0.0
    assert result.balance["boundary_source"] == pytest.approx(boundary_source)
    assert result.balance["absorption"] == pytest.approx(expected_absorption)
    assert result.balance["radial_leakage"] == pytest.approx(
        boundary_source - expected_absorption
    )
    assert result.balance["residual"] == pytest.approx(0.0, abs=1.0e-12)


@pytest.mark.parametrize(
    ("reference_boundary", "partial_return_boundary"),
    [
        (BoundaryCondition.vacuum(), BoundaryCondition.partial_current_return(0.0)),
        (
            BoundaryCondition.reflective(),
            BoundaryCondition.partial_current_return(1.0),
        ),
    ],
)
def test_partial_current_return_limits_match_boundary_presets(
    reference_boundary: BoundaryCondition,
    partial_return_boundary: BoundaryCondition,
) -> None:
    """Partial-current return limits should reproduce vacuum and reflection."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    medium = Material(
        "medium",
        xs=CrossSections(
            D=[1.2],
            sigma_a=[0.02],
            sigma_s=np.zeros((len([1.2]), len([1.2]))),
            fission=None,
        ),
    )

    def solve(boundary: BoundaryCondition):
        """Solve the shared one-cell fixed-source comparison case."""
        configuration = ProblemConfiguration(
            mesh=mesh,
            materials={"medium": medium},
            material_mesh=MaterialMesh.stack(
                (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),),
            ),
            boundary=BoundaryConditionSet(boundary.globally()),
            source=UniformSource([1.0]),
        )
        return solve_fixed_source(configuration)

    reference = solve(reference_boundary)
    partial_return = solve(partial_return_boundary)

    np.testing.assert_allclose(partial_return.flux_layer(0), reference.flux_layer(0))
    assert dict(partial_return.balance) == pytest.approx(dict(reference.balance))


def test_zero_return_with_incidence_matches_pure_incoming_current() -> None:
    """The beta-zero partial-current form should equal incoming-current input."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    medium = Material(
        "medium",
        xs=CrossSections(
            D=[1.2],
            sigma_a=[0.02],
            sigma_s=np.zeros((len([1.2]), len([1.2]))),
            fission=None,
        ),
    )

    def solve(boundary: BoundaryCondition):
        """Solve the shared one-cell incident-current comparison case."""
        configuration = ProblemConfiguration(
            mesh=mesh,
            materials={"medium": medium},
            material_mesh=MaterialMesh.stack(
                (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),),
            ),
            boundary=BoundaryConditionSet(boundary.globally()),
            source=UniformSource([0.0]),
        )
        return solve_fixed_source(configuration)

    partial_return = solve(BoundaryCondition.partial_current_return(0.0, current=[3.0]))
    incoming_current = solve(BoundaryCondition.incoming_current([3.0]))

    np.testing.assert_allclose(
        partial_return.flux_layer(0),
        incoming_current.flux_layer(0),
    )
    assert dict(partial_return.balance) == pytest.approx(dict(incoming_current.balance))


def test_heterogeneous_symmetric_domain_has_positive_symmetric_flux() -> None:
    """A centered source should preserve sixfold symmetry across unlike media."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    medium = Material(
        "medium",
        xs=CrossSections(
            D=[1.2],
            sigma_a=[0.02],
            sigma_s=np.zeros((len([1.2]), len([1.2]))),
            fission=None,
        ),
    )
    reflector = Material(
        "reflector",
        xs=CrossSections(
            D=[0.7],
            sigma_a=[0.005],
            sigma_s=np.zeros((len([0.7]), len([0.7]))),
            fission=None,
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium, "reflector": reflector},
        material_mesh=MaterialMesh.stack(
            (
                MaterialSlice.from_openmc_rings(
                    mesh, [["reflector"] * 6, ["medium"]], height=1.0
                ),
            ),
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
        source=MaterialSource(
            {
                "medium": [1.0],
                "reflector": [0.0],
            }
        ),
    )

    result = solve_fixed_source(configuration)
    flux = result.flux_layer(0)[0]

    assert np.all(flux > 0.0)
    np.testing.assert_allclose(flux[:6], np.full(6, flux[0]), rtol=1.0e-13)
    assert flux[6] > flux[0]
    assert result.balance["radial_leakage"] == pytest.approx(0.0, abs=1.0e-10)


def test_irregular_testing_domain_mixes_boundary_regions_and_closes_balance() -> None:
    """A ragged testing domain should resolve distinct boundary physics safely."""
    mesh = HexPlanarMesh(3, pitch=10.0)
    testing = Material(
        "testing",
        xs=CrossSections(
            D=[1.2],
            sigma_a=[0.02],
            sigma_s=np.zeros((len([1.2]), len([1.2]))),
            fission=None,
        ),
    )
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(
                mesh,
                [
                    [
                        "testing",
                        "gap",
                        "testing",
                        "0",
                        "testing",
                        "gap2",
                        "0",
                        "testing",
                        "0",
                        "0",
                        "testing",
                        "0",
                    ],
                    ["0", "testing", "gap2", "testing", "0", "gap"],
                    ["testing"],
                ],
                height=1.0,
            ),
        ),
        excluded_regions={
            "gap": ExcludedRegion(kind="void"),
            "gap2": ExcludedRegion(kind="void"),
        },
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"testing": testing},
        material_mesh=material_mesh,
        boundary=BoundaryConditionSet(
            BoundaryCondition.reflective().globally(),
            BoundaryCondition.vacuum().on_radial(),
            BoundaryCondition.partial_current_return(0.4, current=[0.5]).on_excluded(
                key="0"
            ),
            BoundaryCondition.reflective().on_excluded(key="0", direction="v+"),
            BoundaryCondition.robin(0.2).on_excluded(key="gap"),
            BoundaryCondition.partial_current_return(0.7, current=[0.25]).on_excluded(
                kind="void"
            ),
            BoundaryCondition.incoming_current([1.0]).on_excluded(
                kind="void", direction="u+"
            ),
        ),
        source=UniformSource([1.0]),
    )
    resolved_kinds = {
        configuration.boundary.resolve(material_mesh.face(0, active_id, direction)).kind
        for active_id in range(material_mesh.n_active_cells(0))
        for direction in ("x+", "x-", "u+", "u-", "v+", "v-")
        if material_mesh.face(0, active_id, direction).kind != "internal"
    }

    result = solve_fixed_source(configuration)
    flux = result.flux_layer(0)[0]

    assert resolved_kinds == {
        "reflective",
        "vacuum",
        "robin",
        "partial_current",
        "incoming_current",
    }
    assert flux.shape == (8,)
    assert np.all(np.isfinite(flux))
    assert np.all(flux > 0.0)
    assert not np.allclose(flux, flux[0], rtol=1.0e-12, atol=0.0)
    assert result.balance["boundary_source"] > 0.0
    assert result.balance["radial_leakage"] > 0.0
    assert result.balance["residual"] == pytest.approx(0.0, abs=1.0e-11)
    assert result.balance["source"] + result.balance[
        "boundary_source"
    ] == pytest.approx(result.balance["absorption"] + result.balance["radial_leakage"])
    assert result.balance["residual"] == pytest.approx(0.0, abs=1.0e-10)


def test_nonzero_dirichlet_solution_and_balance_match_analytic_case() -> None:
    """Prescribed boundary flux should enter both solution and net balance."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    diffusion = 1.2
    sigma_a = 0.02
    boundary_flux = 2.0
    medium = Material(
        "medium",
        xs=CrossSections(
            D=[diffusion],
            sigma_a=[sigma_a],
            sigma_s=np.zeros((len([diffusion]), len([diffusion]))),
            fission=None,
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),),
        ),
        boundary=BoundaryConditionSet(
            BoundaryCondition.dirichlet([boundary_flux]).globally(),
            BoundaryCondition.reflective().on_bottom(),
            BoundaryCondition.reflective().on_top(),
        ),
    )
    volume = configuration.material_mesh.cell_volume(0)
    face_conductance = (
        diffusion
        * configuration.material_mesh.radial_face_area(0)
        / mesh.center_to_face
    )
    boundary_source = 6.0 * face_conductance * boundary_flux
    expected_flux = boundary_source / (sigma_a * volume + 6.0 * face_conductance)
    expected_absorption = sigma_a * volume * expected_flux

    result = solve_fixed_source(configuration)

    assert result.flux_layer(0)[0, 0] == pytest.approx(expected_flux)
    assert result.balance["source"] == 0.0
    assert result.balance["boundary_source"] == pytest.approx(boundary_source)
    assert result.balance["absorption"] == pytest.approx(expected_absorption)
    assert result.balance["radial_leakage"] == pytest.approx(
        boundary_source - expected_absorption
    )
    assert result.balance["residual"] == pytest.approx(0.0, abs=1.0e-12)
    assert result.execution_report.true_relative_residual < 1.0e-12


def test_fixed_source_solve_without_source_and_homogeneous_boundaries_is_zero() -> None:
    """A missing volumetric source should contribute a zero fixed-source RHS."""
    configuration = make_configuration()
    configuration.set_source(None)

    result = solve_fixed_source(configuration)

    np.testing.assert_allclose(result.flux_layer(0), 0.0)
    assert result.balance["source"] == 0.0
    assert result.balance["boundary_source"] == 0.0


def test_fixed_source_preparation_retains_one_coherent_operator_set() -> None:
    """Preparation should retain the operators and sources used by the solve."""
    configuration = make_configuration()
    snapshot = configuration.snapshot()

    problem = finite_volume_module._prepare_fixed_source_problem(snapshot)

    expected_data = extract_cross_section_data(snapshot)
    expected_loss = assemble_loss_matrix(snapshot, expected_data)
    expected_fission = assemble_fission_matrix(snapshot, expected_data)
    expected_source = assemble_source_rhs(snapshot, expected_data)
    expected_boundary = assemble_boundary_rhs(snapshot, expected_data)

    np.testing.assert_allclose(
        problem.matrix.toarray(), (expected_loss - expected_fission).toarray()
    )
    np.testing.assert_allclose(
        problem.fission_matrix.toarray(), expected_fission.toarray()
    )
    np.testing.assert_allclose(problem.source_rhs, expected_source)
    np.testing.assert_allclose(problem.boundary_rhs, expected_boundary)
    assert problem.layout.groups == expected_data.groups


def test_fixed_source_solves_coupled_two_group_downscatter() -> None:
    """Fixed source solves the packed two-group system and returns group-major flux."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    medium = Material(
        "medium",
        xs=CrossSections(
            D=[1.2, 0.4],
            sigma_a=[0.02, 0.08],
            sigma_s=[[0.0, 0.05], [0.0, 0.0]],
            fission=None,
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),),
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
        source=UniformSource([1.0, 0.0]),
    )

    result = solve_fixed_source(configuration)

    volume = configuration.material_mesh.cell_volume(0)
    expected = np.array([[1.0 / 0.07], [0.05 / (0.08 * 0.07)]])
    np.testing.assert_allclose(result.flux_layer(0), expected)
    np.testing.assert_allclose(
        result.balance.by_group["scattering_coupling"],
        [0.0, volume * 0.05 * expected[0, 0]],
    )
    assert result.balance["residual"] == pytest.approx(0.0, abs=1.0e-12)


def test_fixed_source_multiplicity_satisfies_assembled_operator_identity() -> None:
    """Nonunit scattering multiplicity must satisfy ``(A - F) phi = q``."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    medium = Material(
        "medium",
        xs=CrossSections(
            D=[1.2, 0.4],
            sigma_a=[0.2, 0.08],
            sigma_s=[[0.1, 0.05], [0.0, 0.0]],
            multiplicity_matrix=[[2.0, 3.0], [1.0, 1.0]],
            fission=None,
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),)
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
        source=UniformSource([1.0, 0.0]),
    )

    result = solve_fixed_source(configuration)
    data = extract_cross_section_data(configuration)
    flux = result.flux_layer(0)[:, 0]

    np.testing.assert_allclose(
        assemble_loss_matrix(configuration, data) @ flux,
        assemble_source_rhs(configuration, data),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(result.balance.by_group["residual"], 0.0, atol=1.0e-12)


@pytest.mark.parametrize(
    "neutron_production",
    (
        SeparableFission(nu_sigma_f=[0.05, 0.0], chi=[0.0, 1.0]),
        FissionTransfer([[0.0, 0.05], [0.0, 0.0]]),
    ),
)
def test_fixed_source_solves_subcritical_multiplying_response(
    neutron_production: SeparableFission | FissionTransfer,
) -> None:
    """Equivalent fission forms must give the same fixed-source response."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    fuel = Material(
        "fuel",
        xs=CrossSections(
            D=[1.2, 0.4],
            sigma_a=[0.1, 0.2],
            sigma_s=[[0.0, 0.0], [0.0, 0.0]],
            fission=FissionData(neutron_production=neutron_production),
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"fuel": fuel},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["fuel"]], height=1.0),),
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
        source=UniformSource([1.0, 0.0]),
    )

    result = solve_fixed_source(configuration)

    np.testing.assert_allclose(result.flux_layer(0), [[10.0], [2.5]])
    group_balance = result.balance.by_group
    production = 0.05 * configuration.material_mesh.cell_volume(0) * 10.0
    np.testing.assert_allclose(group_balance["fission_production"], [production, 0.0])
    np.testing.assert_allclose(group_balance["fission_emission"], [0.0, production])
    np.testing.assert_allclose(group_balance["residual"], [0.0, 0.0])
    assert result.balance["fission_production"] == pytest.approx(
        result.balance["fission_emission"]
    )


def test_fixed_source_rejects_significantly_negative_multiplied_response() -> None:
    """An algebraically solvable supercritical response is not admissible flux."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    fuel = Material(
        "fuel",
        xs=CrossSections(
            D=[1.2],
            sigma_a=[0.1],
            sigma_s=[[0.0]],
            fission=FissionData(
                neutron_production=SeparableFission(nu_sigma_f=[0.2], chi=[1.0])
            ),
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"fuel": fuel},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["fuel"]], height=1.0),),
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
        source=UniformSource([1.0]),
    )

    with pytest.raises(ValueError, match="significantly negative flux"):
        solve_fixed_source(configuration)


def test_fixed_source_matches_two_group_upscatter_downscatter_reference() -> None:
    """Both transfer directions are solved together rather than iterated."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    medium = Material(
        "medium",
        xs=CrossSections(
            D=[1.2, 0.4],
            sigma_a=[0.1, 0.2],
            sigma_s=[[0.0, 0.05], [0.04, 0.0]],
            fission=None,
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),),
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
        source=UniformSource([1.0, 2.0]),
    )

    result = solve_fixed_source(configuration)

    expected = np.linalg.solve([[0.15, -0.04], [-0.05, 0.24]], [1.0, 2.0])
    np.testing.assert_allclose(result.flux_layer(0)[:, 0], expected)
    np.testing.assert_allclose(
        result.balance.by_group["residual"], [0.0, 0.0], atol=1.0e-12
    )


def test_fixed_source_applies_group_resolved_boundary_sources() -> None:
    """Prescribed incoming current contributes only to its specified group."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    medium = Material(
        "medium",
        xs=CrossSections(
            D=[1.2, 0.4],
            sigma_a=[0.1, 0.2],
            sigma_s=[[0.0, 0.0], [0.0, 0.0]],
            fission=None,
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),),
        ),
        boundary=BoundaryConditionSet(
            BoundaryCondition.incoming_current([0.0, 1.0]).globally()
        ),
        source=UniformSource([0.0, 0.0]),
    )

    result = solve_fixed_source(configuration)

    assert result.flux_layer(0)[0, 0] == pytest.approx(0.0)
    assert result.flux_layer(0)[1, 0] > 0.0
    np.testing.assert_allclose(result.balance.by_group["source"], [0.0, 0.0])
    assert result.balance.by_group["boundary_source"][1] > 0.0


def test_fixed_source_rejects_singular_multiplied_system() -> None:
    """A critical one-cell multiplying system cannot be accepted as a response."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    fuel = Material(
        "fuel",
        xs=CrossSections(
            D=[1.2],
            sigma_a=[0.1],
            sigma_s=[[0.0]],
            fission=FissionData(
                neutron_production=SeparableFission(nu_sigma_f=[0.1], chi=[1.0])
            ),
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"fuel": fuel},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["fuel"]], height=1.0),),
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
        source=UniformSource([1.0]),
    )

    with pytest.raises(ValueError, match="singular or unsolvable"):
        solve_fixed_source(configuration)


def test_fixed_source_rejects_overmultiplying_same_group_scatter() -> None:
    """Multiplying scatter must not create a nonphysical steady response."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    medium = Material(
        "medium",
        xs=CrossSections(
            D=[1.2],
            sigma_a=[0.1],
            sigma_s=[[1.0]],
            multiplicity_matrix=[[3.0]],
            fission=None,
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),)
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
        source=UniformSource([1.0]),
    )

    with pytest.raises(ValueError, match="diagonal entries must be positive"):
        solve_fixed_source(configuration)


def test_fixed_source_solve_supports_reflective_multiple_axial_layers() -> None:
    """A uniform reflected stack retains the one-cell analytic flux by layer."""
    configuration = make_configuration()
    mesh = configuration.mesh
    configuration.set_material_mesh(
        MaterialMesh.stack(
            (
                MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),
                MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=2.0),
            )
        )
    )

    result = solve_fixed_source(configuration)

    assert result.n_axial_layers == 2
    np.testing.assert_allclose(result.flux_layer(0), [[50.0]])
    np.testing.assert_allclose(result.flux_layer(1), [[50.0]])
    assert result.configuration_snapshot.material_mesh.n_active_cells(0) == 1
    assert result.configuration_snapshot.material_mesh.n_active_cells(1) == 1
    assert result.balance.scalar["axial_leakage"] == pytest.approx(0.0, abs=1.0e-12)
    for name, values in result.balance.by_layer_group.items():
        np.testing.assert_allclose(
            np.sum(values, axis=0), result.balance.by_group[name]
        )


def test_fixed_source_matches_hand_assembled_unequal_height_axial_system() -> None:
    """A two-layer source response matches its hand-assembled hex-z system."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    lower = Material(
        "lower",
        CrossSections(
            D=[1.0],
            sigma_a=[1.0],
            sigma_s=[[0.0]],
            fission=None,
        ),
    )
    upper = Material(
        "upper",
        CrossSections(
            D=[1.0],
            sigma_a=[1.0],
            sigma_s=[[0.0]],
            fission=None,
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"lower": lower, "upper": upper},
        material_mesh=MaterialMesh.stack(
            (
                MaterialSlice.from_openmc_rings(mesh, [["lower"]], height=1.0),
                MaterialSlice.from_openmc_rings(mesh, [["upper"]], height=2.0),
            )
        ),
        boundary=BoundaryConditionSet(
            BoundaryCondition.reflective().globally(),
            BoundaryCondition.dirichlet([3.0]).on_top(),
        ),
        source=MaterialSource({"lower": [1.0], "upper": [2.0]}),
    )

    result = solve_fixed_source(configuration)

    np.testing.assert_allclose(result.flux_layer(0), [[25.0 / 17.0]])
    np.testing.assert_allclose(result.flux_layer(1), [[37.0 / 17.0]])
    for values in result.balance.by_layer_group["residual"]:
        np.testing.assert_allclose(values, [0.0], atol=1.0e-12)
    assert result.balance.scalar["residual"] == pytest.approx(0.0, abs=1.0e-12)


def test_keff_solve_supports_reflective_multiple_axial_layers() -> None:
    """A homogeneous reflected stack has the material infinite-medium eigenvalue."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    medium = Material(
        "medium",
        CrossSections(
            D=[1.2],
            sigma_a=[0.02],
            sigma_s=[[0.0]],
            fission=FissionData(
                neutron_production=SeparableFission(nu_sigma_f=[0.025], chi=[1.0])
            ),
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=MaterialMesh.stack(
            (
                MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),
                MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=2.0),
            )
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
    )

    settings = KeffSettings()
    normalization = FissionSourceNormalization(rate=7.0)
    result = solve_keff(configuration, normalization, settings)

    assert result.keff == pytest.approx(1.25)
    assert result.solve_mode == "keff"
    assert result.solve_settings is settings
    assert result.normalization is normalization
    assert isinstance(result.execution_report, KeffSolveReport)
    assert result.execution_report.iterations == len(
        result.execution_report.outer_iterations
    )
    assert (
        result.execution_report.final_outer_iteration
        is result.execution_report.outer_iterations[-1]
    )
    assert all(
        isinstance(report, KeffOuterIterationReport)
        for report in result.execution_report.outer_iterations
    )
    assert result.n_axial_layers == 2
    np.testing.assert_allclose(result.flux_layer(0), result.flux_layer(1))
    assert result.configuration_snapshot.material_mesh.n_active_cells(0) == 1
    assert result.configuration_snapshot.material_mesh.n_active_cells(1) == 1
    assert result.balance.scalar["axial_leakage"] == pytest.approx(0.0, abs=1.0e-12)
    for name, values in result.balance.by_layer_group.items():
        np.testing.assert_allclose(
            np.sum(values, axis=0), result.balance.by_group[name]
        )


def test_keff_solve_rejects_independent_source() -> None:
    """Criticality execution must not silently ignore a fixed source."""
    settings = KeffSettings()
    normalization = FissionSourceNormalization(rate=1.0)

    with pytest.raises(ValueError, match="does not accept an independent source"):
        solve_keff(make_configuration(), normalization, settings)


@pytest.mark.parametrize("normalization", [None, object(), "power"])
def test_keff_solve_requires_supported_normalization(normalization: object) -> None:
    """Criticality execution requires a typed physical normalization."""
    configuration = make_configuration()
    configuration.set_source(None)

    with pytest.raises(TypeError, match="FissionSourceNormalization"):
        solve_keff(configuration, normalization)  # type: ignore[arg-type]


@pytest.mark.parametrize("settings", [FixedSourceSettings(), object(), "direct"])
def test_keff_solve_requires_supported_settings(settings: object) -> None:
    """Criticality execution must require its own settings type or ``None``."""
    configuration = make_configuration()
    configuration.set_source(None)

    with pytest.raises(TypeError, match="KeffSettings"):
        solve_keff(
            configuration,
            FissionSourceNormalization(rate=1.0),
            settings,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "neutron_production",
    [
        SeparableFission(nu_sigma_f=[0.2, 0.1], chi=[0.0, 1.0]),
        FissionTransfer([[0.0, 0.2], [0.0, 0.1]]),
    ],
)
def test_keff_solves_two_group_fission_data_with_one_factorization(
    monkeypatch: pytest.MonkeyPatch,
    neutron_production: SeparableFission | FissionTransfer,
) -> None:
    """Criticality iteration uses equivalent separable and transfer fission data."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    fuel = Material(
        "fuel",
        xs=CrossSections(
            D=[1.2, 0.4],
            sigma_a=[0.1, 0.2],
            sigma_s=[[0.0, 0.0], [0.0, 0.0]],
            fission=FissionData(neutron_production=neutron_production),
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"fuel": fuel},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["fuel"]], height=1.0),),
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
    )
    settings = KeffSettings(max_outer_iterations=20)

    factorization_count = 0
    original_splu = finite_volume_module.splu

    def count_factorizations(matrix):
        """Count sparse factorizations while retaining the production routine."""
        nonlocal factorization_count
        factorization_count += 1
        return original_splu(matrix)

    monkeypatch.setattr(finite_volume_module, "splu", count_factorizations)

    result = solve_keff(configuration, FissionSourceNormalization(rate=1.0), settings)

    assert result.keff == pytest.approx(0.5)
    assert result.flux_layer(0).shape == (2, 1)
    assert result.flux_layer(0)[0, 0] == pytest.approx(0.0)
    assert result.flux_layer(0)[1, 0] > 0.0
    reports = result.execution_report.outer_iterations
    assert (
        tuple(report.keff_change for report in reports)[-1]
        <= settings.keff_change_tolerance
    )
    assert (
        tuple(report.flux_change for report in reports)[-1]
        <= settings.flux_change_tolerance
    )
    assert factorization_count == 1
    group_balance = result.balance.by_group
    np.testing.assert_allclose(group_balance["fission_production"].sum(), 1.0)
    np.testing.assert_allclose(group_balance["keff_source"].sum(), 1.0 / result.keff)
    np.testing.assert_allclose(group_balance["residual"], [0.0, 0.0], atol=1.0e-12)


def test_keff_multiplicity_satisfies_assembled_operator_identity() -> None:
    """Nonunit scattering multiplicity must satisfy ``A phi = F phi / k``."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    fuel = Material(
        "fuel",
        xs=CrossSections(
            D=[1.2, 0.4],
            sigma_a=[0.4, 0.4],
            sigma_s=[[0.1, 0.0], [0.0, 0.0]],
            multiplicity_matrix=[[1.5, 1.0], [1.0, 1.0]],
            fission=FissionData(
                neutron_production=SeparableFission(
                    nu_sigma_f=[0.1, 0.2], chi=[0.5, 0.5]
                )
            ),
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"fuel": fuel},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["fuel"]], height=1.0),)
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
    )

    result = solve_keff(configuration, FissionSourceNormalization(rate=1.0))
    data = extract_cross_section_data(configuration)
    flux = result.flux_layer(0)[:, 0]

    np.testing.assert_allclose(
        assemble_loss_matrix(configuration, data) @ flux,
        assemble_fission_matrix(configuration, data) @ flux / result.keff,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(result.balance.by_group["residual"], 0.0, atol=1.0e-12)


def test_keff_gmres_reuses_its_ilu_preconditioner_within_one_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Criticality GMRES should prepare ILU once and report each inner solve."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    medium = Material(
        "medium",
        xs=CrossSections(
            D=[1.2],
            sigma_a=[0.02],
            sigma_s=[[0.0]],
            fission=FissionData(
                neutron_production=SeparableFission(nu_sigma_f=[0.025], chi=[1.0])
            ),
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),),
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
    )
    linear_solve = GmresLinearSolveSettings(
        relative_residual_tolerance=1.0e-12,
        max_krylov_iterations=10,
        restart=5,
        preconditioner=IluPreconditioner(),
    )
    settings = KeffSettings(
        inner_linear_solve=linear_solve,
        keff_change_tolerance=1.0e-12,
        flux_change_tolerance=1.0e-12,
        keff_relative_residual_tolerance=1.0e-12,
    )

    preconditioner_count = 0
    original_spilu = finite_volume_module.spilu

    def count_preconditioners(*args, **kwargs):
        """Count production ILU setup calls."""
        nonlocal preconditioner_count
        preconditioner_count += 1
        return original_spilu(*args, **kwargs)

    monkeypatch.setattr(finite_volume_module, "spilu", count_preconditioners)

    result = solve_keff(configuration, FissionSourceNormalization(rate=1.0), settings)

    assert result.keff == pytest.approx(1.25)
    assert preconditioner_count == 1
    reports = result.execution_report.outer_iterations
    assert len(reports) == 2
    assert all(report.linear_solve.linear_solve is linear_solve for report in reports)
    assert all(report.linear_solve.strategy == "gmres" for report in reports)
    assert all(
        report.linear_solve.preconditioner is linear_solve.preconditioner
        for report in reports
    )
    assert all(1 <= report.linear_solve.iterations <= 10 for report in reports)


@pytest.mark.parametrize(
    "boundary",
    [
        BoundaryCondition.dirichlet([1.0]),
        BoundaryCondition.partial_current_return(0.5, current=[1.0]),
        BoundaryCondition.incoming_current([1.0]),
    ],
)
def test_keff_solve_rejects_nonhomogeneous_boundary_data(
    boundary: BoundaryCondition,
) -> None:
    """Only the agreed homogeneous boundary representations are eligible."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    medium = Material(
        "medium",
        xs=CrossSections(
            D=[1.2],
            sigma_a=[0.02],
            sigma_s=np.zeros((len([1.2]), len([1.2]))),
            fission=FissionData(
                neutron_production=SeparableFission(nu_sigma_f=[0.025], chi=[1.0])
            ),
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),),
        ),
        boundary=BoundaryConditionSet(boundary.globally()),
    )

    with pytest.raises(ValueError, match="requires homogeneous boundary"):
        solve_keff(configuration, FissionSourceNormalization(rate=1.0))


def test_keff_boundary_checking_identifies_the_ineligible_face() -> None:
    """Boundary checking errors should identify the resolved offending face."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    medium = Material(
        "medium",
        xs=CrossSections(
            D=[1.2],
            sigma_a=[0.02],
            sigma_s=np.zeros((len([1.2]), len([1.2]))),
            fission=FissionData(
                neutron_production=SeparableFission(nu_sigma_f=[0.025], chi=[1.0])
            ),
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),),
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.dirichlet([1.0]).globally()),
    )

    with pytest.raises(
        ValueError,
        match=(
            "axial_index=0, active_id=0, direction='x\\+', "
            "face_kind='outer', boundary_kind='dirichlet'"
        ),
    ):
        solve_keff(configuration, FissionSourceNormalization(rate=1.0))


def test_keff_solve_requires_positive_fission_production() -> None:
    """A nonfissionable domain cannot define a criticality eigenproblem."""
    configuration = make_configuration()
    configuration.set_source(None)

    with pytest.raises(ValueError, match="requires positive fission production"):
        solve_keff(configuration, FissionSourceNormalization(rate=1.0))


def test_keff_solve_rejects_singular_loss_matrix() -> None:
    """A loss-free reflected domain has no finite criticality solution."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    medium = Material(
        "medium",
        xs=CrossSections(
            D=[1.2],
            sigma_a=[0.0],
            sigma_s=np.zeros((len([1.2]), len([1.2]))),
            fission=FissionData(
                neutron_production=SeparableFission(nu_sigma_f=[0.025], chi=[1.0])
            ),
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),),
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
    )

    with pytest.raises(ValueError, match="diagonal entries must be positive"):
        solve_keff(configuration, FissionSourceNormalization(rate=1.0))


def test_keff_solve_matches_reflected_homogeneous_analytic_case() -> None:
    """Power iteration should recover analytic keff and source-normalized flux."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    sigma_a = 0.02
    nu_sigma_f = 0.025
    target_fission_source = 7.0
    medium = Material(
        "medium",
        xs=CrossSections(
            D=[1.2],
            sigma_a=[sigma_a],
            sigma_s=[[0.0]],
            fission=FissionData(
                neutron_production=SeparableFission(nu_sigma_f=[nu_sigma_f], chi=[1.0])
            ),
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),),
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
    )
    settings = KeffSettings(
        keff_change_tolerance=1.0e-12,
        flux_change_tolerance=1.0e-12,
        keff_relative_residual_tolerance=1.0e-12,
    )

    result = solve_keff(
        configuration, FissionSourceNormalization(rate=target_fission_source), settings
    )
    expected_flux = target_fission_source / (
        nu_sigma_f * configuration.material_mesh.cell_volume(0)
    )

    assert result.keff == pytest.approx(nu_sigma_f / sigma_a)
    np.testing.assert_allclose(result.flux_layer(0), [[expected_flux]])
    assert result.execution_report.iterations == 2
    assert (
        result.execution_report.final_outer_iteration.keff_relative_residual
        <= settings.keff_relative_residual_tolerance
    )
    assert result.normalization.rate == target_fission_source
    assert result.balance["fission_production"] == pytest.approx(target_fission_source)
    assert result.balance["keff_source"] == pytest.approx(
        target_fission_source / result.keff
    )
    assert result.balance.scalar["residual"] == pytest.approx(0.0, abs=1.0e-12)
    assert result.balance.loss_fractions["absorption"] == pytest.approx(1.0)
    assert result.balance.source_normalized["absorption"] == pytest.approx(1.0)
    assert result.execution_report.outer_iterations[-1].keff_relative_residual < 1.0e-12


def test_keff_solve_supports_recoverable_power_normalization() -> None:
    """Power normalization should scale the eigenfunction by kappa-sigma-f."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    energy_production = 2.0e6
    target_power = 900.0
    medium = Material(
        "medium",
        xs=CrossSections(
            D=[1.2],
            sigma_a=[0.02],
            sigma_s=[[0.0]],
            fission=FissionData(
                neutron_production=SeparableFission(nu_sigma_f=[0.025], chi=[1.0]),
                kappa_sigma_f=[energy_production],
            ),
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),)
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
    )

    normalization = PowerNormalization(power=target_power)
    result = solve_keff(configuration, normalization)
    volume = configuration.material_mesh.cell_volume(0)

    assert result.normalization is normalization
    np.testing.assert_allclose(
        result.flux_layer(0),
        [[target_power / (energy_production * 1.602176634e-19 * volume)]],
    )
    assert result.balance["fission_production"] == pytest.approx(
        0.025 * target_power / (energy_production * 1.602176634e-19)
    )


def test_keff_power_normalization_rejects_zero_recoverable_energy() -> None:
    """A checked but zero power functional cannot scale a criticality result."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={
            "medium": Material(
                "medium",
                CrossSections(
                    D=[1.2],
                    sigma_a=[0.02],
                    sigma_s=[[0.0]],
                    fission=FissionData(
                        neutron_production=SeparableFission(
                            nu_sigma_f=[0.025], chi=[1.0]
                        ),
                        kappa_sigma_f=[0.0],
                    ),
                ),
            )
        },
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),)
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
    )

    with pytest.raises(ValueError, match="positive recoverable fission energy"):
        solve_keff(configuration, PowerNormalization(power=1.0))


def test_power_normalization_requires_fission_energy_production_data() -> None:
    """Power targets must not silently assume fission energy or multiplicity."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={
            "medium": Material(
                "medium",
                xs=CrossSections(
                    D=[1.2],
                    sigma_a=[0.02],
                    sigma_s=[[0.0]],
                    fission=FissionData(
                        neutron_production=SeparableFission(
                            nu_sigma_f=[0.025], chi=[1.0]
                        )
                    ),
                ),
            )
        },
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),)
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
    )

    with pytest.raises(ValueError, match="kappa_sigma_f"):
        solve_keff(configuration, PowerNormalization(power=1.0))


def test_power_normalization_checks_energy_data_before_boundary_assembly() -> None:
    """Missing energy data is a power-mode preflight failure, not a late error."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={
            "medium": Material(
                "medium",
                xs=CrossSections(
                    D=[1.2],
                    sigma_a=[0.02],
                    sigma_s=[[0.0]],
                    fission=FissionData(
                        neutron_production=SeparableFission(
                            nu_sigma_f=[0.025], chi=[1.0]
                        )
                    ),
                ),
            )
        },
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),)
        ),
        boundary=BoundaryConditionSet(),
    )

    with pytest.raises(
        ValueError,
        match="kappa_sigma_f for every active fissionable material",
    ):
        solve_keff(configuration, PowerNormalization(power=1.0))


@pytest.mark.parametrize(
    "boundary",
    [
        BoundaryCondition.vacuum(),
        BoundaryCondition.incoming_current([0.0]),
    ],
)
def test_keff_solve_matches_leaky_homogeneous_analytic_case(
    boundary: BoundaryCondition,
) -> None:
    """Vacuum leakage should appear in the one-cell analytic eigenvalue."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    diffusion = 1.2
    sigma_a = 0.02
    nu_sigma_f = 0.025
    target_fission_source = 7.0
    medium = Material(
        "medium",
        xs=CrossSections(
            D=[diffusion],
            sigma_a=[sigma_a],
            sigma_s=[[0.0]],
            fission=FissionData(
                neutron_production=SeparableFission(nu_sigma_f=[nu_sigma_f], chi=[1.0])
            ),
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),),
        ),
        boundary=BoundaryConditionSet(
            boundary.globally(),
            BoundaryCondition.reflective().on_bottom(),
            BoundaryCondition.reflective().on_top(),
        ),
    )
    settings = KeffSettings(
        keff_change_tolerance=1.0e-12,
        flux_change_tolerance=1.0e-12,
        keff_relative_residual_tolerance=1.0e-12,
    )

    result = solve_keff(
        configuration, FissionSourceNormalization(rate=target_fission_source), settings
    )

    cell_volume = configuration.material_mesh.cell_volume(0)
    face_conductance = (
        diffusion
        * configuration.material_mesh.radial_face_area(0)
        / (mesh.center_to_face + 2.0 * diffusion)
    )
    expected_loss_coefficient = sigma_a * cell_volume + 6.0 * face_conductance
    expected_flux = target_fission_source / (nu_sigma_f * cell_volume)
    expected_keff = nu_sigma_f * cell_volume / expected_loss_coefficient

    assert result.keff == pytest.approx(expected_keff)
    np.testing.assert_allclose(result.flux_layer(0), [[expected_flux]])
    assert result.balance["absorption"] == pytest.approx(
        sigma_a * cell_volume * expected_flux
    )
    assert result.balance["radial_leakage"] == pytest.approx(
        6.0 * face_conductance * expected_flux
    )
    assert result.balance["keff_source"] == pytest.approx(
        target_fission_source / expected_keff
    )
    assert result.balance["residual"] == pytest.approx(0.0, abs=1.0e-12)


def test_wielandt_shift_accelerates_a_deterministic_weak_mode_case() -> None:
    """A fixed shift should reduce outer iterations without changing the eigenpair."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    dominant = Material(
        "dominant",
        xs=CrossSections(
            D=[1.0e-5],
            sigma_a=[0.02],
            sigma_s=[[0.0]],
            fission=FissionData(
                neutron_production=SeparableFission(nu_sigma_f=[0.025], chi=[1.0])
            ),
        ),
    )
    weak_mode = Material(
        "weak_mode",
        xs=CrossSections(
            D=[1.0e-5],
            sigma_a=[0.02],
            sigma_s=[[0.0]],
            fission=FissionData(
                neutron_production=SeparableFission(nu_sigma_f=[0.024], chi=[1.0])
            ),
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"dominant": dominant, "weak_mode": weak_mode},
        material_mesh=MaterialMesh.stack(
            (
                MaterialSlice.from_openmc_rings(
                    mesh,
                    [["weak_mode"] * 6, ["dominant"]],
                    height=1.0,
                ),
            )
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
    )
    common_controls = {
        "max_outer_iterations": 1_000,
        "keff_change_tolerance": 1.0e-10,
        "flux_change_tolerance": 1.0e-10,
        "keff_relative_residual_tolerance": 1.0e-10,
    }
    power = solve_keff(
        configuration,
        FissionSourceNormalization(rate=1.0),
        KeffSettings(**common_controls),
    )
    policy = WielandtShiftSettings(shift_inverse_keff=0.79)
    shifted = solve_keff(
        configuration,
        FissionSourceNormalization(rate=1.0),
        KeffSettings(**common_controls, eigenvalue_iteration=policy),
    )

    cross_sections = extract_cross_section_data(configuration)
    eigenvalues = np.linalg.eigvals(
        np.linalg.solve(
            assemble_loss_matrix(configuration, cross_sections).toarray(),
            assemble_fission_matrix(configuration, cross_sections).toarray(),
        )
    )
    eigenvalues = np.sort(eigenvalues.real)[::-1]

    assert eigenvalues[1] / eigenvalues[0] > 0.95
    assert power.keff == pytest.approx(eigenvalues[0], rel=1.0e-10)
    assert shifted.keff == pytest.approx(power.keff, rel=1.0e-10)
    np.testing.assert_allclose(
        shifted.flux_layer(0), power.flux_layer(0), rtol=1.0e-5, atol=1.0e-12
    )
    assert shifted.execution_report.eigenvalue_iteration is policy
    assert shifted.execution_report.iterations < power.execution_report.iterations
    assert shifted.execution_report.iterations <= 20
    assert (
        shifted.execution_report.final_outer_iteration.keff_relative_residual
        <= common_controls["keff_relative_residual_tolerance"]
    )


def test_wielandt_shift_rejects_a_singular_shifted_operator() -> None:
    """A singular fixed shift must fail rather than silently falling back."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    medium = Material(
        "medium",
        xs=CrossSections(
            D=[1.2],
            sigma_a=[0.02],
            sigma_s=[[0.0]],
            fission=FissionData(
                neutron_production=SeparableFission(nu_sigma_f=[0.025], chi=[1.0])
            ),
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),)
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
    )

    with pytest.raises(
        ValueError, match="Wielandt-shifted power-iteration k-effective"
    ):
        solve_keff(
            configuration,
            FissionSourceNormalization(rate=1.0),
            KeffSettings(
                eigenvalue_iteration=WielandtShiftSettings(shift_inverse_keff=0.8)
            ),
        )


def test_keff_solve_matches_heterogeneous_dense_eigenvalue_reference() -> None:
    """Coupled power iteration should match a dense heterogeneous reference."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    medium_a = Material(
        "medium_a",
        xs=CrossSections(
            D=[1.2, 0.35],
            sigma_a=[0.02, 0.08],
            sigma_s=[[0.0, 0.04], [0.005, 0.0]],
            fission=FissionData(
                neutron_production=SeparableFission(
                    nu_sigma_f=[0.01, 0.11], chi=[0.95, 0.05]
                )
            ),
        ),
    )
    medium_b = Material(
        "medium_b",
        xs=CrossSections(
            D=[0.75, 0.25],
            sigma_a=[0.035, 0.1],
            sigma_s=[[0.0, 0.03], [0.002, 0.0]],
            fission=FissionData(
                neutron_production=SeparableFission(
                    nu_sigma_f=[0.02, 0.12], chi=[0.9, 0.1]
                )
            ),
        ),
    )
    reflector = Material(
        "reflector",
        xs=CrossSections(
            D=[1.0, 0.4],
            sigma_a=[0.006, 0.03],
            sigma_s=[[0.0, 0.05], [0.001, 0.0]],
            fission=None,
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={
            "medium_a": medium_a,
            "medium_b": medium_b,
            "reflector": reflector,
        },
        material_mesh=MaterialMesh.stack(
            (
                MaterialSlice.from_openmc_rings(
                    mesh,
                    [
                        [
                            "medium_a",
                            "reflector",
                            "medium_b",
                            "reflector",
                            "medium_a",
                            "medium_b",
                        ],
                        ["medium_a"],
                    ],
                    height=1.0,
                ),
            ),
        ),
        boundary=BoundaryConditionSet(
            BoundaryCondition.vacuum().globally(),
            BoundaryCondition.reflective().on_bottom(),
            BoundaryCondition.reflective().on_top(),
        ),
    )
    settings = KeffSettings(
        keff_change_tolerance=1.0e-11,
        flux_change_tolerance=1.0e-11,
        keff_relative_residual_tolerance=1.0e-11,
        max_outer_iterations=500,
    )

    result = solve_keff(configuration, FissionSourceNormalization(rate=3.2), settings)

    cross_sections = extract_cross_section_data(configuration)
    loss_matrix = assemble_loss_matrix(configuration, cross_sections).toarray()
    fission_matrix = assemble_fission_matrix(configuration, cross_sections).toarray()
    reference_eigenvalues = np.linalg.eigvals(
        np.linalg.solve(loss_matrix, fission_matrix)
    )
    dominant_index = int(np.argmax(reference_eigenvalues.real))
    expected_keff = float(reference_eigenvalues[dominant_index].real)
    packed_flux = result.flux_layer(0).T.reshape(-1)
    production = np.asarray(fission_matrix.sum(axis=0)).ravel()
    fission_source = float(np.dot(production, packed_flux))
    residual = loss_matrix @ packed_flux - fission_matrix @ packed_flux / result.keff

    assert abs(reference_eigenvalues[dominant_index].imag) < 1.0e-12
    assert result.keff == pytest.approx(expected_keff, rel=1.0e-10)
    assert np.all(packed_flux > 0.0)
    assert fission_source == pytest.approx(3.2)
    assert result.balance["keff_source"] == pytest.approx(fission_source / result.keff)
    assert np.linalg.norm(residual) < 1.0e-10


def test_keff_solve_reports_power_iteration_failure() -> None:
    """A finite iteration limit must not return an unconverged eigenpair."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    medium = Material(
        "medium",
        xs=CrossSections(
            D=[1.2],
            sigma_a=[0.02],
            sigma_s=np.zeros((len([1.2]), len([1.2]))),
            fission=FissionData(
                neutron_production=SeparableFission(nu_sigma_f=[0.025], chi=[1.0])
            ),
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),),
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
    )

    with pytest.raises(RuntimeError, match="max_outer_iterations"):
        solve_keff(
            configuration,
            FissionSourceNormalization(rate=1.0),
            KeffSettings(max_outer_iterations=1),
        )


def _snapshot_fixed_source_configuration() -> ProblemConfiguration:
    """Return one reflected one-cell fixed-source problem."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material = Material(
        "medium",
        CrossSections(
            D=[1.2],
            sigma_a=[0.02],
            sigma_s=[[0.0]],
            fission=None,
        ),
    )
    return ProblemConfiguration(
        mesh=mesh,
        materials={"medium": material},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),)
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
        source=UniformSource([1.0]),
    )


def _snapshot_keff_configuration() -> ProblemConfiguration:
    """Return one reflected one-cell criticality problem."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material = Material(
        "medium",
        CrossSections(
            D=[1.2],
            sigma_a=[0.02],
            sigma_s=[[0.0]],
            fission=FissionData(
                neutron_production=SeparableFission(nu_sigma_f=[0.025], chi=[1.0])
            ),
        ),
    )
    return ProblemConfiguration(
        mesh=mesh,
        materials={"medium": material},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),)
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
    )


def test_fixed_source_accepts_a_configuration_snapshot() -> None:
    """A fixed-source rerun may consume completed-result provenance directly."""
    first = solve_fixed_source(_snapshot_fixed_source_configuration())

    rerun = solve_fixed_source(first.configuration_snapshot)

    np.testing.assert_allclose(rerun.flux_layer(0), first.flux_layer(0))
    assert rerun.configuration_snapshot is first.configuration_snapshot


def test_keff_accepts_a_configuration_snapshot() -> None:
    """A criticality solve may consume immutable problem provenance directly."""
    snapshot = _snapshot_keff_configuration().snapshot()

    result = solve_keff(snapshot, FissionSourceNormalization(rate=1.0))

    assert result.keff == pytest.approx(1.25)
    assert result.configuration_snapshot is snapshot


def test_fixed_source_uses_its_entry_configuration_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller mutation during assembly cannot affect one fixed-source solve."""
    configuration = _snapshot_fixed_source_configuration()
    original_extract = finite_volume_module.extract_cross_section_data

    def mutate_caller_configuration(
        solve_configuration: ProblemConfigurationSnapshot,
    ) -> finite_volume_module.CrossSectionData:
        extracted = original_extract(solve_configuration)
        configuration.set_source(UniformSource([3.0]))
        return extracted

    monkeypatch.setattr(
        finite_volume_module, "extract_cross_section_data", mutate_caller_configuration
    )

    result = solve_fixed_source(configuration)

    np.testing.assert_allclose(result.flux_layer(0), [[50.0]])
    assert isinstance(configuration.source, UniformSource)
    np.testing.assert_allclose(configuration.source.strength, [3.0])
    assert isinstance(result.configuration_snapshot.source, UniformSource)
    np.testing.assert_allclose(result.configuration_snapshot.source.strength, [1.0])


def test_keff_uses_its_entry_configuration_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller mutation during assembly cannot affect one criticality solve."""
    configuration = _snapshot_keff_configuration()
    original_extract = finite_volume_module.extract_cross_section_data

    def mutate_caller_configuration(
        solve_configuration: ProblemConfigurationSnapshot,
    ) -> finite_volume_module.CrossSectionData:
        extracted = original_extract(solve_configuration)
        configuration.set_source(UniformSource([3.0]))
        return extracted

    monkeypatch.setattr(
        finite_volume_module, "extract_cross_section_data", mutate_caller_configuration
    )

    result = solve_keff(configuration, FissionSourceNormalization(rate=1.0))

    assert result.keff == pytest.approx(1.25)
    assert isinstance(configuration.source, UniformSource)
    assert result.configuration_snapshot.source is None


def test_solves_do_not_reconstruct_a_mutable_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finite-volume solves consume snapshots without mutable reconstruction."""
    fixed_source_snapshot = _snapshot_fixed_source_configuration().snapshot()
    keff_snapshot = _snapshot_keff_configuration().snapshot()

    def fail_reconstruction(
        _snapshot: ProblemConfigurationSnapshot,
    ) -> ProblemConfiguration:
        raise AssertionError("solve must not reconstruct a mutable configuration")

    monkeypatch.setattr(
        ProblemConfigurationSnapshot, "to_configuration", fail_reconstruction
    )

    fixed_source = solve_fixed_source(fixed_source_snapshot)
    criticality = solve_keff(keff_snapshot, FissionSourceNormalization(rate=1.0))

    np.testing.assert_allclose(fixed_source.flux_layer(0), [[50.0]])
    assert criticality.keff == pytest.approx(1.25)


@pytest.mark.parametrize("strength", [1e-200, 1.0, 1e200])
def test_fixed_source_residual_check_is_scale_independent(strength, monkeypatch):
    """A wrong backend solution must fail at every source magnitude."""
    configuration = make_configuration()
    configuration.set_source(UniformSource([strength]))
    monkeypatch.setattr(
        finite_volume_module, "spsolve", lambda matrix, rhs: rhs / matrix.diagonal() * 2
    )
    with pytest.raises(ValueError, match="residual"):
        solve_fixed_source(configuration)


@pytest.mark.parametrize("strength", [1e-200, 1e200])
def test_fixed_source_accepts_finite_extreme_source_strength(strength):
    """A finite analytic solution must survive the independent norm check."""
    configuration = make_configuration()
    configuration.set_source(UniformSource([strength]))
    result = solve_fixed_source(configuration)
    np.testing.assert_allclose(result.flux[0] / strength, [[50.0]])


def test_self_scattering_does_not_erase_absorption():
    """Unit-multiplicity self-scattering cancels before numerical summation."""
    configuration = make_configuration()
    configuration.replace_material(
        Material(
            "medium",
            CrossSections(
                D=[0.0],
                sigma_a=[0.02],
                sigma_s=[[1e20]],
                fission=None,
            ),
        )
    )
    result = solve_fixed_source(configuration)
    np.testing.assert_allclose(result.flux[0], [[50.0]])


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_volume_weighted_norm_remains_representable(scale):
    """Weighted norms must not square unscaled extreme flux values."""
    actual = finite_volume_module._volume_weighted_norm(
        np.array([3.0, 4.0]) * scale, np.array([4.0, 4.0])
    )
    assert actual / scale == pytest.approx(10.0)
