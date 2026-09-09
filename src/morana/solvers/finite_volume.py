"""Finite-volume multigroup diffusion solve entry points."""

# pylint: disable=duplicate-code,too-many-instance-attributes,too-many-lines

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import (
    LinearOperator,
    MatrixRankWarning,
    SuperLU,
    gmres,
    spilu,
    splu,
    spsolve,
)

from morana.boundary import _ResolvedBoundaryCondition
from morana.configuration import ProblemConfiguration, ProblemConfigurationSnapshot
from morana.execution_reports import (
    KeffSolveReport,
    KeffOuterIterationReport,
    LinearSolveReport,
)
from morana.normalization import (
    FissionSourceNormalization,
    PowerNormalization,
)
from morana.operators import (
    CrossSectionData,
    CrossSectionLayerData,
    _FiniteVolumeAssemblyContext,
    _FiniteVolumeLayout,
    _assemble_boundary_rhs,
    _assemble_fission_matrix,
    _assemble_fission_power_functional,
    _assemble_loss_matrix,
    _assemble_source_rhs,
    _checked_layout,
    _exposed_face_conductance,
    _fission_production_functional,
    _finite_volume_assembly_context,
    _finite_volume_assembly_context_from_layout,
    _internal_interface_conductance,
    _require_power_normalization_availability,
    extract_cross_section_data,
)
from morana.results import FixedSourceBalance, KeffBalance, Result
from morana.solve_settings import (
    DirectLinearSolveSettings,
    FixedSourceSettings,
    GmresLinearSolveSettings,
    IluPreconditioner,
    JacobiPreconditioner,
    KeffSettings,
    LinearSolveSettings,
    NoPreconditioner,
    WielandtShiftSettings,
)

_KEFF_ITERATION_SOLVE_LABELS = {
    "power": "power-iteration k-effective",
    "wielandt": "Wielandt-shifted power-iteration k-effective",
}

__all__ = ["solve_fixed_source", "solve_keff"]


@dataclass(frozen=True)
class _FixedSourceProblem:
    """Checked finite-volume operators and sources for one fixed-source solve."""

    cross_sections: CrossSectionData
    context: _FiniteVolumeAssemblyContext
    fission_matrix: csr_matrix
    matrix: csr_matrix
    source_rhs: np.ndarray
    boundary_rhs: np.ndarray

    @property
    def layout(self) -> _FiniteVolumeLayout:
        """Return the checked global finite-volume layout."""
        return self.context.layout


@dataclass(frozen=True)
class _KeffProblem:
    """Prepared finite-volume operators and inner solve for criticality."""

    cross_sections: CrossSectionData
    context: _FiniteVolumeAssemblyContext
    loss_matrix: csr_matrix
    fission_matrix: csr_matrix
    fission_production_functional: np.ndarray
    fission_power_functional: np.ndarray | None
    flux_norm_weights: np.ndarray
    inner_solve_matrix: csr_matrix
    inner_linear_solve: "_PreparedLinearSolve"
    settings: KeffSettings

    @property
    def layout(self) -> _FiniteVolumeLayout:
        """Return the checked global finite-volume layout."""
        return self.context.layout


@dataclass(frozen=True)
class _PreparedLinearSolve:
    """Per-call reusable direct factorization or GMRES preconditioner."""

    linear_solve: LinearSolveSettings
    factorization: SuperLU | None = None
    preconditioner: LinearOperator | None = None


def solve_fixed_source(
    configuration: ProblemConfiguration | ProblemConfigurationSnapshot,
    settings: FixedSourceSettings | None = None,
) -> Result:
    """Solve the layered multigroup hex-z fixed-source problem.

    The optional configured volumetric source and additive boundary terms form
    the right-hand side. The solver applies the configured direct or GMRES
    policy to the coupled loss-minus-fission system,
    accepts only negative flux roundoff within
    ``flux_nonnegativity_tolerance``, and checks the relative linear
    residual against ``settings.linear_solve``'s relative-residual
    tolerance. The completed result has ``keff=None`` and one
    linear-solve report. A mutable configuration is snapshotted at entry; a
    supplied snapshot is retained directly as result provenance. Calculation
    uses that immutable snapshot directly.

    Parameters
    ----------
    configuration
        Mutable problem definition or immutable snapshot to solve.
    settings
        Immutable fixed-source controls. ``None`` creates default
        ``FixedSourceSettings`` for this call only.

    Returns
    -------
    Result
        Completed flux, balance, convergence, and provenance data.

    Raises
    ------
    TypeError
        If configuration is neither ``ProblemConfiguration`` nor
        ``ProblemConfigurationSnapshot``, or settings are not
        ``FixedSourceSettings`` or ``None``.
    ValueError
        If the active domain, material data, boundary coverage, or configured
        source is invalid; a linear system or preconditioner is unusable; or
        flux and residual numerical checks fail.
    """
    configuration_snapshot = _configuration_snapshot_for_solve(configuration)
    settings = _checked_fixed_source_settings(settings)
    problem = _prepare_fixed_source_problem(configuration_snapshot)
    rhs = problem.source_rhs + problem.boundary_rhs
    flux, linear_iterations = _solve_linear_system(
        problem.matrix, rhs, settings.linear_solve, "fixed-source"
    )
    flux = _clean_fixed_source_flux(flux, rhs, settings)
    _, relative_residual = _require_linear_residual(
        problem.matrix, flux, rhs, "fixed-source", settings.linear_solve
    )
    layer_group_balance = _fixed_source_layer_group_balance(
        configuration=configuration_snapshot,
        cross_sections=problem.cross_sections,
        context=problem.context,
        fission_matrix=problem.fission_matrix,
        source_rhs=problem.source_rhs,
        boundary_rhs=problem.boundary_rhs,
        flux=flux,
    )
    group_balance = _contract_layer_group_balance(layer_group_balance)
    balance = FixedSourceBalance._from_validated(  # pylint: disable=protected-access
        by_group=group_balance,
        by_layer_group=layer_group_balance,
    )

    return Result._from_validated(  # pylint: disable=protected-access
        flux=problem.layout.unpack(flux),
        balance=balance,
        configuration_snapshot=configuration_snapshot,
        solve_settings=settings,
        execution_report=LinearSolveReport._from_validated(  # pylint: disable=protected-access
            linear_solve=settings.linear_solve,
            iterations=linear_iterations,
            true_relative_residual=relative_residual,
        ),
    )


def solve_keff(
    configuration: ProblemConfiguration | ProblemConfigurationSnapshot,
    normalization: FissionSourceNormalization | PowerNormalization,
    settings: KeffSettings | None = None,
) -> Result:
    """Solve a physically normalized fission eigenvalue diffusion problem.

    A configured positive fission-source rate or recoverable thermal-power
    target scales the exposed result after convergence; it does not alter
    the returned ``keff``. Power normalization requires
    ``kappa_sigma_f`` for every fissionable active material. The solver
    verifies that requirement before it assembles finite-volume operators
    or prepares an inner linear solve. The solver
    starts from a positive unit-fission-source vector, then requires all
    three effective convergence criteria—multiplication-factor change,
    volume-weighted flux change, and relative eigenvalue-equation
    residual—to meet their explicit tolerances within
    ``max_outer_iterations``. A mutable configuration is snapshotted at entry;
    a supplied snapshot is retained directly as result provenance. Calculation
    uses that immutable snapshot directly.

    Parameters
    ----------
    configuration
        Mutable problem definition or immutable snapshot to solve.
    normalization
        Immutable target fission-neutron source rate or recoverable
        thermal power for the completed result.
    settings
        Optional immutable criticality numerical controls. ``None`` uses
        default ``KeffSettings`` for this call only.

    Returns
    -------
    Result
        Completed physically normalized flux, multiplication factor,
        balance, convergence, and provenance data.

    Raises
    ------
    TypeError
        If configuration is neither ``ProblemConfiguration`` nor
        ``ProblemConfigurationSnapshot``, normalization is neither
        ``FissionSourceNormalization`` nor ``PowerNormalization``, or settings
        are neither ``KeffSettings`` nor ``None``.
    ValueError
        If an independent source is configured; the active domain,
        material data, fission production, power data, or exposed boundary
        data is ineligible; the loss matrix is singular; or an inner solve,
        flux, or residual fails numerical checking.
    RuntimeError
        If power iteration does not satisfy all three convergence criteria
        within ``settings.max_outer_iterations``.
    """
    configuration_snapshot = _configuration_snapshot_for_solve(configuration)
    normalization = _checked_normalization(normalization)
    settings = _checked_keff_settings(settings)
    problem = _prepare_keff_problem(configuration_snapshot, settings, normalization)
    flux, execution_report = _iterate_keff(problem)
    flux = _normalize_keff_flux(problem, normalization, flux)
    layer_group_balance = _keff_layer_group_balance(
        problem,
        flux,
        execution_report.final_outer_iteration.keff,
    )
    group_balance = _contract_layer_group_balance(layer_group_balance)
    balance = KeffBalance._from_validated(  # pylint: disable=protected-access
        by_group=group_balance,
        by_layer_group=layer_group_balance,
    )

    return Result._from_validated(  # pylint: disable=protected-access
        flux=problem.layout.unpack(flux),
        balance=balance,
        configuration_snapshot=configuration_snapshot,
        solve_settings=settings,
        normalization=normalization,
        execution_report=execution_report,
    )


def _configuration_snapshot_for_solve(
    configuration: ProblemConfiguration | ProblemConfigurationSnapshot,
) -> ProblemConfigurationSnapshot:
    """Capture one mutable configuration or accept immutable solve provenance."""
    if isinstance(configuration, ProblemConfigurationSnapshot):
        return configuration
    if isinstance(configuration, ProblemConfiguration):
        return configuration.snapshot()
    raise TypeError(
        "configuration must be a ProblemConfiguration or "
        "ProblemConfigurationSnapshot"
    )


def _checked_fixed_source_settings(
    settings: FixedSourceSettings | None,
) -> FixedSourceSettings:
    """Return checked fixed-source settings, defaulting when absent."""
    if settings is None:
        return FixedSourceSettings()
    if not isinstance(settings, FixedSourceSettings):
        raise TypeError("fixed-source settings must be FixedSourceSettings or None")
    return settings


def _checked_keff_settings(settings: KeffSettings | None) -> KeffSettings:
    """Return checked criticality settings, defaulting when absent."""
    if settings is None:
        return KeffSettings()
    if not isinstance(settings, KeffSettings):
        raise TypeError("k-effective settings must be KeffSettings or None")
    return settings


def _checked_normalization(
    normalization: FissionSourceNormalization | PowerNormalization,
) -> FissionSourceNormalization | PowerNormalization:
    """Require one supported physical criticality normalization."""
    if not isinstance(normalization, (FissionSourceNormalization, PowerNormalization)):
        raise TypeError(
            "k-effective normalization must be FissionSourceNormalization "
            "or PowerNormalization"
        )
    return normalization


def _normalize_keff_flux(
    problem: _KeffProblem,
    normalization: FissionSourceNormalization | PowerNormalization,
    flux: np.ndarray,
) -> np.ndarray:
    """Return flux scaled to the requested physical criticality normalization."""
    if isinstance(normalization, FissionSourceNormalization):
        functional = problem.fission_production_functional
        target = normalization.rate
    else:
        if problem.fission_power_functional is None:
            raise ValueError("power normalization functional is unavailable")
        functional = problem.fission_power_functional
        target = normalization.power
    private_normalization = float(np.dot(functional, flux))
    if not np.isfinite(private_normalization) or private_normalization <= 0.0:
        raise ValueError(
            "k-effective solve produced nonpositive physical normalization"
        )
    return flux * (target / private_normalization)


def _prepare_fixed_source_problem(
    configuration: ProblemConfigurationSnapshot,
) -> _FixedSourceProblem:
    """Prepare checked finite-volume operators and sources for one solve.

    This private preparation boundary resolves every exposed boundary face and
    evaluates the configured volumetric source once.  The returned state is
    local to one finite-volume solve; it is not configuration provenance or a
    public representation shared with other discretizations.
    """
    cross_sections = extract_cross_section_data(configuration)
    context = _finite_volume_assembly_context(configuration, cross_sections)
    loss_matrix = _assemble_loss_matrix(context)
    fission_matrix = _assemble_fission_matrix(
        configuration, cross_sections, context.layout
    )
    source_rhs = _assemble_source_rhs(configuration, cross_sections, context.layout)
    boundary_rhs = _assemble_boundary_rhs(context)
    return _FixedSourceProblem(
        cross_sections=cross_sections,
        context=context,
        fission_matrix=fission_matrix,
        matrix=loss_matrix - fission_matrix,
        source_rhs=source_rhs,
        boundary_rhs=boundary_rhs,
    )


def _prepare_keff_problem(
    configuration: ProblemConfigurationSnapshot,
    settings: KeffSettings,
    normalization: FissionSourceNormalization | PowerNormalization,
) -> _KeffProblem:
    """Prepare checked finite-volume operators and an inner solve for criticality."""
    if configuration.source is not None:
        raise ValueError("k-effective solve does not accept an independent source")
    cross_sections = extract_cross_section_data(configuration)
    layout = _checked_layout(configuration, cross_sections)
    if isinstance(normalization, PowerNormalization):
        _require_power_normalization_availability(configuration, cross_sections)
    context = _finite_volume_assembly_context_from_layout(
        configuration, cross_sections, layout
    )
    _check_keff_boundaries(context)
    loss_matrix = _assemble_loss_matrix(context)
    fission_matrix = _assemble_fission_matrix(configuration, cross_sections, layout)
    fission_production_functional = _fission_production_functional(
        configuration, cross_sections, layout
    )
    if float(np.sum(fission_production_functional)) <= 0.0:
        raise ValueError("k-effective solve requires positive fission production")
    fission_power_functional = (
        _assemble_fission_power_functional(configuration, cross_sections, layout)
        if isinstance(normalization, PowerNormalization)
        else None
    )
    if (
        fission_power_functional is not None
        and float(np.sum(fission_power_functional)) <= 0.0
    ):
        raise ValueError(
            "power normalization requires positive recoverable fission energy"
        )
    inner_solve_matrix = _keff_inner_solve_matrix(loss_matrix, fission_matrix, settings)
    solve_label = _KEFF_ITERATION_SOLVE_LABELS[settings.eigenvalue_iteration.kind]
    inner_linear_solve = _prepare_keff_linear_solve(
        inner_solve_matrix,
        settings.inner_linear_solve,
        solve_label,
    )
    flux_norm_weights = _packed_cell_volumes(context.material_mesh, context.layout)
    return _KeffProblem(
        cross_sections=cross_sections,
        context=context,
        loss_matrix=loss_matrix,
        fission_matrix=fission_matrix,
        fission_production_functional=fission_production_functional,
        fission_power_functional=fission_power_functional,
        flux_norm_weights=flux_norm_weights,
        inner_solve_matrix=inner_solve_matrix,
        inner_linear_solve=inner_linear_solve,
        settings=settings,
    )


def _keff_inner_solve_matrix(
    loss_matrix: csr_matrix,
    fission_matrix: csr_matrix,
    settings: KeffSettings,
) -> csr_matrix:
    """Return the checked ordinary or fixed-Wielandt inner operator."""
    policy = settings.eigenvalue_iteration
    if not isinstance(policy, WielandtShiftSettings):
        return loss_matrix
    return (loss_matrix - policy.shift_inverse_keff * fission_matrix).tocsr()


def _iterate_keff(
    problem: _KeffProblem,
) -> tuple[np.ndarray, KeffSolveReport]:
    """Return fission-normalized flux and report from the selected policy."""
    flux = np.ones(problem.loss_matrix.shape[0], dtype=float)
    flux /= float(np.dot(problem.fission_production_functional, flux))
    previous_keff = 1.0
    outer_iterations = []

    settings = problem.settings
    solve_label = _KEFF_ITERATION_SOLVE_LABELS[settings.eigenvalue_iteration.kind]
    for iteration in range(1, settings.max_outer_iterations + 1):
        fission_source = problem.fission_matrix @ flux
        candidate, linear_iterations = _solve_prepared_linear_system(
            problem.inner_solve_matrix,
            fission_source,
            problem.inner_linear_solve,
            solve_label,
        )
        candidate = _clean_keff_flux(candidate, settings, solve_label)
        _, linear_relative_residual = _require_linear_residual(
            problem.inner_solve_matrix,
            candidate,
            fission_source,
            solve_label,
            problem.inner_linear_solve.linear_solve,
        )
        linear_solve_report = (
            LinearSolveReport._from_validated(  # pylint: disable=protected-access
                linear_solve=problem.inner_linear_solve.linear_solve,
                iterations=linear_iterations,
                true_relative_residual=linear_relative_residual,
            )
        )
        candidate_fission_production = float(
            np.dot(problem.fission_production_functional, candidate)
        )
        if (
            not np.isfinite(candidate_fission_production)
            or candidate_fission_production <= 0.0
        ):
            raise ValueError(
                "k-effective iteration produced a nonpositive or non-finite "
                "fission production"
            )
        keff = _keff_from_iteration_production(
            candidate_fission_production, settings.eigenvalue_iteration
        )
        candidate /= candidate_fission_production
        candidate_norm = _volume_weighted_norm(candidate, problem.flux_norm_weights)
        if not np.isfinite(candidate_norm) or candidate_norm == 0.0:
            raise ValueError(
                "k-effective iteration produced a zero or non-finite flux norm"
            )
        keff_change = abs(keff - previous_keff) / abs(keff)
        flux_change = (
            _volume_weighted_norm(candidate - flux, problem.flux_norm_weights)
            / candidate_norm
        )
        loss_action = problem.loss_matrix @ candidate
        keff_source = problem.fission_matrix @ candidate / keff
        _, keff_relative_residual = _equation_residual(
            keff_source,
            loss_action,
        )
        outer_iterations.append(
            KeffOuterIterationReport._from_validated(  # pylint: disable=protected-access
                iteration=iteration,
                linear_solve=linear_solve_report,
                keff=keff,
                keff_change=keff_change,
                flux_change=flux_change,
                keff_relative_residual=keff_relative_residual,
            )
        )
        if (
            keff_change <= settings.keff_change_tolerance
            and flux_change <= settings.flux_change_tolerance
            and keff_relative_residual <= settings.keff_relative_residual_tolerance
        ):
            return (
                np.array(candidate, copy=True),
                KeffSolveReport._from_validated(  # pylint: disable=protected-access
                    tuple(outer_iterations), settings.eigenvalue_iteration
                ),
            )
        flux = candidate
        previous_keff = keff

    raise RuntimeError(
        "k-effective power iteration did not converge within "
        f"max_outer_iterations={settings.max_outer_iterations}; final keff_change="
        f"{outer_iterations[-1].keff_change:.6e} "
        f"(tolerance {settings.keff_change_tolerance:.6e}), "
        f"flux_change={outer_iterations[-1].flux_change:.6e} "
        f"(tolerance {settings.flux_change_tolerance:.6e}), "
        "keff_relative_residual="
        f"{outer_iterations[-1].keff_relative_residual:.6e} "
        f"(tolerance {settings.keff_relative_residual_tolerance:.6e})"
    )


def _keff_from_iteration_production(
    candidate_fission_production: float,
    policy: object,
) -> float:
    """Recover the physical multiplication factor from candidate production."""
    if not isinstance(policy, WielandtShiftSettings):
        return candidate_fission_production
    inverse_keff = policy.shift_inverse_keff + 1.0 / candidate_fission_production
    if not np.isfinite(inverse_keff) or inverse_keff <= 0.0:
        raise ValueError(
            "Wielandt-shifted k-effective iteration produced an unusable "
            "inverse multiplication factor"
        )
    return 1.0 / inverse_keff


def _check_keff_boundaries(
    context: _FiniteVolumeAssemblyContext,
) -> None:
    """Require boundary data with no additive term for criticality."""
    for face in context.exposed_faces:
        boundary = face.boundary
        if _is_homogeneous_boundary(boundary):
            continue
        topology = face.topology
        raise ValueError(
            "k-effective solve requires homogeneous boundary conditions "
            f"at axial_index={topology.axial_index}, "
            f"active_id={topology.active_id}, "
            f"direction={topology.direction!r}, "
            f"face_kind={topology.kind!r}, "
            f"boundary_kind={boundary.kind!r}"
        )


def _factor_loss_matrix(loss_matrix: csr_matrix, solve_name: str) -> SuperLU:
    """Factor one loss matrix and use the factorization as the solvability check."""
    try:
        return splu(loss_matrix.tocsc())
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"{solve_name} loss matrix is singular or unsolvable") from exc


def _prepare_keff_linear_solve(
    matrix: csr_matrix,
    linear_solve: LinearSolveSettings,
    solve_name: str,
) -> _PreparedLinearSolve:
    """Prepare one criticality inner-solve resource for this call only."""
    if isinstance(linear_solve, DirectLinearSolveSettings):
        return _PreparedLinearSolve(
            linear_solve=linear_solve,
            factorization=_factor_loss_matrix(matrix, solve_name),
        )
    return _PreparedLinearSolve(
        linear_solve=linear_solve,
        preconditioner=_build_gmres_preconditioner(matrix, linear_solve, solve_name),
    )


def _solve_linear_system(
    matrix: csr_matrix,
    rhs: np.ndarray,
    linear_solve: LinearSolveSettings,
    solve_name: str,
) -> tuple[np.ndarray, int]:
    """Execute one fixed-source direct or GMRES linear solve."""
    if isinstance(linear_solve, DirectLinearSolveSettings):
        return _solve_direct_loss_system(matrix, rhs, solve_name), 1
    preconditioner = _build_gmres_preconditioner(matrix, linear_solve, solve_name)
    return _solve_gmres_system(matrix, rhs, linear_solve, preconditioner, solve_name)


def _solve_prepared_linear_system(
    matrix: csr_matrix,
    rhs: np.ndarray,
    prepared: _PreparedLinearSolve,
    solve_name: str,
) -> tuple[np.ndarray, int]:
    """Execute one criticality inner solve with per-call reusable setup."""
    if isinstance(prepared.linear_solve, DirectLinearSolveSettings):
        if prepared.factorization is None:
            raise RuntimeError("direct criticality solve is missing its factorization")
        return (
            _solve_factorized_system(prepared.factorization, rhs, solve_name),
            1,
        )
    return _solve_gmres_system(
        matrix,
        rhs,
        prepared.linear_solve,
        prepared.preconditioner,
        solve_name,
    )


def _build_gmres_preconditioner(
    matrix: csr_matrix,
    settings: GmresLinearSolveSettings,
    solve_name: str,
) -> LinearOperator | None:
    """Build one checked left preconditioner for restarted GMRES."""
    if isinstance(settings.preconditioner, NoPreconditioner):
        return None
    if isinstance(settings.preconditioner, JacobiPreconditioner):
        diagonal = np.asarray(matrix.diagonal(), dtype=float)
        if not np.all(np.isfinite(diagonal)) or np.any(diagonal == 0.0):
            raise ValueError(
                f"{solve_name} Jacobi preconditioner requires finite nonzero "
                "diagonal entries"
            )
        inverse_diagonal = 1.0 / diagonal
        return LinearOperator(
            matrix.shape,
            matvec=lambda vector: inverse_diagonal * vector,
            dtype=float,
        )
    if isinstance(settings.preconditioner, IluPreconditioner):
        try:
            factorization = spilu(
                matrix.tocsc(),
                drop_tol=settings.preconditioner.drop_tolerance,
                fill_factor=settings.preconditioner.fill_factor,
            )
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                f"{solve_name} threshold-ILU preconditioner is unusable"
            ) from exc
        return LinearOperator(
            matrix.shape,
            matvec=lambda vector: np.asarray(factorization.solve(vector), dtype=float),
            dtype=float,
        )
    raise RuntimeError("unsupported GMRES preconditioner")


def _solve_gmres_system(
    matrix: csr_matrix,
    rhs: np.ndarray,
    settings: GmresLinearSolveSettings,
    preconditioner: LinearOperator | None,
    solve_name: str,
) -> tuple[np.ndarray, int]:
    """Run restarted GMRES from zero and return its completed iteration count."""
    krylov_iterations = 0

    def count_iteration(_residual: float) -> None:
        """Record one inner Krylov iteration reported by SciPy."""
        nonlocal krylov_iterations
        krylov_iterations += 1

    try:
        flux, status = gmres(
            matrix,
            rhs,
            x0=np.zeros_like(rhs),
            rtol=settings.relative_residual_tolerance,
            atol=0.0,
            restart=settings.restart,
            maxiter=settings.max_krylov_iterations,
            M=preconditioner,
            callback=count_iteration,
            callback_type="legacy",
        )
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"{solve_name} GMRES inner solve failed") from exc
    if status != 0:
        raise ValueError(
            f"{solve_name} GMRES did not converge within "
            f"max_krylov_iterations={settings.max_krylov_iterations}"
        )
    flux = np.asarray(flux, dtype=float)
    if not np.all(np.isfinite(flux)):
        raise ValueError(f"{solve_name} GMRES inner solve produced non-finite flux")
    return flux, krylov_iterations


def _solve_factorized_system(
    factorization: SuperLU, rhs: np.ndarray, solve_name: str
) -> np.ndarray:
    """Solve one right-hand side through an existing sparse factorization."""
    try:
        solution = np.asarray(factorization.solve(rhs), dtype=float)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"{solve_name} inner solve failed") from exc
    if not np.all(np.isfinite(solution)):
        raise ValueError(f"{solve_name} inner solve produced non-finite flux")
    return solution


def _clean_keff_flux(
    flux: np.ndarray,
    settings: KeffSettings,
    solve_label: str,
) -> np.ndarray:
    """Accept nonnegative criticality candidates up to numerical roundoff."""
    scale = float(np.max(np.abs(flux), initial=0.0))
    threshold = settings.flux_nonnegativity_tolerance * scale
    if np.any(flux < -threshold):
        raise ValueError(
            f"{solve_label} iteration produced a significantly negative flux"
        )
    cleaned = np.array(flux, copy=True)
    cleaned[cleaned < 0.0] = 0.0
    return cleaned


def _require_linear_residual(
    matrix: csr_matrix,
    flux: np.ndarray,
    rhs: np.ndarray,
    solve_name: str,
    settings: LinearSolveSettings,
) -> tuple[float, float]:
    """Require a cleaned linear solution to meet its true residual bound."""
    residual, relative_residual = _equation_residual(rhs, matrix @ flux)
    if relative_residual > settings.relative_residual_tolerance:
        raise ValueError(
            f"{solve_name} inner linear residual exceeds tolerance: "
            f"{relative_residual:.6e} > "
            f"{settings.relative_residual_tolerance:.6e}"
        )
    return residual, relative_residual


def _equation_residual(
    rhs_action: np.ndarray,
    lhs_action: np.ndarray,
) -> tuple[float, float]:
    """Return absolute and symmetric relative residual for two equation actions."""
    if not np.all(np.isfinite(rhs_action)) or not np.all(np.isfinite(lhs_action)):
        raise ValueError("equation residual requires finite balance terms")
    scale = max(
        float(np.max(np.abs(rhs_action), initial=0.0)),
        float(np.max(np.abs(lhs_action), initial=0.0)),
    )
    if scale == 0.0:
        return 0.0, 0.0
    scaled_rhs = rhs_action / scale
    scaled_lhs = lhs_action / scale
    residual = float(np.linalg.norm(scaled_rhs - scaled_lhs))
    denominator = float(np.linalg.norm(scaled_rhs) + np.linalg.norm(scaled_lhs))
    return residual * scale, residual / denominator


def _solve_direct_loss_system(
    loss_matrix: csr_matrix,
    rhs: np.ndarray,
    solve_name: str,
) -> np.ndarray:
    """Solve a direct loss system shared by fixed-source and keff modes."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=MatrixRankWarning)
        try:
            flux = np.asarray(spsolve(loss_matrix, rhs), dtype=float)
        except (MatrixRankWarning, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"{solve_name} loss matrix is singular or unsolvable"
            ) from exc
    if not np.all(np.isfinite(flux)):
        raise ValueError(f"{solve_name} loss matrix is singular or unsolvable")
    return flux


def _clean_fixed_source_flux(
    flux: np.ndarray, rhs: np.ndarray, settings: FixedSourceSettings
) -> np.ndarray:
    """Accept numerical roundoff only within the configured flux tolerance."""
    scale = float(np.max(np.abs(flux), initial=0.0))
    threshold = settings.flux_nonnegativity_tolerance * scale
    if np.any(flux < -threshold):
        raise ValueError("fixed-source solve produced a significantly negative flux")
    cleaned = np.array(flux, copy=True)
    cleaned[cleaned < 0.0] = 0.0
    if np.all(cleaned == 0.0) and np.any(rhs != 0.0):
        raise ValueError("fixed-source solve produced zero flux for a nonzero RHS")
    return cleaned


def _fixed_source_layer_group_balance(
    *,
    configuration: ProblemConfigurationSnapshot,
    cross_sections: CrossSectionData,
    context: _FiniteVolumeAssemblyContext,
    fission_matrix: csr_matrix,
    source_rhs: np.ndarray,
    boundary_rhs: np.ndarray,
    flux: np.ndarray,
) -> dict[str, tuple[np.ndarray, ...]]:
    """Return layer-resolved fixed-source balance terms."""
    common = _layer_balance_terms(cross_sections, context, flux)
    source_layers = context.layout.unpack(source_rhs)
    boundary_layers = context.layout.unpack(boundary_rhs)
    emission_layers = context.layout.unpack(fission_matrix @ flux)
    production_layers = context.layout.unpack(
        _fission_production_functional(configuration, cross_sections, context.layout)
        * flux
    )
    result = {
        "source": tuple(np.sum(values, axis=1) for values in source_layers),
        "boundary_source": tuple(np.sum(values, axis=1) for values in boundary_layers),
        "fission_production": tuple(
            np.sum(values, axis=1) for values in production_layers
        ),
        "fission_emission": tuple(np.sum(values, axis=1) for values in emission_layers),
        **common,
    }
    result["residual"] = tuple(
        result["source"][index]
        + result["boundary_source"][index]
        + result["fission_emission"][index]
        + common["net_scattering"][index]
        - common["absorption"][index]
        - common["radial_leakage"][index]
        - common["axial_leakage"][index]
        for index in range(len(context.layout.active_cells_by_layer))
    )
    return result


def _keff_layer_group_balance(
    problem: _KeffProblem,
    flux: np.ndarray,
    keff: float,
) -> dict[str, tuple[np.ndarray, ...]]:
    """Return layer-resolved criticality balance terms."""
    common = _layer_balance_terms(problem.cross_sections, problem.context, flux)
    layout = problem.layout
    production_layers = layout.unpack(problem.fission_production_functional * flux)
    source_layers = layout.unpack(problem.fission_matrix @ flux / keff)
    result = {
        "fission_production": tuple(
            np.sum(values, axis=1) for values in production_layers
        ),
        "keff_source": tuple(np.sum(values, axis=1) for values in source_layers),
        **common,
    }
    result["residual"] = tuple(
        result["keff_source"][index]
        + common["net_scattering"][index]
        - common["absorption"][index]
        - common["radial_leakage"][index]
        - common["axial_leakage"][index]
        for index in range(len(layout.active_cells_by_layer))
    )
    return result


def _layer_balance_terms(
    cross_sections: CrossSectionData,
    context: _FiniteVolumeAssemblyContext,
    flux: np.ndarray,
) -> dict[str, tuple[np.ndarray, ...]]:
    """Return shared material and directional-leakage layer balance terms."""
    flux_layers = context.layout.unpack(flux)
    absorption = []
    removal = []
    scattering_coupling = []
    radial_leakage = []
    axial_leakage = []
    directional_leakage = _directional_leakage_by_layer(context, flux)
    material_mesh = context.material_mesh
    for layer, flux_layer in zip(cross_sections.layers, flux_layers, strict=True):
        volume = material_mesh.cell_volume(layer.axial_index)
        absorption.append(np.sum(layer.sigma_a * volume * flux_layer, axis=1))
        removal_values = np.sum(layer.sigma_r * volume * flux_layer, axis=1)
        removal.append(removal_values)
        coupling = _layer_scattering_coupling(layer, flux_layer, volume)
        scattering_coupling.append(coupling)
        radial, axial = directional_leakage[layer.axial_index]
        radial_leakage.append(radial)
        axial_leakage.append(axial)
    return {
        "scattering_coupling": tuple(scattering_coupling),
        "removal": tuple(removal),
        "absorption": tuple(absorption),
        "radial_leakage": tuple(radial_leakage),
        "axial_leakage": tuple(axial_leakage),
        "net_scattering": tuple(
            coupling - (removed - absorbed)
            for coupling, removed, absorbed in zip(
                scattering_coupling, removal, absorption, strict=True
            )
        ),
    }


def _is_homogeneous_boundary(boundary: _ResolvedBoundaryCondition) -> bool:
    """Return whether group-resolved boundary data has no additive source."""
    if boundary.kind == "dirichlet":
        return bool(np.all(boundary.flux == 0.0))
    return bool(np.all(boundary.source == 0.0))


def _packed_cell_volumes(material_mesh, layout: _FiniteVolumeLayout) -> np.ndarray:
    """Return ragged node-major/group-fastest finite-volume weights."""
    weights = np.empty(layout.size, dtype=float)
    for axial_index, active_cells in enumerate(layout.active_cells_by_layer):
        volume = material_mesh.cell_volume(axial_index)
        for active_id in range(active_cells):
            for group in range(layout.groups):
                weights[layout.index(axial_index, active_id, group)] = volume
    return weights


def _contract_layer_group_balance(
    layer_group_balance: dict[str, tuple[np.ndarray, ...]],
) -> dict[str, np.ndarray]:
    """Sum layer-resolved group vectors into established group summaries."""
    return {
        name: np.sum(values, axis=0) for name, values in layer_group_balance.items()
    }


def _layer_scattering_coupling(
    layer: CrossSectionLayerData,
    flux_layer: np.ndarray,
    volume: float,
) -> np.ndarray:
    """Return destination-group scattering-neutron coupling for one layer."""
    transfer = np.empty(layer.groups, dtype=float)
    scattering_coupling = layer.scattering_coupling
    for group_to in range(layer.groups):
        transfer[group_to] = sum(
            np.sum(
                scattering_coupling[group_from, group_to]
                * volume
                * flux_layer[group_from]
            )
            for group_from in range(layer.groups)
        )
    return transfer


def _directional_leakage_by_layer(
    context: _FiniteVolumeAssemblyContext,
    flux: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return signed radial and axial outward leakage for all axial layers."""
    material_mesh = context.material_mesh
    layout = context.layout
    directional = [
        (np.zeros(layout.groups, dtype=float), np.zeros(layout.groups, dtype=float))
        for _ in layout.active_cells_by_layer
    ]
    for interface in context.internal_interfaces:
        if interface.radial:
            continue
        for group in range(layout.groups):
            primary = layout.index(
                interface.primary_axial_index, interface.primary_active_id, group
            )
            secondary = layout.index(
                interface.secondary_axial_index,
                interface.secondary_active_id,
                group,
            )
            contribution = _internal_interface_conductance(
                context, interface, group=group
            ) * (flux[primary] - flux[secondary])
            directional[interface.primary_axial_index][1][group] += contribution
            directional[interface.secondary_axial_index][1][group] -= contribution
    for face in context.exposed_faces:
        topology = face.topology
        radial = topology.direction in material_mesh.mesh.direction_labels
        target = directional[topology.axial_index][0 if radial else 1]
        layer = context.cross_sections.layer(topology.axial_index)
        for group in range(layout.groups):
            row = layout.index(topology.axial_index, topology.active_id, group)
            target[group] += _exposed_face_conductance(face, layer, group) * flux[row]
    return tuple(directional)


def _volume_weighted_norm(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the finite-volume norm induced by one node-major volume vector."""
    return float(norm(np.sqrt(weights) * values))
