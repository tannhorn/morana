"""Immutable typed execution diagnostics retained by solver results."""

from __future__ import annotations

from dataclasses import dataclass
from morana._validation import (
    require_finite_nonnegative_real,
    require_finite_positive_real,
    require_nonnegative_integer,
    require_positive_integer,
)
from morana.solve_settings import (
    DirectLinearSolveSettings,
    EigenvalueIterationSettings,
    LinearPreconditioner,
    LinearSolveSettings,
    PowerIterationSettings,
    _check_eigenvalue_iteration,
)


@dataclass(frozen=True, init=False)
class LinearSolveReport:
    """Describe one completed finite-volume linear solve.

    Instances are retained by solver-produced results or restored from checked
    result archives. Direct construction is not supported.

    Attributes
    ----------
    linear_solve
        Immutable strategy and, for GMRES, preconditioner policy used by the
        completed solve.
    iterations
        Nonnegative number of direct or Krylov iterations reported for this
        solve.
    true_relative_residual
        Finite nonnegative true residual calculated by Morana from the final
        flux, operator, and right-hand side.

    """

    linear_solve: LinearSolveSettings
    iterations: int
    true_relative_residual: float

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject direct construction; reports belong to completed results."""
        _ = args, kwargs
        raise TypeError(
            "LinearSolveReport instances are retained by completed results or "
            "Result.load_from_disk()"
        )

    @classmethod
    def _from_validated(
        cls,
        *,
        linear_solve: LinearSolveSettings,
        iterations: int,
        true_relative_residual: float,
    ) -> "LinearSolveReport":
        """Build one checked report from internal solver or archive data."""
        iterations = require_nonnegative_integer("iterations", iterations)
        true_relative_residual = require_finite_nonnegative_real(
            "true_relative_residual", true_relative_residual
        )
        if true_relative_residual > linear_solve.relative_residual_tolerance:
            raise ValueError(
                "true_relative_residual must not exceed "
                "linear_solve.relative_residual_tolerance"
            )
        instance = object.__new__(cls)
        object.__setattr__(instance, "linear_solve", linear_solve)
        object.__setattr__(instance, "iterations", iterations)
        object.__setattr__(instance, "true_relative_residual", true_relative_residual)
        return instance

    @property
    def strategy(self) -> str:
        """Return the stable strategy identifier used by this solve."""
        return self.linear_solve.strategy

    @property
    def preconditioner(self) -> LinearPreconditioner | None:
        """Return the typed GMRES preconditioner, or ``None`` for direct solves."""
        if isinstance(self.linear_solve, DirectLinearSolveSettings):
            return None
        return self.linear_solve.preconditioner


@dataclass(frozen=True, init=False)
class KeffOuterIterationReport:
    """Describe one completed finite-volume criticality outer iteration.

    Instances are retained by solver-produced criticality results or restored
    from checked result archives. Direct construction is not supported.

    Attributes
    ----------
    iteration
        Positive one-based power-iteration index.
    linear_solve
        Completed loss-system solve nested in this outer iteration.
    keff
        Finite positive multiplication-factor estimate.
    keff_change
        Finite nonnegative relative multiplication-factor change.
    flux_change
        Finite nonnegative volume-weighted normalized-flux change.
    keff_relative_residual
        Finite nonnegative relative eigenvalue-equation residual.

    """

    iteration: int
    linear_solve: LinearSolveReport
    keff: float
    keff_change: float
    flux_change: float
    keff_relative_residual: float

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject direct construction; reports belong to completed results."""
        _ = args, kwargs
        raise TypeError(
            "KeffOuterIterationReport instances are retained by completed "
            "results or Result.load_from_disk()"
        )

    @classmethod
    def _from_validated(
        cls,
        *,
        iteration: int,
        linear_solve: LinearSolveReport,
        keff: float,
        keff_change: float,
        flux_change: float,
        keff_relative_residual: float,
    ) -> "KeffOuterIterationReport":
        """Build one checked record from internal solver or archive data."""
        iteration = require_positive_integer("iteration", iteration)
        keff = require_finite_positive_real("keff", keff)
        keff_change = require_finite_nonnegative_real("keff_change", keff_change)
        flux_change = require_finite_nonnegative_real("flux_change", flux_change)
        keff_relative_residual = require_finite_nonnegative_real(
            "keff_relative_residual", keff_relative_residual
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "iteration", iteration)
        object.__setattr__(instance, "linear_solve", linear_solve)
        object.__setattr__(instance, "keff", keff)
        object.__setattr__(instance, "keff_change", keff_change)
        object.__setattr__(instance, "flux_change", flux_change)
        object.__setattr__(instance, "keff_relative_residual", keff_relative_residual)
        return instance


@dataclass(frozen=True, init=False)
class KeffSolveReport:
    """Retain completed records for every criticality outer iteration.

    Instances are retained by solver-produced criticality results or restored
    from checked result archives. Direct construction is not supported.

    Attributes
    ----------
    outer_iterations
        Nonempty consecutive one-based outer-iteration records.
    eigenvalue_iteration
        Immutable ordinary-power or fixed-Wielandt-shift policy used for the
        recorded criticality solve.

    iterations
        Number of completed criticality outer iterations.
    final_outer_iteration
        Final completed outer-iteration record.

    """

    outer_iterations: tuple[KeffOuterIterationReport, ...]
    eigenvalue_iteration: EigenvalueIterationSettings = PowerIterationSettings()

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject direct construction; reports belong to completed results."""
        _ = args, kwargs
        raise TypeError(
            "KeffSolveReport instances are retained by completed results or "
            "Result.load_from_disk()"
        )

    @classmethod
    def _from_validated(
        cls,
        outer_iterations: tuple[KeffOuterIterationReport, ...],
        eigenvalue_iteration: EigenvalueIterationSettings,
    ) -> "KeffSolveReport":
        """Build one checked history from internal solver or archive data."""
        if not outer_iterations:
            raise ValueError("outer_iterations must not be empty")
        expected_indices = tuple(range(1, len(outer_iterations) + 1))
        observed_indices = tuple(report.iteration for report in outer_iterations)
        if observed_indices != expected_indices:
            raise ValueError("outer_iterations must have consecutive one-based indices")
        _check_eigenvalue_iteration("eigenvalue_iteration", eigenvalue_iteration)
        instance = object.__new__(cls)
        object.__setattr__(instance, "outer_iterations", outer_iterations)
        object.__setattr__(instance, "eigenvalue_iteration", eigenvalue_iteration)
        return instance

    @property
    def iterations(self) -> int:
        """Return the number of completed criticality outer iterations."""
        return len(self.outer_iterations)

    @property
    def final_outer_iteration(self) -> KeffOuterIterationReport:
        """Return the final completed criticality outer-iteration record."""
        return self.outer_iterations[-1]
