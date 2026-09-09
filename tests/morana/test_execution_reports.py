"""Tests for internal execution-report reconstruction."""

import numpy as np
import pytest

from morana import (
    DirectLinearSolveSettings,
    KeffOuterIterationReport,
    KeffSolveReport,
    LinearSolveReport,
    PowerIterationSettings,
)


def _linear_report() -> LinearSolveReport:
    """Return one valid minimal direct-solve report."""
    return LinearSolveReport._from_validated(
        linear_solve=DirectLinearSolveSettings(),
        iterations=0,
        true_relative_residual=0.0,
    )


def _outer_report(iteration: int = 1) -> KeffOuterIterationReport:
    """Return one valid minimal outer-iteration report."""
    return KeffOuterIterationReport._from_validated(
        iteration=iteration,
        linear_solve=_linear_report(),
        keff=1.0,
        keff_change=0.0,
        flux_change=0.0,
        keff_relative_residual=0.0,
    )


def test_linear_solve_report_normalizes_numeric_subclasses() -> None:
    """Accepted numerical subclasses should be retained as built-in types."""
    report = LinearSolveReport._from_validated(
        linear_solve=DirectLinearSolveSettings(relative_residual_tolerance=0.5),
        iterations=np.int64(2),
        true_relative_residual=np.float64(0.25),
    )

    assert report.iterations == 2
    assert type(report.iterations) is int
    assert report.true_relative_residual == 0.25
    assert type(report.true_relative_residual) is float


@pytest.mark.parametrize(
    ("argument", "value", "exception"),
    [
        ("iterations", True, TypeError),
        ("true_relative_residual", "zero", TypeError),
        ("iterations", -1, ValueError),
        ("true_relative_residual", -1.0, ValueError),
    ],
)
def test_linear_solve_report_rejects_invalid_inputs(
    argument: str, value: object, exception: type[Exception]
) -> None:
    """Internal reconstruction should reject invalid scalar diagnostics."""
    arguments: dict[str, object] = {
        "linear_solve": DirectLinearSolveSettings(),
        "iterations": 0,
        "true_relative_residual": 0.0,
    }
    arguments[argument] = value

    with pytest.raises(exception):
        LinearSolveReport._from_validated(**arguments)  # type: ignore[arg-type]


def test_linear_solve_report_rejects_residual_above_policy_tolerance() -> None:
    """A completed report must be consistent with its selected policy."""
    with pytest.raises(ValueError, match="must not exceed"):
        LinearSolveReport._from_validated(
            linear_solve=DirectLinearSolveSettings(relative_residual_tolerance=1.0e-10),
            iterations=1,
            true_relative_residual=0.1,
        )


def test_outer_iteration_report_normalizes_numeric_subclasses() -> None:
    """Accepted outer diagnostics should be retained as built-in types."""
    report = KeffOuterIterationReport._from_validated(
        iteration=np.int64(1),
        linear_solve=_linear_report(),
        keff=np.float64(1.0),
        keff_change=np.float64(0.1),
        flux_change=np.float64(0.2),
        keff_relative_residual=np.float64(0.3),
    )

    assert type(report.iteration) is int
    assert all(
        type(value) is float
        for value in (
            report.keff,
            report.keff_change,
            report.flux_change,
            report.keff_relative_residual,
        )
    )


@pytest.mark.parametrize(
    ("argument", "value", "exception"),
    [
        ("iteration", False, TypeError),
        ("keff", "one", TypeError),
        ("iteration", 0, ValueError),
        ("keff", 0.0, ValueError),
        ("keff_change", -1.0, ValueError),
    ],
)
def test_outer_iteration_report_rejects_invalid_inputs(
    argument: str, value: object, exception: type[Exception]
) -> None:
    """Internal reconstruction should reject invalid outer diagnostics."""
    arguments: dict[str, object] = {
        "iteration": 1,
        "linear_solve": _linear_report(),
        "keff": 1.0,
        "keff_change": 0.0,
        "flux_change": 0.0,
        "keff_relative_residual": 0.0,
    }
    arguments[argument] = value

    with pytest.raises(exception):
        KeffOuterIterationReport._from_validated(**arguments)  # type: ignore[arg-type]


def test_execution_reports_reject_direct_construction() -> None:
    """Execution reports originate from solvers or checked archives only."""
    for report_type in (
        LinearSolveReport,
        KeffOuterIterationReport,
        KeffSolveReport,
    ):
        with pytest.raises(TypeError, match="retained by completed results"):
            report_type()


def test_keff_solve_report_rejects_invalid_eigenvalue_iteration() -> None:
    """Criticality histories retain a checked eigenvalue-iteration policy."""
    with pytest.raises(TypeError, match="eigenvalue_iteration"):
        KeffSolveReport._from_validated(
            (_outer_report(),), eigenvalue_iteration=None  # type: ignore[arg-type]
        )


def test_keff_solve_report_rejects_empty_or_nonconsecutive_history() -> None:
    """Criticality report histories must be nonempty and one-based."""
    with pytest.raises(ValueError, match="must not be empty"):
        KeffSolveReport._from_validated((), PowerIterationSettings())
    with pytest.raises(ValueError, match="consecutive one-based indices"):
        KeffSolveReport._from_validated((_outer_report(2),), PowerIterationSettings())
