"""Compare supported finite-volume solver strategies on MMS level 1.

This compact reporting example reuses the independently manufactured
fixed-source and k-effective cases. Timings are informational and are never
used as acceptance criteria; residual closure and manufactured-reference
errors remain the verification evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from examples import fixed_source_mms, keff_mms
from examples._hex_z_mms import volume_weighted_relative_l2

from morana import FissionSourceNormalization
from morana.solvers.finite_volume import solve_fixed_source, solve_keff

REPRESENTATIVE_LEVEL = 1


@dataclass(frozen=True)
# The reporting table has one field per user-facing column.
# pylint: disable=too-many-instance-attributes
class ComparisonRow:
    """Retain one informational strategy-comparison measurement."""

    case: str
    strategy: str
    setup_seconds: float
    solve_seconds: float
    outer_iterations: int | None
    inner_iterations: int
    final_residual: float
    reference_error: float


def _fixed_source_rows() -> tuple[ComparisonRow, ...]:
    """Measure all supported fixed-source linear policies once."""
    rows = []
    for strategy, settings in fixed_source_mms.strategy_settings():
        started = perf_counter()
        configuration, exact_layers, _ = fixed_source_mms.build_configuration(
            REPRESENTATIVE_LEVEL
        )
        setup_seconds = perf_counter() - started
        started = perf_counter()
        result = solve_fixed_source(configuration, settings)
        solve_seconds = perf_counter() - started
        report = result.execution_report
        rows.append(
            ComparisonRow(
                case="fixed-source",
                strategy=strategy,
                setup_seconds=setup_seconds,
                solve_seconds=solve_seconds,
                outer_iterations=None,
                inner_iterations=report.iterations,
                final_residual=report.true_relative_residual,
                reference_error=volume_weighted_relative_l2(
                    configuration,
                    result,
                    exact_layers,
                ),
            )
        )
    return tuple(rows)


def _keff_rows() -> tuple[ComparisonRow, ...]:
    """Measure every maintained k-effective comparison policy once."""
    rows = []
    for strategy, settings in keff_mms.strategy_settings():
        started = perf_counter()
        configuration, _, _ = keff_mms.build_configuration(REPRESENTATIVE_LEVEL)
        setup_seconds = perf_counter() - started
        started = perf_counter()
        result = solve_keff(
            configuration,
            FissionSourceNormalization(rate=keff_mms.FISSION_SOURCE_RATE),
            settings,
        )
        solve_seconds = perf_counter() - started
        report = result.execution_report
        rows.append(
            ComparisonRow(
                case="k-effective",
                strategy=strategy,
                setup_seconds=setup_seconds,
                solve_seconds=solve_seconds,
                outer_iterations=report.iterations,
                inner_iterations=sum(
                    iteration.linear_solve.iterations
                    for iteration in report.outer_iterations
                ),
                final_residual=report.final_outer_iteration.keff_relative_residual,
                reference_error=abs(result.keff - keff_mms.TARGET_KEFF)
                / keff_mms.TARGET_KEFF,
            )
        )
    return tuple(rows)


def _print_rows(rows: tuple[ComparisonRow, ...]) -> None:
    """Print a stable compact table without making timing an assertion."""
    print(f"Representative MMS refinement level: {REPRESENTATIVE_LEVEL}")
    print(
        "case          strategy            setup [s]  solve [s]  outer  "
        "inner  final residual  reference error"
    )
    for row in rows:
        outer = "—" if row.outer_iterations is None else str(row.outer_iterations)
        print(
            f"{row.case:12}  {row.strategy:18}  {row.setup_seconds:9.3f}  "
            f"{row.solve_seconds:9.3f}  {outer:>5}  {row.inner_iterations:5d}  "
            f"{row.final_residual:14.6e}  {row.reference_error:15.6e}"
        )


def main() -> None:
    """Run one measured comparison for every maintained comparison policy."""
    _print_rows(_fixed_source_rows() + _keff_rows())


if __name__ == "__main__":
    main()
