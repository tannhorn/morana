"""Frozen solver cases for the many-group performance study."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from morana import (
    GmresLinearSolveSettings,
    IluPreconditioner,
    JacobiPreconditioner,
    NoPreconditioner,
)

from studies.finite_volume_performance.workload import solve_settings

REFERENCE_CASE_ID = "direct_node_colamd"


@dataclass(frozen=True)
class SolverCase:
    """Describe one frozen solver-and-ordering case."""

    case_id: str
    packing: str
    column_ordering: str | None
    preconditioner: object | None = None

    @property
    def strategy(self) -> str:
        """Return the inner linear-solve strategy name."""
        return "direct" if self.preconditioner is None else "gmres"

    def settings(self) -> Any:
        """Return complete criticality settings for this case."""
        baseline = solve_settings()
        if self.strategy == "direct":
            inner = baseline.inner_linear_solve
        else:
            inner = GmresLinearSolveSettings(
                relative_residual_tolerance=(
                    baseline.inner_linear_solve.relative_residual_tolerance
                ),
                max_krylov_iterations=1_000,
                restart=50,
                preconditioner=self.preconditioner,
            )
        return replace(baseline, inner_linear_solve=inner)

    def record(self) -> dict[str, object]:
        """Return the deterministic JSON definition for this case."""
        settings = self.settings()
        settings_record = asdict(settings)
        settings_record["inner_linear_solve"]["strategy"] = self.strategy
        settings_record["eigenvalue_iteration"][
            "kind"
        ] = settings.eigenvalue_iteration.kind
        if self.preconditioner is not None:
            settings_record["inner_linear_solve"]["preconditioner"][
                "kind"
            ] = self.preconditioner.kind
        return {
            "case_id": self.case_id,
            "strategy": self.strategy,
            "packing": self.packing,
            "column_ordering": self.column_ordering,
            "settings": settings_record,
        }


_CASES = (
    SolverCase(REFERENCE_CASE_ID, "node_major_group_fastest", "COLAMD"),
    SolverCase("direct_node_mmd_ata", "node_major_group_fastest", "MMD_ATA"),
    SolverCase(
        "direct_node_mmd_at_plus_a",
        "node_major_group_fastest",
        "MMD_AT_PLUS_A",
    ),
    SolverCase("direct_group_colamd", "group_major_node_fastest", "COLAMD"),
    SolverCase("gmres_none", "node_major_group_fastest", None, NoPreconditioner()),
    SolverCase(
        "gmres_jacobi", "node_major_group_fastest", None, JacobiPreconditioner()
    ),
    SolverCase("gmres_ilu", "node_major_group_fastest", None, IluPreconditioner()),
)


def solver_case(case_id: str) -> SolverCase:
    """Return one frozen solver case by identifier."""
    for case in _CASES:
        if case.case_id == case_id:
            return case
    raise ValueError(f"unknown solver case {case_id!r}")


def case_records() -> list[dict[str, object]]:
    """Return all frozen solver-case definitions in protocol order."""
    return [case.record() for case in _CASES]
