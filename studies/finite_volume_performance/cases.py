"""The direct and GMRES/Jacobi linear-solve cases for the performance study."""

from __future__ import annotations

from dataclasses import asdict, replace

from morana import GmresLinearSolveSettings, JacobiPreconditioner

from studies.finite_volume_performance.workload import solve_settings

DIRECT_CASE_ID = "direct"
GMRES_JACOBI_CASE_ID = "gmres_jacobi"
CASE_IDS = (DIRECT_CASE_ID, GMRES_JACOBI_CASE_ID)


def settings_for(case_id: str):
    """Return the frozen criticality settings for one study case."""
    if case_id == DIRECT_CASE_ID:
        return solve_settings()
    if case_id == GMRES_JACOBI_CASE_ID:
        baseline = solve_settings()
        return replace(
            baseline,
            inner_linear_solve=GmresLinearSolveSettings(
                relative_residual_tolerance=(
                    baseline.inner_linear_solve.relative_residual_tolerance
                ),
                max_krylov_iterations=1_000,
                restart=50,
                preconditioner=JacobiPreconditioner(),
            ),
        )
    raise ValueError(f"case_id must be one of {CASE_IDS}")


def records() -> list[dict[str, object]]:
    """Return deterministic provenance records for the study cases."""
    result = []
    for case_id in CASE_IDS:
        settings = settings_for(case_id)
        settings_record = asdict(settings)
        settings_record["inner_linear_solve"]["strategy"] = (
            "direct" if case_id == DIRECT_CASE_ID else "gmres"
        )
        settings_record["eigenvalue_iteration"][
            "kind"
        ] = settings.eigenvalue_iteration.kind
        if case_id == GMRES_JACOBI_CASE_ID:
            settings_record["inner_linear_solve"]["preconditioner"]["kind"] = "jacobi"
        result.append({"case_id": case_id, "settings": settings_record})
    return result
