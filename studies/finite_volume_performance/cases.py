"""Independent linear-solve and eigenvalue-iteration study policies."""

# Numerical imports are deliberately delayed until after worker resource and
# thread controls are established.
# pylint: disable=import-outside-toplevel

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace

DIRECT_CASE_ID = "direct"
GMRES_JACOBI_CASE_ID = "gmres_jacobi"
BICGSTAB_JACOBI_CASE_ID = "bicgstab_jacobi"
CASE_IDS = (DIRECT_CASE_ID, GMRES_JACOBI_CASE_ID, BICGSTAB_JACOBI_CASE_ID)
POWER_ITERATION_ID = "power"
WIELANDT_ITERATION_ID = "wielandt"
ITERATION_IDS = (POWER_ITERATION_ID, WIELANDT_ITERATION_ID)


def iteration_record(
    iteration_id: str, shift_inverse_keff: float | None = None
) -> dict[str, object]:
    """Return one checked ordinary or explicitly fixed-shift policy."""
    if iteration_id == POWER_ITERATION_ID:
        if shift_inverse_keff is not None:
            raise ValueError("a fixed shift applies only to Wielandt iteration")
        return {"kind": POWER_ITERATION_ID}
    if iteration_id == WIELANDT_ITERATION_ID:
        if shift_inverse_keff is None:
            raise ValueError("Wielandt iteration requires an explicit fixed shift")
        from morana import WielandtShiftSettings

        policy = WielandtShiftSettings(shift_inverse_keff)
        return {
            "kind": WIELANDT_ITERATION_ID,
            "shift_inverse_keff": policy.shift_inverse_keff,
        }
    raise ValueError(f"iteration_id must be one of {ITERATION_IDS}")


def iteration_identifier(iteration: Mapping[str, object]) -> str:
    """Return the complete stable identity of one checked iteration policy."""
    checked = iteration_record(
        str(iteration.get("kind")), iteration.get("shift_inverse_keff")
    )
    if checked["kind"] == POWER_ITERATION_ID:
        return POWER_ITERATION_ID
    return f"{WIELANDT_ITERATION_ID}-s{checked['shift_inverse_keff']}"


def _base_settings(case_id: str):
    """Return settings carrying only one frozen linear-solve case choice."""
    from morana import (
        BicgstabLinearSolveSettings,
        GmresLinearSolveSettings,
        JacobiPreconditioner,
    )
    from studies.finite_volume_performance.workload import solve_settings

    baseline = solve_settings()
    if case_id == DIRECT_CASE_ID:
        return baseline
    if case_id == GMRES_JACOBI_CASE_ID:
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
    if case_id == BICGSTAB_JACOBI_CASE_ID:
        return replace(
            baseline,
            inner_linear_solve=BicgstabLinearSolveSettings(
                relative_residual_tolerance=(
                    baseline.inner_linear_solve.relative_residual_tolerance
                ),
                max_krylov_iterations=1_000,
                preconditioner=JacobiPreconditioner(),
            ),
        )
    raise ValueError(f"case_id must be one of {CASE_IDS}")


def resolve_case(
    case_id: str,
    iteration_id: str = POWER_ITERATION_ID,
    shift_inverse_keff: float | None = None,
):
    """Return resolved settings and their checked iteration record."""
    from morana import PowerIterationSettings, WielandtShiftSettings

    baseline = _base_settings(case_id)
    policy = iteration_record(iteration_id, shift_inverse_keff)
    settings = replace(
        baseline,
        eigenvalue_iteration=(
            PowerIterationSettings()
            if policy["kind"] == POWER_ITERATION_ID
            else WielandtShiftSettings(policy["shift_inverse_keff"])
        ),
    )
    return settings, policy


def records() -> list[dict[str, object]]:
    """Return deterministic linear-solve provenance records for the study cases."""
    result = []
    for case_id in CASE_IDS:
        linear_solve = asdict(_base_settings(case_id).inner_linear_solve)
        strategy = {
            DIRECT_CASE_ID: "direct",
            GMRES_JACOBI_CASE_ID: "gmres",
            BICGSTAB_JACOBI_CASE_ID: "bicgstab",
        }[case_id]
        linear_solve["strategy"] = strategy
        if case_id in {GMRES_JACOBI_CASE_ID, BICGSTAB_JACOBI_CASE_ID}:
            linear_solve["preconditioner"]["kind"] = "jacobi"
        result.append({"case_id": case_id, "linear_solve": linear_solve})
    return result


def solve_controls_record() -> dict[str, object]:
    """Return the explicitly owned shared convergence and acceptance controls."""
    from studies.finite_volume_performance.workload import solve_settings

    settings = solve_settings()
    return {
        "max_outer_iterations": settings.max_outer_iterations,
        "keff_change_tolerance": settings.keff_change_tolerance,
        "flux_change_tolerance": settings.flux_change_tolerance,
        "keff_relative_residual_tolerance": (settings.keff_relative_residual_tolerance),
        "flux_nonnegativity_tolerance": settings.flux_nonnegativity_tolerance,
    }
