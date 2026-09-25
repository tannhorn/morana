"""Worker-local iterative measurements for the performance study."""

# This study observes private finite-volume stages. It remains separate from
# Morana's public solver API.
# pylint: disable=protected-access

from __future__ import annotations

from contextlib import ExitStack
import cProfile
import resource
import time
from typing import Any
from unittest.mock import patch

from morana import FissionSourceNormalization
from morana.solvers import finite_volume

from studies.finite_volume_performance import measurement
from studies.finite_volume_performance.cases import (
    BICGSTAB_JACOBI_CASE_ID,
    GMRES_JACOBI_CASE_ID,
    resolve_case,
)

_TIMING_FIELDS = (
    "loss_assembly",
    "fission_assembly",
    "linear_solve_setup",
    "linear_rhs_solves",
)
_CPU_FIELDS = tuple(
    f"{stage}_{kind}"
    for stage in ("linear_solve_setup", "linear_rhs_solves")
    for kind in ("user", "system")
)


class _Capture:
    """Collect iterative setup and solve timings alongside assembled matrices."""

    def __init__(self) -> None:
        self.wall_seconds = {name: 0.0 for name in _TIMING_FIELDS}
        self.cpu_seconds = {name: 0.0 for name in _CPU_FIELDS}
        self.rss: dict[str, int] = {}
        self.matrices: dict[str, dict[str, object]] = {}
        self.rhs_solve_count = 0
        self.krylov_iterations: list[int] = []

    def timed_assembly(self, name: str, function: Any) -> Any:
        """Wrap one matrix-assembly stage."""

        def wrapper(*args: object, **kwargs: object) -> Any:
            start = time.perf_counter_ns()
            value = function(*args, **kwargs)
            self.wall_seconds[name] += (time.perf_counter_ns() - start) / 1.0e9
            matrix_name = "loss" if name == "loss_assembly" else "fission"
            self.matrices[matrix_name] = measurement._sparse_record(value)
            checkpoint = name if name == "loss_assembly" else "fission_matrix_assembly"
            self.rss[checkpoint] = measurement._current_rss_bytes()
            return value

        return wrapper

    def timed_linear(self, name: str, function: Any, *, repeated: bool) -> Any:
        """Wrap iterative setup or repeated RHS work with CPU and wall timing."""

        def wrapper(*args: object, **kwargs: object) -> Any:
            usage_before = resource.getrusage(resource.RUSAGE_SELF)
            start = time.perf_counter_ns()
            value = function(*args, **kwargs)
            usage_after = resource.getrusage(resource.RUSAGE_SELF)
            self.wall_seconds[name] += (time.perf_counter_ns() - start) / 1.0e9
            self.cpu_seconds[f"{name}_user"] += (
                usage_after.ru_utime - usage_before.ru_utime
            )
            self.cpu_seconds[f"{name}_system"] += (
                usage_after.ru_stime - usage_before.ru_stime
            )
            if repeated:
                self.rhs_solve_count += 1
                self.krylov_iterations.append(int(value[1]))
            else:
                self.rss[name] = measurement._current_rss_bytes()
            return value

        return wrapper


def _instrumented_solve(
    configuration: Any,
    capture: _Capture,
    *,
    settings: Any,
    solve_attribute: str,
) -> Any:
    """Run one iterative solve with measurement wrappers."""
    wrappers = {
        "_assemble_loss_matrix": capture.timed_assembly(
            "loss_assembly", finite_volume._assemble_loss_matrix
        ),
        "_assemble_fission_matrix": capture.timed_assembly(
            "fission_assembly", finite_volume._assemble_fission_matrix
        ),
        "_build_iterative_preconditioner": capture.timed_linear(
            "linear_solve_setup",
            finite_volume._build_iterative_preconditioner,
            repeated=False,
        ),
        solve_attribute: capture.timed_linear(
            "linear_rhs_solves",
            getattr(finite_volume, solve_attribute),
            repeated=True,
        ),
    }
    with ExitStack() as stack:
        for name, wrapper in wrappers.items():
            stack.enter_context(patch.object(finite_volume, name, wrapper))
        return finite_volume.solve_keff(
            configuration,
            FissionSourceNormalization(rate=1.0),
            settings,
        )


def measure_gmres_jacobi(
    groups: int,
    axial_layers: int,
    *,
    requested_threads: int,
    kind: str,
    repetition: int | None,
    iteration_id: str = "power",
    shift_inverse_keff: float | None = None,
) -> dict[str, object]:
    """Measure one GMRES/Jacobi workload in the current fresh worker."""
    return _measure_iterative(
        groups,
        axial_layers,
        requested_threads=requested_threads,
        kind=kind,
        repetition=repetition,
        iteration_id=iteration_id,
        shift_inverse_keff=shift_inverse_keff,
        case_id=GMRES_JACOBI_CASE_ID,
        solve_attribute="_solve_gmres_system",
    )


def measure_bicgstab_jacobi(
    groups: int,
    axial_layers: int,
    *,
    requested_threads: int,
    kind: str,
    repetition: int | None,
    iteration_id: str = "power",
    shift_inverse_keff: float | None = None,
) -> dict[str, object]:
    """Measure one BiCGSTAB/Jacobi workload in the current fresh worker."""
    return _measure_iterative(
        groups,
        axial_layers,
        requested_threads=requested_threads,
        kind=kind,
        repetition=repetition,
        iteration_id=iteration_id,
        shift_inverse_keff=shift_inverse_keff,
        case_id=BICGSTAB_JACOBI_CASE_ID,
        solve_attribute="_solve_bicgstab_system",
    )


def _measure_iterative(
    groups: int,
    axial_layers: int,
    *,
    requested_threads: int,
    kind: str,
    repetition: int | None,
    iteration_id: str,
    shift_inverse_keff: float | None,
    case_id: str,
    solve_attribute: str,
) -> dict[str, object]:
    """Measure one checked iterative workload in the current fresh worker."""
    if kind not in {"measurement", "profile"}:
        raise ValueError("kind must be 'measurement' or 'profile'")
    settings, iteration = resolve_case(case_id, iteration_id, shift_inverse_keff)
    capture = _Capture()
    profile = cProfile.Profile() if kind == "profile" else None

    def solve(configuration: Any) -> tuple[Any, dict[str, int]]:
        try:
            if profile is not None:
                profile.enable()
            result = _instrumented_solve(
                configuration,
                capture,
                settings=settings,
                solve_attribute=solve_attribute,
            )
        finally:
            if profile is not None:
                profile.disable()
        capture.rss["completed_result"] = measurement._current_rss_bytes()
        return result, capture.rss

    result, common = measurement._measure_worker_call(groups, axial_layers, solve)
    report = result.execution_report
    if capture.rhs_solve_count != report.iterations:
        raise RuntimeError("iterative RHS-solve count does not match outer iterations")
    reported_krylov = [item.linear_solve.iterations for item in report.outer_iterations]
    if capture.krylov_iterations != reported_krylov:
        raise RuntimeError("captured Krylov counts do not match execution report")
    return measurement._successful_outcome(
        groups,
        axial_layers,
        case_id,
        iteration=iteration,
        requested_threads=requested_threads,
        kind=kind,
        repetition=repetition,
        common=common,
        solve_fields={
            "rss_checkpoints_bytes": capture.rss,
            "timings_seconds": capture.wall_seconds,
            "cpu_times_seconds": capture.cpu_seconds,
            "matrices": capture.matrices,
            "factorization": None,
            "outer_iterations": report.iterations,
            "krylov_iterations_by_outer": capture.krylov_iterations,
            "total_krylov_iterations": sum(capture.krylov_iterations),
            "linear_relative_residuals_by_outer": [
                item.linear_solve.true_relative_residual
                for item in report.outer_iterations
            ],
            "numerical_checks": measurement._numerical_record(result),
        },
        profile=profile,
    )
