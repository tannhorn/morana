"""Checked solver-and-ordering screen within the many-group study."""

# The screen deliberately observes private finite-volume stages and substitutes
# equivalent worker-local sparse factorizations.  No instrumentation becomes a
# Morana API.
# pylint: disable=duplicate-code,protected-access,too-many-lines
# pylint: disable=too-many-instance-attributes

from __future__ import annotations

from contextlib import ExitStack
import cProfile
import resource
import time
from typing import Any
from unittest.mock import patch

import numpy as np

from morana import FissionSourceNormalization
from morana.solvers import finite_volume

from examples.many_group_performance.cases import (
    SolverCase,
    solver_case,
)
from examples.many_group_performance import measurement
from examples.many_group_performance.orchestration import outcome_identifier

_TIMING_FIELDS = {
    "loss_assembly",
    "fission_assembly",
    "linear_solve_setup",
    "linear_rhs_solves",
}
_CPU_FIELDS = {
    "linear_solve_setup_user",
    "linear_solve_setup_system",
    "linear_rhs_solves_user",
    "linear_rhs_solves_system",
}


def packing_permutation(node_count: int, group_count: int) -> np.ndarray:
    """Return group-major new indices expressed as node-major old indices."""
    if node_count <= 0 or group_count <= 0:
        raise ValueError("packing dimensions must be positive")
    node_major = np.arange(node_count * group_count, dtype=np.int64)
    return node_major.reshape(node_count, group_count).T.ravel()


class _PermutedFactorization:
    """Expose a group-major factorization through node-major solve vectors."""

    def __init__(self, factorization: Any, new_to_old: np.ndarray) -> None:
        self._factorization = factorization
        self._new_to_old = new_to_old

    @property
    def L(self) -> Any:  # pylint: disable=invalid-name
        """Return the underlying lower factor."""
        return self._factorization.L

    @property
    def U(self) -> Any:  # pylint: disable=invalid-name
        """Return the underlying upper factor."""
        return self._factorization.U

    @property
    def perm_c(self) -> Any:
        """Return the underlying column permutation."""
        return self._factorization.perm_c

    @property
    def perm_r(self) -> Any:
        """Return the underlying row permutation."""
        return self._factorization.perm_r

    def solve(self, right_hand_side: np.ndarray) -> np.ndarray:
        """Permute a right-hand side and return its node-major solution."""
        solution = np.empty_like(right_hand_side)
        group_major = self._factorization.solve(right_hand_side[self._new_to_old])
        solution[self._new_to_old] = group_major
        return solution


class _Capture:
    """Collect diagnostic timings and retained sparse state."""

    def __init__(self) -> None:
        self.wall_seconds = {name: 0.0 for name in _TIMING_FIELDS}
        self.cpu_seconds = {name: 0.0 for name in _CPU_FIELDS}
        self.rss: dict[str, int] = {}
        self.matrices: dict[str, dict[str, object]] = {}
        self.factorization: Any | None = None
        self.packing_new_to_old: np.ndarray | None = None
        self.rhs_solve_count = 0
        self.krylov_iterations: list[int] = []

    def timed_assembly(self, name: str, function: Any) -> Any:
        """Wrap one matrix assembly stage."""

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

    def timed_linear(
        self, name: str, function: Any, *, gmres: bool | None = None
    ) -> Any:
        """Wrap setup or repeated RHS work with wall and process-CPU timing."""

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
            if gmres is not None:
                self.rhs_solve_count += 1
                if gmres:
                    self.krylov_iterations.append(int(value[1]))
            else:
                self.rss[name] = measurement._current_rss_bytes()
            return value

        return wrapper


def _direct_factorizer(case: SolverCase, capture: _Capture, groups: int) -> Any:
    """Return the requested worker-local SuperLU factorization function."""
    ordering = str(case.column_ordering)

    def factor(matrix: Any, solve_name: str) -> Any:
        del solve_name
        if case.packing == "group_major_node_fastest":
            new_to_old = packing_permutation(matrix.shape[0] // groups, groups)
            capture.packing_new_to_old = new_to_old
            matrix = matrix[new_to_old, :][:, new_to_old]
        factorization = finite_volume.splu(matrix.tocsc(), permc_spec=ordering)
        if capture.packing_new_to_old is not None:
            factorization = _PermutedFactorization(
                factorization, capture.packing_new_to_old
            )
        capture.factorization = factorization
        return factorization

    return factor


def _instrumented_solve(
    configuration: Any,
    case: SolverCase,
    capture: _Capture,
    groups: int,
) -> Any:
    """Run one public criticality solve with diagnostic wrappers installed."""
    wrappers = {
        "_assemble_loss_matrix": capture.timed_assembly(
            "loss_assembly", finite_volume._assemble_loss_matrix
        ),
        "_assemble_fission_matrix": capture.timed_assembly(
            "fission_assembly", finite_volume._assemble_fission_matrix
        ),
    }
    if case.strategy == "direct":
        wrappers["_factor_loss_matrix"] = capture.timed_linear(
            "linear_solve_setup", _direct_factorizer(case, capture, groups)
        )
        wrappers["_solve_factorized_system"] = capture.timed_linear(
            "linear_rhs_solves",
            finite_volume._solve_factorized_system,
            gmres=False,
        )
    else:
        wrappers["_build_gmres_preconditioner"] = capture.timed_linear(
            "linear_solve_setup", finite_volume._build_gmres_preconditioner
        )
        wrappers["_solve_gmres_system"] = capture.timed_linear(
            "linear_rhs_solves", finite_volume._solve_gmres_system, gmres=True
        )

    original_spilu = finite_volume.spilu

    def capture_spilu(*args: object, **kwargs: object) -> Any:
        factorization = original_spilu(*args, **kwargs)
        capture.factorization = factorization
        return factorization

    with ExitStack() as stack:
        for name, wrapper in wrappers.items():
            stack.enter_context(patch.object(finite_volume, name, wrapper))
        if case.case_id == "gmres_ilu":
            stack.enter_context(patch.object(finite_volume, "spilu", capture_spilu))
        return finite_volume.solve_keff(
            configuration,
            FissionSourceNormalization(rate=1.0),
            case.settings(),
        )


def _factorization_record(case: SolverCase, capture: _Capture) -> object:
    """Serialize complete or incomplete factor statistics and permutations."""
    if capture.factorization is None:
        return None
    factorization = capture.factorization
    sparse = measurement._factor_record(factorization)
    return {
        "kind": "complete" if case.strategy == "direct" else "incomplete",
        "lower": sparse["lower"],
        "upper": sparse["upper"],
        "row_permutation": [int(value) for value in factorization.perm_r],
        "column_permutation": [int(value) for value in factorization.perm_c],
        "packing_new_to_old": (
            None
            if capture.packing_new_to_old is None
            else [int(value) for value in capture.packing_new_to_old]
        ),
    }


def measure_solver_case(
    groups: int,
    axial_layers: int,
    *,
    case_id: str,
    requested_threads: int,
    kind: str,
    repetition: int | None,
) -> dict[str, object]:
    """Measure one solver-screen case in the current fresh worker."""
    if kind not in {"measurement", "profile"}:
        raise ValueError("kind must be 'measurement' or 'profile'")
    case = solver_case(case_id)
    capture = _Capture()
    profile = cProfile.Profile() if kind == "profile" else None

    def solve(configuration: Any) -> tuple[Any, dict[str, int]]:
        try:
            if profile is not None:
                profile.enable()
            result = _instrumented_solve(configuration, case, capture, groups)
        finally:
            if profile is not None:
                profile.disable()
        capture.rss["completed_result"] = measurement._current_rss_bytes()
        return result, capture.rss

    result, common = measurement._measure_worker_call(groups, axial_layers, solve)
    report = result.execution_report
    if capture.rhs_solve_count != report.iterations:
        raise RuntimeError("RHS-solve count does not match completed outer iterations")
    if case.strategy == "gmres":
        reported_krylov = [
            item.linear_solve.iterations for item in report.outer_iterations
        ]
        if capture.krylov_iterations != reported_krylov:
            raise RuntimeError("captured Krylov counts do not match execution report")
    factorization = _factorization_record(case, capture)
    outcome_id = outcome_identifier(
        groups,
        axial_layers,
        case_id,
        requested_threads=requested_threads,
        kind=kind,
        repetition=repetition,
    )
    return {
        "outcome_id": outcome_id,
        "case_id": case_id,
        "workload_id": f"g{groups}-z{axial_layers}",
        "kind": kind,
        "repetition": repetition,
        "requested_threads": requested_threads,
        "status": "success",
        **measurement._worker_fields(
            common,
            {
                "rss_checkpoints_bytes": capture.rss,
                "timings_seconds": capture.wall_seconds,
                "cpu_times_seconds": capture.cpu_seconds,
                "matrices": capture.matrices,
                "factorization": factorization,
                "outer_iterations": report.iterations,
                "numerical_checks": measurement._numerical_record(result),
                **(
                    {
                        "krylov_iterations_by_outer": capture.krylov_iterations,
                        "total_krylov_iterations": sum(capture.krylov_iterations),
                        "linear_relative_residuals_by_outer": [
                            item.linear_solve.true_relative_residual
                            for item in report.outer_iterations
                        ],
                    }
                    if case.strategy == "gmres"
                    else {}
                ),
            },
        ),
        "profile": (
            None
            if profile is None
            else {
                "profiler": "cProfile",
                "entries": measurement._profile_entries(profile),
            }
        ),
        "failure": None,
    }
