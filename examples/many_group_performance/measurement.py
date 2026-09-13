"""Worker-local measurement of the finite-volume criticality solve."""

# This maintained benchmark deliberately observes private finite-volume stages.
# It fails on call-graph drift instead of exposing instrumentation in Morana.
# pylint: disable=protected-access,too-many-lines

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
import cProfile
from dataclasses import asdict
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import pstats
import resource
import subprocess
import time
from typing import Any
from unittest.mock import patch

import numpy as np
import scipy

from morana import FissionSourceNormalization
from morana.results import KeffBalance, Result
from morana.solvers import finite_volume

from examples.many_group_performance.orchestration import (
    THREAD_ENVIRONMENT_VARIABLES,
)
from examples.many_group_performance.results import STAGE_NAMES
from examples.many_group_performance.workload import (
    FROZEN_ARRAY_DIGESTS,
    build_configuration,
    solve_settings,
)

_EXPECTED_CALLS = {
    "configuration_snapshot": 1,
    "cross_section_extraction": 1,
    "layout": 1,
    "topology": 1,
    "boundary_check": 1,
    "loss_matrix": 1,
    "fission_matrix": 1,
    "fission_functional": 1,
    "factorization": 1,
    "volume_weights": 1,
    "iteration": 1,
    "normalization": 1,
    "balance_terms": 1,
    "balance_contraction": 1,
    "balance_value": 1,
    "result": 1,
}
_PROFILE_STAGE_BY_FUNCTION = {
    "_configuration_snapshot_for_solve": "snapshot_and_input_extraction",
    "extract_cross_section_data": "snapshot_and_input_extraction",
    "_checked_layout": "layout_and_topology",
    "_finite_volume_assembly_context_from_layout": "layout_and_topology",
    "_resolve_finite_volume_interfaces": "layout_and_topology",
    "_assemble_loss_matrix": "loss_assembly",
    "_assemble_fission_matrix": "fission_assembly",
    "_fission_production_functional": "fission_assembly",
    "_factor_loss_matrix": "direct_factorization",
    "splu": "direct_factorization",
    "_iterate_keff": "eigenvalue_iteration",
    "_solve_factorized_system": "eigenvalue_iteration",
    "_normalize_keff_flux": "normalization_balance_and_result",
    "_keff_layer_group_balance": "normalization_balance_and_result",
}


def normalize_peak_rss_bytes(raw_value: int | float, system: str) -> int:
    """Normalize ``resource.ru_maxrss`` to bytes for a supported platform."""
    value = int(raw_value)
    if value < 0:
        raise ValueError("peak RSS must be nonnegative")
    if system == "Linux":
        return value * 1024
    if system == "Darwin":
        return value
    raise RuntimeError(f"peak RSS measurement is unsupported on {system}")


def _current_rss_bytes(system: str | None = None) -> int:
    """Return current worker RSS, failing rather than changing metrics silently."""
    system = platform.system() if system is None else system
    if system == "Linux":
        try:
            fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
            return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, IndexError, ValueError) as exc:
            raise RuntimeError(
                "current RSS is unavailable from /proc/self/statm"
            ) from exc
    if system == "Darwin":
        return normalize_peak_rss_bytes(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, system
        )
    raise RuntimeError(f"current RSS measurement is unsupported on {system}")


def _git(*arguments: str) -> str | None:
    """Return stripped Git output, or ``None`` outside a usable worktree."""
    completed = subprocess.run(
        ("git", *arguments), check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _repository_record() -> dict[str, object]:
    """Capture repository identity and tracked-worktree state."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    changes = [] if status is None or not status else status.splitlines()
    return {
        "commit": _git("rev-parse", "HEAD"),
        "tracked_worktree_state": (
            "unknown" if status is None else ("clean" if not changes else "dirty")
        ),
        "tracked_changes": changes,
    }


def _environment_record() -> dict[str, object]:
    """Capture the execution environment visible to this worker."""
    return {
        "operating_system": platform.platform(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "thread_environment": {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES
        },
    }


def _sparse_record(matrix: Any) -> dict[str, object]:
    """Return retained-array statistics for one SciPy sparse matrix."""
    return {
        "shape": list(matrix.shape),
        "stored_nonzeros": int(matrix.nnz),
        "data_bytes": int(matrix.data.nbytes),
        "indices_bytes": int(matrix.indices.nbytes),
        "indptr_bytes": int(matrix.indptr.nbytes),
    }


def _factor_record(factorization: Any) -> dict[str, object]:
    """Return retained sparse statistics for one SuperLU factorization."""
    lower = _sparse_record(factorization.L)
    upper = _sparse_record(factorization.U)
    return {
        "lower": lower,
        "upper": upper,
    }


class _Recorder:
    """Accumulate exclusive stage intervals and captured solver state."""

    def __init__(self) -> None:
        self.stage_nanoseconds = {name: 0 for name in STAGE_NAMES}
        self.calls = {name: 0 for name in _EXPECTED_CALLS}
        self.checkpoints: dict[str, int] = {}
        self.matrices: dict[str, dict[str, object]] = {}
        self.factorization_handle: Any | None = None
        self.factorized_rhs_solves = 0

    def timed(
        self,
        *,
        stage: str,
        call: str,
        function: Callable[..., Any],
        checkpoint: str | None = None,
        capture: Callable[[Any], None] | None = None,
    ) -> Callable[..., Any]:
        """Return a wrapper that records one nonnested private call."""

        def wrapper(*args: object, **kwargs: object) -> Any:
            start = time.perf_counter_ns()
            value = function(*args, **kwargs)
            self.stage_nanoseconds[stage] += time.perf_counter_ns() - start
            self.calls[call] += 1
            if capture is not None:
                capture(value)
            if checkpoint is not None:
                self.checkpoints[checkpoint] = _current_rss_bytes()
            return value

        return wrapper

    def counted(self, function: Callable[..., Any]) -> Callable[..., Any]:
        """Return an untimed call-counting wrapper for a nested operation."""

        def wrapper(*args: object, **kwargs: object) -> Any:
            self.factorized_rhs_solves += 1
            return function(*args, **kwargs)

        return wrapper

    def require_expected_calls(self) -> None:
        """Fail clearly when solver-private call-graph assumptions drift."""
        mismatches = {
            name: (self.calls[name], expected)
            for name, expected in _EXPECTED_CALLS.items()
            if self.calls[name] != expected
        }
        if mismatches:
            raise RuntimeError(
                f"benchmark instrumentation call graph changed: {mismatches}"
            )


def _instrumented_solve(configuration: Any, recorder: _Recorder) -> Result:
    """Call public ``solve_keff`` while observing worker-local private stages."""
    original_balance = KeffBalance
    original_result = Result

    class TimedBalance:
        """Worker-local proxy for the internal criticality balance factory."""

        _from_validated = staticmethod(
            recorder.timed(
                stage="normalization_balance_and_result",
                call="balance_value",
                function=original_balance._from_validated,
            )
        )

    class TimedResult:
        """Worker-local proxy for the internal result factory."""

        _from_validated = staticmethod(
            recorder.timed(
                stage="normalization_balance_and_result",
                call="result",
                function=original_result._from_validated,
                checkpoint="completed_result",
            )
        )

    def capture_loss(matrix: Any) -> None:
        recorder.matrices["loss"] = _sparse_record(matrix)

    def capture_fission(matrix: Any) -> None:
        recorder.matrices["fission"] = _sparse_record(matrix)

    def capture_factor(factorization: Any) -> None:
        # Retain the already-created factorization without materializing its L
        # and U views inside the solve's peak-RSS measurement window.
        recorder.factorization_handle = factorization

    wrappers = {
        "_configuration_snapshot_for_solve": recorder.timed(
            stage="snapshot_and_input_extraction",
            call="configuration_snapshot",
            function=finite_volume._configuration_snapshot_for_solve,
        ),
        "extract_cross_section_data": recorder.timed(
            stage="snapshot_and_input_extraction",
            call="cross_section_extraction",
            function=finite_volume.extract_cross_section_data,
            checkpoint="snapshot_and_input_extraction",
        ),
        "_checked_layout": recorder.timed(
            stage="layout_and_topology",
            call="layout",
            function=finite_volume._checked_layout,
        ),
        "_finite_volume_assembly_context_from_layout": recorder.timed(
            stage="layout_and_topology",
            call="topology",
            function=finite_volume._finite_volume_assembly_context_from_layout,
        ),
        "_check_keff_boundaries": recorder.timed(
            stage="layout_and_topology",
            call="boundary_check",
            function=finite_volume._check_keff_boundaries,
            checkpoint="topology_preparation",
        ),
        "_packed_cell_volumes": recorder.timed(
            stage="layout_and_topology",
            call="volume_weights",
            function=finite_volume._packed_cell_volumes,
        ),
        "_assemble_loss_matrix": recorder.timed(
            stage="loss_assembly",
            call="loss_matrix",
            function=finite_volume._assemble_loss_matrix,
            checkpoint="loss_assembly",
            capture=capture_loss,
        ),
        "_assemble_fission_matrix": recorder.timed(
            stage="fission_assembly",
            call="fission_matrix",
            function=finite_volume._assemble_fission_matrix,
            checkpoint="fission_matrix_assembly",
            capture=capture_fission,
        ),
        "_fission_production_functional": recorder.timed(
            stage="fission_assembly",
            call="fission_functional",
            function=finite_volume._fission_production_functional,
        ),
        "_factor_loss_matrix": recorder.timed(
            stage="direct_factorization",
            call="factorization",
            function=finite_volume._factor_loss_matrix,
            checkpoint="factorization",
            capture=capture_factor,
        ),
        "_solve_factorized_system": recorder.counted(
            finite_volume._solve_factorized_system
        ),
        "_iterate_keff": recorder.timed(
            stage="eigenvalue_iteration",
            call="iteration",
            function=finite_volume._iterate_keff,
        ),
        "_normalize_keff_flux": recorder.timed(
            stage="normalization_balance_and_result",
            call="normalization",
            function=finite_volume._normalize_keff_flux,
        ),
        "_keff_layer_group_balance": recorder.timed(
            stage="normalization_balance_and_result",
            call="balance_terms",
            function=finite_volume._keff_layer_group_balance,
        ),
        "_contract_layer_group_balance": recorder.timed(
            stage="normalization_balance_and_result",
            call="balance_contraction",
            function=finite_volume._contract_layer_group_balance,
        ),
    }
    patches = [
        patch.object(finite_volume, name, value) for name, value in wrappers.items()
    ]
    patches.extend(
        (
            patch.object(finite_volume, "KeffBalance", TimedBalance),
            patch.object(finite_volume, "Result", TimedResult),
        )
    )
    with ExitStack() as stack:
        for active_patch in patches:
            stack.enter_context(active_patch)
        return finite_volume.solve_keff(
            configuration,
            FissionSourceNormalization(rate=1.0),
            solve_settings(),
        )


def _profile_entries(profile: cProfile.Profile) -> list[dict[str, object]]:
    """Return every profiler entry with deterministic cumulative-time ordering."""
    statistics = pstats.Stats(profile)
    entries = []
    for (filename, line, function), values in statistics.stats.items():
        primitive_calls, total_calls, total_time, cumulative_time, _ = values
        entries.append(
            {
                "file": filename,
                "line": line,
                "function": function,
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "total_time_seconds": total_time,
                "cumulative_time_seconds": cumulative_time,
                "stage": _PROFILE_STAGE_BY_FUNCTION.get(function, "uncategorized"),
            }
        )
    return sorted(
        entries,
        key=lambda entry: (
            -float(entry["cumulative_time_seconds"]),
            str(entry["file"]),
            int(entry["line"]),
            str(entry["function"]),
        ),
    )


def _settings_record() -> dict[str, object]:
    """Serialize every frozen direct-power control explicitly."""
    settings = solve_settings()
    record = asdict(settings)
    record["inner_linear_solve"]["strategy"] = settings.inner_linear_solve.strategy
    record["eigenvalue_iteration"]["kind"] = settings.eigenvalue_iteration.kind
    return record


def _numerical_record(result: Result) -> dict[str, object]:
    """Return final convergence and scalar-balance checks."""
    final = result.execution_report.final_outer_iteration
    scalar = result.balance.scalar
    lhs = scalar["absorption"] + scalar["radial_leakage"] + scalar["axial_leakage"]
    rhs = scalar["keff_source"] + scalar["net_scattering"]
    scale = abs(lhs) + abs(rhs)
    return {
        "keff": result.keff,
        "minimum_flux": min(float(np.min(layer)) for layer in result.flux),
        "final_keff_relative_residual": final.keff_relative_residual,
        "scalar_balance_residual": scalar["residual"],
        "scalar_balance_relative_closure": abs(scalar["residual"]) / scale,
    }


def measure_workload(
    groups: int,
    axial_layers: int,
    *,
    requested_threads: int,
    kind: str,
    repetition: int | None,
) -> dict[str, object]:
    """Measure one workload in the current fresh worker process."""
    if kind not in {"measurement", "profile"}:
        raise ValueError("kind must be 'measurement' or 'profile'")
    started_at_utc = datetime.now(timezone.utc).isoformat()
    repository = _repository_record()
    environment = _environment_record()
    imported_rss = _current_rss_bytes()
    configuration_start = time.perf_counter_ns()
    configuration = build_configuration(groups, axial_layers).snapshot()
    configuration_seconds = (time.perf_counter_ns() - configuration_start) / 1.0e9
    configuration_rss = _current_rss_bytes()

    recorder = _Recorder()
    profile = cProfile.Profile() if kind == "profile" else None
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    solve_start = time.perf_counter_ns()
    try:
        if profile is not None:
            profile.enable()
        result = _instrumented_solve(configuration, recorder)
    finally:
        if profile is not None:
            profile.disable()
    solve_seconds = (time.perf_counter_ns() - solve_start) / 1.0e9
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    recorder.require_expected_calls()

    user_seconds = usage_after.ru_utime - usage_before.ru_utime
    system_seconds = usage_after.ru_stime - usage_before.ru_stime
    peak_rss = max(
        normalize_peak_rss_bytes(usage_after.ru_maxrss, platform.system()),
        imported_rss,
        configuration_rss,
        *recorder.checkpoints.values(),
    )
    if recorder.factorization_handle is None:
        raise RuntimeError("benchmark did not retain the direct factorization")
    # Freeze peak RSS above, then inspect L and U: SciPy may materialize sparse
    # views while exposing these statistics, which is observer overhead rather
    # than solver memory.
    factorization_record = _factor_record(recorder.factorization_handle)
    recorder.factorization_handle = None
    stages = {name: recorder.stage_nanoseconds[name] / 1.0e9 for name in STAGE_NAMES}
    stage_total = sum(stages.values())
    if solve_seconds - stage_total < -1.0e-9:
        raise RuntimeError("exclusive stage times exceed end-to-end solve time")
    report = result.execution_report
    if recorder.factorized_rhs_solves != report.iterations:
        raise RuntimeError(
            "direct RHS-solve count does not match completed outer iterations"
        )
    observation_id = (
        f"g{groups}-z{axial_layers}-t{requested_threads}-{kind}-"
        f"{'none' if repetition is None else repetition}"
    )
    return {
        "observation_id": observation_id,
        "workload_id": f"g{groups}-z{axial_layers}",
        "kind": kind,
        "repetition": repetition,
        "requested_threads": requested_threads,
        "started_at_utc": started_at_utc,
        "repository": repository,
        "environment": environment,
        "configuration": {
            "wall_time_seconds": configuration_seconds,
            "import_baseline_rss_bytes": imported_rss,
            "completed_rss_bytes": configuration_rss,
        },
        "solve": {
            "wall_time_seconds": solve_seconds,
            "user_cpu_time_seconds": user_seconds,
            "system_cpu_time_seconds": system_seconds,
            "peak_rss_bytes": peak_rss,
            "rss_checkpoints_bytes": recorder.checkpoints,
            "stages_seconds": stages,
            "matrices": recorder.matrices,
            "factorization": factorization_record,
            "outer_iterations": report.iterations,
            "numerical_checks": _numerical_record(result),
        },
        "profile": (
            None
            if profile is None
            else {
                "profiler": "cProfile",
                "entries": _profile_entries(profile),
            }
        ),
    }


def workload_record(groups: int, axial_layers: int) -> dict[str, object]:
    """Return the frozen workload definition referenced by observations."""
    cells = 61 * axial_layers
    return {
        "workload_id": f"g{groups}-z{axial_layers}",
        "groups": groups,
        "axial_layers": axial_layers,
        "active_cells": cells,
        "unknowns": cells * groups,
        "array_digests": dict(FROZEN_ARRAY_DIGESTS[groups]),
        "settings": _settings_record(),
    }
