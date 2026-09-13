"""Checked JSON records for the many-group performance study."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
import json
import math
from pathlib import Path
import re
from statistics import median
from typing import Any

from examples.many_group_performance.orchestration import (
    THREAD_ENVIRONMENT_VARIABLES,
)

STAGE_NAMES = (
    "snapshot_and_input_extraction",
    "layout_and_topology",
    "loss_assembly",
    "fission_assembly",
    "direct_factorization",
    "eigenvalue_iteration",
    "normalization_balance_and_result",
)
RSS_CHECKPOINT_NAMES = (
    "snapshot_and_input_extraction",
    "topology_preparation",
    "loss_assembly",
    "fission_matrix_assembly",
    "factorization",
    "completed_result",
)

_DOCUMENT_FIELDS = {"workloads", "observations"}
_WORKLOAD_FIELDS = {
    "workload_id",
    "groups",
    "axial_layers",
    "active_cells",
    "unknowns",
    "array_digests",
    "settings",
}
_DIGEST_FIELDS = {"diffusion", "absorption", "scattering", "fission_transfer"}
_SETTINGS_FIELDS = {
    "inner_linear_solve",
    "max_outer_iterations",
    "keff_change_tolerance",
    "flux_change_tolerance",
    "keff_relative_residual_tolerance",
    "flux_nonnegativity_tolerance",
    "eigenvalue_iteration",
}
_OBSERVATION_FIELDS = {
    "observation_id",
    "workload_id",
    "kind",
    "repetition",
    "requested_threads",
    "started_at_utc",
    "repository",
    "environment",
    "resource_limits",
    "configuration",
    "solve",
    "profile",
}
_REPOSITORY_FIELDS = {"commit", "tracked_worktree_state", "tracked_changes"}
_ENVIRONMENT_FIELDS = {
    "operating_system",
    "processor",
    "logical_cpu_count",
    "python_version",
    "numpy_version",
    "scipy_version",
    "thread_environment",
}
_THREAD_FIELDS = set(THREAD_ENVIRONMENT_VARIABLES)
_RESOURCE_LIMIT_FIELDS = {
    "timeout_seconds",
    "address_space_bytes",
    "captured_output_bytes_per_stream",
}
_CONFIGURATION_FIELDS = {
    "wall_time_seconds",
    "import_baseline_rss_bytes",
    "completed_rss_bytes",
}
_SOLVE_FIELDS = {
    "wall_time_seconds",
    "user_cpu_time_seconds",
    "system_cpu_time_seconds",
    "peak_rss_bytes",
    "rss_checkpoints_bytes",
    "stages_seconds",
    "matrices",
    "factorization",
    "outer_iterations",
    "numerical_checks",
}
_STAGE_FIELDS = set(STAGE_NAMES)
_RSS_CHECKPOINT_FIELDS = set(RSS_CHECKPOINT_NAMES)
_SPARSE_FIELDS = {
    "shape",
    "stored_nonzeros",
    "data_bytes",
    "indices_bytes",
    "indptr_bytes",
}
_NUMERICAL_FIELDS = {
    "keff",
    "minimum_flux",
    "final_keff_relative_residual",
    "scalar_balance_residual",
    "scalar_balance_relative_closure",
}
_PROFILE_ENTRY_FIELDS = {
    "file",
    "line",
    "function",
    "primitive_calls",
    "total_calls",
    "total_time_seconds",
    "cumulative_time_seconds",
    "stage",
}
_PROFILE_STAGES = _STAGE_FIELDS | {"uncategorized"}


def _record(value: object, fields: set[str], location: str) -> dict[str, Any]:
    """Return a mapping after enforcing its exact field set."""
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object")
    actual = set(value)
    if actual != fields:
        missing = sorted(fields - actual)
        unexpected = sorted(actual - fields)
        raise ValueError(
            f"{location} has invalid fields; missing={missing}, unexpected={unexpected}"
        )
    return value


def _number(value: object, location: str, *, positive: bool = False) -> float:
    """Return one finite nonnegative numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a number")
    checked = float(value)
    if not math.isfinite(checked) or checked < 0.0 or (positive and checked == 0.0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{location} must be a finite {qualifier} number")
    return checked


def _integer(value: object, location: str, *, positive: bool = False) -> int:
    """Return one checked nonnegative integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location} must be an integer")
    if value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{location} must be a {qualifier} integer")
    return value


def _timestamp(value: object, location: str) -> None:
    """Require one timezone-aware ISO timestamp."""
    if not isinstance(value, str):
        raise ValueError(f"{location} must be a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{location} must be an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{location} must include a timezone")


def _check_sparse(value: object, location: str, expected_size: int) -> None:
    """Validate one retained sparse-matrix storage record."""
    record = _record(value, _SPARSE_FIELDS, location)
    if record["shape"] != [expected_size, expected_size]:
        raise ValueError(f"{location}.shape does not match workload unknowns")
    for field in _SPARSE_FIELDS - {"shape"}:
        _integer(record[field], f"{location}.{field}")


def _check_workload(value: object) -> tuple[str, dict[str, Any]]:
    """Validate and return one workload definition keyed by its identifier."""
    workload = _record(value, _WORKLOAD_FIELDS, "workload")
    groups = _integer(workload["groups"], "workload.groups", positive=True)
    layers = _integer(workload["axial_layers"], "workload.axial_layers", positive=True)
    cells = _integer(workload["active_cells"], "workload.active_cells", positive=True)
    unknowns = _integer(workload["unknowns"], "workload.unknowns", positive=True)
    expected_id = f"g{groups}-z{layers}"
    if workload["workload_id"] != expected_id:
        raise ValueError(f"workload.workload_id must be {expected_id!r}")
    if cells != 61 * layers or unknowns != cells * groups:
        raise ValueError("workload dimensions are inconsistent")

    digests = _record(workload["array_digests"], _DIGEST_FIELDS, "array_digests")
    for name, digest in digests.items():
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"array_digests.{name} must be a SHA-256 digest")

    settings = _record(workload["settings"], _SETTINGS_FIELDS, "settings")
    inner = _record(
        settings["inner_linear_solve"],
        {"relative_residual_tolerance", "strategy"},
        "settings.inner_linear_solve",
    )
    if inner["strategy"] != "direct":
        raise ValueError("settings.inner_linear_solve.strategy must be 'direct'")
    _number(
        inner["relative_residual_tolerance"],
        "settings.inner_linear_solve.relative_residual_tolerance",
        positive=True,
    )
    _integer(
        settings["max_outer_iterations"],
        "settings.max_outer_iterations",
        positive=True,
    )
    for field in (
        "keff_change_tolerance",
        "flux_change_tolerance",
        "keff_relative_residual_tolerance",
    ):
        _number(settings[field], f"settings.{field}", positive=True)
    _number(
        settings["flux_nonnegativity_tolerance"],
        "settings.flux_nonnegativity_tolerance",
    )
    iteration = _record(
        settings["eigenvalue_iteration"], {"kind"}, "settings.eigenvalue_iteration"
    )
    if iteration["kind"] != "power":
        raise ValueError("settings.eigenvalue_iteration.kind must be 'power'")
    return expected_id, workload


def _check_profile(value: object, kind: str) -> None:
    """Validate the optional profile belonging to an observation kind."""
    if kind == "measurement":
        if value is not None:
            raise ValueError("measurement observations must not contain a profile")
        return
    profile = _record(value, {"profiler", "entries"}, "observation.profile")
    if profile["profiler"] != "cProfile" or not isinstance(profile["entries"], list):
        raise ValueError("profile must contain cProfile entries")
    for index, candidate in enumerate(profile["entries"]):
        location = f"observation.profile.entries[{index}]"
        entry = _record(candidate, _PROFILE_ENTRY_FIELDS, location)
        if not isinstance(entry["file"], str) or not isinstance(entry["function"], str):
            raise ValueError(f"{location} file and function must be strings")
        for field in ("line", "primitive_calls", "total_calls"):
            _integer(entry[field], f"{location}.{field}")
        for field in ("total_time_seconds", "cumulative_time_seconds"):
            _number(entry[field], f"{location}.{field}")
        if entry["stage"] not in _PROFILE_STAGES:
            raise ValueError(f"{location}.stage is unknown")


def _check_repository(value: object) -> None:
    """Validate repository provenance for one observation."""
    repository = _record(value, _REPOSITORY_FIELDS, "observation.repository")
    commit = repository["commit"]
    if commit is not None and (
        not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None
    ):
        raise ValueError(
            "observation.repository.commit must be a full Git hash or null"
        )
    if repository["tracked_worktree_state"] not in {"clean", "dirty", "unknown"}:
        raise ValueError("observation.repository.tracked_worktree_state is unknown")
    changes = repository["tracked_changes"]
    if not isinstance(changes, list) or not all(
        isinstance(item, str) for item in changes
    ):
        raise ValueError("observation.repository.tracked_changes must be strings")


def _check_environment(value: object, requested_threads: int) -> None:
    """Validate machine and numerical-library context for one observation."""
    environment = _record(value, _ENVIRONMENT_FIELDS, "observation.environment")
    for field in (
        "operating_system",
        "python_version",
        "numpy_version",
        "scipy_version",
    ):
        if not isinstance(environment[field], str):
            raise ValueError(f"observation.environment.{field} must be a string")
    if environment["processor"] is not None and not isinstance(
        environment["processor"], str
    ):
        raise ValueError("observation.environment.processor must be a string or null")
    if environment["logical_cpu_count"] is not None:
        _integer(
            environment["logical_cpu_count"],
            "observation.environment.logical_cpu_count",
            positive=True,
        )
    thread_environment = _record(
        environment["thread_environment"],
        _THREAD_FIELDS,
        "observation.environment.thread_environment",
    )
    if any(
        thread_value != str(requested_threads)
        for thread_value in thread_environment.values()
    ):
        raise ValueError("worker thread environment does not match requested_threads")


def _check_configuration(value: object) -> dict[str, Any]:
    """Validate and return configuration timing and memory observations."""
    configuration = _record(value, _CONFIGURATION_FIELDS, "observation.configuration")
    _number(configuration["wall_time_seconds"], "configuration.wall_time_seconds")
    for field in ("import_baseline_rss_bytes", "completed_rss_bytes"):
        _integer(configuration[field], f"configuration.{field}")
    return configuration


def _check_solve(
    value: object,
    *,
    configuration: Mapping[str, Any],
    expected_size: int,
) -> None:
    """Validate solve measurements, retained arrays, and numerical checks."""
    solve = _record(value, _SOLVE_FIELDS, "observation.solve")
    wall_time = _number(
        solve["wall_time_seconds"], "solve.wall_time_seconds", positive=True
    )
    for field in ("user_cpu_time_seconds", "system_cpu_time_seconds"):
        _number(solve[field], f"solve.{field}")
    peak_rss = _integer(solve["peak_rss_bytes"], "solve.peak_rss_bytes")
    checkpoints = _record(
        solve["rss_checkpoints_bytes"],
        _RSS_CHECKPOINT_FIELDS,
        "solve.rss_checkpoints_bytes",
    )
    for name, checkpoint in checkpoints.items():
        _integer(checkpoint, f"solve.rss_checkpoints_bytes.{name}")
    if peak_rss < max(
        configuration["import_baseline_rss_bytes"],
        configuration["completed_rss_bytes"],
        *checkpoints.values(),
    ):
        raise ValueError("solve.peak_rss_bytes is smaller than an RSS observation")

    stages = _record(solve["stages_seconds"], _STAGE_FIELDS, "solve.stages_seconds")
    for name, seconds in stages.items():
        _number(seconds, f"solve.stages_seconds.{name}")
    if sum(stages.values()) > wall_time + 1.0e-9:
        raise ValueError("exclusive stage times exceed solve wall time")

    matrices = _record(solve["matrices"], {"loss", "fission"}, "solve.matrices")
    for name, matrix in matrices.items():
        _check_sparse(matrix, f"solve.matrices.{name}", expected_size)
    factorization = _record(
        solve["factorization"], {"lower", "upper"}, "solve.factorization"
    )
    for name, matrix in factorization.items():
        _check_sparse(matrix, f"solve.factorization.{name}", expected_size)
    _integer(solve["outer_iterations"], "solve.outer_iterations", positive=True)

    checks = _record(
        solve["numerical_checks"], _NUMERICAL_FIELDS, "solve.numerical_checks"
    )
    _number(checks["keff"], "solve.numerical_checks.keff", positive=True)
    for field in (
        "minimum_flux",
        "final_keff_relative_residual",
        "scalar_balance_relative_closure",
    ):
        _number(checks[field], f"solve.numerical_checks.{field}")
    residual = checks["scalar_balance_residual"]
    if (
        isinstance(residual, bool)
        or not isinstance(residual, (int, float))
        or not math.isfinite(residual)
    ):
        raise ValueError(
            "solve.numerical_checks.scalar_balance_residual must be finite"
        )


def _check_observation(
    value: object, workloads: Mapping[str, Mapping[str, Any]]
) -> str:
    """Validate and return one observation identifier."""
    observation = _record(value, _OBSERVATION_FIELDS, "observation")
    observation_id = observation["observation_id"]
    workload_id = observation["workload_id"]
    if not isinstance(observation_id, str) or not isinstance(workload_id, str):
        raise ValueError("observation identifiers must be strings")
    if workload_id not in workloads:
        raise ValueError(f"observation references unknown workload_id {workload_id!r}")
    kind = observation["kind"]
    if kind not in {"measurement", "profile"}:
        raise ValueError("observation.kind must be 'measurement' or 'profile'")
    repetition = observation["repetition"]
    if kind == "measurement":
        _integer(repetition, "observation.repetition")
    elif repetition is not None:
        raise ValueError("profile observation repetition must be null")
    threads = _integer(
        observation["requested_threads"], "observation.requested_threads", positive=True
    )
    _timestamp(observation["started_at_utc"], "observation.started_at_utc")
    _check_repository(observation["repository"])
    _check_environment(observation["environment"], threads)
    limits = _record(
        observation["resource_limits"],
        _RESOURCE_LIMIT_FIELDS,
        "observation.resource_limits",
    )
    _number(
        limits["timeout_seconds"],
        "observation.resource_limits.timeout_seconds",
        positive=True,
    )
    for field in ("address_space_bytes", "captured_output_bytes_per_stream"):
        _integer(limits[field], f"observation.resource_limits.{field}", positive=True)
    configuration = _check_configuration(observation["configuration"])
    _check_solve(
        observation["solve"],
        configuration=configuration,
        expected_size=workloads[workload_id]["unknowns"],
    )
    _check_profile(observation["profile"], kind)
    return observation_id


def check_document(document: object) -> dict[str, Any]:
    """Check a current-checkout performance result document."""
    checked = _record(document, _DOCUMENT_FIELDS, "performance result")
    if not isinstance(checked["workloads"], list):
        raise ValueError("performance result workloads must be an array")
    if not isinstance(checked["observations"], list):
        raise ValueError("performance result observations must be an array")
    workloads: dict[str, dict[str, Any]] = {}
    for candidate in checked["workloads"]:
        workload_id, workload = _check_workload(candidate)
        if workload_id in workloads:
            raise ValueError("performance result contains duplicate workload_id values")
        workloads[workload_id] = workload
    observation_ids = []
    for candidate in checked["observations"]:
        observation_ids.append(_check_observation(candidate, workloads))
    if len(set(observation_ids)) != len(observation_ids):
        raise ValueError("performance result contains duplicate observation_id values")
    return checked


def read_document(path: str | Path) -> dict[str, Any]:
    """Read and check one current-checkout performance result JSON file."""
    candidate = Path(path)
    try:
        document = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read performance result {candidate}") from exc
    return check_document(document)


def write_document(path: str | Path, document: Mapping[str, object]) -> None:
    """Check and deterministically serialize one performance result."""
    checked = check_document(dict(document))
    try:
        serialized = json.dumps(checked, indent=2, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("performance result contains a non-JSON value") from exc
    Path(path).write_text(serialized + "\n", encoding="utf-8")


def observation_metrics(observation: Mapping[str, object]) -> dict[str, float]:
    """Derive reporting metrics omitted from the stored observation."""
    configuration = observation["configuration"]
    solve = observation["solve"]
    total_cpu = float(solve["user_cpu_time_seconds"]) + float(
        solve["system_cpu_time_seconds"]
    )
    factorization = solve["factorization"]
    factorization_bytes = sum(
        int(factorization[part][field])
        for part in ("lower", "upper")
        for field in ("data_bytes", "indices_bytes", "indptr_bytes")
    )
    wall_time = float(solve["wall_time_seconds"])
    return {
        "configuration_wall_time_seconds": float(configuration["wall_time_seconds"]),
        "wall_time_seconds": wall_time,
        "user_cpu_time_seconds": float(solve["user_cpu_time_seconds"]),
        "system_cpu_time_seconds": float(solve["system_cpu_time_seconds"]),
        "total_cpu_time_seconds": total_cpu,
        "cpu_time_to_wall_time_ratio": total_cpu / wall_time,
        "peak_rss_bytes": float(solve["peak_rss_bytes"]),
        "import_relative_peak_rss_bytes": float(
            solve["peak_rss_bytes"] - configuration["import_baseline_rss_bytes"]
        ),
        "factorization_retained_array_bytes": float(factorization_bytes),
        "unattributed_seconds": max(
            0.0,
            wall_time - sum(float(value) for value in solve["stages_seconds"].values()),
        ),
    }


def summarize_observations(
    observations: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Derive timing summaries from repeated headline observations."""
    grouped: dict[tuple[str, int], list[Mapping[str, object]]] = {}
    for observation in observations:
        if observation["kind"] != "measurement":
            continue
        key = (str(observation["workload_id"]), int(observation["requested_threads"]))
        grouped.setdefault(key, []).append(observation)

    summaries = []
    for (workload_id, requested_threads), records in sorted(grouped.items()):
        if len(records) < 3:
            continue
        metrics = [observation_metrics(record) for record in records]
        statistics = {
            name: _summary_statistics(record[name] for record in metrics)
            for name in metrics[0]
        }
        stage_statistics = {
            stage: _summary_statistics(
                float(record["solve"]["stages_seconds"][stage]) for record in records
            )
            for stage in records[0]["solve"]["stages_seconds"]
        }
        summaries.append(
            {
                "workload_id": workload_id,
                "requested_threads": requested_threads,
                "observation_count": len(records),
                **statistics,
                "stages_seconds": stage_statistics,
            }
        )
    return summaries


def _summary_statistics(values: Iterable[float]) -> dict[str, float]:
    """Return deterministic median, extrema, and range for numeric values."""
    ordered = sorted(values)
    return {
        "median": median(ordered),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "range": ordered[-1] - ordered[0],
    }


def build_document(
    *,
    workloads: Iterable[Mapping[str, object]],
    observations: Iterable[Mapping[str, object]],
) -> dict[str, Any]:
    """Construct and check one current-checkout result document."""
    document = {
        "workloads": [dict(record) for record in workloads],
        "observations": [dict(record) for record in observations],
    }
    return check_document(document)
