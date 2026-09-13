"""Tests for many-group fresh-process measurement and result records."""

# pylint: disable=duplicate-code,import-error,protected-access,redefined-outer-name

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile

import pytest

from examples.many_group_performance import measurement, orchestration, results


@pytest.fixture(scope="module")
def measured_observation() -> dict[str, object]:
    """Return one real headline observation from an isolated interpreter."""
    return orchestration.launch_observation(
        6, 2, requested_threads=1, kind="measurement", repetition=0
    )


@pytest.fixture(scope="module")
def profile_observation() -> dict[str, object]:
    """Return one separately profiled observation from an isolated interpreter."""
    return orchestration.launch_observation(
        6, 2, requested_threads=1, kind="profile", repetition=None
    )


def _document(observations: list[dict[str, object]]) -> dict[str, object]:
    """Build a checked smoke-size document for tests."""
    return results.build_document(
        workloads=[measurement.workload_record(6, 2)],
        observations=observations,
    )


def test_worker_environment_sets_every_thread_limit() -> None:
    """The parent controls common BLAS and OpenMP limits before worker import."""
    environment = orchestration.worker_environment(4)
    assert all(
        environment[name] == "4" for name in orchestration.THREAD_ENVIRONMENT_VARIABLES
    )
    with pytest.raises(TypeError, match="integer"):
        orchestration.worker_environment(True)
    with pytest.raises(ValueError, match="positive"):
        orchestration.worker_environment(0)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("timeout_seconds", True, TypeError),
        ("timeout_seconds", float("nan"), ValueError),
        ("address_space_limit_bytes", False, TypeError),
        ("address_space_limit_bytes", 0, ValueError),
    ),
)
def test_launcher_rejects_invalid_resource_limits(
    field: str, value: object, error: type[Exception]
) -> None:
    """Invalid resource guards fail before a worker is launched."""
    arguments = {field: value}
    with pytest.raises(error):
        orchestration.launch_observation(
            6, 2, requested_threads=1, repetition=0, **arguments
        )


def test_launcher_terminates_and_reaps_timed_out_worker() -> None:
    """A deadline kills the isolated worker rather than leaving it running."""
    with pytest.raises(RuntimeError, match="exceeded"):
        orchestration.launch_observation(
            6, 2, requested_threads=1, repetition=0, timeout_seconds=1.0e-6
        )


def test_captured_output_has_a_hard_size_limit(monkeypatch) -> None:
    """Disk-backed worker output is rejected before an unbounded read."""
    monkeypatch.setattr(orchestration, "MAX_CAPTURED_OUTPUT_BYTES", 3)
    with tempfile.TemporaryFile(mode="w+b") as stream:
        stream.write(b"four")
        with pytest.raises(RuntimeError, match="captured bytes"):
            orchestration._captured_text(stream, "stdout")


def test_measurement_uses_fresh_worker_and_records_thread_context(
    measured_observation: dict[str, object],
) -> None:
    """One observation records the thread controls set before worker import."""
    environment = measured_observation["environment"]
    assert measured_observation["requested_threads"] == 1
    assert all(value == "1" for value in environment["thread_environment"].values())
    assert measured_observation["resource_limits"] == {
        "timeout_seconds": float(orchestration.DEFAULT_TIMEOUT_SECONDS),
        "address_space_bytes": orchestration.DEFAULT_ADDRESS_SPACE_LIMIT_BYTES,
        "captured_output_bytes_per_stream": orchestration.MAX_CAPTURED_OUTPUT_BYTES,
    }
    assert (
        environment["logical_cpu_count"] is None
        or environment["logical_cpu_count"] >= 1
    )


def test_measurement_covers_and_reconciles_exclusive_stages(
    measured_observation: dict[str, object],
) -> None:
    """Every private boundary is observed once and exclusive time reconciles."""
    solve = measured_observation["solve"]
    assert set(solve["stages_seconds"]) == set(measurement.STAGE_NAMES)
    assert all(value >= 0.0 for value in solve["stages_seconds"].values())
    assert sum(solve["stages_seconds"].values()) <= (
        solve["wall_time_seconds"] + 1.0e-9
    )
    assert (
        results.observation_metrics(measured_observation)["unattributed_seconds"] >= 0.0
    )


def test_measurement_records_memory_sparse_state_and_numerics(
    measured_observation: dict[str, object],
) -> None:
    """The real observation retains RSS, operators, factors, and solve checks."""
    solve = measured_observation["solve"]
    assert set(solve["rss_checkpoints_bytes"]) == set(results.RSS_CHECKPOINT_NAMES)
    assert solve["peak_rss_bytes"] >= max(solve["rss_checkpoints_bytes"].values())
    assert (
        solve["peak_rss_bytes"]
        >= measured_observation["configuration"]["import_baseline_rss_bytes"]
    )
    for matrix in solve["matrices"].values():
        assert matrix["shape"] == [732, 732]
        assert matrix["stored_nonzeros"] > 0
        assert matrix["data_bytes"] > 0
    factorization = solve["factorization"]
    assert factorization["lower"]["stored_nonzeros"] > 0
    assert factorization["upper"]["stored_nonzeros"] > 0
    assert (
        sum(
            factorization[part][field]
            for part in ("lower", "upper")
            for field in ("data_bytes", "indices_bytes", "indptr_bytes")
        )
        > 0
    )
    assert solve["outer_iterations"] == 66
    checks = solve["numerical_checks"]
    assert checks["keff"] == pytest.approx(1.001_363_867_359_448_3, rel=1.0e-12)
    assert checks["minimum_flux"] >= 0.0
    assert checks["final_keff_relative_residual"] <= 1.0e-10
    assert checks["scalar_balance_relative_closure"] <= 1.0e-12


def test_profile_is_separate_complete_and_deterministically_sorted(
    profile_observation: dict[str, object],
) -> None:
    """Diagnostic profiling retains every entry in deterministic order."""
    assert profile_observation["kind"] == "profile"
    profile = profile_observation["profile"]
    assert profile["profiler"] == "cProfile"
    entries = profile["entries"]
    assert entries
    assert [entry["cumulative_time_seconds"] for entry in entries] == sorted(
        (entry["cumulative_time_seconds"] for entry in entries), reverse=True
    )
    assert any(entry["stage"] == "eigenvalue_iteration" for entry in entries)
    assert any(entry["stage"] == "uncategorized" for entry in entries)
    document = _document([profile_observation])
    assert set(document) == {"workloads", "observations"}


def test_checked_document_round_trip_is_deterministic(
    tmp_path: Path, measured_observation: dict[str, object]
) -> None:
    """The reader checks exact structure and serialization is stable."""
    document = _document([measured_observation])
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    results.write_document(first, document)
    results.write_document(second, document)
    assert first.read_bytes() == second.read_bytes()
    assert results.read_document(first) == document
    assert json.loads(first.read_text(encoding="utf-8")) == document


def test_checked_document_rejects_broken_record_relationships(
    measured_observation: dict[str, object],
) -> None:
    """Validation checks exact fields and cross-record relationships."""
    unknown = deepcopy(measured_observation)
    unknown["workload_id"] = "missing"
    with pytest.raises(ValueError, match="unknown workload_id"):
        _document([unknown])

    document = _document([measured_observation])
    document["unexpected"] = None
    with pytest.raises(ValueError, match="unexpected"):
        results.check_document(document)


def test_summary_derivation_uses_only_repeated_measurements(
    measured_observation: dict[str, object],
    profile_observation: dict[str, object],
) -> None:
    """Three raw headline records produce exact robust wall-time statistics."""
    measurements = []
    for index, wall_time in enumerate((3.0, 1.0, 2.0)):
        record = deepcopy(measured_observation)
        record["observation_id"] = f"repeat-{index}"
        record["solve"]["wall_time_seconds"] = wall_time
        measurements.append(record)
    summary = results.summarize_observations([*measurements, profile_observation])
    assert len(summary) == 1
    assert summary[0]["workload_id"] == "g6-z2"
    assert summary[0]["requested_threads"] == 1
    assert summary[0]["observation_count"] == 3
    assert summary[0]["wall_time_seconds"] == {
        "median": 2.0,
        "minimum": 1.0,
        "maximum": 3.0,
        "range": 2.0,
    }
    assert "total_cpu_time_seconds" not in measurements[0]["solve"]
    assert "cpu_time_to_wall_time_ratio" not in measurements[0]["solve"]
    assert "import_relative_peak_rss_bytes" not in measurements[0]["solve"]
    assert "unattributed_seconds" not in measurements[0]["solve"]
    assert "total_cpu_time_seconds" in summary[0]
    assert "cpu_time_to_wall_time_ratio" in summary[0]
    assert "import_relative_peak_rss_bytes" in summary[0]
    assert "factorization_retained_array_bytes" in summary[0]
    assert "unattributed_seconds" in summary[0]
    assert set(summary[0]["stages_seconds"]) == set(measurement.STAGE_NAMES)


@pytest.mark.parametrize(
    ("system", "raw", "expected"),
    (("Linux", 7, 7 * 1024), ("Darwin", 7, 7)),
)
def test_peak_rss_unit_normalization(system: str, raw: int, expected: int) -> None:
    """Platform-specific ru_maxrss units normalize to bytes."""
    assert measurement.normalize_peak_rss_bytes(raw, system) == expected


def test_rss_measurement_rejects_unsupported_platform() -> None:
    """Unsupported total-memory metrics fail instead of changing meaning."""
    with pytest.raises(RuntimeError, match="unsupported"):
        measurement.normalize_peak_rss_bytes(1, "Plan9")
    with pytest.raises(RuntimeError, match="unsupported"):
        measurement._current_rss_bytes("Plan9")


def test_instrumentation_call_graph_drift_fails_clearly() -> None:
    """Missing private calls cannot silently produce incomplete attribution."""
    recorder = measurement._Recorder()
    with pytest.raises(RuntimeError, match="call graph changed"):
        recorder.require_expected_calls()


def test_factor_views_are_inspected_after_peak_rss_is_frozen(monkeypatch) -> None:
    """Potentially allocating L/U inspection follows the final usage sample."""
    events = []
    getrusage = measurement.resource.getrusage
    factor_record = measurement._factor_record

    def recording_getrusage(who: int):
        events.append("usage")
        return getrusage(who)

    def recording_factor_record(factorization):
        events.append("factor")
        return factor_record(factorization)

    monkeypatch.setattr(measurement.resource, "getrusage", recording_getrusage)
    monkeypatch.setattr(measurement, "_factor_record", recording_factor_record)
    measurement.measure_workload(
        6, 2, requested_threads=1, kind="measurement", repetition=0
    )
    assert events == ["usage", "usage", "factor"]
