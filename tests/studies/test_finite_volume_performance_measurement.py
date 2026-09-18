"""Tests for many-group fresh-process measurements and records."""

# pylint: disable=import-error,protected-access,redefined-outer-name

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile

import pytest

from studies.finite_volume_performance import (
    measurement,
    orchestration,
    results,
    runner,
    workload,
)
from studies.finite_volume_performance.cases import (
    DIRECT_CASE_ID,
    GMRES_JACOBI_CASE_ID,
    records,
)


@pytest.fixture(scope="module")
def direct_observation() -> dict[str, object]:
    """Return one direct measurement from an isolated interpreter."""
    return orchestration.launch_outcome(
        6,
        2,
        case_id=DIRECT_CASE_ID,
        requested_threads=1,
        kind="measurement",
        repetition=0,
    )


@pytest.fixture(scope="module")
def profile_observation() -> dict[str, object]:
    """Return one separately profiled direct observation."""
    return orchestration.launch_outcome(
        6,
        2,
        case_id=DIRECT_CASE_ID,
        requested_threads=1,
        kind="profile",
        repetition=None,
    )


@pytest.fixture(scope="module")
def gmres_observation() -> dict[str, object]:
    """Return one GMRES/Jacobi measurement."""
    return orchestration.launch_outcome(
        6,
        2,
        case_id=GMRES_JACOBI_CASE_ID,
        requested_threads=1,
        kind="measurement",
        repetition=0,
    )


def _document(outcomes: list[dict[str, object]]) -> dict[str, object]:
    """Build a checked smoke-size document for tests."""
    return results.build_document(
        workloads=[measurement.workload_record(6, 2)], outcomes=outcomes
    )


def test_worker_environment_sets_every_thread_limit() -> None:
    """The parent controls common BLAS and OpenMP limits before worker import."""
    environment = orchestration.worker_environment(1)
    assert all(
        environment[name] == "1" for name in orchestration.THREAD_ENVIRONMENT_VARIABLES
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
    with pytest.raises(error):
        orchestration.launch_outcome(
            6,
            2,
            case_id=DIRECT_CASE_ID,
            requested_threads=1,
            kind="measurement",
            repetition=0,
            **{field: value},
        )


def test_launcher_terminates_and_reaps_timed_out_worker() -> None:
    """A deadline kills the isolated worker rather than leaving it running."""
    outcome = orchestration.launch_outcome(
        6,
        2,
        case_id=DIRECT_CASE_ID,
        requested_threads=1,
        kind="measurement",
        repetition=0,
        timeout_seconds=1.0e-6,
    )
    assert outcome["status"] == "failed"
    assert outcome["failure"]["kind"] == "timeout"


def test_captured_output_has_a_hard_size_limit(monkeypatch) -> None:
    """Disk-backed worker output is rejected before an unbounded read."""
    monkeypatch.setattr(orchestration, "MAX_CAPTURED_OUTPUT_BYTES", 3)
    with tempfile.TemporaryFile(mode="w+b") as stream:
        stream.write(b"four")
        with pytest.raises(RuntimeError, match="captured bytes"):
            orchestration._captured_text(stream, "stdout")


def test_direct_measurement_records_thread_context_and_checked_stages(
    direct_observation: dict[str, object],
) -> None:
    """Direct observation retains controlled context, stages, and numerics."""
    environment = direct_observation["environment"]
    solve = direct_observation["solve"]
    assert direct_observation["requested_threads"] == 1
    assert all(value == "1" for value in environment["thread_environment"].values())
    assert set(solve["stages_seconds"]) == set(measurement.STAGE_NAMES)
    assert sum(solve["stages_seconds"].values()) <= solve["wall_time_seconds"] + 1e-9
    assert set(solve["rss_checkpoints_bytes"]) == set(results.RSS_CHECKPOINT_NAMES)
    assert solve["factorization"]["lower"]["stored_nonzeros"] > 0
    assert solve["factorization"]["upper"]["stored_nonzeros"] > 0
    assert solve["outer_iterations"] == 187
    assert solve["numerical_checks"]["keff"] == pytest.approx(
        1.039_943_512_290_362, rel=1.0e-12
    )


def test_gmres_jacobi_measurement_records_its_retained_diagnostics(
    gmres_observation: dict[str, object],
) -> None:
    """GMRES/Jacobi records setup, Krylov, and numerical acceptance evidence."""
    assert gmres_observation["status"] == "success"
    solve = gmres_observation["solve"]
    assert solve["factorization"] is None
    assert len(solve["krylov_iterations_by_outer"]) == solve["outer_iterations"]
    assert solve["total_krylov_iterations"] == sum(solve["krylov_iterations_by_outer"])
    assert solve["timings_seconds"]["linear_solve_setup"] >= 0.0
    assert solve["timings_seconds"]["linear_rhs_solves"] >= 0.0
    assert solve["numerical_checks"]["final_keff_relative_residual"] <= 1.0e-10


def test_profile_is_separate_complete_and_deterministically_sorted(
    profile_observation: dict[str, object],
) -> None:
    """Profiling is a distinct smoke outcome."""
    profile = profile_observation["profile"]
    assert profile["profiler"] == "cProfile"
    assert profile["entries"]
    assert [entry["cumulative_time_seconds"] for entry in profile["entries"]] == sorted(
        (entry["cumulative_time_seconds"] for entry in profile["entries"]),
        reverse=True,
    )


def test_checked_document_round_trip_is_deterministic(
    tmp_path: Path,
    direct_observation: dict[str, object],
    gmres_observation: dict[str, object],
) -> None:
    """The reader accepts the two study cases and stable serialization."""
    document = _document([direct_observation, gmres_observation])
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    results.write_document(first, document)
    results.write_document(second, document)
    assert first.read_bytes() == second.read_bytes()
    assert results.read_document(first) == document
    assert json.loads(first.read_text(encoding="utf-8"))["cases"] == records()
    workload_record = document["workloads"][0]
    assert (
        workload_record["material_family_digest"]
        == workload.FROZEN_MATERIAL_FAMILY_DIGESTS[6]
    )
    assert workload_record["placement_digest"] == workload.FROZEN_PLACEMENT_DIGESTS[2]


def test_checked_document_rejects_broken_record_relationships(
    direct_observation: dict[str, object],
) -> None:
    """Validation rejects unknown workloads and unsupported output shapes."""
    unknown = deepcopy(direct_observation)
    unknown["workload_id"] = "missing"
    with pytest.raises(ValueError, match="unknown workload"):
        _document([unknown])
    document = _document([direct_observation])
    document["unexpected"] = None
    with pytest.raises(ValueError, match="unexpected"):
        results.check_document(document)


def test_summary_derivation_uses_repeated_measurements(
    direct_observation: dict[str, object],
) -> None:
    """Three raw records produce robust wall-time statistics."""
    measurements = []
    for index, wall_time in enumerate((3.0, 1.0, 2.0)):
        record = deepcopy(direct_observation)
        record["outcome_id"] = f"repeat-{index}"
        record["solve"]["wall_time_seconds"] = wall_time
        measurements.append(record)
    summary = results.summarize_outcomes(measurements)
    assert summary[0]["wall_time_seconds"] == {
        "median": 2.0,
        "minimum": 1.0,
        "maximum": 3.0,
        "range": 2.0,
    }
    assert set(summary[0]["stages_seconds"]) == set(measurement.STAGE_NAMES)


@pytest.mark.parametrize(
    ("system", "raw", "expected"),
    (("Linux", 7, 7 * 1024), ("Darwin", 7, 7)),
)
def test_peak_rss_unit_normalization(system: str, raw: int, expected: int) -> None:
    """Platform-specific ru_maxrss units normalize to bytes."""
    assert measurement.normalize_peak_rss_bytes(raw, system) == expected


def test_instrumentation_call_graph_drift_fails_clearly() -> None:
    """Missing private calls cannot silently produce incomplete attribution."""
    with pytest.raises(RuntimeError, match="call graph changed"):
        measurement._Recorder().require_expected_calls()


def test_factor_views_are_inspected_after_peak_rss_is_frozen(monkeypatch) -> None:
    """L/U inspection follows the final usage sample."""
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
        6,
        2,
        case_id=DIRECT_CASE_ID,
        requested_threads=1,
        kind="measurement",
        repetition=0,
    )
    assert events == ["usage", "usage", "factor"]


def test_study_modes_and_cases_are_exact() -> None:
    """Smoke and the two single-thread baselines have fixed definitions."""
    smoke = runner._plan("smoke")
    assert smoke.output_name == "smoke.json"
    assert [request.record for request in smoke.requests] == [False, True, True]
    assert {request.case_id for request in smoke.requests} == {DIRECT_CASE_ID}
    direct = runner._plan("baseline")
    gmres = runner._plan("baseline", GMRES_JACOBI_CASE_ID)
    assert direct.output_name == "baseline_direct.json"
    assert gmres.output_name == "baseline_gmres_jacobi.json"
    assert len(direct.requests) == len(gmres.requests) == 60
    assert {request.requested_threads for request in gmres.requests} == {1}
    with pytest.raises(ValueError, match="one of"):
        runner._plan("baseline", "gmres_ilu")


def test_baseline_resume_runs_only_missing_endpoint_after_smaller_warmup() -> None:
    """A partial baseline resumes its endpoint without replaying the matrix."""
    baseline = runner._plan("baseline").requests
    completed = [
        {"outcome_id": runner._request_id(request), "status": "success"}
        for request in baseline
        if request.record and (request.groups, request.axial_layers) != (72, 24)
    ]
    resumed = runner._pending_requests(baseline, completed)
    assert resumed[0] == runner.WorkerRequest(72, 6, DIRECT_CASE_ID, record=False)
    assert [(request.kind, request.repetition) for request in resumed[1:]] == [
        ("measurement", 0),
        ("measurement", 1),
        ("measurement", 2),
        ("profile", None),
    ]


def test_smoke_runner_discards_warmup_and_checkpoints_results(
    tmp_path: Path,
    monkeypatch,
    direct_observation: dict[str, object],
    profile_observation: dict[str, object],
) -> None:
    """The maintained smoke path writes only its measurement and profile."""
    returned = iter((direct_observation, direct_observation, profile_observation))
    monkeypatch.setattr(
        runner,
        "launch_outcome",
        lambda *_args, **_kwargs: deepcopy(next(returned)),
    )
    document = results.read_document(runner.run("smoke", tmp_path))
    assert [record["kind"] for record in document["outcomes"]] == [
        "measurement",
        "profile",
    ]
