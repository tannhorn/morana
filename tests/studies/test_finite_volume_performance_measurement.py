"""Tests for many-group fresh-process measurement and result records."""

# pylint: disable=duplicate-code,import-error,protected-access,redefined-outer-name

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
    solver_screen,
)
from studies.finite_volume_performance.cases import REFERENCE_CASE_ID, case_records


@pytest.fixture(scope="module")
def measured_observation() -> dict[str, object]:
    """Return one real headline observation from an isolated interpreter."""
    return orchestration.launch_outcome(
        6,
        2,
        case_id=REFERENCE_CASE_ID,
        requested_threads=1,
        kind="measurement",
        repetition=0,
    )


@pytest.fixture(scope="module")
def profile_observation() -> dict[str, object]:
    """Return one separately profiled observation from an isolated interpreter."""
    return orchestration.launch_outcome(
        6,
        2,
        case_id=REFERENCE_CASE_ID,
        requested_threads=1,
        kind="profile",
        repetition=None,
    )


@pytest.fixture(scope="module")
def direct_ordering_outcome() -> dict[str, object]:
    """Return one real group-major direct-screen outcome."""
    return orchestration.launch_outcome(
        6,
        2,
        case_id="direct_group_colamd",
        requested_threads=1,
        kind="measurement",
        repetition=0,
    )


@pytest.fixture(scope="module")
def gmres_ilu_outcome() -> dict[str, object]:
    """Return one real incomplete-LU GMRES screen outcome."""
    return orchestration.launch_outcome(
        6,
        2,
        case_id="gmres_ilu",
        requested_threads=1,
        kind="measurement",
        repetition=0,
    )


def _document(outcomes: list[dict[str, object]]) -> dict[str, object]:
    """Build a checked smoke-size document for tests."""
    return results.build_document(
        workloads=[measurement.workload_record(6, 2)],
        outcomes=outcomes,
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
        orchestration.launch_outcome(
            6,
            2,
            case_id=REFERENCE_CASE_ID,
            requested_threads=1,
            kind="measurement",
            repetition=0,
            **arguments,
        )


def test_launcher_terminates_and_reaps_timed_out_worker() -> None:
    """A deadline kills the isolated worker rather than leaving it running."""
    outcome = orchestration.launch_outcome(
        6,
        2,
        case_id=REFERENCE_CASE_ID,
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
    assert results.outcome_metrics(measured_observation)["unattributed_seconds"] >= 0.0


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
    assert set(document) == {"workloads", "cases", "outcomes"}


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
    with pytest.raises(ValueError, match="unknown workload"):
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
        record["outcome_id"] = f"repeat-{index}"
        record["solve"]["wall_time_seconds"] = wall_time
        measurements.append(record)
    summary = results.summarize_outcomes([*measurements, profile_observation])
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
        6,
        2,
        case_id=REFERENCE_CASE_ID,
        requested_threads=1,
        kind="measurement",
        repetition=0,
    )
    assert events == ["usage", "usage", "factor"]


def test_maintained_mode_worker_sequences_are_exact() -> None:
    """Modes retain the resolved warm-up, repetition, and profiling policy."""
    smoke = runner._plan("smoke").requests
    assert len(smoke) == 3
    assert [request.retain for request in smoke] == [False, True, True]
    assert [request.kind for request in smoke] == [
        "measurement",
        "measurement",
        "profile",
    ]

    baseline = runner._plan("baseline").requests
    assert len(baseline) == 60
    assert sum(request.retain for request in baseline) == 48
    assert sum(request.kind == "profile" for request in baseline) == 12
    endpoint = [
        request
        for request in baseline
        if (request.groups, request.axial_layers) == (72, 20)
    ]
    assert [(request.kind, request.retain) for request in endpoint] == [
        ("measurement", True),
        ("measurement", True),
        ("measurement", True),
        ("profile", True),
    ]
    assert [request.repetition for request in endpoint] == [0, 1, 2, None]
    endpoint_index = baseline.index(endpoint[0])
    assert baseline[endpoint_index - 1] == runner.WorkerRequest(
        72, 5, REFERENCE_CASE_ID, retain=False
    )

    thread_screen = runner._plan("thread-screen").requests
    assert len(thread_screen) == 16
    assert sum(request.retain for request in thread_screen) == 12
    assert {request.requested_threads for request in thread_screen} == {1, 2, 4, 6}

    solver_requests = runner._plan("solver-screen").requests
    assert len(solver_requests) == 42
    assert sum(request.retain for request in solver_requests) == 28
    assert {request.kind for request in solver_requests} == {"measurement"}
    assert {request.requested_threads for request in solver_requests} == {1}
    assert {request.case_id for request in solver_requests} == {
        case["case_id"] for case in case_records()
    }
    assert {(request.groups, request.axial_layers) for request in solver_requests} == {
        (36, 10),
        (72, 5),
    }


def test_solver_case_definitions_and_explicit_thread_requests_are_exact() -> None:
    """The diagnostic matrix and operator-selected thread policy stay bounded."""
    cases = case_records()
    assert [case["case_id"] for case in cases] == [
        "direct_node_colamd",
        "direct_node_mmd_ata",
        "direct_node_mmd_at_plus_a",
        "direct_group_colamd",
        "gmres_none",
        "gmres_jacobi",
        "gmres_ilu",
    ]
    assert [case["column_ordering"] for case in cases[:4]] == [
        "COLAMD",
        "MMD_ATA",
        "MMD_AT_PLUS_A",
        "COLAMD",
    ]
    requests = runner._plan("thread-screen", "gmres_ilu").requests
    assert len(requests) == 16
    assert sum(request.retain for request in requests) == 12
    assert {request.requested_threads for request in requests} == {1, 2, 4, 6}
    assert {request.case_id for request in requests} == {"gmres_ilu"}
    assert runner._plan("thread-screen").output_name == (
        "thread_screen_direct_node_colamd.json"
    )
    assert runner._plan("thread-screen", "gmres_ilu").output_name == (
        "thread_screen_gmres_ilu.json"
    )
    assert runner._plan("baseline").output_name == "baseline_direct_node_colamd.json"
    assert runner._plan("baseline", "gmres_ilu").output_name == (
        "baseline_gmres_ilu.json"
    )
    alternate_baseline = runner._plan("baseline", "gmres_ilu").requests
    assert len(alternate_baseline) == 60
    assert sum(request.retain for request in alternate_baseline) == 48
    assert {request.kind for request in alternate_baseline} == {
        "measurement",
        "profile",
    }
    assert {request.case_id for request in alternate_baseline} == {"gmres_ilu"}


def test_group_major_permutation_round_trip_and_direct_factor_statistics(
    direct_ordering_outcome: dict[str, object],
    measured_observation: dict[str, object],
) -> None:
    """Explicit packing preserves results and records applied permutations."""
    permutation = solver_screen.packing_permutation(3, 2)
    assert permutation.tolist() == [0, 2, 4, 1, 3, 5]
    group_major = list(range(6))
    restored = [None] * 6
    for new, old in enumerate(permutation):
        restored[old] = group_major[new]
    assert [group_major[index] for index in permutation] == [0, 2, 4, 1, 3, 5]
    assert [restored[index] for index in permutation] == group_major

    assert direct_ordering_outcome["status"] == "success"
    solve = direct_ordering_outcome["solve"]
    factor = solve["factorization"]
    assert factor["kind"] == "complete"
    assert len(factor["packing_new_to_old"]) == 732
    assert len(factor["row_permutation"]) == 732
    assert len(factor["column_permutation"]) == 732
    assert factor["lower"]["stored_nonzeros"] > 0
    assert factor["upper"]["stored_nonzeros"] > 0
    assert "krylov_iterations_by_outer" not in solve
    assert "total_krylov_iterations" not in solve
    assert "linear_relative_residuals_by_outer" not in solve
    assert solve["numerical_checks"]["keff"] == pytest.approx(
        measured_observation["solve"]["numerical_checks"]["keff"], rel=1.0e-12
    )


def test_gmres_screen_aggregates_iterations_and_incomplete_factor_statistics(
    gmres_ilu_outcome: dict[str, object],
) -> None:
    """One reused ILU setup and all per-outer Krylov counts are retained."""
    assert gmres_ilu_outcome["status"] == "success"
    solve = gmres_ilu_outcome["solve"]
    assert len(solve["krylov_iterations_by_outer"]) == solve["outer_iterations"]
    assert solve["total_krylov_iterations"] == sum(solve["krylov_iterations_by_outer"])
    assert solve["total_krylov_iterations"] > solve["outer_iterations"]
    assert solve["factorization"]["kind"] == "incomplete"
    assert solve["timings_seconds"]["linear_solve_setup"] >= 0.0
    assert solve["timings_seconds"]["linear_rhs_solves"] >= 0.0


def test_outcome_summary_labels_outer_and_optional_krylov(
    capsys: pytest.CaptureFixture[str],
    direct_ordering_outcome: dict[str, object],
    gmres_ilu_outcome: dict[str, object],
) -> None:
    """Console summaries distinguish outer and optional Krylov work."""
    runner._print_outcome(direct_ordering_outcome)
    direct_text = capsys.readouterr().out
    assert "outer=" in direct_text
    assert "krylov=" not in direct_text

    runner._print_outcome(gmres_ilu_outcome)
    gmres_text = capsys.readouterr().out
    assert "outer=" in gmres_text
    assert "krylov=" in gmres_text


def test_solver_failures_are_structured_and_terminal_on_resume(monkeypatch) -> None:
    """A bounded worker failure is retained and suppresses the rest of its cell."""

    def fail(*_args, **_kwargs):
        raise orchestration.PerformanceWorkerError("timeout", "deadline")

    monkeypatch.setattr(orchestration, "_launch_worker", fail)
    outcome = orchestration.launch_outcome(
        36,
        10,
        case_id="gmres_none",
        requested_threads=1,
        kind="measurement",
        repetition=0,
    )
    assert outcome["status"] == "failed"
    assert outcome["failure"] == {"kind": "timeout", "message": "deadline"}
    requests = tuple(
        request
        for request in runner._plan("solver-screen").requests
        if request.case_id == "gmres_none"
        and (request.groups, request.axial_layers) == (36, 10)
    )
    assert not runner._pending_requests(requests, [outcome])


def test_document_accepts_success_and_failure_records(
    direct_ordering_outcome: dict[str, object],
) -> None:
    """One checked envelope accepts every strategy and terminal failure."""
    failure = orchestration.failed_outcome(
        6,
        2,
        "gmres_none",
        requested_threads=1,
        measurement_kind="measurement",
        repetition=0,
        failure_kind="worker_error",
        message="failed",
    )
    failure["resource_limits"] = direct_ordering_outcome["resource_limits"]
    document = results.build_document(
        workloads=[measurement.workload_record(6, 2)],
        outcomes=[direct_ordering_outcome, failure],
    )
    results.check_document(document)
    assert set(document) == {"workloads", "cases", "outcomes"}
    assert set(_document([])) == {"workloads", "cases", "outcomes"}


def test_baseline_resume_runs_only_missing_endpoint_after_smaller_warmup() -> None:
    """A partial baseline resumes its endpoint without replaying the matrix."""
    baseline = runner._plan("baseline").requests
    completed = [
        {"outcome_id": runner._request_id(request), "status": "success"}
        for request in baseline
        if request.retain and (request.groups, request.axial_layers) != (72, 20)
    ]

    resumed = runner._pending_requests(baseline, completed)

    assert resumed[0] == runner.WorkerRequest(72, 5, REFERENCE_CASE_ID, retain=False)
    endpoint_requests = [
        (request.groups, request.axial_layers, request.kind, request.repetition)
        for request in resumed[1:]
    ]
    assert endpoint_requests == [
        (72, 20, "measurement", 0),
        (72, 20, "measurement", 1),
        (72, 20, "measurement", 2),
        (72, 20, "profile", None),
    ]


def test_resume_rejects_unexpected_observations_and_skips_completed_work() -> None:
    """Resume retains only recognized observations and skips completed work."""
    baseline = runner._plan("baseline").requests
    completed = [
        {"outcome_id": runner._request_id(request), "status": "success"}
        for request in baseline
        if request.retain
    ]

    assert not runner._pending_requests(baseline, completed)
    with pytest.raises(ValueError, match="unexpected"):
        runner._pending_requests(
            baseline, [{"outcome_id": "unknown", "status": "success"}]
        )


def test_resume_replays_required_warmups_for_every_mode() -> None:
    """Resume handles a partial smoke record without baseline-specific rules."""
    smoke = runner._plan("smoke").requests
    profile = next(request for request in smoke if request.kind == "profile")
    measurement = next(
        request for request in smoke if request.retain and request.kind == "measurement"
    )

    assert runner._pending_requests(smoke, []) == smoke
    assert runner._pending_requests(
        smoke,
        [{"outcome_id": runner._request_id(measurement), "status": "success"}],
    ) == (profile,)


def test_smoke_runner_discards_warmup_and_checkpoints_results(
    tmp_path: Path,
    monkeypatch,
    measured_observation: dict[str, object],
    profile_observation: dict[str, object],
) -> None:
    """The maintained smoke path writes only its measurement and profile."""
    launched = []
    returned = iter((measured_observation, measured_observation, profile_observation))

    def fake_launch(*args, **kwargs):
        launched.append((args, kwargs))
        return deepcopy(next(returned))

    monkeypatch.setattr(runner, "launch_outcome", fake_launch)
    output_path = runner.run("smoke", tmp_path)
    document = results.read_document(output_path)
    assert len(launched) == 3
    assert len(document["workloads"]) == 1
    assert [record["kind"] for record in document["outcomes"]] == [
        "measurement",
        "profile",
    ]
