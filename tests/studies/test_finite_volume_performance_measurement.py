# pylint: disable=import-error,protected-access,redefined-outer-name

"""Tests for many-group fresh-process measurements and records."""

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
    BICGSTAB_JACOBI_CASE_ID,
    DIRECT_CASE_ID,
    GMRES_JACOBI_CASE_ID,
    WIELANDT_ITERATION_ID,
    iteration_record,
    records,
    resolve_case,
)


def _plan(mode: str, solver_case: str | None = None, **overrides) -> runner.RunPlan:
    """Resolve a test plan while keeping production planning inputs explicit."""
    options = {
        "workloads": (),
        "warmup": None,
        "output_name": None,
        "repetitions": 1,
        "profile": False,
        "iteration_id": "power",
        "shift_inverse_keff": None,
        "placement": "structured",
    }
    options.update(overrides)
    return runner._plan(mode, solver_case, **options)


def _request(
    groups: int, axial_layers: int, case_id: str, **overrides
) -> runner.WorkerRequest:
    """Build one complete resolved request for plan assertions."""
    fields = {
        "groups": groups,
        "axial_layers": axial_layers,
        "case_id": case_id,
        "placement": "structured",
        "requested_threads": 1,
        "iteration_id": "power",
        "shift_inverse_keff": None,
        "kind": "measurement",
        "repetition": None,
        "record": True,
    }
    fields.update(overrides)
    return runner.WorkerRequest(**fields)


def _launch_outcome(groups: int, axial_layers: int, **overrides):
    """Launch a test worker with every resolved control explicit."""
    options = {
        "case_id": DIRECT_CASE_ID,
        "iteration_id": "power",
        "shift_inverse_keff": None,
        "requested_threads": 1,
        "kind": "measurement",
        "repetition": 0,
        "timeout_seconds": orchestration.DEFAULT_TIMEOUT_SECONDS,
        "address_space_limit_bytes": (orchestration.DEFAULT_ADDRESS_SPACE_LIMIT_BYTES),
        "placement": "structured",
    }
    options.update(overrides)
    return orchestration.launch_outcome(groups, axial_layers, **options)


@pytest.fixture(scope="module")
def direct_observation() -> dict[str, object]:
    """Return one direct measurement from an isolated interpreter."""
    return _launch_outcome(
        6,
        2,
        case_id=DIRECT_CASE_ID,
    )


@pytest.fixture(scope="module")
def profile_observation() -> dict[str, object]:
    """Return one separately profiled direct observation."""
    return _launch_outcome(
        6,
        2,
        case_id=DIRECT_CASE_ID,
        kind="profile",
        repetition=None,
    )


@pytest.fixture(scope="module")
def gmres_observation() -> dict[str, object]:
    """Return one GMRES/Jacobi measurement."""
    return _launch_outcome(
        6,
        2,
        case_id=GMRES_JACOBI_CASE_ID,
    )


@pytest.fixture(scope="module")
def bicgstab_observation() -> dict[str, object]:
    """Return one BiCGSTAB/Jacobi measurement."""
    return _launch_outcome(
        6,
        2,
        case_id=BICGSTAB_JACOBI_CASE_ID,
    )


def _document(outcomes: list[dict[str, object]]) -> dict[str, object]:
    """Build a checked smoke-size document for tests."""
    return results.build_document(
        workloads=[measurement.workload_record(6, 2, "structured")],
        outcomes=outcomes,
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
        _launch_outcome(
            6,
            2,
            case_id=DIRECT_CASE_ID,
            **{field: value},
        )


def test_launcher_requires_a_complete_iteration_policy() -> None:
    """Incomplete or inconsistent policies fail before a worker is created."""
    with pytest.raises(ValueError, match="requires an explicit fixed shift"):
        _launch_outcome(
            6,
            2,
            case_id=DIRECT_CASE_ID,
            iteration_id=WIELANDT_ITERATION_ID,
        )
    with pytest.raises(ValueError, match="only to Wielandt"):
        _launch_outcome(
            6,
            2,
            case_id=DIRECT_CASE_ID,
            shift_inverse_keff=0.95,
        )


def test_launcher_terminates_and_reaps_timed_out_worker() -> None:
    """A deadline kills the isolated worker rather than leaving it running."""
    outcome = _launch_outcome(
        6,
        2,
        case_id=DIRECT_CASE_ID,
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


def test_bicgstab_jacobi_measurement_records_its_retained_diagnostics(
    bicgstab_observation: dict[str, object],
) -> None:
    """BiCGSTAB/Jacobi records setup and numerical acceptance evidence."""
    assert bicgstab_observation["status"] == "success"
    solve = bicgstab_observation["solve"]
    assert solve["factorization"] is None
    assert len(solve["krylov_iterations_by_outer"]) == solve["outer_iterations"]
    assert solve["total_krylov_iterations"] == sum(solve["krylov_iterations_by_outer"])
    assert solve["timings_seconds"]["linear_solve_setup"] >= 0.0
    assert solve["timings_seconds"]["linear_rhs_solves"] >= 0.0
    assert solve["numerical_checks"]["final_keff_relative_residual"] <= 1.0e-10


def test_checked_document_accepts_a_zero_iteration_continued_inner_solve(
    gmres_observation: dict[str, object],
) -> None:
    """A converged continuation guess may require no additional Krylov work."""
    observation = deepcopy(gmres_observation)
    solve = observation["solve"]
    replaced = solve["krylov_iterations_by_outer"][-1]
    solve["krylov_iterations_by_outer"][-1] = 0
    solve["total_krylov_iterations"] -= replaced

    assert _document([observation])["outcomes"] == [observation]


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
    serialized = json.loads(first.read_text(encoding="utf-8"))
    assert serialized["cases"] == records()
    assert all("eigenvalue_iteration" not in case for case in serialized["cases"])
    assert "inner_linear_solve" not in serialized["solve_controls"]
    assert "eigenvalue_iteration" not in serialized["solve_controls"]
    workload_record = document["workloads"][0]
    assert (
        workload_record["material_family_digest"]
        == workload.FROZEN_MATERIAL_FAMILY_DIGESTS[6]
    )
    assert workload_record["placement_digest"] == workload.FROZEN_PLACEMENT_DIGESTS[2]
    assert workload_record["placement"] == {
        "kind": "structured",
        "algorithm": None,
        "seed": None,
    }


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
    assert summary[0]["eigenvalue_iteration"] == {"kind": "power"}


def test_summary_keeps_iteration_policies_separate(
    direct_observation: dict[str, object],
) -> None:
    """Ordinary and shifted measurements cannot form one timing population."""
    measurements = []
    for iteration in (
        iteration_record("power", None),
        iteration_record("wielandt", 0.95),
    ):
        for index in range(3):
            record = deepcopy(direct_observation)
            record["outcome_id"] = f"summary-only-{iteration['kind']}-{index}"
            record["eigenvalue_iteration"] = iteration
            measurements.append(record)
    summaries = results.summarize_outcomes(measurements)
    assert [item["eigenvalue_iteration"] for item in summaries] == [
        {"kind": "power"},
        {"kind": "wielandt", "shift_inverse_keff": 0.95},
    ]


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
    measurement.measure_direct(
        6,
        2,
        requested_threads=1,
        kind="measurement",
        repetition=0,
        iteration_id="power",
        shift_inverse_keff=None,
        placement="structured",
    )
    assert events == ["usage", "usage", "factor"]


def test_study_modes_and_cases_are_exact() -> None:
    """Smoke and the three single-thread baselines have fixed definitions."""
    smoke = _plan("smoke")
    assert smoke.output_name == "smoke.json"
    assert [request.record for request in smoke.requests] == [False, True, True]
    assert {request.case_id for request in smoke.requests} == {DIRECT_CASE_ID}
    direct = _plan("baseline")
    gmres = _plan("baseline", GMRES_JACOBI_CASE_ID)
    bicgstab = _plan("baseline", BICGSTAB_JACOBI_CASE_ID)
    assert direct.output_name == "baseline_direct.json"
    assert gmres.output_name == "baseline_gmres_jacobi.json"
    assert bicgstab.output_name == "baseline_bicgstab_jacobi.json"
    assert len(direct.requests) == len(gmres.requests) == len(bicgstab.requests) == 60
    assert {request.requested_threads for request in gmres.requests} == {1}
    expected_warmup = _request(6, 2, DIRECT_CASE_ID, record=False)
    assert direct.requests[::5] == (expected_warmup,) * 12
    with pytest.raises(ValueError, match="one of"):
        _plan("baseline", "gmres_ilu")


def test_selected_protocol_accepts_reusable_workload_and_warmup_inputs() -> None:
    """Selected runs accept checked workload and warm-up dimensions."""
    plan = _plan(
        "selected",
        workloads=((18, 6), (36, 12)),
        warmup=(6, 2),
        output_name="selected_direct.json",
    )
    assert plan.output_name == "selected_direct.json"
    assert plan.requests == (
        _request(6, 2, DIRECT_CASE_ID, record=False),
        _request(18, 6, DIRECT_CASE_ID, repetition=0),
        _request(6, 2, DIRECT_CASE_ID, record=False),
        _request(36, 12, DIRECT_CASE_ID, repetition=0),
    )
    assert [item["workload_id"] for item in runner._workloads(plan.requests)] == [
        "g6-z2-structured",
        "g18-z6-structured",
        "g36-z12-structured",
    ]

    iterative = _plan(
        "selected",
        GMRES_JACOBI_CASE_ID,
        workloads=((18, 12),),
        warmup=(6, 2),
        output_name="selected_iterative.json",
    )
    assert iterative.output_name == "selected_iterative.json"
    assert iterative.requests == (
        _request(6, 2, GMRES_JACOBI_CASE_ID, record=False),
        _request(18, 12, GMRES_JACOBI_CASE_ID, repetition=0),
    )
    with pytest.raises(ValueError, match="at least one workload"):
        _plan("selected", output_name="empty.json")
    with pytest.raises(ValueError, match="output name"):
        _plan("selected", workloads=((6, 2),))
    with pytest.raises(ValueError, match="groups must be"):
        _plan(
            "selected",
            workloads=((7, 2),),
            output_name="invalid.json",
        )
    with pytest.raises(ValueError, match="axial_layers must be"):
        _plan(
            "selected",
            workloads=((6, 0),),
            output_name="invalid.json",
        )
    with pytest.raises(ValueError, match="must be unique"):
        _plan(
            "selected",
            workloads=((6, 2), (6, 2)),
            output_name="duplicate.json",
        )


def test_selected_protocol_exposes_only_the_canonical_permuted_placement() -> None:
    """Selected runs can request the frozen permutation without seed controls."""
    plan = _plan(
        "selected",
        workloads=((18, 6),),
        warmup=(6, 2),
        output_name="permuted.json",
        placement="permuted",
    )
    assert plan.requests[0].placement == "permuted"
    measured = plan.requests[1]
    assert measured.placement == "permuted"
    assert runner._request_id(measured).startswith("g18-z6-permuted-")
    assert [item["workload_id"] for item in runner._workloads(plan.requests)] == [
        "g6-z2-permuted",
        "g18-z6-permuted",
    ]
    with pytest.raises(ValueError, match="placement must be"):
        _plan(
            "selected",
            workloads=((18, 6),),
            output_name="invalid.json",
            placement="arbitrary",
        )


def test_selected_protocol_supports_repetition_and_profile() -> None:
    """One selected protocol supports repeated and profiled GMRES assessment."""
    plan = _plan(
        "selected",
        GMRES_JACOBI_CASE_ID,
        workloads=((18, 12),),
        warmup=(6, 2),
        repetitions=3,
        profile=True,
        output_name="repeated.json",
    )
    assert plan.requests == (
        _request(6, 2, GMRES_JACOBI_CASE_ID, record=False),
        _request(18, 12, GMRES_JACOBI_CASE_ID, repetition=0),
        _request(18, 12, GMRES_JACOBI_CASE_ID, repetition=1),
        _request(18, 12, GMRES_JACOBI_CASE_ID, repetition=2),
        _request(18, 12, GMRES_JACOBI_CASE_ID, kind="profile"),
    )
    with pytest.raises(ValueError, match="positive"):
        _plan(
            "selected",
            workloads=((18, 12),),
            repetitions=0,
            output_name="bad.json",
        )


def test_selected_protocol_accepts_an_explicit_wielandt_policy() -> None:
    """Any checked workload can carry an explicit fixed shift."""
    plan = _plan(
        "selected",
        GMRES_JACOBI_CASE_ID,
        workloads=((18, 6),),
        warmup=(6, 2),
        output_name="shifted.json",
        iteration_id=WIELANDT_ITERATION_ID,
        shift_inverse_keff=0.95,
    )
    assert plan.requests[0] == _request(6, 2, GMRES_JACOBI_CASE_ID, record=False)
    measured = plan.requests[1]
    assert measured.iteration_id == WIELANDT_ITERATION_ID
    assert measured.shift_inverse_keff == pytest.approx(0.95)
    assert runner._request_id(measured).startswith(
        "g18-z6-structured-gmres_jacobi-wielandt-s0.95-"
    )
    settings, iteration = resolve_case(
        GMRES_JACOBI_CASE_ID, WIELANDT_ITERATION_ID, 0.95
    )
    assert iteration == iteration_record(WIELANDT_ITERATION_ID, 0.95)
    assert settings.eigenvalue_iteration.shift_inverse_keff == pytest.approx(0.95)
    with pytest.raises(ValueError, match="requires an explicit fixed shift"):
        _plan(
            "selected",
            workloads=((18, 6),),
            output_name="missing-shift.json",
            iteration_id=WIELANDT_ITERATION_ID,
        )


def test_result_reader_requires_an_exact_operator_policy(
    direct_observation: dict[str, object],
) -> None:
    """An outcome cannot add fields to its complete operator policy."""
    invalid = deepcopy(direct_observation)
    invalid["eigenvalue_iteration"] = {
        **iteration_record("power", None),
        "shift_inverse_keff": 1.0,
    }
    with pytest.raises(ValueError, match="invalid"):
        _document([invalid])


def test_baseline_resume_runs_only_missing_endpoint_after_cheap_warmup() -> None:
    """A partial baseline resumes its endpoint after the uniform cheap warm-up."""
    plan = _plan("baseline")
    baseline = plan.requests
    completed = [
        {"outcome_id": runner._request_id(request), "status": "success"}
        for request in baseline
        if request.record and (request.groups, request.axial_layers) != (72, 24)
    ]
    pending = runner._pending_groups(plan.groups, completed)
    resumed = (pending[0].warmup,) + pending[0].measurements
    assert resumed[0] == _request(6, 2, DIRECT_CASE_ID, record=False)
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


@pytest.mark.parametrize("warmup", [None, (6, 2)])
def test_terminal_failures_stop_dependent_workers(tmp_path, monkeypatch, warmup):
    """Fresh and resumed runs share the same terminal-failure dependencies."""
    calls = []

    def fail(*args, **_kwargs):
        calls.append(args)
        raise orchestration.PerformanceWorkerError("timeout", "worker deadline")

    monkeypatch.setattr(orchestration, "_launch_worker", fail)
    options = {
        "workloads": ((6, 6), (6, 12)),
        "warmup": warmup,
        "repetitions": 3,
        "profile": True,
        "output_name": "failures.json",
    }
    path = runner.run("selected", tmp_path, **options)
    document = results.read_document(path)
    expected_calls = [(6, 6), (6, 12)] if warmup is None else [(6, 2)]
    assert calls == expected_calls
    assert len(document["outcomes"]) == len(expected_calls)
    assert all(outcome["status"] == "failed" for outcome in document["outcomes"])
    runner.run("selected", tmp_path, resume=True, **options)
    assert calls == expected_calls
    assert results.read_document(path) == document


def test_profile_only_resume_keeps_its_warmup():
    """Profiling follows the same warm-up protocol after an interrupted run."""
    plan = _plan(
        "selected",
        workloads=((6, 6),),
        warmup=(6, 2),
        profile=True,
        output_name="profile.json",
    )
    measured = plan.groups[0].measurements[0]
    pending = runner._pending_groups(
        plan.groups,
        [
            {"outcome_id": runner._request_id(measured), "status": "success"},
        ],
    )
    assert len(pending) == 1
    assert pending[0].warmup == plan.groups[0].warmup
    assert [request.kind for request in pending[0].measurements] == ["profile"]


@pytest.mark.parametrize("dimensions", [(6, 1), (6, 3), (6, True), (6, 2.0)])
@pytest.mark.parametrize("as_warmup", [False, True])
def test_unsupported_dimensions_fail_during_planning(dimensions, as_warmup):
    """Workloads and warm-ups obey the frozen workload's dimension checks."""
    with pytest.raises((TypeError, ValueError), match="axial_layers"):
        _plan(
            "selected",
            workloads=((6, 2),) if as_warmup else (dimensions,),
            warmup=dimensions if as_warmup else None,
            output_name="invalid.json",
        )


def test_selected_repetitions_print_summaries(
    tmp_path,
    monkeypatch,
    direct_observation,
    capsys,
):
    """Repeated selected measurements expose their computed medians."""

    def measured(groups, axial_layers, **options):
        outcome = deepcopy(direct_observation)
        outcome.update(
            orchestration.outcome_fields(
                groups,
                axial_layers,
                options["case_id"],
                iteration=iteration_record(options["iteration_id"], None),
                requested_threads=options["requested_threads"],
                kind=options["kind"],
                repetition=options["repetition"],
                placement=options["placement"],
            )
        )
        return outcome

    monkeypatch.setattr(runner, "launch_outcome", measured)
    path = runner.run(
        "selected",
        tmp_path,
        workloads=((6, 2),),
        repetitions=3,
        output_name="repeated.json",
    )
    assert len(results.read_document(path)["outcomes"]) == 3
    output = capsys.readouterr().out
    assert "Measurement medians:" in output
    assert "g6-z2-structured, direct" in output
