"""Maintained orchestration for the many-group performance study."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from studies.finite_volume_performance.orchestration import (
    DEFAULT_ADDRESS_SPACE_LIMIT_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    launch_outcome,
    outcome_identifier,
)
from studies.finite_volume_performance.results import (
    build_document,
    outcome_metrics,
    read_document,
    summarize_outcomes,
    write_document,
)

DEFAULT_OUTPUT_DIR = Path("artifacts/studies/finite_volume_performance")
_THREAD_COUNTS = (1, 2, 4, 6)
_THREAD_SCREEN_WORKLOAD = (72, 10)
_BASELINE_ENDPOINT = (72, 20)
_BASELINE_ENDPOINT_WARMUP = (72, 5)
_SOLVER_SCREEN_WORKLOADS = ((36, 10, 3), (72, 5, 1))


@dataclass(frozen=True)
class WorkerRequest:
    """Describe one fresh worker and whether its outcome is retained."""

    groups: int
    axial_layers: int
    case_id: str
    requested_threads: int = 1
    kind: str = "measurement"
    repetition: int | None = None
    retain: bool = True


@dataclass(frozen=True)
class RunPlan:
    """Describe one resolved maintained-mode execution."""

    mode: str
    output_name: str
    requests: tuple[WorkerRequest, ...]


def _resolve_case_id(case_id: str | None) -> str:
    """Resolve and validate the CLI's silent solver default."""
    # pylint: disable-next=import-outside-toplevel
    from studies.finite_volume_performance.cases import REFERENCE_CASE_ID, solver_case

    resolved = REFERENCE_CASE_ID if case_id is None else case_id
    return solver_case(resolved).case_id


def _cell_requests(
    groups: int,
    axial_layers: int,
    case_id: str,
    *,
    requested_threads: int = 1,
    repetitions: int = 3,
    warmup: bool = True,
    profile: bool = False,
) -> tuple[WorkerRequest, ...]:
    """Return the standard fresh-worker protocol for one measured cell."""
    requests = []
    if warmup:
        requests.append(
            WorkerRequest(
                groups,
                axial_layers,
                case_id,
                requested_threads=requested_threads,
                retain=False,
            )
        )
    requests.extend(
        WorkerRequest(
            groups,
            axial_layers,
            case_id,
            requested_threads=requested_threads,
            repetition=repetition,
        )
        for repetition in range(repetitions)
    )
    if profile:
        requests.append(
            WorkerRequest(
                groups,
                axial_layers,
                case_id,
                requested_threads=requested_threads,
                kind="profile",
            )
        )
    return tuple(requests)


def _baseline_requests(case_id: str) -> tuple[WorkerRequest, ...]:
    """Return the complete Cartesian baseline protocol for one solver case."""
    # pylint: disable-next=import-outside-toplevel
    from studies.finite_volume_performance.workload import baseline_workloads

    requests = []
    for groups, layers in baseline_workloads():
        if (groups, layers) == _BASELINE_ENDPOINT:
            requests.append(
                WorkerRequest(
                    *_BASELINE_ENDPOINT_WARMUP,
                    case_id,
                    retain=False,
                )
            )
        requests.extend(
            _cell_requests(
                groups,
                layers,
                case_id,
                warmup=(groups, layers) != _BASELINE_ENDPOINT,
                profile=True,
            )
        )
    return tuple(requests)


def _thread_requests(case_id: str) -> tuple[WorkerRequest, ...]:
    """Return the workstation thread-screen protocol for one solver case."""
    groups, layers = _THREAD_SCREEN_WORKLOAD
    return tuple(
        request
        for threads in _THREAD_COUNTS
        for request in _cell_requests(
            groups, layers, case_id, requested_threads=threads
        )
    )


def _solver_screen_requests() -> tuple[WorkerRequest, ...]:
    """Return the bounded single-thread solver-and-ordering protocol."""
    # pylint: disable-next=import-outside-toplevel
    from studies.finite_volume_performance.cases import case_records

    requests = []
    for case in case_records():
        case_id = str(case["case_id"])
        for groups, layers, repetitions in _SOLVER_SCREEN_WORKLOADS:
            requests.extend(
                _cell_requests(
                    groups,
                    layers,
                    case_id,
                    repetitions=repetitions,
                )
            )
    return tuple(requests)


def _plan(mode: str, solver_case: str | None = None) -> RunPlan:
    """Resolve one CLI mode into an explicit immutable execution plan."""
    if mode == "solver-screen":
        if solver_case is not None:
            raise ValueError("--solver does not apply to solver-screen mode")
        return RunPlan(mode, "solver_screen.json", _solver_screen_requests())
    case_id = _resolve_case_id(solver_case)
    if mode == "smoke":
        if solver_case is not None:
            raise ValueError("--solver does not apply to smoke mode")
        return RunPlan(
            mode,
            "smoke.json",
            _cell_requests(6, 2, case_id, repetitions=1, profile=True),
        )
    if mode == "baseline":
        return RunPlan(mode, f"baseline_{case_id}.json", _baseline_requests(case_id))
    if mode == "thread-screen":
        return RunPlan(mode, f"thread_screen_{case_id}.json", _thread_requests(case_id))
    raise ValueError(
        "mode must be 'smoke', 'baseline', 'thread-screen', or 'solver-screen'"
    )


def _request_id(request: WorkerRequest) -> str:
    """Return the deterministic identifier for one worker request."""
    return outcome_identifier(
        request.groups,
        request.axial_layers,
        request.case_id,
        requested_threads=request.requested_threads,
        kind=request.kind,
        repetition=request.repetition,
    )


def _cell(request: WorkerRequest) -> tuple[str, int, int, int]:
    """Return the terminal-failure cell for one request."""
    return (
        request.case_id,
        request.groups,
        request.axial_layers,
        request.requested_threads,
    )


def _pending_requests(
    requests: tuple[WorkerRequest, ...], outcomes: Iterable[dict[str, object]]
) -> tuple[WorkerRequest, ...]:
    """Return missing work and only warm-ups preceding missing measurements."""
    by_id = {_request_id(request): request for request in requests}
    recorded = {str(outcome["outcome_id"]): outcome for outcome in outcomes}
    unexpected = set(recorded) - set(by_id)
    if unexpected:
        raise ValueError(
            f"cannot resume with unexpected outcomes: {sorted(unexpected)}"
        )
    failed_cells = {
        _cell(by_id[outcome_id])
        for outcome_id, outcome in recorded.items()
        if outcome["status"] == "failed"
    }
    missing = {
        request_id
        for request_id, request in by_id.items()
        if request.retain
        and _cell(request) not in failed_cells
        and request_id not in recorded
    }
    pending = []
    for index, request in enumerate(requests):
        if _cell(request) in failed_cells:
            continue
        if request.retain:
            if _request_id(request) in missing:
                pending.append(request)
            continue
        following = []
        for candidate in requests[index + 1 :]:
            if not candidate.retain:
                break
            following.append(candidate)
        if any(
            candidate.kind == "measurement" and _request_id(candidate) in missing
            for candidate in following
        ):
            pending.append(request)
    return tuple(pending)


def _workloads(requests: Iterable[WorkerRequest]) -> list[dict[str, object]]:
    """Return each workload definition once in first-use order."""
    # pylint: disable-next=import-outside-toplevel
    from studies.finite_volume_performance.measurement import workload_record

    dimensions = []
    for request in requests:
        candidate = (request.groups, request.axial_layers)
        if candidate not in dimensions:
            dimensions.append(candidate)
    return [workload_record(*item) for item in dimensions]


def _print_outcome(outcome: dict[str, object]) -> None:
    """Print one concise retained or terminal outcome."""
    if outcome["status"] == "failed":
        print(f"  terminal {outcome['outcome_id']}: {outcome['failure']['kind']}")
        return
    solve = outcome["solve"]
    metrics = outcome_metrics(outcome)
    detail = f"outer={solve['outer_iterations']}"
    if "total_krylov_iterations" in solve:
        detail += f", krylov={solve['total_krylov_iterations']}"
    print(
        f"  retained {outcome['outcome_id']}: "
        f"wall={metrics['wall_time_seconds']:.3f} s, "
        f"peak={metrics['peak_rss_bytes'] / 1024**2:.1f} MiB, {detail}"
    )


def _print_summaries(plan: RunPlan, outcomes: list[dict[str, object]]) -> None:
    """Print repeated-run summaries appropriate to the resolved plan."""
    summaries = summarize_outcomes(outcomes)
    if plan.mode == "baseline":
        print("Baseline medians:")
        for summary in summaries:
            wall = summary["wall_time_seconds"]["median"]
            peak = summary["peak_rss_bytes"]["median"] / 1024**2
            print(f"  {summary['workload_id']}: wall={wall:.3f} s, peak={peak:.1f} MiB")
    elif plan.mode == "thread-screen" and summaries:
        reference = next(
            item["wall_time_seconds"]["median"]
            for item in summaries
            if item["requested_threads"] == 1
        )
        print("Thread-screen medians:")
        for summary in summaries:
            threads = summary["requested_threads"]
            wall = summary["wall_time_seconds"]["median"]
            speedup = reference / wall
            cpu_ratio = summary["cpu_time_to_wall_time_ratio"]["median"]
            peak = summary["peak_rss_bytes"]["median"] / 1024**2
            print(
                f"  threads={threads}: wall={wall:.3f} s, "
                f"speedup={speedup:.3f}, efficiency={speedup / threads:.3f}, "
                f"CPU/wall={cpu_ratio:.3f}, peak={peak:.1f} MiB"
            )


def run(
    mode: str,
    output_dir: Path,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    address_space_limit_bytes: int = DEFAULT_ADDRESS_SPACE_LIMIT_BYTES,
    resume: bool = False,
    solver_case: str | None = None,
) -> Path:
    """Execute one maintained mode and return its checked JSON path."""
    plan = _plan(mode, solver_case)
    workloads = _workloads(plan.requests)
    output_path = output_dir / plan.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        document = read_document(output_path)
        if document["workloads"] != workloads:
            raise ValueError(
                "cannot resume: stored workloads differ from this checkout"
            )
        outcomes = list(document["outcomes"])
    else:
        outcomes = []
        write_document(output_path, build_document(workloads=workloads, outcomes=[]))
    pending = _pending_requests(plan.requests, outcomes)
    disposition = "Resuming" if resume else "Running"
    print(f"{disposition} many-group performance mode: {plan.mode}")
    for index, request in enumerate(pending, start=1):
        retention = "retained" if request.retain else "discarded warm-up"
        print(
            f"[{index}/{len(pending)}] {request.case_id}, g={request.groups}, "
            f"z={request.axial_layers}, threads={request.requested_threads}, "
            f"{request.kind}, {retention}"
        )
        outcome = launch_outcome(
            request.groups,
            request.axial_layers,
            case_id=request.case_id,
            requested_threads=request.requested_threads,
            kind=request.kind,
            repetition=request.repetition,
            timeout_seconds=timeout_seconds,
            address_space_limit_bytes=address_space_limit_bytes,
        )
        if request.retain or outcome["status"] == "failed":
            outcomes.append(outcome)
            write_document(
                output_path,
                build_document(workloads=workloads, outcomes=outcomes),
            )
            _print_outcome(outcome)
    _print_summaries(plan, outcomes)
    print(f"Checked outcomes: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse the maintained performance-study interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("smoke", "baseline", "thread-screen", "solver-screen"),
        nargs="?",
        default="smoke",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--solver",
        dest="solver_case",
        metavar="CASE_ID",
        help=(
            "solver case for baseline or thread-screen mode; omitted uses "
            "ordinary factorized power iteration"
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(DEFAULT_TIMEOUT_SECONDS),
        help="finite deadline applied independently to every worker",
    )
    parser.add_argument(
        "--address-space-limit-gib",
        type=float,
        default=DEFAULT_ADDRESS_SPACE_LIMIT_BYTES / 1024**3,
        help="address-space ceiling applied before numerical-library imports",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append only missing outcomes to the selected mode's JSON",
    )
    return parser.parse_args()


def main() -> None:
    """Run the selected maintained performance mode."""
    arguments = parse_args()
    run(
        arguments.mode,
        arguments.output_dir,
        timeout_seconds=arguments.timeout_seconds,
        address_space_limit_bytes=int(arguments.address_space_limit_gib * 1024**3),
        resume=arguments.resume,
        solver_case=arguments.solver_case,
    )
