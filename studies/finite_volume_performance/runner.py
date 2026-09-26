"""Maintained orchestration for the many-group performance study."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from studies.finite_volume_performance.cases import (
    CASE_IDS,
    DIRECT_CASE_ID,
    iteration_record,
)
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
_BASELINE_WARMUP = (6, 2)


@dataclass(frozen=True)
# The request deliberately carries all independently persisted outcome axes.
# pylint: disable-next=too-many-instance-attributes
class WorkerRequest:
    """Describe one fresh worker and whether its outcome is recorded."""

    groups: int
    axial_layers: int
    case_id: str
    placement: str
    requested_threads: int
    iteration_id: str
    shift_inverse_keff: float | None
    kind: str
    repetition: int | None
    record: bool


@dataclass(frozen=True)
class ExecutionGroup:
    """Own one workload's warm-up and ordered recorded workers."""

    warmup: WorkerRequest | None
    measurements: tuple[WorkerRequest, ...]

    @property
    def requests(self) -> tuple[WorkerRequest, ...]:
        """Return the prerequisite followed by recorded workers."""
        if self.warmup is None:
            return self.measurements
        return (self.warmup,) + self.measurements


@dataclass(frozen=True)
class RunPlan:
    """Describe one resolved maintained-mode execution."""

    mode: str
    output_name: str
    groups: tuple[ExecutionGroup, ...]

    @property
    def requests(self) -> tuple[WorkerRequest, ...]:
        """Return all workers in execution order for workload provenance."""
        return tuple(request for group in self.groups for request in group.requests)


def _resolve_case_id(case_id: str | None) -> str:
    """Resolve and validate the selected baseline case."""
    resolved = DIRECT_CASE_ID if case_id is None else case_id
    if resolved not in CASE_IDS:
        raise ValueError(f"--solver must be one of {CASE_IDS}")
    return resolved


def _execution_groups(
    workloads: tuple[tuple[int, int], ...],
    case_id: str,
    warmup: tuple[int, int] | None,
    *,
    repetitions: int,
    profile: bool,
    iteration_id: str,
    shift_inverse_keff: float | None,
    placement: str,
) -> tuple[ExecutionGroup, ...]:
    """Build the same checked worker protocol for every selected workload."""
    # pylint: disable-next=import-outside-toplevel,protected-access
    from studies.finite_volume_performance.workload import (
        _require_group_count,
        _require_axial_layers,
        _require_placement,
    )

    iteration_record(iteration_id, shift_inverse_keff)
    _require_placement(placement)
    if len(set(workloads)) != len(workloads):
        raise ValueError("selected workloads must be unique")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int):
        raise TypeError("selected repetitions must be an integer")
    if repetitions < 1:
        raise ValueError("selected repetitions must be positive")
    for groups, layers in workloads + (() if warmup is None else (warmup,)):
        _require_group_count(groups)
        _require_axial_layers(layers)
    warmup_request = (
        None
        if warmup is None
        else WorkerRequest(
            *warmup,
            case_id,
            placement,
            1,
            "power",
            None,
            "measurement",
            None,
            False,
        )
    )
    result = []
    for groups, layers in workloads:
        request = WorkerRequest(
            groups,
            layers,
            case_id,
            placement,
            1,
            iteration_id,
            shift_inverse_keff,
            "measurement",
            None,
            True,
        )
        measurements = tuple(
            replace(request, repetition=index) for index in range(repetitions)
        )
        if profile:
            measurements += (replace(request, kind="profile"),)
        result.append(ExecutionGroup(warmup_request, measurements))
    return tuple(result)


def _plan(
    mode: str,
    solver_case: str | None,
    *,
    workloads: tuple[tuple[int, int], ...],
    warmup: tuple[int, int] | None,
    output_name: str | None,
    repetitions: int,
    profile: bool,
    iteration_id: str,
    shift_inverse_keff: float | None,
    placement: str,
) -> RunPlan:
    """Resolve one CLI mode into an explicit immutable execution plan."""
    selected_options = (
        workloads,
        warmup,
        output_name,
        repetitions != 1,
        profile,
        iteration_id != "power",
        shift_inverse_keff is not None,
        placement != "structured",
    )
    if mode == "smoke":
        if solver_case is not None:
            raise ValueError("--solver applies only to baseline or selected mode")
        if any(selected_options):
            raise ValueError("workload selection applies only to selected mode")
        return RunPlan(
            mode,
            "smoke.json",
            _execution_groups(
                ((6, 2),),
                DIRECT_CASE_ID,
                (6, 2),
                repetitions=1,
                profile=True,
                iteration_id="power",
                shift_inverse_keff=None,
                placement=placement,
            ),
        )
    if mode == "baseline":
        if any(selected_options):
            raise ValueError("workload selection applies only to selected mode")
        case_id = _resolve_case_id(solver_case)
        # pylint: disable-next=import-outside-toplevel
        from studies.finite_volume_performance.workload import baseline_workloads

        return RunPlan(
            mode,
            f"baseline_{case_id}.json",
            _execution_groups(
                tuple(baseline_workloads()),
                case_id,
                _BASELINE_WARMUP,
                repetitions=3,
                profile=True,
                iteration_id="power",
                shift_inverse_keff=None,
                placement=placement,
            ),
        )
    if mode == "selected":
        if not workloads:
            raise ValueError("selected mode requires at least one workload")
        if output_name is None:
            raise ValueError("selected mode requires an output name")
        if Path(output_name).name != output_name or not output_name.endswith(".json"):
            raise ValueError("output name must be a JSON filename")
        case_id = _resolve_case_id(solver_case)
        return RunPlan(
            mode,
            output_name,
            _execution_groups(
                workloads,
                case_id,
                warmup,
                repetitions=repetitions,
                profile=profile,
                iteration_id=iteration_id,
                shift_inverse_keff=shift_inverse_keff,
                placement=placement,
            ),
        )
    raise ValueError("mode must be 'smoke', 'baseline', or 'selected'")


def _request_id(request: WorkerRequest) -> str:
    """Return the deterministic identifier for one worker request."""
    return outcome_identifier(
        request.groups,
        request.axial_layers,
        request.case_id,
        iteration_record(request.iteration_id, request.shift_inverse_keff),
        requested_threads=request.requested_threads,
        kind=request.kind,
        repetition=request.repetition,
        placement=request.placement,
    )


def _pending_groups(
    groups: tuple[ExecutionGroup, ...], outcomes: Iterable[dict[str, object]]
) -> tuple[ExecutionGroup, ...]:
    """Resume incomplete groups with their warm-ups, preserving terminal failures."""
    recorded = {str(outcome["outcome_id"]): outcome for outcome in outcomes}
    expected = {_request_id(request) for group in groups for request in group.requests}
    unexpected = set(recorded) - expected
    if unexpected:
        raise ValueError(
            f"cannot resume with unexpected outcomes: {sorted(unexpected)}"
        )
    failed = {
        identifier
        for identifier, outcome in recorded.items()
        if outcome["status"] == "failed"
    }
    pending = []
    for group in groups:
        identifiers = {_request_id(request) for request in group.measurements}
        if group.warmup is not None:
            identifiers.add(_request_id(group.warmup))
        if identifiers & failed:
            continue
        missing = tuple(
            request
            for request in group.measurements
            if _request_id(request) not in recorded
        )
        if missing:
            pending.append(replace(group, measurements=missing))
    return tuple(pending)


def _workloads(requests: Iterable[WorkerRequest]) -> list[dict[str, object]]:
    """Return each workload definition once in first-use order."""
    # pylint: disable-next=import-outside-toplevel
    from studies.finite_volume_performance.measurement import workload_record

    dimensions = []
    for request in requests:
        candidate = (request.groups, request.axial_layers, request.placement)
        if candidate not in dimensions:
            dimensions.append(candidate)
    return [workload_record(*item) for item in dimensions]


def _print_outcome(outcome: dict[str, object]) -> None:
    """Print one concise recorded or terminal outcome."""
    if outcome["status"] == "failed":
        print(f"  terminal {outcome['outcome_id']}: {outcome['failure']['kind']}")
        return
    solve = outcome["solve"]
    metrics = outcome_metrics(outcome)
    detail = f"outer={solve['outer_iterations']}"
    if "total_krylov_iterations" in solve:
        detail += f", krylov={solve['total_krylov_iterations']}"
    print(
        f"  recorded {outcome['outcome_id']}: "
        f"wall={metrics['wall_time_seconds']:.3f} s, "
        f"peak={metrics['peak_rss_bytes'] / 1024**2:.1f} MiB, {detail}"
    )


def _print_summaries(outcomes: list[dict[str, object]]) -> None:
    """Print summaries whenever enough repeated measurements are available."""
    summaries = summarize_outcomes(outcomes)
    if summaries:
        print("Measurement medians:")
    for summary in summaries:
        wall = summary["wall_time_seconds"]["median"]
        peak = summary["peak_rss_bytes"]["median"] / 1024**2
        print(
            f"  {summary['workload_id']}, {summary['case_id']}, "
            f"{summary['eigenvalue_iteration']}: "
            f"wall={wall:.3f} s, peak={peak:.1f} MiB"
        )


def run(
    mode: str,
    output_dir: Path,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    address_space_limit_bytes: int = DEFAULT_ADDRESS_SPACE_LIMIT_BYTES,
    resume: bool = False,
    solver_case: str | None = None,
    workloads: tuple[tuple[int, int], ...] = (),
    warmup: tuple[int, int] | None = None,
    output_name: str | None = None,
    repetitions: int = 1,
    profile: bool = False,
    iteration_id: str = "power",
    shift_inverse_keff: float | None = None,
    placement: str = "structured",
) -> Path:
    """Execute one maintained mode and return its checked JSON path."""
    plan = _plan(
        mode,
        solver_case,
        workloads=workloads,
        warmup=warmup,
        output_name=output_name,
        repetitions=repetitions,
        profile=profile,
        iteration_id=iteration_id,
        shift_inverse_keff=shift_inverse_keff,
        placement=placement,
    )
    workload_records = _workloads(plan.requests)
    output_path = output_dir / plan.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        document = read_document(output_path)
        if document["workloads"] != workload_records:
            raise ValueError(
                "cannot resume: stored workloads differ from this checkout"
            )
        outcomes = list(document["outcomes"])
    else:
        outcomes = []
        write_document(
            output_path,
            build_document(workloads=workload_records, outcomes=[]),
        )
    pending = _pending_groups(plan.groups, outcomes)
    disposition = "Resuming" if resume else "Running"
    print(f"{disposition} many-group performance mode: {plan.mode}")
    while pending:
        group = pending[0]
        for request in group.requests:
            disposition = "recorded" if request.record else "warm-up"
            print(
                f"  {request.case_id}, g={request.groups}, "
                f"z={request.axial_layers}, {request.kind}, {disposition}"
                f", placement={request.placement}"
            )
            outcome = launch_outcome(
                request.groups,
                request.axial_layers,
                case_id=request.case_id,
                iteration_id=request.iteration_id,
                shift_inverse_keff=request.shift_inverse_keff,
                requested_threads=request.requested_threads,
                kind=request.kind,
                repetition=request.repetition,
                timeout_seconds=timeout_seconds,
                address_space_limit_bytes=address_space_limit_bytes,
                placement=request.placement,
            )
            if request.record or outcome["status"] == "failed":
                outcomes.append(outcome)
                write_document(
                    output_path,
                    build_document(workloads=workload_records, outcomes=outcomes),
                )
                _print_outcome(outcome)
            if outcome["status"] == "failed":
                break
        pending = _pending_groups(plan.groups, outcomes)
    _print_summaries(outcomes)
    print(f"Checked outcomes: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse the maintained performance-study interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("smoke", "baseline", "selected"),
        nargs="?",
        default="smoke",
    )
    parser.add_argument(
        "--iteration",
        choices=("power", "wielandt"),
        default="power",
        help="criticality operator for selected mode (default: power)",
    )
    parser.add_argument(
        "--shift-inverse-keff",
        type=float,
        help="explicit fixed inverse-keff shift for selected Wielandt runs",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--solver",
        dest="solver_case",
        metavar="CASE_ID",
        help=("solver case for baseline or selected mode"),
    )
    parser.add_argument(
        "--workload",
        nargs=2,
        type=int,
        action="append",
        default=[],
        metavar=("GROUPS", "LAYERS"),
        help="workload to record in selected mode; may be repeated",
    )
    parser.add_argument(
        "--placement",
        choices=("structured", "permuted"),
        default="structured",
        help="placement family for selected workloads (default: structured)",
    )
    parser.add_argument(
        "--warmup",
        nargs=2,
        type=int,
        metavar=("GROUPS", "LAYERS"),
        help="optional unrecorded workload before each selected workload",
    )
    parser.add_argument(
        "--output-name",
        help="JSON filename required by selected mode",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="recorded measurements per selected workload (default: 1)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="also record one function profile per selected workload",
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
        help="append only missing outcomes to the chosen mode's JSON",
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
        workloads=tuple(tuple(item) for item in arguments.workload),
        warmup=None if arguments.warmup is None else tuple(arguments.warmup),
        output_name=arguments.output_name,
        repetitions=arguments.repetitions,
        profile=arguments.profile,
        iteration_id=arguments.iteration,
        shift_inverse_keff=arguments.shift_inverse_keff,
        placement=arguments.placement,
    )
