"""Maintained orchestration for the many-group performance study."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from examples.many_group_performance.orchestration import (
    DEFAULT_ADDRESS_SPACE_LIMIT_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    launch_observation,
)
from examples.many_group_performance.results import (
    build_document,
    observation_metrics,
    summarize_observations,
    write_document,
)

DEFAULT_OUTPUT_DIR = Path("artifacts/examples/many_group_performance")
_THREAD_COUNTS = (1, 2, 4, 6)
_THREAD_SCREEN_WORKLOAD = (72, 10)


@dataclass(frozen=True)
class WorkerRequest:
    """Describe one fresh worker and whether its result is retained."""

    groups: int
    axial_layers: int
    requested_threads: int
    kind: str
    repetition: int | None
    retain: bool


def _requests(mode: str) -> tuple[WorkerRequest, ...]:
    """Return the frozen worker sequence for one maintained mode."""
    # Keep numerical imports out of the command's help path.
    # pylint: disable-next=import-outside-toplevel
    from examples.many_group_performance import workload

    if mode == "smoke":
        groups, layers = 6, workload.SMOKE_AXIAL_LAYER_COUNT
        return (
            WorkerRequest(groups, layers, 1, "measurement", 0, False),
            WorkerRequest(groups, layers, 1, "measurement", 0, True),
            WorkerRequest(groups, layers, 1, "profile", None, True),
        )
    if mode == "full":
        requests = []
        endpoint = (72, 20)
        for groups, layers in workload.baseline_workloads():
            if (groups, layers) != endpoint:
                requests.append(
                    WorkerRequest(groups, layers, 1, "measurement", 0, False)
                )
            repetitions = 1 if (groups, layers) == endpoint else 3
            requests.extend(
                WorkerRequest(groups, layers, 1, "measurement", repetition, True)
                for repetition in range(repetitions)
            )
            requests.append(WorkerRequest(groups, layers, 1, "profile", None, True))
        return tuple(requests)
    if mode == "thread-screen":
        groups, layers = _THREAD_SCREEN_WORKLOAD
        requests = []
        for threads in _THREAD_COUNTS:
            requests.append(
                WorkerRequest(groups, layers, threads, "measurement", 0, False)
            )
            requests.extend(
                WorkerRequest(groups, layers, threads, "measurement", repetition, True)
                for repetition in range(3)
            )
        return tuple(requests)
    raise ValueError("mode must be 'smoke', 'full', or 'thread-screen'")


def _workloads(requests: Iterable[WorkerRequest]) -> list[dict[str, object]]:
    """Return each workload definition once in first-use order."""
    # pylint: disable-next=import-outside-toplevel
    from examples.many_group_performance.measurement import workload_record

    dimensions = []
    for request in requests:
        candidate = (request.groups, request.axial_layers)
        if candidate not in dimensions:
            dimensions.append(candidate)
    return [workload_record(*dimensions_) for dimensions_ in dimensions]


def _print_observation(observation: dict[str, object]) -> None:
    """Print one concise retained-observation summary."""
    solve = observation["solve"]
    checks = solve["numerical_checks"]
    metrics = observation_metrics(observation)
    print(
        f"  retained {observation['observation_id']}: "
        f"wall={metrics['wall_time_seconds']:.3f} s, "
        f"peak={metrics['peak_rss_bytes'] / 1024**2:.1f} MiB, "
        f"iterations={solve['outer_iterations']}, "
        f"k_eff={checks['keff']:.12f}"
    )


def _print_repeated_summaries(mode: str, observations: list[dict[str, object]]) -> None:
    """Print repeated-observation statistics for full measurement modes."""
    summaries = summarize_observations(observations)
    if mode == "full":
        print("Repeated-workload medians:")
        for summary in summaries:
            wall = summary["wall_time_seconds"]["median"]
            peak = summary["peak_rss_bytes"]["median"] / 1024**2
            print(f"  {summary['workload_id']}: wall={wall:.3f} s, peak={peak:.1f} MiB")
        return
    if mode == "thread-screen":
        by_threads = {summary["requested_threads"]: summary for summary in summaries}
        reference = by_threads[1]["wall_time_seconds"]["median"]
        print("Thread-screen medians:")
        for threads in _THREAD_COUNTS:
            summary = by_threads[threads]
            wall = summary["wall_time_seconds"]["median"]
            speedup = reference / wall
            efficiency = speedup / threads
            cpu_ratio = summary["cpu_time_to_wall_time_ratio"]["median"]
            peak = summary["peak_rss_bytes"]["median"] / 1024**2
            print(
                f"  threads={threads}: wall={wall:.3f} s, speedup={speedup:.3f}, "
                f"efficiency={efficiency:.3f}, CPU/wall={cpu_ratio:.3f}, "
                f"peak={peak:.1f} MiB"
            )


def run(
    mode: str,
    output_dir: Path,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    address_space_limit_bytes: int = DEFAULT_ADDRESS_SPACE_LIMIT_BYTES,
) -> Path:
    """Execute one maintained mode and return its checked JSON path."""
    requests = _requests(mode)
    workloads = _workloads(requests)
    observations: list[dict[str, object]] = []
    output_path = output_dir / f"{mode.replace('-', '_')}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_document(
        output_path,
        build_document(workloads=workloads, observations=observations),
    )

    print(f"Running many-group performance mode: {mode}")
    for index, request in enumerate(requests, start=1):
        disposition = "retained" if request.retain else "discarded warm-up"
        print(
            f"[{index}/{len(requests)}] g={request.groups}, "
            f"z={request.axial_layers}, threads={request.requested_threads}, "
            f"{request.kind}, {disposition}"
        )
        observation = launch_observation(
            request.groups,
            request.axial_layers,
            requested_threads=request.requested_threads,
            kind=request.kind,
            repetition=request.repetition,
            timeout_seconds=timeout_seconds,
            address_space_limit_bytes=address_space_limit_bytes,
        )
        if request.retain:
            observations.append(observation)
            document = build_document(
                workloads=workloads,
                observations=observations,
            )
            write_document(output_path, document)
            _print_observation(observation)

    _print_repeated_summaries(mode, observations)
    print(f"Checked raw observations: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse the maintained performance-example interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("smoke", "full", "thread-screen"),
        nargs="?",
        default="smoke",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    return parser.parse_args()


def main() -> None:
    """Run the selected maintained performance mode."""
    arguments = parse_args()
    run(
        arguments.mode,
        arguments.output_dir,
        timeout_seconds=arguments.timeout_seconds,
        address_space_limit_bytes=int(arguments.address_space_limit_gib * 1024**3),
    )
