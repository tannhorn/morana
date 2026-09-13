"""Internal fresh-process entry point for one performance observation."""

from __future__ import annotations

import argparse
import json
import resource


def _arguments() -> argparse.Namespace:
    """Parse one internal worker request without importing numerical libraries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--axial-layers", type=int, required=True)
    parser.add_argument("--requested-threads", type=int, required=True)
    parser.add_argument("--kind", choices=("measurement", "profile"), required=True)
    parser.add_argument("--repetition", type=int)
    parser.add_argument("--address-space-limit-bytes", type=int, required=True)
    parser.add_argument("--output-file-limit-bytes", type=int, required=True)
    return parser.parse_args()


def _limit_address_space(limit_bytes: int) -> None:
    """Bound worker virtual memory before importing numerical libraries."""
    if limit_bytes <= 0:
        raise ValueError("address-space-limit-bytes must be positive")
    _, inherited_hard_limit = resource.getrlimit(resource.RLIMIT_AS)
    limit = limit_bytes
    if inherited_hard_limit != resource.RLIM_INFINITY:
        limit = min(limit, inherited_hard_limit)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def _limit_output_files(limit_bytes: int) -> None:
    """Bound regular-file output before the worker emits captured JSON."""
    if limit_bytes <= 0:
        raise ValueError("output-file-limit-bytes must be positive")
    _, inherited_hard_limit = resource.getrlimit(resource.RLIMIT_FSIZE)
    limit = limit_bytes
    if inherited_hard_limit != resource.RLIM_INFINITY:
        limit = min(limit, inherited_hard_limit)
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))


def main() -> None:
    """Import the measurement implementation only after worker startup."""
    arguments = _arguments()
    _limit_address_space(arguments.address_space_limit_bytes)
    _limit_output_files(arguments.output_file_limit_bytes)
    # pylint: disable-next=import-outside-toplevel
    from examples.many_group_performance.measurement import measure_workload

    observation = measure_workload(
        arguments.groups,
        arguments.axial_layers,
        requested_threads=arguments.requested_threads,
        kind=arguments.kind,
        repetition=arguments.repetition,
    )
    print(json.dumps(observation, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
