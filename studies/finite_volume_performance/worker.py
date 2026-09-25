"""Internal fresh-process entry point for one performance outcome."""

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
    parser.add_argument("--solver-case", required=True)
    parser.add_argument("--iteration", required=True)
    parser.add_argument("--shift-inverse-keff", type=float)
    parser.add_argument("--address-space-limit-bytes", type=int, required=True)
    parser.add_argument("--output-file-limit-bytes", type=int, required=True)
    return parser.parse_args()


def _set_resource_limit(resource_id: int, limit_bytes: int, name: str) -> None:
    """Set one positive byte limit without exceeding an inherited hard limit."""
    if limit_bytes <= 0:
        raise ValueError(f"{name} must be positive")
    _, inherited_hard_limit = resource.getrlimit(resource_id)
    limit = limit_bytes
    if inherited_hard_limit != resource.RLIM_INFINITY:
        limit = min(limit, inherited_hard_limit)
    resource.setrlimit(resource_id, (limit, limit))


def main() -> None:
    """Import the measurement implementation only after worker startup."""
    arguments = _arguments()
    _set_resource_limit(
        resource.RLIMIT_AS,
        arguments.address_space_limit_bytes,
        "address-space-limit-bytes",
    )
    _set_resource_limit(
        resource.RLIMIT_FSIZE,
        arguments.output_file_limit_bytes,
        "output-file-limit-bytes",
    )
    if arguments.solver_case == "direct":
        # pylint: disable-next=import-outside-toplevel
        from studies.finite_volume_performance.measurement import measure_direct

        measure = measure_direct
    elif arguments.solver_case == "gmres_jacobi":
        # pylint: disable-next=import-outside-toplevel
        from studies.finite_volume_performance.gmres import measure_gmres_jacobi

        measure = measure_gmres_jacobi
    else:
        raise ValueError("solver-case must be 'direct' or 'gmres_jacobi'")
    outcome = measure(
        arguments.groups,
        arguments.axial_layers,
        requested_threads=arguments.requested_threads,
        kind=arguments.kind,
        repetition=arguments.repetition,
        iteration_id=arguments.iteration,
        shift_inverse_keff=arguments.shift_inverse_keff,
    )
    print(json.dumps(outcome, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
