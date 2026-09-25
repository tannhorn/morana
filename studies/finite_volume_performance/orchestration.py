"""Fresh-process orchestration for many-group performance outcomes."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from typing import Any

from studies.finite_volume_performance.cases import (
    iteration_identifier,
    iteration_record,
)

THREAD_ENVIRONMENT_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_REPOSITORY_ROOT = Path(__file__).parents[2]
_WORKER_MODULE = "studies.finite_volume_performance.worker"
DEFAULT_TIMEOUT_SECONDS = 2 * 60 * 60
DEFAULT_ADDRESS_SPACE_LIMIT_BYTES = 22 * 1024**3
MAX_CAPTURED_OUTPUT_BYTES = 64 * 1024**2
_TERMINATION_GRACE_SECONDS = 5.0


class PerformanceWorkerError(RuntimeError):
    """Report a classified fresh-worker failure."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def worker_environment(requested_threads: int) -> dict[str, str]:
    """Return a copied worker environment with explicit library thread limits."""
    if isinstance(requested_threads, bool) or not isinstance(requested_threads, int):
        raise TypeError("requested_threads must be an integer")
    if requested_threads <= 0:
        raise ValueError("requested_threads must be positive")
    environment = os.environ.copy()
    value = str(requested_threads)
    for name in THREAD_ENVIRONMENT_VARIABLES:
        environment[name] = value
    return environment


def _positive_number(value: object, name: str) -> float:
    """Return one finite positive non-boolean number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    checked = float(value)
    if not math.isfinite(checked) or checked <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return checked


def _positive_integer(value: object, name: str) -> int:
    """Return one positive non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    """Terminate and reap one isolated worker process group."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=_TERMINATION_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _captured_text(stream: Any, name: str) -> str:
    """Read bounded captured worker output from a temporary file."""
    stream.seek(0)
    content = stream.read(MAX_CAPTURED_OUTPUT_BYTES + 1)
    if len(content) > MAX_CAPTURED_OUTPUT_BYTES:
        raise PerformanceWorkerError(
            "output_limit",
            f"performance worker {name} exceeded "
            f"{MAX_CAPTURED_OUTPUT_BYTES} captured bytes",
        )
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PerformanceWorkerError(
            "invalid_output", f"performance worker {name} was not UTF-8"
        ) from exc


def _worker_failure_kind(detail: str, returncode: int) -> str:
    """Classify one worker failure without depending on exception types."""
    lowered = detail.lower()
    if "gmres did not converge" in lowered or "max_krylov_iterations" in lowered:
        return "krylov_failure"
    if "preconditioner" in lowered:
        return "preconditioner_failure"
    if any(
        phrase in lowered
        for phrase in (
            "relative residual",
            "negative flux",
            "outer iterations",
            "scalar balance",
            "non-finite",
        )
    ):
        return "numerical_failure"
    if returncode < 0 or "memoryerror" in lowered or "cannot allocate" in lowered:
        return "resource_limit"
    return "worker_error"


def _launch_worker(
    groups: int,
    axial_layers: int,
    *,
    requested_threads: int,
    kind: str = "measurement",
    repetition: int | None = None,
    timeout_seconds: int | float = DEFAULT_TIMEOUT_SECONDS,
    address_space_limit_bytes: int = DEFAULT_ADDRESS_SPACE_LIMIT_BYTES,
    solver_case: str,
    iteration_id: str,
    shift_inverse_keff: float | None,
) -> dict[str, Any]:
    """Run one resource-bounded outcome in a new Python interpreter."""
    if kind not in {"measurement", "profile"}:
        raise ValueError("kind must be 'measurement' or 'profile'")
    command = [
        sys.executable,
        "-m",
        _WORKER_MODULE,
        "--groups",
        str(groups),
        "--axial-layers",
        str(axial_layers),
        "--requested-threads",
        str(requested_threads),
        "--kind",
        kind,
        "--address-space-limit-bytes",
        str(address_space_limit_bytes),
        "--output-file-limit-bytes",
        str(MAX_CAPTURED_OUTPUT_BYTES),
        "--solver-case",
        solver_case,
        "--iteration",
        iteration_id,
    ]
    if repetition is not None:
        command.extend(("--repetition", str(repetition)))
    if shift_inverse_keff is not None:
        command.extend(("--shift-inverse-keff", str(shift_inverse_keff)))
    with (
        tempfile.TemporaryFile(mode="w+b") as stdout_file,
        tempfile.TemporaryFile(mode="w+b") as stderr_file,
    ):
        process = subprocess.Popen(  # pylint: disable=consider-using-with
            command,
            stdout=stdout_file,
            stderr=stderr_file,
            env=worker_environment(requested_threads),
            cwd=_REPOSITORY_ROOT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process)
            raise PerformanceWorkerError(
                "timeout", f"performance worker exceeded {timeout_seconds:g} seconds"
            ) from exc
        except BaseException:
            _terminate_process_group(process)
            raise
        stdout = _captured_text(stdout_file, "stdout")
        stderr = _captured_text(stderr_file, "stderr")
    if returncode != 0:
        detail = stderr.strip() or stdout.strip()
        raise PerformanceWorkerError(
            _worker_failure_kind(detail, returncode),
            f"performance worker failed with status {returncode}: {detail}",
        )
    try:
        outcome = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise PerformanceWorkerError(
            "invalid_output", "performance worker returned invalid JSON"
        ) from exc
    if not isinstance(outcome, dict):
        raise PerformanceWorkerError(
            "invalid_output", "performance worker did not return a JSON object"
        )
    outcome["resource_limits"] = {
        "timeout_seconds": timeout_seconds,
        "address_space_bytes": address_space_limit_bytes,
        "captured_output_bytes_per_stream": MAX_CAPTURED_OUTPUT_BYTES,
    }
    return outcome


def outcome_identifier(
    groups: int,
    axial_layers: int,
    case_id: str,
    iteration: dict[str, object],
    *,
    requested_threads: int,
    kind: str,
    repetition: int | None,
) -> str:
    """Return one deterministic performance-outcome identifier."""
    policy_id = iteration_identifier(iteration)
    suffix = "none" if repetition is None else str(repetition)
    return (
        f"g{groups}-z{axial_layers}-{case_id}-{policy_id}-t{requested_threads}-"
        f"{kind}-r{suffix}"
    )


def outcome_fields(
    groups: int,
    axial_layers: int,
    case_id: str,
    *,
    iteration: dict[str, object],
    requested_threads: int,
    kind: str,
    repetition: int | None,
) -> dict[str, object]:
    """Return the shared identity fields for one terminal outcome."""
    return {
        "outcome_id": outcome_identifier(
            groups,
            axial_layers,
            case_id,
            iteration,
            requested_threads=requested_threads,
            kind=kind,
            repetition=repetition,
        ),
        "case_id": case_id,
        "eigenvalue_iteration": iteration,
        "workload_id": f"g{groups}-z{axial_layers}",
        "kind": kind,
        "repetition": repetition,
        "requested_threads": requested_threads,
    }


def launch_outcome(
    groups: int,
    axial_layers: int,
    *,
    case_id: str,
    iteration_id: str = "power",
    shift_inverse_keff: float | None = None,
    requested_threads: int,
    kind: str,
    repetition: int | None,
    timeout_seconds: int | float = DEFAULT_TIMEOUT_SECONDS,
    address_space_limit_bytes: int = DEFAULT_ADDRESS_SPACE_LIMIT_BYTES,
) -> dict[str, Any]:
    """Run one case and convert bounded worker failures into outcome records."""
    iteration = iteration_record(iteration_id, shift_inverse_keff)
    checked_timeout = _positive_number(timeout_seconds, "timeout_seconds")
    checked_memory = _positive_integer(
        address_space_limit_bytes, "address_space_limit_bytes"
    )
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        return _launch_worker(
            groups,
            axial_layers,
            requested_threads=requested_threads,
            kind=kind,
            repetition=repetition,
            timeout_seconds=checked_timeout,
            address_space_limit_bytes=checked_memory,
            solver_case=case_id,
            iteration_id=iteration_id,
            shift_inverse_keff=shift_inverse_keff,
        )
    except PerformanceWorkerError as exc:
        return {
            **outcome_fields(
                groups,
                axial_layers,
                case_id,
                iteration=iteration,
                requested_threads=requested_threads,
                kind=kind,
                repetition=repetition,
            ),
            "started_at_utc": started_at,
            "status": "failed",
            "repository": None,
            "environment": None,
            "resource_limits": {
                "timeout_seconds": checked_timeout,
                "address_space_bytes": checked_memory,
                "captured_output_bytes_per_stream": MAX_CAPTURED_OUTPUT_BYTES,
            },
            "configuration": None,
            "solve": None,
            "profile": None,
            "failure": {"kind": exc.kind, "message": str(exc)[-4096:]},
        }
