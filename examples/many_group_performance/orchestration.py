"""Fresh-process orchestration for many-group performance observations."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from typing import Any

THREAD_ENVIRONMENT_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_REPOSITORY_ROOT = Path(__file__).parents[2]
_WORKER_MODULE = "examples.many_group_performance.worker"
DEFAULT_TIMEOUT_SECONDS = 2 * 60 * 60
DEFAULT_ADDRESS_SPACE_LIMIT_BYTES = 16 * 1024**3
MAX_CAPTURED_OUTPUT_BYTES = 64 * 1024**2
_TERMINATION_GRACE_SECONDS = 5.0


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
        raise RuntimeError(
            f"performance worker {name} exceeded "
            f"{MAX_CAPTURED_OUTPUT_BYTES} captured bytes"
        )
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"performance worker {name} was not UTF-8") from exc


def launch_observation(
    groups: int,
    axial_layers: int,
    *,
    requested_threads: int,
    kind: str = "measurement",
    repetition: int | None = None,
    timeout_seconds: int | float = DEFAULT_TIMEOUT_SECONDS,
    address_space_limit_bytes: int = DEFAULT_ADDRESS_SPACE_LIMIT_BYTES,
) -> dict[str, Any]:
    """Run one resource-bounded observation in a new Python interpreter."""
    if kind not in {"measurement", "profile"}:
        raise ValueError("kind must be 'measurement' or 'profile'")
    checked_timeout = _positive_number(timeout_seconds, "timeout_seconds")
    checked_memory = _positive_integer(
        address_space_limit_bytes, "address_space_limit_bytes"
    )
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
        str(checked_memory),
        "--output-file-limit-bytes",
        str(MAX_CAPTURED_OUTPUT_BYTES),
    ]
    if repetition is not None:
        command.extend(("--repetition", str(repetition)))
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
            returncode = process.wait(timeout=checked_timeout)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process)
            raise RuntimeError(
                f"performance worker exceeded {checked_timeout:g} seconds"
            ) from exc
        except BaseException:
            _terminate_process_group(process)
            raise
        stdout = _captured_text(stdout_file, "stdout")
        stderr = _captured_text(stderr_file, "stderr")
    if returncode != 0:
        detail = stderr.strip() or stdout.strip()
        raise RuntimeError(f"performance worker failed: {detail}")
    try:
        observation = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("performance worker returned invalid JSON") from exc
    if not isinstance(observation, dict):
        raise RuntimeError("performance worker did not return a JSON object")
    observation["resource_limits"] = {
        "timeout_seconds": checked_timeout,
        "address_space_bytes": checked_memory,
        "captured_output_bytes_per_stream": MAX_CAPTURED_OUTPUT_BYTES,
    }
    return observation
