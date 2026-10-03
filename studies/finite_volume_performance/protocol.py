"""Shared default protocol for maintained finite-volume study runs."""

from __future__ import annotations

from dataclasses import dataclass
import math

from studies.finite_volume_performance.cases import CASE_IDS, DIRECT_CASE_ID


@dataclass(frozen=True)
class StudyProtocol:
    """Own the single set of maintained execution defaults."""

    timeout_seconds: float
    address_space_limit_bytes: int
    requested_threads: int
    warmup: tuple[int, int]
    direct_repetitions: int
    iterative_repetitions: int
    profile: bool

    def __post_init__(self) -> None:
        """Reject invalid protocol controls at construction."""
        if isinstance(self.timeout_seconds, bool) or not isinstance(
            self.timeout_seconds, (int, float)
        ):
            raise TypeError("timeout_seconds must be a number")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be finite and positive")
        for name in (
            "address_space_limit_bytes",
            "requested_threads",
            "direct_repetitions",
            "iterative_repetitions",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            not isinstance(self.warmup, tuple)
            or len(self.warmup) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in self.warmup
            )
        ):
            raise TypeError("warmup must contain two integers")
        if any(value <= 0 for value in self.warmup):
            raise ValueError("warmup values must be positive")
        if not isinstance(self.profile, bool):
            raise TypeError("profile must be a boolean")

    def repetitions(self, case_id: str) -> int:
        """Return the default repetition count for a checked solver case."""
        if case_id not in CASE_IDS:
            raise ValueError(f"case_id must be one of {CASE_IDS}")
        if case_id == DIRECT_CASE_ID:
            return self.direct_repetitions
        return self.iterative_repetitions


DEFAULT_STUDY_PROTOCOL = StudyProtocol(
    timeout_seconds=300.0,
    address_space_limit_bytes=16 * 1024**3,
    requested_threads=1,
    warmup=(6, 2),
    direct_repetitions=1,
    iterative_repetitions=3,
    profile=True,
)
