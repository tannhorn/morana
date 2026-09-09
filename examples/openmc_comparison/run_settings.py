"""Shared validation for OpenMC eigenvalue statistical controls."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable


def active_history_label(active_histories: int) -> str:
    """Return the compact artifact label for an active-history count."""

    if not isinstance(active_histories, int) or isinstance(active_histories, bool):
        raise TypeError("active_histories must be an integer")
    if active_histories <= 0:
        raise ValueError("active_histories must be positive")
    if active_histories % 1_000_000 == 0:
        return f"{active_histories // 1_000_000}m"
    return f"{active_histories}h"


def newest_compatible_statepoint(
    output_dir: Path,
    summary_filename: str,
    is_compatible: Callable[[dict[str, object]], bool],
) -> tuple[Path | None, int | None]:
    """Return the newest statepoint whose summary is restart-compatible."""

    candidates: list[tuple[int, Path]] = []
    if not output_dir.exists():
        return None, None
    for summary_path in output_dir.glob(f"*/{summary_filename}"):
        try:
            record = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or not is_compatible(record):
            continue
        statepoint = summary_path.parent / str(record.get("statepoint", ""))
        if statepoint.is_file():
            candidates.append((int(record["batches"]), statepoint))
    if not candidates:
        return None, None
    batches, statepoint = max(candidates, key=lambda candidate: candidate[0])
    return statepoint.resolve(), batches


def newest_compatible_openmc_statepoint(
    output_dir: Path,
    summary_filename: str,
    is_compatible: Callable[[dict[str, object]], bool],
    *,
    openmc_version: str,
    cross_sections: str,
) -> tuple[Path | None, int | None]:
    """Select a restart with matching controls and OpenMC provenance."""

    def matches_record(record: dict[str, object]) -> bool:
        return (
            is_compatible(record)
            and record.get("openmc_version") == openmc_version
            and record.get("cross_sections") == cross_sections
        )

    return newest_compatible_statepoint(
        output_dir,
        summary_filename,
        matches_record,
    )


@dataclass(frozen=True)
class EigenvalueRunSettings:
    """Statistical controls common to CE and MGXS eigenvalue runs."""

    particles: int
    batches: int
    inactive: int
    seed: int

    def __post_init__(self) -> None:
        """Validate the common OpenMC controls."""

        for name in ("particles", "batches", "inactive", "seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if self.particles <= 0:
            raise ValueError("particles must be positive")
        if self.inactive < 0:
            raise ValueError("inactive must be non-negative")
        if self.batches <= self.inactive:
            raise ValueError("batches must exceed inactive")
        if self.seed <= 0:
            raise ValueError("seed must be positive")

    @property
    def artifact_label(self) -> str:
        """Return the calculation-specific artifact-directory label."""

        raise NotImplementedError

    def artifact_directory(self, output_dir: Path) -> Path:
        """Return this run's directory below an artifact base directory."""

        if not isinstance(output_dir, Path):
            raise TypeError("output_dir must be a pathlib.Path")
        return output_dir.expanduser().resolve() / self.artifact_label

    def statepoint_path(self, output_dir: Path) -> Path:
        """Return this run's final OpenMC statepoint path."""

        return self.artifact_directory(output_dir) / f"statepoint.{self.batches}.h5"
