"""Shared artifact and cache paths for the comparison helpers."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import gettempdir

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "examples" / "openmc_comparison"


def configure_matplotlib_cache() -> None:
    """Select a writable shared Matplotlib configuration directory."""

    config_directory = Path(gettempdir()) / "morana_examples" / "matplotlib_config"
    config_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config_directory))
