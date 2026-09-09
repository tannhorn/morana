"""Regression test for the installation-guide quickstart program."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = REPOSITORY_ROOT / "examples" / "quickstart.py"
GETTING_STARTED_GUIDE = REPOSITORY_ROOT / "docs" / "getting_started.md"


def test_quickstart_program_runs_as_documented() -> None:
    """The quickstart program should solve its stated uniform-flux case."""
    assert '--8<-- "examples/quickstart.py"' in GETTING_STARTED_GUIDE.read_text(
        encoding="utf-8"
    )

    completed = subprocess.run(
        [sys.executable, str(QUICKSTART)],
        check=True,
        capture_output=True,
        cwd=REPOSITORY_ROOT,
        text=True,
    )

    assert completed.stdout == "[50. 50. 50. 50. 50. 50.]\n"
