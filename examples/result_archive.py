"""Save and reload a completed result without pickle serialization."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from examples.multigroup_fixed_source import build_configuration
from morana import Result
from morana.solvers.finite_volume import solve_fixed_source

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_DIR = REPO_ROOT / "artifacts" / "examples" / "result_archive"


def parse_args() -> argparse.Namespace:
    """Parse the archive output-directory option."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ARCHIVE_DIR,
        help="Directory for the non-pickle result archive.",
    )
    return parser.parse_args()


def main() -> None:
    """Solve, archive, reload, and verify a completed fixed-source result."""
    output_dir = parse_args().output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "multigroup_fixed_source.morana-result"

    result = solve_fixed_source(build_configuration())
    result.save_to_disk(archive_path)
    loaded = Result.load_from_disk(archive_path)

    for axial_index in range(result.n_axial_layers):
        np.testing.assert_allclose(
            loaded.flux_layer(axial_index), result.flux_layer(axial_index)
        )
    assert loaded.configuration_snapshot is not None
    print(f"Saved result archive: {archive_path}")
    print(f"Loaded configuration: {loaded.configuration_snapshot.name}")
    print(
        f"Loaded flux range: {loaded.flux_layer(0).min():.6f} to "
        f"{loaded.flux_layer(0).max():.6f}"
    )


if __name__ == "__main__":
    main()
