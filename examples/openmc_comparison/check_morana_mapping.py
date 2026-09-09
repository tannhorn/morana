"""Visually check the SRE-derived mini-core mapping to Morana planar cell IDs.

Run from the repository root:

``python examples/openmc_comparison/check_morana_mapping.py``

The check uses a deliberately nonuniform diagnostic value. Every plotted cell
labels its source ``(q, r)`` coordinate, Morana planar ID, and diagnostic
value, so a rotation, reflection, or ring-order mismatch is apparent without
running a transport or diffusion calculation.
"""

# Matplotlib configuration must precede its imports.
# pylint: disable=wrong-import-position

from __future__ import annotations

import argparse
from math import isclose
from pathlib import Path

from artifact_paths import ARTIFACT_ROOT, configure_matplotlib_cache
from geometry import (
    NUM_CORE_RINGS,
    lattice_pitch_cm,
    mini_core_coordinates,
)

DEFAULT_OUTPUT = ARTIFACT_ROOT / "morana_coordinate_mapping.png"
configure_matplotlib_cache()

import matplotlib  # pylint: disable=wrong-import-position

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
from matplotlib.collections import (
    PatchCollection,
)  # pylint: disable=wrong-import-position
from matplotlib.patches import Polygon  # pylint: disable=wrong-import-position
from morana import HexPlanarMesh  # pylint: disable=wrong-import-position

_TOLERANCE = 1.0e-12


def diagnostic_value(q_coord: int, r_coord: int) -> int:
    """Return a visibly nonuniform value for one source lattice coordinate."""

    return 100 * q_coord + r_coord


def mapped_planar_ids(mesh: HexPlanarMesh) -> dict[tuple[int, int], int]:
    """Map every SRE-derived coordinate to its Morana planar ID and verify centres."""

    mapping: dict[tuple[int, int], int] = {}
    for q_coord, r_coord, source_x, source_y in mini_core_coordinates():
        # The shared point-up axial bases use q -> x and r -> u directly.
        planar_id = mesh.index_of[(q_coord, r_coord)]
        morana_x, morana_y = mesh.cartesian_center(planar_id)
        if not (
            isclose(morana_x, source_x, rel_tol=0.0, abs_tol=_TOLERANCE)
            and isclose(morana_y, source_y, rel_tol=0.0, abs_tol=_TOLERANCE)
        ):
            raise ValueError(
                "Morana coordinate mapping does not preserve the shared "
                f"centre for ({q_coord}, {r_coord})."
            )
        mapping[(q_coord, r_coord)] = planar_id
    if len(mapping) != mesh.n_cells or len(set(mapping.values())) != mesh.n_cells:
        raise ValueError(
            "SRE-derived coordinates do not map one-to-one to Morana cells."
        )
    return mapping


def _arguments() -> argparse.Namespace:
    """Parse command-line options for the generated diagnostic figure."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for the diagnostic PNG.",
    )
    return parser.parse_args()


def _write_figure(
    mesh: HexPlanarMesh,
    mapping: dict[tuple[int, int], int],
    output_path: Path,
) -> None:
    """Write a labelled nonuniform Morana planar-layout diagnostic."""

    figure, axes = plt.subplots(figsize=(10.0, 8.6))
    planar_ids = tuple(mapping.values())
    values_by_id = {
        planar_id: diagnostic_value(q_coord, r_coord)
        for (q_coord, r_coord), planar_id in mapping.items()
    }
    patches = [
        Polygon(mesh.cell_vertices(planar_id), closed=True) for planar_id in planar_ids
    ]
    collection = PatchCollection(
        patches,
        array=[values_by_id[planar_id] for planar_id in planar_ids],
        cmap="viridis",
        edgecolor="#243036",
        linewidth=0.8,
    )
    axes.add_collection(collection)
    for (q_coord, r_coord), planar_id in mapping.items():
        x_coord, y_coord = mesh.cartesian_center(planar_id)
        axes.text(
            x_coord,
            y_coord,
            (
                f"({q_coord:+d}, {r_coord:+d})\n"
                f"ID {planar_id}\n{values_by_id[planar_id]:+d}"
            ),
            ha="center",
            va="center",
            color="white",
            fontsize=6.5,
        )
    figure.colorbar(collection, ax=axes, label="Diagnostic value: 100q + r")
    axes.set_aspect("equal", adjustable="box")
    axes.autoscale_view()
    axes.margins(0.08)
    axes.set_xlabel("x [cm]")
    axes.set_ylabel("y [cm]")
    axes.set_title("SRE-derived (q, r) to Morana (x, u) mapping: q -> x, r -> u")
    axes.grid(True, color="#e6e6e6", linewidth=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Verify and render the SRE-derived-to-Morana planar coordinate mapping."""

    arguments = _arguments()
    output_path = arguments.output.expanduser().resolve()
    mesh = HexPlanarMesh(NUM_CORE_RINGS, pitch=lattice_pitch_cm())
    mapping = mapped_planar_ids(mesh)
    _write_figure(mesh, mapping, output_path)
    print("Verified q -> x and r -> u for all 61 SRE-derived mini-core cells.")
    print(f"Wrote nonuniform mapping diagnostic: {output_path}")


if __name__ == "__main__":
    main()
