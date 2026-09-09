"""Export and render the SRE-derived comparison geometry using OpenMC.

Run from the repository root with an active Python environment that provides
OpenMC:

``python examples/openmc_comparison/render_geometry.py``

The script does not run particle transport.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import openmc

from artifact_paths import ARTIFACT_ROOT
from case import (
    build_mini_core_model,
    build_unit_cell_model,
)
from geometry import NUM_CORE_RINGS, lattice_pitch_cm
from plotting import MATERIAL_COLORS_RGB

DEFAULT_OUTPUT = ARTIFACT_ROOT / "geometry"


def _add_openmc_plot(
    model: openmc.Model,
    *,
    filename: str,
    width: float,
    pixels: int,
) -> None:
    """Attach a direct OpenMC material plot to a geometry model."""

    materials = {material.name: material for material in model.materials}
    plot = openmc.Plot(name=filename)
    plot.filename = filename
    plot.color_by = "material"
    plot.origin = (0.0, 0.0, 0.0)
    plot.width = (width, width)
    plot.pixels = (pixels, pixels)
    plot.background = (255, 255, 255)
    plot.show_overlaps = True
    plot.colors = {
        materials[name]: color for name, color in MATERIAL_COLORS_RGB.items()
    }
    model.plots = openmc.Plots([plot])


def _export_and_render(
    model: openmc.Model,
    *,
    model_directory: Path,
    output_directory: Path,
) -> None:
    """Write one model's XML and generate its OpenMC plot in the output root."""

    model.export_to_xml(model_directory)
    openmc.plot_geometry(
        output=True,
        cwd=output_directory,
        path_input=model_directory,
    )


def main() -> None:
    """Export geometry XML and render OpenMC's direct material plots."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_directory = args.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    unit_model = build_unit_cell_model()
    mini_core_model = build_mini_core_model()
    pitch = lattice_pitch_cm()
    _add_openmc_plot(
        unit_model,
        filename="unit_cell",
        width=1.3 * pitch,
        pixels=1400,
    )
    _add_openmc_plot(
        mini_core_model,
        filename="mini_core",
        width=2 * NUM_CORE_RINGS * pitch,  # one extra for margin
        pixels=1800,
    )

    _export_and_render(
        unit_model,
        model_directory=output_directory / "unit_cell",
        output_directory=output_directory,
    )
    _export_and_render(
        mini_core_model,
        model_directory=output_directory / "mini_core",
        output_directory=output_directory,
    )

    print(f"Lattice pitch [cm]: {pitch:.6f}")
    print("Unit-cell XML:", output_directory / "unit_cell")
    print("61-cell mini-core XML:", output_directory / "mini_core")
    print("OpenMC unit-cell plot:", output_directory / "unit_cell.png")
    print("OpenMC mini-core plot:", output_directory / "mini_core.png")
    print(
        "Fuel-can prisms use OpenMC orientation='y'; 61 translated universe "
        "fills share explicit complete-lattice-cell CSG edges."
    )


if __name__ == "__main__":
    main()
