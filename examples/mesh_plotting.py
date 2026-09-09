"""HexPlanarMesh and MaterialMesh plotting/export example.

The example writes generated files outside the repository by default so the
artifacts are easy to inspect without polluting the working tree.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from tempfile import gettempdir

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "examples" / "mesh_plotting"
DEFAULT_MATPLOTLIB_CONFIG_DIR = (
    Path(gettempdir()) / "morana_examples" / "matplotlib_config"
)
DEFAULT_MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MATPLOTLIB_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

from morana import (
    HexPlanarMesh,
    Material,
    MaterialMesh,
    MaterialSlice,
)  # pylint: disable=wrong-import-position


def parse_args() -> argparse.Namespace:
    """Parse example command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated PNG, HTML, and VTK files.",
    )
    return parser.parse_args()


def main() -> None:
    """Create Matplotlib, Plotly, and VTK outputs for small meshes."""
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = HexPlanarMesh(num_rings=3, pitch=10.0)
    png_path = output_dir / "hex_planar_mesh.png"
    html_path = output_dir / "hex_planar_mesh.html"
    vtu_path = output_dir / "hex_planar_mesh.vtu"

    axes = mesh.plot_matplotlib()
    axes.figure.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(axes.figure)

    figure = mesh.to_plotly()
    figure.write_html(html_path)

    mesh.export_vtu(vtu_path)

    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(
                mesh,
                [
                    [
                        "0",
                        "reflector",
                        "reflector",
                        "0",
                        "reflector",
                        "reflector",
                        "0",
                        "reflector",
                        "reflector",
                        "0",
                        "reflector",
                        "reflector",
                    ],
                    ["medium"] * 6,
                    ["medium"],
                ],
                height=1.0,
            ),
        )
    )
    materials = {
        "medium": Material(name="medium", color="#d62728"),
        "reflector": Material(name="reflector", color="#1f77b4"),
    }
    material_png_path = output_dir / "material_mesh_slice.png"
    material_html_path = output_dir / "material_mesh_slice.html"
    material_vtm_path = output_dir / "material_mesh.vtm"

    material_axes = material_mesh.plot_matplotlib(0, materials=materials)
    material_axes.figure.savefig(material_png_path, dpi=200, bbox_inches="tight")
    plt.close(material_axes.figure)

    material_figure = material_mesh.to_plotly(0, materials=materials)
    material_figure.write_html(material_html_path)

    material_mesh.export_vtm(material_vtm_path)

    print("HexPlanarMesh plotting example")
    print(f"Full lattice positions: {mesh.n_cells}")
    print("Generated files:")
    print(f"  Matplotlib PNG: {png_path}")
    print(f"  Plotly HTML:    {html_path}")
    print(f"  VTU mesh:       {vtu_path}")
    print(f"  Material PNG:   {material_png_path}")
    print(f"  Material HTML:  {material_html_path}")
    print(f"  Material VTM:   {material_vtm_path}")


if __name__ == "__main__":
    main()
