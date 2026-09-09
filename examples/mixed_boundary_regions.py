"""Solve a small mixed-boundary domain with several boundary conditions.

The active domain is a three-ring region with a central instrument
channel, an offset beam port, and one omitted reflector block. These excluded
regions demonstrate key, kind, and directional selection with reflective,
vacuum, Robin, partial-current-return, and incoming-current physics.
"""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
from tempfile import gettempdir

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "examples" / "mixed_boundary_regions"
DEFAULT_MATPLOTLIB_CONFIG_DIR = (
    Path(gettempdir()) / "morana_examples" / "matplotlib_config"
)
DEFAULT_MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MATPLOTLIB_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

from morana.solvers.finite_volume import solve_fixed_source
from morana import (  # pylint: disable=wrong-import-position
    BoundaryCondition,
    BoundaryConditionSet,
    CrossSections,
    ExcludedRegion,
    Material,
    MaterialMesh,
    MaterialSlice,
    HexPlanarMesh,
    ProblemConfiguration,
    UniformSource,
)


def parse_args() -> argparse.Namespace:
    """Parse example command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated result PNG, HTML, and VTM files.",
    )
    return parser.parse_args()


def build_configuration() -> ProblemConfiguration:
    """Build a legible asymmetric mixed-boundary domain."""
    mesh = HexPlanarMesh(num_rings=3, pitch=10.0)
    medium = Material(
        "medium",
        xs=CrossSections(
            D=[1.2],
            sigma_a=[0.02],
            sigma_s=[[0.0]],
            fission=None,
        ),
    )
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(
                mesh,
                [
                    ["beam_port"] + ["medium"] * 11,
                    [
                        "medium",
                        "medium",
                        "medium",
                        "reflector_block",
                        "medium",
                        "medium",
                    ],
                    ["instrument_channel"],
                ],
                height=1.0,
            ),
        ),
        excluded_regions={
            "instrument_channel": ExcludedRegion(
                kind="channel",
                color="#d9d9d9",
            ),
            "beam_port": ExcludedRegion(
                kind="source_port",
                color="#f4c27a",
            ),
            "reflector_block": ExcludedRegion(
                kind="reflector",
                color="#c7d4e8",
            ),
        },
    )
    return ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=material_mesh,
        boundary=BoundaryConditionSet(
            BoundaryCondition.reflective().globally(),
            BoundaryCondition.vacuum().on_radial(),
            BoundaryCondition.reflective().on_excluded(kind="reflector"),
            BoundaryCondition.partial_current_return(0.5).on_excluded(kind="channel"),
            BoundaryCondition.robin(0.15).on_excluded(kind="channel", direction="u+"),
            BoundaryCondition.incoming_current([2.0]).on_excluded(key="beam_port"),
        ),
        source=UniformSource([1.0]),
        name="mixed_boundary_domain",
    )


def resolved_boundary_counts(
    configuration: ProblemConfiguration,
) -> Counter[str]:
    """Count the condition kinds actually resolved on exposed faces."""
    counts: Counter[str] = Counter()
    material_mesh = configuration.material_mesh
    active_cells = configuration.material_mesh.n_active_cells(0)
    for active_id in range(active_cells):
        for direction in ("x+", "x-", "u+", "u-", "v+", "v-"):
            face = material_mesh.face(0, active_id, direction)
            if face.kind != "internal":
                counts[configuration.boundary.resolve(face).kind] += 1
    return counts


def main() -> None:
    """Solve the mixed-boundary problem and write inspection artifacts."""
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configuration = build_configuration()
    result = solve_fixed_source(configuration)

    material_png_path = output_dir / "materials.png"
    flux_png_path = output_dir / "flux.png"
    html_path = output_dir / "flux.html"
    vtm_path = output_dir / "result.vtm"

    axes = configuration.material_mesh.plot_matplotlib(
        axial_index=0,
        materials=configuration.materials,
    )
    axes.figure.savefig(material_png_path, dpi=200, bbox_inches="tight")
    plt.close(axes.figure)

    axes = result.plot_matplotlib(group=0, axial_index=0)
    axes.figure.savefig(flux_png_path, dpi=200, bbox_inches="tight")
    plt.close(axes.figure)
    result.plot_plotly(group=0, axial_index=0).write_html(html_path)
    result.export_vtm(vtm_path)

    print(f"Configuration: {configuration.name}")
    print("Boundary assignments:")
    print("  radial exterior: vacuum")
    print("  omitted reflector block: reflective")
    print("  instrument channel: partial-current return beta=0.5")
    print("  instrument channel u+ face: Robin alpha=0.15 override")
    print("  beam port: incoming partial current=2.0")
    print("Resolved exposed faces by condition:")
    for kind, count in sorted(resolved_boundary_counts(configuration).items()):
        print(f"  {kind}: {count}")
    print("Active solver cells: " f"{configuration.material_mesh.n_active_cells(0)}")
    flux = result.flux_layer(0)[0]
    print(f"Flux by active ID: {flux}")
    print(f"Flux range: {flux.min():.6f} to {flux.max():.6f}")
    print("Neutron balance:")
    for name, value in result.balance.items():
        print(f"  {name}: {value:.6e}")
    print("Generated files:")
    print(f"  Material PNG:   {material_png_path}")
    print(f"  Flux PNG:       {flux_png_path}")
    print(f"  Plotly HTML:    {html_path}")
    print(f"  Result VTM:     {vtm_path}")


if __name__ == "__main__":
    main()
