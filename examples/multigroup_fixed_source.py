"""Compare unit and multiplying scatter in a layered two-group fixed-source case.

An external fast-group source drives downscatter, thermal fission production,
and fission-spectrum emission in a compact fuel island surrounded by a leaky
reflector. The matched cases use compact unit multiplicity and explicit
multiplying scatter. Top is also vacuum. Cross sections are illustrative, not
design data.
"""

# Matplotlib configuration must precede its import.
# pylint: disable=wrong-import-position

from __future__ import annotations

import argparse
import os
from pathlib import Path
from tempfile import gettempdir

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "examples" / "multigroup_fixed_source"
DEFAULT_MATPLOTLIB_CONFIG_DIR = (
    Path(gettempdir()) / "morana_examples" / "matplotlib_config"
)
DEFAULT_MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MATPLOTLIB_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from morana.solvers.finite_volume import solve_fixed_source
from morana import (
    FissionData,
    SeparableFission,
    BoundaryCondition,
    BoundaryConditionSet,
    CrossSections,
    Material,
    MaterialMesh,
    MaterialSlice,
    MaterialSource,
    HexPlanarMesh,
    ProblemConfiguration,
)


def parse_args() -> argparse.Namespace:
    """Parse the output-directory option."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_configuration(
    multiplicity_matrix: list[list[float]] | None = None,
    *,
    name: str = "two_group_externally_driven_fuel_island",
) -> ProblemConfiguration:
    """Build an axially heterogeneous fuel island with an external fast source."""
    mesh = HexPlanarMesh(num_rings=3, pitch=12.0)
    fuel = Material(
        "fuel",
        xs=CrossSections(
            D=[1.35, 0.36],
            sigma_a=[0.010, 0.075],
            sigma_s=[[0.018, 0.070], [0.003, 0.180]],
            multiplicity_matrix=multiplicity_matrix,
            fission=FissionData(
                neutron_production=SeparableFission(
                    nu_sigma_f=[0.006, 0.095], chi=[0.995, 0.005]
                )
            ),
        ),
        color="#d62728",
    )
    reflector = Material(
        "reflector",
        xs=CrossSections(
            D=[1.55, 0.42],
            sigma_a=[0.002, 0.018],
            sigma_s=[[0.090, 0.050], [0.002, 0.300]],
            multiplicity_matrix=multiplicity_matrix,
            fission=None,
        ),
        color="#4c78a8",
    )
    return ProblemConfiguration(
        mesh=mesh,
        materials={"fuel": fuel, "reflector": reflector},
        material_mesh=MaterialMesh.stack(
            (
                MaterialSlice.from_openmc_rings(
                    mesh,
                    [["reflector"] * 12, ["fuel"] * 6, ["fuel"]],
                    height=1.0,
                ),
                MaterialSlice.from_openmc_rings(
                    mesh,
                    [["reflector"] * 12, ["fuel"] * 6, ["reflector"]],
                    height=1.5,
                ),
            )
        ),
        boundary=BoundaryConditionSet(
            BoundaryCondition.reflective().globally(),
            BoundaryCondition.vacuum().on_radial(),
            BoundaryCondition.vacuum().on_top(),
        ),
        source=MaterialSource({"fuel": [2.0e5, 0.0], "reflector": [0.0, 0.0]}),
        name=name,
    )


def write_artifacts(result, output_dir: Path) -> None:
    """Write slice plots and VTM output for one completed response."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for axial_index in range(result.n_axial_layers):
        for group in range(result.groups):
            axes = result.plot_matplotlib(group=group, axial_index=axial_index)
            axes.figure.savefig(
                output_dir / f"flux_g{group}_z{axial_index}.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(axes.figure)
            result.plot_plotly(group=group, axial_index=axial_index).write_html(
                output_dir / f"flux_g{group}_z{axial_index}.html"
            )
    result.export_vtm(output_dir / "result.vtm")


def _group_flux_totals(result) -> tuple[float, ...]:
    """Return unnormalized compact-cell flux sums by energy group."""
    return tuple(
        sum(
            float(result.flux_layer(axial_index)[group].sum())
            for axial_index in range(result.n_axial_layers)
        )
        for group in range(result.groups)
    )


def report_case(case_name: str, configuration: ProblemConfiguration, result) -> None:
    """Report one case's flux range and neutron-balance interpretation."""
    print(f"Case: {case_name}")
    print(f"Configuration: {configuration.name}")
    print(f"Group flux sums: {_group_flux_totals(result)}")
    print(
        f"Scattering coupling [n / s]: {result.balance.by_group['scattering_coupling']}"
    )
    print(f"Net scattering [n / s]: {result.balance.by_group['net_scattering']}")
    print(
        "Relative linear residual: "
        f"{result.execution_report.true_relative_residual:.3e}"
    )


def main() -> None:
    """Solve matched scattering cases, report balances, and write artifacts."""
    output_dir = parse_args().output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = (
        ("unit_multiplicity", None),
        ("multiplying_scatter", [[1.05, 1.20], [1.00, 1.05]]),
    )

    print("Model: variable-height axially heterogeneous fuel island, radial/top vacuum")
    for case_name, multiplicity_matrix in cases:
        configuration = build_configuration(
            multiplicity_matrix,
            name=f"two_group_externally_driven_fuel_island_{case_name}",
        )
        result = solve_fixed_source(configuration)
        write_artifacts(result, output_dir / case_name)
        report_case(case_name, configuration, result)
    print(f"Generated per-case group PNG/HTML and VTM artifacts in: {output_dir}")


if __name__ == "__main__":
    main()
