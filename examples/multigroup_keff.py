"""Compare equivalent fission representations in a layered criticality solve.

The example uses fast/thermal scattering, heterogeneous fuel and reflector
regions, radial and top vacuum boundaries, and recoverable-power
normalization. It solves identical physics with compact separable fission data
and its equivalent event-oriented fission-transfer matrix, then verifies that
the assembled fission matrices and criticality results agree. Cross sections
are illustrative, not design data.
"""

# Matplotlib configuration must precede its imports.
# pylint: disable=wrong-import-position,ungrouped-imports

from __future__ import annotations

import argparse
import os
from pathlib import Path
from tempfile import gettempdir

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "examples" / "multigroup_keff"
DEFAULT_MATPLOTLIB_CONFIG_DIR = (
    Path(gettempdir()) / "morana_examples" / "matplotlib_config"
)
DEFAULT_MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MATPLOTLIB_CONFIG_DIR))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

from morana.solvers.finite_volume import solve_keff
from morana import (  # pylint: disable=wrong-import-position
    BoundaryCondition,
    BoundaryConditionSet,
    CrossSections,
    FissionData,
    FissionTransfer,
    SeparableFission,
    Material,
    MaterialMesh,
    MaterialSlice,
    HexPlanarMesh,
    ProblemConfiguration,
    KeffSettings,
    PowerNormalization,
)
from morana.operators import assemble_fission_matrix, extract_cross_section_data

FUEL_NU_SIGMA_F = (0.007, 0.110)
FUEL_CHI = (0.995, 0.005)


def parse_args() -> argparse.Namespace:
    """Parse the output-directory option."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_configuration(*, representation: str) -> ProblemConfiguration:
    """Build a fuel island using one equivalent fission representation."""
    if representation == "separable":
        neutron_production = SeparableFission(nu_sigma_f=FUEL_NU_SIGMA_F, chi=FUEL_CHI)
    elif representation == "transfer":
        neutron_production = FissionTransfer(
            fission_transfer=[
                [nu_sigma_f * chi for chi in FUEL_CHI] for nu_sigma_f in FUEL_NU_SIGMA_F
            ]
        )
    else:
        raise ValueError(f"unsupported fission representation: {representation!r}")

    mesh = HexPlanarMesh(num_rings=3, pitch=12.0)
    fuel = Material(
        "fuel",
        xs=CrossSections(
            D=[1.30, 0.34],
            sigma_a=[0.009, 0.065],
            sigma_s=[[0.016, 0.075], [0.003, 0.190]],
            fission=FissionData(
                neutron_production=neutron_production,
                kappa_sigma_f=[5.6e5, 8.8e6],
            ),
        ),
        color="#d62728",
    )
    reflector = Material(
        "reflector",
        xs=CrossSections(
            D=[1.55, 0.42],
            sigma_a=[0.002, 0.016],
            sigma_s=[[0.090, 0.052], [0.002, 0.310]],
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
        name="two_group_fuel_island_with_reflector",
    )


def check_equivalent_results(separable, transfer) -> None:
    """Require equivalent fission inputs to retain one physical response."""
    np.testing.assert_allclose(separable.keff, transfer.keff, rtol=1.0e-11)
    for axial_index in range(separable.n_axial_layers):
        np.testing.assert_allclose(
            separable.flux_layer(axial_index),
            transfer.flux_layer(axial_index),
            rtol=1.0e-10,
            atol=1.0e-12,
        )
    for name, values in separable.balance.by_group.items():
        np.testing.assert_allclose(
            values, transfer.balance.by_group[name], rtol=1.0e-10, atol=1.0e-12
        )


def main() -> None:
    """Solve, normalize, report group balances, and write output artifacts."""
    output_dir = parse_args().output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = KeffSettings(
        keff_change_tolerance=1.0e-10,
        flux_change_tolerance=1.0e-10,
        keff_relative_residual_tolerance=1.0e-10,
    )
    separable_configuration = build_configuration(representation="separable")
    transfer_configuration = build_configuration(representation="transfer")
    separable_data = extract_cross_section_data(separable_configuration)
    transfer_data = extract_cross_section_data(transfer_configuration)
    np.testing.assert_allclose(
        assemble_fission_matrix(separable_configuration, separable_data).toarray(),
        assemble_fission_matrix(transfer_configuration, transfer_data).toarray(),
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    result = solve_keff(
        separable_configuration, PowerNormalization(power=1.0e5), settings
    )
    transfer_result = solve_keff(
        transfer_configuration, PowerNormalization(power=1.0e5), settings
    )
    check_equivalent_results(result, transfer_result)

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

    print(f"Configuration: {result.configuration_snapshot.name}")
    print(
        "Model: variable-height axially heterogeneous fuel/reflector "
        "criticality, radial/top vacuum"
    )
    print(
        f"k_eff: {result.keff:.8f} after "
        f"{result.execution_report.iterations} iterations"
    )
    print(f"Recoverable-power target [W]: {result.normalization.power:.6e}")
    print(
        "Resulting fission-neutron source [n / s]: "
        f"{result.balance['fission_production']:.6e}"
    )
    for axial_index in range(result.n_axial_layers):
        values = result.flux_layer(axial_index)
        print(f"Axial layer {axial_index}: {values.shape[1]} active cells")
    print("Group balance [n / s]:")
    for name, values in result.balance.by_group.items():
        print(f"  {name}: {values}")
    print("Layer axial leakage [n / s]:")
    for axial_index, values in enumerate(
        result.balance.by_layer_group["axial_leakage"]
    ):
        print(f"  layer {axial_index}: {values}")
    print(
        "Relative k_eff residual: "
        f"{result.execution_report.final_outer_iteration.keff_relative_residual:.3e}"
    )
    print(
        "Equivalent separable and transfer fission representations agree in "
        "the assembled fission matrix, k_eff, flux, and group balances."
    )
    print(f"Generated group PNG/HTML and VTM artifacts in: {output_dir}")


if __name__ == "__main__":
    main()
