# pylint: disable=wrong-import-position

"""Compare a one-dimensional axial core-reflector eigenproblem with diffusion.

The single planar hexagon has reflective radial faces, reducing the model to a
one-dimensional z problem. A fissionable core occupies the lower part of the
column, a nonfissioning reflector occupies its upper part, the bottom is
reflective, and the reflector top has a Marshak-vacuum condition. The
example compares the computed multiplication factor and cell-average flux with
the analytic two-region diffusion eigenfunction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import cos, cosh, pi, sin, sinh, sqrt, tan
import os
from pathlib import Path
from tempfile import gettempdir

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "examples" / "one_group_keff"
DEFAULT_MATPLOTLIB_CONFIG_DIR = (
    Path(gettempdir()) / "morana_examples" / "matplotlib_config"
)
DEFAULT_MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MATPLOTLIB_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np
from scipy.optimize import brentq

from morana.solvers.finite_volume import solve_keff
from morana import (  # pylint: disable=wrong-import-position
    BoundaryCondition,
    BoundaryConditionSet,
    CrossSections,
    FissionData,
    SeparableFission,
    FissionSourceNormalization,
    HexPlanarMesh,
    Material,
    MaterialMesh,
    MaterialSlice,
    ProblemConfiguration,
    KeffSettings,
)

CORE_HEIGHT = 100.0
REFLECTOR_HEIGHT = 50.0
CORE_LAYERS = 80
REFLECTOR_LAYERS = 40
CORE_DIFFUSION = 1.20
CORE_ABSORPTION = 0.010
CORE_NU_SIGMA_F = 0.012
REFLECTOR_DIFFUSION = 1.50
REFLECTOR_ABSORPTION = 0.002
FISSION_SOURCE_RATE = 1.0e15


@dataclass(frozen=True)
class AxialReference:
    """Represent the analytic core-reflector flux shape and eigenvalue."""

    axial_buckling: float
    reflector_decay: float

    @property
    def keff(self) -> float:
        """Return the analytic multiplication factor."""
        return CORE_NU_SIGMA_F / (
            CORE_ABSORPTION + CORE_DIFFUSION * self.axial_buckling**2
        )

    @property
    def reflector_boundary_factor(self) -> float:
        """Return the vacuum-compatible coefficient multiplying ``sinh``."""
        return 1.0 / (2.0 * REFLECTOR_DIFFUSION * self.reflector_decay)

    @property
    def reflector_amplitude(self) -> float:
        """Return the reflector amplitude for unit core-bottom flux."""
        return cos(self.axial_buckling * CORE_HEIGHT) / self._reflector_shape(
            REFLECTOR_HEIGHT
        )

    def _reflector_shape(self, distance_from_top: float) -> float:
        """Return the reflector shape satisfying its top vacuum condition."""
        decay_distance = self.reflector_decay * distance_from_top
        return cosh(decay_distance) + self.reflector_boundary_factor * sinh(
            decay_distance
        )

    def flux(self, z: float | np.ndarray) -> float | np.ndarray:
        """Return the unit-amplitude analytic scalar-flux shape at ``z``."""
        values = np.asarray(z, dtype=float)
        core_flux = np.cos(self.axial_buckling * values)
        reflector_flux = self.reflector_amplitude * (
            np.cosh(self.reflector_decay * (CORE_HEIGHT + REFLECTOR_HEIGHT - values))
            + self.reflector_boundary_factor
            * np.sinh(self.reflector_decay * (CORE_HEIGHT + REFLECTOR_HEIGHT - values))
        )
        result = np.where(values <= CORE_HEIGHT, core_flux, reflector_flux)
        return float(result) if result.ndim == 0 else result

    def integral(self, lower: float, upper: float) -> float:
        """Return the unit-amplitude flux integral between two axial points."""
        if not 0.0 <= lower <= upper <= CORE_HEIGHT + REFLECTOR_HEIGHT:
            raise ValueError("integration bounds must lie within the axial column")
        if upper <= CORE_HEIGHT:
            return self._core_integral(lower, upper)
        if lower >= CORE_HEIGHT:
            return self._reflector_integral(lower, upper)
        return self._core_integral(lower, CORE_HEIGHT) + self._reflector_integral(
            CORE_HEIGHT, upper
        )

    def _core_integral(self, lower: float, upper: float) -> float:
        """Return the exact integral through one core interval."""
        buckling = self.axial_buckling
        return (sin(buckling * upper) - sin(buckling * lower)) / buckling

    def _reflector_integral(self, lower: float, upper: float) -> float:
        """Return the exact integral through one reflector interval."""
        decay = self.reflector_decay
        factor = self.reflector_boundary_factor
        distance_lower = CORE_HEIGHT + REFLECTOR_HEIGHT - lower
        distance_upper = CORE_HEIGHT + REFLECTOR_HEIGHT - upper

        def antiderivative(distance: float) -> float:
            """Return the primitive of the vacuum-compatible reflector shape."""
            return (sinh(decay * distance) + factor * cosh(decay * distance)) / decay

        return self.reflector_amplitude * (
            antiderivative(distance_lower) - antiderivative(distance_upper)
        )


def parse_args() -> argparse.Namespace:
    """Parse artifact and optional documentation-asset output directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the axial-comparison PNG and result VTM files.",
    )
    parser.add_argument(
        "--documentation-assets-dir",
        type=Path,
        help="Optional directory for the tracked axial-comparison PNG asset.",
    )
    return parser.parse_args()


def analytic_reference() -> AxialReference:
    """Solve the analytic core-reflector interface equation for the fundamental mode."""
    reflector_decay = sqrt(REFLECTOR_ABSORPTION / REFLECTOR_DIFFUSION)
    boundary_factor = 1.0 / (2.0 * REFLECTOR_DIFFUSION * reflector_decay)
    decay_height = reflector_decay * REFLECTOR_HEIGHT
    reflector_current_to_flux = (
        REFLECTOR_DIFFUSION
        * reflector_decay
        * (sinh(decay_height) + boundary_factor * cosh(decay_height))
        / (cosh(decay_height) + boundary_factor * sinh(decay_height))
    )

    def interface_residual(axial_buckling: float) -> float:
        return (
            CORE_DIFFUSION * axial_buckling * tan(axial_buckling * CORE_HEIGHT)
            - reflector_current_to_flux
        )

    axial_buckling = brentq(
        interface_residual,
        1.0e-12,
        pi / (2.0 * CORE_HEIGHT) * (1.0 - 1.0e-12),
    )
    return AxialReference(axial_buckling, reflector_decay)


def build_configuration() -> ProblemConfiguration:
    """Build the reflected one-column core with an upper axial reflector."""
    mesh = HexPlanarMesh(num_rings=1, pitch=10.0)
    core = Material(
        "core",
        xs=CrossSections(
            D=[CORE_DIFFUSION],
            sigma_a=[CORE_ABSORPTION],
            sigma_s=[[0.0]],
            fission=FissionData(
                neutron_production=SeparableFission(
                    nu_sigma_f=[CORE_NU_SIGMA_F], chi=[1.0]
                )
            ),
        ),
        color="#d62728",
    )
    reflector = Material(
        "reflector",
        xs=CrossSections(
            D=[REFLECTOR_DIFFUSION],
            sigma_a=[REFLECTOR_ABSORPTION],
            sigma_s=[[0.0]],
            fission=None,
        ),
        color="#4c78a8",
    )
    core_slices = tuple(
        MaterialSlice.from_openmc_rings(
            mesh,
            [["core"]],
            height=CORE_HEIGHT / CORE_LAYERS,
        )
        for _ in range(CORE_LAYERS)
    )
    reflector_slices = tuple(
        MaterialSlice.from_openmc_rings(
            mesh,
            [["reflector"]],
            height=REFLECTOR_HEIGHT / REFLECTOR_LAYERS,
        )
        for _ in range(REFLECTOR_LAYERS)
    )
    return ProblemConfiguration(
        mesh=mesh,
        materials={"core": core, "reflector": reflector},
        material_mesh=MaterialMesh.stack(core_slices + reflector_slices),
        boundary=BoundaryConditionSet(
            BoundaryCondition.reflective().globally(),
            BoundaryCondition.vacuum().on_top(),
        ),
        name="one_dimensional_axial_core_reflector",
    )


def _normalized_reference_averages(
    configuration: ProblemConfiguration,
    reference: AxialReference,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return axial centres, source-normalized reference averages, and a scale."""
    scale = FISSION_SOURCE_RATE / (
        CORE_NU_SIGMA_F * configuration.mesh.area * reference.integral(0.0, CORE_HEIGHT)
    )
    centres = []
    averages = []
    for axial_index in range(configuration.material_mesh.n_axial_layers):
        lower, upper = configuration.material_mesh.z_bounds(axial_index)
        centres.append((lower + upper) / 2.0)
        averages.append(scale * reference.integral(lower, upper) / (upper - lower))
    return np.asarray(centres), np.asarray(averages), scale


def plot_comparison(
    output_path: Path,
    centres: np.ndarray,
    numerical_flux: np.ndarray,
    reference_averages: np.ndarray,
    *,
    reference: AxialReference,
    scale: float,
) -> None:
    """Write a transparent axial cell-average flux comparison figure."""
    z_values = np.linspace(0.0, CORE_HEIGHT + REFLECTOR_HEIGHT, 1_001)
    figure, axes = plt.subplots()
    figure.patch.set_alpha(0.0)
    axes.patch.set_alpha(0.0)
    axes.plot(
        z_values,
        scale * reference.flux(z_values),
        color="#1f77b4",
        label="analytic flux",
    )
    axes.plot(
        centres,
        numerical_flux,
        "o",
        color="#d62728",
        markersize=3,
        label="Morana cell average",
    )
    axes.plot(
        centres,
        reference_averages,
        "x",
        color="#2ca02c",
        markersize=3,
        label="analytic cell average",
    )
    axes.axvline(CORE_HEIGHT, color="black", linestyle="--", linewidth=1.0)
    axes.text(
        CORE_HEIGHT / 2.0,
        axes.get_ylim()[1] * 0.92,
        "fissionable core",
        ha="center",
    )
    axes.text(
        CORE_HEIGHT + REFLECTOR_HEIGHT / 2.0,
        axes.get_ylim()[1] * 0.92,
        "reflector",
        ha="center",
    )
    axes.set_xlabel("axial position z [cm]")
    axes.set_ylabel("scalar flux [n cm⁻² s⁻¹]")
    axes.set_title("One-dimensional axial core-reflector eigenfunction")
    axes.legend()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        transparent=True,
        facecolor="none",
    )
    plt.close(figure)


def main() -> None:
    """Solve the axial eigenproblem and compare it with its analytic reference."""
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = build_configuration()
    reference = analytic_reference()
    settings = KeffSettings(
        max_outer_iterations=1_000,
        keff_change_tolerance=1.0e-10,
        flux_change_tolerance=1.0e-10,
        keff_relative_residual_tolerance=1.0e-10,
    )
    result = solve_keff(
        configuration, FissionSourceNormalization(rate=FISSION_SOURCE_RATE), settings
    )
    centres, reference_averages, scale = _normalized_reference_averages(
        configuration, reference
    )
    numerical_flux = np.array(
        [
            result.flux_layer(axial_index)[0, 0]
            for axial_index in range(result.n_axial_layers)
        ]
    )
    relative_flux_error = np.linalg.norm(
        numerical_flux - reference_averages
    ) / np.linalg.norm(reference_averages)
    relative_keff_error = abs(result.keff - reference.keff) / reference.keff

    flux_png_path = output_dir / "axial_flux_comparison.png"
    vtm_path = output_dir / "result.vtm"
    plot_comparison(
        flux_png_path,
        centres,
        numerical_flux,
        reference_averages,
        reference=reference,
        scale=scale,
    )
    result.export_vtm(vtm_path)
    if args.documentation_assets_dir is not None:
        documentation_dir = args.documentation_assets_dir.expanduser().resolve()
        documentation_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison(
            documentation_dir / "one_group_keff_axial_flux_comparison.png",
            centres,
            numerical_flux,
            reference_averages,
            reference=reference,
            scale=scale,
        )

    print(f"Configuration: {configuration.name}")
    print("Model: reflected radial column, core below a nonfissioning reflector")
    print("Boundary conditions: reflective radial/bottom, Marshak-vacuum top")
    print(f"Axial layers: {CORE_LAYERS} core + {REFLECTOR_LAYERS} reflector")
    print(f"Analytic k_eff: {reference.keff:.10f}")
    print(f"Morana k_eff:   {result.keff:.10f}")
    print(f"Relative k_eff error: {relative_keff_error:.3e}")
    print(f"Relative cell-average flux error: {relative_flux_error:.3e}")
    print(f"Power iterations: {result.execution_report.iterations}")
    print(
        "Relative k_eff residual: "
        f"{result.execution_report.final_outer_iteration.keff_relative_residual:.3e}"
    )
    print("Generated files:")
    print(f"  Axial flux comparison PNG: {flux_png_path}")
    print(f"  Result VTM:                {vtm_path}")


if __name__ == "__main__":
    main()
