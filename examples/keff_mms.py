"""Verify hex-z criticality diffusion with a refining manufactured eigenpair.

The active regular-hexagonal core is surrounded by an inactive excluded shell.
Unique shell keys and directional excluded-face selectors apply exact local
homogeneous Robin coefficients. Cell-local fission-production cross sections
make a positive Gaussian field an exact continuous eigenfunction at a selected
``k_eff``. See the k-effective manufactured-solution verification page for the
complete derivation.
"""

# pylint: disable=duplicate-code,wrong-import-position

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import log
import os
from pathlib import Path
from tempfile import gettempdir

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "examples" / "keff_mms"
DEFAULT_MATPLOTLIB_CONFIG_DIR = (
    Path(gettempdir()) / "morana_examples" / "matplotlib_config"
)
DEFAULT_MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MATPLOTLIB_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
from matplotlib.ticker import NullLocator  # pylint: disable=wrong-import-position
import numpy as np

from examples._hex_z_mms import (  # pylint: disable=wrong-import-position
    APOTHEM,
    HEIGHT,
    LEVELS,
    cell_average,
    excluded_radial_faces,
    layer_heights,
    level_rings,
    maximum_pitch,
    midplane_axial_index,
    volume_weighted_relative_l2,
)
from morana.solvers.finite_volume import solve_keff
from morana import (  # pylint: disable=wrong-import-position
    BoundaryCondition,
    BoundaryConditionSet,
    CrossSections,
    DirectLinearSolveSettings,
    ExcludedRegion,
    FissionData,
    FissionSourceNormalization,
    GmresLinearSolveSettings,
    HexPlanarMesh,
    Material,
    MaterialMesh,
    MaterialSlice,
    ProblemConfiguration,
    KeffSettings,
    SeparableFission,
    IluPreconditioner,
    JacobiPreconditioner,
    NoPreconditioner,
    WielandtShiftSettings,
)

TARGET_KEFF = 1.075
FISSION_SOURCE_RATE = 1.0e15
DIFFUSION = 1.20
ABSORPTION = 0.020
RADIAL_CURVATURE = 0.60
AXIAL_CURVATURE = 1.00
SHELL_KIND = "keff_mms_boundary"
SHELL_COLOR = "#d9d9d9"
MEDIUM_COLOR = "#4c78a8"


@dataclass(frozen=True)
class MMSField:
    """Evaluate the independently defined positive manufactured eigenfunction."""

    radial_scale: float = APOTHEM
    height: float = HEIGHT

    def flux(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return the dimensionless exact flux shape at Cartesian points."""
        radial_term = (
            RADIAL_CURVATURE
            * (np.asarray(x) ** 2 + np.asarray(y) ** 2)
            / self.radial_scale**2
        )
        axial_term = (
            AXIAL_CURVATURE * (np.asarray(z) - self.height / 2.0) ** 2 / self.height**2
        )
        return np.exp(-radial_term - axial_term)

    def laplacian(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return the analytic Laplacian of the exact flux shape."""
        x_values = np.asarray(x)
        y_values = np.asarray(y)
        axial_offset = np.asarray(z) - self.height / 2.0
        radial_ratio = (
            4.0
            * RADIAL_CURVATURE**2
            * (x_values**2 + y_values**2)
            / self.radial_scale**4
            - 4.0 * RADIAL_CURVATURE / self.radial_scale**2
        )
        axial_ratio = (
            4.0 * AXIAL_CURVATURE**2 * axial_offset**2 / self.height**4
            - 2.0 * AXIAL_CURVATURE / self.height**2
        )
        return self.flux(x_values, y_values, z) * (radial_ratio + axial_ratio)

    def fission_production(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        """Return ``nu_sigma_f * flux`` manufactured at ``TARGET_KEFF``."""
        flux = self.flux(x, y, z)
        return TARGET_KEFF * (-DIFFUSION * self.laplacian(x, y, z) + ABSORPTION * flux)

    def radial_robin_alpha(
        self,
        face_center: np.ndarray,
        outward_normal: np.ndarray,
    ) -> float:
        """Return the exact homogeneous Robin coefficient on one radial face."""
        return float(
            2.0
            * DIFFUSION
            * RADIAL_CURVATURE
            * np.dot(outward_normal, face_center)
            / self.radial_scale**2
        )

    @property
    def axial_robin_alpha(self) -> float:
        """Return the common exact homogeneous Robin coefficient at both ends."""
        return DIFFUSION * AXIAL_CURVATURE / self.height


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class RefinementResult:
    """Store independently compared eigenpair evidence for one mesh level."""

    level: int
    pitch: float
    max_height: float
    relative_l2_error: float
    max_relative_error: float
    keff_relative_error: float
    keff_relative_residual: float
    balance_relative_residual: float
    iterations: int
    radial_boundary_faces: int


# pylint: enable=too-many-instance-attributes


def parse_args() -> argparse.Namespace:
    """Parse artifact and optional documentation-asset output directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--documentation-assets-dir",
        type=Path,
        help="Optional directory for the tracked MMS documentation PNG assets.",
    )
    parser.add_argument(
        "--compare-strategies",
        action="store_true",
        help="Also verify the maintained strategy set at every refinement level.",
    )
    return parser.parse_args()


def _shell_key(axial_index: int, ring_position: int) -> str:
    """Return one unique inactive-shell key for a layer and ring position."""
    return f"keff_mms_shell_z{axial_index}_p{ring_position}"


def _material_key(axial_index: int, planar_id: int) -> str:
    """Return one unique active-material key for a manufactured cell."""
    return f"keff_mms_medium_z{axial_index}_c{planar_id}"


def _build_material_mesh(
    mesh: HexPlanarMesh,
    heights: tuple[float, ...],
) -> MaterialMesh:
    """Build cell-local active materials and a uniquely keyed inactive shell."""
    slices = []
    excluded_regions = {}
    for axial_index, height in enumerate(heights):
        material_keys = {}
        for index in mesh.openmc_indices:
            if index.ring == 0:
                key = _shell_key(axial_index, index.position)
                material_keys[index] = key
                excluded_regions[key] = ExcludedRegion(
                    kind=SHELL_KIND,
                    color=SHELL_COLOR,
                )
                continue
            planar_id = mesh.planar_id_at(index)
            if planar_id is None:
                raise RuntimeError("active MMS position has no planar ID")
            material_keys[index] = _material_key(axial_index, planar_id)
        slices.append(MaterialSlice(mesh, material_keys, height))
    return MaterialMesh.stack(tuple(slices), excluded_regions=excluded_regions)


def _manufactured_materials_and_flux(
    field: MMSField,
    material_mesh: MaterialMesh,
) -> tuple[dict[str, Material], tuple[np.ndarray, ...]]:
    """Return reaction-preserving cell materials and normalized exact flux."""
    materials = {}
    shape_layers = []
    production = 0.0
    mesh = material_mesh.mesh
    for axial_index in range(material_mesh.n_axial_layers):
        z_lower, _ = material_mesh.z_bounds(axial_index)
        height = material_mesh.layer_height(axial_index)
        material_by_id = material_mesh.material_by_active_id(axial_index)
        shape_values = []
        for active_id in range(material_mesh.n_active_cells(axial_index)):
            index = material_mesh.openmc_index_for_active_id(axial_index, active_id)
            planar_id = mesh.planar_id_at(index)
            if planar_id is None:
                raise RuntimeError("active MMS material has no planar ID")
            shape_average = float(
                cell_average(field.flux, mesh, planar_id, z_lower, height)
            )
            production_average = float(
                cell_average(
                    field.fission_production,
                    mesh,
                    planar_id,
                    z_lower,
                    height,
                )
            )
            nu_sigma_f = production_average / shape_average
            if not np.isfinite(nu_sigma_f) or nu_sigma_f <= 0.0:
                raise RuntimeError(
                    "manufactured fission cross section must be finite and positive"
                )
            key = material_by_id[active_id]
            materials[key] = Material(
                key,
                xs=CrossSections(
                    D=[DIFFUSION],
                    sigma_a=[ABSORPTION],
                    sigma_s=[[0.0]],
                    fission=FissionData(
                        neutron_production=SeparableFission(
                            nu_sigma_f=[nu_sigma_f], chi=[1.0]
                        )
                    ),
                ),
                color=MEDIUM_COLOR,
            )
            shape_values.append(shape_average)
            production += (
                nu_sigma_f * shape_average * material_mesh.cell_volume(axial_index)
            )
        shape_layers.append(np.array([shape_values]))
    scale = FISSION_SOURCE_RATE / production
    return materials, tuple(layer * scale for layer in shape_layers)


def _radial_boundary_assignments(
    field: MMSField,
    material_mesh: MaterialMesh,
) -> tuple[list, dict[tuple[int, int, str], float]]:
    """Return one exact homogeneous Robin assignment per radial boundary face."""
    assignments = []
    values_by_face = {}
    mesh = material_mesh.mesh
    for axial_index, active_id, direction, face in excluded_radial_faces(material_mesh):
        active_index = material_mesh.openmc_index_for_active_id(
            axial_index,
            active_id,
        )
        active_planar_id = mesh.planar_id_at(active_index)
        neighbor_planar_id = mesh.planar_id_at(face.neighbor_openmc_index)
        if active_planar_id is None or neighbor_planar_id is None:
            raise RuntimeError("boundary face has no planar identity")
        active_center = np.array(mesh.cartesian_center(active_planar_id))
        neighbor_center = np.array(mesh.cartesian_center(neighbor_planar_id))
        normal = (neighbor_center - active_center) / mesh.center_distance
        face_center = active_center + mesh.center_to_face * normal
        alpha = field.radial_robin_alpha(face_center, normal)
        if alpha < 0.0:
            raise RuntimeError("manufactured Robin coefficient must be passive")
        assignments.append(
            BoundaryCondition.robin(alpha).on_excluded(
                key=face.neighbor_key,
                direction=direction,
            )
        )
        values_by_face[(axial_index, active_id, direction)] = alpha
    return assignments, values_by_face


def _check_radial_boundary_resolution(
    material_mesh: MaterialMesh,
    boundary: BoundaryConditionSet,
    values_by_face: dict[tuple[int, int, str], float],
) -> None:
    """Assert that every active radial face resolves to its local Robin data."""
    resolved_faces = 0
    for axial_index, active_id, direction, face in excluded_radial_faces(material_mesh):
        condition = boundary.resolve(face)
        expected = values_by_face[(axial_index, active_id, direction)]
        if condition.kind != "robin" or condition.alpha != expected:
            raise RuntimeError(
                "MMS radial face did not resolve to its local Robin data"
            )
        resolved_faces += 1
    if resolved_faces != len(values_by_face):
        raise RuntimeError("MMS radial boundary-face accounting is inconsistent")


def build_configuration(
    level: int,
) -> tuple[ProblemConfiguration, tuple[np.ndarray, ...], int]:
    """Build one fully three-dimensional k-effective MMS refinement case."""
    active_rings = level_rings(level)
    mesh = HexPlanarMesh(
        num_rings=active_rings + 1,
        pitch=maximum_pitch(active_rings),
    )
    material_mesh = _build_material_mesh(mesh, layer_heights(level))
    field = MMSField()
    materials, exact_layers = _manufactured_materials_and_flux(
        field,
        material_mesh,
    )
    radial_assignments, values_by_face = _radial_boundary_assignments(
        field,
        material_mesh,
    )
    boundary = BoundaryConditionSet(
        BoundaryCondition.robin(field.axial_robin_alpha).on_bottom(),
        BoundaryCondition.robin(field.axial_robin_alpha).on_top(),
        *radial_assignments,
    )
    _check_radial_boundary_resolution(material_mesh, boundary, values_by_face)
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials=materials,
        material_mesh=material_mesh,
        boundary=boundary,
        name=f"one_group_keff_mms_level_{level}",
    )
    return configuration, exact_layers, len(values_by_face)


def strategy_settings() -> tuple[tuple[str, KeffSettings], ...]:
    """Return the compact supported k-effective strategy comparison set."""
    tolerance = 1.0e-11
    common = {
        "max_outer_iterations": 500,
        "keff_change_tolerance": tolerance,
        "flux_change_tolerance": tolerance,
        "keff_relative_residual_tolerance": tolerance,
    }
    direct = DirectLinearSolveSettings(relative_residual_tolerance=tolerance)
    return (
        ("direct-power", KeffSettings(inner_linear_solve=direct, **common)),
        (
            "gmres-none-power",
            KeffSettings(
                inner_linear_solve=GmresLinearSolveSettings(
                    relative_residual_tolerance=tolerance,
                    preconditioner=NoPreconditioner(),
                ),
                **common,
            ),
        ),
        (
            "gmres-jacobi-power",
            KeffSettings(
                inner_linear_solve=GmresLinearSolveSettings(
                    relative_residual_tolerance=tolerance,
                    preconditioner=JacobiPreconditioner(),
                ),
                **common,
            ),
        ),
        (
            "gmres-ilu-power",
            KeffSettings(
                inner_linear_solve=GmresLinearSolveSettings(
                    relative_residual_tolerance=tolerance,
                    preconditioner=IluPreconditioner(),
                ),
                **common,
            ),
        ),
        (
            "direct-wielandt",
            KeffSettings(
                inner_linear_solve=direct,
                eigenvalue_iteration=WielandtShiftSettings(0.9),
                **common,
            ),
        ),
        (
            "gmres-ilu-wielandt",
            KeffSettings(
                inner_linear_solve=GmresLinearSolveSettings(
                    relative_residual_tolerance=tolerance,
                    preconditioner=IluPreconditioner(),
                ),
                eigenvalue_iteration=WielandtShiftSettings(0.9),
                **common,
            ),
        ),
    )


def _solve_level(
    level: int,
    settings: KeffSettings | None = None,
) -> tuple[RefinementResult, ProblemConfiguration, object]:
    """Solve one level and compare it with independent eigenpair data."""
    configuration, exact_layers, boundary_faces = build_configuration(level)
    if settings is None:
        settings = strategy_settings()[0][1]
    result = solve_keff(
        configuration, FissionSourceNormalization(rate=FISSION_SOURCE_RATE), settings
    )
    maximum = 0.0
    for axial_index, exact in enumerate(exact_layers):
        difference = result.flux_layer(axial_index) - exact
        maximum = max(maximum, float(np.max(np.abs(difference) / exact)))
    balance_driving = sum(
        abs(result.balance[name])
        for name in ("keff_source", "absorption", "radial_leakage", "axial_leakage")
    )
    return (
        RefinementResult(
            level=level,
            pitch=configuration.mesh.pitch,
            max_height=max(configuration.material_mesh.axial_layer_heights),
            relative_l2_error=volume_weighted_relative_l2(
                configuration,
                result,
                exact_layers,
            ),
            max_relative_error=maximum,
            keff_relative_error=abs(result.keff - TARGET_KEFF) / TARGET_KEFF,
            keff_relative_residual=(
                result.execution_report.final_outer_iteration.keff_relative_residual
            ),
            balance_relative_residual=(
                abs(result.balance["residual"]) / balance_driving
            ),
            iterations=result.execution_report.iterations,
            radial_boundary_faces=boundary_faces,
        ),
        configuration,
        result,
    )


def _observed_order(
    coarser_error: float,
    finer_error: float,
    coarser_pitch: float,
    finer_pitch: float,
) -> float:
    """Return an observed order using the actual planar-pitch ratio."""
    return log(coarser_error / finer_error) / log(coarser_pitch / finer_pitch)


def _write_convergence_csv(rows: list[RefinementResult], output_dir: Path) -> None:
    """Write refinement and eigenpair evidence as comma-separated data."""
    lines = [
        "level,pitch_cm,max_axial_height_cm,relative_l2_error,"
        "max_relative_error,keff_relative_error,keff_relative_residual,"
        "balance_relative_residual,iterations,radial_boundary_faces,"
        "relative_l2_observed_order,max_relative_error_observed_order,"
        "keff_observed_order"
    ]
    for index, row in enumerate(rows):
        orders = ("", "", "")
        if index:
            previous = rows[index - 1]
            orders = tuple(
                str(
                    _observed_order(
                        getattr(previous, name),
                        getattr(row, name),
                        previous.pitch,
                        row.pitch,
                    )
                )
                for name in (
                    "relative_l2_error",
                    "max_relative_error",
                    "keff_relative_error",
                )
            )
        lines.append(
            f"{row.level},{row.pitch:.16e},{row.max_height:.16e},"
            f"{row.relative_l2_error:.16e},{row.max_relative_error:.16e},"
            f"{row.keff_relative_error:.16e},{row.keff_relative_residual:.16e},"
            f"{row.balance_relative_residual:.16e},{row.iterations},"
            f"{row.radial_boundary_faces},{orders[0]},{orders[1]},{orders[2]}"
        )
    (output_dir / "convergence.csv").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _plot_convergence(rows: list[RefinementResult], output_path: Path) -> None:
    """Write transparent normalized flux-shape and eigenvalue-error evidence."""
    pitches = np.array([row.pitch for row in rows])
    series = (
        (
            np.array([row.relative_l2_error for row in rows]),
            "o-",
            "#16697a",
            r"relative $E_2$ / level 0",
        ),
        (
            np.array([row.max_relative_error for row in rows]),
            "o-",
            "#b24c2f",
            "maximum relative flux error / level 0",
        ),
        (
            np.array([row.keff_relative_error for row in rows]),
            "s-",
            "#6c4f8d",
            r"relative $k_{\mathrm{eff}}$ error / level 0",
        ),
    )
    with plt.rc_context(
        {
            "axes.edgecolor": "#34495e",
            "axes.labelcolor": "#17202a",
            "axes.titlecolor": "#17202a",
            "font.size": 10.5,
            "xtick.color": "#34495e",
            "ytick.color": "#34495e",
        }
    ):
        figure, axes = plt.subplots(figsize=(8.2, 4.7))
        figure.patch.set_alpha(0.0)
        axes.patch.set_alpha(0.0)
        axes.loglog(
            pitches,
            (pitches / pitches[0]) ** 2,
            "--",
            color="#7f8c8d",
            linewidth=2.0,
            label=r"second-order guide, $O(p^2)$",
        )
        for values, marker, color, label in series:
            axes.loglog(
                pitches,
                values / values[0],
                marker,
                color=color,
                linewidth=2.8,
                label=label,
            )
        axes.set_xticks(pitches, [f"{pitch:.0f}" for pitch in pitches])
        axes.xaxis.set_minor_locator(NullLocator())
        y_ticks = np.arange(0.2, 1.01, 0.1)
        axes.set_yticks(y_ticks, [f"{value:.1f}" for value in y_ticks])
        axes.yaxis.set_minor_locator(NullLocator())
        axes.invert_xaxis()
        axes.set_xlabel(r"planar pitch $p$ [cm]")
        axes.set_ylabel("normalized value")
        axes.set_title("k-effective MMS convergence", loc="left", pad=9)
        axes.grid(color="#d5d8dc", linewidth=0.8, alpha=0.8)
        axes.spines[["top", "right"]].set_visible(False)
        axes.legend(frameon=True, facecolor="white", edgecolor="#7f8c8d")
        figure.tight_layout()
        figure.savefig(
            output_path,
            format="png",
            dpi=180,
            facecolor="none",
        )
        plt.close(figure)


def _save_midplane_flux(
    configuration: ProblemConfiguration,
    result,
    output_path: Path,
) -> None:
    """Write the representative finest-level midplane scalar-flux plot."""
    axes = result.plot_matplotlib(
        group=0,
        axial_index=midplane_axial_index(configuration, result),
    )
    axes.figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(axes.figure)


def _check_acceptance(rows: list[RefinementResult]) -> None:
    """Assert residual closure and monotonic eigenpair error reduction."""
    if any(row.keff_relative_residual > 1.0e-10 for row in rows):
        raise RuntimeError(
            "MMS relative eigenvalue residual did not meet the threshold"
        )
    if any(row.balance_relative_residual > 1.0e-10 for row in rows):
        raise RuntimeError("MMS balance residual did not meet the threshold")
    for name, label in (
        ("relative_l2_error", "relative-E2 flux"),
        ("max_relative_error", "maximum-relative flux"),
        ("keff_relative_error", "k-effective"),
    ):
        if any(
            getattr(later, name) >= getattr(earlier, name)
            for earlier, later in zip(rows, rows[1:])
        ):
            raise RuntimeError(f"MMS {label} error did not decrease under refinement")


def compare_strategies(
    levels: tuple[int, ...] = LEVELS,
) -> dict[str, tuple[RefinementResult, ...]]:
    """Verify each finite-volume strategy against the manufactured eigenpair.

    The maintained default covers all refinement levels. One representative
    level may be selected for compact execution-report demonstrations.
    """
    comparisons = {}
    for name, settings in strategy_settings():
        rows = tuple(_solve_level(level, settings)[0] for level in levels)
        if levels == LEVELS:
            _check_acceptance(list(rows))
        elif any(
            row.keff_relative_residual > 1.0e-10
            or row.balance_relative_residual > 1.0e-10
            for row in rows
        ):
            raise RuntimeError("MMS strategy did not meet residual acceptance")
        comparisons[name] = rows
    return comparisons


def _write_strategy_comparison(
    comparisons: dict[str, tuple[RefinementResult, ...]],
    output_dir: Path,
) -> None:
    """Write k-effective MMS strategy evidence as a portable CSV table."""
    lines = [
        "strategy,level,relative_l2_error,keff_relative_error,"
        "keff_relative_residual,"
        "balance_relative_residual,outer_iterations"
    ]
    for strategy, rows in comparisons.items():
        for row in rows:
            lines.append(
                f"{strategy},{row.level},{row.relative_l2_error:.16e},"
                f"{row.keff_relative_error:.16e},"
                f"{row.keff_relative_residual:.16e},"
                f"{row.balance_relative_residual:.16e},{row.iterations}"
            )
    (output_dir / "strategy_comparison.csv").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _print_strategy_comparison(
    comparisons: dict[str, tuple[RefinementResult, ...]],
) -> None:
    """Print compact k-effective strategy evidence after successful checks."""
    print("MMS strategy comparison")
    print("strategy             level  k-effective error  outer  residual")
    for strategy, rows in comparisons.items():
        for row in rows:
            print(
                f"{strategy:19}  {row.level:5d}  {row.keff_relative_error:16.8e}"
                f"  {row.iterations:5d}  {row.keff_relative_residual:15.8e}"
            )


def main() -> None:
    """Run the refinement study, assert evidence, and write artifacts."""
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    solved = [_solve_level(level) for level in LEVELS]
    rows = [entry[0] for entry in solved]
    finest_configuration, finest_result = solved[-1][1:]
    _check_acceptance(rows)
    if args.compare_strategies:
        comparisons = compare_strategies()
        _write_strategy_comparison(comparisons, output_dir)
        _print_strategy_comparison(comparisons)
    _write_convergence_csv(rows, output_dir)
    _plot_convergence(rows, output_dir / "convergence.png")
    _save_midplane_flux(
        finest_configuration,
        finest_result,
        output_dir / "flux_midplane_finest.png",
    )
    finest_result.export_vtm(output_dir / "result_finest.vtm")
    if args.documentation_assets_dir is not None:
        documentation_dir = args.documentation_assets_dir.expanduser().resolve()
        documentation_dir.mkdir(parents=True, exist_ok=True)
        _plot_convergence(rows, documentation_dir / "keff_mms_convergence.png")
        _save_midplane_flux(
            finest_configuration,
            finest_result,
            documentation_dir / "keff_mms_flux_midplane.png",
        )

    print("Configuration: one-group k-effective manufactured solution")
    print("Target: refining regular hexagonal prism with inactive excluded shell")
    print(
        "level  pitch [cm]  relative L2 error  maximum relative flux error  "
        "relative k_eff error"
    )
    for row in rows:
        print(
            f"{row.level:5d}  {row.pitch:10.4f}  {row.relative_l2_error:17.8e}"
            f"  {row.max_relative_error:27.8e}"
            f"  {row.keff_relative_error:20.8e}"
        )
    print(f"Target k_eff: {TARGET_KEFF:.8f}")
    print(f"Generated MMS artifacts in: {output_dir}")


if __name__ == "__main__":
    main()
