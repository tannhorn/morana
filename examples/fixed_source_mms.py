"""Verify coupled fixed-source diffusion with a refining manufactured solution.

The active regular-hexagonal core is surrounded by an inactive excluded shell.
Unique shell keys and directional excluded-face selectors apply independently
integrated Dirichlet values to every radial active face. Non-unity scattering
multiplicity exercises both intergroup and same-group scattering coupling. See
the fixed-source manufactured-solution verification page for the complete
derivation.
"""

# pylint: disable=wrong-import-position

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import log, sqrt
import os
from pathlib import Path
from tempfile import gettempdir

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "examples" / "fixed_source_mms"
DEFAULT_MATPLOTLIB_CONFIG_DIR = (
    Path(gettempdir()) / "morana_examples" / "matplotlib_config"
)
DEFAULT_MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MATPLOTLIB_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
from matplotlib.patches import Patch  # pylint: disable=wrong-import-position
from matplotlib.ticker import NullLocator  # pylint: disable=wrong-import-position
import numpy as np

from _hex_z_mms import (  # pylint: disable=wrong-import-position
    APOTHEM,
    HEIGHT,
    LEVELS,
    QUADRATURE_ORDER,
    cell_average,
    excluded_radial_faces,
    layer_heights,
    level_rings,
    maximum_pitch,
    midplane_axial_index,
    volume_weighted_relative_l2,
)

from morana.solvers.finite_volume import solve_fixed_source
from morana import (  # pylint: disable=wrong-import-position
    BoundaryCondition,
    BoundaryConditionSet,
    CellSource,
    CrossSections,
    DirectLinearSolveSettings,
    ExcludedRegion,
    FixedSourceSettings,
    GmresLinearSolveSettings,
    HexPlanarMesh,
    IluPreconditioner,
    JacobiPreconditioner,
    Material,
    MaterialMesh,
    MaterialSlice,
    ProblemConfiguration,
    NoPreconditioner,
)

MEDIUM_KEY = "mms_medium"
SHELL_KIND = "mms_boundary"
SHELL_COLOR = "#d9d9d9"
PHI_BASE = np.array([1.20e12, 0.85e12, 0.65e12])
MODE_A = np.array([0.12, -0.09, 0.08])
MODE_B = np.array([0.07, 0.11, -0.10])
DIFFUSION = np.array([1.40, 0.80, 0.35])
ABSORPTION = np.array([0.030, 0.040, 0.055])
SCATTER = np.array(
    [
        [0.0030, 0.0040, 0.0010],
        [0.0005, 0.0025, 0.0060],
        [0.0002, 0.0008, 0.0040],
    ]
)
SCATTERING_MULTIPLICITY = np.array(
    [
        [1.10, 1.15, 1.05],
        [1.00, 0.90, 1.10],
        [1.20, 1.00, 1.05],
    ]
)


@dataclass(frozen=True)
class MMSField:
    """Evaluate the independently defined multigroup manufactured field."""

    radial_scale: float = APOTHEM
    height: float = HEIGHT

    @property
    def removal(self) -> np.ndarray:
        """Return the groupwise removal cross sections in inverse centimetres."""
        return ABSORPTION + np.sum(SCATTER, axis=1) - np.diag(SCATTER)

    @property
    def scattering_coupling(self) -> np.ndarray:
        """Return independent signed scattering-neutron coupling coefficients."""
        coupling = SCATTERING_MULTIPLICITY * SCATTER
        diagonal = np.diag_indices_from(coupling)
        coupling[diagonal] -= SCATTER[diagonal]
        return coupling

    def flux(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return exact group fluxes at broadcast-compatible Cartesian points."""
        x_phase = np.pi * np.asarray(x) / self.radial_scale
        y_phase = np.pi * np.asarray(y) / (sqrt(3.0) * self.radial_scale)
        z_mode = np.sin(np.pi * np.asarray(z) / self.height)
        cosine_mode = np.cos(x_phase) * np.cos(y_phase)
        sine_mode = np.sin(2.0 * x_phase) * np.cos(y_phase)
        mode = np.array(
            [
                MODE_A[group] * cosine_mode + MODE_B[group] * sine_mode
                for group in range(PHI_BASE.size)
            ]
        )
        return PHI_BASE.reshape((-1,) + (1,) * z_mode.ndim) * (1.0 + z_mode * mode)

    def source(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return the exact nonnegative volumetric source in n / cm^3 / s."""
        x_phase = np.pi * np.asarray(x) / self.radial_scale
        y_phase = np.pi * np.asarray(y) / (sqrt(3.0) * self.radial_scale)
        z_mode = np.sin(np.pi * np.asarray(z) / self.height)
        cosine_mode = np.cos(x_phase) * np.cos(y_phase)
        sine_mode = np.sin(2.0 * x_phase) * np.cos(y_phase)
        cosine_laplacian = (
            -(
                (np.pi / self.radial_scale) ** 2
                + (np.pi / (sqrt(3.0) * self.radial_scale)) ** 2
            )
            * cosine_mode
        )
        sine_laplacian = (
            -(
                (2.0 * np.pi / self.radial_scale) ** 2
                + (np.pi / (sqrt(3.0) * self.radial_scale)) ** 2
            )
            * sine_mode
        )
        mode = np.array(
            [
                MODE_A[group] * cosine_mode + MODE_B[group] * sine_mode
                for group in range(PHI_BASE.size)
            ]
        )
        mode_laplacian = np.array(
            [
                MODE_A[group] * cosine_laplacian + MODE_B[group] * sine_laplacian
                for group in range(PHI_BASE.size)
            ]
        )
        flux = self.flux(x, y, z)
        laplacian = (
            PHI_BASE.reshape((-1,) + (1,) * z_mode.ndim)
            * z_mode
            * (mode_laplacian - (np.pi / self.height) ** 2 * mode)
        )
        incoming_scatter = np.einsum(
            "hg,h...->g...",
            self.scattering_coupling,
            flux,
        )
        return -DIFFUSION.reshape((-1,) + (1,) * z_mode.ndim) * laplacian + (
            self.removal.reshape((-1,) + (1,) * z_mode.ndim) * flux - incoming_scatter
        )


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class RefinementResult:
    """Store one independently compared refinement result."""

    level: int
    pitch: float
    max_height: float
    relative_l2_error: float
    max_relative_error: float
    linear_residual: float
    balance_relative_residual: float
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
        help="Also verify all supported linear strategies at every refinement level.",
    )
    return parser.parse_args()


def _shell_key(axial_index: int, ring_position: int) -> str:
    """Return one unique inactive-shell key for a layer and outer-ring position."""
    return f"mms_shell_z{axial_index}_p{ring_position}"


def _build_material_mesh(
    mesh: HexPlanarMesh, heights: tuple[float, ...]
) -> MaterialMesh:
    """Build an active core and one uniquely keyed inactive excluded shell."""
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
            else:
                material_keys[index] = MEDIUM_KEY
        slices.append(MaterialSlice(mesh, material_keys, height))
    return MaterialMesh.stack(tuple(slices), excluded_regions=excluded_regions)


def _radial_face_average(
    field: MMSField,
    mesh: HexPlanarMesh,
    active_planar_id: int,
    neighbor_planar_id: int,
    *,
    z_lower: float,
    height: float,
) -> np.ndarray:
    """Return an independently quadrature-averaged exact radial face flux."""
    nodes, weights = np.polynomial.legendre.leggauss(QUADRATURE_ORDER)
    active_center = np.array(mesh.cartesian_center(active_planar_id))
    neighbor_center = np.array(mesh.cartesian_center(neighbor_planar_id))
    normal = (neighbor_center - active_center) / mesh.center_distance
    tangent = np.array((-normal[1], normal[0]))
    face_center = active_center + mesh.center_to_face * normal
    edge_values, z_values = np.meshgrid(nodes, nodes, indexing="ij")
    edge_weights, z_weights = np.meshgrid(weights, weights, indexing="ij")
    points = face_center + (
        mesh.face_length / 2.0 * edge_values[..., np.newaxis] * tangent
    )
    z_points = z_lower + height / 2.0 * (z_values + 1.0)
    face_weights = edge_weights * z_weights * mesh.face_length * height / 4.0
    integral = field.flux(
        points[..., 0].ravel(),
        points[..., 1].ravel(),
        z_points.ravel(),
    )
    return integral @ face_weights.ravel() / (mesh.face_length * height)


def _source_layers(
    field: MMSField, material_mesh: MaterialMesh
) -> tuple[np.ndarray, ...]:
    """Return independent cell-average manufactured-source arrays by layer."""
    layers = []
    mesh = material_mesh.mesh
    for axial_index in range(material_mesh.n_axial_layers):
        z_lower, _ = material_mesh.z_bounds(axial_index)
        height = material_mesh.layer_height(axial_index)
        values = []
        for active_id in range(material_mesh.n_active_cells(axial_index)):
            index = material_mesh.openmc_index_for_active_id(axial_index, active_id)
            planar_id = mesh.planar_id_at(index)
            if planar_id is None:
                raise RuntimeError("active material has no planar ID")
            values.append(cell_average(field.source, mesh, planar_id, z_lower, height))
        layer = np.column_stack(values)
        if np.any(layer < 0.0):
            raise RuntimeError("manufactured source must be non-negative")
        layers.append(layer)
    return tuple(layers)


def _radial_boundary_assignments(
    field: MMSField, material_mesh: MaterialMesh
) -> tuple[list, dict[tuple[int, int, str], np.ndarray]]:
    """Return one directional excluded-shell Dirichlet assignment per radial face."""
    assignments = []
    values_by_face = {}
    mesh = material_mesh.mesh
    for axial_index, active_id, direction, face in excluded_radial_faces(material_mesh):
        z_lower, _ = material_mesh.z_bounds(axial_index)
        height = material_mesh.layer_height(axial_index)
        active_index = material_mesh.openmc_index_for_active_id(
            axial_index,
            active_id,
        )
        active_planar_id = mesh.planar_id_at(active_index)
        neighbor_planar_id = mesh.planar_id_at(face.neighbor_openmc_index)
        if active_planar_id is None or neighbor_planar_id is None:
            raise RuntimeError("boundary face has no planar identity")
        flux = _radial_face_average(
            field,
            mesh,
            active_planar_id,
            neighbor_planar_id,
            z_lower=z_lower,
            height=height,
        )
        assignments.append(
            BoundaryCondition.dirichlet(flux).on_excluded(
                key=face.neighbor_key,
                direction=direction,
            )
        )
        values_by_face[(axial_index, active_id, direction)] = flux
    return assignments, values_by_face


def _check_radial_boundary_resolution(
    material_mesh: MaterialMesh,
    boundary: BoundaryConditionSet,
    values_by_face: dict[tuple[int, int, str], np.ndarray],
) -> None:
    """Assert that every active radial face resolves to its local Dirichlet data."""
    resolved_faces = 0
    for axial_index, active_id, direction, face in excluded_radial_faces(material_mesh):
        condition = boundary.resolve(face)
        expected = values_by_face[(axial_index, active_id, direction)]
        if condition.kind != "dirichlet" or not np.array_equal(
            condition.flux,
            expected,
        ):
            raise RuntimeError(
                "MMS radial face did not resolve to its local Dirichlet data"
            )
        resolved_faces += 1
    if resolved_faces != len(values_by_face):
        raise RuntimeError("MMS radial boundary-face accounting is inconsistent")


def build_configuration(
    level: int,
) -> tuple[ProblemConfiguration, tuple[np.ndarray, ...], int]:
    """Build one fully coupled fixed-source MMS refinement configuration."""
    active_rings = level_rings(level)
    mesh = HexPlanarMesh(
        num_rings=active_rings + 1,
        pitch=maximum_pitch(active_rings),
    )
    material_mesh = _build_material_mesh(mesh, layer_heights(level))
    field = MMSField()
    source_layers = _source_layers(field, material_mesh)
    radial_assignments, values_by_face = _radial_boundary_assignments(
        field,
        material_mesh,
    )
    boundary = BoundaryConditionSet(
        BoundaryCondition.dirichlet(PHI_BASE).on_bottom(),
        BoundaryCondition.dirichlet(PHI_BASE).on_top(),
        *radial_assignments,
    )
    _check_radial_boundary_resolution(material_mesh, boundary, values_by_face)
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={
            MEDIUM_KEY: Material(
                MEDIUM_KEY,
                xs=CrossSections(
                    D=DIFFUSION,
                    sigma_a=ABSORPTION,
                    sigma_s=SCATTER,
                    multiplicity_matrix=SCATTERING_MULTIPLICITY,
                    fission=None,
                ),
                color="#4c78a8",
            )
        },
        material_mesh=material_mesh,
        boundary=boundary,
        source=CellSource(source_layers),
        name=f"three_group_fixed_source_mms_level_{level}",
    )
    return configuration, _exact_flux_layers(field, material_mesh), len(values_by_face)


def _exact_flux_layers(
    field: MMSField,
    material_mesh: MaterialMesh,
) -> tuple[np.ndarray, ...]:
    """Return independently quadrature-averaged exact flux layers."""
    layers = []
    mesh = material_mesh.mesh
    for axial_index in range(material_mesh.n_axial_layers):
        z_lower, _ = material_mesh.z_bounds(axial_index)
        height = material_mesh.layer_height(axial_index)
        values = []
        for active_id in range(material_mesh.n_active_cells(axial_index)):
            index = material_mesh.openmc_index_for_active_id(axial_index, active_id)
            planar_id = mesh.planar_id_at(index)
            if planar_id is None:
                raise RuntimeError("active material has no planar ID")
            values.append(cell_average(field.flux, mesh, planar_id, z_lower, height))
        layers.append(np.column_stack(values))
    return tuple(layers)


def strategy_settings() -> tuple[tuple[str, FixedSourceSettings], ...]:
    """Return the compact supported fixed-source strategy comparison set."""
    tolerance = 1.0e-11
    return (
        (
            "direct",
            FixedSourceSettings(
                linear_solve=DirectLinearSolveSettings(tolerance),
            ),
        ),
        (
            "gmres-none",
            FixedSourceSettings(
                linear_solve=GmresLinearSolveSettings(
                    relative_residual_tolerance=tolerance,
                    preconditioner=NoPreconditioner(),
                ),
            ),
        ),
        (
            "gmres-jacobi",
            FixedSourceSettings(
                linear_solve=GmresLinearSolveSettings(
                    relative_residual_tolerance=tolerance,
                    preconditioner=JacobiPreconditioner(),
                ),
            ),
        ),
        (
            "gmres-ilu",
            FixedSourceSettings(
                linear_solve=GmresLinearSolveSettings(
                    relative_residual_tolerance=tolerance,
                    preconditioner=IluPreconditioner(),
                ),
            ),
        ),
    )


def _solve_level(
    level: int,
    settings: FixedSourceSettings | None = None,
) -> tuple[RefinementResult, ProblemConfiguration, object]:
    """Solve one level and compare it with independent exact cell averages."""
    configuration, exact_layers, boundary_faces = build_configuration(level)
    result = solve_fixed_source(configuration, settings)
    maximum = 0.0
    for axial_index, exact in enumerate(exact_layers):
        difference = result.flux_layer(axial_index) - exact
        maximum = max(maximum, float(np.max(np.abs(difference) / exact)))
    driving = sum(
        abs(result.balance[name])
        for name in ("source", "boundary_source", "fission_emission")
    )
    balance_relative_residual = abs(result.balance["residual"]) / driving
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
            linear_residual=result.execution_report.true_relative_residual,
            balance_relative_residual=balance_relative_residual,
            radial_boundary_faces=boundary_faces,
        ),
        configuration,
        result,
    )


def _write_convergence_csv(rows: list[RefinementResult], output_dir: Path) -> None:
    """Write refinement evidence in a simple portable comma-separated table."""
    lines = [
        "level,pitch_cm,max_axial_height_cm,relative_l2_error,"
        "max_relative_error,linear_residual,balance_relative_residual,"
        "radial_boundary_faces,relative_l2_observed_order,"
        "max_relative_error_observed_order"
    ]
    for index, row in enumerate(rows):
        l2_order = "" if index == 0 else str(_observed_order(rows[index - 1], row))
        maximum_order = (
            ""
            if index == 0
            else str(_maximum_error_observed_order(rows[index - 1], row))
        )
        lines.append(
            f"{row.level},{row.pitch:.16e},{row.max_height:.16e},"
            f"{row.relative_l2_error:.16e},{row.max_relative_error:.16e},"
            f"{row.linear_residual:.16e},{row.balance_relative_residual:.16e},"
            f"{row.radial_boundary_faces},{l2_order},{maximum_order}"
        )
    (output_dir / "convergence.csv").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _plot_convergence(rows: list[RefinementResult], output_path: Path) -> None:
    """Write transparent normalized RMS and maximum flux-error evidence."""
    pitches = np.array([row.pitch for row in rows])
    errors = np.array([row.relative_l2_error for row in rows])
    maximum_errors = np.array([row.max_relative_error for row in rows])
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
            pitches / pitches[0],
            "--",
            color="#7f8c8d",
            linewidth=2.0,
            label=r"first-order guide, $O(p)$",
        )
        axes.loglog(
            pitches,
            errors / errors[0],
            "o-",
            color="#16697a",
            linewidth=2.8,
            label=r"relative $E_2$ / level 0",
        )
        axes.loglog(
            pitches,
            maximum_errors / maximum_errors[0],
            "o-",
            color="#b24c2f",
            linewidth=2.8,
            label="maximum relative flux error / level 0",
        )
        axes.set_xticks(pitches, [f"{pitch:.0f}" for pitch in pitches])
        axes.xaxis.set_minor_locator(NullLocator())
        y_ticks = np.arange(0.4, 1.01, 0.1)
        axes.set_yticks(y_ticks, [f"{value:.1f}" for value in y_ticks])
        axes.yaxis.set_minor_locator(NullLocator())
        axes.invert_xaxis()
        axes.set_xlabel(r"planar pitch $p$ [cm]")
        axes.set_ylabel("normalized value")
        axes.set_title("Fixed-source MMS convergence", loc="left", pad=9)
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


def _save_material_layout(
    configuration: ProblemConfiguration,
    output_path: Path,
) -> None:
    """Write a two-entry active-core and inactive-shell material-layout plot."""
    axes = configuration.material_mesh.plot_matplotlib(
        axial_index=0,
        materials=configuration.materials,
    )
    axes.get_legend().remove()
    axes.legend(
        handles=(
            Patch(
                facecolor="#4c78a8",
                edgecolor="black",
                label="active MMS medium",
            ),
            Patch(
                facecolor=SHELL_COLOR,
                edgecolor="black",
                label="inactive shell",
            ),
        ),
        title="layout",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    axes.figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(axes.figure)


def _save_midplane_fluxes(
    configuration: ProblemConfiguration,
    result,
    output_directory: Path,
    stem: str,
) -> None:
    """Write representative fast and thermal midplane scalar-flux plots.

    The exported PNGs use transparent figure and axes backgrounds so they
    integrate with the documentation page background.
    """
    axial_index = midplane_axial_index(configuration, result)
    for group in (0, 2):
        axes = result.plot_matplotlib(group=group, axial_index=axial_index)
        axes.figure.savefig(
            output_directory / f"{stem}_g{group}.png",
            dpi=200,
            bbox_inches="tight",
            transparent=True,
        )
        plt.close(axes.figure)


def _check_acceptance(rows: list[RefinementResult]) -> None:
    """Assert residual closure and monotonic error-reduction evidence."""
    if any(row.linear_residual > 1.0e-11 for row in rows):
        raise RuntimeError("MMS linear residual did not meet the acceptance threshold")
    if any(row.balance_relative_residual > 1.0e-11 for row in rows):
        raise RuntimeError("MMS balance residual did not meet the acceptance threshold")
    if any(
        later.relative_l2_error >= earlier.relative_l2_error
        for earlier, later in zip(rows, rows[1:])
    ):
        raise RuntimeError("MMS relative-E2 error did not decrease under refinement")
    if any(
        later.max_relative_error >= earlier.max_relative_error
        for earlier, later in zip(rows, rows[1:])
    ):
        raise RuntimeError(
            "MMS maximum relative error did not decrease under refinement"
        )


def compare_strategies(
    levels: tuple[int, ...] = LEVELS,
) -> dict[str, tuple[RefinementResult, ...]]:
    """Verify every fixed-source execution strategy against the MMS reference.

    The default covers the full maintained refinement study. Callers that only
    need a compact comparison may select one representative level.
    """
    comparisons = {}
    for name, settings in strategy_settings():
        rows = tuple(_solve_level(level, settings)[0] for level in levels)
        if levels == LEVELS:
            _check_acceptance(list(rows))
        elif any(
            row.linear_residual > 1.0e-11 or row.balance_relative_residual > 1.0e-11
            for row in rows
        ):
            raise RuntimeError("MMS strategy did not meet residual acceptance")
        comparisons[name] = rows
    return comparisons


def _write_strategy_comparison(
    comparisons: dict[str, tuple[RefinementResult, ...]],
    output_dir: Path,
) -> None:
    """Write fixed-source MMS strategy evidence as a portable CSV table."""
    lines = [
        "strategy,level,relative_l2_error,max_relative_error,linear_residual,"
        "balance_relative_residual"
    ]
    for strategy, rows in comparisons.items():
        for row in rows:
            lines.append(
                f"{strategy},{row.level},{row.relative_l2_error:.16e},"
                f"{row.max_relative_error:.16e},{row.linear_residual:.16e},"
                f"{row.balance_relative_residual:.16e}"
            )
    (output_dir / "strategy_comparison.csv").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _print_strategy_comparison(
    comparisons: dict[str, tuple[RefinementResult, ...]],
) -> None:
    """Print compact fixed-source strategy evidence after successful checks."""
    print("MMS strategy comparison")
    print("strategy       level  relative L2 error  linear residual")
    for strategy, rows in comparisons.items():
        for row in rows:
            print(
                f"{strategy:13}  {row.level:5d}  {row.relative_l2_error:17.8e}"
                f"  {row.linear_residual:15.8e}"
            )


def _observed_order(coarser: RefinementResult, finer: RefinementResult) -> float:
    """Return the relative-E2 order using the actual planar pitch ratio."""
    return _pitch_order(
        coarser.relative_l2_error,
        finer.relative_l2_error,
        coarser.pitch,
        finer.pitch,
    )


def _maximum_error_observed_order(
    coarser: RefinementResult,
    finer: RefinementResult,
) -> float:
    """Return the maximum-relative-flux-error order using the pitch ratio."""
    return _pitch_order(
        coarser.max_relative_error,
        finer.max_relative_error,
        coarser.pitch,
        finer.pitch,
    )


def _pitch_order(
    coarser_error: float,
    finer_error: float,
    coarser_pitch: float,
    finer_pitch: float,
) -> float:
    """Return an observed order for two error values and planar pitches."""
    return log(coarser_error / finer_error) / log(coarser_pitch / finer_pitch)


def main() -> None:
    """Run the full refinement study, assert evidence, and write artifacts."""
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
    _save_material_layout(
        finest_configuration,
        output_dir / "material_layout_finest.png",
    )
    _save_midplane_fluxes(
        finest_configuration,
        finest_result,
        output_dir,
        "flux_midplane_finest",
    )
    finest_result.export_vtm(output_dir / "result_finest.vtm")
    if args.documentation_assets_dir is not None:
        documentation_dir = args.documentation_assets_dir.expanduser().resolve()
        documentation_dir.mkdir(parents=True, exist_ok=True)
        _plot_convergence(rows, documentation_dir / "fixed_source_mms_convergence.png")
        _save_midplane_fluxes(
            finest_configuration,
            finest_result,
            documentation_dir,
            "fixed_source_mms_flux_midplane",
        )

    print("Configuration: three-group fixed-source manufactured solution")
    print("Target: refining regular hexagonal prism with inactive excluded shell")
    print("level  pitch [cm]  relative L2 error  maximum relative flux error")
    for row in rows:
        print(
            f"{row.level:5d}  {row.pitch:10.4f}  {row.relative_l2_error:17.8e}"
            f"  {row.max_relative_error:27.8e}"
        )
    print(f"Generated MMS artifacts in: {output_dir}")


if __name__ == "__main__":
    main()
