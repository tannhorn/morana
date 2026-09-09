"""Generate the tracked figures for the OpenMC--Morana comparison page.

The script reads the locally generated CE reference and Morana study records.
Raw OpenMC and Morana calculation artifacts remain outside version control;
only the resulting publication figures belong in ``docs/assets``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from artifact_paths import ARTIFACT_ROOT, REPOSITORY_ROOT, configure_matplotlib_cache
from compare_study_cases import (
    _load_ce_reference,
    _load_morana_case,
    compare_morana_cases,
)
from geometry import full_pitch_hex_vertices
from plot_unit_cell import plot_unit_cell

configure_matplotlib_cache()

# Matplotlib must read this script-local configuration before it is imported.
import matplotlib  # pylint: disable=wrong-import-position

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
from matplotlib.collections import (
    PatchCollection,
)  # pylint: disable=wrong-import-position
from matplotlib.colors import TwoSlopeNorm  # pylint: disable=wrong-import-position
from matplotlib.patches import Polygon  # pylint: disable=wrong-import-position

# The checked loaders are shared by scripts in this example directory rather
# than being package API.
# pylint: disable=protected-access

DEFAULT_STUDY_DIRECTORY = ARTIFACT_ROOT / "study" / "morana"
DEFAULT_CE_REFERENCE = (
    ARTIFACT_ROOT
    / "ce_reference"
    / "100m_seed31415_profiles_z100"
    / "reference_summary.json"
)
DEFAULT_DOCUMENTATION_ASSETS_DIRECTORY = REPOSITORY_ROOT / "docs" / "assets"

GROUP_STRUCTURES = (8, 25, 40, 70)
AXIAL_REFINEMENTS = (5, 10, 20, 50, 100)
SELECTED_SEEDS = (16180, 27182, 31415)


def _summary_path(
    study_directory: Path,
    *,
    active_histories_millions: int,
    seed: int,
    groups: int,
    axial_layers: int,
) -> Path:
    """Return the conventional path of one Morana study summary."""

    return (
        study_directory
        / f"{active_histories_millions}m_seed{seed}"
        / f"casmo-{groups}"
        / f"z{axial_layers}"
        / "summary.json"
    )


def _comparison_series(
    candidate_paths: tuple[Path, ...],
    reference_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Morana-to-Morana eigenvalue and production differences."""

    comparisons = tuple(
        compare_morana_cases(candidate_path, reference_path)
        for candidate_path in candidate_paths
    )
    return (
        np.asarray([float(comparison["delta_keff_pcm"]) for comparison in comparisons]),
        np.asarray(
            [
                float(comparison["axial_production_rms_percent"])
                for comparison in comparisons
            ]
        ),
        np.asarray(
            [
                float(comparison["planar_production_rms_percent"])
                for comparison in comparisons
            ]
        ),
    )


def _plot_convergence(study_directory: Path, output_path: Path) -> None:
    """Plot energy-group and axial-mesh convergence diagnostics."""

    group_paths = tuple(
        _summary_path(
            study_directory,
            active_histories_millions=20,
            seed=31415,
            groups=groups,
            axial_layers=20,
        )
        for groups in GROUP_STRUCTURES
    )
    group_keff, group_axial, group_planar = _comparison_series(
        group_paths,
        group_paths[-1],
    )

    axial_paths = tuple(
        _summary_path(
            study_directory,
            active_histories_millions=20,
            seed=31415,
            groups=25,
            axial_layers=axial_layers,
        )
        for axial_layers in AXIAL_REFINEMENTS
    )
    axial_keff, axial_profile, axial_planar = _comparison_series(
        axial_paths,
        axial_paths[-1],
    )

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    color_keff = "#1565c0"
    color_axial = "#d1495b"
    color_planar = "#2e7d32"

    axes[0, 0].plot(GROUP_STRUCTURES, group_keff, "o-", color=color_keff)
    axes[0, 0].axhline(0.0, color="#555555", linewidth=0.8)
    axes[0, 0].set(
        title="Energy-group convergence of $k_{eff}$",
        xlabel="Number of energy groups",
        ylabel="Difference from CASMO-70 [pcm]",
        xticks=GROUP_STRUCTURES,
    )
    axes[1, 0].plot(
        GROUP_STRUCTURES,
        group_axial,
        "o-",
        color=color_axial,
        label="Axial profile",
    )
    axes[1, 0].plot(
        GROUP_STRUCTURES,
        group_planar,
        "s-",
        color=color_planar,
        label="Planar profile",
    )
    axes[1, 0].set(
        title="Energy-group convergence of production",
        xlabel="Number of energy groups",
        ylabel="Normalized RMS difference [%]",
        xticks=GROUP_STRUCTURES,
    )
    axes[1, 0].legend()

    axes[0, 1].plot(AXIAL_REFINEMENTS, axial_keff, "o-", color=color_keff)
    axes[0, 1].axhline(0.0, color="#555555", linewidth=0.8)
    axes[0, 1].set(
        title="Axial-mesh convergence of $k_{eff}$",
        xlabel="Uniform axial layers",
        ylabel="Difference from 100 layers [pcm]",
        xticks=AXIAL_REFINEMENTS,
    )
    axes[1, 1].plot(
        AXIAL_REFINEMENTS,
        axial_profile,
        "o-",
        color=color_axial,
        label="Axial profile",
    )
    axes[1, 1].plot(
        AXIAL_REFINEMENTS,
        axial_planar,
        "s-",
        color=color_planar,
        label="Planar profile",
    )
    axes[1, 1].set(
        title="Axial-mesh convergence of production",
        xlabel="Uniform axial layers",
        ylabel="Normalized RMS difference [%]",
        xticks=AXIAL_REFINEMENTS,
    )
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(True, color="#e0e0e0", linewidth=0.7)
    figure.savefig(output_path, dpi=220, transparent=True)
    plt.close(figure)


def _selected_profiles(
    study_directory: Path,
) -> tuple[np.ndarray, float]:
    """Return the mean selected production field and multiplication factor."""

    cases = tuple(
        _load_morana_case(
            _summary_path(
                study_directory,
                active_histories_millions=40,
                seed=seed,
                groups=70,
                axial_layers=20,
            )
        )
        for seed in SELECTED_SEEDS
    )
    production = np.mean(np.stack([case.production for case in cases]), axis=0)
    keff = float(np.mean([float(case.record["keff"]) for case in cases]))
    return production, keff


def _hex_collection(values: np.ndarray) -> PatchCollection:
    """Return a colored patch collection in the checked shared cell order."""

    from geometry import (
        mini_core_coordinates,
    )  # pylint: disable=import-outside-toplevel

    coordinates = mini_core_coordinates()
    if values.shape != (len(coordinates),):
        raise ValueError("planar field does not span the 61-cell mini-core")
    patches = [
        Polygon(full_pitch_hex_vertices(x_coord, y_coord), closed=True)
        for _, _, x_coord, y_coord in coordinates
    ]
    collection = PatchCollection(
        patches,
        cmap="inferno",
        edgecolor="#263238",
        linewidth=0.45,
    )
    collection.set_array(values)
    return collection


def _plot_radial_production(
    morana_production: np.ndarray,
    ce_planar_production: np.ndarray,
    output_path: Path,
) -> None:
    """Plot CE, Morana, and difference maps of integrated production."""

    morana_profile = np.sum(morana_production, axis=0)
    ce_relative = ce_planar_production / np.mean(ce_planar_production)
    morana_relative = morana_profile / np.mean(morana_profile)
    error_percent = (
        100.0 * (morana_profile - ce_planar_production) / np.mean(ce_planar_production)
    )

    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.8), constrained_layout=True)
    collections = (
        _hex_collection(ce_relative),
        _hex_collection(morana_relative),
        _hex_collection(error_percent),
    )
    shared_minimum = float(min(np.min(ce_relative), np.min(morana_relative)))
    shared_maximum = float(max(np.max(ce_relative), np.max(morana_relative)))
    for collection in collections[:2]:
        collection.set_clim(shared_minimum, shared_maximum)
    error_limit = float(np.max(np.abs(error_percent)))
    collections[2].set_cmap("coolwarm")
    collections[2].set_norm(
        TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit)
    )

    titles = (
        "OpenMC continuous energy",
        "Morana CASMO-70",
        "Morana minus OpenMC",
    )
    for axis, collection, title in zip(axes, collections, titles, strict=True):
        axis.add_collection(collection)
        axis.autoscale_view()
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title)
        axis.set_xlabel("x [cm]")
        axis.set_ylabel("y [cm]")
    figure.colorbar(
        collections[0],
        ax=axes[:2],
        label="Production relative to cell mean",
        shrink=0.82,
    )
    figure.colorbar(
        collections[2],
        ax=axes[2],
        label="Difference relative to OpenMC cell mean [%]",
        shrink=0.82,
    )
    figure.savefig(output_path, dpi=220, transparent=True)
    plt.close(figure)


def _plot_axial_production(
    morana_production: np.ndarray,
    ce_axial_production: np.ndarray,
    active_height_cm: float,
    output_path: Path,
) -> None:
    """Plot axial production profiles and their rebinned difference."""

    morana_profile = np.sum(morana_production, axis=1)
    morana_bins = morana_profile.size
    ce_bins = ce_axial_production.size
    if ce_bins % morana_bins:
        raise ValueError("CE axial bins cannot be exactly rebinned to Morana layers")
    ce_rebinned = np.sum(
        ce_axial_production.reshape(morana_bins, ce_bins // morana_bins),
        axis=1,
    )
    ce_edges = np.linspace(0.0, active_height_cm, ce_bins + 1)
    morana_edges = np.linspace(0.0, active_height_cm, morana_bins + 1)
    difference_percent = 100.0 * morana_bins * (morana_profile - ce_rebinned)

    figure, (profile_axis, error_axis) = plt.subplots(
        2,
        1,
        figsize=(9.0, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": (2.2, 1.0)},
        constrained_layout=True,
    )
    profile_axis.stairs(
        ce_bins * ce_axial_production,
        ce_edges,
        color="#222222",
        linewidth=1.4,
        label="OpenMC CE (100 bins)",
    )
    profile_axis.stairs(
        morana_bins * morana_profile,
        morana_edges,
        color="#1565c0",
        linewidth=2.0,
        label="Morana CASMO-70 (20 layers)",
    )
    profile_axis.set_ylabel("Production relative to axial mean")
    profile_axis.legend()
    profile_axis.grid(True, color="#e0e0e0", linewidth=0.7)

    error_axis.stairs(
        difference_percent,
        morana_edges,
        color="#d1495b",
        linewidth=1.7,
        fill=True,
        alpha=0.30,
    )
    error_axis.axhline(0.0, color="#555555", linewidth=0.8)
    error_axis.set(
        xlabel="Axial position [cm]",
        ylabel="Difference [% of\nOpenMC bin mean]",
        xlim=(0.0, active_height_cm),
    )
    error_axis.grid(True, color="#e0e0e0", linewidth=0.7)
    figure.savefig(output_path, dpi=220, transparent=True)
    plt.close(figure)


def _arguments() -> argparse.Namespace:
    """Parse documentation-figure input and output paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=DEFAULT_STUDY_DIRECTORY,
        help="Directory containing Morana history/seed study records.",
    )
    parser.add_argument(
        "--ce-reference",
        type=Path,
        default=DEFAULT_CE_REFERENCE,
        help="Qualified CE reference summary.",
    )
    parser.add_argument(
        "--documentation-assets-dir",
        type=Path,
        default=DEFAULT_DOCUMENTATION_ASSETS_DIRECTORY,
        help="Destination for the four tracked PNG figures.",
    )
    return parser.parse_args()


def main() -> None:
    """Validate local study records and write all documentation figures."""

    arguments = _arguments()
    study_directory = arguments.study_dir.expanduser().resolve()
    ce_reference = _load_ce_reference(arguments.ce_reference)
    output_directory = arguments.documentation_assets_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    outputs = {
        "unit cell": output_directory / "openmc_comparison_unit_cell.png",
        "convergence": output_directory / "openmc_comparison_convergence.png",
        "radial production": (
            output_directory / "openmc_comparison_radial_production.png"
        ),
        "axial production": output_directory / "openmc_comparison_axial_production.png",
    }
    morana_production, _ = _selected_profiles(study_directory)
    plot_unit_cell(outputs["unit cell"], transparent=True)
    _plot_convergence(study_directory, outputs["convergence"])
    _plot_radial_production(
        morana_production,
        ce_reference.planar_production,
        outputs["radial production"],
    )
    _plot_axial_production(
        morana_production,
        ce_reference.axial_production,
        ce_reference.active_height_cm,
        outputs["axial production"],
    )
    for description, output in outputs.items():
        print(f"Wrote {description} figure: {output}")


if __name__ == "__main__":
    main()
