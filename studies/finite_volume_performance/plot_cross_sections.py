"""Plot the frozen synthetic cross sections for development review.

The rendered PNG files are local study evidence.  By default they are written
below ``artifacts/`` and are not tracked.
"""

# Matplotlib's noninteractive backend must be selected before importing pyplot.
# pylint: disable=wrong-import-position

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from morana import CrossSections, FissionData, SeparableFission
from studies.finite_volume_performance import workload

DEFAULT_OUTPUT_DIRECTORY = Path(
    "artifacts/studies/finite_volume_performance/cross_sections"
)

MATERIAL_LABELS = {
    workload.REFERENCE_MATERIAL: "Reference",
    workload.HIGH_LEAKAGE_MATERIAL: "High leakage",
    workload.HIGH_REACTIVITY_MATERIAL: "High reactivity",
    workload.FUEL_REMOVAL_MATERIAL: "Fuel removal",
    workload.LOCALIZED_ABSORBER_MATERIAL: "Localized absorber",
}


def _materials(groups: int) -> dict[str, CrossSections]:
    """Build every synthetic material at one supported group count."""
    return {
        name: workload.build_cross_sections(groups, name)
        for name in workload.SYNTHETIC_MATERIAL_NAMES
    }


def _fission(cross_sections: CrossSections) -> FissionData:
    """Return fission data, enforcing the synthetic-family invariant."""
    if cross_sections.fission is None:
        raise RuntimeError("synthetic cross sections unexpectedly lack fission data")
    return cross_sections.fission


def _separable_fission(cross_sections: CrossSections) -> SeparableFission:
    """Return the synthetic family's compact fission representation."""
    neutron_production = _fission(cross_sections).neutron_production
    if not isinstance(neutron_production, SeparableFission):
        raise RuntimeError("synthetic fission data are unexpectedly nonseparable")
    return neutron_production


def _save_line_plot(
    materials: dict[str, CrossSections],
    groups: int,
    output_directory: Path,
    *,
    filename_component: str,
    value: Callable[[CrossSections], np.ndarray],
    title: str,
    ylabel: str,
) -> Path:
    """Plot one groupwise quantity for every synthetic material."""
    figure, axis = plt.subplots(figsize=(9.0, 5.25), constrained_layout=True)
    group_numbers = np.arange(1, groups + 1)
    for name, cross_sections in materials.items():
        axis.plot(
            group_numbers,
            value(cross_sections),
            marker="o" if groups <= 18 else None,
            markersize=3.5,
            linewidth=1.8,
            label=MATERIAL_LABELS[name],
        )
    axis.set(
        xlabel="Energy group (fast to thermal)",
        ylabel=ylabel,
        title=f"{title} ({groups} groups)",
        xlim=(1, groups),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    path = output_directory / f"g{groups}_{filename_component}.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _save_matrix_plot(
    materials: dict[str, CrossSections],
    groups: int,
    output_directory: Path,
    *,
    filename_component: str,
    value: Callable[[CrossSections], np.ndarray],
    title: str,
    colorbar_label: str,
) -> Path:
    """Plot one transfer matrix per material using one logarithmic scale."""
    matrices = {name: value(xs) for name, xs in materials.items()}
    positive_values = np.concatenate(
        [matrix[matrix > 0.0] for matrix in matrices.values()]
    )
    normalization = LogNorm(
        vmin=float(np.min(positive_values)),
        vmax=float(np.max(positive_values)),
    )
    color_map = matplotlib.colormaps["viridis"].with_extremes(bad="white")

    figure, axes = plt.subplots(
        1,
        len(materials),
        figsize=(17.0, 4.3),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for axis, (name, matrix) in zip(axes, matrices.items(), strict=True):
        image = axis.imshow(
            np.ma.masked_less_equal(matrix, 0.0),
            cmap=color_map,
            norm=normalization,
            origin="upper",
            extent=(0.5, groups + 0.5, groups + 0.5, 0.5),
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(MATERIAL_LABELS[name])
        axis.set_xlabel("Outgoing group")
    axes[0].set_ylabel("Incident group")
    figure.suptitle(f"{title} ({groups} groups; fast to thermal)")
    if image is None:  # pragma: no cover - the material family is nonempty
        raise RuntimeError("no synthetic material matrices were plotted")
    figure.colorbar(image, ax=axes, label=colorbar_label, shrink=0.78)
    path = output_directory / f"g{groups}_{filename_component}.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_group_count(groups: int, output_directory: Path) -> tuple[Path, ...]:
    """Create all cross-section comparison plots for one group count."""
    materials = _materials(groups)
    return (
        _save_line_plot(
            materials,
            groups,
            output_directory,
            filename_component="diffusion",
            value=lambda cross_sections: cross_sections.D,
            title="Synthetic diffusion coefficient",
            ylabel="Diffusion coefficient [cm]",
        ),
        _save_line_plot(
            materials,
            groups,
            output_directory,
            filename_component="absorption",
            value=lambda cross_sections: cross_sections.sigma_a,
            title="Synthetic absorption cross section",
            ylabel=r"$\Sigma_a$ [1 / cm]",
        ),
        _save_line_plot(
            materials,
            groups,
            output_directory,
            filename_component="fission_production",
            value=lambda cross_sections: _fission(cross_sections).fission_production,
            title="Synthetic fission-neutron production cross section",
            ylabel=r"$\nu\Sigma_f$ [1 / cm]",
        ),
        _save_line_plot(
            materials,
            groups,
            output_directory,
            filename_component="fission_spectrum",
            value=lambda cross_sections: _separable_fission(cross_sections).chi,
            title="Synthetic fission emission spectrum",
            ylabel=r"$\chi$",
        ),
        _save_matrix_plot(
            materials,
            groups,
            output_directory,
            filename_component="scattering",
            value=lambda cross_sections: cross_sections.sigma_s,
            title="Synthetic scattering cross section",
            colorbar_label=r"$\Sigma_s(g \rightarrow g')$ [1 / cm]",
        ),
        _save_matrix_plot(
            materials,
            groups,
            output_directory,
            filename_component="fission_transfer",
            value=lambda cross_sections: _fission(cross_sections).fission_transfer,
            title="Synthetic fission-neutron transfer cross section",
            colorbar_label=r"$\nu\Sigma_f(g \rightarrow g')$ [1 / cm]",
        ),
    )


def generate_plots(groups: Sequence[int], output_directory: Path) -> tuple[Path, ...]:
    """Write all selected cross-section plots and return their paths."""
    output_directory.mkdir(parents=True, exist_ok=True)
    return tuple(
        path
        for group_count in groups
        for path in plot_group_count(group_count, output_directory)
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups",
        nargs="+",
        type=int,
        choices=workload.GROUP_COUNTS,
        default=workload.GROUP_COUNTS,
        metavar="GROUPS",
        help="supported group counts to plot (default: all)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=f"PNG directory (default: {DEFAULT_OUTPUT_DIRECTORY})",
    )
    return parser.parse_args()


def main() -> None:
    """Generate the selected PNG comparisons."""
    arguments = _parse_arguments()
    paths = generate_plots(arguments.groups, arguments.output_directory)
    print(f"Wrote {len(paths)} plots to {arguments.output_directory.resolve()}")


if __name__ == "__main__":
    main()
