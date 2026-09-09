"""Generate the partial-current/affine-Robin parameter figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "assets"
    / "boundary_parameter_relations.png"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output PNG path (default: docs/assets/boundary_parameter_relations.png)",
    )
    return parser.parse_args()


def generate(output: Path) -> None:
    """Write the boundary-parameter relation figure to *output*."""
    beta = np.linspace(0.0, 1.0, 501)
    alpha = (1.0 - beta) / (2.0 * (1.0 + beta))
    source_multiplier = 2.0 / (1.0 + beta)

    mpl.rcParams.update(
        {
            "axes.edgecolor": "#34495e",
            "axes.labelcolor": "#17202a",
            "axes.titlecolor": "#17202a",
            "font.size": 10.5,
            "xtick.color": "#34495e",
            "ytick.color": "#34495e",
        }
    )

    figure, (alpha_axis, source_axis) = plt.subplots(
        2,
        1,
        figsize=(8.2, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": (1.08, 1.0), "hspace": 0.18},
    )
    figure.patch.set_alpha(0.0)
    figure.suptitle(
        "Partial-current return and affine Robin parameters",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )

    for axis in (alpha_axis, source_axis):
        axis.patch.set_alpha(0.0)
        axis.set_xlim(-0.02, 1.02)
        axis.grid(color="#d5d8dc", linewidth=0.8, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.axvline(0.0, color="#7f8c8d", linestyle="--", linewidth=1.0)
        axis.axvline(1.0, color="#7f8c8d", linestyle="--", linewidth=1.0)

    alpha_color = "#16697a"
    source_color = "#b24c2f"

    alpha_axis.plot(beta, alpha, color=alpha_color, linewidth=2.8)
    alpha_axis.scatter(
        [0.0, 1.0], [0.5, 0.0], color=alpha_color, edgecolor="white", zorder=3
    )
    alpha_axis.set_ylim(-0.035, 0.57)
    alpha_axis.set_ylabel(r"Robin response $\alpha$")
    alpha_axis.set_title(
        r"Flux-dependent response: $\alpha=(1-\beta)/[2(1+\beta)]$",
        loc="left",
        fontsize=11.5,
        pad=9,
    )
    alpha_axis.text(
        0.035,
        0.475,
        "No return\n"
        r"$\beta=0,\ \alpha=1/2$"
        "\n"
        r"$q_{\mathrm{in}}=0$: Marshak vacuum",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": alpha_color},
        va="top",
    )
    alpha_axis.text(
        0.965,
        0.055,
        "Perfect return\n"
        r"$\beta=1,\ \alpha=0$"
        "\n"
        r"$q_{\mathrm{in}}=0$: reflective",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": alpha_color},
        ha="right",
        va="bottom",
    )
    alpha_axis.text(
        0.5,
        0.535,
        r"Zero-flux Dirichlet: $\alpha\to\infty$ "
        r"(outside the passive $0\leq\beta\leq1$ interval)",
        ha="center",
        va="center",
        color="#566573",
        fontsize=9.5,
    )

    source_axis.plot(beta, source_multiplier, color=source_color, linewidth=2.8)
    source_axis.scatter(
        [0.0, 1.0], [2.0, 1.0], color=source_color, edgecolor="white", zorder=3
    )
    source_axis.set_ylim(0.82, 2.16)
    source_axis.set_xlabel(r"Partial-current return ratio $\beta$")
    source_axis.set_ylabel(r"Source multiplier $s/q_{\mathrm{in}}$")
    source_axis.set_title(
        r"Imposed-current scaling for $q_{\mathrm{in}}>0$: "
        r"$s/q_{\mathrm{in}}=2/(1+\beta)$",
        loc="left",
        fontsize=11.5,
        pad=9,
    )
    source_axis.text(
        0.035,
        1.96,
        "No returned current\n"
        r"$\beta=0,\ s=2q_{\mathrm{in}}$"
        "\n"
        "pure prescribed incoming current",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": source_color},
        va="top",
    )
    source_axis.text(
        0.965,
        1.055,
        "Perfect return plus incident source\n" r"$\beta=1,\ s=q_{\mathrm{in}}$",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": source_color},
        ha="right",
        va="bottom",
    )
    source_axis.text(
        0.5,
        0.89,
        r"For a homogeneous boundary, $q_{\mathrm{in}}=0$ and therefore "
        r"$s=0$ for every $\beta$.",
        ha="center",
        va="center",
        color="#566573",
        fontsize=9.5,
    )

    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.095, top=0.91)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() != ".png":
        raise ValueError("figure output must use a .png suffix")
    figure.savefig(
        output,
        format="png",
        dpi=180,
        facecolor="none",
    )
    plt.close(figure)


def main() -> None:
    """Generate the requested figure from command-line arguments."""
    generate(_arguments().output.resolve())


if __name__ == "__main__":
    main()
