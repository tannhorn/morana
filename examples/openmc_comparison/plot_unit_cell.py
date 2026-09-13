"""Render the comparison unit-cell geometry with Matplotlib."""

# Matplotlib configuration must precede its imports.
# pylint: disable=wrong-import-order,wrong-import-position

from __future__ import annotations

import argparse
from math import hypot
from pathlib import Path

from examples.openmc_comparison.artifact_paths import (
    ARTIFACT_ROOT,
    configure_matplotlib_cache,
)
from examples.openmc_comparison.geometry import (
    FUEL_DIAMETER_IN,
    FUEL_ELEMENT_PITCH_IN,
    FUEL_ROD_OUTER_DIAMETER_IN,
    FUEL_CAN_ORIENTATION,
    INTER_CAN_SODIUM_GAP_IN,
    NAK_BOND_THICKNESS_IN,
    PROCESS_CHANNEL_INNER_DIAMETER_IN,
    PROCESS_CHANNEL_WALL_THICKNESS_IN,
    STEEL_TUBE_THICKNESS_IN,
    WIRE_DIAMETER_IN,
    ZR_CAN_WALL_IN,
    can_across_flats_cm,
    graphite_across_flats_cm,
    hex_edge_from_across_flats,
    inches_to_cm,
    lattice_pitch_cm,
    point_up_hex_vertices,
    rod_centers,
    wire_to_channel_wall_clearance_in,
    wire_centers,
)
from examples.openmc_comparison.plotting import material_color_hex

DEFAULT_OUTPUT = ARTIFACT_ROOT / "unit_cell_matplotlib.png"
configure_matplotlib_cache()

# Matplotlib must read this script-local configuration before it is imported.
import matplotlib  # pylint: disable=wrong-import-position

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
from matplotlib.patches import (  # pylint: disable=wrong-import-position
    Circle,
    Polygon,
    Rectangle,
)

_COLORS = {
    "fuel": material_color_hex("fuel"),
    "nak": material_color_hex("nak"),
    "steel": material_color_hex("ss304"),
    "sodium": material_color_hex("sodium"),
    "graphite": material_color_hex("graphite"),
    "zirconium": material_color_hex("zirconium"),
}
_DARK_EDGE = "#172026"
_HEX_EDGE = "#243036"
_SODIUM_EDGE = "#23576f"


def _hex_vertices(across_flats_cm: float) -> tuple[tuple[float, float], ...]:
    """Return vertices matching the case's point-up OpenMC hex orientation."""

    if FUEL_CAN_ORIENTATION != "y":
        raise ValueError("This drawing expects the case's orientation='y' fuel can.")
    return point_up_hex_vertices(across_flats_cm)


def _add_hexagons(axes: plt.Axes) -> None:
    """Draw the complete 11-inch lattice cell's can, graphite, and sodium."""

    for across_flats, color, edge_color, linewidth in (
        (lattice_pitch_cm(), _COLORS["sodium"], _SODIUM_EDGE, 1.2),
        (can_across_flats_cm(), _COLORS["zirconium"], _HEX_EDGE, 2.0),
        (graphite_across_flats_cm(), _COLORS["graphite"], _HEX_EDGE, 1.2),
    ):
        axes.add_patch(
            Polygon(
                _hex_vertices(across_flats),
                closed=True,
                facecolor=color,
                edgecolor=edge_color,
                linewidth=linewidth,
            )
        )


def _add_bundle(axes: plt.Axes) -> None:
    """Draw the case's process channel, rods, and wire surrogate."""

    fuel_radius = inches_to_cm(FUEL_DIAMETER_IN / 2.0)
    nak_radius = fuel_radius + inches_to_cm(NAK_BOND_THICKNESS_IN)
    tube_radius = inches_to_cm(FUEL_ROD_OUTER_DIAMETER_IN / 2.0)
    wire_radius = inches_to_cm(WIRE_DIAMETER_IN / 2.0)
    channel_inner_radius = inches_to_cm(PROCESS_CHANNEL_INNER_DIAMETER_IN / 2.0)
    channel_outer_radius = channel_inner_radius + inches_to_cm(
        PROCESS_CHANNEL_WALL_THICKNESS_IN
    )

    axes.add_patch(
        Circle(
            (0.0, 0.0),
            channel_outer_radius,
            facecolor=_COLORS["zirconium"],
            edgecolor=_HEX_EDGE,
            linewidth=1.0,
        )
    )
    axes.add_patch(
        Circle(
            (0.0, 0.0),
            channel_inner_radius,
            facecolor=_COLORS["sodium"],
            edgecolor=_SODIUM_EDGE,
            linewidth=1.0,
        )
    )
    for x_coord, y_coord in rod_centers():
        axes.add_patch(
            Circle(
                (x_coord, y_coord),
                tube_radius,
                facecolor=_COLORS["steel"],
                edgecolor=_DARK_EDGE,
                linewidth=0.9,
            )
        )
        axes.add_patch(
            Circle(
                (x_coord, y_coord),
                nak_radius,
                facecolor=_COLORS["nak"],
                edgecolor=_DARK_EDGE,
                linewidth=0.7,
            )
        )
        axes.add_patch(
            Circle(
                (x_coord, y_coord),
                fuel_radius,
                facecolor=_COLORS["fuel"],
                edgecolor=_DARK_EDGE,
                linewidth=0.7,
            )
        )
    for x_coord, y_coord in wire_centers():
        axes.add_patch(
            Circle(
                (x_coord, y_coord),
                wire_radius,
                facecolor=_COLORS["steel"],
                edgecolor=_DARK_EDGE,
                linewidth=0.6,
            )
        )


def _configure_axes(axes: plt.Axes) -> None:
    """Apply the shared Cartesian appearance for each panel."""

    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x [cm]")
    axes.set_ylabel("y [cm]")
    axes.grid(True, color="#e6e6e6", linewidth=0.6)


def _legend_handles() -> list[Circle | Polygon]:
    """Return material handles matching the visualized case regions."""

    return [
        Circle(
            (0, 0),
            1,
            facecolor=_COLORS["fuel"],
            edgecolor=_DARK_EDGE,
            label="U metal fuel",
        ),
        Circle(
            (0, 0),
            1,
            facecolor=_COLORS["nak"],
            edgecolor=_DARK_EDGE,
            label="NaK bond",
        ),
        Circle(
            (0, 0),
            1,
            facecolor=_COLORS["steel"],
            edgecolor=_DARK_EDGE,
            label="SS-304",
        ),
        Circle(
            (0, 0),
            1,
            facecolor=_COLORS["sodium"],
            edgecolor=_SODIUM_EDGE,
            label="Sodium coolant",
        ),
        Polygon(
            [(0, 0), (1, 0), (0, 1)],
            facecolor=_COLORS["graphite"],
            edgecolor=_HEX_EDGE,
            label="Graphite",
        ),
        Polygon(
            [(0, 0), (1, 0), (0, 1)],
            facecolor=_COLORS["zirconium"],
            edgecolor=_HEX_EDGE,
            label="Zirconium",
        ),
    ]


def plot_unit_cell(path: Path, *, transparent: bool = False) -> Path:
    """Create the labeled two-panel unit-cell geometry figure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, (full_axes, zoom_axes) = plt.subplots(1, 2, figsize=(13, 7))
    figure.subplots_adjust(
        bottom=0.10,
        left=0.06,
        right=0.84,
        top=0.88,
        wspace=0.13,
    )

    zoom_limit = inches_to_cm(2.1)
    zoom_axes.add_patch(
        Rectangle(
            (-zoom_limit, -zoom_limit),
            2.0 * zoom_limit,
            2.0 * zoom_limit,
            facecolor=_COLORS["graphite"],
            edgecolor="none",
            zorder=-10,
        )
    )
    _add_hexagons(full_axes)
    _add_bundle(full_axes)
    _add_bundle(zoom_axes)
    for axes in (full_axes, zoom_axes):
        _configure_axes(axes)

    full_limit = 1.3 * hex_edge_from_across_flats(lattice_pitch_cm())
    full_axes.set(xlim=(-full_limit, full_limit), ylim=(-full_limit, full_limit))
    full_axes.set_title(f"Full {FUEL_ELEMENT_PITCH_IN:g} in-pitch unit cell")
    channel_inner_radius = inches_to_cm(PROCESS_CHANNEL_INNER_DIAMETER_IN / 2.0)
    channel_outer_radius = channel_inner_radius + inches_to_cm(
        PROCESS_CHANNEL_WALL_THICKNESS_IN
    )
    inter_can_sodium_target = (
        0.0,
        (
            hex_edge_from_across_flats(can_across_flats_cm())
            + hex_edge_from_across_flats(lattice_pitch_cm())
        )
        / 2.0,
    )
    full_axes.annotate(
        f"{INTER_CAN_SODIUM_GAP_IN / 2.0:g} in apportioned\ninter-can Na",
        xytext=(-inches_to_cm(4.8), inches_to_cm(6.5)),
        xy=inter_can_sodium_target,
        fontsize=9,
    )
    full_axes.annotate(
        "",
        xy=inter_can_sodium_target,
        xytext=(-inches_to_cm(1.9), inches_to_cm(6.65)),
        arrowprops={"arrowstyle": "->", "linewidth": 1.0},
    )
    graphite_half_width = graphite_across_flats_cm() / 2.0
    graphite_dimension_y = -inches_to_cm(2.7)
    full_axes.annotate(
        "",
        xy=(-graphite_half_width, graphite_dimension_y),
        xytext=(graphite_half_width, graphite_dimension_y),
        arrowprops={"arrowstyle": "<->", "linewidth": 1.0},
    )
    full_axes.text(
        0.0,
        graphite_dimension_y - inches_to_cm(0.35),
        (
            f"{graphite_across_flats_cm() / inches_to_cm(1.0):g} in "
            "graphite across flats"
        ),
        fontsize=9,
        horizontalalignment="center",
        verticalalignment="top",
    )
    can_wall_target = (
        (can_across_flats_cm() + graphite_across_flats_cm()) / 4.0,
        inches_to_cm(2.2),
    )
    full_axes.annotate(
        f"{ZR_CAN_WALL_IN:g} in zirconium\ncan wall",
        xy=can_wall_target,
        xytext=(inches_to_cm(6.2), inches_to_cm(3.8)),
        arrowprops={"arrowstyle": "->", "linewidth": 1.0},
        fontsize=9,
        horizontalalignment="center",
    )
    full_axes.annotate(
        f"{PROCESS_CHANNEL_WALL_THICKNESS_IN:g} in zirconium\n" "process-channel wall",
        xy=(-(channel_inner_radius + channel_outer_radius) / 2.0, 0.0),
        xytext=(-inches_to_cm(4.7), inches_to_cm(2.0)),
        arrowprops={"arrowstyle": "->", "linewidth": 1.0},
        fontsize=9,
    )

    zoom_axes.set(xlim=(-zoom_limit, zoom_limit), ylim=(-zoom_limit, zoom_limit))
    zoom_axes.set_title("Seven-rod fuel bundle detail")
    fuel_radius = inches_to_cm(FUEL_DIAMETER_IN / 2.0)
    tube_radius = inches_to_cm(FUEL_ROD_OUTER_DIAMETER_IN / 2.0)
    wire_radius = inches_to_cm(WIRE_DIAMETER_IN / 2.0)
    outer_wire_x, outer_wire_y = wire_centers()[9]
    outer_wire_distance = hypot(outer_wire_x, outer_wire_y)
    clearance_target_distance = outer_wire_distance + wire_radius
    wire_to_channel_wall_clearance = (
        outer_wire_x * clearance_target_distance / outer_wire_distance,
        outer_wire_y * clearance_target_distance / outer_wire_distance,
    )
    zoom_axes.annotate(
        f"{FUEL_DIAMETER_IN:g} in fuel diameter",
        xy=(fuel_radius, 0.0),
        xytext=(inches_to_cm(0.55), inches_to_cm(1.6)),
        arrowprops={"arrowstyle": "->", "linewidth": 1.0},
        fontsize=9,
    )
    zoom_axes.annotate(
        f"{NAK_BOND_THICKNESS_IN:g} in NaK +\n"
        f"{STEEL_TUBE_THICKNESS_IN:g} in SS-304",
        xy=(tube_radius, 0.0),
        xytext=(inches_to_cm(1.65), -inches_to_cm(1.7)),
        arrowprops={"arrowstyle": "->", "linewidth": 1.0},
        fontsize=9,
        horizontalalignment="center",
    )
    zoom_axes.annotate(
        f"{PROCESS_CHANNEL_INNER_DIAMETER_IN:g} in Na process channel ID",
        xy=(0.0, channel_inner_radius),
        xytext=(-inches_to_cm(1.9), inches_to_cm(1.85)),
        arrowprops={"arrowstyle": "->", "linewidth": 1.0},
        fontsize=9,
    )
    zoom_axes.text(
        -inches_to_cm(2.0),
        -inches_to_cm(1.8),
        f"{WIRE_DIAMETER_IN:g} in SS-304 wire\n"
        f"{wire_to_channel_wall_clearance_in():g} in wire-to-channel-wall clearance",
        fontsize=9,
    )
    zoom_axes.annotate(
        "",
        xy=wire_to_channel_wall_clearance,
        xytext=(-inches_to_cm(0.75), -inches_to_cm(1.66)),
        arrowprops={
            "arrowstyle": "->",
            "connectionstyle": "arc3,rad=0.08",
            "linewidth": 1.0,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
    )
    figure.suptitle(
        "Transverse geometry of the SRE-derived fuel bundle and graphite can",
        y=0.94,
    )
    zoom_axes.legend(
        handles=_legend_handles(),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
    )
    figure.savefig(path, dpi=220, transparent=transparent)
    plt.close(figure)
    return path


def main() -> None:
    """Parse the output path and write the geometry figure."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = plot_unit_cell(args.output.expanduser().resolve())
    print("Wrote Matplotlib unit-cell figure:", output)


if __name__ == "__main__":
    main()
