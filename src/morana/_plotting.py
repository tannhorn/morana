"""Shared private helpers and conventions for hexagonal plots.

Excluded positions have no solver value and are represented by ``NaN`` in
full-lattice result fields. Result plots render those positions with
``EXCLUDED_FLUX_COLOR`` outside the scalar colormap.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from matplotlib.axes import Axes
from matplotlib.patches import Polygon
import plotly.graph_objects as go
from plotly.graph_objects import Figure

if TYPE_CHECKING:
    from morana.hex_planar_mesh import HexPlanarMesh

EXCLUDED_FLUX_COLOR = "#d9d9d9"
FLUX_COLORMAP = "inferno"
DEFAULT_MATERIAL_COLORS = (
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
)
PLOTLY_HEX_FIGURE_WIDTH = 900
PLOTLY_HEX_FIGURE_HEIGHT = 700
PLOTLY_HEX_FIGURE_MARGIN = {"l": 80, "r": 180, "t": 70, "b": 70}


def add_hex_patch(
    ax: Axes,
    vertices: Sequence[tuple[float, float]],
    facecolor: str | tuple[float, float, float, float],
) -> Polygon:
    """Add one consistently styled hexagonal patch to Matplotlib axes."""
    polygon = Polygon(
        vertices,
        closed=True,
        facecolor=facecolor,
        edgecolor="black",
        linewidth=1.0,
    )
    ax.add_patch(polygon)
    return polygon


def add_planar_matplotlib_cells(
    mesh: HexPlanarMesh,
    ax: Axes,
    facecolor_for: Callable[[int], str | tuple[float, float, float, float]],
    label_for: Callable[[int], str] | None = None,
) -> None:
    """Add complete-lattice hexagons and optional centered labels.

    The caller owns colors and labels; ``mesh`` supplies the regular planar
    geometry indexed by planar ID.
    """
    for planar_id in range(mesh.n_cells):
        add_hex_patch(
            ax,
            mesh.cell_vertices(planar_id),
            facecolor=facecolor_for(planar_id),
        )
        if label_for is not None:
            center_x, center_y = mesh.cartesian_center(planar_id)
            ax.text(
                center_x,
                center_y,
                label_for(planar_id),
                ha="center",
                va="center",
                fontsize=7,
            )


def configure_hex_axes(ax: Axes) -> None:
    """Apply common Cartesian geometry settings to Matplotlib hex plots."""
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")


def hex_fill_trace(
    cells: Sequence[Sequence[tuple[float, float]]],
    fillcolor: str,
    *,
    legendgroup: str | None = None,
    name: str = "",
    showlegend: bool = False,
) -> go.Scatter:
    """Return one non-hoverable Plotly trace containing styled hexagons."""
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    for vertices in cells:
        closed_vertices = tuple(vertices) + (vertices[0],)
        x_values.extend(x_value for x_value, _ in closed_vertices)
        y_values.extend(y_value for _, y_value in closed_vertices)
        x_values.append(None)
        y_values.append(None)
    return go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines",
        fill="toself",
        fillcolor=fillcolor,
        line={"color": "black", "width": 1},
        hoverinfo="skip",
        name=name,
        legendgroup=legendgroup,
        showlegend=showlegend,
    )


def add_planar_plotly_cells(
    mesh: HexPlanarMesh,
    figure: Figure,
    fillcolor_for: Callable[[int], str],
    legendgroup_for: Callable[[int], str | None] | None = None,
) -> None:
    """Add complete-lattice hexagons batched by visible Plotly style.

    The caller owns colors and legend grouping; ``mesh`` supplies the regular
    planar geometry indexed by planar ID. Hover targets are added separately by
    ``add_planar_plotly_hover_targets`` because Plotly cannot fill-hover
    disconnected polygons in one trace.
    """
    cells_by_style: dict[
        tuple[str, str | None],
        list[Sequence[tuple[float, float]]],
    ] = {}
    for planar_id in range(mesh.n_cells):
        legendgroup = (
            legendgroup_for(planar_id) if legendgroup_for is not None else None
        )
        fillcolor = fillcolor_for(planar_id)
        cells_by_style.setdefault((fillcolor, legendgroup), []).append(
            mesh.cell_vertices(planar_id)
        )

    for (fillcolor, legendgroup), cells in cells_by_style.items():
        figure.add_trace(
            hex_fill_trace(
                cells,
                fillcolor,
                legendgroup=legendgroup,
                name=legendgroup or "",
                showlegend=legendgroup is not None,
            )
        )


def add_planar_plotly_hover_targets(
    mesh: HexPlanarMesh,
    figure: Figure,
    hover_text_for: Callable[[int], str],
    legendgroup_for: Callable[[int], str | None] | None = None,
) -> None:
    """Add transparent hex-marker hover targets, optionally by legend group."""
    planar_ids_by_legend: dict[str | None, list[int]] = {}
    for planar_id in range(mesh.n_cells):
        legendgroup = (
            legendgroup_for(planar_id) if legendgroup_for is not None else None
        )
        planar_ids_by_legend.setdefault(legendgroup, []).append(planar_id)

    marker_size = _plotly_hex_marker_size(mesh)
    for legendgroup, planar_ids in planar_ids_by_legend.items():
        centers = [mesh.cartesian_center(planar_id) for planar_id in planar_ids]
        figure.add_trace(
            go.Scatter(
                x=[center_x for center_x, _ in centers],
                y=[center_y for _, center_y in centers],
                mode="markers",
                marker={
                    "symbol": "hexagon",
                    "size": marker_size,
                    "color": "rgba(0,0,0,0)",
                    "line": {"width": 0},
                },
                customdata=[hover_text_for(planar_id) for planar_id in planar_ids],
                hovertemplate="%{customdata}<extra></extra>",
                legendgroup=legendgroup,
                showlegend=False,
            )
        )


def configure_hex_figure(figure: Figure, mesh: HexPlanarMesh, title: str) -> None:
    """Apply common Cartesian geometry and hover-target settings to a figure."""
    x_min, x_max, y_min, y_max = _plotly_hex_bounds(mesh)
    figure.update_layout(
        title=title,
        width=PLOTLY_HEX_FIGURE_WIDTH,
        height=PLOTLY_HEX_FIGURE_HEIGHT,
        margin=PLOTLY_HEX_FIGURE_MARGIN,
        hovermode="closest",
        xaxis={
            "title": "x [cm]",
            "range": [x_min, x_max],
            "scaleanchor": "y",
            "scaleratio": 1,
        },
        yaxis={"title": "y [cm]", "range": [y_min, y_max]},
        legend={"groupclick": "togglegroup"},
        template="plotly_white",
    )


def _plotly_hex_bounds(mesh: HexPlanarMesh) -> tuple[float, float, float, float]:
    """Return symmetric one-half-pitch-padded Cartesian plot bounds."""
    vertices = [
        vertex
        for planar_id in range(mesh.n_cells)
        for vertex in mesh.cell_vertices(planar_id)
    ]
    x_values, y_values = zip(*vertices, strict=True)
    return (
        min(x_values) - mesh.pitch / 2.0,
        max(x_values) + mesh.pitch / 2.0,
        min(y_values) - mesh.pitch / 2.0,
        max(y_values) + mesh.pitch / 2.0,
    )


def _plotly_hex_marker_size(mesh: HexPlanarMesh) -> float:
    """Return the fixed-canvas pixel diameter for one transparent hex marker."""
    x_min, x_max, _, _ = _plotly_hex_bounds(mesh)
    plot_width = (
        PLOTLY_HEX_FIGURE_WIDTH
        - PLOTLY_HEX_FIGURE_MARGIN["l"]
        - PLOTLY_HEX_FIGURE_MARGIN["r"]
    )
    return mesh.pitch * plot_width / (x_max - x_min)
