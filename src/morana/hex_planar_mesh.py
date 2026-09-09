"""Hexagonal mesh topology and geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi, sin, sqrt
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import plotly.graph_objects as go
from plotly.graph_objects import Figure

from morana._validation import (
    require_finite_positive_real,
    require_integer,
    require_nonnegative_integer,
    require_positive_integer,
)
from morana._plotting import (
    add_planar_matplotlib_cells,
    add_planar_plotly_cells,
    add_planar_plotly_hover_targets,
    configure_hex_axes,
    configure_hex_figure,
)
from morana._vtk import VTKDataArray, prepare_output_path, write_vtu

_X_PLUS = (1, 0)
_X_MINUS = (-1, 0)
_U_PLUS = (0, 1)
_U_MINUS = (0, -1)
_V_PLUS = (-1, 1)
_V_MINUS = (1, -1)

HEX_DIRECTIONS: tuple[tuple[str, tuple[int, int]], ...] = (
    ("x+", _X_PLUS),
    ("x-", _X_MINUS),
    ("u+", _U_PLUS),
    ("u-", _U_MINUS),
    ("v+", _V_PLUS),
    ("v-", _V_MINUS),
)
HEX_DIRECTION_LABELS = tuple(label for label, _ in HEX_DIRECTIONS)

_RING_WALK_DIRECTIONS: tuple[tuple[int, int], ...] = (
    _U_MINUS,
    _X_MINUS,
    _V_PLUS,
    _U_PLUS,
    _X_PLUS,
    _V_MINUS,
)


@dataclass(frozen=True, order=True)
class OpenMCIndex:
    """Identify one serialized position in an OpenMC-style hexagonal ring.

    Parameters
    ----------
    ring
        Zero-based serialized ring index. For a ``HexPlanarMesh``, zero is the
        outermost ring and increasing values move inward.
    position
        Zero-based position within ``ring``. Positions start at the positive
        ``x`` direction and proceed clockwise in Morana's supported OpenMC
        ``orientation="x"`` convention.

    Raises
    ------
    TypeError
        If ``ring`` or ``position`` is not an integer or is a boolean.
    ValueError
        If ``ring`` or ``position`` is negative.

    Notes
    -----
    This is an immutable, hashable value object. Its dataclass ordering is
    lexicographic by ``(ring, position)``, which matches a mesh's outer-to-
    inner serialized ring order but is not the same as spatial ordering around
    the lattice.

    ``OpenMCIndex`` validates that both fields are nonnegative integers, but
    membership remains mesh-relative. For a mesh with ``n`` rings, a valid
    ring is ``0 <= ring < n`` and its valid positions are
    ``0 <= position < max(6 * (n - 1 - ring), 1)``. ``HexPlanarMesh`` creates
    its complete valid sequence as ``openmc_indices``;
    ``HexPlanarMesh.planar_id_at()`` returns ``None`` for an index outside that
    sequence, and ``MaterialSlice`` rejects such indices as sparse keys.

    The type records a ring-position identity only. It does not require
    OpenMC at runtime and does not identify a compact material-mesh
    ``active_id``; use the latter only with an explicit axial layer.
    """

    ring: int
    position: int

    def __post_init__(self) -> None:
        """Check and normalize one mesh-relative ring-position identity."""
        ring = require_nonnegative_integer("ring", self.ring)
        position = require_nonnegative_integer("position", self.position)
        object.__setattr__(self, "ring", ring)
        object.__setattr__(self, "position", position)


@dataclass(frozen=True)
class HexPlanarMesh:
    """Represent a regular 2D hexagonal mesh.

    Parameters
    ----------
    num_rings
        Positive integer OpenMC-style ring count. ``1`` is center-only, ``2``
        has 7 cells, and ``3`` has 19 cells. A mesh with ``n`` rings has
        ``1 + 3 * n * (n - 1)`` full-lattice cells.
    pitch
        Finite positive flat-to-flat hexagon pitch in cm.

    Attributes
    ----------
    n_cells
        Number of positions in the complete regular lattice.
    coords
        Integer ``(x, u)`` lattice coordinates in planar-ID order.
    openmc_indices
        ``OpenMCIndex`` values in the same planar-ID order.
    direction_labels
        Radial neighbor directions in the fixed order ``("x+", "x-", "u+",
        "u-", "v+", "v-")``.
    area, face_length, center_distance, center_to_face
        Per-cell planar area in ``cm^2`` and geometric lengths in cm derived
        from ``pitch``.
    index_of
        Newly constructed mapping from a lattice coordinate to its planar ID.

    Raises
    ------
    TypeError
        If ``num_rings`` is not an integer or is a boolean, or if ``pitch``
        is not a real non-Boolean number.
    ValueError
        If ``num_rings`` is less than one, or ``pitch`` is not finite and
        positive.

    Notes
    -----
    The mesh is a full regular lattice. OpenMC-style ring indices and integer
    lattice coordinates are constructed internally from ``num_rings``
    using the locked OpenMC ``orientation="x"`` ring-order convention. The
    coordinate basis is aligned with Morana's ``x`` and ``u`` face directions:
    ``x+`` points right and ``u+`` points upper-right.

    Planar IDs enumerate rings outermost to innermost; positions within a ring
    start at ``x+`` and proceed clockwise. ``planar_id_at()`` is the safe
    reverse lookup and returns ``None`` for an index outside this mesh.
    ``neighbors()`` returns full-lattice planar IDs in ``direction_labels``
    order and uses ``None`` for a physical lattice perimeter. The mesh has no
    material, active-domain, or axial information; ``MaterialMesh`` supplies
    those layers of the problem definition.
    """

    num_rings: int
    pitch: float
    _coords: tuple[tuple[int, int], ...] = field(init=False, repr=False, compare=False)
    _openmc_indices: tuple[OpenMCIndex, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _planar_id_by_coord: Mapping[tuple[int, int], int] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _planar_id_by_openmc_index: Mapping[OpenMCIndex, int] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _neighbors: tuple[tuple[int | None, ...], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Check dimensions and construct immutable mesh topology."""
        num_rings = require_positive_integer("num_rings", self.num_rings)
        pitch = require_finite_positive_real("pitch", self.pitch)
        object.__setattr__(self, "num_rings", num_rings)
        object.__setattr__(self, "pitch", pitch)

        coords = tuple(
            coord
            for radius in range(self.num_rings - 1, -1, -1)
            for coord in _ring_coords(radius)
        )
        openmc_indices = tuple(
            OpenMCIndex(serialized_ring, position)
            for serialized_ring, radius in enumerate(range(self.num_rings - 1, -1, -1))
            for position in range(max(6 * radius, 1))
        )
        planar_id_by_coord = {
            coord: planar_id for planar_id, coord in enumerate(coords)
        }
        planar_id_by_openmc_index = {
            index: planar_id for planar_id, index in enumerate(openmc_indices)
        }
        neighbors = tuple(
            tuple(
                planar_id_by_coord.get((x_coord + dx, u_coord + du))
                for _, (dx, du) in HEX_DIRECTIONS
            )
            for x_coord, u_coord in coords
        )
        object.__setattr__(self, "_coords", coords)
        object.__setattr__(self, "_openmc_indices", openmc_indices)
        object.__setattr__(
            self,
            "_planar_id_by_coord",
            MappingProxyType(planar_id_by_coord),
        )
        object.__setattr__(
            self,
            "_planar_id_by_openmc_index",
            MappingProxyType(planar_id_by_openmc_index),
        )
        object.__setattr__(self, "_neighbors", neighbors)

    @property
    def n_cells(self) -> int:
        """Return the number of positions in the complete regular lattice."""
        return len(self._coords)

    @property
    def direction_labels(self) -> tuple[str, ...]:
        """Return the mesh-owned face-direction labels in neighbor order."""
        return HEX_DIRECTION_LABELS

    @property
    def coords(self) -> tuple[tuple[int, int], ...]:
        """Return lattice ``(x, u)`` coordinates in outer-to-inner ID order."""
        return self._coords

    @property
    def openmc_indices(self) -> tuple[OpenMCIndex, ...]:
        """Return orientation-x OpenMC ring indices in planar-ID order."""
        return self._openmc_indices

    @property
    def area(self) -> float:
        """Return the area of a regular 2D hex cell in ``cm^2``."""
        return sqrt(3.0) / 2.0 * self.pitch**2

    @property
    def face_length(self) -> float:
        """Return the length of one hex face in cm."""
        return self.pitch / sqrt(3.0)

    @property
    def center_distance(self) -> float:
        """Return the center-to-center distance across a face in cm."""
        return self.pitch

    @property
    def center_to_face(self) -> float:
        """Return the perpendicular center-to-face distance in cm."""
        return self.pitch / 2.0

    @property
    def index_of(self) -> Mapping[tuple[int, int], int]:
        """Return the immutable mapping from lattice coordinate to planar ID."""
        return self._planar_id_by_coord

    def lattice_coord(self, planar_id: int) -> tuple[int, int]:
        """Return the ``(x, u)`` lattice coordinate for a planar ID.

        Raises
        ------
        TypeError
            If ``planar_id`` is not an integer or is a boolean.
        IndexError
            If ``planar_id`` is outside the complete lattice.
        """
        planar_id = self._check_planar_id(planar_id)
        return self.coords[planar_id]

    def openmc_index(self, planar_id: int) -> OpenMCIndex:
        """Return the orientation-x OpenMC ring index for a planar ID.

        Raises
        ------
        TypeError
            If ``planar_id`` is not an integer or is a boolean.
        IndexError
            If ``planar_id`` is outside the complete lattice.
        """
        planar_id = self._check_planar_id(planar_id)
        return self.openmc_indices[planar_id]

    def planar_id_at(self, openmc_index: OpenMCIndex) -> int | None:
        """Return the planar ID at an OpenMC index, or ``None`` if absent.

        Raises
        ------
        TypeError
            If ``openmc_index`` is not an ``OpenMCIndex``.
        """
        if not isinstance(openmc_index, OpenMCIndex):
            raise TypeError("openmc_index must be an OpenMCIndex")
        return self._planar_id_by_openmc_index.get(openmc_index)

    def neighbors(self, planar_id: int) -> tuple[int | None, ...]:
        """Return neighbors in ``direction_labels`` order, using ``None`` outside.

        Raises
        ------
        TypeError
            If ``planar_id`` is not an integer or is a boolean.
        IndexError
            If ``planar_id`` is outside the complete lattice.
        """
        planar_id = self._check_planar_id(planar_id)
        return self._neighbors[planar_id]

    def cartesian_center(self, planar_id: int) -> tuple[float, float]:
        """Return a center using the ``(x, u)`` Cartesian transform.

        Raises
        ------
        TypeError
            If ``planar_id`` is not an integer or is a boolean.
        IndexError
            If ``planar_id`` is outside the complete lattice.
        """
        planar_id = self._check_planar_id(planar_id)
        x_index, u_index = self.coords[planar_id]
        x_coord = self.pitch * (x_index + 0.5 * u_index)
        y_coord = sqrt(3.0) / 2.0 * self.pitch * u_index
        return (x_coord, y_coord)

    def cell_vertices(self, planar_id: int) -> tuple[tuple[float, float], ...]:
        """Return Cartesian vertices for one hexagonal cell.

        Vertices are ordered counter-clockwise.

        Raises
        ------
        TypeError
            If ``planar_id`` is not an integer or is a boolean.
        IndexError
            If ``planar_id`` is outside the complete lattice.
        """
        center_x, center_y = self.cartesian_center(planar_id)
        # A regular hexagon's center-to-vertex distance equals its side length.
        center_to_vertex = self.face_length
        return tuple(
            (
                center_x + center_to_vertex * cos(pi / 6.0 + angle_index * pi / 3.0),
                center_y + center_to_vertex * sin(pi / 6.0 + angle_index * pi / 3.0),
            )
            for angle_index in range(6)
        )

    def _check_planar_id(self, planar_id: int) -> int:
        """Check and return one complete-lattice planar ID."""
        planar_id = require_integer("planar_id", planar_id)
        if planar_id < 0 or planar_id >= self.n_cells:
            raise IndexError("planar_id is outside the complete lattice")
        return planar_id

    def plot_matplotlib(self, ax: Axes | None = None) -> Axes:
        """Plot the full lattice with planar, lattice, and OpenMC labels.

        Parameters
        ----------
        ax
            Optional axes to populate. A new figure and axes are created when
            omitted.

        Returns
        -------
        matplotlib.axes.Axes
            The populated axes. Each cell label lists planar ID, ``(x, u)``,
            and ``ring/position`` on separate lines.

        Raises
        ------
        TypeError
            If ``ax`` is neither an ``Axes`` nor ``None``.
        """
        if ax is None:
            _, ax = plt.subplots()
        elif not isinstance(ax, Axes):
            raise TypeError("ax must be an Axes or None")

        add_planar_matplotlib_cells(
            self,
            ax,
            facecolor_for=lambda _planar_id: "white",
            label_for=lambda planar_id: self._compact_planar_label(
                planar_id,
                line_break="\n",
            ),
        )

        configure_hex_axes(ax)
        ax.set_title(
            f"HexPlanarMesh: {self.num_rings} rings, {self.n_cells} cells, "
            f"pitch {self.pitch:g} cm"
        )
        ax.text(
            1.02,
            0.5,
            (
                "Line 1: planar ID\n"
                "Line 2: lattice (x, u)\n"
                "Line 3: OpenMC ring/position"
            ),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=8,
        )
        return ax

    def to_plotly(self) -> Figure:
        """Return a Plotly full-lattice figure with per-cell inspection labels.

        The visible cell label lists planar ID, ``(x, u)``, and
        ``ring/position``. Hover text also gives the Cartesian center in cm.
        """
        figure = go.Figure()
        label_x: list[float] = []
        label_y: list[float] = []
        labels: list[str] = []
        add_planar_plotly_cells(
            self,
            figure,
            fillcolor_for=lambda _planar_id: "white",
        )
        for planar_id in range(self.n_cells):
            center_x, center_y = self.cartesian_center(planar_id)
            label_x.append(center_x)
            label_y.append(center_y)
            labels.append(self._compact_planar_label(planar_id, line_break="<br>"))

        figure.add_trace(
            go.Scatter(
                x=label_x,
                y=label_y,
                mode="text",
                text=labels,
                textfont={"size": 12},
                name="",
                hoverinfo="skip",
                showlegend=False,
            )
        )
        add_planar_plotly_hover_targets(
            self,
            figure,
            hover_text_for=lambda planar_id: self._descriptive_cell_label(
                planar_id,
                line_break="<br>",
            ),
        )
        configure_hex_figure(
            figure,
            self,
            (
                f"HexPlanarMesh: {self.num_rings} rings, {self.n_cells} cells, "
                f"pitch {self.pitch:g} cm"
            ),
        )
        return figure

    def export_vtu(self, path: str | Path) -> None:
        """Export the full lattice as a VTK XML unstructured grid.

        Each cell is one planar VTK polygon. Cell data comprises ``planar_id``,
        ``lattice_x``, ``lattice_u``, ``openmc_ring``, and
        ``openmc_position`` in planar-ID order. The mesh pitch is encoded in
        geometry. Parent directories are
        created when needed; an existing file at ``path`` is replaced.

        Raises
        ------
        ValueError
            If ``path`` does not have a ``.vtu`` suffix.
        IsADirectoryError
            If ``path`` identifies an existing directory.
        OSError
            If the parent directory cannot be created or the VTU file cannot
            be written.
        """
        path = prepare_output_path(path, ".vtu")
        points: list[tuple[float, float, float]] = []
        connectivity: list[int] = []
        offsets: list[int] = []
        cell_types: list[int] = []
        for planar_id in range(self.n_cells):
            for x_value, y_value in self.cell_vertices(planar_id):
                points.append((x_value, y_value, 0.0))
                connectivity.append(len(points) - 1)
            offsets.append(len(connectivity))
            cell_types.append(7)  # VTK_POLYGON

        write_vtu(
            path,
            points=points,
            connectivity=connectivity,
            offsets=offsets,
            cell_types=cell_types,
            cell_data=(
                VTKDataArray("Int64", "planar_id", range(self.n_cells)),
                VTKDataArray(
                    "Int64",
                    "lattice_x",
                    (coord[0] for coord in self.coords),
                ),
                VTKDataArray(
                    "Int64",
                    "lattice_u",
                    (coord[1] for coord in self.coords),
                ),
                VTKDataArray(
                    "Int64",
                    "openmc_ring",
                    (index.ring for index in self.openmc_indices),
                ),
                VTKDataArray(
                    "Int64",
                    "openmc_position",
                    (index.position for index in self.openmc_indices),
                ),
            ),
        )

    def _compact_planar_label(self, planar_id: int, line_break: str) -> str:
        """Return a compact in-plot inspection label for one planar cell."""
        lattice_x, lattice_u = self.lattice_coord(planar_id)
        openmc_index = self.openmc_index(planar_id)
        return line_break.join(
            (
                str(planar_id),
                f"({lattice_x}, {lattice_u})",
                f"{openmc_index.ring}/{openmc_index.position}",
            )
        )

    def _descriptive_cell_label(self, planar_id: int, line_break: str) -> str:
        """Return a compact inspection label for one planar cell."""
        lattice_x, lattice_u = self.lattice_coord(planar_id)
        openmc_index = self.openmc_index(planar_id)
        center_x, center_y = self.cartesian_center(planar_id)
        return line_break.join(
            (
                f"planar_id {planar_id}",
                f"lattice ({lattice_x}, {lattice_u})",
                f"ring {openmc_index.ring}, pos {openmc_index.position}",
                f"center ({center_x:.2f}, {center_y:.2f}) cm",
            )
        )


def _ring_coords(radius: int) -> tuple[tuple[int, int], ...]:
    """Return coordinates for one OpenMC ``orientation="x"`` serialized ring."""
    if radius == 0:
        return ((0, 0),)
    coords: list[tuple[int, int]] = []
    q_coord, r_coord = radius, 0
    for dq, dr in _RING_WALK_DIRECTIONS:
        for _ in range(radius):
            coords.append((q_coord, r_coord))
            q_coord += dq
            r_coord += dr
    return tuple(coords)
