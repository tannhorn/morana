"""Check the mini-core construction, including its exact ragged boundary.

Run this OpenMC-only structural check from the repository root with an active
Python environment that provides OpenMC:

``python examples/openmc_comparison/check_structure.py``

It performs no particle transport and writes no artifacts.
"""

from __future__ import annotations

from collections import Counter
from math import isclose, sqrt

import openmc

from case import build_mini_core_model
from geometry import full_pitch_hex_vertices, lattice_pitch_cm, mini_core_coordinates

_EXPECTED_CORE_CELLS = 61
_EXPECTED_BOUNDARY_EDGES = 54
_EXPECTED_INTERNAL_EDGES = 156
_EXPECTED_UNIQUE_EDGES = _EXPECTED_BOUNDARY_EDGES + _EXPECTED_INTERNAL_EDGES
_TOLERANCE = 1.0e-9


def _mini_core_boundary_points(
    coordinates: tuple[tuple[int, int, float, float], ...],
) -> tuple[tuple[float, float], ...]:
    """Return the independently reconstructed mini-core perimeter."""

    edge_map: dict[
        tuple[tuple[float, float], tuple[float, float]],
        tuple[tuple[float, float], tuple[float, float]] | None,
    ] = {}

    def point_key(point: tuple[float, float]) -> tuple[float, float]:
        return (round(point[0], 10), round(point[1], 10))

    for _, _, x_coord, y_coord in coordinates:
        vertices = full_pitch_hex_vertices(x_coord, y_coord)
        for start, end in zip(vertices, vertices[1:] + vertices[:1]):
            key = tuple(sorted((point_key(start), point_key(end))))
            edge_map[key] = None if key in edge_map else (start, end)

    boundary_edges = [edge for edge in edge_map.values() if edge is not None]
    successors = {point_key(start): end for start, end in boundary_edges}
    if len(successors) != len(boundary_edges):
        raise ValueError("Mini-core perimeter is not a single edge loop.")

    start_key = min(successors)
    points = [start_key]
    current_key = start_key
    while True:
        end = successors[current_key]
        end_key = point_key(end)
        if end_key == start_key:
            break
        points.append(end)
        current_key = end_key
        if len(points) > len(boundary_edges):
            raise ValueError("Mini-core perimeter could not be closed.")
    if len(points) != len(boundary_edges):
        raise ValueError("Mini-core perimeter has disconnected edges.")
    return tuple(points)


def _polygon_area(points: tuple[tuple[float, float], ...]) -> float:
    """Return the positive area enclosed by an ordered planar polygon."""

    return (
        abs(
            sum(
                start[0] * end[1] - start[1] * end[0]
                for start, end in zip(points, points[1:] + points[:1])
            )
        )
        / 2.0
    )


def _surface_contains_edge(
    surface: openmc.Surface,
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    """Return whether one CSG surface contains the complete planar edge."""

    return (
        abs(surface.evaluate((*start, 0.0))) < _TOLERANCE
        and abs(surface.evaluate((*end, 0.0))) < _TOLERANCE
    )


def main() -> None:
    """Assert the exact tile-union outline and its OpenMC boundary assignment."""

    coordinates = mini_core_coordinates()
    assert len(coordinates) == _EXPECTED_CORE_CELLS
    assert len({(q_coord, r_coord) for q_coord, r_coord, _, _ in coordinates}) == len(
        coordinates
    )

    points = _mini_core_boundary_points(coordinates)
    assert len(points) == _EXPECTED_BOUNDARY_EDGES
    pitch = lattice_pitch_cm()
    full_pitch_hexagon_area = sqrt(3.0) * pitch**2 / 2.0
    assert isclose(
        _polygon_area(points),
        _EXPECTED_CORE_CELLS * full_pitch_hexagon_area,
        rel_tol=0.0,
        abs_tol=_TOLERANCE,
    )
    model = build_mini_core_model()
    root_cells = tuple(model.geometry.root_universe.cells.values())
    assert len(root_cells) == _EXPECTED_CORE_CELLS
    assert all(isinstance(cell.fill, openmc.Universe) for cell in root_cells)
    assert len({id(cell.fill) for cell in root_cells}) == 1

    expected_centres = {
        f"sre_full_pitch_q{q_coord:+d}_r{r_coord:+d}": (x_coord, y_coord, 0.0)
        for q_coord, r_coord, x_coord, y_coord in coordinates
    }
    assert {cell.name for cell in root_cells} == set(expected_centres)
    for cell in root_cells:
        assert cell.translation is not None
        assert all(
            isclose(actual, expected, rel_tol=0.0, abs_tol=_TOLERANCE)
            for actual, expected in zip(cell.translation, expected_centres[cell.name])
        )
        assert len(cell.region.get_surfaces()) == 6

    surfaces = {
        surface.id: surface
        for cell in root_cells
        for surface in cell.region.get_surfaces().values()
    }
    assert len(surfaces) == _EXPECTED_UNIQUE_EDGES
    incidence = Counter(
        surface.id
        for cell in root_cells
        for surface in cell.region.get_surfaces().values()
    )
    assert Counter(incidence.values()) == {
        1: _EXPECTED_BOUNDARY_EDGES,
        2: _EXPECTED_INTERNAL_EDGES,
    }
    vacuum_surfaces = tuple(
        surface for surface in surfaces.values() if surface.boundary_type == "vacuum"
    )
    assert len(vacuum_surfaces) == _EXPECTED_BOUNDARY_EDGES
    assert all(incidence[surface.id] == 1 for surface in vacuum_surfaces)
    assert all(
        surface.boundary_type
        == ("vacuum" if incidence[surface.id] == 1 else "transmission")
        for surface in surfaces.values()
    )
    boundary_edges = tuple(zip(points, points[1:] + points[:1]))
    for start, end in boundary_edges:
        assert any(
            _surface_contains_edge(surface, start, end) for surface in vacuum_surfaces
        )
    for surface in vacuum_surfaces:
        assert any(
            _surface_contains_edge(surface, start, end) for start, end in boundary_edges
        )

    print(
        "Verified 61 explicit 11-inch lattice cells, 54 exposed edges, and 156 "
        "shared internal edges."
    )


if __name__ == "__main__":
    main()
