"""Pure transverse geometry specification for the SRE-derived comparison.

This module deliberately has no OpenMC dependency. Both the OpenMC model in
`case.py` and the Matplotlib unit-cell drawing use these dimensions and
coordinate helpers.
"""

from __future__ import annotations

from math import cos, pi, sin, sqrt

INCH_TO_CM = 2.54
NUM_CORE_RINGS = 5
FUEL_CAN_ORIENTATION = "y"
BUNDLE_AZIMUTH_RADIANS = pi / 6.0

FUEL_ELEMENT_PITCH_IN = 11.0
ACTIVE_HEIGHT_IN = 72.0
ZR_CAN_WALL_IN = 0.035
INTER_CAN_SODIUM_GAP_IN = 0.170
FUEL_DIAMETER_IN = 0.750
NAK_BOND_THICKNESS_IN = 0.010
STEEL_TUBE_THICKNESS_IN = 0.010
FUEL_ROD_OUTER_DIAMETER_IN = FUEL_DIAMETER_IN + 2.0 * (
    NAK_BOND_THICKNESS_IN + STEEL_TUBE_THICKNESS_IN
)
WIRE_DIAMETER_IN = 0.091
OUTER_ROD_CENTER_SPAN_IN = 2.0 * (FUEL_ROD_OUTER_DIAMETER_IN + WIRE_DIAMETER_IN)
PROCESS_CHANNEL_INNER_DIAMETER_IN = 2.80
PROCESS_CHANNEL_WALL_THICKNESS_IN = 0.035


def inches_to_cm(value: float) -> float:
    """Convert an inch value to the shared centimetre geometry unit."""

    return value * INCH_TO_CM


def hex_edge_from_across_flats(across_flats: float) -> float:
    """Return a regular hexagon side length from its flat-to-flat width."""

    return across_flats / sqrt(3.0)


def graphite_across_flats_cm() -> float:
    """Return the graphite-prism width derived from the sourced lattice pitch."""

    return inches_to_cm(
        FUEL_ELEMENT_PITCH_IN - 2.0 * ZR_CAN_WALL_IN - INTER_CAN_SODIUM_GAP_IN
    )


def can_across_flats_cm() -> float:
    """Return the derived zirconium-can outer width across flats."""

    return inches_to_cm(FUEL_ELEMENT_PITCH_IN - INTER_CAN_SODIUM_GAP_IN)


def lattice_pitch_cm() -> float:
    """Return the sourced centre-to-centre lattice pitch across flats."""

    return inches_to_cm(FUEL_ELEMENT_PITCH_IN)


def active_height_cm() -> float:
    """Return the active height."""

    return inches_to_cm(ACTIVE_HEIGHT_IN)


def point_up_hex_vertices(
    across_flats: float,
    x_coord: float = 0.0,
    y_coord: float = 0.0,
) -> tuple[tuple[float, float], ...]:
    """Return counterclockwise vertices for a point-up regular hexagon."""

    half_width = across_flats / 2.0
    half_height = across_flats / (2.0 * sqrt(3.0))
    full_height = across_flats / sqrt(3.0)
    return (
        (x_coord + half_width, y_coord + half_height),
        (x_coord, y_coord + full_height),
        (x_coord - half_width, y_coord + half_height),
        (x_coord - half_width, y_coord - half_height),
        (x_coord, y_coord - full_height),
        (x_coord + half_width, y_coord - half_height),
    )


def full_pitch_hex_vertices(
    x_coord: float = 0.0,
    y_coord: float = 0.0,
) -> tuple[tuple[float, float], ...]:
    """Return vertices for one point-up 11-inch lattice cell."""

    return point_up_hex_vertices(lattice_pitch_cm(), x_coord, y_coord)


def mini_core_coordinates() -> tuple[tuple[int, int, float, float], ...]:
    """Return planar axial coordinates and centres for the 61-cell mini-core."""

    pitch = lattice_pitch_cm()
    return tuple(
        (
            q_coord,
            r_coord,
            pitch * (q_coord + r_coord / 2.0),
            pitch * sqrt(0.75) * r_coord,
        )
        for q_coord in range(-(NUM_CORE_RINGS - 1), NUM_CORE_RINGS)
        for r_coord in range(-(NUM_CORE_RINGS - 1), NUM_CORE_RINGS)
        if max(abs(q_coord), abs(r_coord), abs(q_coord + r_coord)) < NUM_CORE_RINGS
    )


def rod_centers() -> tuple[tuple[float, float], ...]:
    """Return seven rod centres directed toward fuel-can vertices."""

    radius = inches_to_cm(OUTER_ROD_CENTER_SPAN_IN / 2.0)
    return ((0.0, 0.0),) + tuple(
        (
            radius * cos(BUNDLE_AZIMUTH_RADIANS + index * pi / 3.0),
            radius * sin(BUNDLE_AZIMUTH_RADIANS + index * pi / 3.0),
        )
        for index in range(6)
    )


def wire_centers() -> tuple[tuple[float, float], ...]:
    """Return the fixed transverse surrogate locations for the spiral wire."""

    outer_rod_radius = inches_to_cm(OUTER_ROD_CENTER_SPAN_IN / 2.0)
    fuel_rod_radius = inches_to_cm(FUEL_ROD_OUTER_DIAMETER_IN / 2.0)
    wire_radius = inches_to_cm(WIRE_DIAMETER_IN / 2.0)
    inner_wire_radius = outer_rod_radius / 2.0
    outer_wire_radius = outer_rod_radius + fuel_rod_radius + wire_radius
    return tuple(
        coordinate
        for index in range(6)
        for coordinate in (
            (
                inner_wire_radius * cos(BUNDLE_AZIMUTH_RADIANS + index * pi / 3.0),
                inner_wire_radius * sin(BUNDLE_AZIMUTH_RADIANS + index * pi / 3.0),
            ),
            (
                outer_wire_radius * cos(BUNDLE_AZIMUTH_RADIANS + index * pi / 3.0),
                outer_wire_radius * sin(BUNDLE_AZIMUTH_RADIANS + index * pi / 3.0),
            ),
        )
    )


def wire_to_channel_wall_clearance_in() -> float:
    """Return the clearance implied by tangent outer wires and channel dimensions."""

    return (
        PROCESS_CHANNEL_INNER_DIAMETER_IN / 2.0
        - OUTER_ROD_CENTER_SPAN_IN / 2.0
        - FUEL_ROD_OUTER_DIAMETER_IN / 2.0
        - WIRE_DIAMETER_IN
    )
