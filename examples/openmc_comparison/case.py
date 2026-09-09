"""OpenMC materials and geometry for the SRE-derived comparison case.

The transverse unit cell is the radial homogenization domain. The mini-core
supports structural checks and all 72-inch three-dimensional continuous-energy
calculations, including the direct-profile CE reference tallies.
"""

from __future__ import annotations

from math import hypot, isfinite

import openmc

from geometry import (
    FUEL_DIAMETER_IN,
    FUEL_ROD_OUTER_DIAMETER_IN,
    FUEL_CAN_ORIENTATION,
    NAK_BOND_THICKNESS_IN,
    NUM_CORE_RINGS,
    PROCESS_CHANNEL_INNER_DIAMETER_IN,
    PROCESS_CHANNEL_WALL_THICKNESS_IN,
    WIRE_DIAMETER_IN,
    can_across_flats_cm,
    full_pitch_hex_vertices,
    graphite_across_flats_cm,
    hex_edge_from_across_flats,
    inches_to_cm,
    lattice_pitch_cm,
    mini_core_coordinates,
    rod_centers,
    wire_centers,
)
from specification import TEMPERATURE_K


def material_definitions() -> openmc.Materials:
    """Build the material definitions for the comparison case."""

    graphite = openmc.Material(name="graphite")
    graphite.temperature = TEMPERATURE_K
    graphite.set_density("g/cm3", 1.700)
    graphite.add_element("C", 0.999999, "wo")
    graphite.add_element("B", 0.000001, "wo")
    graphite.add_s_alpha_beta("c_Graphite")

    zirconium = openmc.Material(name="zirconium")
    zirconium.temperature = TEMPERATURE_K
    zirconium.set_density("g/cm3", 6.52)
    zirconium.add_element("Zr", 1.0)

    sodium = openmc.Material(name="sodium")
    sodium.temperature = TEMPERATURE_K
    sodium.set_density("g/cm3", 0.861170)
    sodium.add_element("Na", 1.0)

    steel = openmc.Material(name="ss304")
    steel.temperature = TEMPERATURE_K
    steel.set_density("g/cm3", 8.03)
    for element, fraction in (
        ("C", 0.000800),
        ("Mn", 0.020000),
        ("P", 0.000450),
        ("S", 0.000300),
        ("Si", 0.010000),
        ("Cr", 0.190000),
        ("Ni", 0.095000),
        ("Fe", 0.683450),
    ):
        steel.add_element(element, fraction, "wo")

    nak = openmc.Material(name="nak")
    nak.temperature = TEMPERATURE_K
    nak.set_density("g/cm3", 0.78143)
    nak.add_element("K", 0.786, "wo")
    nak.add_element("Na", 0.214, "wo")

    fuel = openmc.Material(name="fuel")
    fuel.temperature = TEMPERATURE_K
    fuel.set_density("g/cm3", 18.944)
    fuel.add_element("U", 1.0, "wo", enrichment=2.8)
    return openmc.Materials([graphite, zirconium, sodium, steel, nak, fuel])


def _materials_by_name(materials: openmc.Materials) -> dict[str, openmc.Material]:
    """Index the case materials by their stable case-local names."""

    return {material.name: material for material in materials}


def unit_cell_universe(
    materials: openmc.Materials,
    *,
    outer_boundary_type: str | None = "transmission",
) -> openmc.Universe:
    """Build one complete 11-inch lattice unit-cell universe."""

    material = _materials_by_name(materials)
    graphite_hex = openmc.model.HexagonalPrism(
        edge_length=hex_edge_from_across_flats(graphite_across_flats_cm()),
        orientation=FUEL_CAN_ORIENTATION,
    )
    can_hex = openmc.model.HexagonalPrism(
        edge_length=hex_edge_from_across_flats(can_across_flats_cm()),
        orientation=FUEL_CAN_ORIENTATION,
    )
    pitch_hex = None
    if outer_boundary_type is not None:
        pitch_hex = openmc.model.HexagonalPrism(
            edge_length=hex_edge_from_across_flats(lattice_pitch_cm()),
            orientation=FUEL_CAN_ORIENTATION,
            boundary_type=outer_boundary_type,
        )

    channel_inner = openmc.ZCylinder(
        r=inches_to_cm(PROCESS_CHANNEL_INNER_DIAMETER_IN / 2.0)
    )
    channel_outer = openmc.ZCylinder(
        r=inches_to_cm(
            PROCESS_CHANNEL_INNER_DIAMETER_IN / 2.0 + PROCESS_CHANNEL_WALL_THICKNESS_IN
        )
    )
    fuel_radius = inches_to_cm(FUEL_DIAMETER_IN / 2.0)
    nak_radius = fuel_radius + inches_to_cm(NAK_BOND_THICKNESS_IN)
    tube_radius = inches_to_cm(FUEL_ROD_OUTER_DIAMETER_IN / 2.0)
    wire_radius = inches_to_cm(WIRE_DIAMETER_IN / 2.0)

    rod_or_wire = None
    cells: list[openmc.Cell] = []
    for x_coord, y_coord in rod_centers():
        fuel_surface = openmc.ZCylinder(x0=x_coord, y0=y_coord, r=fuel_radius)
        nak_surface = openmc.ZCylinder(x0=x_coord, y0=y_coord, r=nak_radius)
        tube_surface = openmc.ZCylinder(x0=x_coord, y0=y_coord, r=tube_radius)
        cells.extend(
            (
                openmc.Cell(fill=material["fuel"], region=-fuel_surface),
                openmc.Cell(
                    fill=material["nak"],
                    region=+fuel_surface & -nak_surface,
                ),
                openmc.Cell(
                    fill=material["ss304"],
                    region=+nak_surface & -tube_surface,
                ),
            )
        )
        rod_or_wire = (
            -tube_surface if rod_or_wire is None else rod_or_wire | -tube_surface
        )

    for x_coord, y_coord in wire_centers():
        wire_surface = openmc.ZCylinder(x0=x_coord, y0=y_coord, r=wire_radius)
        cells.append(openmc.Cell(fill=material["ss304"], region=-wire_surface))
        rod_or_wire = rod_or_wire | -wire_surface

    assert rod_or_wire is not None
    inter_can_region = +can_hex
    if pitch_hex is not None:
        inter_can_region &= -pitch_hex
    cells.extend(
        (
            openmc.Cell(
                fill=material["sodium"],
                region=-channel_inner & ~rod_or_wire,
            ),
            openmc.Cell(
                fill=material["zirconium"],
                region=+channel_inner & -channel_outer,
            ),
            openmc.Cell(
                fill=material["graphite"],
                region=-graphite_hex & +channel_outer,
            ),
            openmc.Cell(
                fill=material["zirconium"],
                region=+graphite_hex & -can_hex,
            ),
            openmc.Cell(fill=material["sodium"], region=inter_can_region),
        )
    )
    return openmc.Universe(name="sre_fuel_can", cells=cells)


def _default_settings(
    half_width: float,
    axial_height: float | None = None,
) -> openmc.Settings:
    """Return common source and placeholder eigenvalue settings."""

    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.particles = 1_000
    settings.batches = 20
    settings.inactive = 0
    settings.temperature = {"method": "interpolation"}
    settings.source_rejection_fraction = 0.01
    lower_z = -1.0 if axial_height is None else 0.0
    upper_z = 1.0 if axial_height is None else axial_height
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Box(
            lower_left=(-half_width, -half_width, lower_z),
            upper_right=(half_width, half_width, upper_z),
        ),
        constraints={"fissionable": True},
    )
    return settings


def configure_eigenvalue_settings(
    model: openmc.Model,
    *,
    particles: int,
    batches: int,
    inactive: int,
    seed: int | None = None,
    generations_per_batch: int | None = None,
) -> None:
    """Set common eigenvalue settings for a reproducible CE calculation.

    Parameters are deliberately supplied by each calculation script so it can
    choose its own statistical controls.  All comparison calculations use
    temperature interpolation and suppress unneeded tally output.
    """

    settings = model.settings
    settings.run_mode = "eigenvalue"
    settings.particles = particles
    settings.batches = batches
    settings.inactive = inactive
    settings.temperature = {"method": "interpolation"}
    settings.output = {"tallies": False}
    if generations_per_batch is not None:
        settings.generations_per_batch = generations_per_batch
    if seed is not None:
        settings.seed = seed


def build_unit_cell_model() -> openmc.Model:
    """Build the reflecting, transverse 2-D unit-cell model for MGXS work."""

    materials = material_definitions()
    universe = unit_cell_universe(materials, outer_boundary_type="reflective")
    return openmc.Model(
        geometry=openmc.Geometry(universe),
        materials=materials,
        settings=_default_settings(lattice_pitch_cm() / 2.0),
    )


# pylint: disable=too-many-branches,too-many-statements
def build_mini_core_model(
    *,
    axial_height: float | None = None,
) -> openmc.Model:
    """Build the mini-core with vacuum on every exposed face.

    Parameters
    ----------
    axial_height
        Active height in cm, spanning ``z=0`` to ``z=axial_height``. Omit it
        for the transverse structural model.
    """

    if axial_height is not None and (not isfinite(axial_height) or axial_height <= 0.0):
        raise ValueError("axial_height must be finite and positive when provided")

    materials = material_definitions()
    fuel_can = unit_cell_universe(materials, outer_boundary_type=None)
    coordinates = mini_core_coordinates()
    tile_vertices = tuple(
        full_pitch_hex_vertices(x_coord, y_coord)
        for _, _, x_coord, y_coord in coordinates
    )

    def point_key(point: tuple[float, float]) -> tuple[float, float]:
        return (round(point[0], 10), round(point[1], 10))

    edge_owners: dict[tuple[tuple[float, float], tuple[float, float]], list[int]] = {}
    edge_points: dict[
        tuple[tuple[float, float], tuple[float, float]],
        tuple[tuple[float, float], tuple[float, float]],
    ] = {}
    tile_edge_keys: list[
        tuple[tuple[tuple[float, float], tuple[float, float]], ...]
    ] = []
    for tile_index, vertices in enumerate(tile_vertices):
        keys = []
        for start, end in zip(vertices, vertices[1:] + vertices[:1]):
            key = tuple(sorted((point_key(start), point_key(end))))
            edge_owners.setdefault(key, []).append(tile_index)
            edge_points.setdefault(key, (start, end))
            keys.append(key)
        tile_edge_keys.append(tuple(keys))

    if any(len(owners) not in (1, 2) for owners in edge_owners.values()):
        raise ValueError("Mini-core tile edges do not form a valid shared topology.")

    edge_surfaces: dict[
        tuple[tuple[float, float], tuple[float, float]], openmc.Plane
    ] = {}
    for key, owners in edge_owners.items():
        start, end = edge_points[key]
        delta_x = end[0] - start[0]
        delta_y = end[1] - start[1]
        length = hypot(delta_x, delta_y)
        normal_x = delta_y / length
        normal_y = -delta_x / length
        surface = openmc.Plane(
            a=normal_x,
            b=normal_y,
            d=normal_x * start[0] + normal_y * start[1],
            boundary_type="vacuum" if len(owners) == 1 else "transmission",
        )
        edge_surfaces[key] = surface

    lower_plane = None
    upper_plane = None
    if axial_height is not None:
        lower_plane = openmc.ZPlane(
            z0=0.0,
            boundary_type="vacuum",
        )
        upper_plane = openmc.ZPlane(
            z0=axial_height,
            boundary_type="vacuum",
        )

    cells: list[openmc.Cell] = []
    for (q_coord, r_coord, x_coord, y_coord), edge_keys in zip(
        coordinates, tile_edge_keys
    ):
        region = None
        for key in edge_keys:
            surface = edge_surfaces[key]
            halfspace = (
                -surface
                if surface.evaluate((x_coord, y_coord, 0.0)) < 0.0
                else +surface
            )
            region = halfspace if region is None else region & halfspace
        assert region is not None
        if lower_plane is not None and upper_plane is not None:
            region &= +lower_plane & -upper_plane
        cell = openmc.Cell(
            name=f"sre_full_pitch_q{q_coord:+d}_r{r_coord:+d}",
            fill=fuel_can,
            region=region,
        )
        cell.translation = (x_coord, y_coord, 0.0)
        cells.append(cell)

    radial_half_width = (
        NUM_CORE_RINGS - 1
    ) * lattice_pitch_cm() + lattice_pitch_cm() / 2.0
    root = openmc.Universe(cells=cells)
    return openmc.Model(
        geometry=openmc.Geometry(root),
        materials=materials,
        settings=_default_settings(radial_half_width, axial_height),
    )
