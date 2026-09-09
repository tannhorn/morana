"""Share target-domain refinement and quadrature for the hex-z MMS cases."""

from __future__ import annotations

from collections.abc import Iterator
from math import sqrt

import numpy as np

from morana import DomainFace, HexPlanarMesh, MaterialMesh

APOTHEM = 60.0
CIRCUMRADIUS = 2.0 * APOTHEM / sqrt(3.0)
HEIGHT = 108.0
LEVELS = (0, 1, 2)
CORE_RINGS = (3, 4, 6)
AXIAL_SUBDIVISIONS = (2, 3, 4)
QUADRATURE_ORDER = 3


def level_rings(level: int) -> int:
    """Return the active-core ring count for one refinement level."""
    return CORE_RINGS[level]


def maximum_pitch(active_rings: int) -> float:
    """Return the largest pitch whose active cluster lies inside the target."""
    return CIRCUMRADIUS / (active_rings - 1.0 / 3.0)


def layer_heights(level: int) -> tuple[float, ...]:
    """Return unequal layer heights refined with the corresponding core size."""
    subdivisions = AXIAL_SUBDIVISIONS[level]
    bands = (2.0 * HEIGHT / 9.0, 3.0 * HEIGHT / 9.0, 4.0 * HEIGHT / 9.0)
    return tuple(height / subdivisions for height in bands for _ in range(subdivisions))


def hex_prism_quadrature(
    mesh: HexPlanarMesh,
    planar_id: int,
    z_lower: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return independent tensor-product quadrature points and prism weights."""
    nodes, weights = np.polynomial.legendre.leggauss(QUADRATURE_ORDER)
    parameters = 0.5 * (nodes + 1.0)
    parameter_weights = 0.5 * weights
    center = np.array(mesh.cartesian_center(planar_id))
    vertices = np.array(mesh.cell_vertices(planar_id))
    t_values, s_values, z_values = np.meshgrid(
        parameters, parameters, parameters, indexing="ij"
    )
    t_weights, s_weights, z_weights = np.meshgrid(
        parameter_weights, parameter_weights, parameter_weights, indexing="ij"
    )
    x_parts = []
    y_parts = []
    z_parts = []
    weight_parts = []
    for vertex_index in range(6):
        first = vertices[vertex_index] - center
        second = vertices[(vertex_index + 1) % 6] - center
        radial = center + t_values[..., np.newaxis] * (
            (1.0 - s_values)[..., np.newaxis] * first
            + s_values[..., np.newaxis] * second
        )
        triangle_jacobian = abs(first[0] * second[1] - first[1] * second[0])
        x_parts.append(radial[..., 0].ravel())
        y_parts.append(radial[..., 1].ravel())
        z_parts.append((z_lower + height * z_values).ravel())
        weight_parts.append(
            (
                t_weights
                * s_weights
                * z_weights
                * t_values
                * triangle_jacobian
                * height
            ).ravel()
        )
    return (
        np.concatenate(x_parts),
        np.concatenate(y_parts),
        np.concatenate(z_parts),
        np.concatenate(weight_parts),
    )


def cell_average(
    function,
    mesh: HexPlanarMesh,
    planar_id: int,
    z_lower: float,
    height: float,
) -> np.ndarray:
    """Return a scalar or group-vector average over one hex-z control volume."""
    x_values, y_values, z_values, weights = hex_prism_quadrature(
        mesh, planar_id, z_lower, height
    )
    integral = np.asarray(function(x_values, y_values, z_values)) @ weights
    return integral / (mesh.area * height)


def excluded_radial_faces(
    material_mesh: MaterialMesh,
) -> Iterator[tuple[int, int, str, DomainFace]]:
    """Yield validated active radial faces selected through the excluded shell."""
    for axial_index in range(material_mesh.n_axial_layers):
        for active_id in range(material_mesh.n_active_cells(axial_index)):
            for direction in material_mesh.mesh.direction_labels:
                face = material_mesh.face(axial_index, active_id, direction)
                if face.kind == "internal":
                    continue
                if face.kind != "to_excluded":
                    raise RuntimeError(
                        "MMS active radial boundary must use excluded shell"
                    )
                if face.neighbor_openmc_index is None or face.neighbor_key is None:
                    raise RuntimeError(
                        "excluded-shell face is missing neighbor identity"
                    )
                yield axial_index, active_id, direction, face


def volume_weighted_relative_l2(
    configuration,
    result,
    exact_layers: tuple[np.ndarray, ...],
) -> float:
    """Return independent volume-weighted relative L2 error for MMS layers."""
    numerator = 0.0
    denominator = 0.0
    for axial_index, exact in enumerate(exact_layers):
        difference = result.flux_layer(axial_index) - exact
        volume = configuration.material_mesh.cell_volume(axial_index)
        numerator += volume * float(np.sum(difference**2))
        denominator += volume * float(np.sum(exact**2))
    return sqrt(numerator / denominator)


def midplane_axial_index(configuration, result) -> int:
    """Return the axial slice whose centre is closest to the target midplane."""
    return min(
        range(result.n_axial_layers),
        key=lambda index: abs(
            sum(configuration.material_mesh.z_bounds(index)) / 2.0 - HEIGHT / 2.0
        ),
    )
