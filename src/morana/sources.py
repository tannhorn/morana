"""Fixed-source definitions for diffusion calculations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from morana._arrays import freeze_owned_float_array, readonly_float_array
from morana._validation import require_nonnegative_integer
from morana.materials import check_material_name


def _checked_active_ids(material_by_active_id: Mapping[int, str]) -> tuple[int, ...]:
    """Validate and return compact slice-local IDs, allowing an empty layer."""
    if not isinstance(material_by_active_id, Mapping):
        raise TypeError("material_by_active_id must be a mapping")
    active_ids = tuple(
        sorted(
            require_nonnegative_integer("material_by_active_id key", active_id)
            for active_id in material_by_active_id
        )
    )
    for material in material_by_active_id.values():
        check_material_name(material)
    expected = tuple(range(len(active_ids)))
    if active_ids != expected:
        raise ValueError(
            "material_by_active_id must use slice-local active IDs starting at 0"
        )
    return active_ids


@dataclass(frozen=True)
class UniformSource:
    """Apply one volumetric source vector to every active cell.

    Parameters
    ----------
    strength
        Nonempty one-dimensional group-major source vector in
        ``n / cm^3 / s``. It is copied into a read-only floating-point array.

    Raises
    ------
    TypeError
        If ``strength`` contains values other than real non-Boolean numbers.
    ValueError
        If ``strength`` is not a nonempty one-dimensional vector.

    Notes
    -----
    The vector is broadcast to every active cell in the selected layer,
    including a valid zero-column result for an empty layer. Finiteness,
    nonnegativity, and compatibility with the problem group count are checked
    during source assembly rather than construction. ``UniformSource`` is
    immutable; construct a replacement source to change its strength.
    """

    strength: np.ndarray | list[float]

    def __post_init__(self) -> None:
        """Convert source strength to a one-dimensional NumPy array."""
        strength = readonly_float_array(self.strength)
        if strength.ndim != 1 or strength.size == 0:
            raise ValueError("uniform source strength must be a non-empty vector")
        object.__setattr__(self, "strength", strength)

    def values(
        self, axial_index: int, material_by_active_id: Mapping[int, str]
    ) -> np.ndarray:
        """Broadcast source strength over the selected layer's active cells.

        The returned read-only array has shape
        ``(len(strength), len(material_by_active_id))``. ``axial_index`` is
        checked only as a nonnegative explicit index because this source does
        not own an axial stack.
        """
        require_nonnegative_integer("axial_index", axial_index)
        n_cells = len(_checked_active_ids(material_by_active_id))
        return np.broadcast_to(
            self.strength[:, np.newaxis],
            (self.strength.size, n_cells),
        )


@dataclass(frozen=True)
class MaterialSource:
    """Assign volumetric source vectors by material name.

    Parameters
    ----------
    values_by_material
        Nonempty mapping from nonempty material names to common-length
        nonempty group-major vectors in ``n / cm^3 / s``. Values are copied
        into read-only floating-point arrays.

    Raises
    ------
    TypeError
        If ``values_by_material`` is not a mapping or a material name is not a
        string, or a source vector contains values other than real non-Boolean
        numbers.
    ValueError
        If a material name is empty, the mapping is empty, a vector is empty
        or not one-dimensional, or vectors do not have one common group count.

    Notes
    -----
    Every material in a selected active layer must have an entry; evaluation
    otherwise raises ``ValueError`` naming the missing material. The mapping
    may contain unused entries. Finiteness, nonnegativity, and compatibility
    with the problem group count are checked during source assembly rather
    than construction. The stored vectors and mapping are immutable; construct
    a replacement source to change material-wise values.
    """

    values_by_material: Mapping[str, np.ndarray | list[float]]

    def __post_init__(self) -> None:
        """Convert material source vectors to owned read-only arrays."""
        if not isinstance(self.values_by_material, Mapping):
            raise TypeError("values_by_material must be a mapping")
        converted = {
            material: readonly_float_array(values)
            for material, values in self.values_by_material.items()
        }
        for material in converted:
            check_material_name(material)
        for values in converted.values():
            if values.ndim != 1 or values.size == 0:
                raise ValueError("material source values must be non-empty vectors")
        sizes = {values.size for values in converted.values()}
        if not converted or len(sizes) != 1:
            raise ValueError("material source vectors must have the same group count")
        object.__setattr__(self, "values_by_material", MappingProxyType(converted))

    def values(
        self, axial_index: int, material_by_active_id: Mapping[int, str]
    ) -> np.ndarray:
        """Return material-wise source values in compact active-ID order.

        The returned array is read-only with one column for each selected
        active ID. An empty layer produces a read-only ``(groups, 0)`` array.
        """
        require_nonnegative_integer("axial_index", axial_index)
        active_ids = _checked_active_ids(material_by_active_id)
        groups = next(iter(self.values_by_material.values())).size
        try:
            columns = [
                self.values_by_material[material_by_active_id[active_id]]
                for active_id in active_ids
            ]
        except KeyError as exc:
            raise ValueError(f"source missing material {exc.args[0]!r}") from exc
        if not columns:
            return freeze_owned_float_array(np.empty((groups, 0)))
        return freeze_owned_float_array(np.column_stack(columns))


@dataclass(frozen=True)
class CellSource:
    """Store explicit group/cell source values by axial layer.

    Parameters
    ----------
    layers
        Bottom-to-top tuple of group-major arrays shaped
        ``(groups, active_cells_in_layer)`` in ``n / cm^3 / s`` and slice-local
        active-ID order. The tuple must be nonempty, every array must be
        two-dimensional with one common nonzero group count, and arrays may be
        ragged in their active-cell dimension. Inputs are copied into read-only
        floating-point arrays.

    Raises
    ------
    TypeError
        If a layer is not iterable or contains values other than real
        non-Boolean numbers.
    ValueError
        If no layer is supplied, a layer is not two-dimensional, no group is
        supplied, or layer group counts differ.

    Notes
    -----
    A cell source does not own a material mesh. Assembly requires exactly one
    supplied source layer per material-mesh layer and checks each column count
    against that selected layer's active-cell count. It checks finiteness and
    nonnegativity then, not at construction. ``CellSource`` is immutable;
    construct a replacement source to change its layers.
    """

    layers: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        """Own checked group-major source arrays for every supplied layer."""
        layers = tuple(readonly_float_array(layer) for layer in self.layers)
        if not layers:
            raise ValueError("cell source must contain at least one layer")
        if any(layer.ndim != 2 for layer in layers):
            raise ValueError(
                "cell source layers must have shape (groups, active_cells)"
            )
        groups = layers[0].shape[0]
        if groups == 0:
            raise ValueError("cell source layers must contain at least one group")
        if any(layer.shape[0] != groups for layer in layers[1:]):
            raise ValueError("cell source layers must have the same group count")
        object.__setattr__(self, "layers", layers)

    def values(
        self, axial_index: int, material_by_active_id: Mapping[int, str]
    ) -> np.ndarray:
        """Return explicit read-only source values for one selected layer.

        The selected array must have exactly one column for every compact
        active ID. Requesting an index outside this source's layer tuple raises
        ``ValueError``.
        """
        require_nonnegative_integer("axial_index", axial_index)
        if axial_index >= len(self.layers):
            raise ValueError(f"cell source has no values for axial layer {axial_index}")
        active_ids = _checked_active_ids(material_by_active_id)
        layer = self.layers[axial_index]
        if layer.shape[1] != len(active_ids):
            raise ValueError(
                "cell source column count must match active cells in the selected layer"
            )
        return layer
