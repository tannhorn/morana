"""Material-key layout over a full hexagonal mesh."""

# pylint: disable=duplicate-code,protected-access,too-many-lines

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Mapping

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch
import plotly.graph_objects as go
from plotly.graph_objects import Figure

from morana._plotting import (
    DEFAULT_MATERIAL_COLORS,
    add_planar_matplotlib_cells,
    add_planar_plotly_cells,
    add_planar_plotly_hover_targets,
    configure_hex_axes,
    configure_hex_figure,
)
from morana._vtk import VTKDataArray, prepare_output_path, write_vtm, write_vtu
from morana._validation import (
    require_finite_positive_real,
    require_human_readable_identifier,
    require_integer,
    require_nonnegative_integer,
    require_positive_integer,
)
from morana.materials import Material
from morana.materials import (
    ExcludedRegion,
    INACTIVE_EXCLUDED_KEY,
    excluded_key_color,
    excluded_key_kind,
    is_active_material_key,
    is_excluded_material_key,
    normalize_excluded_regions,
)
from morana.hex_planar_mesh import HexPlanarMesh, OpenMCIndex
from morana._validation import require_nonempty_string

DOMAIN_FACE_KIND_INTERNAL = "internal"
DOMAIN_FACE_KIND_OUTER = "outer"
DOMAIN_FACE_KIND_TO_EXCLUDED = "to_excluded"
EXPOSED_DOMAIN_FACE_KINDS = frozenset(
    {DOMAIN_FACE_KIND_OUTER, DOMAIN_FACE_KIND_TO_EXCLUDED}
)
DOMAIN_GROUP_NAMES = ("active", "excluded")
AXIAL_DIRECTION_OFFSETS = MappingProxyType({"bottom": -1, "top": 1})
AXIAL_DIRECTION_LABELS = tuple(AXIAL_DIRECTION_OFFSETS)


@dataclass(frozen=True, init=False)
# pylint: disable=too-many-instance-attributes
class DomainFace:
    """Store a discretization-neutral active-cell face description.

    Instances are returned by ``MaterialMesh.face``. Direct construction is
    not supported.

    Attributes
    ----------
    axial_index, active_id, openmc_index
        Axial, slice-local, and geometric identity of the owning active cell.
    direction, kind
        Canonical face direction and its internal, outer, or excluded-interface
        topology classification.
    neighbor_axial_index, neighbor_active_id, neighbor_openmc_index
        In-stack axial, slice-local, and geometric identity of an internal or
        excluded-interface neighbor.
    neighbor_key, neighbor_key_kind
        Excluded-region identity available only for an excluded interface.

    Notes
    -----
    A domain face describes topology only. Boundary physics, face geometry, and
    finite-volume conductance are resolved separately by their owning layers.
    ``MaterialMesh.face()`` constructs topology-consistent values. For an
    ``"internal"`` face, every neighbor-identity field is present and both
    excluded-region fields are ``None``. For a ``"to_excluded"`` face, the
    neighbor axial and OpenMC indices and both excluded-region fields are
    present, while ``neighbor_active_id`` is ``None``. For an ``"outer"``
    face, every neighbor field is ``None``.

    ``DomainFace`` is an immutable value record. It carries a snapshot of one
    topology query and has no live reference to a ``MaterialMesh``.
    """

    axial_index: int
    active_id: int
    openmc_index: OpenMCIndex
    direction: str
    neighbor_axial_index: int | None
    neighbor_active_id: int | None
    neighbor_openmc_index: OpenMCIndex | None
    kind: str
    neighbor_key: str | None = None
    neighbor_key_kind: str | None = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject direct construction; obtain faces from ``MaterialMesh``."""
        _ = args, kwargs
        raise TypeError("DomainFace instances are returned by MaterialMesh.face()")

    @classmethod
    def _from_trusted(
        cls,
        *,
        axial_index: int,
        active_id: int,
        openmc_index: OpenMCIndex,
        direction: str,
        neighbor_axial_index: int | None,
        neighbor_active_id: int | None,
        neighbor_openmc_index: OpenMCIndex | None,
        kind: str,
        neighbor_key: str | None = None,
        neighbor_key_kind: str | None = None,
    ) -> "DomainFace":
        """Construct one face from ``MaterialMesh``-owned checked topology.

        ``MaterialMesh.face()`` derives every field from its immutable complete
        layout.
        """
        instance = object.__new__(cls)
        object.__setattr__(instance, "axial_index", axial_index)
        object.__setattr__(instance, "active_id", active_id)
        object.__setattr__(instance, "openmc_index", openmc_index)
        object.__setattr__(instance, "direction", direction)
        object.__setattr__(instance, "neighbor_axial_index", neighbor_axial_index)
        object.__setattr__(instance, "neighbor_active_id", neighbor_active_id)
        object.__setattr__(instance, "neighbor_openmc_index", neighbor_openmc_index)
        object.__setattr__(instance, "kind", kind)
        object.__setattr__(instance, "neighbor_key", neighbor_key)
        object.__setattr__(instance, "neighbor_key_kind", neighbor_key_kind)
        return instance


@dataclass(frozen=True)
class MaterialSlice:
    """Immutable height-bearing planar material-layout input.

    Parameters
    ----------
    mesh
        Authoritative ``HexPlanarMesh`` for this slice.
    material_keys
        Sparse mapping from OpenMC positions to active or excluded keys.
        Keys must be ``OpenMCIndex`` values belonging to ``mesh``; values must
        be human-readable identifiers: they must contain a non-whitespace
        character and only printable characters. The mapping is copied to a
        read-only mapping.
        Unspecified positions become the built-in inactive key when stacked.
    height
        Finite positive axial height in cm.

    Raises
    ------
    TypeError
        If ``mesh`` is not a ``HexPlanarMesh``, ``material_keys`` is not a
        mapping, a mapping key is not an ``OpenMCIndex``, or a material key is
        not a string, or ``height`` is not a real non-Boolean number.
    ValueError
        If ``height`` is not finite and positive, a material key is empty,
        whitespace-only, or non-printable, or a mapping key is outside
        ``mesh``.

    Notes
    -----
    A slice is reusable construction data. It deliberately does not own the
    excluded-region catalog or compact active-cell IDs; ``MaterialMesh.stack``
    applies those domain-level decisions to a bottom-to-top slice stack. Thus,
    a key is not classified as active or excluded until stacking, when the
    material mesh receives its excluded-region catalog.
    """

    mesh: HexPlanarMesh
    material_keys: Mapping[OpenMCIndex, str]
    height: float

    def __post_init__(self) -> None:
        """Check and own one sparse planar material-key mapping."""
        if not isinstance(self.mesh, HexPlanarMesh):
            raise TypeError("mesh must be a HexPlanarMesh")
        if not isinstance(self.material_keys, Mapping):
            raise TypeError("material_keys must be a mapping")
        height = require_finite_positive_real("height", self.height)
        valid_indices = set(self.mesh.openmc_indices)
        normalized: dict[OpenMCIndex, str] = {}
        for openmc_index, material_key in self.material_keys.items():
            if not isinstance(openmc_index, OpenMCIndex):
                raise TypeError("material_keys keys must be OpenMCIndex values")
            if openmc_index not in valid_indices:
                raise ValueError(f"unknown OpenMC index: {openmc_index!r}")
            require_human_readable_identifier("material key", material_key)
            normalized[openmc_index] = material_key
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "material_keys", MappingProxyType(normalized))

    @classmethod
    def from_openmc_rings(
        cls,
        mesh: HexPlanarMesh,
        rings: list[list[str]],
        *,
        height: float,
    ) -> "MaterialSlice":
        """Build one reusable complete slice from OpenMC-style ring data.

        Parameters
        ----------
        mesh
            Authoritative ``HexPlanarMesh`` for the returned slice.
        rings
            List of lists of human-readable material keys for every mesh ring,
            ordered outermost to innermost. Ring ``i`` must contain
            ``max(6 * (mesh.num_rings - 1 - i), 1)`` values. Positions within
            each ring follow the mesh's OpenMC ``orientation="x"`` order.
            Unlike direct construction, this form assigns every planar
            position explicitly.
        height
            Finite positive slice height in cm.

        Returns
        -------
        MaterialSlice
            An immutable slice with a complete material-key mapping.

        Raises
        ------
        TypeError
            If ``mesh`` is not a ``HexPlanarMesh``, ``rings`` is not a list of
            lists, or a ring value is not a string.
        ValueError
            If the number of rings or a ring length does not match ``mesh``,
            a ring value is empty, whitespace-only, or non-printable, or the
            resulting slice has an invalid height.

        Notes
        -----
        Ring values are human-readable identifiers. Their active or excluded
        classification remains a ``MaterialMesh.stack`` decision.
        """
        if not isinstance(mesh, HexPlanarMesh):
            raise TypeError("mesh must be a HexPlanarMesh")
        if not isinstance(rings, list) or any(
            not isinstance(ring, list) for ring in rings
        ):
            raise TypeError("rings must be a list of lists")
        if len(rings) != mesh.num_rings:
            raise ValueError("ring count must match mesh.num_rings")
        material_keys: dict[OpenMCIndex, str] = {}
        for ring_index, ring in enumerate(rings):
            radius = mesh.num_rings - 1 - ring_index
            expected_size = max(6 * radius, 1)
            if len(ring) != expected_size:
                raise ValueError(
                    f"ring {ring_index} must contain {expected_size} positions"
                )
            for position, material_key in enumerate(ring):
                material_keys[OpenMCIndex(ring_index, position)] = material_key
        return cls(mesh, material_keys, height)

    def extrude(
        self,
        *,
        count: int | None = None,
        heights: tuple[float, ...] | None = None,
    ) -> tuple["MaterialSlice", ...]:
        """Return a composable uniform or explicitly height-bearing fragment.

        Parameters
        ----------
        count
            Positive integer number of copies retaining this slice's height.
            Provide this argument or ``heights``, but not both.
        heights
            Nonempty tuple of finite positive heights in cm. Provide this
            argument or ``count``, but not both.

        Returns
        -------
        tuple[MaterialSlice, ...]
            A bottom-to-top fragment suitable for concatenation and passing to
            ``MaterialMesh.stack``. Uniform extrusion repeats this immutable
            slice value; explicit heights create one independent slice per
            requested height with the same mesh and material keys.

        Raises
        ------
        TypeError
            If ``count`` is not an integer or is a boolean.
        ValueError
            If neither or both forms are supplied, ``count`` is nonpositive,
            ``heights`` is not a nonempty tuple, or an explicit height is
            invalid.
        """
        if (count is None) == (heights is None):
            raise ValueError("provide exactly one of count or heights")
        if count is not None:
            return (self,) * require_positive_integer("count", count)
        if not isinstance(heights, tuple) or not heights:
            raise ValueError("heights must be a nonempty tuple")
        return tuple(
            MaterialSlice(self.mesh, self.material_keys, height) for height in heights
        )


# pylint: disable=too-many-public-methods
class MaterialMesh:
    """Store material keys for every lattice position.

    Attributes
    ----------
    mesh
        Authoritative planar mesh shared by all layers.
    layers
        Read-only completed material-key mappings in bottom-to-top order.
    axial_layer_heights
        Immutable positive heights in bottom-to-top order.
    excluded_regions
        Read-only excluded-region catalog.

    Notes
    -----
    ``HexPlanarMesh`` owns only the full regular geometry. ``MaterialMesh`` overlays
    material keys on that geometry and defines the active solution domain.
    Unassigned positions default to the built-in inactive excluded key ``"0"``.
    Direct construction is rejected; use
    [`MaterialMesh.stack`][morana.material_mesh.MaterialMesh.stack].

    The public API is persistent: public properties expose read-only mappings
    and immutable height tuples, and supported assignment changes construct an
    independent material mesh rather than modifying this one. Treat the
    underscore-prefixed owned state as internal implementation detail.
    """

    _mesh: HexPlanarMesh
    _layers: tuple[dict[OpenMCIndex, str], ...]
    _axial_layer_heights: tuple[float, ...]
    _excluded_regions: dict[str, ExcludedRegion]
    _active_indices: tuple[tuple[OpenMCIndex, ...], ...]
    _active_id_by_openmc_index: tuple[dict[OpenMCIndex, int], ...]
    _material_by_active_id: tuple[Mapping[int, str], ...]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject direct construction; completed meshes are stacked slices."""
        _ = args, kwargs
        raise TypeError("MaterialMesh must be constructed with MaterialMesh.stack()")

    @property
    def mesh(self) -> HexPlanarMesh:
        """Return the authoritative planar mesh for the complete stack."""
        return self._mesh

    @property
    def layers(self) -> tuple[Mapping[OpenMCIndex, str], ...]:
        """Return read-only complete material-key mappings in axial order."""
        return tuple(MappingProxyType(layer) for layer in self._layers)

    @property
    def axial_layer_heights(self) -> tuple[float, ...]:
        """Return the immutable positive layer heights in axial order."""
        return self._axial_layer_heights

    @property
    def excluded_regions(self) -> Mapping[str, ExcludedRegion]:
        """Return the read-only excluded-region catalog."""
        return MappingProxyType(self._excluded_regions)

    @classmethod
    def _from_owned_state(
        cls,
        mesh: HexPlanarMesh,
        layers: tuple[dict[OpenMCIndex, str], ...],
        axial_layer_heights: tuple[float, ...],
        excluded_regions: dict[str, ExcludedRegion],
    ) -> "MaterialMesh":
        """Construct one completed mesh from already-owned checked state."""
        instance = object.__new__(cls)
        instance._mesh = mesh
        instance._layers = layers
        instance._axial_layer_heights = axial_layer_heights
        instance._excluded_regions = excluded_regions
        active_indices = tuple(
            tuple(
                openmc_index
                for openmc_index in mesh.openmc_indices
                if is_active_material_key(layer[openmc_index], excluded_regions)
            )
            for layer in layers
        )
        instance._active_indices = active_indices
        instance._active_id_by_openmc_index = tuple(
            {openmc_index: active_id for active_id, openmc_index in enumerate(indices)}
            for indices in active_indices
        )
        instance._material_by_active_id = tuple(
            MappingProxyType(
                {
                    active_id: layer[openmc_index]
                    for active_id, openmc_index in enumerate(indices)
                }
            )
            for layer, indices in zip(layers, active_indices, strict=True)
        )
        return instance

    @classmethod
    def stack(
        cls,
        slices: tuple[MaterialSlice, ...],
        *,
        excluded_regions: Mapping[str, ExcludedRegion] | None = None,
    ) -> "MaterialMesh":
        """Complete a bottom-to-top stack of reusable material slices.

        Parameters
        ----------
        slices
            Nonempty tuple of ``MaterialSlice`` values in bottom-to-top order.
            Every slice must reference the identical ``HexPlanarMesh`` instance.
            Each sparse mapping is completed independently: positions absent from
            a slice receive the built-in inactive key ``"0"``.
        excluded_regions
            Optional mapping from keys to ``ExcludedRegion`` descriptions for
            additional inactive regions. Keys must be nonempty strings. The
            built-in key ``"0"`` always denotes the ``"inactive"`` kind and
            may not be redefined. A key present in this catalog is excluded
            from the active solver domain in every layer where it occurs.

        Returns
        -------
        MaterialMesh
            An independent completed layout with read-only layer mappings and
            excluded-region catalog.

        Raises
        ------
        TypeError
            If ``slices`` contains a value other than ``MaterialSlice``,
            ``excluded_regions`` is not a mapping, or an excluded-region key
            is not a string.
        ValueError
            If ``slices`` is not a nonempty tuple, slices use different mesh
            instances, or the excluded-region mapping is invalid.
        """
        if not isinstance(slices, tuple) or not slices:
            raise ValueError("slices must be a nonempty tuple of MaterialSlice values")
        if any(
            not isinstance(material_slice, MaterialSlice) for material_slice in slices
        ):
            raise TypeError("slices must contain only MaterialSlice values")
        mesh = slices[0].mesh
        if any(material_slice.mesh is not mesh for material_slice in slices):
            raise ValueError("every slice must reference the identical HexPlanarMesh")
        region_map = normalize_excluded_regions(
            {} if excluded_regions is None else excluded_regions
        )
        normalized_layers: list[dict[OpenMCIndex, str]] = []
        for material_slice in slices:
            normalized = {
                openmc_index: INACTIVE_EXCLUDED_KEY
                for openmc_index in mesh.openmc_indices
            }
            for openmc_index, key in material_slice.material_keys.items():
                normalized[openmc_index] = key
            normalized_layers.append(normalized)
        return cls._from_owned_state(
            mesh,
            tuple(normalized_layers),
            tuple(material_slice.height for material_slice in slices),
            region_map,
        )

    @property
    def n_axial_layers(self) -> int:
        """Return the number of axial material layers."""
        return len(self._layers)

    @property
    def face_direction_labels(self) -> tuple[str, ...]:
        """Return radial and axial face labels in canonical domain order."""
        return (*self.mesh.direction_labels, *AXIAL_DIRECTION_LABELS)

    @property
    def z_min(self) -> float:
        """Return the physical lower coordinate of the complete slice stack."""
        return 0.0

    @property
    def z_max(self) -> float:
        """Return the physical upper coordinate of the complete slice stack."""
        return sum(self.axial_layer_heights)

    def _check_axial_index(self, axial_index: int) -> int:
        """Check and return one nonnegative in-stack axial layer index."""
        axial_index = require_integer("axial_index", axial_index)
        if axial_index < 0 or axial_index >= self.n_axial_layers:
            raise IndexError("axial_index is outside the material-slice stack")
        return axial_index

    def layer_height(self, axial_index: int) -> float:
        """Return the height of one axial material layer in cm."""
        axial_index = self._check_axial_index(axial_index)
        return self.axial_layer_heights[axial_index]

    def z_bounds(self, axial_index: int) -> tuple[float, float]:
        """Return cumulative lower and upper coordinates for one layer."""
        axial_index = self._check_axial_index(axial_index)
        z_min = sum(self.axial_layer_heights[:axial_index])
        return z_min, z_min + self.axial_layer_heights[axial_index]

    def cell_volume(self, axial_index: int) -> float:
        """Return one hex-z cell volume for an axial material layer."""
        return self.mesh.area * self.layer_height(axial_index)

    def radial_face_area(self, axial_index: int) -> float:
        """Return one radial face area for an axial material layer."""
        return self.mesh.face_length * self.layer_height(axial_index)

    def axial_face_area(self, axial_index: int) -> float:
        """Return one axial top/bottom face area for an axial material layer."""
        _ = self._check_axial_index(axial_index)
        return self.mesh.area

    def key_at(self, axial_index: int, openmc_index: OpenMCIndex) -> str:
        """Return the completed material key at one axial-layer position.

        Unassigned positions have the built-in inactive key ``"0"``. A
        position outside the full planar mesh raises ``KeyError``.

        Raises
        ------
        TypeError
            If ``axial_index`` is not an integer or ``openmc_index`` is not an
            ``OpenMCIndex``.
        IndexError
            If ``axial_index`` is outside the material-slice stack.
        """
        axial_index = self._check_axial_index(axial_index)
        if not isinstance(openmc_index, OpenMCIndex):
            raise TypeError("openmc_index must be an OpenMCIndex")
        return self._layers[axial_index][openmc_index]

    def _with_assignment(
        self,
        axial_index: int,
        openmc_index: OpenMCIndex,
        material_key: str,
    ) -> "MaterialMesh":
        """Return an independent mesh with one checked assignment replaced."""
        axial_index = self._check_axial_index(axial_index)
        if openmc_index not in self._layers[axial_index]:
            raise ValueError(f"unknown OpenMC index: {openmc_index!r}")
        require_human_readable_identifier("material key", material_key)
        layers = tuple(dict(layer) for layer in self._layers)
        layers[axial_index][openmc_index] = material_key
        return self._from_owned_state(
            self._mesh,
            layers,
            self._axial_layer_heights,
            dict(self._excluded_regions),
        )

    def active_indices(self, axial_index: int) -> tuple[OpenMCIndex, ...]:
        """Return active OpenMC positions in compact solver order.

        The order is the selected subset of ``mesh.openmc_indices``. It is the
        canonical order for slice-local active IDs and the active-cell axis of
        solver arrays for this layer.
        """
        axial_index = self._check_axial_index(axial_index)
        return self._active_indices[axial_index]

    def n_active_cells(self, axial_index: int) -> int:
        """Return the number of active solver cells in one axial layer."""
        return len(self.active_indices(axial_index))

    def material_by_active_id(self, axial_index: int) -> Mapping[int, str]:
        """Return a read-only material mapping keyed by slice-local active ID.

        The mapping insertion order matches ``active_indices(axial_index)`` and
        therefore the selected layer's compact solver order. The returned
        mapping is a stable view of state derived when the material mesh is
        constructed; copy it with ``dict(...)`` if mutation is required.
        """
        axial_index = self._check_axial_index(axial_index)
        return self._material_by_active_id[axial_index]

    def active_id_at(
        self,
        axial_index: int,
        openmc_index: OpenMCIndex,
    ) -> int | None:
        """Return a slice-local active ID at a lattice position, or ``None``.

        ``axial_index`` selects the layer before ``openmc_index`` identifies
        its planar position.

        ``None`` means that the position is excluded in the selected layer or
        is not a position of the complete planar mesh. Use ``key_at`` when
        those cases need to be distinguished.

        Raises
        ------
        TypeError
            If ``axial_index`` is not an integer or ``openmc_index`` is not an
            ``OpenMCIndex``.
        IndexError
            If ``axial_index`` is outside the material-slice stack.
        """
        axial_index = self._check_axial_index(axial_index)
        if not isinstance(openmc_index, OpenMCIndex):
            raise TypeError("openmc_index must be an OpenMCIndex")
        return self._active_id_by_openmc_index[axial_index].get(openmc_index)

    def openmc_index_for_active_id(
        self,
        axial_index: int,
        active_id: int,
    ) -> OpenMCIndex:
        """Return the lattice position for a slice-local active ID.

        ``axial_index`` selects the layer before ``active_id`` identifies a
        cell in its compact order. The ID must be a nonnegative integer in the
        selected layer. Another input type raises ``TypeError``; a negative or
        out-of-range integer raises ``ValueError``.
        """
        axial_index = self._check_axial_index(axial_index)
        active_indices = self._active_indices[axial_index]
        active_id = require_nonnegative_integer("active_id", active_id)
        if active_id >= len(active_indices):
            raise ValueError("active_id is outside the selected material-mesh layer")
        return active_indices[active_id]

    def face(
        self,
        axial_index: int,
        active_id: int,
        direction: str,
    ) -> DomainFace:
        """Return the topology description for one active-cell face.

        ``axial_index`` and ``active_id`` identify the cell in that order.
        The result is independent of discretization details. Solvers decide
        how to turn the face classification into matrix coefficients or
        response relations.

        Raises
        ------
        TypeError
            If ``direction`` is not a string.
        ValueError
            If ``direction`` is empty or is not one of the radial or axial
            face-direction labels.
        """
        axial_index = self._check_axial_index(axial_index)
        require_nonempty_string("direction", direction)
        if direction in AXIAL_DIRECTION_LABELS:
            return self._axial_face(active_id, direction, axial_index)
        if direction not in self.mesh.direction_labels:
            raise ValueError(f"unknown hex direction: {direction!r}")
        return self._radial_face(active_id, direction, axial_index)

    def _radial_face(
        self,
        active_id: int,
        direction: str,
        axial_index: int,
    ) -> DomainFace:
        """Return the topology description for one radial hex face."""
        openmc_index = self.openmc_index_for_active_id(axial_index, active_id)
        planar_id = self.mesh.planar_id_at(openmc_index)
        if planar_id is None:
            raise ValueError(f"active ID has no mesh position: {active_id}")

        direction_index = self.mesh.direction_labels.index(direction)
        neighbor_planar_id = self.mesh.neighbors(planar_id)[direction_index]
        if neighbor_planar_id is None:
            return DomainFace._from_trusted(
                active_id=active_id,
                axial_index=axial_index,
                openmc_index=openmc_index,
                direction=direction,
                neighbor_axial_index=None,
                neighbor_openmc_index=None,
                neighbor_active_id=None,
                kind=DOMAIN_FACE_KIND_OUTER,
            )

        neighbor_openmc_index = self.mesh.openmc_index(neighbor_planar_id)
        neighbor_key = self.key_at(axial_index, neighbor_openmc_index)
        neighbor_kind = excluded_key_kind(neighbor_key, self.excluded_regions)
        if neighbor_kind is not None:
            return DomainFace._from_trusted(
                active_id=active_id,
                axial_index=axial_index,
                openmc_index=openmc_index,
                direction=direction,
                neighbor_axial_index=axial_index,
                neighbor_openmc_index=neighbor_openmc_index,
                neighbor_active_id=None,
                kind=DOMAIN_FACE_KIND_TO_EXCLUDED,
                neighbor_key=neighbor_key,
                neighbor_key_kind=neighbor_kind,
            )

        return DomainFace._from_trusted(
            active_id=active_id,
            axial_index=axial_index,
            openmc_index=openmc_index,
            direction=direction,
            neighbor_axial_index=axial_index,
            neighbor_openmc_index=neighbor_openmc_index,
            neighbor_active_id=self.active_id_at(axial_index, neighbor_openmc_index),
            kind=DOMAIN_FACE_KIND_INTERNAL,
        )

    def _axial_face(
        self,
        active_id: int,
        direction: str,
        axial_index: int,
    ) -> DomainFace:
        """Return the topology description for a bottom or top face."""
        openmc_index = self.openmc_index_for_active_id(axial_index, active_id)
        neighbor_axial_index = axial_index + AXIAL_DIRECTION_OFFSETS[direction]
        if neighbor_axial_index < 0 or neighbor_axial_index >= self.n_axial_layers:
            return DomainFace._from_trusted(
                active_id=active_id,
                axial_index=axial_index,
                openmc_index=openmc_index,
                direction=direction,
                neighbor_axial_index=None,
                neighbor_openmc_index=None,
                neighbor_active_id=None,
                kind=DOMAIN_FACE_KIND_OUTER,
            )

        neighbor_key = self.key_at(neighbor_axial_index, openmc_index)
        neighbor_kind = excluded_key_kind(neighbor_key, self.excluded_regions)
        if neighbor_kind is not None:
            return DomainFace._from_trusted(
                active_id=active_id,
                axial_index=axial_index,
                openmc_index=openmc_index,
                direction=direction,
                neighbor_axial_index=neighbor_axial_index,
                neighbor_openmc_index=openmc_index,
                neighbor_active_id=None,
                kind=DOMAIN_FACE_KIND_TO_EXCLUDED,
                neighbor_key=neighbor_key,
                neighbor_key_kind=neighbor_kind,
            )

        return DomainFace._from_trusted(
            active_id=active_id,
            axial_index=axial_index,
            openmc_index=openmc_index,
            direction=direction,
            neighbor_axial_index=neighbor_axial_index,
            neighbor_openmc_index=openmc_index,
            neighbor_active_id=self.active_id_at(neighbor_axial_index, openmc_index),
            kind=DOMAIN_FACE_KIND_INTERNAL,
        )

    def plot_matplotlib(
        self,
        axial_index: int,
        materials: Mapping[str, Material] | None = None,
        ax: Axes | None = None,
    ) -> Axes:
        """Plot one axial material slice with resolved material-key colors.

        ``materials`` optionally supplies ``Material`` definitions by active key;
        its explicit colors are resolved by ``material_colors``. When ``ax`` is
        omitted, the method creates and returns a new Matplotlib axes.

        Raises
        ------
        TypeError
            If ``materials`` is not a mapping of string keys to ``Material``
            values, or ``ax`` is neither an ``Axes`` nor ``None``.
        ValueError
            If a material-mapping key is empty or differs from its
            ``Material.name``.
        """
        axial_index = self._check_axial_index(axial_index)
        if ax is None:
            _, ax = plt.subplots()
        elif not isinstance(ax, Axes):
            raise TypeError("ax must be an Axes or None")
        colors = self.material_colors(materials)
        layer = self.layers[axial_index]
        add_planar_matplotlib_cells(
            self.mesh,
            ax,
            facecolor_for=lambda planar_id: colors[
                layer[self.mesh.openmc_indices[planar_id]]
            ],
        )

        configure_hex_axes(ax)
        z_min, z_max = self.z_bounds(axial_index)
        title = f"MaterialMesh axial slice {axial_index}, z={z_min:g}..{z_max:g} cm"
        ax.set_title(title)
        handles = [
            Patch(facecolor=colors[key], edgecolor="black", label=key)
            for key in self._material_keys_in_order(axial_index)
        ]
        ax.legend(
            handles=handles,
            title="material",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )
        return ax

    def to_plotly(
        self,
        axial_index: int,
        materials: Mapping[str, Material] | None = None,
    ) -> Figure:
        """Return a Plotly figure for one axial material slice.

        ``materials`` optionally supplies ``Material`` definitions by active key.
        Color resolution follows ``material_colors``.

        Raises
        ------
        TypeError
            If ``materials`` is not a mapping of string keys to ``Material``
            values.
        ValueError
            If a material-mapping key is empty or differs from its
            ``Material.name``.
        """
        axial_index = self._check_axial_index(axial_index)
        colors = self.material_colors(materials)
        layer = self.layers[axial_index]
        figure = go.Figure()
        add_planar_plotly_cells(
            self.mesh,
            figure,
            fillcolor_for=lambda planar_id: colors[
                layer[self.mesh.openmc_indices[planar_id]]
            ],
            legendgroup_for=lambda planar_id: layer[
                self.mesh.openmc_indices[planar_id]
            ],
        )
        add_planar_plotly_hover_targets(
            self.mesh,
            figure,
            hover_text_for=lambda planar_id: self._material_hover_label(
                planar_id,
                self.mesh.openmc_indices[planar_id],
                layer[self.mesh.openmc_indices[planar_id]],
                axial_index,
                line_break="<br>",
            ),
            legendgroup_for=lambda planar_id: layer[
                self.mesh.openmc_indices[planar_id]
            ],
        )
        z_min, z_max = self.z_bounds(axial_index)
        title = f"MaterialMesh axial slice {axial_index}, z={z_min:g}..{z_max:g} cm"
        configure_hex_figure(figure, self.mesh, title)
        figure.update_layout(legend={"title": {"text": "material"}})
        return figure

    def export_vtm(self, path: str | Path) -> None:
        """Export active and excluded material cells as VTK multiblock data.

        The ``.vtm`` file references one ``.vtu`` leaf dataset per domain group.
        ParaView can toggle those leaf datasets through the MultiBlock Inspector.
        The adjacent ``material_keys.json`` maps each exported ``material_key_id``
        to its material key.

        Raises
        ------
        ValueError
            If ``path`` does not have a ``.vtm`` suffix.
        IsADirectoryError
            If ``path`` identifies an existing directory.
        OSError
            If an output directory cannot be created or an output file cannot
            be written.
        """
        self._export_vtm_with_cell_data(path)

    def _export_vtm_with_cell_data(
        self,
        path: str | Path,
        extra_cell_data: Mapping[str, tuple[tuple[float, ...], ...]] | None = None,
    ) -> None:
        """Export material data plus optional full-lattice scalar cell data.

        Raises
        ------
        ValueError
            If ``path`` does not have a ``.vtm`` suffix or ``extra_cell_data``
            does not cover every axial layer and planar position.
        IsADirectoryError
            If ``path`` identifies an existing directory.
        OSError
            If an output directory cannot be created or an output file cannot
            be written.
        """
        path = prepare_output_path(path, ".vtm")
        extra_data = dict(extra_cell_data or {})
        for name, layers in extra_data.items():
            if len(layers) != self.n_axial_layers:
                raise ValueError(f"cell data {name!r} must provide every axial layer")
            if any(len(layer) != self.mesh.n_cells for layer in layers):
                raise ValueError(
                    f"cell data {name!r} must provide every planar position"
                )

        active_cells, excluded_cells = self._vtu_cells_by_domain()
        material_ids = self._material_key_ids()
        block_specs = tuple(
            zip(
                DOMAIN_GROUP_NAMES,
                (active_cells, excluded_cells),
                strict=True,
            )
        )
        block_dir = path.with_suffix("")
        block_dir.mkdir(parents=True, exist_ok=True)
        for block_name, cells in block_specs:
            self._replace_output_file(
                block_dir / f"{block_name}.vtu",
                lambda temporary_path, cells=cells: self._write_vtu(
                    temporary_path,
                    cells,
                    material_ids,
                    extra_data,
                ),
            )
        self._replace_output_file(
            block_dir / "material_keys.json",
            lambda temporary_path: self._write_material_key_mapping(
                temporary_path,
                material_ids,
            ),
        )
        self._replace_output_file(
            path,
            lambda temporary_path: write_vtm(
                temporary_path,
                tuple(
                    (
                        block_name,
                        (block_dir / f"{block_name}.vtu")
                        .relative_to(path.parent)
                        .as_posix(),
                    )
                    for block_name, _ in block_specs
                ),
            ),
        )

    @staticmethod
    def _write_material_key_mapping(
        path: Path,
        material_ids: Mapping[str, int],
    ) -> None:
        """Write the companion mapping from exported IDs to material keys."""
        identifier_to_key = {
            str(material_id): material_key
            for material_key, material_id in material_ids.items()
        }
        with path.open("w", encoding="utf-8") as mapping_file:
            json.dump(identifier_to_key, mapping_file, ensure_ascii=False, indent=2)
            mapping_file.write("\n")

    @staticmethod
    def _replace_output_file(path: Path, write: Callable[[Path], None]) -> None:
        """Write a temporary sibling file, then replace ``path`` atomically."""
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}-",
            suffix=".tmp",
            dir=path.parent,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            write(temporary_path)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _write_vtu(
        self,
        path: Path,
        cells: list[tuple[int, int, str]],
        material_ids: Mapping[str, int],
        extra_cell_data: Mapping[str, tuple[tuple[float, ...], ...]],
    ) -> None:
        """Write material-mesh cells as one VTK XML unstructured grid."""
        points: list[tuple[float, float, float]] = []
        connectivity: list[int] = []
        offsets: list[int] = []
        cell_types: list[int] = []
        for axial_index, planar_id, _ in cells:
            z_min, z_max = self.z_bounds(axial_index)
            for x_value, y_value in self.mesh.cell_vertices(planar_id):
                points.append((x_value, y_value, z_min))
                connectivity.append(len(points) - 1)
            for x_value, y_value in self.mesh.cell_vertices(planar_id):
                points.append((x_value, y_value, z_max))
                connectivity.append(len(points) - 1)
            offsets.append(len(connectivity))
            cell_types.append(16)  # VTK_HEXAGONAL_PRISM

        cell_data = [
            VTKDataArray("Int64", "planar_id", (cell[1] for cell in cells)),
            VTKDataArray(
                "Float64",
                "active_id",
                (self._vtu_compact_active_id(*cell) for cell in cells),
            ),
            VTKDataArray(
                "Int64",
                "axial_index",
                (cell[0] for cell in cells),
            ),
            VTKDataArray(
                "Int64",
                "domain_group_id",
                (
                    0 if is_excluded_material_key(cell[2], self.excluded_regions) else 1
                    for cell in cells
                ),
            ),
            VTKDataArray(
                "Int64",
                "material_key_id",
                (material_ids[cell[2]] for cell in cells),
            ),
            VTKDataArray(
                "Int64",
                "lattice_x",
                (self.mesh.lattice_coord(cell[1])[0] for cell in cells),
            ),
            VTKDataArray(
                "Int64",
                "lattice_u",
                (self.mesh.lattice_coord(cell[1])[1] for cell in cells),
            ),
            VTKDataArray(
                "Int64",
                "openmc_ring",
                (self.mesh.openmc_index(cell[1]).ring for cell in cells),
            ),
            VTKDataArray(
                "Int64",
                "openmc_position",
                (self.mesh.openmc_index(cell[1]).position for cell in cells),
            ),
        ]
        for name, layers in extra_cell_data.items():
            cell_data.append(
                VTKDataArray(
                    "Float64",
                    name,
                    tuple(layers[cell[0]][cell[1]] for cell in cells),
                )
            )
        write_vtu(
            path,
            points=points,
            connectivity=connectivity,
            offsets=offsets,
            cell_types=cell_types,
            cell_data=cell_data,
        )

    def material_colors(
        self,
        materials: Mapping[str, Material] | None = None,
    ) -> dict[str, str]:
        """Return resolved plotting colors for every key in the layout.

        Excluded-region colors take precedence and use the region's configured
        color or its default. For active keys, a supplied ``Material.color``
        takes precedence over the repeating default palette. Active palette
        entries are assigned in first-use order while scanning layers bottom to
        top and positions in planar order.

        Raises
        ------
        TypeError
            If ``materials`` is not a mapping of string keys to ``Material``
            values.
        ValueError
            If a material-mapping key is empty or differs from its
            ``Material.name``.
        """
        material_map = self._checked_materials(materials)
        colors: dict[str, str] = {}
        active_color_index = 0
        for material_key in self._material_keys_in_order():
            excluded_color = excluded_key_color(material_key, self.excluded_regions)
            if excluded_color is not None:
                colors[material_key] = excluded_color
                continue
            material = material_map.get(material_key)
            if material is not None and material.color is not None:
                colors[material_key] = material.color
                continue
            colors[material_key] = DEFAULT_MATERIAL_COLORS[
                active_color_index % len(DEFAULT_MATERIAL_COLORS)
            ]
            active_color_index += 1
        return colors

    @staticmethod
    def _checked_materials(
        materials: Mapping[str, Material] | None,
    ) -> dict[str, Material]:
        """Copy optional plotting materials after checking their identities."""
        if materials is None:
            return {}
        if not isinstance(materials, Mapping):
            raise TypeError("materials must be a mapping or None")
        material_map = dict(materials)
        for material_key, material in material_map.items():
            require_nonempty_string("materials key", material_key)
            if not isinstance(material, Material):
                raise TypeError("materials values must be Material objects")
            if material.name != material_key:
                raise ValueError("materials mapping keys must match Material.name")
        return material_map

    def _material_keys_in_order(
        self,
        axial_index: int | None = None,
    ) -> tuple[str, ...]:
        """Return material keys in first-use order."""
        layers = self.layers if axial_index is None else (self.layers[axial_index],)
        keys: list[str] = []
        for layer in layers:
            for openmc_index in self.mesh.openmc_indices:
                key = layer[openmc_index]
                if key not in keys:
                    keys.append(key)
        return tuple(keys)

    def _material_key_ids(self) -> dict[str, int]:
        """Return deterministic numeric IDs for material keys."""
        return {
            material_key: material_id
            for material_id, material_key in enumerate(self._material_keys_in_order())
        }

    def _material_hover_label(
        self,
        planar_id: int,
        openmc_index: OpenMCIndex,
        material_key: str,
        axial_index: int,
        line_break: str,
    ) -> str:
        """Return a Plotly hover label for one material-mesh position."""
        center_x, center_y = self.mesh.cartesian_center(planar_id)
        z_min, z_max = self.z_bounds(axial_index)
        return line_break.join(
            (
                f"material {material_key}",
                f"planar_id {planar_id}",
                f"axial layer {axial_index}",
                f"ring {openmc_index.ring}, pos {openmc_index.position}",
                f"center ({center_x:.2f}, {center_y:.2f}) cm",
                f"z {z_min:.2f}..{z_max:.2f} cm",
                f"domain {self._domain_group_name(material_key)}",
            )
        )

    def _vtu_cells(self) -> list[tuple[int, int, str]]:
        """Return cells ordered by active group, then excluded group."""
        active, excluded = self._vtu_cells_by_domain()
        return active + excluded

    def _vtu_cells_by_domain(
        self,
    ) -> tuple[list[tuple[int, int, str]], list[tuple[int, int, str]]]:
        """Return active and excluded cells in separate domain lists."""
        active: list[tuple[int, int, str]] = []
        excluded: list[tuple[int, int, str]] = []
        for axial_index, layer in enumerate(self.layers):
            for planar_id, openmc_index in enumerate(self.mesh.openmc_indices):
                material_key = layer[openmc_index]
                cell = (axial_index, planar_id, material_key)
                if is_excluded_material_key(material_key, self.excluded_regions):
                    excluded.append(cell)
                else:
                    active.append(cell)
        return active, excluded

    def _vtu_compact_active_id(
        self,
        axial_index: int,
        planar_id: int,
        material_key: str,
    ) -> float:
        """Return slice-local active ID, or NaN for excluded cells."""
        if is_excluded_material_key(material_key, self.excluded_regions):
            return float("nan")
        openmc_index = self.mesh.openmc_index(planar_id)
        active_id = self.active_id_at(axial_index, openmc_index)
        if active_id is None:
            raise ValueError(
                f"active material has no slice-local active ID: {openmc_index}"
            )
        return float(active_id)

    def _domain_group_name(self, material_key: str) -> str:
        """Return the VTK/plotting domain group name."""
        is_excluded = is_excluded_material_key(material_key, self.excluded_regions)
        group_index = int(is_excluded)
        return DOMAIN_GROUP_NAMES[group_index]
