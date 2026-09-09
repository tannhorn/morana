"""Problem-configuration objects for Morana."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from morana.boundary import (
    BoundaryAssignment,
    BoundaryCondition,
    BoundaryConditionSet,
    BoundarySelector,
)
from morana._validation import require_integer, require_optional_string
from morana.material_mesh import MaterialMesh, MaterialSlice
from morana.materials import (
    CrossSections,
    ExcludedRegion,
    FissionData,
    FissionTransfer,
    Material,
    SeparableFission,
    check_material_name,
)
from morana.hex_planar_mesh import HexPlanarMesh, OpenMCIndex
from morana.sources import CellSource, MaterialSource, UniformSource


@dataclass(frozen=True)
class _CrossSectionsRecord:
    """Immutable cross-section definition retained by a configuration snapshot.

    The tuples use the same fast-to-thermal group ordering and units as
    [`CrossSections`][morana.materials.CrossSections]. Use
    ``to_cross_sections()`` to construct an independent checked runtime value.
    """

    D: tuple[float, ...]
    sigma_a: tuple[float, ...]
    sigma_s: tuple[tuple[float, ...], ...]
    multiplicity_matrix: tuple[tuple[float, ...], ...] | None
    fission: _FissionDataRecord | None

    @classmethod
    def from_cross_sections(
        cls, cross_sections: CrossSections
    ) -> "_CrossSectionsRecord":
        """Capture one checked cross-section definition as immutable tuples."""
        return cls(
            D=tuple(float(value) for value in cross_sections.D),
            sigma_a=tuple(float(value) for value in cross_sections.sigma_a),
            sigma_s=tuple(
                tuple(float(value) for value in row) for row in cross_sections.sigma_s
            ),
            multiplicity_matrix=(
                None
                if cross_sections.multiplicity_matrix is None
                else tuple(
                    tuple(float(value) for value in row)
                    for row in cross_sections.multiplicity_matrix
                )
            ),
            fission=(
                None
                if cross_sections.fission is None
                else _FissionDataRecord.from_fission_data(cross_sections.fission)
            ),
        )

    def to_cross_sections(self) -> CrossSections:
        """Return an independent checked runtime cross-section value."""
        return CrossSections(
            D=list(self.D),
            sigma_a=list(self.sigma_a),
            sigma_s=[list(row) for row in self.sigma_s],
            multiplicity_matrix=(
                None
                if self.multiplicity_matrix is None
                else [list(row) for row in self.multiplicity_matrix]
            ),
            fission=None if self.fission is None else self.fission.to_fission_data(),
        )


@dataclass(frozen=True)
class _SeparableFissionRecord:
    """Immutable separable neutron-production data in a snapshot."""

    nu_sigma_f: tuple[float, ...]
    chi: tuple[float, ...]
    chi_normalization_tolerance: float

    @classmethod
    def from_separable_fission(
        cls, fission: SeparableFission
    ) -> "_SeparableFissionRecord":
        """Capture checked separable fission data as immutable tuples."""
        return cls(
            nu_sigma_f=tuple(float(value) for value in fission.nu_sigma_f),
            chi=tuple(float(value) for value in fission.chi),
            chi_normalization_tolerance=fission.chi_normalization_tolerance,
        )

    def to_separable_fission(self) -> SeparableFission:
        """Return an independent checked separable fission value."""
        return SeparableFission(
            nu_sigma_f=list(self.nu_sigma_f),
            chi=list(self.chi),
            chi_normalization_tolerance=self.chi_normalization_tolerance,
        )


@dataclass(frozen=True)
class _FissionTransferRecord:
    """Immutable transfer neutron-production data in a snapshot."""

    fission_transfer: tuple[tuple[float, ...], ...]

    @classmethod
    def from_fission_transfer(
        cls, fission: FissionTransfer
    ) -> "_FissionTransferRecord":
        """Capture checked transfer fission data as immutable tuples."""
        return cls(
            fission_transfer=tuple(
                tuple(float(value) for value in row) for row in fission.fission_transfer
            )
        )

    def to_fission_transfer(self) -> FissionTransfer:
        """Return an independent checked transfer fission value."""
        return FissionTransfer(
            fission_transfer=[list(row) for row in self.fission_transfer]
        )


@dataclass(frozen=True)
class _FissionDataRecord:
    """Immutable fission-physics data in a configuration snapshot."""

    neutron_production: _SeparableFissionRecord | _FissionTransferRecord
    kappa_sigma_f: tuple[float, ...] | None

    @classmethod
    def from_fission_data(cls, fission: FissionData) -> "_FissionDataRecord":
        """Capture one checked fission-physics bundle as immutable tuples."""
        neutron_production = fission.neutron_production
        if isinstance(neutron_production, SeparableFission):
            record = _SeparableFissionRecord.from_separable_fission(neutron_production)
        else:
            record = _FissionTransferRecord.from_fission_transfer(neutron_production)
        return cls(
            neutron_production=record,
            kappa_sigma_f=(
                None
                if fission.kappa_sigma_f is None
                else tuple(float(value) for value in fission.kappa_sigma_f)
            ),
        )

    def to_fission_data(self) -> FissionData:
        """Return an independent checked fission-physics bundle."""
        return FissionData(
            neutron_production=(
                self.neutron_production.to_separable_fission()
                if isinstance(self.neutron_production, _SeparableFissionRecord)
                else self.neutron_production.to_fission_transfer()
            ),
            kappa_sigma_f=(
                None if self.kappa_sigma_f is None else list(self.kappa_sigma_f)
            ),
        )


@dataclass(frozen=True)
class _MaterialRecord:
    """Immutable material definition retained by a configuration snapshot."""

    name: str
    xs: _CrossSectionsRecord | None
    color: str | None

    @classmethod
    def from_material(cls, material: Material) -> "_MaterialRecord":
        """Capture one material independently of its runtime object."""
        return cls(
            name=material.name,
            xs=(
                None
                if material.xs is None
                else _CrossSectionsRecord.from_cross_sections(material.xs)
            ),
            color=material.color,
        )

    def to_material(self) -> Material:
        """Return an independent checked runtime material value."""
        return Material(
            name=self.name,
            xs=None if self.xs is None else self.xs.to_cross_sections(),
            color=self.color,
        )


@dataclass(frozen=True)
class _UniformSourceRecord:
    """Immutable definition of a uniform fixed source."""

    strength: tuple[float, ...]

    def to_source(self) -> UniformSource:
        """Return an independent runtime source."""
        return UniformSource(list(self.strength))


@dataclass(frozen=True)
class _MaterialSourceRecord:
    """Immutable definition of a material-wise fixed source."""

    values_by_material: tuple[tuple[str, tuple[float, ...]], ...]

    def to_source(self) -> MaterialSource:
        """Return an independent runtime source."""
        return MaterialSource(
            {material: list(values) for material, values in self.values_by_material}
        )


@dataclass(frozen=True)
class _CellSourceRecord:
    """Immutable definition of an explicit layered fixed source."""

    layers: tuple[tuple[tuple[float, ...], ...], ...]

    def to_source(self) -> CellSource:
        """Return an independent runtime source."""
        return CellSource(
            tuple([list(group) for group in layer] for layer in self.layers)
        )


_SourceRecord = _UniformSourceRecord | _MaterialSourceRecord | _CellSourceRecord


def _snapshot_source(
    source: UniformSource | MaterialSource | CellSource | None,
) -> _SourceRecord | None:
    """Capture one built-in source as a fully independent immutable value."""
    if source is None:
        return None
    if isinstance(source, UniformSource):
        return _UniformSourceRecord(tuple(float(value) for value in source.strength))
    if isinstance(source, MaterialSource):
        return _MaterialSourceRecord(
            tuple(
                (material, tuple(float(value) for value in values))
                for material, values in sorted(source.values_by_material.items())
            )
        )
    if isinstance(source, CellSource):
        return _CellSourceRecord(
            tuple(
                tuple(tuple(float(value) for value in group) for group in layer)
                for layer in source.layers
            )
        )
    raise ValueError("checked source has an unsupported type")


@dataclass(frozen=True)
class _BoundaryConditionRecord:
    """Deeply owned primitive representation of boundary physics."""

    kind: str
    flux: tuple[float, ...] | None
    alpha: float | None
    beta: float | None
    current: tuple[float, ...] | None

    @classmethod
    def from_condition(cls, condition: BoundaryCondition) -> "_BoundaryConditionRecord":
        """Capture one boundary condition without retaining its arrays."""
        return cls(
            kind=condition.kind,
            flux=(
                None
                if condition.flux is None
                else tuple(float(value) for value in condition.flux)
            ),
            alpha=condition.alpha,
            beta=condition.beta,
            current=(
                None
                if condition.current is None
                else tuple(float(value) for value in condition.current)
            ),
        )

    def to_condition(self) -> BoundaryCondition:
        """Restore a fresh checked boundary condition."""
        return BoundaryCondition(
            kind=self.kind,
            flux=None if self.flux is None else list(self.flux),
            alpha=self.alpha,
            beta=self.beta,
            current=None if self.current is None else list(self.current),
        )


@dataclass(frozen=True)
class _BoundaryAssignmentRecord:
    """Deeply owned primitive representation of one boundary assignment."""

    scope: str
    direction: str | None
    excluded_key: str | None
    excluded_kind: str | None
    condition: _BoundaryConditionRecord

    @classmethod
    def from_assignment(
        cls, assignment: BoundaryAssignment
    ) -> "_BoundaryAssignmentRecord":
        """Capture one selector and condition without live references."""
        selector = assignment.selector
        return cls(
            scope=selector.scope,
            direction=selector.direction,
            excluded_key=selector.excluded_key,
            excluded_kind=selector.excluded_kind,
            condition=_BoundaryConditionRecord.from_condition(assignment.condition),
        )

    def to_assignment(self) -> BoundaryAssignment:
        """Restore one fresh checked boundary assignment."""
        return BoundaryAssignment(
            selector=BoundarySelector(
                scope=self.scope,
                direction=self.direction,
                excluded_key=self.excluded_key,
                excluded_kind=self.excluded_kind,
            ),
            condition=self.condition.to_condition(),
        )


@dataclass(frozen=True)
class _AxialLayerRecord:
    """Immutable material-layout provenance for one axial layer.

    Parameters
    ----------
    material_keys
        Complete material-key sequence in the parent mesh's OpenMC ordering.
    ``material_keys`` follows the parent mesh's public OpenMC-index ordering.
    The compact active-cell mapping is derived from these keys when the public
    material mesh is reconstructed.
    """

    material_keys: tuple[str, ...]


@dataclass(frozen=True)
class _ExcludedRegionRecord:
    """Primitive excluded-region definition retained in result provenance.

    Parameters
    ----------
    key
        Material-mesh key.
    kind
        Excluded-region classification.
    color
        Optional plotting color.
    """

    key: str
    kind: str
    color: str | None


@dataclass(init=False)
# pylint: disable=too-many-instance-attributes
class ProblemConfiguration:
    """In-memory representation of a neutronics problem definition.

    Parameters
    ----------
    mesh
        Hexagonal mesh topology and geometry.
    materials
        Mapping from public material name to material object. Every mapping key
        must match its ``Material.name``.
    material_mesh
        Material-key layout over the full lattice. This defines the active
        solution domain and must use ``mesh``. Every active key must identify
        an entry in ``materials``; material names cannot collide with
        excluded-region keys.
    boundary
        Boundary-condition assignments for exposed mesh faces. The default is
        empty; a solve requires complete boundary coverage.
    source
        Optional volumetric fixed-source object. A fixed-source solve may use
        this source, an inhomogeneous boundary contribution, or both;
        eigenvalue solves reject an external source.
    name
        Optional nonempty configuration name for provenance and serialization.

    Raises
    ------
    TypeError
        If an input does not have its documented type.
    ValueError
        If checked values are incompatible, a name is invalid, or a material
        layout refers to an unknown material.

    Notes
    -----
    Public state is read-only. Use ``set_materials()``, ``replace_material()``,
    ``set_material_mesh()``, ``set_boundary()``, ``add_boundary()``,
    ``assign_material()``, ``set_source()``, and ``set_name()`` to change an
    existing configuration.
    """

    _mesh: HexPlanarMesh
    _materials: dict[str, Material]
    _material_mesh: MaterialMesh
    _boundary: BoundaryConditionSet
    _source: UniformSource | MaterialSource | CellSource | None
    _name: str | None

    def __init__(
        self,
        mesh: HexPlanarMesh,
        materials: Mapping[str, Material],
        material_mesh: MaterialMesh,
        *,
        boundary: BoundaryConditionSet | None = None,
        source: UniformSource | MaterialSource | CellSource | None = None,
        name: str | None = None,
    ) -> None:
        """Check and own one mutable problem definition."""
        if not isinstance(mesh, HexPlanarMesh):
            raise TypeError("mesh must be a HexPlanarMesh")
        checked_material_mesh = self._checked_material_mesh(mesh, material_mesh)
        checked_materials = self._checked_materials(materials, checked_material_mesh)
        self._mesh = mesh
        self._materials = checked_materials
        self._material_mesh = checked_material_mesh
        self._boundary = (
            BoundaryConditionSet()
            if boundary is None
            else self._checked_boundary(boundary)
        )
        self._source = self._checked_source(source)
        self._name = require_optional_string("name", name)

    @property
    def mesh(self) -> HexPlanarMesh:
        """Return the immutable planar geometry for this configuration."""
        return self._mesh

    @property
    def materials(self) -> Mapping[str, Material]:
        """Return the read-only name-to-immutable-material mapping."""
        return MappingProxyType(self._materials)

    @property
    def material_mesh(self) -> MaterialMesh:
        """Return the immutable material layout for this configuration."""
        return self._material_mesh

    @property
    def unused_material_names(self) -> frozenset[str]:
        """Return configured material names absent from every active mesh cell.

        The returned set is derived from all axial layers of ``material_mesh``.
        Excluded positions do not use material definitions.
        """
        used_names = set()
        for axial_index in range(self._material_mesh.n_axial_layers):
            used_names.update(
                self._material_mesh.material_by_active_id(axial_index).values()
            )
        return frozenset(self._materials).difference(used_names)

    @property
    def boundary(self) -> BoundaryConditionSet:
        """Return the immutable boundary-assignment set."""
        return self._boundary

    @property
    def source(self) -> UniformSource | MaterialSource | CellSource | None:
        """Return the optional fixed-source definition."""
        return self._source

    @property
    def name(self) -> str | None:
        """Return the optional configuration provenance name."""
        return self._name

    @staticmethod
    def _checked_boundary(
        boundary: BoundaryConditionSet,
    ) -> BoundaryConditionSet:
        """Return one checked explicit boundary set."""
        if not isinstance(boundary, BoundaryConditionSet):
            raise TypeError("boundary must be a BoundaryConditionSet")
        return boundary

    @staticmethod
    def _checked_source(
        source: UniformSource | MaterialSource | CellSource | None,
    ) -> UniformSource | MaterialSource | CellSource | None:
        """Return one checked optional built-in fixed-source definition."""
        if source is not None and type(source) not in (
            UniformSource,
            MaterialSource,
            CellSource,
        ):
            raise TypeError(
                "source must be UniformSource, MaterialSource, CellSource, or None"
            )
        return source

    @staticmethod
    def _checked_material_mesh(
        mesh: HexPlanarMesh, material_mesh: MaterialMesh
    ) -> MaterialMesh:
        """Require a material layout over the configuration geometry."""
        if not isinstance(material_mesh, MaterialMesh):
            raise TypeError("material_mesh must be a MaterialMesh")
        if material_mesh.mesh != mesh:
            raise ValueError("material_mesh must be defined on configuration mesh")
        return material_mesh

    @classmethod
    def _checked_materials(
        cls, materials: Mapping[str, Material], material_mesh: MaterialMesh
    ) -> dict[str, Material]:
        """Copy and validate material definitions against one layout."""
        if not isinstance(materials, Mapping):
            raise TypeError("materials must be a mapping")
        normalized = dict(materials)
        for name, material in normalized.items():
            check_material_name(name)
            if not isinstance(material, Material):
                raise TypeError("materials values must be Material objects")
            if material.name != name:
                raise ValueError("materials mapping keys must match Material.name")
            if name in material_mesh.excluded_regions:
                raise ValueError(
                    f"material name {name!r} collides with an excluded-region key"
                )
        cls._check_material_assignments(material_mesh, normalized)
        return normalized

    def set_boundary(self, boundary: BoundaryConditionSet) -> None:
        """Replace all boundary assignments.

        Parameters
        ----------
        boundary
            Complete replacement boundary-condition set.

        Raises
        ------
        TypeError
            If ``boundary`` is not a ``BoundaryConditionSet``.
        """
        self._boundary = self._checked_boundary(boundary)

    def set_materials(self, materials: Mapping[str, Material]) -> None:
        """Replace all material definitions.

        Parameters
        ----------
        materials
            Complete name-to-material replacement mapping. Every key must
            match its immutable ``Material.name``, every active layout key
            must remain present, and no name may collide with an excluded key.

        Raises
        ------
        TypeError
            If ``materials`` is not a mapping, a key is not a string, or a
            value is not a ``Material``.
        ValueError
            If a name is invalid, a key does not match ``Material.name``, a
            name collides with an excluded key, or an active layout key is
            absent from the replacement mapping.
        """
        self._materials = self._checked_materials(materials, self._material_mesh)

    def replace_material(self, material: Material) -> None:
        """Replace one existing material definition.

        Parameters
        ----------
        material
            Replacement immutable material. Its name must already identify a
            material in this configuration.

        Raises
        ------
        TypeError
            If ``material`` is not a ``Material``.
        ValueError
            If its name is not present.
        """
        if not isinstance(material, Material):
            raise TypeError("replacement material must be a Material")
        if material.name not in self._materials:
            raise ValueError(f"unknown material: {material.name!r}")
        materials = dict(self._materials)
        materials[material.name] = material
        self.set_materials(materials)

    def set_material_mesh(self, material_mesh: MaterialMesh) -> None:
        """Replace the full material layout.

        Parameters
        ----------
        material_mesh
            Completed immutable material layout over this configuration's
            planar mesh. Every active material key must identify a configured
            material.

        Raises
        ------
        TypeError
            If ``material_mesh`` is not a ``MaterialMesh``.
        ValueError
            If it uses different geometry or assigns an unknown material.
        """
        checked_material_mesh = self._checked_material_mesh(self._mesh, material_mesh)
        self._checked_materials(self._materials, checked_material_mesh)
        self._material_mesh = checked_material_mesh

    def add_boundary(self, assignment: BoundaryAssignment) -> None:
        """Add one boundary assignment.

        Parameters
        ----------
        assignment
            Assignment appended through ``BoundaryConditionSet.with_assignment``.
            Its selector is checked for duplicate identity and its condition is
            retained by the resulting immutable boundary set.

        Raises
        ------
        TypeError
            If ``assignment`` is not a ``BoundaryAssignment``.
        ValueError
            If its selector duplicates an existing assignment.
        """
        self._boundary = self._boundary.with_assignment(assignment)

    def check_boundary_coverage(self) -> None:
        """Require conditions for every exposed face of the configured layout.

        Raises
        ------
        ValueError
            If one or more physical-exterior or excluded-region faces are
            uncovered.
        """
        self._boundary.check_coverage(self._material_mesh)

    def check_no_unused_materials(self) -> None:
        """Require every configured material to occur in an active mesh cell.

        Raises
        ------
        ValueError
            If one or more configured material names are absent from every
            active cell in every axial layer.
        """
        unused_names = self.unused_material_names
        if unused_names:
            formatted_names = ", ".join(repr(name) for name in sorted(unused_names))
            raise ValueError(f"unused material names: {formatted_names}")

    def assign_material(
        self,
        axial_index: int,
        openmc_index: OpenMCIndex,
        material: str,
    ) -> None:
        """Assign a known material to one axial-layer lattice position.

        Parameters
        ----------
        axial_index
            Bottom-to-top axial-layer index.
        openmc_index
            Public OpenMC-style position in the shared planar mesh.
        material
            Public name of a material in ``materials``.

        Raises
        ------
        TypeError
            If ``axial_index`` is not an integer, ``openmc_index`` is not an
            ``OpenMCIndex``, or ``material`` is not a string.
        ValueError
            If ``material`` is invalid or unknown, or either location index
            is invalid.

        Notes
        -----
        The replacement updates only the specified layer. The previous
        immutable material mesh remains unchanged.
        """
        axial_index = require_integer("axial_index", axial_index)
        if not isinstance(openmc_index, OpenMCIndex):
            raise TypeError("openmc_index must be an OpenMCIndex")
        check_material_name(material)
        if material not in self._materials:
            raise ValueError(f"unknown material: {material!r}")
        self._material_mesh = (
            self._material_mesh._with_assignment(  # pylint: disable=protected-access
                axial_index, openmc_index, material
            )
        )

    def set_source(
        self,
        source: UniformSource | MaterialSource | CellSource | None,
    ) -> None:
        """Set or clear the fixed-source definition.

        Parameters
        ----------
        source
            Volumetric fixed-source definition, or ``None`` to clear it.
            Fixed-source solves may instead be driven by inhomogeneous boundary
            data; eigenvalue solves reject an external source.

        Raises
        ------
        TypeError
            If ``source`` is not a built-in source or ``None``.
        """
        self._source = self._checked_source(source)

    def set_name(self, name: str | None) -> None:
        """Replace the provenance name.

        Raises
        ------
        TypeError
            If ``name`` is neither a string nor ``None``.
        ValueError
            If ``name`` is empty.
        """
        self._name = require_optional_string("name", name)

    def snapshot(self) -> "ProblemConfigurationSnapshot":
        """Return an immutable, non-aliasing problem-definition snapshot.

        Returns
        -------
        ProblemConfigurationSnapshot
            Non-aliasing snapshot of geometry, material layout, material and
            excluded-region definitions, boundary assignments, source, and
            name. Its public values are freshly reconstructed, not live
            references to this mutable configuration. It can be passed to a
            supported solve function or retained as result provenance.
        """
        return ProblemConfigurationSnapshot._from_configuration(  # pylint: disable=protected-access
            self
        )

    @staticmethod
    def _check_material_assignments(
        material_mesh: MaterialMesh, materials: Mapping[str, Material]
    ) -> None:
        """Require known active materials in every axial material-mesh layer."""
        for axial_index in range(material_mesh.n_axial_layers):
            assignments = material_mesh.material_by_active_id(axial_index)
            for active_id, material in assignments.items():
                if material not in materials:
                    raise ValueError(
                        "unknown material in material_mesh: "
                        f"{material!r} at axial_index={axial_index}, "
                        f"active_id={active_id}"
                    )


@dataclass(frozen=True)
class _ConfigurationRecord:
    """Deeply owned configuration state for one public snapshot."""

    mesh_num_rings: int
    mesh_pitch: float
    axial_layer_heights: tuple[float, ...]
    layers: tuple[_AxialLayerRecord, ...]
    excluded_regions: tuple[_ExcludedRegionRecord, ...]
    boundary_assignments: tuple[_BoundaryAssignmentRecord, ...]
    materials: tuple[_MaterialRecord, ...]
    source: _SourceRecord | None
    name: str | None


@dataclass(frozen=True, init=False)
class ProblemConfigurationSnapshot:
    """Immutable, non-aliasing problem definition and result provenance.

    The snapshot privately owns geometry, material-layout, definition,
    boundary, source, and provenance records. Its public properties restore
    fresh immutable Morana values on each access, never a live object or NumPy
    array from the configuration it captures. Pass it to a supported solve
    function as immutable input, retain it as result provenance, or call
    ``to_configuration()`` to create an independent mutable problem owner.

    ``UniformSource``, ``MaterialSource``, and ``CellSource`` are the complete
    supported fixed-source set. Each is snapshotable and reconstructible.
    """

    _record: _ConfigurationRecord

    @classmethod
    def _from_configuration(
        cls, configuration: ProblemConfiguration
    ) -> "ProblemConfigurationSnapshot":
        """Capture one configuration without retaining public object references."""
        material_mesh = configuration.material_mesh
        record = _ConfigurationRecord(
            mesh_num_rings=configuration.mesh.num_rings,
            mesh_pitch=configuration.mesh.pitch,
            axial_layer_heights=material_mesh.axial_layer_heights,
            layers=tuple(
                _AxialLayerRecord(
                    material_keys=tuple(
                        layer[index] for index in configuration.mesh.openmc_indices
                    ),
                )
                for layer in material_mesh.layers
            ),
            excluded_regions=tuple(
                _ExcludedRegionRecord(key, region.kind, region.color)
                for key, region in sorted(material_mesh.excluded_regions.items())
            ),
            boundary_assignments=tuple(
                _BoundaryAssignmentRecord.from_assignment(assignment)
                for assignment in configuration.boundary.assignments
            ),
            materials=tuple(
                _MaterialRecord.from_material(material)
                for _, material in sorted(configuration.materials.items())
            ),
            source=_snapshot_source(configuration.source),
            name=configuration.name,
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "_record", record)
        return instance

    @property
    def name(self) -> str | None:
        """Return the captured optional configuration name."""
        return self._record.name

    @property
    def mesh(self) -> HexPlanarMesh:
        """Return a fresh immutable planar geometry value."""
        return HexPlanarMesh(self._record.mesh_num_rings, self._record.mesh_pitch)

    @property
    def materials(self) -> Mapping[str, Material]:
        """Return a fresh read-only mapping of immutable material values."""
        return MappingProxyType(self._material_map())

    @property
    def material_mesh(self) -> MaterialMesh:
        """Return a fresh immutable material-layout value."""
        mesh = self.mesh
        return self._material_mesh_for(mesh)

    @property
    def boundary(self) -> BoundaryConditionSet:
        """Return a fresh immutable boundary-assignment set."""
        return BoundaryConditionSet(
            *(
                assignment.to_assignment()
                for assignment in self._record.boundary_assignments
            )
        )

    @property
    def source(self) -> UniformSource | MaterialSource | CellSource | None:
        """Return a fresh built-in source definition, when present."""
        if self._record.source is None:
            return None
        return self._record.source.to_source()

    def excluded_region_map(self) -> dict[str, ExcludedRegion]:
        """Return fresh excluded-region descriptions keyed by layout key."""
        return {
            entry.key: ExcludedRegion(entry.kind, entry.color)
            for entry in self._record.excluded_regions
        }

    def _material_map(self) -> dict[str, Material]:
        """Restore fresh material values from private records."""
        materials = {
            material.name: material.to_material() for material in self._record.materials
        }
        if len(materials) != len(self._record.materials):
            raise ValueError("snapshot material definitions must have unique names")
        return materials

    def _material_mesh_for(self, mesh: HexPlanarMesh) -> MaterialMesh:
        """Restore a material layout on one supplied mesh value."""
        if len(self._record.axial_layer_heights) != len(self._record.layers):
            raise ValueError("snapshot layers must match axial layer heights")
        if any(
            len(layer.material_keys) != mesh.n_cells for layer in self._record.layers
        ):
            raise ValueError("snapshot material keys must cover the complete mesh")
        slices = tuple(
            MaterialSlice(
                mesh=mesh,
                material_keys=dict(zip(mesh.openmc_indices, layer.material_keys)),
                height=height,
            )
            for layer, height in zip(
                self._record.layers, self._record.axial_layer_heights
            )
        )
        return MaterialMesh.stack(slices, excluded_regions=self.excluded_region_map())

    def to_configuration(self) -> ProblemConfiguration:
        """Reconstruct an independent mutable configuration owner."""
        mesh = self.mesh
        return ProblemConfiguration(
            mesh=mesh,
            materials=self._material_map(),
            material_mesh=self._material_mesh_for(mesh),
            boundary=self.boundary,
            source=self.source,
            name=self.name,
        )
