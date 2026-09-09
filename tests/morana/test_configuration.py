"""Tests for problem-configuration ownership and mutation boundaries."""

import numpy as np
import pytest

from morana import (
    BoundaryCondition,
    BoundaryConditionSet,
    CellSource,
    CrossSections,
    ExcludedRegion,
    FissionData,
    FissionTransfer,
    HexPlanarMesh,
    Material,
    MaterialMesh,
    MaterialSlice,
    MaterialSource,
    OpenMCIndex,
    ProblemConfiguration,
    SeparableFission,
    UniformSource,
)
from morana.solvers.finite_volume import solve_fixed_source


def make_configuration() -> ProblemConfiguration:
    """Create a small one-material configuration for skeleton tests."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    xs = CrossSections(
        D=[1.2],
        sigma_a=[0.02],
        sigma_s=[[0.0]],
        fission=None,
    )
    medium = Material(name="medium", xs=xs)
    return ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),),
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
        source=UniformSource([1.0]),
    )


def test_configuration_owns_problem_definition_mutation() -> None:
    """Configuration mutation should update the owned problem definition."""
    configuration = make_configuration()

    configuration.set_boundary(
        BoundaryConditionSet(BoundaryCondition.vacuum().globally())
    )

    assert configuration.boundary.assignments[0].condition.kind == "vacuum"
    assert configuration.material_mesh.material_by_active_id(0) == {0: "medium"}


def test_configuration_public_state_is_read_only() -> None:
    """Invariant-bearing configuration properties must not be directly assigned."""
    configuration = make_configuration()

    with pytest.raises(AttributeError):
        configuration.source = None
    with pytest.raises(AttributeError):
        configuration.materials = {}
    with pytest.raises(TypeError):
        configuration.materials["other"] = Material(name="other")


def test_configuration_reports_and_checks_unused_materials() -> None:
    """Unused definitions should be inspectable and optionally rejectable."""
    configuration = make_configuration()
    assert configuration.unused_material_names == frozenset()
    configuration.check_no_unused_materials()

    configuration.set_materials(
        {**configuration.materials, "spare": Material(name="spare")}
    )

    assert configuration.unused_material_names == frozenset({"spare"})
    with pytest.raises(ValueError, match="unused material names: 'spare'"):
        configuration.check_no_unused_materials()

    configuration.assign_material(0, OpenMCIndex(0, 0), "spare")

    assert configuration.unused_material_names == frozenset({"medium"})
    with pytest.raises(ValueError, match="unused material names: 'medium'"):
        configuration.check_no_unused_materials()


def test_configuration_requires_material_mapping() -> None:
    """Material definitions should not defer a wrong container type to iteration."""
    configuration = make_configuration()
    with pytest.raises(TypeError, match="materials must be a mapping"):
        configuration.set_materials([("medium", configuration.materials["medium"])])  # type: ignore[arg-type]


def test_configuration_public_inputs_reject_wrong_types() -> None:
    """Configuration APIs should reject invalid types before using their values."""
    configuration = make_configuration()
    mesh = configuration.mesh
    material = configuration.materials["medium"]
    material_mesh = configuration.material_mesh

    with pytest.raises(TypeError, match="mesh must be a HexPlanarMesh"):
        ProblemConfiguration(
            "mesh", {"medium": material}, material_mesh
        )  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="material_mesh must be a MaterialMesh"):
        ProblemConfiguration(
            mesh, {"medium": material}, "material mesh"
        )  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="boundary must be a BoundaryConditionSet"):
        ProblemConfiguration(
            mesh,
            {"medium": material},
            material_mesh,
            boundary="boundary",
        )  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="source must be UniformSource"):
        ProblemConfiguration(
            mesh,
            {"medium": material},
            material_mesh,
            source="source",
        )  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="materials values must be Material objects"):
        ProblemConfiguration(
            mesh, {"medium": "material"}, material_mesh
        )  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="boundary must be a BoundaryConditionSet"):
        configuration.set_boundary(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="material_mesh must be a MaterialMesh"):
        configuration.set_material_mesh("material mesh")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="replacement material must be a Material"):
        configuration.replace_material("material")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="source must be UniformSource"):
        configuration.set_source("source")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="axial_index must be an integer"):
        configuration.assign_material(True, OpenMCIndex(0, 0), "medium")
    with pytest.raises(TypeError, match="openmc_index must be an OpenMCIndex"):
        configuration.assign_material(0, (0, 0), "medium")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="material name must be a string"):
        configuration.assign_material(0, OpenMCIndex(0, 0), 1)  # type: ignore[arg-type]


def test_configuration_rejects_custom_source_definitions() -> None:
    """Only the closed built-in source set may define a fixed source."""

    class CustomSource:
        """Mimic the former source-extension protocol."""

        def values(self, axial_index: int, material_by_active_id: object) -> np.ndarray:
            _ = axial_index, material_by_active_id
            return np.array([[1.0]])

    class DerivedUniformSource(UniformSource):
        """Mimic a source variation through inheritance."""

    configuration = make_configuration()

    for source in (CustomSource(), DerivedUniformSource([1.0])):
        with pytest.raises(TypeError, match="source must be UniformSource"):
            configuration.set_source(source)  # type: ignore[arg-type]


def test_configuration_rejects_empty_provenance_name() -> None:
    """Optional configuration labels should not preserve an empty string."""
    mesh = HexPlanarMesh(1, 10.0)
    with pytest.raises(ValueError, match="name must not be empty"):
        ProblemConfiguration(
            mesh=mesh,
            materials={},
            material_mesh=MaterialMesh.stack((MaterialSlice(mesh, {}, 1.0),)),
            name="",
        )


def test_configuration_replacement_methods_validate() -> None:
    """Checked configuration replacements should preserve complete invariants."""
    configuration = make_configuration()
    mesh = configuration.mesh
    replacement = Material(
        name="medium",
        xs=CrossSections(
            D=[1.0],
            sigma_a=[0.01],
            sigma_s=[[0.0]],
            fission=None,
        ),
    )

    configuration.replace_material(replacement)
    assert configuration.materials["medium"] is replacement

    configuration.set_materials({"medium": replacement})

    replacement_mesh = MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),
            MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=2.0),
        )
    )
    configuration.set_material_mesh(replacement_mesh)
    assert configuration.material_mesh is replacement_mesh

    configuration.set_name("replacement case")
    assert configuration.name == "replacement case"


def test_configuration_allows_incremental_boundary_construction() -> None:
    """An incomplete configuration should accept boundary assignments later."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    medium = Material(
        name="medium",
        xs=CrossSections(
            D=[1.2],
            sigma_a=[0.02],
            sigma_s=[[0.0]],
            fission=None,
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),),
        ),
        source=UniformSource([1.0]),
    )

    assert configuration.boundary.assignments == ()
    configuration.add_boundary(BoundaryCondition.reflective().globally())
    configuration.check_boundary_coverage()


def test_configuration_preserves_source_definition_without_layer_inference() -> None:
    """Source evaluation selects its material context and axial layer explicitly."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    medium = Material(
        name="medium",
        xs=CrossSections(
            D=[1.2],
            sigma_a=[0.02],
            sigma_s=[[0.0]],
            fission=None,
        ),
    )
    material_mesh = MaterialMesh.stack(
        (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),),
    )
    source = MaterialSource({"medium": [2.0]})
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=material_mesh,
        source=source,
    )

    assert configuration.source is source
    np.testing.assert_allclose(
        source.values(0, material_mesh.material_by_active_id(0)), [[2.0]]
    )


def test_configuration_rejects_material_excluded_key_collision() -> None:
    """Active material names should not collide with excluded-region keys."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    xs = CrossSections(
        D=[1.2],
        sigma_a=[0.02],
        sigma_s=[[0.0]],
        fission=None,
    )
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(
                mesh,
                [["medium", "gap", "medium", "0", "medium", "gap"], ["medium"]],
                height=1.0,
            ),
        ),
        excluded_regions={"gap": ExcludedRegion(kind="void")},
    )

    with pytest.raises(ValueError, match="collides"):
        ProblemConfiguration(
            mesh=mesh,
            materials={
                "medium": Material(name="medium", xs=xs),
                "gap": Material(name="gap", xs=xs),
            },
            material_mesh=material_mesh,
            boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
        )


def test_configuration_rejects_unknown_material_in_later_layer() -> None:
    """Configuration checking should cover every axial material layer."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    medium = Material(
        name="medium",
        xs=CrossSections(
            D=[1.2],
            sigma_a=[0.02],
            sigma_s=[[0.0]],
            fission=None,
        ),
    )
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice(mesh, {OpenMCIndex(0, 0): "medium"}, 1.0),
            MaterialSlice(mesh, {OpenMCIndex(0, 0): "ghost"}, 1.0),
        )
    )

    with pytest.raises(
        ValueError,
        match="'ghost' at axial_index=1, active_id=0",
    ):
        ProblemConfiguration(
            mesh=mesh,
            materials={"medium": medium},
            material_mesh=material_mesh,
        )


def test_configuration_snapshot_preserves_layered_active_assignments() -> None:
    """Snapshot layout should retain its explicit axial-layer identity."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    medium = Material(
        name="medium",
        xs=CrossSections(
            D=[1.2],
            sigma_a=[0.02],
            sigma_s=[[0.0]],
            fission=None,
        ),
    )
    reflector = Material(
        name="reflector",
        xs=CrossSections(
            D=[0.8],
            sigma_a=[0.01],
            sigma_s=[[0.0]],
            fission=None,
        ),
    )
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice(mesh, {OpenMCIndex(1, 0): "medium"}, 1.0),
            MaterialSlice(
                mesh,
                {
                    OpenMCIndex(0, 0): "reflector",
                    OpenMCIndex(1, 0): "medium",
                },
                1.0,
            ),
        )
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium, "reflector": reflector},
        material_mesh=material_mesh,
    )

    snapshot = configuration.snapshot()
    configuration.assign_material(
        1,
        OpenMCIndex(0, 0),
        "medium",
    )

    snapshot_mesh = snapshot.material_mesh
    assert tuple(
        tuple(layer[index] for index in snapshot.mesh.openmc_indices)
        for layer in snapshot_mesh.layers
    ) == (
        ("0", "0", "0", "0", "0", "0", "medium"),
        ("reflector", "0", "0", "0", "0", "0", "medium"),
    )
    assert snapshot_mesh.material_by_active_id(0) == {0: "medium"}
    assert snapshot_mesh.material_by_active_id(1) == {
        0: "reflector",
        1: "medium",
    }


def test_configuration_snapshot_reconstructs_an_independent_configuration() -> None:
    """A snapshot should retain every configuration input needed to rebuild it."""
    configuration = make_configuration()
    original_result = solve_fixed_source(configuration)

    configuration.set_source(UniformSource([3.0]))

    reconstructed = original_result.configuration_snapshot.to_configuration()

    assert reconstructed is not configuration
    assert reconstructed.name == configuration.name
    assert reconstructed.mesh == configuration.mesh
    assert reconstructed.material_mesh.layers == configuration.material_mesh.layers
    assert reconstructed.boundary.assignments == configuration.boundary.assignments
    assert isinstance(reconstructed.source, UniformSource)
    np.testing.assert_allclose(reconstructed.source.strength, [1.0])

    reconstructed_result = solve_fixed_source(reconstructed)
    np.testing.assert_allclose(
        reconstructed_result.flux_layer(0), original_result.flux_layer(0)
    )
    assert reconstructed.materials["medium"] is not configuration.materials["medium"]
    assert (
        reconstructed.materials["medium"].xs is not configuration.materials["medium"].xs
    )
    np.testing.assert_allclose(
        reconstructed.materials["medium"].xs.D,
        [1.2],
    )


def test_configuration_snapshot_restores_fresh_immutable_values() -> None:
    """Snapshot access should never share numerical storage with its record."""
    configuration = make_configuration()
    configuration.set_boundary(
        BoundaryConditionSet(BoundaryCondition.dirichlet([1.0]).globally())
    )
    snapshot = configuration.snapshot()

    first = snapshot.materials["medium"]
    first.xs.D.setflags(write=True)
    first.xs.D[0] = 9.0

    second = snapshot.materials["medium"]
    np.testing.assert_allclose(second.xs.D, [1.2])
    assert first is not second

    boundary = snapshot.boundary.assignments[0].condition
    boundary.flux.setflags(write=True)
    boundary.flux[0] = 9.0
    np.testing.assert_allclose(snapshot.boundary.assignments[0].condition.flux, [1.0])

    source = snapshot.source
    source.strength.setflags(write=True)
    source.strength[0] = 9.0
    np.testing.assert_allclose(snapshot.source.strength, [1.0])


def test_configuration_snapshot_preserves_scattering_multiplicity() -> None:
    """Snapshot reconstruction should retain optional scattering multiplicity."""
    configuration = make_configuration()
    configuration.replace_material(
        Material(
            "medium",
            CrossSections(
                D=[1.2],
                sigma_a=[0.02],
                sigma_s=[[0.1]],
                fission=None,
                multiplicity_matrix=[[2.0]],
            ),
        )
    )

    restored = configuration.snapshot().to_configuration().materials["medium"].xs

    assert restored is not None
    np.testing.assert_allclose(restored.multiplicity_matrix, [[2.0]])


@pytest.mark.parametrize(
    "neutron_production",
    (
        SeparableFission(nu_sigma_f=[0.025], chi=[1.0]),
        FissionTransfer(fission_transfer=[[0.025]]),
    ),
)
def test_configuration_snapshot_preserves_selected_fission_representation(
    neutron_production: SeparableFission | FissionTransfer,
) -> None:
    """Snapshots should rebuild the caller-selected fission representation."""
    configuration = make_configuration()
    configuration.replace_material(
        Material(
            "medium",
            CrossSections(
                D=[1.2],
                sigma_a=[0.02],
                sigma_s=[[0.0]],
                fission=FissionData(neutron_production=neutron_production),
            ),
        )
    )

    restored = configuration.snapshot().to_configuration().materials["medium"].xs

    assert restored is not None
    assert restored.fission is not None
    assert isinstance(restored.fission.neutron_production, type(neutron_production))


@pytest.mark.parametrize(
    "source",
    (
        MaterialSource({"medium": [2.0]}),
        CellSource([[[3.0]]]),
    ),
)
def test_configuration_snapshot_reconstructs_other_builtin_sources(
    source: MaterialSource | CellSource,
) -> None:
    """Built-in nonuniform source definitions should retain their input data."""
    configuration = make_configuration()
    configuration.set_source(source)

    reconstructed = configuration.snapshot().to_configuration()

    np.testing.assert_allclose(
        reconstructed.source.values(
            0, reconstructed.material_mesh.material_by_active_id(0)
        ),
        source.values(0, configuration.material_mesh.material_by_active_id(0)),
    )


def test_configuration_assign_material_requires_explicit_axial_layer() -> None:
    """Configuration mutation should target one explicit axial material layer."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    medium = Material(name="medium")
    reflector = Material(name="reflector")
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium, "reflector": reflector},
        material_mesh=MaterialMesh.stack(
            (
                MaterialSlice(mesh, {OpenMCIndex(0, 0): "medium"}, 1.0),
                MaterialSlice(mesh, {OpenMCIndex(0, 0): "reflector"}, 1.0),
            )
        ),
    )

    original_mesh = configuration.material_mesh
    configuration.assign_material(1, OpenMCIndex(0, 0), "medium")

    assert configuration.material_mesh.material_by_active_id(1) == {0: "medium"}
    assert original_mesh.material_by_active_id(1) == {0: "reflector"}
