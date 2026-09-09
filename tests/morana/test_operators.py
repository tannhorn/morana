"""Tests for global multigroup finite-volume operator assembly."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from morana import (
    FissionData,
    FissionTransfer,
    SeparableFission,
    BoundaryCondition,
    BoundaryConditionSet,
    CellSource,
    CrossSections,
    HexPlanarMesh,
    Material,
    MaterialMesh,
    MaterialSlice,
    ProblemConfiguration,
)
from morana.operators import (
    CrossSectionData,
    CrossSectionLayerData,
    _checked_layout,
    _fission_power_functional,
    _fission_production_functional,
    assemble_boundary_rhs,
    assemble_fission_matrix,
    assemble_loss_matrix,
    assemble_source_rhs,
    extract_cross_section_data,
)


def _configuration(
    *,
    source=None,
    boundary=None,
    layers=None,
    mesh: HexPlanarMesh | None = None,
    kappa_sigma_f=None,
    multiplicity_matrix=None,
) -> ProblemConfiguration:
    mesh = mesh or HexPlanarMesh(1, pitch=10.0)
    material = Material(
        "medium",
        CrossSections(
            D=[1.0, 2.0],
            sigma_a=[0.1, 0.2],
            sigma_s=[[0.3, 0.4], [0.5, 0.6]],
            fission=FissionData(
                neutron_production=SeparableFission(
                    nu_sigma_f=[0.7, 0.8], chi=[0.25, 0.75]
                ),
                kappa_sigma_f=kappa_sigma_f,
            ),
            multiplicity_matrix=multiplicity_matrix,
        ),
    )
    return ProblemConfiguration(
        mesh=mesh,
        materials={"medium": material},
        material_mesh=layers
        or MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),),
        ),
        boundary=boundary
        or BoundaryConditionSet(BoundaryCondition.reflective().globally()),
        source=source,
    )


def test_source_rhs_packs_ragged_explicit_layer_sources() -> None:
    """Explicit layer sources retain per-layer volumes and global ordering."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    layers = MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(
                mesh, [["medium"] * 6, ["medium"]], height=2.0
            ),
            MaterialSlice.from_openmc_rings(mesh, [["0"] * 6, ["medium"]], height=3.0),
        )
    )
    configuration = _configuration(
        mesh=mesh,
        layers=layers,
        source=CellSource(
            (
                [list(range(1, 8)), list(range(11, 18))],
                [[30.0], [40.0]],
            )
        ),
    )

    rhs = assemble_source_rhs(configuration, extract_cross_section_data(configuration))
    area = mesh.area

    np.testing.assert_allclose(
        rhs,
        [
            2.0 * area,
            22.0 * area,
            4.0 * area,
            24.0 * area,
            6.0 * area,
            26.0 * area,
            8.0 * area,
            28.0 * area,
            10.0 * area,
            30.0 * area,
            12.0 * area,
            32.0 * area,
            14.0 * area,
            34.0 * area,
            90.0 * area,
            120.0 * area,
        ],
    )


def test_source_rhs_does_not_require_unrelated_boundary_coverage() -> None:
    """Volumetric source assembly depends only on layout, materials, and source."""
    configuration = _configuration(
        source=CellSource(([[2.0], [3.0]],)),
        boundary=BoundaryConditionSet(),
    )

    rhs = assemble_source_rhs(configuration, extract_cross_section_data(configuration))

    np.testing.assert_allclose(
        rhs,
        [
            2.0 * configuration.material_mesh.cell_volume(0),
            3.0 * configuration.material_mesh.cell_volume(0),
        ],
    )


def test_source_rhs_without_source_is_zero() -> None:
    """Absent volumetric source data should contribute a zero RHS vector."""
    configuration = _configuration()

    rhs = assemble_source_rhs(configuration, extract_cross_section_data(configuration))

    np.testing.assert_allclose(rhs, 0.0)


def test_source_rhs_requires_exact_cell_source_layer_count() -> None:
    """Explicit cell sources must describe exactly the configured axial stack."""
    configuration = _configuration(source=CellSource(([[1.0], [2.0]], [[3.0], [4.0]])))

    with pytest.raises(ValueError, match="layer count"):
        assemble_source_rhs(configuration, extract_cross_section_data(configuration))


def test_global_one_cell_operators_have_hand_calculated_coupling_entries() -> None:
    """Removal, transfer, and separable fission use the contracted signs."""
    configuration = _configuration()
    data = extract_cross_section_data(configuration)
    snapshot = configuration.snapshot()
    volume = configuration.material_mesh.cell_volume(0)
    loss = assemble_loss_matrix(configuration, data).toarray()
    fission = assemble_fission_matrix(configuration, data).toarray()
    np.testing.assert_allclose(
        loss, [[0.5 * volume, -0.5 * volume], [-0.4 * volume, 0.7 * volume]]
    )
    np.testing.assert_allclose(
        fission, [[0.175 * volume, 0.2 * volume], [0.525 * volume, 0.6 * volume]]
    )
    np.testing.assert_allclose(
        _fission_production_functional(snapshot, data, _checked_layout(snapshot, data)),
        [0.7 * volume, 0.8 * volume],
    )
    np.testing.assert_allclose(
        np.sum(fission, axis=0),
        _fission_production_functional(snapshot, data, _checked_layout(snapshot, data)),
    )


def test_power_vector_uses_checked_configuration_fission_energy_data() -> None:
    """The power functional reads energy data after compact layout extraction."""
    configuration = _configuration(kappa_sigma_f=[2.0e6, 3.0e6])
    data = extract_cross_section_data(configuration)
    snapshot = configuration.snapshot()
    volume = configuration.material_mesh.cell_volume(0)

    np.testing.assert_allclose(
        _fission_power_functional(snapshot, data, _checked_layout(snapshot, data)),
        np.array([2.0e6, 3.0e6]) * 1.602176634e-19 * volume,
    )


def test_global_loss_uses_conventional_removal_and_unit_coupling() -> None:
    """Unit multiplicity preserves conventional removal and zero diagonal coupling."""
    configuration = _configuration()
    layer = extract_cross_section_data(configuration).layer(0)
    np.testing.assert_allclose(layer.sigma_r[:, 0], [0.5, 0.7])
    np.testing.assert_allclose(layer.multiplicity_matrix[:, :, 0], 1.0)
    np.testing.assert_allclose(
        np.diagonal(layer.scattering_coupling[:, :, 0]), [0.0, 0.0]
    )
    assert assemble_loss_matrix(
        configuration, extract_cross_section_data(configuration)
    ).shape == (2, 2)


def test_explicit_unit_multiplicity_recovers_the_compact_none_operator() -> None:
    """An explicit all-ones matrix must exactly recover ``None`` behavior."""
    compact_configuration = _configuration()
    explicit_configuration = _configuration(multiplicity_matrix=np.ones((2, 2)))

    compact_data = extract_cross_section_data(compact_configuration)
    explicit_data = extract_cross_section_data(explicit_configuration)

    np.testing.assert_allclose(
        assemble_loss_matrix(compact_configuration, compact_data).toarray(),
        assemble_loss_matrix(explicit_configuration, explicit_data).toarray(),
    )
    np.testing.assert_allclose(
        compact_data.layer(0).scattering_coupling,
        explicit_data.layer(0).scattering_coupling,
    )


def test_global_loss_uses_off_diagonal_scattering_multiplication() -> None:
    """Off-diagonal multiplicity changes only scattering-neutron coupling."""
    configuration = _configuration(multiplicity_matrix=[[1.0, 3.0], [2.0, 1.0]])
    layer = extract_cross_section_data(configuration).layer(0)
    volume = configuration.material_mesh.cell_volume(0)

    np.testing.assert_allclose(layer.sigma_r[:, 0], [0.5, 0.7])
    np.testing.assert_allclose(
        layer.scattering_coupling[:, :, 0], [[0.0, 1.2], [1.0, 0.0]]
    )
    np.testing.assert_allclose(
        assemble_loss_matrix(configuration, CrossSectionData((layer,))).toarray(),
        [[0.5 * volume, -volume], [-1.2 * volume, 0.7 * volume]],
    )


def test_global_loss_uses_same_group_scattering_multiplication() -> None:
    """Same-group multiplicity produces signed diagonal coupling."""
    configuration = _configuration(multiplicity_matrix=[[2.0, 1.0], [1.0, 0.5]])
    layer = extract_cross_section_data(configuration).layer(0)
    volume = configuration.material_mesh.cell_volume(0)

    np.testing.assert_allclose(
        layer.scattering_coupling[:, :, 0], [[0.3, 0.4], [0.5, -0.3]]
    )
    np.testing.assert_allclose(
        assemble_loss_matrix(configuration, CrossSectionData((layer,))).toarray(),
        [[0.2 * volume, -0.5 * volume], [-0.4 * volume, volume]],
    )


def test_global_loss_uses_full_signed_scattering_coupling() -> None:
    """Multiplying scatter changes coupling without changing event removal."""
    configuration = _configuration(multiplicity_matrix=[[2.0, 3.0], [4.0, 0.5]])
    layer = extract_cross_section_data(configuration).layer(0)
    volume = configuration.material_mesh.cell_volume(0)

    np.testing.assert_allclose(layer.sigma_r[:, 0], [0.5, 0.7])
    np.testing.assert_allclose(
        layer.multiplicity_matrix[:, :, 0], [[2.0, 3.0], [4.0, 0.5]]
    )
    np.testing.assert_allclose(
        layer.scattering_coupling[:, :, 0], [[0.3, 1.2], [2.0, -0.3]]
    )
    np.testing.assert_allclose(
        assemble_loss_matrix(configuration, CrossSectionData((layer,))).toarray(),
        [[0.2 * volume, -2.0 * volume], [-1.2 * volume, volume]],
    )


def test_three_group_extraction_preserves_heterogeneous_compact_data() -> None:
    """Extraction owns ordered derived data across independent material policies."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    fuel_a = Material(
        "fuel_a",
        CrossSections(
            D=[1.4, 0.8, 0.3],
            sigma_a=[0.01, 0.03, 0.08],
            sigma_s=[
                [0.2, 0.04, 0.01],
                [0.002, 0.3, 0.05],
                [0.001, 0.003, 0.4],
            ],
            fission=FissionData(
                neutron_production=SeparableFission(
                    nu_sigma_f=[0.01, 0.03, 0.1],
                    chi=[0.9995, 0.0004, 0.0002],
                    chi_normalization_tolerance=1.0e-3,
                )
            ),
        ),
    )
    medium = Material(
        "medium",
        CrossSections(
            D=[1.2, 0.6, 0.2],
            sigma_a=[0.02, 0.04, 0.1],
            sigma_s=np.zeros((3, 3)),
            fission=None,
        ),
    )
    fuel_b = Material(
        "fuel_b",
        CrossSections(
            D=[1.3, 0.7, 0.25],
            sigma_a=[0.015, 0.035, 0.09],
            sigma_s=np.zeros((3, 3)),
            fission=FissionData(
                neutron_production=SeparableFission(
                    nu_sigma_f=[0.02, 0.04, 0.08],
                    chi=[0.98, 0.01, 0.01],
                    chi_normalization_tolerance=1.0e-10,
                )
            ),
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"fuel_a": fuel_a, "medium": medium, "fuel_b": fuel_b},
        material_mesh=MaterialMesh.stack(
            (
                MaterialSlice.from_openmc_rings(
                    mesh,
                    [["fuel_a", "medium", "0", "0", "0", "0"], ["fuel_b"]],
                    height=1.0,
                ),
            ),
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
    )

    layer = extract_cross_section_data(configuration).layer(0)

    assert layer.material_by_active_id == {0: "fuel_a", 1: "medium", 2: "fuel_b"}
    assert layer.diffusion.shape == (3, 3)
    assert layer.scattering_coupling.shape == (3, 3, 3)
    np.testing.assert_allclose(
        layer.sigma_r[:, 0],
        fuel_a.xs.sigma_a
        + np.sum(fuel_a.xs.sigma_s, axis=1)
        - np.diagonal(fuel_a.xs.sigma_s),
    )
    np.testing.assert_allclose(layer.fission_transfer[:, :, 1], np.zeros((3, 3)))
    np.testing.assert_allclose(
        layer.fission_production[:, [0, 2]], [[0.01, 0.02], [0.03, 0.04], [0.1, 0.08]]
    )
    assert all(
        not values.flags.writeable
        for values in (
            layer.diffusion,
            layer.sigma_a,
            layer.sigma_r,
            layer.sigma_s,
            layer.multiplicity_matrix,
            layer.fission_transfer,
            layer.scattering_coupling,
            layer.fission_production,
        )
    )


def test_extraction_preserves_transfer_fission_data_and_derives_production() -> None:
    """Extraction should stack canonical event-oriented transfer data directly."""
    configuration = _configuration()
    transfer = [[0.01, 0.02], [0.03, 0.04]]
    configuration.replace_material(
        Material(
            "medium",
            CrossSections(
                D=[1.0, 2.0],
                sigma_a=[0.1, 0.2],
                sigma_s=[[0.3, 0.4], [0.5, 0.6]],
                fission=FissionData(neutron_production=FissionTransfer(transfer)),
            ),
        )
    )

    layer = extract_cross_section_data(configuration).layer(0)

    np.testing.assert_allclose(layer.fission_transfer[:, :, 0], transfer)
    np.testing.assert_allclose(layer.fission_production[:, 0], [0.03, 0.07])


def test_compact_cross_section_data_are_immutable() -> None:
    """Checked compact cross sections must not permit replacement or map edits."""
    data = extract_cross_section_data(_configuration())
    layer = data.layer(0)

    with pytest.raises(FrozenInstanceError):
        layer.diffusion = np.ones((2, 1))
    with pytest.raises(TypeError):
        layer.material_by_active_id[0] = "other"
    with pytest.raises(FrozenInstanceError):
        data.layers = ()


@pytest.mark.parametrize(
    ("replacement", "exception"),
    [
        ({"axial_index": "0"}, TypeError),
        ({"axial_index": -1}, ValueError),
        ({"material_by_active_id": [(0, "medium")]}, TypeError),
        ({"material_by_active_id": {False: "medium"}}, TypeError),
        ({"material_by_active_id": {0: 1}}, TypeError),
    ],
)
def test_compact_cross_sections_check_structural_input_types(
    replacement: dict[str, object], exception: type[Exception]
) -> None:
    """Direct compact data rejects malformed structural input before assembly."""
    layer = extract_cross_section_data(_configuration()).layer(0)
    fields: dict[str, object] = {
        "axial_index": layer.axial_index,
        "diffusion": layer.diffusion,
        "sigma_a": layer.sigma_a,
        "sigma_s": layer.sigma_s,
        "multiplicity_matrix": layer.multiplicity_matrix,
        "fission_transfer": layer.fission_transfer,
        "material_by_active_id": layer.material_by_active_id,
    }
    fields.update(replacement)

    with pytest.raises(exception):
        CrossSectionLayerData(**fields)  # type: ignore[arg-type]


def test_compact_cross_sections_reject_invalid_resolved_multiplicity() -> None:
    """Compact resolved multiplicity must retain physical array invariants."""
    layer = extract_cross_section_data(_configuration()).layer(0)

    with pytest.raises(ValueError, match="multiplicity_matrix"):
        CrossSectionLayerData(
            axial_index=layer.axial_index,
            diffusion=layer.diffusion,
            sigma_a=layer.sigma_a,
            sigma_s=layer.sigma_s,
            multiplicity_matrix=np.array([[[-1.0], [1.0]], [[1.0], [1.0]]]),
            fission_transfer=layer.fission_transfer,
            material_by_active_id=layer.material_by_active_id,
        )


@pytest.mark.parametrize("layers", [None, (object(),)])
def test_cross_section_data_requires_compact_layers(layers: object) -> None:
    """The layered wrapper reports malformed input at construction."""
    with pytest.raises(TypeError):
        CrossSectionData(layers)  # type: ignore[arg-type]


def test_cross_section_data_requires_layers_and_active_cells() -> None:
    """Compact data must describe a nonempty active axial domain."""
    empty_layer = CrossSectionLayerData(
        axial_index=0,
        diffusion=np.empty((1, 0)),
        sigma_a=np.empty((1, 0)),
        sigma_s=np.empty((1, 1, 0)),
        multiplicity_matrix=np.empty((1, 1, 0)),
        fission_transfer=np.empty((1, 1, 0)),
        material_by_active_id={},
    )

    with pytest.raises(ValueError, match="at least one layer"):
        CrossSectionData(())
    with pytest.raises(ValueError, match="at least one active cell"):
        CrossSectionData((empty_layer,))


def test_compact_cross_section_layer_requires_a_group() -> None:
    """Even an empty axial layer must retain the problem group count."""
    with pytest.raises(ValueError, match="at least one group"):
        CrossSectionLayerData(
            axial_index=0,
            diffusion=np.empty((0, 0)),
            sigma_a=np.empty((0, 0)),
            sigma_s=np.empty((0, 0, 0)),
            multiplicity_matrix=np.empty((0, 0, 0)),
            fission_transfer=np.empty((0, 0, 0)),
            material_by_active_id={},
        )


def test_cross_section_extraction_requires_an_active_cell() -> None:
    """All-excluded material stacks cannot produce compact cross sections."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    configuration = _configuration(
        mesh=mesh,
        layers=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["0"]], height=1.0),)
        ),
    )

    with pytest.raises(ValueError, match="at least one active cell"):
        extract_cross_section_data(configuration)


def test_cross_section_data_layer_requires_nonnegative_integer_index() -> None:
    """Layer retrieval follows the material-mesh index contract."""
    data = extract_cross_section_data(_configuration())

    with pytest.raises(TypeError, match="axial_index must be an integer"):
        data.layer("0")  # type: ignore[arg-type]
    with pytest.raises(IndexError, match="outside cross-section data"):
        data.layer(-1)


def test_global_two_cell_loss_places_same_group_radial_coupling_by_node() -> None:
    """Radial leakage connects matching group slots at neighboring nodes."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    material = Material(
        "medium",
        CrossSections(
            D=[1.0, 2.0],
            sigma_a=[0.1, 0.2],
            sigma_s=[[0.0, 0.4], [0.5, 0.0]],
            fission=None,
        ),
    )
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(
                mesh, [["medium", "0", "0", "0", "0", "0"], ["medium"]], height=1.0
            ),
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": material},
        material_mesh=material_mesh,
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
    )
    volume = material_mesh.cell_volume(0)
    conductance = material_mesh.radial_face_area(0) / mesh.center_distance
    matrix = assemble_loss_matrix(
        configuration, extract_cross_section_data(configuration)
    ).toarray()
    np.testing.assert_allclose(matrix[0, 2], -conductance)
    np.testing.assert_allclose(matrix[2, 0], -conductance)
    np.testing.assert_allclose(matrix[1, 3], -2.0 * conductance)
    np.testing.assert_allclose(matrix[3, 1], -2.0 * conductance)
    np.testing.assert_allclose(matrix[1, 0], -0.4 * volume)
    np.testing.assert_allclose(matrix[2, 3], -0.5 * volume)


def test_global_loss_assembles_unequal_axial_coupling_by_layered_node() -> None:
    """Axial coupling uses half-cell resistances and cross-layer compact IDs."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    lower = Material(
        "lower",
        CrossSections(
            D=[2.0, 4.0],
            sigma_a=[0.0, 0.0],
            sigma_s=[[0.0, 0.0], [0.0, 0.0]],
            fission=None,
        ),
    )
    upper = Material(
        "upper",
        CrossSections(
            D=[3.0, 6.0],
            sigma_a=[0.0, 0.0],
            sigma_s=[[0.0, 0.0], [0.0, 0.0]],
            fission=None,
        ),
    )
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(
                mesh, [["lower", "lower", "0", "0", "0", "0"], ["0"]], height=2.0
            ),
            MaterialSlice.from_openmc_rings(
                mesh, [["0", "upper", "0", "0", "0", "0"], ["0"]], height=3.0
            ),
        )
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"lower": lower, "upper": upper},
        material_mesh=material_mesh,
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
    )

    matrix = assemble_loss_matrix(
        configuration, extract_cross_section_data(configuration)
    ).toarray()
    area = mesh.area

    # The shared second outer position is active_id=1 below and active_id=0 above.
    np.testing.assert_allclose(matrix[2, 4], -area)
    np.testing.assert_allclose(matrix[4, 2], -area)
    np.testing.assert_allclose(matrix[3, 5], -2.0 * area)
    np.testing.assert_allclose(matrix[5, 3], -2.0 * area)
    np.testing.assert_allclose(matrix[4, 4], area)
    np.testing.assert_allclose(matrix[5, 5], 2.0 * area)


def test_source_and_boundary_rhs_are_node_major_and_group_resolved() -> None:
    """Volumetric and Dirichlet data contribute only to their selected groups."""
    boundary = BoundaryConditionSet(BoundaryCondition.dirichlet([3.0, 5.0]).globally())
    configuration = _configuration(
        source=CellSource(([[2.0], [7.0]],)), boundary=boundary
    )
    data = extract_cross_section_data(configuration)
    volume = configuration.material_mesh.cell_volume(0)
    np.testing.assert_allclose(
        assemble_source_rhs(configuration, data), [2.0 * volume, 7.0 * volume]
    )
    rhs = assemble_boundary_rhs(configuration, data)
    assert rhs[0] > 0.0 and rhs[1] > 0.0
    assert rhs[1] / rhs[0] == pytest.approx((2.0 * 5.0) / (1.0 * 3.0))


def test_axial_outer_dirichlet_contributes_loss_and_rhs() -> None:
    """Top Dirichlet data uses the planar area and half-layer distance."""
    reflective = _configuration()
    top_dirichlet = _configuration(
        boundary=BoundaryConditionSet(
            BoundaryCondition.reflective().globally(),
            BoundaryCondition.dirichlet([3.0, 5.0]).on_top(),
        )
    )
    reflective_data = extract_cross_section_data(reflective)
    top_data = extract_cross_section_data(top_dirichlet)
    area = top_dirichlet.mesh.area

    loss_increment = (
        assemble_loss_matrix(top_dirichlet, top_data)
        - assemble_loss_matrix(reflective, reflective_data)
    ).toarray()
    np.testing.assert_allclose(np.diag(loss_increment), [2.0 * area, 4.0 * area])
    np.testing.assert_allclose(
        assemble_boundary_rhs(top_dirichlet, top_data), [6.0 * area, 20.0 * area]
    )


def test_axial_excluded_incoming_current_contributes_loss_and_rhs() -> None:
    """An axial excluded interface uses its directional selected condition."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    layers = MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=2.0),
            MaterialSlice.from_openmc_rings(mesh, [["0"]], height=3.0),
        )
    )
    reflective = _configuration(mesh=mesh, layers=layers)
    excluded_current = _configuration(
        mesh=mesh,
        layers=layers,
        boundary=BoundaryConditionSet(
            BoundaryCondition.reflective().globally(),
            BoundaryCondition.incoming_current([3.0, 4.0]).on_excluded(direction="top"),
        ),
    )
    reflective_data = extract_cross_section_data(reflective)
    excluded_data = extract_cross_section_data(excluded_current)
    area = mesh.area

    loss_increment = (
        assemble_loss_matrix(excluded_current, excluded_data)
        - assemble_loss_matrix(reflective, reflective_data)
    ).toarray()
    np.testing.assert_allclose(np.diag(loss_increment), [area / 3.0, 0.4 * area])
    np.testing.assert_allclose(
        assemble_boundary_rhs(excluded_current, excluded_data), [4.0 * area, 6.4 * area]
    )


def test_fission_assembly_rejects_nonfinite_integrated_values() -> None:
    """Finite compact coefficients must not overflow into an exposed operator."""
    configuration = _configuration()
    layer = extract_cross_section_data(configuration).layer(0)
    overflowing = CrossSectionLayerData(
        axial_index=0,
        diffusion=layer.diffusion,
        sigma_a=layer.sigma_a,
        sigma_s=layer.sigma_s,
        multiplicity_matrix=layer.multiplicity_matrix,
        fission_transfer=np.full((2, 2, 1), 1.0e308),
        material_by_active_id=layer.material_by_active_id,
    )
    data = CrossSectionData((overflowing,))
    snapshot = configuration.snapshot()

    with pytest.raises(
        ValueError, match="fission matrix entries must contain only finite values"
    ):
        assemble_fission_matrix(configuration, data)
    with pytest.raises(
        ValueError, match="fission production values must contain only finite values"
    ):
        _fission_production_functional(snapshot, data, _checked_layout(snapshot, data))


def test_rhs_assembly_rejects_nonfinite_integrated_values() -> None:
    """Volume and face integration must retain the finite RHS contract."""
    source_configuration = _configuration(source=CellSource(([[1.0e308], [0.0]],)))
    with pytest.raises(
        ValueError, match="source RHS values must contain only finite values"
    ):
        assemble_source_rhs(
            source_configuration, extract_cross_section_data(source_configuration)
        )

    boundary_configuration = _configuration(
        boundary=BoundaryConditionSet(
            BoundaryCondition.dirichlet([1.0e308, 0.0]).globally()
        )
    )
    with pytest.raises(
        ValueError, match="boundary RHS values must contain only finite values"
    ):
        assemble_boundary_rhs(
            boundary_configuration, extract_cross_section_data(boundary_configuration)
        )


def test_boundary_vectors_require_exact_group_count_at_assembly() -> None:
    """Scalar broadcasting and wrong-length prescribed spectra are rejected."""
    boundary = BoundaryConditionSet(BoundaryCondition.dirichlet([1.0]).globally())
    configuration = _configuration(boundary=boundary)
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        assemble_boundary_rhs(configuration, extract_cross_section_data(configuration))


def test_global_assembly_rejects_layer_data_instead_of_complete_input() -> None:
    """Public operators require complete layered compact cross-section data."""
    configuration = _configuration()
    with pytest.raises(TypeError, match="complete CrossSectionData"):
        assemble_loss_matrix(
            configuration, extract_cross_section_data(configuration).layer(0)
        )


@pytest.mark.parametrize(
    "operation",
    [
        extract_cross_section_data,
        lambda configuration: assemble_fission_matrix(configuration, object()),
        lambda configuration: assemble_loss_matrix(configuration, object()),
        lambda configuration: assemble_source_rhs(configuration, object()),
        lambda configuration: assemble_boundary_rhs(configuration, object()),
    ],
)
def test_public_operator_functions_require_problem_definition(operation) -> None:
    """Public extraction and assembly reject lookalike problem definitions."""
    with pytest.raises(TypeError, match="configuration must be a ProblemConfiguration"):
        operation(object())


def test_public_operator_functions_accept_one_configuration_snapshot() -> None:
    """Snapshot input supports cross-section extraction and every assembly form."""
    configuration = _configuration(source=CellSource(([[2.0], [3.0]],)))
    snapshot = configuration.snapshot()
    data = extract_cross_section_data(snapshot)

    np.testing.assert_allclose(
        assemble_fission_matrix(snapshot, data).toarray(),
        assemble_fission_matrix(configuration, data).toarray(),
    )
    np.testing.assert_allclose(
        assemble_loss_matrix(snapshot, data).toarray(),
        assemble_loss_matrix(configuration, data).toarray(),
    )
    np.testing.assert_allclose(
        assemble_source_rhs(snapshot, data),
        assemble_source_rhs(configuration, data),
    )
    np.testing.assert_allclose(
        assemble_boundary_rhs(snapshot, data),
        assemble_boundary_rhs(configuration, data),
    )


@pytest.mark.parametrize(
    "operation",
    [
        assemble_fission_matrix,
        assemble_loss_matrix,
        assemble_boundary_rhs,
        assemble_source_rhs,
    ],
)
def test_public_assemblers_require_complete_cross_section_data(operation) -> None:
    """Public assemblers reject compact-data lookalikes before accessing them."""
    configuration = _configuration(source=CellSource(([[1.0], [1.0]],)))

    with pytest.raises(TypeError, match="complete CrossSectionData"):
        operation(configuration, object())


def test_global_assembly_checks_layer_configuration_correspondence() -> None:
    """A compact layer must retain its exact material-mesh provenance."""
    configuration = _configuration()
    layer = extract_cross_section_data(configuration).layer(0)
    incorrect = CrossSectionLayerData(
        axial_index=0,
        diffusion=layer.diffusion,
        sigma_a=layer.sigma_a,
        sigma_s=layer.sigma_s,
        multiplicity_matrix=layer.multiplicity_matrix,
        fission_transfer=layer.fission_transfer,
        material_by_active_id={0: "other"},
    )
    with pytest.raises(ValueError, match="does not match"):
        assemble_loss_matrix(configuration, CrossSectionData((incorrect,)))


def test_removal_retains_small_outscatter_beside_large_self_scattering():
    """Self-scattering must not swamp absorption or off-diagonal events."""
    configuration = _configuration()
    configuration.replace_material(
        Material(
            "medium",
            CrossSections(
                D=[1.0, 1.0],
                sigma_a=[0.1, 0.2],
                sigma_s=[[1e20, 0.4], [0.5, 1e20]],
                fission=None,
            ),
        )
    )
    layer = extract_cross_section_data(configuration).layer(0)
    np.testing.assert_allclose(layer.sigma_r, [[0.5], [0.7]])


def test_loss_matrix_rejects_overflow_when_entries_are_combined():
    """Individually finite face terms can overflow the assembled diagonal."""
    configuration = _configuration(
        mesh=HexPlanarMesh(1, pitch=1.0),
        boundary=BoundaryConditionSet(BoundaryCondition.zero_dirichlet().globally()),
    )
    configuration.replace_material(
        Material(
            "medium",
            CrossSections(
                D=[3e307, 3e307],
                sigma_a=[0.1, 0.2],
                sigma_s=[[0.0, 0.0], [0.0, 0.0]],
                fission=None,
            ),
        )
    )
    with pytest.raises(ValueError, match="finite"):
        assemble_loss_matrix(configuration, extract_cross_section_data(configuration))
