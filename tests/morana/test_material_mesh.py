"""Tests for material-key layouts over hex meshes."""

from xml.etree import ElementTree as ET
import json
import math
from pathlib import Path

from matplotlib.colors import to_hex
import numpy as np
import pytest

from morana import (
    DomainFace,
    ExcludedRegion,
    HexPlanarMesh,
    Material,
    MaterialMesh,
    MaterialSlice,
    OpenMCIndex,
)
from morana.materials import INACTIVE_EXCLUDED_REGION_COLOR


def _vtm_leaf_paths(path: Path) -> tuple[Path, Path]:
    """Return active and excluded leaf paths referenced by a VTM index."""
    root = ET.parse(path).getroot()
    datasets = root.findall("./vtkMultiBlockDataSet/Block/DataSet")
    assert len(datasets) == 2
    return (
        path.parent / datasets[0].attrib["file"],
        path.parent / datasets[1].attrib["file"],
    )


def test_material_mesh_defaults_to_inactive_positions() -> None:
    """Unassigned lattice positions should default to excluded key '0'."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    material_mesh = MaterialMesh.stack((MaterialSlice(mesh, {}, 1.0),))

    assert material_mesh.key_at(0, OpenMCIndex(0, 0)) == "0"
    assert material_mesh.active_indices(0) == ()
    assert material_mesh.n_active_cells(0) == 0
    assert material_mesh.material_by_active_id(0) == {}
    assert material_mesh.active_id_at(0, OpenMCIndex(0, 0)) is None


def test_material_mesh_owns_complete_domain_face_direction_order() -> None:
    """Consumers should obtain radial and axial topology from MaterialMesh."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material_mesh = MaterialMesh.stack((MaterialSlice(mesh, {}, 1.0),))

    assert material_mesh.face_direction_labels == (
        "x+",
        "x-",
        "u+",
        "u-",
        "v+",
        "v-",
        "bottom",
        "top",
    )


def test_domain_face_rejects_direct_construction() -> None:
    """Face records originate from completed material-mesh topology only."""
    with pytest.raises(TypeError, match="returned by MaterialMesh.face"):
        DomainFace()


def test_material_slice_requires_openmc_mapping_keys() -> None:
    """Sparse layouts should reject untyped keys before mesh-membership lookup."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    with pytest.raises(TypeError, match="OpenMCIndex"):
        MaterialSlice(mesh, {(0, 0): "medium"}, 1.0)  # type: ignore[dict-item]


def test_openmc_ring_factory_requires_typed_container_inputs() -> None:
    """The direct ring factory should validate before accessing its inputs."""
    with pytest.raises(TypeError, match="mesh"):
        MaterialSlice.from_openmc_rings(None, [["medium"]], height=1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="rings"):
        MaterialSlice.from_openmc_rings(HexPlanarMesh(1, 10.0), (("medium",),), height=1.0)  # type: ignore[arg-type]


def test_openmc_index_queries_require_typed_identity() -> None:
    """Public material-layout queries should reject untyped lattice identities."""
    mesh = MaterialMesh.stack((MaterialSlice(HexPlanarMesh(1, 10.0), {}, 1.0),))
    with pytest.raises(TypeError, match="openmc_index"):
        mesh.key_at(0, (0, 0))  # type: ignore[arg-type]


def test_material_mesh_stacks_independent_height_bearing_slices() -> None:
    """Stacking should complete independent layers from reusable slice inputs."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    source_mapping = {OpenMCIndex(0, 0): "medium"}
    lower = MaterialSlice(mesh, source_mapping, height=2.5)
    upper = MaterialSlice(mesh, source_mapping, height=4.0)

    material_mesh = MaterialMesh.stack((lower, upper))
    source_mapping[OpenMCIndex(0, 0)] = "reflector"

    assert lower.material_keys[OpenMCIndex(0, 0)] == "medium"
    assert material_mesh.key_at(0, OpenMCIndex(0, 0)) == "medium"
    assert material_mesh.key_at(1, OpenMCIndex(0, 0)) == "medium"
    assert material_mesh.axial_layer_heights == (2.5, 4.0)
    assert material_mesh.z_min == pytest.approx(0.0)
    assert material_mesh.z_max == pytest.approx(6.5)
    assert material_mesh.z_bounds(0) == pytest.approx((0.0, 2.5))
    assert material_mesh.z_bounds(1) == pytest.approx((2.5, 6.5))


def test_material_mesh_stack_requires_one_shared_mesh_instance() -> None:
    """A stack must retain one authoritative planar-index universe."""
    first = HexPlanarMesh(1, pitch=10.0)
    second = HexPlanarMesh(1, pitch=10.0)

    with pytest.raises(ValueError, match="identical HexPlanarMesh"):
        MaterialMesh.stack(
            (MaterialSlice(first, {}, 1.0), MaterialSlice(second, {}, 1.0))
        )


@pytest.mark.parametrize("axial_index", [-1, 1])
def test_material_mesh_rejects_axial_indices_outside_stack(axial_index: int) -> None:
    """Layer-specific queries must select a nonnegative in-stack layer."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material_mesh = MaterialMesh.stack((MaterialSlice(mesh, {}, 1.0),))

    with pytest.raises(IndexError, match="outside the material-slice stack"):
        material_mesh.axial_face_area(axial_index)


def test_material_mesh_rejects_noninteger_axial_index() -> None:
    """Layer-specific queries should reject ambiguous axial selectors."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material_mesh = MaterialMesh.stack((MaterialSlice(mesh, {}, 1.0),))

    with pytest.raises(TypeError, match="axial_index must be an integer"):
        material_mesh.z_bounds(True)


def test_material_slice_extrusion_is_a_composable_stack_fragment() -> None:
    """Slice extrusion should preserve the one immutable slice value."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material_slice = MaterialSlice(mesh, {OpenMCIndex(0, 0): "medium"}, 3.0)

    material_mesh = MaterialMesh.stack(material_slice.extrude(count=3))

    assert material_mesh.axial_layer_heights == (3.0, 3.0, 3.0)
    assert all(
        material_mesh.key_at(axial_index, OpenMCIndex(0, 0)) == "medium"
        for axial_index in range(3)
    )


def test_material_slice_extrusion_accepts_unequal_heights() -> None:
    """Explicit extrusion heights should create a heterogeneous stack fragment."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material_slice = MaterialSlice(mesh, {OpenMCIndex(0, 0): "medium"}, 3.0)

    material_mesh = MaterialMesh.stack(material_slice.extrude(heights=(1.5, 4.0, 2.5)))

    assert material_mesh.axial_layer_heights == (1.5, 4.0, 2.5)
    assert material_mesh.z_max == pytest.approx(8.0)
    assert material_mesh.z_bounds(1) == pytest.approx((1.5, 5.5))


def test_material_mesh_stacks_custom_slices_and_extrusion_fragments() -> None:
    """Composed bottom-to-top fragments should retain independent layers."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    bottom_reflector = MaterialSlice(mesh, {OpenMCIndex(0, 0): "bottom_reflector"}, 2.0)
    core = MaterialSlice(mesh, {OpenMCIndex(0, 0): "fuel"}, 4.0)
    top_reflector = MaterialSlice(mesh, {OpenMCIndex(0, 0): "top_reflector"}, 1.5)

    material_mesh = MaterialMesh.stack(
        bottom_reflector.extrude(heights=(1.0, 2.0))
        + (core,)
        + top_reflector.extrude(count=2)
    )

    assert material_mesh.axial_layer_heights == (1.0, 2.0, 4.0, 1.5, 1.5)
    np.testing.assert_allclose(
        tuple(material_mesh.z_bounds(index) for index in range(5)),
        ((0.0, 1.0), (1.0, 3.0), (3.0, 7.0), (7.0, 8.5), (8.5, 10.0)),
    )
    assert tuple(
        material_mesh.key_at(index, OpenMCIndex(0, 0)) for index in range(5)
    ) == (
        "bottom_reflector",
        "bottom_reflector",
        "fuel",
        "top_reflector",
        "top_reflector",
    )
    assert tuple(material_mesh.material_by_active_id(index) for index in range(5)) == (
        {0: "bottom_reflector"},
        {0: "bottom_reflector"},
        {0: "fuel"},
        {0: "top_reflector"},
        {0: "top_reflector"},
    )


def test_material_slice_extrusion_requires_one_fragment_specification() -> None:
    """Extrusion must select exactly one uniform or explicit-height form."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material_slice = MaterialSlice(mesh, {}, 1.0)

    with pytest.raises(ValueError, match="exactly one"):
        material_slice.extrude()
    with pytest.raises(ValueError, match="exactly one"):
        material_slice.extrude(count=2, heights=(1.0, 1.0))
    with pytest.raises(ValueError, match="nonempty tuple"):
        material_slice.extrude(heights=())


def test_material_mesh_derives_compact_active_ids() -> None:
    """Active positions should be compacted in OpenMC flattened order."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(
                mesh,
                [["0", "medium", "0", "reflector", "0", "medium"], ["medium"]],
                height=1.0,
            ),
        ),
    )

    assert material_mesh.active_indices(0) == (
        OpenMCIndex(0, 1),
        OpenMCIndex(0, 3),
        OpenMCIndex(0, 5),
        OpenMCIndex(1, 0),
    )
    assert material_mesh.n_active_cells(0) == 4
    material_by_active_id = material_mesh.material_by_active_id(0)
    assert material_by_active_id == {
        0: "medium",
        1: "reflector",
        2: "medium",
        3: "medium",
    }
    with pytest.raises(TypeError):
        material_by_active_id[0] = "other"  # type: ignore[index]
    assert material_mesh.active_id_at(0, OpenMCIndex(0, 0)) is None
    assert material_mesh.active_id_at(0, OpenMCIndex(0, 3)) == 1
    assert material_mesh.openmc_index_for_active_id(0, 2) == OpenMCIndex(0, 5)


def test_material_slice_rejects_non_string_material_keys() -> None:
    """Sparse and ring-based material layouts should require string keys."""
    mesh = HexPlanarMesh(1, pitch=10.0)

    with pytest.raises(TypeError, match="material key must be a string"):
        MaterialSlice(mesh, {OpenMCIndex(0, 0): 0}, 1.0)  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="material key must be a string"):
        MaterialSlice.from_openmc_rings(mesh, [[0]], height=1.0)  # type: ignore[list-item]


@pytest.mark.parametrize("key", ["", "   ", "fuel\n1", "fuel\u200b1"])
def test_material_slice_rejects_nonhuman_readable_material_keys(key: str) -> None:
    """Material-layout keys must remain legible identifiers."""
    mesh = HexPlanarMesh(1, pitch=10.0)

    with pytest.raises(ValueError, match="material key"):
        MaterialSlice(mesh, {OpenMCIndex(0, 0): key}, 1.0)


def test_material_mesh_rejects_non_string_excluded_region_keys() -> None:
    """Custom excluded-region catalogs should require string keys."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material_slice = MaterialSlice(mesh, {}, 1.0)

    with pytest.raises(TypeError, match="material key must be a string"):
        MaterialMesh.stack(
            (material_slice,),
            excluded_regions={0: ExcludedRegion(kind="void")},  # type: ignore[dict-item]
        )


def test_material_mesh_requires_excluded_region_mapping() -> None:
    """Excluded-region normalization should reject non-mapping containers."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material_slice = MaterialSlice(mesh, {}, 1.0)

    with pytest.raises(TypeError, match="excluded_regions must be a mapping"):
        MaterialMesh.stack((material_slice,), excluded_regions=False)  # type: ignore[arg-type]


def test_material_mesh_checks_ring_shape() -> None:
    """OpenMC-style ring input must match the full mesh shape."""
    mesh = HexPlanarMesh(2, pitch=10.0)

    with pytest.raises(ValueError, match="ring 0"):
        MaterialMesh.stack(
            (
                MaterialSlice.from_openmc_rings(
                    mesh, [["medium"], ["medium"]], height=1.0
                ),
            ),
        )


def test_material_mesh_face_reports_internal_neighbor() -> None:
    """Face descriptions should identify active neighboring cells."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(
                mesh, [["medium", "0", "0", "0", "0", "0"], ["medium"]], height=1.0
            ),
        ),
    )

    face = material_mesh.face(axial_index=0, active_id=1, direction="x+")

    assert (
        face.active_id,
        face.axial_index,
        face.openmc_index,
        face.direction,
        face.neighbor_axial_index,
        face.neighbor_openmc_index,
        face.neighbor_active_id,
        face.kind,
    ) == (
        1,
        0,
        OpenMCIndex(1, 0),
        "x+",
        0,
        OpenMCIndex(0, 0),
        0,
        "internal",
    )


def test_material_mesh_face_reports_outer_boundary() -> None:
    """Face descriptions should identify neighbors outside the full mesh."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(
                mesh, [["medium", "0", "0", "0", "0", "0"], ["medium"]], height=1.0
            ),
        ),
    )

    face = material_mesh.face(axial_index=0, active_id=0, direction="x+")

    assert face.kind == "outer"
    assert face.axial_index == 0
    assert face.neighbor_axial_index is None
    assert face.openmc_index == OpenMCIndex(0, 0)
    assert face.neighbor_openmc_index is None
    assert face.neighbor_active_id is None


def test_material_mesh_outer_boundary_directions_are_deterministic() -> None:
    """Representative outer cells should expose fixed direction labels."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(
                mesh,
                [
                    ["medium", "medium", "medium", "medium", "medium", "medium"],
                    ["medium"],
                ],
                height=1.0,
            ),
        ),
    )

    outer_directions_by_active_id = {
        0: {"x+", "u+", "v-"},
        1: {"x+", "u-", "v-"},
        6: set(),
    }

    for active_id, expected_outer_directions in outer_directions_by_active_id.items():
        actual_outer_directions = {
            direction
            for direction in ("x+", "x-", "u+", "u-", "v+", "v-")
            if material_mesh.face(0, active_id, direction).kind == "outer"
        }
        assert actual_outer_directions == expected_outer_directions


def test_material_mesh_face_reports_excluded_neighbor() -> None:
    """Face descriptions should preserve excluded-neighbor semantics."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(
                mesh, [["medium", "0", "0", "0", "0", "0"], ["medium"]], height=1.0
            ),
        ),
    )

    face = material_mesh.face(axial_index=0, active_id=1, direction="v-")

    assert face.kind == "to_excluded"
    assert face.axial_index == 0
    assert face.neighbor_axial_index == 0
    assert face.openmc_index == OpenMCIndex(1, 0)
    assert face.neighbor_openmc_index == OpenMCIndex(0, 1)
    assert face.neighbor_active_id is None
    assert face.neighbor_key == "0"
    assert face.neighbor_key_kind == "inactive"


def test_material_mesh_axial_faces_translate_cross_layer_active_ids() -> None:
    """Axial neighbors should match planar identity, not slice-local IDs."""
    mesh = HexPlanarMesh(2, pitch=10.0)
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

    top_face = material_mesh.face(axial_index=0, active_id=0, direction="top")
    bottom_face = material_mesh.face(axial_index=1, active_id=1, direction="bottom")

    assert (
        top_face.active_id,
        top_face.axial_index,
        top_face.openmc_index,
        top_face.direction,
        top_face.neighbor_axial_index,
        top_face.neighbor_openmc_index,
        top_face.neighbor_active_id,
        top_face.kind,
    ) == (0, 0, OpenMCIndex(1, 0), "top", 1, OpenMCIndex(1, 0), 1, "internal")
    assert (
        bottom_face.active_id,
        bottom_face.axial_index,
        bottom_face.openmc_index,
        bottom_face.direction,
        bottom_face.neighbor_axial_index,
        bottom_face.neighbor_openmc_index,
        bottom_face.neighbor_active_id,
        bottom_face.kind,
    ) == (1, 1, OpenMCIndex(1, 0), "bottom", 0, OpenMCIndex(1, 0), 0, "internal")
    assert material_mesh.face(0, 0, "bottom").kind == "outer"
    assert material_mesh.face(1, 1, "top").kind == "outer"


def test_material_mesh_axial_face_reports_excluded_neighbor() -> None:
    """An in-stack excluded axial position is not a physical outer face."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice(mesh, {OpenMCIndex(0, 0): "medium"}, 1.0),
            MaterialSlice(mesh, {OpenMCIndex(0, 0): "gap"}, 1.0),
            MaterialSlice(mesh, {OpenMCIndex(0, 0): "medium"}, 1.0),
        ),
        excluded_regions={"gap": ExcludedRegion(kind="void")},
    )

    lower_face = material_mesh.face(axial_index=0, active_id=0, direction="top")
    upper_face = material_mesh.face(axial_index=2, active_id=0, direction="bottom")

    for face, axial_index, direction, neighbor_axial_index in (
        (lower_face, 0, "top", 1),
        (upper_face, 2, "bottom", 1),
    ):
        assert face.kind == "to_excluded"
        assert face.axial_index == axial_index
        assert face.direction == direction
        assert face.neighbor_axial_index == neighbor_axial_index
        assert face.neighbor_openmc_index == OpenMCIndex(0, 0)
        assert face.neighbor_active_id is None
        assert face.neighbor_key == "gap"
        assert face.neighbor_key_kind == "void"


def test_material_mesh_supports_custom_excluded_region_keys() -> None:
    """Custom excluded keys should be excluded from active IDs."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(
                mesh,
                [["medium", "gap", "medium", "0", "medium", "gap"], ["medium"]],
                height=1.0,
            ),
        ),
        excluded_regions={"gap": ExcludedRegion(kind="void", color="#eeeeee")},
    )

    assert material_mesh.key_at(0, OpenMCIndex(0, 1)) == "gap"
    assert material_mesh.active_indices(0) == (
        OpenMCIndex(0, 0),
        OpenMCIndex(0, 2),
        OpenMCIndex(0, 4),
        OpenMCIndex(1, 0),
    )
    assert material_mesh.excluded_regions["0"].kind == "inactive"
    assert material_mesh.excluded_regions["gap"].kind == "void"


def test_material_mesh_face_reports_custom_excluded_region_details() -> None:
    """Faces next to custom excluded keys should carry key and kind details."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(
                mesh, [["gap", "0", "0", "0", "0", "0"], ["medium"]], height=1.0
            ),
        ),
        excluded_regions={"gap": ExcludedRegion(kind="void")},
    )

    face = material_mesh.face(axial_index=0, active_id=0, direction="x+")

    assert face.kind == "to_excluded"
    assert face.neighbor_key == "gap"
    assert face.neighbor_key_kind == "void"


def test_material_mesh_face_rejects_unknown_direction() -> None:
    """Face queries should check direction labels."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),),
    )

    with pytest.raises(ValueError, match="unknown hex direction"):
        material_mesh.face(axial_index=0, active_id=0, direction="bad")


@pytest.mark.parametrize("height", [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_material_slice_rejects_invalid_height(height: object) -> None:
    """Material slices require a finite positive height."""
    mesh = HexPlanarMesh(1, pitch=10.0)

    with pytest.raises(ValueError, match="finite and positive"):
        MaterialSlice(mesh, {}, height)


@pytest.mark.parametrize("height", [True, "1.0"])
def test_material_slice_rejects_nonreal_height(height: object) -> None:
    """Material slices require a real height scalar."""
    mesh = HexPlanarMesh(1, pitch=10.0)

    with pytest.raises(TypeError, match="height must be a real numeric value"):
        MaterialSlice(mesh, {}, height)


def test_material_mesh_reports_control_volume_geometry() -> None:
    """Material mesh should provide hex-z geometry by axial layer."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=2.5),),
    )

    assert material_mesh.layer_height(0) == pytest.approx(2.5)
    assert material_mesh.cell_volume(0) == pytest.approx(mesh.area * 2.5)
    assert material_mesh.radial_face_area(0) == pytest.approx(mesh.face_length * 2.5)
    assert material_mesh.axial_face_area(0) == pytest.approx(mesh.area)


def test_material_mesh_exposes_read_only_completed_stack_state() -> None:
    """Completed topology and geometry should not permit unchecked mutation."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice(mesh, {OpenMCIndex(0, 0): "medium"}, 1.0),
            MaterialSlice(mesh, {OpenMCIndex(0, 0): "reflector"}, 1.0),
        )
    )

    with pytest.raises(TypeError):
        material_mesh.layers[1][OpenMCIndex(0, 0)] = "medium"
    with pytest.raises(AttributeError):
        material_mesh.axial_layer_heights = (1.0, -1.0)
    with pytest.raises(TypeError):
        material_mesh.excluded_regions["reflector"] = ExcludedRegion(kind="void")


@pytest.mark.parametrize("active_id", [True, 0.5])
def test_material_mesh_rejects_noninteger_active_ids(active_id: object) -> None:
    """Active-cell queries must reject invalid active-ID types."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (MaterialSlice(mesh, {OpenMCIndex(0, 0): "medium"}, 1.0),)
    )

    with pytest.raises(TypeError, match="active_id"):
        material_mesh.face(0, active_id, "top")  # type: ignore[arg-type]


@pytest.mark.parametrize("active_id", [-1, 1])
def test_material_mesh_rejects_outside_active_ids(active_id: int) -> None:
    """Active-cell queries must reject negative IDs and overflow."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (MaterialSlice(mesh, {OpenMCIndex(0, 0): "medium"}, 1.0),)
    )

    with pytest.raises(ValueError, match="active_id"):
        material_mesh.face(0, active_id, "top")


@pytest.mark.parametrize(
    ("direction", "exception", "match"),
    (
        (None, TypeError, "direction must be a string"),
        ([], TypeError, "direction must be a string"),
        ("", ValueError, "direction must not be empty"),
    ),
)
def test_material_mesh_face_checks_direction_type_and_presence(
    direction: object,
    exception: type[Exception],
    match: str,
) -> None:
    """Face queries should reject invalid directions before topology dispatch."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (MaterialSlice(mesh, {OpenMCIndex(0, 0): "medium"}, 1.0),)
    )

    with pytest.raises(exception, match=match):
        material_mesh.face(0, 0, direction)  # type: ignore[arg-type]


def test_material_mesh_colors_use_material_and_excluded_defaults() -> None:
    """Material colors should combine user, palette, and excluded colors."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(
                mesh,
                [["medium", "0", "reflector", "0", "medium", "0"], ["medium"]],
                height=1.0,
            ),
        ),
    )

    colors = material_mesh.material_colors(
        {"medium": Material(name="medium", color="#ff0000")}
    )

    assert colors["medium"] == "#ff0000"
    assert colors["0"] == INACTIVE_EXCLUDED_REGION_COLOR


def test_material_mesh_colors_use_excluded_region_colors() -> None:
    """Excluded-region colors should come from their descriptions."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(
                mesh, [["medium", "gap", "0", "0", "0", "0"], ["medium"]], height=1.0
            ),
        ),
        excluded_regions={"gap": ExcludedRegion(kind="void", color="#eeeeee")},
    )

    colors = material_mesh.material_colors()

    assert colors["gap"] == "#eeeeee"
    assert colors["0"] == INACTIVE_EXCLUDED_REGION_COLOR


@pytest.mark.parametrize(
    ("materials", "exception", "match"),
    (
        (False, TypeError, "materials must be a mapping"),
        ({1: Material(name="medium")}, TypeError, "materials key must be a string"),
        ({"medium": "not a material"}, TypeError, "values must be Material"),
        ({"medium": Material(name="other")}, ValueError, "keys must match"),
    ),
)
def test_material_mesh_checks_plotting_material_mappings(
    materials: object,
    exception: type[Exception],
    match: str,
) -> None:
    """Plotting material mappings should have complete typed identities."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (MaterialSlice(mesh, {OpenMCIndex(0, 0): "medium"}, 1.0),)
    )

    with pytest.raises(exception, match=match):
        material_mesh.material_colors(materials)  # type: ignore[arg-type]


def test_material_mesh_plot_matplotlib_checks_axes_type() -> None:
    """Matplotlib plotting should reject objects that are not axes."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (MaterialSlice(mesh, {OpenMCIndex(0, 0): "medium"}, 1.0),)
    )

    with pytest.raises(TypeError, match="ax must be an Axes or None"):
        material_mesh.plot_matplotlib(0, ax="not axes")  # type: ignore[arg-type]


def test_material_mesh_matplotlib_plots_selected_axial_slice() -> None:
    """Matplotlib slice plots should color cells by material key."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),),
    )

    axes = material_mesh.plot_matplotlib(
        axial_index=0, materials={"medium": Material(name="medium", color="#ff0000")}
    )

    assert len(axes.patches) == 1
    assert to_hex(axes.patches[0].get_facecolor()) == "#ff0000"
    assert axes.get_legend() is not None


def test_material_mesh_plotly_plots_selected_axial_slice() -> None:
    """Plotly slice plots should expose material hover data and colors."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),),
    )

    figure = material_mesh.to_plotly(
        axial_index=0, materials={"medium": Material(name="medium", color="#ff0000")}
    )

    assert len(figure.data) == 2
    assert figure.data[0].fillcolor == "#ff0000"
    assert figure.data[0].hoverinfo == "skip"
    assert figure.data[0].name == "medium"
    assert figure.data[0].legendgroup == "medium"
    assert figure.data[0].showlegend
    assert "material medium" in figure.data[1].customdata[0]
    assert "domain active" in figure.data[1].customdata[0]
    assert "ring 0, pos 0" in figure.data[1].customdata[0]
    assert "center (0.00, 0.00) cm" in figure.data[1].customdata[0]
    assert figure.data[1].legendgroup == "medium"
    assert figure.data[1].marker.symbol == "hexagon"
    assert figure.layout.legend.groupclick == "togglegroup"


def test_material_mesh_export_vtm_writes_true_hex_prisms(tmp_path) -> None:
    """VTM export leaves should preserve true hexagonal-prism material cells."""
    import vtk

    mesh = HexPlanarMesh(1, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice(mesh, {OpenMCIndex(0, 0): "medium"}, 2.5),
            MaterialSlice(mesh, {OpenMCIndex(0, 0): "0"}, 2.5),
        )
    )
    path = tmp_path / "material_mesh.vtm"

    material_mesh.export_vtm(path)

    assert path.exists()
    active_path, excluded_path = _vtm_leaf_paths(path)
    root = ET.parse(active_path).getroot()
    piece = root.find("./UnstructuredGrid/Piece")
    assert piece is not None
    assert piece.attrib["NumberOfCells"] == "1"
    assert piece.attrib["NumberOfPoints"] == "12"

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(active_path))
    reader.Update()
    grid = reader.GetOutput()
    assert grid.GetNumberOfCells() == 1
    assert grid.GetNumberOfPoints() == 12
    assert grid.GetCellType(0) == vtk.VTK_HEXAGONAL_PRISM
    cell_data = grid.GetCellData()
    assert [
        cell_data.GetArrayName(array_index)
        for array_index in range(cell_data.GetNumberOfArrays())
    ] == [
        "planar_id",
        "active_id",
        "axial_index",
        "domain_group_id",
        "material_key_id",
        "lattice_x",
        "lattice_u",
        "openmc_ring",
        "openmc_position",
    ]
    assert cell_data.GetArray("active_id").GetTuple1(0) == 0
    assert cell_data.GetArray("domain_group_id").GetTuple1(0) == 1
    assert cell_data.GetArray("lattice_x").GetTuple1(0) == 0
    assert cell_data.GetArray("lattice_u").GetTuple1(0) == 0
    assert cell_data.GetArray("openmc_ring").GetTuple1(0) == 0
    assert cell_data.GetArray("openmc_position").GetTuple1(0) == 0

    reader.SetFileName(str(excluded_path))
    reader.Update()
    excluded_grid = reader.GetOutput()
    assert excluded_grid.GetNumberOfCells() == 1
    assert excluded_grid.GetCellType(0) == vtk.VTK_HEXAGONAL_PRISM
    excluded_data = excluded_grid.GetCellData()
    assert math.isnan(excluded_data.GetArray("active_id").GetTuple1(0))
    assert excluded_data.GetArray("domain_group_id").GetTuple1(0) == 0


def test_material_mesh_export_vtm_uses_slice_local_active_ids(tmp_path) -> None:
    """Compact active IDs should reset on each axial layer."""
    import vtk

    mesh = HexPlanarMesh(1, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice(mesh, {OpenMCIndex(0, 0): "medium"}, 1.0),
            MaterialSlice(mesh, {OpenMCIndex(0, 0): "medium"}, 1.0),
        )
    )
    path = tmp_path / "material_mesh.vtm"

    material_mesh.export_vtm(path)

    active_path, _ = _vtm_leaf_paths(path)

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(active_path))
    reader.Update()
    cell_data = reader.GetOutput().GetCellData()

    assert cell_data.GetArray("active_id").GetTuple1(0) == 0
    assert cell_data.GetArray("active_id").GetTuple1(1) == 0
    assert cell_data.GetArray("axial_index").GetTuple1(0) == 0
    assert cell_data.GetArray("axial_index").GetTuple1(1) == 1


def test_material_mesh_export_vtm_writes_domain_blocks(tmp_path) -> None:
    """VTM export should provide active/excluded leaf datasets."""
    import vtk

    mesh = HexPlanarMesh(2, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(
                mesh, [["medium", "0", "0", "0", "0", "0"], ["medium"]], height=1.0
            ),
        ),
    )
    path = tmp_path / "material_mesh.vtm"

    material_mesh.export_vtm(path)

    active_path, excluded_path = _vtm_leaf_paths(path)
    assert path.exists()
    assert active_path.exists()
    assert excluded_path.exists()

    assert active_path == tmp_path / "material_mesh" / "active.vtu"
    assert excluded_path == tmp_path / "material_mesh" / "excluded.vtu"

    reader = vtk.vtkXMLMultiBlockDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    multiblock = reader.GetOutput()
    assert multiblock.GetNumberOfBlocks() == 2

    active_grid = multiblock.GetBlock(0).GetBlock(0)
    excluded_grid = multiblock.GetBlock(1).GetBlock(0)
    assert active_grid.GetNumberOfCells() == 2
    assert excluded_grid.GetNumberOfCells() == 5
    assert active_grid.GetCellType(0) == vtk.VTK_HEXAGONAL_PRISM
    assert excluded_grid.GetCellType(0) == vtk.VTK_HEXAGONAL_PRISM
    assert active_grid.GetCellData().GetArray("domain_group_id").GetTuple1(0) == 1
    assert excluded_grid.GetCellData().GetArray("domain_group_id").GetTuple1(0) == 0


def test_material_mesh_export_vtm_writes_material_key_mapping(tmp_path) -> None:
    """VTM export should preserve the meaning of material-key IDs."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(
                mesh,
                [["fuel", "reflector", "0", "0", "0", "0"], ["coolant"]],
                height=1.0,
            ),
        )
    )
    path = tmp_path / "layout.vtm"

    material_mesh.export_vtm(path)

    mapping_path = tmp_path / "layout" / "material_keys.json"
    assert json.loads(mapping_path.read_text(encoding="utf-8")) == {
        "0": "fuel",
        "1": "reflector",
        "2": "0",
        "3": "coolant",
    }


def test_material_mesh_export_vtm_validates_destination(tmp_path) -> None:
    """VTM export should reject ambiguous or directory destinations."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material_mesh = MaterialMesh.stack((MaterialSlice(mesh, {}, 1.0),))

    with pytest.raises(ValueError, match="\\.vtm"):
        material_mesh.export_vtm(tmp_path / "layout")

    directory_path = tmp_path / "layout.vtm"
    directory_path.mkdir()
    with pytest.raises(IsADirectoryError, match="output path is a directory"):
        material_mesh.export_vtm(directory_path)


def test_material_mesh_export_vtm_reuses_stable_leaf_directory(tmp_path) -> None:
    """Repeated VTM exports should replace one stable leaf directory."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    material_mesh = MaterialMesh.stack(
        (MaterialSlice(mesh, {OpenMCIndex(0, 0): "fuel"}, 1.0),)
    )
    path = tmp_path / "layout.vtm"
    material_mesh.export_vtm(path)
    first_active_path, _ = _vtm_leaf_paths(path)

    material_mesh.export_vtm(path)
    second_active_path, _ = _vtm_leaf_paths(path)

    assert first_active_path == second_active_path == tmp_path / "layout" / "active.vtu"
    assert sorted(child.name for child in tmp_path.iterdir()) == [
        "layout",
        "layout.vtm",
    ]
