"""Tests for OpenMC-compatible hex mesh ordering."""

from math import sqrt
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from morana import HexPlanarMesh, OpenMCIndex

EXPECTED_COORDS_BY_RINGS = {
    1: ((0, 0),),
    2: (
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (0, 0),
    ),
    3: (
        (2, 0),
        (2, -1),
        (2, -2),
        (1, -2),
        (0, -2),
        (-1, -1),
        (-2, 0),
        (-2, 1),
        (-2, 2),
        (-1, 2),
        (0, 2),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (0, 0),
    ),
}


def test_num_rings_cell_counts() -> None:
    """Regular hex meshes should use OpenMC-style ring counts."""
    assert HexPlanarMesh(1, pitch=10.0).n_cells == 1
    assert HexPlanarMesh(2, pitch=10.0).n_cells == 7
    assert HexPlanarMesh(3, pitch=10.0).n_cells == 19


@pytest.mark.parametrize("field", ["ring", "position"])
@pytest.mark.parametrize("value", [True, 1.5, "0"])
def test_openmc_index_rejects_noninteger_fields(field: str, value: object) -> None:
    """Standalone OpenMC identities distinguish invalid field types."""
    arguments = {"ring": 0, "position": 0, field: value}
    with pytest.raises(TypeError, match=f"{field} must be an integer"):
        OpenMCIndex(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["ring", "position"])
def test_openmc_index_rejects_negative_fields(field: str) -> None:
    """Standalone OpenMC identities require nonnegative integer fields."""
    arguments = {"ring": 0, "position": 0, field: -1}
    with pytest.raises(ValueError, match=f"{field} must be a nonnegative integer"):
        OpenMCIndex(**arguments)  # type: ignore[arg-type]


def test_planar_id_at_requires_openmc_index() -> None:
    """Reverse planar lookup must reject values outside its typed identity API."""
    with pytest.raises(TypeError, match="openmc_index"):
        HexPlanarMesh(1, pitch=10.0).planar_id_at((0, 0))  # type: ignore[arg-type]


@pytest.mark.parametrize("num_rings", [True, 1.5, "1"])
def test_hex_planar_mesh_rejects_noninteger_ring_counts(num_rings: object) -> None:
    """Ring counts should be integer topology values rather than coercible data."""
    with pytest.raises(TypeError, match="num_rings must be an integer"):
        HexPlanarMesh(num_rings, pitch=10.0)  # type: ignore[arg-type]


def test_hex_planar_mesh_rejects_nonpositive_ring_counts() -> None:
    """Ring counts must retain their positive value constraint."""
    with pytest.raises(ValueError, match="num_rings must be a positive integer"):
        HexPlanarMesh(0, pitch=10.0)


@pytest.mark.parametrize("pitch", [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_hex_planar_mesh_rejects_invalid_pitch(pitch: object) -> None:
    """Mesh geometry requires a finite positive pitch."""
    with pytest.raises(ValueError, match="finite and positive"):
        HexPlanarMesh(1, pitch=pitch)  # type: ignore[arg-type]


def test_hex_planar_mesh_rejects_nonreal_pitch() -> None:
    """Mesh geometry requires a real pitch scalar."""
    with pytest.raises(TypeError, match="pitch must be a real numeric value"):
        HexPlanarMesh(1, pitch=True)  # type: ignore[arg-type]


def test_hex_planar_mesh_geometry_factors_follow_pitch() -> None:
    """Regular hex mesh geometry factors should derive from pitch."""
    mesh = HexPlanarMesh(1, pitch=10.0)

    assert mesh.center_distance == 10.0
    assert mesh.center_to_face == 5.0
    assert mesh.face_length > 0.0
    assert mesh.area > 0.0


def test_hex_planar_mesh_owns_direction_order() -> None:
    """Mesh consumers should obtain the six face labels from the mesh."""
    mesh = HexPlanarMesh(1, pitch=10.0)

    assert mesh.direction_labels == ("x+", "x-", "u+", "u-", "v+", "v-")


@pytest.mark.parametrize(
    "lookup",
    (
        HexPlanarMesh.lattice_coord,
        HexPlanarMesh.openmc_index,
        HexPlanarMesh.neighbors,
        HexPlanarMesh.cartesian_center,
        HexPlanarMesh.cell_vertices,
    ),
)
@pytest.mark.parametrize("planar_id", [True, 1.5, "0"])
def test_planar_id_lookups_reject_noninteger_ids(
    lookup: object, planar_id: object
) -> None:
    """Public planar-ID lookups should reject non-integer inputs consistently."""
    mesh = HexPlanarMesh(2, pitch=10.0)

    with pytest.raises(TypeError, match="planar_id must be an integer"):
        lookup(mesh, planar_id)  # type: ignore[operator]


@pytest.mark.parametrize(
    "lookup",
    (
        HexPlanarMesh.lattice_coord,
        HexPlanarMesh.openmc_index,
        HexPlanarMesh.neighbors,
        HexPlanarMesh.cartesian_center,
        HexPlanarMesh.cell_vertices,
    ),
)
@pytest.mark.parametrize("planar_id", [-1, 7])
def test_planar_id_lookups_reject_outside_lattice_ids(
    lookup: object, planar_id: int
) -> None:
    """Public planar-ID lookups should not use Python negative indexing."""
    mesh = HexPlanarMesh(2, pitch=10.0)

    with pytest.raises(IndexError, match="planar_id is outside the complete lattice"):
        lookup(mesh, planar_id)  # type: ignore[operator]


def test_cell_vertices_follow_hex_geometry() -> None:
    """Each full-lattice cell should expose six Cartesian vertices."""
    mesh = HexPlanarMesh(1, pitch=10.0)

    vertices = mesh.cell_vertices(0)

    assert len(vertices) == 6
    assert vertices[0] == pytest.approx((5.0, 5.0 / sqrt(3.0)))
    assert vertices[3] == pytest.approx((-5.0, -5.0 / sqrt(3.0)))


@pytest.mark.parametrize("num_rings", [1, 2, 3])
def test_openmc_x_ordering_oracle(num_rings: int) -> None:
    """Planar IDs should follow OpenMC orientation-x ring order exactly."""
    mesh = HexPlanarMesh(num_rings, pitch=10.0)
    expected_coords = EXPECTED_COORDS_BY_RINGS[num_rings]
    expected_indices = []
    for serialized_ring, radius in enumerate(range(num_rings - 1, -1, -1)):
        ring_size = max(6 * radius, 1)
        expected_indices.extend(
            OpenMCIndex(serialized_ring, position) for position in range(ring_size)
        )

    assert mesh.coords == expected_coords
    assert mesh.openmc_indices == tuple(expected_indices)
    for planar_id, openmc_index in enumerate(expected_indices):
        assert mesh.openmc_index(planar_id) == openmc_index
        assert mesh.lattice_coord(planar_id) == expected_coords[planar_id]
        assert mesh.planar_id_at(openmc_index) == planar_id


def test_cartesian_centers_match_lattice_coordinates() -> None:
    """Cartesian centers should use standard nodal x/u lattice coordinates."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    expected_centers = (
        (10.0, 0.0),
        (5.0, -5.0 * sqrt(3.0)),
        (-5.0, -5.0 * sqrt(3.0)),
        (-10.0, 0.0),
        (-5.0, 5.0 * sqrt(3.0)),
        (5.0, 5.0 * sqrt(3.0)),
        (0.0, 0.0),
    )

    for planar_id, expected in enumerate(expected_centers):
        assert mesh.cartesian_center(planar_id) == pytest.approx(expected)


def test_interior_cell_has_six_neighbors() -> None:
    """The center of a three-ring mesh should have six active neighbors."""
    mesh = HexPlanarMesh(3, pitch=1.0)

    assert mesh.neighbors(18) == (12, 15, 17, 14, 16, 13)


def test_corner_cell_has_three_missing_full_lattice_neighbors() -> None:
    """Outer-ring corner cells should have three missing mesh neighbors."""
    mesh = HexPlanarMesh(3, pitch=1.0)

    assert mesh.neighbors(0).count(None) == 3


def test_neighbor_relation_is_symmetric() -> None:
    """If one cell names another as a neighbor, the relation is symmetric."""
    mesh = HexPlanarMesh(3, pitch=1.0)

    for planar_id in range(mesh.n_cells):
        for neighbor in mesh.neighbors(planar_id):
            if neighbor is not None:
                assert planar_id in mesh.neighbors(neighbor)


def test_matplotlib_plot_contains_one_patch_and_label_per_cell() -> None:
    """Mesh inspection plots should label every full-lattice cell."""
    mesh = HexPlanarMesh(2, pitch=1.0)

    axes = mesh.plot_matplotlib()

    assert len(axes.patches) == mesh.n_cells
    assert len(axes.texts) == mesh.n_cells + 1
    assert axes.texts[0].get_text() == "0\n(1, 0)\n0/0"
    assert "Line 1: planar ID" in axes.texts[-1].get_text()


@pytest.mark.parametrize("ax", ["axes", object()])
def test_matplotlib_plot_requires_axes_or_none(ax: object) -> None:
    """Planar mesh plotting should validate an explicitly supplied axes."""
    with pytest.raises(TypeError, match="ax must be an Axes or None"):
        HexPlanarMesh(1, pitch=1.0).plot_matplotlib(ax=ax)  # type: ignore[arg-type]


def test_plotly_figure_contains_mesh_traces() -> None:
    """Plotly inspection output should contain polygons and labels."""
    mesh = HexPlanarMesh(1, pitch=1.0)

    figure = mesh.to_plotly()

    assert len(figure.data) == 3
    assert figure.data[1].text[0] == "0<br>(0, 0)<br>0/0"
    assert figure.data[1].textfont.size == 12
    assert figure.data[0].hoverinfo == "skip"
    assert figure.data[2].customdata[0] == (
        "planar_id 0<br>lattice (0, 0)<br>ring 0, pos 0<br>" "center (0.00, 0.00) cm"
    )
    assert figure.data[2].hovertemplate == "%{customdata}<extra></extra>"
    assert figure.data[2].marker.symbol == "hexagon"


def test_plotly_mesh_cells_with_one_style_share_one_fill_trace() -> None:
    """Plotly mesh inspection should batch same-style visible hexagons."""
    mesh = HexPlanarMesh(2, pitch=1.0)

    figure = mesh.to_plotly()

    assert len(figure.data) == 3
    assert len(figure.data[0].x) == 8 * mesh.n_cells


def test_export_vtu_writes_full_lattice_cell_arrays(tmp_path) -> None:
    """VTU mesh export should include full-lattice IDs and OpenMC indices."""
    import vtk

    mesh = HexPlanarMesh(2, pitch=1.0)
    path = tmp_path / "mesh.vtu"

    mesh.export_vtu(path)

    root = ET.parse(path).getroot()
    piece = root.find("./UnstructuredGrid/Piece")
    assert piece is not None
    assert piece.attrib["NumberOfCells"] == str(mesh.n_cells)

    cell_data = piece.find("CellData")
    assert cell_data is not None
    arrays = {
        data_array.attrib["Name"]: data_array.text
        for data_array in cell_data.findall("DataArray")
    }
    assert arrays["planar_id"].split() == [str(planar_id) for planar_id in range(7)]
    assert arrays["openmc_ring"].split() == ["0", "0", "0", "0", "0", "0", "1"]
    assert arrays["openmc_position"].split() == ["0", "1", "2", "3", "4", "5", "0"]

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()
    assert grid.GetNumberOfCells() == mesh.n_cells
    assert grid.GetNumberOfPoints() == 6 * mesh.n_cells
    assert grid.GetCellType(0) == vtk.VTK_POLYGON
    assert grid.GetCellData().GetArray("planar_id") is not None


def test_export_vtu_validates_destination_and_creates_parent_directory(
    tmp_path,
) -> None:
    """VTU export should prepare only valid file destinations."""
    mesh = HexPlanarMesh(1, pitch=1.0)

    nested_path = tmp_path / "nested" / "mesh.vtu"
    mesh.export_vtu(nested_path)
    assert nested_path.exists()

    with pytest.raises(ValueError, match="\\.vtu"):
        mesh.export_vtu(tmp_path / "mesh.vtm")

    directory_path = tmp_path / "directory.vtu"
    directory_path.mkdir()
    with pytest.raises(IsADirectoryError, match="output path is a directory"):
        mesh.export_vtu(directory_path)
