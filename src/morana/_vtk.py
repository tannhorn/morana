"""Shared private VTK XML serialization helpers."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class VTKDataArray:
    """Describe one ASCII VTK XML data array."""

    data_type: str
    name: str
    values: Iterable[object]
    number_of_components: int | None = None


def prepare_output_path(path: str | Path, suffix: str) -> Path:
    """Validate a VTK destination and create its parent directory.

    Parameters
    ----------
    path
        Destination file path.
    suffix
        Required filename suffix, including the leading dot.

    Returns
    -------
    pathlib.Path
        Checked destination path.

    Raises
    ------
    ValueError
        If ``path`` does not use ``suffix``.
    IsADirectoryError
        If ``path`` identifies an existing directory.
    OSError
        If the destination parent cannot be created.
    """
    output_path = Path(path)
    if output_path.suffix.lower() != suffix:
        raise ValueError(f"output path must have the {suffix!r} suffix: {output_path}")
    if output_path.exists() and output_path.is_dir():
        raise IsADirectoryError(f"output path is a directory: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def write_vtu(
    path: Path,
    *,
    points: Sequence[tuple[float, float, float]],
    connectivity: Sequence[int],
    offsets: Sequence[int],
    cell_types: Sequence[int],
    cell_data: Sequence[VTKDataArray],
) -> None:
    """Write one VTK XML unstructured-grid file."""
    if len(offsets) != len(cell_types):
        raise ValueError("VTU offsets and cell types must have the same length")

    root = ET.Element(
        "VTKFile",
        type="UnstructuredGrid",
        version="0.1",
        byte_order="LittleEndian",
    )
    grid = ET.SubElement(root, "UnstructuredGrid")
    piece = ET.SubElement(
        grid,
        "Piece",
        NumberOfPoints=str(len(points)),
        NumberOfCells=str(len(cell_types)),
    )
    points_element = ET.SubElement(piece, "Points")
    _add_data_array(
        points_element,
        VTKDataArray("Float64", "Points", points, number_of_components=3),
    )

    cells_element = ET.SubElement(piece, "Cells")
    _add_data_array(
        cells_element,
        VTKDataArray("Int64", "connectivity", connectivity),
    )
    _add_data_array(cells_element, VTKDataArray("Int64", "offsets", offsets))
    _add_data_array(cells_element, VTKDataArray("UInt8", "types", cell_types))

    cell_data_element = ET.SubElement(piece, "CellData")
    for array in cell_data:
        _add_data_array(cell_data_element, array)

    _write_xml(path, root)


def write_vtm(path: Path, blocks: Sequence[tuple[str, str]]) -> None:
    """Write a VTK XML multiblock index referencing named leaf datasets."""
    root = ET.Element(
        "VTKFile",
        type="vtkMultiBlockDataSet",
        version="1.0",
        byte_order="LittleEndian",
    )
    multiblock = ET.SubElement(root, "vtkMultiBlockDataSet")
    for block_index, (block_name, relative_file) in enumerate(blocks):
        block = ET.SubElement(
            multiblock,
            "Block",
            index=str(block_index),
            name=block_name,
        )
        ET.SubElement(
            block,
            "DataSet",
            index="0",
            file=relative_file,
        )

    _write_xml(path, root)


def _add_data_array(parent: ET.Element, array: VTKDataArray) -> None:
    """Append one ASCII DataArray to a VTK XML element."""
    attributes = {
        "type": array.data_type,
        "Name": array.name,
        "format": "ascii",
    }
    if array.number_of_components is not None:
        attributes["NumberOfComponents"] = str(array.number_of_components)
    data_array = ET.SubElement(parent, "DataArray", attributes)
    data_array.text = f"\n{_format_values(array.values)}\n"


def _format_values(values: Iterable[object]) -> str:
    """Return a VTK ASCII payload from scalar or tuple values."""
    entries: list[str] = []
    for value in values:
        if isinstance(value, tuple):
            entries.extend(_format_value(component) for component in value)
        else:
            entries.append(_format_value(value))
    return " ".join(entries)


def _format_value(value: object) -> str:
    """Return one VTK-friendly ASCII scalar."""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.15g}"
    return str(value)


def _write_xml(path: Path, root: ET.Element) -> None:
    """Write one consistently formatted VTK XML document."""
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
