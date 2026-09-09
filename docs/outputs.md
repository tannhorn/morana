# Inspection and output

Plotting and export methods live on the Morana object that owns the data being
inspected. Maintained examples write generated artifacts under
`artifacts/examples/` by default.

Morana's lightweight serializer writes VTU and VTM output directly as VTK XML,
so export does not require the Python `vtk` package. The development
environment includes `vtk` only to reopen generated files with an independent
reader during testing. ParaView and Python VTK are optional tools for
interactive inspection and downstream processing.

## Planar mesh

[`HexPlanarMesh`](reference/core.md#morana.HexPlanarMesh) provides
`plot_matplotlib()`, `to_plotly()`, and `export_vtu(path)` for the full regular
lattice with planar IDs, lattice coordinates, and OpenMC ring-position
indices. The official
[VTK XML format documentation](https://docs.vtk.org/en/latest/vtk_file_formats/vtkxml_file_format.html)
defines `.vtu` as the serial `UnstructuredGrid` format and describes its point,
cell, and data-array structure.

The planar VTU cell arrays are `planar_id`, `lattice_x`, `lattice_u`,
`openmc_ring`, and `openmc_position`.

## Material layout

[`MaterialMesh`](reference/core.md#morana.MaterialMesh) provides
`plot_matplotlib(axial_index, materials)` and
`to_plotly(axial_index, materials)` for one selected axial slice, and
`export_vtm(path)` for true hexagonal-prism cells in separate `active` and
`excluded` leaf datasets. VTK's
[`vtkXMLMultiBlockDataWriter` reference](https://vtk.org/doc/nightly/html/classvtkXMLMultiBlockDataWriter.html)
describes the corresponding multiblock writer and its recursively written
child datasets. For `layout.vtm`, Morana creates the referenced
`layout/active.vtu` and `layout/excluded.vtu` leaves. That directory also
contains `material_keys.json`, which maps `material_key_id` values to their
original material keys. Repeated exports replace these stable files; keep the
index and its directory together.

For material-layout plots, excluded-region colors take precedence. The built-in
inactive key `"0"` is white; a custom region uses its configured
`ExcludedRegion.color` or the default light-gray `#d9d9d9`. Active keys use
the explicit `Material.color` supplied through `materials` when present; all
other active keys receive the repeating default palette in first-use order,
scanning layers bottom-to-top and positions in planar order. Call
[`MaterialMesh.material_colors()`](reference/core.md#morana.MaterialMesh.material_colors)
to inspect the resolved mapping used by both plotting backends.

Both leaves contain these cell arrays:

| Array | Meaning |
| --- | --- |
| `planar_id` | Full-lattice planar ID |
| `active_id` | Slice-local active ID, or `NaN` for an excluded cell |
| `axial_index` | Bottom-to-top layer index |
| `domain_group_id` | `1` for active cells and `0` for excluded cells |
| `material_key_id` | Deterministic integer ID assigned on first encounter, scanning layers bottom-to-top and cells in planar order |
| `lattice_x`, `lattice_u` | Integer lattice coordinates |
| `openmc_ring`, `openmc_position` | OpenMC-style serialized position |

## Plotting conventions

All Matplotlib and Plotly hex plots use black cell edges, Cartesian axis
labels, and equal aspect ratio. Plotly also provides per-cell hover labels.

Plotly renders visible hexagons in batched Cartesian `Scatter` traces grouped
by fill color and, for material layouts, legend group. A transparent hex-marker
overlay supplies the per-cell hover target and remains available while zooming.
Material legend toggles hide both the visible region and its matching hover
targets. Plotly hex figures use a fixed initial canvas so these pixel-sized
hover markers align with the cells.

[`Result`](reference/core.md#morana.Result) flux is compact over active cells.
When it is mapped onto the full planar lattice, excluded positions contain
`NaN`. Both plotting backends render those positions with the fixed light-gray
color `#d9d9d9`. This color is outside the scalar-flux colormap and means “no
solver value,” not zero flux. Finite scalar flux uses the centralized `inferno`
colormap in both backends.

## Solver results

`Result` provides `plot_matplotlib(group, axial_index)` and
`plot_plotly(group, axial_index)` for a selected scalar-flux slice using the
conventions above.

For one active planar position, `result.cell_at(axial_index, openmc_index)`
returns a read-only
[`CellInspection`](reference/core.md#morana.CellInspection) value with its
material key, cross sections, and complete fast-to-thermal flux vector without
exposing the layer-local compact cell ID. Obtain this value through
`cell_at()`; direct construction is not supported.

`Result.export_vtm("solution.vtm")` writes material-aware active and excluded
blocks with the material-layout arrays above and one `flux_gN` field per energy
group, numbered from fast to thermal beginning at `flux_g0`. Excluded-cell
flux fields contain `NaN`. Its `solution/material_keys.json` maps each
`material_key_id` value to its original material key. Open the VTM file in
ParaView and use
**View → MultiBlock Inspector** to toggle blocks, as described in the official
[ParaView MultiBlock documentation](https://docs.paraview.org/en/latest/Tutorials/ClassroomTutorials/advancedMultiBlock.html#multiblock-inspector).

The [geometry guide](geometry.md) defines the IDs and axial ordering preserved
by these outputs. See the [maintained examples](examples.md) for runnable
output workflows, the [result archive guide](result_archives.md) for completed
result persistence, and the [Python reference](reference/index.md) for exact
method signatures.
