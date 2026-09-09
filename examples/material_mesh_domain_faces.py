"""Material-mesh active-domain and face-classification example.

This example demonstrates the current mesh/material-domain workflow: build a
full regular ``HexPlanarMesh``, overlay material keys with ``MaterialMesh``, and
query active-position faces without introducing finite-volume assembly details.
"""

from morana import HexPlanarMesh, MaterialMesh, MaterialSlice, OpenMCIndex


def build_material_mesh() -> MaterialMesh:
    """Build a small ragged active domain on a full two-ring lattice."""
    mesh = HexPlanarMesh(num_rings=2, pitch=10.0)
    return MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(
                mesh,
                [
                    ["medium", "0", "reflector", "0", "medium", "0"],
                    ["medium"],
                ],
                height=1.0,
            ),
        )
    )


def describe_face(
    material_mesh: MaterialMesh,
    active_id: int,
    direction: str,
    axial_index: int,
) -> str:
    """Return a compact text description for one active-position face."""
    face = material_mesh.face(
        axial_index=axial_index,
        active_id=active_id,
        direction=direction,
    )
    if face.neighbor_active_id is None:
        neighbor = "none"
    else:
        neighbor = f"active_id {face.neighbor_active_id}"

    reserved = ""
    if face.neighbor_key_kind is not None:
        reserved = f", key={face.neighbor_key!r}, kind={face.neighbor_key_kind}"

    return (
        f"active_id {active_id} {direction}: "
        f"{face.kind}, neighbor={neighbor}{reserved}"
    )


def main() -> None:
    """Print active-material assignments and representative face descriptions."""
    material_mesh = build_material_mesh()
    mesh = material_mesh.mesh
    axial_index = 0
    center_active_id = material_mesh.active_id_at(axial_index, OpenMCIndex(1, 0))
    right_active_id = material_mesh.active_id_at(axial_index, OpenMCIndex(0, 0))
    if center_active_id is None or right_active_id is None:
        raise RuntimeError("example material mesh did not create expected cells")

    print("MaterialMesh domain-face example")
    print(f"Full lattice positions: {mesh.n_cells}")
    print("Active solver cells: " f"{material_mesh.n_active_cells(axial_index)}")
    print("Active material assignments:")
    for active_id, material_name in material_mesh.material_by_active_id(
        axial_index
    ).items():
        openmc_index = material_mesh.openmc_index_for_active_id(axial_index, active_id)
        print(f"  active_id {active_id}: {openmc_index} -> {material_name}")

    print("Representative faces:")
    for active_id, direction in (
        (right_active_id, "x+"),
        (center_active_id, "x+"),
        (center_active_id, "v-"),
    ):
        print(f"  {describe_face(material_mesh, active_id, direction, axial_index)}")


if __name__ == "__main__":
    main()
