"""The reflected six-cell-ring quickstart case."""

from morana.solvers.finite_volume import solve_fixed_source
from morana import (
    BoundaryCondition,
    BoundaryConditionSet,
    CrossSections,
    HexPlanarMesh,
    Material,
    MaterialMesh,
    MaterialSlice,
    ProblemConfiguration,
    UniformSource,
)

mesh = HexPlanarMesh(num_rings=2, pitch=10.0)
medium = Material(
    "medium",
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
        (
            MaterialSlice.from_openmc_rings(
                mesh,
                [["medium"] * 6, ["0"]],
                height=1.0,
            ),
        )
    ),
    boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
    source=UniformSource([1.0]),
)

result = solve_fixed_source(configuration)
print(result.flux_layer(0)[0])  # [50. 50. 50. 50. 50. 50.]
