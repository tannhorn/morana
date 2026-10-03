"""Use public finite-volume operators with a custom block preconditioner.

The example assembles one small two-group problem, groups each node's energy
unknowns into a dense Jacobi block, and uses that preconditioner with SciPy
GMRES for an external fixed-source solve and power iteration. The candidates
are plain NumPy arrays, not checked Morana results.
"""

# Matplotlib configuration must precede Morana's plotting-capable imports.
# pylint: disable=wrong-import-position

from __future__ import annotations

import os
from pathlib import Path
from tempfile import gettempdir
import warnings

DEFAULT_MATPLOTLIB_CONFIG_DIR = (
    Path(gettempdir()) / "morana_examples" / "matplotlib_config"
)
DEFAULT_MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MATPLOTLIB_CONFIG_DIR))

import numpy as np
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, gmres

from morana import (
    BoundaryCondition,
    BoundaryConditionSet,
    CrossSections,
    FissionData,
    HexPlanarMesh,
    Material,
    MaterialMesh,
    MaterialSlice,
    ProblemConfiguration,
    SeparableFission,
    UniformSource,
)
from morana.operators import (
    CrossSectionData,
    assemble_boundary_rhs,
    assemble_fission_matrix,
    assemble_loss_matrix,
    assemble_source_rhs,
    extract_cross_section_data,
)

LINEAR_TOLERANCE = 1.0e-11
EIGENVALUE_TOLERANCE = 1.0e-10
MAX_POWER_ITERATIONS = 200


def build_configuration() -> ProblemConfiguration:
    """Build a small ragged two-group fuel-reflector problem."""
    mesh = HexPlanarMesh(num_rings=2, pitch=12.0)
    fuel = Material(
        "fuel",
        xs=CrossSections(
            D=[1.30, 0.34],
            sigma_a=[0.009, 0.065],
            sigma_s=[[0.016, 0.075], [0.003, 0.190]],
            fission=FissionData(
                neutron_production=SeparableFission(
                    nu_sigma_f=[0.007, 0.110], chi=[0.995, 0.005]
                )
            ),
        ),
    )
    reflector = Material(
        "reflector",
        xs=CrossSections(
            D=[1.55, 0.42],
            sigma_a=[0.002, 0.016],
            sigma_s=[[0.090, 0.052], [0.002, 0.310]],
            fission=None,
        ),
    )
    material_mesh = MaterialMesh.stack(
        (
            MaterialSlice.from_openmc_rings(
                mesh, [["reflector"] * 6, ["fuel"]], height=1.0
            ),
            MaterialSlice.from_openmc_rings(mesh, [["0"] * 6, ["fuel"]], height=1.5),
        )
    )
    return ProblemConfiguration(
        mesh=mesh,
        materials={"fuel": fuel, "reflector": reflector},
        material_mesh=material_mesh,
        boundary=BoundaryConditionSet(BoundaryCondition.vacuum().globally()),
        source=UniformSource([1.0, 0.0]),
        name="custom_node_block_operator_example",
    )


def node_block_jacobi(matrix: csr_matrix, groups: int) -> LinearOperator:
    """Return a node-block Jacobi approximate-inverse action.

    Morana's group-fastest packing places the ``groups`` unknowns for each
    spatial node next to one another. For an assembled matrix ``K``, this
    function retains each node-local ``groups x groups`` diagonal block and
    discards couplings between different nodes:

    ``B = blockdiag(K[0:G, 0:G], K[G:2G, G:2G], ...)``.

    A loss-matrix block contains the node's local loss and energy-transfer
    coupling; a fixed-source ``loss - fission`` block also contains local
    fission coupling. Spatial leakage between nodes is outside the blocks.

    SciPy's ``M`` argument must approximate the inverse action of ``K``, so the
    returned ``LinearOperator`` maps a vector to the solution of
    ``B @ result = vector``. Each small block is LU-factorized once during
    construction and those factors are reused by every GMRES preconditioner
    application. A singular node block therefore makes preconditioner
    construction fail rather than silently weakening the method.
    """
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % groups:
        raise ValueError("matrix shape must contain complete node blocks")
    factors = []
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=LinAlgWarning)
        # Group-fastest packing makes every contiguous G-by-G diagonal slice
        # the complete energy-space block for exactly one spatial node.
        for start in range(0, matrix.shape[0], groups):
            block = matrix[start : start + groups, start : start + groups]
            factors.append(lu_factor(block.toarray(), check_finite=True))

    def apply(vector: np.ndarray) -> np.ndarray:
        values = np.empty_like(vector, dtype=float)
        for node, factor in enumerate(factors):
            start = node * groups
            # Apply the stored inverse block; SciPy expects M to approximate
            # the inverse operator, rather than the block-diagonal matrix B.
            values[start : start + groups] = lu_solve(
                factor, vector[start : start + groups], check_finite=False
            )
        return values

    return LinearOperator(matrix.shape, matvec=apply, dtype=float)


def unpack_layers(
    packed: np.ndarray, cross_sections: CrossSectionData
) -> tuple[np.ndarray, ...]:
    """Convert documented node-major packing to group-major layer arrays."""
    layers = []
    node_offset = 0
    for layer in cross_sections.layers:
        start = node_offset * cross_sections.groups
        stop = start + layer.active_cells * cross_sections.groups
        layers.append(
            packed[start:stop]
            .reshape(layer.active_cells, cross_sections.groups)
            .T.copy()
        )
        node_offset += layer.active_cells
    return tuple(layers)


def relative_residual(matrix: csr_matrix, vector: np.ndarray, rhs: np.ndarray) -> float:
    """Return a scale-aware residual for one original assembled equation."""
    lhs = matrix @ vector
    scale = np.linalg.norm(lhs) + np.linalg.norm(rhs)
    return 0.0 if scale == 0.0 else float(np.linalg.norm(lhs - rhs) / scale)


def solve_fixed_source_external(
    operator: csr_matrix, rhs: np.ndarray, groups: int
) -> tuple[np.ndarray, float]:
    """Solve the fixed-source equation with custom node-block Jacobi."""
    flux, info = gmres(
        operator,
        rhs,
        M=node_block_jacobi(operator, groups),
        rtol=LINEAR_TOLERANCE,
        atol=0.0,
    )
    if info != 0:
        raise RuntimeError(f"external fixed-source GMRES failed with info={info}")
    return flux, relative_residual(operator, flux, rhs)


def solve_keff_external(
    loss: csr_matrix, fission: csr_matrix, groups: int
) -> tuple[float, np.ndarray, int, float]:
    """Run a compact external power iteration with block-preconditioned GMRES."""
    preconditioner = node_block_jacobi(loss, groups)
    flux = np.ones(loss.shape[0], dtype=float)
    # A unit total fission-emission source makes its next magnitude the
    # unshifted power-iteration estimate of k-effective.
    flux /= np.sum(fission @ flux)
    keff = 1.0

    for iteration in range(1, MAX_POWER_ITERATIONS + 1):
        candidate, info = gmres(
            loss,
            fission @ flux,
            M=preconditioner,
            rtol=LINEAR_TOLERANCE,
            atol=0.0,
        )
        if info != 0:
            raise RuntimeError(f"external criticality GMRES failed with info={info}")
        next_keff = float(np.sum(fission @ candidate))
        if not np.isfinite(next_keff) or next_keff <= 0.0:
            raise RuntimeError(
                "external power iteration produced an invalid k-effective"
            )
        flux = candidate / next_keff
        rhs = (fission @ flux) / next_keff
        residual = relative_residual(loss, flux, rhs)
        change = abs(next_keff - keff) / next_keff
        keff = next_keff
        if change <= EIGENVALUE_TOLERANCE and residual <= EIGENVALUE_TOLERANCE:
            return keff, flux, iteration, residual

    raise RuntimeError("external power iteration did not converge")


def main() -> None:
    """Assemble, solve, unpack, and verify both original equations."""
    snapshot = build_configuration().snapshot()
    cross_sections = extract_cross_section_data(snapshot)
    loss = assemble_loss_matrix(snapshot, cross_sections)
    fission = assemble_fission_matrix(snapshot, cross_sections)
    rhs = assemble_source_rhs(snapshot, cross_sections) + assemble_boundary_rhs(
        snapshot, cross_sections
    )

    fixed_operator = loss - fission
    fixed_flux, fixed_residual = solve_fixed_source_external(
        fixed_operator, rhs, cross_sections.groups
    )
    keff, critical_flux, iterations, critical_residual = solve_keff_external(
        loss, fission, cross_sections.groups
    )

    fixed_layers = unpack_layers(fixed_flux, cross_sections)
    critical_layers = unpack_layers(critical_flux, cross_sections)
    if fixed_residual > LINEAR_TOLERANCE or critical_residual > EIGENVALUE_TOLERANCE:
        raise RuntimeError("an external candidate failed its original equation")
    if np.min(fixed_flux) < 0.0 or np.min(critical_flux) < 0.0:
        raise RuntimeError("an external candidate contains negative flux")

    print("Custom preconditioner: dense node-block Jacobi")
    print("Packed mapping: (axial_index, active_id, group), with group fastest")
    print(f"Layer shapes: {[values.shape for values in fixed_layers]}")
    print(f"Fixed-source original-equation residual: {fixed_residual:.3e}")
    print(f"External k-effective: {keff:.8f} after {iterations} power iterations")
    print(f"Criticality original-equation residual: {critical_residual:.3e}")
    print(
        "Candidate ownership: plain NumPy arrays; no checked Morana Result was created."
    )
    if [values.shape for values in critical_layers] != [
        values.shape for values in fixed_layers
    ]:
        raise RuntimeError("unpacked candidates do not use one common layout")


if __name__ == "__main__":
    main()
