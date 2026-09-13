#!/usr/bin/env bash
# Run every maintained Morana example from the repository root.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

for example in \
    material_mesh_domain_faces \
    mesh_plotting \
    quickstart \
    mixed_boundary_regions \
    result_archive \
    one_group_keff \
    multigroup_fixed_source \
    multigroup_keff \
    fixed_source_mms \
    keff_mms \
    solver_comparison
do
    echo "Running examples/$example.py"
    python -m "examples.$example"
done
