#!/usr/bin/env bash
# Run every maintained Morana example from the repository root.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

for example in \
    material_mesh_domain_faces.py \
    mesh_plotting.py \
    quickstart.py \
    mixed_boundary_regions.py \
    result_archive.py \
    one_group_keff.py \
    multigroup_fixed_source.py \
    multigroup_keff.py \
    fixed_source_mms.py \
    keff_mms.py \
    solver_comparison.py
do
    echo "Running examples/$example"
    python "examples/$example"
done
