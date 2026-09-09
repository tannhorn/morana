#!/usr/bin/env bash
# Run the OpenMC and Morana study stages in separate Python environments.

set -euo pipefail

if [[ -z "${OPENMC_CONDA_ENV:-}" ]]; then
    echo "OPENMC_CONDA_ENV must name the conda environment that provides OpenMC." >&2
    exit 2
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "conda must be available to run the OpenMC stage." >&2
    exit 2
fi
if ! command -v python >/dev/null 2>&1; then
    echo "Activate a Python environment that provides Morana before running this script." >&2
    exit 2
fi

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runner="${script_directory}/run_study_case.py"

for argument in "$@"; do
    if [[ "${argument}" == "-h" || "${argument}" == "--help" ]]; then
        python "${runner}" --help
        exit 0
    fi
done

conda run --no-capture-output -n "${OPENMC_CONDA_ENV}" \
    python "${runner}" "$@" --stage mgxs

python "${runner}" "$@" --stage morana
