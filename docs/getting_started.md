# Installation and quickstart

## Requirements

Morana requires Python 3.12 or newer. The source tree currently has package
version `0.1.0`, but that version has not been released and no package-index
distribution is available. Install it from a source checkout.

!!! warning "Supported platforms"
    Morana is developed and verified on Linux. Native Windows and macOS
    execution are not claimed as supported platforms. Windows users can use
    [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install),
    which provides a Linux distribution and Bash environment on Windows. Run
    the Linux installation commands below inside that environment. WSL itself
    has not been separately verified for Morana.

## User installation

Create an isolated virtual environment and install a regular, non-editable
copy from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

The installation brings in Morana's runtime dependencies from
`pyproject.toml`, including `h5py`, which supports the
[OpenMC runtime-MGXS material importer](openmc_mgxs.md). OpenMC itself is not
a Morana runtime dependency; the maintained transport comparison needs it for
model construction, geometry checks, and CE/MGXS calculations. Python VTK is
also unnecessary at runtime because Morana writes VTU and VTM files directly
as VTK XML. Install ParaView or Python VTK only to
inspect those files with an external tool.

## Development installation

Contributors and maintainers use the project Conda environment. It installs
Morana in editable mode together with its test, documentation, and file-reader
verification dependencies.

Create the environment from the repository root:

```bash
conda env create -f environment.yml
conda activate morana-dev
```

Update an existing environment after `environment.yml` changes:

```bash
conda env update -n morana-dev -f environment.yml --prune
conda activate morana-dev
```

Changes under `src/morana` are immediately importable. See the
[contributor workflow](contributor_workflow.md) for required development
checks.

## Quickstart

The following reflected six-cell ring has the uniform analytic flux
$\phi=Q/\Sigma_a=50$ in every active cell:

```python
--8<-- "examples/quickstart.py"
```

Even a one-layer mesh is a finite hex-z layer. The global reflective condition
above covers its radial, bottom, top, and excluded-region faces. To model a
two-dimensional radial problem with different radial conditions, assign
reflective `bottom` and `top` conditions explicitly, then assign the radial
condition. The six active cells are the outer ring; the center entry `"0"` is
the built-in excluded-region key.

## Capability boundaries

Before building a larger calculation, review the
[modeling and solver workflow](modeling_workflow.md#build-a-problem) and its
[capability boundaries](modeling_workflow.md#capability-boundaries).

Continue with the [maintained examples](examples.md), then consult
[geometry and indexing](geometry.md) and the
[modeling and solver workflow](modeling_workflow.md). For nonuniform boundary
assignments, use [boundary conditions and face selection](boundary_conditions.md).
