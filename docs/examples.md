# Maintained examples

Run examples from the repository root in an environment where Morana is
installed. The regular user installation in the
[installation and quickstart guide](getting_started.md#user-installation) is
sufficient. Examples that generate artifacts write them under
`artifacts/examples/` by default; that directory is ignored by Git. Those
examples accept `--output-dir PATH` to choose another destination. The
domain-face inspection example reports to the console and does not create
artifacts.

The source repository's
[`examples/`](https://github.com/tannhorn/morana/tree/main/examples) directory
contains the examples; they are excluded from the installed runtime package.
Run them from a checkout that matches the Morana version being evaluated.

## Run the routine example suite

The repository
[`run_all.sh`](https://github.com/tannhorn/morana/blob/main/examples/run_all.sh)
runner executes the routine examples in this catalog, including the
verification studies. The computationally expensive OpenMC comparison is the
sole exception; it has a separate staged workflow.

```bash
examples/run_all.sh
```

## Start with a small solve

### Quickstart

[`quickstart.py`](https://github.com/tannhorn/morana/blob/main/examples/quickstart.py)
is the executable source rendered in the
[installation and quickstart guide](getting_started.md#quickstart). It solves
the reflected six-cell ring and reports its uniform analytic flux.

```bash
python examples/quickstart.py
```

## Build and inspect geometry

### Mesh inspection and export

[`mesh_plotting.py`](https://github.com/tannhorn/morana/blob/main/examples/mesh_plotting.py)
writes full-lattice and axial-slice Matplotlib/Plotly output,
planar VTU data, and material-layout VTM data.

```bash
python examples/mesh_plotting.py
```

### Domain-face inspection

[`material_mesh_domain_faces.py`](https://github.com/tannhorn/morana/blob/main/examples/material_mesh_domain_faces.py)
demonstrates ring-ordered material input,
slice-local active IDs, and representative `internal`, `outer`, and
`to_excluded` face classifications.

```bash
python examples/material_mesh_domain_faces.py
```

### Mixed boundary regions

[`mixed_boundary_regions.py`](https://github.com/tannhorn/morana/blob/main/examples/mixed_boundary_regions.py)
demonstrates boundary selection by radial exterior,
excluded key, excluded kind, and excluded-face direction. Reflective, vacuum,
Robin, partial-current return, and incoming-current conditions all resolve on
represented faces.

```bash
python examples/mixed_boundary_regions.py
```

## Solve and inspect responses

### Multigroup fixed source

[`multigroup_fixed_source.py`](https://github.com/tannhorn/morana/blob/main/examples/multigroup_fixed_source.py)
demonstrates an axially heterogeneous two-layer,
variable-height external-source problem with top vacuum, downscatter, thermal
fission emission, nonzero axial leakage, and matched unit-multiplicity and
multiplying-scatter cases. It reports group flux sums with signed
`scattering_coupling` and `net_scattering` balances, then writes separate
axial-slice PNG/HTML and VTM artifacts for each case.

```bash
python examples/multigroup_fixed_source.py
```

### Multigroup criticality

[`multigroup_keff.py`](https://github.com/tannhorn/morana/blob/main/examples/multigroup_keff.py)
demonstrates a two-group, axially heterogeneous
fuel/reflector eigenvalue solve across two variable-height layers. It includes
top vacuum, explicit P0 scattering, nonzero axial leakage,
recoverable-power-normalized flux, group balances, axial-slice inspection, and
VTM output. It solves the same fuel data once as
`SeparableFission(nu_sigma_f, chi)` and once as the equivalent event-oriented
`FissionTransfer(fission_transfer[g_from, g_to])`, then verifies agreement of
the fission matrices, `k_eff`, flux, and group balances. Its illustrative fuel
data includes groupwise `kappa_sigma_f`; each completed result is normalized to
100 kW.

```bash
python examples/multigroup_keff.py
```

### Result archive

[`result_archive.py`](https://github.com/tannhorn/morana/blob/main/examples/result_archive.py)
solves the layered multigroup fixed-source case, writes a
non-pickle `.morana-result` archive, reloads it through `Result.load_from_disk`,
and checks every group and flux layer against the original result.

```bash
python examples/result_archive.py
```

## Verify and compare

### One-group criticality

[`one_group_keff.py`](https://github.com/tannhorn/morana/blob/main/examples/one_group_keff.py)
solves a reflected-radial, one-dimensional axial
core-reflector criticality problem. It compares `k_eff` and a
source-normalized cell-average axial flux with an analytic two-region
eigenfunction, and writes an axial-comparison PNG and material-aware VTM
output. The [one-dimensional axial core-reflector case](one_group_keff.md)
derives the reference solution and records the maintained comparison.

```bash
python examples/one_group_keff.py
```

### Fixed-source manufactured solution

[`fixed_source_mms.py`](https://github.com/tannhorn/morana/blob/main/examples/fixed_source_mms.py)
is a verification-focused three-group refinement study.
It exercises non-unit scattering multiplicity and uses a uniquely keyed
inactive excluded shell with directional excluded-face selectors to apply
independently evaluated local Dirichlet data to a refining variable-height
hex-z core. The derivation, acceptance criteria, and recorded convergence
evidence are in the
[fixed-source manufactured-solution case](fixed_source_mms.md).

```bash
python examples/fixed_source_mms.py
```

### k-effective manufactured solution

[`keff_mms.py`](https://github.com/tannhorn/morana/blob/main/examples/keff_mms.py)
is a verification-focused one-group eigenvalue refinement
study. It manufactures cell-local fission-production cross sections and
face-local homogeneous Robin coefficients for a positive three-dimensional
Gaussian mode on the same refining variable-height hex-z domain as the
fixed-source example above. The derivation, acceptance criteria, and recorded
convergence evidence are in the
[k-effective manufactured-solution case](keff_mms.md).

```bash
python examples/keff_mms.py
```

### Finite-volume strategy comparison

[`solver_comparison.py`](https://github.com/tannhorn/morana/blob/main/examples/solver_comparison.py)
runs the maintained strategy set once on refinement
level 1 of the independently manufactured cases. The fixed-source set covers
direct solving and GMRES with no, Jacobi, or ILU preconditioning. The
criticality set applies those four linear policies to ordinary power iteration
and also exercises fixed-Wielandt iteration with direct and GMRES–ILU inner
solves. Its compact console table reports configuration setup and solve times,
outer and total inner iterations, final residuals, and the relevant
manufactured-reference errors. Timing is informative only: portability and
correctness rely on the MMS residual and reference-error checks, not relative
wall-clock performance. Run either MMS example with `--compare-strategies` to
apply its full three-level acceptance criteria to that example's maintained
set and write a `strategy_comparison.csv` evidence table beside its usual
artifacts.

```bash
python examples/solver_comparison.py
```

## Import and compare external material data

### OpenMC–Morana SRE-derived comparison

The repository's
[`examples/openmc_comparison/`](https://github.com/tannhorn/morana/tree/main/examples/openmc_comparison)
directory demonstrates the complete path from a
heterogeneous OpenMC continuous-energy model through locally generated
runtime-MGXS data to a Morana diffusion calculation. The
[documented comparison](openmc_comparison.md) explains the problem definition,
figures, numerical results, and interpretation; the
[reproduction workflow](openmc_comparison_workflow.md) gives the staged commands.

OpenMC, a configured evaluated-data library, and substantial particle
transport are required to reproduce the documented result, so this workflow is
not part of `examples/run_all.sh`. Its deterministic geometry plot, coordinate
mapping, and comparison-reader checks can be run independently of OpenMC.

The [OpenMC MGXS import guide](openmc_mgxs.md) separately defines the accepted
runtime file, diffusion conventions, data conversion, and rejection
boundaries.

See [inspection and output](outputs.md) for artifact conventions and the
[verification guide](verification.md) for the regression evidence behind these
workflows.
