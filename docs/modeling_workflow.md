# Modeling and solver workflow

This guide connects Morana's public objects into a calculation. Use
the [quickstart](getting_started.md#quickstart) for the introductory runnable
calculation and the [maintained examples](examples.md) for larger cases. The
[source-derived Python reference](reference/index.md) is canonical for exact
signatures, attributes, and methods. The
[theory and numerical conventions](theory_references.md) explain the governing
equations and numerical derivations.

The workflow is:

1. [Define the planar lattice and material layout](#define-geometry-and-material-layout).
2. [Define or import materials](#add-materials),
   [assign boundaries](#assign-boundaries), and
   [add a source](#add-a-fixed-source) when needed.
3. [Create and solve a `ProblemConfiguration`](#configure-and-run-the-solver).
4. [Inspect the returned `Result`](#interpret-a-result), including its
   immutable input snapshot, balance, and execution report.

## API stability

Morana is in early development. Its public Python API may change, including
through incompatible changes, without a backward-compatibility guarantee.
For the unreleased `0.1.0` source tree, record the exact Git commit and
dependency environment: the package version alone does not identify a source
revision. Once releases are available, pin exact versions and review the
[changelog](changelog.md) before upgrading.

## Build a problem

[`ProblemConfiguration`](reference/core.md#morana.ProblemConfiguration) is the
mutable owner of a problem definition, while
[`ProblemConfigurationSnapshot`](reference/core.md#morana.ProblemConfigurationSnapshot)
is its immutable, independently owned representation.
[`morana.solvers.finite_volume`](reference/finite_volume.md) provides stateless
finite-volume solve functions that accept either a configuration or its
snapshot. Solver-produced [`Result`](reference/core.md#morana.Result) objects
retain the snapshot used for the calculation, so later edits to the original
configuration cannot change result provenance. A snapshot can also be solved
again or converted to an independent mutable configuration with
`to_configuration()`.

### Define geometry and material layout

Create one [`HexPlanarMesh`](reference/core.md#morana.HexPlanarMesh), then
assign keys to one or more immutable
[`MaterialSlice`](reference/core.md#morana.MaterialSlice) objects and stack
them with [`MaterialMesh.stack(...)`](reference/core.md#morana.MaterialMesh.stack).
Slices are ordered bottom to top and must all reference the same planar mesh.
A one-layer stack represents a finite hex-z problem. Reflective bottom and top
faces reduce it to the usual two-dimensional radial problem.

[`MaterialSlice.from_openmc_rings(...)`](reference/core.md#morana.MaterialSlice.from_openmc_rings)
accepts the documented OpenMC-style ring order. Unassigned positions receive
the built-in excluded key `"0"`; additional
[`ExcludedRegion`](reference/core.md#morana.ExcludedRegion) definitions are
local to the material mesh. Each layer has its own compact active-cell order,
so active-cell counts may differ between layers. See
[geometry and indexing](geometry.md) for lattice IDs, face topology, axial
ordering, and the distinction between `planar_id` and slice-local `active_id`.

### Add materials

Each [`Material`](reference/core.md#morana.Material) associates one layout key
with optional cross sections and a display color. Material names, layout keys,
and excluded-region kinds are human-readable identifiers: they must contain a
non-whitespace character and only printable characters, while preserving exact
spelling. Material names are also the keys used by slices, sources, and the
configuration mapping. `xs=None` is useful for layout inspection, but a solve
requires cross sections for every active material and one common group count
across them.

[`CrossSections`](reference/core.md#morana.CrossSections) requires group vectors
`D` in cm and `sigma_a` in `1 / cm`, a square `sigma_s` matrix in `1 / cm`,
and an explicit `fission` value. These numerical inputs must be finite and
nonnegative; zero diffusion is accepted. Supply a zero `sigma_s` matrix when
there is no scattering.

Ordinary P0 scattering events use `sigma_s[g_from, g_to]`. The optional
`multiplicity_matrix[g_from, g_to]` specifies scattering-neutron emission per
event; `None` is the compact unit-multiplicity convention. A nonfissile value
uses `fission=None`. A fissionable value supplies `FissionData` containing
either compact `SeparableFission(nu_sigma_f, chi)` or general
`FissionTransfer(fission_transfer=...)` data. Both expose canonical
`fission_transfer[g_from, g_to]`, and the selected representation is retained
in configuration snapshots and result archives.

Scattering and fission therefore share an incident-to-outgoing array
orientation. Assembly places a transfer from `g_from` to `g_to` in the
`g_to` row and `g_from` column. The
[theory guide](theory_references.md#scattering-fission-and-operator-split)
defines the resulting signed scattering coupling, fission emission, and
balances. One important modeling constraint is checked during loss assembly:
zero and other nonnegative scattering multiplicities are valid, including
same-group values above one, but the assembled loss diagonal must remain
positive after scattering coupling and leakage are included. In a one-group
material, self-scattering cancels only at unit multiplicity; the
[theory guide](theory_references.md#scattering-fission-and-operator-split)
derives this limit explicitly.

Materials and cross sections are immutable. Construct a replacement, then use
`ProblemConfiguration.replace_material()` or `set_materials()` to install it.
See [inspection and output](outputs.md#material-layout) for color precedence.

### Import OpenMC MGXS material data

`CrossSections.from_openmc_mgxs_hdf5(...)` reads one explicitly selected
macroscopic material and temperature from a supported OpenMC runtime-MGXS HDF5
library. It constructs the same checked `CrossSections` used by native Python
inputs, including optional scattering multiplicity, separable or general
fission production, and recoverable-energy data. The caller wraps the returned
cross sections in a `Material` with a Morana name and display color.

The required `diffusion` argument selects total-cross-section or P1
transport-corrected coefficients. OpenMC is not imported at runtime, and no
geometry, layout, source, or boundary data are read. See the
[OpenMC MGXS import guide](openmc_mgxs.md) for the accepted HDF5 artifact,
data and unit contracts, warnings, limitations, and a runnable example.

### Add a fixed source

Fixed-source calculations may use a built-in volumetric source, an
inhomogeneous boundary contribution, or both. The available built-in
volumetric source definitions are:

- [`UniformSource`](reference/core.md#morana.UniformSource) applies the same
  group vector to every active cell;
- [`MaterialSource`](reference/core.md#morana.MaterialSource) assigns a group
  vector by material name; and
- [`CellSource`](reference/core.md#morana.CellSource) supplies explicit,
  possibly ragged, group-major arrays for the bottom-to-top layers.

Source values are volumetric in `n / cm^3 / s` and follow each layer's compact
active-ID order. Assembly checks finiteness, nonnegativity, and compatibility
with the active layout and group count; constructing a source alone does not
establish those properties. `MaterialSource` requires an entry for every active
material, including an explicit zero vector for materials without a source.

Omit `source` (or clear it with `set_source(None)`) for a boundary-only
fixed-source calculation. A criticality solve does not accept an
independent source.
The [Python reference](reference/core.md) defines each source form's exact
array contract. Sources are immutable; construct a replacement and install it
with `ProblemConfiguration.set_source()`.

### Assign boundaries

[`BoundaryConditionSet`](reference/core.md#morana.BoundaryConditionSet) pairs
a [`BoundaryCondition`](reference/core.md#morana.BoundaryCondition) with a
topology-only selector. A broad `globally()` assignment can provide a fallback,
while physical-exterior and excluded-interface selectors refine individual
parts of the domain. Every exposed face must resolve before loss assembly or a
solve.

Morana supports reflective, vacuum, prescribed-flux, scalar Robin,
partial-current-return, and incoming-current conditions. Nonzero prescribed
flux or current contributes to a fixed-source right-hand side; criticality
requires homogeneous resolved boundary data.

The [boundary conditions and face selection](boundary_conditions.md) guide
explains the selector hierarchy, condition catalog, precedence rules, coverage
checks, and supported scope. The
[exposed-boundary theory](theory_references.md#exposed-boundary-conditions)
derives the corresponding finite-volume equations.

### Configure and run the solver

#### Assemble the configuration

Create `ProblemConfiguration` from the mesh, material mapping, material mesh,
boundary set, and optional source. It checks that every active key identifies a
known material, that material names do not collide with excluded-region keys,
and that the material mesh uses the same mesh. Use
[`set_boundary()`](reference/core.md#morana.ProblemConfiguration.set_boundary),
[`add_boundary()`](reference/core.md#morana.ProblemConfiguration.add_boundary),
[`assign_material()`](reference/core.md#morana.ProblemConfiguration.assign_material),
and [`set_source()`](reference/core.md#morana.ProblemConfiguration.set_source)
when changing an existing problem.
`set_materials()`, `replace_material()`, and `set_material_mesh()` replace
checked material definitions or layout, while `set_name()` changes its
provenance label. Its public state is read-only: direct field replacement raises
`AttributeError`, and its material mapping is read-only. `assign_material()`
takes an explicit bottom-to-top axial index and OpenMC lattice position, and
changes only that layer.

The material mapping may also contain definitions that no active mesh cell
uses. `unused_material_names` reports those names across all axial
layers; call `check_no_unused_materials()` when a workflow requires every
configured material to be assigned. `check_no_unused_materials()` raises
`ValueError` listing the unused names.

#### Select the solve mode

Import `solve_fixed_source` or `solve_keff` from `morana.solvers.finite_volume`.
Each function accepts a `ProblemConfiguration` or a
`ProblemConfigurationSnapshot`. A mutable configuration is captured at entry;
a supplied snapshot is used directly. The functions retain no matrix,
factorization, or preconditioner after they return; a successful `Result`
contains the same snapshot and immutable settings. Snapshot capture is not
synchronized, so capture the solve input before another thread can edit the
configuration:

```python
solve_input = configuration.snapshot()
result = solve_fixed_source(solve_input)
```

| Solve mode | Solve function call | Problem prerequisite | Boundary and source restrictions |
| --- | --- | --- | --- |
| Fixed source | `solve_fixed_source(configuration, settings=None)`; omitting `settings` creates default `FixedSourceSettings` for that call | A usable loss-minus-fission system | Volumetric and boundary sources are optional; with neither, a nonsingular system returns zero flux. |
| Criticality | `solve_keff(configuration, normalization, settings=None)`; `normalization` is a `FissionSourceNormalization` or `PowerNormalization`, and omitting `settings` creates default `KeffSettings` for that call | Positive fission production | Boundaries must be homogeneous, and the configuration must not contain an independent source. |

#### Choose numerical policies

`FixedSourceSettings` and `KeffSettings` are separate immutable value objects.
Their `linear_solve` and `inner_linear_solve` fields respectively own a typed
`DirectLinearSolveSettings` or `GmresLinearSolveSettings` policy. Direct sparse
solving is the default; `GmresLinearSolveSettings` selects restarted SciPy
GMRES from a zero initial guess with one typed
`NoPreconditioner`, `JacobiPreconditioner`, or `IluPreconditioner`. Jacobi
uses the assembled diagonal, while threshold-ILU uses its configured drop
tolerance and fill-factor limit. Fixed-source GMRES constructs its
preconditioner for that one solve. Criticality constructs a direct
factorization or GMRES preconditioner once per `solve_keff()` call and reuses
that setup for the call's inner right-hand sides.
The [theory and numerical conventions](theory_references.md#linear-algebra-execution)
define the direct/GMRES residual, left preconditioning, ordinary power
iteration, and fixed-Wielandt transformation. All numerical controls are
finite; residual controls must be positive, while the flux-roundoff tolerance
may be zero. `KeffSettings` owns explicit `max_outer_iterations` and the three
criticality convergence tolerances. Its `eigenvalue_iteration` policy defaults
to `PowerIterationSettings()` and may instead be
`WielandtShiftSettings(shift_inverse_keff=...)`. The shift is never adapted;
an unusable shifted operator, failed inner solve, nonfinite inverse
multiplication factor, or candidate flux below the negative roundoff limit
raises `ValueError` rather than falling back to ordinary power iteration.
In both solve modes, the flux-roundoff threshold is relative to the candidate
vector's largest absolute component.

#### Understand execution and convergence

Direct fixed-source execution uses SciPy
[`spsolve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html).
GMRES execution uses SciPy
[`gmres`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html)
and records its completed Krylov-iteration count. Criticality direct solving
reuses one SciPy
[`splu`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html)
factorization during power iteration. Every completed direct or GMRES solve
must meet the selected policy's independently calculated true-relative-residual
bound after SciPy returns; backend convergence alone is insufficient.
Criticality records multiplication-factor, flux-shape, and equation-residual
histories. The independently checked linear residual applies to the selected
ordinary or shifted inner operator; the outer residual always checks the
original unshifted eigenvalue equation.

For fixed source, the optional configured volumetric source and any
inhomogeneous resolved boundary terms form the right-hand side of the coupled
loss-minus-fission system. The configured direct or GMRES solution must meet
its typed policy's true-residual bound, and only negative flux within
`flux_nonnegativity_tolerance` times the solution's largest magnitude is
cleaned to zero. The result has `keff=None` and one recorded linear solve.

For criticality, Morana starts fission-source-normalized power iteration from
a positive vector. Multiplication-factor change, volume-weighted flux-shape
change, and the relative eigenvalue-equation residual must all meet their
explicit tolerances within `max_outer_iterations`.

#### Normalize criticality flux

A criticality solve requires either `FissionSourceNormalization` with a
positive rate or `PowerNormalization` with a positive target in W. The
requested normalization scales the converged flux without changing `k_eff`.
`FissionSourceNormalization` applies to the physical fission-neutron
production functional derived from the selected fission data, and
`PowerNormalization` applies to the recoverable-power functional,
\(\int \kappa\Sigma_f\phi\,dV\). Neither includes a
\(1/k_\mathrm{eff}\) factor. That factor belongs only to the effective
fission source, \(F\phi/k_\mathrm{eff}\), in the eigenvalue equation and
the corresponding criticality balance diagnostic.
`PowerNormalization(power=...)` uses W and requires every fissionable active
material to provide `CrossSections.fission.kappa_sigma_f` in eV / cm. Missing
data reject the solve before finite-volume assembly, preventing a partial
power response.
Morana converts the input to joules using the exact SI eV-to-J relation; it
neither assumes a fission multiplicity nor converts separable `nu_sigma_f` to
a fission rate. Failure to converge raises `RuntimeError`, while invalid scope,
singular operators, missing power data, or failed numerical checks raise
`ValueError`.

The [multigroup criticality example](examples.md#multigroup-criticality)
provides groupwise `kappa_sigma_f` alongside the fission-neutron data and
normalizes the result to 100 kW. Use it as a complete example of specifying
materials, the physical target, and iteration controls together.

## Interpret a result

[`Result`](reference/core.md#morana.Result) owns a bottom-to-top tuple of
read-only group-major scalar-flux arrays in `n / cm^2 / s`. Each layer has
shape `(groups, active_cells_in_layer)` and is selected explicitly with
`flux_layer(axial_index)`. Solver and archive reconstruction checks match every
layer's compact active-cell count to the retained configuration snapshot.

That snapshot captures named layer assignments, excluded-region definitions,
boundary assignments, materials and cross sections, and the optional built-in
source. Its public object properties return fresh immutable values rather than
live references to retained numerical storage. Use `to_configuration()` when
an independent mutable problem definition is needed.

The immutable result also owns mode-specific settings, normalization, balance,
and execution diagnostics. `normalization` is absent for fixed-source results
and required for criticality results; `solve_mode` is derived from the typed
settings and report. A fixed-source result stores its single
`LinearSolveReport` directly. A `KeffSolveReport` stores one
`KeffOuterIterationReport` per power iteration, including the multiplication
factor, flux convergence, and nested linear report. Its `iterations` property
counts those records, `final_outer_iteration` accesses the last one, and
`eigenvalue_iteration` retains the selected ordinary or fixed-Wielandt policy.
Each linear report records its strategy and preconditioner, iteration count,
and independently calculated true relative residual.

For position-based inspection, use `result.cell_at(axial_index, openmc_index)`.
It resolves the layer-local solver ID and returns a read-only
[`CellInspection`](reference/core.md#morana.CellInspection) value containing
the assigned `material_key`, the material's `cross_sections`, and the read-only
all-group `flux` vector. Group ordering remains fast to thermal, so, for
example, `cell.cross_sections.sigma_a[group]` and `cell.flux[group]` describe
the same energy group. Excluded positions have no flux or cross sections and
raise `ValueError`; use `result.configuration_snapshot.material_mesh.key_at()`
when an excluded-region key itself is needed.

Use `Result.save_to_disk(...)` to retain a completed calculation in a versioned
non-pickle archive, and `Result.load_from_disk(...)` to restore it later. The
[result archives guide](result_archives.md) defines the portable archive
contents and validation behavior.

Use `Result.plot_matplotlib(group, axial_index)` or
`Result.plot_plotly(group, axial_index)` to inspect a flux slice, and
`Result.export_vtm(path)` to export flux with the reconstructed material layout.
Plot indices must be in range, and a layer without active cells cannot be
plotted. See [inspection and output](outputs.md#solver-results) for artifacts
and display conventions.

`Result.balance` is a `FixedSourceBalance` or `KeffBalance` record. Its
`by_group` mapping contains immutable group arrays in `n / s`, and its
`by_layer_group` mapping contains their bottom-to-top layer decomposition.
Contracting `by_layer_group` over its layers produces `by_group`.
`balance.scalar` derives the corresponding scalar volume-integrated terms, so
scalar and detailed values cannot disagree. `balance.loss_fractions` and
`balance.source_normalized` are likewise derived views; a zero-response
fixed-source result returns `None` for the affected view.

Criticality-result completion also independently recomputes its physical
normalization from the retained flux and configuration snapshot. A source-rate
result must reproduce the requested integrated fission-neutron production
within the result-validation tolerance. A power result must reproduce its converted `kappa_sigma_f * flux` response and
retain `kappa_sigma_f` for every fissionable active material. Consequently,
those checks apply to solver results and loaded archives.

The [theory guide's balance section](theory_references.md#balance-terms)
defines the balance terms and signs.

The group and layer-group balance mappings use these keys:

- Fixed source: `source`, `boundary_source`, `scattering_coupling`,
  `fission_production`, `fission_emission`, `removal`, `absorption`,
  `radial_leakage`, `axial_leakage`, `net_scattering`, and `residual`.
- Criticality: `fission_production`, `keff_source`, `scattering_coupling`,
  `removal`, `absorption`, `radial_leakage`, `axial_leakage`,
  `net_scattering`, and `residual`.

`scattering_coupling` is the signed scattering-neutron emission term derived
from ordinary scattering events and multiplicity; it is distinct from the
ordinary event-loss contribution retained in `removal`. Fixed-source
`fission_emission` and criticality `keff_source` are fission-neutron emission
terms, not scattering production.

## Advanced operator access

Most workflows should use the solve functions above. For numerical inspection,
verification, or integration with another solver, `morana.operators` exposes
the compact cross-section, loss, fission, volumetric-source, and boundary-source
assembly stages independently. See
[direct finite-volume operator assembly](operator_assembly.md) for their data
flow, packing, prerequisites, and ownership rules.

## Capability boundaries

- Boundary selection supports scalar affine Robin, partial-current-return, and
  incoming-current response over the documented selector hierarchy. Its scope
  excludes arbitrary per-cell-face maps, exterior radial-direction splitting,
  and group-coupled boundary response; see the
  [boundary guide](boundary_conditions.md#scope-and-related-guidance).
- OpenMC integration is limited to selected macroscopic material data from the
  supported runtime-MGXS format. Geometry, material placement, sources,
  boundaries, tally-postprocessing stores, and microscopic number densities
  remain part of the surrounding application.
- `solve_keff()` supports multigroup layered direct or GMRES inner solves with
  ordinary or fixed-Wielandt-shifted fission-source iteration. Start with the
  direct ordinary policy as a reference. Consider GMRES and its typed
  preconditioner for larger sparse systems, and use a fixed Wielandt shift only
  after a reference run identifies slow outer convergence and provides a safe
  inverse-`k_eff` estimate below the dominant value. It rejects independent
  sources, nonhomogeneous boundary data, missing fission normalization, meshes
  without active or fissionable cells, and unusable ordinary or shifted
  operators. `PowerNormalization` additionally requires recoverable
  `kappa_sigma_f` data for every fissionable active material.
