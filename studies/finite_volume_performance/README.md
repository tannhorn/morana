# Finite-volume performance study

This maintained development study evaluates finite-volume solver behavior on
deterministic synthetic many-group workloads. It informs implementation and
solver-policy decisions; it is not a user example, a numerical-verification
claim, a recognized reference problem, or a portable performance guarantee.

Run the commands below from the repository root.

## Smoke and baseline runs

The routine smoke workflow is:

```bash
python -m studies.finite_volume_performance
```

The long-running production-policy baselines are explicit:

```bash
python -m studies.finite_volume_performance baseline
python -m studies.finite_volume_performance baseline --solver gmres_jacobi
python -m studies.finite_volume_performance baseline --solver bicgstab_jacobi
```

Every baseline workload uses the same unrecorded `(6 groups, 2 layers)`
warm-up before its recorded workers. Each worker starts in a fresh process;
numerical-library imports precede its measured solve interval.
Smoke, baseline, and complete-matrix presets share one execution protocol:
one numerical-library thread, a five-minute deadline, a 16 GiB address-space
limit, the `(6, 2)` warm-up, and one profile per case. Direct cases record one
timing observation; iterative cases record three.

## Selected workloads

For a bounded selection that does not warrant another durable runner mode, pass
the checked workloads, optional warm-up, resource guards, and output name
explicitly. For example:

```bash
python -m studies.finite_volume_performance selected \
  --solver gmres_jacobi --workload 18 6 --warmup 6 2 \
  --repetitions 1 --no-profile \
  --timeout-seconds 300 --address-space-limit-gib 16 \
  --output-name selected_gmres_jacobi.json
```

Add `--placement permuted` to use the single maintained seed-0 inventory
permutation. It preserves every material-role count while changing planar and
axial adjacency. Its SHA-256 ordering algorithm, seed, and placement digest are
recorded with the workload. The runner intentionally provides no arbitrary
seed interface, and routine smoke and baseline runs remain structured.

```bash
python -m studies.finite_volume_performance selected \
  --solver bicgstab_jacobi --workload 18 12 --placement permuted \
  --no-warmup --repetitions 1 --no-profile \
  --timeout-seconds 300 --address-space-limit-gib 16 \
  --output-name selected_permuted_bicgstab.json
```

Selected mode requires explicit warm-up, repetition, and profiling choices:
pass either `--warmup GROUPS LAYERS` or `--no-warmup`, a positive
`--repetitions` value, and either `--profile` or `--no-profile`. This prevents
an ad hoc assessment from silently inheriting the complete production-matrix
workload. Each recorded workload is preceded by the requested unrecorded
warm-up. The shared five-minute and 16 GiB resource guards remain defaults and
may be overridden explicitly. They apply independently to every worker; a
timeout or memory-limit failure is retained as study evidence. Repeat
`--workload` to record several selections with the same solver and guards.
Runs with at least three successful measurements print timing and memory
medians. Supported layer counts are 2, 6, 12, and 24; both selected workloads
and warm-ups use the frozen workload dimensions.

The frozen study controls permit 500 outer iterations for both placement
families. This assessment ceiling does not change Morana's package defaults.

The ordinary operator is the default. To record a fixed-Wielandt case, select
`--iteration wielandt` and provide its explicit nonnegative
`--shift-inverse-keff`. The same policy applies to every workload in that run,
and the runner writes the complete policy into every outcome. It never derives
or adapts a shift while measuring. For example:

```bash
python -m studies.finite_volume_performance selected \
  --solver gmres_jacobi --iteration wielandt --shift-inverse-keff 1.02 \
  --workload 36 6 --workload 18 12 --warmup 6 2 \
  --repetitions 3 --profile \
  --timeout-seconds 300 --address-space-limit-gib 16 \
  --output-name shifted_gmres_jacobi.json
```

Supplying a shift with ordinary power iteration, or selecting Wielandt
iteration without a shift, is rejected before any worker starts.

The optional warm-up is deliberately ordinary power iteration: it is
unrecorded and uses a separate fresh worker, so it cannot provide solver state
or alter the selected shifted measurement.

A failed measurement or profile is retained and stops the remaining workers for
that workload. A failed warm-up is retained once and blocks all workloads that
require the same warm-up. Other independent workloads continue. `--resume`
retains these terminal decisions and executes only missing work, including the
warm-up when only a profile remains. To retry a terminal failure, start a new
run with a separate output name.

## Complete production-path assessment

Run the frozen complete production-path matrix with:

```bash
python -m studies.finite_volume_performance.production_matrix
```

The launcher covers the three selected workload sizes, structured and
canonical seed-0 permuted placement, direct, Jacobi-GMRES, and
Jacobi-BiCGSTAB solves, and ordinary and fixed-Wielandt criticality. It owns
the six checked workload-specific shifts, uses one direct or three iterative
timing repetitions, records one function profile for every matrix entry, and
applies a five-minute and 16 GiB limit to each fresh worker. In total it plans
36 matrix entries, 120 recorded workers, and 36 unrecorded warm-ups.
Direct solves are retained as expensive reference observations, while three
iterative observations expose timing variability and support the study's
median, minimum, and maximum summaries.

Inspect the complete plan without starting workers:

```bash
python -m studies.finite_volume_performance.production_matrix --dry-run
```

If execution is interrupted, resume the complete campaign with:

```bash
python -m studies.finite_volume_performance.production_matrix --resume
```

A fresh run refuses to overwrite any existing matrix document. Resume checks
and completes existing documents and starts entries whose output documents do
not yet exist. Terminal worker failures remain evidence and are not retried;
use a separate output directory for a deliberately new campaign.

## Generated artifacts and workload scope

The workflow writes checked, machine-specific results below
`artifacts/studies/finite_volume_performance/`. These ignored records are local
development evidence rather than package data.

Each result document records shared convergence and acceptance controls once,
linear-solve cases independently, and the complete eigenvalue-iteration policy
on every outcome. Outcome identifiers and repeated-run summaries include that
iteration policy, so measurements from different operators or fixed shifts
cannot be conflated.

The heterogeneous role materials are deterministic synthetic formulas over an
abstract fast-to-thermal group coordinate. They are workload contrasts spanning
diffusion, loss, transfer, production, and fast-emission structure; they are not
evaluated or condensed cross sections.

Review every role material together in groupwise and transfer-matrix plots
with:

```bash
python -m studies.finite_volume_performance.plot_cross_sections
```

Use `--groups` to select supported group counts. The command writes PNG files
below `artifacts/studies/finite_volume_performance/cross_sections/` by default.
