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

The long-running direct-reference and GMRES/Jacobi baselines are explicit:

```bash
python -m studies.finite_volume_performance baseline
python -m studies.finite_volume_performance baseline --solver gmres_jacobi
```

Every baseline cell starts in a fresh worker after the same unrecorded `(6
groups, 2 layers)` warm-up. This keeps numerical-library initialization out of
the recorded runs.

## Selected workloads

For a bounded selection that does not warrant another durable runner mode, pass
the checked workloads, optional warm-up, resource guards, and output name
explicitly. For example:

```bash
python -m studies.finite_volume_performance selected \
  --solver gmres_jacobi --workload 18 6 --warmup 6 2 \
  --timeout-seconds 300 --address-space-limit-gib 16 \
  --output-name selected_gmres_jacobi.json
```

Each recorded workload is preceded by the requested unrecorded warm-up. The
resource guards apply independently to every worker; a timeout or memory-limit
failure is retained as study evidence. Repeat `--workload` to record several
selections with the same solver and guards. Selected runs record one observation
per workload; use `baseline` for the maintained repeated and profiled matrix.

## Generated artifacts and workload scope

The workflow writes checked, machine-specific results below
`artifacts/studies/finite_volume_performance/`. These ignored records are local
development evidence rather than package data.

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
