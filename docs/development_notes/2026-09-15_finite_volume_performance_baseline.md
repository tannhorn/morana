# Finite-volume performance baseline

*Recorded 2026-09-15.*

*Updated 2026-09-26: linked the completed production-policy study and updated
current runner terminology after the maintained `baseline` preset was renamed
`scaling`.*

This note records a controlled workstation study used to guide finite-volume
development. It is not a portable performance guarantee or a comparison of
hardware or solver libraries.

## Study

The study exercised the finite-volume criticality solver on deterministic,
synthetic many-group cross sections and a 61-cell hex-z mesh with 5, 10, or 20
axial layers. It covered 6, 18, 36, and 72 energy groups, using fresh worker
processes, fixed convergence controls, and independent residual and balance
checks. Measurements recorded wall-clock and process CPU time, peak resident
set size (RSS), and exclusive stage timings.

The direct reference, bounded direct-ordering comparison, existing GMRES
policies, and numerical-library thread limits were all assessed on the
same frozen workload family.

Measurements were collected using the `devel` branch at commit
`82535ddb1a0192525a4c9ad2ee73b6e1c00fe8b7` on a six-core, 12-thread x86-64
development laptop. The principal solver comparisons used a controlled
one-thread numerical-library environment.

## Major results

- Sparse direct factorization and its fill dominated the larger direct solves.
  When using 72 groups and 20 layers, the direct
  case took about 14 minutes and reached about 11 GiB peak RSS; factorization
  accounted for about 90 percent of its solve time.
- GMRES with Jacobi scaling was substantially faster and smaller on the large
  tested workloads. In the corresponding single endpoint observation, it took
  about half a minute and reached about 1 GiB peak RSS: approximately 27 times
  faster and 11 times smaller than the direct case. Direct solving remained
  faster for the two smallest six-group workloads, so this evidence does not
  support changing the default reference path.
- On the replicated 36-group, 10-layer matched-unknown workload, median
  GMRES/Jacobi execution was approximately 13 times faster than direct solving
  and used about one-fifth of its peak memory. Alternative direct orderings did
  not provide a consistent time improvement. Default ILU reduced Krylov
  iterations while taking roughly 2--4 times longer than Jacobi without a
  useful peak-RSS benefit.
- Increasing numerical-library thread limits increased process CPU use without
  a material, reliable wall-time benefit for the tested direct, GMRES/Jacobi,
  or GMRES/ILU paths.
- With GMRES/Jacobi in place of the sparse-direct inner solve, inner Krylov
  solves occupied roughly 45--57 percent of total solve time on the larger
  cases. Loss and fission assembly together occupied roughly 35--45 percent.

## Implications for further development

The next experiment should test a previous-iterate GMRES initial guess while
preserving the existing independent residual checks, then repeat the stage and
function profiling. If Morana-side assembly or result work is then dominant,
that work should take priority over stronger preconditioning. This study does
not justify threaded-BLAS tuning, a parallel direct backend, or a broader
solver-policy surface.

That experiment and the later bounded solver screen are concluded in the
[production-policy study](2026-09-26_finite_volume_production_policy.md).
This section records the earlier decision point rather than the current solver
recommendation.

## Retained study

The repository retains the deterministic workload builder, checked result
reader, and controlled one-thread smoke and scaling workflows under
`studies/finite_volume_performance/`. Its
[README](https://github.com/tannhorn/morana/blob/main/studies/finite_volume_performance/README.md)
owns the current commands and artifact-location guidance. Raw,
machine-specific study records remain local rather than becoming package data.
