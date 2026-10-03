# Finite-volume acceleration assessment

*Recorded 2026-10-03.*

This note records the bounded study used to assess spatial
multilevel preconditioning, MPI linear solves, and an external criticality
eigensolver after the finite-volume assembly improvements. It is machine-local
development evidence, not a portable performance guarantee, a supported backend
contract, or an application-valid reactor calculation.

## Study design

The study asked whether PyAMG spatial preconditioning, PETSc linear solves
with power iteration, or SLEPc eigensolving with PETSc inner solves could
reduce complete criticality solve time enough to justify further integration
work. PETSc and SLEPc candidates used Open MPI for parallel execution. Candidates
had to pass independent checks of the original equation, normalization, and
flux admissibility. Ordinary and fixed-Wielandt operators were assessed
separately; ordinary acceleration did not establish shifted reliability.
The preceding
[production-policy study](2026-09-26_finite_volume_production_policy.md)
provided the profiling motivation, while fresh SciPy measurements supplied
the comparison baselines.

### Synthetic workloads

Problems used the maintained study's five deterministic synthetic material
roles, a fixed 27.94 cm pitch, and a fixed 182.88 cm total height. Radial growth
enlarged the domain; axial refinement preserved height. Structured and
inventory-preserving seed-0 permuted placement were assessed with ordinary
and fixed-Wielandt operators, using geometry-specific fixed shifts. These
formula-only many-group problems are synthetic solver stresses, not evaluated
reactor materials or a physically representative benchmark suite.

The 36-group screens covered five-, seven-, and nine-ring radial cases
and a six-, twelve-, and 24-layer axial ladder. The large structured ordinary
workload used 20 rings, 24 layers, and 72 groups, with 1,971,648 packed unknowns. Its loss
matrix stored 65,898,984 entries and its fission matrix stored 43,376,256.

### Measurement protocol

Measurements were collected using the `devel` branch at commit
`db1a09af93250dad531d4174b2f564d8ad1b05ef` on an Intel i7-10750H development
laptop with six physical cores, twelve hardware threads, approximately
32 GiB RAM, and one NUMA node. MPI rank counts were 1, 2, 4, and 6, with one
numerical-library thread per rank. Large workers bound ranks to physical cores
and used fresh 16 GiB aggregate memory scopes with swap disabled.

Complete worker time included construction, serial assembly, conversion and
distribution, setup, iterations, collection, and independent numerical checks.
Launcher elapsed time additionally included MPI and interpreter startup and
worker finalization, but excluded outer campaign startup between workers.
Memory measurements were aggregate resource-scope peaks rather than per-rank
RSS. These definitions apply to the external comparisons below.

## Small candidate screens

The small screens assessed two complementary questions on the 36-group
workload family: whether external linear solvers or spatial preconditioning
improved on SciPy, and whether SLEPc eigensolving improved on PETSc power
iteration. Both comparisons covered ordinary and fixed-Wielandt operators.

The linear-solver comparison contained 76 cases using SciPy Jacobi-BiCGSTAB,
PyAMG spatial preconditioning, and PETSc BiCGStab or GMRES with point Jacobi
or group GAMG, including selected two-rank PETSc runs. The eigensolver
comparison used 14 cases for each of SLEPc and PETSc power iteration across
ten geometries and four selected two-rank cases, with ten one-rank SciPy
baselines.

### Reliability

All 34 ordinary cases in the linear-solver comparison passed. Of its 42
fixed-Wielandt cases, 15 passed and 27 failed. SciPy Jacobi-BiCGSTAB passed
all ten one-rank baseline cases, while PyAMG passed eight. PETSc GMRES
reached the 1,000-inner-iteration limit in every shifted case at both rank
counts; other shifted failures included BiCGStab breakdown. No case timed
out or recorded an out-of-memory kill.

In the eigensolver comparison, SLEPc and PETSc power iteration each passed
the six ordinary cases and failed the eight shifted cases at the PETSc inner
iteration limit. SciPy passed all ten baseline cases. Neither external
approach established a shifted reliability advantage over SciPy.

### Complete solve time

Successful one-rank PyAMG cases took about 5.2–12.5 times the matched SciPy
launcher elapsed time, and the successful one-rank PETSc linear-solver cases
were also slower than SciPy. Spatial AMG supplied no small-case acceleration
role, and PyAMG was not pursued on the large workload.

SLEPc improved complete worker time over PETSc power iteration on some
ordinary cases, but its launcher elapsed time exceeded SciPy on all five
ordinary one-rank cases. MPI and interpreter startup therefore mattered at
these sizes. The small screens supplied no complete-time advantage over
SciPy for the tested one-rank external candidates; the large workload assessed
whether that changed as solve work grew.

## Large ordinary assessment

The campaign used a two-hour deadline, 1,000 permitted
outer iterations, relaxed $10^{-7}$ on the three outer criteria and final
original-equation/group-balance checks, and a $10^{-10}$ independent inner
residual requirement. GMRES restart was 50 with at most 1,000 inner
iterations; PETSc KSP relative stopping tolerance was $10^{-12}$.
The SLEPc large workers used EPS target $10^{-8}$, the same final equation
and group-balance ceilings, and a multiplication-factor check against the
frozen SciPy reference. Every successful large observation stayed within
the aggregate memory limit.

### Results

Only a very small number of repetitions was used, so these measurements
indicate the scale of the benefit rather than statistical confidence or an
optimal rank count. Timings below are rounded to the nearest minute, using
medians where available. The successful large runs passed their recorded
checks without timeout or out-of-memory failure.

| Method | Ranks | Complete solve time, min | Aggregate peak, GiB |
| --- | ---: | ---: | ---: |
| SciPy GMRES/Jacobi | 1 | 56 | 9.3 |
| PETSc BiCGStab/Jacobi | 1 | 36 | 10.2 |
| PETSc BiCGStab/Jacobi | 2 | 27 | 10.1 |
| PETSc BiCGStab/Jacobi | 4 | 27 | 10.2 |
| PETSc BiCGStab/Jacobi | 6 | 28 | 9.8 |
| PETSc GMRES/Jacobi | 1 | 42 | 10.1 |
| PETSc GMRES/Jacobi | 2 | 34 | 10.2 |
| PETSc GMRES/Jacobi | 4 | 34 | 10.2 |
| PETSc GMRES/Jacobi | 6 | 36 | 9.8 |
| SLEPc + PETSc GMRES/Jacobi | 1 | 16 | 9.5 |
| SLEPc + PETSc GMRES/Jacobi | 2 | 12 | 9.5 |
| SLEPc + PETSc GMRES/Jacobi | 4 | 11 | 9.6 |
| SLEPc + PETSc GMRES/Jacobi | 6 | 12 | 9.8 |

The six-rank group-GAMG cases took about 60 minutes with BiCGStab and
51 minutes with GMRES, with aggregate peaks of about 10 GiB. They supplied
no compelling complete-time advantage over point Jacobi, so no further AMG
search was undertaken.

### Interpretation

The large-case results show benefits from both distributed inner solving and
changing the criticality algorithm. Against the roughly 56-minute SciPy GMRES
baseline, PETSc-backed power iteration took about 27 minutes with
BiCGStab/Jacobi and 34 minutes with GMRES/Jacobi at two or four ranks,
reductions of roughly 50% and 40%. Using SLEPc eigensolving with PETSc
GMRES/Jacobi inner solves brought complete time to about 11–12 minutes at
those rank counts. Aggregate memory remained comparable across these
approaches. The results suggest that the eigensolver improves on power
iteration for this large ordinary case, making it a promising development
direction.

Across the external approaches, most parallel benefit came from moving from
one rank to two; four ranks gave similar or slightly better timings, while
six ranks supplied no further gain. More ranks also consume more allocated
rank-seconds, so lower elapsed time does not imply proportionate improvement
in CPU efficiency. The study does not establish a preferred rank count.

The modest slowdown from four to six ranks is consistent with shared memory
bandwidth or MPI overhead limiting scaling, possibly compounded by lower
sustained clock speeds when all six physical cores are active. Intel documents
that turbo frequency depends on active-core count, power, and temperature
([Turbo Boost overview](https://www.intel.com/content/www/us/en/support/articles/000007359/processors/intel-core-processors.html)).
The study did not measure which mechanism dominated.

Note that the small screens and large case used
different accuracy requirements, so their timings do not establish a precise
size crossover.

## Implications for further development

The measurements support retaining SLEPc and pure PETSc as candidates for
future acceleration work. They do not justify changing Morana's
maintained SciPy policies or adding supported PETSc/SLEPc integration. Small-case
startup costs and shifted failures remain practical limits to investigate.

Any future integration requires a concrete user need, installation and
compatibility ownership, numerical reliability over its explicit operator
scope, and fresh matched complete-solve evidence. Distributed assembly and
stronger preconditioning require their own measured justification; ordinary
speedups and iteration reductions alone do not establish those priorities.
