# Finite-volume solver production-policy study

*Recorded 2026-09-26.*

This note records the bounded many-group study used to refine criticality
continuation, screen Krylov methods and preconditioners, select the maintained
finite-volume solver policy, and identify the next optimization target. It is
machine-local development evidence, not a portable performance guarantee, a
numerical-verification claim, or an application-valid reactor calculation.

## Study design

The study asked which sparse linear-solve policies deserved ownership in
Morana's maintained fixed-source and criticality entry points. A candidate had
to demonstrate a supported performance, memory, or robustness role against a
simpler policy; convergence in isolation was not enough. Direct solving
remained the checked reference and default. Experimental algorithms stayed in
an ignored external harness until the evidence justified implementing an
actual production path.

### Synthetic workloads

Five deterministic formula-only many-group cross section sets were
used in the reference problem: `reference`, `high_leakage`,
`high_reactivity`, `fuel_removal`, and `localized_absorber`.
They exercise diffusion, absorption, transfer, fission
production, upscatter, and fast-emission contrasts. They are synthetic solver stresses, not
reconstructed MGXS materials or evaluated reactor data.

Each problem uses the same 61-cell planar hexagonal mesh. The bounded matrix
contains `(36 groups, 6 layers)` and `(18 groups, 12 layers)`, each with 13,176
packed unknowns, and `(72 groups, 24 layers)` with 105,408 unknowns. Every size
has a structured placement and one canonical seed-0 SHA-256 permutation. The
permutation preserves the complete material-role inventory while changing
planar and axial adjacency; its algorithm, seed, and placement digest are
frozen provenance.

Every placement was exercised with ordinary power iteration and a fixed
Wielandt operator. Each fixed inverse-$k$ shift was set once to
$0.995/k_\mathrm{ref}$ from a checked ordinary reference and was never adapted
during measurement. The shifted matrix is $A-\sigma F$, so shifted convergence
was assessed independently rather than inferred from ordinary behavior.

### Screening protocol

Every worker used one numerical-library thread, a five-minute deadline, and a
16 GiB address-space limit. The private screen allowed at most 500 outer
iterations and 1,000 candidate-owned inner iterations. It recorded setup and
solve time, process wall and CPU time, peak RSS, matrix products,
preconditioner applications, outer progress, backend failures, and original-
equation residuals. Successful results also had to pass Morana's independent
eigen-residual, balance, finiteness, and nonnegativity checks. Unlike backend
callback counts were not treated as comparable Krylov iterations.

The work proceeded as a funnel:

1. Establish previous-solution continuation for ordinary GMRES and keep its
   shifted behavior as separate operator evidence.
2. Screen restarted GMRES, LGMRES, CGS, BiCGSTAB, TFQMR, and GCROT(m,k), each
   with no preconditioner and point Jacobi, over all 12 combinations of size,
   placement, and operator.
3. On representative structured medium and large cases, assess cold starts,
   native LGMRES and GCROT(m,k) recycling, two threshold-ILU strengths, and a
   node-block-Jacobi prototype.
4. Repeat only the selected finalist across the full 12-case matrix, implement
   the justified production policy, and then measure the actual maintained
   direct, GMRES, and BiCGSTAB paths.

The broad candidate adapters and their raw JSON records were intentionally
untracked screening evidence, so this note owns their retained conclusions.
The deterministic workload, checked result format, worker controls, and final
production matrix are maintained under `studies/finite_volume_performance/`.
The final campaign can be inspected without launching workers with:

```bash
python -m studies.finite_volume_performance.production_matrix --dry-run
```

Machine-specific result documents remain ignored under
`artifacts/studies/finite_volume_performance/`; they are not package data.

## Continuation result

Passing the preceding cleaned inner solution to the next ordinary GMRES solve
reduced total Krylov callbacks from 4,089 to 1,993 for `(36, 6)`, from 3,782 to
1,879 for `(18, 12)`, and from 5,362 to 2,627 for `(72, 24)`. The checked
$k_\mathrm{eff}$ values were unchanged and repeated wall time fell in every
case. Continuation therefore became the sole production criticality behavior;
it introduced no public setting and its state remains local to one
`solve_keff()` call.

The fixed-Wielandt comparison did not support a general continuation claim.
Both cold and continued Jacobi-GMRES failed on the large shifted case,
continued GMRES failed on every `(36, 6)` repetition, and its successful
`(18, 12)` runs increased rather than reduced Krylov work. Those outcomes
motivated the broader shifted-operator screen rather than reversal of the
ordinary continuation decision.

## Common candidate screen

The one-repetition common baseline completed 144 fresh-worker cases. Ninety-
three succeeded and 51 retained terminal candidate failures; no worker reached
the time or memory limit. The success counts were:

| Candidate | No preconditioner | Point Jacobi | Total | Disposition |
| --- | ---: | ---: | ---: | --- |
| Restarted GMRES | 7/12 | 7/12 | 14/24 | Retain narrowly for ordinary solver diversity |
| LGMRES | 12/12 | 12/12 | 24/24 | Focused recycling assessment |
| CGS | 1/12 | 3/12 | 4/24 | Reject |
| BiCGSTAB | 9/12 | 12/12 | 21/24 | Advance Jacobi form as finalist |
| TFQMR | 6/12 | 0/12 | 6/24 | Reject |
| GCROT(m,k) | 12/12 | 12/12 | 24/24 | Focused recycling assessment |

Jacobi-BiCGSTAB passed every ordinary and shifted case. Unpreconditioned
BiCGSTAB supplied the fastest single observation in four conditions but broke
down in three shifted conditions; Jacobi-BiCGSTAB supplied the other eight
timing leaders. This established point Jacobi as the recommended policy while
leaving no preconditioning as an explicit fallback for matrices whose diagonal
makes Jacobi invalid.

Restarted GMRES passed every ordinary case but only one of six shifted cases
for each preconditioner choice. CGS had 20 backend nonconvergence outcomes,
including one breakdown; a separately labelled 10,000-iteration smoke
diagnostic still did not establish a viable route. TFQMR's six successes were
all unpreconditioned ordinary cases. Sixteen other TFQMR outcomes were rejected
by the independent true-residual check despite backend completion, and two
reported backend failure. These reliability results rejected CGS and TFQMR
without repeated measurement.

LGMRES and GCROT(m,k) were numerically robust but did not beat the simpler
BiCGSTAB policies in the common timing catalogue. They advanced only to test
the concrete hypothesis that native cross-inner-solve recycling might create a
distinct role.

## Focused candidate and preconditioner screen

The focused Stage 2 matrix contained 52 structured workers: 16 recycling
comparisons, 12 cold-start comparisons, 16 threshold-ILU cases, and eight
node-block-Jacobi cases. All completed successfully and passed the independent
checks.

### Initial guesses and native recycling

Cold starts increased candidate-owned matrix products by 1.28–2.30 times and
end-to-end elapsed time by 1.01–1.80 times across the tested BiCGSTAB, LGMRES,
and GCROT(m,k) cases. They supplied no robustness benefit, so
previous-solution continuation was retained for every credible criticality
candidate.

Native LGMRES recycling was slower in all eight comparisons: elapsed-time
ratios against its non-native form ranged from 1.04 to 1.25, with a median of
1.10, and matrix-product changes were inconsistent. Native GCROT(m,k) ranged
from 0.92 to 1.23 times its non-native elapsed time, with a median of 0.98;
matrix products ranged from 0.93 to 1.13 times baseline. Its small wins did not
repeat across sizes, operators, or preconditioners. Neither method justified
production recycling state or another maintained Krylov implementation.

### Threshold ILU

The weak threshold-ILU choice used `drop_tol=1e-2` and `fill_factor=2`; the
strong choice used `drop_tol=1e-4` and `fill_factor=10`. Both materially reduced
matrix products and made the tested shifted GMRES cases converge, but setup
cost dominated the saved iterations.

Against the same successful point-Jacobi candidate, weak ILU took 1.20–1.95
times as long end to end and strong ILU took 2.68–3.71 times as long. Against
the applicable Jacobi-BiCGSTAB alternative, weak ILU cases took 1.20–2.04
times as long and strong cases took 2.98–4.00 times as long. Strong-ILU setup
alone reached 74–88 seconds on the large workload. ILU therefore supplied no
performance or robustness role that the simpler Jacobi-BiCGSTAB path lacked,
so the former production GMRES-ILU policy was removed.

### Node-block Jacobi

The prototype exactly factorized each node-local multigroup block, capturing
local removal, scattering, and shifted-fission coupling. Setup was inexpensive
at 0.03–0.18 seconds, and matrix products fell to 19–60% of point-Jacobi
BiCGSTAB and 38–62% of point-Jacobi GMRES on their shared successes. The
current per-node application path nevertheless made BiCGSTAB 1.20–1.94 times
slower and GMRES 1.66–1.80 times slower end to end than their point-Jacobi
forms. It also rescued shifted GMRES, but took 1.24–2.17 times as long as the
already robust point-Jacobi BiCGSTAB alternative.

This rejects the current Python prototype, not the mathematical idea. A
compiled or batched block action would perform materially more arithmetic than
point Jacobi and requires fresh measured justification before becoming another
optimization project.

## Finalist confirmation

Continued Jacobi-BiCGSTAB was the sole finalist. Three fresh-worker repetitions
of all 12 size, placement, and operator combinations succeeded. Within each
case, outer iterations, matrix products, $k_\mathrm{eff}$, and final residuals
were identical across repetitions. Timing coefficients of variation were
0.2–6.9% in the private confirmation, with ten of 12 below 3%. This was enough
to promote BiCGSTAB into the tracked Morana code base.

## Final production assessment

The maintained production matrix exercised three synthetic workload sizes,
structured and canonical seed-0 permuted material placement, ordinary power
iteration and fixed-Wielandt iteration, and all three retained solver paths:
sparse direct, Jacobi-GMRES, and Jacobi-BiCGSTAB. The fixed shifts were checked
before measurement and held constant. Every worker used one numerical-library
thread, a five-minute deadline, and a 16 GiB address-space limit. Direct cases
had one timing observation; iterative cases had three. Every successful case
also had a separate function-profile worker.

The campaign produced 36 complete entry documents. There were 65 successful
timing observations, nine terminal failed observations, and 27 successful
profiles. A terminal failure deliberately stopped later repetitions and the
profile for that entry. All successful outcomes passed the independent linear,
eigen-residual, balance, finiteness, and nonnegativity checks.

## Results

The table gives median end-to-end solve time in seconds. `timeout` is the
checked five-minute worker limit; `Krylov failure` is retained backend
nonconvergence rather than missing evidence.

| Placement | Groups × layers | Iteration | Direct | Jacobi-GMRES | Jacobi-BiCGSTAB |
| --- | ---: | --- | ---: | ---: | ---: |
| Structured | 36 × 6 | Power | 25.752 | 2.630 | 2.047 |
| Structured | 36 × 6 | Wielandt | 17.335 | Krylov failure | 1.790 |
| Structured | 18 × 12 | Power | 12.088 | 2.157 | 1.457 |
| Structured | 18 × 12 | Wielandt | 6.718 | 3.397 | 1.174 |
| Structured | 72 × 24 | Power | timeout | 38.995 | 30.827 |
| Structured | 72 × 24 | Wielandt | timeout | Krylov failure | 25.197 |
| Permuted | 36 × 6 | Power | 26.836 | 4.159 | 2.774 |
| Permuted | 36 × 6 | Wielandt | 15.493 | Krylov failure | 2.014 |
| Permuted | 18 × 12 | Power | 12.265 | 3.189 | 2.022 |
| Permuted | 18 × 12 | Wielandt | 7.721 | Krylov failure | 1.374 |
| Permuted | 72 × 24 | Power | timeout | 42.377 | 36.010 |
| Permuted | 72 × 24 | Wielandt | timeout | Krylov failure | 27.478 |

Across the eight cases where direct and BiCGSTAB both succeeded, BiCGSTAB was
5.62–12.58 times faster, with a median factor of 8.00, and direct solving used
2.58–3.36 times as much peak RSS, with a median factor of 2.98. Across the
seven shared GMRES successes, BiCGSTAB was 1.18–2.89 times faster, with a
median factor of 1.48. The two iterative paths had effectively the same peak
RSS because the common workload matrices dominated their resident memory.

BiCGSTAB succeeded in all 12 workload, placement, and operator combinations.
GMRES succeeded in all six ordinary cases and only the structured 18-group,
12-layer shifted case. Direct solving succeeded for the two smaller workload
sizes and reached the fixed deadline in all four 72-group, 24-layer cases.
These outcomes confirm the selected roles: sparse direct remains the small-case
reference, Jacobi-BiCGSTAB is the recommended scalable ordinary and shifted
path, and Jacobi-GMRES is a secondary ordinary-criticality route.

The largest within-case successful multiplication-factor spread was
$1.23\times10^{-11}$. The largest final eigen relative residual was
$6.23\times10^{-11}$, and the largest scalar-balance relative closure was
$1.15\times10^{-12}$. Iterative timing coefficients of variation were
1.2–3.4% for successful GMRES cases and 0.2–8.3% for BiCGSTAB cases. The 8.3%
endpoint occurred in the short structured 36-group, 6-layer shifted case;
absolute observations there ranged only from 1.70 to 2.06 seconds.

## Profiling conclusion

Sparse-direct factorization remains the direct path's dominant cost. Summing
the per-case median stage times across its eight successes, factorization
accounted for 75.7% of end-to-end solve time and repeated triangular solves for
17.4%.

For the recommended BiCGSTAB path across all 12 cases, inner linear solves
accounted for 47.1% of aggregate end-to-end time. Loss and fission assembly
together accounted for 43.3%, split into 27.2% and 16.2%; Jacobi setup was
negligible. The function profiles corroborate those stage measurements. GMRES
showed the same broad division on its seven successes, with 59.8% in inner
solves and 31.3% in operator assembly.

The next optimization target should therefore be the common finite-volume
loss and fission assembly path. It is nearly co-dominant with the already
screened BiCGSTAB solve work and benefits every solver policy. A stronger
preconditioner or spatial multilevel dependency is not justified next: the
bounded candidates either lost end to end or added complexity without a
distinct production role. Re-profile after any material assembly improvement
before reconsidering the linear-solve path.

## Interpretation

The mechanisms below are interpretations consistent with the measured stage,
operation-count, and failure evidence. The study did not isolate each
low-level cause experimentally, so they should not be treated as portable
performance laws.

The earlier thread-limit screen is consistent with the structure of these
sparse calculations. Assembly, sparse matrix-vector products, sparse
triangular solves, and sparse factorization have irregular memory access and
expose much less threaded dense-linear-algebra work than a large matrix-matrix
calculation. Additional numerical-library threads can therefore add scheduling
and memory-bandwidth pressure without shortening the limiting operation. This
explains the observed increase in CPU use without a reliable wall-time gain;
it does not establish that every sparse backend or future machine should run
with one thread.

Sparse direct memory growth is governed by factor fill rather than by the
stored size of the original loss matrix. Elimination connects previously
uncoupled unknowns, so the triangular factors can contain far more values and
indices than the assembled operator and also require factorization workspace.
Energy-group transfer and spatial coupling make that growth superlinear. The
two medium workloads have the same number of unknowns but different group and
layer counts, and their different direct costs show why total dimension alone
is not an adequate predictor: denser node-local group coupling changes the
elimination graph.

Previous-solution continuation is useful because successive criticality outer
iterations solve the same prepared operator with progressively related fission
sources. The preceding cleaned inner solution is therefore a cheap
approximation to the next solution and reduces the initial error presented to
GMRES. Native LGMRES and GCROT(m,k) recycling asks a stronger question: whether
a small retained subspace remains useful as the source evolves. In this study,
orthogonalization and recycling-state overhead were not repaid by consistent
reductions in matrix products. A useful initial vector did not imply that a
more elaborate recycled subspace would be useful.

Fixed Wielandt iteration illustrates the difference between outer and inner
difficulty. Moving the shift toward the dominant eigenvalue sharply reduces
the number of outer iterations, but the inner matrix $A-\sigma F$ approaches a
more difficult spectral regime. A better initial guess cannot repair poor
conditioning, nonnormal behavior, or restarted-GMRES stagnation. BiCGSTAB's
short recurrence avoided GMRES's restarted orthogonalization pattern and was
more robust on the bounded shifted matrix, but the study does not establish a
universal advantage for BiCGSTAB on other operators.

Point Jacobi succeeded as a policy because it corrects basic diagonal scaling
with one inexpensive setup and application. Threshold ILU and node-block
Jacobi did not fail numerically; they failed the end-to-end policy criterion.
ILU exchanged fewer matrix products for sparse factorization, fill, and
triangular applications. Node-block Jacobi captured useful group-local
coupling, but the prototype performed many small per-node solves in Python and
did more arithmetic than point scaling. Its reduced product counts therefore
did not translate into reduced elapsed time. A compiled or batched block
application remains a separate hypothesis rather than a conclusion of this
study.

The unpreconditioned results also explain why no preconditioning remains an
explicit fallback rather than the recommendation. It occasionally avoided
Jacobi overhead and produced the fastest isolated observation, but it was less
robust on shifted operators. Conversely, subtracting shifted fission can make
the diagonal unusable even when the full matrix remains nonsingular, so an
automatic assumption that Jacobi always exists would be incorrect.

Finally, the structured and permuted problems separate material inventory
from adjacency. Their different work counts show that placement and coupling
topology matter even when size and global material counts are unchanged. Once
the iterative linear solve was reduced to roughly half of total time, the
common assembly path became comparably important. That is the expected
Amdahl-law effect behind the profiling conclusion: further solver-only gains
cannot improve the portion already spent constructing the operators.
