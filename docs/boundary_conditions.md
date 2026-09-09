# Boundary conditions and face selection

Morana assigns boundary physics to exposed hex-z faces by combining an
immutable [`BoundaryCondition`](reference/core.md#morana.BoundaryCondition)
with a topology-based selector. This guide explains how to select faces,
compose a [`BoundaryConditionSet`](reference/core.md#morana.BoundaryConditionSet),
and check that a material layout has complete coverage. The
[theory guide](theory_references.md#exposed-boundary-conditions) derives the
finite-volume equations; the [Python reference](reference/core.md) defines
exact signatures and accepted argument types.

## Boundary model

Internal active-to-active faces use conservative two-cell diffusion coupling
and never receive a boundary assignment. Every other face of an active cell is
exposed: it lies on the physical radial, bottom, or top exterior, or it borders
an excluded material-mesh position. Every exposed face must resolve to a
condition before loss or boundary-source assembly.

Selectors describe topology only; they do not imply boundary physics. The
same selector can therefore be paired with any compatible condition.
Conversely, one condition can be assigned independently to several selectors.

## Select exposed faces

Use the fluent methods on a boundary condition for ordinary configurations:

| Fluent method | Selected exposed faces | Does not select |
| --- | --- | --- |
| `globally()` | Every physical exterior and excluded interface | Internal active-to-active faces |
| `on_outer()` | Every physical exterior face, radial and axial | Excluded interfaces |
| `on_radial()` | Physical lateral exterior faces | Bottom, top, and excluded interfaces |
| `on_bottom()` / `on_top()` | Physical exterior faces at that axial end | In-stack excluded interfaces in that direction |
| `on_excluded(...)` | Interfaces adjoining excluded positions, optionally filtered | Physical exterior faces |

The fluent methods construct assignments containing a
[`BoundarySelector`](reference/core.md#morana.BoundarySelector). Direct
selector construction is available when an assignment must be built
explicitly. Its `scope` uses the closed vocabulary `global`, `outer`,
`radial`, `to_excluded`, `bottom`, and `top`. Only `to_excluded` accepts
`direction`, `excluded_key`, or `excluded_kind` filters.

An excluded region has no active-cell unknown or material data.
`on_excluded(...)` selects the face of the adjacent active cell, not the
excluded position itself. Its optional filters are combined:

- `key` selects one exact excluded-region identity;
- `kind` selects every identity represented by that
  [`ExcludedRegion`](reference/core.md#morana.ExcludedRegion) kind; and
- `direction` selects one radial (`"x+"`, `"x-"`, `"u+"`, `"u-"`, `"v+"`,
  or `"v-"`) or axial (`"bottom"` or `"top"`) face direction.

Specify `key` or `kind`, but not both. Omit every filter to select all excluded
interfaces; omit an individual filter to accept any value for it. For example:

```python
from morana import BoundaryCondition

BoundaryCondition.dirichlet([1.0]).on_excluded(
    key="beam_port", direction="u+"
)
BoundaryCondition.reflective().on_excluded(kind="reflector")
```

## Define face physics

Use the named `BoundaryCondition` constructors rather than direct
construction. Spectrum arguments are nonempty, finite, nonnegative
one-dimensional vectors in fast-to-thermal order. Morana copies them into
read-only arrays and checks their group count during assembly.

| Constructor | Face condition | Input and limits |
| --- | --- | --- |
| `reflective()` | $J_{\mathrm{out}}=0$ | Homogeneous; equivalent to a return ratio of 1. |
| `vacuum()` | $J_{\mathrm{out}}=\phi_b/2$ | Homogeneous Marshak vacuum; equivalent to `robin(0.5)`. |
| `zero_dirichlet()` | $\phi_b=0$ | Homogeneous zero physical face flux; distinct from Marshak vacuum. |
| `dirichlet(flux)` | Prescribed $\phi_b$ | `flux` is in `n / cm^2 / s`; nonzero values are a boundary source. |
| `robin(alpha)` | $J_{\mathrm{out}}=\alpha\phi_b$ | Finite, nonnegative scalar $\alpha$; 0 is reflective and 0.5 is Marshak vacuum. |
| `partial_current_return(beta, current=None)` | $j^-=\beta j^+ + q_{\mathrm{in}}$ | $0\leq\beta\leq1$; optional `current` is a group vector in `n / cm^2 / s`. |
| `incoming_current(current)` | $J_{\mathrm{out}}=\phi_b/2-2q_{\mathrm{in}}$ | `current` is a group vector in `n / cm^2 / s`. |

Here $\phi_b$ is the physical face flux, $J_{\mathrm{out}}$ is net outward
current, and $j^+$ and $j^-$ are outgoing and incoming partial currents.
For `partial_current_return`, $\beta=1$ is reflective only when the imposed
current is absent or zero; otherwise it prescribes a net inward current.

Prescribed fluxes and currents are group-resolved. Robin response and
partial-current-return coefficients are scalar and energy independent;
group-coupled boundary response is not implemented.

`solve_keff()` accepts only homogeneous resolved boundary data. Reflective,
vacuum, zero-Dirichlet, and Robin conditions qualify directly, as do a zero
Dirichlet spectrum and an absent or zero imposed-current spectrum. Nonzero
prescribed flux or current contributes to a fixed-source right-hand side and
requires `solve_fixed_source()`.

## Compose and resolve assignments

Construct a boundary set from fluent assignments. A broad fallback followed
by more specific rules is easy to read, although construction order does not
control resolution:

```python
from morana import BoundaryCondition, BoundaryConditionSet

boundaries = BoundaryConditionSet(
    BoundaryCondition.reflective().globally(),
    BoundaryCondition.vacuum().on_radial(),
    BoundaryCondition.reflective().on_bottom(),
    BoundaryCondition.reflective().on_top(),
)
```

An exact duplicate selector is rejected; overlapping distinct selectors are
intentional refinements. `with_assignment(...)` returns a new set. When
editing a `ProblemConfiguration`, use `add_boundary(...)` to install that new
set on the configuration.

Resolution uses selector specificity, not tuple order. For an excluded
interface, precedence is:

1. key and direction;
2. key only;
3. kind and direction;
4. kind only;
5. direction only;
6. unfiltered excluded interface; and
7. global fallback.

For a physical exterior face, a radial or matching axial rule takes
precedence over `outer`, which takes precedence over `global`.

Call `ProblemConfiguration.check_boundary_coverage()` to validate the current
material layout; an uncovered face is reported with its topology. Loss
assembly and both solve functions require complete coverage.
`BoundaryConditionSet.resolve(face)` is useful when inspecting one exposed
[`DomainFace`](reference/core.md#morana.DomainFace); it rejects internal and
uncovered faces.

## Scope and related guidance

The selector hierarchy supports whole-domain fallbacks, physical exteriors,
and excluded interfaces filtered by identity, kind, and direction. It does
not provide arbitrary per-cell-face maps or separate physical radial rules by
direction. Distinct excluded keys and directional filters provide local rules
where the supported topology permits them.

See [geometry and indexing](geometry.md#faces-and-geometric-measures) for face
classification, [theory and numerical conventions](theory_references.md#exposed-boundary-conditions)
for boundary reductions and limiting cases, and the
[`mixed_boundary_regions.py`](examples.md#mixed-boundary-regions) example for
a solved problem using several selector and condition types.
