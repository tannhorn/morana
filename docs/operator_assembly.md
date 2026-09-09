# Direct finite-volume operator assembly

The public functions in `morana.operators` expose Morana's finite-volume
assembly stages independently of the solvers. They are intended for numerical
inspection, verification, and advanced integrations. For ordinary
calculations, prefer `solve_fixed_source()` or `solve_keff()`: those functions
also enforce solve-mode prerequisites, convergence, physical normalization,
and result construction.

The [coupled-operator reference](reference/operators.md) is canonical for
exact signatures and returned types. The
[finite-volume theory](theory_references.md#finite-volume-discretization)
defines the equations and signs assembled here.

## Capture one stable problem definition

Each helper accepts a mutable `ProblemConfiguration` or an immutable
`ProblemConfigurationSnapshot`. A mutable configuration is captured
independently at every call. When several operators must describe the same
problem state, capture one snapshot and reuse it:

```python
from morana.operators import (
    assemble_boundary_rhs,
    assemble_fission_matrix,
    assemble_loss_matrix,
    assemble_source_rhs,
    extract_cross_section_data,
)

snapshot = configuration.snapshot()
cross_sections = extract_cross_section_data(snapshot)

loss = assemble_loss_matrix(snapshot, cross_sections)
fission = assemble_fission_matrix(snapshot, cross_sections)
source = assemble_source_rhs(snapshot, cross_sections)
boundary_source = assemble_boundary_rhs(snapshot, cross_sections)
```

`extract_cross_section_data()` creates immutable, layout-matching compact data
in bottom-to-top axial order. Its layer arrays are group-major and use each
slice's local `active_id` order. The extraction resolves compact unit
scattering multiplicity to explicit ones and nonfissile material data to zero
fission-transfer blocks.

## Assembly products

| Function | Product | Independent prerequisites |
| --- | --- | --- |
| `assemble_loss_matrix()` | Diffusion, conventional removal, signed scattering coupling, and boundary-response conductance | Complete boundary coverage; no volumetric source required |
| `assemble_fission_matrix()` | Fission-neutron emission | No source or boundary coverage required |
| `assemble_source_rhs()` | Volume-integrated volumetric source | No boundary coverage required; returns zero when no source is configured |
| `assemble_boundary_rhs()` | Face-integrated prescribed-flux and incoming-current terms | Complete boundary coverage; no volumetric source required |

Matrices are fresh mutable SciPy CSR matrices; right-hand sides are fresh
mutable NumPy vectors. Global products use node-major, group-fastest packing,
while the extracted layer arrays remain group-major. For active node $n$ and
group $g$, the global index is $I(n,g)=nG+g$.

With `loss` denoted by $A$, `fission` by $F$, and
`rhs = source + boundary_source`, the assembled fixed-source system is

$$
(A-F)\boldsymbol\phi=\mathbf{rhs},
$$

while criticality uses

$$
A\boldsymbol\phi=\frac{1}{k_{\mathrm{eff}}}F\boldsymbol\phi.
$$

These expressions explain how the products relate; the public solve functions
remain the supported path for producing a checked `Result`.

## Validation and ownership

Every assembler checks that the compact data match the selected
configuration's layer count and active-ID-to-material mappings. The compact
data define the group count and cross-section values; assemblers do not compare
them with the configuration's material cross sections. Reuse data extracted
from the same snapshot unless you deliberately supply replacement compact data.
Loss and
boundary-source assembly additionally resolve every exposed face and check
group-compatible boundary data. Fission assembly checks nonnegative emission
and verifies that matrix column sums reproduce the volume-integrated
incident-group production functional.

Extracted numerical arrays are read-only snapshots. Assemblers return new
mutable matrices or vectors, so modifying a returned product does not alter
the configuration or compact input data. Construct new compact data when an
advanced workflow needs different cross sections; do not mutate extracted
values in place.

Direct use of these products does not create execution reports, apply
criticality normalization, or reproduce the solver's independent residual and
flux-admissibility checks. Consult the
[modeling and solver workflow](modeling_workflow.md#configure-and-run-the-solver)
for solver behavior and the [Python reference](reference/operators.md) for all
validation errors and data-container attributes.
