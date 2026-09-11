# Verification and comparisons

Morana's verification evidence combines data-invariant tests, analytic
problems, independently assembled references, and file-reader checks. These
activities test whether the code implements its documented equations
consistently. They do not validate the model against experimental reactor
measurements; verification and experimental validation are distinct activities
in computational science ([Oberkampf and Trucano,
2007](#oberkampf-trucano-2007)).

!!! warning "Not validated for reactor decisions"
    Morana is intended for research, teaching, and software development. It
    has not been validated against experimental reactor measurements or
    qualified for design, operational, licensing, or regulatory decisions.
    Results must be independently reviewed and validated for their intended
    use. The example cross sections are illustrative and are not
    reactor-design data.

## Verification cases

The detailed derivations, acceptance criteria, and reproducible artifacts for
the maintained numerical cases are on separate pages:

| Case | Evidence | Documentation | Source |
| --- | --- | --- | --- |
| One-dimensional axial core-reflector | Analytic fundamental eigenvalue and source-normalized axial cell averages | [Case details](one_group_keff.md) | [`one_group_keff.py`](https://github.com/tannhorn/morana/blob/main/examples/one_group_keff.py) |
| Fixed-source MMS | Three-group, variable-height refinement with independently evaluated source and Dirichlet data | [Case details](fixed_source_mms.md) | [`fixed_source_mms.py`](https://github.com/tannhorn/morana/blob/main/examples/fixed_source_mms.py) |
| k-effective MMS | One-group, three-dimensional eigenpair refinement with manufactured fission and Robin data | [Case details](keff_mms.md) | [`keff_mms.py`](https://github.com/tannhorn/morana/blob/main/examples/keff_mms.py) |

The [theory and numerical conventions](theory_references.md) page defines the
equations and conventions these cases exercise.

## Comparative evidence

Comparison cases exercise a complete modeling workflow against an independent
calculation without claiming an exact solution, formal benchmark, or
experimental validation.

| Case | Comparison | Documentation | Source |
| --- | --- | --- | --- |
| OpenMC–Morana SRE-derived mini-core | Heterogeneous OpenMC continuous-energy reference against homogenized multigroup Morana diffusion, including energy and axial convergence | [Comparison details](openmc_comparison.md) | [`examples/openmc_comparison/`](https://github.com/tannhorn/morana/tree/main/examples/openmc_comparison) |

The OpenMC case reports agreement and remaining systematic differences. It
appears in a separate comparison table because neither code supplies an exact
reference solution for that model.

## Reproducing the checks

The [contributor workflow](contributor_workflow.md) gives the exact commands
for `pytest` and the documentation checks. The repository's
[`tests/`](https://github.com/tannhorn/morana/tree/main/tests) suite exercises
the Python implementation; the documentation checks build the public site,
verify links and generated reference pages, and check spelling and terminology.
Maintained examples provide numerical evidence; the browser-based smoke check
verifies that representative documentation pages and equations render offline.

## What automated checks cover

### Problem definition and geometry

- Material, source, boundary, and solver-setting inputs are checked for shape,
  type, finite values, and physical admissibility at construction or assembly,
  according to each input's contract. The public value objects
  retain owned read-only numerical data.
- Planar and material-mesh checks cover hexagonal geometry, OpenMC ordering,
  neighbors, variable-height axial stacks, excluded regions, and boundary-face
  classification.
- Configuration and boundary checks cover complete material coverage,
  immutable snapshots, group-resolved boundary data, and deterministic
  boundary-selection precedence.
- OpenMC runtime-MGXS reader tests use an OpenMC-written synthetic fixture and
  dynamic malformed files. They cover file identity, exact record and
  temperature selection, both diffusion conventions, compact scattering,
  optional multiplicity, nonfissionable, separable, and general-transfer
  fission data, recoverable energy, warnings, and rejected ambiguity or
  inconsistent metadata without making OpenMC a test dependency.

### Operators and solutions

- Hand-calculated operator references check removal, scattering, fission,
  radial and axial leakage, and source and boundary contributions. They cover
  ragged axial layouts as well as group-major data and packed solver ordering.
- Fixed-source tests compare analytic and direct linear-reference solutions,
  including downscatter, upscatter, subcritical multiplication, and the
  supported boundary conditions. They also reject singular, negative-flux, and
  over-multiplying same-group-scatter responses.
- Criticality tests compare analytic homogeneous cases and a heterogeneous
  dense-eigenvalue reference. They check normalization, convergence, residuals,
  balance closure, and supported acceleration behavior.
- Fission-data tests check separable and event-oriented transfer inputs,
  transfer orientation and production/emission conservation in assembly,
  equivalent fixed-source and criticality responses, and snapshot and tagged
  archive preservation of the selected representation.
- Scattering-multiplicity tests establish the unit-multiplicity compatibility
  limit, off-diagonal and same-group effects, archive provenance, and the
  assembled fixed-source and criticality identities.

### Results and output

- Result and archive checks cover immutable layered flux, balance diagnostics,
  solve provenance, normalization, and corrupted or incompatible archive data.
- Plot checks verify selected group and axial slices, labels, hover data, and
  excluded-cell handling. Python VTK readers reopen VTU and VTM output to
  confirm hexagonal-prism geometry, material layout, and flux fields.

## Limits of the evidence

The automated suite does not constitute experimental validation,
qualification for a particular application, or a performance and scalability
benchmark.

For exact test and documentation commands, continue with the
[contributor workflow](contributor_workflow.md). For numerical derivations and
reproducible artifacts, use the linked verification and comparison pages
above.

## References

<a id="oberkampf-trucano-2007"></a>
**Oberkampf and Trucano (2007).** W. L. Oberkampf and T. G. Trucano,
*Verification and Validation Benchmarks*, SAND2007-0853, Sandia National
Laboratories, 2007. [OSTI bibliographic record](https://www.osti.gov/biblio/901974)
and [open full text](https://www.osti.gov/servlets/purl/901974).
