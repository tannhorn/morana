# Changelog

This page records dated releases that help a package user decide whether or how
to install, upgrade, or use Morana. Entries follow the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) categories
where they apply: Added, Changed, Deprecated, Removed, Fixed, and Security.
Routine documentation corrections, citation-metadata updates, repository
maintenance, and maintainer-only tooling do not receive separate entries.
Release entries summarize notable changes, required upgrade actions, and
release-specific limitations. Current package behavior is documented in the
[modeling and solver workflow](modeling_workflow.md),
[verification guide](verification.md), and source-derived
[Python reference](reference/index.md). Release-note candidates remain in
committed change fragments during development and enter this page only as part
of a dated release. Subscribe to the
[Morana releases RSS feed](https://pypi.org/rss/project/morana/releases.xml)
to receive new-version notifications when installable packages are published.

## 0.2.0 - 2026-10-03

Version `0.2.0` adds BiCGSTAB solving and revises the iterative solver policies
and maintained examples. Release locations are
[GitHub](https://github.com/tannhorn/morana/releases/tag/v0.2.0),
[PyPI](https://pypi.org/project/morana/0.2.0/), and the archived source under
[DOI 10.5281/zenodo.23118502](https://doi.org/10.5281/zenodo.23118502).

### Added

- BiCGSTAB linear solves for fixed-source and criticality calculations, with
  Jacobi preconditioning as the recommended scalable iterative policy.
- A [custom-preconditioner example](examples.md#custom-finite-volume-operator-solve)
  using public finite-volume operators and checking its solution against the
  direct reference.

### Changed

- Iterative settings now default to Jacobi preconditioning. Criticality inner
  solves continue from the preceding cleaned inner solution after the first
  outer iteration; restarted GMRES remains available.
- Finite-volume operator assembly is faster and uses less memory in the
  maintained many-group comparisons, with unchanged numerical settings and
  checked-result requirements. Gains depend on the workload and environment.
- Result archives now use schema 9, and maintained examples use module execution.

### Removed

- Built-in threshold-ILU preconditioning.
- The former solver-comparison example and the MMS examples' strategy-comparison
  options. The MMS examples retain direct-reference refinement.

### Upgrade actions

- Replace `IluPreconditioner` with `JacobiPreconditioner` or `NoPreconditioner`.
  GMRES callers requiring the previous unpreconditioned default must select
  `NoPreconditioner()` explicitly. See the
  [solver policies](modeling_workflow.md#choose-numerical-policies).
- Regenerate [result archives](result_archives.md) with schema 9; readers reject
  earlier schemas.
- Run maintained examples with `python -m examples.<module>` instead of
  `python examples/<module>.py`.
- Replace `examples.solver_comparison` with `examples.custom_operator_solve` for
  custom-preconditioner exploration. Remove `--compare-strategies` from MMS
  commands.

## 0.1.0 - 2026-09-09

Version `0.1.0` is Morana's initial source release. It establishes the
object-based Python API and finite-volume reference implementation for
multigroup fixed-source and `k_eff` calculations on variable-height hex-z
lattices. The release is available as a
[GitHub release](https://github.com/tannhorn/morana/releases/tag/v0.1.0) and
archived under
[DOI 10.5281/zenodo.22681316](https://doi.org/10.5281/zenodo.22681316). Its
source distribution and pure-Python wheel are also available from
[PyPI](https://pypi.org/project/morana/0.1.0/).

Morana 0.1.0 is its first public release; development before this release
occurred in a private repository, so the public Git history begins with version
0.1.0.

### Added

- Direct and preconditioned restarted-GMRES linear solves, power iteration,
  and optional fixed Wielandt-shift acceleration.
- Material, source, boundary-condition, normalization, balance, execution
  report, result-archive, plotting, and VTK-output APIs.
- Dependency-neutral import of selected macroscopic material data from OpenMC
  runtime-MGXS HDF5 libraries.
- Maintained examples, theory and workflow documentation, analytic and
  manufactured-solution verification cases, and an OpenMC comparison.

### Verification

- Release evidence includes conservation and residual checks, analytic axial
  and manufactured-solution cases, independently assembled operator
  comparisons, result-archive checks, maintained examples, and an end-to-end
  OpenMC comparison. The [verification guide](verification.md) defines the
  evidence and its limits.

### Limitations

- Morana has not been validated against experimental reactor measurements or
  qualified for design, operational, licensing, regulatory, or safety
  decisions.
- The supported platform is Linux with Python 3.12 or newer. The public API and
  result-archive schema may change incompatibly during early development.
- Version `0.1.0` is distributed through
  [PyPI](https://pypi.org/project/morana/0.1.0/) and as source through
  [GitHub](https://github.com/tannhorn/morana/releases/tag/v0.1.0) and Zenodo.
  Conda packages, native binaries, containers, and bundled nuclear data are not
  provided.
