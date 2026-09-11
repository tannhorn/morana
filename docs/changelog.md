# Changelog

This page records dated releases and subsequent unreleased changes. Entries
follow the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) categories
where they apply: Added, Changed, Deprecated, Removed, Fixed, and Security.
Each release section summarizes capabilities, limitations, verification
evidence, and upgrade actions. Current package behavior is documented in the
[modeling and solver workflow](modeling_workflow.md),
[verification guide](verification.md), and source-derived
[Python reference](reference/index.md).

## Unreleased

### Changed

- Added the Zenodo all-versions DOI to project-level citation guidance while
  retaining the version DOI for citations of Morana 0.1.0.

## 0.1.0 - 2026-09-09

Version `0.1.0` is Morana's initial source release. It establishes the
object-based Python API and finite-volume reference implementation for
multigroup fixed-source and `k_eff` calculations on variable-height hex-z
lattices. The release is available from GitHub and archived under
[DOI 10.5281/zenodo.22681316](https://doi.org/10.5281/zenodo.22681316); no
package-index distribution is available.

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
- Version `0.1.0` is distributed as source through GitHub and Zenodo; PyPI,
  Conda, native binaries, containers, and bundled nuclear data are not
  provided.
