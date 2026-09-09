# Changelog

This page records unreleased changes and, once available, dated releases. Entries
follow the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) categories
where they apply: Added, Changed, Deprecated, Removed, Fixed, and Security.
Each dated release section summarizes capabilities, limitations, verification
evidence, and upgrade actions. Current package behavior is documented in the
[modeling and solver workflow](modeling_workflow.md),
[verification guide](verification.md), and source-derived
[Python reference](reference/index.md).

## Unreleased

Version `0.1.0` is prepared as Morana's initial release but has not been
published. It establishes the object-based Python API and finite-volume
reference implementation for multigroup fixed-source and `k_eff` calculations
on variable-height hex-z lattices.

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
