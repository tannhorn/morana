# Morana {.morana-home-title}

<p class="morana-home-logo" align="center">
  <span class="morana-home-logo__art">
    <img src="assets/morana.png" alt="Morana" width="220">
  </span>
</p>

*Morana* (**MO**dern **R**eactor **ANA**lysis) is a Python package for
finite-volume neutron-diffusion calculations on regular hex-z lattices. It
provides a transparent Python API and serves as a reference implementation for
research, teaching, and numerical-method development. The package import name
is `morana`. The [source code](https://github.com/tannhorn/morana) is available
on GitHub.

## Supported calculations

Morana provides:

- one-group and multigroup fixed-source and `k_eff` calculations;
- variable-height axial stacks with coupled radial and axial diffusion;
- material-dependent multigroup scattering with optional neutron multiplicity
  and either separable or general incident-to-outgoing fission production;
- reflective, vacuum, prescribed-flux, Robin, partial-current-return, and
  incoming-current boundary conditions;
- scalar, group, and axial-layer balance diagnostics with immutable input
  provenance;
- selected macroscopic material-data import from OpenMC runtime-MGXS HDF5
  libraries, without importing OpenMC at runtime; and
- Matplotlib, Plotly, VTU, and material-aware VTM inspection output.

The [modeling and solver workflow](modeling_workflow.md#capability-boundaries)
defines the exact limits of these capabilities.

## Choose a starting point

- **Install and run a first calculation:** follow
  [installation and quickstart](getting_started.md), then execute a
  [maintained example](examples.md).
- **Understand the model:** begin with
  [geometry and indexing](geometry.md), continue through the
  [theory and numerical conventions](theory_references.md), and then review
  [inspection and output](outputs.md).
- **Build and solve:** use the
  [modeling and solver workflow](modeling_workflow.md) for cross-object
  behavior, [boundary conditions and face selection](boundary_conditions.md)
  for boundary assignments, the [OpenMC MGXS import guide](openmc_mgxs.md) for
  external material data, and the
  [source-derived reference](reference/index.md) for exact signatures and
  public members.
- **Assess the evidence:** read
  [verification and comparisons](verification.md) for tested behavior,
  analytic references, manufactured-solution studies, the end-to-end
  OpenMC–Morana comparison, and their limitations.
- **Develop or track changes:** read the
  [contributor workflow](contributor_workflow.md) and
  [changelog](changelog.md). The latest source release is version `0.1.0`;
  install its package distribution from PyPI.

!!! warning "Project status and intended use"
    Morana is intended for research, teaching, and software development. It
    has not been validated against experimental reactor measurements or
    qualified for design, operational, licensing, or regulatory decisions.
    Results must be independently reviewed and validated for their intended
    use. The example cross sections are illustrative, not reactor-design data.

!!! warning "Early-development API"
    Morana's public Python API may change incompatibly without a
    backward-compatibility guarantee. Record the source commit and dependency
    environment for reproducible work, and review the
    [API stability policy](modeling_workflow.md#api-stability) and
    [changelog](changelog.md) before upgrading.

Morana is distributed under the Apache License 2.0 without warranties or
conditions beyond those stated in the license. See
[licenses and third-party notices](licenses.md).
