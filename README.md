# Morana

<p align="center">
  <img src="docs/assets/morana.png" alt="Morana" width="220">
</p>

*Morana* (**MO**dern **R**eactor **ANA**lysis) is a Python package for
finite-volume neutron-diffusion calculations on regular hex-z lattices. It is
a transparent reference implementation for research, teaching, and
numerical-method development. The package import name is `morana`.

Read the [published documentation](https://www.lubomirbures.com/morana/) for
installation, modeling guidance, examples, verification evidence, and the
Python API reference.

Morana supports one-group and multigroup fixed-source and `k_eff`
calculations on variable-height hex-z material meshes. See the
[documentation home](docs/index.md) for a concise capability overview and the
[modeling and solver workflow](docs/modeling_workflow.md) for the supported
workflow and capability limits.

## Project status

Morana is intended for research, teaching, and software development. It has
not been validated against experimental reactor measurements or qualified for
design, operational, licensing, or regulatory decisions. Results must be
independently reviewed and validated for their intended use. Example cross
sections are illustrative and are not reactor-design data. See
[verification and comparisons](docs/verification.md) for the evidence and its
limitations.

Morana is also in early development: its public Python API may change
incompatibly without a backward-compatibility guarantee. Record the exact source
commit and dependency environment for reproducible work, and review the
[API stability policy](docs/modeling_workflow.md#api-stability) and
[changelog](docs/changelog.md) before upgrading.

Morana 0.1.0 is its first public release; development before this release
occurred in a private repository, so the public Git history begins with version
0.1.0.

## Development disclosure

Generative-AI coding tools assist Morana development. Project maintainers
remain responsible for design decisions, review, testing, documentation, and
released code. This disclosure does not alter the Apache-2.0 license or its
warranty disclaimer; it records the maintainers' responsibility for reviewing
and releasing AI-assisted contributions.

## Installation

Morana requires Python 3.12 or newer. Version `0.1.0` is available as a
[source release from GitHub](https://github.com/tannhorn/morana/releases/tag/v0.1.0)
and is archived on Zenodo under
[DOI 10.5281/zenodo.22681316](https://doi.org/10.5281/zenodo.22681316). No
package-index distribution is currently available. Install the exact release
into an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install \
  https://github.com/tannhorn/morana/releases/download/v0.1.0/morana-0.1.0.tar.gz
```

Follow [installation and quickstart](docs/getting_started.md) for platform
requirements, the WSL option for Windows users, and a first analytic problem.
Contributors should use the documented Conda development environment. The
[maintained examples](docs/examples.md) demonstrate the supported solver,
boundary, geometry, and output workflows.

## Documentation

- **Start:** [installation and quickstart](docs/getting_started.md) and
  [maintained examples](docs/examples.md)
- **Understand:** [geometry and indexing](docs/geometry.md),
  [theory and numerical conventions](docs/theory_references.md), and
  [inspection and output](docs/outputs.md)
- **Build and solve:** [modeling and solver workflow](docs/modeling_workflow.md),
  [boundary conditions and face selection](docs/boundary_conditions.md), and
  [source-derived Python reference](docs/reference/index.md)
- **Evaluate and contribute:**
  [verification and comparisons](docs/verification.md),
  [changelog](docs/changelog.md), and
  [contributor workflow](docs/contributor_workflow.md)

The [contributor workflow](docs/contributor_workflow.md#previewing-documentation)
explains how to build and preview the complete documentation locally.

## Citation

Machine-readable citation metadata is provided in
[`CITATION.cff`](CITATION.cff). The version DOI for Morana 0.1.0 is
[10.5281/zenodo.22681316](https://doi.org/10.5281/zenodo.22681316).
For the latest Morana release rather than a specific version, use the
[all-versions DOI 10.5281/zenodo.22681315](https://doi.org/10.5281/zenodo.22681315).

## License

Morana source code and authored documentation are licensed under
[Apache-2.0](LICENSE) and provided without warranties or conditions beyond
those stated in the license. The documentation site includes
[third-party notices](docs/licenses.md) for its distributed frontend assets.
