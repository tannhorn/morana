# OpenMC–Morana SRE-derived comparison case

This directory contains the executable OpenMC–Morana comparison. It follows a
heterogeneous unit cell through MGXS generation, a 61-cell Morana diffusion
calculation, and comparison with an independent continuous-energy reference.

- [Model and
  results](https://www.lubomirbures.com/morana/openmc_comparison.html) describes
  the physical inputs, numerical evidence, figures, interpretation, and
  references.
- [Reproduction
  workflow](https://www.lubomirbures.com/morana/openmc_comparison_workflow.html)
  gives the environment setup, staged commands, restart requirements, artifact
  layout, and script file map. The corresponding Markdown sources remain under
  [`docs/`](../../docs/) in the repository.

Run commands from the repository root. Generated artifacts belong under the
ignored `artifacts/examples/openmc_comparison/` directory. OpenMC transport
requires a separately configured evaluated-data library; raw calculation data
are not distributed with Morana.
