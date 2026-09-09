# OpenMC comparison workflow

Run the commands on this page from the repository root. The scripts live in
`examples/openmc_comparison/` and share the geometry and material definitions
used by the CE reference, MGXS generation, and Morana model. See the
[comparison results](openmc_comparison.md) for the physical model, figures,
numerical evidence, and interpretation.

## Generated artifacts

The directory contains authored inputs only. Generated figures, OpenMC XML,
statepoints, runtime-MGXS libraries, Morana result archives, NumPy arrays, and
JSON summaries are written below
`artifacts/examples/openmc_comparison/` by default. That directory is ignored
by Git.

Do not add raw calculation artifacts or evaluated nuclear data to the
repository. Only the final documentation figures under `docs/assets/` are
tracked.

## Environments

Run the workflow from an active Python environment containing the checked-out
version of Morana. The canonical
[user-installation instructions](getting_started.md#user-installation)
cover environment creation and package installation. A regular, non-editable
installation is sufficient for the direct plotters, coordinate and reader
checks, Morana solves, comparisons, and documentation-figure generator.

OpenMC construction, CSG checks, rendering, CE transport, and MGXS generation
also require an OpenMC environment with a configured continuous-energy data
library. OpenMC and Morana may share one environment, but the supplied wrapper
also supports separate environments.

For the Conda commands below, replace `your-openmc-environment` with the
name of your OpenMC environment. The wrapper uses this same variable:

```bash
export OPENMC_CONDA_ENV="your-openmc-environment"
```

## Inspect the model

Render the unit cell directly with Matplotlib and verify the coordinate
mapping without OpenMC:

```bash
python examples/openmc_comparison/plot_unit_cell.py
python examples/openmc_comparison/check_morana_mapping.py
```

The mapping diagnostic labels every source `(q, r)` coordinate, Morana planar
ID, and deliberately nonuniform value. It makes rotations, reflections, and
ordering mistakes visible.

The deterministic comparison-reader check also needs neither OpenMC nor
persistent artifacts:

```bash
python examples/openmc_comparison/check_comparison_reader.py
```

With OpenMC available, check the shared-surface CSG structure and render the
unit cell and mini-core without running particle transport:

```bash
python examples/openmc_comparison/check_structure.py
python examples/openmc_comparison/render_geometry.py
```

## Generate the CE reference

`generate_ce_reference.py` runs the heterogeneous 61-cell mini-core and
directly tallies `nu-fission` in 100 uniform axial bins and in the 61 root
cells. Its defaults reproduce the documented 100-million-active-history
request:

```bash
conda run -n "$OPENMC_CONDA_ENV" \
  python examples/openmc_comparison/generate_ce_reference.py
```

The default output is
`artifacts/examples/openmc_comparison/ce_reference/100m_seed31415_profiles_z100/`.
The JSON summary qualifies a reference only when the multiplication-factor
standard deviation is at most 10 pcm and every directly tallied production bin
has at most 1% relative standard deviation. Smaller particle or batch controls
are useful for workflow tests but do not define the published reference.

The runner is restart-aware. A continuation must match the source model,
particle count, inactive batches, seed, tally layout, OpenMC version, and
nuclear-data library path recorded beside the statepoint. These metadata
checks do not hash the library contents or the Python model; keep both
unchanged when reusing or continuing a run.

## Generate MGXS directly

`generate_mgxs.py` runs the reflected two-dimensional unit cell. It tallies
CASMO-70 and condenses the same statepoint to CASMO-40, CASMO-25, and CASMO-8:

```bash
conda run -n "$OPENMC_CONDA_ENV" \
  python examples/openmc_comparison/generate_mgxs.py
```

The defaults request 20 million active histories with seed 31415 and write
four runtime libraries plus XML, statepoint, and `generation_summary.json`
below `artifacts/examples/openmc_comparison/mgxs/20m_seed31415/`.

Use `--from-statepoint` to recreate all four runtime libraries from the
selected run's statepoint without particle transport. The generated
`homogenized_cell` record stores OpenMC `TransportXS` in the runtime `total`
field; Morana must therefore import it with `diffusion="total"`.

## Run a Morana study case

`run_study_case.py` prepares or reuses one exact history/seed MGXS sample,
then solves every requested group structure and axial mesh. Each Morana case
writes:

- `result.morana-result`, a portable result archive;
- `normalized_production.npy`, unit-normalized volume-integrated
  fission-neutron production; and
- `summary.json`, including model provenance and an optional checked CE
  comparison.

When OpenMC and Morana share the active environment, run both stages directly:

```bash
python examples/openmc_comparison/run_study_case.py \
  --active-histories 20000000 \
  --seed 31415 \
  --group-structures CASMO-70 \
  --axial-layers 20 \
  --ce-reference \
  artifacts/examples/openmc_comparison/ce_reference/100m_seed31415_profiles_z100/reference_summary.json
```

For separate environments, use the wrapper after activating the virtual
environment containing Morana and setting `OPENMC_CONDA_ENV`:

```bash
examples/openmc_comparison/run_study.sh \
  --active-histories 20000000 \
  --seed 31415 \
  --group-structures CASMO-70 \
  --axial-layers 20 \
  --ce-reference \
  artifacts/examples/openmc_comparison/ce_reference/100m_seed31415_profiles_z100/reference_summary.json
```

The wrapper always runs the MGXS preparation stage through `conda run`, then
the Morana stage with the active Python interpreter. It appends the two
`--stage` selections itself, so do not pass `--stage` to the wrapper. Other
model and output options pass through to `run_study_case.py`; use
`--output-dir PATH` to select a nondefault study root. To execute only one
stage, call `run_study_case.py` directly with `--stage mgxs` under the OpenMC
interpreter or `--stage morana` under the Morana interpreter.

## Reproduce the discretization studies

The energy-group screen uses one common 20-million-history seed and 20 axial
layers:

```bash
examples/openmc_comparison/run_study.sh \
  --active-histories 20000000 \
  --seed 31415 \
  --group-structures CASMO-8 CASMO-25 CASMO-40 CASMO-70 \
  --axial-layers 20 \
  --ce-reference \
  artifacts/examples/openmc_comparison/ce_reference/100m_seed31415_profiles_z100/reference_summary.json
```

The axial screen reuses those existing libraries and solves CASMO-25 on the
documented meshes with the active Morana interpreter:

```bash
python examples/openmc_comparison/run_study_case.py \
  --stage morana \
  --active-histories 20000000 \
  --seed 31415 \
  --group-structures CASMO-25 \
  --axial-layers 5 10 20 50 100 \
  --ce-reference \
  artifacts/examples/openmc_comparison/ce_reference/100m_seed31415_profiles_z100/reference_summary.json
```

Automatic MGXS continuation with `generations_per_batch > 1` requires an
OpenMC implementation that restores eigenvalue statistics by generation.
The [OpenMC 0.16.0 state-point implementation](https://github.com/openmc-dev/openmc/blob/v0.16.0/src/state_point.cpp)
instead uses batch indices in this calculation. The documented continuations
used a locally corrected build. Fresh runs and statepoint-only MGXS export do
not require that correction.

For the final sampling study, create or confirm a 20-million-history sample
for each selected seed, then continue each matching sample to 40 million:

```bash
for seed in 16180 27182 31415; do
  examples/openmc_comparison/run_study.sh \
    --active-histories 20000000 \
    --seed "$seed" \
    --group-structures CASMO-70 \
    --axial-layers 20 \
    --ce-reference \
    artifacts/examples/openmc_comparison/ce_reference/100m_seed31415_profiles_z100/reference_summary.json

  examples/openmc_comparison/run_study.sh \
    --active-histories 40000000 \
    --seed "$seed" \
    --group-structures CASMO-70 \
    --axial-layers 20 \
    --ce-reference \
    artifacts/examples/openmc_comparison/ce_reference/100m_seed31415_profiles_z100/reference_summary.json
done
```


## Compare saved cases

`compare_study_cases.py` compares two compatible Morana summaries or one
Morana summary with a qualified CE reference. Pass the candidate first and
reference second. For example:

```bash
python examples/openmc_comparison/compare_study_cases.py \
  artifacts/examples/openmc_comparison/study/morana/20m_seed31415/casmo-40/z20/summary.json \
  artifacts/examples/openmc_comparison/study/morana/20m_seed31415/casmo-70/z20/summary.json
```

To compare with CE and save the checked comparison:

```bash
python examples/openmc_comparison/compare_study_cases.py \
  artifacts/examples/openmc_comparison/study/morana/40m_seed31415/casmo-70/z20/summary.json \
  artifacts/examples/openmc_comparison/ce_reference/100m_seed31415_profiles_z100/reference_summary.json \
  --output artifacts/examples/openmc_comparison/study/comparisons/casmo-70-z20.json
```

The reader validates profile layout, raw-to-normalized consistency,
uncertainties, physical dimensions, recorded OpenMC version and nuclear-data
path, diffusion convention, and canonical cell order before reporting
differences. A matching data path is not a checksum of the library contents.

## Regenerate the documentation figures

After the complete local study exists at its default paths, write the four
tracked documentation figures directly to the documentation asset directory:

```bash
python examples/openmc_comparison/plot_documentation.py \
  --documentation-assets-dir docs/assets
```

The generator validates the CE and Morana records before plotting the unit
cell, convergence studies, mean selected planar production and error, and mean
selected axial production and error. The tracked PNG canvases are transparent,
the production maps use `inferno`, and the signed error map uses a diverging
scale centered on zero. Pass `--study-dir PATH` or `--ce-reference PATH` when
the source artifacts are elsewhere. Review figure changes and the associated
numerical summaries before committing them.

## File map

| Files | Responsibility |
| --- | --- |
| `specification.py`, `geometry.py`, `plotting.py`, `artifact_paths.py`, `run_settings.py` | OpenMC-independent conditions, geometry, colors, paths, and statistical controls |
| `case.py` | OpenMC materials, reflected unit cell, three-dimensional mini-core, and eigenvalue settings |
| `plot_unit_cell.py`, `render_geometry.py` | Direct Matplotlib and OpenMC geometry rendering |
| `check_structure.py`, `check_morana_mapping.py`, `check_comparison_reader.py` | CSG, coordinate-order, and portable-reader checks |
| `generate_ce_reference.py` | Restart-aware CE calculation and direct axial and planar profiles |
| `mgxs_generation.py`, `generate_mgxs.py` | MGXS run identity, restart selection, tallying, condensation, and export |
| `run_study.sh`, `run_study_case.py`, `compare_study_cases.py` | Split-environment orchestration, Morana execution, and saved-case comparisons |
| `plot_documentation.py` | Checked generation of the four tracked publication figures |
