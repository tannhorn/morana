# Contributor workflow

This page is for contributors and maintainers changing Morana's source code,
tests, documentation, examples, or build dependencies. Package users do not
need these checks to run Morana; start instead with
[installation and quickstart](getting_started.md). Development takes place in
the [Morana repository](https://github.com/tannhorn/morana); proposed changes
are submitted as [pull requests](https://github.com/tannhorn/morana/pulls), and
bugs or scoped feature requests can be reported through the
[issue tracker](https://github.com/tannhorn/morana/issues).

## Development disclosure

Generative-AI coding tools assist Morana development. Project maintainers
remain responsible for design decisions, review, testing, documentation, and
released code. This disclosure does not alter the Apache-2.0 license or its
warranty disclaimer; it records the maintainers' responsibility for reviewing
and releasing AI-assisted contributions.

## Development environment

Use the editable Conda environment described in
[installation and quickstart](getting_started.md). Update it after dependency
or documentation-tool changes:

```bash
conda env update -n morana-dev -f environment.yml --prune
conda activate morana-dev
```

The development environment includes Python VTK so tests can reopen Morana's
VTU and VTM output. Morana's export implementation does not depend on VTK at
runtime.

## Checks before submitting changes

After changing Python source, run:

```bash
pytest
pylint src/morana
python -m compileall src examples tests
```

Run `reuse lint` after adding, moving, or renaming repository files and before
every completed milestone handoff:

```bash
reuse lint
```

After changing `CITATION.cff`, validate it against the Citation File Format
schema:

```bash
cffconvert --validate -i CITATION.cff
```

After adding or editing a release-note fragment or changing its allocation
counter, validate the fragment collection:

```bash
python scripts/check_change_fragments.py
```

After changing tracked documentation, public NumPy-style docstrings, exported
API declarations, or MkDocs configuration, build the complete site strictly:

```bash
mkdocs build --strict
python scripts/check_internal_links.py --site-path /morana/
python scripts/check_reference_exports.py
python scripts/check_spelling_and_terms.py
```

The link check scans the generated HTML, verifies every local `href` target,
and requires each URL fragment to match an ID in its target page. It also scans
tracked Markdown outside the configured documentation source directory. Links
below the published `site_url` map back to the generated site, while GitHub
`blob/main` and `tree/main` links below the configured `repo_url` map back to
the checkout. These project cross-links are checked locally without network
requests; other external links are ignored. The check therefore catches stale
heading anchors and project cross-links that the strict MkDocs build does not
reject.
The reference-export check compares the runtime `__all__` declarations of
`morana`, `morana.solvers.finite_volume`, and `morana.operators` with exact
mkdocstrings IDs in their generated reference pages, rejecting missing runtime
attributes, missing anchors, and duplicate anchors.
The spelling and terminology check scans authored documentation, public source
and docstrings, maintained examples, release-note fragments, and maintenance
scripts. Add legitimate technical words to `scripts/spelling_vocabulary.txt`;
keep discouraged forms and their canonical replacements in
`scripts/terminology_rules.toml`.

After a substantial change, also run every maintained example in scope.
The routine suite is:

```bash
examples/run_all.sh
```

Computationally expensive staged workflows, such as the OpenMC comparison,
document their own reproduction commands and are run separately when in scope.

## Continuous integration

Verification is local-first. Run the commands on this page before integration
or release work. Make new changes on the long-lived `devel` branch; keep
`main` for integrated, release-ready work. Pushes to `main` and pull requests
into `main` run tests on Python 3.12, 3.13, and 3.14, plus static, licensing,
spelling, terminology, documentation, internal-link, and reference-export
checks. Run the local checks during ordinary `devel` work, or dispatch a
workflow manually when clean-environment verification is useful before a pull
request. Once a pull request is open, each update to it runs the hosted checks.
Only pushes to `main` deploy documentation. The protected `main` branch
requires every hosted check on an up-to-date pull request and allows
squash-merging only.

Because a squash merge gives the integrated change a new commit identity,
synchronize the long-lived `devel` branch immediately after each pull request
merge and before beginning new work:

```bash
scripts/sync_devel_after_pr.sh <PR_NUMBER>
```

The script requires a clean worktree, fetches `origin`, merges `origin/main`
into `devel` with a descriptive synchronization commit, and pushes `devel`.
Keeping this merge-back step adjacent to the squash merge prevents the branch
histories from accumulating unrelated versions of the same change.

Merging a pull request into `main` integrates the change; it does not publish a
Morana release. Ordinary pull requests may update project-level documentation
or contributor automation. A change that affects a specific release—the
package version, `CITATION.cff` version or version DOI, dated changelog entry,
release URL, source archive, tag, or GitHub and Zenodo publication—must use the
explicit release procedure below. During development, record relevant public
changes as the committed fragments described below instead of editing the
changelog. When the availability of an already released version changes,
update that dated release entry instead of creating a fragment for a future
release.

Morana currently uses sole-maintainer release approval. The maintainer may
approve publication without a second-person review, but the protected
pull-request and verification gates still apply.

## Change fragments

Change fragments are temporary, reviewable inputs to the next changelog, not a
second authority for implemented behavior. Add one with the implementation of
each notable user-facing capability, behavior or compatibility change, fix,
deprecation, removal, or security correction. Do not add fragments for routine
documentation corrections, citation metadata, repository maintenance, CI,
tests, internal planning, or maintainer-only tooling.

Fragments live under `changes/` and use
`NNNNNN.category.md`, where `NNNNNN` is a six-digit repository sequence number
and `category` is one of `added`, `changed`, `deprecated`, `removed`, `fixed`,
or `security`. Allocate the value currently stored in
`changes/next_id.txt`, then increment that file in the same change. Never
decrement the counter, fill an old gap, or reuse the number of a released
fragment. Sequence numbers identify fragments; GitHub issue numbers remain
separate metadata.

Each fragment consists of YAML front matter followed by concise Markdown:

```markdown
---
issues: []
breaking: false
upgrade: null
documentation:
  - docs/modeling_workflow.md
---
Added a reproducible many-group performance workflow for evaluating
finite-volume solver scaling on deterministic synthetic workloads.
```

The four metadata fields are required:

- `issues` is a list of positive GitHub issue numbers and may be empty.
- `breaking` is `true` only when existing users may need to change their code,
  data, or workflow.
- `upgrade` is `null` when no action is needed; otherwise it is a short,
  actionable instruction. A breaking fragment must provide one.
- `documentation` lists repository-relative paths to the tracked pages or
  public source files that own the changed behavior and may be empty when no
  such page applies. Every listed path must exist.

Write the body for package users and describe observable impact rather than
commits, file edits, tests, or implementation mechanics. Keep related effects
in one fragment when they will naturally form one release-note item. A fragment
may be revised or combined before release if its implementation changes.

Run `python scripts/check_change_fragments.py` before submitting the change.
The checker validates filenames, allocation state, metadata types, referenced
paths, and nonempty bodies. The ordinary test suite and CI also exercise this
check.

## Publishing package distributions

Package-index publication is separate from the GitHub and Zenodo source-release
procedure. The dedicated `.github/workflows/publish.yml` workflow never runs
for a branch push. It checks out an annotated `v<VERSION>` tag, confirms the
project metadata, builds exactly one source distribution and one pure-Python
wheel, validates their contents, and transfers those checked files to a
separate publishing job.

Configure the PyPI Trusted Publisher for the repository, `publish.yml`, and the
`pypi` GitHub environment. Require maintainer approval for that environment.
Only the publishing job receives permission to request short-lived OIDC
credentials; do not store a package-index API token in GitHub.

Publish the GitHub release only after its annotated tag, source metadata,
release evidence, and distribution checks are complete. The `release:
published` event selects the protected production path. After publication,
perform a clean, no-cache installation from PyPI and run the installed
metadata, quickstart, and result-archive smoke paths. Package files and versions
are immutable: do not rerun a successful upload or move its tag.

## Publishing a source release

Only the following explicit procedure turns an exact `main` commit into a
Morana release through GitHub and Zenodo. It is separate from ordinary `main`
integration.

Before opening that release pull request, inspect the Git range since the
previous tag and reconcile it with every pending fragment. Account for notable
public changes, combine related fragments into concise user-facing entries,
and retain any required upgrade actions and important limitations. Add the
result as the new dated section of `docs/changelog.md`; do not mechanically
concatenate fragment bodies. Delete all consumed fragment files in the same
release change, leave `changes/next_id.txt` at its current value, and run
`python scripts/check_change_fragments.py` again.

Prepare a release on `devel`, then use a pull request from `devel` into `main`
to integrate the package version, dated changelog, citation metadata,
installation guidance, and public URLs. The Zenodo version DOI must already be
present in the source. After the pull request is squash-merged, update local
`main` without creating another commit and record the exact release commit:

```bash
git switch main
git pull --ff-only
git status --short
git rev-parse HEAD
```

The status output must be empty. Run the complete local verification suite from
this page, including all maintained routine examples and the strict
documentation checks. Confirm that the hosted verification and documentation
workflows also pass on the same commit. Any failure requires a new pull request
and a complete rerun on its merged commit.

Build the release archive from the verified commit rather than from the working
directory. Replace `<VERSION>` with the version being published and
`<RELEASE_COMMIT>` with the recorded full commit hash before running each
command block:

```bash
release_version="<VERSION>"
release_commit="<RELEASE_COMMIT>"
archive_name="morana-${release_version}.tar.gz"
mkdir -p dist
test ! -e "dist/${archive_name}"
test ! -e "dist/${archive_name}.sha256"
git archive \
  --format=tar.gz \
  --prefix="morana-${release_version}/" \
  --output="dist/${archive_name}" \
  "$release_commit"
cd dist
sha256sum "$archive_name" > "${archive_name}.sha256"
cd ..
```

Inspect the archive before publication:

```bash
release_version="<VERSION>"
archive_name="morana-${release_version}.tar.gz"
tar -tzf "dist/${archive_name}"
gzip -dc "dist/${archive_name}" | git get-tar-commit-id
cd dist
sha256sum --check "${archive_name}.sha256"
cd ..
```

It must have one `morana-<VERSION>/` top-level directory and contain the
package source, `pyproject.toml`, README, citation metadata, canonical license
files, documentation, examples, and tests. It must exclude generated sites,
caches, local environments, result archives, raw study data, and other
untracked artifacts.

Clean-install and test the archive outside the repository before tagging:

```bash
release_version="<VERSION>"
archive_name="morana-${release_version}.tar.gz"
archive_path="$(pwd)/dist/${archive_name}"
release_test_dir="$(mktemp -d)"
python -m venv "$release_test_dir/venv"
"$release_test_dir/venv/bin/python" -m pip install "$archive_path"
"$release_test_dir/venv/bin/python" -c \
  'import sys; from importlib.metadata import version; import morana; assert version("morana") == sys.argv[1]' \
  "$release_version"
"$release_test_dir/venv/bin/python" examples/quickstart.py
"$release_test_dir/venv/bin/python" examples/result_archive.py \
  --output-dir "$release_test_dir/output"
```

Create a signed tag if signing is configured; otherwise create an annotated
tag. The tag must point to `release_commit` and must never be moved after
publication:

```bash
release_version="<VERSION>"
release_commit="<RELEASE_COMMIT>"
git tag -a "v${release_version}" "$release_commit" \
  -m "Morana ${release_version}"
git show --no-patch --decorate "v${release_version}"
```

During one coordinated release window:

1. Push the verified commit and `v<VERSION>` tag.
2. Upload `morana-<VERSION>.tar.gz` and its SHA-256 file to the prepared Zenodo
   draft, then publish it and verify the version DOI.
3. Create the GitHub release from the same tag, attach the identical two files,
   link the Zenodo record and documentation, and use the dated changelog as the
   release-note basis.
4. Approve the protected PyPI publishing job, verify its distribution files,
   and clean-install the exact version from PyPI.
5. Verify the deployed documentation and clean-install again from the
   published GitHub release archive.
6. Record Zenodo's concept DOI for project-level citation links while retaining
   the version DOI for citations of the specific release.

Do not use automatic GitHub-release ingestion: the archived source must already
contain its version DOI. Do not publish either channel after a failed gate,
rebuild the archive between channels, replace an accepted archive, reuse the
version, or move the public tag.

## Licensing files and dependencies

Morana source code, authored documentation, tests, examples, and ordinary
project assets are Apache-2.0. The root
[`CONTRIBUTING.md`](https://github.com/tannhorn/morana/blob/main/CONTRIBUTING.md)
records the contribution agreement. The root
[`LICENSE`](https://github.com/tannhorn/morana/blob/main/LICENSE) is the
canonical project license,
[`pyproject.toml`](https://github.com/tannhorn/morana/blob/main/pyproject.toml)
declares the package license, and
[`LICENSES/`](https://github.com/tannhorn/morana/tree/main/LICENSES) contains
canonical SPDX license texts.

`REUSE.toml` supplies copyright and SPDX license annotations for the
repository paths. When adding or moving a file:

1. Check whether its new path is covered by the intended `REUSE.toml`
   annotation. Extend the path list or add an explicit annotation when it is
   not.
2. For third-party, generated, or separately licensed material, record its
   actual provenance, copyright holder, and SPDX identifier with a
   higher-precedence override. Do not let it inherit Morana's authored-file
   annotation merely because it sits under `docs/` or another covered path.
3. Preserve the upstream license and required notices. Add a canonical text
   under `LICENSES/` when the SPDX license is not already present; keep bundled
   asset licenses with the assets when upstream distribution requires it.
4. If an asset or frontend component is copied into the generated site, update
   [licenses and third-party notices](licenses.md) so deployed documentation
   carries the required notice. Merely declaring a Python dependency does not
   make its source part of Morana's distributed documentation.
5. Run `reuse lint` and inspect its file count and license summary before
   handing off the change.

The Morana logo is the exception to the authored Apache-2.0 default:
`REUSE.toml` records its ChatGPT provenance and CC0-1.0
dedication. Vendored MathJax has its own Apache-2.0 override under The MathJax
Consortium. Keep these overrides synchronized with the assets they cover.

## Previewing documentation

The generated `site/` directory is an untracked build artifact. The tracked
Markdown and source docstrings remain the documentation sources. The build uses
flat `.html` links, so `site/index.html` and its navigation also work when
opened directly through a local `file://` URL without a web server.

Use the live local preview while reviewing interactive documentation features:

```bash
mkdocs serve
```

Open the local address printed by MkDocs (normally `http://127.0.0.1:8000/`)
and stop the preview with <kbd>Ctrl</kbd>+<kbd>C</kbd>. Material for MkDocs 9.x
provides the responsive light theme, navigation, search presentation, and code
controls. Morana-specific colors, content width, table treatment, and
generated-signature styling live in `docs/stylesheets/extra.css`; keep those
overrides small and verify the light presentation after changing them. The
generated site includes the required
[third-party notices](licenses.md) for Material for MkDocs and MkDocs.

## Regenerating authored figures

The maintained verification examples generate their tracked result figures
when given the documentation asset directory:

```bash
python examples/one_group_keff.py --documentation-assets-dir docs/assets
python examples/fixed_source_mms.py --documentation-assets-dir docs/assets
python examples/keff_mms.py --documentation-assets-dir docs/assets
python examples/openmc_comparison/plot_documentation.py \
  --documentation-assets-dir docs/assets
```

Each owning verification page documents which figures and numerical evidence
its command regenerates. The OpenMC comparison command additionally requires
the complete local CE and Morana study records described by the
[comparison workflow](openmc_comparison.md#reproduce-the-example). Review the
numerical output before accepting an asset change.

The affine-Robin parameter figure has a separate Matplotlib generator. After
changing the boundary equations or figure presentation, run:

```bash
python scripts/generate_boundary_parameter_figure.py
```

This deterministically rewrites the tracked Matplotlib-generated image
`docs/assets/boundary_parameter_relations.png`. The script also accepts
`--output path.png` for inspection copies.

## Regenerating the synthetic OpenMC MGXS fixture

The checked-in OpenMC runtime-MGXS fixture is a small integration artifact for
the material-data importer and is exercised by its routine tests. Regenerate
it only when intentionally changing its synthetic arrays or updating the
documented OpenMC writer version:

```bash
conda run -n OPENMC_ENVIRONMENT python scripts/generate_openmc_mgxs_fixture.py
```

Replace `OPENMC_ENVIRONMENT` with a separately managed environment that
provides OpenMC. It is used solely for fixture generation; Morana's importer
and normal development environment do not depend on OpenMC. Review the
generated HDF5 change, update the OpenMC version recorded in the generator and
REUSE provenance when appropriate, and run `reuse lint`.

## Maintaining the documentation toolchain

The tasks in this section are for maintainers changing documentation
dependencies or vendored rendering assets. Routine prose and docstring changes
do not require refreshing the toolchain.

The active documentation toolchain is deliberately constrained to MkDocs 1.x
(`>=1.6,<2`) and Material for MkDocs 9.x. Do not upgrade to MkDocs 2 without a
separately verified migration: the source-derived reference depends on the
MkDocs plugin interface through `mkdocstrings-python`.

Mathematics uses `pymdownx.arithmatex` and the pinned MathJax 4.1.3 component,
New Computer Modern font ranges, and `boldsymbol` extension under
`docs/assets/mathjax/` (Apache-2.0; see its bundled license). Upgrade those
assets together. Do not replace them with a CDN reference: the complete
rendered site must retain equation support when opened through `file://`
without network access.

Refresh the selected component and font assets with an exact matching release:

```bash
bash scripts/refresh_mathjax.sh 4.1.3
```

The script downloads `mathjax` and `@mathjax/mathjax-newcm-font` from npm,
stages only the component, `boldsymbol` extension, license, and CommonHTML
font files used by this site, then synchronizes that selection into
`docs/assets/mathjax/`. When changing the version, update the pin in this
section and the MathJax comment in `REUSE.toml`, then run the applicable checks
on this page. The site retains semantic MathJax markup but disables the optional
speech and braille worker, which is not vendored.

### Documentation rendering smoke check

After building the site, run the offline browser check when changing the theme,
representative content structures, generated reference, equations, MathJax
configuration, or vendored frontend assets:

```bash
BROWSER=chromium bash scripts/check_documentation_rendering.sh
```

It opens the built home, quickstart, core-reference, and theory pages directly
from `file://` with host-name resolution disabled. The check verifies Material
JavaScript initialization, representative content and code-copy controls,
mkdocstrings signatures, and the project-status presentation. On the theory
page it additionally confirms that every arithmatex expression becomes a
MathJax container, rejects MathJax errors and literal `\boldsymbol` text, and
verifies retained TeX annotations plus semantic, keyboard-accessible MathJax
markup.

## Where documentation changes belong

Use these ownership rules when adding or revising user-facing information.
Verify technical claims against the relevant implementation, tests, and project
README. Documentation ownership determines where to explain that behavior; it
does not make another documentation page evidence for an implementation claim.

### Documentation ownership

- The modeling and solver workflow owns the implemented capability inventory,
  cross-object behavior, result conventions, and capability boundaries.
- The geometry, theory, and output pages own their named conventions; the
  examples page alone catalogs maintained runnable workflows.
- The OpenMC MGXS import guide owns its accepted external artifact, selection,
  conversion, warnings, units, and interoperability limits.
- The OpenMC–Morana comparison page owns its physical model, published
  numerical evidence, figures, interpretation, and references. The
  [OpenMC comparison workflow](openmc_comparison_workflow.md) owns reproduction
  commands and artifact handling; the example README links to both pages.
- The verification page owns test-coverage claims and the distinction between
  verification and experimental validation.
- The licenses page owns notices for assets distributed with the generated
  site; `REUSE.toml` owns repository file annotations.
- The changelog owns dated, user-relevant release notes; it does not duplicate
  the modeling and solver workflow's capability inventory or record routine
  documentation, metadata, repository-process, CI, or maintainer-tooling
  changes.
- The README and documentation home provide concise summaries and route
  readers to these owners without duplicating detailed capability contracts.
- `docs/reference/` is rendered from explicit `__all__` declarations, type
  annotations, and NumPy-style public docstrings using `mkdocstrings-python`.
- Generated HTML is never edited directly.

### Writing and navigation

Use sentence case for page and section headings: capitalize the first word and
retain capitals only for proper nouns and established acronyms such as Morana,
OpenMC, API, VTK, and MathJax.

Use lowercase snake-case filenames that describe the rendered page purpose.
Every tracked Markdown page under `docs/` must appear in `mkdocs.yml`
navigation. Prefer relative Markdown links between sources; MkDocs produces
the flat `.html` targets.

Link the first explanatory mention of a public Python object in each section
to its exact anchor in `docs/reference/` when readers may need its constructor,
attributes, or methods. Do not mechanically link repeated mentions or code
examples. Keep cross-object behavior in the modeling and solver workflow, and
link to the owning geometry, theory, output, or verification page when that
context matters more than the object interface.

Document externally defined theory and interoperability conventions with
open-access technical references or official project documentation where
available. State clearly which conclusions come from those sources and which
are Morana-specific numerical or API conventions. Reproducible authored
figures should keep their generator scripts under `scripts/` and document the
regeneration command on this page.

### Citations and references

Use a direct inline link for official project documentation or an authoritative
online reference work when it supports a specific API, file-format,
interoperability, definition, or formula claim. Link to the relevant stable
page or section, not a search result or a project home page.

For a technical paper, report, textbook, or other scholarly source, use an
author--year citation at the claim and link it to a page-local `## References`
section. Give each cited work a stable anchor and a complete bibliographic
record, including authors, title, venue or report number, publisher or
institution, and year as applicable. Include a link to open full text; a DOI or
publisher record may accompany it but is not a substitute for open access.
Name the supporting figure, section, or page when that specificity matters.
Pages that cite at least one technical source must include this bibliography;
pages that only link to official project documentation need not add one.

See [verification and comparisons](verification.md) for numerical and
file-format test coverage.
