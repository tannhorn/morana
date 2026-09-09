# Result archives

`Result.save_to_disk("case.morana-result")` writes a portable result archive;
`Result.load_from_disk(...)` restores it as a new, checked `Result`. The format
retains a completed calculation for later inspection, plotting, export, or
provenance review.

The conventional suffix is `.morana-result`, but save and load use the exact
path supplied by the caller. Thus `result.save_to_disk("case")` writes an
archive named exactly `case`; the manifest, rather than the filename,
identifies the format. The maintained
[`result_archive.py`](examples.md#result-archive) example demonstrates a
round trip.

## Contents and compatibility

An archive is a standard
[DEFLATE-compressed ZIP container](https://docs.python.org/3/library/zipfile.html)
containing an explicit UTF-8 JSON manifest and named
[NumPy `.npy` payloads](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html).
It records flux and typed balance diagnostics, configuration provenance,
tagged mode-specific solve settings, and the typed execution report. A
criticality archive always records one physical normalization: either a
fission-source rate or recoverable power. Fixed-source archives have no
normalization. Configuration provenance retains whether each material uses the
compact `null` or explicit-array scattering-multiplicity representation, along
with its fission data. Fission data are `null` for a nonfissile material;
otherwise their `neutron_production` object is explicitly tagged `separable`
or `transfer`. A separable record retains `nu_sigma_f`, `chi`, and its
normalization tolerance, while a transfer record retains only the canonical
event-oriented `fission_transfer[g_from, g_to]` array. Thus an archive restores
the representation the caller selected instead of converting it to a flattened
or alternate form. `solve_mode` is derived from the retained values when the
result is reconstructed; it is not stored separately.

Archives use pre-release schema version 8, which has no backward-compatibility
guarantee. Readers accept only this representation and reject every other
schema version or malformed tagged record.

## Integrity and consistency checks

Payloads are loaded with `allow_pickle=False`. The manifest records every
payload's name, shape, dtype, and SHA-256 checksum. The loader checks the ZIP
member list before reconstruction, verifies payloads as they are read, and
rejects unreferenced payloads before returning the result. Duplicate,
encrypted, missing, unexpected, and checksum-mismatched members are rejected,
as are members with unsafe paths.

Stored flux must be finite and nonnegative, and its energy-group count must
match the active materials in the retained configuration. Criticality reports
must satisfy their recorded outer-iteration limit and final convergence
tolerances, in addition to the linear residual checks.

When completing or loading a criticality result, Morana recomputes the declared
physical normalization from the retained flux and configuration snapshot. A
fission-source target must agree with the integrated fission-production response
within a relative tolerance of `1e-10` (with zero absolute tolerance). A
power target must meet the same tolerance against the integrated
`kappa_sigma_f * flux` response after conversion from eV/s to W, and
every fissionable active material must retain `kappa_sigma_f`. A mismatch is
rejected. Loading checks the stored result's structure and normalization; it
does not rerun the solve or independently recompute every stored balance or
execution-report value.

The restored result owns fresh read-only arrays and a configuration snapshot.
It can therefore be plotted or exported in the same way as an in-memory solve;
see [inspection and output](outputs.md) for those artifacts and
[modeling and solver workflow](modeling_workflow.md) for result interpretation.
