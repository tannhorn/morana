# OpenMC MGXS import

Morana can construct one immutable
[`CrossSections`](reference/core.md#morana.CrossSections) value from a selected
macroscopic material record in an OpenMC runtime multigroup cross-section
library. The supported input is the HDF5 file written by
[`openmc.MGXSLibrary.export_to_hdf5(...)`](https://docs.openmc.org/en/stable/pythonapi/generated/openmc.MGXSLibrary.html#openmc.MGXSLibrary.export_to_hdf5),
with `filetype="mgxs"` and format version 1.0. Morana reads the file directly
through `h5py`; OpenMC is not a Morana runtime dependency.

The adapter reads only macroscopic material data. Geometry, material
assignments, sources, boundary conditions, and tally post-processing remain the
responsibility of the surrounding application. In particular,
[`openmc.mgxs.Library.build_hdf5_store(...)`](https://docs.openmc.org/en/stable/pythonapi/generated/openmc.mgxs.Library.html#openmc.mgxs.Library.build_hdf5_store)
writes a different HDF5 hierarchy and is not accepted.

## Create a runtime library

Library creation is an OpenMC-side step. For example, this synthetic
two-group record uses the scalar-flux Legendre representation accepted by
Morana:

```python
import numpy as np
import openmc

groups = openmc.mgxs.EnergyGroups([0.0, 1.0e6, 2.0e7])
fuel = openmc.XSdata("fuel", groups, temperatures=[600.0])
fuel.order = 1
fuel.set_total(np.array([0.30, 0.80]), temperature=600.0)
fuel.set_absorption(np.array([0.01, 0.06]), temperature=600.0)
fuel.set_scatter_matrix(
    np.array(
        [
            [[0.15, 0.03], [0.08, 0.01]],
            [[0.00, 0.00], [0.55, 0.11]],
        ]
    ),
    temperature=600.0,
)
fuel.set_multiplicity_matrix(
    np.array([[1.0, 1.2], [0.0, 1.0]]), temperature=600.0
)
fuel.set_nu_fission(np.array([0.02, 0.10]), temperature=600.0)
fuel.set_chi(np.array([1.0, 0.0]), temperature=600.0)
fuel.set_kappa_fission(np.array([1.0e6, 8.0e6]), temperature=600.0)

library = openmc.MGXSLibrary(groups)
library.add_xsdatas([fuel])
library.export_to_hdf5("mgxs.h5")
```

The `XSdata` name `"fuel"` becomes an HDF5 group directly below the file root,
selected through Morana's `dataset` argument. The value `600.0` is stored as
one of that record's temperatures. Those are
the values selected explicitly during import below. The invented cross
sections serve only to demonstrate the workflow. A tally-based OpenMC workflow
can construct the runtime library with
[`openmc.mgxs.Library.create_mg_library(...)`](https://docs.openmc.org/en/stable/pythonapi/generated/openmc.mgxs.Library.html#openmc.mgxs.Library.create_mg_library)
and export that `MGXSLibrary`. In either case, select macroscopic data when
converting tally results. Do not pass the separate HDF5 tally store produced by
`build_hdf5_store(...)` to Morana.

## Import one material record

Select the library record and its physical temperature explicitly, then choose
how Morana should derive the diffusion coefficient:

```python
from morana import CrossSections, Material

fuel_xs = CrossSections.from_openmc_mgxs_hdf5(
    path="mgxs.h5",
    dataset="fuel",
    temperature=600.0,
    diffusion="p1-outscatter",
)
fuel = Material("fuel", xs=fuel_xs, color="#d62728")
```

`dataset` is the exact name of one direct child record in the runtime library.
`temperature` is in K and must match exactly one stored temperature, apart
from floating-point representation error; nearest-record selection and
interpolation are outside the adapter's scope. The returned `CrossSections`
contains owned read-only arrays and enters the same material, configuration,
operator, and solver workflow as native Python cross-section data.

The returned value retains the selected fission representation but not the
source file or selection metadata. Record the path, dataset name, temperature,
and energy grid in the surrounding application when that provenance is needed.
A saved Morana result retains the imported numerical values; the OpenMC
selection remains application metadata.

The [OpenMC–Morana comparison](openmc_comparison.md) demonstrates the complete
path from tally-based runtime-library creation through explicit import and a
Morana solve.

## Choose the diffusion convention

The `diffusion` argument is required because the runtime format does not record
how its `total` vector was produced. In particular, the OpenMC
[`XSdata.set_total_mgxs(...)` API](https://docs.openmc.org/en/stable/pythonapi/generated/openmc.XSdata.html#openmc.XSdata.set_total_mgxs)
accepts either a `TotalXS` or a `TransportXS` for the same `XSdata` total cross
section, which is exported under the `total` dataset name.

| Value | Imported coefficient | Diffusion-specific requirement |
| --- | --- | --- |
| `"total"` | $D_g=1/(3\Sigma_{\mathrm{stored},g})$, using `total` exactly as stored | Finite positive `total` in every group |
| `"p1-outscatter"` | $D_g=1/(3\Sigma_{tr,g})$, where $\Sigma_{tr,g}=\Sigma_{t,g}-\sum_h\Sigma_{s1,g\to h}$ | An uncorrected finite positive `TotalXS` in `total`, plus a P1 scattering moment producing positive $\Sigma_{tr}$ |

With `"total"`, a stored `TotalXS` produces the uncorrected coefficient while a
stored `TransportXS` produces the transport-corrected coefficient already
calculated by OpenMC. Morana cannot distinguish those provenances and does not
retain them in `CrossSections`.

The `"p1-outscatter"` convention fixes the incident group and sums its outgoing
ordinary-event P1 scattering row; scattering multiplicity does not enter the
correction. This is the group-discrete form of the outscatter approximation
described by [Ványi et al. (2021), §2.1](theory_references.md#vanyi-et-al-2021).
Its stored `total` must be an uncorrected `TotalXS`. Selecting
`"p1-outscatter"` when `total` was populated from `TransportXS` would apply a
second transport correction and is invalid. The runtime file does not contain
enough metadata for Morana to detect that misuse. The
[theory guide](theory_references.md#imported-diffusion-coefficients) places both
conversions in Morana's multigroup convention.

## Supported record data

Morana accepts scalar-flux `representation="isotropic"` records with
Legendre scattering and the OpenMC writer's compact
`scatter_shape="[G][G'][Order]"` layout. It validates the one-based
`g_min`/`g_max` bands, expands them to complete incoming-to-outgoing arrays,
requires finite nonnegative P0 scattering entries and finite signed higher
Legendre moments, and imports group data in the shared fast-to-thermal order.
The
[`openmc.XSdata` reference](https://docs.openmc.org/en/stable/pythonapi/generated/openmc.XSdata.html)
defines the OpenMC fields and their allowed shapes.

| Runtime-library data | Morana value |
| --- | --- |
| P0 `scatter_matrix` | `sigma_s[g_from, g_to]` ordinary scattering-event data |
| Optional `multiplicity_matrix` | `multiplicity_matrix[g_from, g_to]`; absence uses Morana's compact unit-multiplicity representation |
| Vector `nu-fission` plus `chi` | `FissionData(SeparableFission(...))` |
| Matrix `nu-fission` without `chi` | `FissionData(FissionTransfer(...))` |
| Optional `kappa-fission` on a valid fissionable record | `FissionData.kappa_sigma_f` in eV / cm |
| `fissionable=false` with no nonzero fission data | `fission=None` |

Fission-production arrays must be finite, nonnegative, and contain at least
one nonzero value. A separable `chi` must also be finite and nonnegative, have
a positive sum, and lie within `SeparableFission`'s normalization tolerance of
one; Morana normalizes an accepted spectrum before storing it. Optional
`kappa-fission` values must be finite and nonnegative.

OpenMC describes scattering multiplicity as the ratio of scattering-neutron
production to ordinary scattering events. Morana imports the ordinary event
matrix and the separate multiplicity directly. Multiplicity entries are
finite and nonnegative. A zero entry means that the corresponding ordinary
event contributes no emitted neutron to that destination; same-group values
need not equal one. The complete assembled loss diagonal must nevertheless
remain positive, as described under
[scattering and operator assembly](theory_references.md#scattering-fission-and-operator-split).

The selected record must be macroscopic. Morana rejects a nuclide-like entry
that carries `atomic_weight_ratio`, but the file format cannot prove units for
every hand-authored record. The caller remains responsible for ensuring that
`total`, `absorption`, scattering, and fission-production data are macroscopic
in `1 / cm`, and that `kappa-fission`, when present, is macroscopic in
`eV / cm`.

## Warnings and rejection boundaries

Morana warns when an otherwise supported record contains data it cannot retain:
Legendre moments above P0 for `"total"` or above P1 for `"p1-outscatter"`,
reaction-rate `fission`, inverse velocity,
prompt/delayed-neutron fields, or an unrecognized selected-temperature field.
Review these warnings because discarded data may matter to the surrounding
analysis even though they do not prevent this steady-state diffusion import.

The importer rejects:

- a different HDF5 file identity or runtime-library version;
- missing, ambiguous, or nuclide-like material and temperature selections;
- invalid energy grids, required vectors, scattering bands, or diffusion data;
- angle-dependent, tabular, histogram, or unsupported scattering layouts;
- inconsistent `fissionable` metadata, zero or incomplete neutron production,
  and matrix `nu-fission` mixed with an independent `chi`; and
- nonfissionable records containing any nonzero fission-related field.

OpenMC geometry and layout import, microscopic-to-macroscopic conversion,
temperature interpolation, delayed-neutron physics, and reflector-response or
albedo calibration are outside this adapter's scope. Continue with the
[modeling and solver workflow](modeling_workflow.md) to place imported cross
sections in a Morana problem.
