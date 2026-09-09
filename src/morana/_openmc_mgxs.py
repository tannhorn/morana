"""Private readers for OpenMC runtime multigroup cross-section libraries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import h5py
import numpy as np

from morana._arrays import (
    require_finite_array,
    require_finite_nonnegative_array,
    require_finite_positive_array,
    require_nonzero_array,
)
from morana._validation import (
    require_finite_positive_real,
    require_human_readable_identifier,
    require_nonempty_string,
)

_BOLTZMANN_CONSTANT_EV_PER_KELVIN = 8.617333262e-5
_RUNTIME_FILETYPE = "mgxs"
_RUNTIME_FORMAT_VERSION = (1, 0)
_TEMPERATURE_MATCH_ULPS = 4
_TOTAL_DIFFUSION = "total"
_P1_OUTSCATTER_DIFFUSION = "p1-outscatter"
_DIFFUSION_CONVENTIONS = frozenset({_TOTAL_DIFFUSION, _P1_OUTSCATTER_DIFFUSION})
_FISSION_DATASET_NAMES = (
    "fission",
    "nu-fission",
    "chi",
    "kappa-fission",
    "prompt-nu-fission",
    "delayed-nu-fission",
    "chi-prompt",
    "chi-delayed",
    "beta",
    "decay-rate",
)
_DISCARDED_DATASET_MESSAGES = {
    "fission": "Morana uses nu-fission for neutron production",
    "inverse-velocity": "time-dependent transport data are outside Morana's scope",
    "prompt-nu-fission": "Morana imports total nu-fission only",
    "delayed-nu-fission": "delayed-neutron data are outside Morana's scope",
    "chi-prompt": "Morana imports total chi only",
    "chi-delayed": "delayed-neutron data are outside Morana's scope",
    "beta": "delayed-neutron data are outside Morana's scope",
    "decay-rate": "delayed-neutron data are outside Morana's scope",
}
_RECOGNIZED_TEMPERATURE_DATASET_NAMES = frozenset(
    {
        "absorption",
        "total",
        "scatter_data",
        "nu-fission",
        "chi",
        "kappa-fission",
        *_DISCARDED_DATASET_MESSAGES,
    }
)


@dataclass(frozen=True)
class _OpenMCMGXSSelection:
    """Checked group count and label for one runtime-library temperature record."""

    groups: int
    temperature_label: str


@dataclass(frozen=True)
class _OpenMCMGXSImport:
    """Owned Morana arrays decoded from one runtime-MGXS temperature record."""

    coefficients: np.ndarray
    absorption: np.ndarray
    scatter: np.ndarray
    multiplicity: np.ndarray | None
    fission: _OpenMCFissionImport | None


@dataclass(frozen=True)
class _OpenMCFissionImport:
    """Owned fission arrays decoded from one runtime-MGXS temperature record."""

    nu_sigma_f: np.ndarray | None
    chi: np.ndarray | None
    fission_transfer: np.ndarray | None
    kappa_sigma_f: np.ndarray | None


def read_openmc_runtime_mgxs(
    path: str | Path,
    dataset: str,
    temperature: float,
    diffusion: str,
) -> _OpenMCMGXSImport:
    """Read one runtime-MGXS record into owned Morana-compatible arrays.

    The reader validates selection and decodes physical data under one open
    archive handle, then returns owned arrays without HDF5 handles or import
    provenance. Nonfissionable records return ``None`` fission arrays;
    fissionable records retain either separable or transfer production data.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or pathlib.Path")
    dataset = require_human_readable_identifier("OpenMC MGXS dataset", dataset)
    temperature = require_finite_positive_real("temperature", temperature)
    diffusion = require_nonempty_string("diffusion", diffusion)
    if diffusion not in _DIFFUSION_CONVENTIONS:
        raise ValueError(
            "diffusion must be " f"{_TOTAL_DIFFUSION!r} or {_P1_OUTSCATTER_DIFFUSION!r}"
        )

    try:
        with h5py.File(path, "r") as library:
            selection, record, temperature_record = (
                _select_openmc_runtime_mgxs_from_library(library, dataset, temperature)
            )
            dataset_location = f"dataset {dataset!r}"
            temperature_location = (
                f"{dataset_location} temperature {selection.temperature_label!r}"
            )
            fission = _read_fission_data(
                record,
                temperature_record,
                selection.groups,
                dataset_location,
                temperature_location,
            )
            _require_scalar_flux_legendre_record(record, dataset_location)
            absorption = _required_real_vector(
                temperature_record,
                "absorption",
                selection.groups,
                temperature_location,
            )
            total = _required_real_vector(
                temperature_record,
                "total",
                selection.groups,
                temperature_location,
            )
            require_finite_nonnegative_array("OpenMC MGXS absorption", absorption)
            require_finite_positive_array("OpenMC MGXS total", total)
            moments, multiplicity = _read_scattering_data(
                record,
                temperature_record,
                selection.groups,
                diffusion,
                dataset_location=dataset_location,
                temperature_location=temperature_location,
            )
            temperature_dataset_names = tuple(temperature_record)
            discarded_dataset_names = tuple(
                name
                for name in _DISCARDED_DATASET_MESSAGES
                if name in temperature_dataset_names
            )
            unrecognized_dataset_names = tuple(
                name
                for name in temperature_dataset_names
                if name not in _RECOGNIZED_TEMPERATURE_DATASET_NAMES
            )
    except OSError as error:
        raise ValueError(
            "path is not a readable OpenMC runtime-MGXS HDF5 file"
        ) from error

    scatter = moments[:, :, 0]
    transport = total
    if diffusion == _P1_OUTSCATTER_DIFFUSION:
        transport = total - np.sum(moments[:, :, 1], axis=1)
        require_finite_positive_array(
            "OpenMC MGXS P1 outscatter transport cross section", transport
        )
    with np.errstate(over="ignore", divide="ignore"):
        coefficients = (1.0 / 3.0) / transport
    require_finite_positive_array("OpenMC MGXS diffusion coefficients", coefficients)
    _warn_discarded_scattering_moments(moments, diffusion)
    _warn_discarded_temperature_data(discarded_dataset_names)
    _warn_unrecognized_temperature_data(unrecognized_dataset_names)
    return _OpenMCMGXSImport(
        coefficients=coefficients,
        absorption=absorption,
        scatter=scatter,
        multiplicity=multiplicity,
        fission=fission,
    )


def _select_openmc_runtime_mgxs_from_library(
    library: h5py.File, dataset: str, temperature: float
) -> tuple[_OpenMCMGXSSelection, h5py.Group, h5py.Group]:
    """Validate and select one runtime-MGXS record from an already-open library.

    The returned HDF5 groups are valid only while the caller keeps ``library``
    open. They are an internal decoding convenience and must not escape into
    Morana calculation values.
    """
    _check_runtime_library_identity(library)
    groups = _checked_runtime_library_group_count(library)
    record = _select_dataset(library, dataset)
    _reject_nuclide_record(record, dataset)
    temperature_label = _select_temperature_label(record, dataset, temperature)
    temperature_record = _required_temperature_group(record, temperature_label, dataset)
    return (
        _OpenMCMGXSSelection(groups=groups, temperature_label=temperature_label),
        record,
        temperature_record,
    )


def _read_fission_data(
    record: h5py.Group,
    temperature_record: h5py.Group,
    groups: int,
    dataset_location: str,
    temperature_location: str,
) -> _OpenMCFissionImport | None:
    """Decode nonfissionable, separable, or transfer fission production."""
    fissionable = _required_boolean_attribute(record, "fissionable", dataset_location)
    if not fissionable:
        for name in _FISSION_DATASET_NAMES:
            if name in temperature_record:
                values = _required_real_dataset(
                    temperature_record, name, temperature_location
                )
                if np.any(values != 0.0):
                    raise ValueError(
                        f"nonfissionable OpenMC MGXS {temperature_location} "
                        f"has nonzero {name!r}"
                    )
        return None

    nu_fission = _required_real_dataset(
        temperature_record, "nu-fission", temperature_location
    )
    nu_sigma_f = None
    chi = None
    fission_transfer = None
    if nu_fission.shape == (groups,):
        nu_sigma_f = nu_fission
        chi = _required_real_vector(
            temperature_record, "chi", groups, temperature_location
        )
        require_finite_nonnegative_array("OpenMC MGXS nu-fission", nu_sigma_f)
        require_finite_nonnegative_array("OpenMC MGXS chi", chi)
        require_nonzero_array("OpenMC MGXS nu-fission", nu_sigma_f)
        if float(np.sum(chi)) <= 0.0:
            raise ValueError(
                f"fissionable OpenMC MGXS {temperature_location} "
                "must have a positive chi sum"
            )
    elif nu_fission.shape == (groups, groups):
        if "chi" in temperature_record:
            raise ValueError(
                f"fissionable OpenMC MGXS {temperature_location} has mixed "
                "matrix nu-fission and chi production data"
            )
        fission_transfer = nu_fission
        require_finite_nonnegative_array(
            "OpenMC MGXS nu-fission transfer", fission_transfer
        )
        require_nonzero_array("OpenMC MGXS nu-fission transfer", fission_transfer)
    else:
        raise ValueError(
            f"OpenMC MGXS {temperature_location} 'nu-fission' data must have "
            f"shape ({groups},) or ({groups}, {groups})"
        )
    kappa_sigma_f = None
    if "kappa-fission" in temperature_record:
        kappa_sigma_f = _required_real_vector(
            temperature_record, "kappa-fission", groups, temperature_location
        )
        require_finite_nonnegative_array("OpenMC MGXS kappa-fission", kappa_sigma_f)
    return _OpenMCFissionImport(
        nu_sigma_f=nu_sigma_f,
        chi=chi,
        fission_transfer=fission_transfer,
        kappa_sigma_f=kappa_sigma_f,
    )


def _warn_discarded_temperature_data(dataset_names: tuple[str, ...]) -> None:
    """Warn after validation for optional datasets outside Morana's scope."""
    for name in dataset_names:
        warnings.warn(
            f"discarding OpenMC MGXS {name} data; "
            f"{_DISCARDED_DATASET_MESSAGES[name]}",
            UserWarning,
            stacklevel=3,
        )


def _warn_unrecognized_temperature_data(dataset_names: tuple[str, ...]) -> None:
    """Warn after validation for selected-record fields Morana does not know."""
    for name in dataset_names:
        warnings.warn(
            f"discarding unrecognized OpenMC MGXS temperature data {name!r}",
            UserWarning,
            stacklevel=3,
        )


def _require_scalar_flux_legendre_record(
    record: h5py.Group, dataset_location: str
) -> None:
    """Require the scalar-flux Legendre scattering representation."""
    representation = _required_text_attribute(
        record, "representation", dataset_location
    )
    if representation != "isotropic":
        raise ValueError("OpenMC MGXS representation must be 'isotropic'")
    scatter_format = _required_text_attribute(
        record, "scatter_format", dataset_location
    )
    if scatter_format != "legendre":
        raise ValueError("OpenMC MGXS scatter_format must be 'legendre'")


def _read_scattering_data(
    record: h5py.Group,
    temperature_record: h5py.Group,
    groups: int,
    diffusion: str,
    *,
    dataset_location: str,
    temperature_location: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read supported compact scattering and multiplicity data into Morana arrays.

    The supported ``[G][G'][Order]`` layout stores one-based inclusive outgoing
    bands per incoming group, with contiguous Legendre moments for each stored
    transfer. The returned transient array uses Morana's
    ``[g_from, g_to, moment]`` convention. Its P0 slice is retained for the
    imported material; the caller uses P1 only for the selected transport
    correction and discards all higher moments after warning.
    """
    order = _required_nonnegative_integer_attribute(record, "order", dataset_location)
    needed_order = 1 if diffusion == _P1_OUTSCATTER_DIFFUSION else 0
    if order < needed_order:
        raise ValueError(
            "OpenMC MGXS scattering order must include "
            f"P{needed_order} for {diffusion!r} diffusion"
        )
    shape = _required_text_attribute(record, "scatter_shape", dataset_location)
    if shape != "[G][G'][Order]":
        raise ValueError(
            f"unsupported or ambiguous OpenMC MGXS scatter_shape {shape!r}"
        )
    scatter_data = temperature_record.get("scatter_data", getlink=False)
    if not isinstance(scatter_data, h5py.Group):
        raise ValueError(
            f"OpenMC MGXS {temperature_location} is missing 'scatter_data'"
        )
    scatter_location = f"{temperature_location} scatter_data"
    g_min = _required_integer_vector(scatter_data, "g_min", groups, scatter_location)
    g_max = _required_integer_vector(scatter_data, "g_max", groups, scatter_location)
    bands = _checked_scatter_bands(g_min, g_max, groups)
    moments = _expand_compact_scatter(
        _required_real_dataset(scatter_data, "scatter_matrix", scatter_location),
        bands,
        groups,
        order,
    )
    multiplicity = None
    if "multiplicity_matrix" in scatter_data:
        multiplicity = _expand_compact_scalar(
            _required_real_dataset(
                scatter_data, "multiplicity_matrix", scatter_location
            ),
            bands,
            groups,
        )
        require_finite_nonnegative_array(
            "OpenMC MGXS multiplicity_matrix", multiplicity
        )
    return moments, multiplicity


def _warn_discarded_scattering_moments(moments: np.ndarray, diffusion: str) -> None:
    """Warn after validation when imported Legendre moments exceed the need."""
    needed_order = 1 if diffusion == _P1_OUTSCATTER_DIFFUSION else 0
    if moments.shape[2] - 1 > needed_order:
        warnings.warn(
            "discarding OpenMC MGXS Legendre scattering moments above "
            "the selected diffusion convention",
            UserWarning,
            stacklevel=3,
        )


def _checked_scatter_bands(
    g_min: np.ndarray, g_max: np.ndarray, groups: int
) -> tuple[tuple[int, int], ...]:
    """Validate OpenMC's one-based inclusive outgoing scattering bands."""
    bands = []
    for minimum, maximum in zip(g_min, g_max, strict=True):
        if minimum < 1 or maximum < minimum or maximum > groups:
            raise ValueError("OpenMC MGXS scatter_data g_min/g_max bands are invalid")
        bands.append((int(minimum) - 1, int(maximum)))
    return tuple(bands)


def _expand_compact_scatter(
    values: np.ndarray,
    bands: tuple[tuple[int, int], ...],
    groups: int,
    order: int,
) -> np.ndarray:
    """Expand supported compact values to ``[g_from, g_to, moment]`` order."""
    expected = sum(stop - start for start, stop in bands) * (order + 1)
    if values.shape != (expected,):
        raise ValueError(
            "OpenMC MGXS scatter_matrix shape conflicts with scatter bands"
        )
    require_finite_array("OpenMC MGXS scatter_matrix", values)
    expanded = np.zeros((groups, groups, order + 1), dtype=float)
    index = 0
    for incoming, (start, stop) in enumerate(bands):
        width = stop - start
        compact = values[index : index + width * (order + 1)]
        expanded[incoming, start:stop, :] = compact.reshape(width, order + 1)
        index += width * (order + 1)
    require_finite_nonnegative_array("OpenMC MGXS P0 scatter_matrix", expanded[:, :, 0])
    return expanded


def _expand_compact_scalar(
    values: np.ndarray, bands: tuple[tuple[int, int], ...], groups: int
) -> np.ndarray:
    """Expand compact band values to Morana's complete transfer matrix."""
    expected = sum(stop - start for start, stop in bands)
    if values.shape != (expected,):
        raise ValueError(
            "OpenMC MGXS multiplicity_matrix shape conflicts with scatter bands"
        )
    expanded = np.zeros((groups, groups), dtype=float)
    index = 0
    for incoming, (start, stop) in enumerate(bands):
        width = stop - start
        expanded[incoming, start:stop] = values[index : index + width]
        index += width
    return expanded


def _checked_runtime_library_group_count(library: h5py.File) -> int:
    """Return the checked group count after validating the energy grid."""
    groups = _required_positive_integer_attribute(library, "energy_groups", "root")
    group_structure = _required_numeric_attribute(library, "group structure", "root")
    if group_structure.shape != (groups + 1,):
        raise ValueError(
            "OpenMC MGXS root group structure must have energy_groups + 1 values"
        )
    require_finite_nonnegative_array(
        "OpenMC MGXS root group structure", group_structure
    )
    if not np.all(np.diff(group_structure) > 0.0):
        raise ValueError("OpenMC MGXS root group structure must be strictly increasing")
    return groups


def _check_runtime_library_identity(library: h5py.File) -> None:
    """Require the supported OpenMC runtime-library file identity."""
    filetype = _required_text_attribute(library, "filetype", "root")
    if filetype != _RUNTIME_FILETYPE:
        raise ValueError("OpenMC MGXS filetype must be 'mgxs'")

    version = np.asarray(_required_attribute(library, "version", "root"))
    if version.shape != (2,) or not np.array_equal(version, _RUNTIME_FORMAT_VERSION):
        raise ValueError("OpenMC MGXS format version must be 1.0")


def _select_dataset(library: h5py.File, dataset: str) -> h5py.Group:
    """Return one explicitly selected direct child dataset record."""
    record = library.get(dataset, getlink=False)
    if not isinstance(record, h5py.Group) or record.parent != library:
        raise ValueError(f"OpenMC MGXS dataset {dataset!r} is not present")
    return record


def _reject_nuclide_record(record: h5py.Group, dataset: str) -> None:
    """Reject a selected nuclide-like record before any unit conversion."""
    if "atomic_weight_ratio" in record.attrs:
        raise ValueError(
            f"OpenMC MGXS dataset {dataset!r} is nuclide-like, not macroscopic"
        )


def _select_temperature_label(
    record: h5py.Group,
    dataset: str,
    temperature: float,
) -> str:
    """Return the uniquely stored temperature matching Kelvin input."""
    temperatures = record.get("kTs", getlink=False)
    if not isinstance(temperatures, h5py.Group):
        raise ValueError(f"OpenMC MGXS dataset {dataset!r} has no kTs temperature data")

    requested_kt = temperature * _BOLTZMANN_CONSTANT_EV_PER_KELVIN
    matches = [
        label
        for label, value in temperatures.items()
        if _matches_requested_temperature(value, requested_kt)
    ]
    if not matches:
        raise ValueError(
            f"OpenMC MGXS dataset {dataset!r} has no {temperature:g} K record"
        )
    if len(matches) != 1:
        raise ValueError(
            f"OpenMC MGXS dataset {dataset!r} has ambiguous {temperature:g} K records"
        )

    label = matches[0]
    return label


def _required_temperature_group(
    record: h5py.Group, label: str, dataset: str
) -> h5py.Group:
    """Return the required direct-child data group for a selected temperature."""
    record_at_temperature = record.get(label, getlink=False)
    if not isinstance(record_at_temperature, h5py.Group):
        raise ValueError(
            f"OpenMC MGXS dataset {dataset!r} has no data group for {label!r}"
        )
    return record_at_temperature


def _matches_requested_temperature(
    value: h5py.Dataset | h5py.Group, requested_kt: float
) -> bool:
    """Return whether one scalar stored kT represents the requested temperature.

    The finite ULP-scale tolerance accounts only for floating-point
    representation of the same physical temperature. It does not select a
    nearby temperature or interpolate between stored temperature records.
    """
    if not isinstance(value, h5py.Dataset):
        return False
    array = np.asarray(value[()])
    if (
        array.shape != ()
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        return False
    stored_kt = float(array)
    if not np.isfinite(stored_kt):
        return False
    tolerance = _TEMPERATURE_MATCH_ULPS * max(
        abs(np.spacing(stored_kt)), abs(np.spacing(requested_kt))
    )
    return abs(stored_kt - requested_kt) <= tolerance


def _required_attribute(
    owner: h5py.File | h5py.Group,
    name: str,
    location: str,
) -> object:
    """Return one mandatory HDF5 attribute with a contextual error."""
    if name not in owner.attrs:
        raise ValueError(f"OpenMC MGXS {location} is missing {name!r} metadata")
    return owner.attrs[name]


def _required_text_attribute(
    owner: h5py.File | h5py.Group,
    name: str,
    location: str,
) -> str:
    """Return one required scalar UTF-8 attribute."""
    value = _required_attribute(owner, name, location)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"OpenMC MGXS {location} {name!r} metadata must be UTF-8 text"
            ) from error
    if isinstance(value, str):
        return value
    raise ValueError(f"OpenMC MGXS {location} {name!r} metadata must be text")


def _required_boolean_attribute(
    owner: h5py.File | h5py.Group,
    name: str,
    location: str,
) -> bool:
    """Return one mandatory Boolean HDF5 attribute."""
    value = _required_attribute(owner, name, location)
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"OpenMC MGXS {location} {name!r} metadata must be Boolean")
    return bool(value)


def _required_positive_integer_attribute(
    owner: h5py.File | h5py.Group,
    name: str,
    location: str,
) -> int:
    """Return one mandatory positive scalar integer HDF5 attribute."""
    value = np.asarray(_required_attribute(owner, name, location))
    if value.shape != () or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(
            f"OpenMC MGXS {location} {name!r} metadata must be a positive integer"
        )
    integer = int(value)
    if integer < 1:
        raise ValueError(
            f"OpenMC MGXS {location} {name!r} metadata must be a positive integer"
        )
    return integer


def _required_numeric_attribute(
    owner: h5py.File | h5py.Group,
    name: str,
    location: str,
) -> np.ndarray:
    """Return one mandatory real-valued HDF5 attribute as a float array."""
    value = np.asarray(_required_attribute(owner, name, location))
    if not np.issubdtype(value.dtype, np.number) or np.issubdtype(
        value.dtype, np.complexfloating
    ):
        raise ValueError(
            f"OpenMC MGXS {location} {name!r} metadata must be real-valued"
        )
    return np.asarray(value, dtype=float)


def _required_nonnegative_integer_attribute(
    owner: h5py.File | h5py.Group,
    name: str,
    location: str,
) -> int:
    """Return one mandatory nonnegative scalar integer HDF5 attribute."""
    value = np.asarray(_required_attribute(owner, name, location))
    if value.shape != () or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(
            f"OpenMC MGXS {location} {name!r} metadata must be a nonnegative integer"
        )
    integer = int(value)
    if integer < 0:
        raise ValueError(
            f"OpenMC MGXS {location} {name!r} metadata must be a nonnegative integer"
        )
    return integer


def _required_real_dataset(owner: h5py.Group, name: str, location: str) -> np.ndarray:
    """Read one mandatory real HDF5 dataset into an owned float array."""
    value = _required_dataset(owner, name, location)
    array = np.asarray(value[()])
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(f"OpenMC MGXS {location} {name!r} data must be real-valued")
    return np.array(array, dtype=float, copy=True)


def _required_dataset(owner: h5py.Group, name: str, location: str) -> h5py.Dataset:
    """Return one mandatory HDF5 dataset with a contextual error."""
    value = owner.get(name, getlink=False)
    if not isinstance(value, h5py.Dataset):
        raise ValueError(f"OpenMC MGXS {location} is missing {name!r} data")
    return value


def _required_real_vector(
    owner: h5py.Group, name: str, groups: int, location: str
) -> np.ndarray:
    """Read one mandatory group-indexed real vector."""
    values = _required_real_dataset(owner, name, location)
    if values.shape != (groups,):
        raise ValueError(
            f"OpenMC MGXS {location} {name!r} data must have shape ({groups},)"
        )
    return values


def _required_integer_vector(
    owner: h5py.Group, name: str, groups: int, location: str
) -> np.ndarray:
    """Read one mandatory group-indexed integer vector."""
    value = _required_dataset(owner, name, location)
    values = np.asarray(value[()])
    if values.shape != (groups,) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(
            "OpenMC MGXS "
            f"{location} {name!r} data must be an integer vector of length {groups}"
        )
    return np.array(values, dtype=int, copy=True)
