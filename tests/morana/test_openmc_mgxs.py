"""Tests for the private OpenMC runtime-MGXS reader boundary."""

from pathlib import Path
import shutil
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from morana._openmc_mgxs import (
    _OpenMCMGXSSelection,
    _required_text_attribute,
    _select_openmc_runtime_mgxs_from_library,
)
from morana.materials import CrossSections, FissionTransfer

_FIXTURE_PATH = Path(__file__).parent / "fixtures/openmc_runtime_mgxs_synthetic.h5"


def test_open_library_selection_validates_and_selects_fixture_record() -> None:
    """The reader should select one requested material-temperature record."""
    selection = _select_runtime_library(_FIXTURE_PATH, "synthetic_fuel", 600.0)

    assert selection.groups == 3
    assert selection.temperature_label == "600K"


def test_open_library_selection_tolerates_temperature_rounding(
    tmp_path: Path,
) -> None:
    """A few ULPs of stored kT rounding should retain the same temperature."""
    path = _write_minimal_runtime_library(tmp_path / "rounded_temperature.h5")
    with h5py.File(path, "r+") as library:
        stored_temperature = library["material/kTs/600K"]
        stored_temperature[()] = np.nextafter(stored_temperature[()], np.inf)

    selection = _select_runtime_library(path, "material", 600.0)

    assert selection.temperature_label == "600K"


@pytest.mark.parametrize(
    ("dataset", "temperature", "exception", "message"),
    [
        ("missing", 600.0, ValueError, "not present"),
        ("synthetic_fuel", 700.0, ValueError, "no 700 K record"),
        ("synthetic_fuel", 0.0, ValueError, "finite and positive"),
        ("synthetic_fuel", True, TypeError, "real numeric"),
        (None, 600.0, TypeError, "dataset"),
    ],
)
def test_cross_sections_checks_explicit_selection(
    dataset: object,
    temperature: object,
    exception: type[Exception],
    message: str,
) -> None:
    """The reader should not infer a material name or nearby temperature."""
    with pytest.raises(exception, match=message):
        CrossSections.from_openmc_mgxs_hdf5(  # type: ignore[arg-type]
            _FIXTURE_PATH, dataset, temperature, diffusion="total"
        )


def test_open_library_selection_rejects_non_runtime_filetype(
    tmp_path: Path,
) -> None:
    """Only the OpenMC runtime-MGXS HDF5 library identity is accepted."""
    path = _write_minimal_runtime_library(tmp_path / "wrong_filetype.h5")
    with h5py.File(path, "r+") as library:
        library.attrs["filetype"] = "not-mgxs"

    with pytest.raises(ValueError, match="filetype"):
        _select_runtime_library(path, "material", 600.0)


def test_open_library_selection_rejects_non_utf8_filetype() -> None:
    """Malformed byte attributes raise a contextual validation error."""
    with pytest.raises(ValueError, match="UTF-8"):
        _required_text_attribute(
            SimpleNamespace(attrs={"filetype": b"\xff"}),
            "filetype",
            "root",
        )


@pytest.mark.parametrize(
    ("version", "message"),
    [
        (np.array([2, 0]), "version"),
        (np.array([1]), "version"),
    ],
)
def test_open_library_selection_rejects_other_format_versions(
    tmp_path: Path, version: np.ndarray, message: str
) -> None:
    """The reader should reject unsupported runtime-library versions."""
    path = _write_minimal_runtime_library(tmp_path / "wrong_version.h5")
    with h5py.File(path, "r+") as library:
        library.attrs["version"] = version

    with pytest.raises(ValueError, match=message):
        _select_runtime_library(path, "material", 600.0)


@pytest.mark.parametrize(
    ("group_structure", "message"),
    [
        (np.array([0.0, 1.0]), r"energy_groups \+ 1"),
        (np.array([0.0, 2.0, 1.0]), "strictly increasing"),
        (np.array([0.0, 1.0, np.inf]), "finite"),
        (np.array([-1.0, 1.0, 2.0]), "non-negative"),
        (np.array([0.0j, 1.0j, 2.0j]), "real-valued"),
    ],
)
def test_open_library_selection_validates_energy_grid(
    tmp_path: Path, group_structure: np.ndarray, message: str
) -> None:
    """The reader should validate the root energy grid before decoding data."""
    path = _write_minimal_runtime_library(tmp_path / "bad_grid.h5")
    with h5py.File(path, "r+") as library:
        library.attrs["group structure"] = group_structure

    with pytest.raises(ValueError, match=message):
        _select_runtime_library(path, "material", 600.0)


def test_open_library_selection_rejects_nuclide_like_record(tmp_path: Path) -> None:
    """Nuclide records are not silently converted to macroscopic data."""
    path = _write_minimal_runtime_library(tmp_path / "nuclide.h5")
    with h5py.File(path, "r+") as library:
        library["material"].attrs["atomic_weight_ratio"] = 235.0

    with pytest.raises(ValueError, match="nuclide-like"):
        _select_runtime_library(path, "material", 600.0)


def test_cross_sections_imports_nonfissionable_fixture_with_unit_multiplicity() -> None:
    """The moderator path expands P0 bands and retains no fission data."""
    with pytest.warns(UserWarning, match="discarding"):
        cross_sections = CrossSections.from_openmc_mgxs_hdf5(
            _FIXTURE_PATH, "synthetic_moderator", 600.0, diffusion="total"
        )

    np.testing.assert_allclose(
        cross_sections.D, 1.0 / (3.0 * np.array([0.41, 0.63, 1.09]))
    )
    np.testing.assert_allclose(
        cross_sections.sigma_a, np.array([0.0004, 0.0012, 0.0075])
    )
    np.testing.assert_allclose(
        cross_sections.sigma_s,
        np.array([[0.27, 0.10, 0.01], [0.0, 0.44, 0.13], [0.0, 0.0, 0.86]]),
    )
    assert cross_sections.fission is None
    assert cross_sections.multiplicity_matrix is None
    assert not cross_sections.D.flags.writeable
    assert not cross_sections.sigma_s.flags.writeable


def test_cross_sections_imports_p1_outscatter_moderator() -> None:
    """P1 outscatter correction sums each incident group's outgoing row."""
    with pytest.warns(UserWarning, match="discarding"):
        cross_sections = CrossSections.from_openmc_mgxs_hdf5(
            _FIXTURE_PATH,
            "synthetic_moderator",
            600.0,
            diffusion="p1-outscatter",
        )

    p1 = np.array([[0.054, 0.020, 0.002], [0.0, 0.088, 0.026], [0.0, 0.0, 0.172]])
    np.testing.assert_allclose(
        cross_sections.D, 1.0 / (3.0 * (np.array([0.41, 0.63, 1.09]) - p1.sum(axis=1)))
    )


def test_cross_sections_accepts_signed_higher_scattering_moments(
    tmp_path: Path,
) -> None:
    """P1 Legendre moments may be finite and signed while P0 remains physical."""
    path = tmp_path / "signed_p1.h5"
    shutil.copy(_FIXTURE_PATH, path)
    with h5py.File(path, "r+") as library:
        matrix = library["synthetic_moderator/600K/scatter_data/scatter_matrix"]
        values = matrix[()]
        values[1] = -0.054
        matrix[...] = values

    cross_sections = CrossSections.from_openmc_mgxs_hdf5(
        path, "synthetic_moderator", 600.0, diffusion="p1-outscatter"
    )

    np.testing.assert_allclose(cross_sections.D[0], 1.0 / (3.0 * (0.41 + 0.032)))


def test_cross_sections_rejects_negative_p0_scattering(tmp_path: Path) -> None:
    """The ordinary P0 scattering-event matrix must remain nonnegative."""
    path = tmp_path / "negative_p0.h5"
    shutil.copy(_FIXTURE_PATH, path)
    with h5py.File(path, "r+") as library:
        matrix = library["synthetic_moderator/600K/scatter_data/scatter_matrix"]
        values = matrix[()]
        values[0] = -0.27
        matrix[...] = values

    with pytest.raises(ValueError, match="P0 scatter_matrix.*non-negative"):
        CrossSections.from_openmc_mgxs_hdf5(
            path, "synthetic_moderator", 600.0, diffusion="total"
        )


def test_cross_sections_imports_separable_fissionable_fixture() -> None:
    """Fuel records retain checked separable neutron-production data."""
    with pytest.warns(UserWarning, match="discarding"):
        cross_sections = CrossSections.from_openmc_mgxs_hdf5(
            _FIXTURE_PATH, "synthetic_fuel", 600.0, diffusion="total"
        )

    assert cross_sections.fission is not None
    production = cross_sections.fission.neutron_production
    np.testing.assert_allclose(production.nu_sigma_f, [0.014, 0.043, 0.112])
    np.testing.assert_allclose(production.chi, [0.72, 0.25, 0.03])
    np.testing.assert_allclose(
        production.fission_transfer,
        np.array([0.014, 0.043, 0.112])[:, np.newaxis]
        * np.array([0.72, 0.25, 0.03])[np.newaxis, :],
    )
    np.testing.assert_allclose(
        cross_sections.fission.kappa_sigma_f, [1.20e6, 3.80e6, 10.10e6]
    )
    assert not cross_sections.fission.kappa_sigma_f.flags.writeable
    assert not production.nu_sigma_f.flags.writeable
    assert not production.chi.flags.writeable


def test_cross_sections_warns_for_each_known_discarded_dataset(
    tmp_path: Path,
) -> None:
    """Every recognized optional runtime datum warns when it is discarded."""
    path = tmp_path / "discarded_data.h5"
    shutil.copy(_FIXTURE_PATH, path)
    with h5py.File(path, "r+") as library:
        record = library["synthetic_fuel/600K"]
        for name in (
            "prompt-nu-fission",
            "delayed-nu-fission",
            "chi-prompt",
            "chi-delayed",
            "beta",
            "decay-rate",
        ):
            record.create_dataset(name, data=np.array([0.0]))

    with pytest.warns(UserWarning) as warnings_record:
        CrossSections.from_openmc_mgxs_hdf5(
            path, "synthetic_fuel", 600.0, diffusion="total"
        )

    messages = {str(warning.message) for warning in warnings_record}
    for name in (
        "fission",
        "inverse-velocity",
        "prompt-nu-fission",
        "delayed-nu-fission",
        "chi-prompt",
        "chi-delayed",
        "beta",
        "decay-rate",
    ):
        assert any(f"{name} data" in message for message in messages)


def test_cross_sections_warns_for_unrecognized_temperature_data(
    tmp_path: Path,
) -> None:
    """Extra selected-temperature fields are visible to an importing caller."""
    path = tmp_path / "unrecognized_data.h5"
    shutil.copy(_FIXTURE_PATH, path)
    with h5py.File(path, "r+") as library:
        library["synthetic_moderator/600K"].create_dataset(
            "future-data", data=np.array([1.0])
        )

    with pytest.warns(UserWarning, match="unrecognized.*future-data"):
        cross_sections = CrossSections.from_openmc_mgxs_hdf5(
            path, "synthetic_moderator", 600.0, diffusion="total"
        )

    assert cross_sections.fission is None


def test_cross_sections_rejects_nonzero_delayed_data_for_nonfissionable_record(
    tmp_path: Path,
) -> None:
    """A nonfissionable record cannot silently carry delayed production data."""
    path = tmp_path / "nonfissionable_delayed_data.h5"
    shutil.copy(_FIXTURE_PATH, path)
    with h5py.File(path, "r+") as library:
        library["synthetic_moderator/600K"].create_dataset(
            "delayed-nu-fission", data=np.array([0.001])
        )

    with pytest.raises(ValueError, match="nonfissionable.*delayed-nu-fission"):
        CrossSections.from_openmc_mgxs_hdf5(
            path, "synthetic_moderator", 600.0, diffusion="total"
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda library: library["synthetic_fuel/600K"].__delitem__("fission"),
        lambda library: library["synthetic_fuel/600K/fission"].__setitem__(
            Ellipsis, np.zeros(3)
        ),
    ],
)
def test_cross_sections_discards_optional_fission_data(
    tmp_path: Path, mutate: object
) -> None:
    """Reaction-rate fission data is not needed for Morana production data."""
    path = tmp_path / "surplus_fission.h5"
    shutil.copy(_FIXTURE_PATH, path)
    with h5py.File(path, "r+") as library:
        mutate(library)  # type: ignore[operator]

    with pytest.warns(UserWarning, match="discarding OpenMC MGXS"):
        cross_sections = CrossSections.from_openmc_mgxs_hdf5(
            path, "synthetic_fuel", 600.0, diffusion="total"
        )

    assert cross_sections.fission is not None


def test_cross_sections_imports_transfer_fissionable_fixture() -> None:
    """Matrix nu-fission records retain general transfer production data."""
    with pytest.warns(UserWarning, match="discarding"):
        cross_sections = CrossSections.from_openmc_mgxs_hdf5(
            _FIXTURE_PATH, "synthetic_transfer_fuel", 600.0, diffusion="total"
        )

    assert cross_sections.fission is not None
    production = cross_sections.fission.neutron_production
    assert isinstance(production, FissionTransfer)
    np.testing.assert_allclose(
        production.fission_transfer,
        [[0.011, 0.003, 0.0], [0.029, 0.011, 0.002], [0.066, 0.026, 0.007]],
    )
    np.testing.assert_allclose(
        cross_sections.fission.kappa_sigma_f, [1.05e6, 3.40e6, 9.30e6]
    )
    assert not production.fission_transfer.flags.writeable


def test_cross_sections_allows_absent_kappa_fission(tmp_path: Path) -> None:
    """Missing recoverable-energy data leaves power normalization unavailable."""
    path = tmp_path / "without_kappa_fission.h5"
    shutil.copy(_FIXTURE_PATH, path)
    with h5py.File(path, "r+") as library:
        del library["synthetic_fuel/600K/kappa-fission"]

    with pytest.warns(UserWarning, match="discarding"):
        cross_sections = CrossSections.from_openmc_mgxs_hdf5(
            path, "synthetic_fuel", 600.0, diffusion="total"
        )

    assert cross_sections.fission is not None
    assert cross_sections.fission.kappa_sigma_f is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda library: library["synthetic_transfer_fuel/600K"].create_dataset(
                "chi", data=np.array([0.72, 0.25, 0.03])
            ),
            "mixed.*matrix nu-fission.*chi",
        ),
        (
            lambda library: library[
                "synthetic_transfer_fuel/600K/nu-fission"
            ].__setitem__(Ellipsis, np.zeros((3, 3))),
            "nu-fission transfer.*at least one nonzero",
        ),
    ],
)
def test_cross_sections_rejects_invalid_transfer_fission_data(
    tmp_path: Path, mutate: object, message: str
) -> None:
    """Transfer production must be unambiguous and produce neutrons."""
    path = tmp_path / "invalid_transfer_fission.h5"
    shutil.copy(_FIXTURE_PATH, path)
    with h5py.File(path, "r+") as library:
        mutate(library)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        CrossSections.from_openmc_mgxs_hdf5(
            path, "synthetic_transfer_fuel", 600.0, diffusion="total"
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda library: library["synthetic_fuel/600K"].__delitem__("nu-fission"),
            "nu-fission",
        ),
        (
            lambda library: library["synthetic_fuel/600K"].__delitem__("chi"),
            "chi",
        ),
        (
            lambda library: library["synthetic_fuel/600K/nu-fission"].__setitem__(
                Ellipsis, np.zeros(3)
            ),
            "OpenMC MGXS nu-fission must contain at least one nonzero value",
        ),
        (
            lambda library: library["synthetic_fuel/600K/chi"].__setitem__(
                Ellipsis, np.zeros(3)
            ),
            "positive chi sum",
        ),
        (
            lambda library: library["synthetic_fuel/600K/kappa-fission"].__setitem__(
                Ellipsis, np.array([1.20e6, -3.80e6, 10.10e6])
            ),
            "kappa-fission.*non-negative",
        ),
        (
            lambda library: library["synthetic_fuel"].attrs.__setitem__(
                "fissionable", False
            ),
            "nonfissionable.*nonzero 'fission'",
        ),
    ],
)
def test_cross_sections_rejects_incomplete_or_contradictory_fissionable_data(
    tmp_path: Path, mutate: object, message: str
) -> None:
    """Fissionable records require complete, positive separable production."""
    path = tmp_path / "invalid_fissionable.h5"
    shutil.copy(_FIXTURE_PATH, path)
    with h5py.File(path, "r+") as library:
        mutate(library)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        CrossSections.from_openmc_mgxs_hdf5(
            path, "synthetic_fuel", 600.0, diffusion="total"
        )


def test_cross_sections_requires_boolean_fissionable_metadata(tmp_path: Path) -> None:
    """The reusable Boolean-attribute reader rejects numeric stand-ins."""
    path = tmp_path / "numeric_fissionable.h5"
    shutil.copy(_FIXTURE_PATH, path)
    with h5py.File(path, "r+") as library:
        library["synthetic_moderator"].attrs["fissionable"] = 0

    with pytest.raises(ValueError, match="fissionable.*Boolean"):
        CrossSections.from_openmc_mgxs_hdf5(
            path, "synthetic_moderator", 600.0, diffusion="total"
        )


@pytest.mark.parametrize(
    ("diffusion", "message"),
    [
        ("unsupported", "diffusion must be"),
        ("p1-outscatter", "P1 outscatter transport cross section"),
    ],
)
def test_cross_sections_rejects_unavailable_or_invalid_diffusion_data(
    tmp_path: Path, diffusion: str, message: str
) -> None:
    """Diffusion conventions must be explicit and physically defined."""
    path = tmp_path / "invalid_diffusion_data.h5"
    shutil.copy(_FIXTURE_PATH, path)
    if diffusion == "p1-outscatter":
        with h5py.File(path, "r+") as library:
            matrix = library["synthetic_moderator/600K/scatter_data/scatter_matrix"]
            values = matrix[()]
            values[1] = 1.0
            matrix[...] = values

    with pytest.raises(ValueError, match=message):
        CrossSections.from_openmc_mgxs_hdf5(
            path, "synthetic_moderator", 600.0, diffusion=diffusion
        )


def test_cross_sections_reports_a_missing_temperature_group(tmp_path: Path) -> None:
    """Selection reports a missing data group for the matching temperature."""
    path = tmp_path / "missing_temperature_group.h5"
    shutil.copy(_FIXTURE_PATH, path)
    with h5py.File(path, "r+") as library:
        del library["synthetic_moderator/600K"]

    with pytest.raises(ValueError, match="no data group"):
        CrossSections.from_openmc_mgxs_hdf5(
            path, "synthetic_moderator", 600.0, diffusion="total"
        )


def test_cross_sections_physical_data_errors_identify_selected_record(
    tmp_path: Path,
) -> None:
    """Malformed selected-temperature data reports the record and label."""
    path = tmp_path / "missing_absorption.h5"
    shutil.copy(_FIXTURE_PATH, path)
    with h5py.File(path, "r+") as library:
        del library["synthetic_moderator/600K/absorption"]

    with pytest.raises(
        ValueError,
        match=r"dataset 'synthetic_moderator' temperature '600K'.*absorption",
    ):
        CrossSections.from_openmc_mgxs_hdf5(
            path, "synthetic_moderator", 600.0, diffusion="total"
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda library: library["synthetic_moderator"].attrs.__setitem__(
                "representation", "angle"
            ),
            "representation",
        ),
        (
            lambda library: library["synthetic_moderator"].attrs.__setitem__(
                "scatter_format", "histogram"
            ),
            "scatter_format",
        ),
        (
            lambda library: library["synthetic_moderator"].attrs.__setitem__(
                "scatter_shape", "[G][G']"
            ),
            "scatter_shape",
        ),
        (
            lambda library: library["synthetic_moderator/600K"].__delitem__(
                "absorption"
            ),
            "absorption",
        ),
        (
            lambda library: library[
                "synthetic_moderator/600K/scatter_data"
            ].__delitem__("g_min"),
            "g_min",
        ),
        (
            lambda library: library[
                "synthetic_moderator/600K/scatter_data"
            ].__delitem__("scatter_matrix"),
            "scatter_matrix",
        ),
        (
            lambda library: library[
                "synthetic_moderator/600K/scatter_data/g_max"
            ].__setitem__(Ellipsis, np.array([3, 2, 1])),
            "bands are invalid",
        ),
        (
            lambda library: library[
                "synthetic_moderator/600K/scatter_data/scatter_matrix"
            ].__setitem__(Ellipsis, np.array([np.nan] * 18)),
            "finite",
        ),
        (
            lambda library: library[
                "synthetic_moderator/600K/scatter_data"
            ].create_dataset("multiplicity_matrix", data=np.array([-1.0] * 6)),
            "multiplicity_matrix.*non-negative",
        ),
        (
            lambda library: library["synthetic_moderator/600K/total"].__setitem__(
                Ellipsis, np.array([0.41, 0.0, 1.09])
            ),
            "strictly positive",
        ),
    ],
)
def test_cross_sections_rejects_incomplete_or_invalid_nonfissionable_data(
    tmp_path: Path, mutate: object, message: str
) -> None:
    """Required physical datasets and scalar-flux metadata are never inferred."""
    path = tmp_path / "invalid_nonfissionable.h5"
    shutil.copy(_FIXTURE_PATH, path)
    with h5py.File(path, "r+") as library:
        mutate(library)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        CrossSections.from_openmc_mgxs_hdf5(
            path, "synthetic_moderator", 600.0, diffusion="total"
        )


def _write_minimal_runtime_library(path: Path) -> Path:
    """Write only the metadata required by the reader-boundary tests."""
    with h5py.File(path, "w") as library:
        library.attrs["filetype"] = "mgxs"
        library.attrs["version"] = np.array([1, 0])
        library.attrs["energy_groups"] = 2
        library.attrs["group structure"] = np.array([0.0, 1.0, 2.0])
        record = library.create_group("material")
        temperatures = record.create_group("kTs")
        temperatures.create_dataset("600K", data=600.0 * 8.617333262e-5)
        record.create_group("600K")
    return path


def _select_runtime_library(
    path: Path, dataset: str, temperature: float
) -> _OpenMCMGXSSelection:
    """Run the private open-library selector under a checked HDF5 handle."""
    with h5py.File(path, "r") as library:
        selection, _, _ = _select_openmc_runtime_mgxs_from_library(
            library, dataset, temperature
        )
        return selection


@pytest.mark.parametrize("diffusion", ["total", "p1-outscatter"])
def test_imported_diffusion_avoids_intermediate_overflow(tmp_path, diffusion):
    """Large finite total data must retain positive representable diffusion."""
    path = tmp_path / "large_total.h5"
    shutil.copyfile(_FIXTURE_PATH, path)
    with h5py.File(path, "r+") as library:
        library["synthetic_fuel/600K/total"][...] = 1e308
    with pytest.warns(UserWarning):
        xs = CrossSections.from_openmc_mgxs_hdf5(
            path, "synthetic_fuel", 600.0, diffusion=diffusion
        )
    np.testing.assert_allclose(xs.D * 1e308, np.full(3, 1.0 / 3.0))
