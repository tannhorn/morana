"""Tests for material and cross-section containers."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from morana import (
    CrossSections,
    ExcludedRegion,
    FissionData,
    FissionTransfer,
    Material,
    SeparableFission,
)
from morana.materials import (
    DEFAULT_EXCLUDED_REGION_COLOR,
    INACTIVE_EXCLUDED_KIND,
    INACTIVE_EXCLUDED_REGION_COLOR,
    excluded_key_color,
    excluded_key_kind,
    is_active_material_key,
    is_excluded_material_key,
    normalize_excluded_regions,
)


def test_separable_fission_owns_normalized_data_and_derives_transfer() -> None:
    """Separable fission should normalize owned inputs into canonical data."""
    production = np.array([0.025, 0.01])
    spectrum = np.array([0.700000001, 0.299999999])

    fission = SeparableFission(nu_sigma_f=production, chi=spectrum)
    production[0] = 9.9
    spectrum[0] = 9.9

    assert fission.groups == 2
    np.testing.assert_allclose(fission.nu_sigma_f, [0.025, 0.01])
    np.testing.assert_allclose(fission.chi, [0.700000001, 0.299999999])
    np.testing.assert_allclose(
        fission.fission_transfer,
        [[0.017500000025, 0.007499999975], [0.00700000001, 0.00299999999]],
    )
    np.testing.assert_allclose(fission.fission_production, [0.025, 0.01])
    assert fission.nu_sigma_f.flags.writeable is False
    assert fission.chi.flags.writeable is False
    assert fission.fission_transfer.flags.writeable is False
    assert fission.fission_production.flags.writeable is False


@pytest.mark.parametrize(
    ("nu_sigma_f", "chi", "message"),
    [
        ([], [], "at least one"),
        ([0.025], [1.0, 0.0], "same group count"),
        ([[0.025]], [1.0], "one-dimensional"),
        ([0.0], [1.0], "at least one nonzero value"),
        ([0.025], [0.0], "positive sum"),
        ([0.025], [0.8], "normalization_tolerance"),
    ],
)
def test_separable_fission_validates_input_contract(
    nu_sigma_f: list[float] | list[list[float]],
    chi: list[float],
    message: str,
) -> None:
    """Separable fission should enforce its compact representation contract."""
    with pytest.raises(ValueError, match=message):
        SeparableFission(nu_sigma_f=nu_sigma_f, chi=chi)


@pytest.mark.parametrize("field_name", ["nu_sigma_f", "chi"])
def test_separable_fission_rejects_non_numeric_data(field_name: str) -> None:
    """Separable fission inputs must contain actual real numeric values."""
    arguments: dict[str, list[object]] = {
        "nu_sigma_f": [0.025],
        "chi": [1.0],
    }
    arguments[field_name] = [True]

    with pytest.raises(TypeError, match=field_name):
        SeparableFission(**arguments)  # type: ignore[arg-type]


def test_fission_transfer_owns_data_and_derives_incident_production() -> None:
    """Transfer fission should retain event orientation and derive row totals."""
    transfer = np.array([[0.01, 0.015], [0.005, 0.0]])

    fission = FissionTransfer(fission_transfer=transfer)
    transfer[0, 0] = 9.9

    assert fission.groups == 2
    np.testing.assert_allclose(fission.fission_transfer, [[0.01, 0.015], [0.005, 0.0]])
    np.testing.assert_allclose(fission.fission_production, [0.025, 0.005])
    assert fission.fission_transfer.flags.writeable is False
    assert fission.fission_production.flags.writeable is False


@pytest.mark.parametrize(
    ("transfer", "message"),
    [
        ([], "square two-dimensional"),
        ([[0.025, 0.0]], "square two-dimensional"),
        ([[0.0]], "at least one nonzero value"),
        ([[float("nan")]], "finite"),
        ([[-0.025]], "non-negative"),
        ([[True]], "real numeric"),
    ],
)
def test_fission_transfer_validates_input_contract(
    transfer: list[object] | list[list[object]], message: str
) -> None:
    """Transfer fission should enforce square nonnegative production data."""
    exception = TypeError if message == "real numeric" else ValueError
    with pytest.raises(exception, match=message):
        FissionTransfer(fission_transfer=transfer)  # type: ignore[arg-type]


def test_fission_data_forwards_canonical_production_and_owns_energy_data() -> None:
    """Fission data should expose selected production data and own energy input."""
    energy_production = np.array([2.0e6, 1.5e6])
    fission = FissionData(
        neutron_production=FissionTransfer([[0.01, 0.015], [0.005, 0.0]]),
        kappa_sigma_f=energy_production,
    )
    energy_production[0] = 9.9

    assert fission.groups == 2
    np.testing.assert_allclose(fission.fission_production, [0.025, 0.005])
    np.testing.assert_allclose(fission.kappa_sigma_f, [2.0e6, 1.5e6])
    assert fission.kappa_sigma_f is not None
    assert fission.kappa_sigma_f.flags.writeable is False


@pytest.mark.parametrize(
    ("neutron_production", "kappa_sigma_f", "exception", "message"),
    [
        (None, None, TypeError, "SeparableFission or FissionTransfer"),
        (
            SeparableFission([0.025], [1.0]),
            [2.0e6, 1.5e6],
            ValueError,
            "expected shape",
        ),
        (SeparableFission([0.025], [1.0]), [float("nan")], ValueError, "finite"),
        (SeparableFission([0.025], [1.0]), [True], TypeError, "real numeric"),
    ],
)
def test_fission_data_validates_selected_production_and_energy_data(
    neutron_production: object,
    kappa_sigma_f: list[object] | None,
    exception: type[Exception],
    message: str,
) -> None:
    """Fission data should require one checked production representation."""
    with pytest.raises(exception, match=message):
        FissionData(
            neutron_production=neutron_production,  # type: ignore[arg-type]
            kappa_sigma_f=kappa_sigma_f,  # type: ignore[arg-type]
        )


def test_cross_sections_convert_lists_to_numpy_arrays() -> None:
    """Cross-section inputs should immediately become NumPy arrays."""
    xs = CrossSections(
        D=[1.2],
        sigma_a=[0.02],
        sigma_s=[[0.0]],
        fission=None,
    )

    assert xs.groups == 1
    assert isinstance(xs.D, np.ndarray)
    assert xs.sigma_s.shape == (1, 1)


def test_cross_sections_own_input_arrays() -> None:
    """Mutating caller-owned arrays should not change stored cross sections."""
    diffusion = np.array([1.2])
    absorption = np.array([0.02])
    scatter = np.array([[0.0]])
    production = np.array([0.025])
    spectrum = np.array([1.0])

    xs = CrossSections(
        D=diffusion,
        sigma_a=absorption,
        sigma_s=scatter,
        fission=FissionData(
            neutron_production=SeparableFission(nu_sigma_f=production, chi=spectrum)
        ),
    )
    diffusion[0] = 9.9
    absorption[0] = 9.9
    scatter[0, 0] = 9.9
    production[0] = 9.9
    spectrum[0] = 9.9

    np.testing.assert_allclose(xs.D, [1.2])
    np.testing.assert_allclose(xs.sigma_a, [0.02])
    np.testing.assert_allclose(xs.sigma_s, [[0.0]])
    assert xs.fission is not None
    np.testing.assert_allclose(xs.fission.fission_production, [0.025])


def test_cross_sections_own_optional_fission_energy_production_data() -> None:
    """Recoverable fission-energy production data should be immutable input."""
    energy_production = np.array([2.0e6])
    xs = CrossSections(
        D=[1.2],
        sigma_a=[0.02],
        sigma_s=[[0.0]],
        fission=FissionData(
            neutron_production=SeparableFission(nu_sigma_f=[0.025], chi=[1.0]),
            kappa_sigma_f=energy_production,
        ),
    )
    energy_production[0] = 9.9

    assert xs.fission is not None
    np.testing.assert_allclose(xs.fission.kappa_sigma_f, [2.0e6])
    assert xs.fission.kappa_sigma_f is not None
    assert xs.fission.kappa_sigma_f.flags.writeable is False


def test_cross_sections_own_optional_scattering_multiplicity_data() -> None:
    """Scattering multiplicity should be immutable input when supplied."""
    multiplicity = np.array([[1.0, 1.5], [0.0, 2.0]])

    xs = CrossSections(
        D=[1.2, 0.8],
        sigma_a=[0.02, 0.04],
        sigma_s=[[0.1, 0.2], [0.0, 0.3]],
        fission=None,
        multiplicity_matrix=multiplicity,
    )
    multiplicity[0, 1] = 9.9

    np.testing.assert_allclose(xs.multiplicity_matrix, [[1.0, 1.5], [0.0, 2.0]])
    assert xs.multiplicity_matrix is not None
    assert xs.multiplicity_matrix.flags.writeable is False


def test_cross_sections_use_none_for_compact_unit_scattering_multiplicity() -> None:
    """Omitted multiplicity should retain the public compact representation."""
    xs = CrossSections(
        D=[1.2],
        sigma_a=[0.02],
        sigma_s=[[0.0]],
        fission=None,
    )

    assert xs.multiplicity_matrix is None


def test_cross_sections_require_scattering_multiplicity_shape_to_match_groups() -> None:
    """Supplied scattering multiplicity must cover every group pair."""
    with pytest.raises(ValueError, match=r"expected shape \(2, 2\)"):
        CrossSections(
            D=[1.2, 0.8],
            sigma_a=[0.02, 0.04],
            sigma_s=[[0.1, 0.2], [0.0, 0.3]],
            fission=None,
            multiplicity_matrix=[[1.0]],
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("D", [np.nan]),
        ("D", [np.inf]),
        ("D", [-1.0]),
        ("sigma_a", [np.nan]),
        ("sigma_a", [np.inf]),
        ("sigma_a", [-1.0]),
        ("sigma_s", [[np.nan]]),
        ("sigma_s", [[np.inf]]),
        ("sigma_s", [[-1.0]]),
        ("multiplicity_matrix", [[np.nan]]),
        ("multiplicity_matrix", [[np.inf]]),
        ("multiplicity_matrix", [[-1.0]]),
    ],
)
def test_cross_sections_reject_nonfinite_or_negative_data(
    field_name: str,
    field_value: list[float] | list[list[float]],
) -> None:
    """Every cross-section field should contain finite non-negative values."""
    arguments = {
        "D": [1.2],
        "sigma_a": [0.02],
        "sigma_s": [[0.0]],
        "fission": None,
        field_name: field_value,
    }

    with pytest.raises(ValueError, match=field_name):
        CrossSections(**arguments)


def test_cross_section_arrays_are_read_only() -> None:
    """Stored cross-section arrays should retain construction-time invariants."""
    xs = CrossSections(
        D=[1.2],
        sigma_a=[0.02],
        sigma_s=[[0.0]],
        fission=None,
    )

    with pytest.raises(ValueError, match="read-only"):
        xs.sigma_a[0] = np.inf


def test_cross_sections_are_immutable() -> None:
    """Validated cross sections must not permit field replacement."""
    xs = CrossSections(
        D=[1.2],
        sigma_a=[0.02],
        sigma_s=[[0.0]],
        fission=None,
    )

    with pytest.raises(FrozenInstanceError):
        xs.D = np.array([1.0])


def test_cross_sections_retains_checked_fission_data() -> None:
    """Cross sections should retain the selected checked fission bundle."""
    xs = CrossSections(
        D=[1.2, 0.4],
        sigma_a=[0.02, 0.08],
        sigma_s=[[0.1, 0.15], [0.02, 0.2]],
        fission=FissionData(
            neutron_production=SeparableFission(
                nu_sigma_f=[0.025, 0.0], chi=[0.700000001, 0.299999999]
            )
        ),
    )

    assert xs.fission is not None
    np.testing.assert_allclose(
        xs.fission.neutron_production.chi, [0.700000001, 0.299999999]
    )


@pytest.mark.parametrize(
    ("fission", "message"),
    [
        (object(), "FissionData or None"),
        (
            FissionData(neutron_production=SeparableFission([0.025], [1.0])),
            "same group count",
        ),
    ],
)
def test_cross_sections_validate_optional_fission_bundle(
    fission: object, message: str
) -> None:
    """Cross sections should delegate fission details and check group alignment."""
    with pytest.raises((TypeError, ValueError), match=message):
        CrossSections(
            D=[1.2, 0.8],
            sigma_a=[0.02, 0.04],
            sigma_s=[[0.0, 0.0], [0.0, 0.0]],
            fission=fission,  # type: ignore[arg-type]
        )


def test_material_rejects_builtin_inactive_excluded_name() -> None:
    """Material name '0' is the built-in inactive excluded key."""
    with pytest.raises(ValueError):
        Material(name="0")


def test_builtin_excluded_material_key_is_generalized() -> None:
    """The inactive key should be queried through excluded-region helpers."""
    assert INACTIVE_EXCLUDED_KIND == "inactive"
    assert excluded_key_kind("0") == INACTIVE_EXCLUDED_KIND
    assert excluded_key_color("0") == INACTIVE_EXCLUDED_REGION_COLOR
    assert is_excluded_material_key("0")
    assert not is_active_material_key("0")
    assert is_active_material_key("medium")


def test_custom_excluded_regions_are_normalized() -> None:
    """User excluded-region descriptions should classify additional keys."""
    regions = normalize_excluded_regions(
        {"gap": ExcludedRegion(kind="void", color="#eeeeee")}
    )

    assert excluded_key_kind("0", regions) == INACTIVE_EXCLUDED_KIND
    assert excluded_key_kind("gap", regions) == "void"
    assert excluded_key_color("gap", regions) == "#eeeeee"
    assert is_excluded_material_key("gap", regions)
    assert not is_active_material_key("gap", regions)


def test_custom_excluded_region_without_color_uses_central_default() -> None:
    """Excluded keys without explicit colors should use one central default."""
    regions = normalize_excluded_regions({"gap": ExcludedRegion(kind="void")})

    assert DEFAULT_EXCLUDED_REGION_COLOR == "#d9d9d9"
    assert excluded_key_color("gap", regions) == DEFAULT_EXCLUDED_REGION_COLOR


@pytest.mark.parametrize("kind", ["", "   ", "void\nport", "void\u200bport"])
def test_excluded_region_rejects_nonhuman_readable_kind(kind: str) -> None:
    """Excluded-region kinds must remain legible selector identifiers."""
    with pytest.raises(ValueError, match="excluded-region kind"):
        ExcludedRegion(kind=kind)


@pytest.mark.parametrize("kind", [None, 1])
def test_excluded_region_requires_string_kind(kind: object) -> None:
    """Excluded-region kinds must satisfy their public string contract."""
    with pytest.raises(TypeError, match="kind must be a string"):
        ExcludedRegion(kind=kind)  # type: ignore[arg-type]


@pytest.mark.parametrize("color", [0, False])
def test_excluded_region_requires_optional_string_color(color: object) -> None:
    """Colors retain flexible syntax but must have the documented type."""
    with pytest.raises(TypeError, match="color must be a string or None"):
        ExcludedRegion(kind="void", color=color)  # type: ignore[arg-type]


def test_excluded_regions_reject_builtin_redefinition() -> None:
    """The built-in inactive key should stay defined by default."""
    with pytest.raises(ValueError, match="cannot redefine"):
        normalize_excluded_regions({"0": ExcludedRegion(kind="inactive")})


def test_excluded_regions_reject_non_string_keys() -> None:
    """Excluded-region catalogs should require string keys."""
    with pytest.raises(TypeError, match="material key must be a string"):
        normalize_excluded_regions({1: ExcludedRegion(kind="void")})  # type: ignore[dict-item]


@pytest.mark.parametrize("key", ["", "   ", "beam\nport", "beam\u200bport"])
def test_excluded_regions_reject_nonhuman_readable_keys(key: str) -> None:
    """Excluded-region keys must remain legible material-mesh identifiers."""
    with pytest.raises(ValueError, match="material key"):
        normalize_excluded_regions({key: ExcludedRegion(kind="void")})


@pytest.mark.parametrize(
    "lookup",
    [
        excluded_key_kind,
        excluded_key_color,
        is_excluded_material_key,
        is_active_material_key,
    ],
)
def test_excluded_region_queries_reject_nonmapping_catalogs(lookup: object) -> None:
    """Excluded-region queries should reject nonmapping catalogs explicitly."""
    with pytest.raises(TypeError, match="excluded_regions must be a mapping"):
        lookup("gap", [])  # type: ignore[operator]


@pytest.mark.parametrize(
    "lookup",
    [
        excluded_key_kind,
        excluded_key_color,
        is_excluded_material_key,
        is_active_material_key,
    ],
)
def test_excluded_region_queries_reject_invalid_selected_values(lookup: object) -> None:
    """Excluded-region queries should not fail incidentally on malformed values."""
    with pytest.raises(TypeError, match="values must be ExcludedRegion"):
        lookup("gap", {"gap": object()})  # type: ignore[operator, dict-item]


def test_excluded_regions_reject_invalid_value_types() -> None:
    """Excluded-region catalogs should report invalid values as type errors."""
    with pytest.raises(TypeError, match="values must be ExcludedRegion"):
        normalize_excluded_regions({"gap": object()})  # type: ignore[dict-item]


def test_material_can_define_plotting_color() -> None:
    """Materials may carry an optional plotting color."""
    material = Material(name="medium", color="#ff0000")

    assert material.color == "#ff0000"


@pytest.mark.parametrize("name", [None, 1, "", "   ", "fuel\n1", "fuel\u200b1"])
def test_material_requires_human_readable_name(name: object) -> None:
    """Material identities cannot defer invalid names to configuration use."""
    with pytest.raises((TypeError, ValueError), match="material name"):
        Material(name=name)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["xs", "color"])
def test_material_requires_typed_optional_fields(field: str) -> None:
    """Optional material fields should reject untyped construction values."""
    with pytest.raises(TypeError):
        Material(name="medium", **{field: 1})  # type: ignore[arg-type]


def test_material_is_immutable() -> None:
    """Validated material identity must not be replaceable."""
    material = Material(name="medium", color="#ff0000")

    with pytest.raises(FrozenInstanceError):
        material.color = "#00ff00"


def test_fission_transfer_rejects_nonfinite_derived_production():
    """Finite transfer entries must also produce a finite incident-group sum."""
    with pytest.raises(ValueError, match="fission_production"):
        FissionTransfer([[1e308, 1e308], [0.0, 0.0]])
