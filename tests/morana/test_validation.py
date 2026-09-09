"""Tests for shared private scalar validation contracts."""

import numpy as np
import pytest

from morana._validation import (
    require_finite_nonnegative_real,
    require_finite_positive_real,
    require_finite_real,
    require_integer,
    require_nonnegative_integer,
    require_nonempty_string,
    require_optional_string,
    require_positive_integer,
)


@pytest.mark.parametrize("value", ["name", "#ddeeff"])
def test_string_contracts_accept_nonempty_strings(value: str) -> None:
    """Shared string contracts retain ordinary nonempty labels unchanged."""
    assert require_nonempty_string("value", value) == value
    assert require_optional_string("value", value) == value
    assert require_optional_string("value", None) is None


@pytest.mark.parametrize("value", [None, 1, ""])
def test_nonempty_string_contract_rejects_invalid_values(value: object) -> None:
    """Required strings reject absent, untyped, and empty values."""
    with pytest.raises((TypeError, ValueError), match="value"):
        require_nonempty_string("value", value)


@pytest.mark.parametrize("value", [1, ""])
def test_optional_string_contract_rejects_invalid_values(value: object) -> None:
    """Optional strings accept only ``None`` or nonempty string labels."""
    with pytest.raises((TypeError, ValueError), match="value"):
        require_optional_string("value", value)


@pytest.mark.parametrize("value", [0, 1, np.int64(2)])
def test_nonnegative_integer_contract_accepts_integer_scalars(value: object) -> None:
    """Shared index contracts should accept Python and NumPy integer scalars."""
    assert require_nonnegative_integer("value", value) == int(value)


@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_nonnegative_integer_contract_rejects_noninteger_types(value: object) -> None:
    """Shared index contracts reject invalid types before range checking."""
    with pytest.raises(TypeError, match="value must be an integer"):
        require_nonnegative_integer("value", value)


def test_nonnegative_integer_contract_rejects_negative_values() -> None:
    """Shared index contracts retain their nonnegative lower bound."""
    with pytest.raises(ValueError, match="value must be a nonnegative integer"):
        require_nonnegative_integer("value", -1)


@pytest.mark.parametrize("value", [1, np.int64(2)])
def test_positive_integer_contract_normalizes_integral_values(value: object) -> None:
    """Positive integer contracts return ordinary Python integers."""
    assert require_positive_integer("count", value) == int(value)


@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_positive_integer_contract_rejects_noninteger_types(value: object) -> None:
    """Positive integer contracts reject invalid types before range checking."""
    with pytest.raises(TypeError, match="count must be an integer"):
        require_positive_integer("count", value)


@pytest.mark.parametrize("value", [0, -1])
def test_positive_integer_contract_rejects_nonpositive_values(value: int) -> None:
    """Positive integer contracts retain their strict lower bound."""
    with pytest.raises(ValueError, match="count must be a positive integer"):
        require_positive_integer("count", value)


@pytest.mark.parametrize("value", [True, 0.5, "0"])
def test_integer_contract_rejects_noninteger_types(value: object) -> None:
    """Integer contracts always raise TypeError for a non-integer input."""
    with pytest.raises(TypeError, match="axial_index must be an integer"):
        require_integer("axial_index", value)


@pytest.mark.parametrize("value", [1, 1.5, np.float64(2.0)])
def test_finite_real_contracts_normalize_numeric_scalars(value: object) -> None:
    """Finite-real contracts accept ordinary and NumPy real scalars."""
    assert require_finite_real("value", value) == float(value)
    assert require_finite_positive_real("value", value) == float(value)
    assert require_finite_nonnegative_real("value", value) == float(value)


@pytest.mark.parametrize("value", [True, "1"])
def test_finite_real_contract_rejects_nonreal_values(
    value: object,
) -> None:
    """Finite-real contracts use TypeError for non-real scalar inputs."""
    with pytest.raises(TypeError, match="value must be a real numeric value"):
        require_finite_real("value", value)


@pytest.mark.parametrize(
    "validator",
    [require_finite_positive_real, require_finite_nonnegative_real],
)
@pytest.mark.parametrize("value", [True, "1"])
def test_bounded_real_contracts_reject_nonreal_values(
    validator: object, value: object
) -> None:
    """Bounded real contracts retain the shared type distinction."""
    with pytest.raises(TypeError, match="value must be a real numeric value"):
        validator("value", value)  # type: ignore[operator]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_finite_real_contract_rejects_nonfinite_values(value: object) -> None:
    """Finite-real contracts use ValueError for nonfinite real scalars."""
    with pytest.raises(ValueError, match="value must be finite"):
        require_finite_real("value", value)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_positive_real_contract_rejects_nonpositive_values(value: float) -> None:
    """Positive real contracts use a strict lower bound."""
    with pytest.raises(ValueError, match="value must be finite and positive"):
        require_finite_positive_real("value", value)


@pytest.mark.parametrize("value", [-1.0, -np.inf])
def test_nonnegative_real_contract_rejects_negative_values(value: float) -> None:
    """Nonnegative real contracts retain zero and reject negative values."""
    with pytest.raises(ValueError, match="value must be finite and nonnegative"):
        require_finite_nonnegative_real("value", value)
