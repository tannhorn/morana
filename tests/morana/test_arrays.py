"""Tests for shared private immutable-array ownership helpers."""

import numpy as np
import pytest

from morana._arrays import (
    readonly_float_array,
    require_finite_array,
    require_finite_nonnegative_array,
    require_finite_positive_array,
    require_nonzero_array,
)


def test_readonly_float_array_owns_and_freezes_converted_values() -> None:
    """The helper should isolate inputs while normalizing its floating dtype."""
    values = np.array([1, 2], dtype=int)

    owned = readonly_float_array(values)
    values[0] = 3

    assert owned.dtype == float
    assert not owned.flags.writeable
    np.testing.assert_allclose(owned, [1.0, 2.0])


@pytest.mark.parametrize("values", [[True], ["1.0"], [1.0 + 0.0j]])
def test_readonly_float_array_rejects_non_real_numeric_values(
    values: list[object],
) -> None:
    """Shared floating-array ownership must not silently coerce its input."""
    with pytest.raises(TypeError, match="real numeric"):
        readonly_float_array(values)


@pytest.mark.parametrize("values", [np.array([np.nan]), np.array([np.inf])])
def test_finite_array_contract_rejects_nonfinite_entries(values: np.ndarray) -> None:
    """Finite-array contracts should reject every nonfinite representation."""
    with pytest.raises(ValueError, match="values must contain only finite values"):
        require_finite_array("values", values)


def test_finite_nonnegative_array_contract_rejects_negative_entries() -> None:
    """Finite nonnegative arrays should reject negative entries after finiteness."""
    with pytest.raises(ValueError, match="values must be non-negative"):
        require_finite_nonnegative_array("values", np.array([0.0, -1.0]))


@pytest.mark.parametrize("values", [np.array([0.0]), np.array([-1.0])])
def test_finite_positive_array_contract_rejects_nonpositive_entries(
    values: np.ndarray,
) -> None:
    """Finite positive arrays should reject zero and negative entries."""
    with pytest.raises(ValueError, match="values must be strictly positive"):
        require_finite_positive_array("values", values)


def test_finite_positive_array_contract_accepts_positive_entries() -> None:
    """Finite positive arrays should accept every strictly positive value."""
    require_finite_positive_array("values", np.array([1.0, 2.0]))


def test_nonzero_array_contract_rejects_an_all_zero_array() -> None:
    """Nonzero-array contracts should reject arrays with no nonzero entries."""
    with pytest.raises(
        ValueError, match="values must contain at least one nonzero value"
    ):
        require_nonzero_array("values", np.array([0.0, 0.0]))


def test_nonzero_array_contract_accepts_an_array_with_a_nonzero_entry() -> None:
    """Nonzero-array contracts should accept arrays containing a nonzero entry."""
    require_nonzero_array("values", np.array([0.0, 2.0]))
