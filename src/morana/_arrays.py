"""Private helpers for owned immutable NumPy arrays."""

from __future__ import annotations

from numbers import Real

import numpy as np


def float_array(name: str, values: object) -> np.ndarray:
    """Return an owned floating-point array made only from real values.

    Strings, booleans, complex values, and arbitrary objects with a
    ``__float__`` conversion are deliberately rejected instead of silently
    coercing them to numerical input.
    """
    raw = np.asarray(values, dtype=object)
    if any(
        isinstance(value, bool) or not isinstance(value, Real) for value in raw.flat
    ):
        raise TypeError(f"{name} must contain only real numeric values")
    return np.array(values, dtype=float, copy=True)


def readonly_float_array(values: object) -> np.ndarray:
    """Return an owned read-only floating-point copy of real numeric values."""
    array = float_array("array", values)
    return freeze_owned_float_array(array)


def freeze_owned_float_array(values: np.ndarray) -> np.ndarray:
    """Mark a trusted owned floating-point array read-only without copying.

    Callers must retain ownership of ``values`` and have already established
    its floating-point dtype and value contract.  Use ``readonly_float_array``
    for untrusted input instead.
    """
    values.setflags(write=False)
    return values


def require_finite_array(name: str, values: np.ndarray) -> None:
    """Require every entry in one array to be finite."""
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")


def require_finite_nonnegative_array(name: str, values: np.ndarray) -> None:
    """Require every entry in one array to be finite and nonnegative."""
    require_finite_array(name, values)
    if np.any(values < 0.0):
        raise ValueError(f"{name} must be non-negative")


def require_finite_positive_array(name: str, values: np.ndarray) -> None:
    """Require every entry in one array to be finite and strictly positive."""
    require_finite_array(name, values)
    if np.any(values <= 0.0):
        raise ValueError(f"{name} must be strictly positive")


def require_nonzero_array(name: str, values: np.ndarray) -> None:
    """Require an array to contain at least one nonzero entry."""
    if not np.any(values):
        raise ValueError(f"{name} must contain at least one nonzero value")
