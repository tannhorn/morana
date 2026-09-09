"""Private reusable contracts for scalar numerical and string inputs."""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real


def require_nonempty_string(name: str, value: object) -> str:
    """Return one nonempty string, rejecting other values and empty strings."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def require_human_readable_identifier(name: str, value: object) -> str:
    """Return one legible identifier string without changing its spelling.

    Identifiers may contain spaces, punctuation, Unicode, and leading digits,
    but must contain a non-whitespace character and only printable characters.
    """
    value = require_nonempty_string(name, value)
    if not value.strip():
        raise ValueError(f"{name} must not be whitespace only")
    if not value.isprintable():
        raise ValueError(f"{name} must contain only printable characters")
    return value


def require_optional_string(name: str, value: object) -> str | None:
    """Return one nonempty string or ``None``."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def require_integer(name: str, value: object) -> int:
    """Return one non-Boolean integer, rejecting other types with ``TypeError``."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def require_positive_integer(name: str, value: object) -> int:
    """Return one positive non-Boolean integer.

    Non-integer and Boolean inputs raise ``TypeError``; nonpositive integers
    raise ``ValueError``.
    """
    value = require_integer(name, value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def require_nonnegative_integer(name: str, value: object) -> int:
    """Return one nonnegative non-Boolean integer.

    Non-integer and Boolean inputs raise ``TypeError``; negative integers
    raise ``ValueError``.
    """
    value = require_integer(name, value)
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def require_finite_real(name: str, value: object) -> float:
    """Return one finite non-Boolean real value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric value")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def require_finite_positive_real(name: str, value: object) -> float:
    """Return one finite positive non-Boolean real value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric value")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def require_finite_nonnegative_real(name: str, value: object) -> float:
    """Return one finite nonnegative non-Boolean real value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric value")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite and nonnegative") from error
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value
