"""Immutable numerical controls for individual diffusion solves."""

# pylint: disable=too-many-instance-attributes

from __future__ import annotations

from dataclasses import dataclass, field
from morana._validation import (
    require_finite_nonnegative_real,
    require_finite_positive_real,
    require_positive_integer,
)


@dataclass(frozen=True)
class NoPreconditioner:
    """Select unpreconditioned GMRES execution."""

    @property
    def kind(self) -> str:
        """Return the stable preconditioner identifier ``"none"``."""
        return "none"


@dataclass(frozen=True)
class JacobiPreconditioner:
    """Select diagonal (Jacobi) preconditioning for GMRES."""

    @property
    def kind(self) -> str:
        """Return the stable preconditioner identifier ``"jacobi"``."""
        return "jacobi"


@dataclass(frozen=True)
class IluPreconditioner:
    """Select threshold incomplete-LU preconditioning for GMRES.

    Parameters
    ----------
    drop_tolerance
        Finite nonnegative real threshold used to drop incomplete-factor
        entries. Boolean values are not accepted.
    fill_factor
        Finite positive real upper bound on incomplete-factor fill relative to
        the original sparse matrix. Boolean values are not accepted.

    Raises
    ------
    TypeError
        If either control is not a real number or is Boolean.
    ValueError
        If either control is not finite after conversion to ``float``, or if
        ``drop_tolerance`` is negative or ``fill_factor`` is nonpositive.
    """

    drop_tolerance: float = 1.0e-4
    fill_factor: float = 10.0

    def __post_init__(self) -> None:
        """Check incomplete-LU controls."""
        drop_tolerance = require_finite_nonnegative_real(
            "drop_tolerance", self.drop_tolerance
        )
        fill_factor = require_finite_positive_real("fill_factor", self.fill_factor)
        object.__setattr__(self, "drop_tolerance", drop_tolerance)
        object.__setattr__(self, "fill_factor", fill_factor)

    @property
    def kind(self) -> str:
        """Return the stable preconditioner identifier ``"ilu"``."""
        return "ilu"


LinearPreconditioner = NoPreconditioner | JacobiPreconditioner | IluPreconditioner


@dataclass(frozen=True)
class DirectLinearSolveSettings:
    """Configure the sparse-direct reference linear-solve path.

    Parameters
    ----------
    relative_residual_tolerance
        Finite positive real bound applied to Morana's independently
        calculated true relative residual after SciPy returns a candidate
        solution. Boolean values are not accepted.

    Raises
    ------
    TypeError
        If ``relative_residual_tolerance`` is not a real number or is Boolean.
    ValueError
        If ``relative_residual_tolerance`` is not finite and positive after
        conversion to ``float``.
    """

    relative_residual_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        """Check direct linear-solve controls."""
        relative_residual_tolerance = require_finite_positive_real(
            "relative_residual_tolerance", self.relative_residual_tolerance
        )
        object.__setattr__(
            self, "relative_residual_tolerance", relative_residual_tolerance
        )

    @property
    def strategy(self) -> str:
        """Return the stable linear-solve strategy identifier ``"direct"``."""
        return "direct"


@dataclass(frozen=True)
class GmresLinearSolveSettings:
    """Configure restarted GMRES and its one typed preconditioner.

    Parameters
    ----------
    relative_residual_tolerance
        Finite positive real bound supplied to GMRES and applied again to
        Morana's independently calculated true relative residual. Boolean
        values are not accepted.
    max_krylov_iterations
        Positive non-Boolean integer maximum number of Krylov iterations
        across all restarts.
    restart
        Positive non-Boolean integer number of Krylov vectors retained in one
        GMRES cycle. It may not exceed ``max_krylov_iterations``.
    preconditioner
        One immutable no, Jacobi, or threshold-ILU preconditioner policy.

    Raises
    ------
    TypeError
        If a numerical control has an unsupported type or is Boolean, or if
        ``preconditioner`` is not a supported preconditioner policy.
    ValueError
        If a numerical control is out of range, cannot be represented as a
        finite ``float``, or ``restart`` exceeds ``max_krylov_iterations``.
    """

    relative_residual_tolerance: float = 1.0e-10
    max_krylov_iterations: int = 1_000
    restart: int = 50
    preconditioner: LinearPreconditioner = field(default_factory=NoPreconditioner)

    def __post_init__(self) -> None:
        """Check GMRES controls and its typed preconditioner."""
        relative_residual_tolerance = require_finite_positive_real(
            "relative_residual_tolerance", self.relative_residual_tolerance
        )
        max_krylov_iterations = require_positive_integer(
            "max_krylov_iterations", self.max_krylov_iterations
        )
        restart = require_positive_integer("restart", self.restart)
        if restart > max_krylov_iterations:
            raise ValueError("restart must not exceed max_krylov_iterations")
        if not isinstance(
            self.preconditioner,
            (NoPreconditioner, JacobiPreconditioner, IluPreconditioner),
        ):
            raise TypeError(
                "preconditioner must be NoPreconditioner, JacobiPreconditioner, "
                "or IluPreconditioner"
            )
        object.__setattr__(
            self, "relative_residual_tolerance", relative_residual_tolerance
        )
        object.__setattr__(self, "max_krylov_iterations", max_krylov_iterations)
        object.__setattr__(self, "restart", restart)

    @property
    def strategy(self) -> str:
        """Return the stable linear-solve strategy identifier ``"gmres"``."""
        return "gmres"


LinearSolveSettings = DirectLinearSolveSettings | GmresLinearSolveSettings


@dataclass(frozen=True)
class PowerIterationSettings:
    """Select ordinary fission-source-normalized power iteration."""

    @property
    def kind(self) -> str:
        """Return the stable eigenvalue-iteration identifier ``"power"``."""
        return "power"


@dataclass(frozen=True)
class WielandtShiftSettings:
    """Select fixed Wielandt-shifted fission-source power iteration.

    Parameters
    ----------
    shift_inverse_keff
        Finite nonnegative real fixed shift applied to the inverse
        multiplication factor in the shifted operator ``A - shift_inverse_keff
        * F``. Boolean values are not accepted.
        The solver does not adapt this value or fall back to ordinary power
        iteration when the shifted operator is unusable.

    Raises
    ------
    TypeError
        If ``shift_inverse_keff`` is not a real number or is Boolean.
    ValueError
        If ``shift_inverse_keff`` is not finite and nonnegative after
        conversion to ``float``.
    """

    shift_inverse_keff: float

    def __post_init__(self) -> None:
        """Check the fixed inverse-multiplication-factor shift."""
        shift_inverse_keff = require_finite_nonnegative_real(
            "shift_inverse_keff", self.shift_inverse_keff
        )
        object.__setattr__(self, "shift_inverse_keff", shift_inverse_keff)

    @property
    def kind(self) -> str:
        """Return the stable eigenvalue-iteration identifier ``"wielandt"``."""
        return "wielandt"


EigenvalueIterationSettings = PowerIterationSettings | WielandtShiftSettings


@dataclass(frozen=True)
class FixedSourceSettings:
    """Configure one fixed-source solve.

    Parameters
    ----------
    linear_solve
        Immutable per-call direct or GMRES policy. Direct solving is the
        default reference path.
    flux_nonnegativity_tolerance
        Finite nonnegative real relative tolerance for accepting and cleaning
        negative roundoff in the solved scalar-flux vector. The threshold is
        this value times the candidate vector's largest absolute component.
        Zero rejects every negative candidate; Boolean values are not
        accepted.

    Raises
    ------
    TypeError
        If ``linear_solve`` is not a supported linear-solve policy, or if
        ``flux_nonnegativity_tolerance`` is not a real number or is Boolean.
    ValueError
        If ``flux_nonnegativity_tolerance`` is not finite and nonnegative
        after conversion to ``float``.
    """

    linear_solve: LinearSolveSettings = field(default_factory=DirectLinearSolveSettings)
    flux_nonnegativity_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        """Check fixed-source numerical controls."""
        _check_linear_solve("linear_solve", self.linear_solve)
        flux_nonnegativity_tolerance = require_finite_nonnegative_real(
            "flux_nonnegativity_tolerance", self.flux_nonnegativity_tolerance
        )
        object.__setattr__(
            self, "flux_nonnegativity_tolerance", flux_nonnegativity_tolerance
        )


@dataclass(frozen=True)
class KeffSettings:
    """Configure one source-normalized fission eigenvalue solve.

    Parameters
    ----------
    inner_linear_solve
        Immutable direct or GMRES policy applied separately to each outer
        iteration. Direct solving is the default reference path.
    max_outer_iterations
        Positive non-Boolean integer maximum power-iteration count.
    keff_change_tolerance
        Finite positive real relative multiplication-factor change tolerance.
        Boolean values are not accepted.
    flux_change_tolerance
        Finite positive real volume-weighted normalized-flux change tolerance.
        Boolean values are not accepted.
    keff_relative_residual_tolerance
        Finite positive real relative eigenvalue-equation residual tolerance.
        Boolean values are not accepted.
    flux_nonnegativity_tolerance
        Finite nonnegative real relative tolerance for accepting and cleaning
        negative roundoff in a power-iteration flux vector. The threshold is
        this value times the candidate vector's largest absolute component.
        Zero rejects every negative candidate; Boolean values are not
        accepted.
    eigenvalue_iteration
        Immutable ordinary-power or fixed-Wielandt-shift policy. Ordinary
        power iteration is the default reference path.

    Raises
    ------
    TypeError
        If a policy input is unsupported, or if a numerical control has an
        unsupported type or is Boolean.
    ValueError
        If a numerical control is out of range or cannot be represented as a
        finite ``float``.
    """

    inner_linear_solve: LinearSolveSettings = field(
        default_factory=DirectLinearSolveSettings
    )
    max_outer_iterations: int = 100
    keff_change_tolerance: float = 1.0e-10
    flux_change_tolerance: float = 1.0e-10
    keff_relative_residual_tolerance: float = 1.0e-10
    flux_nonnegativity_tolerance: float = 1.0e-12
    eigenvalue_iteration: EigenvalueIterationSettings = field(
        default_factory=PowerIterationSettings
    )

    def __post_init__(self) -> None:
        """Check criticality numerical controls."""
        _check_linear_solve("inner_linear_solve", self.inner_linear_solve)
        max_outer_iterations = require_positive_integer(
            "max_outer_iterations", self.max_outer_iterations
        )
        keff_change_tolerance = require_finite_positive_real(
            "keff_change_tolerance", self.keff_change_tolerance
        )
        flux_change_tolerance = require_finite_positive_real(
            "flux_change_tolerance", self.flux_change_tolerance
        )
        keff_relative_residual_tolerance = require_finite_positive_real(
            "keff_relative_residual_tolerance", self.keff_relative_residual_tolerance
        )
        flux_nonnegativity_tolerance = require_finite_nonnegative_real(
            "flux_nonnegativity_tolerance", self.flux_nonnegativity_tolerance
        )
        _check_eigenvalue_iteration("eigenvalue_iteration", self.eigenvalue_iteration)
        object.__setattr__(self, "max_outer_iterations", max_outer_iterations)
        object.__setattr__(self, "keff_change_tolerance", keff_change_tolerance)
        object.__setattr__(self, "flux_change_tolerance", flux_change_tolerance)
        object.__setattr__(
            self, "keff_relative_residual_tolerance", keff_relative_residual_tolerance
        )
        object.__setattr__(
            self, "flux_nonnegativity_tolerance", flux_nonnegativity_tolerance
        )


def _check_linear_solve(name: str, value: object) -> None:
    """Require one supported immutable linear-solve policy."""
    if not isinstance(value, (DirectLinearSolveSettings, GmresLinearSolveSettings)):
        raise TypeError(
            f"{name} must be DirectLinearSolveSettings or GmresLinearSolveSettings"
        )


def _check_eigenvalue_iteration(name: str, value: object) -> None:
    """Require one supported immutable eigenvalue-iteration policy."""
    if not isinstance(value, (PowerIterationSettings, WielandtShiftSettings)):
        raise TypeError(
            f"{name} must be PowerIterationSettings or WielandtShiftSettings"
        )
