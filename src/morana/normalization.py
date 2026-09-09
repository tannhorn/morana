"""Physical normalization declarations for fission eigenvalue results."""

from __future__ import annotations

from dataclasses import dataclass
from morana._validation import require_finite_positive_real


@dataclass(frozen=True)
class FissionSourceNormalization:
    """Scale a converged eigenfunction to a target fission-source rate.

    Parameters
    ----------
    rate
        Finite positive total fission-neutron source rate in ``n / s``. The
        target scales the returned flux and balance terms but does not alter
        the returned ``k_eff``. Boolean values are not accepted.

    Raises
    ------
    TypeError
        If ``rate`` is not a real number or is Boolean.
    ValueError
        If ``rate`` is not finite and positive after conversion to ``float``.

    Notes
    -----
    Use `PowerNormalization` when recoverable fission-energy production
    cross sections are available and a thermal-power target is required.
    """

    rate: float

    def __post_init__(self) -> None:
        """Require one finite positive source-rate target."""
        rate = require_finite_positive_real("rate", self.rate)
        object.__setattr__(self, "rate", rate)


@dataclass(frozen=True)
class PowerNormalization:
    """Scale a converged eigenfunction to a target recoverable thermal power.

    Parameters
    ----------
    power
        Finite positive target recoverable thermal power in W (J / s). Every
        fissionable material in the active domain must provide
        ``CrossSections.fission.kappa_sigma_f`` for this normalization to be
        usable.
        Boolean values are not accepted.

    Raises
    ------
    TypeError
        If ``power`` is not a real number or is Boolean.
    ValueError
        If ``power`` is not finite and positive after conversion to ``float``.

    Notes
    -----
    The target scales the returned flux and neutron-balance terms but does
    not alter the returned ``k_eff``. ``kappa_sigma_f`` directly represents
    recoverable fission-energy production, avoiding an implied or fixed value
    for neutrons emitted per fission.
    """

    power: float

    def __post_init__(self) -> None:
        """Require one finite positive thermal-power target."""
        power = require_finite_positive_real("power", self.power)
        object.__setattr__(self, "power", power)
