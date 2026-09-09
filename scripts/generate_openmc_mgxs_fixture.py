"""Generate Morana's synthetic OpenMC runtime-MGXS fixture.

Run in a separately managed environment that provides OpenMC. OpenMC
is deliberately a fixture-generation tool, not a Morana dependency. The
numerical values below are invented test values, not evaluated nuclear data or
benchmark results.
"""

from pathlib import Path

import numpy as np
import openmc

FIXTURE_PATH = (
    Path(__file__).parents[1] / "tests/morana/fixtures/openmc_runtime_mgxs_synthetic.h5"
)
GROUP_EDGES_EV = np.array([0.0, 1.0e3, 1.0e6, 2.0e7])


def _fuel() -> openmc.XSdata:
    """Return synthetic fissionable material data at two temperatures."""
    groups = openmc.mgxs.EnergyGroups(GROUP_EDGES_EV)
    fuel = openmc.XSdata("synthetic_fuel", groups, temperatures=[600.0, 900.0])
    fuel.order = 2

    for temperature, scale in ((600.0, 1.0), (900.0, 1.05)):
        fuel.set_total(scale * np.array([0.32, 0.57, 0.84]), temperature)
        fuel.set_absorption(scale * np.array([0.012, 0.024, 0.062]), temperature)
        fuel.set_scatter_matrix(
            scale
            * np.array(
                [
                    [[0.18, 0.036, 0.009], [0.09, 0.018, 0.004], [0.01, 0.002, 0.001]],
                    [[0.00, 0.000, 0.000], [0.31, 0.062, 0.012], [0.13, 0.026, 0.005]],
                    [[0.00, 0.000, 0.000], [0.00, 0.000, 0.000], [0.48, 0.096, 0.019]],
                ]
            ),
            temperature,
        )
        fuel.set_multiplicity_matrix(
            np.array([[1.0, 1.5, 0.0], [0.0, 1.0, 1.25], [0.0, 0.0, 1.0]]),
            temperature,
        )
        fuel.set_fission(scale * np.array([0.006, 0.019, 0.051]), temperature)
        fuel.set_nu_fission(scale * np.array([0.014, 0.043, 0.112]), temperature)
        fuel.set_kappa_fission(scale * np.array([1.20e6, 3.80e6, 10.10e6]), temperature)
        fuel.set_chi(np.array([0.72, 0.25, 0.03]), temperature)
        fuel.set_inverse_velocity(np.array([1.0e-7, 2.5e-7, 8.0e-7]), temperature)
    return fuel


def _transfer_fuel() -> openmc.XSdata:
    """Return synthetic fissionable data with general transfer production."""
    groups = openmc.mgxs.EnergyGroups(GROUP_EDGES_EV)
    fuel = openmc.XSdata("synthetic_transfer_fuel", groups, temperatures=[600.0])
    fuel.order = 2
    fuel.set_total(np.array([0.29, 0.53, 0.79]), temperature=600.0)
    fuel.set_absorption(np.array([0.011, 0.021, 0.058]), temperature=600.0)
    fuel.set_scatter_matrix(
        np.array(
            [
                [[0.16, 0.032, 0.008], [0.08, 0.016, 0.004], [0.01, 0.002, 0.001]],
                [[0.00, 0.000, 0.000], [0.29, 0.058, 0.012], [0.12, 0.024, 0.005]],
                [[0.00, 0.000, 0.000], [0.00, 0.000, 0.000], [0.44, 0.088, 0.018]],
            ]
        ),
        temperature=600.0,
    )
    fuel.set_fission(np.array([0.005, 0.017, 0.047]), temperature=600.0)
    fuel.set_nu_fission(
        np.array([[0.011, 0.003, 0.000], [0.029, 0.011, 0.002], [0.066, 0.026, 0.007]]),
        temperature=600.0,
    )
    fuel.set_kappa_fission(np.array([1.05e6, 3.40e6, 9.30e6]), temperature=600.0)
    fuel.set_inverse_velocity(np.array([1.1e-7, 2.6e-7, 8.2e-7]), temperature=600.0)
    return fuel


def _moderator() -> openmc.XSdata:
    """Return synthetic nonfissionable material data at one temperature."""
    groups = openmc.mgxs.EnergyGroups(GROUP_EDGES_EV)
    moderator = openmc.XSdata("synthetic_moderator", groups, temperatures=[600.0])
    moderator.order = 2
    moderator.set_total(np.array([0.41, 0.63, 1.09]), temperature=600.0)
    moderator.set_absorption(np.array([0.0004, 0.0012, 0.0075]), temperature=600.0)
    moderator.set_scatter_matrix(
        np.array(
            [
                [[0.27, 0.054, 0.011], [0.10, 0.020, 0.004], [0.01, 0.002, 0.001]],
                [[0.00, 0.000, 0.000], [0.44, 0.088, 0.018], [0.13, 0.026, 0.005]],
                [[0.00, 0.000, 0.000], [0.00, 0.000, 0.000], [0.86, 0.172, 0.034]],
            ]
        ),
        temperature=600.0,
    )
    moderator.set_inverse_velocity(
        np.array([1.3e-7, 3.1e-7, 9.4e-7]), temperature=600.0
    )
    return moderator


def main() -> None:
    """Write the checked-in OpenMC runtime-MGXS library fixture."""
    library = openmc.MGXSLibrary(openmc.mgxs.EnergyGroups(GROUP_EDGES_EV))
    library.add_xsdatas([_fuel(), _transfer_fuel(), _moderator()])
    library.export_to_hdf5(FIXTURE_PATH)


if __name__ == "__main__":
    main()
