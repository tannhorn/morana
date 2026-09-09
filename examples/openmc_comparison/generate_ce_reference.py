"""Generate continuous-energy mini-core production-profile references.

The reference is a bare 61-cell, 72-inch OpenMC eigenvalue calculation with
vacuum conditions on every exposed face.  Separate direct ``nu-fission``
tallies produce the 100-bin axial profile integrated over all 61 cells and
the 61-cell planar profile integrated over the full active height.  This
preserves the one-sigma uncertainty of each quantity used by the
OpenMC--Morana comparison.

Run this script from the repository root in an environment that provides
OpenMC and configured continuous-energy data.  It writes all generated inputs,
statepoints, and portable arrays below ``artifacts/`` by default.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import openmc

from artifact_paths import ARTIFACT_ROOT
from case import build_mini_core_model, configure_eigenvalue_settings
from geometry import (
    NUM_CORE_RINGS,
    active_height_cm,
    lattice_pitch_cm,
    mini_core_coordinates,
)
from run_settings import (
    EigenvalueRunSettings,
    active_history_label,
    newest_compatible_openmc_statepoint,
)
from specification import TEMPERATURE_K

DEFAULT_PARTICLES = 20_000
DEFAULT_BATCHES = 5_200
DEFAULT_INACTIVE = 200
DEFAULT_SEED = 31415
DEFAULT_AXIAL_BINS = 100
DEFAULT_KEFF_STANDARD_DEVIATION_PCM = 10.0
DEFAULT_PRODUCTION_RELATIVE_STANDARD_DEVIATION_PERCENT = 1.0
DEFAULT_OUTPUT_DIR = ARTIFACT_ROOT / "ce_reference"
TALLY_LAYOUT = "direct_profiles_v1"
AXIAL_TALLY_NAME = "axial_nu_fission"
PLANAR_TALLY_NAME = "planar_cell_nu_fission"


@dataclass(frozen=True)
class CeReferenceRunSettings(EigenvalueRunSettings):
    """Describe one reproducible CE production-profile calculation.

    The default controls define the 100-million-active-history reference. A
    higher-history request with compatible particle, inactive-batch, and RNG
    settings restarts from the largest qualifying lower-history statepoint.
    """

    particles: int = DEFAULT_PARTICLES
    batches: int = DEFAULT_BATCHES
    inactive: int = DEFAULT_INACTIVE
    seed: int = DEFAULT_SEED
    axial_bins: int = DEFAULT_AXIAL_BINS

    def __post_init__(self) -> None:
        """Check statistical controls and the axial tally discretization."""

        super().__post_init__()
        if not isinstance(self.axial_bins, int) or isinstance(self.axial_bins, bool):
            raise TypeError("axial_bins must be an integer")
        if self.axial_bins <= 0:
            raise ValueError("axial_bins must be positive")

    @property
    def active_histories(self) -> int:
        """Return the total number of active source-particle histories."""

        return self.particles * (self.batches - self.inactive)

    @property
    def artifact_label(self) -> str:
        """Return a collision-free label for this tally configuration."""

        label = (
            f"{active_history_label(self.active_histories)}_seed{self.seed}"
            f"_profiles_z{self.axial_bins}"
        )
        if (self.particles, self.inactive) != (
            DEFAULT_PARTICLES,
            DEFAULT_INACTIVE,
        ):
            label += f"_p{self.particles}_i{self.inactive}"
        return label

    def as_record(self) -> dict[str, int | str]:
        """Return JSON-ready stochastic and tally controls."""

        return {
            "particles": self.particles,
            "batches": self.batches,
            "inactive_batches": self.inactive,
            "active_histories": self.active_histories,
            "seed": self.seed,
            "axial_bins": self.axial_bins,
            "tally_layout": TALLY_LAYOUT,
        }

    def is_compatible_restart(self, record: dict[str, object]) -> bool:
        """Return whether ``record`` can restart to this larger request."""

        previous_batches = record.get("batches")
        return (
            record.get("particles") == self.particles
            and record.get("inactive_batches") == self.inactive
            and record.get("seed") == self.seed
            and record.get("axial_bins") == self.axial_bins
            and record.get("tally_layout") == TALLY_LAYOUT
            and isinstance(previous_batches, int)
            and not isinstance(previous_batches, bool)
            and self.inactive < previous_batches < self.batches
        )


def _root_cells_in_coordinate_order(
    model: openmc.Model,
) -> tuple[openmc.Cell, ...]:
    """Return root cells in the shared `(q, r)` coordinate order."""

    root_cells = model.geometry.root_universe.cells
    cells = []
    for q_coord, r_coord, _, _ in mini_core_coordinates():
        name = f"sre_full_pitch_q{q_coord:+d}_r{r_coord:+d}"
        matching = [cell for cell in root_cells.values() if cell.name == name]
        if len(matching) != 1:
            raise ValueError(f"could not identify one mini-core cell named {name!r}")
        cells.append(matching[0])
    return tuple(cells)


def _axial_mesh(axial_bins: int) -> tuple[openmc.RegularMesh, np.ndarray]:
    """Return a z-only tally mesh spanning the mini-core and its edges."""

    half_width = (NUM_CORE_RINGS - 0.5) * lattice_pitch_cm()
    axial_edges = np.linspace(0.0, active_height_cm(), axial_bins + 1)
    mesh = openmc.RegularMesh()
    mesh.lower_left = (-half_width, -half_width, axial_edges[0])
    mesh.upper_right = (half_width, half_width, axial_edges[-1])
    mesh.dimension = (1, 1, axial_bins)
    return mesh, axial_edges


def _add_production_tallies(
    model: openmc.Model,
    axial_bins: int,
) -> tuple[tuple[openmc.Cell, ...], np.ndarray]:
    """Attach direct axial and planar fission-neutron production tallies."""

    root_cells = _root_cells_in_coordinate_order(model)
    mesh, axial_edges = _axial_mesh(axial_bins)
    axial_tally = openmc.Tally(name=AXIAL_TALLY_NAME)
    axial_tally.filters = [openmc.MeshFilter(mesh)]
    axial_tally.scores = ["nu-fission"]
    planar_tally = openmc.Tally(name=PLANAR_TALLY_NAME)
    planar_tally.filters = [openmc.CellFilter(root_cells)]
    planar_tally.scores = ["nu-fission"]
    model.tallies.extend((axial_tally, planar_tally))
    return root_cells, axial_edges


def _restart_statepoint(
    settings: CeReferenceRunSettings,
    output_dir: Path,
) -> tuple[Path | None, int]:
    """Return the newest compatible lower-history statepoint, if available."""

    statepoint, previous_batches = newest_compatible_openmc_statepoint(
        output_dir,
        "reference_summary.json",
        settings.is_compatible_restart,
        openmc_version=openmc.__version__,
        cross_sections=str(openmc.config.get("cross_sections")),
    )
    if statepoint is None or previous_batches is None:
        return None, 0
    inherited_histories = settings.particles * (previous_batches - settings.inactive)
    return statepoint, inherited_histories


def _profile_arrays(
    statepoint: openmc.StatePoint,
    tally_name: str,
    expected_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one direct nu-fission profile and one-sigma uncertainties."""

    tally = statepoint.get_tally(name=tally_name)
    if tally.mean.shape != (expected_bins, 1, 1):
        raise ValueError(
            f"CE {tally_name!r} tally has unexpected shape "
            f"{tally.mean.shape}; expected ({expected_bins}, 1, 1)"
        )
    means = tally.mean[:, 0, 0]
    standard_deviations = tally.std_dev[:, 0, 0]
    if not np.all(np.isfinite(means)) or np.any(means < 0.0):
        raise ValueError(f"CE {tally_name!r} tally must be finite and nonnegative")
    if not np.all(np.isfinite(standard_deviations)) or np.any(
        standard_deviations < 0.0
    ):
        raise ValueError(
            f"CE {tally_name!r} uncertainties must be finite and nonnegative"
        )
    return means, standard_deviations


def _write_profile_arrays(
    output_dir: Path,
    production: np.ndarray,
    standard_deviations: np.ndarray,
    profile_name: str,
) -> tuple[Path, Path, Path, float]:
    """Write one direct profile and return paths and its maximum RSD."""

    total = float(np.sum(production))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"CE {profile_name} production tally has nonpositive total")
    normalized_production = production / total
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_standard_deviations = np.divide(
            standard_deviations,
            production,
            out=np.full_like(production, np.inf),
            where=production > 0.0,
        )
    mean_path = output_dir / f"{profile_name}_nu_fission_mean.npy"
    standard_deviation_path = output_dir / f"{profile_name}_nu_fission_std_dev.npy"
    normalized_path = output_dir / f"normalized_{profile_name}_production.npy"
    np.save(mean_path, production, allow_pickle=False)
    np.save(standard_deviation_path, standard_deviations, allow_pickle=False)
    np.save(normalized_path, normalized_production, allow_pickle=False)
    return (
        mean_path,
        standard_deviation_path,
        normalized_path,
        100.0 * float(np.max(relative_standard_deviations)),
    )


def _validate_precision_targets(
    keff_standard_deviation_pcm: float,
    production_relative_standard_deviation_percent: float,
) -> None:
    """Validate CE statistical precision limits."""

    for name, value in (
        ("keff_standard_deviation_pcm", keff_standard_deviation_pcm),
        (
            "production_relative_standard_deviation_percent",
            production_relative_standard_deviation_percent,
        ),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be a real number")
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")


def generate_ce_reference(
    *,
    settings: CeReferenceRunSettings,
    output_dir: Path,
    keff_standard_deviation_pcm: float = DEFAULT_KEFF_STANDARD_DEVIATION_PCM,
    production_relative_standard_deviation_percent: float = (
        DEFAULT_PRODUCTION_RELATIVE_STANDARD_DEVIATION_PERCENT
    ),
) -> Path:
    """Run or reuse one CE reference and write its summary.

    The axial profile has shape ``(axial_bins,)`` and increases from the
    physical bottom to the top of the 72-inch active interval. The planar
    profile has shape ``(61,)`` and follows :func:`geometry.mini_core_coordinates`.

    Parameters
    ----------
    settings
        Checked stochastic controls and axial tally resolution.
    output_dir
        Base artifact directory. A settings-specific child directory is used.
    keff_standard_deviation_pcm
        Acceptance limit recorded in the summary for the CE eigenvalue.
    production_relative_standard_deviation_percent
        Acceptance limit recorded for every directly tallied `nu-fission`
        profile bin.

    Returns
    -------
    pathlib.Path
        The generated ``reference_summary.json`` path. The summary records
        whether the sampled result meets both requested precision limits.
    """

    if not isinstance(settings, CeReferenceRunSettings):
        raise TypeError("settings must be a CeReferenceRunSettings")
    _validate_precision_targets(
        keff_standard_deviation_pcm,
        production_relative_standard_deviation_percent,
    )

    run_output_dir = settings.artifact_directory(output_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    statepoint_path = settings.statepoint_path(output_dir)
    summary_path = run_output_dir / "reference_summary.json"
    existing_summary: dict[str, object] = {}
    if statepoint_path.is_file() and summary_path.is_file():
        try:
            loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded_summary = None
        if (
            isinstance(loaded_summary, dict)
            and loaded_summary.get("statepoint") == statepoint_path.name
        ):
            existing_summary = loaded_summary
    model = build_mini_core_model(axial_height=active_height_cm())
    root_cells, axial_edges = _add_production_tallies(model, settings.axial_bins)
    configure_eigenvalue_settings(
        model,
        particles=settings.particles,
        batches=settings.batches,
        inactive=settings.inactive,
        seed=settings.seed,
    )
    inherited_histories = 0
    restart_path = None
    if not statepoint_path.is_file():
        model.export_to_xml(run_output_dir)
        restart_path, inherited_histories = _restart_statepoint(settings, output_dir)
        openmc.run(cwd=run_output_dir, restart_file=restart_path)
    else:
        recorded_histories = existing_summary.get("inherited_active_histories", 0)
        if isinstance(recorded_histories, int) and not isinstance(
            recorded_histories, bool
        ):
            inherited_histories = recorded_histories

    with openmc.StatePoint(statepoint_path) as statepoint:
        axial_production, axial_standard_deviations = _profile_arrays(
            statepoint,
            AXIAL_TALLY_NAME,
            settings.axial_bins,
        )
        planar_production, planar_standard_deviations = _profile_arrays(
            statepoint,
            PLANAR_TALLY_NAME,
            len(root_cells),
        )
        keff = float(statepoint.keff.nominal_value)
        keff_standard_deviation = float(statepoint.keff.std_dev)
        openmc_version = ".".join(str(part) for part in statepoint.version)
    (
        axial_mean_path,
        axial_standard_deviation_path,
        normalized_axial_path,
        maximum_axial_relative_standard_deviation_percent,
    ) = _write_profile_arrays(
        run_output_dir,
        axial_production,
        axial_standard_deviations,
        "axial",
    )
    (
        planar_mean_path,
        planar_standard_deviation_path,
        normalized_planar_path,
        maximum_planar_relative_standard_deviation_percent,
    ) = _write_profile_arrays(
        run_output_dir,
        planar_production,
        planar_standard_deviations,
        "planar",
    )
    axial_edges_path = run_output_dir / "axial_bin_edges_cm.npy"
    np.save(axial_edges_path, axial_edges, allow_pickle=False)
    restart_record = None
    if restart_path is not None:
        restart_record = str(Path("..") / restart_path.parent.name / restart_path.name)
    elif isinstance(existing_summary.get("restart_from"), str):
        restart_record = str(existing_summary["restart_from"])
    keff_standard_deviation_pcm_actual = keff_standard_deviation * 1.0e5
    accepted = (
        keff_standard_deviation_pcm_actual <= keff_standard_deviation_pcm
        and maximum_axial_relative_standard_deviation_percent
        <= production_relative_standard_deviation_percent
        and maximum_planar_relative_standard_deviation_percent
        <= production_relative_standard_deviation_percent
    )
    summary = {
        **settings.as_record(),
        "temperature_k": TEMPERATURE_K,
        "lattice_pitch_cm": lattice_pitch_cm(),
        "active_height_cm": active_height_cm(),
        "tallies": {
            "axial_profile": {
                "name": AXIAL_TALLY_NAME,
                "score": "nu-fission",
                "shape": list(axial_production.shape),
                "axis_order": ["axial_bin_bottom_to_top"],
                "axial_bin_edges_cm": axial_edges.tolist(),
            },
            "planar_profile": {
                "name": PLANAR_TALLY_NAME,
                "score": "nu-fission",
                "shape": list(planar_production.shape),
                "axis_order": ["planar_cell"],
                "planar_coordinates": [
                    [q_coord, r_coord]
                    for q_coord, r_coord, _, _ in mini_core_coordinates()
                ],
                "root_cell_ids": [cell.id for cell in root_cells],
            },
        },
        "keff": keff,
        "keff_standard_deviation": keff_standard_deviation,
        "keff_standard_deviation_pcm": keff_standard_deviation_pcm_actual,
        "maximum_nu_fission_relative_standard_deviation_percent": (
            max(
                maximum_axial_relative_standard_deviation_percent,
                maximum_planar_relative_standard_deviation_percent,
            )
        ),
        "maximum_axial_nu_fission_relative_standard_deviation_percent": (
            maximum_axial_relative_standard_deviation_percent
        ),
        "maximum_planar_nu_fission_relative_standard_deviation_percent": (
            maximum_planar_relative_standard_deviation_percent
        ),
        "precision_targets": {
            "keff_standard_deviation_pcm": keff_standard_deviation_pcm,
            "nu_fission_relative_standard_deviation_percent": (
                production_relative_standard_deviation_percent
            ),
        },
        "meets_precision_targets": accepted,
        "inherited_active_histories": inherited_histories,
        "restart_from": restart_record,
        "statepoint": statepoint_path.name,
        "axial_nu_fission_mean": axial_mean_path.name,
        "axial_nu_fission_std_dev": axial_standard_deviation_path.name,
        "normalized_axial_production": normalized_axial_path.name,
        "planar_nu_fission_mean": planar_mean_path.name,
        "planar_nu_fission_std_dev": planar_standard_deviation_path.name,
        "normalized_planar_production": normalized_planar_path.name,
        "axial_bin_edges_cm": axial_edges_path.name,
        "openmc_version": openmc_version,
        "cross_sections": (
            existing_summary.get("cross_sections")
            if isinstance(existing_summary.get("cross_sections"), str)
            else str(openmc.config.get("cross_sections"))
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary_path


def _arguments() -> tuple[CeReferenceRunSettings, Path, float, float]:
    """Parse CE-reference run controls and precision targets."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=DEFAULT_PARTICLES)
    parser.add_argument("--batches", type=int, default=DEFAULT_BATCHES)
    parser.add_argument("--inactive", type=int, default=DEFAULT_INACTIVE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--axial-bins", type=int, default=DEFAULT_AXIAL_BINS)
    parser.add_argument(
        "--keff-standard-deviation-pcm",
        type=float,
        default=DEFAULT_KEFF_STANDARD_DEVIATION_PCM,
    )
    parser.add_argument(
        "--production-relative-standard-deviation-percent",
        type=float,
        default=DEFAULT_PRODUCTION_RELATIVE_STANDARD_DEVIATION_PERCENT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()
    try:
        settings = CeReferenceRunSettings(
            particles=arguments.particles,
            batches=arguments.batches,
            inactive=arguments.inactive,
            seed=arguments.seed,
            axial_bins=arguments.axial_bins,
        )
        _validate_precision_targets(
            arguments.keff_standard_deviation_pcm,
            arguments.production_relative_standard_deviation_percent,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    return (
        settings,
        arguments.output_dir,
        arguments.keff_standard_deviation_pcm,
        arguments.production_relative_standard_deviation_percent,
    )


def main() -> None:
    """Generate the requested CE reference and report its precision summary."""

    (
        settings,
        output_dir,
        keff_standard_deviation_pcm,
        production_relative_standard_deviation_percent,
    ) = _arguments()
    summary_path = generate_ce_reference(
        settings=settings,
        output_dir=output_dir,
        keff_standard_deviation_pcm=keff_standard_deviation_pcm,
        production_relative_standard_deviation_percent=(
            production_relative_standard_deviation_percent
        ),
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"CE reference summary: {summary_path}")
    print(
        f"k_eff={summary['keff']:.7f} +/- " f"{summary['keff_standard_deviation']:.7f}"
    )
    axial_rsd = summary["maximum_axial_nu_fission_relative_standard_deviation_percent"]
    planar_rsd = summary[
        "maximum_planar_nu_fission_relative_standard_deviation_percent"
    ]
    print(
        "Maximum axial/planar nu-fission relative standard deviation [%]: "
        f"{axial_rsd:.4f}/{planar_rsd:.4f}"
    )
    print(f"Meets precision targets: {summary['meets_precision_targets']}")


if __name__ == "__main__":
    main()
