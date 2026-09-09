"""Generate the lattice-unit-cell MGXS libraries.

Run from the repository root with an active Python environment that provides
OpenMC. The continuous-energy transport calculation and every generated OpenMC
artifact are written outside the tracked source tree by default:

``python examples/openmc_comparison/generate_mgxs.py``

One transport calculation tallies the CASMO-70 group structure. OpenMC then
condenses those results to CASMO-8, CASMO-25, and CASMO-40. Each exported
library contains
one macroscopic record homogenized over the complete lattice unit cell and is
compatible with ``CrossSections.from_openmc_mgxs_hdf5`` at the case
temperature.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import openmc

from artifact_paths import ARTIFACT_ROOT
from case import build_unit_cell_model, configure_eigenvalue_settings
from mgxs_generation import (
    DEFAULT_BATCHES,
    DEFAULT_GENERATIONS_PER_BATCH,
    DEFAULT_INACTIVE,
    DEFAULT_PARTICLES,
    DEFAULT_SEED,
    MgxsRunSettings,
    RUNTIME_DATASET,
    SUPPORTED_GROUP_STRUCTURES,
    TALLIED_GROUP_STRUCTURE,
)
from specification import TEMPERATURE_K
from run_settings import newest_compatible_openmc_statepoint

DEFAULT_OUTPUT_DIR = ARTIFACT_ROOT / "mgxs"
MGXS_TYPES = (
    "transport",
    "absorption",
    "consistent scatter matrix",
    "multiplicity matrix",
    "nu-fission matrix",
)


def _energy_groups(structure: str) -> openmc.mgxs.EnergyGroups:
    """Return the requested OpenMC built-in CASMO group structure."""

    return openmc.mgxs.EnergyGroups(openmc.mgxs.GROUP_STRUCTURES[structure])


def _build_library(
    model: openmc.Model,
    energy_groups: openmc.mgxs.EnergyGroups,
) -> openmc.mgxs.Library:
    """Attach complete-lattice-unit-cell universe-domain tallies to ``model``.

    The unit-cell universe, rather than its constituent materials, is the one
    tally domain.  Consequently the exported record includes the volume- and
    flux-weighted contribution of fuel, structural materials, graphite, and
    apportioned inter-can sodium across the whole transverse lattice pitch.
    """

    library = openmc.mgxs.Library(
        model.geometry,
        by_nuclide=False,
        mgxs_types=MGXS_TYPES,
    )
    library.domain_type = "universe"
    library.domains = [model.geometry.root_universe]
    library.energy_groups = energy_groups
    library.correction = None
    library.scatter_format = "legendre"
    library.legendre_order = 0
    library.build_library()
    library.add_to_tallies(model.tallies, merge=True)
    return library


def _export_runtime_library(
    library: openmc.mgxs.Library,
    output_path: Path,
) -> None:
    """Write one explicitly temperature-labelled MGXS record."""

    domain = library.domains[0]
    runtime_data = openmc.XSdata(
        RUNTIME_DATASET,
        library.energy_groups,
        temperatures=[TEMPERATURE_K],
    )
    runtime_data.order = 0
    mgxs_arguments = {
        "temperature": TEMPERATURE_K,
        "nuclide": "total",
        "xs_type": "macro",
        "subdomain": "all",
    }
    # OpenMC's TransportXS owns the P1 in-scatter correction. The runtime
    # format stores either TotalXS or TransportXS in its common `total` field.
    runtime_data.set_total_mgxs(library.get_mgxs(domain, "transport"), **mgxs_arguments)
    runtime_data.set_absorption_mgxs(
        library.get_mgxs(domain, "absorption"), **mgxs_arguments
    )
    runtime_data.set_scatter_matrix_mgxs(
        library.get_mgxs(domain, "consistent scatter matrix"), **mgxs_arguments
    )
    runtime_data.set_multiplicity_matrix_mgxs(
        library.get_mgxs(domain, "multiplicity matrix"), **mgxs_arguments
    )
    runtime_data.set_nu_fission_mgxs(
        library.get_mgxs(domain, "nu-fission matrix"), **mgxs_arguments
    )
    runtime_library = openmc.MGXSLibrary(library.energy_groups)
    runtime_library.add_xsdatas([runtime_data])
    runtime_library.export_to_hdf5(output_path)


def _arguments() -> tuple[MgxsRunSettings, Path, bool]:
    """Parse command-line arguments for the MGXS production calculation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--particles",
        type=int,
        default=DEFAULT_PARTICLES,
        help=f"Source particles per generation (default: {DEFAULT_PARTICLES:,}).",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=DEFAULT_BATCHES,
        help=f"Total batches (default: {DEFAULT_BATCHES}).",
    )
    parser.add_argument(
        "--inactive",
        type=int,
        default=DEFAULT_INACTIVE,
        help=f"Inactive batches (default: {DEFAULT_INACTIVE}).",
    )
    parser.add_argument(
        "--generations-per-batch",
        type=int,
        default=DEFAULT_GENERATIONS_PER_BATCH,
        help=(
            "Fission generations per statistical batch "
            f"(default: {DEFAULT_GENERATIONS_PER_BATCH})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--from-statepoint",
        action="store_true",
        help=(
            "Re-export every supported group structure from the statepoint "
            "selected by these controls, without rerunning transport."
        ),
    )
    args = parser.parse_args()
    try:
        settings = MgxsRunSettings(
            particles=args.particles,
            batches=args.batches,
            inactive=args.inactive,
            generations_per_batch=args.generations_per_batch,
            seed=args.seed,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    return settings, args.output_dir, args.from_statepoint


def _restart_statepoint(
    settings: MgxsRunSettings,
    output_dir: Path,
) -> tuple[Path | None, int]:
    """Return the newest compatible lower-history statepoint, if one exists."""

    statepoint, previous_batches = newest_compatible_openmc_statepoint(
        output_dir,
        "generation_summary.json",
        settings.is_compatible_restart,
        openmc_version=openmc.__version__,
        cross_sections=str(openmc.config.get("cross_sections")),
    )
    if statepoint is None or previous_batches is None:
        return None, 0
    inherited_histories = (
        settings.particles
        * settings.generations_per_batch
        * (previous_batches - settings.inactive)
    )
    return statepoint, inherited_histories


def _runtime_paths(settings: MgxsRunSettings, output_dir: Path) -> dict[str, Path]:
    """Return paths for every supported runtime-MGXS library."""

    return {
        structure: settings.runtime_library_path(output_dir, structure)
        for structure in SUPPORTED_GROUP_STRUCTURES
    }


def _load_summary(summary_path: Path) -> dict[str, object]:
    """Read one checked JSON generation summary."""

    try:
        record = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read generation summary: {summary_path}") from error
    if not isinstance(record, dict):
        raise ValueError("generation summary must contain a JSON object")
    return record


def _export_from_statepoint(
    fine_library: openmc.mgxs.Library,
    statepoint_path: Path,
    summary_path: Path,
    runtime_paths: dict[str, Path],
) -> None:
    """Load a fine-group statepoint and export every supported condensation."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", openmc.IDWarning)
        with openmc.StatePoint(statepoint_path) as statepoint:
            if statepoint.summary is None:
                statepoint.link_with_summary(summary_path)
            fine_library.load_from_statepoint(statepoint)
            for structure, runtime_path in runtime_paths.items():
                library = (
                    fine_library
                    if structure == TALLIED_GROUP_STRUCTURE
                    else fine_library.get_condensed_library(_energy_groups(structure))
                )
                _export_runtime_library(library, runtime_path)


def _write_generation_summary(
    settings: MgxsRunSettings,
    run_output_dir: Path,
    statepoint_path: Path,
    runtime_paths: dict[str, Path],
    *,
    inherited_histories: int,
    restart_record: str | None,
    openmc_version: str,
    cross_sections: str,
) -> None:
    """Write provenance for the statepoint and its complete export set."""

    summary = {
        **settings.as_record(),
        "inherited_active_histories": inherited_histories,
        "restart_from": restart_record,
        "statepoint": statepoint_path.name,
        "runtime_libraries": {
            structure: path.name for structure, path in runtime_paths.items()
        },
        "openmc_version": openmc_version,
        "cross_sections": cross_sections,
    }
    summary_path = run_output_dir / "generation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def reexport_mgxs_from_statepoint(
    *,
    settings: MgxsRunSettings,
    output_dir: Path,
) -> dict[str, Path]:
    """Export all supported libraries from one existing compatible statepoint.

    This post-processes the recorded CASMO-70 tally data. It does not invoke
    :func:`openmc.run` and therefore does not add particle histories.
    """

    if not isinstance(settings, MgxsRunSettings):
        raise TypeError("settings must be an MgxsRunSettings")
    run_output_dir = settings.artifact_directory(output_dir)
    summary_path = run_output_dir / "generation_summary.json"
    record = _load_summary(summary_path)
    if not settings.matches_tally_record(record):
        raise ValueError("generation summary is incompatible with requested tally")
    statepoint_path = settings.statepoint_path(output_dir)
    if not statepoint_path.is_file():
        raise FileNotFoundError(f"selected statepoint is missing: {statepoint_path}")
    summary_hdf5_path = run_output_dir / "summary.h5"
    if not summary_hdf5_path.is_file():
        raise FileNotFoundError(f"OpenMC summary is missing: {summary_hdf5_path}")

    model = build_unit_cell_model()
    configure_eigenvalue_settings(
        model,
        particles=settings.particles,
        batches=settings.batches,
        inactive=settings.inactive,
        generations_per_batch=settings.generations_per_batch,
        seed=settings.seed,
    )
    fine_library = _build_library(model, _energy_groups(TALLIED_GROUP_STRUCTURE))
    runtime_paths = _runtime_paths(settings, output_dir)
    _export_from_statepoint(
        fine_library,
        statepoint_path,
        summary_hdf5_path,
        runtime_paths,
    )
    inherited_histories = record.get("inherited_active_histories", 0)
    if not isinstance(inherited_histories, int) or isinstance(
        inherited_histories, bool
    ):
        raise ValueError("generation summary has invalid inherited history count")
    restart_record = record.get("restart_from")
    if restart_record is not None and not isinstance(restart_record, str):
        raise ValueError("generation summary has invalid restart path")
    openmc_version = record.get("openmc_version")
    cross_sections = record.get("cross_sections")
    if not isinstance(openmc_version, str) or not openmc_version:
        raise ValueError("generation summary has invalid OpenMC version")
    if not isinstance(cross_sections, str) or not cross_sections:
        raise ValueError("generation summary has invalid cross-sections provenance")
    _write_generation_summary(
        settings,
        run_output_dir,
        statepoint_path,
        runtime_paths,
        inherited_histories=inherited_histories,
        restart_record=restart_record,
        openmc_version=openmc_version,
        cross_sections=cross_sections,
    )
    return runtime_paths


def generate_mgxs(
    *,
    settings: MgxsRunSettings,
    output_dir: Path,
) -> dict[str, Path]:
    """Run one lattice-unit-cell CE tally and return all runtime libraries.

    Parameters
    ----------
    settings
        Checked statistical-control and RNG declaration.
    output_dir
        Base artifact directory. A history- and seed-specific child directory
        is created below it.

    Returns
    -------
    dict[str, pathlib.Path]
        Generated OpenMC runtime-MGXS HDF5 paths keyed by group structure.
    """

    if not isinstance(settings, MgxsRunSettings):
        raise TypeError("settings must be an MgxsRunSettings")
    run_output_dir = settings.artifact_directory(output_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    model = build_unit_cell_model()
    configure_eigenvalue_settings(
        model,
        particles=settings.particles,
        batches=settings.batches,
        inactive=settings.inactive,
        generations_per_batch=settings.generations_per_batch,
        seed=settings.seed,
    )
    fine_library = _build_library(model, _energy_groups(TALLIED_GROUP_STRUCTURE))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", openmc.IDWarning)
        model.export_to_xml(run_output_dir)
    restart_path, inherited_histories = _restart_statepoint(settings, output_dir)
    openmc.run(cwd=run_output_dir, restart_file=restart_path)

    statepoint_path = settings.statepoint_path(output_dir)
    runtime_paths = _runtime_paths(settings, output_dir)
    _export_from_statepoint(
        fine_library,
        statepoint_path,
        run_output_dir / "summary.h5",
        runtime_paths,
    )
    restart_record = None
    if restart_path is not None:
        restart_record = str(Path("..") / restart_path.parent.name / restart_path.name)
    _write_generation_summary(
        settings,
        run_output_dir,
        statepoint_path,
        runtime_paths,
        inherited_histories=inherited_histories,
        restart_record=restart_record,
        openmc_version=openmc.__version__,
        cross_sections=str(openmc.config.get("cross_sections")),
    )
    return runtime_paths


def main() -> None:
    """Parse CLI inputs, generate all MGXS libraries, and report their paths."""

    settings, output_base, from_statepoint = _arguments()
    if from_statepoint:
        runtime_paths = reexport_mgxs_from_statepoint(
            settings=settings,
            output_dir=output_base,
        )
    else:
        runtime_paths = generate_mgxs(
            settings=settings,
            output_dir=output_base,
        )
    output_dir = settings.artifact_directory(output_base)
    statepoint_path = settings.statepoint_path(output_base)

    print(f"Tallied energy-group structure: {TALLIED_GROUP_STRUCTURE}")
    print("Exported energy-group structures:", ", ".join(SUPPORTED_GROUP_STRUCTURES))
    print(
        "Production histories:",
        settings.active_histories,
    )
    print(f"Unit-cell runtime dataset: {RUNTIME_DATASET}")
    print(f"Stored temperature [K]: {TEMPERATURE_K:.2f}")
    print("Statepoint:", statepoint_path)
    for structure, runtime_path in runtime_paths.items():
        print(f"{structure} runtime-MGXS library:", runtime_path)
    print("Generation summary:", output_dir / "generation_summary.json")


if __name__ == "__main__":
    main()
