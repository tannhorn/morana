"""Prepare MGXS data and run Morana study cases.

One invocation owns an active-history count and RNG seed. A single OpenMC run
generates all four CASMO libraries, and each library is reused for every
requested axial mesh because Morana solves are cheap compared with transport.

Use ``--stage`` to run both stages in one environment or to split MGXS
generation and Morana execution between environments. Pass a
precision-qualified CE summary with ``--ce-reference`` to validate it and
record a checked comparison for every Morana case. This runner does not
generate the CE reference; see the directory's ``README.md`` for the staged
procedure.
"""

# Matplotlib configuration must precede Morana's plotting imports.
# pylint: disable=wrong-import-position,wrong-import-order

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from examples.openmc_comparison.artifact_paths import (
    ARTIFACT_ROOT,
    configure_matplotlib_cache,
)
from examples.openmc_comparison.compare_study_cases import compare_morana_to_ce
from examples.openmc_comparison.geometry import (
    NUM_CORE_RINGS,
    active_height_cm,
    lattice_pitch_cm,
    mini_core_coordinates,
)
from examples.openmc_comparison.mgxs_generation import (
    DEFAULT_GENERATIONS_PER_BATCH,
    DEFAULT_INACTIVE,
    DEFAULT_PARTICLES,
    DEFAULT_SEED,
    MgxsRunSettings,
    RUNTIME_DATASET,
    SUPPORTED_GROUP_STRUCTURES,
    TALLIED_GROUP_STRUCTURE,
)
from examples.openmc_comparison.specification import TEMPERATURE_K

if TYPE_CHECKING:
    from morana import (
        CrossSections,
        HexPlanarMesh,
        MaterialMesh,
        ProblemConfiguration,
    )

DEFAULT_OUTPUT_DIR = ARTIFACT_ROOT / "study"
DEFAULT_GROUP_STRUCTURES = (TALLIED_GROUP_STRUCTURE,)
DEFAULT_AXIAL_LAYERS = (20,)


def _libraries_are_reusable(
    settings: MgxsRunSettings,
    mgxs_output_dir: Path,
    runtime_paths: dict[str, Path],
) -> bool:
    """Return whether all libraries use the requested run configuration."""

    if not all(path.is_file() for path in runtime_paths.values()):
        return False
    summary_path = (
        settings.artifact_directory(mgxs_output_dir) / "generation_summary.json"
    )
    try:
        record = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected_libraries = {
        structure: path.name for structure, path in runtime_paths.items()
    }
    return (
        isinstance(record, dict)
        and settings.matches_record(record)
        and record.get("runtime_libraries") == expected_libraries
    )


def _configuration(
    cross_sections: CrossSections,
    axial_layers: int,
) -> ProblemConfiguration:
    """Build the homogeneous 61-cell Morana mini-core."""

    # Morana remains optional when this entry point runs only the MGXS stage.
    # pylint: disable-next=import-outside-toplevel
    from morana import (
        BoundaryCondition,
        BoundaryConditionSet,
        HexPlanarMesh,
        Material,
        MaterialMesh,
        MaterialSlice,
        ProblemConfiguration,
    )

    mesh = HexPlanarMesh(num_rings=NUM_CORE_RINGS, pitch=lattice_pitch_cm())
    material = Material("homogenized_cell", xs=cross_sections, color="#7f7f7f")
    layer = MaterialSlice(
        mesh,
        {index: material.name for index in mesh.openmc_indices},
        active_height_cm() / axial_layers,
    )
    return ProblemConfiguration(
        mesh=mesh,
        materials={material.name: material},
        material_mesh=MaterialMesh.stack(layer.extrude(count=axial_layers)),
        boundary=BoundaryConditionSet(
            BoundaryCondition.vacuum().on_radial(),
            BoundaryCondition.vacuum().on_bottom(),
            BoundaryCondition.vacuum().on_top(),
        ),
        name=f"openmc_comparison_z{axial_layers}",
    )


def _normalized_production(
    flux: tuple[np.ndarray, ...],
    cross_sections: CrossSections,
    material_mesh: MaterialMesh,
) -> np.ndarray:
    """Return volume-integrated cell production normalized to unit total."""

    if cross_sections.fission is None:
        raise ValueError("the homogenized MGXS record must contain fission data")
    production_xs = cross_sections.fission.fission_production
    production = np.stack(
        [
            production_xs @ layer_flux * material_mesh.cell_volume(axial_index)
            for axial_index, layer_flux in enumerate(flux)
        ]
    )
    total = float(np.sum(production))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("Morana returned nonpositive total fission production")
    return production / total


def _shared_planar_production_order(
    production: np.ndarray,
    mesh: HexPlanarMesh,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    """Reorder Morana planar columns into the shared mini-core order.

    Morana uses OpenMC-compatible outer-to-inner ring IDs internally.  The CE
    root-cell tally uses the shared ``(q, r)`` coordinate scan, so portable
    comparison artifacts must reorder both their production columns and their
    coordinate metadata to that shared order.
    """

    coordinates = tuple(
        (q_coord, r_coord) for q_coord, r_coord, _, _ in mini_core_coordinates()
    )
    try:
        planar_ids = tuple(mesh.index_of[coordinate] for coordinate in coordinates)
    except KeyError as error:
        raise ValueError(
            "Morana mesh does not contain every shared mini-core coordinate"
        ) from error
    if production.shape[1] != len(planar_ids):
        raise ValueError("Morana production does not span the shared mini-core")
    return production[:, planar_ids], coordinates


def _run_morana(
    arguments: argparse.Namespace,
    settings: MgxsRunSettings,
    run_directory: Path,
    runtime_paths: dict[str, Path],
) -> None:
    """Run and record every requested group and axial discretization."""

    configure_matplotlib_cache()
    # pylint: disable-next=import-outside-toplevel
    from morana import (
        CrossSections,
        FissionSourceNormalization,
        KeffSettings,
    )

    # pylint: disable-next=import-outside-toplevel
    from morana.solvers.finite_volume import solve_keff

    for group_structure in arguments.group_structures:
        mgxs_path = runtime_paths[group_structure]
        cross_sections = CrossSections.from_openmc_mgxs_hdf5(
            mgxs_path,
            RUNTIME_DATASET,
            TEMPERATURE_K,
            diffusion="total",
        )
        for axial_layers in arguments.axial_layers:
            case_directory = (
                run_directory / group_structure.lower() / f"z{axial_layers}"
            )
            case_directory.mkdir(parents=True, exist_ok=True)
            configuration = _configuration(
                cross_sections,
                axial_layers,
            )
            started = perf_counter()
            result = solve_keff(
                configuration,
                FissionSourceNormalization(rate=1.0),
                KeffSettings(max_outer_iterations=5_000),
            )
            elapsed_seconds = perf_counter() - started
            production = _normalized_production(
                result.flux,
                cross_sections,
                configuration.material_mesh,
            )
            production, planar_coordinates = _shared_planar_production_order(
                production,
                configuration.mesh,
            )
            result_path = case_directory / "result.morana-result"
            production_path = case_directory / "normalized_production.npy"
            summary_path = case_directory / "summary.json"
            result.save_to_disk(result_path)
            np.save(production_path, production, allow_pickle=False)
            relative_mgxs_path = os.path.relpath(mgxs_path, start=case_directory)
            record: dict[str, object] = {
                "temperature_k": TEMPERATURE_K,
                "lattice_pitch_cm": lattice_pitch_cm(),
                "active_height_cm": active_height_cm(),
                "group_structure": group_structure,
                "groups": cross_sections.groups,
                "active_histories": settings.active_histories,
                "seed": settings.seed,
                "particles": settings.particles,
                "generations_per_batch": settings.generations_per_batch,
                "inactive_batches": settings.inactive,
                "total_batches": settings.batches,
                "axial_layers": axial_layers,
                "diffusion_convention": "openmc-transport",
                "keff": result.keff,
                "morana_elapsed_seconds": elapsed_seconds,
                "mgxs_path": relative_mgxs_path,
                "result_path": result_path.name,
                "normalized_production_path": production_path.name,
                "planar_coordinates": [list(coords) for coords in planar_coordinates],
            }
            summary_path.write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
            if arguments.ce_reference is not None:
                ce_reference_path = arguments.ce_reference.expanduser().resolve()
                record["ce_comparison"] = compare_morana_to_ce(
                    summary_path,
                    ce_reference_path,
                )
                summary_path.write_text(
                    json.dumps(record, indent=2) + "\n", encoding="utf-8"
                )
            print(
                f"Morana {group_structure}, z={axial_layers}: "
                f"k_eff={result.keff:.9f}; summary={summary_path}"
            )


def _arguments() -> tuple[argparse.Namespace, MgxsRunSettings]:
    """Parse one stochastic-sample execution request."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group-structures",
        choices=SUPPORTED_GROUP_STRUCTURES,
        nargs="+",
        default=DEFAULT_GROUP_STRUCTURES,
        help=(
            "Libraries to solve with Morana; OpenMC tallies CASMO-70 and "
            "exports all four supported structures."
        ),
    )
    parser.add_argument(
        "--active-histories",
        type=int,
        default=MgxsRunSettings().active_histories,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--axial-layers",
        type=int,
        nargs="+",
        default=DEFAULT_AXIAL_LAYERS,
    )
    parser.add_argument("--particles", type=int, default=DEFAULT_PARTICLES)
    parser.add_argument("--inactive", type=int, default=DEFAULT_INACTIVE)
    parser.add_argument(
        "--generations-per-batch",
        type=int,
        default=DEFAULT_GENERATIONS_PER_BATCH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--ce-reference",
        type=Path,
        help=(
            "Checked CE reference summary to compare with and record in every "
            "Morana case summary."
        ),
    )
    parser.add_argument(
        "--force-mgxs",
        action="store_true",
        help="Regenerate an existing history/seed MGXS sample.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "mgxs", "morana"),
        default="all",
        help=(
            "Run both stages, prepare/reuse MGXS only, or solve with existing "
            "MGXS only (default: all)."
        ),
    )
    arguments = parser.parse_args()
    if any(layer <= 0 for layer in arguments.axial_layers):
        parser.error("--axial-layers values must be positive")
    if len(set(arguments.axial_layers)) != len(arguments.axial_layers):
        parser.error("--axial-layers values must be unique")
    if len(set(arguments.group_structures)) != len(arguments.group_structures):
        parser.error("--group-structures values must be unique")
    try:
        settings = MgxsRunSettings.from_active_histories(
            active_histories=arguments.active_histories,
            seed=arguments.seed,
            particles=arguments.particles,
            inactive=arguments.inactive,
            generations_per_batch=arguments.generations_per_batch,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    return arguments, settings


def main() -> None:
    """Generate/reuse MGXS, execute Morana, and write durable case records."""

    arguments, settings = _arguments()
    output_directory = arguments.output_dir.expanduser().resolve()
    run_directory = output_directory / "morana" / settings.artifact_label
    mgxs_output_dir = output_directory / "mgxs"
    runtime_paths = {
        structure: settings.runtime_library_path(mgxs_output_dir, structure)
        for structure in SUPPORTED_GROUP_STRUCTURES
    }
    libraries_exist = _libraries_are_reusable(
        settings,
        mgxs_output_dir,
        runtime_paths,
    )
    generate_libraries = arguments.stage != "morana" and (
        arguments.force_mgxs or not libraries_exist
    )
    if generate_libraries:
        # OpenMC remains optional for the Morana-only stage and when a complete
        # generated library set is reused.
        # pylint: disable-next=import-outside-toplevel
        from generate_mgxs import generate_mgxs

        runtime_paths = generate_mgxs(
            settings=settings,
            output_dir=mgxs_output_dir,
        )
        libraries_exist = True
    elif libraries_exist:
        mgxs_directory = settings.artifact_directory(mgxs_output_dir)
        print(f"Reusing MGXS libraries in: {mgxs_directory}")
    else:
        missing = ", ".join(
            str(path) for path in runtime_paths.values() if not path.is_file()
        )
        raise FileNotFoundError(
            "the Morana-only stage requires all MGXS libraries; missing: " + missing
        )
    if arguments.stage != "mgxs":
        run_directory.mkdir(parents=True, exist_ok=True)
        _run_morana(arguments, settings, run_directory, runtime_paths)


if __name__ == "__main__":
    main()
