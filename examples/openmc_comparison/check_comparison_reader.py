"""Check the CE comparison reader with deterministic temporary artifacts.

Run from the repository root:

``python -m examples.openmc_comparison.check_comparison_reader``

The check creates no persistent artifacts. It verifies exact axial rebinning,
raw-to-normalized profile consistency, CE provenance reporting, rejection of
a noncanonical 61-cell order, and the Morana-to-shared-order production
conversion used by the study runner.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from examples.openmc_comparison.artifact_paths import configure_matplotlib_cache
from examples.openmc_comparison.compare_study_cases import (
    CE_AXIAL_BINS,
    CE_PLANAR_CELLS,
    CE_TALLY_LAYOUT,
    EXPECTED_PLANAR_COORDINATES,
    compare_morana_to_ce,
)
from examples.openmc_comparison.geometry import (
    NUM_CORE_RINGS,
    active_height_cm,
    lattice_pitch_cm,
)
from examples.openmc_comparison.run_study_case import (
    _shared_planar_production_order,
)
from examples.openmc_comparison.specification import TEMPERATURE_K

configure_matplotlib_cache()

# Morana imports Matplotlib, so the cache must be configured first.
from morana import HexPlanarMesh  # pylint: disable=wrong-import-position


def _write_reference(directory: Path, coordinates: list[list[int]]) -> Path:
    """Write one self-consistent 100-bin CE reference fixture."""

    axial_mean = np.linspace(1.0, 2.0, CE_AXIAL_BINS)
    planar_mean = np.linspace(1.0, 2.0, CE_PLANAR_CELLS)
    arrays = {
        "axial_nu_fission_mean": axial_mean,
        "axial_nu_fission_std_dev": axial_mean * 0.001,
        "normalized_axial_production": axial_mean / np.sum(axial_mean),
        "planar_nu_fission_mean": planar_mean,
        "planar_nu_fission_std_dev": planar_mean * 0.001,
        "normalized_planar_production": planar_mean / np.sum(planar_mean),
        "axial_bin_edges_cm": np.linspace(
            0.0,
            active_height_cm(),
            CE_AXIAL_BINS + 1,
        ),
    }
    for name, values in arrays.items():
        np.save(directory / f"{name}.npy", values, allow_pickle=False)
    summary: dict[str, object] = {
        "tally_layout": CE_TALLY_LAYOUT,
        "axial_bins": CE_AXIAL_BINS,
        "meets_precision_targets": True,
        "temperature_k": TEMPERATURE_K,
        "lattice_pitch_cm": lattice_pitch_cm(),
        "active_height_cm": active_height_cm(),
        "keff": 0.98,
        "keff_standard_deviation": 5.0e-5,
        "keff_standard_deviation_pcm": 5.0,
        "maximum_axial_nu_fission_relative_standard_deviation_percent": 0.1,
        "maximum_planar_nu_fission_relative_standard_deviation_percent": 0.1,
        "precision_targets": {
            "keff_standard_deviation_pcm": 10.0,
            "nu_fission_relative_standard_deviation_percent": 1.0,
        },
        "openmc_version": "0.15.3",
        "cross_sections": "/data/cross_sections.xml",
        "tallies": {
            "axial_profile": {
                "name": "axial_nu_fission",
                "score": "nu-fission",
                "shape": [CE_AXIAL_BINS],
                "axis_order": ["axial_bin_bottom_to_top"],
                "axial_bin_edges_cm": arrays["axial_bin_edges_cm"].tolist(),
            },
            "planar_profile": {
                "name": "planar_cell_nu_fission",
                "score": "nu-fission",
                "shape": [CE_PLANAR_CELLS],
                "axis_order": ["planar_cell"],
                "planar_coordinates": coordinates,
            },
        },
    }
    for name in arrays:
        summary[name] = f"{name}.npy"
    summary_path = directory / "reference_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path


def _write_morana_case(
    directory: Path,
    coordinates: list[list[int]],
) -> Path:
    """Write a five-layer Morana field matching the rebinned CE fixture."""

    axial = np.linspace(1.0, 2.0, CE_AXIAL_BINS)
    axial /= np.sum(axial)
    planar = np.linspace(1.0, 2.0, CE_PLANAR_CELLS)
    planar /= np.sum(planar)
    production = np.outer(
        axial.reshape(5, CE_AXIAL_BINS // 5).sum(axis=1),
        planar,
    )
    np.save(directory / "normalized_production.npy", production, allow_pickle=False)
    summary = {
        "active_height_cm": active_height_cm(),
        "axial_layers": 5,
        "diffusion_convention": "openmc-transport",
        "keff": 0.9801,
        "lattice_pitch_cm": lattice_pitch_cm(),
        "normalized_production_path": "normalized_production.npy",
        "planar_coordinates": coordinates,
        "temperature_k": TEMPERATURE_K,
    }
    summary_path = directory / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path


def _check_valid_reference(directory: Path, coordinates: list[list[int]]) -> None:
    """Verify exact CE rebinning and recorded CE provenance."""

    reference_path = _write_reference(directory, coordinates)
    candidate_path = _write_morana_case(directory, coordinates)
    result = compare_morana_to_ce(candidate_path, reference_path)
    assert result["comparison_type"] == "morana_to_ce"
    assert result["axial_profile_comparison_bins"] == 5
    assert np.isclose(float(result["delta_keff_pcm"]), 10.0)
    assert np.isclose(float(result["axial_production_rms_percent"]), 0.0)
    assert np.isclose(float(result["planar_production_rms_percent"]), 0.0)
    assert np.isclose(float(result["reference_keff_standard_deviation_pcm"]), 5.0)
    assert result["reference_openmc_version"] == "0.15.3"
    assert result["reference_cross_sections"] == "/data/cross_sections.xml"


def _check_noncanonical_coordinates(
    directory: Path,
    coordinates: list[list[int]],
) -> None:
    """Verify that the reader rejects a noncanonical CE cell order."""

    invalid_coordinates = coordinates.copy()
    invalid_coordinates[0], invalid_coordinates[1] = (
        invalid_coordinates[1],
        invalid_coordinates[0],
    )
    reference_path = _write_reference(directory, invalid_coordinates)
    candidate_path = _write_morana_case(directory, coordinates)
    try:
        compare_morana_to_ce(candidate_path, reference_path)
    except ValueError as error:
        if "shared mini-core ordering" not in str(error):
            raise
    else:
        raise AssertionError("reader accepted a noncanonical CE coordinate order")


def _check_morana_shared_order_conversion() -> None:
    """Verify runner artifacts use the CE root-cell coordinate order."""

    mesh = HexPlanarMesh(NUM_CORE_RINGS, pitch=lattice_pitch_cm())
    production = np.arange(2 * mesh.n_cells, dtype=float).reshape(2, mesh.n_cells)
    converted, coordinates = _shared_planar_production_order(production, mesh)
    planar_ids = tuple(mesh.index_of[coordinate] for coordinate in coordinates)
    assert coordinates == EXPECTED_PLANAR_COORDINATES
    assert np.array_equal(converted, production[:, planar_ids])
    assert planar_ids != tuple(range(mesh.n_cells))


def _check_inconsistent_normalized_profile(
    directory: Path,
    coordinates: list[list[int]],
) -> None:
    """Verify that normalized data must agree with the raw direct tally."""

    reference_path = _write_reference(directory, coordinates)
    candidate_path = _write_morana_case(directory, coordinates)
    normalized_path = directory / "normalized_axial_production.npy"
    normalized = np.load(normalized_path, allow_pickle=False)
    normalized[[0, 1]] = normalized[[1, 0]]
    np.save(normalized_path, normalized, allow_pickle=False)
    try:
        compare_morana_to_ce(candidate_path, reference_path)
    except ValueError as error:
        if "raw tally means" not in str(error):
            raise
    else:
        raise AssertionError("reader accepted an inconsistent normalized profile")


def _check_incompatible_diffusion_convention(
    directory: Path,
    coordinates: list[list[int]],
) -> None:
    """Verify that the reader requires the OpenMC transport convention."""

    reference_path = _write_reference(directory, coordinates)
    candidate_path = _write_morana_case(directory, coordinates)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["diffusion_convention"] = "p1-outscatter"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    try:
        compare_morana_to_ce(candidate_path, reference_path)
    except ValueError as error:
        if "diffusion_convention" not in str(error):
            raise
    else:
        raise AssertionError("reader accepted an incompatible diffusion convention")


def _check_invalid_production_array(
    directory: Path, coordinates: list[list[int]]
) -> None:
    """Reject complex payloads and containers in place of real array data."""
    reference_path = _write_reference(directory, coordinates)
    candidate_path = _write_morana_case(directory, coordinates)
    array_path = directory / "normalized_production.npy"
    production = np.load(array_path, allow_pickle=False)
    for kind in ("complex", "container"):
        with array_path.open("wb") as stream:
            if kind == "complex":
                np.save(stream, production + 1j, allow_pickle=False)
            else:
                np.savez(stream, production=production)
        try:
            compare_morana_to_ce(candidate_path, reference_path)
        except ValueError as error:
            expected = "real numeric" if kind == "complex" else "NumPy array"
            if expected not in str(error):
                raise
        else:
            raise AssertionError(f"reader accepted {kind} production data")


def main() -> None:
    """Run deterministic CE-reader checks without persistent artifacts."""

    coordinates = [list(pair) for pair in EXPECTED_PLANAR_COORDINATES]
    with TemporaryDirectory(prefix="morana_ce_reader_") as temporary_directory:
        _check_valid_reference(Path(temporary_directory), coordinates)
    with TemporaryDirectory(prefix="morana_ce_reader_") as temporary_directory:
        _check_noncanonical_coordinates(Path(temporary_directory), coordinates)
    with TemporaryDirectory(prefix="morana_ce_reader_") as temporary_directory:
        _check_invalid_production_array(Path(temporary_directory), coordinates)
    _check_morana_shared_order_conversion()
    with TemporaryDirectory(prefix="morana_ce_reader_") as temporary_directory:
        _check_inconsistent_normalized_profile(Path(temporary_directory), coordinates)
    with TemporaryDirectory(prefix="morana_ce_reader_") as temporary_directory:
        _check_incompatible_diffusion_convention(Path(temporary_directory), coordinates)
    print(
        "Verified CE-reader rebinning, profile consistency, provenance, "
        "coordinate order, and diffusion convention."
    )


if __name__ == "__main__":
    main()
