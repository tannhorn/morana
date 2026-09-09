"""Compare Morana study cases with each other or a checked CE reference.

The production comparisons are the radially integrated axial profile and the
axially integrated 61-cell profile. A CE reference must contain the versioned
direct-profile layout emitted by :mod:`generate_ce_reference`; its 100-bin
axial profile is exactly rebinned to the candidate Morana mesh.
"""

# This data object deliberately mirrors a complete portable CE record.
# pylint: disable=too-many-instance-attributes

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import isclose, isfinite
from pathlib import Path

import numpy as np

from geometry import mini_core_coordinates

CE_TALLY_LAYOUT = "direct_profiles_v1"
CE_AXIAL_BINS = 100
EXPECTED_PLANAR_COORDINATES = tuple(
    (q_coord, r_coord) for q_coord, r_coord, _, _ in mini_core_coordinates()
)
CE_PLANAR_CELLS = len(EXPECTED_PLANAR_COORDINATES)


@dataclass(frozen=True)
class MoranaStudyCase:
    """Store a checked portable Morana study case."""

    summary_path: Path
    record: dict[str, object]
    production: np.ndarray
    planar_coordinates: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class CeReference:
    """Store the checked direct-profile CE reference quantities."""

    summary_path: Path
    keff: float
    keff_standard_deviation_pcm: float
    temperature_k: float
    lattice_pitch_cm: float
    active_height_cm: float
    axial_production: np.ndarray
    planar_production: np.ndarray
    axial_bin_edges_cm: np.ndarray
    planar_coordinates: tuple[tuple[int, int], ...]
    maximum_axial_relative_standard_deviation_percent: float
    maximum_planar_relative_standard_deviation_percent: float
    openmc_version: str
    cross_sections: str


def _load_json_object(path: Path, description: str) -> tuple[Path, dict[str, object]]:
    """Load one JSON-object artifact and return its resolved path."""

    resolved_path = path.expanduser().resolve()
    try:
        value = json.loads(resolved_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read {description}: {resolved_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON {description}: {resolved_path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {resolved_path}")
    return resolved_path, value


def _required(record: dict[str, object], field: str, description: str) -> object:
    """Return one required named record field."""

    try:
        return record[field]
    except KeyError as error:
        raise ValueError(f"{description} is missing field: {field}") from error


def _finite_float(value: object, description: str, *, positive: bool = False) -> float:
    """Return one finite JSON number, optionally requiring positivity."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{description} must be finite")
    if positive and number <= 0.0:
        raise ValueError(f"{description} must be positive")
    return number


def _coordinates(
    value: object,
    expected_count: int,
    description: str,
) -> tuple[tuple[int, int], ...]:
    """Check and normalize ordered integer axial-coordinate pairs."""

    if not isinstance(value, list) or len(value) != expected_count:
        raise ValueError(f"{description} must contain {expected_count} coordinates")
    coordinates: list[tuple[int, int]] = []
    for coordinate in value:
        if (
            not isinstance(coordinate, list)
            or len(coordinate) != 2
            or any(
                isinstance(component, bool) or not isinstance(component, int)
                for component in coordinate
            )
        ):
            raise ValueError(f"{description} must contain integer coordinate pairs")
        coordinates.append((coordinate[0], coordinate[1]))
    if len(set(coordinates)) != len(coordinates):
        raise ValueError(f"{description} contains duplicate coordinates")
    normalized_coordinates = tuple(coordinates)
    if normalized_coordinates != EXPECTED_PLANAR_COORDINATES:
        raise ValueError(f"{description} does not use the shared mini-core ordering")
    return normalized_coordinates


def _load_array(
    summary_path: Path,
    record: dict[str, object],
    field: str,
    shape: tuple[int, ...],
    description: str,
) -> np.ndarray:
    """Load and validate a finite nonnegative non-pickle NumPy array."""

    value = _required(record, field, description)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} field {field} must name an array file")
    array_path = Path(value)
    if not array_path.is_absolute():
        array_path = summary_path.parent / array_path
    try:
        array = np.load(array_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load {description} array: {array_path}") from error
    if not isinstance(array, np.ndarray):
        array.close()
        raise ValueError(f"{description} must contain a NumPy array")
    if array.shape != shape:
        raise ValueError(
            f"{description} array has shape {array.shape}, expected {shape}"
        )
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(f"{description} array must have a real numeric dtype")
    numeric_array = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(numeric_array)) or np.any(numeric_array < 0.0):
        raise ValueError(f"{description} array must be finite and nonnegative")
    return numeric_array


def _normalized_array(
    summary_path: Path,
    record: dict[str, object],
    field: str,
    shape: tuple[int, ...],
    description: str,
) -> np.ndarray:
    """Load a normalized production array with a checked unit total."""

    array = _load_array(summary_path, record, field, shape, description)
    if not np.isclose(np.sum(array), 1.0, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError(f"{description} must sum to one")
    return array


def _check_normalized_profile(
    normalized: np.ndarray,
    raw: np.ndarray,
    description: str,
) -> None:
    """Require a stored normalized profile to match its raw tally means."""

    expected = raw / np.sum(raw)
    if not np.allclose(normalized, expected, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError(f"{description} is inconsistent with its raw tally means")


def _check_close(actual: float, expected: float, description: str) -> None:
    """Require two independently recorded scalar values to agree."""

    if not isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError(
            f"{description} is inconsistent with the associated array data"
        )


def _load_morana_case(path: Path) -> MoranaStudyCase:
    """Load one checked Morana study summary and production field."""

    summary_path, record = _load_json_object(path, "Morana study summary")
    required_fields = {
        "active_height_cm",
        "axial_layers",
        "diffusion_convention",
        "keff",
        "lattice_pitch_cm",
        "normalized_production_path",
        "planar_coordinates",
        "temperature_k",
    }
    missing_fields = required_fields.difference(record)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"Morana study summary is missing fields: {missing}")
    if record["diffusion_convention"] != "openmc-transport":
        raise ValueError("Morana study diffusion_convention must be 'openmc-transport'")
    axial_layers_value = _required(record, "axial_layers", "Morana study summary")
    if (
        isinstance(axial_layers_value, bool)
        or not isinstance(axial_layers_value, int)
        or axial_layers_value <= 0
    ):
        raise ValueError("Morana study axial_layers must be a positive integer")
    planar_coordinates = _coordinates(
        _required(record, "planar_coordinates", "Morana study summary"),
        CE_PLANAR_CELLS,
        "Morana study planar_coordinates",
    )
    _finite_float(_required(record, "keff", "Morana study summary"), "Morana k_eff")
    _finite_float(
        _required(record, "temperature_k", "Morana study summary"),
        "Morana temperature_k",
        positive=True,
    )
    _finite_float(
        _required(record, "lattice_pitch_cm", "Morana study summary"),
        "Morana lattice_pitch_cm",
        positive=True,
    )
    _finite_float(
        _required(record, "active_height_cm", "Morana study summary"),
        "Morana active_height_cm",
        positive=True,
    )
    production = _normalized_array(
        summary_path,
        record,
        "normalized_production_path",
        (axial_layers_value, CE_PLANAR_CELLS),
        "Morana normalized production",
    )
    return MoranaStudyCase(summary_path, record, production, planar_coordinates)


def _tally_record(
    tallies: dict[str, object],
    name: str,
    shape: list[int],
    axis_order: list[str],
) -> dict[str, object]:
    """Validate one versioned CE direct-profile tally metadata record."""

    value = _required(tallies, name, "CE reference tallies")
    if not isinstance(value, dict):
        raise ValueError(f"CE reference tally {name} must be an object")
    if value.get("shape") != shape or value.get("axis_order") != axis_order:
        raise ValueError(f"CE reference tally {name} has unexpected layout")
    return value


# pylint: disable=too-many-branches,too-many-statements
def _load_ce_reference(path: Path) -> CeReference:
    """Load and fully validate the CE direct-profile comparison reference."""

    summary_path, record = _load_json_object(path, "CE reference summary")
    if _required(record, "tally_layout", "CE reference summary") != CE_TALLY_LAYOUT:
        raise ValueError("CE reference has an unsupported tally layout")
    if _required(record, "axial_bins", "CE reference summary") != CE_AXIAL_BINS:
        raise ValueError("CE reference must use 100 direct axial bins")
    if _required(record, "meets_precision_targets", "CE reference summary") is not True:
        raise ValueError("CE reference does not meet its precision targets")
    tallies = _required(record, "tallies", "CE reference summary")
    if not isinstance(tallies, dict):
        raise ValueError("CE reference tallies must be an object")
    axial_tally = _tally_record(
        tallies,
        "axial_profile",
        [CE_AXIAL_BINS],
        ["axial_bin_bottom_to_top"],
    )
    planar_tally = _tally_record(
        tallies,
        "planar_profile",
        [CE_PLANAR_CELLS],
        ["planar_cell"],
    )
    if axial_tally.get("name") != "axial_nu_fission":
        raise ValueError("CE reference axial tally has unexpected name")
    if planar_tally.get("name") != "planar_cell_nu_fission":
        raise ValueError("CE reference planar tally has unexpected name")
    if axial_tally.get("score") != "nu-fission":
        raise ValueError("CE reference axial tally has unexpected score")
    if planar_tally.get("score") != "nu-fission":
        raise ValueError("CE reference planar tally has unexpected score")
    planar_coordinates = _coordinates(
        _required(planar_tally, "planar_coordinates", "CE planar tally"),
        CE_PLANAR_CELLS,
        "CE planar coordinates",
    )
    axial_mean = _load_array(
        summary_path,
        record,
        "axial_nu_fission_mean",
        (CE_AXIAL_BINS,),
        "CE axial nu-fission mean",
    )
    axial_standard_deviation = _load_array(
        summary_path,
        record,
        "axial_nu_fission_std_dev",
        (CE_AXIAL_BINS,),
        "CE axial nu-fission standard deviation",
    )
    planar_mean = _load_array(
        summary_path,
        record,
        "planar_nu_fission_mean",
        (CE_PLANAR_CELLS,),
        "CE planar nu-fission mean",
    )
    planar_standard_deviation = _load_array(
        summary_path,
        record,
        "planar_nu_fission_std_dev",
        (CE_PLANAR_CELLS,),
        "CE planar nu-fission standard deviation",
    )
    if np.any(axial_mean <= 0.0) or np.any(planar_mean <= 0.0):
        raise ValueError("CE nu-fission means must be positive")
    axial_production = _normalized_array(
        summary_path,
        record,
        "normalized_axial_production",
        (CE_AXIAL_BINS,),
        "CE normalized axial production",
    )
    planar_production = _normalized_array(
        summary_path,
        record,
        "normalized_planar_production",
        (CE_PLANAR_CELLS,),
        "CE normalized planar production",
    )
    _check_normalized_profile(
        axial_production,
        axial_mean,
        "CE normalized axial production",
    )
    _check_normalized_profile(
        planar_production,
        planar_mean,
        "CE normalized planar production",
    )
    axial_edges = _load_array(
        summary_path,
        record,
        "axial_bin_edges_cm",
        (CE_AXIAL_BINS + 1,),
        "CE axial-bin edges",
    )
    if not np.all(np.diff(axial_edges) > 0.0):
        raise ValueError("CE axial-bin edges must be strictly increasing")
    if not np.allclose(
        np.diff(axial_edges),
        axial_edges[1] - axial_edges[0],
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError("CE axial-bin edges must be uniform")
    metadata_edges = axial_tally.get("axial_bin_edges_cm")
    if not isinstance(metadata_edges, list) or len(metadata_edges) != CE_AXIAL_BINS + 1:
        raise ValueError("CE axial tally metadata has inconsistent bin edges")
    try:
        numeric_metadata_edges = np.asarray(metadata_edges, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "CE axial tally metadata has inconsistent bin edges"
        ) from error
    if not np.allclose(
        numeric_metadata_edges,
        axial_edges,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError("CE axial tally metadata has inconsistent bin edges")
    active_height = _finite_float(
        _required(record, "active_height_cm", "CE reference summary"),
        "CE active_height_cm",
        positive=True,
    )
    if not np.allclose(
        axial_edges[[0, -1]],
        [0.0, active_height],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("CE axial-bin edges do not span the recorded active height")
    targets = _required(record, "precision_targets", "CE reference summary")
    if not isinstance(targets, dict):
        raise ValueError("CE precision_targets must be an object")
    keff_target_pcm = _finite_float(
        _required(targets, "keff_standard_deviation_pcm", "CE precision targets"),
        "CE k_eff standard-deviation target",
        positive=True,
    )
    production_target_percent = _finite_float(
        _required(
            targets,
            "nu_fission_relative_standard_deviation_percent",
            "CE precision targets",
        ),
        "CE nu-fission relative standard-deviation target",
        positive=True,
    )
    keff = _finite_float(_required(record, "keff", "CE reference summary"), "CE k_eff")
    keff_standard_deviation = _finite_float(
        _required(record, "keff_standard_deviation", "CE reference summary"),
        "CE k_eff standard deviation",
        positive=True,
    )
    keff_standard_deviation_pcm = keff_standard_deviation * 1.0e5
    _check_close(
        keff_standard_deviation_pcm,
        _finite_float(
            _required(
                record,
                "keff_standard_deviation_pcm",
                "CE reference summary",
            ),
            "recorded CE k_eff standard deviation [pcm]",
            positive=True,
        ),
        "recorded CE k_eff standard deviation [pcm]",
    )
    axial_relative_standard_deviation_percent = float(
        100.0 * np.max(axial_standard_deviation / axial_mean)
    )
    planar_relative_standard_deviation_percent = float(
        100.0 * np.max(planar_standard_deviation / planar_mean)
    )
    _check_close(
        axial_relative_standard_deviation_percent,
        _finite_float(
            _required(
                record,
                "maximum_axial_nu_fission_relative_standard_deviation_percent",
                "CE reference summary",
            ),
            "recorded maximum CE axial relative standard deviation",
            positive=True,
        ),
        "recorded maximum CE axial relative standard deviation",
    )
    _check_close(
        planar_relative_standard_deviation_percent,
        _finite_float(
            _required(
                record,
                "maximum_planar_nu_fission_relative_standard_deviation_percent",
                "CE reference summary",
            ),
            "recorded maximum CE planar relative standard deviation",
            positive=True,
        ),
        "recorded maximum CE planar relative standard deviation",
    )
    if keff_standard_deviation_pcm > keff_target_pcm:
        raise ValueError("CE k_eff standard deviation exceeds its precision target")
    if axial_relative_standard_deviation_percent > production_target_percent:
        raise ValueError("CE axial profile uncertainty exceeds its precision target")
    if planar_relative_standard_deviation_percent > production_target_percent:
        raise ValueError("CE planar profile uncertainty exceeds its precision target")
    openmc_version = _required(record, "openmc_version", "CE reference summary")
    cross_sections = _required(record, "cross_sections", "CE reference summary")
    if not isinstance(openmc_version, str) or not openmc_version:
        raise ValueError("CE reference OpenMC version must be a nonempty string")
    if not isinstance(cross_sections, str) or not cross_sections:
        raise ValueError("CE reference cross-sections provenance must be a string")
    return CeReference(
        summary_path=summary_path,
        keff=keff,
        keff_standard_deviation_pcm=keff_standard_deviation_pcm,
        temperature_k=_finite_float(
            _required(record, "temperature_k", "CE reference summary"),
            "CE temperature_k",
            positive=True,
        ),
        lattice_pitch_cm=_finite_float(
            _required(record, "lattice_pitch_cm", "CE reference summary"),
            "CE lattice_pitch_cm",
            positive=True,
        ),
        active_height_cm=active_height,
        axial_production=axial_production,
        planar_production=planar_production,
        axial_bin_edges_cm=axial_edges,
        planar_coordinates=planar_coordinates,
        maximum_axial_relative_standard_deviation_percent=(
            axial_relative_standard_deviation_percent
        ),
        maximum_planar_relative_standard_deviation_percent=(
            planar_relative_standard_deviation_percent
        ),
        openmc_version=openmc_version,
        cross_sections=cross_sections,
    )


# pylint: enable=too-many-branches,too-many-statements


def _relative_rms_percent(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Return RMS difference relative to the reference-field mean."""

    if candidate.shape != reference.shape:
        raise ValueError("comparison arrays must have identical shapes")
    reference_mean = float(np.mean(reference))
    if reference_mean <= 0.0:
        raise ValueError("reference production mean must be positive")
    difference_rms = float(np.sqrt(np.mean((candidate - reference) ** 2)))
    return 100.0 * difference_rms / reference_mean


def _axial_profile(production: np.ndarray, bins: int) -> np.ndarray:
    """Return the radially integrated profile exactly rebinned to ``bins``."""

    return _rebin_axial_profile(np.sum(production, axis=1), bins)


def _rebin_axial_profile(profile: np.ndarray, bins: int) -> np.ndarray:
    """Exactly rebin a normalized one-dimensional axial profile."""

    source_bins = profile.size
    if source_bins % bins:
        raise ValueError(
            f"cannot exactly rebin {source_bins} axial layers to {bins} layers"
        )
    return np.sum(profile.reshape(bins, source_bins // bins), axis=1)


def _cell_profile(production: np.ndarray) -> np.ndarray:
    """Return production integrated over all axial layers for each cell."""

    return np.sum(production, axis=0)


def _check_shared_model(
    candidate: MoranaStudyCase,
    reference: MoranaStudyCase,
) -> None:
    """Require two Morana records to use the same physical model."""

    if candidate.planar_coordinates != reference.planar_coordinates:
        raise ValueError(
            "study cases use different planar coordinates or cell ordering"
        )
    for field, description in (
        ("temperature_k", "material temperatures"),
        ("lattice_pitch_cm", "lattice pitches"),
        ("active_height_cm", "active heights"),
    ):
        if not isclose(
            _finite_float(candidate.record[field], f"candidate {field}"),
            _finite_float(reference.record[field], f"reference {field}"),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"study cases use different {description}")
    if (
        candidate.record["diffusion_convention"]
        != reference.record["diffusion_convention"]
    ):
        raise ValueError("study cases use different diffusion conventions")


def compare_morana_cases(
    candidate_path: Path,
    reference_path: Path,
) -> dict[str, object]:
    """Compare two compatible Morana study cases."""

    candidate = _load_morana_case(candidate_path)
    reference = _load_morana_case(reference_path)
    _check_shared_model(candidate, reference)
    common_layers = min(
        candidate.production.shape[0],
        reference.production.shape[0],
    )
    return {
        "comparison_type": "morana_to_morana",
        "candidate_summary": str(candidate.summary_path),
        "reference_summary": str(reference.summary_path),
        "axial_profile_comparison_bins": common_layers,
        "delta_keff_pcm": (
            _finite_float(candidate.record["keff"], "candidate k_eff")
            - _finite_float(reference.record["keff"], "reference k_eff")
        )
        * 1.0e5,
        "axial_production_rms_percent": _relative_rms_percent(
            _axial_profile(candidate.production, common_layers),
            _axial_profile(reference.production, common_layers),
        ),
        "planar_production_rms_percent": _relative_rms_percent(
            _cell_profile(candidate.production),
            _cell_profile(reference.production),
        ),
    }


def compare_morana_to_ce(
    candidate_path: Path,
    reference_path: Path,
) -> dict[str, object]:
    """Compare one Morana case with a checked direct-profile CE reference."""

    candidate = _load_morana_case(candidate_path)
    reference = _load_ce_reference(reference_path)
    if candidate.planar_coordinates != reference.planar_coordinates:
        raise ValueError(
            "Morana and CE use different planar coordinates or cell ordering"
        )
    candidate_temperature = _finite_float(
        candidate.record["temperature_k"], "Morana temperature_k", positive=True
    )
    if not isclose(
        candidate_temperature,
        reference.temperature_k,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Morana and CE use different material temperatures")
    candidate_height = _finite_float(
        candidate.record["active_height_cm"],
        "Morana active_height_cm",
        positive=True,
    )
    if not isclose(
        candidate_height,
        reference.active_height_cm,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Morana and CE use different active heights")
    candidate_pitch = _finite_float(
        candidate.record["lattice_pitch_cm"],
        "Morana lattice_pitch_cm",
        positive=True,
    )
    if not isclose(
        candidate_pitch,
        reference.lattice_pitch_cm,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Morana and CE use different lattice pitches")
    axial_layers = candidate.production.shape[0]
    if CE_AXIAL_BINS % axial_layers:
        raise ValueError(
            "Morana axial layers must exactly divide the 100-bin CE reference"
        )
    return {
        "comparison_type": "morana_to_ce",
        "candidate_summary": str(candidate.summary_path),
        "reference_summary": str(reference.summary_path),
        "axial_profile_comparison_bins": axial_layers,
        "delta_keff_pcm": (
            _finite_float(candidate.record["keff"], "Morana k_eff") - reference.keff
        )
        * 1.0e5,
        "reference_keff_standard_deviation_pcm": reference.keff_standard_deviation_pcm,
        "axial_production_rms_percent": _relative_rms_percent(
            _axial_profile(candidate.production, axial_layers),
            _rebin_axial_profile(reference.axial_production, axial_layers),
        ),
        "planar_production_rms_percent": _relative_rms_percent(
            _cell_profile(candidate.production),
            reference.planar_production,
        ),
        "reference_maximum_axial_relative_standard_deviation_percent": (
            reference.maximum_axial_relative_standard_deviation_percent
        ),
        "reference_maximum_planar_relative_standard_deviation_percent": (
            reference.maximum_planar_relative_standard_deviation_percent
        ),
        "reference_openmc_version": reference.openmc_version,
        "reference_cross_sections": reference.cross_sections,
    }


def _arguments() -> argparse.Namespace:
    """Parse candidate/reference summaries and optional output path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="Morana case summary")
    parser.add_argument(
        "reference",
        type=Path,
        help="Morana case summary or a checked CE reference summary",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON path for the checked comparison result",
    )
    return parser.parse_args()


def _print_result(result: dict[str, object]) -> None:
    """Print a compact human-readable comparison result."""

    comparison_type = result["comparison_type"]
    if comparison_type == "morana_to_ce":
        print("Reference: checked continuous-energy direct-profile result")
        print(
            "CE k_eff one-sigma uncertainty [pcm]: "
            f"{float(result['reference_keff_standard_deviation_pcm']):.3f}"
        )
        axial_relative_standard_deviation = float(
            result["reference_maximum_axial_relative_standard_deviation_percent"]
        )
        planar_relative_standard_deviation = float(
            result["reference_maximum_planar_relative_standard_deviation_percent"]
        )
        print(
            "CE maximum axial/planar nu-fission relative standard deviation [%]: "
            f"{axial_relative_standard_deviation:.4f}/"
            f"{planar_relative_standard_deviation:.4f}"
        )
    else:
        print("Reference: Morana study case")
    print(
        "Axial-profile comparison bins: " f"{result['axial_profile_comparison_bins']}"
    )
    print(
        "Candidate - reference k_eff [pcm]: " f"{float(result['delta_keff_pcm']):.3f}"
    )
    print(
        "Radially integrated axial-production RMS [%]: "
        f"{float(result['axial_production_rms_percent']):.6f}"
    )
    print(
        "Axially integrated cell-production RMS [%]: "
        f"{float(result['planar_production_rms_percent']):.6f}"
    )


def main() -> None:
    """Compare one Morana candidate to a Morana or CE reference."""

    arguments = _arguments()
    _, reference_record = _load_json_object(arguments.reference, "reference summary")
    if "tally_layout" in reference_record:
        result = compare_morana_to_ce(arguments.candidate, arguments.reference)
    else:
        result = compare_morana_cases(arguments.candidate, arguments.reference)
    _print_result(result)
    if arguments.output is not None:
        output_path = arguments.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Comparison summary: {output_path}")


if __name__ == "__main__":
    main()
