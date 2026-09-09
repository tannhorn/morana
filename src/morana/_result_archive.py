"""Versioned non-pickle archive support for completed Morana results."""

# pylint: disable=protected-access,too-many-branches,too-many-lines,too-many-return-statements

from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
import zipfile

import numpy as np

from morana.boundary import (
    BoundaryAssignment,
    BoundaryCondition,
    BoundaryConditionSet,
    BoundarySelector,
)
from morana.configuration import ProblemConfiguration, ProblemConfigurationSnapshot
from morana.hex_planar_mesh import HexPlanarMesh
from morana.material_mesh import MaterialMesh, MaterialSlice
from morana.materials import (
    CrossSections,
    ExcludedRegion,
    FissionData,
    FissionTransfer,
    Material,
    SeparableFission,
)
from morana.normalization import (
    FissionSourceNormalization,
    PowerNormalization,
)
from morana.results import FixedSourceBalance, KeffBalance, Result
from morana.sources import CellSource, MaterialSource, UniformSource
from morana.execution_reports import (
    KeffSolveReport,
    KeffOuterIterationReport,
    LinearSolveReport,
)
from morana.solve_settings import (
    DirectLinearSolveSettings,
    FixedSourceSettings,
    GmresLinearSolveSettings,
    IluPreconditioner,
    JacobiPreconditioner,
    KeffSettings,
    NoPreconditioner,
    PowerIterationSettings,
    WielandtShiftSettings,
)

_FORMAT = "morana-result"
_SCHEMA_VERSION = 8
_MANIFEST_MEMBER = "manifest.json"


class _ArrayStore:
    """Build named non-pickle ``.npy`` payloads for one archive."""

    def __init__(self) -> None:
        self.payloads: dict[str, bytes] = {}
        self.descriptors: list[dict[str, object]] = []
        self._counter = 0

    def add(self, label: str, array: np.ndarray) -> str:
        """Store one NumPy array without object-dtype fields.

        Return its archive member name.
        """
        if array.dtype.hasobject:
            raise ValueError("result archives do not support object-dtype arrays")
        member = f"arrays/{label}_{self._counter:08d}.npy"
        self._counter += 1
        stream = io.BytesIO()
        np.save(stream, array, allow_pickle=False)
        payload = stream.getvalue()
        self.payloads[member] = payload
        self.descriptors.append(
            {
                "member": member,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "dtype": array.dtype.str,
                "shape": list(array.shape),
            }
        )
        return member


class _ArrayReader:
    """Validate and restore declared ``.npy`` payloads from one archive."""

    def __init__(self, archive: zipfile.ZipFile, descriptors: list[object]) -> None:
        self._archive = archive
        self._descriptors = _descriptor_map(descriptors)
        self._used: set[str] = set()

    @property
    def members(self) -> set[str]:
        """Return all manifest-declared array members."""
        return set(self._descriptors)

    def load(self, member: object) -> np.ndarray:
        """Load one declared payload after digest, dtype, and shape checks."""
        if not isinstance(member, str) or member not in self._descriptors:
            raise ValueError("archive refers to an undeclared array payload")
        descriptor = self._descriptors[member]
        try:
            payload = self._archive.read(member)
        except KeyError as exc:
            raise ValueError(f"archive payload is missing: {member}") from exc
        if hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
            raise ValueError(f"archive payload checksum mismatch: {member}")
        try:
            array = np.load(io.BytesIO(payload), allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"archive payload is not a valid NumPy array: {member}"
            ) from exc
        if not isinstance(array, np.ndarray) or array.dtype.hasobject:
            raise ValueError(f"archive payload has unsupported dtype: {member}")
        if (
            array.dtype.str != descriptor["dtype"]
            or list(array.shape) != descriptor["shape"]
        ):
            raise ValueError(f"archive payload descriptor mismatch: {member}")
        self._used.add(member)
        return array

    def check_all_used(self) -> None:
        """Reject manifest payloads that are not referenced by the result data."""
        unused = self.members - self._used
        if unused:
            raise ValueError("archive declares unreferenced array payloads")


def save_result(result: Result, path: str | Path) -> None:
    """Write one completed result as an atomic versioned ZIP archive."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = _ArrayStore()
    manifest = {
        "format": _FORMAT,
        "schema_version": _SCHEMA_VERSION,
        "result": _result_to_data(result, arrays),
        "arrays": arrays.descriptors,
    }
    manifest_bytes = json.dumps(
        manifest,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    temporary_name = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
        with zipfile.ZipFile(
            temporary_name,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            archive.writestr(_MANIFEST_MEMBER, manifest_bytes)
            for member, payload in arrays.payloads.items():
                archive.writestr(member, payload)
        os.replace(temporary_name, destination)
    except OSError:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def load_result(path: str | Path) -> Result:
    """Load one checked completed result from a versioned ZIP archive."""
    try:
        with zipfile.ZipFile(Path(path), mode="r") as archive:
            manifest = _load_manifest(archive)
            result_data = manifest["result"]
            arrays = _ArrayReader(archive, manifest["arrays"])
            expected_members = {_MANIFEST_MEMBER, *arrays.members}
            _check_archive_members(archive, expected_members)
            result = _result_from_data(result_data, arrays)
            arrays.check_all_used()
            return result
    except (json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise ValueError("invalid Morana result archive") from exc


def _load_manifest(archive: zipfile.ZipFile) -> dict[str, object]:
    """Read and validate one archive manifest."""
    _check_archive_members(archive, {_MANIFEST_MEMBER}, allow_array_members=True)
    try:
        raw_manifest = archive.read(_MANIFEST_MEMBER)
    except KeyError as exc:
        raise ValueError("archive is missing manifest.json") from exc
    try:
        manifest = json.loads(raw_manifest.decode("utf-8"), parse_constant=_bad_json)
    except UnicodeDecodeError as exc:
        raise ValueError("archive manifest must be UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("archive manifest must be a JSON object")
    return _check_manifest(manifest)


def _bad_json(value: str) -> None:
    """Reject non-standard JSON constants such as NaN and Infinity."""
    raise ValueError(f"archive manifest contains invalid JSON constant {value!r}")


def _check_manifest(manifest: dict[str, object]) -> dict[str, object]:
    """Require one manifest in the current archive schema."""
    if manifest.get("format") != _FORMAT:
        raise ValueError("archive does not contain a Morana result")
    version = manifest.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("archive schema_version must be an integer")
    if version != _SCHEMA_VERSION:
        raise ValueError(f"unsupported archive schema version {version}")
    _require_keys(
        manifest,
        {"format", "schema_version", "result", "arrays"},
        "manifest",
    )
    if not isinstance(manifest["result"], dict) or not isinstance(
        manifest["arrays"], list
    ):
        raise ValueError("archive manifest has invalid result or arrays entries")
    return manifest


def _descriptor_map(descriptors: list[object]) -> dict[str, dict[str, object]]:
    """Return checked array descriptors keyed by their ZIP member name."""
    mapped: dict[str, dict[str, object]] = {}
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise ValueError("archive array descriptors must be JSON objects")
        _require_keys(
            descriptor,
            {"member", "sha256", "dtype", "shape"},
            "array descriptor",
        )
        member = descriptor["member"]
        digest = descriptor["sha256"]
        dtype = descriptor["dtype"]
        shape = descriptor["shape"]
        if (
            not isinstance(member, str)
            or not member.startswith("arrays/")
            or not member.endswith(".npy")
            or "/" in member[len("arrays/") : -len(".npy")]
            or member in mapped
        ):
            raise ValueError("archive array member name is invalid")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("archive array checksum is invalid")
        try:
            parsed_dtype = np.dtype(dtype)
        except TypeError as exc:
            raise ValueError("archive array dtype is invalid") from exc
        if parsed_dtype.hasobject or parsed_dtype.str != dtype:
            raise ValueError("archive array dtype is unsupported")
        if not isinstance(shape, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in shape
        ):
            raise ValueError("archive array shape is invalid")
        mapped[member] = descriptor
    return mapped


def _check_archive_members(
    archive: zipfile.ZipFile,
    expected_members: set[str],
    *,
    allow_array_members: bool = False,
) -> None:
    """Reject encrypted, duplicate, path-like, or unexpected archive members."""
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise ValueError("archive contains duplicate members")
    for info in infos:
        if info.flag_bits & 0x1:
            raise ValueError("archive members must not be encrypted")
        if info.filename.startswith("/") or ".." in Path(info.filename).parts:
            raise ValueError("archive member path is unsafe")
    if allow_array_members:
        if any(
            name != _MANIFEST_MEMBER and not name.startswith("arrays/")
            for name in names
        ):
            raise ValueError("archive contains unexpected members")
        return
    if set(names) != expected_members:
        raise ValueError("archive members do not match its manifest")


def _result_to_data(result: Result, arrays: _ArrayStore) -> dict[str, object]:
    """Convert one completed result to manifest data and register its arrays."""
    return {
        "flux": [arrays.add("flux_layer", layer) for layer in result.flux],
        "balance": _balance_to_data(result.balance, arrays),
        "configuration_snapshot": _snapshot_to_data(
            result.configuration_snapshot, arrays
        ),
        "solve_settings": _settings_to_data(result.solve_settings),
        "normalization": _normalization_to_data(result.normalization),
        "execution_report": _execution_report_to_data(result.execution_report),
    }


def _result_from_data(data: object, arrays: _ArrayReader) -> Result:
    """Restore one validated internal ``Result`` from manifest data."""
    if not isinstance(data, dict):
        raise ValueError("archive result must be a JSON object")
    required_keys = {
        "flux",
        "balance",
        "configuration_snapshot",
        "solve_settings",
        "normalization",
        "execution_report",
    }
    _require_keys(data, required_keys, "result")
    flux_members = data["flux"]
    if not isinstance(flux_members, list):
        raise ValueError("archive result flux must be an array-member list")
    flux = tuple(arrays.load(member) for member in flux_members)
    snapshot = _snapshot_from_data(data["configuration_snapshot"], arrays)
    solve_settings = _settings_from_data(data["solve_settings"])
    normalization = _normalization_from_data(data["normalization"])
    execution_report = _execution_report_from_data(data["execution_report"])
    balance = _balance_from_data(data["balance"], arrays, solve_settings)
    try:
        result = Result._from_validated(  # pylint: disable=protected-access
            flux=flux,
            balance=balance,
            configuration_snapshot=snapshot,
            solve_settings=solve_settings,
            normalization=normalization,
            execution_report=execution_report,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"archive result data is invalid: {exc}") from exc
    return result


def _balance_to_data(
    balance: FixedSourceBalance | KeffBalance, arrays: _ArrayStore
) -> dict[str, object]:
    """Convert one balance record to manifest data and register its arrays."""
    return {
        "by_group": {
            name: arrays.add(f"balance_group_{name}", values)
            for name, values in balance.by_group.items()
        },
        "by_layer_group": {
            name: [arrays.add(f"balance_layer_{name}", values) for values in layers]
            for name, layers in balance.by_layer_group.items()
        },
    }


def _balance_from_data(
    data: object,
    arrays: _ArrayReader,
    settings: FixedSourceSettings | KeffSettings,
) -> FixedSourceBalance | KeffBalance:
    """Restore one typed balance record from archive data and solve settings."""
    if not isinstance(data, dict):
        raise ValueError("archive balance must be a JSON object")
    _require_keys(
        data,
        {"by_group", "by_layer_group"},
        "balance",
    )
    by_group_data = data["by_group"]
    by_layer_group_data = data["by_layer_group"]
    if not isinstance(by_group_data, dict) or not isinstance(by_layer_group_data, dict):
        raise ValueError("archive balance vectors must be JSON objects")
    by_group = {name: arrays.load(member) for name, member in by_group_data.items()}
    by_layer_group = {}
    for name, members in by_layer_group_data.items():
        if not isinstance(members, list):
            raise ValueError("archive balance layer vectors must be lists")
        by_layer_group[name] = tuple(arrays.load(member) for member in members)
    if isinstance(settings, FixedSourceSettings):
        return FixedSourceBalance._from_validated(  # pylint: disable=protected-access
            by_group=by_group,
            by_layer_group=by_layer_group,
        )
    return KeffBalance._from_validated(  # pylint: disable=protected-access
        by_group=by_group,
        by_layer_group=by_layer_group,
    )


def _settings_to_data(
    settings: FixedSourceSettings | KeffSettings,
) -> dict[str, object]:
    """Return explicit mode-specific solve-settings provenance."""
    if isinstance(settings, FixedSourceSettings):
        return {
            "kind": "fixed_source",
            "linear_solve": _linear_solve_to_data(settings.linear_solve),
            "flux_nonnegativity_tolerance": settings.flux_nonnegativity_tolerance,
        }
    if isinstance(settings, KeffSettings):
        return {
            "kind": "keff",
            "inner_linear_solve": _linear_solve_to_data(settings.inner_linear_solve),
            "max_outer_iterations": settings.max_outer_iterations,
            "keff_change_tolerance": settings.keff_change_tolerance,
            "flux_change_tolerance": settings.flux_change_tolerance,
            "keff_relative_residual_tolerance": (
                settings.keff_relative_residual_tolerance
            ),
            "flux_nonnegativity_tolerance": settings.flux_nonnegativity_tolerance,
            "eigenvalue_iteration": _eigenvalue_iteration_to_data(
                settings.eigenvalue_iteration
            ),
        }
    raise ValueError("result solve_settings has an unsupported type")


def _settings_from_data(data: object) -> FixedSourceSettings | KeffSettings:
    """Restore checked mode-specific solve settings from manifest fields."""
    if not isinstance(data, dict):
        raise ValueError("archive solve_settings must be a JSON object")
    kind = data.get("kind")
    if kind == "fixed_source":
        fields = {"kind", "linear_solve", "flux_nonnegativity_tolerance"}
        _require_keys(data, fields, "solve_settings")
        return FixedSourceSettings(
            linear_solve=_linear_solve_from_data(data["linear_solve"]),
            flux_nonnegativity_tolerance=data["flux_nonnegativity_tolerance"],
        )
    if kind == "keff":
        fields = {
            "kind",
            "inner_linear_solve",
            "max_outer_iterations",
            "keff_change_tolerance",
            "flux_change_tolerance",
            "keff_relative_residual_tolerance",
            "flux_nonnegativity_tolerance",
            "eigenvalue_iteration",
        }
        _require_keys(data, fields, "solve_settings")
        return KeffSettings(
            inner_linear_solve=_linear_solve_from_data(data["inner_linear_solve"]),
            max_outer_iterations=data["max_outer_iterations"],
            keff_change_tolerance=data["keff_change_tolerance"],
            flux_change_tolerance=data["flux_change_tolerance"],
            keff_relative_residual_tolerance=data["keff_relative_residual_tolerance"],
            flux_nonnegativity_tolerance=data["flux_nonnegativity_tolerance"],
            eigenvalue_iteration=_eigenvalue_iteration_from_data(
                data["eigenvalue_iteration"]
            ),
        )
    raise ValueError("archive solve_settings has an unknown kind")


def _eigenvalue_iteration_to_data(policy: object) -> dict[str, object]:
    """Convert one checked eigenvalue-iteration policy to tagged archive data."""
    if isinstance(policy, PowerIterationSettings):
        return {"kind": "power"}
    if isinstance(policy, WielandtShiftSettings):
        return {
            "kind": "wielandt",
            "shift_inverse_keff": policy.shift_inverse_keff,
        }
    raise ValueError("eigenvalue iteration has an unsupported type")


def _eigenvalue_iteration_from_data(
    data: object,
) -> PowerIterationSettings | WielandtShiftSettings:
    """Restore one checked tagged eigenvalue-iteration policy from archive data."""
    if not isinstance(data, dict):
        raise ValueError("archive eigenvalue iteration must be a JSON object")
    if data.get("kind") == "power":
        _require_keys(data, {"kind"}, "eigenvalue iteration")
        return PowerIterationSettings()
    if data.get("kind") == "wielandt":
        _require_keys(
            data,
            {"kind", "shift_inverse_keff"},
            "eigenvalue iteration",
        )
        return WielandtShiftSettings(shift_inverse_keff=data["shift_inverse_keff"])
    raise ValueError("archive eigenvalue iteration has an unknown kind")


def _linear_solve_to_data(settings: object) -> dict[str, object]:
    """Convert one checked linear-solve policy to tagged archive data."""
    if isinstance(settings, DirectLinearSolveSettings):
        return {
            "kind": "direct",
            "relative_residual_tolerance": settings.relative_residual_tolerance,
        }
    if isinstance(settings, GmresLinearSolveSettings):
        return {
            "kind": "gmres",
            "relative_residual_tolerance": settings.relative_residual_tolerance,
            "max_krylov_iterations": settings.max_krylov_iterations,
            "restart": settings.restart,
            "preconditioner": _preconditioner_to_data(settings.preconditioner),
        }
    raise ValueError("linear solve has an unsupported type")


def _linear_solve_from_data(
    data: object,
) -> DirectLinearSolveSettings | GmresLinearSolveSettings:
    """Restore one checked tagged linear-solve policy from archive data."""
    if not isinstance(data, dict):
        raise ValueError("archive linear solve must be a JSON object")
    if data.get("kind") == "direct":
        _require_keys(data, {"kind", "relative_residual_tolerance"}, "linear solve")
        return DirectLinearSolveSettings(
            relative_residual_tolerance=data["relative_residual_tolerance"]
        )
    if data.get("kind") == "gmres":
        _require_keys(
            data,
            {
                "kind",
                "relative_residual_tolerance",
                "max_krylov_iterations",
                "restart",
                "preconditioner",
            },
            "linear solve",
        )
        return GmresLinearSolveSettings(
            relative_residual_tolerance=data["relative_residual_tolerance"],
            max_krylov_iterations=data["max_krylov_iterations"],
            restart=data["restart"],
            preconditioner=_preconditioner_from_data(data["preconditioner"]),
        )
    raise ValueError("archive linear solve has an unknown kind")


def _preconditioner_to_data(preconditioner: object) -> dict[str, object]:
    """Convert one typed GMRES preconditioner to tagged archive data."""
    if isinstance(preconditioner, NoPreconditioner):
        return {"kind": "none"}
    if isinstance(preconditioner, JacobiPreconditioner):
        return {"kind": "jacobi"}
    if isinstance(preconditioner, IluPreconditioner):
        return {
            "kind": "ilu",
            "drop_tolerance": preconditioner.drop_tolerance,
            "fill_factor": preconditioner.fill_factor,
        }
    raise ValueError("preconditioner has an unsupported type")


def _preconditioner_from_data(
    data: object,
) -> NoPreconditioner | JacobiPreconditioner | IluPreconditioner:
    """Restore one checked tagged GMRES preconditioner from archive data."""
    if not isinstance(data, dict):
        raise ValueError("archive preconditioner must be a JSON object")
    if data.get("kind") == "none":
        _require_keys(data, {"kind"}, "preconditioner")
        return NoPreconditioner()
    if data.get("kind") == "jacobi":
        _require_keys(data, {"kind"}, "preconditioner")
        return JacobiPreconditioner()
    if data.get("kind") == "ilu":
        _require_keys(data, {"kind", "drop_tolerance", "fill_factor"}, "preconditioner")
        return IluPreconditioner(
            drop_tolerance=data["drop_tolerance"], fill_factor=data["fill_factor"]
        )
    raise ValueError("archive preconditioner has an unknown kind")


def _execution_report_to_data(
    report: LinearSolveReport | KeffSolveReport,
) -> dict[str, object]:
    """Convert typed completed execution diagnostics to tagged archive data."""
    if isinstance(report, LinearSolveReport):
        return {
            "kind": "linear_solve",
            "report": _linear_report_to_data(report),
        }
    if isinstance(report, KeffSolveReport):
        return {
            "kind": "keff",
            "eigenvalue_iteration": _eigenvalue_iteration_to_data(
                report.eigenvalue_iteration
            ),
            "outer_iterations": [
                _outer_iteration_report_to_data(item)
                for item in report.outer_iterations
            ],
        }
    raise ValueError("execution report has an unsupported type")


def _execution_report_from_data(
    data: object,
) -> LinearSolveReport | KeffSolveReport:
    """Restore typed completed execution diagnostics from archive data."""
    if not isinstance(data, dict):
        raise ValueError("archive execution_report must be a JSON object")
    if data.get("kind") == "linear_solve":
        _require_keys(data, {"kind", "report"}, "execution_report")
        return _linear_report_from_data(data["report"])
    if data.get("kind") == "keff":
        _require_keys(
            data,
            {"kind", "eigenvalue_iteration", "outer_iterations"},
            "execution_report",
        )
        values = data["outer_iterations"]
        if not isinstance(values, list):
            raise ValueError("archive outer_iterations must be a list")
        return KeffSolveReport._from_validated(  # pylint: disable=protected-access
            tuple(_outer_iteration_report_from_data(item) for item in values),
            _eigenvalue_iteration_from_data(data["eigenvalue_iteration"]),
        )
    raise ValueError("archive execution_report has an unknown kind")


def _linear_report_to_data(report: LinearSolveReport) -> dict[str, object]:
    """Convert one completed linear report to explicit archive data."""
    return {
        "linear_solve": _linear_solve_to_data(report.linear_solve),
        "iterations": report.iterations,
        "true_relative_residual": report.true_relative_residual,
    }


def _linear_report_from_data(data: object) -> LinearSolveReport:
    """Restore one checked completed linear report from archive data."""
    if not isinstance(data, dict):
        raise ValueError("archive linear solve report must be a JSON object")
    _require_keys(
        data,
        {"linear_solve", "iterations", "true_relative_residual"},
        "linear solve report",
    )
    return LinearSolveReport._from_validated(  # pylint: disable=protected-access
        linear_solve=_linear_solve_from_data(data["linear_solve"]),
        iterations=data["iterations"],
        true_relative_residual=data["true_relative_residual"],
    )


def _outer_iteration_report_to_data(
    report: KeffOuterIterationReport,
) -> dict[str, object]:
    """Convert one criticality outer-iteration report to archive data."""
    return {
        "iteration": report.iteration,
        "linear_solve": _linear_report_to_data(report.linear_solve),
        "keff": report.keff,
        "keff_change": report.keff_change,
        "flux_change": report.flux_change,
        "keff_relative_residual": report.keff_relative_residual,
    }


def _outer_iteration_report_from_data(data: object) -> KeffOuterIterationReport:
    """Restore one checked criticality outer-iteration report from archive data."""
    if not isinstance(data, dict):
        raise ValueError("archive outer iteration report must be a JSON object")
    _require_keys(
        data,
        {
            "iteration",
            "linear_solve",
            "keff",
            "keff_change",
            "flux_change",
            "keff_relative_residual",
        },
        "outer iteration report",
    )
    return KeffOuterIterationReport._from_validated(  # pylint: disable=protected-access
        iteration=data["iteration"],
        linear_solve=_linear_report_from_data(data["linear_solve"]),
        keff=data["keff"],
        keff_change=data["keff_change"],
        flux_change=data["flux_change"],
        keff_relative_residual=data["keff_relative_residual"],
    )


def _normalization_to_data(
    normalization: FissionSourceNormalization | PowerNormalization | None,
) -> dict[str, object] | None:
    """Return explicit eigenvalue normalization provenance or ``None``."""
    if normalization is None:
        return None
    if isinstance(normalization, FissionSourceNormalization):
        return {"kind": "fission_source_rate", "rate": normalization.rate}
    return {"kind": "power", "power": normalization.power}


def _normalization_from_data(
    data: object,
) -> FissionSourceNormalization | PowerNormalization | None:
    """Restore checked eigenvalue normalization from manifest data."""
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("archive normalization must be a JSON object or null")
    if data.get("kind") == "fission_source_rate":
        _require_keys(data, {"kind", "rate"}, "normalization")
        return FissionSourceNormalization(rate=data["rate"])
    if data.get("kind") == "power":
        _require_keys(data, {"kind", "power"}, "normalization")
        return PowerNormalization(power=data["power"])
    raise ValueError("archive normalization has an unknown kind")


def _snapshot_to_data(
    snapshot: ProblemConfigurationSnapshot,
    arrays: _ArrayStore,
) -> dict[str, object]:
    """Convert a snapshot to manifest data and register its contained arrays."""
    mesh = snapshot.mesh
    material_mesh = snapshot.material_mesh
    return {
        "name": snapshot.name,
        "mesh": {"num_rings": mesh.num_rings, "pitch": mesh.pitch},
        "layers": [
            {
                "height": material_mesh.axial_layer_heights[axial_index],
                "material_keys": [layer[index] for index in mesh.openmc_indices],
            }
            for axial_index, layer in enumerate(material_mesh.layers)
        ],
        "excluded_regions": [
            {"key": key, "kind": region.kind, "color": region.color}
            for key, region in sorted(snapshot.excluded_region_map().items())
        ],
        "materials": [
            _material_to_data(material, arrays)
            for _, material in sorted(snapshot.materials.items())
        ],
        "boundary": [
            _boundary_assignment_to_data(assignment, arrays)
            for assignment in snapshot.boundary.assignments
        ],
        "source": _source_to_data(snapshot.source, arrays),
    }


def _snapshot_from_data(
    data: object,
    arrays: _ArrayReader,
) -> ProblemConfigurationSnapshot:
    """Restore and validate complete configuration provenance from a manifest."""
    if not isinstance(data, dict):
        raise ValueError("archive configuration_snapshot must be a JSON object")
    _require_keys(
        data,
        {
            "name",
            "mesh",
            "layers",
            "excluded_regions",
            "materials",
            "boundary",
            "source",
        },
        "configuration_snapshot",
    )
    mesh_data = data["mesh"]
    if not isinstance(mesh_data, dict):
        raise ValueError("archive snapshot mesh must be a JSON object")
    _require_keys(mesh_data, {"num_rings", "pitch"}, "snapshot mesh")
    mesh = HexPlanarMesh(**mesh_data)
    materials_data = data["materials"]
    if not isinstance(materials_data, list):
        raise ValueError("archive snapshot materials must be a list")
    materials = [_material_from_data(item, arrays) for item in materials_data]
    material_map = {material.name: material for material in materials}
    if len(material_map) != len(materials):
        raise ValueError("archive snapshot material names must be unique")
    excluded_data = data["excluded_regions"]
    if not isinstance(excluded_data, list):
        raise ValueError("archive snapshot excluded_regions must be a list")
    excluded_regions = _excluded_regions_from_data(excluded_data)
    layers_data = data["layers"]
    if not isinstance(layers_data, list) or not layers_data:
        raise ValueError("archive snapshot layers must be a non-empty list")
    slices = tuple(_slice_from_data(item, mesh) for item in layers_data)
    boundary_data = data["boundary"]
    if not isinstance(boundary_data, list):
        raise ValueError("archive snapshot boundary must be a list")
    boundary = BoundaryConditionSet(
        *(_boundary_assignment_from_data(item, arrays) for item in boundary_data)
    )
    source = _source_from_data(data["source"], arrays)
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials=material_map,
        material_mesh=MaterialMesh.stack(slices, excluded_regions=excluded_regions),
        boundary=boundary,
        source=source,
        name=data["name"],
    )
    return configuration.snapshot()


def _material_to_data(material: Material, arrays: _ArrayStore) -> dict[str, object]:
    """Convert one material to manifest data and register its cross-section arrays."""
    return {
        "name": material.name,
        "cross_sections": _cross_sections_to_data(material.xs, arrays),
        "color": material.color,
    }


def _material_from_data(data: object, arrays: _ArrayReader) -> Material:
    """Restore one checked immutable material definition."""
    if not isinstance(data, dict):
        raise ValueError("archive material must be a JSON object")
    _require_keys(data, {"name", "cross_sections", "color"}, "material")
    return Material(
        name=data["name"],
        xs=_cross_sections_from_data(data["cross_sections"], arrays),
        color=data["color"],
    )


def _cross_sections_to_data(
    cross_sections: CrossSections | None,
    arrays: _ArrayStore,
) -> dict[str, object] | None:
    """Convert cross sections to manifest data and register their arrays."""
    if cross_sections is None:
        return None
    return {
        "D": arrays.add("cross_sections_D", cross_sections.D),
        "sigma_a": arrays.add("cross_sections_sigma_a", cross_sections.sigma_a),
        "sigma_s": arrays.add("cross_sections_sigma_s", cross_sections.sigma_s),
        "multiplicity_matrix": (
            None
            if cross_sections.multiplicity_matrix is None
            else arrays.add(
                "cross_sections_multiplicity_matrix",
                cross_sections.multiplicity_matrix,
            )
        ),
        "fission": _fission_data_to_data(cross_sections.fission, arrays),
    }


def _cross_sections_from_data(
    data: object,
    arrays: _ArrayReader,
) -> CrossSections | None:
    """Restore checked cross sections from array payload references."""
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("archive cross_sections must be a JSON object or null")
    _require_keys(
        data,
        {
            "D",
            "sigma_a",
            "sigma_s",
            "multiplicity_matrix",
            "fission",
        },
        "cross_sections",
    )
    return CrossSections(
        D=arrays.load(data["D"]),
        sigma_a=arrays.load(data["sigma_a"]),
        sigma_s=arrays.load(data["sigma_s"]),
        multiplicity_matrix=(
            None
            if data["multiplicity_matrix"] is None
            else arrays.load(data["multiplicity_matrix"])
        ),
        fission=_fission_data_from_data(data["fission"], arrays),
    )


def _fission_data_to_data(
    fission: FissionData | None, arrays: _ArrayStore
) -> dict[str, object] | None:
    """Convert one optional tagged fission-physics bundle to manifest data."""
    if fission is None:
        return None
    neutron_production = fission.neutron_production
    if isinstance(neutron_production, SeparableFission):
        production_data = {
            "kind": "separable",
            "nu_sigma_f": arrays.add(
                "fission_nu_sigma_f", neutron_production.nu_sigma_f
            ),
            "chi": arrays.add("fission_chi", neutron_production.chi),
            "chi_normalization_tolerance": (
                neutron_production.chi_normalization_tolerance
            ),
        }
    else:
        production_data = {
            "kind": "transfer",
            "fission_transfer": arrays.add(
                "fission_transfer", neutron_production.fission_transfer
            ),
        }
    return {
        "neutron_production": production_data,
        "kappa_sigma_f": (
            None
            if fission.kappa_sigma_f is None
            else arrays.add("fission_kappa_sigma_f", fission.kappa_sigma_f)
        ),
    }


def _fission_data_from_data(data: object, arrays: _ArrayReader) -> FissionData | None:
    """Restore one optional checked tagged fission-physics bundle."""
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("archive fission must be a JSON object or null")
    _require_keys(data, {"neutron_production", "kappa_sigma_f"}, "fission")
    production_data = data["neutron_production"]
    if not isinstance(production_data, dict):
        raise ValueError("archive fission neutron_production must be a JSON object")
    kind = production_data.get("kind")
    if kind == "separable":
        _require_keys(
            production_data,
            {"kind", "nu_sigma_f", "chi", "chi_normalization_tolerance"},
            "fission neutron_production",
        )
        production = SeparableFission(
            nu_sigma_f=arrays.load(production_data["nu_sigma_f"]),
            chi=arrays.load(production_data["chi"]),
            chi_normalization_tolerance=production_data["chi_normalization_tolerance"],
        )
    elif kind == "transfer":
        _require_keys(
            production_data,
            {"kind", "fission_transfer"},
            "fission neutron_production",
        )
        production = FissionTransfer(
            fission_transfer=arrays.load(production_data["fission_transfer"])
        )
    else:
        raise ValueError("archive fission neutron_production kind is unsupported")
    return FissionData(
        neutron_production=production,
        kappa_sigma_f=(
            None
            if data["kappa_sigma_f"] is None
            else arrays.load(data["kappa_sigma_f"])
        ),
    )


def _boundary_assignment_to_data(
    assignment: BoundaryAssignment,
    arrays: _ArrayStore,
) -> dict[str, object]:
    """Convert one boundary assignment to manifest data and register its arrays."""
    selector = assignment.selector
    condition = assignment.condition
    return {
        "selector": {
            "scope": selector.scope,
            "direction": selector.direction,
            "excluded_key": selector.excluded_key,
            "excluded_kind": selector.excluded_kind,
        },
        "condition": {
            "kind": condition.kind,
            "flux": (
                None
                if condition.flux is None
                else arrays.add("boundary_flux", condition.flux)
            ),
            "alpha": condition.alpha,
            "beta": condition.beta,
            "current": (
                None
                if condition.current is None
                else arrays.add("boundary_current", condition.current)
            ),
        },
    }


def _boundary_assignment_from_data(
    data: object,
    arrays: _ArrayReader,
) -> BoundaryAssignment:
    """Restore one checked selector-bound condition from archive fields."""
    if not isinstance(data, dict):
        raise ValueError("archive boundary assignment must be a JSON object")
    _require_keys(data, {"selector", "condition"}, "boundary assignment")
    selector_data = data["selector"]
    condition_data = data["condition"]
    if not isinstance(selector_data, dict) or not isinstance(condition_data, dict):
        raise ValueError("archive boundary assignment fields must be JSON objects")
    _require_keys(
        selector_data,
        {"scope", "direction", "excluded_key", "excluded_kind"},
        "boundary selector",
    )
    _require_keys(
        condition_data,
        {"kind", "flux", "alpha", "beta", "current"},
        "boundary condition",
    )
    return BoundaryAssignment(
        selector=BoundarySelector(**selector_data),
        condition=BoundaryCondition(
            kind=condition_data["kind"],
            flux=(
                None
                if condition_data["flux"] is None
                else arrays.load(condition_data["flux"])
            ),
            alpha=condition_data["alpha"],
            beta=condition_data["beta"],
            current=(
                None
                if condition_data["current"] is None
                else arrays.load(condition_data["current"])
            ),
        ),
    )


def _source_to_data(
    source: UniformSource | MaterialSource | CellSource | None,
    arrays: _ArrayStore,
) -> dict[str, object] | None:
    """Convert one built-in source to manifest data and register its arrays."""
    if source is None:
        return None
    if isinstance(source, UniformSource):
        return {
            "kind": "uniform",
            "strength": arrays.add("source_uniform", source.strength),
        }
    if isinstance(source, MaterialSource):
        return {
            "kind": "material",
            "values_by_material": [
                {
                    "material": material,
                    "values": arrays.add("source_material", values),
                }
                for material, values in sorted(source.values_by_material.items())
            ],
        }
    if isinstance(source, CellSource):
        return {
            "kind": "cell",
            "layers": [
                arrays.add("source_cell_layer", layer) for layer in source.layers
            ],
        }
    raise ValueError("checked source has an unsupported type")


def _source_from_data(
    data: object,
    arrays: _ArrayReader,
) -> UniformSource | MaterialSource | CellSource | None:
    """Restore one checked built-in source from archive fields."""
    if data is None:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("kind"), str):
        raise ValueError("archive source must be a JSON object or null")
    kind = data["kind"]
    if kind == "uniform":
        _require_keys(data, {"kind", "strength"}, "uniform source")
        return UniformSource(arrays.load(data["strength"]))
    if kind == "material":
        _require_keys(data, {"kind", "values_by_material"}, "material source")
        values = data["values_by_material"]
        if not isinstance(values, list):
            raise ValueError("archive material source values must be a list")
        restored: dict[str, np.ndarray] = {}
        for entry in values:
            if not isinstance(entry, dict):
                raise ValueError("archive material source entry must be a JSON object")
            _require_keys(entry, {"material", "values"}, "material source entry")
            material = entry["material"]
            if not isinstance(material, str) or material in restored:
                raise ValueError("archive material source material name is invalid")
            restored[material] = arrays.load(entry["values"])
        return MaterialSource(restored)
    if kind == "cell":
        _require_keys(data, {"kind", "layers"}, "cell source")
        layers = data["layers"]
        if not isinstance(layers, list):
            raise ValueError("archive cell source layers must be a list")
        return CellSource(tuple(arrays.load(member) for member in layers))
    raise ValueError(f"archive source kind is unsupported: {kind!r}")


def _excluded_regions_from_data(data: list[object]) -> dict[str, ExcludedRegion]:
    """Restore the checked excluded-region catalog from manifest entries."""
    regions: dict[str, ExcludedRegion] = {}
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError("archive excluded-region entry must be a JSON object")
        _require_keys(entry, {"key", "kind", "color"}, "excluded region")
        key = entry["key"]
        if not isinstance(key, str) or key in regions:
            raise ValueError("archive excluded-region key is invalid")
        regions[key] = ExcludedRegion(kind=entry["kind"], color=entry["color"])
    return regions


def _slice_from_data(data: object, mesh: HexPlanarMesh) -> MaterialSlice:
    """Restore one complete material slice from ordered material keys."""
    if not isinstance(data, dict):
        raise ValueError("archive material layer must be a JSON object")
    _require_keys(data, {"height", "material_keys"}, "material layer")
    material_keys = data["material_keys"]
    if not isinstance(material_keys, list) or len(material_keys) != mesh.n_cells:
        raise ValueError("archive material layer keys must cover the mesh")
    return MaterialSlice(
        mesh=mesh,
        material_keys=dict(zip(mesh.openmc_indices, material_keys)),
        height=data["height"],
    )


def _require_keys(
    data: Mapping[str, object],
    expected: set[str],
    context: str,
) -> None:
    """Require an exact schema object key set."""
    if set(data) != expected:
        raise ValueError(f"archive {context} has unexpected or missing fields")
