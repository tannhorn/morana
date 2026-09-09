"""Tests for structured solver results."""

from dataclasses import FrozenInstanceError
from fractions import Fraction
import json
import math
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
from plotly.colors import get_colorscale
import pytest

from morana import (
    FissionData,
    FissionTransfer,
    SeparableFission,
    CellInspection,
    BoundaryCondition,
    BoundaryConditionSet,
    CellSource,
    CrossSections,
    DirectLinearSolveSettings,
    ExcludedRegion,
    FixedSourceSettings,
    FissionSourceNormalization,
    FixedSourceBalance,
    HexPlanarMesh,
    Material,
    MaterialMesh,
    MaterialSource,
    MaterialSlice,
    KeffSettings,
    KeffBalance,
    KeffSolveReport,
    KeffOuterIterationReport,
    LinearSolveReport,
    OpenMCIndex,
    PowerIterationSettings,
    ProblemConfiguration,
    Result,
    PowerNormalization,
    UniformSource,
    WielandtShiftSettings,
)
from morana._plotting import EXCLUDED_FLUX_COLOR, FLUX_COLORMAP
from morana.results import _flux_normalize
from morana.solvers.finite_volume import solve_fixed_source, solve_keff
from tests.morana.test_configuration import make_configuration


def _rewrite_archive(
    source: Path,
    destination: Path,
    transform: callable,
) -> None:
    """Copy a ZIP archive while transforming selected member payloads."""
    with zipfile.ZipFile(source) as input_archive:
        with zipfile.ZipFile(destination, mode="w") as output_archive:
            for info in input_archive.infolist():
                output_archive.writestr(
                    info.filename,
                    transform(info.filename, input_archive.read(info.filename)),
                )


def _manual_linear_report() -> LinearSolveReport:
    """Return a minimal completed direct-solve report for manual results."""
    return LinearSolveReport._from_validated(
        linear_solve=DirectLinearSolveSettings(),
        iterations=1,
        true_relative_residual=0.0,
    )


def _manual_fixed_source_balance(flux: tuple[object, ...]) -> FixedSourceBalance:
    """Return complete zero fixed-source balance provenance."""
    groups = np.asarray(flux[0]).shape[0]
    terms = (
        "source",
        "boundary_source",
        "scattering_coupling",
        "fission_production",
        "fission_emission",
        "removal",
        "absorption",
        "radial_leakage",
        "axial_leakage",
        "net_scattering",
        "residual",
    )
    by_layer_group = {name: tuple(np.zeros(groups) for _ in flux) for name in terms}
    return FixedSourceBalance._from_validated(
        by_group={name: np.zeros(groups) for name in terms},
        by_layer_group=by_layer_group,
    )


def _manual_keff_balance(flux: tuple[object, ...], keff: float) -> KeffBalance:
    """Return a complete criticality balance satisfying the scalar equation."""
    groups = np.asarray(flux[0]).shape[0]
    source = tuple(np.ones(groups) for _ in flux)
    zeros = tuple(np.zeros(groups) for _ in flux)
    by_layer_group = {
        "fission_production": tuple(keff * values for values in source),
        "keff_source": source,
        "scattering_coupling": zeros,
        "removal": zeros,
        "absorption": source,
        "radial_leakage": zeros,
        "axial_leakage": zeros,
        "net_scattering": zeros,
        "residual": zeros,
    }
    return KeffBalance._from_validated(
        by_group={
            name: np.sum(values, axis=0) for name, values in by_layer_group.items()
        },
        by_layer_group=by_layer_group,
    )


def _manual_fixed_source_result(
    flux: tuple[object, ...] = (np.ones((1, 1)),),
    *,
    configuration: ProblemConfiguration | None = None,
    balance: FixedSourceBalance | None = None,
) -> Result:
    """Build a complete fixed-source record for targeted synthetic tests."""
    result_configuration = (
        make_configuration() if configuration is None else configuration
    )
    return Result._from_validated(
        flux=flux,  # type: ignore[arg-type]
        execution_report=_manual_linear_report(),
        balance=_manual_fixed_source_balance(flux) if balance is None else balance,
        configuration_snapshot=result_configuration.snapshot(),
        solve_settings=FixedSourceSettings(),
    )


def test_result_cell_at_returns_material_cross_sections_and_flux() -> None:
    """Cell inspection should hide layer-local active-ID lookup."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    cross_sections = CrossSections(
        D=[1.2, 0.9],
        sigma_a=[0.02, 0.03],
        sigma_s=[[0.0, 0.0], [0.0, 0.0]],
        fission=None,
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": Material("medium", xs=cross_sections)},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),)
        ),
    )
    result = _manual_fixed_source_result(
        flux=(np.array([[1.5], [2.5]]),),
        configuration=configuration,
    )

    cell = result.cell_at(0, OpenMCIndex(0, 0))

    assert isinstance(cell, CellInspection)
    assert cell.material_key == "medium"
    np.testing.assert_allclose(cell.cross_sections.sigma_a, cross_sections.sigma_a)
    np.testing.assert_allclose(cell.flux, [1.5, 2.5])
    assert not cell.flux.flags.writeable


def test_cell_inspection_rejects_direct_construction() -> None:
    """Inspection values originate from completed result queries only."""
    with pytest.raises(TypeError, match="returned by Result.cell_at"):
        CellInspection()


def test_result_cell_at_rejects_excluded_position() -> None:
    """Excluded positions have neither flux nor material cross sections."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    cross_sections = make_configuration().materials["medium"].xs
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": Material("medium", xs=cross_sections)},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice(mesh, {OpenMCIndex(0, 0): "medium"}, height=1.0),)
        ),
    )
    result = _manual_fixed_source_result(
        flux=(np.array([[1.0]]),), configuration=configuration
    )

    with pytest.raises(ValueError, match="selected cell is excluded"):
        result.cell_at(0, OpenMCIndex(1, 0))


def _keff_configuration(*, kappa_sigma_f: float | None = None) -> ProblemConfiguration:
    """Return one fissionable one-cell criticality configuration."""
    mesh = HexPlanarMesh(1, pitch=10.0)
    medium = Material(
        "medium",
        xs=CrossSections(
            D=[1.2],
            sigma_a=[0.02],
            sigma_s=[[0.0]],
            fission=FissionData(
                neutron_production=SeparableFission(nu_sigma_f=[0.025], chi=[1.0]),
                kappa_sigma_f=None if kappa_sigma_f is None else [kappa_sigma_f],
            ),
        ),
    )
    return ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=MaterialMesh.stack(
            (MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),)
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
    )


def test_result_rejects_direct_construction() -> None:
    """Completed results originate from solvers or checked archives only."""
    with pytest.raises(TypeError, match="returned by solve functions"):
        Result()


def test_balances_reject_direct_construction() -> None:
    """Balances originate from solvers or checked archives only."""
    for balance_type in (FixedSourceBalance, KeffBalance):
        with pytest.raises(TypeError, match="retained by completed results"):
            balance_type()


def test_result_snapshot_is_immutable_and_not_live_configuration() -> None:
    """Mutating a configuration after result creation must not alter provenance."""
    configuration = make_configuration()
    result = _manual_fixed_source_result(configuration=configuration)

    configuration.set_boundary(
        BoundaryConditionSet(BoundaryCondition.vacuum().globally())
    )

    assert result.configuration_snapshot.boundary.assignments[0].condition.kind == (
        "reflective"
    )
    with pytest.raises(FrozenInstanceError):
        result.configuration_snapshot.name = "changed"


def test_result_snapshot_preserves_custom_excluded_regions() -> None:
    """Result checking and inspection should use snapshot excluded-region details."""
    configuration = make_configuration()
    mesh = configuration.mesh
    configuration.set_material_mesh(
        MaterialMesh.stack(
            (
                MaterialSlice(mesh, {mesh.openmc_indices[0]: "medium"}, 1.0),
                MaterialSlice(mesh, {mesh.openmc_indices[0]: "gap"}, 1.0),
            ),
            excluded_regions={"gap": ExcludedRegion(kind="void")},
        )
    )

    result = _manual_fixed_source_result(
        flux=(np.ones((1, 1)), np.empty((1, 0))), configuration=configuration
    )

    assert result.configuration_snapshot is not None
    assert result.configuration_snapshot.excluded_region_map()["gap"].kind == "void"


def test_result_archive_round_trip_preserves_checked_provenance(tmp_path: Path) -> None:
    """A non-pickle archive should restore independent solved-result inputs."""
    configuration = make_configuration()
    configuration.set_name("archived fixed-source case")
    result = solve_fixed_source(configuration)
    archive = tmp_path / "nested" / "result.morana-result"

    result.save_to_disk(archive)
    loaded = Result.load_from_disk(archive)

    assert archive.is_file()
    assert loaded is not result
    assert loaded.keff == result.keff
    assert loaded.solve_settings == result.solve_settings
    assert loaded.solve_mode == result.solve_mode
    assert loaded.execution_report == result.execution_report
    assert loaded.configuration_snapshot is not result.configuration_snapshot
    assert loaded.configuration_snapshot.name == result.configuration_snapshot.name
    assert (
        loaded.configuration_snapshot.boundary.assignments
        == result.configuration_snapshot.boundary.assignments
    )
    np.testing.assert_allclose(loaded.flux_layer(0), result.flux_layer(0))
    for name, values in result.balance.by_group.items():
        np.testing.assert_allclose(loaded.balance.by_group[name], values)
        assert loaded.balance.scalar[name] == result.balance.scalar[name]

    with zipfile.ZipFile(archive) as container:
        manifest = json.loads(container.read("manifest.json"))
        assert manifest["format"] == "morana-result"
        assert manifest["schema_version"] == 8
        assert "revision" not in manifest["result"]["configuration_snapshot"]
        assert "solve_mode" not in manifest["result"]
        assert any(
            name.startswith("arrays/flux_layer") for name in container.namelist()
        )


@pytest.mark.parametrize("multiplicity_matrix", (None, [[1.5]]))
def test_result_archive_round_trips_scattering_multiplicity(
    tmp_path: Path,
    multiplicity_matrix: list[list[float]] | None,
) -> None:
    """Archive provenance should preserve compact and explicit multiplicity."""
    configuration = make_configuration()
    configuration.replace_material(
        Material(
            "medium",
            CrossSections(
                D=[1.2],
                sigma_a=[0.02],
                sigma_s=[[0.1]],
                fission=None,
                multiplicity_matrix=multiplicity_matrix,
            ),
        )
    )
    result = _manual_fixed_source_result(configuration=configuration)
    archive = tmp_path / "multiplicity.morana-result"

    result.save_to_disk(archive)
    restored = Result.load_from_disk(archive)

    restored_xs = restored.configuration_snapshot.materials["medium"].xs
    assert restored_xs is not None
    if multiplicity_matrix is None:
        assert restored_xs.multiplicity_matrix is None
    else:
        np.testing.assert_allclose(restored_xs.multiplicity_matrix, multiplicity_matrix)


@pytest.mark.parametrize(
    ("source", "source_type"),
    (
        (MaterialSource({"medium": [2.0]}), MaterialSource),
        (CellSource([[[3.0]]]), CellSource),
    ),
)
def test_result_archive_round_trip_preserves_other_builtin_sources(
    tmp_path: Path,
    source: MaterialSource | CellSource,
    source_type: type[MaterialSource] | type[CellSource],
) -> None:
    """Archive provenance should retain each supported built-in source form."""
    configuration = make_configuration()
    configuration.set_source(source)
    result = _manual_fixed_source_result(
        flux=(np.array([[1.0]]),), configuration=configuration
    )
    archive = tmp_path / "source.morana-result"

    result.save_to_disk(archive)
    loaded = Result.load_from_disk(archive)

    restored_source = loaded.configuration_snapshot.source
    assert isinstance(restored_source, source_type)
    if isinstance(restored_source, MaterialSource):
        np.testing.assert_allclose(restored_source.values_by_material["medium"], [2.0])
    else:
        np.testing.assert_allclose(restored_source.layers[0], [[3.0]])


def test_result_archive_rejects_modified_array_payload(tmp_path: Path) -> None:
    """Payload tampering should fail SHA-256 validation before reconstruction."""
    result = solve_fixed_source(make_configuration())
    archive = tmp_path / "result.morana-result"
    tampered = tmp_path / "tampered.morana-result"
    result.save_to_disk(archive)

    _rewrite_archive(
        archive,
        tampered,
        lambda name, payload: (
            b"not a NumPy payload" if name.startswith("arrays/") else payload
        ),
    )

    with pytest.raises(ValueError, match="checksum"):
        Result.load_from_disk(tampered)


def test_result_archive_rejects_linear_residual_above_policy_tolerance(
    tmp_path: Path,
) -> None:
    """Archive reports must remain consistent with their linear-solve policy."""
    result = solve_fixed_source(make_configuration())
    archive = tmp_path / "result.morana-result"
    inconsistent = tmp_path / "inconsistent.morana-result"
    result.save_to_disk(archive)

    def make_inconsistent_report(name: str, payload: bytes) -> bytes:
        if name != "manifest.json":
            return payload
        manifest = json.loads(payload)
        manifest["result"]["execution_report"]["report"]["true_relative_residual"] = 1.0
        return json.dumps(manifest).encode("utf-8")

    _rewrite_archive(archive, inconsistent, make_inconsistent_report)

    with pytest.raises(ValueError, match="must not exceed"):
        Result.load_from_disk(inconsistent)


@pytest.mark.parametrize("schema_version", [1, 2, 3, 4, 5, 6, 7, 9])
def test_result_archive_rejects_other_schema(
    tmp_path: Path, schema_version: int
) -> None:
    """A loader must require exactly the current manifest schema."""
    result = _manual_fixed_source_result(flux=(np.array([[1.0]]),))
    archive = tmp_path / "result.morana-result"
    newer = tmp_path / "newer.morana-result"
    result.save_to_disk(archive)

    def make_newer_manifest(name: str, payload: bytes) -> bytes:
        if name != "manifest.json":
            return payload
        manifest = json.loads(payload)
        manifest["schema_version"] = schema_version
        return json.dumps(manifest).encode("utf-8")

    _rewrite_archive(archive, newer, make_newer_manifest)

    with pytest.raises(ValueError, match="unsupported"):
        Result.load_from_disk(newer)


def test_result_archive_requires_execution_report_field(tmp_path: Path) -> None:
    """A current archive must retain its explicit execution-report field."""
    result = _manual_fixed_source_result(flux=(np.array([[1.0]]),))
    archive = tmp_path / "result.morana-result"
    incomplete = tmp_path / "incomplete.morana-result"
    result.save_to_disk(archive)

    def remove_execution_report(name: str, payload: bytes) -> bytes:
        if name != "manifest.json":
            return payload
        manifest = json.loads(payload)
        del manifest["result"]["execution_report"]
        return json.dumps(manifest).encode("utf-8")

    _rewrite_archive(archive, incomplete, remove_execution_report)

    with pytest.raises(ValueError, match="result has unexpected or missing fields"):
        Result.load_from_disk(incomplete)


def test_result_archive_rejects_unexpected_member(tmp_path: Path) -> None:
    """The manifest must account for every member in the ZIP container."""
    result = _manual_fixed_source_result(flux=(np.array([[1.0]]),))
    archive = tmp_path / "result.morana-result"
    result.save_to_disk(archive)

    with zipfile.ZipFile(archive, mode="a") as container:
        container.writestr("surprise.txt", "not declared by the manifest")

    with pytest.raises(ValueError, match="members"):
        Result.load_from_disk(archive)


def test_fixed_source_balance_owns_immutable_diagnostics() -> None:
    """Fixed-source balance should own checked read-only group and layer values."""
    balance = _manual_fixed_source_balance((np.ones((1, 1)),))

    with pytest.raises(TypeError):
        balance.by_group["new"] = np.array([1.0])
    with pytest.raises(ValueError, match="read-only"):
        balance.by_layer_group["source"][0][0] = 4.0


def test_fixed_source_balance_rejects_incomplete_schema() -> None:
    """Fixed-source balances must retain every documented diagnostic term."""
    with pytest.raises(ValueError, match="required schema"):
        FixedSourceBalance._from_validated(
            by_group={"source": np.zeros(1)},
            by_layer_group={"source": (np.zeros(1),)},
        )


def test_result_archive_round_trips_keff_settings(tmp_path: Path) -> None:
    """Archive provenance should preserve the criticality settings variant."""
    policy = WielandtShiftSettings(shift_inverse_keff=0.79)
    settings = KeffSettings(max_outer_iterations=12, eigenvalue_iteration=policy)
    normalization = FissionSourceNormalization(rate=3.0)
    result = solve_keff(_keff_configuration(), normalization, settings)
    archive = tmp_path / "keff.morana-result"

    result.save_to_disk(archive)
    loaded = Result.load_from_disk(archive)

    assert loaded.solve_settings == settings
    assert loaded.normalization == normalization
    assert loaded.execution_report == result.execution_report
    assert loaded.execution_report.eigenvalue_iteration == policy


def test_result_archive_round_trips_power_normalization(tmp_path: Path) -> None:
    """Archive provenance should preserve a thermal-power normalization."""
    normalization = PowerNormalization(power=1.0e6)
    result = solve_keff(_keff_configuration(kappa_sigma_f=2.0e6), normalization)
    archive = tmp_path / "power.morana-result"

    result.save_to_disk(archive)
    loaded = Result.load_from_disk(archive)

    assert loaded.normalization == normalization


def test_result_reconstruction_requires_matching_criticality_normalization() -> None:
    """Internal reconstruction enforces both physical normalizations."""
    configuration = _keff_configuration()
    execution_report = KeffSolveReport._from_validated(
        (
            KeffOuterIterationReport._from_validated(
                iteration=1,
                linear_solve=_manual_linear_report(),
                keff=1.1,
                keff_change=0.0,
                flux_change=0.0,
                keff_relative_residual=0.0,
            ),
        ),
        PowerIterationSettings(),
    )
    common = {
        "flux": (np.ones((1, 1)),),
        "execution_report": execution_report,
        "balance": _manual_keff_balance((np.ones((1, 1)),), keff=1.1),
        "configuration_snapshot": configuration,
        "solve_settings": KeffSettings(),
    }

    common["configuration_snapshot"] = configuration.snapshot()

    with pytest.raises(ValueError, match="fission-source rate normalization"):
        Result._from_validated(
            normalization=FissionSourceNormalization(rate=1.0), **common
        )
    with pytest.raises(ValueError, match="kappa_sigma_f"):
        Result._from_validated(normalization=PowerNormalization(power=1.0), **common)


def test_result_archive_rejects_inconsistent_fission_source_normalization(
    tmp_path: Path,
) -> None:
    """An archive rate must match fission production from its stored flux."""
    result = solve_keff(_keff_configuration(), FissionSourceNormalization(rate=3.0))
    archive = tmp_path / "source.morana-result"
    inconsistent = tmp_path / "inconsistent-source.morana-result"
    result.save_to_disk(archive)

    def change_rate(name: str, payload: bytes) -> bytes:
        if name != "manifest.json":
            return payload
        manifest = json.loads(payload)
        manifest["result"]["normalization"]["rate"] = 4.0
        return json.dumps(manifest).encode("utf-8")

    _rewrite_archive(archive, inconsistent, change_rate)

    with pytest.raises(ValueError, match="archive result data is invalid"):
        Result.load_from_disk(inconsistent)


def test_result_archive_rejects_power_normalization_without_energy_data(
    tmp_path: Path,
) -> None:
    """A power archive must retain kappa-sigma-f for its active fuel."""
    result = solve_keff(_keff_configuration(), FissionSourceNormalization(rate=3.0))
    archive = tmp_path / "source.morana-result"
    inconsistent = tmp_path / "inconsistent-power.morana-result"
    result.save_to_disk(archive)

    def select_power(name: str, payload: bytes) -> bytes:
        if name != "manifest.json":
            return payload
        manifest = json.loads(payload)
        manifest["result"]["normalization"] = {"kind": "power", "power": 1.0}
        return json.dumps(manifest).encode("utf-8")

    _rewrite_archive(archive, inconsistent, select_power)

    with pytest.raises(ValueError, match="archive result data is invalid"):
        Result.load_from_disk(inconsistent)


def test_result_archive_rejects_inconsistent_power_normalization(
    tmp_path: Path,
) -> None:
    """An archive power target must match its retained power response."""
    result = solve_keff(
        _keff_configuration(kappa_sigma_f=2.0e6), PowerNormalization(power=1.0e6)
    )
    archive = tmp_path / "power.morana-result"
    inconsistent = tmp_path / "inconsistent-power.morana-result"
    result.save_to_disk(archive)

    def change_power(name: str, payload: bytes) -> bytes:
        if name != "manifest.json":
            return payload
        manifest = json.loads(payload)
        manifest["result"]["normalization"]["power"] = 2.0e6
        return json.dumps(manifest).encode("utf-8")

    _rewrite_archive(archive, inconsistent, change_power)

    with pytest.raises(ValueError, match="archive result data is invalid"):
        Result.load_from_disk(inconsistent)


def test_result_archive_retains_fission_energy_production_data(tmp_path: Path) -> None:
    """Archived configuration provenance should retain kappa-sigma-f data."""
    configuration = make_configuration()
    configuration.set_materials(
        {
            "medium": Material(
                "medium",
                xs=CrossSections(
                    D=[1.2],
                    sigma_a=[0.02],
                    sigma_s=[[0.0]],
                    fission=FissionData(
                        neutron_production=SeparableFission(
                            nu_sigma_f=[0.025], chi=[1.0]
                        ),
                        kappa_sigma_f=[2.0e6],
                    ),
                ),
            )
        }
    )
    result = _manual_fixed_source_result(configuration=configuration)
    archive = tmp_path / "fission-energy.morana-result"

    result.save_to_disk(archive)
    loaded = Result.load_from_disk(archive)
    loaded_xs = loaded.configuration_snapshot.materials["medium"].xs

    assert loaded_xs is not None
    assert loaded_xs.fission is not None
    np.testing.assert_allclose(loaded_xs.fission.kappa_sigma_f, [2.0e6])


@pytest.mark.parametrize(
    ("neutron_production", "kind"),
    (
        (SeparableFission(nu_sigma_f=[0.025], chi=[1.0]), "separable"),
        (FissionTransfer(fission_transfer=[[0.025]]), "transfer"),
    ),
)
def test_result_archive_preserves_tagged_fission_representation(
    tmp_path: Path,
    neutron_production: SeparableFission | FissionTransfer,
    kind: str,
) -> None:
    """Archives must round-trip the selected fission representation itself."""
    configuration = make_configuration()
    configuration.replace_material(
        Material(
            "medium",
            xs=CrossSections(
                D=[1.2],
                sigma_a=[0.02],
                sigma_s=[[0.0]],
                fission=FissionData(neutron_production=neutron_production),
            ),
        )
    )
    result = _manual_fixed_source_result(configuration=configuration)
    archive = tmp_path / f"{kind}.morana-result"

    result.save_to_disk(archive)
    loaded = Result.load_from_disk(archive)

    with zipfile.ZipFile(archive) as container:
        manifest = json.loads(container.read("manifest.json"))
    material_data = manifest["result"]["configuration_snapshot"]["materials"][0]
    production_data = material_data["cross_sections"]["fission"]["neutron_production"]
    assert production_data["kind"] == kind
    assert set(production_data) == (
        {"kind", "nu_sigma_f", "chi", "chi_normalization_tolerance"}
        if kind == "separable"
        else {"kind", "fission_transfer"}
    )
    loaded_xs = loaded.configuration_snapshot.materials["medium"].xs
    assert loaded_xs is not None
    assert loaded_xs.fission is not None
    assert isinstance(loaded_xs.fission.neutron_production, type(neutron_production))


def test_result_archive_rejects_unknown_fission_representation_kind(
    tmp_path: Path,
) -> None:
    """Archive input must not guess how to interpret an unknown fission tag."""
    result = _manual_fixed_source_result(configuration=_keff_configuration())
    archive = tmp_path / "result.morana-result"
    malformed = tmp_path / "unknown-fission-kind.morana-result"
    result.save_to_disk(archive)

    def replace_fission_kind(name: str, payload: bytes) -> bytes:
        if name != "manifest.json":
            return payload
        manifest = json.loads(payload)
        material_data = manifest["result"]["configuration_snapshot"]["materials"][0]
        material_data["cross_sections"]["fission"]["neutron_production"][
            "kind"
        ] = "legacy_flattened"
        return json.dumps(manifest).encode("utf-8")

    _rewrite_archive(archive, malformed, replace_fission_kind)

    with pytest.raises(ValueError, match="kind is unsupported"):
        Result.load_from_disk(malformed)


def test_result_archive_rejects_malformed_tagged_fission_data(tmp_path: Path) -> None:
    """Each fission tag must retain exactly its declared representation fields."""
    result = _manual_fixed_source_result(configuration=_keff_configuration())
    archive = tmp_path / "result.morana-result"
    malformed = tmp_path / "incomplete-fission-data.morana-result"
    result.save_to_disk(archive)

    def remove_separable_spectrum(name: str, payload: bytes) -> bytes:
        if name != "manifest.json":
            return payload
        manifest = json.loads(payload)
        material_data = manifest["result"]["configuration_snapshot"]["materials"][0]
        del material_data["cross_sections"]["fission"]["neutron_production"]["chi"]
        return json.dumps(manifest).encode("utf-8")

    _rewrite_archive(archive, malformed, remove_separable_spectrum)

    with pytest.raises(ValueError, match="fission neutron_production"):
        Result.load_from_disk(malformed)


def test_keff_balance_derives_required_ratio_diagnostics() -> None:
    """Criticality balance ratios should derive from its retained scalar terms."""
    balance = _manual_keff_balance((np.ones((1, 1)),), keff=1.1)

    assert balance.loss_fractions["absorption"] == pytest.approx(1.0)
    assert balance.source_normalized["absorption"] == pytest.approx(1.0)


def test_result_flux_is_layered_and_group_major() -> None:
    """Result flux should store one group-major array per axial layer."""
    configuration = make_configuration()
    mesh = configuration.mesh
    configuration.set_material_mesh(
        MaterialMesh.stack(
            (
                MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),
                MaterialSlice.from_openmc_rings(mesh, [["medium"]], height=1.0),
            )
        )
    )
    result = _manual_fixed_source_result(
        flux=([[1.0]], [[3.0]]), configuration=configuration
    )

    assert result.groups == 1
    assert result.n_axial_layers == 2
    assert result.flux_layer(0).shape == (1, 1)
    assert result.flux_layer(1).shape == (1, 1)
    np.testing.assert_allclose(result.flux_layer(1), [[3.0]])


def test_result_owns_flux_array() -> None:
    """Mutating caller-owned flux arrays should not change a result."""
    flux = np.array([[1.0]])
    result = _manual_fixed_source_result(flux=(flux,))

    flux[0, 0] = 2.0

    np.testing.assert_allclose(result.flux_layer(0), [[1.0]])
    with pytest.raises(ValueError, match="read-only"):
        result.flux_layer(0)[0, 0] = 3.0


def test_result_is_immutable() -> None:
    """A completed result must not permit replacing checked fields."""
    result = _manual_fixed_source_result()

    with pytest.raises(FrozenInstanceError):
        result.flux = (np.zeros((1, 1)),)
    with pytest.raises(TypeError):
        result.balance["residual"] = 1.0


def test_result_owns_multigroup_flux_arrays() -> None:
    """Multigroup result ownership preserves each group-major component."""
    flux = np.array([[1.0], [3.0]])
    configuration = make_configuration()
    configuration.replace_material(
        Material(
            "medium",
            CrossSections(
                D=[1.0, 1.0],
                sigma_a=[0.1, 0.1],
                sigma_s=[[0.0, 0.0], [0.0, 0.0]],
                fission=None,
            ),
        )
    )
    result = _manual_fixed_source_result(flux=(flux,), configuration=configuration)

    flux[:, :] = 0.0

    assert result.groups == 2
    np.testing.assert_allclose(result.flux_layer(0), [[1.0], [3.0]])
    with pytest.raises(ValueError, match="read-only"):
        result.flux_layer(0)[1, 0] = 5.0


def test_result_requires_consistent_group_count_across_layers() -> None:
    """Every axial layer should use the same energy-group structure."""
    with pytest.raises(ValueError, match="same group count"):
        _manual_fixed_source_result(
            flux=(
                np.ones((1, 2)),
                np.ones((2, 1)),
            ),
        )


@pytest.mark.parametrize("axial_index", [True, 0.5])
def test_result_flux_layer_rejects_noninteger_axial_indices(
    axial_index: object,
) -> None:
    """Flux-layer selection must reject invalid index types."""
    result = _manual_fixed_source_result()

    with pytest.raises(TypeError, match="axial_index"):
        result.flux_layer(axial_index)  # type: ignore[arg-type]


@pytest.mark.parametrize("axial_index", [-1, 1])
def test_result_flux_layer_rejects_outside_axial_indices(axial_index: int) -> None:
    """Flux-layer selection must reject negative indices and overflow."""
    result = _manual_fixed_source_result()

    with pytest.raises(ValueError, match="axial_index"):
        result.flux_layer(axial_index)


@pytest.mark.parametrize("flux", [np.array([[np.nan]]), np.array([[np.inf]])])
def test_result_rejects_nonfinite_flux(flux: np.ndarray) -> None:
    """Completed scalar-flux fields should contain only finite values."""
    with pytest.raises(ValueError, match="flux"):
        _manual_fixed_source_result(flux=(flux,))


def test_result_rejects_empty_group_axis() -> None:
    """Every result layer should contain at least one energy group."""
    with pytest.raises(ValueError, match="at least one group"):
        Result._from_validated(
            flux=(np.empty((0, 1)),),
            execution_report=_manual_linear_report(),
            balance=_manual_fixed_source_balance((np.ones((1, 1)),)),
            configuration_snapshot=make_configuration().snapshot(),
            solve_settings=FixedSourceSettings(),
        )


def test_result_rejects_balance_with_invalid_residual_equation() -> None:
    """A balance residual must agree with the retained group balance terms."""
    balance = _manual_fixed_source_balance((np.ones((1, 1)),))
    by_group = dict(balance.by_group)
    by_layer_group = dict(balance.by_layer_group)
    by_group["residual"] = np.ones(1)
    by_layer_group["residual"] = (np.ones(1),)
    inconsistent = FixedSourceBalance._from_validated(
        by_group=by_group,
        by_layer_group=by_layer_group,
    )

    with pytest.raises(ValueError, match="residual"):
        _manual_fixed_source_result(balance=inconsistent)


def test_near_zero_uniform_flux_normalization_uses_unit_range() -> None:
    """Roundoff around zero should not be visually amplified."""
    normalization = _flux_normalize(np.array([-1.0e-15, 1.0e-15]))

    assert (normalization.vmin, normalization.vmax) == (0.0, 1.0)


def _make_ragged_result() -> Result:
    """Return a solved result with active and excluded planar positions."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    medium = Material(
        "medium",
        xs=CrossSections(
            D=[1.2],
            sigma_a=[0.02],
            sigma_s=[[0.0]],
            fission=None,
        ),
        color="#ff0000",
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=MaterialMesh.stack(
            (
                MaterialSlice.from_openmc_rings(
                    mesh, [["medium", "0", "0", "0", "0", "0"], ["medium"]], height=1.0
                ),
            ),
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
        source=UniformSource([1.0]),
    )
    return solve_fixed_source(configuration)


def _make_multigroup_ragged_result() -> Result:
    """Return a two-group result with active and excluded planar positions."""
    mesh = HexPlanarMesh(2, pitch=10.0)
    medium = Material(
        "medium",
        xs=CrossSections(
            D=[1.2, 0.4],
            sigma_a=[0.02, 0.04],
            sigma_s=[[0.0, 0.0], [0.0, 0.0]],
            fission=None,
        ),
    )
    configuration = ProblemConfiguration(
        mesh=mesh,
        materials={"medium": medium},
        material_mesh=MaterialMesh.stack(
            (
                MaterialSlice.from_openmc_rings(
                    mesh, [["medium", "0", "0", "0", "0", "0"], ["medium"]], height=1.0
                ),
            ),
        ),
        boundary=BoundaryConditionSet(BoundaryCondition.reflective().globally()),
        source=UniformSource([1.0, 1.0]),
    )
    return solve_fixed_source(configuration)


def test_result_matplotlib_plots_flux_slice_with_excluded_cells() -> None:
    """Matplotlib result plots should cover the full planar snapshot."""
    result = _make_ragged_result()

    axes = result.plot_matplotlib(group=0, axial_index=0)

    assert len(axes.patches) == 7
    assert axes.get_title() == "Flux group 0, axial slice 0"
    assert len(axes.figure.axes) == 2
    assert axes.patches[0].get_facecolor() == axes.patches[6].get_facecolor()
    assert axes.patches[1].get_facecolor() == to_rgba(EXCLUDED_FLUX_COLOR)
    assert axes.figure.axes[1].get_ylim() == pytest.approx((45.0, 55.0))
    assert axes.figure.axes[1].get_ylabel() == "scalar flux [n cm⁻² s⁻¹]"
    assert axes.figure.axes[1]._colorbar.mappable.cmap.name == FLUX_COLORMAP
    plt.close(axes.figure)


@pytest.mark.parametrize("ax", ["axes", object()])
def test_result_matplotlib_plot_requires_axes_or_none(ax: object) -> None:
    """Result plots should validate an explicitly supplied axes."""
    result = _make_ragged_result()

    with pytest.raises(TypeError, match="ax must be an Axes or None"):
        result.plot_matplotlib(group=0, axial_index=0, ax=ax)  # type: ignore[arg-type]


def test_result_plotly_plots_flux_slice_with_nan_exclusions() -> None:
    """Plotly result plots should expose flux and excluded-cell hover data."""
    result = _make_ragged_result()

    figure = result.plot_plotly(group=0, axial_index=0)

    assert len(figure.data) == 4
    assert "flux group 0: nan n cm⁻² s⁻¹" in figure.data[2].customdata[1]
    assert "material 0" in figure.data[2].customdata[1]
    assert "domain excluded" in figure.data[2].customdata[1]
    assert "flux group 0: 50 n cm⁻² s⁻¹" in figure.data[2].customdata[0]
    assert "ring 0, pos 0" in figure.data[2].customdata[0]
    assert "center (10.00, 0.00) cm" in figure.data[2].customdata[0]
    assert "z 0.00..1.00 cm" in figure.data[2].customdata[0]
    assert "domain active" in figure.data[2].customdata[0]
    assert figure.data[0].hoverinfo == "skip"
    assert figure.data[0].name == ""
    assert figure.data[1].fillcolor == EXCLUDED_FLUX_COLOR
    assert figure.data[2].marker.symbol == "hexagon"
    assert tuple(color for _, color in figure.data[-1].marker.colorscale) == tuple(
        color for _, color in get_colorscale(FLUX_COLORMAP)
    )
    assert figure.data[-1].marker.colorbar.title.text == "scalar flux [n cm⁻² s⁻¹]"


def test_result_multigroup_plots_select_requested_group_and_reject_bad_indices() -> (
    None
):
    """Both plotting backends preserve group selection for coupled results."""
    result = _make_multigroup_ragged_result()

    axes = result.plot_matplotlib(group=1, axial_index=0)
    figure = result.plot_plotly(group=1, axial_index=0)

    assert axes.get_title() == "Flux group 1, axial slice 0"
    assert "flux group 1: 25 n cm⁻² s⁻¹" in figure.data[2].customdata[0]
    plt.close(axes.figure)
    with pytest.raises(ValueError, match="group index 2"):
        result.plot_matplotlib(group=2, axial_index=0)
    with pytest.raises(ValueError, match="group index -1"):
        result.plot_plotly(group=-1, axial_index=0)


@pytest.mark.parametrize("group", [True, 0.5])
@pytest.mark.parametrize("method_name", ["plot_matplotlib", "plot_plotly"])
def test_result_plots_reject_noninteger_group_indices(
    group: object,
    method_name: str,
) -> None:
    """Result plots should reject Boolean and nonintegral group selectors."""
    result = _make_multigroup_ragged_result()

    with pytest.raises(TypeError, match="group must be an integer"):
        getattr(result, method_name)(group=group, axial_index=0)  # type: ignore[arg-type]


def test_result_export_vtm_writes_material_and_flux_blocks(tmp_path) -> None:
    """Result VTM should include material colors and NaN-fill exclusions."""
    import vtk

    result = _make_ragged_result()
    path = tmp_path / "result.vtm"

    result.export_vtm(path)

    reader = vtk.vtkXMLMultiBlockDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    multiblock = reader.GetOutput()
    active = multiblock.GetBlock(0).GetBlock(0)
    excluded = multiblock.GetBlock(1).GetBlock(0)
    active_data = active.GetCellData()
    excluded_data = excluded.GetCellData()

    assert multiblock.GetNumberOfBlocks() == 2
    assert active.GetNumberOfCells() == 2
    assert excluded.GetNumberOfCells() == 5
    assert active_data.GetArray("material_key_id") is not None
    assert active_data.GetArray("active_id") is not None
    assert active_data.GetArray("flux_g0").GetTuple1(0) == pytest.approx(50.0)
    assert math.isnan(excluded_data.GetArray("flux_g0").GetTuple1(0))


def test_result_export_vtm_writes_each_multigroup_flux_array(tmp_path) -> None:
    """VTM export preserves group order and NaN-fills every excluded cell."""
    import vtk

    result = _make_multigroup_ragged_result()
    path = tmp_path / "multigroup_result.vtm"

    result.export_vtm(path)

    reader = vtk.vtkXMLMultiBlockDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    multiblock = reader.GetOutput()
    active = multiblock.GetBlock(0).GetBlock(0).GetCellData()
    excluded = multiblock.GetBlock(1).GetBlock(0).GetCellData()

    assert active.GetArray("flux_g0").GetTuple1(0) == pytest.approx(50.0)
    assert active.GetArray("flux_g1").GetTuple1(0) == pytest.approx(25.0)
    assert math.isnan(excluded.GetArray("flux_g0").GetTuple1(0))
    assert math.isnan(excluded.GetArray("flux_g1").GetTuple1(0))


@pytest.mark.parametrize(
    "field", ["keff_change", "flux_change", "keff_relative_residual"]
)
def test_archive_requires_converged_final_iteration(tmp_path, field):
    """A completed criticality result must satisfy its recorded controls."""
    result = solve_keff(_keff_configuration(), FissionSourceNormalization(1.0))
    archive = tmp_path / "result.morana-result"
    inconsistent = tmp_path / "inconsistent.morana-result"
    result.save_to_disk(archive)

    def change_final_iteration(name, payload):
        if name != "manifest.json":
            return payload
        manifest = json.loads(payload)
        manifest["result"]["execution_report"]["outer_iterations"][-1][field] = 1.0
        return json.dumps(manifest).encode("utf-8")

    _rewrite_archive(archive, inconsistent, change_final_iteration)
    with pytest.raises(ValueError, match="convergence"):
        Result.load_from_disk(inconsistent)


def test_result_rejects_negative_flux():
    """Stored results contain the nonnegative flux after solver cleanup."""
    with pytest.raises(ValueError, match="non-negative"):
        _manual_fixed_source_result((np.array([[-1.0]]),))


@pytest.mark.parametrize(
    "condition",
    [
        BoundaryCondition.robin(Fraction(1, 2)),
        BoundaryCondition.partial_current_return(Fraction(1, 2)),
    ],
)
def test_archive_round_trips_real_boundary_coefficients(tmp_path, condition):
    """Accepted real boundary scalars must have portable JSON representations."""
    configuration = make_configuration()
    configuration.set_boundary(BoundaryConditionSet(condition.globally()))
    result = solve_fixed_source(configuration)
    path = tmp_path / "boundary.morana-result"
    result.save_to_disk(path)
    restored = Result.load_from_disk(path)
    np.testing.assert_allclose(restored.flux, result.flux)


def test_result_groups_must_match_retained_cross_sections():
    """A structurally valid flux cannot use a different energy-group count."""
    with pytest.raises(ValueError, match="groups must match"):
        _manual_fixed_source_result((np.ones((2, 1)),))
