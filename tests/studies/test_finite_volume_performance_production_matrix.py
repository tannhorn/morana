"""Tests for the complete production-path assessment launcher."""

from pathlib import Path

import pytest

from studies.finite_volume_performance import production_matrix
from studies.finite_volume_performance.cases import (
    BICGSTAB_JACOBI_CASE_ID,
    DIRECT_CASE_ID,
    GMRES_JACOBI_CASE_ID,
    POWER_ITERATION_ID,
    WIELANDT_ITERATION_ID,
)
from studies.finite_volume_performance.protocol import DEFAULT_STUDY_PROTOCOL


def test_matrix_covers_every_resolved_axis_with_frozen_controls() -> None:
    """The campaign is the exact 3-by-2-by-3-by-2 resolved matrix."""
    entries = production_matrix.matrix_runs()
    assert len(entries) == 36
    assert len({entry.output_name for entry in entries}) == len(entries)
    assert {entry.solver for entry in entries} == {
        DIRECT_CASE_ID,
        GMRES_JACOBI_CASE_ID,
        BICGSTAB_JACOBI_CASE_ID,
    }
    assert {entry.placement for entry in entries} == {"structured", "permuted"}
    assert {entry.iteration for entry in entries} == {
        POWER_ITERATION_ID,
        WIELANDT_ITERATION_ID,
    }
    assert {
        (entry.workload.groups, entry.workload.axial_layers) for entry in entries
    } == {(36, 6), (18, 12), (72, 24)}
    assert all(
        entry.repetitions == (1 if entry.solver == DIRECT_CASE_ID else 3)
        for entry in entries
    )
    assert all(
        (entry.shift_inverse_keff is None) == (entry.iteration == POWER_ITERATION_ID)
        for entry in entries
    )
    assert DEFAULT_STUDY_PROTOCOL.timeout_seconds == 300.0
    assert DEFAULT_STUDY_PROTOCOL.address_space_limit_bytes == 16 * 1024**3
    assert DEFAULT_STUDY_PROTOCOL.requested_threads == 1
    assert DEFAULT_STUDY_PROTOCOL.warmup == (6, 2)
    assert DEFAULT_STUDY_PROTOCOL.profile is True


def test_matrix_freezes_all_six_checked_wielandt_shifts() -> None:
    """Each workload and placement resolves to its calibrated fixed shift."""
    shifts = {
        (
            entry.workload.groups,
            entry.workload.axial_layers,
            entry.placement,
        ): entry.shift_inverse_keff
        for entry in production_matrix.matrix_runs()
        if entry.iteration == WIELANDT_ITERATION_ID
    }
    assert shifts == {
        (36, 6, "structured"): 1.021339391553522,
        (18, 12, "structured"): 1.0407799590610955,
        (72, 24, "structured"): 1.0207685354094909,
        (36, 6, "permuted"): 0.9900506307635286,
        (18, 12, "permuted"): 1.0045898337826535,
        (72, 24, "permuted"): 0.9907602987675709,
    }


def test_fresh_matrix_refuses_to_overwrite_any_existing_entry(
    tmp_path: Path, monkeypatch
) -> None:
    """A fresh campaign fails before dispatch rather than losing evidence."""
    first = production_matrix.matrix_runs()[0]
    (tmp_path / first.output_name).write_text("existing", encoding="utf-8")
    dispatched = []
    monkeypatch.setattr(
        production_matrix, "run", lambda *args, **kwargs: dispatched.append(args)
    )

    with pytest.raises(FileExistsError, match="--resume"):
        production_matrix.run_matrix(tmp_path)
    assert not dispatched


def test_resume_dispatches_existing_and_new_entries_correctly(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Campaign resume is per output document and supports partial creation."""
    first = production_matrix.matrix_runs()[0]
    (tmp_path / first.output_name).write_text("existing", encoding="utf-8")
    calls = []

    def fake_run(mode, output_dir, **options):
        calls.append((mode, output_dir, options))
        return output_dir / options["output_name"]

    monkeypatch.setattr(production_matrix, "run", fake_run)
    paths = production_matrix.run_matrix(tmp_path, resume=True)

    assert len(paths) == len(calls) == 36
    assert calls[0][2]["resume"] is True
    assert all(call[2]["resume"] is False for call in calls[1:])
    assert all(call[0] == "selected" and call[1] == tmp_path for call in calls)
    assert all(call[2]["warmup"] == (6, 2) for call in calls)
    assert all(call[2]["profile"] is True for call in calls)
    assert all(
        call[2]["timeout_seconds"] == DEFAULT_STUDY_PROTOCOL.timeout_seconds
        for call in calls
    )
    assert all(
        call[2]["address_space_limit_bytes"]
        == DEFAULT_STUDY_PROTOCOL.address_space_limit_bytes
        for call in calls
    )
    output = capsys.readouterr().out
    assert "Production matrix entry [1/36]" in output
    assert "Production matrix entry [36/36]" in output


def test_dry_run_has_no_filesystem_or_worker_side_effects(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Dry-run prints the complete worker count without dispatching work."""
    monkeypatch.setattr(
        production_matrix,
        "run",
        lambda *args, **kwargs: pytest.fail("dry-run dispatched a worker"),
    )
    paths = production_matrix.run_matrix(tmp_path, dry_run=True)
    output = capsys.readouterr().out

    assert len(paths) == 36
    assert "36 entries, 120 recorded workers, 36 warm-ups" in output
    assert not tuple(tmp_path.iterdir())
