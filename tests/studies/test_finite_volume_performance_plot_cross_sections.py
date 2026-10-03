"""Tests for the maintained synthetic cross-section plotter."""

from __future__ import annotations

from studies.finite_volume_performance import plot_cross_sections


def test_generate_plots_writes_every_comparison(tmp_path) -> None:
    """One selected group count produces every expected nonempty PNG."""
    paths = plot_cross_sections.generate_plots((6,), tmp_path)

    assert {path.name for path in paths} == {
        "g6_absorption.png",
        "g6_diffusion.png",
        "g6_fission_production.png",
        "g6_fission_spectrum.png",
        "g6_fission_transfer.png",
        "g6_scattering.png",
    }
    for path in paths:
        assert path.parent == tmp_path
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
