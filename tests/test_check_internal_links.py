"""Tests for the generated documentation internal-link checker."""

from pathlib import Path

import pytest

from scripts.check_internal_links import check_site


def test_check_site_accepts_local_files_and_fragments(tmp_path: Path) -> None:
    """Existing relative, root-relative, and same-page targets should pass."""
    (tmp_path / "index.html").write_text(
        '<a href="guide.html#section">Guide</a>'
        '<a href="/assets/logo.svg">Logo</a>'
        '<a href="https://example.com/elsewhere">External</a>',
        encoding="utf-8",
    )
    (tmp_path / "guide.html").write_text(
        '<h1 id="section">Section</h1><a href="#section">Permalink</a>',
        encoding="utf-8",
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "logo.svg").write_text("<svg/>", encoding="utf-8")

    result = check_site(tmp_path)

    assert result.pages == 2
    assert result.links == 3
    assert not result.failures


def test_check_site_accepts_project_site_root_relative_targets(tmp_path: Path) -> None:
    """Root-relative links may include the configured hosted project path."""
    (tmp_path / "index.html").write_text(
        '<a href="/morana/guide.html#section">Guide</a>'
        '<a href="/morana/assets/logo.svg">Logo</a>',
        encoding="utf-8",
    )
    (tmp_path / "guide.html").write_text(
        '<h1 id="section">Section</h1>',
        encoding="utf-8",
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "logo.svg").write_text("<svg/>", encoding="utf-8")

    result = check_site(tmp_path, "/morana/")

    assert result.links == 2
    assert not result.failures


def test_check_site_reports_missing_targets_fragments_and_traversal(
    tmp_path: Path,
) -> None:
    """Broken files, fragments, and paths outside the site should be reported."""
    (tmp_path / "index.html").write_text(
        '<a href="missing.html">Missing</a>'
        '<a href="guide.html#missing">Bad fragment</a>'
        '<a href="../outside.html">Outside</a>',
        encoding="utf-8",
    )
    (tmp_path / "guide.html").write_text(
        '<h1 id="present">Present</h1>',
        encoding="utf-8",
    )

    result = check_site(tmp_path)

    assert tuple(failure.reason for failure in result.failures) == (
        "missing target",
        "missing fragment",
        "target escapes the site directory",
    )


def test_check_site_rejects_empty_generated_site(tmp_path: Path) -> None:
    """An empty output directory must not produce a false successful check."""
    with pytest.raises(ValueError, match="contains no HTML pages"):
        check_site(tmp_path)
