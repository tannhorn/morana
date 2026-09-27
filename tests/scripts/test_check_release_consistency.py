"""Tests for the release-consistency quality gate."""

from pathlib import Path

from scripts.check_release_consistency import check_release_consistency


VERSION = "1.2.3"
DOI = "10.5281/zenodo.1234567"
DATE = "2026-09-27"
RELEASE_URL = f"https://github.com/tannhorn/morana/releases/tag/v{VERSION}"
PYPI_URL = f"https://pypi.org/project/morana/{VERSION}/"
DOI_URL = f"https://doi.org/{DOI}"


def _repository(tmp_path: Path) -> Path:
    """Create a minimal repository with internally consistent release data."""
    (tmp_path / "pyproject.toml").write_text(
        f'''[project]
version = "{VERSION}"

[project.urls]
Release = "{RELEASE_URL}"
DOI = "{DOI_URL}"
''',
        encoding="utf-8",
    )
    (tmp_path / "CITATION.cff").write_text(
        f"version: {VERSION}\ndate-released: {DATE}\ndoi: {DOI}\n",
        encoding="utf-8",
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (tmp_path / "README.md").write_text(
        f"{RELEASE_URL} {PYPI_URL} {DOI_URL} morana=={VERSION}\n",
        encoding="utf-8",
    )
    (docs / "getting_started.md").write_text(
        f"{RELEASE_URL} {PYPI_URL} {DOI_URL} morana=={VERSION} "
        f"morana-{VERSION}.tar.gz.sha256\n",
        encoding="utf-8",
    )
    (docs / "index.md").write_text(
        f"{RELEASE_URL} {PYPI_URL} version `{VERSION}`\n",
        encoding="utf-8",
    )
    (docs / "citation.md").write_text(
        f"Morana {VERSION} {DOI_URL}\n",
        encoding="utf-8",
    )
    (docs / "changelog.md").write_text(
        f"# Changelog\n\n## {VERSION} - {DATE}\n\n"
        f"{RELEASE_URL} {PYPI_URL} {DOI_URL}\n",
        encoding="utf-8",
    )
    return tmp_path


def test_release_consistency_accepts_matching_metadata(tmp_path: Path) -> None:
    """All synchronized current-release fields may agree."""
    root = _repository(tmp_path)

    version, failures = check_release_consistency(root, VERSION)

    assert version == VERSION
    assert not failures


def test_release_consistency_reports_metadata_disagreement(tmp_path: Path) -> None:
    """Version, DOI, date, tag, and requested-version mismatches should fail."""
    root = _repository(tmp_path)
    pyproject = root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            RELEASE_URL,
            "https://github.com/tannhorn/morana/releases/tag/v1.2.2",
        ),
        encoding="utf-8",
    )
    (root / "CITATION.cff").write_text(
        "version: 1.2.2\ndate-released: 2026-09-26\n"
        "doi: 10.5281/zenodo.7654321\n",
        encoding="utf-8",
    )
    changelog = root / "docs/changelog.md"
    changelog.write_text(
        changelog.read_text(encoding="utf-8").replace(
            f"## {VERSION} - {DATE}",
            f"## 1.2.2 - {DATE}",
        ),
        encoding="utf-8",
    )

    _, failures = check_release_consistency(root, "1.2.4")

    reasons = {failure.reason for failure in failures}
    assert any("does not match expected version" in reason for reason in reasons)
    assert any("does not match project.version" in reason for reason in reasons)
    assert any("project.urls.Release must be" in reason for reason in reasons)
    assert any("project.urls.DOI must be" in reason for reason in reasons)
    assert any("latest release" in reason for reason in reasons)
    assert any("does not match latest changelog date" in reason for reason in reasons)


def test_release_consistency_reports_stale_current_documentation(
    tmp_path: Path,
) -> None:
    """A missing current-release reference in user documentation should fail."""
    root = _repository(tmp_path)
    (root / "docs/index.md").write_text("Version 1.2.2.\n", encoding="utf-8")

    _, failures = check_release_consistency(root)

    index_reasons = [
        failure.reason
        for failure in failures
        if failure.path == root / "docs/index.md"
    ]
    assert len(index_reasons) == 3
    assert all("missing current-release text" in reason for reason in index_reasons)
