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


def test_check_site_accepts_known_documentation_and_repository_links(
    tmp_path: Path,
) -> None:
    """Known absolute project links should resolve against local artifacts."""
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    examples = repository_root / "examples"
    examples.mkdir()
    (examples / "quickstart.py").write_text("", encoding="utf-8")

    site_dir = tmp_path / "site"
    site_dir.mkdir()
    (site_dir / "index.html").write_text(
        '<a href="https://docs.example/morana/guide.html#section">Guide</a>'
        '<a href="https://github.example/owner/project/blob/main/'
        'examples/quickstart.py">Source</a>'
        '<a href="https://github.example/owner/project/tree/main/examples">'
        "Examples</a>",
        encoding="utf-8",
    )
    (site_dir / "guide.html").write_text(
        '<h1 id="section">Section</h1>', encoding="utf-8"
    )

    result = check_site(
        site_dir,
        site_url="https://docs.example/morana/",
        repository_root=repository_root,
        repository_url="https://github.example/owner/project",
    )

    assert result.links == 3
    assert not result.failures


def test_check_site_checks_known_links_in_repository_markdown(
    tmp_path: Path,
) -> None:
    """Project links outside the generated documentation should also be checked."""
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    readme = repository_root / "README.md"
    readme.write_text(
        "[Guide](https://docs.example/morana/guide.html#section)\n\n"
        "[Source](https://github.example/owner/project/blob/main/example.py)\n\n"
        "[External](https://example.com/ignored)\n",
        encoding="utf-8",
    )
    (repository_root / "example.py").write_text("", encoding="utf-8")

    site_dir = tmp_path / "site"
    site_dir.mkdir()
    (site_dir / "index.html").write_text("", encoding="utf-8")
    (site_dir / "guide.html").write_text(
        '<h1 id="section">Section</h1>', encoding="utf-8"
    )

    result = check_site(
        site_dir,
        site_url="https://docs.example/morana/",
        repository_root=repository_root,
        repository_url="https://github.example/owner/project",
        markdown_files=(readme,),
    )

    assert result.links == 2
    assert not result.failures


def test_check_site_reports_missing_known_project_targets(tmp_path: Path) -> None:
    """Missing published pages, fragments, files, and directories should fail."""
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    (site_dir / "index.html").write_text(
        '<a href="https://docs.example/morana/missing.html">Missing page</a>'
        '<a href="https://docs.example/morana/index.html#missing">'
        "Missing fragment</a>"
        '<a href="https://github.example/owner/project/blob/main/missing.py">'
        "Missing file</a>"
        '<a href="https://github.example/owner/project/tree/main/missing">'
        "Missing directory</a>",
        encoding="utf-8",
    )

    result = check_site(
        site_dir,
        site_url="https://docs.example/morana/",
        repository_root=repository_root,
        repository_url="https://github.example/owner/project",
    )

    assert tuple(failure.reason for failure in result.failures) == (
        "missing target",
        "missing fragment",
        "missing repository file",
        "missing repository directory",
    )


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
