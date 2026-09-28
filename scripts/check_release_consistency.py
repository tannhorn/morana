"""Check that current-release metadata and documentation agree."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import tomllib

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
CHANGELOG_HEADING = re.compile(
    r"^## (?P<version>\S+) - (?P<date>\d{4}-\d{2}-\d{2})$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class ReleaseConsistencyFailure:
    """Describe one inconsistent release field or reference."""

    path: Path
    reason: str


def _citation_scalar(text: str, field: str) -> str:
    """Return one plain or quoted top-level scalar from CFF text."""
    matches = re.findall(rf"^{re.escape(field)}:\s*(.+?)\s*$", text, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"{field} must occur exactly once as a top-level scalar")
    value = matches[0]
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    if not value:
        raise ValueError(f"{field} must be a nonempty scalar")
    return value


def _required_text_failures(
    path: Path,
    required: tuple[str, ...],
) -> list[ReleaseConsistencyFailure]:
    """Return failures for expected current-release text missing from a file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [ReleaseConsistencyFailure(path, str(exc))]
    return [
        ReleaseConsistencyFailure(path, f"missing current-release text: {item}")
        for item in required
        if item not in text
    ]


def check_release_consistency(
    repository_root: Path,
    expected_version: str | None = None,
) -> tuple[str | None, tuple[ReleaseConsistencyFailure, ...]]:
    """Validate synchronized release metadata and return the package version."""
    failures: list[ReleaseConsistencyFailure] = []
    pyproject_path = repository_root / "pyproject.toml"
    citation_path = repository_root / "CITATION.cff"
    changelog_path = repository_root / "docs/changelog.md"

    try:
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        project = pyproject["project"]
        if not isinstance(project, dict):
            raise ValueError("[project] must be a table")
        version = project["version"]
        urls = project["urls"]
        if not isinstance(version, str) or not version:
            raise ValueError("project.version must be a nonempty string")
        if not isinstance(urls, dict):
            raise ValueError("project.urls must be a table")
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        return None, (ReleaseConsistencyFailure(pyproject_path, str(exc)),)

    if expected_version is not None and version != expected_version:
        failures.append(
            ReleaseConsistencyFailure(
                pyproject_path,
                f"project.version {version!r} does not match expected version "
                f"{expected_version!r}",
            )
        )

    release_url = f"https://github.com/tannhorn/morana/releases/tag/v{version}"
    if urls.get("Release") != release_url:
        failures.append(
            ReleaseConsistencyFailure(
                pyproject_path,
                f"project.urls.Release must be {release_url!r}",
            )
        )

    try:
        citation_text = citation_path.read_text(encoding="utf-8")
        citation_version = _citation_scalar(citation_text, "version")
        citation_doi = _citation_scalar(citation_text, "doi")
        citation_date = _citation_scalar(citation_text, "date-released")
    except (OSError, ValueError) as exc:
        failures.append(ReleaseConsistencyFailure(citation_path, str(exc)))
        return version, tuple(failures)

    if citation_version != version:
        failures.append(
            ReleaseConsistencyFailure(
                citation_path,
                f"version {citation_version!r} does not match project.version "
                f"{version!r}",
            )
        )
    doi_url = f"https://doi.org/{citation_doi}"
    if urls.get("DOI") != doi_url:
        failures.append(
            ReleaseConsistencyFailure(
                pyproject_path,
                f"project.urls.DOI must be {doi_url!r}",
            )
        )

    try:
        changelog_text = changelog_path.read_text(encoding="utf-8")
        heading = CHANGELOG_HEADING.search(changelog_text)
        if heading is None:
            raise ValueError("no dated release heading found")
    except (OSError, ValueError) as exc:
        failures.append(ReleaseConsistencyFailure(changelog_path, str(exc)))
        return version, tuple(failures)

    changelog_version = heading.group("version")
    changelog_date = heading.group("date")
    if changelog_version != version:
        failures.append(
            ReleaseConsistencyFailure(
                changelog_path,
                f"latest release {changelog_version!r} does not match "
                f"project.version {version!r}",
            )
        )
    if citation_date != changelog_date:
        failures.append(
            ReleaseConsistencyFailure(
                citation_path,
                f"date-released {citation_date!r} does not match latest "
                f"changelog date {changelog_date!r}",
            )
        )

    pypi_url = f"https://pypi.org/project/morana/{version}/"
    current_release_text = {
        repository_root
        / "README.md": (
            release_url,
            pypi_url,
            doi_url,
            f"morana=={version}",
        ),
        repository_root
        / "docs/getting_started.md": (
            release_url,
            pypi_url,
            doi_url,
            f"morana=={version}",
            f"morana-{version}.tar.gz.sha256",
        ),
        repository_root
        / "docs/index.md": (
            release_url,
            pypi_url,
            f"version `{version}`",
        ),
        repository_root
        / "docs/citation.md": (
            f"Morana {version}",
            doi_url,
        ),
        changelog_path: (release_url, pypi_url, doi_url),
    }
    for path, required in current_release_text.items():
        failures.extend(_required_text_failures(path, required))

    return version, tuple(failures)


def main() -> int:
    """Check release consistency for the repository."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--expected-version")
    args = parser.parse_args()
    version, failures = check_release_consistency(
        args.repository_root,
        args.expected_version,
    )
    for failure in failures:
        try:
            label = failure.path.resolve().relative_to(args.repository_root.resolve())
        except ValueError:
            label = failure.path
        print(f"{label}: {failure.reason}")
    if failures:
        print("Release metadata is inconsistent.")
    else:
        print(f"Release metadata is consistent for Morana {version}.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
