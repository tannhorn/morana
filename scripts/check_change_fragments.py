"""Validate pending release-note fragments and their allocation counter."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHANGES_DIR = REPOSITORY_ROOT / "changes"
CATEGORIES = (
    "added",
    "changed",
    "deprecated",
    "removed",
    "fixed",
    "security",
)
FILENAME_PATTERN = re.compile(
    rf"(?P<number>[0-9]{{6}})\.(?P<category>{'|'.join(CATEGORIES)})\.md"
)
REQUIRED_FIELDS = frozenset({"issues", "breaking", "upgrade", "documentation"})


@dataclass(frozen=True)
class FragmentFailure:
    """Describe one invalid fragment or allocation-counter condition."""

    path: Path
    reason: str


def _positive_integer(value: Any) -> bool:
    """Return whether a value is a positive integer but not a Boolean."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _parse_fragment(path: Path) -> tuple[dict[str, Any], str]:
    """Return one fragment's front matter and Markdown body."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError("must begin with YAML front matter delimited by ---")
    try:
        closing_index = lines[1:].index("---") + 1
    except ValueError as exc:
        raise ValueError("front matter has no closing --- delimiter") from exc
    metadata = yaml.safe_load("\n".join(lines[1:closing_index]))
    if not isinstance(metadata, dict):
        raise ValueError("front matter must be a YAML mapping")
    body = "\n".join(lines[closing_index + 1 :]).strip()
    return metadata, body


def _documentation_failures(
    path: Path,
    documentation: Any,
    repository_root: Path,
) -> list[FragmentFailure]:
    """Return failures for a fragment's documentation references."""
    if not isinstance(documentation, list) or any(
        not isinstance(item, str) or not item.strip() for item in documentation
    ):
        return [FragmentFailure(path, "documentation must be a list of paths")]

    failures = []
    for item in documentation:
        candidate = repository_root / item
        try:
            candidate.resolve().relative_to(repository_root.resolve())
        except ValueError:
            failures.append(
                FragmentFailure(path, f"documentation escapes root: {item}")
            )
            continue
        if not candidate.is_file():
            failures.append(
                FragmentFailure(path, f"documentation does not exist: {item}")
            )
    return failures


def _metadata_failures(
    path: Path,
    metadata: dict[str, Any],
    repository_root: Path,
) -> list[FragmentFailure]:
    """Return schema and referenced-path failures for one fragment."""
    failures = []
    fields = frozenset(metadata)
    if fields != REQUIRED_FIELDS:
        missing = sorted(REQUIRED_FIELDS - fields)
        unexpected = sorted(fields - REQUIRED_FIELDS)
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected fields: {', '.join(unexpected)}")
        failures.append(FragmentFailure(path, "; ".join(details)))
        return failures

    issues = metadata["issues"]
    if not isinstance(issues, list) or any(
        not _positive_integer(issue) for issue in issues
    ):
        failures.append(FragmentFailure(path, "issues must be positive integers"))
    elif len(issues) != len(set(issues)):
        failures.append(FragmentFailure(path, "issues must not contain duplicates"))

    breaking = metadata["breaking"]
    if not isinstance(breaking, bool):
        failures.append(FragmentFailure(path, "breaking must be true or false"))

    upgrade = metadata["upgrade"]
    if upgrade is not None and (not isinstance(upgrade, str) or not upgrade.strip()):
        failures.append(FragmentFailure(path, "upgrade must be nonempty or null"))
    if breaking is True and upgrade is None:
        failures.append(FragmentFailure(path, "breaking changes require upgrade text"))

    failures.extend(
        _documentation_failures(path, metadata["documentation"], repository_root)
    )
    return failures


def check_fragments(
    changes_dir: Path,
    repository_root: Path,
) -> tuple[int | None, tuple[FragmentFailure, ...]]:
    """Validate a change-fragment directory and return its next identifier."""
    failures: list[FragmentFailure] = []
    counter_path = changes_dir / "next_id.txt"
    try:
        counter_text = counter_path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, (FragmentFailure(counter_path, str(exc)),)
    if not re.fullmatch(r"[0-9]{6}\n", counter_text):
        failures.append(
            FragmentFailure(counter_path, "must contain six digits and one newline")
        )
        next_id = None
    else:
        next_id = int(counter_text)
        if next_id < 1:
            failures.append(FragmentFailure(counter_path, "must be greater than zero"))

    numbers: dict[int, Path] = {}
    for path in sorted(changes_dir.iterdir()):
        if path.name == "next_id.txt":
            continue
        match = FILENAME_PATTERN.fullmatch(path.name)
        if not path.is_file() or match is None:
            failures.append(
                FragmentFailure(path, "unexpected change-fragment filename")
            )
            continue
        number = int(match.group("number"))
        if number < 1:
            failures.append(
                FragmentFailure(path, "fragment number must be greater than zero")
            )
        previous = numbers.get(number)
        if previous is not None:
            failures.append(
                FragmentFailure(path, f"fragment number duplicates {previous.name}")
            )
        numbers[number] = path
        if next_id is not None and number >= next_id:
            failures.append(
                FragmentFailure(path, "fragment number must be below next_id.txt")
            )
        try:
            metadata, body = _parse_fragment(path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            failures.append(FragmentFailure(path, str(exc)))
            continue
        failures.extend(_metadata_failures(path, metadata, repository_root))
        if not body:
            failures.append(FragmentFailure(path, "Markdown body must not be empty"))
    return next_id, tuple(failures)


def main() -> int:
    """Validate the repository's pending release-note fragments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changes-dir", type=Path, default=DEFAULT_CHANGES_DIR)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    args = parser.parse_args()
    next_id, failures = check_fragments(args.changes_dir, args.repository_root)
    for failure in failures:
        try:
            label = failure.path.resolve().relative_to(args.repository_root.resolve())
        except ValueError:
            label = failure.path
        print(f"{label}: {failure.reason}")
    fragment_count = sum(1 for path in args.changes_dir.glob("*.md") if path.is_file())
    if next_id is None:
        print(f"Checked {fragment_count} change fragment(s); counter is invalid.")
    else:
        print(
            f"Checked {fragment_count} change fragment(s); "
            f"next identifier is {next_id:06d}."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
