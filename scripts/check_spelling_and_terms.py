"""Check authored text for spelling mistakes and Morana terminology."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import tomllib
from typing import Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VOCABULARY = REPOSITORY_ROOT / "scripts" / "spelling_vocabulary.txt"
DEFAULT_RULES = REPOSITORY_ROOT / "scripts" / "terminology_rules.toml"
AUTHORED_FILES = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "CONTRIBUTING.md",
)
AUTHORED_TREES = {
    REPOSITORY_ROOT / "docs": frozenset({".md"}),
    REPOSITORY_ROOT / "src": frozenset({".py"}),
    REPOSITORY_ROOT / "examples": frozenset({".py", ".sh"}),
    REPOSITORY_ROOT / "scripts": frozenset({".py", ".sh"}),
    REPOSITORY_ROOT / "changes": frozenset({".md"}),
}
TERMINOLOGY_TREES = frozenset({"docs", "src", "examples", "changes"})


@dataclass(frozen=True)
class TerminologyRule:
    """Define one discouraged expression and its preferred replacement."""

    pattern: re.Pattern[str]
    preferred: str


@dataclass(frozen=True)
class TerminologyFailure:
    """Describe one terminology-rule match."""

    path: Path
    line: int
    matched: str
    preferred: str


def authored_files() -> tuple[Path, ...]:
    """Return the deterministic default authored-text scope."""
    files = [path for path in AUTHORED_FILES if path.is_file()]
    for root, suffixes in AUTHORED_TREES.items():
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix in suffixes
        )
    return tuple(sorted(files))


def terminology_files(files: Iterable[Path]) -> tuple[Path, ...]:
    """Return public prose and source files from the authored-text scope."""
    selected = []
    for path in files:
        relative = path.resolve().relative_to(REPOSITORY_ROOT)
        if len(relative.parts) == 1 or relative.parts[0] in TERMINOLOGY_TREES:
            selected.append(path)
    return tuple(selected)


def load_rules(path: Path) -> tuple[TerminologyRule, ...]:
    """Load and compile terminology rules from TOML."""
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    raw_rules = document.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError(f"terminology rules must define a nonempty rules list: {path}")
    rules = []
    for index, raw_rule in enumerate(raw_rules):
        if not isinstance(raw_rule, dict):
            raise ValueError(f"terminology rule {index} must be a table")
        pattern = raw_rule.get("pattern")
        preferred = raw_rule.get("preferred")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(f"terminology rule {index} requires a pattern")
        if not isinstance(preferred, str) or not preferred:
            raise ValueError(f"terminology rule {index} requires a preferred term")
        rules.append(TerminologyRule(re.compile(pattern), preferred))
    return tuple(rules)


def check_terminology(
    files: Iterable[Path],
    rules: Iterable[TerminologyRule],
) -> tuple[TerminologyFailure, ...]:
    """Return every terminology-rule match in the selected files."""
    failures = []
    for path in files:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            for rule in rules:
                failures.extend(
                    TerminologyFailure(
                        path=path,
                        line=line_number,
                        matched=match.group(0),
                        preferred=rule.preferred,
                    )
                    for match in rule.pattern.finditer(line)
                )
    return tuple(failures)


def run_codespell(files: Iterable[Path], vocabulary: Path) -> int:
    """Run codespell over the selected authored files."""
    executable = shutil.which("codespell")
    if executable is None:
        raise RuntimeError(
            "codespell is required; update the morana-dev environment first"
        )
    command = [
        executable,
        "--quiet-level=2",
        f"--ignore-words={vocabulary}",
        *(str(path) for path in files),
    ]
    return subprocess.run(command, check=False).returncode


def _display_path(path: Path) -> Path:
    """Prefer a repository-relative path in diagnostics."""
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT)
    except ValueError:
        return path


def main() -> int:
    """Run spelling and terminology checks from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="files to check")
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=DEFAULT_VOCABULARY,
        help="codespell ignore-word file",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=DEFAULT_RULES,
        help="Morana terminology rules TOML file",
    )
    args = parser.parse_args()
    files = tuple(path.resolve() for path in args.paths) or authored_files()
    missing = tuple(path for path in files if not path.is_file())
    if missing:
        parser.error(f"selected path is not a file: {missing[0]}")

    try:
        rules = load_rules(args.rules)
    except (OSError, ValueError, re.error, tomllib.TOMLDecodeError) as exc:
        parser.error(str(exc))

    try:
        spelling_status = run_codespell(files, args.vocabulary)
    except RuntimeError as exc:
        parser.error(str(exc))
    selected_terminology_files = files if args.paths else terminology_files(files)
    terminology_failures = check_terminology(selected_terminology_files, rules)
    for failure in terminology_failures:
        print(
            f'{_display_path(failure.path)}:{failure.line}: "{failure.matched}" '
            f'is discouraged; use "{failure.preferred}"'
        )
    print(
        f"Checked spelling and terminology in {len(files)} authored files; "
        f"found {len(terminology_failures)} terminology failure(s)."
    )
    return 1 if spelling_status or terminology_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
