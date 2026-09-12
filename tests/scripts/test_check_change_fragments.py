"""Tests for the release-note fragment quality gate."""

from pathlib import Path

from scripts.check_change_fragments import check_fragments

VALID_FRAGMENT = """---
issues: [42]
breaking: false
upgrade: null
documentation:
  - docs/guide.md
---
Added a user-visible capability.
"""


def _repository(tmp_path: Path, counter: str = "000002\n") -> tuple[Path, Path]:
    """Create a minimal repository and return its root and changes directory."""
    changes = tmp_path / "changes"
    changes.mkdir()
    (changes / "next_id.txt").write_text(counter, encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\n", encoding="utf-8")
    return tmp_path, changes


def test_check_fragments_accepts_valid_pending_fragment(tmp_path: Path) -> None:
    """A numbered fragment with complete metadata and prose should pass."""
    root, changes = _repository(tmp_path)
    (changes / "000001.added.md").write_text(VALID_FRAGMENT, encoding="utf-8")

    next_id, failures = check_fragments(changes, root)

    assert next_id == 2
    assert not failures


def test_check_fragments_accepts_empty_pending_collection(tmp_path: Path) -> None:
    """A release may consume every fragment without resetting its counter."""
    root, changes = _repository(tmp_path)

    next_id, failures = check_fragments(changes, root)

    assert next_id == 2
    assert not failures


def test_check_fragments_reports_filename_counter_and_schema_failures(
    tmp_path: Path,
) -> None:
    """Malformed names, reused allocation state, and metadata should fail."""
    root, changes = _repository(tmp_path, counter="000001\n")
    (changes / "note.md").write_text(VALID_FRAGMENT, encoding="utf-8")
    (changes / "000001.fixed.md").write_text(
        VALID_FRAGMENT.replace("breaking: false", "breaking: true").replace(
            "  - docs/guide.md", "  - docs/missing.md"
        ),
        encoding="utf-8",
    )

    _, failures = check_fragments(changes, root)

    reasons = {failure.reason for failure in failures}
    assert "unexpected change-fragment filename" in reasons
    assert "fragment number must be below next_id.txt" in reasons
    assert "breaking changes require upgrade text" in reasons
    assert "documentation does not exist: docs/missing.md" in reasons
