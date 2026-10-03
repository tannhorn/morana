"""Tests for the generated-reference public-export checker."""

# The documentation checkers intentionally share a small command-line shape.
# pylint: disable=duplicate-code

from pathlib import Path
from types import ModuleType

import pytest

from scripts.check_reference_exports import (
    check_reference_page,
    module_exports,
)


def _module(*exports: str) -> ModuleType:
    """Return a synthetic module with matching runtime exports."""
    module = ModuleType("sample")
    module.__all__ = list(exports)
    for export in exports:
        setattr(module, export, object())
    return module


def test_check_reference_page_accepts_exact_export_anchors(tmp_path: Path) -> None:
    """Every runtime export with one exact generated anchor should pass."""
    module = _module("Thing", "VALUE")
    page = tmp_path / "reference.html"
    page.write_text(
        '<h2 id="sample.Thing">Thing</h2><h2 id="sample.VALUE">Value</h2>',
        encoding="utf-8",
    )

    failures = check_reference_page(module, module_exports(module), page)

    assert not failures


def test_check_reference_page_rejects_missing_and_similar_anchors(
    tmp_path: Path,
) -> None:
    """A longer similarly named anchor must not satisfy one missing export."""
    module = _module("Thing")
    page = tmp_path / "reference.html"
    page.write_text('<h2 id="sample.ThingExtra">Other</h2>', encoding="utf-8")

    failures = check_reference_page(module, module_exports(module), page)

    assert len(failures) == 1
    assert failures[0].reason == "generated anchor is missing"
    assert failures[0].expected_anchor == "sample.Thing"


def test_check_reference_page_reports_duplicate_anchor_and_missing_attribute(
    tmp_path: Path,
) -> None:
    """Duplicate IDs and stale runtime export declarations should both fail."""
    module = _module("Thing", "STALE")
    delattr(module, "STALE")
    page = tmp_path / "reference.html"
    page.write_text(
        '<h2 id="sample.Thing">Thing</h2><a id="sample.Thing"></a>'
        '<h2 id="sample.STALE">Stale</h2>',
        encoding="utf-8",
    )

    failures = check_reference_page(module, module_exports(module), page)

    assert tuple(failure.reason for failure in failures) == (
        "generated anchor occurs 2 times",
        "export is missing from the runtime module",
    )


@pytest.mark.parametrize("exports", [None, [], [""], ["Thing", "Thing"]])
def test_module_exports_rejects_invalid_declarations(exports: object) -> None:
    """The checker should reject malformed or duplicate ``__all__`` values."""
    module = ModuleType("sample")
    module.__all__ = exports  # type: ignore[assignment]

    with pytest.raises(ValueError, match="sample.__all__"):
        module_exports(module)
