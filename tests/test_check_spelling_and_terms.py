"""Tests for the spelling and terminology quality gate."""

from pathlib import Path
import re

from scripts.check_spelling_and_terms import (
    TerminologyRule,
    check_terminology,
    load_rules,
    run_codespell,
)


def test_check_terminology_reports_preferred_terms(tmp_path: Path) -> None:
    """Rule matches should retain file, line, text, and preferred wording."""
    source = tmp_path / "guide.md"
    source.write_text("A multi-group model.\nAn axial layer.\n", encoding="utf-8")
    rules = (
        TerminologyRule(
            pattern=re.compile("multi-group"),
            preferred="multigroup",
        ),
    )

    failures = check_terminology((source,), rules)

    assert len(failures) == 1
    assert failures[0].path == source
    assert failures[0].line == 1
    assert failures[0].matched == "multi-group"
    assert failures[0].preferred == "multigroup"


def test_load_rules_reads_project_terminology_toml(tmp_path: Path) -> None:
    """Terminology rules should be data owned by a checked TOML file."""
    path = tmp_path / "rules.toml"
    path.write_text(
        "[[rules]]\npattern = '(?i)\\bhex\\s+z\\b'\npreferred = \"hex-z\"\n",
        encoding="utf-8",
    )

    rules = load_rules(path)

    assert len(rules) == 1
    assert rules[0].preferred == "hex-z"
    assert rules[0].pattern.search("hex z")


def test_codespell_uses_reviewed_vocabulary(tmp_path: Path) -> None:
    """Codespell should reject ordinary typos while allowing reviewed words."""
    vocabulary = tmp_path / "vocabulary.txt"
    vocabulary.write_text("coo\n", encoding="utf-8")
    clean = tmp_path / "clean.md"
    clean.write_text("Use a COO sparse matrix.\n", encoding="utf-8")
    misspelled = tmp_path / "misspelled.md"
    misspelled.write_text("Teh calculation converged.\n", encoding="utf-8")

    assert run_codespell((clean,), vocabulary) == 0
    assert run_codespell((misspelled,), vocabulary) != 0
