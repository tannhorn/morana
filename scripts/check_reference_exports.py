"""Check that runtime public exports appear in generated API references."""

# Documentation checkers intentionally share a small command-line shape.
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
import importlib
from pathlib import Path
from types import ModuleType
from typing import Iterable


@dataclass(frozen=True)
class ReferenceSpec:
    """Map one public module to its generated reference page."""

    module: str
    page: Path


@dataclass(frozen=True)
class ReferenceFailure:
    """Describe one runtime-export or generated-anchor failure."""

    module: str
    export: str | None
    reason: str
    expected_anchor: str | None = None


@dataclass(frozen=True)
class ReferenceCheckResult:
    """Store generated-reference check counts and failures."""

    modules: int
    exports: int
    failures: tuple[ReferenceFailure, ...]


REFERENCE_SPECS = (
    ReferenceSpec("morana", Path("reference/core.html")),
    ReferenceSpec("morana.solvers.finite_volume", Path("reference/finite_volume.html")),
    ReferenceSpec("morana.operators", Path("reference/operators.html")),
)


class _IdParser(HTMLParser):
    """Count exact ID attributes in one generated HTML page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: Counter[str] = Counter()

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Count an element ID when present."""
        del tag
        target_id = dict(attrs).get("id")
        if target_id is not None:
            self.ids[target_id] += 1


def module_exports(module: ModuleType) -> tuple[str, ...]:
    """Return a checked runtime ``__all__`` declaration."""
    exports = getattr(module, "__all__", None)
    if not isinstance(exports, (list, tuple)) or not exports:
        raise ValueError(f"{module.__name__}.__all__ must be a nonempty list or tuple")
    if any(not isinstance(name, str) or not name for name in exports):
        raise ValueError(f"{module.__name__}.__all__ must contain nonempty strings")
    if len(set(exports)) != len(exports):
        raise ValueError(f"{module.__name__}.__all__ must not contain duplicates")
    return tuple(exports)


def check_reference_page(
    module: ModuleType,
    exports: Iterable[str],
    page: Path,
) -> tuple[ReferenceFailure, ...]:
    """Check runtime attributes and exact generated anchors for one module."""
    parser = _parse_page(page)
    failures = []
    for export in exports:
        anchor = f"{module.__name__}.{export}"
        if not hasattr(module, export):
            failures.append(
                ReferenceFailure(
                    module=module.__name__,
                    export=export,
                    reason="export is missing from the runtime module",
                    expected_anchor=anchor,
                )
            )
        anchor_count = parser.ids[anchor]
        if anchor_count == 0:
            failures.append(
                ReferenceFailure(
                    module=module.__name__,
                    export=export,
                    reason="generated anchor is missing",
                    expected_anchor=anchor,
                )
            )
        elif anchor_count > 1:
            failures.append(
                ReferenceFailure(
                    module=module.__name__,
                    export=export,
                    reason=f"generated anchor occurs {anchor_count} times",
                    expected_anchor=anchor,
                )
            )
    return tuple(failures)


def check_references(
    site_dir: Path,
    specs: Iterable[ReferenceSpec] = REFERENCE_SPECS,
) -> ReferenceCheckResult:
    """Check every configured runtime module and generated reference page."""
    site_dir = site_dir.resolve()
    if not site_dir.is_dir():
        raise ValueError(f"generated site directory does not exist: {site_dir}")
    spec_list = tuple(specs)
    failures = []
    export_count = 0
    for spec in spec_list:
        module = importlib.import_module(spec.module)
        exports = module_exports(module)
        export_count += len(exports)
        page = site_dir / spec.page
        if not page.is_file():
            failures.append(
                ReferenceFailure(
                    module=spec.module,
                    export=None,
                    reason=f"generated reference page is missing: {spec.page}",
                )
            )
            continue
        failures.extend(check_reference_page(module, exports, page))
    return ReferenceCheckResult(len(spec_list), export_count, tuple(failures))


def _parse_page(path: Path) -> _IdParser:
    """Parse IDs from one generated reference page."""
    parser = _IdParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return parser


def main() -> int:
    """Run the generated-reference export check from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "site_dir",
        nargs="?",
        type=Path,
        default=Path("site"),
        help="generated site directory (default: site)",
    )
    args = parser.parse_args()
    try:
        result = check_references(args.site_dir)
    except (ImportError, ValueError) as exc:
        parser.error(str(exc))

    for failure in result.failures:
        export = f".{failure.export}" if failure.export is not None else ""
        anchor = (
            f'; expected id="{failure.expected_anchor}"'
            if failure.expected_anchor is not None
            else ""
        )
        print(f"{failure.module}{export}: {failure.reason}{anchor}")
    print(
        f"Checked {result.exports} exports from {result.modules} modules; "
        f"found {len(result.failures)} reference failure(s)."
    )
    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
