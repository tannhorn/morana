"""Check local links and fragments in a generated HTML documentation site."""

# Documentation checkers intentionally share a small command-line shape.
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


@dataclass(frozen=True)
class LinkFailure:
    """Describe one invalid generated-site link."""

    source: Path
    href: str
    reason: str


@dataclass(frozen=True)
class LinkCheckResult:
    """Store generated-site link-check counts and failures."""

    pages: int
    links: int
    failures: tuple[LinkFailure, ...]


class _PageParser(HTMLParser):
    """Collect links and named targets from one HTML page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []
        self.targets: set[str] = set()

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Collect href, id, and legacy anchor-name attributes."""
        attributes = dict(attrs)
        href = attributes.get("href")
        if href is not None:
            self.hrefs.append(href)
        target_id = attributes.get("id")
        if target_id is not None:
            self.targets.add(target_id)
        if tag == "a":
            target_name = attributes.get("name")
            if target_name is not None:
                self.targets.add(target_name)


def check_site(site_dir: Path, site_path: str = "/") -> LinkCheckResult:
    """Check local links and fragments below one generated site directory.

    Parameters
    ----------
    site_dir
        Directory containing generated HTML documentation.
    site_path
        URL path where the generated site is hosted. For a GitHub Pages project
        site, this is normally ``/OWNER-REPOSITORY/`` or ``/REPOSITORY/``.
    """
    site_dir = site_dir.resolve()
    if not site_dir.is_dir():
        raise ValueError(f"generated site directory does not exist: {site_dir}")
    normalized_site_path = _normalize_site_path(site_path)

    pages = tuple(sorted(site_dir.rglob("*.html")))
    if not pages:
        raise ValueError(f"generated site contains no HTML pages: {site_dir}")
    parsed_pages: dict[Path, _PageParser] = {}
    failures: list[LinkFailure] = []
    link_count = 0

    for source in pages:
        parser = _parse_page(source)
        parsed_pages[source] = parser
        for href in parser.hrefs:
            parts = urlsplit(href)
            if parts.scheme or parts.netloc:
                continue
            link_count += 1
            target = _resolve_target(
                site_dir,
                source,
                unquote(parts.path),
                normalized_site_path,
            )
            relative_source = source.relative_to(site_dir)
            if target is None:
                failures.append(
                    LinkFailure(
                        relative_source,
                        href,
                        "target escapes the site directory",
                    )
                )
                continue
            if target.is_dir():
                target = target / "index.html"
            if not target.is_file():
                failures.append(LinkFailure(relative_source, href, "missing target"))
                continue
            fragment = unquote(parts.fragment)
            if not fragment:
                continue
            if target.suffix.lower() != ".html":
                failures.append(
                    LinkFailure(
                        relative_source,
                        href,
                        "fragment target is not an HTML page",
                    )
                )
                continue
            target_parser = parsed_pages.get(target)
            if target_parser is None:
                target_parser = _parse_page(target)
                parsed_pages[target] = target_parser
            if fragment not in target_parser.targets:
                failures.append(LinkFailure(relative_source, href, "missing fragment"))

    return LinkCheckResult(len(pages), link_count, tuple(failures))


def _parse_page(path: Path) -> _PageParser:
    """Parse one generated HTML page."""
    parser = _PageParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return parser


def _normalize_site_path(site_path: str) -> str:
    """Return a canonical absolute hosting path."""
    if not site_path.startswith("/"):
        raise ValueError("site path must start with '/'")
    stripped_path = site_path.strip("/")
    return "/" if not stripped_path else f"/{stripped_path}/"


def _resolve_target(
    site_dir: Path,
    source: Path,
    link_path: str,
    site_path: str,
) -> Path | None:
    """Resolve one local link path and reject traversal outside the site."""
    if not link_path:
        target = source
    elif link_path.startswith("/"):
        relative_path = link_path.lstrip("/")
        site_prefix = site_path.strip("/")
        if site_prefix and (
            relative_path == site_prefix
            or relative_path.startswith(f"{site_prefix}/")
        ):
            relative_path = relative_path[len(site_prefix) :].lstrip("/")
        target = site_dir / relative_path
    else:
        target = source.parent / link_path
    target = target.resolve()
    try:
        target.relative_to(site_dir)
    except ValueError:
        return None
    return target


def main() -> int:
    """Run the generated-site link check from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "site_dir",
        nargs="?",
        type=Path,
        default=Path("site"),
        help="generated site directory (default: site)",
    )
    parser.add_argument(
        "--site-path",
        default="/",
        help="absolute URL path where the site is hosted (default: /)",
    )
    args = parser.parse_args()
    try:
        result = check_site(args.site_dir, args.site_path)
    except ValueError as exc:
        parser.error(str(exc))

    for failure in result.failures:
        print(f"{failure.source}: {failure.href}: {failure.reason}")
    print(
        f"Checked {result.links} local links across {result.pages} HTML pages; "
        f"found {len(result.failures)} failure(s)."
    )
    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
