"""Check local documentation links and recognized project cross-links."""

# Documentation checkers intentionally share a small command-line shape.
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import SplitResult, unquote, urlsplit

import markdown
import yaml


@dataclass(frozen=True)
class LinkFailure:
    """Describe one invalid documentation or project link."""

    source: Path
    href: str
    reason: str


@dataclass(frozen=True)
class LinkCheckResult:
    """Store generated-site link-check counts and failures."""

    pages: int
    links: int
    failures: tuple[LinkFailure, ...]


@dataclass(frozen=True)
class _LinkContext:
    """Store local roots and public URLs used to resolve checked links."""

    site_dir: Path
    site_path: str
    site_url: str | None
    repository_root: Path | None
    repository_url: str | None
    repository_ref: str


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


def check_site(
    site_dir: Path,
    site_path: str = "/",
    *,
    site_url: str | None = None,
    repository_root: Path | None = None,
    repository_url: str | None = None,
    repository_ref: str = "main",
    markdown_files: tuple[Path, ...] = (),
) -> LinkCheckResult:
    """Check local links and fragments below one generated site directory.

    Parameters
    ----------
    site_dir
        Directory containing generated HTML documentation.
    site_path
        URL path where the generated site is hosted. For a GitHub Pages project
        site, this is normally ``/OWNER-REPOSITORY/`` or ``/REPOSITORY/``.
    site_url
        Published documentation URL. Absolute links below this URL are checked
        against ``site_dir`` without making network requests.
    repository_root
        Source checkout root used to validate recognized repository links.
    repository_url
        Canonical GitHub repository URL. ``blob`` and ``tree`` links for
        ``repository_ref`` are checked against ``repository_root``.
    repository_ref
        GitHub branch or tag recognized in repository links.
    markdown_files
        Repository Markdown files whose recognized absolute links are checked.
    """
    site_dir = site_dir.resolve()
    if not site_dir.is_dir():
        raise ValueError(f"generated site directory does not exist: {site_dir}")
    normalized_site_path = _normalize_site_path(site_path)
    if (repository_root is None) != (repository_url is None):
        raise ValueError("repository_root and repository_url must be provided together")
    if repository_root is not None:
        repository_root = repository_root.resolve()
        if not repository_root.is_dir():
            raise ValueError(
                f"repository root directory does not exist: {repository_root}"
            )
    if markdown_files and repository_root is None:
        raise ValueError("markdown_files require repository_root")
    context = _LinkContext(
        site_dir,
        normalized_site_path,
        site_url,
        repository_root,
        repository_url,
        repository_ref,
    )

    pages = tuple(sorted(site_dir.rglob("*.html")))
    if not pages:
        raise ValueError(f"generated site contains no HTML pages: {site_dir}")
    parsed_pages: dict[Path, _PageParser] = {}
    failures: list[LinkFailure] = []
    link_count = 0

    for source in pages:
        parser = _parse_page(source)
        parsed_pages[source] = parser
        checked_count, page_failures = _check_hrefs(
            parser.hrefs,
            source,
            source.relative_to(site_dir),
            context,
            parsed_pages,
        )
        link_count += checked_count
        failures.extend(page_failures)

    for source in markdown_files:
        source = source.resolve()
        assert repository_root is not None
        try:
            relative_source = source.relative_to(repository_root)
        except ValueError as exc:
            raise ValueError(
                f"Markdown source is outside repository root: {source}"
            ) from exc
        parser = _parse_markdown(source)
        checked_count, source_failures = _check_hrefs(
            parser.hrefs,
            site_dir / "index.html",
            relative_source,
            context,
            parsed_pages,
            recognized_absolute_only=True,
        )
        link_count += checked_count
        failures.extend(source_failures)

    return LinkCheckResult(len(pages), link_count, tuple(failures))


def _check_hrefs(
    hrefs: list[str],
    source_page: Path,
    source_label: Path,
    context: _LinkContext,
    parsed_pages: dict[Path, _PageParser],
    *,
    recognized_absolute_only: bool = False,
) -> tuple[int, list[LinkFailure]]:
    """Check a collection of links and return its count and failures."""
    checked_count = 0
    failures = []
    for href in hrefs:
        is_checked, failure_reason = _check_href(
            href,
            source_page,
            context,
            parsed_pages,
            recognized_absolute_only=recognized_absolute_only,
        )
        if not is_checked:
            continue
        checked_count += 1
        if failure_reason is not None:
            failures.append(LinkFailure(source_label, href, failure_reason))
    return checked_count, failures


def _check_href(
    href: str,
    source: Path,
    context: _LinkContext,
    parsed_pages: dict[Path, _PageParser],
    *,
    recognized_absolute_only: bool = False,
) -> tuple[bool, str | None]:
    """Return whether a link was checked and its optional failure reason."""
    parts = urlsplit(href)
    if parts.scheme or parts.netloc:
        resolved = _resolve_known_absolute(parts, context)
        if resolved is None:
            return False, None
        target, expected_kind, root = resolved
    else:
        if recognized_absolute_only:
            return False, None
        target = _resolve_target(
            context.site_dir,
            source,
            unquote(parts.path),
            context.site_path,
        )
        if target is None:
            return True, "target escapes the site directory"
        expected_kind = "site"
        root = context.site_dir

    target = target.resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return True, f"target escapes the {expected_kind} root"

    if expected_kind != "site":
        return True, _repository_failure(target, expected_kind)
    return True, _site_failure(target, parts.fragment, parsed_pages)


def _repository_failure(target: Path, expected_kind: str) -> str | None:
    """Return a failure when a repository link has the wrong target kind."""
    if expected_kind == "repository file" and not target.is_file():
        return "missing repository file"
    if expected_kind == "repository directory" and not target.is_dir():
        return "missing repository directory"
    return None


def _site_failure(
    target: Path,
    raw_fragment: str,
    parsed_pages: dict[Path, _PageParser],
) -> str | None:
    """Return a failure for a missing generated-site target or fragment."""
    if target.is_dir():
        target = target / "index.html"
    if not target.is_file():
        return "missing target"
    fragment = unquote(raw_fragment)
    if not fragment:
        return None
    if target.suffix.lower() != ".html":
        return "fragment target is not an HTML page"
    target_parser = parsed_pages.get(target)
    if target_parser is None:
        target_parser = _parse_page(target)
        parsed_pages[target] = target_parser
    if fragment not in target_parser.targets:
        return "missing fragment"
    return None


def _resolve_known_absolute(
    parts: SplitResult,
    context: _LinkContext,
) -> tuple[Path, str, Path] | None:
    """Map a known published-site or GitHub URL to a local target."""
    if context.site_url is not None:
        site_parts = urlsplit(context.site_url)
        site_prefix = site_parts.path.rstrip("/")
        if (
            parts.scheme == site_parts.scheme
            and parts.netloc == site_parts.netloc
            and (parts.path == site_prefix or parts.path.startswith(f"{site_prefix}/"))
        ):
            relative_path = unquote(parts.path[len(site_prefix) :].lstrip("/"))
            return context.site_dir / relative_path, "site", context.site_dir

    if context.repository_root is None or context.repository_url is None:
        return None
    repository_parts = urlsplit(context.repository_url)
    if (
        parts.scheme != repository_parts.scheme
        or parts.netloc != repository_parts.netloc
    ):
        return None
    repository_path = repository_parts.path.rstrip("/")
    for link_kind, expected_kind in (
        ("blob", "repository file"),
        ("tree", "repository directory"),
    ):
        prefix = f"{repository_path}/{link_kind}/{context.repository_ref}/"
        if parts.path.startswith(prefix):
            relative_path = unquote(parts.path[len(prefix) :])
            return (
                context.repository_root / relative_path,
                expected_kind,
                context.repository_root,
            )
    return None


def _parse_page(path: Path) -> _PageParser:
    """Parse one generated HTML page."""
    parser = _PageParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return parser


def _parse_markdown(path: Path) -> _PageParser:
    """Render one Markdown source sufficiently to collect its links."""
    parser = _PageParser()
    parser.feed(markdown.markdown(path.read_text(encoding="utf-8")))
    parser.close()
    return parser


def _tracked_markdown_files(repository_root: Path) -> tuple[Path, ...]:
    """Return Markdown files tracked by the repository."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), "ls-files", "-z", "--", "*.md"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("could not enumerate tracked Markdown files") from exc
    paths = completed.stdout.decode("utf-8").split("\0")
    return tuple(repository_root / path for path in paths if path)


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
            relative_path == site_prefix or relative_path.startswith(f"{site_prefix}/")
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
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("mkdocs.yml"),
        help="MkDocs configuration supplying site_url and repo_url",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
        help="source checkout root (default: current directory)",
    )
    parser.add_argument(
        "--repository-ref",
        default="main",
        help="GitHub branch or tag used by checked source links (default: main)",
    )
    args = parser.parse_args()
    try:
        config = yaml.safe_load(args.config_file.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("MkDocs config must contain a mapping")
        site_url = config.get("site_url")
        repository_url = config.get("repo_url")
        docs_dir = config.get("docs_dir", "docs")
        if (
            not isinstance(site_url, str)
            or not isinstance(repository_url, str)
            or not isinstance(docs_dir, str)
        ):
            raise ValueError(
                "MkDocs site_url, repo_url, and docs_dir values must be strings"
            )
        repository_root = args.repository_root.resolve()
        docs_root = (repository_root / docs_dir).resolve()
        markdown_files = tuple(
            path
            for path in _tracked_markdown_files(repository_root)
            if not path.resolve().is_relative_to(docs_root)
        )
        result = check_site(
            args.site_dir,
            args.site_path,
            site_url=site_url,
            repository_root=repository_root,
            repository_url=repository_url,
            repository_ref=args.repository_ref,
            markdown_files=markdown_files,
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    for failure in result.failures:
        print(f"{failure.source}: {failure.href}: {failure.reason}")
    print(
        f"Checked {result.links} local and project cross-links across "
        f"{result.pages} HTML pages; "
        f"found {len(result.failures)} failure(s)."
    )
    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
