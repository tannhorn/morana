#!/usr/bin/env bash
# Smoke-test representative generated documentation pages in an offline browser.
set -euo pipefail

site_dir="$(cd "${1:-site}" && pwd)"
browser="${BROWSER:-chromium}"
temporary_files=()

cleanup() {
    rm -f "${temporary_files[@]}"
}

trap cleanup EXIT

if ! command -v "$browser" >/dev/null 2>&1; then
    echo "A Chromium-compatible browser is required; set BROWSER if needed." >&2
    exit 1
fi

require_marker() {
    local page="$1"
    local rendered="$2"
    local marker="$3"
    local expectation="$4"

    if ! rg -q "$marker" "$rendered"; then
        echo "$page: rendered DOM is missing $expectation" >&2
        exit 1
    fi
}

check_page() {
    local page="$1"
    local source="$site_dir/$page"
    local homepage="index.html"
    local rendered
    local browser_log

    if [[ "$page" == */* ]]; then
        homepage="../index.html"
    fi

    if [[ ! -f "$source" ]]; then
        echo "Missing built page: $source" >&2
        exit 1
    fi
    if [[ "$page" != "index.html" ]]; then
        require_marker "$page" "$source" \
            "<a href=\"$homepage\" class=\"md-nav__link\"" \
            "a direct homepage link in the narrow-screen navigation drawer"
    fi

    rendered="$(mktemp)"
    browser_log="$(mktemp)"
    temporary_files+=("$rendered" "$browser_log")
    if ! "$browser" \
        --headless \
        --no-sandbox \
        --disable-gpu \
        --host-resolver-rules='MAP * ~NOTFOUND' \
        --virtual-time-budget=5000 \
        --dump-dom "file://$source" 2>"$browser_log" \
        | tee "$rendered" >/dev/null; then
        cat "$browser_log" >&2
        echo "$page: headless browser rendering failed" >&2
        exit 1
    fi

    require_marker "$page" "$rendered" '<html[^>]*class="[^"]*js' \
        "the initialized JavaScript document class"
    require_marker "$page" "$rendered" 'data-md-component="main"' \
        "the Material main container"
    require_marker "$page" "$rendered" 'data-md-component="content"' \
        "the Material content container"
    require_marker "$page" "$rendered" \
        "<a href=\"$homepage\" title=\"Morana\" class=\"md-header__button md-logo\"" \
        "an explicit local-file homepage target in the header logo"
    if rg -q 'class="no-js"|::: morana' "$rendered"; then
        echo "$page: JavaScript initialization or source rendering is incomplete" >&2
        exit 1
    fi

    case "$page" in
        index.html)
            require_marker "$page" "$rendered" 'id="supported-calculations"' \
                "the supported-calculations section"
            require_marker "$page" "$rendered" \
                '<p class="admonition-title">Project status and intended use</p>' \
                "the project-status warning"
            require_marker "$page" "$rendered" 'src="assets/morana.png"' \
                "the Morana logo"
            echo "$page: home content rendered offline"
            ;;
        getting_started.html)
            require_marker "$page" "$rendered" 'id="quickstart"' \
                "the quickstart section"
            require_marker "$page" "$rendered" 'language-python highlight' \
                "the highlighted Python quickstart"
            require_marker "$page" "$rendered" 'data-md-type="copy"' \
                "the JavaScript-initialized code-copy control"
            echo "$page: quickstart and code controls rendered offline"
            ;;
        reference/core.html)
            require_marker "$page" "$rendered" 'id="morana.Result"' \
                "the generated Result reference"
            require_marker "$page" "$rendered" 'doc-signature highlight' \
                "generated API signatures"
            require_marker "$page" "$rendered" 'data-md-type="copy"' \
                "the JavaScript-initialized signature-copy control"
            echo "$page: generated API reference rendered offline"
            ;;
        theory_references.html)
            check_math "$page" "$source" "$rendered"
            ;;
        *)
            echo "$page: no page-specific smoke assertions are defined" >&2
            exit 1
            ;;
    esac
}

check_math() {
    local page="$1"
    local source="$2"
    local rendered="$3"
    local source_count
    local rendered_count

    source_count="$(rg -o 'class="arithmatex"' "$source" | wc -l)"
    rendered_count="$(rg -o '<mjx-container' "$rendered" | wc -l)"
    if [[ "$source_count" -ne "$rendered_count" ]]; then
        echo "$page: rendered $rendered_count of $source_count math expressions" >&2
        exit 1
    fi

    if rg -q '<mjx-merror|data-mjx-error|<mjx-mtext[^>]*data-latex="\\boldsymbol"' "$rendered"; then
        echo "$page: MathJax reported an error or left \\boldsymbol literal" >&2
        exit 1
    fi

    for marker in \
        '<mjx-math[^>]*data-latex=' \
        'data-semantic-type=' \
        '<mjx-container[^>]*tabindex="0"'
    do
        require_marker "$page" "$rendered" "$marker" \
            "MathJax TeX annotations and accessibility details"
    done

    if ! rg -q 'data-latex="[^"]*\\boldsymbol' "$rendered"; then
        echo "$page: vector TeX source was not retained in MathJax annotations" >&2
        exit 1
    fi

    echo "$page: $rendered_count math expressions rendered offline"
}

check_page index.html
check_page getting_started.html
check_page reference/core.html
check_page theory_references.html
