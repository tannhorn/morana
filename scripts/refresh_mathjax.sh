#!/usr/bin/env bash
# Refresh the locally vendored MathJax component and New Computer Modern font.
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 VERSION" >&2
    exit 1
fi

version="$1"
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "VERSION must be an exact release, for example 4.1.3." >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_dir="$(cd "$script_dir/.." && pwd)"
asset_dir="$repository_dir/docs/assets/mathjax"
temporary_dir="$(mktemp -d)"
stage_dir="$temporary_dir/mathjax"

cleanup() {
    rm -rf "$temporary_dir"
}
trap cleanup EXIT

npm install \
    --ignore-scripts \
    --no-package-lock \
    --no-save \
    --prefix "$temporary_dir" \
    "mathjax@$version" \
    "@mathjax/mathjax-newcm-font@$version"

mathjax_dir="$temporary_dir/node_modules/mathjax"
font_dir="$temporary_dir/node_modules/@mathjax/mathjax-newcm-font"

for required_path in \
    "$mathjax_dir/LICENSE" \
    "$mathjax_dir/tex-chtml.js" \
    "$mathjax_dir/input/tex/extensions/boldsymbol.js" \
    "$font_dir/chtml.js" \
    "$font_dir/chtml"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required MathJax asset is missing: $required_path" >&2
        exit 1
    fi
done

mathjax_version="$(node -p "require(process.argv[1]).version" "$mathjax_dir/package.json")"
font_version="$(node -p "require(process.argv[1]).version" "$font_dir/package.json")"
if [[ "$mathjax_version" != "$version" || "$font_version" != "$version" ]]; then
    echo "Downloaded package versions do not both match $version." >&2
    exit 1
fi

mkdir -p "$stage_dir/input/tex/extensions" "$stage_dir/font"
cp "$mathjax_dir/LICENSE" "$stage_dir/LICENSE"
cp "$mathjax_dir/tex-chtml.js" "$stage_dir/tex-chtml.js"
cp "$mathjax_dir/input/tex/extensions/boldsymbol.js" \
    "$stage_dir/input/tex/extensions/boldsymbol.js"
cp "$font_dir/chtml.js" "$stage_dir/font/chtml.js"
cp -a "$font_dir/chtml" "$stage_dir/font/chtml"

rsync --archive --delete --checksum "$stage_dir/" "$asset_dir/"

echo "Refreshed $asset_dir from MathJax $version."
