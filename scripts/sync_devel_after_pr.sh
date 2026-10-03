#!/usr/bin/env bash
# Fast-forward the long-lived development branch after a merge pull request.
set -euo pipefail

if [[ $# -ne 0 ]]; then
    echo "Usage: $0" >&2
    exit 2
fi

repository_root="$(git rev-parse --show-toplevel)"
cd "$repository_root"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "The worktree must be clean before synchronizing devel." >&2
    exit 1
fi

git fetch origin
git switch devel
if ! git merge-base --is-ancestor HEAD origin/main; then
    echo "devel contains work not integrated into main; merge its pull request first." >&2
    exit 1
fi
git merge --ff-only origin/main
git push origin devel
