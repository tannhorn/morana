#!/usr/bin/env bash
# Synchronize the long-lived development branch after a squash merge.
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 <PR_NUMBER>" >&2
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
git merge origin/main -m "Sync devel with main after PR #$1"
git push origin devel
