#!/usr/bin/env bash
# Pre-push hook: quality checks before allowing push.
# Install:
#   ln -sf ../../scripts/ci/pre-push.sh .git/hooks/pre-push

set -euo pipefail

echo "Running pre-push checks..."

# Shell syntax sanity across scripts/ — catches bashisms and typos before CI.
while IFS= read -r script; do
  bash -n "$script"
done < <(git ls-files | rg '^scripts/.*\.sh$' || true)

# Workflow action pin check — keeps .github/actions-lock.yaml honest.
just check-actions

# Host fmt/check/clippy/docs.
just check

echo "All checks passed."
