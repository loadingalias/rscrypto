#!/usr/bin/env bash
# Fast, deterministic checks before publishing the current topic branch.
# Compilation, tests, and exhaustive assurance belong to explicit validation
# recipes and hosted CI, not the transport path.

set -euo pipefail

export PATH="$HOME/.cargo/bin:$PATH"

if [[ $# -ne 0 ]]; then
  echo "Error: scripts/ci/pre-push.sh does not accept arguments" >&2
  exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  echo "Error: unable to find repository root" >&2
  exit 1
fi

# shellcheck source=scripts/lib/common.sh
source "$REPO_ROOT/scripts/lib/common.sh"

cd "$REPO_ROOT"

BRANCH="$(git branch --show-current)"
if [[ -z "$BRANCH" ]]; then
  echo "Error: refusing to push from a detached HEAD" >&2
  exit 1
fi
if [[ "$BRANCH" == "main" ]]; then
  echo "Error: refusing to push directly from main" >&2
  exit 1
fi

BASE_REF=refs/remotes/origin/main
if ! git rev-parse --verify --quiet "$BASE_REF" >/dev/null; then
  echo "Error: $BASE_REF is unavailable; run 'git fetch origin main' first" >&2
  exit 1
fi

MERGE_BASE="$(git merge-base HEAD "$BASE_REF")"
if [[ -z "$MERGE_BASE" ]]; then
  echo "Error: HEAD and $BASE_REF have no merge base" >&2
  exit 1
fi

CHANGED_FILES=()
while IFS= read -r -d '' path; do
  CHANGED_FILES+=("$path")
done < <(git diff --name-only --diff-filter=ACDMRTUXB -z "$MERGE_BASE"..HEAD)

changed_file_matches() {
  local pattern=$1
  local path

  for path in "${CHANGED_FILES[@]}"; do
    if [[ "$path" =~ $pattern ]]; then
      return 0
    fi
  done

  return 1
}

run_shell_syntax_checks() {
  local scripts=()
  local path

  for path in "${CHANGED_FILES[@]}"; do
    if [[ "$path" == scripts/*.sh && -f "$path" ]]; then
      scripts+=("$path")
    fi
  done

  if [[ ${#scripts[@]} -eq 0 ]]; then
    skip "Shell syntax" "no changed shell scripts"
    return 0
  fi

  step "Shell syntax"
  for path in "${scripts[@]}"; do
    bash -n "$path"
  done
  ok
}

is_cargo_rail_release_branch() {
  [[ "$BRANCH" == rail/release-* ]] || return 1
  git log --format=%B "$BASE_REF"..HEAD \
    | grep -Fxq 'Rail-Release-Mode: prepare'
}

echo "Running fast pre-push checks..."
echo "branch: $BRANCH"
echo "outgoing files: ${#CHANGED_FILES[@]}"

step "Outgoing diff hygiene"
git diff --check "$MERGE_BASE"..HEAD
ok

run_shell_syntax_checks

if changed_file_matches '^justfile$'; then
  step "Justfile parse"
  just --list >/dev/null
  ok
else
  skip "Justfile parse" "justfile unchanged"
fi

if changed_file_matches '^\.config/rail\.toml$'; then
  step "Cargo-rail config"
  cargo rail config validate --strict
  cargo rail config migrate --check
  ok
else
  skip "Cargo-rail config" "rail config unchanged"
fi

if changed_file_matches '(^|/)(Cargo\.toml|Cargo\.lock|build\.rs)$|^rust-toolchain\.toml$|^\.cargo/|^\.config/(rail\.toml|target-matrix\.json|toolchains\.toml)$'; then
  step "Cargo graph consistency"
  cargo rail unify --check
  ok
else
  skip "Cargo graph consistency" "cargo graph inputs unchanged"
fi

if [[ ${#CHANGED_FILES[@]} -eq 0 ]]; then
  skip "Release intent coverage" "no outgoing changes"
elif is_cargo_rail_release_branch; then
  skip "Release intent coverage" "Cargo Rail release branch has consumed its reviewed intent"
else
  step "Release intent coverage"
  cargo rail change check --merge-base --required
  ok
fi

echo "Fast pre-push checks passed."
