#!/usr/bin/env bash
set -euo pipefail

# GitHub Actions SHA Pinning Script
#
# Purpose:
#   Resolves semantic versions from .github/actions-lock.yaml to commit SHAs,
#   then rewrites all workflow files to use SHA-pinned references.
#
# Usage:
#   ./scripts/ci/pin-actions.sh [--verify-only] [--update-lock] [--root PATH]
#
# Options:
#   --verify-only    Check if workflows match lock file (CI mode)
#   --update-lock    Resolve the refs already in the lock file to SHAs
#   --root PATH      Operate on another repository root (validation fixtures)
#
# Requirements:
#   - yq (YAML processor): brew install yq OR apt-get install yq
#   - jq (JSON processor)
#   - gh (GitHub CLI) OR curl
#

VERIFY_ONLY=false
UPDATE_LOCK=false
ROOT_OVERRIDE=""

# Parse arguments.
while [[ $# -gt 0 ]]; do
  case $1 in
    --verify-only)
      VERIFY_ONLY=true
      shift
      ;;
    --update-lock)
      UPDATE_LOCK=true
      shift
      ;;
    --root)
      ROOT_OVERRIDE=${2:?missing path after --root}
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--verify-only] [--update-lock] [--root PATH]"
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "$ROOT_OVERRIDE" ]]; then
  REPO_ROOT="$(cd "$ROOT_OVERRIDE" && pwd)"
else
  REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
LOCK_FILE="$REPO_ROOT/.github/actions-lock.yaml"
WORKFLOWS_DIR="$REPO_ROOT/.github/workflows"
ACTIONS_DIR="$REPO_ROOT/.github/actions"

# Dependency Checks

check_dependencies() {
  local missing=()

  if ! command -v yq &> /dev/null; then
    missing+=("yq (install: brew install yq)")
  fi

  if ! command -v jq &> /dev/null; then
    missing+=("jq (install: brew install jq)")
  fi

  if [ ${#missing[@]} -gt 0 ]; then
    echo "ERROR: Missing required dependencies:"
    printf '  - %s\n' "${missing[@]}"
    exit 1
  fi
}

workflow_files() {
  find "$WORKFLOWS_DIR" "$ACTIONS_DIR" \( -name "*.yaml" -o -name "*.yml" \) -type f 2>/dev/null | sort
}

# GitHub API - Resolve ref to SHA

action_repository() {
  local action="$1"
  local owner=${action%%/*}
  local remainder=${action#*/}
  local repository=${remainder%%/*}

  if [[ -z "$owner" || -z "$repository" || "$remainder" == "$action" ]]; then
    echo "ERROR: invalid GitHub Action name '$action'" >&2
    return 1
  fi

  printf '%s/%s\n' "$owner" "$repository"
}

resolve_ref_to_sha() {
  local action="$1"  # e.g., "actions/checkout"
  local ref="$2"     # e.g., "v4" or "master"
  local repository
  repository=$(action_repository "$action")

  echo "  Resolving $action@$ref..." >&2

  # Try gh CLI first (respects auth, higher rate limits)
  if command -v gh &> /dev/null && gh auth status &> /dev/null; then
    local sha
    sha=$(gh api "repos/$repository/commits/$ref" --jq '.sha' 2>/dev/null || echo "")
    if [ -n "$sha" ]; then
      echo "$sha"
      return 0
    fi
  fi

  # Fallback to curl (unauthenticated, 60 req/hour limit)
  local sha
  sha=$(curl -sSL "https://api.github.com/repos/$repository/commits/$ref" 2>/dev/null | jq -r '.sha // empty')

  if [ -z "$sha" ]; then
    echo "ERROR: Failed to resolve $action@$ref" >&2
    return 1
  fi

  echo "$sha"
}

# Update Lock File with SHAs for Locked Refs

update_lock_file() {
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Updating actions-lock.yaml with SHAs for locked refs"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""

  local actions
  actions=$(yq eval 'keys | .[]' "$LOCK_FILE" | grep -v '^#')

  local temp_lock
  temp_lock=$(mktemp)
  cp "$LOCK_FILE" "$temp_lock"

  for action in $actions; do
    echo "Processing: $action"

    local ref
    ref=$(yq eval ".\"$action\".ref" "$LOCK_FILE")

    if [ "$ref" = "null" ] || [ -z "$ref" ]; then
      echo "  ⚠️  Skipping: No ref defined"
      continue
    fi

    local sha
    if ! sha=$(resolve_ref_to_sha "$action" "$ref"); then
      echo "  ❌ Failed to resolve SHA"
      continue
    fi

    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # Update YAML in-place
    yq eval -i ".\"$action\".sha = \"$sha\"" "$temp_lock"
    yq eval -i ".\"$action\".updated = \"$timestamp\"" "$temp_lock"

    echo "  ✅ $ref → $sha"
    echo ""
  done

  mv "$temp_lock" "$LOCK_FILE"
  echo "✅ Lock file updated successfully"
  echo ""
}

# Rewrite Workflow Files to Use SHA Pins

rewrite_workflows() {
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Rewriting workflows to use SHA-pinned actions"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""

  local files
  files=$(workflow_files)

  for file in $files; do
    echo "Processing: $(basename "$file")"

    # Read all actions from lock file
    local actions
    actions=$(yq eval 'keys | .[]' "$LOCK_FILE" | grep -v '^#')

    local modified=false
    local temp_file
    temp_file=$(mktemp)
    cp "$file" "$temp_file"

    for action in $actions; do
      local ref sha
      ref=$(yq eval ".\"$action\".ref" "$LOCK_FILE")
      sha=$(yq eval ".\"$action\".sha" "$LOCK_FILE")

      if [ "$sha" = "null" ] || [ -z "$sha" ]; then
        echo "  ⚠️  Skipping $action: No SHA in lock file"
        continue
      fi

      # Pattern: uses: actions/checkout@v4
      # Replace with: uses: actions/checkout@abc123...  # v4

      # Check if action exists in file
      if ! grep -q "uses: $action@" "$temp_file"; then
        continue
      fi

      # Perform replacement using sed
      # Match: uses: actions/checkout@<anything>
      # Replace: uses: actions/checkout@<sha>  # <ref>
      sed -i.bak -E "s|(uses: $action)@[a-zA-Z0-9._-]+( *#.*)?$|\1@$sha  # $ref|g" "$temp_file"

      modified=true
      echo "  ✅ Pinned $action@$ref → $sha"
    done

    if [ "$modified" = true ]; then
      mv "$temp_file" "$file"
      rm -f "$file.bak"
      echo "  💾 Saved changes"
    else
      rm -f "$temp_file" "$temp_file.bak"
      echo "  ℹ️  No changes needed"
    fi

    echo ""
  done

  echo "✅ All workflows updated"
  echo ""
}

# Verify Mode - Check if workflows are properly pinned

verify_workflows() {
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Verifying workflows are properly pinned"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""

  local files
  files=$(workflow_files)

  local unpinned=()
  local unlocked=()
  local mismatched=()

  for file in $files; do
    # Find lines with "uses:" that are NOT pinned to a SHA
    # SHA pattern: 40 hex characters
    # Exclude local actions: composite actions (./.github/actions/) and reusable workflows (./.github/workflows/)
    local issues
    issues=$(grep -n "uses:" "$file" | grep -v "@[0-9a-f]\{40\}" | grep -v "^#" | grep -v "uses: \\./\\.github/actions/" | grep -v "uses: \\./\\.github/workflows/" || true)

    if [ -n "$issues" ]; then
      unpinned+=("$file")
      echo "❌ UNPINNED: $(basename "$file")"
      while IFS= read -r issue; do
        echo "     $issue"
      done <<< "$issues"
      echo ""
    fi

    local external_uses
    external_uses=$(grep -n "uses:" "$file" \
      | grep -v "uses: \\./\\.github/actions/" \
      | grep -v "uses: \\./\\.github/workflows/" \
      | grep -v '^docker://' \
      | grep -v '^$' \
      || true)

    local line use_ref action workflow_sha workflow_ref lock_sha lock_ref
    while IFS= read -r line; do
      [ -z "$line" ] && continue
      use_ref=$(sed -E 's/^[0-9]+:[[:space:]-]*uses:[[:space:]]*([^[:space:]#]+).*/\1/' <<<"$line")
      action="${use_ref%@*}"
      if [ "$action" = "$use_ref" ]; then
        continue
      fi
      if ! yq eval -e ".\"$action\"" "$LOCK_FILE" >/dev/null 2>&1; then
        unlocked+=("$file:$action")
        continue
      fi

      workflow_sha="${use_ref##*@}"
      workflow_ref=$(sed -nE 's/.*#[[:space:]]*([^[:space:]]+)[[:space:]]*$/\1/p' <<<"$line")
      lock_sha=$(yq eval -r ".\"$action\".sha // \"\"" "$LOCK_FILE")
      lock_ref=$(yq eval -r ".\"$action\".ref // \"\"" "$LOCK_FILE")

      if [[ "$workflow_sha" != "$lock_sha" ]]; then
        mismatched+=("$file:$action SHA $workflow_sha != $lock_sha")
      fi
      if [[ -z "$workflow_ref" || "$workflow_ref" != "$lock_ref" ]]; then
        mismatched+=("$file:$action ref ${workflow_ref:-<missing>} != $lock_ref")
      fi
    done <<< "$external_uses"
  done

  if [ ${#unlocked[@]} -gt 0 ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "❌ LOCK COVERAGE FAILED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Found external actions missing from .github/actions-lock.yaml:"
    printf '  - %s\n' "${unlocked[@]}"
    echo ""
  fi

  if [ ${#mismatched[@]} -gt 0 ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "❌ LOCK CONSISTENCY FAILED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Found workflow pins that differ from .github/actions-lock.yaml:"
    printf '  - %s\n' "${mismatched[@]}"
    echo ""
  fi

  if [ ${#unpinned[@]} -gt 0 ] || [ ${#unlocked[@]} -gt 0 ] || [ ${#mismatched[@]} -gt 0 ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "❌ VERIFICATION FAILED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Found ${#unpinned[@]} unpinned file(s), ${#unlocked[@]} unlocked action(s), and ${#mismatched[@]} lock mismatch(es)."
    echo ""
    echo "To fix, run:"
    echo "  1. Add missing external actions to .github/actions-lock.yaml"
    echo "  2. just pin-actions"
    echo ""
    exit 1
  else
    echo "✅ All workflows properly pinned"
    echo ""
  fi
}

# Main Execution

main() {
  check_dependencies

  if [ "$VERIFY_ONLY" = true ]; then
    verify_workflows
    exit 0
  fi

  if [ "$UPDATE_LOCK" = true ]; then
    update_lock_file
  fi

  rewrite_workflows

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "✅ GitHub Actions pinning complete"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  echo "Next steps:"
  echo "  1. Review changes: git diff"
  echo "  2. Test workflows locally if possible"
  echo "  3. Commit changes: git add -A && git commit -m 'ci: pin GitHub Actions to commit SHAs'"
  echo ""
}

main
