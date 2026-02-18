#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CHECK_ONLY=false
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=true
fi

if ! command -v cargo-upgrade >/dev/null 2>&1; then
  echo "Error: cargo-upgrade is required but not installed."
  echo "Install with: cargo install cargo-edit"
  exit 1
fi

update_workspace() {
  local workspace_path="$1"
  local workspace_name="$2"

  if [[ ! -f "$workspace_path/Cargo.toml" ]]; then
    return 0
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Updating: $workspace_name"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  if [[ "$CHECK_ONLY" == true ]]; then
    echo "  [CHECK MODE] Would run cargo update + cargo upgrade in $workspace_path"
    return 0
  fi

  pushd "$workspace_path" >/dev/null
  cargo update --workspace
  cargo upgrade --recursive
  popd >/dev/null
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "rscrypto dependency update"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ "$CHECK_ONLY" == true ]]; then
  echo "Running in check mode"
fi

update_workspace "$REPO_ROOT" "workspace"

FUZZ_COUNT=0
for fuzz_dir in "$REPO_ROOT"/crates/*/fuzz; do
  if [[ -d "$fuzz_dir" && -f "$fuzz_dir/Cargo.toml" ]]; then
    update_workspace "$fuzz_dir" "fuzz: $(basename "$(dirname "$fuzz_dir")")"
    FUZZ_COUNT=$((FUZZ_COUNT + 1))
  fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "GitHub actions lock/pinning"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ "$CHECK_ONLY" == true ]]; then
  "$REPO_ROOT/scripts/ci/pin-actions.sh" --verify-only
else
  "$REPO_ROOT/scripts/ci/pin-actions.sh" --update-lock
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Workspace updated: yes"
echo "Fuzz workspaces seen: $FUZZ_COUNT"
if [[ "$CHECK_ONLY" == true ]]; then
  echo "Mode: check-only"
else
  echo "Mode: applied"
fi
