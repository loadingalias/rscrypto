#!/usr/bin/env bash
# Complete host, feature, and generic cross-target validation for rscrypto.

set -euo pipefail

[[ $# -eq 0 ]] || { echo "Usage: $0" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Complete rscrypto validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

"$SCRIPT_DIR/policy.sh" --all
"$SCRIPT_DIR/check.sh" --all
"$SCRIPT_DIR/feature-contracts.sh" all
"$SCRIPT_DIR/zeroize-evidence.sh"

rows=()
while IFS= read -r row; do
  rows+=("$row")
done < <(
  jq -r '.variants[] | select(.dimensions.operation == "cross") | .id' \
    "$REPO_ROOT/.config/target-matrix.json"
)
[[ ${#rows[@]} -gt 0 ]] || {
  echo "No generic cross-target proofs are configured" >&2
  exit 1
}

echo ""
echo "Cross targets ${DIM}(parallel)${RESET}"
log_dir=$(mktemp -d)
trap 'rm -rf "$log_dir"' EXIT

pids=()
for i in "${!rows[@]}"; do
  row=${rows[$i]}
  ("$SCRIPT_DIR/../ci/target-contracts.sh" run "$row" deep) >"$log_dir/$row.log" 2>&1 &
  pids[i]=$!
done

failures=0
for i in "${!rows[@]}"; do
  row=${rows[$i]}
  step "$row"
  if wait "${pids[$i]}"; then
    ok
  else
    fail
    show_error "$log_dir/$row.log"
    failures=1
  fi
done

[[ "$failures" -eq 0 ]] || exit 1
echo ""
echo "${GREEN}✓${RESET} Complete rscrypto validation passed"
