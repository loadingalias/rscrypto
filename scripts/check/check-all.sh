#!/usr/bin/env bash
# Complete host, feature, and generic cross-target validation for rscrypto.

set -euo pipefail

[[ $# -eq 0 ]] || { echo "Usage: $0" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Complete rscrypto validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

"$SCRIPT_DIR/policy.sh" --all
"$SCRIPT_DIR/check.sh" --all
"$SCRIPT_DIR/msrv.sh"
"$SCRIPT_DIR/feature-contracts.sh" all
"$SCRIPT_DIR/../test/test-examples.sh"
"$SCRIPT_DIR/zeroize-evidence.sh"
echo ""
echo "${GREEN}✓${RESET} Complete rscrypto validation passed"
