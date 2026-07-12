#!/usr/bin/env bash
set -euo pipefail

# Architecture-independent CI quality checks. Run once on the primary x86_64
# CI host. Native and cross-target validation have separate owners.

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 CI Quality Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Prefer cargo-installed tools over any preinstalled runner tools.
export PATH="$HOME/.cargo/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "🔗 Checking workflow ownership and action pins..."
"$SCRIPT_DIR/check-ci-ownership.sh"
"$SCRIPT_DIR/check-ci-ownership-test.sh"
"$SCRIPT_DIR/pin-actions.sh" --verify-only

echo ""
echo "🚆 Checking cargo-rail config and unified Cargo graph..."
just check-unify

export RSCRYPTO_SKIP_CHECK_SUPPLY_CHAIN=1
"$SCRIPT_DIR/../check/check.sh" --all

echo ""
echo "🔨 Building all targets..."
cargo build --workspace --all-targets --all-features

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All CI checks passed!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
