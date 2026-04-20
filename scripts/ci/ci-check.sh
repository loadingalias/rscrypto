#!/usr/bin/env bash
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ci-check.sh - CI Quality Checks (runs natively on target machines)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# This script runs in CI on actual target machines (x86_64, aarch64, Windows).
# No cross-compilation - each CI runner compiles natively for its platform.
#
# Local development uses `just check` for host checks and `just check-all` for cross-targets.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 CI Quality Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Prefer cargo-installed tools over any preinstalled runner tools.
export PATH="$HOME/.cargo/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

maybe_disable_sccache
apply_ci_resource_profile

# ─────────────────────────────────────────────────────────────────────────────
# Infrastructure correctness checks
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "🧱 Infrastructure checks..."
if [[ "${CI:-}" == "true" ]]; then
  echo "skipped in CI (local-only)"
else
  "$SCRIPT_DIR/check-infra.sh"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Format Check
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "📐 Checking formatting..."
cargo fmt --all -- --check

# ─────────────────────────────────────────────────────────────────────────────
# Cargo Check
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "🔧 Running cargo check..."
cargo check --workspace --all-targets --all-features

echo ""
echo "🧪 Running executable feature matrix..."
"$SCRIPT_DIR/../test/test-feature-matrix.sh"

# ─────────────────────────────────────────────────────────────────────────────
# Clippy
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "📎 Running clippy..."
cargo clippy --workspace --all-targets --all-features -- -D warnings

# ─────────────────────────────────────────────────────────────────────────────
# Documentation
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "📚 Building documentation..."
if [[ "${RSCRYPTO_CI_RESOURCE_PROFILE:-}" == "constrained" ]]; then
  skip "rustdoc on constrained CI runners" "native check/clippy/test coverage retained"
else
  RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All CI checks passed!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
