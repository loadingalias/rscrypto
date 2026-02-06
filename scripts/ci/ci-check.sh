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

# ─────────────────────────────────────────────────────────────────────────────
# Infrastructure correctness checks
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "🧱 Running infrastructure checks..."
"$SCRIPT_DIR/check-infra.sh"

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

# ─────────────────────────────────────────────────────────────────────────────
# Clippy
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "📎 Running clippy..."
cargo clippy --workspace --all-targets --all-features -- -D warnings

# ─────────────────────────────────────────────────────────────────────────────
# Dependency & Security Checks
# ─────────────────────────────────────────────────────────────────────────────
echo ""
if [[ "${RSCRYPTO_SKIP_POLICY_CHECKS:-}" == "true" ]]; then
  echo "🔒 Skipping cargo deny/audit (RSCRYPTO_SKIP_POLICY_CHECKS=true)"
else
  echo "🔒 Running cargo deny..."
  cargo deny check all

  echo ""
  echo "🛡️ Running security audit..."
  command -v cargo-audit >/dev/null 2>&1 && cargo-audit --version || true
  cargo audit
fi

# ─────────────────────────────────────────────────────────────────────────────
# Documentation
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "📚 Building documentation..."
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All CI checks passed!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
