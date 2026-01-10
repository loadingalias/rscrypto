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

maybe_disable_sccache() {
  if [[ -n "${RUSTC_WRAPPER:-}" && "${RUSTC_WRAPPER##*/}" == "sccache" ]]; then
    if ! "$RUSTC_WRAPPER" rustc -vV >/dev/null 2>&1; then
      echo "⚠️  WARNING: sccache is configured but not usable; disabling RUSTC_WRAPPER for this run."
      export RUSTC_WRAPPER=
    fi
  fi
}

maybe_disable_sccache

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
echo "🔒 Running cargo deny..."
cargo deny check all

echo ""
echo "🛡️ Running security audit..."
command -v cargo-audit >/dev/null 2>&1 && cargo-audit --version || true
cargo audit

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
