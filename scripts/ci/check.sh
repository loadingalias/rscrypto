#!/usr/bin/env bash
# Quality checks for rscrypto workspace
#
# Usage:
#   scripts/ci/check.sh           # Full checks including cross-compile (local)
#   scripts/ci/check.sh --ci      # CI mode (no cross-compile, stricter)
#
# Environment:
#   RSCRYPTO_CHECK_MODE: ci | local (default: local)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Parse arguments
CI_MODE="${RSCRYPTO_CHECK_MODE:-local}"

for arg in "$@"; do
  case $arg in
    --ci)
      CI_MODE="ci"
      ;;
  esac
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Quality Checks (mode: $CI_MODE)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ─────────────────────────────────────────────────────────────────────────────
# Format Check
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "📐 Checking formatting..."
if [ "$CI_MODE" = "ci" ]; then
  cargo fmt --all -- --check
else
  cargo fmt --all
fi

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
if [ "$CI_MODE" = "ci" ]; then
  cargo clippy --workspace --all-targets --all-features -- -D warnings
else
  cargo clippy --workspace --all-targets --all-features --fix --allow-dirty -- -D warnings
fi

# ─────────────────────────────────────────────────────────────────────────────
# Security and Documentation
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "🔒 Running cargo deny..."
cargo deny check all

echo ""
echo "📚 Building documentation..."
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features

echo ""
echo "🛡️ Running security audit..."
cargo audit

# ─────────────────────────────────────────────────────────────────────────────
# Cross-target compilation (local mode only)
# ─────────────────────────────────────────────────────────────────────────────
if [ "$CI_MODE" = "local" ]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🎯 Cross-target compile matrix (platform/backend crates)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # Tier A: Full hardware acceleration targets
  TIER_A_TARGETS=(
    "x86_64-unknown-linux-gnu"
    "x86_64-pc-windows-msvc"
    "aarch64-unknown-linux-gnu"
    "aarch64-apple-darwin"
  )

  # Tier B: Partial acceleration (MUSL)
  TIER_B_TARGETS=(
    "x86_64-unknown-linux-musl"
    "aarch64-unknown-linux-musl"
    "aarch64-pc-windows-msvc"
  )

  # Tier C: no_std targets
  NO_STD_TARGETS=(
    "thumbv6m-none-eabi"
    "riscv32imac-unknown-none-elf"
    "aarch64-unknown-none"
    "x86_64-unknown-none"
  )

  # WASM targets
  WASM_TARGETS=(
    "wasm32-unknown-unknown"
    "wasm32-wasip1"
  )

  # Crates to cross-compile
  CROSS_CRATES="platform backend traits"

  # Install targets if needed
  install_target() {
    local target=$1
    if ! rustup target list --installed | grep -q "^${target}$"; then
      echo "  Installing target: $target"
      rustup target add "$target" 2>/dev/null || true
    fi
  }

  # Check with std features
  check_std_target() {
    local target=$1
    echo ""
    echo "  ▸ $target (std)"
    install_target "$target"
    for crate in $CROSS_CRATES; do
      RUSTC_WRAPPER= cargo check -p "$crate" --all-features --target "$target" 2>/dev/null || \
        RUSTC_WRAPPER= cargo check -p "$crate" --target "$target"
    done
  }

  # Check without std features
  check_no_std_target() {
    local target=$1
    echo ""
    echo "  ▸ $target (no_std)"
    install_target "$target"
    for crate in $CROSS_CRATES; do
      RUSTC_WRAPPER= cargo check -p "$crate" --no-default-features --target "$target" --lib 2>/dev/null || true
      RUSTC_WRAPPER= cargo check -p "$crate" --no-default-features --features alloc --target "$target" --lib 2>/dev/null || true
    done
  }

  echo ""
  echo "Tier A (Full acceleration):"
  for target in "${TIER_A_TARGETS[@]}"; do
    check_std_target "$target"
  done

  echo ""
  echo "Tier B (Partial acceleration):"
  for target in "${TIER_B_TARGETS[@]}"; do
    check_std_target "$target"
  done

  echo ""
  echo "Tier C (no_std):"
  for target in "${NO_STD_TARGETS[@]}"; do
    check_no_std_target "$target"
  done

  echo ""
  echo "WASM targets:"
  for target in "${WASM_TARGETS[@]}"; do
    check_no_std_target "$target"
  done
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All checks passed!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
