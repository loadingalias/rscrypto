#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Miri Memory Safety Tests for rscrypto
#
# Miri interprets Rust code and checks for undefined behavior, memory safety
# violations, and data races. By default this script runs the explicit Miri
# surface only: deterministic shadow invariants plus Miri-only tests for paths
# with real UB risk. Correctness/vector/property coverage belongs to nextest and
# fuzz replay; re-running hundreds of safe correctness tests under an interpreter
# is wasted CI money.
#
# How it works:
#   - Code uses #[cfg(miri)] guards to fall back to portable implementations
#   - SIMD intrinsics that Miri can't interpret are automatically bypassed
#   - Integration proptests run in the regular `just test` lane
#   - Boundary-heavy shadow tests and Miri-only tests validate the same unsafe
#     invariants under interpreter isolation
#   - No special RUSTFLAGS needed - works on ARM and x86 identically
#
# Usage:
#   ./scripts/test/test-miri.sh           # Run focused Miri surface
#   ./scripts/test/test-miri.sh --all     # Run exhaustive lib tests under Miri
#   RSCRYPTO_MIRI_SCOPE=exhaustive ./scripts/test/test-miri.sh
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running Memory Safety Tests via Miri..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

maybe_disable_sccache
unset RUSTC_WRAPPER
unset CARGO_BUILD_RUSTC_WRAPPER

# Miri cannot execute SIMD/CLMUL kernels directly. Force CRC families onto their
# portable tiers so the lane spends time validating real pointer/length logic
# instead of re-running selector machinery that Miri cannot meaningfully
# distinguish from the hardware paths.
export RSCRYPTO_CRC16_CCITT_FORCE=portable
export RSCRYPTO_CRC16_IBM_FORCE=portable
export RSCRYPTO_CRC24_FORCE=portable
export RSCRYPTO_CRC32_FORCE=portable
export RSCRYPTO_CRC64_FORCE=portable

# Single-crate layout: run Miri on the whole rscrypto crate.
# The #[cfg(miri)] guards in dispatch/SIMD modules ensure portable code is used.

MIRI_FEATURES="${RSCRYPTO_MIRI_FEATURES:-full,diag}"
MIRI_SCOPE="${RSCRYPTO_MIRI_SCOPE:-focused}"

case "${1:-}" in
  --all | exhaustive)
    MIRI_SCOPE="exhaustive"
    shift
    ;;
  --focused | focused)
    MIRI_SCOPE="focused"
    shift
    ;;
  "")
    ;;
  *)
    echo "Usage: $0 [--focused|--all]"
    echo "Set RSCRYPTO_MIRI_FEATURES to override the default feature set: $MIRI_FEATURES"
    exit 2
    ;;
esac

if [ "$#" -ne 0 ]; then
  echo "Unexpected extra arguments: $*"
  echo "Usage: $0 [--focused|--all]"
  exit 2
fi

run_miri_lib_filter() {
  local label="$1"
  local filter="$2"

  echo ""
  echo "━━━ $label ━━━"
  cargo miri test --lib --features "$MIRI_FEATURES" "$filter"
}

run_miri_test_target() {
  local label="$1"
  local target="$2"
  local features="$3"

  echo ""
  echo "━━━ $label ━━━"
  cargo miri test --test "$target" --features "$features"
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run Miri Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "Testing: rscrypto"
echo "Mode: $MIRI_SCOPE"
echo "Features: $MIRI_FEATURES"
echo "Out of scope: integration proptests in tests/ (run via just test)"
echo ""

case "$MIRI_SCOPE" in
  focused)
    echo "Scope: explicit UB-risk surface + deterministic shadow invariants"

    run_miri_lib_filter "portable dispatch and checksum shadow invariants" "miri_shadow_tests"
    run_miri_lib_filter "platform detection under Miri" "platform::detect::tests::test_miri_returns_portable"
    run_miri_lib_filter "X25519 portable path under Miri" "auth::x25519::tests::miri_uses_portable_x25519_path"
    run_miri_test_target "Argon2 MatrixView/portable kernel under Miri" "argon2_miri" "argon2"
    ;;
  exhaustive)
    echo "Scope: exhaustive lib tests under Miri"
    cargo miri test --lib --features "$MIRI_FEATURES"
    ;;
  *)
    echo "Invalid RSCRYPTO_MIRI_SCOPE: $MIRI_SCOPE"
    echo "Expected: focused or exhaustive"
    exit 2
    ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "All Miri tests passed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
