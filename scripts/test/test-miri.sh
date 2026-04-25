#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Miri Memory Safety Tests for rscrypto
#
# Miri interprets Rust code and checks for undefined behavior, memory safety
# violations, and data races. This script runs the in-crate library test surface,
# including deterministic Miri shadow invariants for paths that are randomized in
# the proptest lane.
#
# How it works:
#   - Code uses #[cfg(miri)] guards to fall back to portable implementations
#   - SIMD intrinsics that Miri can't interpret are automatically bypassed
#   - Integration proptests run in the regular `just test` lane; equivalent
#     boundary-heavy shadow tests live in lib tests so Miri can still validate
#     invariants
#   - No special RUSTFLAGS needed - works on ARM and x86 identically
#
# Usage:
#   ./scripts/test/test-miri.sh           # Run Miri tests
#   ./scripts/test/test-miri.sh backend   # Test specific crate
#   ./scripts/test/test-miri.sh --all     # Test all compatible crates
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run Miri Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "Testing: rscrypto"
echo "Mode: --lib with production feature coverage and Miri shadow invariants"
echo "Features: full,diag"
echo "Scope: unit tests + in-crate deterministic invariant coverage"
echo "Out of scope: integration proptests in tests/ (run via just test)"
echo ""

# Run Miri on library tests only (--lib excludes benchmarks/examples and
# integration tests) while covering the production primitive surface instead of
# default features alone. Deterministic shadow tests keep the core invariant
# surface covered under Miri isolation.
cargo miri test --lib --features full,diag

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "All Miri tests passed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
