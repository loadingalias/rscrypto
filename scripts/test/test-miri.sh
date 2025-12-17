#!/usr/bin/env bash
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Miri Memory Safety Tests for rscrypto
#
# Miri interprets Rust code and checks for undefined behavior, memory safety
# violations, and data races. This script runs Miri on the checksum crate.
#
# How it works:
#   - Code uses #[cfg(miri)] guards to fall back to portable implementations
#   - SIMD intrinsics that Miri can't interpret are automatically bypassed
#   - No special RUSTFLAGS needed - works on ARM and x86 identically
#
# Usage:
#   ./scripts/test/test-miri.sh           # Run Miri tests
#   ./scripts/test/test-miri.sh checksum  # Test specific crate
#   ./scripts/test/test-miri.sh --all     # Test all compatible crates
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running Memory Safety Tests via Miri..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Crates that work with Miri (have #[cfg(miri)] portable fallbacks)
MIRI_CRATES="checksum"

# Crates excluded from Miri testing
# - platform: No tests, just feature detection
# - hash: Not yet implemented
# - traits: Just trait definitions, no unsafe code
EXCLUDED_CRATES="platform hash traits rscrypto"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Argument Parsing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

is_in_list() {
  local item="$1"
  local list="$2"
  for i in $list; do
    [ "$item" = "$i" ] && return 0
  done
  return 1
}

CRATES=()
ALL_FLAG=false

for arg in "$@"; do
  if [ "$arg" = "--all" ]; then
    ALL_FLAG=true
  else
    CRATES+=("$arg")
  fi
done

# Determine which crates to test
if [ ${#CRATES[@]} -gt 0 ]; then
  CRATES_TO_TEST="${CRATES[*]}"
elif [ "$ALL_FLAG" = true ]; then
  CRATES_TO_TEST="$MIRI_CRATES"
else
  CRATES_TO_TEST="$MIRI_CRATES"
fi

# Build package flags, filtering exclusions
PKG_FLAGS=""
PKG_LIST=""

for crate in $CRATES_TO_TEST; do
  if is_in_list "$crate" "$EXCLUDED_CRATES"; then
    echo "Skipping $crate (excluded from Miri)"
  else
    PKG_FLAGS="$PKG_FLAGS -p $crate"
    PKG_LIST="$PKG_LIST $crate"
  fi
done

if [ -z "$PKG_FLAGS" ]; then
  echo "No compatible crates to test"
  exit 0
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run Miri Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "Testing:$PKG_LIST"
echo "Mode: --lib (unit tests only, no benchmarks)"
echo ""

# Run Miri on library tests only (--lib excludes benchmarks/examples)
# The #[cfg(miri)] guards in dispatch functions ensure portable code is used
# shellcheck disable=SC2086
cargo miri test $PKG_FLAGS --lib

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "All Miri tests passed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
