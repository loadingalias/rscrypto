#!/usr/bin/env bash
set -euo pipefail

echo "Running Memory Safety Tests via Miri (strict mode)..."
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Strict Miri Configuration for rscrypto
#
# We run Miri with NO MIRIFLAGS for maximum strictness. This means:
#   - Full isolation (no filesystem access)
#   - Default alignment checks
#   - All UB detection enabled
#
# SIMD Intrinsics:
#   The checksum crate uses SIMD intrinsics which Miri cannot interpret.
#   To test with Miri, we force the portable code path by:
#     1. Disabling `std` feature (prevents runtime SIMD detection)
#     2. Using a target without compile-time SIMD features
#
#   On aarch64-apple-darwin (Apple Silicon), CRC intrinsics are always
#   enabled at compile time, so we MUST cross-compile to x86_64.
#
#   On x86_64 without RUSTFLAGS, no SIMD features are compile-time enabled,
#   so the portable path is used when std is disabled.
#
# Library Tests Only:
#   We only run --lib tests because:
#     - Integration tests (proptests) require filesystem access for
#       failure persistence, which conflicts with strict isolation
#     - Proptests are covered by normal test runs, Miri validates
#       the core memory safety of the implementations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# STRICT MODE: Unset any existing MIRIFLAGS
unset MIRIFLAGS

# Detect host architecture
HOST_ARCH=$(uname -m)

# Always cross-compile to x86_64-unknown-linux-gnu for consistency:
# - On Apple Silicon: Required because aarch64 CRC intrinsics are compile-time enabled
# - On x86_64: Still useful for consistent behavior across platforms
MIRI_TARGET="x86_64-unknown-linux-gnu"

if [ "$HOST_ARCH" = "arm64" ] || [ "$HOST_ARCH" = "aarch64" ]; then
  echo "Host: aarch64 - cross-compiling to $MIRI_TARGET (required: CRC intrinsics)"
else
  echo "Host: $HOST_ARCH - cross-compiling to $MIRI_TARGET (for consistency)"
fi

# Crates that need portable mode (disable std to avoid SIMD intrinsics)
PORTABLE_MODE_CRATES="checksum"

# Crates excluded from Miri entirely (platform detection, SIMD-only code)
EXCLUDED_CRATES="platform hash traits"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Argument Parsing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Helper: Check if a crate is in a space-separated list
is_in_list() {
  local crate="$1"
  local list="$2"
  for item in $list; do
    if [ "$crate" = "$item" ]; then
      return 0
    fi
  done
  return 1
}

# Parse arguments
CRATES=()
for arg in "$@"; do
  if [ "$arg" = "--all" ]; then
    # --all is now just documentation, we always test all compatible crates
    true
  else
    CRATES+=("$arg")
  fi
done

# Determine which crates to test
if [ ${#CRATES[@]} -gt 0 ]; then
  CRATES_TO_TEST="${CRATES[*]}"
else
  # Default: test all portable mode crates
  CRATES_TO_TEST="$PORTABLE_MODE_CRATES"
fi

# Build package flags
PKG_FLAGS=""
PKG_LIST=""

for crate in $CRATES_TO_TEST; do
  if is_in_list "$crate" "$EXCLUDED_CRATES"; then
    echo "⏭️  Skipping $crate (excluded: no testable Miri code)"
  elif is_in_list "$crate" "$PORTABLE_MODE_CRATES"; then
    PKG_FLAGS="$PKG_FLAGS -p $crate"
    PKG_LIST="$PKG_LIST $crate"
  else
    echo "⏭️  Skipping $crate (not in compatible list)"
  fi
done

# Check if anything to test
if [ -z "$PKG_FLAGS" ]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "No compatible crates to test with Miri"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  exit 0
fi

# Ensure Miri sysroot for target
# Note: We don't use +nightly here - rust-toolchain.toml specifies the nightly version
echo "Ensuring Miri sysroot for $MIRI_TARGET..."
cargo miri setup --target "$MIRI_TARGET" 2>/dev/null || true

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run Miri Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running Miri tests (portable mode, strict) for:$PKG_LIST"
echo "Mode: --no-default-features --features alloc --lib"
echo "Target: $MIRI_TARGET"
echo "MIRIFLAGS: (none - strict mode)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# shellcheck disable=SC2086
cargo miri test $PKG_FLAGS \
  --target "$MIRI_TARGET" \
  --no-default-features \
  --features alloc \
  --lib

echo ""
echo "✅ All Miri tests passed (strict mode)!"
