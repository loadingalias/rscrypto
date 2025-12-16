#!/usr/bin/env bash
set -euo pipefail

echo "Running Memory Safety Tests via Miri..."
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Miri Configuration for rscrypto
#
# The checksum crate uses SIMD intrinsics which Miri cannot interpret.
# We force the portable code path by disabling the `std` feature.
#
# On x86_64: Run natively with --no-default-features (no compile-time SIMD)
# On ARM: Skip (CRC intrinsics are compile-time enabled, cross-compile needs
#         x86_64-linux-gnu-gcc which isn't installed). Run in CI on x86_64.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOST_ARCH=$(uname -m)
if [ "$HOST_ARCH" = "arm64" ] || [ "$HOST_ARCH" = "aarch64" ]; then
  echo "Miri skipped on ARM (run in CI on x86_64)"
  exit 0
fi

# Crates to test with Miri
MIRI_CRATES="checksum"
EXCLUDED_CRATES="platform hash traits"

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
for arg in "$@"; do
  [ "$arg" = "--all" ] && continue
  CRATES+=("$arg")
done

if [ ${#CRATES[@]} -gt 0 ]; then
  CRATES_TO_TEST="${CRATES[*]}"
else
  CRATES_TO_TEST="$MIRI_CRATES"
fi

PKG_FLAGS=""
PKG_LIST=""

for crate in $CRATES_TO_TEST; do
  if is_in_list "$crate" "$EXCLUDED_CRATES"; then
    echo "Skipping $crate (excluded)"
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

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running Miri tests for:$PKG_LIST"
echo "Mode: --no-default-features --features alloc --lib"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# shellcheck disable=SC2086
cargo miri test $PKG_FLAGS \
  --no-default-features \
  --features alloc \
  --lib

echo ""
echo "All Miri tests passed"
