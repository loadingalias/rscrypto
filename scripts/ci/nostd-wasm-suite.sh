#!/usr/bin/env bash
# No-std / WASM cross-compilation sanity sweep.
#
# Usage:
#   nostd-wasm-suite.sh <target-triple> <depth>
#
# depth:
#   shallow  per-PR smoke (bare + alloc)
#   deep     weekly full sweep (bare + alloc + every individual + combined features)

set -euo pipefail

TARGET="${1:?usage: nostd-wasm-suite.sh <target-triple> <depth>}"
DEPTH="${2:-shallow}"

rustup target add "$TARGET"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Cross-compile sweep: $TARGET ($DEPTH)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Always: bare no-default-features.
cargo check --target "$TARGET" --no-default-features --lib
cargo build --target "$TARGET" --no-default-features --lib --release

# Always: alloc.
cargo check --target "$TARGET" --no-default-features --features alloc --lib

if [[ "$DEPTH" == "deep" ]]; then
  # Individual feature flags.
  for f in checksums hashes aead auth; do
    cargo check --target "$TARGET" --no-default-features --features "$f" --lib
  done

  # Common feature combinations.
  for combo in "alloc,checksums" "alloc,hashes" "alloc,checksums,hashes,auth,aead"; do
    cargo check --target "$TARGET" --no-default-features --features "$combo" --lib
  done

  # Full no_std release build.
  cargo build --target "$TARGET" --no-default-features --features "alloc,checksums,hashes,auth,aead" --lib --release
fi

# Target-specific smoke (shallow gets a token extra so each target has >0 feature coverage).
if [[ "$DEPTH" == "shallow" ]]; then
  case "$TARGET" in
    thumbv6m-none-eabi)
      cargo check --target "$TARGET" --no-default-features --features checksums --lib
      ;;
    wasm32-unknown-unknown)
      cargo check --target "$TARGET" --no-default-features --features hashes --lib
      ;;
  esac
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ $TARGET ($DEPTH) passed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
