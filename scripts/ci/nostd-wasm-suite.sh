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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/ci-tool-integrity.sh
source "$SCRIPT_DIR/../lib/ci-tool-integrity.sh"

rustup target add "$TARGET"

install_wasmtime() {
  local platform tmpdir install_dir installed_version expected_version
  tmpdir="$(mktemp -d)"
  install_dir="${WASMTIME_HOME:-$HOME/.wasmtime}"

  ci_tool_download wasmtime "$tmpdir"
  platform="${CI_TOOL_HOST_ARCH}-${CI_TOOL_HOST_OS}"
  echo "Installing Wasmtime $CI_TOOL_VERSION for $platform"
  tar -xJf "$CI_TOOL_ARCHIVE_PATH" -C "$tmpdir"
  mkdir -p "$install_dir/bin"
  cp "$tmpdir/wasmtime-${CI_TOOL_VERSION}-${platform}/wasmtime" "$install_dir/bin/wasmtime"
  chmod +x "$install_dir/bin/wasmtime"

  installed_version=$("$install_dir/bin/wasmtime" --version)
  expected_version=${CI_TOOL_VERSION#v}
  if [[ "$installed_version" =~ ([0-9]+\.[0-9]+\.[0-9]+) ]]; then
    installed_version=${BASH_REMATCH[1]}
  else
    echo "Wasmtime version mismatch: expected $expected_version, got $installed_version" >&2
    return 1
  fi
  [[ "$installed_version" == "$expected_version" ]] || {
    echo "Wasmtime version mismatch: expected $expected_version, got $installed_version" >&2
    return 1
  }

  rm -rf "$tmpdir"
  export PATH="$install_dir/bin:$PATH"
}

run_wasm_runtime_vectors() {
  if [[ "$TARGET" != "wasm32-wasip1" ]]; then
    return
  fi

  install_wasmtime
  export CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime"

  local manifest="tools/wasm-runtime-vectors/Cargo.toml"
  cargo run --manifest-path "$manifest" --target "$TARGET"
  RUSTFLAGS="-C target-feature=+simd128" \
    cargo run --manifest-path "$manifest" --target "$TARGET"
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Cross-compile sweep: $TARGET ($DEPTH)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Always: bare no-default-features.
cargo check --target "$TARGET" --no-default-features --lib
cargo build --target "$TARGET" --no-default-features --lib --release

# Always: alloc.
cargo check --target "$TARGET" --no-default-features --features alloc --lib

if [[ "$DEPTH" == "deep" ]]; then
  # Union of the historical check-all facade matrix and the dedicated weekly
  # no_std/WASM combinations. This preserves the old coverage exactly once.
  FEATURE_SETS=(
    "crc16"
    "crc24"
    "crc32"
    "crc64"
    "alloc,crc32"
    "sha2"
    "sha3"
    "xxh3"
    "hmac"
    "hmac-sha3"
    "kmac"
    "hkdf"
    "poly1305"
    "rsa"
    "x25519"
    "ml-kem"
    "chacha20poly1305"
    "ascon-aead"
    "checksums"
    "hashes"
    "macs"
    "kdfs"
    "signatures"
    "key-exchange"
    "auth"
    "aead"
    "full"
    "alloc,checksums"
    "alloc,hashes"
    "alloc,checksums,hashes,auth,aead"
  )

  for feature_set in "${FEATURE_SETS[@]}"; do
    cargo check --target "$TARGET" --no-default-features --features "$feature_set" --lib
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
      RUSTFLAGS="-C target-feature=+simd128" cargo check --target "$TARGET" --no-default-features --features hashes --lib
      ;;
  esac
fi

run_wasm_runtime_vectors

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ $TARGET ($DEPTH) passed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
