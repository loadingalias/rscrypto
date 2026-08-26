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

install_wasm_tools() {
  local platform tmpdir install_dir installed_version
  tmpdir="$(mktemp -d)"
  install_dir="${WASM_TOOLS_HOME:-$HOME/.wasm-tools}"

  ci_tool_download wasm-tools "$tmpdir"
  platform="${CI_TOOL_HOST_ARCH}-${CI_TOOL_HOST_OS}"
  echo "Installing wasm-tools $CI_TOOL_VERSION for $platform"
  tar -xzf "$CI_TOOL_ARCHIVE_PATH" -C "$tmpdir"
  mkdir -p "$install_dir/bin"
  cp "$tmpdir/wasm-tools-${CI_TOOL_VERSION}-${platform}/wasm-tools" "$install_dir/bin/wasm-tools"
  chmod +x "$install_dir/bin/wasm-tools"

  installed_version=$("$install_dir/bin/wasm-tools" --version)
  if [[ "$installed_version" =~ ([0-9]+\.[0-9]+\.[0-9]+) ]]; then
    installed_version=${BASH_REMATCH[1]}
  else
    echo "wasm-tools version mismatch: expected $CI_TOOL_VERSION, got $installed_version" >&2
    return 1
  fi
  [[ "$installed_version" == "$CI_TOOL_VERSION" ]] || {
    echo "wasm-tools version mismatch: expected $CI_TOOL_VERSION, got $installed_version" >&2
    return 1
  }

  rm -rf "$tmpdir"
  export PATH="$install_dir/bin:$PATH"
}

build_validate_run_wasm_vectors() {
  local variant=$1
  local rustflags=$2
  local target_dir="$CI_TOOL_REPO_ROOT/target/wasm-runtime-vectors/$variant"
  local artifact="$target_dir/$TARGET/debug/rscrypto-wasm-runtime-vectors.wasm"
  local wat="$artifact.wat"
  local manifest="tools/wasm-runtime-vectors/Cargo.toml"

  CARGO_TARGET_DIR="$target_dir" RUSTFLAGS="$rustflags" \
    cargo build --locked --manifest-path "$manifest" --target "$TARGET"

  wasm-tools validate "$artifact"
  if [[ "$variant" == simd128 ]]; then
    wasm-tools print "$artifact" >"$wat"
    grep -Eq '\b(v128\.(load|store|const)|i(8x16|16x8|32x4|64x2)\.|f(32x4|64x2)\.)' "$wat" || {
      echo "SIMD WASM artifact contains no SIMD instruction" >&2
      return 1
    }
  fi

  wasmtime "$artifact"
}

run_wasm_runtime_vectors() {
  if [[ "$TARGET" != "wasm32-wasip1" ]]; then
    return
  fi

  install_wasmtime
  install_wasm_tools
  build_validate_run_wasm_vectors default ""
  build_validate_run_wasm_vectors simd128 "-C target-feature=+simd128"
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Cross-compile sweep: $TARGET ($DEPTH)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Always: bare no-default-features.
cargo check --locked --target "$TARGET" --no-default-features --lib
cargo build --locked --target "$TARGET" --no-default-features --lib --release

# Always: alloc.
cargo check --locked --target "$TARGET" --no-default-features --features alloc --lib

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
    "websocket-sha1"
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
    cargo check --locked --target "$TARGET" --no-default-features --features "$feature_set" --lib
  done

  # Full no_std release build.
  cargo build --locked --target "$TARGET" --no-default-features --features "alloc,checksums,hashes,auth,aead" --lib --release
fi

# Target-specific smoke (shallow gets a token extra so each target has >0 feature coverage).
if [[ "$DEPTH" == "shallow" ]]; then
  case "$TARGET" in
    thumbv6m-none-eabi)
      cargo check --locked --target "$TARGET" --no-default-features --features checksums --lib
      ;;
    wasm32-unknown-unknown)
      cargo check --locked --target "$TARGET" --no-default-features --features hashes --lib
      RUSTFLAGS="-C target-feature=+simd128" cargo check --locked --target "$TARGET" --no-default-features --features hashes --lib
      ;;
  esac
fi

run_wasm_runtime_vectors

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ $TARGET ($DEPTH) passed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
