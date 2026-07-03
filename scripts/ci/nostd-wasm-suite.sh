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

install_wasmtime() {
  if command -v wasmtime >/dev/null 2>&1; then
    return
  fi

  local version="${WASMTIME_VERSION:-v46.0.1}"
  local host_os host_arch platform archive_name archive_url tmpdir archive install_dir

  case "$(uname -s)" in
    Linux) host_os="linux" ;;
    Darwin) host_os="macos" ;;
    *)
      echo "unsupported Wasmtime host OS: $(uname -s)" >&2
      return 1
      ;;
  esac

  case "$(uname -m)" in
    x86_64 | amd64) host_arch="x86_64" ;;
    aarch64 | arm64) host_arch="aarch64" ;;
    *)
      echo "unsupported Wasmtime host arch: $(uname -m)" >&2
      return 1
      ;;
  esac

  platform="${host_arch}-${host_os}"
  archive_name="wasmtime-${version}-${platform}.tar.xz"
  archive_url="https://github.com/bytecodealliance/wasmtime/releases/download/${version}/${archive_name}"
  tmpdir="$(mktemp -d)"
  archive="${tmpdir}/${archive_name}"
  install_dir="${WASMTIME_HOME:-$HOME/.wasmtime}"

  echo "Installing Wasmtime ${version} for ${platform}"
  curl --fail --location --show-error --silent --retry 3 --retry-delay 2 --output "$archive" "$archive_url"
  tar -xJf "$archive" -C "$tmpdir"
  mkdir -p "$install_dir/bin"
  cp "$tmpdir/wasmtime-${version}-${platform}/wasmtime" "$install_dir/bin/wasmtime"
  chmod +x "$install_dir/bin/wasmtime"
  rm -rf "$tmpdir"

  export PATH="$install_dir/bin:$PATH"
  wasmtime --version
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
      RUSTFLAGS="-C target-feature=+simd128" cargo check --target "$TARGET" --no-default-features --features hashes --lib
      ;;
  esac
fi

run_wasm_runtime_vectors

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ $TARGET ($DEPTH) passed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
