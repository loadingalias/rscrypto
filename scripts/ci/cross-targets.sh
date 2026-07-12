#!/usr/bin/env bash
set -euo pipefail

DEPTH=${1:-deep}
if [[ "$DEPTH" != "shallow" && "$DEPTH" != "deep" ]]; then
  echo "usage: cross-targets.sh [shallow|deep]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=../lib/targets.sh
source "$SCRIPT_DIR/../lib/targets.sh"

maybe_disable_sccache

ZIG_VERSION="0.17.0-dev.1282+c0f9b51d8"
ZIG_SHA256="6d81dec0152f6f11f8a12a84a73535b65070fe01b14a1ce75423870d65b3f270"

if ! command -v zig >/dev/null 2>&1; then
  if [[ "$(uname -s)-$(uname -m)" != "Linux-x86_64" ]]; then
    echo "error: Zig $ZIG_VERSION is required for MUSL target validation" >&2
    exit 1
  fi

  zig_root="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/zig-$ZIG_VERSION"
  zig_archive="$(mktemp)"
  trap 'rm -f "$zig_archive"' EXIT
  curl -fsSL --proto '=https' --tlsv1.2 \
    "https://ziglang.org/builds/zig-x86_64-linux-$ZIG_VERSION.tar.xz" \
    -o "$zig_archive"
  echo "$ZIG_SHA256  $zig_archive" | sha256sum --check --status
  mkdir -p "$zig_root"
  tar -xJf "$zig_archive" --strip-components=1 -C "$zig_root"
  export PATH="$zig_root:$PATH"
fi

if [[ "$(zig version)" != "$ZIG_VERSION" ]]; then
  echo "error: expected Zig $ZIG_VERSION, found $(zig version)" >&2
  exit 1
fi

ZIG_CC="$REPO_ROOT/scripts/check/zig-cc.sh"
MUSL_TARGETS=()
for target in "${LINUX_TARGETS[@]}"; do
  if [[ "$target" == *-musl ]]; then
    MUSL_TARGETS+=("$target")
  fi
done

if [[ ${#MUSL_TARGETS[@]} -ne 2 ]]; then
  echo "error: expected two MUSL targets, found ${#MUSL_TARGETS[@]}" >&2
  exit 1
fi

for target in "${MUSL_TARGETS[@]}"; do
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "MUSL compile evidence: $target"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  ensure_target "$target"
  target_dir="target/cross-check/$target"
  mkdir -p "$target_dir"

  CC="$ZIG_CC" RUSTC_WRAPPER="" CARGO_TARGET_DIR="$target_dir" \
    cargo check --target "$target" --no-default-features --lib
  CC="$ZIG_CC" RUSTC_WRAPPER="" CARGO_TARGET_DIR="$target_dir" \
    cargo clippy --target "$target" --lib --all-features -- -D warnings
  CC="$ZIG_CC" RUSTC_WRAPPER="" CARGO_TARGET_DIR="$target_dir" \
    cargo build --target "$target" --no-default-features --features alloc --lib --release
done

for target in "${NOSTD_TARGETS[@]}" "${WASM_TARGETS[@]}"; do
  "$SCRIPT_DIR/nostd-wasm-suite.sh" "$target" "$DEPTH"
done

echo "Cross-target validation passed: ${#MUSL_TARGETS[@]} MUSL + ${#NOSTD_TARGETS[@]} no_std + ${#WASM_TARGETS[@]} WASM targets"
