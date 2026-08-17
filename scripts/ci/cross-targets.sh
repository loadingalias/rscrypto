#!/usr/bin/env bash
set -euo pipefail

DEPTH=${1:-deep}
if [[ "$DEPTH" != "shallow" && "$DEPTH" != "deep" ]]; then
  echo "usage: cross-targets.sh [shallow|deep]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=../lib/targets.sh
source "$SCRIPT_DIR/../lib/targets.sh"

maybe_disable_sccache

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

  RUSTC_WRAPPER="" CARGO_TARGET_DIR="$target_dir" \
    cargo check --locked --target "$target" --no-default-features --lib
  RUSTC_WRAPPER="" CARGO_TARGET_DIR="$target_dir" \
    cargo clippy --locked --target "$target" --lib --all-features
  RUSTC_WRAPPER="" CARGO_TARGET_DIR="$target_dir" \
    cargo build --locked --target "$target" --no-default-features --features alloc --lib --release
done

for target in "${NOSTD_TARGETS[@]}" "${WASM_TARGETS[@]}"; do
  "$SCRIPT_DIR/nostd-wasm-suite.sh" "$target" "$DEPTH"
done

echo "Cross-target validation passed: ${#MUSL_TARGETS[@]} MUSL + ${#NOSTD_TARGETS[@]} no_std + ${#WASM_TARGETS[@]} WASM targets"
