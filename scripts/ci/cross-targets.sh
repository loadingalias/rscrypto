#!/usr/bin/env bash
set -euo pipefail

TARGET=${1:-}
DEPTH=${2:-deep}
if [[ -z "$TARGET" ]]; then
  echo "usage: cross-targets.sh <target> [shallow|deep]" >&2
  exit 2
fi
if [[ "$DEPTH" != "shallow" && "$DEPTH" != "deep" ]]; then
  echo "usage: cross-targets.sh <target> [shallow|deep]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=../lib/targets.sh
source "$SCRIPT_DIR/../lib/targets.sh"

target_is_in() {
  local candidate=$1
  shift
  local item
  for item in "$@"; do
    [[ "$candidate" == "$item" ]] && return 0
  done
  return 1
}

if target_is_in "$TARGET" "${LINUX_TARGETS[@]}" && [[ "$TARGET" == *-musl ]]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "MUSL compile evidence: $TARGET"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  ensure_target "$TARGET"
  target_dir="target/cross-check/$TARGET"
  mkdir -p "$target_dir"

  CARGO_TARGET_DIR="$target_dir" \
    cargo check --locked --target "$TARGET" --no-default-features --lib
  CARGO_TARGET_DIR="$target_dir" \
    cargo clippy --locked --target "$TARGET" --lib --all-features
  CARGO_TARGET_DIR="$target_dir" \
    cargo build --locked --target "$TARGET" --no-default-features --features alloc --lib --release
elif target_is_in "$TARGET" "${NOSTD_TARGETS[@]}" "${WASM_TARGETS[@]}"; then
  "$SCRIPT_DIR/nostd-wasm-suite.sh" "$TARGET" "$DEPTH"
else
  echo "target is not a generic cross-contract row: $TARGET" >&2
  exit 2
fi

echo "Cross-target validation passed: $TARGET ($DEPTH)"
