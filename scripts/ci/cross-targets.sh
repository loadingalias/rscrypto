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
# shellcheck source=../lib/ci-tool-integrity.sh
source "$SCRIPT_DIR/../lib/ci-tool-integrity.sh"

maybe_disable_sccache

zig_temp_root=${RUNNER_TEMP:-${TMPDIR:-/tmp}}
zig_root=$(mktemp -d "$zig_temp_root/rscrypto-zig.XXXXXX")
trap 'rm -rf "$zig_root"' EXIT
ci_tool_download zig "$zig_root"
tar -xJf "$CI_TOOL_ARCHIVE_PATH" --strip-components=1 -C "$zig_root"
export PATH="$zig_root:$PATH"

if [[ "$(zig version)" != "$CI_TOOL_VERSION" ]]; then
  echo "error: expected Zig $CI_TOOL_VERSION, found $(zig version)" >&2
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
