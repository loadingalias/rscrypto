#!/usr/bin/env bash
set -euo pipefail

# Linux cross-compilation checks via zig
# Usage: check-linux.sh [--all] [crate1 crate2 ...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck source=lib/targets.sh
source "$SCRIPT_DIR/lib/targets.sh"

# Check for zig
if ! command -v zig >/dev/null 2>&1; then
  echo "Linux targets ${DIM}(skipped)${RESET}"
  skip "zig not installed" "brew install zig"
  exit 0
fi

# Parse args and set CRATE_FLAGS, SCOPE_DESC
get_crate_flags "$@"

export ZIG_CC="$SCRIPT_DIR/zig-cc.sh"

TUNE_IN_SCOPE=false
if [[ "$CRATE_FLAGS" == "--workspace" || "$CRATE_FLAGS" == *"-p tune"* || "$CRATE_FLAGS" == *"-p checksum"* ]]; then
  TUNE_IN_SCOPE=true
fi

LOG_DIR=$(mktemp -d)
trap 'rm -rf "$LOG_DIR"' EXIT

echo "Linux targets ${DIM}($SCOPE_DESC)${RESET}"

for target in "${LINUX_TARGETS[@]}"; do
  # Shorten: x86_64-unknown-linux-gnu -> x86_64-gnu
  short_name="${target/unknown-linux-/}"

  step "$short_name check"
  ensure_target "$target"

  target_dir="target/cross-check/$target"
  mkdir -p "$target_dir"

  # Check
  # shellcheck disable=SC2086
  if ! CC="$ZIG_CC" RUSTC_WRAPPER="" CARGO_TARGET_DIR="$target_dir" \
       cargo check $CRATE_FLAGS --lib --all-features --target "$target" \
       >"$LOG_DIR/$target.log" 2>&1; then
    fail
    show_error "$LOG_DIR/$target.log"
    exit 1
  fi
  ok

  step "$short_name clippy"
  # shellcheck disable=SC2086
  if ! CC="$ZIG_CC" RUSTC_WRAPPER="" CARGO_TARGET_DIR="$target_dir" \
       cargo clippy $CRATE_FLAGS --lib --all-features --target "$target" -- -D warnings \
       >>"$LOG_DIR/$target.log" 2>&1; then
    fail
    show_error "$LOG_DIR/$target.log"
    exit 1
  fi
  ok

  if [[ "$TUNE_IN_SCOPE" == true && "$target" == x86_64-* ]]; then
    step "$short_name rscrypto-tune"
    if ! CC="$ZIG_CC" RUSTC_WRAPPER="" CARGO_TARGET_DIR="$target_dir" \
         cargo clippy -p tune --bin rscrypto-tune --all-features --target "$target" -- -D warnings \
         >>"$LOG_DIR/$target.log" 2>&1; then
      fail
      show_error "$LOG_DIR/$target.log"
      exit 1
    fi
    ok
  fi
done

echo "${GREEN}✓${RESET} Linux targets passed"
