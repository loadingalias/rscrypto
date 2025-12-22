#!/usr/bin/env bash
set -euo pipefail

# Windows cross-compilation checks via cargo-xwin
# Usage: check-win.sh [--all] [crate1 crate2 ...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck source=lib/targets.sh
source "$SCRIPT_DIR/lib/targets.sh"

# Check for cargo-xwin
if ! cargo xwin --version >/dev/null 2>&1; then
  echo "Windows targets ${DIM}(skipped)${RESET}"
  skip "cargo-xwin not installed" "cargo install cargo-xwin && brew install llvm"
  exit 0
fi

maybe_disable_sccache

# Parse args and set CRATE_FLAGS, SCOPE_DESC
get_crate_flags "$@"

CHECKSUM_IN_SCOPE=false
if [[ "$CRATE_FLAGS" == "--workspace" || "$CRATE_FLAGS" == *"-p checksum"* ]]; then
  CHECKSUM_IN_SCOPE=true
fi

LOG_DIR=$(mktemp -d)
trap 'rm -rf "$LOG_DIR"' EXIT

echo "Windows targets ${DIM}($SCOPE_DESC)${RESET}"

# Initialize xwin cache once (avoids race conditions)
step "Initializing SDK cache"
if ! CARGO_TARGET_DIR="target/cross-check/xwin-init" \
     cargo xwin check -p traits --lib --target x86_64-pc-windows-msvc \
     >"$LOG_DIR/xwin-init.log" 2>&1; then
  fail
  show_error "$LOG_DIR/xwin-init.log"
  exit 1
fi
ok

for target in "${WIN_TARGETS[@]}"; do
  short_name="${target%-pc-windows-msvc}"  # x86_64 or aarch64

  step "$short_name check"
  ensure_target "$target"

  target_dir="target/cross-check/$target"
  mkdir -p "$target_dir"

  # Check
  # shellcheck disable=SC2086
  if ! CARGO_TARGET_DIR="$target_dir" \
       cargo xwin check $CRATE_FLAGS --lib --all-features --target "$target" \
       >"$LOG_DIR/$target.log" 2>&1; then
    fail
    show_error "$LOG_DIR/$target.log"
    exit 1
  fi
  ok

  step "$short_name clippy"
  # shellcheck disable=SC2086
  if ! CARGO_TARGET_DIR="$target_dir" \
       cargo xwin clippy $CRATE_FLAGS --lib --all-features --target "$target" -- -D warnings \
       >>"$LOG_DIR/$target.log" 2>&1; then
    fail
    show_error "$LOG_DIR/$target.log"
    exit 1
  fi
  ok

  if [[ "$CHECKSUM_IN_SCOPE" == true && "$target" == "x86_64-pc-windows-msvc" ]]; then
    step "$short_name crc64-tune"
    if ! CARGO_TARGET_DIR="$target_dir" \
         cargo xwin clippy -p checksum --bin crc64-tune --all-features --target "$target" -- -D warnings \
         >>"$LOG_DIR/$target.log" 2>&1; then
      fail
      show_error "$LOG_DIR/$target.log"
      exit 1
    fi
    ok
  fi
done

echo "${GREEN}✓${RESET} Windows targets passed"
