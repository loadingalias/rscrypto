#!/usr/bin/env bash
set -euo pipefail

# Linux cross-compilation checks via zig
# Usage: check-linux.sh [--all] [crate1 crate2 ...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=../lib/targets.sh
source "$SCRIPT_DIR/../lib/targets.sh"

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

if [[ ${#LINUX_TARGETS[@]} -eq 0 ]]; then
  skip "no Linux targets configured" "config/target-matrix.toml"
  exit 0
fi

for target in "${LINUX_TARGETS[@]}"; do
  ensure_target "$target"
done

for target in "${LINUX_TARGETS[@]}"; do
  mkdir -p "target/cross-check/$target"
done

pids=()
logs=()
targets=()

for i in "${!LINUX_TARGETS[@]}"; do
  target="${LINUX_TARGETS[$i]}"
  target_dir="target/cross-check/$target"
  log_file="$LOG_DIR/$target.log"
  logs[$i]="$log_file"
  targets[$i]="$target"

  (
    # shellcheck disable=SC2086
    if ! CC="$ZIG_CC" RUSTC_WRAPPER="" CARGO_TARGET_DIR="$target_dir" \
         cargo clippy $CRATE_FLAGS --lib --all-features --target "$target" -- -D warnings \
         >"$log_file" 2>&1; then
      exit 1
    fi

    if [[ "$TUNE_IN_SCOPE" == true && "$target" == x86_64-* ]]; then
      if ! CC="$ZIG_CC" RUSTC_WRAPPER="" CARGO_TARGET_DIR="$target_dir" \
           cargo clippy -p tune --bin rscrypto-blake3-boundary --all-features --target "$target" -- -D warnings \
           >>"$log_file" 2>&1; then
        exit 1
      fi
    fi
  ) &
  pids[$i]=$!
done

FAILED=0
for i in "${!targets[@]}"; do
  target="${targets[$i]}"
  short_name="${target/unknown-linux-/}"

  step "$short_name clippy"
  if wait "${pids[$i]}"; then
    ok
    if [[ "$TUNE_IN_SCOPE" == true && "$target" == x86_64-* ]]; then
      step "$short_name rscrypto-blake3-boundary"
      ok
    fi
  else
    fail
    show_error "${logs[$i]}"
    FAILED=1
  fi
done

if [ $FAILED -ne 0 ]; then
  exit 1
fi

echo "${GREEN}✓${RESET} Linux targets passed"
