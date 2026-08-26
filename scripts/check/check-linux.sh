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
NIGHTLY_TOOLCHAIN=$("$SCRIPT_DIR/../lib/toolchain.sh" --nightly)

LOG_DIR=$(mktemp -d)
trap 'rm -rf "$LOG_DIR"' EXIT

echo "Linux targets ${DIM}($SCOPE_DESC)${RESET}"

if [[ ${#LINUX_TARGETS[@]} -eq 0 ]]; then
  skip "no Linux targets configured" ".config/target-matrix.json"
  exit 0
fi

for target in "${LINUX_TARGETS[@]}"; do
  if [[ "$target" == riscv64* ]]; then
    ensure_target "$target" "$NIGHTLY_TOOLCHAIN"
  else
    ensure_target "$target"
  fi
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
  logs[i]="$log_file"
  targets[i]="$target"

  (
    # Keep the command prefix non-empty: Bash 3.2 treats an empty array
    # expansion as an unbound variable under `set -u`.
    toolchain_env=(env)
    if [[ "$target" == riscv64* ]]; then
      toolchain_env+=("RUSTUP_TOOLCHAIN=$NIGHTLY_TOOLCHAIN")
    fi
    # shellcheck disable=SC2086
    if ! CC="$ZIG_CC" CARGO_TARGET_DIR="$target_dir" \
         "${toolchain_env[@]}" cargo clippy $CRATE_FLAGS --lib --all-features --locked --target "$target" \
         >"$log_file" 2>&1; then
      exit 1
    fi
  ) &
  pids[i]=$!
done

FAILED=0
for i in "${!targets[@]}"; do
  target="${targets[$i]}"
  short_name="${target/unknown-linux-/}"

  step "$short_name clippy"
  if wait "${pids[$i]}"; then
    ok
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
