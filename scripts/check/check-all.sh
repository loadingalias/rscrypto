#!/usr/bin/env bash
# Complete host and cross-target validation for rscrypto.

set -euo pipefail

[[ $# -eq 0 ]] || { echo "Usage: $0" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=../lib/targets.sh
source "$SCRIPT_DIR/../lib/targets.sh"
# shellcheck source=../lib/feature-profiles.sh
source "$SCRIPT_DIR/../lib/feature-profiles.sh"

NIGHTLY_TOOLCHAIN=$("$SCRIPT_DIR/../lib/toolchain.sh" --nightly)

cargo_for_target() {
  local target=$1
  shift
  if [[ "$target" == riscv32* ]]; then
    RUSTUP_TOOLCHAIN="$NIGHTLY_TOOLCHAIN" cargo "$@"
  else
    cargo "$@"
  fi
}

run_constrained_check() {
  local target=$1
  local target_dir=$2
  local log_file=$3
  local feature_set=${4:-}
  local args=(
    check
    -p rscrypto
    --no-default-features
    --target "$target"
    --lib
    --locked
  )

  if [[ -n "$feature_set" ]]; then
    args+=(--features "$feature_set")
  fi

  CARGO_TARGET_DIR="$target_dir" cargo_for_target "$target" "${args[@]}" \
    >>"$log_file" 2>&1
}

run_constrained_target() {
  local target=$1
  local log_dir=$2
  local target_dir="target/cross-check/$target"
  local log_file="$log_dir/$target.log"

  if [[ "$target" == riscv32* ]]; then
    ensure_target "$target" "$NIGHTLY_TOOLCHAIN"
  else
    ensure_target "$target"
  fi
  mkdir -p "$target_dir"
  : >"$log_file"

  step "$target check (no features)"
  if run_constrained_check "$target" "$target_dir" "$log_file"; then
    ok
  else
    fail
    show_error "$log_file"
    return 1
  fi

  step "$target check (feature contract)"
  for feature_set in alloc "${CONSTRAINED_FEATURE_SETS[@]}"; do
    if ! run_constrained_check "$target" "$target_dir" "$log_file" "$feature_set"; then
      fail
      show_error "$log_file"
      return 1
    fi
  done
  ok

  step "$target release build (no features)"
  if CARGO_TARGET_DIR="$target_dir" cargo_for_target "$target" build --locked \
    -p rscrypto --no-default-features --target "$target" --lib --release \
    >>"$log_file" 2>&1; then
    ok
  else
    fail
    show_error "$log_file"
    return 1
  fi

  step "$target release build (alloc)"
  if CARGO_TARGET_DIR="$target_dir" cargo_for_target "$target" build --locked \
    -p rscrypto --no-default-features --features alloc --target "$target" --lib \
    --release >>"$log_file" 2>&1; then
    ok
  else
    fail
    show_error "$log_file"
    return 1
  fi
}

run_constrained_checks() {
  local log_dir
  log_dir=$(mktemp -d)
  trap 'rm -rf "$log_dir"' EXIT

  local constrained_targets=(
    "${NOSTD_TARGETS[@]:+${NOSTD_TARGETS[@]}}"
    "${WASM_TARGETS[@]:+${WASM_TARGETS[@]}}"
  )

  echo ""
  echo "Constrained targets ${DIM}(rscrypto, parallel)${RESET}"
  if [[ ${#constrained_targets[@]} -eq 0 ]]; then
    skip "no constrained targets configured" ".config/target-matrix.json"
    return 0
  fi

  local pids=()
  local target
  for i in "${!constrained_targets[@]}"; do
    target=${constrained_targets[$i]}
    (run_constrained_target "$target" "$log_dir") &
    pids[i]=$!
  done

  local failures=0
  for i in "${!constrained_targets[@]}"; do
    target=${constrained_targets[$i]}
    step "$target group"
    if wait "${pids[$i]}"; then
      ok
    else
      fail
      failures=1
    fi
  done

  [[ "$failures" -eq 0 ]] || return 1
  echo "${GREEN}✓${RESET} Constrained targets passed"
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Complete rscrypto validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

"$SCRIPT_DIR/check.sh" --all --feature-matrix
"$SCRIPT_DIR/zeroize-evidence.sh"

echo ""
echo "Cross targets ${DIM}(parallel)${RESET}"
log_dir=$(mktemp -d)
trap 'rm -rf "$log_dir"' EXIT

jobs=(windows linux ibm constrained)
("$SCRIPT_DIR/check-win.sh") >"$log_dir/windows.log" 2>&1 &
pids=("$!")
("$SCRIPT_DIR/check-zig.sh" linux) >"$log_dir/linux.log" 2>&1 &
pids+=("$!")
("$SCRIPT_DIR/check-zig.sh" ibm) >"$log_dir/ibm.log" 2>&1 &
pids+=("$!")
(run_constrained_checks) >"$log_dir/constrained.log" 2>&1 &
pids+=("$!")

failures=0
for i in "${!jobs[@]}"; do
  job=${jobs[$i]}
  step "$job group"
  if wait "${pids[$i]}"; then
    ok
  else
    fail
    show_error "$log_dir/$job.log"
    failures=1
  fi
done

[[ "$failures" -eq 0 ]] || exit 1
echo ""
echo "${GREEN}✓${RESET} Complete rscrypto validation passed"
