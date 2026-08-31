#!/usr/bin/env bash
# Cross-compile the rscrypto library with Zig for one target group.

set -euo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 <linux|ibm>" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=../lib/targets.sh
source "$SCRIPT_DIR/../lib/targets.sh"

group=$1
case "$group" in
  linux)
    label=Linux
    targets=("${LINUX_TARGETS[@]:+${LINUX_TARGETS[@]}}")
    ;;
  ibm)
    label=IBM
    targets=("${IBM_TARGETS[@]:+${IBM_TARGETS[@]}}")
    ;;
  *)
    echo "Usage: $0 <linux|ibm>" >&2
    exit 2
    ;;
esac

if ! command -v zig >/dev/null 2>&1; then
  echo "$label targets ${DIM}(skipped)${RESET}"
  skip "zig not installed" "brew install zig"
  exit 0
fi

if [[ ${#targets[@]} -eq 0 ]]; then
  skip "no $label targets configured" ".config/target-matrix.json"
  exit 0
fi

export ZIG_CC="$SCRIPT_DIR/zig-cc.sh"
NIGHTLY_TOOLCHAIN=$("$SCRIPT_DIR/../lib/toolchain.sh" --nightly)

uses_nightly() {
  [[ "$group" == ibm || "$1" == riscv64* ]]
}

for target in "${targets[@]}"; do
  if uses_nightly "$target"; then
    ensure_target "$target" "$NIGHTLY_TOOLCHAIN"
  else
    ensure_target "$target"
  fi
  mkdir -p "target/cross-check/$target"
done

LOG_DIR=$(mktemp -d)
trap 'rm -rf "$LOG_DIR"' EXIT
pids=()
logs=()

echo "$label targets ${DIM}(rscrypto)${RESET}"
for i in "${!targets[@]}"; do
  target=${targets[$i]}
  log_file="$LOG_DIR/$target.log"
  logs[i]=$log_file

  (
    toolchain_env=(env)
    if uses_nightly "$target"; then
      toolchain_env+=("RUSTUP_TOOLCHAIN=$NIGHTLY_TOOLCHAIN")
    fi
    if ! CC="$ZIG_CC" CARGO_TARGET_DIR="target/cross-check/$target" \
      "${toolchain_env[@]}" cargo clippy -p rscrypto --lib --all-features --locked \
      --target "$target" >"$log_file" 2>&1; then
      exit 1
    fi
  ) &
  pids[i]=$!
done

failed=0
for i in "${!targets[@]}"; do
  target=${targets[$i]}
  step "${target/unknown-linux-/} clippy"
  if wait "${pids[$i]}"; then
    ok
  else
    fail
    show_error "${logs[$i]}"
    failed=1
  fi
done

[[ "$failed" -eq 0 ]] || exit 1
echo "${GREEN}✓${RESET} $label targets passed"
