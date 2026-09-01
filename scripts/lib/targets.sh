#!/usr/bin/env bash
# Target definitions for cross-platform checks.
#
# Single source of truth: .config/target-matrix.json
# shellcheck disable=SC2034
# Target arrays are caller-visible outputs for sourced check scripts.

TARGETS_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_MATRIX_SH="$TARGETS_LIB_DIR/target-matrix.sh"
TARGET_MATRIX_JSON="$TARGETS_LIB_DIR/../../.config/target-matrix.json"

# Always define arrays up front so callers using `set -u` never trip on
# unbound vars when target matrix loading fails.
declare -a LINUX_TARGETS=()
declare -a NOSTD_TARGETS=()
declare -a WASM_TARGETS=()

if [[ ! -x "$TARGET_MATRIX_SH" ]]; then
  echo "ERROR: target matrix loader not found: $TARGET_MATRIX_SH" >&2
  exit 1
fi

"$TARGET_MATRIX_SH" --validate

load_target_group() {
  local group=$1
  local target
  while IFS= read -r target; do
    case "$group" in
      linux) LINUX_TARGETS+=("$target") ;;
      no_std) NOSTD_TARGETS+=("$target") ;;
      wasm) WASM_TARGETS+=("$target") ;;
    esac
  done < <(
    jq -r --arg group "$group" '
      [.variants[]
        | select(.dimensions.group == $group and .dimensions.operation != "amx")
        | .dimensions.target]
      | unique[]
    ' "$TARGET_MATRIX_JSON"
  )
}

load_target_group linux
load_target_group no_std
load_target_group wasm
