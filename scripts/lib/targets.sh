#!/usr/bin/env bash
# Target definitions for cross-platform checks.
#
# Single source of truth: .config/target-matrix.json

TARGETS_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_MATRIX_SH="$TARGETS_LIB_DIR/target-matrix.sh"

# Always define arrays up front so callers using `set -u` never trip on
# unbound vars when target matrix loading fails.
declare -a WIN_TARGETS=()
declare -a LINUX_TARGETS=()
declare -a IBM_TARGETS=()
declare -a NOSTD_TARGETS=()
declare -a WASM_TARGETS=()

if [[ ! -x "$TARGET_MATRIX_SH" ]]; then
  echo "ERROR: target matrix loader not found: $TARGET_MATRIX_SH" >&2
  exit 1
fi

if ! matrix_shell="$(bash "$TARGET_MATRIX_SH" --format shell 2>&1)"; then
  echo "ERROR: failed to load target matrix from $TARGET_MATRIX_SH" >&2
  echo "$matrix_shell" >&2
  exit 1
fi
eval "$matrix_shell"
