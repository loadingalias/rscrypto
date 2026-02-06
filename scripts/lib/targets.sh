#!/usr/bin/env bash
# Target definitions for cross-platform checks.
#
# Single source of truth: config/target-matrix.toml

TARGETS_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_MATRIX_PY="$TARGETS_LIB_DIR/target-matrix.py"

if [[ ! -x "$TARGET_MATRIX_PY" ]]; then
  echo "ERROR: target matrix loader not found: $TARGET_MATRIX_PY" >&2
  exit 1
fi

eval "$("$TARGET_MATRIX_PY" --format shell)"
