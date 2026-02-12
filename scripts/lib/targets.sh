#!/usr/bin/env bash
# Target definitions for cross-platform checks.
#
# Single source of truth: config/target-matrix.toml

TARGETS_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_MATRIX_PY="$TARGETS_LIB_DIR/target-matrix.py"

# Always define arrays up front so callers using `set -u` never trip on
# unbound vars when target matrix loading fails.
declare -a WIN_TARGETS=()
declare -a LINUX_TARGETS=()
declare -a IBM_TARGETS=()
declare -a NOSTD_TARGETS=()
declare -a WASM_TARGETS=()

if [[ ! -x "$TARGET_MATRIX_PY" ]]; then
  echo "ERROR: target matrix loader not found: $TARGET_MATRIX_PY" >&2
  exit 1
fi

pick_matrix_python() {
  local candidates=(
    "${PYTHON_BIN:-}"
    "$(command -v python3.13 2>/dev/null || true)"
    "/opt/homebrew/bin/python3.13"
    "$(command -v python3.12 2>/dev/null || true)"
    "$(command -v python3.11 2>/dev/null || true)"
    "$(command -v python3 2>/dev/null || true)"
  )
  local py
  for py in "${candidates[@]}"; do
    [[ -n "$py" ]] || continue
    [[ -x "$py" ]] || continue
    if "$py" -c 'import sys; ok=False
try:
  import tomllib
  ok=True
except ModuleNotFoundError:
  try:
    import tomli  # noqa: F401
    ok=True
  except ModuleNotFoundError:
    ok=False
sys.exit(0 if ok else 1)' >/dev/null 2>&1; then
      echo "$py"
      return 0
    fi
  done
  return 1
}

if ! MATRIX_PYTHON="$(pick_matrix_python)"; then
  echo "ERROR: no usable Python found for target matrix loader." >&2
  echo "Need Python with tomllib (3.11+) or tomli installed." >&2
  echo "Try: brew install python@3.13" >&2
  exit 1
fi

if ! matrix_shell="$("$MATRIX_PYTHON" "$TARGET_MATRIX_PY" --format shell 2>&1)"; then
  echo "ERROR: failed to load target matrix from $TARGET_MATRIX_PY" >&2
  echo "$matrix_shell" >&2
  exit 1
fi
eval "$matrix_shell"
