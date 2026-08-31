#!/usr/bin/env bash
# Validate .config/target-matrix.json and its generated projections.
#
# Single source of truth: .config/target-matrix.json
#
# `.config/ci-plan-variants.json` owns CI runner rows and Cargo Rail selection.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MANIFEST="$REPO_ROOT/.config/target-matrix.json"

if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: target matrix manifest not found: $MANIFEST" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq is required but not found on PATH" >&2
  exit 1
fi

[[ $# -eq 1 && "$1" == --validate ]] \
  || { echo "Usage: $0 --validate" >&2; exit 2; }

validate_manifest() {
  jq -e '
    . as $root |
    ($root.groups | keys) == ["ibm", "linux", "macos", "no_std", "wasm", "win"] and
    all($root.groups[]; type == "array" and length > 0 and . == sort and all(.[]; type == "string" and length > 0)) and
    ([$root.groups[][]] | length) == ([$root.groups[][]] | unique | length) and
    ($root | keys) == ["groups"]
  ' "$MANIFEST" >/dev/null || {
    echo "ERROR: invalid target matrix schema: $MANIFEST" >&2
    return 1
  }

  local matrix_targets
  matrix_targets="$(jq -r '.groups[][]' "$MANIFEST" | LC_ALL=C sort)"

  local projection
  for projection in "$REPO_ROOT/.config/rail.toml" "$REPO_ROOT/deny.toml"; do
    local projected_targets
    projected_targets="$(awk '
      /^targets = \[$/ { in_targets = 1; next }
      in_targets && /^\]$/ { exit }
      in_targets && match($0, /"[^"]+"/) {
        print substr($0, RSTART + 1, RLENGTH - 2)
      }
    ' "$projection" | LC_ALL=C sort)"
    if [[ "$projected_targets" != "$matrix_targets" ]]; then
      echo "ERROR: target projection does not match .config/target-matrix.json: $projection" >&2
      diff -u <(printf '%s\n' "$matrix_targets") <(printf '%s\n' "$projected_targets") >&2 || true
      return 1
    fi
  done
}

validate_manifest
