#!/usr/bin/env bash
# Validate the Cargo Rail platform catalog and its target projections.
#
# Single source of truth: .config/target-matrix.json
#
# Every row is one independently selectable proof unit. The ordinary Linux
# x86-64 row is executed by the core job; AMX is a second proof for that target.
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
    ($root | keys) == ["variant_catalog_version", "variants", "work"] and
    $root.variant_catalog_version == 2 and
    $root.work == "targets.platforms" and
    ($root.variants | type == "array" and length == 18) and
    ([$root.variants[].id] == ([$root.variants[].id] | sort | unique)) and
    all($root.variants[];
      (. | keys | sort) == (["config", "dimensions", "external_paths", "id"] | sort) and
      (.id | type == "string" and test("^[a-z][a-z0-9.-]*$")) and
      (.external_paths | type == "array" and length > 0 and . == sort and length == (unique | length)) and
      (.config == ["targets"]) and
      (.dimensions | keys | sort) == ([
        "cache", "compile", "components", "constant_time", "contract", "group",
        "name", "operation", "performance", "platform", "release", "runner",
        "runner_type", "runtime", "target", "timeout_minutes", "verify_plan"
      ] | sort) and
      (.dimensions.group | IN("ibm", "linux", "macos", "no_std", "wasm", "win")) and
      (.dimensions.operation | IN("amx", "core", "cross", "native")) and
      (.dimensions.runner_type | IN("github", "runson")) and
      (.dimensions.contract | IN("development", "nightly")) and
      (.dimensions.compile | IN("core", "generic", "hosted", "native")) and
      (.dimensions.runtime | IN("emulated", "none", "physical-native", "virtual-native")) and
      (.dimensions.timeout_minutes | type == "number" and . >= 0) and
      (.dimensions.cache | type == "boolean") and
      (.dimensions.constant_time | type == "boolean") and
      (.dimensions.performance | type == "boolean") and
      (.dimensions.release == true) and
      (.dimensions.verify_plan | type == "boolean")
    ) and
    ([$root.variants[] | select(.dimensions.operation != "amx") | .dimensions.target] | length) == 17 and
    ([$root.variants[] | .dimensions.target] | unique | length) == 17 and
    ([$root.variants[] | select(.dimensions.operation == "core")] | length) == 1 and
    ([$root.variants[] | select(.dimensions.operation == "amx")] | length) == 1 and
    ($root.variants[] | select(.id == "aarch64-pc-windows-msvc") | .dimensions.runtime) == "none" and
    ($root.variants[] | select(.id == "x86-64-pc-windows-msvc") | .dimensions.runtime) == "virtual-native" and
    ($root.variants[] | select(.id == "wasm32-wasip1") | .dimensions.runtime) == "emulated" and
    all($root.variants[] | select(.dimensions.verify_plan); .dimensions.runner_type == "runson")
  ' "$MANIFEST" >/dev/null || {
    echo "ERROR: invalid target matrix schema: $MANIFEST" >&2
    return 1
  }

  local matrix_targets
  matrix_targets="$(jq -r '[.variants[].dimensions.target] | unique[]' "$MANIFEST" | LC_ALL=C sort)"
  matrix_targets=${matrix_targets//$'\r'/}

  local projection
  for projection in "$REPO_ROOT/.config/rail.toml" "$REPO_ROOT/deny.toml"; do
    local projected_targets
    projected_targets="$(awk '
      { sub(/\r$/, "") }
      /^targets = \[$/ { in_targets = 1; next }
      in_targets && /^\]$/ { exit }
      in_targets && match($0, /"[^"]+"/) {
        print substr($0, RSTART + 1, RLENGTH - 2)
      }
    ' "$projection" | LC_ALL=C sort)"
    projected_targets=${projected_targets//$'\r'/}
    if [[ "$projected_targets" != "$matrix_targets" ]]; then
      echo "ERROR: target projection does not match .config/target-matrix.json: $projection" >&2
      diff -u <(printf '%s\n' "$matrix_targets") <(printf '%s\n' "$projected_targets") >&2 || true
      return 1
    fi
  done
}

validate_manifest
