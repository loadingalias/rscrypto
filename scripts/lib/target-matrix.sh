#!/usr/bin/env bash
# Read .config/target-matrix.json and emit shell/json views for CI/scripts.
#
# Single source of truth: .config/target-matrix.json
#
# JSON keys:
#   ci      CI host matrix (linux via runs-on, windows via GHA). Injects runs-on
#           routing with the current github run_id so runs-on.com pools route correctly.
#   no_std  Bare-metal no_std targets.
#   wasm    WASM targets.
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

usage() {
  echo "Usage: $0 --validate | --format {shell|json} [--key KEY]" >&2
  exit 1
}

FORMAT=""
KEY=""
VALIDATE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --validate) VALIDATE=true; shift ;;
    --format) FORMAT="$2"; shift 2 ;;
    --key)    KEY="$2";    shift 2 ;;
    *)        usage ;;
  esac
done

if [[ "$VALIDATE" == true && -n "$FORMAT" ]]; then
  usage
fi

validate_manifest() {
  jq -e '
    . as $root |
    ($root.groups | keys) == ["ibm", "linux", "macos", "no_std", "wasm", "win"] and
    all($root.groups[]; type == "array" and length > 0 and . == sort and all(.[]; type == "string" and length > 0)) and
    ([$root.groups[][]] | length) == ([$root.groups[][]] | unique | length) and
    ($root.ci | type == "array" and length > 0) and
    ($root.ci == ($root.ci | sort_by(.name))) and
    ([$root.ci[].name] | length) == ([$root.ci[].name] | unique | length) and
    all($root.ci[];
      .name as $name |
      any($root.groups[][]; . == $name) and
      if .type == "runson" then
        (keys | sort) == ["name", "pool", "type"]
      elif .type == "gha" then
        (keys | sort) == ["name", "runner", "type"]
      else
        false
      end
    )
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

if [[ "$VALIDATE" == true ]]; then
  exit 0
fi

[[ -n "$FORMAT" ]] || usage

print_shell() {
  jq -r '
    def bash_array(name; arr):
      name + "=(" + ([arr[] | "'"'"'" + . + "'"'"'"] | join(" ")) + ")";

    bash_array("WIN_TARGETS"; .groups.win),
    bash_array("LINUX_TARGETS"; .groups.linux),
    bash_array("IBM_TARGETS"; (.groups.ibm // [])),
    bash_array("NOSTD_TARGETS"; .groups.no_std),
    bash_array("WASM_TARGETS"; .groups.wasm)
  ' "$MANIFEST"
}

get_json() {
  local key="$1"
  local run_id="${GITHUB_RUN_ID:-${GH_RUN_ID:-0}}"

  case "$key" in
    ci)
      # Inject runs-on routing for type=runson rows; leave type=gha rows as-is.
      jq -c --arg run_id "$run_id" '
        .ci | map(
          if .type == "runson" then
            .runner = "runs-on=" + $run_id + "/runner=" + .pool
          else
            .
          end
        )
      ' "$MANIFEST"
      ;;
    no_std) jq -c '.groups.no_std' "$MANIFEST" ;;
    wasm)   jq -c '.groups.wasm'   "$MANIFEST" ;;
    *)
      echo "ERROR: unknown json key: $key (supported: ci, no_std, wasm)" >&2
      exit 1
      ;;
  esac
}

case "$FORMAT" in
  shell)
    print_shell
    ;;
  json)
    [[ -n "$KEY" ]] || { echo "ERROR: --key is required for --format json" >&2; exit 1; }
    get_json "$KEY"
    ;;
  *)
    usage
    ;;
esac
