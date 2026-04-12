#!/usr/bin/env bash
# Read .config/target-matrix.json and emit shell/json views for CI/scripts.
#
# Single source of truth: .config/target-matrix.json
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
  echo "Usage: $0 --format {shell|json} [--key KEY]" >&2
  exit 1
}

FORMAT=""
KEY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --format) FORMAT="$2"; shift 2 ;;
    --key)    KEY="$2";    shift 2 ;;
    *)        usage ;;
  esac
done

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
  case "$key" in
    commit_ci)     jq -c '.ci.commit'    "$MANIFEST" ;;
    weekly_ci)     jq -c '.ci.weekly // .ci.commit' "$MANIFEST" ;;
    commit_no_std | weekly_no_std) jq -c '.groups.no_std' "$MANIFEST" ;;
    commit_wasm   | weekly_wasm)   jq -c '.groups.wasm'   "$MANIFEST" ;;
    *)
      echo "ERROR: unknown json key: $key" >&2
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
