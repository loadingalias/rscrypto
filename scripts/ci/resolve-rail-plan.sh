#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/rail-plan.sh
source "$SCRIPT_DIR/../lib/rail-plan.sh"

OUTPUT_PATH=${GITHUB_OUTPUT:?GITHUB_OUTPUT is required}

write_outputs() {
  local valid=$1
  local empty=$2
  local build=$3
  local test=$4
  local infra=$5
  local cargo_graph=$6

  {
    printf 'valid=%s\n' "$valid"
    printf 'empty=%s\n' "$empty"
    printf 'build=%s\n' "$build"
    printf 'test=%s\n' "$test"
    printf 'infra=%s\n' "$infra"
    printf 'cargo_graph=%s\n' "$cargo_graph"
  } >>"$OUTPUT_PATH"
}

if [[ "${RAIL_PLAN_STEP_OUTCOME:-}" != "success" ]]; then
  write_outputs false false true true true true
  exit 0
fi

if ! scope_json="$(rail_scope_json)"; then
  write_outputs false false true true true true
  exit 0
fi

mode="$(jq -r '.mode' <<<"$scope_json")"
build="$(jq -r '.surfaces.build' <<<"$scope_json")"
infra="$(jq -r '.surfaces.infra' <<<"$scope_json")"
cargo_graph="$(jq -r '.surfaces["custom:cargo_graph"]' <<<"$scope_json")"

if [[ "$mode" == "empty" ]]; then
  write_outputs true true "$build" false "$infra" "$cargo_graph"
else
  write_outputs true false "$build" true "$infra" "$cargo_graph"
fi
