#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$("$SCRIPT_DIR/../lib/python.sh" --print)"
PLAN_FILE=${1:?plan file is required}
PLAN_READER=${2:?plan reader is required}
GITHUB_OUTPUT_FILE=${3:?GitHub output file is required}
CATALOG=${4:-.config/ci-plan-variants.json}

"$PYTHON" "$PLAN_READER" validate "$PLAN_FILE"
policy_required=$("$PYTHON" "$PLAN_READER" is-required "$PLAN_FILE" ci-policy)
if [[ "$policy_required" == true ]]; then
  matrix=$(jq -c '{include: [.variants[] | {work: ({id} + .dimensions)}]}' "$CATALOG")
else
  suite_required=$("$PYTHON" "$PLAN_READER" is-required "$PLAN_FILE" ci-suite)
  if [[ "$suite_required" != true ]]; then
    matrix='{"include":[]}'
  else
    matrix=$("$PYTHON" "$PLAN_READER" matrix "$PLAN_FILE" ci-suite)
    if [[ "$matrix" == all ]]; then
      matrix=$(jq -c '{include: [.variants[] | {work: ({id} + .dimensions)}]}' "$CATALOG")
    else
      matrix=$(jq -ce '{include: [.include[] | {work: .}]}' <<<"$matrix")
    fi
  fi
fi

jq -e '
  .include | type == "array" and all(.[].work;
    (.id | type == "string" and length > 0)
    and (.display_name | type == "string" and length > 0)
    and (.operation | type == "string" and length > 0)
    and (.runner | type == "string" and length > 0)
    and (.runner_type == "github" or .runner_type == "runson")
    and (.timeout_minutes | type == "number" and floor == . and . > 0)
    and (.tools_mode | type == "string")
    and (.toolchain_contract | type == "string")
    and (.toolchain_components | type == "string")
  )
' <<<"$matrix" >/dev/null

has_suite=$(jq -r '.include | length > 0' <<<"$matrix")
printf 'matrix=%s\nhas_suite=%s\n' "$matrix" "$has_suite" >>"$GITHUB_OUTPUT_FILE"
