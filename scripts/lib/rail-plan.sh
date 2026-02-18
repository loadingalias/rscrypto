#!/usr/bin/env bash
# cargo-rail v0.10+ plan helpers for repository scripts.

rail_since_args() {
  if [[ -n "${RAIL_SINCE:-}" ]]; then
    printf -- "--since %s" "$RAIL_SINCE"
  else
    printf -- "--merge-base"
  fi
}

rail_plan_json() {
  local since_arg
  since_arg="$(rail_since_args)"
  # shellcheck disable=SC2086
  cargo rail plan $since_arg --json 2>/dev/null || echo ""
}

rail_plan_crates() {
  local plan_output
  plan_output="$(rail_plan_json)"

  if [[ -z "$plan_output" ]]; then
    echo ""
    return 0
  fi

  echo "$plan_output" | jq -r '
    ((.impact.direct_crates // []) + (.impact.transitive_crates // []))
    | unique
    | .[]
  ' 2>/dev/null | tr -d '\r' | tr '\n' ' ' | xargs || echo ""
}
