#!/usr/bin/env bash
# cargo-rail scope helpers for repository scripts.

rail_since_args() {
  if [[ -n "${RAIL_SINCE:-}" ]]; then
    printf -- "--since %s" "$RAIL_SINCE"
  else
    printf -- "--merge-base"
  fi
}

rail_plan_json() {
  if [[ -n "${RAIL_PLAN_JSON_CACHE:-}" ]]; then
    printf '%s\n' "$RAIL_PLAN_JSON_CACHE"
    return 0
  fi

  local since_arg
  since_arg="$(rail_since_args)"

  local plan_output
  # shellcheck disable=SC2086
  plan_output="$(cargo rail plan $since_arg --quiet --json 2>/dev/null || true)"
  RAIL_PLAN_JSON_CACHE="$plan_output"
  printf '%s\n' "$RAIL_PLAN_JSON_CACHE"
}

rail_scope_json() {
  if [[ -n "${RAIL_SCOPE_JSON:-}" ]]; then
    printf '%s\n' "$RAIL_SCOPE_JSON"
    return 0
  fi

  if [[ -n "${RAIL_SCOPE_JSON_CACHE:-}" ]]; then
    printf '%s\n' "$RAIL_SCOPE_JSON_CACHE"
    return 0
  fi

  local plan_output
  plan_output="$(rail_plan_json)"
  if [[ -z "$plan_output" ]]; then
    echo ""
    return 0
  fi

  RAIL_SCOPE_JSON_CACHE="$(echo "$plan_output" | jq -c '.scope // empty' 2>/dev/null || true)"
  printf '%s\n' "$RAIL_SCOPE_JSON_CACHE"
}

rail_scope_mode() {
  local scope_output
  scope_output="$(rail_scope_json)"

  if [[ -z "$scope_output" ]]; then
    echo ""
    return 0
  fi

  echo "$scope_output" | jq -r '.mode // empty' 2>/dev/null || echo ""
}

rail_scope_surface_enabled() {
  local surface=$1
  local scope_output
  scope_output="$(rail_scope_json)"

  if [[ -z "$scope_output" ]]; then
    return 1
  fi

  echo "$scope_output" | jq -e --arg surface "$surface" '.surfaces[$surface] == true' >/dev/null 2>&1
}

rail_plan_crates() {
  local scope_output
  scope_output="$(rail_scope_json)"

  if [[ -z "$scope_output" ]]; then
    echo ""
    return 0
  fi

  echo "$scope_output" | jq -r '
    if .mode == "crates" then
      (.crates // [])[]
    else
      empty
    end
  ' 2>/dev/null | tr -d '\r' | tr '\n' ' ' | xargs || echo ""
}
