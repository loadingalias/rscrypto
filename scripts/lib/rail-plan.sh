#!/usr/bin/env bash
# Strict Cargo Rail v8 plan helpers for repository scripts.

RAIL_PLAN_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_rail_load_plan() {
  if [[ -n "${RAIL_PLAN_JSON_CACHE:-}" && "${RAIL_PLAN_JSON_CACHE_VALIDATED:-false}" == true ]]; then
    return 0
  fi
  if [[ "${RAIL_PLAN_LOAD_ATTEMPTED:-false}" == true ]]; then
    return 1
  fi
  RAIL_PLAN_LOAD_ATTEMPTED=true

  local plan_output
  if [[ -n "${RAIL_PLAN_FILE:-}" || -n "${RAIL_PLAN_READER:-}" ]]; then
    [[ -n "${RAIL_PLAN_FILE:-}" && -n "${RAIL_PLAN_READER:-}" ]] || return 1
    [[ -f "$RAIL_PLAN_FILE" && -f "$RAIL_PLAN_READER" ]] || return 1
    local python
    python="$("$RAIL_PLAN_LIB_DIR/python.sh" --print)" || return 1
    "$python" "$RAIL_PLAN_READER" validate "$RAIL_PLAN_FILE" >/dev/null || return 1
    if [[ "${RAIL_PLAN_CHECKOUT_VERIFIED:-false}" != true ]]; then
      "$python" "$RAIL_PLAN_READER" verify-checkout "$RAIL_PLAN_FILE" >/dev/null || return 1
    fi
    plan_output=$(<"$RAIL_PLAN_FILE")
  else
    local plan_args=(rail plan --quiet --json)
    if [[ -n "${RAIL_SINCE:-}" ]]; then
      plan_args+=(--since "$RAIL_SINCE")
    fi
    plan_output=$(cargo "${plan_args[@]}" 2>/dev/null) || return 1

    local plan_file
    plan_file=$(mktemp)
    printf '%s\n' "$plan_output" >"$plan_file"
    if ! cargo rail plan --verify "$plan_file" >/dev/null 2>&1; then
      rm -f "$plan_file"
      return 1
    fi
    rm -f "$plan_file"
  fi

  jq -e '.plan_contract_version == 8' <<<"$plan_output" >/dev/null 2>&1 || return 1
  RAIL_PLAN_JSON_CACHE=$plan_output
  RAIL_PLAN_JSON_CACHE_VALIDATED=true
}

rail_prime_plan() {
  _rail_load_plan
}

rail_scope_json() {
  local work_id=${1:-${RAIL_WORK_ID:-cargo.build}}
  _rail_load_plan || return 1

  jq -ce --arg work_id "$work_id" '
    .work[$work_id] as $decision
    | if $decision.state == "skipped" then
        {mode: "empty", cargo_args: []}
      elif $decision.state == "required"
        and $decision.scope.kind == "cargo"
        and $decision.scope.selection.kind == "workspace" then
        {mode: "workspace", cargo_args: ["--workspace"]}
      elif $decision.state == "required"
        and $decision.scope.kind == "cargo"
        and $decision.scope.selection.kind == "packages" then
        {
          mode: "packages",
          cargo_args: $decision.scope.selection.cargo_args
        }
      else
        empty
      end
  ' <<<"$RAIL_PLAN_JSON_CACHE"
}

rail_scope_mode() {
  local scope_output
  if ! scope_output=$(rail_scope_json "${1:-}"); then
    echo workspace
    return 0
  fi
  jq -r '.mode' <<<"$scope_output"
}

rail_scope_cargo_args() {
  local scope_output
  if ! scope_output=$(rail_scope_json "${1:-}"); then
    return 0
  fi
  jq -r 'select(.mode != "empty") | .cargo_args[]' <<<"$scope_output"
}
