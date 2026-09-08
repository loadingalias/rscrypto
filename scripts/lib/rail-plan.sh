#!/usr/bin/env bash
# One fail-closed Cargo Rail v8 plan consumer for repository scripts.
# shellcheck disable=SC2034
# CARGO_ARGS, CARGO_SCOPE_KIND, and SCOPE_DESC are outputs for test.sh.

_rail_load_plan() {
  if [[ "${RAIL_PLAN_LOADED:-false}" == true ]]; then
    return 0
  fi
  if [[ "${RAIL_PLAN_LOAD_ATTEMPTED:-false}" == true ]]; then
    echo "Cargo Rail plan loading already failed" >&2
    return 2
  fi
  RAIL_PLAN_LOAD_ATTEMPTED=true

  local plan_args=(rail plan --quiet --json)
  if [[ -n "${RAIL_SINCE:-}" ]]; then
    plan_args+=(--since "$RAIL_SINCE")
  fi
  if [[ "${RAIL_ALL:-false}" == true ]]; then
    plan_args+=(--all)
  fi
  RAIL_PLAN_JSON_CACHE=$(cargo "${plan_args[@]}") || return 2

  local plan_file
  plan_file=$(mktemp "${TMPDIR:-/tmp}/rscrypto-plan-v8.XXXXXX")
  printf '%s\n' "$RAIL_PLAN_JSON_CACHE" >"$plan_file"
  if ! cargo rail plan --verify "$plan_file"; then
    rm -f "$plan_file"
    return 2
  fi
  rm -f "$plan_file"

  jq -e '
    .plan_contract_version == 8
    and (.identity | type == "string" and startswith("plan-v8:sha256:"))
    and (.required | type == "array")
    and (.work | type == "object")
  ' <<<"$RAIL_PLAN_JSON_CACHE" >/dev/null || {
    echo "Cargo Rail emitted an incompatible plan" >&2
    return 2
  }

  RAIL_PLAN_LOADED=true
}

rail_prime_plan() {
  _rail_load_plan
}

rail_scope_json() {
  local work_id=${1:-}
  [[ -n "$work_id" ]] || {
    echo "Cargo Rail work ID is required" >&2
    return 2
  }
  _rail_load_plan || return 2

  jq -ce --arg work_id "$work_id" '
    .work[$work_id] as $decision
    | if $decision == null then
        error("unknown work ID")
      elif $decision.state == "skipped" then
        {mode: "empty", cargo_args: []}
      elif $decision.state == "required"
        and $decision.scope.kind == "cargo"
        and ($decision.scope.selection.kind == "workspace" or $decision.scope.selection.kind == "packages")
        and ($decision.scope.selection.cargo_args | type == "array") then
        {
          mode: $decision.scope.selection.kind,
          cargo_args: $decision.scope.selection.cargo_args
        }
      else
        error("work item does not carry Cargo scope")
      end
  ' <<<"$RAIL_PLAN_JSON_CACHE"
}

rail_scope_mode() {
  _rail_load_plan || return 2
  local scope_output
  scope_output=$(rail_scope_json "$1") || return 2
  jq -r '.mode' <<<"$scope_output"
}

rail_scope_cargo_args() {
  local work_id=$1
  _rail_load_plan || return 2

  rail_scope_json "$work_id" | jq -j '.cargo_args[] | ., "\u0000"'
}

# Select the exact Cargo arguments from one Cargo Rail work decision.
# Usage: select_cargo_scope WORK_ID [true]
# Returns 1 only when Cargo Rail selected no work.
select_cargo_scope() {
  local work_id=$1
  local force_all=${2:-false}
  local arg args_file

  CARGO_ARGS=()
  CARGO_SCOPE_KIND=""
  SCOPE_DESC=""

  if [[ "$force_all" == true ]]; then
    CARGO_ARGS=(--workspace)
    CARGO_SCOPE_KIND=workspace
    SCOPE_DESC=workspace
    return 0
  fi

  # Prime in the caller shell so subsequent process substitutions consume the
  # same verified plan instead of replanning in isolated subshells.
  rail_prime_plan || return 2
  CARGO_SCOPE_KIND="$(rail_scope_mode "$work_id")" || return 2

  case "$CARGO_SCOPE_KIND" in
    empty)
      SCOPE_DESC="no changes"
      return 1
      ;;
    workspace)
      SCOPE_DESC="workspace (Cargo Rail)"
      ;;
    packages)
      args_file=$(mktemp "${TMPDIR:-/tmp}/rscrypto-cargo-args.XXXXXX")
      if ! rail_scope_cargo_args "$work_id" >"$args_file"; then
        rm -f "$args_file"
        return 2
      fi
      while IFS= read -r -d '' arg; do
        CARGO_ARGS+=("$arg")
      done <"$args_file"
      rm -f "$args_file"
      if [[ ${#CARGO_ARGS[@]} -eq 0 ]]; then
        echo "ERROR: Cargo Rail selected packages without Cargo arguments for $work_id" >&2
        return 2
      fi
      SCOPE_DESC="affected packages (Cargo Rail)"
      ;;
    *)
      echo "ERROR: unsupported Cargo Rail scope '$CARGO_SCOPE_KIND' for $work_id" >&2
      return 2
      ;;
  esac
}
