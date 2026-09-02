#!/usr/bin/env bash
# One fail-closed Cargo Rail v8 plan consumer for repository scripts.

RAIL_PLAN_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_rail_load_plan() {
  if [[ "${RAIL_PLAN_LOADED:-false}" == true ]]; then
    return 0
  fi
  if [[ "${RAIL_PLAN_LOAD_ATTEMPTED:-false}" == true ]]; then
    echo "Cargo Rail plan loading already failed" >&2
    return 2
  fi
  RAIL_PLAN_LOAD_ATTEMPTED=true

  local python
  python="$("$RAIL_PLAN_LIB_DIR/python.sh" --print)" || return 2

  if [[ -n "${RAIL_PLAN_FILE:-}" || -n "${RAIL_PLAN_READER:-}" ]]; then
    [[ -n "${RAIL_PLAN_FILE:-}" && -f "$RAIL_PLAN_FILE" ]] || {
      echo "RAIL_PLAN_FILE must name a saved plan" >&2
      return 2
    }
    if [[ -n "${RAIL_PLAN_READER:-}" ]]; then
      [[ -f "$RAIL_PLAN_READER" ]] || {
        echo "RAIL_PLAN_READER must name the matching strict reader" >&2
        return 2
      }
      "$python" "$RAIL_PLAN_READER" validate "$RAIL_PLAN_FILE" || return 2
      "$python" "$RAIL_PLAN_READER" verify-checkout "$RAIL_PLAN_FILE" || return 2
      RAIL_PLAN_USE_READER=true
    elif [[ "${RAIL_PLAN_LOCAL:-false}" == true ]]; then
      cargo rail plan --verify "$RAIL_PLAN_FILE" || return 2
      RAIL_PLAN_USE_READER=false
    else
      echo "A transported plan requires its matching RAIL_PLAN_READER" >&2
      return 2
    fi
    RAIL_PLAN_JSON_CACHE=$(<"$RAIL_PLAN_FILE")
  else
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
    RAIL_PLAN_USE_READER=false
  fi

  jq -e '
    .plan_contract_version == 8
    and (.identity | type == "string" and startswith("plan-v8:sha256:"))
    and (.required | type == "array")
    and (.work | type == "object")
  ' <<<"$RAIL_PLAN_JSON_CACHE" >/dev/null || {
    echo "Cargo Rail emitted an incompatible plan" >&2
    return 2
  }

  if [[ -n "${RAIL_PLAN_IDENTITY:-}" ]]; then
    local actual_identity
    actual_identity=$(jq -r '.identity' <<<"$RAIL_PLAN_JSON_CACHE")
    [[ "$actual_identity" == "$RAIL_PLAN_IDENTITY" ]] || {
      echo "Cargo Rail plan identity mismatch" >&2
      return 2
    }
  fi
  if [[ -n "${RAIL_PLAN_HEAD_COMMIT:-}" ]]; then
    local actual_head
    actual_head=$(jq -r '.inputs.head_commit' <<<"$RAIL_PLAN_JSON_CACHE")
    [[ "$actual_head" == "$RAIL_PLAN_HEAD_COMMIT" ]] || {
      echo "Cargo Rail plan checkout mismatch" >&2
      return 2
    }
  fi

  RAIL_PLAN_LOADED=true
}

rail_prime_plan() {
  _rail_load_plan
}

rail_work_required() {
  local work_id=$1
  _rail_load_plan || return 2

  if [[ "$RAIL_PLAN_USE_READER" == true ]]; then
    local python required status=0
    python="$("$RAIL_PLAN_LIB_DIR/python.sh" --print)" || return 2
    required=$("$python" "$RAIL_PLAN_READER" is-required "$RAIL_PLAN_FILE" "$work_id") || status=$?
    [[ "$status" -eq 0 ]] || return 2
    case "$required" in
      true) return 0 ;;
      false) return 1 ;;
      *)
        echo "Cargo Rail reader emitted an invalid required-work decision" >&2
        return 2
        ;;
    esac
  fi

  local status=0
  jq -e --arg work_id "$work_id" '
    .work[$work_id] as $decision
    | if $decision == null then error("unknown work ID") else $decision.state == "required" end
  ' <<<"$RAIL_PLAN_JSON_CACHE" >/dev/null || status=$?
  case "$status" in
    0) return 0 ;;
    1) return 1 ;;
    *) return 2 ;;
  esac
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
  if [[ "$RAIL_PLAN_USE_READER" == true ]]; then
    local python mode status=0
    python="$("$RAIL_PLAN_LIB_DIR/python.sh" --print)" || return 2
    mode=$("$python" "$RAIL_PLAN_READER" cargo-scope "$RAIL_PLAN_FILE" "$1") || status=$?
    [[ "$status" -eq 0 ]] || return 2
    case "$mode" in
      skipped) printf 'empty\n' ;;
      workspace | packages) printf '%s\n' "$mode" ;;
      *)
        echo "Cargo Rail reader emitted an invalid Cargo scope" >&2
        return 2
        ;;
    esac
    return
  fi

  local scope_output
  scope_output=$(rail_scope_json "$1") || return 2
  jq -r '.mode' <<<"$scope_output"
}

rail_scope_cargo_args() {
  local work_id=$1
  _rail_load_plan || return 2

  if [[ "$RAIL_PLAN_USE_READER" == true ]]; then
    local python
    python="$("$RAIL_PLAN_LIB_DIR/python.sh" --print)" || return 2
    "$python" "$RAIL_PLAN_READER" cargo-args "$RAIL_PLAN_FILE" "$work_id"
    return
  fi

  rail_scope_json "$work_id" | jq -j '.cargo_args[] | ., "\u0000"'
}

rail_variant_matrix() {
  local work_id=${1:-}
  [[ -n "$work_id" ]] || {
    echo "Cargo Rail work ID is required" >&2
    return 2
  }
  _rail_load_plan || return 2

  if [[ "$RAIL_PLAN_USE_READER" == true ]]; then
    local python
    python="$("$RAIL_PLAN_LIB_DIR/python.sh" --print)" || return 2
    "$python" "$RAIL_PLAN_READER" matrix "$RAIL_PLAN_FILE" "$work_id"
    return
  fi

  jq -cer --arg work_id "$work_id" '
    .work[$work_id] as $decision
    | if $decision == null then
        error("unknown work ID")
      elif $decision.state == "skipped" then
        {include: []}
      elif $decision.state != "required" or $decision.scope.kind != "variants" then
        error("work item does not carry variant scope")
      elif $decision.scope.selection.kind == "all" then
        "all"
      elif $decision.scope.selection.kind == "selected"
        and ($decision.scope.selection.variants | type == "array")
        and ($decision.scope.selection.variants | length) > 0 then
        {include: [
          $decision.scope.selection.variants[]
          | {id: .id} + .dimensions
        ]}
      else
        error("work item carries an invalid variant selection")
      end
  ' <<<"$RAIL_PLAN_JSON_CACHE"
}
