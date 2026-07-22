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

  if ! cargo metadata --locked --format-version 1 >/dev/null 2>&1; then
    RAIL_PLAN_JSON_CACHE=""
    return 1
  fi

  local plan_output
  # shellcheck disable=SC2086
  if ! plan_output="$(cargo rail plan $since_arg --quiet --json 2>/dev/null)"; then
    RAIL_PLAN_JSON_CACHE=""
    return 1
  fi
  if [[ -z "$plan_output" ]]; then
    RAIL_PLAN_JSON_CACHE=""
    return 1
  fi

  RAIL_PLAN_JSON_CACHE="$plan_output"
  printf '%s\n' "$RAIL_PLAN_JSON_CACHE"
}

_rail_valid_scope_json() {
  local document_kind=$1

  jq -ce --arg document_kind "$document_kind" '
    def valid_surfaces:
      type == "object"
      and has("bench")
      and has("build")
      and has("custom:cargo_graph")
      and has("docs")
      and has("infra")
      and has("test")
      and all(.[]; type == "boolean");

    def valid_crate:
      type == "string" and test("^[A-Za-z0-9][A-Za-z0-9_-]*$");

    def expected_cargo_args:
      if .mode == "empty" then
        []
      elif .mode == "workspace" then
        ["--workspace"]
      else
        [.crates[] | "-p", .]
      end;

    def valid_scope:
      type == "object"
      and .scope_contract_version == 2
      and (.resolved_base | type == "string" and length > 0)
      and (.resolved_head | type == "string" and length > 0)
      and (.mode == "empty" or .mode == "workspace" or .mode == "crates")
      and (.crates | type == "array")
      and (.cargo_args | type == "array" and all(.[]; type == "string"))
      and (.surfaces | valid_surfaces)
      and (
        if .mode == "crates" then
          (.crates
            | length > 0
            and all(.[]; valid_crate)
            and ((unique | length) == length))
        else
          (.crates | length == 0)
        end
      )
      and (.cargo_args == expected_cargo_args)
      and (
        if .mode == "empty" then
          (.surfaces.bench == false and .surfaces.build == false and .surfaces.test == false)
        else
          true
        end
      );

    if $document_kind == "plan" then
      select(
        type == "object"
        and .schema_version == 1
        and .command == "plan"
        and .mode == "inspect"
        and .result == "success"
        and .exit_code == 0
        and .plan_contract_version == 3
        and (.files | type == "array")
        and all(.files[];
          type == "object"
          and (.path
            | type == "string"
            and length > 0
            and (explode | all(. >= 32 and . != 127))))
      )
      | .scope
      | select(valid_scope)
    elif $document_kind == "scope" then
      select(valid_scope)
    else
      empty
    end
  '
}

rail_plan_is_valid() {
  local plan_output=$1
  [[ -n "$plan_output" ]] || return 1
  printf '%s\n' "$plan_output" | _rail_valid_scope_json plan >/dev/null 2>&1
}

rail_scope_json() {
  local scope_output

  if [[ -n "${RAIL_SCOPE_JSON:-}" ]]; then
    scope_output="$RAIL_SCOPE_JSON"
  elif [[ -n "${RAIL_SCOPE_JSON_CACHE:-}" ]]; then
    scope_output="$RAIL_SCOPE_JSON_CACHE"
  else
    local plan_output
    if ! plan_output="$(rail_plan_json)"; then
      return 1
    fi
    if ! scope_output="$(printf '%s\n' "$plan_output" | _rail_valid_scope_json plan 2>/dev/null)"; then
      return 1
    fi
  fi

  if ! RAIL_SCOPE_JSON_CACHE="$(printf '%s\n' "$scope_output" | _rail_valid_scope_json scope 2>/dev/null)"; then
    RAIL_SCOPE_JSON_CACHE=""
    return 1
  fi
  printf '%s\n' "$RAIL_SCOPE_JSON_CACHE"
}

rail_scope_mode() {
  local scope_output
  if ! scope_output="$(rail_scope_json)"; then
    echo "workspace"
    return 0
  fi

  printf '%s\n' "$scope_output" | jq -r '.mode'
}

rail_scope_surface_enabled() {
  local surface=$1
  local scope_output
  if ! scope_output="$(rail_scope_json)"; then
    return 0
  fi

  printf '%s\n' "$scope_output" \
    | jq -e --arg surface "$surface" '.surfaces[$surface] == true' >/dev/null 2>&1
}

rail_plan_crates() {
  local scope_output
  if ! scope_output="$(rail_scope_json)"; then
    echo ""
    return 0
  fi

  printf '%s\n' "$scope_output" | jq -r '
    if .mode == "crates" then
      .crates[]
    else
      empty
    end
  '
}
