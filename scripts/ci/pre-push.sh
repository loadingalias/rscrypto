#!/usr/bin/env bash
# Pre-push hook: quality checks before allowing push.
# Install:
#   ln -sf ../../scripts/ci/pre-push.sh .git/hooks/pre-push
# Force the full host lane with RSCRYPTO_PRE_PUSH_FULL=1.

set -euo pipefail

export PATH="$HOME/.cargo/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

cd "$REPO_ROOT"

LOG_DIR=$(mktemp -d)
trap 'rm -rf "$LOG_DIR"' EXIT

TASK_NAMES=()
TASK_PIDS=()
TASK_LOGS=()

RAIL_READY=false
RAIL_PLAN_JSON_CACHE="$(rail_plan_json)"
export RAIL_PLAN_JSON_CACHE

if [[ -n "$RAIL_PLAN_JSON_CACHE" ]] \
  && command -v jq >/dev/null 2>&1 \
  && jq -e '.result == "success"' <<<"$RAIL_PLAN_JSON_CACHE" >/dev/null 2>&1; then
  RAIL_READY=true
  RAIL_SCOPE_JSON="$(rail_scope_json)"
  export RAIL_SCOPE_JSON
fi

changed_files() {
  if [[ "$RAIL_READY" != true ]]; then
    return 0
  fi

  jq -r '.files[]?.path // empty' <<<"$RAIL_PLAN_JSON_CACHE"
}

changed_file_matches() {
  local pattern=$1
  local path

  while IFS= read -r path; do
    if [[ "$path" =~ $pattern ]]; then
      return 0
    fi
  done < <(changed_files)

  return 1
}

rail_surface_is_enabled() {
  local surface=$1
  [[ "$RAIL_READY" == true ]] && rail_scope_surface_enabled "$surface"
}

describe_rail_plan() {
  if [[ "$RAIL_READY" != true ]]; then
    echo "cargo-rail plan unavailable; running conservative pre-push lanes."
    return 0
  fi

  local file_count scope surfaces
  file_count="$(jq -r '.files | length' <<<"$RAIL_PLAN_JSON_CACHE")"
  scope="$(jq -r '.scope.mode // "unknown"' <<<"$RAIL_PLAN_JSON_CACHE")"
  surfaces="$(jq -r '[.scope.surfaces // {} | to_entries[] | select(.value == true) | .key] | join(",")' <<<"$RAIL_PLAN_JSON_CACHE")"
  [[ -n "$surfaces" ]] || surfaces="none"

  echo "cargo-rail: ${file_count} changed file(s), scope=${scope}, surfaces=${surfaces}"
}

run_shell_syntax_checks() {
  local scripts=()
  local path

  if [[ "$RAIL_READY" == true ]]; then
    while IFS= read -r path; do
      if [[ "$path" == scripts/*.sh && -f "$path" ]]; then
        scripts+=("$path")
      fi
    done < <(changed_files)
  else
    while IFS= read -r path; do
      if [[ -f "$path" ]]; then
        scripts+=("$path")
      fi
    done < <(git ls-files 'scripts/*.sh')
  fi

  if [[ ${#scripts[@]} -eq 0 ]]; then
    skip "Shell syntax" "no changed shell scripts"
    return 0
  fi

  step "Shell syntax"
  for script in "${scripts[@]}"; do
    bash -n "$script"
  done
  ok
}

run_justfile_check() {
  if [[ "$RAIL_READY" != true ]] || ! changed_file_matches '^justfile$'; then
    return 0
  fi

  step "Justfile parse"
  just --list >/dev/null
  ok
}

needs_actions_check() {
  if [[ "$RAIL_READY" != true ]]; then
    return 0
  fi

  changed_file_matches '^\.github/actions-lock\.yaml$|^\.github/(workflows|actions)/.*\.ya?ml$|^scripts/ci/pin-actions\.sh$'
}

needs_host_checks() {
  if [[ "${RSCRYPTO_PRE_PUSH_FULL:-}" == "1" || "$RAIL_READY" != true ]]; then
    return 0
  fi

  if rail_surface_is_enabled build || rail_surface_is_enabled docs || rail_surface_is_enabled test; then
    return 0
  fi

  changed_file_matches '^scripts/check/|^scripts/lib/(common|rail-plan)\.sh$|^scripts/test/test-feature-matrix\.sh$'
}

run_actions_check() {
  just check-actions
}

run_host_checks() {
  if [[ "${RSCRYPTO_PRE_PUSH_FULL:-}" == "1" ]]; then
    just check --all
  else
    just check
  fi
}

start_task() {
  local name=$1
  shift

  local safe_name log_file
  safe_name="${name//[^A-Za-z0-9_.-]/_}"
  log_file="$LOG_DIR/$safe_name.log"

  echo "  → Starting $name"
  ("$@") >"$log_file" 2>&1 &
  TASK_NAMES+=("$name")
  TASK_PIDS+=("$!")
  TASK_LOGS+=("$log_file")
}

wait_tasks() {
  local status=0
  local i

  for i in "${!TASK_PIDS[@]}"; do
    if wait "${TASK_PIDS[$i]}"; then
      echo "  ${GREEN}✓${RESET} ${TASK_NAMES[$i]}"
    else
      echo "  ${RED}✗${RESET} ${TASK_NAMES[$i]}"
      show_error "${TASK_LOGS[$i]}"
      status=1
    fi
  done

  return "$status"
}

echo "Running pre-push checks..."
describe_rail_plan

run_shell_syntax_checks
run_justfile_check

if needs_actions_check; then
  start_task "workflow action pins" run_actions_check
else
  skip "Workflow action pins" "no workflow/action pin changes"
fi

if needs_host_checks; then
  start_task "host Rust checks" run_host_checks
else
  skip "Host Rust checks" "cargo-rail found no build/docs/test surface"
fi

wait_tasks

echo "All checks passed."
