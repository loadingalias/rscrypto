#!/usr/bin/env bash
# Change-aware quality checks used by the supported push recipes.
# Profiles:
#   --light  skip the exhaustive rscrypto feature matrix (default)
#   --full   preserve the existing full host lane

set -euo pipefail

export PATH="$HOME/.cargo/bin:$PATH"

usage() {
  cat <<'USAGE'
Usage: scripts/ci/pre-push.sh [--light|--full]

Profiles:
  --light  run pre-push checks without the exhaustive rscrypto feature matrix
  --full   run the full host pre-push lane
USAGE
}

PRE_PUSH_PROFILE=light
if [[ "${RSCRYPTO_PRE_PUSH_FULL:-}" == "1" ]]; then
  PRE_PUSH_PROFILE=full
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --light)
      PRE_PUSH_PROFILE=light
      shift
      ;;
    --full)
      PRE_PUSH_PROFILE=full
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Error: unknown pre-push option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      # Git passes remote name/url to hooks; they are not profile options.
      break
      ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  echo "Error: unable to find repository root" >&2
  exit 1
fi

# shellcheck source=scripts/lib/common.sh
source "$REPO_ROOT/scripts/lib/common.sh"

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

needs_rail_config_check() {
  if ! cargo rail --version >/dev/null 2>&1; then
    return 1
  fi

  if [[ "$RAIL_READY" != true ]]; then
    return 0
  fi

  changed_file_matches '^\.config/rail\.toml$|^\.github/(workflows|actions)/.*\.ya?ml$|^scripts/ci/(install-tools|release-preflight)\.sh$|^justfile$'
}

needs_actions_check() {
  if [[ "$RAIL_READY" != true ]]; then
    return 0
  fi

  changed_file_matches '^\.config/target-matrix\.json$|^\.github/actions-lock\.yaml$|^\.github/(workflows|actions)/.*\.ya?ml$|^\.github/(rulesets|repository-settings)/.*\.json$|^scripts/ci/(pin-actions|check-ci-ownership|check-ci-ownership-test|release-evidence-check|release-evidence-check-test|repository-controls-evidence|repository-controls-evidence-test|package-release-source|write-release-manifest|release-identity-test|publish-immutable-release|publish-immutable-release-test|release-recipes-test)\.sh$'
}

needs_host_checks() {
  if [[ "$PRE_PUSH_PROFILE" == "full" || "$RAIL_READY" != true ]]; then
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

run_rail_config_check() {
  cargo rail config validate --strict
  cargo rail config migrate --check
}

run_rail_unify_check() {
  cargo rail unify --check --explain
}

run_rail_change_check() {
  cargo rail change check --merge-base --required
}

run_host_checks() {
  if [[ "$PRE_PUSH_PROFILE" == "full" ]]; then
    just check --all --feature-matrix
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

echo "Running pre-push checks (${PRE_PUSH_PROFILE})..."
describe_rail_plan

run_shell_syntax_checks
run_justfile_check

if needs_rail_config_check; then
  start_task "cargo-rail config" run_rail_config_check
else
  skip "Cargo-rail config" "no release/tooling config changes"
fi

if [[ "$RAIL_READY" != true ]] || rail_surface_is_enabled build || rail_surface_is_enabled test; then
  start_task "cargo-rail unify" run_rail_unify_check
  start_task "release intent coverage" run_rail_change_check
else
  skip "Cargo-rail unify" "no build/test surface"
  skip "Release intent coverage" "no build/test surface"
fi

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
