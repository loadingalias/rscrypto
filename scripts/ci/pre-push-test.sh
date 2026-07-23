#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

fixture="$TMP_ROOT/repository"
fake_bin="$TMP_ROOT/bin"
mock_log="$TMP_ROOT/commands.log"
mkdir -p "$fixture/scripts/ci" "$fixture/scripts/lib" "$fake_bin" "$TMP_ROOT/home/.cargo/bin"
cp "$REPO_ROOT/scripts/ci/pre-push.sh" "$fixture/scripts/ci/pre-push.sh"
cp "$REPO_ROOT/scripts/lib/common.sh" "$REPO_ROOT/scripts/lib/rail-plan.sh" "$fixture/scripts/lib/"

git -C "$fixture" init --quiet

cat >"$fake_bin/cargo" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'cargo %s\n' "$*" >>"$MOCK_LOG"
if [[ "${1:-}" == "rail" && "${2:-}" == "plan" ]]; then
  printf '%s' "${MOCK_PLAN_OUTPUT:-}"
  exit "${MOCK_PLAN_STATUS:-0}"
fi
if [[ "$*" == "rail change check --merge-base --required" ]]; then
  exit "${MOCK_RELEASE_CHECK_STATUS:-42}"
fi
SH
chmod +x "$fake_bin/cargo"

cat >"$fake_bin/just" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'just %s\n' "$*" >>"$MOCK_LOG"
SH
chmod +x "$fake_bin/just"

workspace_scope='{"scope_contract_version":2,"resolved_base":"base","resolved_head":"head","mode":"workspace","crates":[],"cargo_args":["--workspace"],"surfaces":{"bench":false,"build":true,"custom:cargo_graph":true,"docs":false,"infra":false,"test":true}}'
empty_scope='{"scope_contract_version":2,"resolved_base":"base","resolved_head":"head","mode":"empty","crates":[],"cargo_args":[],"surfaces":{"bench":false,"build":false,"custom:cargo_graph":false,"docs":false,"infra":false,"test":false}}'
plan="$(jq -cn --argjson scope "$workspace_scope" '{schema_version:1,command:"plan",mode:"inspect",result:"success",exit_code:0,plan_contract_version:3,files:[{path:"Cargo.toml"}],scope:$scope}')"
empty_plan="$(jq -cn --argjson scope "$empty_scope" '{schema_version:1,command:"plan",mode:"inspect",result:"success",exit_code:0,plan_contract_version:3,files:[],scope:$scope}')"

normal_output="$TMP_ROOT/normal.out"
if (
  cd "$fixture"
  HOME="$TMP_ROOT/home" \
    PATH="$fake_bin:$PATH" \
    MOCK_LOG="$mock_log" \
    RAIL_PLAN_JSON_CACHE="$plan" \
    RAIL_SCOPE_JSON='' \
    RAIL_SCOPE_JSON_CACHE='' \
    scripts/ci/pre-push.sh --light
) >"$normal_output" 2>&1; then
  echo "ordinary pushes must fail when release intent coverage fails" >&2
  exit 1
fi
grep -Fq "cargo rail change check --merge-base --required" "$mock_log"

run_pre_push_case() {
  local name=$1
  local cached_plan=$2
  local planner_output=$3
  local planner_status=$4
  local expect_host=$5
  local output="$TMP_ROOT/$name.out"
  : >"$mock_log"

  if ! (
    unset RAIL_PLAN_JSON_CACHE RAIL_SCOPE_JSON RAIL_SCOPE_JSON_CACHE
    if [[ -n "$cached_plan" ]]; then
      export RAIL_PLAN_JSON_CACHE="$cached_plan"
    fi
    cd "$fixture"
    HOME="$TMP_ROOT/home" \
      PATH="$fake_bin:$PATH" \
      MOCK_LOG="$mock_log" \
      MOCK_PLAN_OUTPUT="$planner_output" \
      MOCK_PLAN_STATUS="$planner_status" \
      MOCK_RELEASE_CHECK_STATUS=0 \
      scripts/ci/pre-push.sh --light
  ) >"$output" 2>&1; then
    cat "$output" >&2
    echo "pre-push case failed: $name" >&2
    exit 1
  fi

  if [[ "$expect_host" == true ]]; then
    grep -Fq "just check" "$mock_log" || {
      echo "pre-push did not select host checks for $name" >&2
      exit 1
    }
    grep -Fq "cargo rail unify --check --explain" "$mock_log"
    grep -Fq "cargo rail change check --merge-base --required" "$mock_log"
  elif grep -Fq "just check" "$mock_log"; then
    echo "valid empty plan selected host checks" >&2
    exit 1
  fi
}

run_pre_push_case planner-failure '' '' 9 true
run_pre_push_case empty-output '' '' 0 true
run_pre_push_case malformed-plan '{' '' 0 true
run_pre_push_case missing-scope "$(jq -c 'del(.scope)' <<<"$plan")" '' 0 true
run_pre_push_case valid-empty "$empty_plan" '' 0 false

installer_scope=$(jq -c '.surfaces.infra = true' <<<"$empty_scope")
installer_plan=$(jq -cn --argjson scope "$installer_scope" \
  '{schema_version:1,command:"plan",mode:"inspect",result:"success",exit_code:0,plan_contract_version:3,files:[{path:"scripts/ci/install-tools.sh"}],scope:$scope}')
: >"$mock_log"
if ! (
  cd "$fixture"
  HOME="$TMP_ROOT/home" \
    PATH="$fake_bin:$PATH" \
    MOCK_LOG="$mock_log" \
    MOCK_RELEASE_CHECK_STATUS=0 \
    RAIL_PLAN_JSON_CACHE="$installer_plan" \
    RAIL_SCOPE_JSON='' \
    RAIL_SCOPE_JSON_CACHE='' \
    scripts/ci/pre-push.sh --light
) >/dev/null 2>&1; then
  echo "pre-push installer-integrity routing failed" >&2
  exit 1
fi
grep -Fq 'just check-actions' "$mock_log" || {
  echo "installer integrity changes did not select check-actions" >&2
  exit 1
}

plan_fixture="$TMP_ROOT/plan-repository"
mkdir -p "$plan_fixture/.config" "$plan_fixture/scripts" "$plan_fixture/src"
cp "$REPO_ROOT/.config/rail.toml" "$plan_fixture/.config/rail.toml"
printf '%s\n' \
  '[package]' \
  'name = "rscrypto"' \
  'version = "0.1.0"' \
  'edition = "2024"' \
  >"$plan_fixture/Cargo.toml"
printf '%s\n' '#![no_std]' >"$plan_fixture/src/lib.rs"
printf '%s\n' '# Scripts' >"$plan_fixture/scripts/README.md"

git -C "$plan_fixture" init --quiet --initial-branch=main
git -C "$plan_fixture" config user.email "ci@example.invalid"
git -C "$plan_fixture" config user.name "CI"
git -C "$plan_fixture" add .
git -C "$plan_fixture" commit --quiet -m "baseline"

printf '%s\n' '# Scripts' '' 'Documentation-only edit.' >"$plan_fixture/scripts/README.md"
git -C "$plan_fixture" add scripts/README.md
git -C "$plan_fixture" commit --quiet -m "edit script documentation"

docs_plan=$(cd "$plan_fixture" && cargo rail plan --from HEAD~1 --to HEAD --json)
jq -e '
  .result == "success"
    and (.files | length == 1)
    and .files[0].path == "scripts/README.md"
    and .files[0].kind == "docs"
    and .scope.mode == "empty"
    and .scope.surfaces.docs == true
    and .scope.surfaces.infra == false
    and .scope.surfaces.build == false
    and .scope.surfaces.test == false
    and .scope.surfaces["custom:cargo_graph"] == false
' >/dev/null <<<"$docs_plan"

printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail' >"$plan_fixture/scripts/check.sh"
git -C "$plan_fixture" add scripts/check.sh
git -C "$plan_fixture" commit --quiet -m "add script"

script_plan=$(cd "$plan_fixture" && cargo rail plan --from HEAD~1 --to HEAD --json)
jq -e '
  .result == "success"
    and (.files | length == 1)
    and .files[0].path == "scripts/check.sh"
    and .files[0].kind == "script"
    and .scope.surfaces.infra == true
    and .scope.surfaces.build == false
    and .scope.surfaces.test == false
' >/dev/null <<<"$script_plan"

echo "Pre-push regression tests passed"
