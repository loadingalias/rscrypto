#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

fail() {
  echo "changed-test planning regression failure: $*" >&2
  exit 1
}

assert_eq() {
  local expected=$1
  local actual=$2
  local description=$3
  if [[ "$actual" != "$expected" ]]; then
    printf 'expected %s: %q\nactual: %q\n' "$description" "$expected" "$actual" >&2
    exit 1
  fi
}

make_scope() {
  local mode=$1
  local crates=$2
  local cargo_args=$3
  local build=${4:-false}
  local test=${5:-false}
  local infra=${6:-false}

  jq -cn \
    --arg mode "$mode" \
    --argjson crates "$crates" \
    --argjson cargo_args "$cargo_args" \
    --argjson build "$build" \
    --argjson test "$test" \
    --argjson infra "$infra" \
    '{
      scope_contract_version: 2,
      resolved_base: "base",
      resolved_head: "head",
      mode: $mode,
      crates: $crates,
      cargo_args: $cargo_args,
      surfaces: {
        bench: false,
        build: $build,
        "custom:cargo_graph": false,
        docs: false,
        infra: $infra,
        test: $test
      }
    }'
}

make_plan() {
  local scope=$1
  local files=${2:-'[]'}

  jq -cn \
    --argjson scope "$scope" \
    --argjson files "$files" \
    '{
      schema_version: 1,
      command: "plan",
      mode: "inspect",
      result: "success",
      exit_code: 0,
      plan_contract_version: 3,
      files: $files,
      scope: $scope
    }'
}

EMPTY_SCOPE="$(make_scope empty '[]' '[]')"
WORKSPACE_SCOPE="$(make_scope workspace '[]' '["--workspace"]' true true false)"
CRATES_SCOPE="$(make_scope crates '["crate-a","crate-b"]' '["-p","crate-a","-p","crate-b"]' true true false)"
EMPTY_PLAN="$(make_plan "$EMPTY_SCOPE")"
WORKSPACE_PLAN="$(make_plan "$WORKSPACE_SCOPE")"
CRATES_PLAN="$(make_plan "$CRATES_SCOPE")"

planner_bin="$TMP_ROOT/planner-bin"
mkdir -p "$planner_bin"
cat >"$planner_bin/cargo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "rail" && "${2:-}" == "plan" ]]; then
  printf '%s' "${MOCK_PLAN_OUTPUT:-}"
  exit "${MOCK_PLAN_STATUS:-0}"
fi
exit 0
EOF
chmod +x "$planner_bin/cargo"

scope_mode_for_cached_plan() (
  unset RAIL_SCOPE_JSON RAIL_SCOPE_JSON_CACHE
  export RAIL_PLAN_JSON_CACHE=$1
  # shellcheck source=../lib/rail-plan.sh
  source "$REPO_ROOT/scripts/lib/rail-plan.sh"
  rail_scope_mode
)

scope_mode_for_planner() (
  unset RAIL_PLAN_JSON_CACHE RAIL_SCOPE_JSON RAIL_SCOPE_JSON_CACHE
  export PATH="$planner_bin:$PATH"
  export MOCK_PLAN_OUTPUT=$1
  export MOCK_PLAN_STATUS=$2
  # shellcheck source=../lib/rail-plan.sh
  source "$REPO_ROOT/scripts/lib/rail-plan.sh"
  rail_scope_mode
)

assert_eq workspace "$(scope_mode_for_planner '' 9)" "nonzero planner exit to fail closed"
assert_eq workspace "$(scope_mode_for_planner '' 0)" "empty planner output to fail closed"
assert_eq workspace "$(scope_mode_for_cached_plan '{')" "malformed JSON to fail closed"

unsuccessful_plan="$(jq -c '.result = "failure" | .exit_code = 1' <<<"$WORKSPACE_PLAN")"
assert_eq workspace "$(scope_mode_for_cached_plan "$unsuccessful_plan")" "unsuccessful result to fail closed"

missing_scope="$(jq -c 'del(.scope)' <<<"$WORKSPACE_PLAN")"
missing_mode="$(jq -c 'del(.scope.mode)' <<<"$WORKSPACE_PLAN")"
unsupported_mode="$(jq -c '.scope.mode = "packages"' <<<"$WORKSPACE_PLAN")"
assert_eq workspace "$(scope_mode_for_cached_plan "$missing_scope")" "missing scope to fail closed"
assert_eq workspace "$(scope_mode_for_cached_plan "$missing_mode")" "missing mode to fail closed"
assert_eq workspace "$(scope_mode_for_cached_plan "$unsupported_mode")" "unsupported mode to fail closed"

empty_crate_selection="$(jq -c '.scope.mode = "crates" | .scope.cargo_args = []' <<<"$WORKSPACE_PLAN")"
non_string_crate="$(jq -c '.scope.crates = [7] | .scope.mode = "crates" | .scope.cargo_args = ["-p", "7"]' <<<"$WORKSPACE_PLAN")"
malformed_surface="$(jq -c '.scope.surfaces.test = "false"' <<<"$WORKSPACE_PLAN")"
assert_eq workspace "$(scope_mode_for_cached_plan "$empty_crate_selection")" "empty crate selection to fail closed"
assert_eq workspace "$(scope_mode_for_cached_plan "$non_string_crate")" "non-string crate selection to fail closed"
assert_eq workspace "$(scope_mode_for_cached_plan "$malformed_surface")" "malformed surface to fail closed"

assert_eq empty "$(scope_mode_for_cached_plan "$EMPTY_PLAN")" "valid explicit empty scope"
assert_eq workspace "$(scope_mode_for_cached_plan "$WORKSPACE_PLAN")" "valid workspace scope"
assert_eq crates "$(scope_mode_for_cached_plan "$CRATES_PLAN")" "valid crate scope"

crate_output="$(
  (
    export RAIL_PLAN_JSON_CACHE="$CRATES_PLAN"
    unset RAIL_SCOPE_JSON RAIL_SCOPE_JSON_CACHE
    # shellcheck source=../lib/rail-plan.sh
    source "$REPO_ROOT/scripts/lib/rail-plan.sh"
    rail_plan_crates
  ) 2>/dev/null
)"
assert_eq $'crate-a\ncrate-b' "$crate_output" "validated crate selection"

command_bin="$TMP_ROOT/command-bin"
command_home="$TMP_ROOT/command-home"
command_log="$TMP_ROOT/commands.log"
jq_dir="$(dirname "$(command -v jq)")"
mkdir -p "$command_bin" "$command_home/.cargo/bin"
cat >"$command_bin/cargo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'cargo %s\n' "$*" >>"$MOCK_LOG"
if [[ "${1:-}" == "rail" && "${2:-}" == "plan" ]]; then
  printf '%s' "${MOCK_PLAN_OUTPUT:-}"
  exit "${MOCK_PLAN_STATUS:-0}"
fi
exit "${MOCK_CARGO_STATUS:-0}"
EOF
chmod +x "$command_bin/cargo"
cp "$command_bin/cargo" "$command_home/.cargo/bin/cargo"

run_test_consumer() {
  local name=$1
  local plan=$2
  local status=$3
  local expected=$4
  local output="$TMP_ROOT/test-$name.out"
  : >"$command_log"

  if ! (
    unset RAIL_PLAN_JSON_CACHE RAIL_SCOPE_JSON RAIL_SCOPE_JSON_CACHE
    export HOME="$command_home"
    export PATH="$command_bin:$jq_dir:/usr/bin:/bin"
    export MOCK_LOG="$command_log"
    export MOCK_PLAN_OUTPUT="$plan"
    export MOCK_PLAN_STATUS="$status"
    export RSCRYPTO_SKIP_DOCTESTS=true
    cd "$REPO_ROOT"
    bash scripts/test/test.sh
  ) >"$output" 2>&1; then
    cat "$output" >&2
    fail "test consumer failed for $name"
  fi

  local actual
  actual="$(grep '^cargo test' "$command_log" || true)"
  assert_eq "$expected" "$actual" "test commands for $name"
}

workspace_test='cargo test --workspace --all-features --lib --tests'
run_test_consumer planner-failure '' 9 "$workspace_test"
run_test_consumer empty-output '' 0 "$workspace_test"
run_test_consumer malformed-json '{' 0 "$workspace_test"
run_test_consumer unsuccessful "$unsuccessful_plan" 0 "$workspace_test"
run_test_consumer missing-scope "$missing_scope" 0 "$workspace_test"
run_test_consumer missing-mode "$missing_mode" 0 "$workspace_test"
run_test_consumer unsupported-mode "$unsupported_mode" 0 "$workspace_test"
run_test_consumer malformed-crates "$empty_crate_selection" 0 "$workspace_test"
run_test_consumer malformed-surface "$malformed_surface" 0 "$workspace_test"
run_test_consumer valid-empty "$EMPTY_PLAN" 0 ''
run_test_consumer valid-workspace "$WORKSPACE_PLAN" 0 "$workspace_test"
run_test_consumer valid-crates "$CRATES_PLAN" 0 $'cargo test -p crate-a --all-features --lib --tests\ncargo test -p crate-b --all-features --lib --tests'

check_fixture="$TMP_ROOT/check-repository"
mkdir -p "$check_fixture/scripts/check" "$check_fixture/scripts/lib" "$check_fixture/scripts/ct" "$check_fixture/scripts/test"
cp "$REPO_ROOT/scripts/check/check.sh" "$check_fixture/scripts/check/check.sh"
cp "$REPO_ROOT/scripts/lib/common.sh" "$REPO_ROOT/scripts/lib/rail-plan.sh" "$check_fixture/scripts/lib/"
for helper in "$check_fixture/scripts/check/asm-ledger.sh" \
  "$check_fixture/scripts/check/check-feature-matrix.sh" \
  "$check_fixture/scripts/ct/python.sh" \
  "$check_fixture/scripts/test/test-feature-matrix.sh"; do
  cat >"$helper" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
  chmod +x "$helper"
done

run_check_consumer() {
  local name=$1
  local plan=$2
  local status=$3
  local expected_scope=$4
  local output="$TMP_ROOT/check-$name.out"
  : >"$command_log"

  if ! (
    unset RAIL_PLAN_JSON_CACHE RAIL_SCOPE_JSON RAIL_SCOPE_JSON_CACHE
    export HOME="$command_home"
    export PATH="$command_bin:$jq_dir:/usr/bin:/bin"
    export MOCK_LOG="$command_log"
    export MOCK_PLAN_OUTPUT="$plan"
    export MOCK_PLAN_STATUS="$status"
    export RSCRYPTO_SKIP_CHECK_SUPPLY_CHAIN=1
    cd "$check_fixture"
    bash scripts/check/check.sh
  ) >"$output" 2>&1; then
    cat "$output" >&2
    fail "check consumer failed for $name"
  fi

  grep -Eq "^cargo check ${expected_scope}([[:space:]]|$)" "$command_log" \
    || fail "check consumer selected the wrong scope for $name"
}

run_check_consumer planner-failure '' 9 '--workspace'
run_check_consumer valid-empty "$EMPTY_PLAN" 0 '--workspace'
run_check_consumer valid-workspace "$WORKSPACE_PLAN" 0 '--workspace'
run_check_consumer valid-crates "$CRATES_PLAN" 0 '-p crate-a -p crate-b'

check_all_fixture="$TMP_ROOT/check-all-repository"
mkdir -p "$check_all_fixture/scripts/check" "$check_all_fixture/scripts/lib"
cp "$REPO_ROOT/scripts/check/check-all.sh" "$check_all_fixture/scripts/check/check-all.sh"
cp "$REPO_ROOT/scripts/lib/common.sh" "$REPO_ROOT/scripts/lib/rail-plan.sh" "$check_all_fixture/scripts/lib/"
cat >"$check_all_fixture/scripts/lib/targets.sh" <<'EOF'
WIN_TARGETS=()
LINUX_TARGETS=()
IBM_TARGETS=()
NOSTD_TARGETS=(mock-target)
WASM_TARGETS=()
EOF
for helper in check.sh zeroize-evidence.sh check-win.sh check-linux.sh check-ibm.sh; do
  cat >"$check_all_fixture/scripts/check/$helper" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
  chmod +x "$check_all_fixture/scripts/check/$helper"
done
cat >"$command_bin/rustup" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == "target list --installed" ]]; then
  echo mock-target
fi
EOF
chmod +x "$command_bin/rustup"
printf '%s\n' '[package]' 'name = "fixture"' 'version = "0.0.0"' >"$check_all_fixture/Cargo.toml"

run_check_all_consumer() {
  local name=$1
  local plan=$2
  local status=$3
  local expected_crate=$4
  local output="$TMP_ROOT/check-all-$name.out"
  : >"$command_log"

  if ! (
    unset RAIL_PLAN_JSON_CACHE RAIL_SCOPE_JSON RAIL_SCOPE_JSON_CACHE
    export HOME="$command_home"
    export PATH="$command_bin:$jq_dir:/usr/bin:/bin"
    export MOCK_LOG="$command_log"
    export MOCK_PLAN_OUTPUT="$plan"
    export MOCK_PLAN_STATUS="$status"
    cd "$check_all_fixture"
    bash scripts/check/check-all.sh
  ) >"$output" 2>&1; then
    cat "$output" >&2
    fail "check-all consumer failed for $name"
  fi

  grep -Fq "cargo check -p $expected_crate --no-default-features --target mock-target --lib" "$command_log" \
    || fail "check-all selected the wrong constrained crate for $name"
}

run_check_all_consumer planner-failure '' 9 rscrypto
run_check_all_consumer valid-empty "$EMPTY_PLAN" 0 rscrypto
run_check_all_consumer valid-workspace "$WORKSPACE_PLAN" 0 rscrypto
run_check_all_consumer valid-crates "$CRATES_PLAN" 0 crate-a

run_workflow_resolver() {
  local name=$1
  local outcome=$2
  local scope=$3
  local expected=$4
  local output="$TMP_ROOT/resolver-$name.outputs"

  : >"$output"
  if ! GITHUB_OUTPUT="$output" \
    RAIL_PLAN_STEP_OUTCOME="$outcome" \
    RAIL_SCOPE_JSON="$scope" \
    bash "$REPO_ROOT/scripts/ci/resolve-rail-plan.sh"; then
    fail "workflow resolver failed for $name"
  fi
  assert_eq "$expected" "$(<"$output")" "workflow outputs for $name"
}

fallback_outputs=$'valid=false\nempty=false\nbuild=true\ntest=true\ninfra=true\ncargo_graph=true'
run_workflow_resolver planner-failure failure '' "$fallback_outputs"
run_workflow_resolver malformed-scope success '{' "$fallback_outputs"
run_workflow_resolver valid-empty success "$EMPTY_SCOPE" $'valid=true\nempty=true\nbuild=false\ntest=false\ninfra=false\ncargo_graph=false'
run_workflow_resolver valid-workspace success "$WORKSPACE_SCOPE" $'valid=true\nempty=false\nbuild=true\ntest=true\ninfra=false\ncargo_graph=false'
run_workflow_resolver valid-crates success "$CRATES_SCOPE" $'valid=true\nempty=false\nbuild=true\ntest=true\ninfra=false\ncargo_graph=false'

echo "Changed-test planning regression tests passed"
