#!/usr/bin/env bash
set -euo pipefail
unset BASH_ENV
unset RAIL_ALL RAIL_PLAN_FILE RAIL_PLAN_HEAD_COMMIT RAIL_PLAN_IDENTITY RAIL_PLAN_JSON_CACHE \
  RAIL_PLAN_JSON_CACHE_VALIDATED RAIL_PLAN_LOAD_ATTEMPTED RAIL_PLAN_LOADED RAIL_PLAN_LOCAL \
  RAIL_PLAN_READER RAIL_PLAN_USE_READER RAIL_SINCE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT
PYTHON="$("$REPO_ROOT/scripts/lib/python.sh" --print)"
export PYTHON

fail() {
  echo "changed-test planning regression failure: $*" >&2
  exit 1
}

make_plan() {
  local selection_kind=$1
  local state=${2:-required}
  jq -cn --arg selection_kind "$selection_kind" --arg state "$state" '
    def decision($work):
      if $state == "skipped" then
        {state: "skipped", evidence: ["evidence:sha256:test"]}
      else
        {
          state: "required",
          cause: "changed_input",
          evidence: ["evidence:sha256:test"],
          scope: {
            kind: "cargo",
            selection: (
              if $selection_kind == "workspace" then
                {kind: "workspace", cargo_args: [], targets: []}
              else
                {
                  kind: "packages",
                  packages: [{key: "rscrypto@0.0.0#path:", name: "rscrypto", cargo_spec: "rscrypto"}],
                  cargo_args: ["-p", "rscrypto"],
                  targets: []
                }
              end
            )
          }
        }
      end;
    {
      plan_contract_version: 8,
      identity: "plan-v8:sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      inputs: {head_commit: "0000000000000000000000000000000000000000"},
      work: {
        "cargo.build": decision("cargo.build"),
        "cargo.doctest": decision("cargo.doctest"),
        "cargo.test": decision("cargo.test")
      },
      required: (
        if $state == "required" then
          ["cargo.build", "cargo.doctest", "cargo.test"]
        else
          []
        end
      )
    }
  '
}

WORKSPACE_PLAN=$(make_plan workspace)
PACKAGE_PLAN=$(make_plan packages)
EMPTY_PLAN=$(make_plan workspace skipped)

BIN="$TMP_ROOT/bin"
LOG="$TMP_ROOT/commands.log"
mkdir -p "$BIN"
cat >"$BIN/cargo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'cargo %s\n' "$*" >>"$MOCK_LOG"
if [[ "${1:-}" == rail && "${2:-}" == plan && "${3:-}" == --verify ]]; then
  exit "${MOCK_VERIFY_STATUS:-0}"
fi
if [[ "${1:-}" == rail && "${2:-}" == plan ]]; then
  printf '%s\n' "${MOCK_PLAN_OUTPUT:-}"
  exit "${MOCK_PLAN_STATUS:-0}"
fi
exit 0
EOF
chmod +x "$BIN/cargo"

scope_mode() (
  export PATH="$BIN:$PATH"
  export MOCK_LOG="$LOG"
  export MOCK_PLAN_OUTPUT=$1
  export MOCK_PLAN_STATUS=${2:-0}
  export MOCK_VERIFY_STATUS=${3:-0}
  unset RAIL_PLAN_FILE RAIL_PLAN_READER RAIL_PLAN_JSON_CACHE RAIL_PLAN_JSON_CACHE_VALIDATED RAIL_PLAN_LOAD_ATTEMPTED
  # shellcheck source=../lib/rail-plan.sh
  source "$REPO_ROOT/scripts/lib/rail-plan.sh"
  rail_scope_mode cargo.test
)

[[ $(scope_mode "$WORKSPACE_PLAN") == workspace ]] || fail "workspace selection was not preserved"
[[ $(scope_mode "$PACKAGE_PLAN") == packages ]] || fail "package selection was not preserved"
[[ $(scope_mode "$EMPTY_PLAN") == empty ]] || fail "skipped work was not preserved"
if scope_mode '' 9 >/dev/null 2>&1; then
  fail "planner failure did not fail closed"
fi
if scope_mode "$PACKAGE_PLAN" 0 9 >/dev/null 2>&1; then
  fail "saved-plan verification failure did not fail closed"
fi
if scope_mode '{"plan_contract_version":7}' 0 0 >/dev/null 2>&1; then
  fail "v7 plan was accepted"
fi

package_args_file="$TMP_ROOT/package-args"
(
  export PATH="$BIN:$PATH"
  export MOCK_LOG="$LOG"
  export MOCK_PLAN_OUTPUT="$PACKAGE_PLAN"
  unset RAIL_PLAN_FILE RAIL_PLAN_READER RAIL_PLAN_JSON_CACHE RAIL_PLAN_JSON_CACHE_VALIDATED RAIL_PLAN_LOAD_ATTEMPTED
  # shellcheck source=../lib/rail-plan.sh
  source "$REPO_ROOT/scripts/lib/rail-plan.sh"
  rail_scope_cargo_args cargo.test
) >"$package_args_file"
[[ $(od -An -tx1 -v "$package_args_file" | tr -d ' \n') == 2d7000727363727970746f00 ]] \
  || fail "exact NUL-delimited Cargo arguments were not exposed"

run_test_consumer() {
  local plan=$1
  local expected=$2
  : >"$LOG"
  env \
    PATH="$BIN:/usr/bin:/bin" \
    MOCK_LOG="$LOG" \
    MOCK_PLAN_OUTPUT="$plan" \
    RSCRYPTO_SKIP_DOCTESTS=true \
    bash "$REPO_ROOT/scripts/test/test.sh" >/dev/null
  grep -Fq "$expected" "$LOG" || fail "test consumer did not use $expected"
  [[ $(grep -Fc 'cargo rail plan --quiet --json' "$LOG") -eq 1 ]] \
    || fail "test consumer did not reuse one validated Cargo Rail plan"
  [[ $(grep -Fc 'cargo rail plan --verify ' "$LOG") -eq 1 ]] \
    || fail "test consumer did not verify its Cargo Rail plan exactly once"
}

run_test_consumer "$PACKAGE_PLAN" 'cargo test --locked -p rscrypto --all-features --lib --tests'
run_test_consumer "$WORKSPACE_PLAN" 'cargo test --locked --all-features --lib --tests'

: >"$LOG"
env \
  PATH="$BIN:/usr/bin:/bin" \
  MOCK_LOG="$LOG" \
  MOCK_PLAN_OUTPUT="$EMPTY_PLAN" \
  RSCRYPTO_SKIP_DOCTESTS=true \
  bash "$REPO_ROOT/scripts/test/test.sh" >/dev/null
if grep -Fq 'cargo test ' "$LOG"; then
  fail "skipped cargo.test work executed tests"
fi

echo "Changed-test planning regression tests passed"
