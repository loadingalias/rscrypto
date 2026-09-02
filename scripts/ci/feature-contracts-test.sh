#!/usr/bin/env bash
# Regression coverage for feature catalog deduplication, command scopes, and
# deterministic shards. Uses a fake Cargo executable; no product code builds.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXECUTOR="$REPO_ROOT/scripts/check/feature-contracts.sh"
# shellcheck source=../lib/feature-profiles.sh
source "$REPO_ROOT/scripts/lib/feature-profiles.sh"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT
# Make ordinary local runs prove that executor cases cannot consume an ambient plan.
export RAIL_PLAN_FILE="$TMP_ROOT/ambient-plan-must-not-be-read"
export RAIL_PLAN_READER="$TMP_ROOT/ambient-reader-must-not-be-read"
export RAIL_PLAN_IDENTITY=ambient-plan-identity-must-not-be-read
export RAIL_PLAN_HEAD_COMMIT=ambient-plan-head-must-not-be-read

fail() {
  echo "feature-contract executor regression failure: $*" >&2
  exit 1
}

fake_bin="$TMP_ROOT/bin"
command_log="$TMP_ROOT/commands.log"
real_cargo=$(command -v cargo)
clean_plan_env=(
  env -u BASH_ENV
  -u RAIL_PLAN_FILE -u RAIL_PLAN_READER
  -u RAIL_PLAN_IDENTITY -u RAIL_PLAN_HEAD_COMMIT
)
mkdir -p "$fake_bin"

for case_entry in "${RUNTIME_TEST_CASES[@]}"; do
  IFS='|' read -r _ case_target case_filter <<<"$case_entry"
  case "$case_target" in all | lib) continue ;; esac
  if [[ "$case_target" == websocket_accept_digest ]]; then
    source_path="$REPO_ROOT/tests/websocket_sha1.rs"
    rg -Uq '\[\[test\]\]\nname = "websocket_accept_digest"\npath = "tests/websocket_sha1.rs"' \
      "$REPO_ROOT/Cargo.toml" || fail "WebSocket test target no longer owns its declared source"
  else
    source_path="$REPO_ROOT/tests/$case_target.rs"
  fi
  [[ -f "$source_path" ]] \
    || fail "runtime case names missing Cargo test target $case_target"
  if [[ -n "$case_filter" ]] && ! grep -Fq "$case_filter" "$source_path"; then
    fail "runtime filter $case_filter is absent from $case_target"
  fi
done

cat >"$fake_bin/cargo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == metadata && " $* " == *" --no-deps "* ]]; then
  exec "$REAL_CARGO" "$@"
fi

printf 'cargo' >>"$MOCK_LOG"
printf ' %s' "$@" >>"$MOCK_LOG"
printf '\n' >>"$MOCK_LOG"

if [[ "${1:-}" == metadata ]]; then
  feature_set=""
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == --features ]]; then
      feature_set=$2
      break
    fi
    shift
  done
  if [[ "${MOCK_ALIAS_DIVERGES:-0}" == 1 && "$feature_set" == alloc,auth ]]; then
    printf '%s\n' '{"resolve":{"nodes":[{"id":"rscrypto","features":["alloc","auth","different"]}]}}'
  else
    printf '%s\n' '{"resolve":{"nodes":[{"id":"rscrypto","features":["alloc","auth"]}]}}'
  fi
fi
EOF
chmod +x "$fake_bin/cargo"
touch "$fake_bin/cargo-nextest"
chmod +x "$fake_bin/cargo-nextest"

run_executor() {
  local output=$1
  shift
  "${clean_plan_env[@]}" PATH="$fake_bin:$PATH" MOCK_LOG="$command_log" REAL_CARGO="$real_cargo" \
    "$EXECUTOR" "$@" >"$output" 2>&1
}

list_output="$TMP_ROOT/list.out"
run_executor "$list_output" list
grep -Fq 'compile (59 unique graphs, 60 named contracts)' "$list_output" \
  || fail "compile catalog counts changed"
grep -Fq 'alias: alloc,auth' "$list_output" || fail "compile alias disappeared"
grep -Fq 'runtime (9 profiles)' "$list_output" || fail "runtime catalog count changed"

matrix_output="$TMP_ROOT/matrix.json"
run_executor "$matrix_output" matrix
jq -e '
  (.include | length) == 11
  and ([.include[].domain] | map(select(. == "compile")) | length) == 2
  and ([.include[].domain] | map(select(. == "runtime")) | length) == 9
  and all(.include[]; (.profiles | type == "string" and length > 0))
  and all(.include[]; (.label | type == "string" and length > 0))
  and ([.include[] | select(.test_runner == "nextest")] | length) == 3
  and all(.include[]; .test_runner == "cargo" or .test_runner == "nextest")
' "$matrix_output" >/dev/null || fail "executor matrix does not name every bounded shard"
compile_matrix_count=$(jq -r '
  [.include[] | select(.domain == "compile") | .profiles | split(",")[]]
  | unique | length
' "$matrix_output")
[[ "$compile_matrix_count" -eq 59 ]] || fail "full matrix omitted compile profiles"
runtime_matrix_count=$(jq -r '
  [.include[] | select(.domain == "runtime") | .profiles | split(",")[]]
  | unique | length
' "$matrix_output")
[[ "$runtime_matrix_count" -eq 9 ]] || fail "full matrix omitted runtime profiles"

fake_plan="$TMP_ROOT/plan.json"
fake_reader="$TMP_ROOT/read.py"
cat >"$fake_plan" <<'EOF'
{"plan_contract_version":8,"identity":"plan-v8:sha256:test","required":["contracts.features"],"work":{"contracts.features":{"state":"required"}}}
EOF
cat >"$fake_reader" <<'EOF'
#!/usr/bin/env python3
import json
import sys

command = sys.argv[1]
if command in {"validate", "verify-checkout"}:
    raise SystemExit(0)
if command == "matrix":
    print(json.dumps({"include": [{
        "id": "checksums",
        "group": "Checksums",
        "feature_roots": "crc16,crc24,crc32,crc64",
        "runtime_profiles": "runtime.diagnostics",
        "full": False,
    }]}))
    raise SystemExit(0)
raise SystemExit(2)
EOF
chmod +x "$fake_reader"
selected_matrix="$TMP_ROOT/selected-matrix.json"
"${clean_plan_env[@]}" PATH="$fake_bin:$PATH" MOCK_LOG="$command_log" REAL_CARGO="$real_cargo" \
  RAIL_PLAN_FILE="$fake_plan" RAIL_PLAN_READER="$fake_reader" \
  "$EXECUTOR" matrix >"$selected_matrix"
jq -e '
  ([.include[].profiles | split(",")[]] | index("compile.crc16")) != null
  and ([.include[].profiles | split(",")[]] | index("compile.full")) != null
  and ([.include[].profiles | split(",")[]] | index("runtime.diagnostics")) != null
  and ([.include[].profiles | split(",")[]] | index("compile.sha2")) == null
  and ([.include[].profiles | split(",")[]] | index("runtime.std-full")) == null
' "$selected_matrix" >/dev/null || fail "selected Cargo Rail groups did not narrow the feature matrix"

: >"$command_log"
run_executor "$TMP_ROOT/selected.out" selected compile 1/2 compile.crc16,compile.crc32
selected_compile_count=$(grep -c '^cargo check ' "$command_log")
[[ "$selected_compile_count" -eq 2 ]] || fail "selected execution did not run exactly two compile profiles"
grep -Fq 'shard 1/2' "$TMP_ROOT/selected.out" || fail "selected execution lost its planned shard identity"
grep -F -- '--features crc16' "$command_log" >/dev/null || fail "selected CRC16 profile did not run"
grep -F -- '--features crc32' "$command_log" >/dev/null || fail "selected CRC32 profile did not run"
if grep -F -- '--features sha2' "$command_log" >/dev/null; then
  fail "selected execution ran an unselected profile"
fi

: >"$command_log"
run_executor "$TMP_ROOT/compile-1.out" compile 1/2
run_executor "$TMP_ROOT/compile-2.out" compile 2/2
compile_count=$(grep -c '^cargo check ' "$command_log")
[[ "$compile_count" -eq 59 ]] || fail "expected 59 compile commands, found $compile_count"
compile_unique=$(grep '^cargo check ' "$command_log" | LC_ALL=C sort -u | wc -l | tr -d ' ')
[[ "$compile_unique" -eq 59 ]] || fail "compile shards overlap or omit a unique graph"
metadata_count=$(grep -c '^cargo metadata ' "$command_log")
[[ "$metadata_count" -eq 2 ]] || fail "the one compile alias was not verified exactly once"
if grep -Fq 'cargo clean' "$command_log"; then
  fail "feature execution still deletes Cargo artifacts"
fi

: >"$command_log"
for ((shard = 1; shard <= FEATURE_RUNTIME_SHARDS; shard++)); do
  run_executor "$TMP_ROOT/runtime-$shard.out" runtime "$shard/$FEATURE_RUNTIME_SHARDS"
done
runtime_count=$(grep -Ec '^cargo (test|nextest run) ' "$command_log")
expected_runtime_count=${#RUNTIME_TEST_CASES[@]}
[[ "$runtime_count" -eq "$expected_runtime_count" ]] \
  || fail "expected $expected_runtime_count focused runtime commands, found $runtime_count"
runtime_unique=$(grep -E '^cargo (test|nextest run) ' "$command_log" | LC_ALL=C sort -u | wc -l | tr -d ' ')
[[ "$runtime_unique" -eq "$expected_runtime_count" ]] \
  || fail "runtime shards overlap or omit a test case"
profile_count=$(cat "$TMP_ROOT"/runtime-*.out | grep -c '^  profile ')
[[ "$profile_count" -eq 9 ]] || fail "runtime shards did not execute nine profiles exactly once"

nextest_count=$(grep -c '^cargo nextest run ' "$command_log")
[[ "$nextest_count" -eq 3 ]] || fail "expected three parallel behavior baselines, found $nextest_count"
grep '^cargo test ' "$command_log" \
  | grep -F -- '--features std,full,serde --test serde_roundtrip' >/dev/null \
  || fail "public Serde delta is not focused"
grep '^cargo test ' "$command_log" \
  | grep -F -- '--features std,parallel --lib' >/dev/null \
  || fail "parallel delta lost its unit tests"
grep '^cargo test ' "$command_log" \
  | grep -F -- '--features std,parallel --test argon2_parallel' >/dev/null \
  || fail "parallel delta lost its integration tests"
grep '^cargo test ' "$command_log" \
  | grep -F -- '--features std,full,getrandom --test rsa_public_key -- private_key_outputs_verify_and_decrypt' >/dev/null \
  || fail "entropy RSA coverage is not filtered to the gated tests"
grep '^cargo test ' "$command_log" \
  | grep -F -- '--features std,full,diag --test rsa_public_key -- pss_encoded_message_oracle_failures_are_opaque' >/dev/null \
  || fail "diagnostic RSA coverage lost an exact gated test"
if grep -Fq 'CARGO_TARGET_DIR' "$command_log"; then
  fail "feature execution still isolates or deletes a target tree"
fi

if run_executor "$TMP_ROOT/invalid-shard.out" compile 0/2; then
  fail "zero-based shard was accepted"
fi

: >"$command_log"
if "${clean_plan_env[@]}" PATH="$fake_bin:$PATH" MOCK_LOG="$command_log" REAL_CARGO="$real_cargo" MOCK_ALIAS_DIVERGES=1 \
  "$EXECUTOR" compile >"$TMP_ROOT/divergent-alias.out" 2>&1; then
  fail "divergent compile alias was accepted"
fi
grep -Fq "no longer resolve identically" "$TMP_ROOT/divergent-alias.out" \
  || fail "divergent alias did not explain the failure"

echo "Feature-contract executor regression tests passed"
