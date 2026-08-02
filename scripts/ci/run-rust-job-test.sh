#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISPATCHER="$SCRIPT_DIR/run-rust-job.sh"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

fail() {
  echo "rust job regression failure: $*" >&2
  exit 1
}

expect_failure() {
  if "$@" >/dev/null 2>&1; then
    fail "command unexpectedly succeeded: $*"
  fi
}

FIXTURE="$TMP_ROOT/repo"
CAPTURE="$TMP_ROOT/capture"
BIN="$TMP_ROOT/bin"
AMX_BIN="$TMP_ROOT/amx-bin"
mkdir -p "$FIXTURE/scripts/ci" "$FIXTURE/scripts/ct" "$CAPTURE" "$BIN" "$AMX_BIN"
cp "$DISPATCHER" "$FIXTURE/scripts/ci/run-rust-job.sh"

cat >"$BIN/just" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" >"$RSCRYPTO_CI_CAPTURE_DIR/just.args"
printf 'jobs=%s concurrency=%s\n' \
  "${RSCRYPTO_FUZZ_JOBS:-}" "${RSCRYPTO_FUZZ_TARGET_CONCURRENCY:-}" \
  >"$RSCRYPTO_CI_CAPTURE_DIR/just.env"
exit "${RSCRYPTO_MOCK_JUST_STATUS:-0}"
EOF

for command in uname lscpu sed rustc cargo; do
  cat >"$BIN/$command" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
done

cat >"$FIXTURE/scripts/ci/run-bench.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s' "$BENCH_ONLY" >"$RSCRYPTO_CI_CAPTURE_DIR/bench-targets"
printf '%s' "$BENCH_FILTER" >"$RSCRYPTO_CI_CAPTURE_DIR/bench-filter"
printf '%s' "$BENCH_QUICK" >"$RSCRYPTO_CI_CAPTURE_DIR/bench-quick"
EOF

cat >"$FIXTURE/scripts/ct/full.py" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" >"$RSCRYPTO_CI_CAPTURE_DIR/ct.args"
EOF

cat >"$FIXTURE/scripts/ct/python.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" >"$RSCRYPTO_CI_CAPTURE_DIR/ct-package.args"
EOF

chmod +x \
  "$BIN/just" \
  "$BIN/uname" \
  "$BIN/lscpu" \
  "$BIN/sed" \
  "$BIN/rustc" \
  "$BIN/cargo" \
  "$FIXTURE/scripts/ci/run-bench.sh" \
  "$FIXTURE/scripts/ct/full.py" \
  "$FIXTURE/scripts/ct/python.sh"

cat >"$AMX_BIN/uname" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  -s) echo Linux ;;
  -m) echo x86_64 ;;
  *) echo "Linux AMX fixture" ;;
esac
EOF

cat >"$AMX_BIN/lscpu" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "Architecture: x86_64"
EOF

cat >"$AMX_BIN/rustc" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "host: x86_64-unknown-linux-gnu"
EOF

cat >"$AMX_BIN/sed" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${*: -1}" == /proc/cpuinfo ]]; then
  if [[ "$*" == *'s/^flags'* ]]; then
    echo "amx_tile"
  else
    echo "flags : amx_tile"
  fi
else
  /usr/bin/sed "$@"
fi
EOF

cat >"$AMX_BIN/cargo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'CARGO_PROFILE_TEST_DEBUG=%s RUSTFLAGS=%s RSCRYPTO_REQUIRE_AMX=%s :: %s\n' \
  "${CARGO_PROFILE_TEST_DEBUG:-}" "${RUSTFLAGS:-}" "${RSCRYPTO_REQUIRE_AMX:-}" "$*" \
  >>"$RSCRYPTO_CI_CAPTURE_DIR/amx-cargo.args"

if [[ " $* " != *" --list "* ]]; then
  exit 0
fi

case "${RSCRYPTO_MOCK_AMX_LIST:-none}" in
  all)
    if [[ " $* " == *" --test platform_amx_permission "* ]]; then
      echo "linux_x86_64_amx_permission_and_cache_are_process_scoped: test"
    elif [[ " $* " == *" --lib "* ]]; then
      echo "platform::detect::tests::no_std_linux_x86_64_masks_compile_time_amx_without_a_permission_probe: test"
    fi
    ;;
  integration-only)
    if [[ " $* " == *" --test platform_amx_permission "* ]]; then
      echo "linux_x86_64_amx_permission_and_cache_are_process_scoped: test"
    fi
    ;;
  none) ;;
  *) exit 91 ;;
esac
EOF

chmod +x "$AMX_BIN/uname" "$AMX_BIN/lscpu" "$AMX_BIN/rustc" "$AMX_BIN/sed" "$AMX_BIN/cargo"

TEST_PATH="$BIN:$PATH"
RUNNER=(env PATH="$TEST_PATH" RSCRYPTO_CI_CAPTURE_DIR="$CAPTURE" bash "$FIXTURE/scripts/ci/run-rust-job.sh")

RSCRYPTO_CI_OPERATION=quality "${RUNNER[@]}"
[[ $(<"$CAPTURE/just.args") == "ci-check" ]] || fail "quality selected the wrong command"

mkdir -p \
  "$FIXTURE/fuzz/corpus/hash_cshake256" \
  "$FIXTURE/fuzz/artifacts/hash_cshake256" \
  "$FIXTURE/fuzz-packages/hash-sha3/corpus/hash_cshake256" \
  "$FIXTURE/fuzz-packages/hash-sha3/artifacts/hash_cshake256"
printf '%s' full-corpus >"$FIXTURE/fuzz/corpus/hash_cshake256/seed"
printf '%s' scoped-corpus >"$FIXTURE/fuzz-packages/hash-sha3/corpus/hash_cshake256/seed"
printf '%s' full-crash >"$FIXTURE/fuzz/artifacts/hash_cshake256/crash-fixture"
printf '%s' scoped-crash >"$FIXTURE/fuzz-packages/hash-sha3/artifacts/hash_cshake256/crash-fixture"

fuzz_status=0
env \
  PATH="$TEST_PATH" \
  RSCRYPTO_CI_CAPTURE_DIR="$CAPTURE" \
  RSCRYPTO_CI_OPERATION=fuzz \
  RSCRYPTO_MOCK_JUST_STATUS=23 \
  bash "$FIXTURE/scripts/ci/run-rust-job.sh" >/dev/null 2>&1 \
  || fuzz_status=$?
[[ "$fuzz_status" -eq 23 ]] || fail "fuzz operation did not preserve the fuzz command's failure status"
[[ $(<"$CAPTURE/just.env") == "jobs=1 concurrency=2" ]] \
  || fail "fuzz operation did not reserve both runner cores for independent targets"
[[ -f "$FIXTURE/fuzz-output/corpus.tar.gz" ]] || fail "fuzz failure did not produce a corpus archive"
tar -tzf "$FIXTURE/fuzz-output/corpus.tar.gz" >"$CAPTURE/fuzz-archive.entries"
grep -Fxq 'fuzz/corpus/hash_cshake256/seed' "$CAPTURE/fuzz-archive.entries" \
  || fail "fuzz failure archive omitted the full-workspace corpus"
grep -Fxq 'fuzz-packages/hash-sha3/corpus/hash_cshake256/seed' "$CAPTURE/fuzz-archive.entries" \
  || fail "fuzz failure archive omitted the scoped corpus"
[[ -f "$FIXTURE/fuzz/artifacts/hash_cshake256/crash-fixture" ]] \
  || fail "fuzz failure removed the full-workspace crash artifact"
[[ -f "$FIXTURE/fuzz-packages/hash-sha3/artifacts/hash_cshake256/crash-fixture" ]] \
  || fail "fuzz failure removed the scoped crash artifact"

sentinel="$TMP_ROOT/injected"
# shellcheck disable=SC2016 # Command substitution is an intentional literal injection payload.
shell_payload='$(touch "'"$sentinel"'"); echo injected; #'
expect_failure env \
  PATH="$TEST_PATH" \
  RSCRYPTO_CI_CAPTURE_DIR="$CAPTURE" \
  RSCRYPTO_CI_OPERATION="quality; $shell_payload" \
  bash "$FIXTURE/scripts/ci/run-rust-job.sh"
[[ ! -e "$sentinel" ]] || fail "operation input was evaluated as shell code"

expect_failure env \
  PATH="$TEST_PATH" \
  RSCRYPTO_CI_CAPTURE_DIR="$CAPTURE" \
  RSCRYPTO_CI_OPERATION=dependabot-smoke \
  RSCRYPTO_CI_BASE_SHA="$shell_payload" \
  bash "$FIXTURE/scripts/ci/run-rust-job.sh"
[[ ! -e "$sentinel" ]] || fail "commit input was evaluated as shell code"

multiline_payload="$shell_payload"$'\n'"second line"
env \
  PATH="$TEST_PATH" \
  RSCRYPTO_CI_CAPTURE_DIR="$CAPTURE" \
  RSCRYPTO_CI_OPERATION=benchmark \
  RSCRYPTO_CI_PLATFORM=amd-zen4 \
  RSCRYPTO_CI_BENCH_TARGETS="$multiline_payload" \
  RSCRYPTO_CI_BENCH_FILTER="$multiline_payload" \
  RSCRYPTO_CI_BENCH_QUICK=true \
  bash "$FIXTURE/scripts/ci/run-rust-job.sh" >/dev/null
[[ $(<"$CAPTURE/bench-targets") == "$multiline_payload" ]] || fail "benchmark targets were not passed literally"
[[ $(<"$CAPTURE/bench-filter") == "$multiline_payload" ]] || fail "benchmark filter was not passed literally"
[[ $(<"$CAPTURE/bench-quick") == "true" ]] || fail "benchmark boolean was not preserved"
[[ ! -e "$sentinel" ]] || fail "benchmark input was evaluated as shell code"

# shellcheck disable=SC2016 # Command substitution is an intentional literal injection payload.
ct_payload='$(touch${IFS}'"$sentinel"')'
env \
  PATH="$TEST_PATH" \
  RSCRYPTO_CI_CAPTURE_DIR="$CAPTURE" \
  RSCRYPTO_CI_OPERATION=constant-time \
  RSCRYPTO_CI_RUNNER=test-runner \
  RSCRYPTO_CI_PLATFORM=amd-zen4 \
  RSCRYPTO_CI_TARGET=x86_64-unknown-linux-gnu \
  RSCRYPTO_CI_DUDECT_TIMEOUT=1800 \
  RSCRYPTO_CI_DUDECT_FILTER="$ct_payload" \
  RSCRYPTO_CI_DUDECT_GATE=required \
  RSCRYPTO_CI_BINSEC_TIMEOUT=900 \
  RSCRYPTO_CI_UPLOAD_RAW_ARTIFACTS=false \
  bash "$FIXTURE/scripts/ci/run-rust-job.sh" >/dev/null
grep -Fxq -- "$ct_payload" "$CAPTURE/ct.args" || fail "DudeCT filter was not passed as one literal argument"
if grep -Fxq -- '--raw' "$CAPTURE/ct-package.args"; then
  fail "compact CT packaging included raw artifacts"
fi
[[ ! -e "$sentinel" ]] || fail "DudeCT filter was evaluated as shell code"

env \
  PATH="$TEST_PATH" \
  RSCRYPTO_CI_CAPTURE_DIR="$CAPTURE" \
  RSCRYPTO_CI_OPERATION=constant-time \
  RSCRYPTO_CI_RUNNER=test-runner \
  RSCRYPTO_CI_PLATFORM=amd-zen4 \
  RSCRYPTO_CI_TARGET=x86_64-unknown-linux-gnu \
  RSCRYPTO_CI_DUDECT_TIMEOUT=1800 \
  RSCRYPTO_CI_DUDECT_GATE=required \
  RSCRYPTO_CI_BINSEC_TIMEOUT=900 \
  RSCRYPTO_CI_UPLOAD_RAW_ARTIFACTS=true \
  bash "$FIXTURE/scripts/ci/run-rust-job.sh" >/dev/null
grep -Fxq -- '--raw' "$CAPTURE/ct-package.args" || fail "release CT packaging omitted raw artifacts"

expect_failure env \
  PATH="$TEST_PATH" \
  RSCRYPTO_CI_CAPTURE_DIR="$CAPTURE" \
  RSCRYPTO_CI_OPERATION=constant-time \
  RSCRYPTO_CI_PLATFORM=amd-zen4 \
  RSCRYPTO_CI_TARGET=x86_64-unknown-linux-gnu \
  RSCRYPTO_CI_DUDECT_TIMEOUT="1800; $shell_payload" \
  RSCRYPTO_CI_BINSEC_TIMEOUT=900 \
  RSCRYPTO_CI_UPLOAD_RAW_ARTIFACTS=false \
  bash "$FIXTURE/scripts/ci/run-rust-job.sh"
[[ ! -e "$sentinel" ]] || fail "numeric input was evaluated as shell code"

AMX_PATH="$AMX_BIN:$PATH"
expect_failure env \
  PATH="$AMX_PATH" \
  RSCRYPTO_CI_CAPTURE_DIR="$CAPTURE" \
  RSCRYPTO_CI_OPERATION=platform-amx \
  RSCRYPTO_CI_RUNNER=intel-spr \
  RSCRYPTO_MOCK_AMX_LIST=none \
  bash "$FIXTURE/scripts/ci/run-rust-job.sh"

expect_failure env \
  PATH="$AMX_PATH" \
  RSCRYPTO_CI_CAPTURE_DIR="$CAPTURE" \
  RSCRYPTO_CI_OPERATION=platform-amx \
  RSCRYPTO_CI_RUNNER=intel-spr \
  RSCRYPTO_MOCK_AMX_LIST=integration-only \
  bash "$FIXTURE/scripts/ci/run-rust-job.sh"

: >"$CAPTURE/amx-cargo.args"
env \
  PATH="$AMX_PATH" \
  RSCRYPTO_CI_CAPTURE_DIR="$CAPTURE" \
  RSCRYPTO_CI_OPERATION=platform-amx \
  RSCRYPTO_CI_RUNNER=intel-spr \
  RSCRYPTO_MOCK_AMX_LIST=all \
  bash "$FIXTURE/scripts/ci/run-rust-job.sh" >/dev/null
[[ "$(wc -l <"$CAPTURE/amx-cargo.args" | tr -d ' ')" == 4 ]] \
  || fail "AMX operation did not list and run both exact tests"
if grep -Fvq 'CARGO_PROFILE_TEST_DEBUG=0' "$CAPTURE/amx-cargo.args"; then
  fail "AMX operation retained full test-profile debug artifacts"
fi
grep -Fq \
  'RSCRYPTO_REQUIRE_AMX=1 :: test --locked --test platform_amx_permission -- --list' \
  "$CAPTURE/amx-cargo.args" \
  || fail "AMX integration test existence was not checked under the required permission contract"
grep -Fq \
  'RUSTFLAGS=-A unstable-features -C target-feature=+amx-tile,+amx-bf16,+amx-int8' \
  "$CAPTURE/amx-cargo.args" \
  || fail "AMX no_std test existence was not checked with forced AMX target features"

if grep -En '(^|[[:space:]])eval[[:space:]]|(^|[[:space:]])(bash|sh)[[:space:]]+-c|<<<' "$DISPATCHER" >/dev/null; then
  fail "dispatcher contains a dynamic shell interpreter"
fi

echo "Rust job dispatcher regression tests passed"
