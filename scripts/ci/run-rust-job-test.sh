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
mkdir -p "$FIXTURE/scripts/ci" "$FIXTURE/scripts/ct" "$CAPTURE" "$BIN"
cp "$DISPATCHER" "$FIXTURE/scripts/ci/run-rust-job.sh"

cat >"$BIN/just" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" >"$RSCRYPTO_CI_CAPTURE_DIR/just.args"
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

TEST_PATH="$BIN:$PATH"
RUNNER=(env PATH="$TEST_PATH" RSCRYPTO_CI_CAPTURE_DIR="$CAPTURE" bash "$FIXTURE/scripts/ci/run-rust-job.sh")

RSCRYPTO_CI_OPERATION=quality "${RUNNER[@]}"
[[ $(<"$CAPTURE/just.args") == "ci-check" ]] || fail "quality selected the wrong command"

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
[[ ! -e "$sentinel" ]] || fail "DudeCT filter was evaluated as shell code"

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

if grep -En '(^|[[:space:]])eval[[:space:]]|(^|[[:space:]])(bash|sh)[[:space:]]+-c|<<<' "$DISPATCHER" >/dev/null; then
  fail "dispatcher contains a dynamic shell interpreter"
fi

echo "Rust job dispatcher regression tests passed"
