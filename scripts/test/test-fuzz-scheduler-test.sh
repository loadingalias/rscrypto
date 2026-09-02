#!/usr/bin/env bash
set -euo pipefail
unset BASH_ENV

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

fail() {
  echo "fuzz scheduler regression failure: $*" >&2
  exit 1
}

FIXTURE="$TMP_ROOT/repo"
CAPTURE="$TMP_ROOT/capture"
BIN="$TMP_ROOT/bin"
mkdir -p "$FIXTURE/.config" "$FIXTURE/scripts/test" "$FIXTURE/scripts/lib" "$FIXTURE/fuzz/corpus/alpha" "$CAPTURE" "$BIN"
cp "$REPO_ROOT/scripts/test/test-fuzz.sh" "$FIXTURE/scripts/test/"
cp "$REPO_ROOT/scripts/lib/common.sh" "$REPO_ROOT/scripts/lib/rail-plan.sh" \
  "$REPO_ROOT/scripts/lib/fuzz-packages.sh" "$REPO_ROOT/scripts/lib/toolchain.sh" \
  "$FIXTURE/scripts/lib/"
cp "$REPO_ROOT/.config/toolchains.toml" "$FIXTURE/.config/toolchains.toml"

cat >"$FIXTURE/fuzz/Cargo.toml" <<'EOF'
[package]
name = "fuzz-scheduler-fixture"
version = "0.0.0"

[package.metadata]
cargo-fuzz = true
EOF
printf '%s' preserved >"$FIXTURE/fuzz/corpus/alpha/seed"

cat >"$BIN/rustc" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "host: x86_64-unknown-linux-gnu"
EOF

cat >"$BIN/getconf" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ "${1:-}" == "_NPROCESSORS_ONLN" ]] || exit 1
printf '%s\n' "${RSCRYPTO_TEST_PROCESSORS:-8}"
EOF

cat >"$BIN/cargo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--version" ]]; then
  echo "cargo fixture"
  exit 0
fi
[[ "${1:-}" == "fuzz" ]] || exit 90
shift

subcommand=${1:-}
shift || true
if [[ "$subcommand" == "--version" ]]; then
  echo "cargo-fuzz fixture"
  exit 0
fi

fuzz_dir=""
target=""
artifact_prefix=""
args=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fuzz-dir)
      fuzz_dir=$2
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -artifact_prefix=*)
      artifact_prefix=${1#-artifact_prefix=}
      shift
      ;;
    --*)
      shift
      ;;
    *)
      if [[ -z "$target" ]]; then
        target=$1
      fi
      shift
      ;;
  esac
done
for arg in "$@"; do
  case "$arg" in
    -artifact_prefix=*) artifact_prefix=${arg#-artifact_prefix=} ;;
  esac
done

case "$subcommand" in
  list)
    printf '%s\n' alpha beta crash delta
    if [[ "${RSCRYPTO_FUZZ_LONG_LIST:-0}" == "1" ]]; then
      for index in {1..10000}; do
        printf 'filler_%s\n' "$index"
      done
    fi
    ;;
  run)
    printf '%s\n' "${args[@]}" >"$RSCRYPTO_FUZZ_CAPTURE/$target.args"

    lock="$RSCRYPTO_FUZZ_CAPTURE/lock"
    lock_capture() {
      while ! mkdir "$lock" 2>/dev/null; do
        sleep 0.01
      done
    }
    unlock_capture() {
      rmdir "$lock"
    }

    lock_capture
    current=$(cat "$RSCRYPTO_FUZZ_CAPTURE/current" 2>/dev/null || echo 0)
    current=$((current + 1))
    printf '%s' "$current" >"$RSCRYPTO_FUZZ_CAPTURE/current"
    maximum=$(cat "$RSCRYPTO_FUZZ_CAPTURE/maximum" 2>/dev/null || echo 0)
    if [[ "$current" -gt "$maximum" ]]; then
      printf '%s' "$current" >"$RSCRYPTO_FUZZ_CAPTURE/maximum"
    fi
    unlock_capture

    cleanup_active() {
      lock_capture
      current=$(cat "$RSCRYPTO_FUZZ_CAPTURE/current")
      printf '%s' "$((current - 1))" >"$RSCRYPTO_FUZZ_CAPTURE/current"
      unlock_capture
    }
    trap cleanup_active EXIT

    sleep 0.2
    mkdir -p "$fuzz_dir/corpus/$target" "$artifact_prefix"
    printf '%s' exercised >"$fuzz_dir/corpus/$target/exercised"
    if [[ "$target" == "crash" ]]; then
      printf '%s' crash >"${artifact_prefix}crash-fixture"
      exit 23
    fi
    ;;
  *)
    exit 91
    ;;
esac
EOF

chmod +x "$BIN/cargo" "$BIN/getconf" "$BIN/rustc"

status=0
env \
  PATH="$BIN:$PATH" \
  RSCRYPTO_FUZZ_CAPTURE="$CAPTURE" \
  RSCRYPTO_FUZZ_DURATION_SECS=7 \
  RSCRYPTO_FUZZ_TARGET_CONCURRENCY=2 \
  RSCRYPTO_FUZZ_JOBS=1 \
  bash "$FIXTURE/scripts/test/test-fuzz.sh" --full >"$CAPTURE/output" 2>&1 \
  || status=$?

[[ "$status" -eq 1 ]] || fail "aggregate status did not report a crashing target"
[[ $(<"$CAPTURE/maximum") == "2" ]] || fail "scheduler did not use exactly two target slots"
[[ $(<"$CAPTURE/current") == "0" ]] || fail "scheduler left a target process active"
grep -Fq 'Summary: 4 targets, 1 failed' "$CAPTURE/output" \
  || fail "scheduler did not aggregate every target result"

for target in alpha beta crash delta; do
  [[ -f "$CAPTURE/$target.args" ]] || fail "scheduler omitted $target"
  grep -Fxq -- '--jobs=1' "$CAPTURE/$target.args" \
    || fail "$target did not retain one independent LibFuzzer worker"
  grep -Fxq -- '-max_total_time=7' "$CAPTURE/$target.args" \
    || fail "$target did not retain its full duration"
  grep -Fxq -- "-artifact_prefix=$FIXTURE/fuzz/artifacts/$target/" "$CAPTURE/$target.args" \
    || fail "$target did not retain its private artifact prefix"
  [[ -f "$FIXTURE/fuzz/corpus/$target/exercised" ]] || fail "$target did not retain its corpus"
done

[[ $(<"$FIXTURE/fuzz/corpus/alpha/seed") == "preserved" ]] \
  || fail "scheduler replaced an existing corpus"
[[ -f "$FIXTURE/fuzz/artifacts/crash/crash-fixture" ]] \
  || fail "scheduler did not preserve the crashing target's artifact"

if env \
  PATH="$BIN:$PATH" \
  RSCRYPTO_FUZZ_CAPTURE="$CAPTURE" \
  RSCRYPTO_FUZZ_TARGET_CONCURRENCY=0 \
  bash "$FIXTURE/scripts/test/test-fuzz.sh" --full >/dev/null 2>&1; then
  fail "scheduler accepted zero concurrency"
fi

SELECTED_CAPTURE="$TMP_ROOT/selected-capture"
mkdir -p "$SELECTED_CAPTURE"
env \
  PATH="$BIN:$PATH" \
  RSCRYPTO_FUZZ_CAPTURE="$SELECTED_CAPTURE" \
  RSCRYPTO_FUZZ_DURATION_SECS=3 \
  RSCRYPTO_FUZZ_TARGET_CONCURRENCY=2 \
  RSCRYPTO_FUZZ_JOBS=1 \
  bash "$FIXTURE/scripts/test/test-fuzz.sh" --targets alpha,delta \
  >"$SELECTED_CAPTURE/output" 2>&1 \
  || fail "exact target selection failed"

for target in alpha delta; do
  [[ -f "$SELECTED_CAPTURE/$target.args" ]] || fail "exact selection omitted $target"
  grep -Fxq -- '-max_total_time=3' "$SELECTED_CAPTURE/$target.args" \
    || fail "$target ignored the exact-selection duration"
done
for target in beta crash; do
  [[ ! -e "$SELECTED_CAPTURE/$target.args" ]] || fail "exact selection ran unselected target $target"
done
grep -Fq 'Summary: 2 targets, 0 failed' "$SELECTED_CAPTURE/output" \
  || fail "exact selection did not aggregate only the requested targets"

CONSTRAINED_CAPTURE="$TMP_ROOT/constrained-capture"
mkdir -p "$CONSTRAINED_CAPTURE"
env \
  PATH="$BIN:$PATH" \
  RSCRYPTO_FUZZ_CAPTURE="$CONSTRAINED_CAPTURE" \
  RSCRYPTO_FUZZ_DURATION_SECS=1 \
  RSCRYPTO_TEST_PROCESSORS=2 \
  bash "$FIXTURE/scripts/test/test-fuzz.sh" --targets alpha,delta \
  >"$CONSTRAINED_CAPTURE/output" 2>&1 \
  || fail "automatic constrained-runner scheduling failed"
[[ $(<"$CONSTRAINED_CAPTURE/maximum") == "1" ]] \
  || fail "automatic scheduling oversubscribed a two-processor runner"

LONG_LIST_CAPTURE="$TMP_ROOT/long-list-capture"
mkdir -p "$LONG_LIST_CAPTURE"
env \
  PATH="$BIN:$PATH" \
  RSCRYPTO_FUZZ_CAPTURE="$LONG_LIST_CAPTURE" \
  RSCRYPTO_FUZZ_DURATION_SECS=1 \
  RSCRYPTO_FUZZ_LONG_LIST=1 \
  RSCRYPTO_FUZZ_TARGET_CONCURRENCY=1 \
  bash "$FIXTURE/scripts/test/test-fuzz.sh" --targets alpha \
  >"$LONG_LIST_CAPTURE/output" 2>&1 \
  || fail "target discovery failed when cargo-fuzz produced more than one pipe buffer"

if env \
  PATH="$BIN:$PATH" \
  RSCRYPTO_FUZZ_CAPTURE="$SELECTED_CAPTURE" \
  bash "$FIXTURE/scripts/test/test-fuzz.sh" --targets alpha,alpha >/dev/null 2>&1; then
  fail "exact selection accepted a duplicate target"
fi

if env \
  PATH="$BIN:$PATH" \
  RSCRYPTO_FUZZ_CAPTURE="$SELECTED_CAPTURE" \
  bash "$FIXTURE/scripts/test/test-fuzz.sh" --targets absent >/dev/null 2>&1; then
  fail "exact selection accepted an unknown target"
fi

echo "Fuzz scheduler regression tests passed"
