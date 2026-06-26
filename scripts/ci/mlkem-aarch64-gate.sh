#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

cd "$REPO_ROOT"
maybe_disable_sccache

ARTIFACT_DIR="${MLKEM_AARCH64_GATE_ARTIFACT_DIR:-mlkem-aarch64-gate}"
LOG_PATH="$ARTIFACT_DIR/output.txt"
mkdir -p "$ARTIFACT_DIR"
: > "$LOG_PATH"

bool_value() {
  local raw="${1:-}"
  raw="$(echo "$raw" | tr '[:upper:]' '[:lower:]' | xargs)"
  case "$raw" in
    1 | true | yes | on | y) echo "true" ;;
    *) echo "false" ;;
  esac
}

require_native_linux_aarch64() {
  local os
  local machine
  os="$(uname -s)"
  machine="$(uname -m)"

  if [[ "$os" != "Linux" || "$machine" != "aarch64" ]]; then
    {
      echo "error: ML-KEM aarch64 gate must run natively on Linux/aarch64"
      echo "       got os=$os machine=$machine"
      echo "       set RSCRYPTO_MLKEM_AARCH64_GATE_ALLOW_NON_LINUX=true only for local smoke tests"
    } | tee -a "$LOG_PATH" >&2
    exit 2
  fi
}

run_step() {
  local name="$1"
  shift

  {
    echo ""
    echo "==> $name"
    echo "    $*"
  } | tee -a "$LOG_PATH"

  "$@" 2>&1 | tee -a "$LOG_PATH"
}

if [[ "$(bool_value "${RSCRYPTO_MLKEM_AARCH64_GATE_ALLOW_NON_LINUX:-false}")" != "true" ]]; then
  require_native_linux_aarch64
else
  echo "warning: native Linux/aarch64 host check bypassed for local smoke testing" | tee -a "$LOG_PATH"
fi

run_step "owned aarch64 NEON NTT scalar oracle" \
  cargo test --lib --features ml-kem ntt_neon -- --nocapture

run_step "owned aarch64 basemul scalar oracle" \
  cargo test --lib --features ml-kem basemul_accumulate -- --nocapture

run_step "ML-KEM ACVP FIPS 203 vectors" \
  cargo test --test mlkem_acvp --features ml-kem -- --nocapture

run_step "ML-KEM operation tests" \
  cargo test --test mlkem_ops --features ml-kem -- --nocapture

run_step "ML-KEM FIPS/property tests" \
  cargo test --test mlkem_properties --features ml-kem -- --nocapture

run_step "ML-KEM fuzz corpus replay" \
  cargo test --manifest-path fuzz/Cargo.toml --test corpus_replay replay_auth_mlkem -- --nocapture

if [[ "$(bool_value "${RSCRYPTO_MLKEM_AARCH64_GATE_SKIP_BENCH:-false}")" == "true" ]]; then
  echo "warning: ML-KEM benchmark gate skipped by RSCRYPTO_MLKEM_AARCH64_GATE_SKIP_BENCH" | tee -a "$LOG_PATH"
  exit 0
fi

RUN_DATE="$(date -u +"%Y-%m-%d")"
RUN_TIME="$(date -u +"%H_%M_%S")"
RUN_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
BENCH_DIR="$ARTIFACT_DIR/bench"
BENCH_RESULTS_DIR="$BENCH_DIR/results"
BENCH_ONLY_VALUE="${MLKEM_AARCH64_GATE_BENCH_ONLY:-mlkem,mlkem-matrix-sample,mlkem-arithmetic,mlkem-pke-phases,mlkem-decap-phases}"

run_step "ML-KEM benchmark gate" \
  env \
    BENCH_OUTPUT_DIR="$BENCH_DIR/output" \
    BENCH_RESULTS_DIR="$BENCH_RESULTS_DIR" \
    BENCH_RUN_DATE="$RUN_DATE" \
    BENCH_RUN_TIME="$RUN_TIME" \
    BENCH_RUN_OS=linux \
    BENCH_RUN_ARCH="${MLKEM_AARCH64_GATE_PLATFORM:-aarch64}" \
    BENCH_RUN_COMMIT="$RUN_COMMIT" \
    BENCH_RUN_MODE=ci \
    BENCH_ONLY="$BENCH_ONLY_VALUE" \
    BENCH_QUICK="${MLKEM_AARCH64_GATE_BENCH_QUICK:-true}" \
    BENCH_PLATFORM="${MLKEM_AARCH64_GATE_PLATFORM:-aarch64}" \
    scripts/ci/run-bench.sh

RESULTS_PATH="$BENCH_RESULTS_DIR/results.txt"
if [[ ! -f "$RESULTS_PATH" ]]; then
  echo "error: benchmark gate did not produce $RESULTS_PATH" | tee -a "$LOG_PATH" >&2
  exit 1
fi

for required in "aws-lc-rs" "mlkem-arithmetic" "mlkem-matrix-sample" "mlkem-pke-phases" "mlkem-decap-phases"; do
  if ! grep -q "$required" "$RESULTS_PATH"; then
    echo "error: benchmark gate results missing required marker: $required" | tee -a "$LOG_PATH" >&2
    exit 1
  fi
done

echo "" | tee -a "$LOG_PATH"
echo "ML-KEM Linux aarch64 promotion gate passed" | tee -a "$LOG_PATH"
