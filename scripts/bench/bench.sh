#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/bench/bench.sh [selector ...] [key=value ...]

Selectors:
  aead
  auth
  blake2
  blake3
  crc32c
  checksum
  hashes

Key=Value Overrides:
  crates=<csv> | crate=<csv>
  benches=<csv> | bench=<csv>
  only=<csv>
  filter=<csv>
  quick=true|false
  warmup_ms=<int>
  measure_ms=<int>
  sample_size=<int>
  profile_time_secs=<num>
  output_dir=<path>
  clean=true|false
USAGE
}

append_csv() {
  local current="${1:-}"
  local token="${2:-}"
  token="$(echo "$token" | xargs)"
  if [[ -z "$token" ]]; then
    echo "$current"
    return 0
  fi
  if [[ -z "$current" ]]; then
    echo "$token"
  else
    echo "$current,$token"
  fi
}

apply_kv() {
  local key="${1:-}"
  local value="${2:-}"
  key="$(echo "$key" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"

  case "$key" in
    crate | crates)
      CRATES="$(append_csv "$CRATES" "$value")"
      ;;
    bench | benches)
      BENCHES="$(append_csv "$BENCHES" "$value")"
      ;;
    only)
      ONLY="$(append_csv "$ONLY" "$value")"
      ;;
    filter)
      FILTER="$(append_csv "$FILTER" "$value")"
      ;;
    quick)
      QUICK="$value"
      ;;
    warmup_ms)
      WARMUP_MS="$value"
      ;;
    measure_ms)
      MEASURE_MS="$value"
      ;;
    sample_size)
      SAMPLE_SIZE="$value"
      ;;
    profile_time_secs)
      PROFILE_TIME_SECS="$value"
      ;;
    output_dir)
      OUTPUT_DIR="$value"
      ;;
    clean)
      CLEAN="$value"
      ;;
    *)
      echo "error: unknown key '$key' in '$key=$value'" >&2
      usage >&2
      exit 2
      ;;
  esac
}

CRATES="${BENCH_CRATES:-}"
BENCHES="${BENCH_BENCHES:-}"
ONLY="${BENCH_ONLY:-}"
FILTER="${BENCH_FILTER:-}"
QUICK="${BENCH_QUICK:-false}"
WARMUP_MS="${BENCH_WARMUP_MS:-}"
MEASURE_MS="${BENCH_MEASURE_MS:-}"
SAMPLE_SIZE="${BENCH_SAMPLE_SIZE:-}"
PROFILE_TIME_SECS="${BENCH_PROFILE_TIME_SECS:-}"
OUTPUT_DIR="${BENCH_OUTPUT_DIR:-benchmark-results}"
CLEAN="${BENCH_CLEAN:-true}"

while [[ $# -gt 0 ]]; do
  token="$1"
  case "$token" in
    -h | --help)
      usage
      exit 0
      ;;
    --quick)
      QUICK="true"
      ;;
    --no-quick)
      QUICK="false"
      ;;
    --clean)
      CLEAN="true"
      ;;
    --no-clean)
      CLEAN="false"
      ;;
    --*=*)
      apply_kv "${token%%=*}" "${token#*=}"
      ;;
    *=*)
      apply_kv "${token%%=*}" "${token#*=}"
      ;;
    --*)
      echo "error: unknown option '$token'" >&2
      usage >&2
      exit 2
      ;;
    *)
      ONLY="$(append_csv "$ONLY" "$token")"
      ;;
  esac
  shift
done

# ── Structured results output ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

detect_bench_os() {
  case "${RUNNER_OS:-$(uname -s)}" in
    Linux) echo "linux" ;;
    Darwin) echo "macos" ;;
    Windows | MINGW* | MSYS* | CYGWIN*) echo "windows" ;;
    *) uname -s | tr '[:upper:]' '[:lower:]' ;;
  esac
}

detect_bench_arch() {
  case "$(uname -m)" in
    arm64) echo "aarch64" ;;
    x86_64 | amd64) echo "x86-64" ;;
    *) uname -m ;;
  esac
}

RUN_DATE="$(date -u +"%Y-%m-%d")"
RUN_TIME="$(date -u +"%H_%M_%S")"
RUN_OS="$(detect_bench_os)"
RUN_ARCH="$(detect_bench_arch)"
RUN_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
RUN_MODE="${RSCRYPTO_BENCH_MODE:-local}"
RESULTS_DIR="$REPO_ROOT/benchmark_results/$RUN_DATE/$RUN_OS/$RUN_ARCH"

BENCH_CRATES="$CRATES" \
  BENCH_BENCHES="$BENCHES" \
  BENCH_ONLY="$ONLY" \
  BENCH_FILTER="$FILTER" \
  BENCH_QUICK="$QUICK" \
  BENCH_WARMUP_MS="$WARMUP_MS" \
  BENCH_MEASURE_MS="$MEASURE_MS" \
  BENCH_SAMPLE_SIZE="$SAMPLE_SIZE" \
  BENCH_PROFILE_TIME_SECS="$PROFILE_TIME_SECS" \
  BENCH_OUTPUT_DIR="$OUTPUT_DIR" \
  BENCH_CLEAN="$CLEAN" \
  BENCH_RESULTS_DIR="$RESULTS_DIR" \
  BENCH_RUN_DATE="$RUN_DATE" \
  BENCH_RUN_TIME="$RUN_TIME" \
  BENCH_RUN_OS="$RUN_OS" \
  BENCH_RUN_ARCH="$RUN_ARCH" \
  BENCH_RUN_COMMIT="$RUN_COMMIT" \
  BENCH_RUN_MODE="$RUN_MODE" \
  scripts/ci/run-bench.sh
