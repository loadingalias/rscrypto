#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/bench/bench.sh [selector ...] [key=value ...]

Selectors:
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
    crate|crates)
      CRATES="$(append_csv "$CRATES" "$value")"
      ;;
    bench|benches)
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
    -h|--help)
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
scripts/ci/run-bench.sh
