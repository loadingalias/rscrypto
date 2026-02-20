#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${TUNE_OUTPUT_DIR:-tune-results}"
BOUNDARY_DIR="$OUT_DIR/boundary"
WARMUP_MS="${TUNE_BOUNDARY_WARMUP_MS:-}"
MEASURE_MS="${TUNE_BOUNDARY_MEASURE_MS:-}"
CARGO_BUILD_JOBS_INPUT="${TUNE_CARGO_BUILD_JOBS:-4}"
LOG_PATH="$OUT_DIR/tune.log"

mkdir -p "$BOUNDARY_DIR"
: > "$LOG_PATH"

if ! [[ "$CARGO_BUILD_JOBS_INPUT" =~ ^[0-9]+$ ]] || [[ "$CARGO_BUILD_JOBS_INPUT" -lt 1 ]]; then
  echo "error: TUNE_CARGO_BUILD_JOBS must be an integer >= 1 (got '$CARGO_BUILD_JOBS_INPUT')" >&2
  exit 2
fi

boundary_cmd=(cargo run -p tune --release --bin rscrypto-blake3-boundary --)
[[ -n "$WARMUP_MS" ]] && boundary_cmd+=(--warmup-ms "$WARMUP_MS")
[[ -n "$MEASURE_MS" ]] && boundary_cmd+=(--measure-ms "$MEASURE_MS")

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running BLAKE3 boundary capture"
echo "Boundary output dir: $BOUNDARY_DIR"
echo "Cargo build jobs: $CARGO_BUILD_JOBS_INPUT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

run_capture() {
  local kernel="$1"
  local output="$2"
  local -a cmd=("${boundary_cmd[@]}")
  if [[ -n "$kernel" ]]; then
    cmd+=(--force-kernel "$kernel")
    echo "Boundary run: forced kernel=$kernel"
  else
    echo "Boundary run: auto"
  fi
  cmd+=(--output "$output")

  CARGO_BUILD_JOBS="$CARGO_BUILD_JOBS_INPUT" RUSTC_WRAPPER='' RUSTFLAGS='-C target-cpu=native'     "${cmd[@]}" 2>&1 | tee -a "$LOG_PATH"
}

run_capture "" "$BOUNDARY_DIR/auto.csv"

forced_kernels=(portable)
case "$(uname -m)" in
  x86_64|amd64) forced_kernels+=(sse41 avx2 avx512) ;;
  aarch64|arm64) forced_kernels+=(neon) ;;
esac

summary_inputs=("$BOUNDARY_DIR/auto.csv")
for kernel in "${forced_kernels[@]}"; do
  out="$BOUNDARY_DIR/$kernel.csv"
  run_capture "$kernel" "$out"
  summary_inputs+=("$out")
done

python3 scripts/bench/blake3-boundary-report.py "${summary_inputs[@]}"   | tee "$BOUNDARY_DIR/summary.txt"   | tee -a "$LOG_PATH" >/dev/null

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo ""
    echo "## Blake3 Boundary Results"
    echo ""
    echo '```'
    cat "$LOG_PATH"
    echo '```'
  } >> "$GITHUB_STEP_SUMMARY"
fi

echo "Retained artifacts: boundary/*.csv boundary/summary.txt tune.log"
