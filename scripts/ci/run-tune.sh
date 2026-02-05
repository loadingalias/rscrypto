#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${TUNE_OUTPUT_DIR:-tune-results}"
mkdir -p "$OUT_DIR"

ONLY_INPUT="${TUNE_ONLY:-}"
CRATES_INPUT="${TUNE_CRATES:-}"
WARMUP_INPUT="${TUNE_WARMUP_MS:-}"
MEASURE_INPUT="${TUNE_MEASURE_MS:-}"
APPLY_INPUT="${TUNE_APPLY:-false}"
SELF_CHECK_INPUT="${TUNE_SELF_CHECK:-false}"
ENFORCE_TARGETS_INPUT="${TUNE_ENFORCE_TARGETS:-false}"
QUICK_INPUT="${TUNE_QUICK:-true}"

ARGS=()
if [[ "$QUICK_INPUT" == "true" ]]; then
  ARGS+=(--quick)
fi
if [[ -n "$ONLY_INPUT" ]]; then
  ARGS+=(--only "$ONLY_INPUT")
fi
if [[ -n "$CRATES_INPUT" ]]; then
  ARGS+=(--crate "$CRATES_INPUT")
fi
if [[ -n "$WARMUP_INPUT" ]]; then
  ARGS+=(--warmup-ms "$WARMUP_INPUT")
fi
if [[ -n "$MEASURE_INPUT" ]]; then
  ARGS+=(--measure-ms "$MEASURE_INPUT")
fi
if [[ "$SELF_CHECK_INPUT" == "true" ]]; then
  ARGS+=(--self-check)
fi
if [[ "$ENFORCE_TARGETS_INPUT" == "true" ]]; then
  ARGS+=(--enforce-targets)
fi
if [[ "$APPLY_INPUT" == "true" ]]; then
  ARGS+=(--apply)
fi

ARGS+=(--report-dir "$OUT_DIR")

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running rscrypto-tune"
echo "RUSTFLAGS: -C target-cpu=native"
echo "Args: ${ARGS[*]}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

RUSTC_WRAPPER='' RUSTFLAGS='-C target-cpu=native' \
  cargo run -p tune --release --bin rscrypto-tune -- "${ARGS[@]}" 2>&1 | tee "$OUT_DIR/rscrypto-tune.txt"

if [[ "$APPLY_INPUT" == "true" ]]; then
  git diff > "$OUT_DIR/patch.diff" || true
fi

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo ""
    echo "## Tuning Results"
    echo ""
    echo '```'
    cat "$OUT_DIR/rscrypto-tune.txt"
    echo '```'
  } >> "$GITHUB_STEP_SUMMARY"
fi
