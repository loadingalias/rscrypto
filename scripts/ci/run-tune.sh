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

# `--apply` for blake3 requires stream-profile results.
# If caller asks for `--only blake3`, auto-include the required stream variants
# so apply can succeed without forcing users to remember the full list.
if [[ "$APPLY_INPUT" == "true" && -n "$ONLY_INPUT" ]]; then
  declare -a only_items=()
  IFS=',' read -ra raw_only_items <<< "$ONLY_INPUT"
  for raw in "${raw_only_items[@]}"; do
    item="$(echo "$raw" | xargs)"
    [[ -n "$item" ]] && only_items+=("$item")
  done

  has_blake3=false
  has_stream64=false
  has_stream4k=false
  for item in "${only_items[@]}"; do
    case "$item" in
      blake3) has_blake3=true ;;
      blake3-stream64|blake3-stream64-keyed|blake3-stream64-derive) has_stream64=true ;;
      blake3-stream4k|blake3-stream4k-keyed|blake3-stream4k-derive) has_stream4k=true ;;
    esac
  done

  if [[ "$has_blake3" == "true" && ( "$has_stream64" == "false" || "$has_stream4k" == "false" ) ]]; then
    echo "note: --apply + --only=blake3 requires stream tuning results; adding blake3 stream variants automatically"
    only_items+=(
      "blake3-stream64"
      "blake3-stream64-keyed"
      "blake3-stream64-derive"
      "blake3-stream4k"
      "blake3-stream4k-keyed"
      "blake3-stream4k-derive"
    )

    # De-duplicate while preserving stable order.
    declare -A seen=()
    declare -a deduped=()
    for item in "${only_items[@]}"; do
      if [[ -z "${seen[$item]+x}" ]]; then
        seen["$item"]=1
        deduped+=("$item")
      fi
    done
    ONLY_INPUT="$(IFS=','; echo "${deduped[*]}")"
  fi
fi

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
  {
    # Capture tracked changes first.
    git diff --binary --full-index || true

    # `rscrypto-tune --apply` emits new files under crates/tune/generated.
    # Plain `git diff` ignores untracked files, so include them explicitly.
    git ls-files --others --exclude-standard -z -- crates/tune/generated \
      | while IFS= read -r -d '' path; do
          git diff --binary --full-index --no-index /dev/null "$path" || true
        done
  } > "$OUT_DIR/patch.diff"

  if [[ -s "$OUT_DIR/patch.diff" ]]; then
    echo "Generated non-empty patch: $OUT_DIR/patch.diff"
  else
    echo "Generated empty patch (no tracked or generated-file changes): $OUT_DIR/patch.diff"
  fi
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
