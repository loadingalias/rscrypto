#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${TUNE_OUTPUT_DIR:-tune-results}"
mkdir -p "$OUT_DIR"

to_bool() {
  local raw="${1:-}"
  raw="$(echo "$raw" | tr '[:upper:]' '[:lower:]' | xargs)"
  case "$raw" in
    1|true|yes|on|y) echo "true" ;;
    0|false|no|off|n|"") echo "false" ;;
    *)
      echo "warning: unrecognized boolean value '$1'; treating as false" >&2
      echo "false"
      ;;
  esac
}

ONLY_INPUT="${TUNE_ONLY:-}"
CRATES_INPUT="${TUNE_CRATES:-}"
WARMUP_INPUT="${TUNE_WARMUP_MS:-}"
MEASURE_INPUT="${TUNE_MEASURE_MS:-}"
CHECKSUM_WARMUP_INPUT="${TUNE_CHECKSUM_WARMUP_MS:-}"
CHECKSUM_MEASURE_INPUT="${TUNE_CHECKSUM_MEASURE_MS:-}"
HASH_WARMUP_INPUT="${TUNE_HASH_WARMUP_MS:-}"
HASH_MEASURE_INPUT="${TUNE_HASH_MEASURE_MS:-}"
REPEATS_INPUT="${TUNE_REPEATS:-}"
AGGREGATION_INPUT="${TUNE_AGGREGATION:-auto}"
APPLY_INPUT="$(to_bool "${TUNE_APPLY:-false}")"
SELF_CHECK_INPUT="$(to_bool "${TUNE_SELF_CHECK:-false}")"
ENFORCE_TARGETS_INPUT="$(to_bool "${TUNE_ENFORCE_TARGETS:-false}")"
QUICK_INPUT="$(to_bool "${TUNE_QUICK:-false}")"
MEASURE_ONLY_INPUT="$(to_bool "${TUNE_MEASURE_ONLY:-false}")"
DERIVE_FROM_INPUT="${TUNE_DERIVE_FROM:-}"

if [[ -z "$REPEATS_INPUT" ]]; then
  if [[ "$QUICK_INPUT" == "true" ]]; then
    REPEATS_INPUT="1"
  else
    REPEATS_INPUT="3"
  fi
fi

if ! [[ "$REPEATS_INPUT" =~ ^[0-9]+$ ]] || [[ "$REPEATS_INPUT" -lt 1 ]]; then
  echo "error: TUNE_REPEATS must be an integer >= 1 (got '$REPEATS_INPUT')" >&2
  exit 2
fi

AGGREGATION_INPUT="$(echo "$AGGREGATION_INPUT" | tr '[:upper:]' '[:lower:]' | xargs)"
case "$AGGREGATION_INPUT" in
  auto|median) ;;
  trimmed-mean|trimmed_mean|trimmedmean|trim) AGGREGATION_INPUT="trimmed-mean" ;;
  *)
    echo "error: TUNE_AGGREGATION must be one of auto|median|trimmed-mean (got '$AGGREGATION_INPUT')" >&2
    exit 2
    ;;
esac

if [[ -z "$DERIVE_FROM_INPUT" && "$QUICK_INPUT" == "true" && "$APPLY_INPUT" == "true" ]]; then
  echo "error: quick mode is developer preview only and cannot be used with --apply" >&2
  exit 2
fi

MEASURE_ARGS=()
if [[ "$QUICK_INPUT" == "true" ]]; then
  MEASURE_ARGS+=(--quick)
fi
if [[ -n "$ONLY_INPUT" ]]; then
  MEASURE_ARGS+=(--only "$ONLY_INPUT")
fi
if [[ -n "$CRATES_INPUT" ]]; then
  MEASURE_ARGS+=(--crate "$CRATES_INPUT")
fi
if [[ -n "$WARMUP_INPUT" ]]; then
  MEASURE_ARGS+=(--warmup-ms "$WARMUP_INPUT")
fi
if [[ -n "$MEASURE_INPUT" ]]; then
  MEASURE_ARGS+=(--measure-ms "$MEASURE_INPUT")
fi
if [[ -n "$CHECKSUM_WARMUP_INPUT" ]]; then
  MEASURE_ARGS+=(--checksum-warmup-ms "$CHECKSUM_WARMUP_INPUT")
fi
if [[ -n "$CHECKSUM_MEASURE_INPUT" ]]; then
  MEASURE_ARGS+=(--checksum-measure-ms "$CHECKSUM_MEASURE_INPUT")
fi
if [[ -n "$HASH_WARMUP_INPUT" ]]; then
  MEASURE_ARGS+=(--hash-warmup-ms "$HASH_WARMUP_INPUT")
fi
if [[ -n "$HASH_MEASURE_INPUT" ]]; then
  MEASURE_ARGS+=(--hash-measure-ms "$HASH_MEASURE_INPUT")
fi
MEASURE_ARGS+=(--repeats "$REPEATS_INPUT" --aggregate "$AGGREGATION_INPUT")

DERIVE_ARGS=()
if [[ "$SELF_CHECK_INPUT" == "true" ]]; then
  DERIVE_ARGS+=(--self-check)
fi
if [[ "$ENFORCE_TARGETS_INPUT" == "true" ]]; then
  DERIVE_ARGS+=(--enforce-targets)
fi
if [[ "$APPLY_INPUT" == "true" ]]; then
  DERIVE_ARGS+=(--apply)
fi
DERIVE_ARGS+=(--report-dir "$OUT_DIR")

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running rscrypto-tune"
echo "RUSTFLAGS: -C target-cpu=native"
echo "Quick mode: $QUICK_INPUT"
if [[ "$QUICK_INPUT" == "true" ]]; then
  echo "Mode: developer preview (not for dispatch decisions)"
else
  echo "Mode: full-quality (dispatch eligible)"
fi
echo "Repeats: $REPEATS_INPUT"
echo "Aggregation: $AGGREGATION_INPUT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

RAW_ARTIFACT_PATH="$OUT_DIR/raw-results.json"
LOG_PATH="$OUT_DIR/rscrypto-tune.txt"
: > "$LOG_PATH"

if [[ -n "$DERIVE_FROM_INPUT" ]]; then
  RAW_ARTIFACT_PATH="$DERIVE_FROM_INPUT"
  echo "Derive source: existing raw artifact ($RAW_ARTIFACT_PATH)"
else
  echo "Measure args: ${MEASURE_ARGS[*]} --measure-only --raw-output $RAW_ARTIFACT_PATH"
  RUSTC_WRAPPER='' RUSTFLAGS='-C target-cpu=native' \
    cargo run -p tune --release --bin rscrypto-tune -- "${MEASURE_ARGS[@]}" --measure-only --raw-output "$RAW_ARTIFACT_PATH" \
    2>&1 | tee -a "$LOG_PATH"
fi

if [[ "$MEASURE_ONLY_INPUT" == "true" ]]; then
  echo "Measurement-only mode enabled; skipping derivation/apply."
  exit 0
fi

echo "Derive args: --derive-from $RAW_ARTIFACT_PATH ${DERIVE_ARGS[*]}"
RUSTC_WRAPPER='' RUSTFLAGS='-C target-cpu=native' \
  cargo run -p tune --release --bin rscrypto-tune -- --derive-from "$RAW_ARTIFACT_PATH" "${DERIVE_ARGS[@]}" \
  2>&1 | tee -a "$LOG_PATH"

if [[ "$APPLY_INPUT" == "true" ]]; then
  TARGET_PATCH_PATHS=("crates/tune/generated")
  PATCH_PATHS_FILE="$OUT_DIR/patch-paths.txt"
  PATCH_PATH="$OUT_DIR/patch.diff"
  MANIFEST_PATH="$OUT_DIR/apply-manifest.json"
  : > "$PATCH_PATHS_FILE"

  append_unique_path() {
    local path="$1"
    [[ -z "$path" ]] && return 0
    if ! grep -Fxq "$path" "$PATCH_PATHS_FILE"; then
      echo "$path" >> "$PATCH_PATHS_FILE"
    fi
  }

  while IFS= read -r path; do
    append_unique_path "$path"
  done < <(git diff --name-only -- "${TARGET_PATCH_PATHS[@]}" || true)

  while IFS= read -r path; do
    append_unique_path "$path"
  done < <(git ls-files --others --exclude-standard -- "${TARGET_PATCH_PATHS[@]}" || true)

  {
    # Capture tracked apply-target changes only.
    git diff --binary --full-index -- "${TARGET_PATCH_PATHS[@]}" || true

    # `rscrypto-tune --apply` emits new files under crates/tune/generated.
    # Plain `git diff` ignores untracked files, so include them explicitly.
    git ls-files --others --exclude-standard -z -- "${TARGET_PATCH_PATHS[@]}" \
      | while IFS= read -r -d '' path; do
          git diff --binary --full-index --no-index /dev/null "$path" || true
        done
  } > "$PATCH_PATH"

  if [[ -s "$PATCH_PATH" ]]; then
    echo "Generated non-empty patch: $PATCH_PATH"
  else
    echo "Generated empty patch (no apply-target changes): $PATCH_PATH"
  fi

  RAW_ARTIFACT_PATH="$RAW_ARTIFACT_PATH" \
  RESULTS_ARTIFACT_PATH="$OUT_DIR/results.json" \
  PATCH_PATH="$PATCH_PATH" \
  PATCH_PATHS_FILE="$PATCH_PATHS_FILE" \
  MANIFEST_PATH="$MANIFEST_PATH" \
  python3 - <<'PY'
import datetime as dt
import json
import os
from pathlib import Path


def load_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


raw_path = os.environ.get("RAW_ARTIFACT_PATH", "")
results_path = os.environ.get("RESULTS_ARTIFACT_PATH", "")
patch_path = os.environ.get("PATCH_PATH", "")
paths_file = os.environ.get("PATCH_PATHS_FILE", "")
manifest_path = os.environ.get("MANIFEST_PATH", "")

raw = load_json(raw_path) if raw_path else None
results = load_json(results_path) if results_path else None

changed_files = []
if paths_file and Path(paths_file).exists():
    changed_files = [line.strip() for line in Path(paths_file).read_text().splitlines() if line.strip()]

changed_profiles = []
for path in changed_files:
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 6 and parts[0] == "crates" and parts[1] == "tune" and parts[2] == "generated":
        changed_profiles.append(
            {
                "family": parts[3],
                "profile": parts[4],
                "tune_kind_file": parts[5],
                "path": path,
            }
        )

manifest = {
    "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "source_run": {
        "raw_artifact": raw_path,
        "results_artifact": results_path,
        "patch_artifact": patch_path,
        "raw_schema_version": raw.get("schema_version") if isinstance(raw, dict) else None,
        "quick_mode": raw.get("quick_mode") if isinstance(raw, dict) else None,
        "run_count": raw.get("run_count") if isinstance(raw, dict) else None,
        "aggregation": raw.get("aggregation") if isinstance(raw, dict) else None,
        "measurement_timestamp": raw.get("timestamp") if isinstance(raw, dict) else None,
        "derived_timestamp": results.get("timestamp") if isinstance(results, dict) else None,
        "platform": raw.get("platform") if isinstance(raw, dict) else None,
    },
    "changed_files": changed_files,
    "changed_profiles": changed_profiles,
}

Path(manifest_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

  echo "Generated apply manifest: $MANIFEST_PATH"
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
