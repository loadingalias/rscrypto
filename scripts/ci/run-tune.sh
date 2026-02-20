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

array_contains() {
  local needle="${1:-}"
  shift
  local item
  for item in "$@"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

normalize_csv() {
  local raw="${1:-}"
  local -a parts=()
  local -a normalized=()
  local token
  IFS=',' read -r -a parts <<< "$raw"
  for token in "${parts[@]-}"; do
    token="$(echo "$token" | xargs)"
    if [[ -z "$token" ]]; then
      continue
    fi
    token="$(echo "$token" | tr '[:upper:]' '[:lower:]')"
    if ! array_contains "$token" "${normalized[@]-}"; then
      normalized+=("$token")
    fi
  done

  if [[ -z "${normalized[*]-}" ]]; then
    echo ""
  else
    (IFS=','; echo "${normalized[*]-}")
  fi
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
BOUNDARY_INPUT="$(to_bool "${TUNE_BLAKE3_BOUNDARY:-false}")"
BOUNDARY_WARMUP_INPUT="${TUNE_BOUNDARY_WARMUP_MS:-}"
BOUNDARY_MEASURE_INPUT="${TUNE_BOUNDARY_MEASURE_MS:-}"
ONLY_INPUT="$(normalize_csv "$ONLY_INPUT")"
CRATES_INPUT="$(normalize_csv "$CRATES_INPUT")"

if [[ -z "$REPEATS_INPUT" ]]; then
  REPEATS_INPUT="1"
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

if [[ -n "$DERIVE_FROM_INPUT" && "$MEASURE_ONLY_INPUT" == "true" ]]; then
  echo "error: TUNE_MEASURE_ONLY cannot be combined with TUNE_DERIVE_FROM" >&2
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
echo "BLAKE3 boundary capture: $BOUNDARY_INPUT"
echo "Repeats: $REPEATS_INPUT"
echo "Aggregation: $AGGREGATION_INPUT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

RAW_ARTIFACT_PATH="$OUT_DIR/raw-results.json"
LOG_PATH="$OUT_DIR/rscrypto-tune.txt"
: > "$LOG_PATH"

if [[ -n "$DERIVE_FROM_INPUT" ]]; then
  RAW_ARTIFACT_PATH="$DERIVE_FROM_INPUT"
  echo "Derive source: existing raw artifact ($RAW_ARTIFACT_PATH)"
elif [[ "$MEASURE_ONLY_INPUT" == "true" ]]; then
  echo "Measure args: ${MEASURE_ARGS[*]} --measure-only --raw-output $RAW_ARTIFACT_PATH"
  RUSTC_WRAPPER='' RUSTFLAGS='-C target-cpu=native' \
    cargo run -p tune --release --bin rscrypto-tune -- "${MEASURE_ARGS[@]}" --measure-only --raw-output "$RAW_ARTIFACT_PATH" \
    2>&1 | tee -a "$LOG_PATH"
  echo "Measurement-only mode enabled; skipping derivation/apply."
  exit 0
fi

if [[ -n "$DERIVE_FROM_INPUT" ]]; then
  echo "Derive args: --derive-from $RAW_ARTIFACT_PATH ${DERIVE_ARGS[*]}"
  RUSTC_WRAPPER='' RUSTFLAGS='-C target-cpu=native' \
    cargo run -p tune --release --bin rscrypto-tune -- --derive-from "$RAW_ARTIFACT_PATH" "${DERIVE_ARGS[@]}" \
    2>&1 | tee -a "$LOG_PATH"
else
  RUN_ARGS=("${MEASURE_ARGS[@]}" "${DERIVE_ARGS[@]}" --raw-output "$RAW_ARTIFACT_PATH")
  echo "Run args: ${RUN_ARGS[*]}"
  RUSTC_WRAPPER='' RUSTFLAGS='-C target-cpu=native' \
    cargo run -p tune --release --bin rscrypto-tune -- "${RUN_ARGS[@]}" \
    2>&1 | tee -a "$LOG_PATH"
fi

# Keep the raw artifact inside OUT_DIR so uploaded artifacts are self-contained.
if [[ "$RAW_ARTIFACT_PATH" != "$OUT_DIR/raw-results.json" ]]; then
  cp "$RAW_ARTIFACT_PATH" "$OUT_DIR/raw-results.json"
  RAW_ARTIFACT_PATH="$OUT_DIR/raw-results.json"
fi

if [[ "$APPLY_INPUT" == "true" ]]; then
  TARGET_PATCH_PATHS=(
    "crates/checksum/src/dispatch.rs"
    "crates/hashes/src/crypto"
    "crates/hashes/src/fast"
  )
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

if [[ "$BOUNDARY_INPUT" == "true" ]]; then
  BOUNDARY_DIR="$OUT_DIR/boundary"
  mkdir -p "$BOUNDARY_DIR"

  boundary_cmd_base=(cargo run -p tune --release --bin rscrypto-blake3-boundary --)
  if [[ -n "$BOUNDARY_WARMUP_INPUT" ]]; then
    boundary_cmd_base+=(--warmup-ms "$BOUNDARY_WARMUP_INPUT")
  fi
  if [[ -n "$BOUNDARY_MEASURE_INPUT" ]]; then
    boundary_cmd_base+=(--measure-ms "$BOUNDARY_MEASURE_INPUT")
  fi

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Running BLAKE3 boundary capture"
  echo "Boundary output dir: $BOUNDARY_DIR"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  echo "Boundary run: auto"
  RUSTC_WRAPPER='' RUSTFLAGS='-C target-cpu=native' \
    "${boundary_cmd_base[@]}" --output "$BOUNDARY_DIR/auto.csv" 2>&1 | tee -a "$LOG_PATH"

  forced_kernels=("portable")
  case "$(uname -m)" in
    x86_64|amd64)
      forced_kernels+=("sse41" "avx2" "avx512")
      ;;
    aarch64|arm64)
      forced_kernels+=("neon")
      ;;
  esac

  for kernel in "${forced_kernels[@]}"; do
    echo "Boundary run: forced kernel=$kernel"
    RUSTC_WRAPPER='' RUSTFLAGS='-C target-cpu=native' \
      "${boundary_cmd_base[@]}" --force-kernel "$kernel" --output "$BOUNDARY_DIR/$kernel.csv" 2>&1 | tee -a "$LOG_PATH"
  done

  summary_inputs=("$BOUNDARY_DIR/auto.csv")
  for kernel in "${forced_kernels[@]}"; do
    summary_inputs+=("$BOUNDARY_DIR/$kernel.csv")
  done

  python3 scripts/bench/blake3-boundary-report.py "${summary_inputs[@]}" \
    | tee "$BOUNDARY_DIR/summary.txt" \
    | tee -a "$LOG_PATH" >/dev/null
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

prune_tune_artifacts() {
  local -a keep=("$OUT_DIR/raw-results.json")
  local -a retained=()
  local path
  local keep_path
  local should_keep

  if [[ "$APPLY_INPUT" == "true" ]]; then
    keep+=("$OUT_DIR/patch.diff" "$OUT_DIR/apply-manifest.json")
  else
    keep+=("$OUT_DIR/results.json" "$OUT_DIR/summary.txt")
  fi

  shopt -s nullglob
  for path in "$OUT_DIR"/*; do
    [[ -f "$path" ]] || continue
    should_keep="false"
    for keep_path in "${keep[@]}"; do
      if [[ "$path" == "$keep_path" ]]; then
        should_keep="true"
        break
      fi
    done
    if [[ "$should_keep" == "true" ]]; then
      retained+=("$(basename "$path")")
    else
      rm -f "$path"
    fi
  done
  shopt -u nullglob

  if [[ -n "${retained[*]-}" ]]; then
    echo "Retained artifacts: ${retained[*]}"
  else
    echo "warning: no retained artifacts found in $OUT_DIR"
  fi
}

prune_tune_artifacts
