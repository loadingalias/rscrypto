#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/tune/apply.sh [--out-dir DIR] [--allow-overlap] <artifact_or_raw> [more...]

Inputs:
  - Directory containing raw-results.json (preferred)
  - Path to raw-results.json

Examples:
  just tune-apply ~/Downloads/tuning-amd-zen4
  just tune-apply ~/Downloads/tuning-amd-zen4 ~/Downloads/tuning-intel-spr
  just tune-apply --out-dir target/tune/imported ~/Downloads/tuning-intel-icl
USAGE
}

OUT_DIR="target/tune/imported"
ALLOW_OVERLAP="false"

ARTIFACT_INPUTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --out-dir)
      shift
      if [[ $# -eq 0 ]]; then
        echo "error: --out-dir requires a value" >&2
        exit 2
      fi
      OUT_DIR="$1"
      ;;
    --allow-overlap)
      ALLOW_OVERLAP="true"
      ;;
    --*)
      echo "error: unknown option '$1'" >&2
      usage >&2
      exit 2
      ;;
    *)
      ARTIFACT_INPUTS+=("$1")
      ;;
  esac
  shift
done

if [[ ${#ARTIFACT_INPUTS[@]} -eq 0 ]]; then
  echo "error: provide at least one artifact directory or raw-results.json path" >&2
  usage >&2
  exit 2
fi

sanitize_name() {
  local raw="${1:-}"
  local name
  name="$(echo "$raw" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '-')"
  name="${name#-}"
  name="${name%-}"
  if [[ -z "$name" ]]; then
    name="artifact"
  fi
  echo "$name"
}

extract_changed_files() {
  local manifest_path="${1:-}"
  python3 - "$manifest_path" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
if not manifest_path.exists():
    raise SystemExit(0)

try:
    manifest = json.loads(manifest_path.read_text())
except Exception:
    raise SystemExit(0)

for path in manifest.get("changed_files", []):
    if isinstance(path, str) and path:
        print(path)
PY
}

RAW_PATHS=()
MANIFEST_PATHS=()
LABELS=()

for input_path in "${ARTIFACT_INPUTS[@]}"; do
  if [[ -d "$input_path" ]]; then
    raw_path="$input_path/raw-results.json"
    manifest_path="$input_path/apply-manifest.json"
    label="$(basename "$input_path")"
  elif [[ -f "$input_path" ]]; then
    if [[ "$(basename "$input_path")" != "raw-results.json" ]]; then
      echo "error: file input must be raw-results.json (got '$input_path')" >&2
      exit 2
    fi
    raw_path="$input_path"
    manifest_path="$(dirname "$input_path")/apply-manifest.json"
    label="$(basename "$(dirname "$input_path")")"
  else
    echo "error: input path does not exist: $input_path" >&2
    exit 2
  fi

  if [[ ! -f "$raw_path" ]]; then
    echo "error: missing raw-results.json at '$raw_path'" >&2
    exit 2
  fi

  RAW_PATHS+=("$raw_path")
  MANIFEST_PATHS+=("$manifest_path")
  LABELS+=("$label")
done

if [[ "$ALLOW_OVERLAP" != "true" ]]; then
  tmp_seen_paths="$(mktemp)"
  trap 'rm -f "$tmp_seen_paths"' EXIT
  for i in "${!RAW_PATHS[@]}"; do
    manifest_path="${MANIFEST_PATHS[$i]}"
    label="${LABELS[$i]}"

    if [[ ! -f "$manifest_path" ]]; then
      echo "warning: $label has no apply-manifest.json; overlap detection for this input is limited" >&2
      continue
    fi

    while IFS= read -r changed_path; do
      [[ -z "$changed_path" ]] && continue
      existing_owner="$(awk -F '\t' -v p="$changed_path" '$1 == p { print $2; exit }' "$tmp_seen_paths")"
      if [[ -n "$existing_owner" && "$existing_owner" != "$label" ]]; then
        echo "error: overlapping apply target '$changed_path' between '$existing_owner' and '$label'" >&2
        echo "hint: rerun with --allow-overlap if you intentionally want last-write-wins behavior" >&2
        exit 2
      fi
      if [[ -z "$existing_owner" ]]; then
        printf '%s\t%s\n' "$changed_path" "$label" >> "$tmp_seen_paths"
      fi
    done < <(extract_changed_files "$manifest_path")
  done
  rm -f "$tmp_seen_paths"
  trap - EXIT
fi

mkdir -p "$OUT_DIR"

echo "Applying tuning artifacts into dispatch tables"
echo "Output root: $OUT_DIR"

for i in "${!RAW_PATHS[@]}"; do
  raw_path="${RAW_PATHS[$i]}"
  label="$(sanitize_name "${LABELS[$i]}")"
  lane_out_dir="$OUT_DIR/$label"
  mkdir -p "$lane_out_dir"

  echo "- applying $raw_path"
  TUNE_OUTPUT_DIR="$lane_out_dir" \
  TUNE_DERIVE_FROM="$raw_path" \
  TUNE_APPLY=true \
  TUNE_SELF_CHECK=false \
  TUNE_ENFORCE_TARGETS=false \
  scripts/ci/run-tune.sh

done

echo "Done. Applied ${#RAW_PATHS[@]} artifact(s)."
