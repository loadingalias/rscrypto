#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET=""
PROFILE="release"
SAMPLES="${RSCRYPTO_CT_DUDECT_SAMPLES:-20000}"
THRESHOLD="${RSCRYPTO_CT_DUDECT_THRESHOLD:-10.0}"
FILTER=""
SMOKE=0
PYTHON="${PYTHON:-}"

usage() {
  cat <<'USAGE'
usage: scripts/ct/dudect.sh [--target TRIPLE] [--profile release] [--samples N] [--threshold T] [--filter CASE] [--smoke]

Runs rscrypto's empirical dudect timing lane and writes:
  target/ct/<target>/<profile>/dudect/dudect-report.json

Notes:
  --target records the host target for evidence placement. Cross-target dudect
  requires a physical runner for that target and is intentionally not emulated.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --samples)
      SAMPLES="$2"
      shift 2
      ;;
    --threshold)
      THRESHOLD="$2"
      shift 2
      ;;
    --filter)
      FILTER="$2"
      shift 2
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$TARGET" ]]; then
  TARGET="$(rustc -vV | awk -F': ' '/^host:/ {print $2}')"
fi

HOST_TARGET="$(rustc -vV | awk -F': ' '/^host:/ {print $2}')"
target_runs_on_host() {
  local target="$1"
  local host="$2"
  [[ "$target" == "$host" ]] && return 0
  case "$host:$target" in
    x86_64-unknown-linux-gnu:x86_64-unknown-linux-musl) return 0 ;;
    aarch64-unknown-linux-gnu:aarch64-unknown-linux-musl) return 0 ;;
    *) return 1 ;;
  esac
}

if ! target_runs_on_host "$TARGET" "$HOST_TARGET"; then
  echo "dudect target must match physical host target: requested $TARGET, host is $HOST_TARGET" >&2
  exit 2
fi

if [[ "$SMOKE" == "1" && "$SAMPLES" == "${RSCRYPTO_CT_DUDECT_SAMPLES:-20000}" ]]; then
  SAMPLES=2000
fi

target_env_name() {
  local suffix="$1"
  local upper_target="${TARGET^^}"
  upper_target="${upper_target//-/_}"
  printf 'CARGO_TARGET_%s_%s\n' "$upper_target" "$suffix"
}

if [[ "$TARGET" != "$HOST_TARGET" && "$TARGET" == *-linux-musl && "$(uname -m)" == "${TARGET%%-*}" ]]; then
  linker_env="$(target_env_name LINKER)"
  if [[ -z "${!linker_env:-}" ]] && command -v musl-gcc >/dev/null 2>&1; then
    export "$linker_env=musl-gcc"
  fi
fi

OUT_DIR="$ROOT/target/ct/$TARGET/$PROFILE/dudect"
STDOUT_PATH="$OUT_DIR/dudect.stdout.txt"
CSV_PATH="$OUT_DIR/dudect-raw.csv"
REPORT_PATH="$OUT_DIR/dudect-report.json"
mkdir -p "$OUT_DIR"
rm -f "$STDOUT_PATH" "$CSV_PATH" "$REPORT_PATH"

CARGO_ARGS=(run --manifest-path "$ROOT/tools/ct-dudect/Cargo.toml" --target-dir "$ROOT/target/ct-dudect-build" --target "$TARGET")
if [[ "$PROFILE" == "release" ]]; then
  CARGO_ARGS+=(--release)
elif [[ "$PROFILE" != "debug" ]]; then
  echo "unsupported dudect profile: $PROFILE" >&2
  exit 2
fi
CARGO_ARGS+=(-- --out "$CSV_PATH")
if [[ -n "$FILTER" ]]; then
  CARGO_ARGS+=(--filter "$FILTER")
fi

COMMAND="RSCRYPTO_CT_DUDECT_SAMPLES=$SAMPLES cargo ${CARGO_ARGS[*]}"
echo "$COMMAND"
(
  cd "$ROOT"
  RSCRYPTO_CT_DUDECT_SAMPLES="$SAMPLES" cargo "${CARGO_ARGS[@]}"
) | tee "$STDOUT_PATH"

PYTHON="$("$ROOT/scripts/ct/python.sh" --print)"

if [[ -z "$PYTHON" ]]; then
  echo "python3 or python is required to generate the dudect report" >&2
  exit 1
fi

"$PYTHON" "$ROOT/scripts/ct/dudect_report.py" \
  --stdout "$STDOUT_PATH" \
  --csv "$CSV_PATH" \
  --out "$REPORT_PATH" \
  --target "$TARGET" \
  --profile "$PROFILE" \
  --threshold "$THRESHOLD" \
  --samples "$SAMPLES" \
  --command "$COMMAND"
