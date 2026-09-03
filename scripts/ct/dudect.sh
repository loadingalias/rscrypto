#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET=""
PROFILE="release"
SAMPLES="${RSCRYPTO_CT_DUDECT_SAMPLES:-20000}"
THRESHOLD="${RSCRYPTO_CT_DUDECT_THRESHOLD:-10.0}"
FILTER=""
SMOKE=0

usage() {
  cat <<'USAGE'
usage: scripts/ct/dudect.sh [--target TRIPLE] [--profile release] [--samples N] [--threshold T] [--filter CASE] [--smoke]

Runs rscrypto's empirical dudect timing lane and writes:
  target/ct/<target>/<profile>/dudect/dudect-report.json

Filtered runs also preserve their report, raw samples, and stdout under:
  target/ct/<target>/<profile>/dudect/cases/<filter>/

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

if [[ -n "$FILTER" ]] && \
  [[ ! "$FILTER" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || "$FILTER" == *..* ]]; then
  echo "invalid DudeCT case filter for evidence path: $FILTER" >&2
  exit 2
fi

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
BINARY_PATH="$OUT_DIR/rscrypto-ct-dudect"
LINKER_COMMAND_PATH="$OUT_DIR/dudect-linker-command.txt"
BINARY_DISASM_PATH="$OUT_DIR/rscrypto-ct-dudect.binary.disasm.txt"
BINARY_SYMBOLS_PATH="$OUT_DIR/rscrypto-ct-dudect.binary.symbols.txt"
mkdir -p "$OUT_DIR"
rm -f "$STDOUT_PATH" "$CSV_PATH" "$REPORT_PATH"

BUILD_TARGET_DIR="$ROOT/target/ct-dudect-build/$TARGET/$PROFILE"
if [[ ! -s "$LINKER_COMMAND_PATH" ]]; then
  rm -rf "$BUILD_TARGET_DIR"
fi
CARGO_ARGS=(--manifest-path "$ROOT/tools/ct-dudect/Cargo.toml" --target-dir "$BUILD_TARGET_DIR" --target "$TARGET")
if [[ "$PROFILE" == "release" ]]; then
  CARGO_ARGS+=(--release)
elif [[ "$PROFILE" != "debug" ]]; then
  echo "unsupported dudect profile: $PROFILE" >&2
  exit 2
fi

linker_log_candidate="$(mktemp "$OUT_DIR/.dudect-linker-command.XXXXXXXX")"
cargo rustc --locked "${CARGO_ARGS[@]}" --bin rscrypto-ct-dudect -- --emit=obj,link --print link-args 2>&1 | tee "$linker_log_candidate"
link_command_count=$(grep -Ec '"-o"|"/OUT:' "$linker_log_candidate" || true)
if [[ "$link_command_count" -gt 1 ]]; then
  echo "expected at most one DudeCT linker command; found $link_command_count" >&2
  rm -f "$linker_log_candidate"
  exit 1
fi
if [[ "$link_command_count" -eq 1 ]]; then
  mv "$linker_log_candidate" "$LINKER_COMMAND_PATH"
else
  rm -f "$linker_log_candidate"
fi
if [[ ! -s "$LINKER_COMMAND_PATH" ]]; then
  echo "DudeCT linker command was not captured" >&2
  exit 1
fi

BUILT_BINARY="$BUILD_TARGET_DIR/$TARGET/$PROFILE/rscrypto-ct-dudect"
if [[ -f "$BUILT_BINARY.exe" ]]; then
  BUILT_BINARY="$BUILT_BINARY.exe"
  BINARY_PATH="$BINARY_PATH.exe"
fi
if [[ ! -f "$BUILT_BINARY" ]]; then
  echo "DudeCT executable missing: $BUILT_BINARY" >&2
  exit 1
fi
cp "$BUILT_BINARY" "$BINARY_PATH"

SYSROOT="$(rustc --print sysroot)"
if command -v cygpath >/dev/null 2>&1; then
  SYSROOT="$(cygpath -u "$SYSROOT")"
fi
LLVM_BIN="$SYSROOT/lib/rustlib/$HOST_TARGET/bin"
resolve_tool() {
  local tool="$1"
  if [[ -x "$tool" ]]; then
    printf '%s\n' "$tool"
  elif [[ -x "$tool.exe" ]]; then
    printf '%s\n' "$tool.exe"
  else
    printf '%s\n' "$tool"
  fi
}

LLVM_OBJDUMP="$(resolve_tool "${LLVM_OBJDUMP:-$LLVM_BIN/llvm-objdump}")"
LLVM_NM="$(resolve_tool "${LLVM_NM:-$LLVM_BIN/llvm-nm}")"
for tool in "$LLVM_OBJDUMP" "$LLVM_NM"; do
  if [[ ! -x "$tool" ]]; then
    echo "missing LLVM tool: $tool" >&2
    exit 1
  fi
done
BINARY_OBJECT_PATH=""
BINARY_OBJECT_ARGS=()
if [[ "$TARGET" == *-windows-* ]]; then
  shopt -s nullglob
  binary_objects=("$BUILD_TARGET_DIR/$TARGET/$PROFILE/deps"/rscrypto_ct_dudect*.o)
  if [[ ${#binary_objects[@]} -ne 1 ]]; then
    echo "expected one preserved Windows DudeCT LTO object; found ${#binary_objects[@]}" >&2
    exit 1
  fi
  BINARY_OBJECT_PATH="$OUT_DIR/$(basename "${binary_objects[0]}")"
  cp "${binary_objects[0]}" "$BINARY_OBJECT_PATH"
  "$LLVM_OBJDUMP" --disassemble --reloc --demangle "$BINARY_OBJECT_PATH" > "$BINARY_DISASM_PATH"
  "$LLVM_NM" --defined-only --demangle "$BINARY_OBJECT_PATH" > "$BINARY_SYMBOLS_PATH"
  BINARY_OBJECT_ARGS=(--binary-object "$BINARY_OBJECT_PATH")
else
  "$LLVM_OBJDUMP" --disassemble --reloc --dynamic-reloc --demangle "$BINARY_PATH" > "$BINARY_DISASM_PATH"
  "$LLVM_NM" --defined-only --demangle "$BINARY_PATH" > "$BINARY_SYMBOLS_PATH"
fi

RUNNER_ARGS=(--out "$CSV_PATH")
if [[ -n "$FILTER" ]]; then
  RUNNER_ARGS+=(--filter "$FILTER")
fi

COMMAND="RSCRYPTO_CT_DUDECT_SAMPLES=$SAMPLES $BINARY_PATH ${RUNNER_ARGS[*]}"
echo "$COMMAND"
(
  cd "$ROOT"
  RSCRYPTO_CT_DUDECT_SAMPLES="$SAMPLES" "$BINARY_PATH" "${RUNNER_ARGS[@]}"
) | tee "$STDOUT_PATH"

PYTHON="$("$ROOT/scripts/lib/python.sh" --print)"

"$PYTHON" "$ROOT/scripts/ct/dudect_report.py" \
  --stdout "$STDOUT_PATH" \
  --csv "$CSV_PATH" \
  --out "$REPORT_PATH" \
  --target "$TARGET" \
  --profile "$PROFILE" \
  --threshold "$THRESHOLD" \
  --samples "$SAMPLES" \
  --command "$COMMAND" \
  --binary "$BINARY_PATH" \
  "${BINARY_OBJECT_ARGS[@]}" \
  --binary-disassembly "$BINARY_DISASM_PATH" \
  --binary-symbols "$BINARY_SYMBOLS_PATH" \
  --linker-command-log "$LINKER_COMMAND_PATH"

if [[ -n "$FILTER" ]]; then
  CASE_OUT_DIR="$OUT_DIR/cases/$FILTER"
  mkdir -p "$CASE_OUT_DIR"
  CASE_EVIDENCE_PATHS=(
    "$REPORT_PATH"
    "$CSV_PATH"
    "$STDOUT_PATH"
    "$BINARY_PATH"
    "$BINARY_DISASM_PATH"
    "$BINARY_SYMBOLS_PATH"
    "$LINKER_COMMAND_PATH"
  )
  if [[ -n "$BINARY_OBJECT_PATH" ]]; then
    CASE_EVIDENCE_PATHS+=("$BINARY_OBJECT_PATH")
  fi
  cp -- "${CASE_EVIDENCE_PATHS[@]}" "$CASE_OUT_DIR/"
  echo "dudect case evidence: $CASE_OUT_DIR"
fi
