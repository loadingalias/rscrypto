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
CARGO_ARGS=(--locked --manifest-path "$ROOT/tools/ct-dudect/Cargo.toml" --target-dir "$BUILD_TARGET_DIR" --target "$TARGET")
if [[ "$PROFILE" == "release" ]]; then
  CARGO_ARGS+=(--release)
elif [[ "$PROFILE" != "debug" ]]; then
  echo "unsupported dudect profile: $PROFILE" >&2
  exit 2
fi

linker_log_candidate="$(mktemp "$OUT_DIR/.dudect-linker-command.XXXXXXXX")"
cargo rustc "${CARGO_ARGS[@]}" --bin rscrypto-ct-dudect -- --print link-args 2>&1 | tee "$linker_log_candidate"
link_command_count=$(grep -c '"-o"' "$linker_log_candidate" || true)
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
LLVM_BIN="$SYSROOT/lib/rustlib/$HOST_TARGET/bin"
LLVM_OBJDUMP="${LLVM_OBJDUMP:-$LLVM_BIN/llvm-objdump}"
LLVM_NM="${LLVM_NM:-$LLVM_BIN/llvm-nm}"
for tool in "$LLVM_OBJDUMP" "$LLVM_NM"; do
  if [[ ! -x "$tool" ]]; then
    echo "missing LLVM tool: $tool" >&2
    exit 1
  fi
done
"$LLVM_OBJDUMP" --disassemble --reloc --demangle "$BINARY_PATH" > "$BINARY_DISASM_PATH"
"$LLVM_NM" --defined-only --demangle "$BINARY_PATH" > "$BINARY_SYMBOLS_PATH"

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
  --command "$COMMAND" \
  --binary "$BINARY_PATH" \
  --binary-disassembly "$BINARY_DISASM_PATH" \
  --binary-symbols "$BINARY_SYMBOLS_PATH" \
  --linker-command-log "$LINKER_COMMAND_PATH"
