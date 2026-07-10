#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/ct/artifacts.sh [--target <triple>] [--profile release]

Build the CT harness and capture provenance, LLVM IR, assembly, object files,
object disassembly, symbol maps, and artifact hashes under target/ct/.
EOF
}

PROFILE="release"
TARGET="$(rustc -vV | awk '/^host:/ { print $2 }')"

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

if [[ "$PROFILE" != "release" ]]; then
  echo "only --profile release is supported for CT artifacts today" >&2
  exit 2
fi

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

HOST="$(rustc -vV | awk '/^host:/ { print $2 }')"
target_env_name() {
  local suffix="$1"
  local upper_target="${TARGET^^}"
  upper_target="${upper_target//-/_}"
  printf 'CARGO_TARGET_%s_%s\n' "$upper_target" "$suffix"
}

rustflags_env="$(target_env_name RUSTFLAGS)"
if [[ -z "${!rustflags_env:-}" && "$TARGET" == "s390x-unknown-linux-gnu" ]]; then
  export "$rustflags_env=-C target-feature=+vector"
fi

if [[ "$TARGET" != "$HOST" && "$TARGET" == *linux* ]]; then
  linker_env="$(target_env_name LINKER)"

  if [[ -z "${!linker_env:-}" && "$TARGET" == *-linux-musl && "$(uname -m)" == "${TARGET%%-*}" ]] \
    && command -v musl-gcc >/dev/null 2>&1; then
    export "$linker_env=musl-gcc"
  elif [[ -z "${!linker_env:-}" && -x "$ROOT/scripts/check/zig-cc.sh" ]] && command -v zig >/dev/null 2>&1; then
    export "$linker_env=$ROOT/scripts/check/zig-cc.sh"
    export ZIG_CC_TARGET="${ZIG_CC_TARGET:-$TARGET}"
  fi

  if [[ -z "${!rustflags_env:-}" ]]; then
    case "$TARGET" in
      x86_64-unknown-linux-gnu)
        export "$rustflags_env=-C target-cpu=x86-64"
        ;;
      aarch64-unknown-linux-gnu)
        export "$rustflags_env=-C target-feature=+neon,+lse"
        ;;
    esac
  fi
fi

SYSROOT="$(rustc --print sysroot)"
LLVM_BIN="$SYSROOT/lib/rustlib/$HOST/bin"
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
LLVM_SIZE="$(resolve_tool "${LLVM_SIZE:-$LLVM_BIN/llvm-size}")"

for tool in "$LLVM_OBJDUMP" "$LLVM_NM" "$LLVM_SIZE"; do
  if [[ ! -x "$tool" ]]; then
    echo "missing LLVM tool: $tool" >&2
    exit 1
  fi
done

PYTHON="$("$ROOT/scripts/ct/python.sh" --print)"

if [[ -z "$PYTHON" ]]; then
  echo "python3 or python is required to generate CT provenance" >&2
  exit 1
fi

OUT_DIR="$ROOT/target/ct/$TARGET/$PROFILE"
ARTIFACT_DIR="$OUT_DIR/artifacts"
BUILD_TARGET_DIR="$ROOT/target/ct-build"
rm -rf "$OUT_DIR"
mkdir -p "$ARTIFACT_DIR"

echo "building CT harness for $TARGET ($PROFILE)"
cargo rustc \
  --locked \
  --manifest-path tools/ct-harness/Cargo.toml \
  --target-dir "$BUILD_TARGET_DIR" \
  --target "$TARGET" \
  --release \
  --lib \
  -- \
  --emit=llvm-ir,asm,obj

DEPS_DIR="$BUILD_TARGET_DIR/$TARGET/$PROFILE/deps"
EMITTED=()
while IFS= read -r artifact; do
  EMITTED+=("$artifact")
done < <(
  find "$DEPS_DIR" -maxdepth 1 -type f \
    \( -name 'rscrypto_ct_harness*.ll' \
    -o -name 'rscrypto_ct_harness*.s' \
    -o -name 'rscrypto_ct_harness*.o' \
    -o -name 'rscrypto_ct_harness*.obj' \) \
    | sort
)

if [[ ${#EMITTED[@]} -eq 0 ]]; then
  echo "no CT harness emitted artifacts found in $DEPS_DIR" >&2
  exit 1
fi

for artifact in "${EMITTED[@]}"; do
  cp "$artifact" "$ARTIFACT_DIR/"
done

OBJECTS=()
while IFS= read -r obj; do
  OBJECTS+=("$obj")
done < <(find "$ARTIFACT_DIR" -maxdepth 1 -type f \( -name '*.o' -o -name '*.obj' \) | sort)
if [[ ${#OBJECTS[@]} -eq 0 ]]; then
  echo "no CT harness object file found in $ARTIFACT_DIR" >&2
  exit 1
fi

for obj in "${OBJECTS[@]}"; do
  base="$(basename "$obj")"
  objdump_args=(--disassemble --reloc --demangle)
  if [[ "$TARGET" == "s390x-unknown-linux-gnu" ]]; then
    objdump_args+=(--mattr=+vector)
  fi
  "$LLVM_OBJDUMP" "${objdump_args[@]}" "$obj" > "$ARTIFACT_DIR/$base.disasm.txt"
  "$LLVM_NM" --defined-only --demangle "$obj" > "$ARTIFACT_DIR/$base.symbols.txt"
  "$LLVM_SIZE" "$obj" > "$ARTIFACT_DIR/$base.size.txt"
done

if command -v rustfilt >/dev/null 2>&1; then
  for symbols in "$ARTIFACT_DIR"/*.symbols.txt; do
    rustfilt < "$symbols" > "$symbols.rustfilt.txt"
  done
fi

"$PYTHON" scripts/ct/asm_heuristics.py \
  --target "$TARGET" \
  --profile "$PROFILE" \
  --artifact-dir "$ARTIFACT_DIR" \
  --out-dir "$OUT_DIR"

"$PYTHON" scripts/ct/provenance.py \
  --target "$TARGET" \
  --profile "$PROFILE" \
  --artifact-dir "$ARTIFACT_DIR" \
  --out-dir "$OUT_DIR" \
  --build-target-dir "$BUILD_TARGET_DIR" \
  --backend llvm \
  --features std,full,parallel

echo "CT artifacts written to $OUT_DIR"
