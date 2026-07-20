#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/ct/artifacts.sh [--target <triple>] [--profile release]

Build the CT harness and capture provenance, LLVM IR, assembly, pre-link objects,
the final linked equality binary, disassembly, symbol maps, and artifact hashes
under target/ct/.
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
BUILD_TARGET_DIR="$ROOT/target/ct-build/$TARGET/$PROFILE"
DEPS_DIR="$BUILD_TARGET_DIR/$TARGET/$PROFILE/deps"
rm -rf "$OUT_DIR"
rm -rf "$BUILD_TARGET_DIR"
mkdir -p "$ARTIFACT_DIR"

# The dedicated target/profile build directory was removed above. Cargo does
# not fingerprint extra `--emit` outputs, so reusing it could report a fresh
# build after those outputs were removed and silently consume stale evidence.

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

LIB_EMITTED=()
while IFS= read -r artifact; do
  LIB_EMITTED+=("$artifact")
done < <(
  find "$DEPS_DIR" -maxdepth 1 -type f \
    \( -name 'rscrypto_ct_harness*.ll' \
    -o -name 'rscrypto_ct_harness*.s' \
    -o -name 'rscrypto_ct_harness*.o' \
    -o -name 'rscrypto_ct_harness*.obj' \) \
    | sort
)

if [[ ${#LIB_EMITTED[@]} -eq 0 ]]; then
  echo "no CT harness emitted artifacts found in $DEPS_DIR" >&2
  exit 1
fi
if [[ ${#LIB_EMITTED[@]} -ne 3 ]]; then
  echo "expected one fresh CT harness each for LLVM IR, assembly, and object output; found ${#LIB_EMITTED[@]}" >&2
  printf '  %s\n' "${LIB_EMITTED[@]}" >&2
  exit 1
fi

for artifact in "${LIB_EMITTED[@]}"; do
  cp "$artifact" "$ARTIFACT_DIR/"
done

LINK_LOG="$ARTIFACT_DIR/linker-command.txt"
LINK_MAP="$ARTIFACT_DIR/rscrypto-ct-evidence.link-map.txt"
FINAL_LINK_ARGS=()
if [[ "$TARGET" == *linux* ]]; then
  linker_env="$(target_env_name LINKER)"
  if [[ "${!linker_env:-}" == "$ROOT/scripts/check/zig-cc.sh" ]]; then
    export RSCRYPTO_CT_LINK_MAP="$LINK_MAP"
    FINAL_LINK_ARGS+=("-C" "link-arg=-Wl,--print-map")
  else
    FINAL_LINK_ARGS+=("-C" "link-arg=-Wl,--Map=$LINK_MAP")
  fi
fi
echo "building final linked equality evidence binary for $TARGET ($PROFILE)"
cargo rustc \
  --color never \
  --locked \
  --manifest-path tools/ct-harness/Cargo.toml \
  --target-dir "$BUILD_TARGET_DIR" \
  --target "$TARGET" \
  --release \
  --bin rscrypto-ct-evidence \
  -- \
  --emit=llvm-ir,asm,obj,link \
  --print link-args \
  "${FINAL_LINK_ARGS[@]}" \
  2>&1 | tee "$LINK_LOG"

link_command_count=$(grep -c '"-o"' "$LINK_LOG" || true)
if [[ "$link_command_count" -ne 1 ]]; then
  echo "expected exactly one final equality linker command; found $link_command_count" >&2
  exit 1
fi
if [[ "$TARGET" == *linux* && ! -s "$LINK_MAP" ]]; then
  echo "final equality linker map is missing or empty: $LINK_MAP" >&2
  exit 1
fi

BIN_EMITTED=()
while IFS= read -r artifact; do
  BIN_EMITTED+=("$artifact")
done < <(
  find "$DEPS_DIR" -maxdepth 1 -type f \
    \( -name 'rscrypto_ct_evidence*.ll' \
    -o -name 'rscrypto_ct_evidence*.s' \
    -o -name 'rscrypto_ct_evidence*.o' \
    -o -name 'rscrypto_ct_evidence*.obj' \) \
    | sort
)

if [[ ${#BIN_EMITTED[@]} -ne 3 ]]; then
  echo "expected one fresh equality binary each for LLVM IR, assembly, and object output; found ${#BIN_EMITTED[@]}" >&2
  printf '  %s\n' "${BIN_EMITTED[@]}" >&2
  exit 1
fi

for artifact in "${BIN_EMITTED[@]}"; do
  cp "$artifact" "$ARTIFACT_DIR/"
done

FINAL_BINARY="$BUILD_TARGET_DIR/$TARGET/$PROFILE/rscrypto-ct-evidence"
if [[ -f "$FINAL_BINARY.exe" ]]; then
  FINAL_BINARY="$FINAL_BINARY.exe"
fi
if [[ ! -f "$FINAL_BINARY" ]]; then
  echo "final linked equality evidence binary missing: $FINAL_BINARY" >&2
  exit 1
fi
cp "$FINAL_BINARY" "$ARTIFACT_DIR/"

OBJECTS=()
while IFS= read -r obj; do
  OBJECTS+=("$obj")
done < <(find "$ARTIFACT_DIR" -maxdepth 1 -type f \( -name '*.o' -o -name '*.obj' \) | sort)
if [[ ${#OBJECTS[@]} -ne 2 ]]; then
  echo "expected exactly two fresh CT harness object files in $ARTIFACT_DIR; found ${#OBJECTS[@]}" >&2
  exit 1
fi

for obj in "${OBJECTS[@]}"; do
  base="$(basename "$obj")"
  objdump_args=(--disassemble --reloc --demangle)
  if [[ "$TARGET" == "s390x-unknown-linux-gnu" ]]; then
    objdump_args+=(--mattr=+vector)
  fi
  "$LLVM_OBJDUMP" "${objdump_args[@]}" "$obj" > "$ARTIFACT_DIR/$base.disasm.txt"
  "$LLVM_NM" --defined-only "$obj" > "$ARTIFACT_DIR/$base.raw-symbols.txt"
  "$LLVM_NM" --defined-only --demangle "$obj" > "$ARTIFACT_DIR/$base.symbols.txt"
  "$LLVM_SIZE" "$obj" > "$ARTIFACT_DIR/$base.size.txt"
done

linked_binary="$ARTIFACT_DIR/$(basename "$FINAL_BINARY")"
linked_base="$(basename "$linked_binary")"
raw_disassembly="$ARTIFACT_DIR/$linked_base.binary.raw-disasm.txt"
nm_symbols="$ARTIFACT_DIR/$linked_base.binary.nm-symbols.txt"
"$LLVM_OBJDUMP" --disassemble --reloc --demangle "$linked_binary" > "$raw_disassembly"
"$LLVM_NM" --defined-only --demangle "$linked_binary" > "$nm_symbols" || true
"$LLVM_SIZE" "$linked_binary" > "$ARTIFACT_DIR/$linked_base.binary.size.txt"
indirect_symbols=""
if [[ "$TARGET" == *apple-darwin ]]; then
  indirect_symbols="$ARTIFACT_DIR/$linked_base.binary.indirect-symbols.txt"
  "$LLVM_OBJDUMP" --macho --indirect-symbols "$linked_binary" > "$indirect_symbols"
fi

symbolizer_args=(
  --artifact-dir "$ARTIFACT_DIR"
  --raw-disassembly "$raw_disassembly"
  --nm-symbols "$nm_symbols"
  --out-disassembly "$ARTIFACT_DIR/$linked_base.binary.disasm.txt"
  --out-symbols "$ARTIFACT_DIR/$linked_base.binary.symbols.txt"
)
if [[ -f "$LINK_MAP" ]]; then
  symbolizer_args+=(--link-map "$LINK_MAP")
fi
if [[ -n "$indirect_symbols" ]]; then
  symbolizer_args+=(--indirect-symbols "$indirect_symbols")
fi
"$PYTHON" scripts/ct/symbolize_linked_binary.py "${symbolizer_args[@]}"

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
  --features std,full,parallel,diag \
  --linker-command-log "$LINK_LOG"

echo "CT artifacts written to $OUT_DIR"
