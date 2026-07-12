#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET_DIR="$ROOT/target/zeroize-evidence"
MANIFEST="$ROOT/Cargo.toml"

RUSTC_WRAPPER="" CARGO_TARGET_DIR="$TARGET_DIR" cargo rustc \
  --manifest-path "$MANIFEST" \
  --release \
  --lib \
  --no-default-features \
  --features alloc,aes-gcm,diag \
  -- \
  -Ccodegen-units=1 \
  --emit=mir,llvm-ir,asm

latest_artifact() {
  local extension="$1"
  local latest=""
  local candidate

  shopt -s nullglob
  for candidate in "$TARGET_DIR"/release/deps/rscrypto-*."$extension"; do
    if [[ -z "$latest" || "$candidate" -nt "$latest" ]]; then
      latest="$candidate"
    fi
  done
  printf '%s\n' "$latest"
}

LLVM_IR="$(latest_artifact ll)"
MIR="$(latest_artifact mir)"
ASSEMBLY="$(latest_artifact s)"

if [[ -z "$LLVM_IR" || -z "$MIR" || -z "$ASSEMBLY" ]]; then
  echo "zeroize evidence artifacts missing" >&2
  exit 1
fi

for symbol in \
  diag_zeroize_fixed_stack \
  diag_zeroize_variable_heap \
  diag_zeroize_hex_success \
  diag_zeroize_hex_error; do
  if ! grep -q "@$symbol" "$LLVM_IR"; then
    echo "zeroize LLVM evidence missing symbol: $symbol" >&2
    exit 1
  fi
  if ! grep -q "$symbol" "$ASSEMBLY"; then
    echo "zeroize assembly evidence missing symbol: $symbol" >&2
    exit 1
  fi
  if ! grep -q "$symbol" "$MIR"; then
    echo "zeroize MIR evidence missing symbol: $symbol" >&2
    exit 1
  fi
done

for symbol in diag_zeroize_fixed_stack diag_zeroize_variable_heap; do
  FUNCTION_IR="$(sed -n "/define .*@$symbol(/,/^}/p" "$LLVM_IR")"
  VOLATILE_STORES="$(grep -c 'store volatile .* 0' <<<"$FUNCTION_IR" || true)"
  if [[ "$VOLATILE_STORES" -lt 1 ]]; then
    echo "zeroize LLVM evidence has no volatile zero store in $symbol" >&2
    exit 1
  fi
done

HEX_ERROR_IR="$(sed -n '/define .*@diag_zeroize_hex_error(/,/^}/p' "$LLVM_IR")"
HEX_SUCCESS_IR="$(sed -n '/define .*@diag_zeroize_hex_success(/,/^}/p' "$LLVM_IR")"
HEX_FROM_STR="$(sed -n 's/.*call .*@\([^ (]*FromStr8from_str\).*/\1/p' <<<"$HEX_ERROR_IR" | head -n 1)"
if [[ -z "$HEX_FROM_STR" || "$HEX_SUCCESS_IR" != *"@$HEX_FROM_STR"* ]]; then
  echo "zeroize hex evidence does not share the audited secret parser" >&2
  exit 1
fi

HEX_FROM_STR_IR="$(sed -n "/define .*@$HEX_FROM_STR(/,/^}/p" "$LLVM_IR")"
HEX_VOLATILE_STORES="$(grep -c 'store volatile .* 0' <<<"$HEX_FROM_STR_IR" || true)"
if [[ "$HEX_VOLATILE_STORES" -lt 2 ]]; then
  echo "zeroize secret parser evidence does not cover both success and error cleanup" >&2
  exit 1
fi

FIXED_ASSEMBLY="$(awk '
  $0 == "diag_zeroize_fixed_stack:" || $0 == "_diag_zeroize_fixed_stack:" { found = 1 }
  found && emitted && $0 ~ /^[[:space:]]*\.globl[[:space:]]/ { exit }
  found { print; emitted = 1 }
' "$ASSEMBLY")"
HOST_ARCH="$(rustc -vV | sed -n 's/^host: \([^-]*\).*/\1/p')"
case "$HOST_ARCH" in
  aarch64)
    if ! grep -Eq 'st(p|r)[[:space:]].*\[sp' <<<"$FIXED_ASSEMBLY" || \
       ! grep -Eq 'str(b|h)?[[:space:]].*(wzr|xzr)' <<<"$FIXED_ASSEMBLY"; then
      echo "zeroize assembly evidence does not show the fixed-size stack spill and wipe" >&2
      exit 1
    fi
    ;;
  x86_64)
    if ! grep -Eq '%rsp' <<<"$FIXED_ASSEMBLY" || ! grep -Eq "mov[bql]?[[:space:]]+\\\$0" <<<"$FIXED_ASSEMBLY"; then
      echo "zeroize assembly evidence does not show the fixed-size stack spill and wipe" >&2
      exit 1
    fi
    ;;
esac

echo "zeroize compiler evidence ok"
