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
  --features alloc,aes-gcm,blake3,hmac,hmac-sha3,parallel,diag \
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
  diag_zeroize_fixed_move \
  diag_zeroize_early_return \
  diag_zeroize_variable_heap \
  diag_zeroize_hex_success \
  diag_zeroize_hex_error \
  diag_zeroize_blake3_drop \
  diag_zeroize_blake3_reuse \
  diag_zeroize_blake3_xof_move \
  diag_zeroize_blake3_xof_consume \
  diag_zeroize_blake3_thread_scratch \
  diag_zeroize_blake3_parallel_scratch \
  diag_zeroize_hmac_sha256_finalize \
  diag_zeroize_hmac_sha3_finalize; do
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

for symbol in \
  diag_zeroize_fixed_stack \
  diag_zeroize_fixed_move \
  diag_zeroize_early_return \
  diag_zeroize_variable_heap \
  diag_zeroize_hex_success \
  diag_zeroize_hex_error \
  diag_zeroize_blake3_drop \
  diag_zeroize_blake3_reuse \
  diag_zeroize_blake3_xof_move \
  diag_zeroize_blake3_xof_consume \
  diag_zeroize_blake3_thread_scratch \
  diag_zeroize_blake3_parallel_scratch \
  diag_zeroize_hmac_sha256_finalize \
  diag_zeroize_hmac_sha3_finalize; do
  FUNCTION_IR="$(sed -n "/define .*@$symbol(/,/^}/p" "$LLVM_IR")"
  VOLATILE_STORES="$(grep -c 'store volatile .* 0' <<<"$FUNCTION_IR" || true)"
  if [[ "$VOLATILE_STORES" -lt 1 ]]; then
    echo "zeroize LLVM evidence has no volatile zero store in $symbol" >&2
    exit 1
  fi
done

BLAKE3_DROP_WRAPPER="$(sed -n '/define .*@diag_zeroize_blake3_drop(/,/^}/p' "$LLVM_IR")"
BLAKE3_DROP_SYMBOL="$(sed -n 's/.*call .*@\([^ (]*drop_in_place[^ (]*Blake3[^ (]*\).*/\1/p' \
  <<<"$BLAKE3_DROP_WRAPPER" | head -n 1)"
if [[ -z "$BLAKE3_DROP_SYMBOL" ]]; then
  echo "zeroize LLVM evidence does not route BLAKE3 cleanup through its production Drop" >&2
  exit 1
fi

BLAKE3_DROP_IR="$(sed -n "/define .*@$BLAKE3_DROP_SYMBOL(/,/^}/p" "$LLVM_IR")"
BLAKE3_DROP_STORES="$(grep -c 'store volatile .* 0' <<<"$BLAKE3_DROP_IR" || true)"
if [[ "$BLAKE3_DROP_STORES" -lt 8 ]] || ! grep -q "$BLAKE3_DROP_SYMBOL" "$ASSEMBLY"; then
  echo "zeroize release evidence does not retain BLAKE3 owner and heap-scratch cleanup" >&2
  exit 1
fi

BLAKE3_REUSE_IR="$(sed -n '/define .*@diag_zeroize_blake3_reuse(/,/^}/p' "$LLVM_IR")"
BLAKE3_REUSE_DROPS="$(grep -c '^[[:space:]]*call .*drop_in_place.*Blake3' <<<"$BLAKE3_REUSE_IR" || true)"
if [[ "$BLAKE3_REUSE_DROPS" -lt 2 ]]; then
  echo "zeroize release evidence does not wipe both replaced and final BLAKE3 state" >&2
  exit 1
fi

BLAKE3_MOVE_IR="$(sed -n '/define .*@diag_zeroize_blake3_xof_move(/,/^}/p' "$LLVM_IR")"
if ! grep -q '@diag_zeroize_blake3_xof_consume' <<<"$BLAKE3_MOVE_IR"; then
  echo "zeroize release evidence does not retain the keyed XOF ownership move" >&2
  exit 1
fi

BLAKE3_THREAD_SCRATCH_IR="$(sed -n '/define .*@diag_zeroize_blake3_thread_scratch(/,/^}/p' "$LLVM_IR")"
if [[ "$(grep -c 'store volatile .* 0' <<<"$BLAKE3_THREAD_SCRATCH_IR" || true)" -lt 10 ]]; then
  echo "zeroize release evidence does not clear both BLAKE3 thread-local CV vectors" >&2
  exit 1
fi

BLAKE3_PARALLEL_SCRATCH_IR="$(sed -n '/define .*@diag_zeroize_blake3_parallel_scratch(/,/^}/p' "$LLVM_IR")"
if [[ "$(grep -c 'store volatile .* 0' <<<"$BLAKE3_PARALLEL_SCRATCH_IR" || true)" -lt 9 ]]; then
  echo "zeroize release evidence does not clear BLAKE3 per-state heap scratch" >&2
  exit 1
fi

HMAC_SHA3_IR="$(sed -n '/define .*@diag_zeroize_hmac_sha3_finalize(/,/^}/p' "$LLVM_IR")"
HMAC_SHA3_STORES="$(grep -c 'store volatile .* 0' <<<"$HMAC_SHA3_IR" || true)"
if [[ "$HMAC_SHA3_STORES" -lt 25 ]]; then
  echo "zeroize release evidence does not retain HMAC-SHA3 finalization cleanup" >&2
  exit 1
fi

HMAC_SHA256_IR="$(sed -n '/define .*@diag_zeroize_hmac_sha256_finalize(/,/^}/p' "$LLVM_IR")"
HMAC_SHA256_STORES="$(grep -c 'store volatile .* 0' <<<"$HMAC_SHA256_IR" || true)"
if [[ "$HMAC_SHA256_STORES" -lt 25 ]]; then
  echo "zeroize release evidence does not retain HMAC-SHA256 finalization cleanup" >&2
  exit 1
fi
HMAC_SHA256_FINALIZE_SYMBOL="$(sed -n 's/.*call .*@\([^ (]*HmacSha256[^ (]*Mac8finalize[^ (]*\).*/\1/p' \
  <<<"$HMAC_SHA256_IR" | head -n 1)"
HMAC_SHA256_FINALIZE_IR="$(sed -n "/define .*@$HMAC_SHA256_FINALIZE_SYMBOL(/,/^}/p" "$LLVM_IR")"
HMAC_SHA256_SECRET_FINALIZE_CALLS="$(grep -c 'call .*finalize_secret' <<<"$HMAC_SHA256_FINALIZE_IR" || true)"
HMAC_SHA256_SECRET_FINALIZE_SYMBOL="$(sed -n 's/.*call .*@\([^ (]*finalize_secret[^ (]*\).*/\1/p' \
  <<<"$HMAC_SHA256_FINALIZE_IR" | head -n 1)"
if [[ -z "$HMAC_SHA256_FINALIZE_SYMBOL" || "$HMAC_SHA256_SECRET_FINALIZE_CALLS" -lt 2 || \
  -z "$HMAC_SHA256_SECRET_FINALIZE_SYMBOL" ]]; then
  echo "zeroize release evidence does not route both HMAC-SHA256 snapshots through secret finalization" >&2
  exit 1
fi
HMAC_SHA256_SECRET_FINALIZE_IR="$(sed -n "/define .*@$HMAC_SHA256_SECRET_FINALIZE_SYMBOL(/,/^}/p" "$LLVM_IR")"
if [[ "$(grep -c 'store volatile .* 0' <<<"$HMAC_SHA256_SECRET_FINALIZE_IR" || true)" -lt 2 ]]; then
  echo "zeroize release evidence does not clear SHA-256 finalization snapshots" >&2
  exit 1
fi

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

function_assembly() {
  local symbol="$1"
  awk -v plain="$symbol:" -v apple="_$symbol:" '
    $0 == plain || $0 == apple { found = 1 }
    found && emitted && $0 ~ /^[[:space:]]*\.globl[[:space:]]/ { exit }
    found { print }
    found { emitted = 1 }
  ' "$ASSEMBLY"
}

FIXED_ASSEMBLY="$(function_assembly diag_zeroize_fixed_stack)"
HOST_ARCH="$(rustc -vV | sed -n 's/^host: \([^-]*\).*/\1/p')"
case "$HOST_ARCH" in
  aarch64)
    if ! grep -Eq 'st(p|r)[[:space:]].*\[sp' <<<"$FIXED_ASSEMBLY" || \
       ! grep -Eq 'str(b|h)?[[:space:]].*(wzr|xzr)' <<<"$FIXED_ASSEMBLY"; then
      echo "zeroize assembly evidence does not show the fixed-size stack spill and wipe" >&2
      exit 1
    fi
    for symbol in \
      diag_zeroize_fixed_move \
      diag_zeroize_early_return \
      diag_zeroize_variable_heap \
      diag_zeroize_hex_success \
      diag_zeroize_hex_error \
      diag_zeroize_blake3_xof_consume \
      diag_zeroize_blake3_thread_scratch \
      diag_zeroize_blake3_parallel_scratch \
      diag_zeroize_hmac_sha256_finalize \
      diag_zeroize_hmac_sha3_finalize; do
      FUNCTION_ASSEMBLY="$(function_assembly "$symbol")"
      if ! grep -Eq 'st(p|r)(b|h)?[[:space:]].*(wzr|xzr)' <<<"$FUNCTION_ASSEMBLY"; then
        echo "zeroize assembly evidence has no zero store in $symbol" >&2
        exit 1
      fi
    done
    ;;
  x86_64)
    if ! grep -Eq '%rsp' <<<"$FIXED_ASSEMBLY" || ! grep -Eq "mov[bql]?[[:space:]]+\\\$0" <<<"$FIXED_ASSEMBLY"; then
      echo "zeroize assembly evidence does not show the fixed-size stack spill and wipe" >&2
      exit 1
    fi
    for symbol in \
      diag_zeroize_fixed_move \
      diag_zeroize_early_return \
      diag_zeroize_variable_heap \
      diag_zeroize_hex_success \
      diag_zeroize_hex_error \
      diag_zeroize_blake3_xof_consume \
      diag_zeroize_blake3_thread_scratch \
      diag_zeroize_blake3_parallel_scratch \
      diag_zeroize_hmac_sha256_finalize \
      diag_zeroize_hmac_sha3_finalize; do
      FUNCTION_ASSEMBLY="$(function_assembly "$symbol")"
      if ! grep -Eq "mov[bql]?[[:space:]]+\\\$0" <<<"$FUNCTION_ASSEMBLY"; then
        echo "zeroize assembly evidence has no zero store in $symbol" >&2
        exit 1
      fi
    done
    ;;
esac

echo "zeroize compiler evidence ok"
