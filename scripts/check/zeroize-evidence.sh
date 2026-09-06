#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="$ROOT/Cargo.toml"
SCOPE="all"

if [[ $# -gt 0 ]]; then
  if [[ $# -ne 2 || "$1" != "--primitive" || "$2" != "p256-ecdh" ]]; then
    echo "usage: scripts/check/zeroize-evidence.sh [--primitive p256-ecdh]" >&2
    exit 2
  fi
  SCOPE="$2"
fi

TARGET_DIR="$ROOT/target/zeroize-evidence"
FEATURES="alloc,aegis256,aes-gcm,aes-siv,ascon-aead,blake3,chacha20poly1305,ecdsa-p256,ecdsa-p384,hkdf,hmac,hmac-sha3,ml-kem,p256-ecdh,parallel,rsa,diag"
SYMBOLS=(
  diag_zeroize_fixed_stack
  diag_zeroize_fixed_move
  diag_zeroize_fixed_fill_error
  diag_zeroize_early_return
  diag_zeroize_variable_heap
  diag_zeroize_variable_fill_error
  diag_zeroize_secret_string
  diag_zeroize_hex_success
  diag_zeroize_hex_error
  diag_zeroize_blake3_drop
  diag_zeroize_blake3_reuse
  diag_zeroize_blake3_xof_move
  diag_zeroize_blake3_xof_consume
  diag_zeroize_blake3_thread_scratch
  diag_zeroize_blake3_parallel_scratch
  diag_zeroize_hmac_sha256_finalize
  diag_zeroize_hmac_sha3_finalize
  diag_rsa_caller_random_signing_success
  diag_rsa_caller_random_signing_error
  diag_hkdf_sha256_derive_portable
  diag_hkdf_sha384_derive_portable
  diag_hkdf_sha512_derive_portable
  diag_poly1305_block_portable_digest
  diag_ascon_aead128_tag_portable
  diag_aegis256_update_portable
  diag_aes128gcm_ghash
  diag_aes256gcm_ghash
  diag_zeroize_aes128_header_protection
  diag_zeroize_aes256_header_protection
  diag_zeroize_chacha20_header_protection
  diag_zeroize_aes_siv_cmac256
  diag_zeroize_ecdsa_p256_public_blinding
  diag_zeroize_ecdsa_p256_signing_blinding
  diag_zeroize_ecdsa_p256_safegcd_scratch
  diag_zeroize_p256_ecdh_generation
  diag_zeroize_p256_ecdh_agreement
  diag_zeroize_ecdsa_p384_public_blinding
  diag_zeroize_ecdsa_p384_signing_blinding
  diag_zeroize_ecdsa_p384_safegcd_scratch
  diag_zeroize_mlkem_sha3_512
  diag_zeroize_mlkem_shake256_scalar
  diag_zeroize_mlkem_shake256_pair
  diag_zeroize_mlkem_shake256_quad
)
if [[ "$SCOPE" == "p256-ecdh" ]]; then
  TARGET_DIR="$TARGET_DIR/p256-ecdh"
  FEATURES="p256-ecdh,diag"
  SYMBOLS=(diag_zeroize_p256_ecdh_generation diag_zeroize_p256_ecdh_agreement)
fi

function_assembly() {
  local symbol="$1"
  awk -v plain="$symbol:" -v apple="_$symbol:" '
    $0 == plain || $0 == apple { found = 1 }
    found && emitted && $0 ~ /^[^[:space:].Ll][^:]*:$/ { exit }
    found { print }
    found { emitted = 1 }
  '
}

function_assembly_containing() {
  local fragment="$1"
  awk -v fragment="$fragment" '
    $0 ~ /^[^[:space:].Ll][^:]*:$/ && index($0, fragment) { found = 1 }
    found && emitted && $0 ~ /^[^[:space:].Ll][^:]*:$/ { exit }
    found { print }
    found { emitted = 1 }
  '
}

# Assembly evidence is a deliberate cold baseline; the custom target directory
# would bypass reuse, and the explicit switch keeps that boundary auditable.
CARGO_RAIL_CACHE=off CARGO_TARGET_DIR="$TARGET_DIR" cargo rustc \
  --locked \
  --manifest-path "$MANIFEST" \
  --release \
  --lib \
  --no-default-features \
  --features "$FEATURES" \
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

for symbol in "${SYMBOLS[@]}"; do
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

# The public P-256 ECDH wrappers remain out of line in release evidence, so
# checking only the diagnostic caller would miss their production owners. Pin
# the target-selected LLVM cleanup shape here. Portable public derivation clears
# two replaced projective states, the final projective state, and the scalar;
# portable agreement clears four doubled states, the added state, the final
# state, the scalar, and the consumed ephemeral secret. Native wrappers instead
# clear their scalar and output owners in Rust. Cleanup inside the embedded
# routines remains owned by the deterministic ASM provenance gate.
P256_ECDH_PUBLIC_IR="$(sed -n '/define .*p256_ecdh.*10public_key(/,/^}/p' "$LLVM_IR")"
P256_ECDH_AGREEMENT_IR="$(sed -n '/define .*p256_ecdh.*14diffie_hellman(/,/^}/p' "$LLVM_IR")"
HOST_TRIPLE="$(rustc -vV | sed -n 's/^host: //p')"
P256_ECDH_PUBLIC_NATIVE=false
P256_ECDH_AGREEMENT_NATIVE=false
case "$HOST_TRIPLE" in
  aarch64-apple-darwin | aarch64-*-linux-*)
    P256_ECDH_PUBLIC_NATIVE=true
    P256_ECDH_AGREEMENT_NATIVE=true
    ;;
  x86_64-*-linux-* | x86_64-pc-windows-msvc)
    P256_ECDH_PUBLIC_NATIVE=true
    P256_ECDH_AGREEMENT_NATIVE=true
    ;;
esac

if [[ "$P256_ECDH_PUBLIC_NATIVE" == true ]]; then
  if [[ -z "$P256_ECDH_PUBLIC_IR" ]] || \
    [[ "$P256_ECDH_PUBLIC_IR" != *"rscrypto_p256_scalarmulbase"* ]] || \
    [[ "$(grep -c 'store volatile i64 0' <<<"$P256_ECDH_PUBLIC_IR" || true)" -lt 4 ]] || \
    [[ "$(grep -c 'fence syncscope("singlethread") seq_cst' <<<"$P256_ECDH_PUBLIC_IR" || true)" -lt 1 ]]; then
    echo "zeroize release evidence does not clear the native P-256 ECDH public-derivation scalar" >&2
    exit 1
  fi
elif [[ -z "$P256_ECDH_PUBLIC_IR" ]] || \
  [[ "$(grep -c 'store volatile i64 0' <<<"$P256_ECDH_PUBLIC_IR" || true)" -lt 40 ]] || \
  [[ "$(grep -c 'fence syncscope("singlethread") seq_cst' <<<"$P256_ECDH_PUBLIC_IR" || true)" -lt 4 ]]; then
  echo "zeroize release evidence does not clear every portable P-256 ECDH public-derivation projective owner and scalar" >&2
  exit 1
fi

if [[ "$P256_ECDH_AGREEMENT_NATIVE" == true ]]; then
  if [[ -z "$P256_ECDH_AGREEMENT_IR" ]] || \
    [[ "$P256_ECDH_AGREEMENT_IR" != *"rscrypto_p256_scalarmul_alt"* ]] || \
    [[ "$(grep -c 'store volatile i64 0' <<<"$P256_ECDH_AGREEMENT_IR" || true)" -lt 13 ]] || \
    [[ "$(grep -c 'store volatile i8 0' <<<"$P256_ECDH_AGREEMENT_IR" || true)" -lt 2 ]] || \
    [[ "$(grep -c 'fence syncscope("singlethread") seq_cst' <<<"$P256_ECDH_AGREEMENT_IR" || true)" -lt 3 ]]; then
    echo "zeroize release evidence does not clear the native P-256 ECDH scalar, output, and consumed secret" >&2
    exit 1
  fi
elif [[ -z "$P256_ECDH_AGREEMENT_IR" ]] || \
  [[ "$(grep -c 'store volatile i64 0' <<<"$P256_ECDH_AGREEMENT_IR" || true)" -lt 77 ]] || \
  [[ "$(grep -c 'fence syncscope("singlethread") seq_cst' <<<"$P256_ECDH_AGREEMENT_IR" || true)" -lt 8 ]]; then
  echo "zeroize release evidence does not clear every portable P-256 ECDH agreement projective owner, scalar, and consumed secret" >&2
  exit 1
fi

for symbol in "${SYMBOLS[@]}"; do
  FUNCTION_IR="$(sed -n "/define .*@$symbol(/,/^}/p" "$LLVM_IR")"
  VOLATILE_STORES="$(grep -c 'store volatile .* 0' <<<"$FUNCTION_IR" || true)"
  if [[ "$VOLATILE_STORES" -lt 1 ]]; then
    echo "zeroize LLVM evidence has no volatile zero store in $symbol" >&2
    exit 1
  fi
done

P256_ECDH_PUBLIC_ASSEMBLY="$(function_assembly_containing 'P256EphemeralSecret10public_key' <"$ASSEMBLY")"
P256_ECDH_AGREEMENT_ASSEMBLY="$(function_assembly_containing 'P256EphemeralSecret14diffie_hellman' <"$ASSEMBLY")"
if [[ -z "$P256_ECDH_PUBLIC_ASSEMBLY" || -z "$P256_ECDH_AGREEMENT_ASSEMBLY" ]]; then
  echo "zeroize assembly evidence is missing the P-256 ECDH production operations" >&2
  exit 1
fi

AARCH64_ZERO_MEMORY_PATTERN='^[[:space:]]*st(p|u?r)(b|h)?[[:space:]]+(wzr|xzr)(,[[:space:]]*(wzr|xzr))?,[[:space:]]*\[[^]]+\]'
X86_ZERO_MEMORY_PATTERN='^[[:space:]]*mov[bql]?[[:space:]]+\$0,[[:space:]]*[[:alnum:]_+.-]*\([^)]*%[^)]*\)'
case "${HOST_TRIPLE%%-*}" in
  aarch64)
    if ! grep -Eq "$AARCH64_ZERO_MEMORY_PATTERN" <<<"$P256_ECDH_PUBLIC_ASSEMBLY" || \
      ! grep -Eq "$AARCH64_ZERO_MEMORY_PATTERN" <<<"$P256_ECDH_AGREEMENT_ASSEMBLY"; then
      echo "zeroize assembly evidence has no P-256 ECDH production-state wipe" >&2
      exit 1
    fi
    ;;
  x86_64)
    if ! grep -Eq "$X86_ZERO_MEMORY_PATTERN" <<<"$P256_ECDH_PUBLIC_ASSEMBLY" || \
      ! grep -Eq "$X86_ZERO_MEMORY_PATTERN" <<<"$P256_ECDH_AGREEMENT_ASSEMBLY"; then
      echo "zeroize assembly evidence has no P-256 ECDH production-state wipe" >&2
      exit 1
    fi
    ;;
esac

if [[ "$SCOPE" == "p256-ecdh" ]]; then
  echo "zeroize compiler evidence ok: p256-ecdh"
  exit 0
fi

for symbol in \
  diag_zeroize_aes128_header_protection \
  diag_zeroize_aes256_header_protection \
  diag_zeroize_chacha20_header_protection; do
  FUNCTION_IR="$(sed -n "/define .*@$symbol(/,/^}/p" "$LLVM_IR")"
  if [[ "$(grep -c 'store volatile .* 0' <<<"$FUNCTION_IR" || true)" -lt 2 ]] || \
    ! grep -q 'fence syncscope("singlethread") seq_cst' <<<"$FUNCTION_IR"; then
    echo "zeroize release evidence does not clear header-protection owners and materialized output in $symbol" >&2
    exit 1
  fi
done

POLY1305_IR="$(sed -n '/define .*@diag_poly1305_block_portable_digest(/,/^}/p' "$LLVM_IR")"
if [[ "$(grep -c 'store volatile i32 0' <<<"$POLY1305_IR" || true)" -lt 14 ]] || \
  ! grep -q 'fence syncscope("singlethread") seq_cst' <<<"$POLY1305_IR"; then
  echo "zeroize release evidence does not clear the complete portable Poly1305 state" >&2
  exit 1
fi

ASCON_IR="$(sed -n '/define .*@diag_ascon_aead128_tag_portable(/,/^}/p' "$LLVM_IR")"
ASCON_STATE_CLEANUP="$(sed -n '1,/fence syncscope("singlethread") seq_cst/p' <<<"$ASCON_IR")"
if [[ "$(grep -c 'store volatile i64 0' <<<"$ASCON_STATE_CLEANUP" || true)" -lt 5 ]] || \
  ! grep -q 'fence syncscope("singlethread") seq_cst' <<<"$ASCON_IR"; then
  echo "zeroize release evidence does not clear the complete portable Ascon-AEAD state" >&2
  exit 1
fi

AEGIS_IR="$(sed -n '/define .*@diag_aegis256_update_portable(/,/^}/p' "$LLVM_IR")"
if [[ "$(grep -c 'store volatile .* 0' <<<"$AEGIS_IR" || true)" -lt 3 ]] || \
  ! grep -q 'fence syncscope("singlethread") seq_cst' <<<"$AEGIS_IR"; then
  echo "zeroize release evidence does not retain the portable AEGIS-256 state wipe" >&2
  exit 1
fi

for symbol in diag_aes128gcm_ghash diag_aes256gcm_ghash; do
  FUNCTION_IR="$(sed -n "/define .*@$symbol(/,/^}/p" "$LLVM_IR")"
  if ! grep -q 'store volatile i128 0' <<<"$FUNCTION_IR" || \
    ! grep -q 'fence syncscope("singlethread") seq_cst' <<<"$FUNCTION_IR"; then
    echo "zeroize release evidence does not retain the GCM authentication-state wipe in $symbol" >&2
    exit 1
  fi
done

MLKEM_SHA3_IR="$(sed -n '/define .*@diag_zeroize_mlkem_sha3_512(/,/^}/p' "$LLVM_IR")"
MLKEM_SHA3_FINALIZE_SYMBOL="$(
  sed -n 's/.*call .*@\([^ (]*KeccakCoreImpl[^ (]*finalize_into_fixed[^ (]*\).*/\1/p' \
    <<<"$MLKEM_SHA3_IR" | head -n 1
)"
MLKEM_SHA3_FINALIZE_IR="$(sed -n "/define .*@$MLKEM_SHA3_FINALIZE_SYMBOL(/,/^}/p" "$LLVM_IR")"
if [[ "$(grep -c 'store volatile i64 0' <<<"$MLKEM_SHA3_IR" || true)" -lt 25 ]] || \
  [[ "$(grep -c 'store volatile i64 0' <<<"$MLKEM_SHA3_FINALIZE_IR" || true)" -lt 25 ]] || \
  ! grep -q 'fence syncscope("singlethread") seq_cst' <<<"$MLKEM_SHA3_IR" || \
  ! grep -q 'fence syncscope("singlethread") seq_cst' <<<"$MLKEM_SHA3_FINALIZE_IR"; then
  echo "zeroize release evidence does not clear ML-KEM SHA3-512 owner and finalization states" >&2
  exit 1
fi

MLKEM_SHAKE_SCALAR_IR="$(sed -n '/define .*@diag_zeroize_mlkem_shake256_scalar(/,/^}/p' "$LLVM_IR")"
MLKEM_SHAKE_SCALAR_SEED_SYMBOL="$(
  grep 'call .*MlKemShake256XofReader.*seeded_32_1' <<<"$MLKEM_SHAKE_SCALAR_IR" |
    grep -v 'quad' |
    sed -n 's/.*@\([^ (]*\).*/\1/p' |
    head -n 1
)"
MLKEM_SHAKE_SCALAR_SEED_IR="$(sed -n "/define .*@$MLKEM_SHAKE_SCALAR_SEED_SYMBOL(/,/^}/p" "$LLVM_IR")"
if [[ "$(grep -c 'store volatile i64 0' <<<"$MLKEM_SHAKE_SCALAR_IR" || true)" -lt 25 ]] || \
  [[ "$(grep -c 'store volatile i64 0' <<<"$MLKEM_SHAKE_SCALAR_SEED_IR" || true)" -lt 25 ]] || \
  ! grep -q 'fence syncscope("singlethread") seq_cst' <<<"$MLKEM_SHAKE_SCALAR_IR" || \
  ! grep -q 'fence syncscope("singlethread") seq_cst' <<<"$MLKEM_SHAKE_SCALAR_SEED_IR"; then
  echo "zeroize release evidence does not clear ML-KEM scalar SHAKE owner and seeded state" >&2
  exit 1
fi

MLKEM_SHAKE_PAIR_IR="$(sed -n '/define .*@diag_zeroize_mlkem_shake256_pair(/,/^}/p' "$LLVM_IR")"
if [[ "$(grep -c 'store volatile i64 0' <<<"$MLKEM_SHAKE_PAIR_IR" || true)" -lt 100 ]] || \
  ! grep -q 'fence syncscope("singlethread") seq_cst' <<<"$MLKEM_SHAKE_PAIR_IR"; then
  echo "zeroize release evidence does not clear both ML-KEM pair owners and seeded states" >&2
  exit 1
fi

MLKEM_SHAKE_QUAD_IR="$(sed -n '/define .*@diag_zeroize_mlkem_shake256_quad(/,/^}/p' "$LLVM_IR")"
MLKEM_SHAKE_QUAD_SEED_SYMBOL="$(
  sed -n 's/.*call .*@\([^ (]*MlKemShake256XofReader[^ (]*seeded_32_1_quad[^ (]*\).*/\1/p' \
    <<<"$MLKEM_SHAKE_QUAD_IR" | head -n 1
)"
MLKEM_SHAKE_QUAD_SEED_IR="$(sed -n "/define .*@$MLKEM_SHAKE_QUAD_SEED_SYMBOL(/,/^}/p" "$LLVM_IR")"
if [[ "$(grep -c 'store volatile i64 0' <<<"$MLKEM_SHAKE_QUAD_IR" || true)" -lt 100 ]] || \
  [[ "$(grep -c 'store volatile i64 0' <<<"$MLKEM_SHAKE_QUAD_SEED_IR" || true)" -lt 100 ]] || \
  ! grep -q 'fence syncscope("singlethread") seq_cst' <<<"$MLKEM_SHAKE_QUAD_IR" || \
  ! grep -q 'fence syncscope("singlethread") seq_cst' <<<"$MLKEM_SHAKE_QUAD_SEED_IR"; then
  echo "zeroize release evidence does not clear all ML-KEM quad owners and seeded states" >&2
  exit 1
fi

BLAKE3_DROP_WRAPPER="$(sed -n '/define .*@diag_zeroize_blake3_drop(/,/^}/p' "$LLVM_IR")"
BLAKE3_DROP_SYMBOL="$(sed -En 's/.*call .*@([^ (]*drop_(in_place|glue)[^ (]*Blake3[^ (]*).*/\1/p' \
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
BLAKE3_REUSE_DROPS="$(grep -Ec '^[[:space:]]*call .*drop_(in_place|glue).*Blake3' <<<"$BLAKE3_REUSE_IR" || true)"
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

for symbol in \
  diag_hkdf_sha256_derive_portable \
  diag_hkdf_sha384_derive_portable \
  diag_hkdf_sha512_derive_portable; do
  FUNCTION_IR="$(sed -n "/define .*@$symbol(/,/^}/p" "$LLVM_IR")"
  if [[ "$(grep -c 'store volatile .* 0' <<<"$FUNCTION_IR" || true)" -lt 8 ]] || \
    ! grep -q 'fence syncscope("singlethread") seq_cst' <<<"$FUNCTION_IR"; then
    echo "zeroize release evidence does not retain HKDF prefix and expansion-scratch cleanup in $symbol" >&2
    exit 1
  fi
done

ECDSA_P256_IR="$(sed -n '/define .*@diag_zeroize_ecdsa_p256_platform_scratch(/,/^}/p' "$LLVM_IR")"
if [[ -z "$ECDSA_P256_IR" ]] || \
  ! grep -q 'diag_zeroize_ecdsa_p256_platform_scratch' "$MIR" || \
  ! grep -q 'diag_zeroize_ecdsa_p256_platform_scratch' "$ASSEMBLY" || \
  [[ "$(grep -c 'store volatile i64 0' <<<"$ECDSA_P256_IR" || true)" -lt 26 ]] || \
  ! grep -q 'fence syncscope("singlethread") seq_cst' <<<"$ECDSA_P256_IR"; then
  echo "zeroize release evidence does not clear P-256 accelerated reduction and inversion scratch" >&2
  exit 1
fi

if [[ "$(rustc -vV | sed -n 's/^host: //p')" == aarch64-* ]]; then
  ECDSA_P384_IR="$(sed -n '/define .*@diag_zeroize_ecdsa_p384_platform_scratch(/,/^}/p' "$LLVM_IR")"
  ECDSA_P384_INVERSE_CLEANUP="$(
    awk '/fence syncscope\("singlethread"\) seq_cst/{fences++; next} fences == 0' \
      <<<"$ECDSA_P384_IR"
  )"
  ECDSA_P384_REDUCED_CLEANUP="$(
    awk '/fence syncscope\("singlethread"\) seq_cst/{fences++; next} fences == 1' \
      <<<"$ECDSA_P384_IR"
  )"
  ECDSA_P384_WIDE_CLEANUP="$(
    awk '/fence syncscope\("singlethread"\) seq_cst/{fences++; next} fences >= 2' \
      <<<"$ECDSA_P384_IR"
  )"
  if [[ -z "$ECDSA_P384_IR" ]] || \
    ! grep -q 'diag_zeroize_ecdsa_p384_platform_scratch' "$MIR" || \
    ! grep -q 'diag_zeroize_ecdsa_p384_platform_scratch' "$ASSEMBLY" || \
    [[ "$(grep -c 'store volatile i64 0' <<<"$ECDSA_P384_INVERSE_CLEANUP" || true)" -lt 6 ]] || \
    [[ "$(grep -c 'store volatile i64 0' <<<"$ECDSA_P384_REDUCED_CLEANUP" || true)" -lt 6 ]] || \
    ! grep -q 'sub nuw nsw i64 96' <<<"$ECDSA_P384_WIDE_CLEANUP" || \
    ! grep -q 'store volatile i64 0' <<<"$ECDSA_P384_WIDE_CLEANUP" || \
    ! grep -q 'store volatile i8 0' <<<"$ECDSA_P384_WIDE_CLEANUP" || \
    [[ "$(grep -c 'fence syncscope("singlethread") seq_cst' <<<"$ECDSA_P384_IR" || true)" -lt 3 ]]; then
    echo "zeroize release evidence does not clear P-384 accelerated input, reduction, and inversion scratch" >&2
    exit 1
  fi
fi

ECDSA_P256_SAFEGCD_IR="$(sed -n '/define .*@diag_zeroize_ecdsa_p256_safegcd_scratch(/,/^}/p' "$LLVM_IR")"
ECDSA_P384_SAFEGCD_IR="$(sed -n '/define .*@diag_zeroize_ecdsa_p384_safegcd_scratch(/,/^}/p' "$LLVM_IR")"
ECDSA_P256_SAFEGCD_INVERSE_IR="$(sed -n '/define .*ecdsa_safegcd23invert_order_montgomeryKj4/,/^}/p' "$LLVM_IR")"
ECDSA_P384_SAFEGCD_INVERSE_IR="$(sed -n '/define .*ecdsa_safegcd23invert_order_montgomeryKj6/,/^}/p' "$LLVM_IR")"
ECDSA_SAFEGCD_DROP_SYMBOL="$(
  sed -n 's/^define .*@\([^ (]*ecdsa_safegcd[^ (]*DivstepState[^ (]*Drop4drop[^ (]*\).*/\1/p' "$LLVM_IR" |
    head -n 1
)"
ECDSA_SAFEGCD_DROP_IR="$(sed -n "/define .*@$ECDSA_SAFEGCD_DROP_SYMBOL(/,/^}/p" "$LLVM_IR")"
if [[ -z "$ECDSA_P256_SAFEGCD_IR" || -z "$ECDSA_P384_SAFEGCD_IR" || \
  -z "$ECDSA_P256_SAFEGCD_INVERSE_IR" || -z "$ECDSA_P384_SAFEGCD_INVERSE_IR" || \
  -z "$ECDSA_SAFEGCD_DROP_SYMBOL" || -z "$ECDSA_SAFEGCD_DROP_IR" ]] || \
  ! grep -q 'ecdsa_safegcd23invert_order_montgomeryKj4' <<<"$ECDSA_P256_SAFEGCD_IR" || \
  ! grep -q 'ecdsa_safegcd23invert_order_montgomeryKj6' <<<"$ECDSA_P384_SAFEGCD_IR" || \
  ! grep -q "@$ECDSA_SAFEGCD_DROP_SYMBOL" <<<"$ECDSA_P256_SAFEGCD_INVERSE_IR" || \
  ! grep -q "@$ECDSA_SAFEGCD_DROP_SYMBOL" <<<"$ECDSA_P384_SAFEGCD_INVERSE_IR" || \
  [[ "$(grep -c 'store volatile i32 0' <<<"$ECDSA_SAFEGCD_DROP_IR" || true)" -lt 101 ]] || \
  ! grep -q 'fence syncscope("singlethread") seq_cst' <<<"$ECDSA_SAFEGCD_DROP_IR"; then
  echo "zeroize release evidence does not clear the complete ECDSA safegcd divstep workspace" >&2
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

RSA_VALIDATION_WRAPPER_IR="$(
  sed -n '/define .*diag_rsa_validate_pkcs8_private_key_der_stage(/,/^}/p' "$LLVM_IR"
)"
llvm_calls() {
  local symbol_pattern="$1"
  grep -E \
    "^[[:space:]]*(%[^=]+=[[:space:]]*)?((musttail|tail|notail)[[:space:]]+)?call[[:space:]].*$symbol_pattern" ||
    true
}

RSA_CALLER_SUCCESS_IR="$(
  sed -n '/define .*@diag_rsa_caller_random_signing_success(/,/^}/p' "$LLVM_IR"
)"
RSA_CALLER_ERROR_IR="$(
  sed -n '/define .*@diag_rsa_caller_random_signing_error(/,/^}/p' "$LLVM_IR"
)"
RSA_CALLER_SUCCESS_CLEAR_SYMBOL="$(
  llvm_calls 'RsaPrivateScratch.*clear' <<<"$RSA_CALLER_SUCCESS_IR" |
    sed -n 's/.*@\([^ (]*RsaPrivateScratch[^ (]*clear[^ (]*\).*/\1/p' |
    head -n 1
)"
RSA_CALLER_ERROR_CLEAR_SYMBOL="$(
  llvm_calls 'RsaPrivateScratch.*clear' <<<"$RSA_CALLER_ERROR_IR" |
    sed -n 's/.*@\([^ (]*RsaPrivateScratch[^ (]*clear[^ (]*\).*/\1/p' |
    head -n 1
)"
if [[ -z "$RSA_CALLER_SUCCESS_CLEAR_SYMBOL" || \
  "$RSA_CALLER_SUCCESS_CLEAR_SYMBOL" != "$RSA_CALLER_ERROR_CLEAR_SYMBOL" ]]; then
  echo "zeroize RSA caller-random signing paths do not share scratch cleanup" >&2
  exit 1
fi

RSA_PRIVATE_SCRATCH_CLEAR_SYMBOL="$RSA_CALLER_SUCCESS_CLEAR_SYMBOL"
RSA_PRIVATE_SCRATCH_CLEAR_IR="$(
  sed -n "/define .*@$RSA_PRIVATE_SCRATCH_CLEAR_SYMBOL(/,/^}/p" "$LLVM_IR"
)"
if [[ "$(grep -c 'store volatile .* 0' <<<"$RSA_PRIVATE_SCRATCH_CLEAR_IR" || true)" -lt 40 ]] || \
  [[ "$(grep -c 'fence syncscope("singlethread") seq_cst' \
    <<<"$RSA_PRIVATE_SCRATCH_CLEAR_IR" || true)" -lt 20 ]]; then
  echo "zeroize release evidence does not clear the complete RSA private scratch" >&2
  exit 1
fi

RSA_VALIDATION_SYMBOL="$(
  llvm_calls 'validate_private_key_components_through_stage' \
    <<<"$RSA_VALIDATION_WRAPPER_IR" |
    sed -n 's/.*@\([^ (]*validate_private_key_components_through_stage[^ (]*\).*/\1/p' |
    head -n 1
)"
if [[ -z "$RSA_VALIDATION_SYMBOL" ]] || \
  ! grep -q 'diag_rsa_validate_pkcs8_private_key_der_stage' "$MIR" || \
  ! grep -q 'diag_rsa_validate_pkcs8_private_key_der_stage' "$ASSEMBLY"; then
  echo "zeroize RSA private-key validation evidence is missing" >&2
  exit 1
fi

RSA_VALIDATION_IR="$(sed -n "/define .*@$RSA_VALIDATION_SYMBOL(/,/^}/p" "$LLVM_IR")"
RSA_SECRET_OWNER_CONSTRUCTION_CALLS="$(
  llvm_calls 'SecretBigEndianBuffer.*zeroed' <<<"$RSA_VALIDATION_IR"
)"
RSA_SECRET_OWNER_DROP_CALLS="$(
  llvm_calls 'drop_(in_place|glue).*SecretBigEndianBuffer' <<<"$RSA_VALIDATION_IR"
)"
RSA_SECRET_OWNER_CONSTRUCTIONS="$(grep -c . <<<"$RSA_SECRET_OWNER_CONSTRUCTION_CALLS" || true)"
RSA_SECRET_OWNER_DROPS="$(grep -c . <<<"$RSA_SECRET_OWNER_DROP_CALLS" || true)"
RSA_SECRET_CONSTRUCTION_OPERANDS="$(
  sed -n 's/.*(ptr [^%]*\(%[^,)]*\).*/\1/p' <<<"$RSA_SECRET_OWNER_CONSTRUCTION_CALLS" |
    sort -u |
    grep -c . || true
)"
RSA_SECRET_DROP_OPERANDS="$(
  sed -n 's/.*(ptr [^%]*\(%[^,)]*\).*/\1/p' <<<"$RSA_SECRET_OWNER_DROP_CALLS" |
    sort -u |
    grep -c . || true
)"
RSA_SECRET_DROP_SYMBOL="$(
  sed -En 's/.*@([^ (]*drop_(in_place|glue)[^ (]*SecretBigEndianBuffer[^ (]*).*/\1/p' \
    <<<"$RSA_SECRET_OWNER_DROP_CALLS" | head -n 1
)"
if [[ "$RSA_SECRET_OWNER_CONSTRUCTIONS" -ne 9 || "$RSA_SECRET_OWNER_DROPS" -ne 9 || \
  "$RSA_SECRET_CONSTRUCTION_OPERANDS" -ne 9 || "$RSA_SECRET_DROP_OPERANDS" -ne 9 || \
  -z "$RSA_SECRET_DROP_SYMBOL" ]] || \
  llvm_calls '__rust_dealloc' <<<"$RSA_VALIDATION_IR" | grep -q .; then
  echo "zeroize RSA private-key validation does not retain all RAII cleanup paths" >&2
  exit 1
fi

RSA_VALIDATION_WITHOUT_ONE_CONSTRUCTION="$(
  awk '
    !removed &&
      /^[[:space:]]*(%[^=]+=[[:space:]]*)?((musttail|tail|notail)[[:space:]]+)?call[[:space:]]/ &&
      /SecretBigEndianBuffer.*zeroed/ {
        removed = 1
        next
      }
    { print }
  ' <<<"$RSA_VALIDATION_IR"
)"
RSA_VALIDATION_WITHOUT_ONE_DROP="$(
  awk '
    !removed &&
      /^[[:space:]]*(%[^=]+=[[:space:]]*)?((musttail|tail|notail)[[:space:]]+)?call[[:space:]]/ &&
      /drop_(in_place|glue).*SecretBigEndianBuffer/ {
        removed = 1
        next
      }
    { print }
  ' <<<"$RSA_VALIDATION_IR"
)"
if [[ "$(llvm_calls 'SecretBigEndianBuffer.*zeroed' \
  <<<"$RSA_VALIDATION_WITHOUT_ONE_CONSTRUCTION" | grep -c .)" -ne 8 ]] || \
  [[ "$(llvm_calls 'drop_(in_place|glue).*SecretBigEndianBuffer' \
    <<<"$RSA_VALIDATION_WITHOUT_ONE_DROP" | grep -c .)" -ne 8 ]]; then
  echo "zeroize RSA private-key validation call parser does not reject a missing owner" >&2
  exit 1
fi

RSA_SECRET_DROP_IR="$(sed -n "/define .*@$RSA_SECRET_DROP_SYMBOL(/,/^}/p" "$LLVM_IR")"
RSA_SECRET_LAST_ZERO_LINE="$(
  grep -n '^[[:space:]]*store volatile .* 0' <<<"$RSA_SECRET_DROP_IR" |
    tail -n 1 |
    cut -d: -f1
)"
RSA_SECRET_FENCE_LINE="$(
  grep -n '^[[:space:]]*fence[[:space:]]' <<<"$RSA_SECRET_DROP_IR" |
    head -n 1 |
    cut -d: -f1
)"
RSA_SECRET_DEALLOC_LINE="$(
  grep -nE \
    '^[[:space:]]*(%[^=]+=[[:space:]]*)?((musttail|tail|notail)[[:space:]]+)?call[[:space:]].*__rust_dealloc' \
    <<<"$RSA_SECRET_DROP_IR" |
    head -n 1 |
    cut -d: -f1
)"
if [[ "$(grep -c '^[[:space:]]*store volatile .* 0' <<<"$RSA_SECRET_DROP_IR" || true)" -lt 3 || \
  -z "$RSA_SECRET_LAST_ZERO_LINE" || -z "$RSA_SECRET_FENCE_LINE" || -z "$RSA_SECRET_DEALLOC_LINE" || \
  "$RSA_SECRET_LAST_ZERO_LINE" -ge "$RSA_SECRET_FENCE_LINE" || \
  "$RSA_SECRET_FENCE_LINE" -ge "$RSA_SECRET_DEALLOC_LINE" ]]; then
  echo "zeroize RSA private-key validation owner does not wipe before deallocation" >&2
  exit 1
fi

ordered_assembly_cleanup() {
  local body="$1"
  local zero_pattern="$2"
  local zero_line
  local barrier_line
  local dealloc_line

  barrier_line="$(grep -n 'MEMBARRIER' <<<"$body" | head -n 1 | cut -d: -f1)"
  zero_line="$(
    grep -nE "$zero_pattern" <<<"$body" |
      cut -d: -f1 |
      awk -v barrier="$barrier_line" '$1 < barrier { line = $1 } END { print line }'
  )"
  dealloc_line="$(
    grep -nE '^[[:space:]]*(b|bl|call|callq|jmp|jmpq)[[:space:]].*__rust_dealloc' <<<"$body" |
      cut -d: -f1 |
      awk -v barrier="$barrier_line" '$1 > barrier { print; exit }'
  )"

  [[ -n "$zero_line" && -n "$barrier_line" && -n "$dealloc_line" &&
    "$zero_line" -lt "$barrier_line" && "$barrier_line" -lt "$dealloc_line" ]]
}

FIXED_ASSEMBLY="$(function_assembly diag_zeroize_fixed_stack <"$ASSEMBLY")"
RSA_CALLER_SUCCESS_ASSEMBLY="$(
  function_assembly diag_rsa_caller_random_signing_success <"$ASSEMBLY"
)"
RSA_CALLER_ERROR_ASSEMBLY="$(
  function_assembly diag_rsa_caller_random_signing_error <"$ASSEMBLY"
)"
RSA_PRIVATE_SCRATCH_CLEAR_ASSEMBLY="$(
  function_assembly "$RSA_PRIVATE_SCRATCH_CLEAR_SYMBOL" <"$ASSEMBLY"
)"
RSA_VALIDATION_ASSEMBLY="$(function_assembly "$RSA_VALIDATION_SYMBOL" <"$ASSEMBLY")"
RSA_SECRET_DROP_ASSEMBLY="$(function_assembly "$RSA_SECRET_DROP_SYMBOL" <"$ASSEMBLY")"
if [[ "$RSA_CALLER_SUCCESS_ASSEMBLY" != *"$RSA_PRIVATE_SCRATCH_CLEAR_SYMBOL"* || \
  "$RSA_CALLER_ERROR_ASSEMBLY" != *"$RSA_PRIVATE_SCRATCH_CLEAR_SYMBOL"* || \
  -z "$RSA_PRIVATE_SCRATCH_CLEAR_ASSEMBLY" ]]; then
  echo "zeroize RSA caller-random signing assembly does not retain scratch cleanup" >&2
  exit 1
fi
if [[ "$RSA_VALIDATION_ASSEMBLY" != *"$RSA_SECRET_DROP_SYMBOL"* || -z "$RSA_SECRET_DROP_ASSEMBLY" ]]; then
  echo "zeroize RSA private-key validation assembly does not retain owner cleanup" >&2
  exit 1
fi

RSA_ASSEMBLY_NEGATIVE_FIXTURE="$(
  printf '%s\n' \
    "$RSA_SECRET_DROP_SYMBOL:" \
    $'\tret' \
    '_later_zeroizing_function:' \
    $'\tstrb\twzr, [x0]' \
    $'\t;MEMBARRIER' \
    $'\tb\t__rust_dealloc'
)"
RSA_ASSEMBLY_NEGATIVE_BODY="$(
  function_assembly "$RSA_SECRET_DROP_SYMBOL" <<<"$RSA_ASSEMBLY_NEGATIVE_FIXTURE"
)"
if [[ "$RSA_ASSEMBLY_NEGATIVE_BODY" == *'_later_zeroizing_function'* ]] || \
  ordered_assembly_cleanup "$RSA_ASSEMBLY_NEGATIVE_BODY" "$AARCH64_ZERO_MEMORY_PATTERN"; then
  echo "zeroize assembly function parser does not reject a later function's cleanup" >&2
  exit 1
fi

for fixture in \
  $'; strb wzr, [x0]\n;MEMBARRIER\nb __rust_dealloc' \
  $'# movq $0, (%rax)\n#MEMBARRIER\njmp __rust_dealloc' \
  $'movl $0, %eax\n#MEMBARRIER\njmp __rust_dealloc' \
  $'b __rust_dealloc\nstrb wzr, [x0]\n;MEMBARRIER'; do
  if ordered_assembly_cleanup "$fixture" "$AARCH64_ZERO_MEMORY_PATTERN" || \
    ordered_assembly_cleanup "$fixture" "$X86_ZERO_MEMORY_PATTERN"; then
    echo "zeroize assembly parser accepts a comment or register-only zero" >&2
    exit 1
  fi
done
if ! ordered_assembly_cleanup \
  $'strb wzr, [x0]\n;MEMBARRIER\nb __rust_dealloc' "$AARCH64_ZERO_MEMORY_PATTERN" || \
  ! ordered_assembly_cleanup \
    $'strb wzr, [x0]\n;MEMBARRIER\nb __rust_dealloc\nstrb wzr, [x1]' "$AARCH64_ZERO_MEMORY_PATTERN" || \
  ! grep -Eq "$AARCH64_ZERO_MEMORY_PATTERN" <<< $'stur xzr, [x0, #-8]' || \
  ! ordered_assembly_cleanup \
    $'movq $0, 8(%rax)\n#MEMBARRIER\njmp __rust_dealloc' "$X86_ZERO_MEMORY_PATTERN"; then
  echo "zeroize assembly parser rejects a valid ordered memory wipe" >&2
  exit 1
fi

HOST_ARCH="${HOST_TRIPLE%%-*}"
case "$HOST_ARCH" in
  aarch64)
    if ! grep -Eq "$AARCH64_ZERO_MEMORY_PATTERN" <<<"$P256_ECDH_PUBLIC_ASSEMBLY" || \
      ! grep -Eq "$AARCH64_ZERO_MEMORY_PATTERN" <<<"$P256_ECDH_AGREEMENT_ASSEMBLY"; then
      echo "zeroize assembly evidence has no P-256 ECDH production-state wipe" >&2
      exit 1
    fi
    if ! ordered_assembly_cleanup \
      "$RSA_SECRET_DROP_ASSEMBLY" "$AARCH64_ZERO_MEMORY_PATTERN"; then
      echo "zeroize RSA private-key validation assembly does not wipe before deallocation" >&2
      exit 1
    fi
    if ! grep -Eq 'st(p|r)[[:space:]].*\[sp' <<<"$FIXED_ASSEMBLY" || \
       ! grep -Eq "$AARCH64_ZERO_MEMORY_PATTERN" <<<"$FIXED_ASSEMBLY"; then
      echo "zeroize assembly evidence does not show the fixed-size stack spill and wipe" >&2
      exit 1
    fi
    for symbol in \
      diag_zeroize_variable_heap \
      diag_zeroize_variable_fill_error \
      diag_zeroize_secret_string; do
      FUNCTION_ASSEMBLY="$(function_assembly "$symbol" <"$ASSEMBLY")"
      if ! ordered_assembly_cleanup "$FUNCTION_ASSEMBLY" "$AARCH64_ZERO_MEMORY_PATTERN"; then
        echo "zeroize assembly evidence does not wipe before deallocation in $symbol" >&2
        exit 1
      fi
    done
    for symbol in \
      diag_zeroize_fixed_move \
      diag_zeroize_fixed_fill_error \
      diag_zeroize_early_return \
      diag_zeroize_variable_heap \
      diag_zeroize_variable_fill_error \
      diag_zeroize_secret_string \
      diag_zeroize_hex_success \
      diag_zeroize_hex_error \
      diag_zeroize_blake3_xof_consume \
      diag_zeroize_blake3_thread_scratch \
      diag_zeroize_blake3_parallel_scratch \
      diag_zeroize_hmac_sha256_finalize \
      diag_zeroize_hmac_sha3_finalize \
      diag_rsa_caller_random_signing_success \
      diag_rsa_caller_random_signing_error \
      "$RSA_PRIVATE_SCRATCH_CLEAR_SYMBOL" \
      diag_hkdf_sha256_derive_portable \
      diag_hkdf_sha384_derive_portable \
      diag_hkdf_sha512_derive_portable \
      diag_poly1305_block_portable_digest \
      diag_ascon_aead128_tag_portable \
      diag_aegis256_update_portable \
      diag_aes128gcm_ghash \
      diag_aes256gcm_ghash \
      diag_zeroize_aes128_header_protection \
      diag_zeroize_aes256_header_protection \
      diag_zeroize_chacha20_header_protection \
      diag_zeroize_aes_siv_cmac256 \
      diag_zeroize_ecdsa_p256_public_blinding \
      diag_zeroize_ecdsa_p256_signing_blinding \
      diag_zeroize_ecdsa_p256_safegcd_scratch \
      diag_zeroize_p256_ecdh_generation \
      diag_zeroize_p256_ecdh_agreement \
      diag_zeroize_ecdsa_p384_public_blinding \
      diag_zeroize_ecdsa_p384_signing_blinding \
      diag_zeroize_ecdsa_p384_safegcd_scratch \
      diag_zeroize_mlkem_sha3_512 \
      diag_zeroize_mlkem_shake256_scalar \
      diag_zeroize_mlkem_shake256_pair \
      diag_zeroize_mlkem_shake256_quad \
      diag_zeroize_ecdsa_p256_platform_scratch \
      diag_zeroize_ecdsa_p384_platform_scratch; do
      FUNCTION_ASSEMBLY="$(function_assembly "$symbol" <"$ASSEMBLY")"
      if ! grep -Eq "$AARCH64_ZERO_MEMORY_PATTERN" <<<"$FUNCTION_ASSEMBLY"; then
        echo "zeroize assembly evidence has no zero store in $symbol" >&2
        exit 1
      fi
    done
    ;;
  x86_64)
    if ! grep -Eq "$X86_ZERO_MEMORY_PATTERN" <<<"$P256_ECDH_PUBLIC_ASSEMBLY" || \
      ! grep -Eq "$X86_ZERO_MEMORY_PATTERN" <<<"$P256_ECDH_AGREEMENT_ASSEMBLY"; then
      echo "zeroize assembly evidence has no P-256 ECDH production-state wipe" >&2
      exit 1
    fi
    if ! ordered_assembly_cleanup \
      "$RSA_SECRET_DROP_ASSEMBLY" "$X86_ZERO_MEMORY_PATTERN"; then
      echo "zeroize RSA private-key validation assembly does not wipe before deallocation" >&2
      exit 1
    fi
    if ! grep -Eq '%rsp' <<<"$FIXED_ASSEMBLY" || \
      ! grep -Eq "$X86_ZERO_MEMORY_PATTERN" <<<"$FIXED_ASSEMBLY"; then
      echo "zeroize assembly evidence does not show the fixed-size stack spill and wipe" >&2
      exit 1
    fi
    for symbol in \
      diag_zeroize_variable_heap \
      diag_zeroize_variable_fill_error \
      diag_zeroize_secret_string; do
      FUNCTION_ASSEMBLY="$(function_assembly "$symbol" <"$ASSEMBLY")"
      if ! ordered_assembly_cleanup "$FUNCTION_ASSEMBLY" "$X86_ZERO_MEMORY_PATTERN"; then
        echo "zeroize assembly evidence does not wipe before deallocation in $symbol" >&2
        exit 1
      fi
    done
    for symbol in \
      diag_zeroize_fixed_move \
      diag_zeroize_fixed_fill_error \
      diag_zeroize_early_return \
      diag_zeroize_variable_heap \
      diag_zeroize_variable_fill_error \
      diag_zeroize_secret_string \
      diag_zeroize_hex_success \
      diag_zeroize_hex_error \
      diag_zeroize_blake3_xof_consume \
      diag_zeroize_blake3_thread_scratch \
      diag_zeroize_blake3_parallel_scratch \
      diag_zeroize_hmac_sha256_finalize \
      diag_zeroize_hmac_sha3_finalize \
      diag_rsa_caller_random_signing_success \
      diag_rsa_caller_random_signing_error \
      "$RSA_PRIVATE_SCRATCH_CLEAR_SYMBOL" \
      diag_hkdf_sha256_derive_portable \
      diag_hkdf_sha384_derive_portable \
      diag_hkdf_sha512_derive_portable \
      diag_poly1305_block_portable_digest \
      diag_ascon_aead128_tag_portable \
      diag_aegis256_update_portable \
      diag_aes128gcm_ghash \
      diag_aes256gcm_ghash \
      diag_zeroize_aes128_header_protection \
      diag_zeroize_aes256_header_protection \
      diag_zeroize_chacha20_header_protection \
      diag_zeroize_aes_siv_cmac256 \
      diag_zeroize_ecdsa_p256_public_blinding \
      diag_zeroize_ecdsa_p256_signing_blinding \
      diag_zeroize_ecdsa_p256_safegcd_scratch \
      diag_zeroize_p256_ecdh_generation \
      diag_zeroize_p256_ecdh_agreement \
      diag_zeroize_ecdsa_p384_public_blinding \
      diag_zeroize_ecdsa_p384_signing_blinding \
      diag_zeroize_ecdsa_p384_safegcd_scratch \
      diag_zeroize_mlkem_sha3_512 \
      diag_zeroize_mlkem_shake256_scalar \
      diag_zeroize_mlkem_shake256_pair \
      diag_zeroize_mlkem_shake256_quad \
      diag_zeroize_ecdsa_p256_platform_scratch; do
      FUNCTION_ASSEMBLY="$(function_assembly "$symbol" <"$ASSEMBLY")"
      if ! grep -Eq "$X86_ZERO_MEMORY_PATTERN" <<<"$FUNCTION_ASSEMBLY"; then
        echo "zeroize assembly evidence has no zero store in $symbol" >&2
        exit 1
      fi
    done
    ;;
esac

echo "zeroize compiler evidence ok"
