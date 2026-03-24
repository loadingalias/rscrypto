//! AVX-512 accelerated XXH3 kernel (x86-64).
//!
//! Vectorizes the 8-stripe accumulator in a single 512-bit register
//! (1 × `__m512i` = 8 × u64). One iteration per stripe, one per scramble.
//!
//! # Safety
//!
//! Uses `unsafe` for AVX-512F intrinsics. Callers must ensure AVX-512F is
//! available before executing the accelerated path (the dispatcher does this).
#![allow(unsafe_code)]
#![allow(clippy::indexing_slicing)]

use core::arch::x86_64::*;

use super::{
  ACC_NB, DEFAULT_SECRET, INITIAL_ACC, MID_SIZE_MAX, PRIME32_1, PRIME64_1, PRIME64_2, SECRET_CONSUME_RATE,
  SECRET_LASTACC_START, SECRET_MERGEACCS_START, STRIPE_LEN,
};

/// Shuffle mask: swap 32-bit pairs within each 64-bit lane.
///
/// `_MM_SHUFFLE(1, 0, 3, 2)` = `0x4E`. Applied per 128-bit lane within the
/// 512-bit register, this swaps adjacent u64 elements (the `idx ^ 1` effect).
const SWAP32: i32 = 0x4E;

// ─────────────────────────────────────────────────────────────────────────────
// SIMD accumulate + scramble
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn load_acc(initial: &[u64; ACC_NB]) -> __m512i {
  // SAFETY: AVX-512F available via target_feature. Pointer valid for 8 × u64.
  unsafe { _mm512_loadu_si512(initial.as_ptr() as *const __m512i) }
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn store_acc(acc: __m512i) -> [u64; ACC_NB] {
  // SAFETY: AVX-512F available via target_feature.
  unsafe {
    let mut out = [0u64; ACC_NB];
    _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, acc);
    out
  }
}

/// Accumulate one 64-byte stripe — single iteration with 512-bit registers.
///
/// Processes all 8 u64 accumulators simultaneously:
/// 1. Load full 64 B stripe + 64 B secret in one shot
/// 2. XOR → shift → multiply → shuffle-swap → accumulate
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn accumulate_512(acc: &mut __m512i, stripe: *const u8, secret: *const u8) {
  // SAFETY: AVX-512F available via target_feature. Caller ensures stripe and
  // secret point to ≥64 valid bytes.
  unsafe {
    let data_vec = _mm512_loadu_si512(stripe as *const __m512i);
    let key_vec = _mm512_loadu_si512(secret as *const __m512i);
    let data_key = _mm512_xor_si512(data_vec, key_vec);

    // u32 × u32 → u64 multiply: low32(data_key) × high32(data_key)
    let data_key_hi = _mm512_srli_epi64::<32>(data_key);
    let product = _mm512_mul_epu32(data_key, data_key_hi);

    // Swap 32-bit pairs → swaps u64 elements within each 128-bit lane
    let data_swap = _mm512_shuffle_epi32::<SWAP32>(data_vec);
    let sum = _mm512_add_epi64(*acc, data_swap);
    *acc = _mm512_add_epi64(product, sum);
  }
}

/// Scramble the accumulator at block boundaries — single iteration.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn scramble_acc(acc: &mut __m512i, secret: *const u8) {
  // SAFETY: AVX-512F available via target_feature. Caller ensures secret
  // points to ≥64 valid bytes.
  unsafe {
    let prime32 = _mm512_set1_epi32(PRIME32_1 as i32);

    let shifted = _mm512_srli_epi64::<47>(*acc);
    let data_vec = _mm512_xor_si512(*acc, shifted);

    let key_vec = _mm512_loadu_si512(secret as *const __m512i);
    let data_key = _mm512_xor_si512(data_vec, key_vec);

    // 64-bit multiply by PRIME32_1 (split into lo + hi halves)
    let data_key_hi = _mm512_srli_epi64::<32>(data_key);
    let prod_lo = _mm512_mul_epu32(data_key, prime32);
    let prod_hi = _mm512_mul_epu32(data_key_hi, prime32);
    *acc = _mm512_add_epi64(prod_lo, _mm512_slli_epi64::<32>(prod_hi));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Long-path loop (SIMD inner, scalar merge)
// ─────────────────────────────────────────────────────────────────────────────

#[target_feature(enable = "avx512f")]
unsafe fn hash_long_internal_loop(input: &[u8], secret: &[u8]) -> [u64; ACC_NB] {
  // SAFETY: AVX-512F available via target_feature. Input/secret bounds checked
  // by caller.
  unsafe {
    let mut acc = load_acc(&INITIAL_ACC);

    let nb_stripes = (secret.len().strict_sub(STRIPE_LEN)) / SECRET_CONSUME_RATE;
    let block_len = STRIPE_LEN.strict_mul(nb_stripes);
    let nb_blocks = (input.len().strict_sub(1)) / block_len;

    let mut block = 0usize;
    while block < nb_blocks {
      let mut stripe = 0usize;
      while stripe < nb_stripes {
        let input_off = block.strict_mul(block_len).strict_add(stripe.strict_mul(STRIPE_LEN));
        let secret_off = stripe.strict_mul(SECRET_CONSUME_RATE);
        accumulate_512(&mut acc, input.as_ptr().add(input_off), secret.as_ptr().add(secret_off));
        stripe = stripe.strict_add(1);
      }
      scramble_acc(&mut acc, secret.as_ptr().add(secret.len().strict_sub(STRIPE_LEN)));
      block = block.strict_add(1);
    }

    // Remaining stripes in final partial block
    let nb_stripes_final = (input.len().strict_sub(1).strict_sub(block_len.strict_mul(nb_blocks))) / STRIPE_LEN;
    let mut stripe = 0usize;
    while stripe < nb_stripes_final {
      let input_off = nb_blocks
        .strict_mul(block_len)
        .strict_add(stripe.strict_mul(STRIPE_LEN));
      let secret_off = stripe.strict_mul(SECRET_CONSUME_RATE);
      accumulate_512(&mut acc, input.as_ptr().add(input_off), secret.as_ptr().add(secret_off));
      stripe = stripe.strict_add(1);
    }

    // Last stripe (may overlap with previous)
    accumulate_512(
      &mut acc,
      input.as_ptr().add(input.len().strict_sub(STRIPE_LEN)),
      secret
        .as_ptr()
        .add(secret.len().strict_sub(STRIPE_LEN).strict_sub(SECRET_LASTACC_START)),
    );

    store_acc(acc)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level kernel functions (safe wrappers)
// ─────────────────────────────────────────────────────────────────────────────

/// XXH3 64-bit hash — AVX-512 kernel.
///
/// Delegates ≤240 B to portable scalar paths; >240 B uses AVX-512 accumulator.
pub fn xxh3_64_with_seed(input: &[u8], seed: u64) -> u64 {
  if input.len() <= 16 {
    return super::xxh3_64_0to16(input, seed, &DEFAULT_SECRET);
  }
  if input.len() <= 128 {
    return super::xxh3_64_7to128(input, seed, &DEFAULT_SECRET);
  }
  if input.len() <= MID_SIZE_MAX {
    return super::xxh3_64_129to240(input, seed, &DEFAULT_SECRET);
  }

  if seed == 0 {
    // SAFETY: Dispatcher verifies AVX-512F before selecting this kernel.
    let acc = unsafe { hash_long_internal_loop(input, &DEFAULT_SECRET) };
    super::merge_accs(
      &acc,
      &DEFAULT_SECRET,
      SECRET_MERGEACCS_START,
      (input.len() as u64).wrapping_mul(PRIME64_1),
    )
  } else {
    let secret = super::custom_default_secret(seed);
    // SAFETY: Dispatcher verifies AVX-512F before selecting this kernel.
    let acc = unsafe { hash_long_internal_loop(input, &secret) };
    super::merge_accs(
      &acc,
      &secret,
      SECRET_MERGEACCS_START,
      (input.len() as u64).wrapping_mul(PRIME64_1),
    )
  }
}

/// XXH3 128-bit hash — AVX-512 kernel.
///
/// Delegates ≤240 B to portable scalar paths; >240 B uses AVX-512 accumulator.
pub fn xxh3_128_with_seed(input: &[u8], seed: u64) -> u128 {
  if input.len() <= 16 {
    return super::xxh3_128_0to16(input, seed, &DEFAULT_SECRET);
  }
  if input.len() <= 128 {
    return super::xxh3_128_7to128(input, seed, &DEFAULT_SECRET);
  }
  if input.len() <= MID_SIZE_MAX {
    return super::xxh3_128_129to240(input, seed, &DEFAULT_SECRET);
  }

  let (acc, secret_ref) = if seed == 0 {
    (
      // SAFETY: Dispatcher verifies AVX-512F before selecting this kernel.
      unsafe { hash_long_internal_loop(input, &DEFAULT_SECRET) },
      &DEFAULT_SECRET[..],
    )
  } else {
    let secret = super::custom_default_secret(seed);
    // SAFETY: Dispatcher verifies AVX-512F before selecting this kernel.
    let acc = unsafe { hash_long_internal_loop(input, &secret) };
    let secret2 = super::custom_default_secret(seed);
    return xxh3_128_long_finalize(&acc, &secret2, input.len());
  };

  xxh3_128_long_finalize(&acc, secret_ref, input.len())
}

#[inline(always)]
fn xxh3_128_long_finalize(acc: &[u64; ACC_NB], secret: &[u8], len: usize) -> u128 {
  let lo = super::merge_accs(
    acc,
    secret,
    SECRET_MERGEACCS_START,
    (len as u64).wrapping_mul(PRIME64_1),
  );
  let hi = super::merge_accs(
    acc,
    secret,
    secret
      .len()
      .strict_sub(ACC_NB.strict_mul(core::mem::size_of::<u64>()))
      .strict_sub(SECRET_MERGEACCS_START),
    !(len as u64).wrapping_mul(PRIME64_2),
  );
  (lo as u128) | ((hi as u128) << 64)
}
