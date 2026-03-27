//! AVX2 accelerated XXH3 kernel (x86-64).
//!
//! Vectorizes the 8-stripe accumulator multiply-accumulate loop using
//! 256-bit AVX2 registers (2 × `__m256i` = 8 × u64).
//!
//! # Safety
//!
//! Uses `unsafe` for AVX2 intrinsics. Callers must ensure AVX2 is available
//! before executing the accelerated path (the dispatcher does this).
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
/// 256-bit register, this swaps adjacent u64 elements (the `idx ^ 1` effect).
const SWAP32: i32 = 0x4E;

// ─────────────────────────────────────────────────────────────────────────────
// SIMD accumulate + scramble
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_acc(initial: &[u64; ACC_NB]) -> [__m256i; 2] {
  // SAFETY: AVX2 available via target_feature. Pointer valid for 8 × u64.
  unsafe {
    [
      _mm256_loadu_si256(initial.as_ptr() as *const __m256i),
      _mm256_loadu_si256(initial.as_ptr().add(4) as *const __m256i),
    ]
  }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_acc(acc: &[__m256i; 2]) -> [u64; ACC_NB] {
  // SAFETY: AVX2 available via target_feature.
  unsafe {
    let mut out = [0u64; ACC_NB];
    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, acc[0]);
    _mm256_storeu_si256(out.as_mut_ptr().add(4) as *mut __m256i, acc[1]);
    out
  }
}

/// Accumulate one 64-byte stripe into the AVX2 accumulator.
///
/// Per iteration (2 total for one stripe):
/// 1. Load 32 B of input and 32 B of secret
/// 2. XOR to get data_key
/// 3. `vpmuludq`: multiply low32(data_key) × high32(data_key) → u64
/// 4. Shuffle data to swap u64 pairs (idx ^ 1)
/// 5. Accumulate product + swapped data into acc
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_512(acc: &mut [__m256i; 2], stripe: *const u8, secret: *const u8) {
  // SAFETY: AVX2 available via target_feature. Caller ensures stripe and secret
  // point to ≥64 valid bytes.
  unsafe {
    let mut i = 0usize;
    while i < 2 {
      let data_vec = _mm256_loadu_si256(stripe.add(i.strict_mul(32)) as *const __m256i);
      let key_vec = _mm256_loadu_si256(secret.add(i.strict_mul(32)) as *const __m256i);
      let data_key = _mm256_xor_si256(data_vec, key_vec);

      // u32 × u32 → u64 multiply: low32(data_key) × high32(data_key)
      let data_key_hi = _mm256_srli_epi64::<32>(data_key);
      let product = _mm256_mul_epu32(data_key, data_key_hi);

      // Swap 32-bit pairs → swaps u64 elements within each 128-bit lane
      let data_swap = _mm256_shuffle_epi32::<SWAP32>(data_vec);
      let sum = _mm256_add_epi64(acc[i], data_swap);
      acc[i] = _mm256_add_epi64(product, sum);

      i = i.strict_add(1);
    }
  }
}

/// Scramble the accumulator at block boundaries.
///
/// Per element: `acc = (xorshift64(acc, 47) ^ secret) * PRIME32_1`
/// The 64-bit multiply by a 32-bit prime is split into lo + hi halves.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn scramble_acc(acc: &mut [__m256i; 2], secret: *const u8) {
  // SAFETY: AVX2 available via target_feature. Caller ensures secret points to
  // ≥64 valid bytes.
  unsafe {
    let prime32 = _mm256_set1_epi32(PRIME32_1 as i32);

    let mut i = 0usize;
    while i < 2 {
      let acc_vec = acc[i];
      let shifted = _mm256_srli_epi64::<47>(acc_vec);
      let data_vec = _mm256_xor_si256(acc_vec, shifted);

      let key_vec = _mm256_loadu_si256(secret.add(i.strict_mul(32)) as *const __m256i);
      let data_key = _mm256_xor_si256(data_vec, key_vec);

      // 64-bit multiply by PRIME32_1:
      // prod_lo = low32(data_key) × PRIME32_1
      // prod_hi = high32(data_key) × PRIME32_1, shifted left 32
      let data_key_hi = _mm256_srli_epi64::<32>(data_key);
      let prod_lo = _mm256_mul_epu32(data_key, prime32);
      let prod_hi = _mm256_mul_epu32(data_key_hi, prime32);
      acc[i] = _mm256_add_epi64(prod_lo, _mm256_slli_epi64::<32>(prod_hi));

      i = i.strict_add(1);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Long-path loop (SIMD inner, scalar merge)
// ─────────────────────────────────────────────────────────────────────────────

#[target_feature(enable = "avx2")]
unsafe fn hash_long_internal_loop(input: &[u8], secret: &[u8]) -> [u64; ACC_NB] {
  // SAFETY: AVX2 available via target_feature. Input/secret bounds checked by caller.
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

    store_acc(&acc)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level kernel functions (safe wrappers)
// ─────────────────────────────────────────────────────────────────────────────

/// Long-path entry point (>240B) — no ≤240B branches.
pub fn xxh3_64_long(input: &[u8], seed: u64) -> u64 {
  if seed == 0 {
    // SAFETY: Dispatcher verifies AVX2 before selecting this kernel.
    let acc = unsafe { hash_long_internal_loop(input, &DEFAULT_SECRET) };
    super::merge_accs(
      &acc,
      &DEFAULT_SECRET,
      SECRET_MERGEACCS_START,
      (input.len() as u64).wrapping_mul(PRIME64_1),
    )
  } else {
    let secret = super::custom_default_secret(seed);
    // SAFETY: Dispatcher verifies AVX2 before selecting this kernel.
    let acc = unsafe { hash_long_internal_loop(input, &secret) };
    super::merge_accs(
      &acc,
      &secret,
      SECRET_MERGEACCS_START,
      (input.len() as u64).wrapping_mul(PRIME64_1),
    )
  }
}

/// XXH3 64-bit hash — AVX2 kernel.
///
/// Delegates ≤240 B to portable scalar paths; >240 B uses AVX2 accumulator.
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
  xxh3_64_long(input, seed)
}

/// Long-path entry point (>240B) — no ≤240B branches.
pub fn xxh3_128_long(input: &[u8], seed: u64) -> u128 {
  if seed == 0 {
    // SAFETY: Dispatcher verifies AVX2 before selecting this kernel.
    let acc = unsafe { hash_long_internal_loop(input, &DEFAULT_SECRET) };
    xxh3_128_long_finalize(&acc, &DEFAULT_SECRET, input.len())
  } else {
    let secret = super::custom_default_secret(seed);
    // SAFETY: Dispatcher verifies AVX2 before selecting this kernel.
    let acc = unsafe { hash_long_internal_loop(input, &secret) };
    xxh3_128_long_finalize(&acc, &secret, input.len())
  }
}

/// XXH3 128-bit hash — AVX2 kernel.
///
/// Delegates ≤240 B to portable scalar paths; >240 B uses AVX2 accumulator.
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
  xxh3_128_long(input, seed)
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
