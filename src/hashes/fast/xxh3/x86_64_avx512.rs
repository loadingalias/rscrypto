//! AVX-512 accelerated XXH3 kernel (x86-64).
//!
//! Vectorizes the 8-stripe accumulator in a single 512-bit register
//! (1 × `__m512i` = 8 × u64). One iteration per stripe, one per scramble.
//!
//! # Safety
//!
//! Uses `unsafe` for AVX-512F intrinsics. Callers must ensure AVX-512F is
//! available before executing the accelerated path (the dispatcher does this).

use core::arch::x86_64::*;

use super::{
  ACC_NB, DEFAULT_SECRET, DEFAULT_SECRET_SIZE, INITIAL_ACC, PRIME32_1, PRIME64_1, PRIME64_2, SECRET_CONSUME_RATE,
  SECRET_LASTACC_START, SECRET_MERGEACCS_START, STRIPE_LEN,
};

/// Shuffle mask: swap 32-bit pairs within each 64-bit lane.
///
/// `_MM_SHUFFLE(1, 0, 3, 2)` = `0x4E`. Applied per 128-bit lane within the
/// 512-bit register, this swaps adjacent u64 elements (the `idx ^ 1` effect).
const SWAP32: i32 = 0x4E;

// SIMD accumulate + scramble

#[inline]
#[target_feature(enable = "avx512f")]
/// # Safety
///
/// The current CPU must support AVX-512F.
unsafe fn load_acc(initial: &[u64; ACC_NB]) -> __m512i {
  // SAFETY: The function attribute enables the unaligned AVX-512F load, and
  // `initial` contains the eight readable lanes loaded here.
  unsafe { _mm512_loadu_si512(initial.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "avx512f")]
/// # Safety
///
/// The current CPU must support AVX-512F.
unsafe fn store_acc(acc: __m512i) -> [u64; ACC_NB] {
  // SAFETY: The function attribute enables the unaligned AVX-512F store, and
  // `out` contains the eight writable lanes stored here.
  unsafe {
    let mut out = [0u64; ACC_NB];
    _mm512_storeu_si512(out.as_mut_ptr().cast(), acc);
    out
  }
}

/// Returns one stripe's accumulator term: `swap(data) + lo32(dk) * hi32(dk)`
/// in each lane, with `dk = data ^ key`.
///
/// Accumulation is lane-wise addition modulo 2^64, so callers may sum terms
/// in any order. Keeping the term off the accumulator leaves one dependent add
/// per stripe instead of two.
///
/// # Safety
///
/// The caller must enable AVX-512F. `stripe` and `secret` must each remain
/// valid to read 64 initialized bytes from one allocation.
#[inline(always)]
unsafe fn stripe_term(stripe: *const u8, secret: *const u8) -> __m512i {
  // SAFETY: the caller enables AVX-512F and keeps both 64-byte reads in bounds.
  unsafe {
    let data_vec = _mm512_loadu_si512(stripe.cast());
    let key_vec = _mm512_loadu_si512(secret.cast());
    let data_key = _mm512_xor_si512(data_vec, key_vec);

    // u32 × u32 → u64 multiply: low32(data_key) × high32(data_key)
    let data_key_hi = _mm512_srli_epi64::<32>(data_key);
    let product = _mm512_mul_epu32(data_key, data_key_hi);

    // Swap 32-bit pairs → swaps u64 elements within each 128-bit lane
    let data_swap = _mm512_shuffle_epi32::<SWAP32>(data_vec);
    _mm512_add_epi64(product, data_swap)
  }
}

/// Returns the term of stripe `stripe`, which reads input at `input + 64 * stripe`
/// and secret at `secret + 8 * stripe`.
///
/// # Safety
///
/// The caller must enable AVX-512F and keep both 64-byte reads in bounds.
#[inline(always)]
unsafe fn stripe_term_at(input: *const u8, secret: *const u8, stripe: usize) -> __m512i {
  // SAFETY: the caller enables AVX-512F and keeps both reads in bounds.
  unsafe {
    stripe_term(
      input.add(stripe.strict_mul(STRIPE_LEN)),
      secret.add(stripe.strict_mul(SECRET_CONSUME_RATE)),
    )
  }
}

/// Adds the terms of `count` consecutive stripes to `acc`.
///
/// Stripe `k` reads input at `input + 64k` and secret at `secret + 8k`. From
/// four stripes on, four partial sums shorten the loop-carried add chain; a
/// constant `count` unrolls.
///
/// # Safety
///
/// The caller must enable AVX-512F, and every stripe and secret read named
/// above must stay within its allocation.
#[inline(always)]
unsafe fn accumulate_stripes(acc: __m512i, input: *const u8, secret: *const u8, count: usize) -> __m512i {
  // SAFETY: the caller enables AVX-512F and keeps every read in bounds.
  unsafe {
    if count < 4 {
      // Too few stripes to pay for the partial sums.
      let mut sum = acc;
      let mut stripe = 0usize;
      while stripe < count {
        sum = _mm512_add_epi64(sum, stripe_term_at(input, secret, stripe));
        stripe = stripe.strict_add(1);
      }
      return sum;
    }
    let mut sum0 = acc;
    let mut sum1 = _mm512_setzero_si512();
    let mut sum2 = _mm512_setzero_si512();
    let mut sum3 = _mm512_setzero_si512();
    let mut stripe = 0usize;
    while stripe.strict_add(4) <= count {
      sum0 = _mm512_add_epi64(sum0, stripe_term_at(input, secret, stripe));
      sum1 = _mm512_add_epi64(sum1, stripe_term_at(input, secret, stripe.strict_add(1)));
      sum2 = _mm512_add_epi64(sum2, stripe_term_at(input, secret, stripe.strict_add(2)));
      sum3 = _mm512_add_epi64(sum3, stripe_term_at(input, secret, stripe.strict_add(3)));
      stripe = stripe.strict_add(4);
    }
    while stripe < count {
      sum0 = _mm512_add_epi64(sum0, stripe_term_at(input, secret, stripe));
      stripe = stripe.strict_add(1);
    }
    _mm512_add_epi64(_mm512_add_epi64(sum0, sum1), _mm512_add_epi64(sum2, sum3))
  }
}

/// Scramble the accumulator at block boundaries — single iteration.
///
/// # Safety
///
/// The current CPU must support AVX-512F. `secret` must remain valid to read
/// 64 initialized bytes from one allocation.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn scramble_acc(acc: &mut __m512i, secret: *const u8) {
  // SAFETY: AVX-512F available via target_feature. Caller ensures secret
  // points to ≥64 valid bytes.
  unsafe {
    let prime32 = _mm512_set1_epi32(PRIME32_1.cast_signed());

    let shifted = _mm512_srli_epi64::<47>(*acc);
    let data_vec = _mm512_xor_si512(*acc, shifted);

    let key_vec = _mm512_loadu_si512(secret.cast());
    let data_key = _mm512_xor_si512(data_vec, key_vec);

    // 64-bit multiply by PRIME32_1 (split into lo + hi halves)
    let data_key_hi = _mm512_srli_epi64::<32>(data_key);
    let prod_lo = _mm512_mul_epu32(data_key, prime32);
    let prod_hi = _mm512_mul_epu32(data_key_hi, prime32);
    *acc = _mm512_add_epi64(prod_lo, _mm512_slli_epi64::<32>(prod_hi));
  }
}

// Long-path loop (SIMD inner, scalar merge)

#[target_feature(enable = "avx512f")]
/// # Safety
///
/// The current CPU must support AVX-512F. `stripes` must be nonzero, and the
/// requested input and secret stripes must be within their slices.
unsafe fn stream_accumulate_inner(
  initial: [u64; ACC_NB],
  input: &[u8],
  input_offset: usize,
  secret: &[u8; DEFAULT_SECRET_SIZE],
  secret_offset: usize,
  stripes: usize,
  scramble_after: bool,
) -> [u64; ACC_NB] {
  debug_assert!(input_offset.strict_add(stripes.strict_mul(STRIPE_LEN)) <= input.len());
  debug_assert!(stripes != 0);
  debug_assert!(
    secret_offset
      .strict_add(stripes.strict_sub(1).strict_mul(SECRET_CONSUME_RATE))
      .strict_add(STRIPE_LEN)
      <= secret.len()
  );
  // SAFETY: Accumulating validated XXH3 stripes because:
  // 1. The dispatcher selects this function only when AVX-512F is available.
  // 2. The caller contract keeps every 64-byte input and secret load in bounds;
  //    the assertions above check that contract in debug builds.
  // 3. `initial` and the returned array each contain exactly eight `u64` lanes.
  unsafe {
    let mut acc = accumulate_stripes(
      load_acc(&initial),
      input.as_ptr().add(input_offset),
      secret.as_ptr().add(secret_offset),
      stripes,
    );
    if scramble_after {
      scramble_acc(
        &mut acc,
        secret.as_ptr().add(DEFAULT_SECRET_SIZE.strict_sub(STRIPE_LEN)),
      );
    }
    store_acc(acc)
  }
}

/// # Safety
///
/// The caller must verify AVX-512F support, pass `stripes > 0`, and keep every
/// requested input and secret stripe within its slice.
pub(crate) unsafe fn stream_accumulate(
  initial: [u64; ACC_NB],
  input: &[u8],
  input_offset: usize,
  secret: &[u8; DEFAULT_SECRET_SIZE],
  secret_offset: usize,
  stripes: usize,
  scramble_after: bool,
) -> [u64; ACC_NB] {
  // SAFETY: The caller contract provides AVX-512F support and proves every raw
  // input and secret load is in bounds.
  unsafe {
    stream_accumulate_inner(
      initial,
      input,
      input_offset,
      secret,
      secret_offset,
      stripes,
      scramble_after,
    )
  }
}

#[target_feature(enable = "avx512f")]
/// # Safety
///
/// The current CPU must support AVX-512F, and `input` must contain more than
/// 240 bytes.
unsafe fn hash_long_internal_loop(input: &[u8], secret: &[u8; DEFAULT_SECRET_SIZE]) -> [u64; ACC_NB] {
  // SAFETY: AVX-512F available via target_feature. Input/secret bounds checked
  // by caller.
  unsafe {
    let mut acc = load_acc(&INITIAL_ACC);

    let nb_stripes = secret.len().strict_sub(STRIPE_LEN).strict_div(SECRET_CONSUME_RATE);
    let block_len = STRIPE_LEN.strict_mul(nb_stripes);
    let nb_blocks = input.len().strict_sub(1).strict_div(block_len);

    // Each block pointer stays within `input`: `nb_blocks` full blocks end
    // before its last byte.
    let mut block = input.as_ptr();
    for _ in 0..nb_blocks {
      acc = accumulate_stripes(acc, block, secret.as_ptr(), nb_stripes);
      scramble_acc(&mut acc, secret.as_ptr().add(secret.len().strict_sub(STRIPE_LEN)));
      block = block.add(block_len);
    }

    // Remaining stripes in final partial block
    let nb_stripes_final = input
      .len()
      .strict_sub(1)
      .strict_sub(block_len.strict_mul(nb_blocks))
      .strict_div(STRIPE_LEN);
    acc = accumulate_stripes(acc, block, secret.as_ptr(), nb_stripes_final);

    // Last stripe (may overlap with previous)
    acc = _mm512_add_epi64(
      acc,
      stripe_term(
        input.as_ptr().add(input.len().strict_sub(STRIPE_LEN)),
        secret
          .as_ptr()
          .add(secret.len().strict_sub(STRIPE_LEN).strict_sub(SECRET_LASTACC_START)),
      ),
    );

    store_acc(acc)
  }
}

// Top-level kernel functions (safe wrappers)

/// Long-path entry point (>240B) — no ≤240B branches.
pub(crate) fn xxh3_64_long_default(input: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies AVX-512F before selecting this kernel.
  let acc = unsafe { hash_long_internal_loop(input, &DEFAULT_SECRET) };
  super::merge_accs(
    &acc,
    &DEFAULT_SECRET,
    SECRET_MERGEACCS_START,
    (input.len() as u64).wrapping_mul(PRIME64_1),
  )
}

pub(crate) fn xxh3_64_long(input: &[u8], seed: u64) -> u64 {
  if seed == 0 {
    xxh3_64_long_default(input)
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

/// Long-path entry point (>240B) — no ≤240B branches.
pub(crate) fn xxh3_128_long_default(input: &[u8]) -> u128 {
  // SAFETY: Dispatcher verifies AVX-512F before selecting this kernel.
  let acc = unsafe { hash_long_internal_loop(input, &DEFAULT_SECRET) };
  xxh3_128_long_finalize(&acc, &DEFAULT_SECRET, input.len())
}

pub(crate) fn xxh3_128_long(input: &[u8], seed: u64) -> u128 {
  if seed == 0 {
    xxh3_128_long_default(input)
  } else {
    let secret = super::custom_default_secret(seed);
    // SAFETY: Dispatcher verifies AVX-512F before selecting this kernel.
    let acc = unsafe { hash_long_internal_loop(input, &secret) };
    xxh3_128_long_finalize(&acc, &secret, input.len())
  }
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
