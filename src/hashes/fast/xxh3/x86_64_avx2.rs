//! AVX2 accelerated XXH3 kernel (x86-64).
//!
//! Vectorizes the 8-stripe accumulator multiply-accumulate loop using
//! 256-bit AVX2 registers (2 × `__m256i` = 8 × u64).
//!
//! # Safety
//!
//! Uses `unsafe` for AVX2 intrinsics. Callers must ensure AVX2 is available
//! before executing the accelerated path (the dispatcher does this).

use core::arch::x86_64::*;

// Only the 128-bit finalization uses PRIME64_2; see `xxh3_128_long_default`.
#[cfg(any(not(target_feature = "avx512f"), all(test, not(miri))))]
use super::PRIME64_2;
use super::{
  ACC_NB, DEFAULT_SECRET, DEFAULT_SECRET_SIZE, INITIAL_ACC, PRIME32_1, PRIME64_1, SECRET_CONSUME_RATE,
  SECRET_LASTACC_START, SECRET_MERGEACCS_START, STRIPE_LEN,
};

/// Shuffle mask: swap 32-bit pairs within each 64-bit lane.
///
/// `_MM_SHUFFLE(1, 0, 3, 2)` = `0x4E`. Applied per 128-bit lane within the
/// 256-bit register, this swaps adjacent u64 elements (the `idx ^ 1` effect).
const SWAP32: i32 = 0x4E;

// SIMD accumulate + scramble

#[inline]
#[target_feature(enable = "avx2")]
/// # Safety
///
/// The current CPU must support AVX2.
unsafe fn load_acc(initial: &[u64; ACC_NB]) -> [__m256i; 2] {
  // SAFETY: The function attribute enables the unaligned AVX2 loads, and
  // `initial` contains the eight readable lanes loaded here.
  unsafe {
    [
      _mm256_loadu_si256(initial.as_ptr().cast()),
      _mm256_loadu_si256(initial.as_ptr().add(4).cast()),
    ]
  }
}

#[inline]
#[target_feature(enable = "avx2")]
/// # Safety
///
/// The current CPU must support AVX2.
unsafe fn store_acc(acc: &[__m256i; 2]) -> [u64; ACC_NB] {
  // SAFETY: The function attribute enables the unaligned AVX2 stores, and
  // `out` contains the eight writable lanes stored here.
  unsafe {
    let mut out = [0u64; ACC_NB];
    _mm256_storeu_si256(out.as_mut_ptr().cast(), acc[0]);
    _mm256_storeu_si256(out.as_mut_ptr().add(4).cast(), acc[1]);
    out
  }
}

/// Returns one stripe's accumulator terms, one per 32-byte half:
/// `swap(data) + lo32(dk) * hi32(dk)` in each lane, with `dk = data ^ key`.
///
/// Accumulation is lane-wise addition modulo 2^64, so callers may sum terms
/// in any order. Keeping the term off the accumulator leaves one dependent add
/// per stripe instead of two.
///
/// # Safety
///
/// The caller must enable AVX2. `stripe` and `secret` must each remain valid
/// to read 64 initialized bytes from one allocation.
#[inline(always)]
unsafe fn stripe_term(stripe: *const u8, secret: *const u8) -> [__m256i; 2] {
  // SAFETY: the caller enables AVX2 and keeps both 64-byte reads in bounds;
  // each half reads 32 of them.
  unsafe {
    let half = |offset: usize| {
      let data_vec = _mm256_loadu_si256(stripe.add(offset).cast());
      let key_vec = _mm256_loadu_si256(secret.add(offset).cast());
      let data_key = _mm256_xor_si256(data_vec, key_vec);
      let data_key_hi = _mm256_srli_epi64::<32>(data_key);
      let product = _mm256_mul_epu32(data_key, data_key_hi);
      let data_swap = _mm256_shuffle_epi32::<SWAP32>(data_vec);
      _mm256_add_epi64(product, data_swap)
    };
    [half(0), half(32)]
  }
}

/// Returns the terms of stripe `stripe`, which reads input at
/// `input + 64 * stripe` and secret at `secret + 8 * stripe`.
///
/// # Safety
///
/// The caller must enable AVX2 and keep both 64-byte reads in bounds.
#[inline(always)]
unsafe fn stripe_term_at(input: *const u8, secret: *const u8, stripe: usize) -> [__m256i; 2] {
  // SAFETY: the caller enables AVX2 and keeps both reads in bounds.
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
/// four stripes on, two partial sums per half shorten the loop-carried add
/// chain; a constant `count` unrolls.
///
/// # Safety
///
/// The caller must enable AVX2, and every stripe and secret read named above
/// must stay within its allocation.
#[inline(always)]
unsafe fn accumulate_stripes(acc: [__m256i; 2], input: *const u8, secret: *const u8, count: usize) -> [__m256i; 2] {
  // SAFETY: the caller enables AVX2 and keeps every read in bounds.
  unsafe {
    if count < 4 {
      // Too few stripes to pay for the partial sums.
      let mut sum = acc;
      let mut stripe = 0usize;
      while stripe < count {
        let term = stripe_term_at(input, secret, stripe);
        sum = [_mm256_add_epi64(sum[0], term[0]), _mm256_add_epi64(sum[1], term[1])];
        stripe = stripe.strict_add(1);
      }
      return sum;
    }
    let mut even = acc;
    let mut odd = [_mm256_setzero_si256(); 2];
    let mut stripe = 0usize;
    while stripe.strict_add(2) <= count {
      let first = stripe_term_at(input, secret, stripe);
      let second = stripe_term_at(input, secret, stripe.strict_add(1));
      even = [_mm256_add_epi64(even[0], first[0]), _mm256_add_epi64(even[1], first[1])];
      odd = [_mm256_add_epi64(odd[0], second[0]), _mm256_add_epi64(odd[1], second[1])];
      stripe = stripe.strict_add(2);
    }
    if stripe < count {
      let last = stripe_term_at(input, secret, stripe);
      even = [_mm256_add_epi64(even[0], last[0]), _mm256_add_epi64(even[1], last[1])];
    }
    [_mm256_add_epi64(even[0], odd[0]), _mm256_add_epi64(even[1], odd[1])]
  }
}

/// Scramble the accumulator at block boundaries.
///
/// Per element: `acc = (xorshift64(acc, 47) ^ secret) * PRIME32_1`
/// The 64-bit multiply by a 32-bit prime is split into lo + hi halves.
///
/// # Safety
///
/// The current CPU must support AVX2. `secret` must remain valid to read 64
/// initialized bytes from one allocation.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn scramble_acc(acc: &mut [__m256i; 2], secret: *const u8) {
  // SAFETY: AVX2 available via target_feature. Caller ensures secret points to
  // ≥64 valid bytes.
  unsafe {
    let prime32 = _mm256_set1_epi32(PRIME32_1.cast_signed());

    let mut i = 0usize;
    while i < 2 {
      let acc_vec = acc[i];
      let shifted = _mm256_srli_epi64::<47>(acc_vec);
      let data_vec = _mm256_xor_si256(acc_vec, shifted);

      let key_vec = _mm256_loadu_si256(secret.add(i.strict_mul(32)).cast());
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

// Long-path loop (SIMD inner, scalar merge)

#[target_feature(enable = "avx2")]
/// # Safety
///
/// The current CPU must support AVX2. `stripes` must be nonzero, and the
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
  // 1. The dispatcher selects this function only when AVX2 is available.
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
    store_acc(&acc)
  }
}

/// # Safety
///
/// The caller must verify AVX2 support, pass `stripes > 0`, and keep every
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
  // SAFETY: The caller contract provides AVX2 support and proves every raw
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

#[target_feature(enable = "avx2")]
/// # Safety
///
/// The current CPU must support AVX2, and `input` must contain more than 240
/// bytes.
unsafe fn hash_long_internal_loop(input: &[u8], secret: &[u8; DEFAULT_SECRET_SIZE]) -> [u64; ACC_NB] {
  // SAFETY: AVX2 available via target_feature. Input/secret bounds checked by caller.
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
    let last = stripe_term(
      input.as_ptr().add(input.len().strict_sub(STRIPE_LEN)),
      secret
        .as_ptr()
        .add(secret.len().strict_sub(STRIPE_LEN).strict_sub(SECRET_LASTACC_START)),
    );
    acc = [_mm256_add_epi64(acc[0], last[0]), _mm256_add_epi64(acc[1], last[1])];

    store_acc(&acc)
  }
}

// Top-level kernel functions (safe wrappers)

/// Long-path entry point (>240B) — no ≤240B branches.
pub(crate) fn xxh3_64_long_default(input: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies AVX2 before selecting this kernel.
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

/// Long-path entry point (>240B) — no ≤240B branches.
// With AVX-512 enabled at compile time, dispatch selects the AVX-512 128-bit kernel directly and the
// runtime kernel table is not built, so these AVX2 128-bit entries are reachable only from tests.
#[cfg(any(not(target_feature = "avx512f"), all(test, not(miri))))]
pub(crate) fn xxh3_128_long_default(input: &[u8]) -> u128 {
  // SAFETY: Dispatcher verifies AVX2 before selecting this kernel.
  let acc = unsafe { hash_long_internal_loop(input, &DEFAULT_SECRET) };
  xxh3_128_long_finalize(&acc, &DEFAULT_SECRET, input.len())
}

#[cfg(any(not(target_feature = "avx512f"), all(test, not(miri))))]
pub(crate) fn xxh3_128_long(input: &[u8], seed: u64) -> u128 {
  if seed == 0 {
    xxh3_128_long_default(input)
  } else {
    let secret = super::custom_default_secret(seed);
    // SAFETY: Dispatcher verifies AVX2 before selecting this kernel.
    let acc = unsafe { hash_long_internal_loop(input, &secret) };
    xxh3_128_long_finalize(&acc, &secret, input.len())
  }
}

#[cfg(any(not(target_feature = "avx512f"), all(test, not(miri))))]
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
