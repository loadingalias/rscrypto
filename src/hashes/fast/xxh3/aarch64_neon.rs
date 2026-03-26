//! NEON accelerated XXH3 kernel (aarch64).
//!
//! Vectorizes the 8-stripe accumulator multiply-accumulate loop using
//! 128-bit NEON registers (4 × `uint64x2_t` = 8 × u64).
//!
//! Optimizations over the naive 1-lane-per-iteration approach:
//! - **Pair-wise processing:** 2 lanes per iteration via `vuzpq_u32` deinterleave
//!   and `vmlal_high_u32`, reducing instruction count per stripe.
//! - **Broken dependency chain:** `vmlal(data_swap, lo, hi)` instead of
//!   `vmlal(acc, lo, hi)` lets the multiply start without waiting for the
//!   previous accumulate.
//! - **Software prefetch:** `prfm pldl1keep` for the next stripe's input in the
//!   long-path loop, reducing cache miss stalls on Neoverse V1 (Graviton3).
//!
//! # Safety
//!
//! Uses `unsafe` for NEON intrinsics. Callers must ensure NEON is available
//! (always true on aarch64 — it is the baseline ISA).
#![allow(unsafe_code)]
#![allow(clippy::indexing_slicing)]

use core::arch::aarch64::*;

use super::{
  ACC_NB, DEFAULT_SECRET, INITIAL_ACC, MID_SIZE_MAX, PRIME32_1, PRIME64_1, PRIME64_2, SECRET_CONSUME_RATE,
  SECRET_LASTACC_START, SECRET_MERGEACCS_START, STRIPE_LEN,
};

// ─────────────────────────────────────────────────────────────────────────────
// SIMD accumulate + scramble
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "neon")]
unsafe fn load_acc(initial: &[u64; ACC_NB]) -> [uint64x2_t; 4] {
  // SAFETY: NEON available via target_feature. Pointer valid for 8 × u64.
  unsafe {
    [
      vld1q_u64(initial.as_ptr()),
      vld1q_u64(initial.as_ptr().add(2)),
      vld1q_u64(initial.as_ptr().add(4)),
      vld1q_u64(initial.as_ptr().add(6)),
    ]
  }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn store_acc(acc: &[uint64x2_t; 4]) -> [u64; ACC_NB] {
  // SAFETY: NEON available via target_feature.
  unsafe {
    let mut out = [0u64; ACC_NB];
    vst1q_u64(out.as_mut_ptr(), acc[0]);
    vst1q_u64(out.as_mut_ptr().add(2), acc[1]);
    vst1q_u64(out.as_mut_ptr().add(4), acc[2]);
    vst1q_u64(out.as_mut_ptr().add(6), acc[3]);
    out
  }
}

/// Accumulate one 64-byte stripe into the NEON accumulator.
///
/// Processes 2 accumulator lanes per iteration (32 bytes each) using:
/// - `vuzpq_u32` to deinterleave lo/hi 32-bit halves from two `data_key`
///   vectors in one instruction (vs 2 separate `vmovn` + `vshrn`)
/// - `vmlal_high_u32` for the second lane (avoids `vget_high_u32` extraction)
/// - `vmlal(data_swap, ...)` base to break the acc→vmlal dependency chain
#[inline]
#[target_feature(enable = "neon")]
unsafe fn accumulate_512(acc: &mut [uint64x2_t; 4], stripe: *const u8, secret: *const u8) {
  // SAFETY: NEON available via target_feature. Caller ensures stripe and secret
  // point to ≥64 valid bytes.
  unsafe {
    let mut i = 0usize;
    while i < 4 {
      // Load 2 × 16B input chunks and 2 × 16B secret chunks.
      let data_vec_1 = vreinterpretq_u64_u8(vld1q_u8(stripe.add(i.strict_mul(16))));
      let data_vec_2 = vreinterpretq_u64_u8(vld1q_u8(stripe.add(i.strict_add(1).strict_mul(16))));
      let key_vec_1 = vreinterpretq_u64_u8(vld1q_u8(secret.add(i.strict_mul(16))));
      let key_vec_2 = vreinterpretq_u64_u8(vld1q_u8(secret.add(i.strict_add(1).strict_mul(16))));

      // Swap u64 lanes (idx ^ 1 effect).
      let data_swap_1 = vextq_u64(data_vec_1, data_vec_1, 1);
      let data_swap_2 = vextq_u64(data_vec_2, data_vec_2, 1);

      // data_key = data ^ secret
      let data_key_1 = veorq_u64(data_vec_1, key_vec_1);
      let data_key_2 = veorq_u64(data_vec_2, key_vec_2);

      // Deinterleave lo/hi 32-bit halves from both keys in one instruction.
      // Input as u32x4: [lo0, hi0, lo1, hi1] from each data_key.
      // Result.0 (even): [lo0_1, lo1_1, lo0_2, lo1_2]
      // Result.1 (odd):  [hi0_1, hi1_1, hi0_2, hi1_2]
      let unzipped = vuzpq_u32(
        vreinterpretq_u32_u64(data_key_1),
        vreinterpretq_u32_u64(data_key_2),
      );
      let data_key_lo = unzipped.0;
      let data_key_hi = unzipped.1;

      // Multiply-accumulate with data_swap as base (not acc) to break the
      // dependency chain: vmlal doesn't wait for the previous vaddq to complete.
      // sum = data_swap + lo32 * hi32 (widening multiply-accumulate)
      let sum_1 = vmlal_u32(data_swap_1, vget_low_u32(data_key_lo), vget_low_u32(data_key_hi));
      let sum_2 = vmlal_high_u32(data_swap_2, data_key_lo, data_key_hi);

      acc[i] = vaddq_u64(acc[i], sum_1);
      acc[i.strict_add(1)] = vaddq_u64(acc[i.strict_add(1)], sum_2);

      i = i.strict_add(2);
    }
  }
}

/// Scramble the accumulator at block boundaries.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn scramble_acc(acc: &mut [uint64x2_t; 4], secret: *const u8) {
  // SAFETY: NEON available via target_feature. Caller ensures secret points to
  // ≥64 valid bytes.
  unsafe {
    let prime_lo = vdup_n_u32(PRIME32_1);
    // prime_hi: [0, PRIME32_1, 0, PRIME32_1] as u32x4
    // When multiplied element-wise, this captures high32 × PRIME32_1 in the
    // high 32-bit position of each u64 lane.
    let prime_hi = vreinterpretq_u32_u64(vdupq_n_u64((PRIME32_1 as u64) << 32));

    let mut i = 0usize;
    while i < 4 {
      let acc_vec = acc[i];
      // xorshift64(acc, 47)
      let shifted = vshrq_n_u64::<47>(acc_vec);
      let data_vec = veorq_u64(acc_vec, shifted);

      // XOR with secret
      let key_vec = vreinterpretq_u64_u8(vld1q_u8(secret.add(i.strict_mul(16))));
      let data_key = veorq_u64(data_vec, key_vec);

      // Full 64-bit multiply by PRIME32_1:
      // result = (high32 × PRIME32_1) << 32 + (low32 × PRIME32_1)
      let prod_hi = vmulq_u32(vreinterpretq_u32_u64(data_key), prime_hi);
      let data_key_lo = vmovn_u64(data_key);
      acc[i] = vmlal_u32(vreinterpretq_u64_u32(prod_hi), data_key_lo, prime_lo);

      i = i.strict_add(1);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Prefetch helper
// ─────────────────────────────────────────────────────────────────────────────

/// Prefetch the next stripe's input data into L1 cache.
///
/// Uses `wrapping_add` because the prefetch address may be past the end of the
/// buffer — `prfm` silently ignores invalid addresses.
#[inline(always)]
unsafe fn prefetch_stripe(input_ptr: *const u8) {
  // SAFETY: PRFM is a CPU hint; invalid addresses are silently ignored.
  unsafe {
    core::arch::asm!(
      "prfm pldl1keep, [{ptr}]",
      ptr = in(reg) input_ptr,
      options(nostack, preserves_flags)
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Long-path loop (SIMD inner, scalar merge)
// ─────────────────────────────────────────────────────────────────────────────

#[target_feature(enable = "neon")]
unsafe fn hash_long_internal_loop(input: &[u8], secret: &[u8]) -> [u64; ACC_NB] {
  // SAFETY: NEON available via target_feature. Input/secret bounds checked by caller.
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
        let input_ptr = input.as_ptr().add(input_off);
        let secret_off = stripe.strict_mul(SECRET_CONSUME_RATE);
        // Prefetch next stripe's input — prfm ignores out-of-bounds addresses.
        prefetch_stripe(input_ptr.wrapping_add(STRIPE_LEN));
        accumulate_512(&mut acc, input_ptr, secret.as_ptr().add(secret_off));
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
      let input_ptr = input.as_ptr().add(input_off);
      let secret_off = stripe.strict_mul(SECRET_CONSUME_RATE);
      prefetch_stripe(input_ptr.wrapping_add(STRIPE_LEN));
      accumulate_512(&mut acc, input_ptr, secret.as_ptr().add(secret_off));
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

/// XXH3 64-bit hash — NEON kernel.
///
/// Delegates ≤240 B to portable scalar paths; >240 B uses NEON accumulator.
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
    // SAFETY: Dispatcher verifies NEON before selecting this kernel
    // (always available on aarch64).
    let acc = unsafe { hash_long_internal_loop(input, &DEFAULT_SECRET) };
    super::merge_accs(
      &acc,
      &DEFAULT_SECRET,
      SECRET_MERGEACCS_START,
      (input.len() as u64).wrapping_mul(PRIME64_1),
    )
  } else {
    let secret = super::custom_default_secret(seed);
    // SAFETY: Dispatcher verifies NEON before selecting this kernel.
    let acc = unsafe { hash_long_internal_loop(input, &secret) };
    super::merge_accs(
      &acc,
      &secret,
      SECRET_MERGEACCS_START,
      (input.len() as u64).wrapping_mul(PRIME64_1),
    )
  }
}

/// XXH3 128-bit hash — NEON kernel.
///
/// Delegates ≤240 B to portable scalar paths; >240 B uses NEON accumulator.
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
      // SAFETY: Dispatcher verifies NEON before selecting this kernel.
      unsafe { hash_long_internal_loop(input, &DEFAULT_SECRET) },
      &DEFAULT_SECRET[..],
    )
  } else {
    let secret = super::custom_default_secret(seed);
    // SAFETY: Dispatcher verifies NEON before selecting this kernel.
    let acc = unsafe { hash_long_internal_loop(input, &secret) };
    // Re-derive to keep borrow alive — custom_default_secret is cheap and
    // only called for seed≠0 long inputs.
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
