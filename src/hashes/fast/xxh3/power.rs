//! POWER VSX accelerated XXH3 kernel (ppc64le).
//!
//! Vectorizes the 8-stripe accumulator multiply-accumulate loop using
//! 128-bit VSX registers (4 × `i64x2` = 8 × u64).
//!
//! # Safety
//!
//! Uses `unsafe` for POWER8+ VSX inline asm. Callers must ensure POWER8+
//! vector support before executing the accelerated path (the dispatcher
//! does this).
#![allow(unsafe_code)]
#![allow(clippy::indexing_slicing)]

use core::simd::i64x2;

use super::{
  ACC_NB, DEFAULT_SECRET, INITIAL_ACC, MID_SIZE_MAX, PRIME32_1, PRIME64_1, PRIME64_2, SECRET_CONSUME_RATE,
  SECRET_LASTACC_START, SECRET_MERGEACCS_START, STRIPE_LEN,
};

// ─────────────────────────────────────────────────────────────────────────────
// VSX primitive operations (inline asm, POWER8+)
// ─────────────────────────────────────────────────────────────────────────────

/// Add u64 lanes: `vaddudm`.
#[inline]
#[target_feature(enable = "altivec", enable = "vsx", enable = "power8-vector")]
unsafe fn vadd_u64(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: POWER8+ VSX available via target_feature.
  unsafe {
    core::arch::asm!(
      "vaddudm {out}, {a}, {b}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// Logical shift right u64 lanes: `vsrd`.
#[inline]
#[target_feature(enable = "altivec", enable = "vsx", enable = "power8-vector")]
unsafe fn vshr_u64(a: i64x2, shift: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: POWER8+ VSX available via target_feature.
  unsafe {
    core::arch::asm!(
      "vsrd {out}, {a}, {shift}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      shift = in(vreg) shift,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// Shift left u64 lanes: `vsld`.
#[inline]
#[target_feature(enable = "altivec", enable = "vsx", enable = "power8-vector")]
unsafe fn vshl_u64(a: i64x2, shift: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: POWER8+ VSX available via target_feature.
  unsafe {
    core::arch::asm!(
      "vsld {out}, {a}, {shift}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      shift = in(vreg) shift,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// Multiply even-indexed u32 lanes → u64: `vmuleuw`.
///
/// On ppc64le, even u32 elements are the low 32 bits of each u64 lane.
/// This gives: `low32(a) × low32(b) → u64` per lane.
#[inline]
#[target_feature(enable = "altivec", enable = "vsx", enable = "power8-vector")]
unsafe fn vmul_even_u32(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: POWER8+ VSX available via target_feature.
  unsafe {
    core::arch::asm!(
      "vmuleuw {out}, {a}, {b}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      options(nomem, nostack, pure)
    );
  }
  out
}

// ─────────────────────────────────────────────────────────────────────────────
// Load, store, swap
// ─────────────────────────────────────────────────────────────────────────────

/// Load 128 bits from memory (unaligned).
#[inline(always)]
unsafe fn vload(ptr: *const u8) -> i64x2 {
  // SAFETY: caller ensures ptr is valid for 16 bytes.
  unsafe { core::ptr::read_unaligned(ptr as *const i64x2) }
}

/// Store 128 bits to memory (unaligned).
#[inline(always)]
unsafe fn vstore(ptr: *mut u8, val: i64x2) {
  // SAFETY: caller ensures ptr is valid for 16 bytes.
  unsafe { core::ptr::write_unaligned(ptr as *mut i64x2, val) }
}

/// Swap u64 lanes (idx ^ 1 effect).
#[inline(always)]
fn vswap(a: i64x2) -> i64x2 {
  core::simd::simd_swizzle!(a, [1, 0])
}

// ─────────────────────────────────────────────────────────────────────────────
// SIMD accumulate + scramble
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "altivec", enable = "vsx", enable = "power8-vector")]
unsafe fn load_acc(initial: &[u64; ACC_NB]) -> [i64x2; 4] {
  // SAFETY: POWER8+ VSX via target_feature. Pointer valid for 8 × u64.
  unsafe {
    let p = initial.as_ptr() as *const u8;
    [vload(p), vload(p.add(16)), vload(p.add(32)), vload(p.add(48))]
  }
}

#[inline]
#[target_feature(enable = "altivec", enable = "vsx", enable = "power8-vector")]
unsafe fn store_acc(acc: &[i64x2; 4]) -> [u64; ACC_NB] {
  // SAFETY: POWER8+ VSX via target_feature.
  unsafe {
    let mut out = [0u64; ACC_NB];
    let p = out.as_mut_ptr() as *mut u8;
    vstore(p, acc[0]);
    vstore(p.add(16), acc[1]);
    vstore(p.add(32), acc[2]);
    vstore(p.add(48), acc[3]);
    out
  }
}

/// Accumulate one 64-byte stripe into the VSX accumulator.
///
/// Per iteration (4 total, one per 16-byte chunk):
/// 1. Load 16 B of input and 16 B of secret
/// 2. XOR to get data_key
/// 3. `vmuleuw`: low32(data_key) × high32(data_key) → u64
/// 4. Swap u64 lanes (idx ^ 1) and add data
/// 5. Accumulate product + swapped data into acc
#[inline]
#[target_feature(enable = "altivec", enable = "vsx", enable = "power8-vector")]
unsafe fn accumulate_512(acc: &mut [i64x2; 4], stripe: *const u8, secret: *const u8) {
  // SAFETY: POWER8+ VSX via target_feature. Caller ensures stripe and secret
  // point to ≥64 valid bytes.
  unsafe {
    let shift_32 = i64x2::splat(32);

    let mut i = 0usize;
    while i < 4 {
      let data_vec = vload(stripe.add(i.strict_mul(16)));
      let key_vec = vload(secret.add(i.strict_mul(16)));
      let data_key = data_vec ^ key_vec;

      // Isolate high32 in low32 position, then multiply even u32 lanes
      let data_key_hi = vshr_u64(data_key, shift_32);
      let product = vmul_even_u32(data_key, data_key_hi);

      // Swap u64 lanes and add data to accumulator
      let data_swap = vswap(data_vec);
      let sum = vadd_u64(acc[i], data_swap);
      acc[i] = vadd_u64(product, sum);

      i = i.strict_add(1);
    }
  }
}

/// Scramble the accumulator at block boundaries.
///
/// Per element: `acc = (xorshift64(acc, 47) ^ secret) * PRIME32_1`
/// The 64-bit multiply by a 32-bit prime is split into lo + hi halves.
#[inline]
#[target_feature(enable = "altivec", enable = "vsx", enable = "power8-vector")]
unsafe fn scramble_acc(acc: &mut [i64x2; 4], secret: *const u8) {
  // SAFETY: POWER8+ VSX via target_feature. Caller ensures secret points
  // to ≥64 valid bytes.
  unsafe {
    let prime_vec = i64x2::splat(PRIME32_1 as i64);
    let shift_47 = i64x2::splat(47);
    let shift_32 = i64x2::splat(32);

    let mut i = 0usize;
    while i < 4 {
      let acc_vec = acc[i];
      let shifted = vshr_u64(acc_vec, shift_47);
      let data_vec = acc_vec ^ shifted;

      let key_vec = vload(secret.add(i.strict_mul(16)));
      let data_key = data_vec ^ key_vec;

      // 64-bit multiply by PRIME32_1:
      // prod_lo = low32(data_key) × PRIME32_1
      // prod_hi = high32(data_key) × PRIME32_1, shifted left 32
      let data_key_hi = vshr_u64(data_key, shift_32);
      let prod_lo = vmul_even_u32(data_key, prime_vec);
      let prod_hi = vmul_even_u32(data_key_hi, prime_vec);
      acc[i] = vadd_u64(prod_lo, vshl_u64(prod_hi, shift_32));

      i = i.strict_add(1);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Long-path loop (SIMD inner, scalar merge)
// ─────────────────────────────────────────────────────────────────────────────

#[target_feature(enable = "altivec", enable = "vsx", enable = "power8-vector")]
unsafe fn hash_long_internal_loop(input: &[u8], secret: &[u8]) -> [u64; ACC_NB] {
  // SAFETY: POWER8+ VSX via target_feature. Input/secret bounds checked by caller.
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

/// XXH3 64-bit hash — POWER VSX kernel.
///
/// Delegates ≤240 B to portable scalar paths; >240 B uses VSX accumulator.
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
    // SAFETY: Dispatcher verifies POWER8+ VSX before selecting this kernel.
    let acc = unsafe { hash_long_internal_loop(input, &DEFAULT_SECRET) };
    super::merge_accs(
      &acc,
      &DEFAULT_SECRET,
      SECRET_MERGEACCS_START,
      (input.len() as u64).wrapping_mul(PRIME64_1),
    )
  } else {
    let secret = super::custom_default_secret(seed);
    // SAFETY: Dispatcher verifies POWER8+ VSX before selecting this kernel.
    let acc = unsafe { hash_long_internal_loop(input, &secret) };
    super::merge_accs(
      &acc,
      &secret,
      SECRET_MERGEACCS_START,
      (input.len() as u64).wrapping_mul(PRIME64_1),
    )
  }
}

/// XXH3 128-bit hash — POWER VSX kernel.
///
/// Delegates ≤240 B to portable scalar paths; >240 B uses VSX accumulator.
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
      // SAFETY: Dispatcher verifies POWER8+ VSX before selecting this kernel.
      unsafe { hash_long_internal_loop(input, &DEFAULT_SECRET) },
      &DEFAULT_SECRET[..],
    )
  } else {
    let secret = super::custom_default_secret(seed);
    // SAFETY: Dispatcher verifies POWER8+ VSX before selecting this kernel.
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
