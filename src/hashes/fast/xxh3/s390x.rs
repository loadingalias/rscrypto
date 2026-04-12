//! s390x z/Vector accelerated XXH3 kernel.
//!
//! Vectorizes the 8-stripe accumulator multiply-accumulate loop using
//! 128-bit z/Vector registers (4 × `i64x2` = 8 × u64).
//!
//! s390x is big-endian. XXH3 interprets data as little-endian u64s, so
//! all data and secret loads are byte-reversed per element using `vperm`.
//!
//! # Safety
//!
//! Uses `unsafe` for z/Vector inline asm. Callers must ensure z13+
//! vector facility before executing the accelerated path (the dispatcher
//! does this).
#![allow(unsafe_code)]
#![allow(clippy::indexing_slicing)]

use core::simd::i64x2;

use super::{
  ACC_NB, DEFAULT_SECRET, INITIAL_ACC, MID_SIZE_MAX, PRIME32_1, PRIME64_1, PRIME64_2, SECRET_CONSUME_RATE,
  SECRET_LASTACC_START, SECRET_MERGEACCS_START, STRIPE_LEN,
};

/// Byte-swap mask: reverses bytes within each u64 element (BE → LE).
///
/// Element 0: bytes 7,6,5,4,3,2,1,0  →  reverses bytes [0..8]
/// Element 1: bytes 15,14,13,12,11,10,9,8  →  reverses bytes [8..16]
const BSWAP_MASK: [u8; 16] = [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8];

// ─────────────────────────────────────────────────────────────────────────────
// z/Vector primitive operations (inline asm, z13+)
// ─────────────────────────────────────────────────────────────────────────────

/// Add u64 lanes: `vag`.
#[inline]
#[target_feature(enable = "vector")]
unsafe fn vag(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z13+ vector facility via target_feature.
  unsafe {
    core::arch::asm!(
      "vag {out}, {a}, {b}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// Logical shift right u64 lanes by immediate: `vesrlg`.
#[inline]
#[target_feature(enable = "vector")]
unsafe fn vesrlg<const SHIFT: u32>(a: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z13+ vector facility via target_feature.
  unsafe {
    core::arch::asm!(
      "vesrlg {out}, {a}, {shift}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      shift = const SHIFT,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// Shift left u64 lanes by immediate: `veslg`.
#[inline]
#[target_feature(enable = "vector")]
unsafe fn veslg<const SHIFT: u32>(a: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z13+ vector facility via target_feature.
  unsafe {
    core::arch::asm!(
      "veslg {out}, {a}, {shift}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      shift = const SHIFT,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// Multiply odd-indexed u32 lanes → u64: `vmlof`.
///
/// On s390x (big-endian), odd u32 elements are the low 32 bits of each
/// u64 lane. This gives: `low32(a) × low32(b) → u64` per lane.
#[inline]
#[target_feature(enable = "vector")]
unsafe fn vmlof(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z13+ vector facility via target_feature.
  unsafe {
    core::arch::asm!(
      "vmlof {out}, {a}, {b}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// Byte-permute: `vperm`.
///
/// Selects bytes from the concatenation of `a:a` according to `mask`.
/// Used to byte-reverse each u64 element (BE → LE).
#[inline]
#[target_feature(enable = "vector")]
unsafe fn vperm(a: i64x2, mask: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z13+ vector facility via target_feature.
  unsafe {
    core::arch::asm!(
      "vperm {out}, {a}, {a}, {mask}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      mask = in(vreg) mask,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// Swap u64 lanes (idx ^ 1 effect): `vpdi` with M3=4.
///
/// Result element 0 = source element 1, result element 1 = source element 0.
#[inline]
#[target_feature(enable = "vector")]
unsafe fn vpdi_swap(a: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z13+ vector facility via target_feature.
  unsafe {
    core::arch::asm!(
      "vpdi {out}, {a}, {a}, 4",
      out = lateout(vreg) out,
      a = in(vreg) a,
      options(nomem, nostack, pure)
    );
  }
  out
}

// ─────────────────────────────────────────────────────────────────────────────
// Load / store helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Load 128 bits from memory (unaligned, native byte order).
#[inline(always)]
unsafe fn vload_raw(ptr: *const u8) -> i64x2 {
  // SAFETY: caller ensures ptr is valid for 16 bytes.
  unsafe { core::ptr::read_unaligned(ptr as *const i64x2) }
}

/// Store 128 bits to memory (unaligned, native byte order).
#[inline(always)]
unsafe fn vstore(ptr: *mut u8, val: i64x2) {
  // SAFETY: caller ensures ptr is valid for 16 bytes.
  unsafe { core::ptr::write_unaligned(ptr as *mut i64x2, val) }
}

/// Load 128 bits with per-element byte-reversal (BE → LE).
#[inline]
#[target_feature(enable = "vector")]
unsafe fn vload_le(ptr: *const u8, bswap: i64x2) -> i64x2 {
  // SAFETY: caller ensures ptr is valid for 16 bytes.
  unsafe { vperm(vload_raw(ptr), bswap) }
}

/// Load the byte-swap permutation mask into a vector register.
#[inline(always)]
unsafe fn load_bswap_mask() -> i64x2 {
  // SAFETY: BSWAP_MASK is a 16-byte constant.
  unsafe { vload_raw(BSWAP_MASK.as_ptr()) }
}

// ─────────────────────────────────────────────────────────────────────────────
// SIMD accumulate + scramble
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "vector")]
unsafe fn load_acc(initial: &[u64; ACC_NB]) -> [i64x2; 4] {
  // SAFETY: z13+ vector facility via target_feature. Pointer valid for 8 × u64.
  // Accumulator values are native u64s — no byte-swap needed.
  unsafe {
    let p = initial.as_ptr() as *const u8;
    [
      vload_raw(p),
      vload_raw(p.add(16)),
      vload_raw(p.add(32)),
      vload_raw(p.add(48)),
    ]
  }
}

#[inline]
#[target_feature(enable = "vector")]
unsafe fn store_acc(acc: &[i64x2; 4]) -> [u64; ACC_NB] {
  // SAFETY: z13+ vector facility via target_feature.
  // Accumulator values are native u64s — no byte-swap needed.
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

/// Accumulate one 64-byte stripe into the z/Vector accumulator.
///
/// Per iteration (4 total, one per 16-byte chunk):
/// 1. Load 16 B of input and 16 B of secret (byte-reversed to LE)
/// 2. XOR to get data_key
/// 3. `vmlof`: low32(data_key) × high32(data_key) → u64
/// 4. Swap u64 lanes (idx ^ 1) and add data
/// 5. Accumulate product + swapped data into acc
#[inline]
#[target_feature(enable = "vector")]
unsafe fn accumulate_512(acc: &mut [i64x2; 4], stripe: *const u8, secret: *const u8) {
  // SAFETY: z13+ vector facility via target_feature. Caller ensures stripe
  // and secret point to ≥64 valid bytes.
  unsafe {
    let bswap = load_bswap_mask();

    let mut i = 0usize;
    while i < 4 {
      let data_vec = vload_le(stripe.add(i.strict_mul(16)), bswap);
      let key_vec = vload_le(secret.add(i.strict_mul(16)), bswap);
      let data_key = data_vec ^ key_vec;

      // Isolate high32 in low32 position (per u64), then multiply odd u32 lanes.
      // On BE, after vesrlg the original high32 lands in the odd (low32) position.
      let data_key_hi = vesrlg::<32>(data_key);
      let product = vmlof(data_key, data_key_hi);

      // Swap u64 lanes and add data to accumulator
      let data_swap = vpdi_swap(data_vec);
      let sum = vag(acc[i], data_swap);
      acc[i] = vag(product, sum);

      i = i.strict_add(1);
    }
  }
}

/// Scramble the accumulator at block boundaries.
///
/// Per element: `acc = (xorshift64(acc, 47) ^ secret) * PRIME32_1`
/// The 64-bit multiply by a 32-bit prime is split into lo + hi halves.
#[inline]
#[target_feature(enable = "vector")]
unsafe fn scramble_acc(acc: &mut [i64x2; 4], secret: *const u8) {
  // SAFETY: z13+ vector facility via target_feature. Caller ensures secret
  // points to ≥64 valid bytes.
  unsafe {
    let bswap = load_bswap_mask();
    let prime_vec = i64x2::splat(PRIME32_1 as i64);

    let mut i = 0usize;
    while i < 4 {
      let acc_vec = acc[i];
      let shifted = vesrlg::<47>(acc_vec);
      let data_vec = acc_vec ^ shifted;

      let key_vec = vload_le(secret.add(i.strict_mul(16)), bswap);
      let data_key = data_vec ^ key_vec;

      // 64-bit multiply by PRIME32_1:
      // prod_lo = low32(data_key) × PRIME32_1
      // prod_hi = high32(data_key) × PRIME32_1, shifted left 32
      let data_key_hi = vesrlg::<32>(data_key);
      let prod_lo = vmlof(data_key, prime_vec);
      let prod_hi = vmlof(data_key_hi, prime_vec);
      acc[i] = vag(prod_lo, veslg::<32>(prod_hi));

      i = i.strict_add(1);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Long-path loop (SIMD inner, scalar merge)
// ─────────────────────────────────────────────────────────────────────────────

#[target_feature(enable = "vector")]
unsafe fn hash_long_internal_loop(input: &[u8], secret: &[u8]) -> [u64; ACC_NB] {
  // SAFETY: z13+ vector facility via target_feature. Input/secret bounds
  // checked by caller.
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
    // SAFETY: Dispatcher verifies z13+ vector facility before selecting this kernel.
    let acc = unsafe { hash_long_internal_loop(input, &DEFAULT_SECRET) };
    super::merge_accs(
      &acc,
      &DEFAULT_SECRET,
      SECRET_MERGEACCS_START,
      (input.len() as u64).wrapping_mul(PRIME64_1),
    )
  } else {
    let secret = super::custom_default_secret(seed);
    // SAFETY: Dispatcher verifies z13+ vector facility before selecting this kernel.
    let acc = unsafe { hash_long_internal_loop(input, &secret) };
    super::merge_accs(
      &acc,
      &secret,
      SECRET_MERGEACCS_START,
      (input.len() as u64).wrapping_mul(PRIME64_1),
    )
  }
}

/// XXH3 64-bit hash — s390x z/Vector kernel.
///
/// Delegates ≤240 B to portable scalar paths; >240 B uses z/Vector accumulator.
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
    // SAFETY: Dispatcher verifies z13+ vector facility before selecting this kernel.
    let acc = unsafe { hash_long_internal_loop(input, &DEFAULT_SECRET) };
    xxh3_128_long_finalize(&acc, &DEFAULT_SECRET, input.len())
  } else {
    let secret = super::custom_default_secret(seed);
    // SAFETY: Dispatcher verifies z13+ vector facility before selecting this kernel.
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
