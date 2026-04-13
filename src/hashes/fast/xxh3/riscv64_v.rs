//! RISC-V Vector (RVV) accelerated XXH3 kernel.
//!
//! Vectorizes the 8-stripe accumulator multiply-accumulate loop using
//! RVV registers at SEW=64, VL=2 (4 × 2-lane iterations per 64 B stripe).
//!
//! Uses fixed VL=2 for portability across all VLEN ≥ 128 implementations.
//! On wider VLEN machines (e.g. VLEN=256 on SpacemiT K1), upper lanes are
//! unused but the code is correct and still significantly faster than the
//! portable scalar fallback.
//!
//! Key advantage over POWER/s390x: RVV has full SEW=64 `vmul.vx`, so the
//! scramble multiply is a single instruction instead of a lo+hi split.
//!
//! # Safety
//!
//! Uses `unsafe` for RVV inline asm. Callers must ensure the V extension
//! is available before executing the accelerated path (the dispatcher
//! does this via `platform::caps::riscv::V`).
#![allow(unsafe_code)]
#![allow(clippy::indexing_slicing)]

use super::{
  ACC_NB, DEFAULT_SECRET, INITIAL_ACC, MID_SIZE_MAX, PRIME32_1, PRIME64_1, PRIME64_2, SECRET_CONSUME_RATE,
  SECRET_LASTACC_START, SECRET_MERGEACCS_START, STRIPE_LEN,
};

// ─────────────────────────────────────────────────────────────────────────────
// SIMD accumulate + scramble
// ─────────────────────────────────────────────────────────────────────────────

/// Accumulate `nstripes` consecutive 64-byte stripes into the accumulator.
///
/// **Fused loop**: loads v16-v19 once, processes all stripes, stores once.
/// This eliminates the per-stripe accumulator load/store that dominated
/// medium-size throughput on the in-order K1 core.
///
/// Per stripe, processes 4 × 16-byte chunks:
/// 1. Load data and secret, XOR → data_key
/// 2. product = lo32(data_key) × hi32(data_key)
/// 3. Swap u64 lanes (`vmv.x.s` + `vslide1down.vx`) for the idx^1 cross-add
/// 4. acc += product + swap (breaks the acc dependency chain)
///
/// Input pointer advances by `STRIPE_LEN` (64) per stripe.
/// Secret pointer advances by `SECRET_CONSUME_RATE` (8) per stripe.
///
/// Vector registers: v1-v5 temporaries, v16-v19 accumulator pairs.
///
/// # Safety
///
/// Caller must ensure `nstripes >= 1`, `input` points to `nstripes * 64`
/// valid bytes, and `secret` points to `(nstripes - 1) * 8 + 64` valid bytes.
#[inline]
#[target_feature(enable = "v")]
unsafe fn accumulate_stripes(acc: &mut [u64; ACC_NB], input: *const u8, secret: *const u8, nstripes: usize) {
  debug_assert!(nstripes >= 1);
  // SAFETY: V extension via target_feature. Caller ensures input/secret
  // bounds and nstripes >= 1, acc is valid for 8 × u64.
  unsafe {
    let mask: u64 = 0xFFFF_FFFF;
    let shift32: u64 = 32;
    core::arch::asm!(
      // Configure VL=2, SEW=64, LMUL=1, tail/mask agnostic
      "vsetivli zero, 2, e64, m1, ta, ma",

      // Load all 4 accumulator pairs from memory (once)
      "vle64.v v16, ({acc})",
      "addi {t1}, {acc}, 16",
      "vle64.v v17, ({t1})",
      "addi {t1}, {acc}, 32",
      "vle64.v v18, ({t1})",
      "addi {t1}, {acc}, 48",
      "vle64.v v19, ({t1})",

      // ── Stripe loop ──
      "2:",

      // Chunk 0: bytes 0..15
      "vle64.v v1, ({input})",
      "vle64.v v2, ({secret})",
      "vmv.x.s {t0}, v1",
      "vslide1down.vx v4, v1, {t0}",
      "vxor.vv v3, v1, v2",
      "vsrl.vx v2, v3, {shift32}",
      "vand.vx v5, v3, {mask}",
      "vmul.vv v3, v5, v2",
      "vadd.vv v3, v3, v4",
      "vadd.vv v16, v16, v3",

      // Chunk 1: bytes 16..31
      "addi {t1}, {input}, 16",
      "vle64.v v1, ({t1})",
      "addi {t1}, {secret}, 16",
      "vle64.v v2, ({t1})",
      "vmv.x.s {t0}, v1",
      "vslide1down.vx v4, v1, {t0}",
      "vxor.vv v3, v1, v2",
      "vsrl.vx v2, v3, {shift32}",
      "vand.vx v5, v3, {mask}",
      "vmul.vv v3, v5, v2",
      "vadd.vv v3, v3, v4",
      "vadd.vv v17, v17, v3",

      // Chunk 2: bytes 32..47
      "addi {t1}, {input}, 32",
      "vle64.v v1, ({t1})",
      "addi {t1}, {secret}, 32",
      "vle64.v v2, ({t1})",
      "vmv.x.s {t0}, v1",
      "vslide1down.vx v4, v1, {t0}",
      "vxor.vv v3, v1, v2",
      "vsrl.vx v2, v3, {shift32}",
      "vand.vx v5, v3, {mask}",
      "vmul.vv v3, v5, v2",
      "vadd.vv v3, v3, v4",
      "vadd.vv v18, v18, v3",

      // Chunk 3: bytes 48..63
      "addi {t1}, {input}, 48",
      "vle64.v v1, ({t1})",
      "addi {t1}, {secret}, 48",
      "vle64.v v2, ({t1})",
      "vmv.x.s {t0}, v1",
      "vslide1down.vx v4, v1, {t0}",
      "vxor.vv v3, v1, v2",
      "vsrl.vx v2, v3, {shift32}",
      "vand.vx v5, v3, {mask}",
      "vmul.vv v3, v5, v2",
      "vadd.vv v3, v3, v4",
      "vadd.vv v19, v19, v3",

      // Advance: input += STRIPE_LEN (64), secret += SECRET_CONSUME_RATE (8)
      "addi {input}, {input}, 64",
      "addi {secret}, {secret}, 8",

      // Loop control
      "addi {n}, {n}, -1",
      "bnez {n}, 2b",

      // Store all 4 accumulator pairs back to memory (once)
      "vse64.v v16, ({acc})",
      "addi {t1}, {acc}, 16",
      "vse64.v v17, ({t1})",
      "addi {t1}, {acc}, 32",
      "vse64.v v18, ({t1})",
      "addi {t1}, {acc}, 48",
      "vse64.v v19, ({t1})",

      acc = in(reg) acc.as_mut_ptr(),
      input = inout(reg) input => _,
      secret = inout(reg) secret => _,
      n = inout(reg) nstripes => _,
      mask = in(reg) mask,
      shift32 = in(reg) shift32,
      t0 = out(reg) _,
      t1 = out(reg) _,
      out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
      out("v16") _, out("v17") _, out("v18") _, out("v19") _,
      options(nostack)
    );
  }
}

/// Scramble the accumulator at block boundaries.
///
/// Per element: `acc = (xorshift64(acc, 47) ^ secret) * PRIME32_1`
///
/// RVV advantage: `vmul.vx` does full SEW=64 multiply in one instruction,
/// unlike POWER/s390x which need a lo32+hi32 split.
///
/// Vector registers: v1-v2 temporaries, v16-v19 accumulator pairs.
#[inline]
#[target_feature(enable = "v")]
unsafe fn scramble_acc(acc: &mut [u64; ACC_NB], secret: *const u8) {
  // SAFETY: V extension via target_feature. Caller ensures secret ≥ 64 B.
  unsafe {
    let shift47: u64 = 47;
    let prime: u64 = PRIME32_1 as u64;
    core::arch::asm!(
      "vsetivli zero, 2, e64, m1, ta, ma",

      // Load accumulator pairs
      "vle64.v v16, ({acc})",
      "addi {t1}, {acc}, 16",
      "vle64.v v17, ({t1})",
      "addi {t1}, {acc}, 32",
      "vle64.v v18, ({t1})",
      "addi {t1}, {acc}, 48",
      "vle64.v v19, ({t1})",

      // ── Pair 0 ──
      "vsrl.vx v1, v16, {shift47}",
      "vxor.vv v1, v16, v1",         // xorshift(acc, 47)
      "vle64.v v2, ({secret})",
      "vxor.vv v1, v1, v2",          // ^ secret
      "vmul.vx v16, v1, {prime}",    // * PRIME32_1

      // ── Pair 1 ──
      "vsrl.vx v1, v17, {shift47}",
      "vxor.vv v1, v17, v1",
      "addi {t1}, {secret}, 16",
      "vle64.v v2, ({t1})",
      "vxor.vv v1, v1, v2",
      "vmul.vx v17, v1, {prime}",

      // ── Pair 2 ──
      "vsrl.vx v1, v18, {shift47}",
      "vxor.vv v1, v18, v1",
      "addi {t1}, {secret}, 32",
      "vle64.v v2, ({t1})",
      "vxor.vv v1, v1, v2",
      "vmul.vx v18, v1, {prime}",

      // ── Pair 3 ──
      "vsrl.vx v1, v19, {shift47}",
      "vxor.vv v1, v19, v1",
      "addi {t1}, {secret}, 48",
      "vle64.v v2, ({t1})",
      "vxor.vv v1, v1, v2",
      "vmul.vx v19, v1, {prime}",

      // Store accumulator pairs back
      "vse64.v v16, ({acc})",
      "addi {t1}, {acc}, 16",
      "vse64.v v17, ({t1})",
      "addi {t1}, {acc}, 32",
      "vse64.v v18, ({t1})",
      "addi {t1}, {acc}, 48",
      "vse64.v v19, ({t1})",

      acc = in(reg) acc.as_mut_ptr(),
      secret = in(reg) secret,
      shift47 = in(reg) shift47,
      prime = in(reg) prime,
      t1 = out(reg) _,
      out("v1") _, out("v2") _,
      out("v16") _, out("v17") _, out("v18") _, out("v19") _,
      options(nostack)
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Long-path loop (SIMD inner, scalar merge)
// ─────────────────────────────────────────────────────────────────────────────

#[target_feature(enable = "v")]
unsafe fn hash_long_internal_loop(input: &[u8], secret: &[u8]) -> [u64; ACC_NB] {
  // SAFETY: V extension via target_feature. Input/secret bounds checked
  // by caller.
  unsafe {
    let mut acc = INITIAL_ACC;

    let nb_stripes = (secret.len().strict_sub(STRIPE_LEN)) / SECRET_CONSUME_RATE;
    let block_len = STRIPE_LEN.strict_mul(nb_stripes);
    let nb_blocks = (input.len().strict_sub(1)) / block_len;

    // Full blocks: accumulate all stripes in one fused asm call, then scramble.
    let mut block = 0usize;
    while block < nb_blocks {
      let input_off = block.strict_mul(block_len);
      accumulate_stripes(&mut acc, input.as_ptr().add(input_off), secret.as_ptr(), nb_stripes);
      scramble_acc(&mut acc, secret.as_ptr().add(secret.len().strict_sub(STRIPE_LEN)));
      block = block.strict_add(1);
    }

    // Remaining stripes in final partial block.
    let final_off = nb_blocks.strict_mul(block_len);
    let nb_stripes_final = (input.len().strict_sub(1).strict_sub(final_off)) / STRIPE_LEN;
    if nb_stripes_final > 0 {
      accumulate_stripes(
        &mut acc,
        input.as_ptr().add(final_off),
        secret.as_ptr(),
        nb_stripes_final,
      );
    }

    // Last stripe (may overlap with previous).
    accumulate_stripes(
      &mut acc,
      input.as_ptr().add(input.len().strict_sub(STRIPE_LEN)),
      secret
        .as_ptr()
        .add(secret.len().strict_sub(STRIPE_LEN).strict_sub(SECRET_LASTACC_START)),
      1,
    );

    acc
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level kernel functions (safe wrappers)
// ─────────────────────────────────────────────────────────────────────────────

/// Long-path entry point (>240B) — no ≤240B branches.
///
/// Called from compile-time dispatch when the caller already knows
/// `input.len() > MID_SIZE_MAX`.
pub fn xxh3_64_long(input: &[u8], seed: u64) -> u64 {
  if seed == 0 {
    // SAFETY: Dispatcher verifies V extension before selecting this kernel.
    let acc = unsafe { hash_long_internal_loop(input, &DEFAULT_SECRET) };
    super::merge_accs(
      &acc,
      &DEFAULT_SECRET,
      SECRET_MERGEACCS_START,
      (input.len() as u64).wrapping_mul(PRIME64_1),
    )
  } else {
    let secret = super::custom_default_secret(seed);
    // SAFETY: Dispatcher verifies V extension before selecting this kernel.
    let acc = unsafe { hash_long_internal_loop(input, &secret) };
    super::merge_accs(
      &acc,
      &secret,
      SECRET_MERGEACCS_START,
      (input.len() as u64).wrapping_mul(PRIME64_1),
    )
  }
}

/// XXH3 64-bit hash — RVV kernel.
///
/// Delegates ≤240 B to portable scalar paths; >240 B uses RVV accumulator.
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
    // SAFETY: Dispatcher verifies V extension before selecting this kernel.
    let acc = unsafe { hash_long_internal_loop(input, &DEFAULT_SECRET) };
    xxh3_128_long_finalize(&acc, &DEFAULT_SECRET, input.len())
  } else {
    let secret = super::custom_default_secret(seed);
    // SAFETY: Dispatcher verifies V extension before selecting this kernel.
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
