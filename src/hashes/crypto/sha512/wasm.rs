//! SHA-512 WebAssembly SIMD128 kernel.
//!
//! Vectorizes the message schedule computation using 128-bit SIMD (2 × u64
//! lanes). Compression rounds remain scalar (sequential data dependency).
//!
//! The schedule computes 2 words at a time: W[t] and W[t+1] are independent
//! (they depend on W[t-2]/W[t-1] respectively, both from prior rounds), so
//! we get true 2-wide parallelism for sigma computations and additions.
//!
//! Expected speedup: ~10-25% over the portable scalar implementation.

#![allow(clippy::indexing_slicing)] // Fixed-size arrays + compression schedule

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

use super::{BLOCK_LEN, K, big_sigma0, big_sigma1, ch, maj};

// ─────────────────────────────────────────────────────────────────────────────
// SIMD sigma functions for message schedule (2 × u64 lanes)
// ─────────────────────────────────────────────────────────────────────────────

/// SIMD rotate right: 3 ops per rotate (no native u64 rotate in WASM SIMD).
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn rotr_v(x: v128, n: u32) -> v128 {
  v128_or(u64x2_shr(x, n), u64x2_shl(x, 64u32.wrapping_sub(n)))
}

/// σ0(x) = ROTR(1) ^ ROTR(8) ^ SHR(7)  (lowercase sigma, message schedule)
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn small_sigma0_v(x: v128) -> v128 {
  v128_xor(v128_xor(rotr_v(x, 1), rotr_v(x, 8)), u64x2_shr(x, 7))
}

/// σ1(x) = ROTR(19) ^ ROTR(61) ^ SHR(6)  (lowercase sigma, message schedule)
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn small_sigma1_v(x: v128) -> v128 {
  v128_xor(v128_xor(rotr_v(x, 19), rotr_v(x, 61)), u64x2_shr(x, 6))
}

// ─────────────────────────────────────────────────────────────────────────────
// Message schedule helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Load 2 big-endian u64 message words from `ptr`, byte-swapping each.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn load_be(ptr: *const u8) -> v128 {
  // SAFETY: caller guarantees `ptr` is valid for a 16-byte read
  // and the simd128 target feature is enabled.
  let raw = unsafe { v128_load(ptr as *const v128) };
  i8x16_shuffle::<7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8>(raw, raw)
}

/// Extract [a[1], b[0]] — cross-register lane extraction for W[t-7] and σ0 inputs.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn cross_lanes(a: v128, b: v128) -> v128 {
  i64x2_shuffle::<1, 2>(a, b)
}

/// Compute 2 message schedule words W[2i] and W[2i+1] using SIMD.
///
/// Ring buffer `w` has 8 slots, each holding `[W[2j], W[2j+1]]`.
/// Called for pairs i = 8..39 (schedule words 16..79).
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn schedule_pair(w: &mut [v128; 8], i: usize) {
  // σ1 input: [W[2i-2], W[2i-1]]
  let s1_in = w[i.wrapping_sub(1) & 7];
  // W[t-7]: [W[2i-7], W[2i-6]] — crosses a pair boundary
  let w_tm7 = cross_lanes(w[i.wrapping_sub(4) & 7], w[i.wrapping_sub(3) & 7]);
  // σ0 input: [W[2i-15], W[2i-14]] — crosses a pair boundary
  let s0_in = cross_lanes(w[i.wrapping_sub(8) & 7], w[i.wrapping_sub(7) & 7]);
  // W[t-16]: [W[2i-16], W[2i-15]] — exact pair
  let w_tm16 = w[i.wrapping_sub(8) & 7];

  w[i & 7] = i64x2_add(
    i64x2_add(small_sigma1_v(s1_in), w_tm7),
    i64x2_add(small_sigma0_v(s0_in), w_tm16),
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Block compression
// ─────────────────────────────────────────────────────────────────────────────

/// SHA-512 multi-block compression using WebAssembly SIMD128.
///
/// Message schedule is computed with SIMD (2 u64 words per v128).
/// Compression rounds are scalar (sequential dependency chain).
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn compress_blocks_wasm_simd(state: &mut [u64; 8], blocks: &[u8]) {
  debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);
  if blocks.is_empty() {
    return;
  }

  let num_blocks = blocks.len() / BLOCK_LEN;
  let mut ptr = blocks.as_ptr();

  for _ in 0..num_blocks {
    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];

    // Load and byte-swap 16 message words into 8 × v128 (ring buffer).
    // SAFETY: `ptr` is valid for 128 bytes (one SHA-512 block).
    let mut wv: [v128; 8] = [
      unsafe { load_be(ptr) },
      unsafe { load_be(ptr.add(16)) },
      unsafe { load_be(ptr.add(32)) },
      unsafe { load_be(ptr.add(48)) },
      unsafe { load_be(ptr.add(64)) },
      unsafe { load_be(ptr.add(80)) },
      unsafe { load_be(ptr.add(96)) },
      unsafe { load_be(ptr.add(112)) },
    ];

    macro_rules! sha_round {
      ($k:expr, $w:expr) => {{
        let t1 = h
          .wrapping_add(big_sigma1(e))
          .wrapping_add(ch(e, f, g))
          .wrapping_add($k)
          .wrapping_add($w);
        let t2 = big_sigma0(a).wrapping_add(maj(a, b, c));
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
      }};
    }

    // Rounds 0-15: extract from loaded vectors.
    for pair in 0..8 {
      let v = wv[pair];
      let w_lo = i64x2_extract_lane::<0>(v) as u64;
      let w_hi = i64x2_extract_lane::<1>(v) as u64;
      let r = pair.strict_mul(2);
      sha_round!(K[r], w_lo);
      sha_round!(K[r + 1], w_hi);
    }

    // Rounds 16-79: expand schedule with SIMD, then extract for scalar rounds.
    for pair in 8..40 {
      schedule_pair(&mut wv, pair);
      let v = wv[pair & 7];
      let w_lo = i64x2_extract_lane::<0>(v) as u64;
      let w_hi = i64x2_extract_lane::<1>(v) as u64;
      let r = pair.strict_mul(2);
      sha_round!(K[r], w_lo);
      sha_round!(K[r + 1], w_hi);
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);

    // SAFETY: advancing by one block; total advance bounded by `num_blocks * 128`.
    ptr = unsafe { ptr.add(128) };
  }
}
