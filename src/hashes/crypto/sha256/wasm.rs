//! SHA-256 WebAssembly SIMD128 kernel.
//!
//! Vectorizes the message schedule computation using 128-bit SIMD (4 × u32
//! lanes). Compression rounds remain scalar (sequential data dependency).
//!
//! Expected speedup: ~1.5-2x over the portable scalar implementation.

#![allow(clippy::indexing_slicing)] // Fixed-size arrays + compression schedule

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

use super::{BLOCK_LEN, K, ch, maj};
use crate::hashes::util::rotr32;

// ─────────────────────────────────────────────────────────────────────────────
// SIMD message schedule helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Load 4 big-endian message words from `ptr` into a v128, byte-swapping each.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn load_be(ptr: *const u8) -> v128 {
  // SAFETY: caller guarantees `ptr` is valid for a 16-byte aligned read
  // and the simd128 target feature is enabled.
  let raw = unsafe { v128_load(ptr as *const v128) };
  i8x16_shuffle::<3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12>(raw, raw)
}

/// Scalar sigma functions for the compression rounds.
#[inline(always)]
fn big_sigma0(x: u32) -> u32 {
  rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22)
}

#[inline(always)]
fn big_sigma1(x: u32) -> u32 {
  rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25)
}

#[inline(always)]
fn small_sigma0(x: u32) -> u32 {
  rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3)
}

#[inline(always)]
fn small_sigma1(x: u32) -> u32 {
  rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn schedule_word(w: &[v128; 16], idx: usize) -> u32 {
  let v = w[(idx >> 2) & 0xF];
  match idx & 3 {
    0 => u32x4_extract_lane::<0>(v),
    1 => u32x4_extract_lane::<1>(v),
    2 => u32x4_extract_lane::<2>(v),
    _ => u32x4_extract_lane::<3>(v),
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Message schedule expansion (SIMD)
// ─────────────────────────────────────────────────────────────────────────────

/// Expand the message schedule: compute W[16..64] from W[0..16] using SIMD.
///
/// W[t] = σ1(W[t-2]) + W[t-7] + σ0(W[t-15]) + W[t-16]
///
/// We process 4 words at a time using a ring buffer. Each invocation computes
/// W[t..t+4), with `i` equal to `t/4`.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn schedule_4(w: &mut [v128; 16], i: usize) {
  let t = i << 2;
  let mut slot = u32x4_splat(0);

  let w16_0 = schedule_word(w, t - 16);
  let w15_0 = schedule_word(w, t - 15);
  let w7_0 = schedule_word(w, t - 7);
  let w2_0 = schedule_word(w, t - 2);
  let w0 = small_sigma1(w2_0)
    .wrapping_add(w7_0)
    .wrapping_add(small_sigma0(w15_0))
    .wrapping_add(w16_0);

  let w16_1 = schedule_word(w, t - 15);
  let w15_1 = schedule_word(w, t - 14);
  let w7_1 = schedule_word(w, t - 6);
  let w2_1 = schedule_word(w, t - 1);
  let w1 = small_sigma1(w2_1)
    .wrapping_add(w7_1)
    .wrapping_add(small_sigma0(w15_1))
    .wrapping_add(w16_1);

  let w16_2 = schedule_word(w, t - 14);
  let w15_2 = schedule_word(w, t - 13);
  let w7_2 = schedule_word(w, t - 5);
  let w2_2 = w0;
  let w2 = small_sigma1(w2_2)
    .wrapping_add(w7_2)
    .wrapping_add(small_sigma0(w15_2))
    .wrapping_add(w16_2);

  let w16_3 = schedule_word(w, t - 13);
  let w15_3 = schedule_word(w, t - 12);
  let w7_3 = schedule_word(w, t - 4);
  let w2_3 = w1;
  let w3 = small_sigma1(w2_3)
    .wrapping_add(w7_3)
    .wrapping_add(small_sigma0(w15_3))
    .wrapping_add(w16_3);

  slot = u32x4_replace_lane::<0>(slot, w0);
  slot = u32x4_replace_lane::<1>(slot, w1);
  slot = u32x4_replace_lane::<2>(slot, w2);
  slot = u32x4_replace_lane::<3>(slot, w3);
  w[i & 0xF] = slot;
}

// ─────────────────────────────────────────────────────────────────────────────
// Block compression
// ─────────────────────────────────────────────────────────────────────────────

/// SHA-256 multi-block compression using WebAssembly SIMD128.
///
/// The message schedule is computed with SIMD (4 words per v128).
/// Compression rounds are scalar (sequential dependency chain).
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn compress_blocks_wasm_simd(state: &mut [u32; 8], blocks: &[u8]) {
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

    // Load and byte-swap message words into 16 v128 schedule slots.
    // Each slot holds one u32 splatted (we use v128 as storage, extract scalars for rounds).
    //
    // SAFETY: `ptr` is valid for reads up to `ptr + 64` (one SHA-256 block).
    // The outer loop iterates `num_blocks` times and advances `ptr` by 64 each iteration.
    let mut wv: [v128; 16] = [
      unsafe { load_be(ptr) },
      unsafe { load_be(ptr.add(16)) },
      unsafe { load_be(ptr.add(32)) },
      unsafe { load_be(ptr.add(48)) },
      // Remaining slots will be filled by schedule expansion.
      u32x4_splat(0),
      u32x4_splat(0),
      u32x4_splat(0),
      u32x4_splat(0),
      u32x4_splat(0),
      u32x4_splat(0),
      u32x4_splat(0),
      u32x4_splat(0),
      u32x4_splat(0),
      u32x4_splat(0),
      u32x4_splat(0),
      u32x4_splat(0),
    ];

    // Process initial 16 words (rounds 0-15) — already loaded.
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
    for r in 0..16 {
      let wi = u32x4_extract_lane::<0>(
        // Shift the desired lane into position 0.
        match r % 4 {
          0 => wv[r / 4],
          1 => i32x4_shuffle::<1, 0, 0, 0>(wv[r / 4], wv[r / 4]),
          2 => i32x4_shuffle::<2, 0, 0, 0>(wv[r / 4], wv[r / 4]),
          _ => i32x4_shuffle::<3, 0, 0, 0>(wv[r / 4], wv[r / 4]),
        },
      );
      sha_round!(K[r], wi);
    }

    // Rounds 16-63: expand schedule with SIMD, then extract scalar for rounds.
    for r in (16..64).step_by(4) {
      schedule_4(&mut wv, r / 4);
      let sched = wv[(r / 4) & 0xF];
      sha_round!(K[r], u32x4_extract_lane::<0>(sched));
      sha_round!(K[r + 1], u32x4_extract_lane::<1>(sched));
      sha_round!(K[r + 2], u32x4_extract_lane::<2>(sched));
      sha_round!(K[r + 3], u32x4_extract_lane::<3>(sched));
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);

    // SAFETY: advancing by one block (64 bytes); total advance bounded by `num_blocks * 64`.
    ptr = unsafe { ptr.add(64) };
  }
}
