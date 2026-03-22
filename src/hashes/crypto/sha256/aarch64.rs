//! SHA-256 aarch64 SHA2 Crypto Extension kernel.
//!
//! Uses ARMv8 SHA2 instructions (`vsha256hq_u32`, `vsha256h2q_u32`,
//! `vsha256su0q_u32`, `vsha256su1q_u32`) for hardware-accelerated compression.
//!
//! # Safety
//!
//! All functions require the `sha2` target feature.
//! Callers must verify CPU capabilities before calling.

#![allow(unsafe_code)]
#![allow(clippy::inline_always)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// Round constants K[0..64] packed as 16 groups of 4 × u32.
#[cfg(target_arch = "aarch64")]
static K4: [[u32; 4]; 16] = [
  [0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5],
  [0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5],
  [0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3],
  [0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174],
  [0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc],
  [0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da],
  [0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7],
  [0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967],
  [0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13],
  [0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85],
  [0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3],
  [0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070],
  [0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5],
  [0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3],
  [0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208],
  [0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2],
];

/// Perform 4 SHA-256 rounds with message schedule update.
///
/// Computes `vsha256hq_u32` + `vsha256h2q_u32` (4 rounds), then advances
/// the message schedule with `vsha256su0q_u32` + `vsha256su1q_u32`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sha2")]
#[inline]
unsafe fn sha256_4rounds(abcd: &mut uint32x4_t, efgh: &mut uint32x4_t, w: uint32x4_t, k: uint32x4_t) {
  let tmp = vaddq_u32(w, k);
  let abcd_prev = *abcd;
  *abcd = vsha256hq_u32(*abcd, *efgh, tmp);
  *efgh = vsha256h2q_u32(*efgh, abcd_prev, tmp);
}

/// SHA-256 multi-block compression using ARM SHA2 Crypto Extension.
///
/// Processes one or more consecutive 64-byte blocks, updating the 8-word state
/// in-place. State lives in two NEON registers (ABCD, EFGH) across blocks.
///
/// # Safety
///
/// Caller must ensure `sha2` CPU feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sha2")]
pub(crate) unsafe fn compress_blocks_aarch64_sha2(state: &mut [u32; 8], blocks: &[u8]) {
  debug_assert_eq!(blocks.len() % 64, 0);
  if blocks.is_empty() {
    return;
  }

  // SAFETY: NEON/SHA2 intrinsics are available via this function's #[target_feature] attribute.
  // Pointer arithmetic on `ptr` is bounded by `blocks.len()`.
  unsafe {
    // Load state: ABCD = [A,B,C,D], EFGH = [E,F,G,H]
    let mut abcd = vld1q_u32(state.as_ptr());
    let mut efgh = vld1q_u32(state.as_ptr().add(4));

    let num_blocks = blocks.len() / 64;
    let mut ptr = blocks.as_ptr();

    for _ in 0..num_blocks {
      let abcd_save = abcd;
      let efgh_save = efgh;

      // Load and byte-swap 4 message vectors (16 words = 1 block).
      let mut w0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(ptr)));
      let mut w1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(ptr.add(16))));
      let mut w2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(ptr.add(32))));
      let mut w3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(ptr.add(48))));

      let k = |i: usize| -> uint32x4_t { vld1q_u32(K4[i].as_ptr()) };

      // Rounds 0-3
      sha256_4rounds(&mut abcd, &mut efgh, w0, k(0));
      // Rounds 4-7
      sha256_4rounds(&mut abcd, &mut efgh, w1, k(1));
      // Rounds 8-11
      sha256_4rounds(&mut abcd, &mut efgh, w2, k(2));
      // Rounds 12-15
      sha256_4rounds(&mut abcd, &mut efgh, w3, k(3));

      // Rounds 16-19 + schedule
      w0 = vsha256su1q_u32(vsha256su0q_u32(w0, w1), w2, w3);
      sha256_4rounds(&mut abcd, &mut efgh, w0, k(4));
      // Rounds 20-23
      w1 = vsha256su1q_u32(vsha256su0q_u32(w1, w2), w3, w0);
      sha256_4rounds(&mut abcd, &mut efgh, w1, k(5));
      // Rounds 24-27
      w2 = vsha256su1q_u32(vsha256su0q_u32(w2, w3), w0, w1);
      sha256_4rounds(&mut abcd, &mut efgh, w2, k(6));
      // Rounds 28-31
      w3 = vsha256su1q_u32(vsha256su0q_u32(w3, w0), w1, w2);
      sha256_4rounds(&mut abcd, &mut efgh, w3, k(7));

      // Rounds 32-35
      w0 = vsha256su1q_u32(vsha256su0q_u32(w0, w1), w2, w3);
      sha256_4rounds(&mut abcd, &mut efgh, w0, k(8));
      // Rounds 36-39
      w1 = vsha256su1q_u32(vsha256su0q_u32(w1, w2), w3, w0);
      sha256_4rounds(&mut abcd, &mut efgh, w1, k(9));
      // Rounds 40-43
      w2 = vsha256su1q_u32(vsha256su0q_u32(w2, w3), w0, w1);
      sha256_4rounds(&mut abcd, &mut efgh, w2, k(10));
      // Rounds 44-47
      w3 = vsha256su1q_u32(vsha256su0q_u32(w3, w0), w1, w2);
      sha256_4rounds(&mut abcd, &mut efgh, w3, k(11));

      // Rounds 48-51
      w0 = vsha256su1q_u32(vsha256su0q_u32(w0, w1), w2, w3);
      sha256_4rounds(&mut abcd, &mut efgh, w0, k(12));
      // Rounds 52-55
      w1 = vsha256su1q_u32(vsha256su0q_u32(w1, w2), w3, w0);
      sha256_4rounds(&mut abcd, &mut efgh, w1, k(13));
      // Rounds 56-59
      w2 = vsha256su1q_u32(vsha256su0q_u32(w2, w3), w0, w1);
      sha256_4rounds(&mut abcd, &mut efgh, w2, k(14));
      // Rounds 60-63
      w3 = vsha256su1q_u32(vsha256su0q_u32(w3, w0), w1, w2);
      sha256_4rounds(&mut abcd, &mut efgh, w3, k(15));

      // Add saved state back.
      abcd = vaddq_u32(abcd, abcd_save);
      efgh = vaddq_u32(efgh, efgh_save);

      ptr = ptr.add(64);
    }

    // Store state back.
    vst1q_u32(state.as_mut_ptr(), abcd);
    vst1q_u32(state.as_mut_ptr().add(4), efgh);
  } // unsafe
}
