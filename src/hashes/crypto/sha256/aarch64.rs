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

/// Flat K[0..64] round constants for SHA-256, used as a `vld1q_u32` source.
///
/// Stored as a flat array (not grouped by 4) to allow simple pointer
/// arithmetic `K32.as_ptr().add(t)` for 128-bit NEON loads.
/// This matches the `sha2` crate's layout and avoids the closure indirection
/// of the previous `K4: [[u32; 4]; 16]` layout.
#[cfg(target_arch = "aarch64")]
#[repr(C, align(64))]
struct AlignedK32([u32; 64]);

#[cfg(target_arch = "aarch64")]
static K32: AlignedK32 = AlignedK32([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

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
    let kp = K32.0.as_ptr();

    for _ in 0..num_blocks {
      let abcd_save = abcd;
      let efgh_save = efgh;

      // Load and byte-swap 4 message vectors (16 words = 1 block).
      let mut s0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(ptr)));
      let mut s1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(ptr.add(16))));
      let mut s2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(ptr.add(32))));
      let mut s3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(ptr.add(48))));

      // Rounds 0-3
      let mut tmp = vaddq_u32(s0, vld1q_u32(kp));
      let mut abcd_prev = abcd;
      abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
      efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);

      // Rounds 4-7
      tmp = vaddq_u32(s1, vld1q_u32(kp.add(4)));
      abcd_prev = abcd;
      abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
      efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);

      // Rounds 8-11
      tmp = vaddq_u32(s2, vld1q_u32(kp.add(8)));
      abcd_prev = abcd;
      abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
      efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);

      // Rounds 12-15
      tmp = vaddq_u32(s3, vld1q_u32(kp.add(12)));
      abcd_prev = abcd;
      abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
      efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);

      // Rounds 16-63: compact loop (3 iterations × 16 rounds each).
      // The loop reduces I-cache footprint vs full unroll — critical on
      // Graviton's Neoverse V1/V2 where the competitor's compact kernel
      // beats our previously fully-unrolled version by ~11%.
      let mut t: usize = 16;
      while t < 64 {
        s0 = vsha256su1q_u32(vsha256su0q_u32(s0, s1), s2, s3);
        tmp = vaddq_u32(s0, vld1q_u32(kp.add(t)));
        abcd_prev = abcd;
        abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
        efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);

        s1 = vsha256su1q_u32(vsha256su0q_u32(s1, s2), s3, s0);
        tmp = vaddq_u32(s1, vld1q_u32(kp.add(t.strict_add(4))));
        abcd_prev = abcd;
        abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
        efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);

        s2 = vsha256su1q_u32(vsha256su0q_u32(s2, s3), s0, s1);
        tmp = vaddq_u32(s2, vld1q_u32(kp.add(t.strict_add(8))));
        abcd_prev = abcd;
        abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
        efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);

        s3 = vsha256su1q_u32(vsha256su0q_u32(s3, s0), s1, s2);
        tmp = vaddq_u32(s3, vld1q_u32(kp.add(t.strict_add(12))));
        abcd_prev = abcd;
        abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
        efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);

        t = t.strict_add(16);
      }

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
