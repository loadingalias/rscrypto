//! SHA-256 x86_64 SHA-NI kernel.
//!
//! Uses Intel SHA Extensions (`_mm_sha256rnds2_epi32`, `_mm_sha256msg1_epi32`,
//! `_mm_sha256msg2_epi32`) for hardware-accelerated compression.
//!
//! # Safety
//!
//! All functions require `sha` and `sse4.1` target features.
//! Callers must verify CPU capabilities before calling.

#![allow(unsafe_code)]
#![allow(clippy::inline_always)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Round constants K[0..64] packed as 16 groups of 4 × u32, pre-arranged
/// for direct `_mm_add_epi32` with message words.
#[cfg(target_arch = "x86_64")]
// SAFETY: [u32; 4] and __m128i have identical size/alignment; transmute is sound.
static K4: [__m128i; 16] = unsafe {
  core::mem::transmute([
    [0x428a2f98u32, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5],
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
  ])
};

/// Byte-swap mask for converting little-endian loads to big-endian message words.
#[cfg(target_arch = "x86_64")]
// SAFETY: [u8; 16] and __m128i have identical size; transmute is sound.
static BSWAP_MASK: __m128i = unsafe { core::mem::transmute([3u8, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12]) };

/// SHA-256 multi-block compression using SHA-NI instructions.
///
/// Processes one or more consecutive 64-byte blocks, updating the 8-word state
/// in-place. State lives in two XMM registers (ABEF, CDGH) across blocks — no
/// reload between blocks.
///
/// # Safety
///
/// Caller must ensure `sha` and `sse4.1` CPU features are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sha,sse4.1")]
pub(crate) unsafe fn compress_blocks_sha_ni(state: &mut [u32; 8], blocks: &[u8]) {
  debug_assert_eq!(blocks.len() % 64, 0);
  if blocks.is_empty() {
    return;
  }

  // SAFETY: SHA-NI/SSE4.1 intrinsics are available via this function's #[target_feature] attribute.
  // Pointer arithmetic on `ptr` is bounded by `blocks.len()`.
  unsafe {
    // SHA-NI state layout:
    //   ABEF = [A, B, E, F]  (lanes 3,2,1,0 after shuffle)
    //   CDGH = [C, D, G, H]
    //
    // Load state and rearrange from [A,B,C,D,E,F,G,H] to ABEF/CDGH form.
    let mut abef;
    let mut cdgh;
    {
      // tmp0 = [D, C, B, A], tmp1 = [H, G, F, E]
      let tmp0 = _mm_shuffle_epi32(_mm_loadu_si128(state.as_ptr().cast()), 0xB1);
      let tmp1 = _mm_shuffle_epi32(_mm_loadu_si128(state.as_ptr().add(4).cast()), 0x1B);
      // ABEF = [A, B, E, F], CDGH = [C, D, G, H]
      abef = _mm_alignr_epi8(tmp0, tmp1, 8);
      cdgh = _mm_blend_epi16(tmp1, tmp0, 0xF0);
    }

    let num_blocks = blocks.len() / 64;
    let mut ptr = blocks.as_ptr();

    for _ in 0..num_blocks {
      let abef_save = abef;
      let cdgh_save = cdgh;

      // Load and byte-swap 4 message vectors (16 words = 1 block).
      let mut msg0 = _mm_shuffle_epi8(_mm_loadu_si128(ptr.cast()), BSWAP_MASK);
      let mut msg1 = _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(16).cast()), BSWAP_MASK);
      let mut msg2 = _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(32).cast()), BSWAP_MASK);
      let mut msg3 = _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(48).cast()), BSWAP_MASK);

      // Rounds 0-3
      let mut tmp = _mm_add_epi32(msg0, K4[0]);
      cdgh = _mm_sha256rnds2_epu32(cdgh, abef, tmp);
      tmp = _mm_shuffle_epi32(tmp, 0x0E);
      abef = _mm_sha256rnds2_epu32(abef, cdgh, tmp);

      // Rounds 4-7
      tmp = _mm_add_epi32(msg1, K4[1]);
      cdgh = _mm_sha256rnds2_epu32(cdgh, abef, tmp);
      tmp = _mm_shuffle_epi32(tmp, 0x0E);
      abef = _mm_sha256rnds2_epu32(abef, cdgh, tmp);
      msg0 = _mm_sha256msg1_epu32(msg0, msg1);

      // Rounds 8-11
      tmp = _mm_add_epi32(msg2, K4[2]);
      cdgh = _mm_sha256rnds2_epu32(cdgh, abef, tmp);
      tmp = _mm_shuffle_epi32(tmp, 0x0E);
      abef = _mm_sha256rnds2_epu32(abef, cdgh, tmp);
      msg1 = _mm_sha256msg1_epu32(msg1, msg2);

      // Rounds 12-15
      tmp = _mm_add_epi32(msg3, K4[3]);
      cdgh = _mm_sha256rnds2_epu32(cdgh, abef, tmp);
      tmp = _mm_shuffle_epi32(tmp, 0x0E);
      abef = _mm_sha256rnds2_epu32(abef, cdgh, tmp);
      msg0 = _mm_add_epi32(msg0, _mm_alignr_epi8(msg3, msg2, 4));
      msg0 = _mm_sha256msg2_epu32(msg0, msg3);
      msg2 = _mm_sha256msg1_epu32(msg2, msg3);

      // Rounds 16-19
      tmp = _mm_add_epi32(msg0, K4[4]);
      cdgh = _mm_sha256rnds2_epu32(cdgh, abef, tmp);
      tmp = _mm_shuffle_epi32(tmp, 0x0E);
      abef = _mm_sha256rnds2_epu32(abef, cdgh, tmp);
      msg1 = _mm_add_epi32(msg1, _mm_alignr_epi8(msg0, msg3, 4));
      msg1 = _mm_sha256msg2_epu32(msg1, msg0);
      msg3 = _mm_sha256msg1_epu32(msg3, msg0);

      // Rounds 20-23
      tmp = _mm_add_epi32(msg1, K4[5]);
      cdgh = _mm_sha256rnds2_epu32(cdgh, abef, tmp);
      tmp = _mm_shuffle_epi32(tmp, 0x0E);
      abef = _mm_sha256rnds2_epu32(abef, cdgh, tmp);
      msg2 = _mm_add_epi32(msg2, _mm_alignr_epi8(msg1, msg0, 4));
      msg2 = _mm_sha256msg2_epu32(msg2, msg1);
      msg0 = _mm_sha256msg1_epu32(msg0, msg1);

      // Rounds 24-27
      tmp = _mm_add_epi32(msg2, K4[6]);
      cdgh = _mm_sha256rnds2_epu32(cdgh, abef, tmp);
      tmp = _mm_shuffle_epi32(tmp, 0x0E);
      abef = _mm_sha256rnds2_epu32(abef, cdgh, tmp);
      msg3 = _mm_add_epi32(msg3, _mm_alignr_epi8(msg2, msg1, 4));
      msg3 = _mm_sha256msg2_epu32(msg3, msg2);
      msg1 = _mm_sha256msg1_epu32(msg1, msg2);

      // Rounds 28-31
      tmp = _mm_add_epi32(msg3, K4[7]);
      cdgh = _mm_sha256rnds2_epu32(cdgh, abef, tmp);
      tmp = _mm_shuffle_epi32(tmp, 0x0E);
      abef = _mm_sha256rnds2_epu32(abef, cdgh, tmp);
      msg0 = _mm_add_epi32(msg0, _mm_alignr_epi8(msg3, msg2, 4));
      msg0 = _mm_sha256msg2_epu32(msg0, msg3);
      msg2 = _mm_sha256msg1_epu32(msg2, msg3);

      // Rounds 32-35
      tmp = _mm_add_epi32(msg0, K4[8]);
      cdgh = _mm_sha256rnds2_epu32(cdgh, abef, tmp);
      tmp = _mm_shuffle_epi32(tmp, 0x0E);
      abef = _mm_sha256rnds2_epu32(abef, cdgh, tmp);
      msg1 = _mm_add_epi32(msg1, _mm_alignr_epi8(msg0, msg3, 4));
      msg1 = _mm_sha256msg2_epu32(msg1, msg0);
      msg3 = _mm_sha256msg1_epu32(msg3, msg0);

      // Rounds 36-39
      tmp = _mm_add_epi32(msg1, K4[9]);
      cdgh = _mm_sha256rnds2_epu32(cdgh, abef, tmp);
      tmp = _mm_shuffle_epi32(tmp, 0x0E);
      abef = _mm_sha256rnds2_epu32(abef, cdgh, tmp);
      msg2 = _mm_add_epi32(msg2, _mm_alignr_epi8(msg1, msg0, 4));
      msg2 = _mm_sha256msg2_epu32(msg2, msg1);
      msg0 = _mm_sha256msg1_epu32(msg0, msg1);

      // Rounds 40-43
      tmp = _mm_add_epi32(msg2, K4[10]);
      cdgh = _mm_sha256rnds2_epu32(cdgh, abef, tmp);
      tmp = _mm_shuffle_epi32(tmp, 0x0E);
      abef = _mm_sha256rnds2_epu32(abef, cdgh, tmp);
      msg3 = _mm_add_epi32(msg3, _mm_alignr_epi8(msg2, msg1, 4));
      msg3 = _mm_sha256msg2_epu32(msg3, msg2);
      msg1 = _mm_sha256msg1_epu32(msg1, msg2);

      // Rounds 44-47
      tmp = _mm_add_epi32(msg3, K4[11]);
      cdgh = _mm_sha256rnds2_epu32(cdgh, abef, tmp);
      tmp = _mm_shuffle_epi32(tmp, 0x0E);
      abef = _mm_sha256rnds2_epu32(abef, cdgh, tmp);
      msg0 = _mm_add_epi32(msg0, _mm_alignr_epi8(msg3, msg2, 4));
      msg0 = _mm_sha256msg2_epu32(msg0, msg3);
      msg2 = _mm_sha256msg1_epu32(msg2, msg3);

      // Rounds 48-51
      tmp = _mm_add_epi32(msg0, K4[12]);
      cdgh = _mm_sha256rnds2_epu32(cdgh, abef, tmp);
      tmp = _mm_shuffle_epi32(tmp, 0x0E);
      abef = _mm_sha256rnds2_epu32(abef, cdgh, tmp);
      msg1 = _mm_add_epi32(msg1, _mm_alignr_epi8(msg0, msg3, 4));
      msg1 = _mm_sha256msg2_epu32(msg1, msg0);
      msg3 = _mm_sha256msg1_epu32(msg3, msg0);

      // Rounds 52-55
      tmp = _mm_add_epi32(msg1, K4[13]);
      cdgh = _mm_sha256rnds2_epu32(cdgh, abef, tmp);
      tmp = _mm_shuffle_epi32(tmp, 0x0E);
      abef = _mm_sha256rnds2_epu32(abef, cdgh, tmp);
      msg2 = _mm_add_epi32(msg2, _mm_alignr_epi8(msg1, msg0, 4));
      msg2 = _mm_sha256msg2_epu32(msg2, msg1);

      // Rounds 56-59
      tmp = _mm_add_epi32(msg2, K4[14]);
      cdgh = _mm_sha256rnds2_epu32(cdgh, abef, tmp);
      tmp = _mm_shuffle_epi32(tmp, 0x0E);
      abef = _mm_sha256rnds2_epu32(abef, cdgh, tmp);
      msg3 = _mm_add_epi32(msg3, _mm_alignr_epi8(msg2, msg1, 4));
      msg3 = _mm_sha256msg2_epu32(msg3, msg2);

      // Rounds 60-63
      tmp = _mm_add_epi32(msg3, K4[15]);
      cdgh = _mm_sha256rnds2_epu32(cdgh, abef, tmp);
      tmp = _mm_shuffle_epi32(tmp, 0x0E);
      abef = _mm_sha256rnds2_epu32(abef, cdgh, tmp);

      // Add saved state back.
      abef = _mm_add_epi32(abef, abef_save);
      cdgh = _mm_add_epi32(cdgh, cdgh_save);

      ptr = ptr.add(64);
    }

    // Store state back: convert from ABEF/CDGH to [A,B,C,D,E,F,G,H].
    let tmp0 = _mm_shuffle_epi32(abef, 0x1B);
    let tmp1 = _mm_shuffle_epi32(cdgh, 0xB1);
    _mm_storeu_si128(state.as_mut_ptr().cast(), _mm_blend_epi16(tmp0, tmp1, 0xF0));
    _mm_storeu_si128(state.as_mut_ptr().add(4).cast(), _mm_alignr_epi8(tmp1, tmp0, 8));
  } // unsafe
}
