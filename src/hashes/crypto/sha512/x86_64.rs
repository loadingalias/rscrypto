//! SHA-512 x86_64 SHA-512 NI kernel.
//!
//! Uses Intel SHA-512 New Instructions (`_mm256_sha512rnds2_epi64`,
//! `_mm256_sha512msg1_epi64`, `_mm256_sha512msg2_epi64`) for
//! hardware-accelerated compression.
//!
//! **First pure-Rust crate with this backend.**
//!
//! Supported: Intel Arrow Lake-S (desktop), Lunar Lake (mobile),
//! Clearwater Forest (Atom server), Panther Lake (2025+).
//! Not supported on any AMD CPU (Zen 5/6 confirmed no SHA-512 NI).
//!
//! # Safety
//!
//! All functions require `sha512` and `avx2` target features.
//! Callers must verify CPU capabilities before calling.

#![allow(unsafe_code)]
#![allow(clippy::inline_always)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// K constants packed as 20 groups of 4 × u64, matching the 4 message vectors.
/// Each group is added to the corresponding msg vector before extracting
/// __m128i pairs for `sha512rnds2`.
#[cfg(target_arch = "x86_64")]
#[repr(C, align(64))]
struct AlignedK256([[u64; 4]; 20]);

#[cfg(target_arch = "x86_64")]
impl core::ops::Deref for AlignedK256 {
  type Target = [[u64; 4]; 20];
  #[inline(always)]
  fn deref(&self) -> &[[u64; 4]; 20] {
    &self.0
  }
}

#[cfg(target_arch = "x86_64")]
static K4: AlignedK256 = AlignedK256([
  [
    0x428a2f98d728ae22,
    0x7137449123ef65cd,
    0xb5c0fbcfec4d3b2f,
    0xe9b5dba58189dbbc,
  ],
  [
    0x3956c25bf348b538,
    0x59f111f1b605d019,
    0x923f82a4af194f9b,
    0xab1c5ed5da6d8118,
  ],
  [
    0xd807aa98a3030242,
    0x12835b0145706fbe,
    0x243185be4ee4b28c,
    0x550c7dc3d5ffb4e2,
  ],
  [
    0x72be5d74f27b896f,
    0x80deb1fe3b1696b1,
    0x9bdc06a725c71235,
    0xc19bf174cf692694,
  ],
  [
    0xe49b69c19ef14ad2,
    0xefbe4786384f25e3,
    0x0fc19dc68b8cd5b5,
    0x240ca1cc77ac9c65,
  ],
  [
    0x2de92c6f592b0275,
    0x4a7484aa6ea6e483,
    0x5cb0a9dcbd41fbd4,
    0x76f988da831153b5,
  ],
  [
    0x983e5152ee66dfab,
    0xa831c66d2db43210,
    0xb00327c898fb213f,
    0xbf597fc7beef0ee4,
  ],
  [
    0xc6e00bf33da88fc2,
    0xd5a79147930aa725,
    0x06ca6351e003826f,
    0x142929670a0e6e70,
  ],
  [
    0x27b70a8546d22ffc,
    0x2e1b21385c26c926,
    0x4d2c6dfc5ac42aed,
    0x53380d139d95b3df,
  ],
  [
    0x650a73548baf63de,
    0x766a0abb3c77b2a8,
    0x81c2c92e47edaee6,
    0x92722c851482353b,
  ],
  [
    0xa2bfe8a14cf10364,
    0xa81a664bbc423001,
    0xc24b8b70d0f89791,
    0xc76c51a30654be30,
  ],
  [
    0xd192e819d6ef5218,
    0xd69906245565a910,
    0xf40e35855771202a,
    0x106aa07032bbd1b8,
  ],
  [
    0x19a4c116b8d2d0c8,
    0x1e376c085141ab53,
    0x2748774cdf8eeb99,
    0x34b0bcb5e19b48a8,
  ],
  [
    0x391c0cb3c5c95a63,
    0x4ed8aa4ae3418acb,
    0x5b9cca4f7763e373,
    0x682e6ff3d6b2b8a3,
  ],
  [
    0x748f82ee5defb2fc,
    0x78a5636f43172f60,
    0x84c87814a1f0ab72,
    0x8cc702081a6439ec,
  ],
  [
    0x90befffa23631e28,
    0xa4506cebde82bde9,
    0xbef9a3f7b2c67915,
    0xc67178f2e372532b,
  ],
  [
    0xca273eceea26619c,
    0xd186b8c721c0c207,
    0xeada7dd6cde0eb1e,
    0xf57d4f7fee6ed178,
  ],
  [
    0x06f067aa72176fba,
    0x0a637dc5a2c898a6,
    0x113f9804bef90dae,
    0x1b710b35131c471b,
  ],
  [
    0x28db77f523047d84,
    0x32caab7b40c72493,
    0x3c9ebe0a15c9bebc,
    0x431d67c49c100d4c,
  ],
  [
    0x4cc5d4becb3e42b6,
    0x597f299cfc657e2a,
    0x5fcb6fab3ad6faec,
    0x6c44198c4a475817,
  ],
]);

/// Byte-swap mask for converting little-endian loads to big-endian u64 words.
#[cfg(target_arch = "x86_64")]
static BSWAP64_MASK: [u8; 32] = [
  7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 23, 22, 21, 20, 19, 18, 17, 16, 31, 30, 29, 28, 27, 26, 25, 24,
];

/// Extract W[t-7] spanning two consecutive __m256i message vectors.
///
/// Given a = [W[n], W[n+1], W[n+2], W[n+3]] and b = [W[n+4], ...],
/// returns [W[n+1], W[n+2], W[n+3], W[n+4]].
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn extract_w_tm7(a: __m256i, b: __m256i) -> __m256i {
  // SAFETY: AVX2 intrinsics are available via this function's #[target_feature] attribute.
  unsafe {
    // Swap halves: [a_hi, b_lo]
    let t = _mm256_permute2x128_si256(a, b, 0x21);
    // Within-lane align by 8 bytes (1 u64):
    //   lo: concat(t_lo, a_lo) >> 64 = [W[n+1], W[n+2]]
    //   hi: concat(t_hi, a_hi) >> 64 = [W[n+3], W[n+4]]
    _mm256_alignr_epi8(t, a, 8)
  }
}

/// SHA-512 multi-block compression using SHA-512 NI instructions.
///
/// Processes one or more consecutive 128-byte blocks, updating the 8-word
/// state in-place. State lives in two YMM registers (ABEF, CDGH) across
/// blocks — no reload between blocks.
///
/// # Safety
///
/// Caller must ensure `sha512` and `avx2` CPU features are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sha512,avx2")]
pub(crate) unsafe fn compress_blocks_sha512_ni(state: &mut [u64; 8], blocks: &[u8]) {
  debug_assert_eq!(blocks.len() % 128, 0);
  if blocks.is_empty() {
    return;
  }

  // SAFETY: SHA-512 NI/AVX2 intrinsics are available via this function's #[target_feature]
  // attribute. Pointer arithmetic on `ptr` is bounded by `blocks.len()`.
  unsafe {
    let bswap = _mm256_loadu_si256(BSWAP64_MASK.as_ptr().cast());

    // State layout:
    //   ABEF = [A, B, E, F]  (__m256i, 4 × u64)
    //   CDGH = [C, D, G, H]  (__m256i, 4 × u64)
    //
    // Load state [A,B,C,D,E,F,G,H] and rearrange.
    let abcd = _mm256_loadu_si256(state.as_ptr().cast()); // [A, B, C, D]
    let efgh = _mm256_loadu_si256(state.as_ptr().add(4).cast()); // [E, F, G, H]

    // ABEF = [A, B, E, F]: lo←abcd_lo, hi←efgh_lo
    let mut abef = _mm256_permute2x128_si256(abcd, efgh, 0x20);
    // CDGH = [C, D, G, H]: lo←abcd_hi, hi←efgh_hi
    let mut cdgh = _mm256_permute2x128_si256(abcd, efgh, 0x31);

    let num_blocks = blocks.len() / 128;
    let mut ptr = blocks.as_ptr();

    for _ in 0..num_blocks {
      let abef_save = abef;
      let cdgh_save = cdgh;

      // Load and byte-swap 4 message vectors (16 × u64 = 1 block).
      let mut msg0 = _mm256_shuffle_epi8(_mm256_loadu_si256(ptr.cast()), bswap);
      let mut msg1 = _mm256_shuffle_epi8(_mm256_loadu_si256(ptr.add(32).cast()), bswap);
      let mut msg2 = _mm256_shuffle_epi8(_mm256_loadu_si256(ptr.add(64).cast()), bswap);
      let mut msg3 = _mm256_shuffle_epi8(_mm256_loadu_si256(ptr.add(96).cast()), bswap);

      // Each iteration processes 4 rounds (2 rnds2 calls × 2 rounds each).
      // Total: 20 iterations × 4 rounds = 80 rounds.
      //
      // rnds2(cdgh, abef, rk) → new ABEF; new CDGH = old ABEF.
      // By alternating which variable receives the result, names stay consistent
      // at even-iteration boundaries.
      macro_rules! rounds4 {
        ($msg:expr, $ki:expr) => {{
          let wk = _mm256_add_epi64($msg, _mm256_loadu_si256(K4[$ki].as_ptr().cast()));
          cdgh = _mm256_sha512rnds2_epi64(cdgh, abef, _mm256_castsi256_si128(wk));
          abef = _mm256_sha512rnds2_epi64(abef, cdgh, _mm256_extracti128_si256(wk, 1));
        }};
      }

      // ---- Iteration 0 (rounds 0-3): msg0, no schedule ----
      rounds4!(msg0, 0);

      // ---- Iteration 1 (rounds 4-7): msg1, start msg0 schedule ----
      rounds4!(msg1, 1);
      msg0 = _mm256_sha512msg1_epi64(msg0, _mm256_castsi256_si128(msg1));

      // ---- Iteration 2 (rounds 8-11): msg2, start msg1 schedule ----
      rounds4!(msg2, 2);
      msg1 = _mm256_sha512msg1_epi64(msg1, _mm256_castsi256_si128(msg2));

      // ---- Iteration 3 (rounds 12-15): msg3, finalize msg0, start msg2 schedule ----
      rounds4!(msg3, 3);
      msg0 = _mm256_add_epi64(msg0, extract_w_tm7(msg2, msg3));
      msg0 = _mm256_sha512msg2_epi64(msg0, msg3);
      msg2 = _mm256_sha512msg1_epi64(msg2, _mm256_castsi256_si128(msg3));

      // ---- Iterations 4-16: steady-state schedule cycle (period 4) ----
      macro_rules! steady_round {
        // $cur=msg used for rounds, $prev3=finalize target, $prev2/$prev1=W[t-7] sources,
        // $cur_for_msg2=sha512msg2 source, $prev1_for_msg1=sha512msg1 start target
        ($cur:ident, $ki:expr, $fin:ident, $tm7a:ident, $tm7b:ident, $msg1_tgt:ident) => {{
          rounds4!($cur, $ki);
          $fin = _mm256_add_epi64($fin, extract_w_tm7($tm7a, $tm7b));
          $fin = _mm256_sha512msg2_epi64($fin, $cur);
          $msg1_tgt = _mm256_sha512msg1_epi64($msg1_tgt, _mm256_castsi256_si128($cur));
        }};
      }

      // iter 4:  msg0 rounds, finalize msg1, start msg3
      steady_round!(msg0, 4, msg1, msg3, msg0, msg3);
      // iter 5:  msg1 rounds, finalize msg2, start msg0
      steady_round!(msg1, 5, msg2, msg0, msg1, msg0);
      // iter 6:  msg2 rounds, finalize msg3, start msg1
      steady_round!(msg2, 6, msg3, msg1, msg2, msg1);
      // iter 7:  msg3 rounds, finalize msg0, start msg2
      steady_round!(msg3, 7, msg0, msg2, msg3, msg2);

      // iter 8-11
      steady_round!(msg0, 8, msg1, msg3, msg0, msg3);
      steady_round!(msg1, 9, msg2, msg0, msg1, msg0);
      steady_round!(msg2, 10, msg3, msg1, msg2, msg1);
      steady_round!(msg3, 11, msg0, msg2, msg3, msg2);

      // iter 12-15
      steady_round!(msg0, 12, msg1, msg3, msg0, msg3);
      steady_round!(msg1, 13, msg2, msg0, msg1, msg0);
      steady_round!(msg2, 14, msg3, msg1, msg2, msg1);
      steady_round!(msg3, 15, msg0, msg2, msg3, msg2);

      // iter 16: msg0 rounds, finalize msg1, start msg3
      steady_round!(msg0, 16, msg1, msg3, msg0, msg3);

      // ---- Iteration 17 (rounds 68-71): msg1 rounds, finalize msg2 (no msg1 start) ----
      rounds4!(msg1, 17);
      msg2 = _mm256_add_epi64(msg2, extract_w_tm7(msg0, msg1));
      msg2 = _mm256_sha512msg2_epi64(msg2, msg1);

      // ---- Iteration 18 (rounds 72-75): msg2 rounds, finalize msg3 (no msg1 start) ----
      rounds4!(msg2, 18);
      msg3 = _mm256_add_epi64(msg3, extract_w_tm7(msg1, msg2));
      msg3 = _mm256_sha512msg2_epi64(msg3, msg2);

      // ---- Iteration 19 (rounds 76-79): msg3 rounds, no schedule ----
      rounds4!(msg3, 19);

      // Add saved state.
      abef = _mm256_add_epi64(abef, abef_save);
      cdgh = _mm256_add_epi64(cdgh, cdgh_save);

      ptr = ptr.add(128);
    }

    // Store state back: convert from ABEF/CDGH to [A,B,C,D,E,F,G,H].
    let abcd_out = _mm256_permute2x128_si256(abef, cdgh, 0x20); // [A,B,C,D]
    let efgh_out = _mm256_permute2x128_si256(abef, cdgh, 0x31); // [E,F,G,H]
    _mm256_storeu_si256(state.as_mut_ptr().cast(), abcd_out);
    _mm256_storeu_si256(state.as_mut_ptr().add(4).cast(), efgh_out);
  } // unsafe
}
