//! x86_64 SHA-NI compression for SHA-1.
//!
//! The instruction schedule follows RustCrypto `sha1` 0.11.0's x86 backend
//! (MIT OR Apache-2.0), narrowed to one exact 64-byte block.

use core::arch::x86_64::*;

macro_rules! rounds4 {
  ($left:ident, $right:ident, $words:expr, $phase:expr) => {
    _mm_sha1rnds4_epu32($left, _mm_sha1nexte_epu32($right, $words), $phase)
  };
}

macro_rules! expand_schedule {
  ($first:expr, $second:expr, $third:expr, $fourth:expr) => {
    _mm_sha1msg2_epu32(_mm_xor_si128(_mm_sha1msg1_epu32($first, $second), $third), $fourth)
  };
}

macro_rules! schedule_rounds4 {
  ($left:ident, $right:ident, $first:expr, $second:expr, $third:expr, $fourth:expr, $next:expr, $phase:expr) => {
    $next = expand_schedule!($first, $second, $third, $fourth);
    $right = rounds4!($left, $right, $next, $phase);
  };
}

/// Compresses one SHA-1 block with x86 SHA-NI.
///
/// # Safety
///
/// The caller must establish `sha`, `ssse3`, and `sse4.1` before entry. SSE2
/// is part of the x86_64 baseline.
#[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
pub(super) unsafe fn compress(state: &mut [u32; 5], block: &[u8; 64]) {
  // SAFETY: The target-feature contract makes every intrinsic legal. The
  // unaligned state load/store touches exactly state[0..4]. Transmuting the
  // exact 64-byte block to four integer vectors preserves every bit, all vector
  // bit patterns are valid, and the by-value conversion requires no source
  // alignment. No pointer or reference escapes.
  unsafe {
    let byte_swap_mask = _mm_set_epi64x(0x0001_0203_0405_0607, 0x0809_0a0b_0c0d_0e0f);
    let mut state_abcd = _mm_loadu_si128(state.as_ptr().cast());
    state_abcd = _mm_shuffle_epi32(state_abcd, 0b0001_1011);
    let state_e_bits = i32::from_ne_bytes(state[4].to_ne_bytes());
    let mut state_e = _mm_set_epi32(state_e_bits, 0, 0, 0);
    let [mut words_zero, mut words_one, mut words_two, mut words_three] =
      core::mem::transmute::<[u8; 64], [__m128i; 4]>(*block);
    words_zero = _mm_shuffle_epi8(words_zero, byte_swap_mask);
    words_one = _mm_shuffle_epi8(words_one, byte_swap_mask);
    words_two = _mm_shuffle_epi8(words_two, byte_swap_mask);
    words_three = _mm_shuffle_epi8(words_three, byte_swap_mask);
    let mut words_four;

    let mut state_zero = state_abcd;
    let mut state_one = _mm_add_epi32(state_e, words_zero);

    state_one = _mm_sha1rnds4_epu32(state_zero, state_one, 0);
    state_zero = rounds4!(state_one, state_zero, words_one, 0);
    state_one = rounds4!(state_zero, state_one, words_two, 0);
    state_zero = rounds4!(state_one, state_zero, words_three, 0);
    schedule_rounds4!(
      state_zero,
      state_one,
      words_zero,
      words_one,
      words_two,
      words_three,
      words_four,
      0
    );

    schedule_rounds4!(
      state_one,
      state_zero,
      words_one,
      words_two,
      words_three,
      words_four,
      words_zero,
      1
    );
    schedule_rounds4!(
      state_zero,
      state_one,
      words_two,
      words_three,
      words_four,
      words_zero,
      words_one,
      1
    );
    schedule_rounds4!(
      state_one,
      state_zero,
      words_three,
      words_four,
      words_zero,
      words_one,
      words_two,
      1
    );
    schedule_rounds4!(
      state_zero,
      state_one,
      words_four,
      words_zero,
      words_one,
      words_two,
      words_three,
      1
    );
    schedule_rounds4!(
      state_one,
      state_zero,
      words_zero,
      words_one,
      words_two,
      words_three,
      words_four,
      1
    );

    schedule_rounds4!(
      state_zero,
      state_one,
      words_one,
      words_two,
      words_three,
      words_four,
      words_zero,
      2
    );
    schedule_rounds4!(
      state_one,
      state_zero,
      words_two,
      words_three,
      words_four,
      words_zero,
      words_one,
      2
    );
    schedule_rounds4!(
      state_zero,
      state_one,
      words_three,
      words_four,
      words_zero,
      words_one,
      words_two,
      2
    );
    schedule_rounds4!(
      state_one,
      state_zero,
      words_four,
      words_zero,
      words_one,
      words_two,
      words_three,
      2
    );
    schedule_rounds4!(
      state_zero,
      state_one,
      words_zero,
      words_one,
      words_two,
      words_three,
      words_four,
      2
    );

    schedule_rounds4!(
      state_one,
      state_zero,
      words_one,
      words_two,
      words_three,
      words_four,
      words_zero,
      3
    );
    schedule_rounds4!(
      state_zero,
      state_one,
      words_two,
      words_three,
      words_four,
      words_zero,
      words_one,
      3
    );
    schedule_rounds4!(
      state_one,
      state_zero,
      words_three,
      words_four,
      words_zero,
      words_one,
      words_two,
      3
    );
    schedule_rounds4!(
      state_zero,
      state_one,
      words_four,
      words_zero,
      words_one,
      words_two,
      words_three,
      3
    );
    schedule_rounds4!(
      state_one,
      state_zero,
      words_zero,
      words_one,
      words_two,
      words_three,
      words_four,
      3
    );

    state_abcd = _mm_add_epi32(state_abcd, state_zero);
    state_e = _mm_sha1nexte_epu32(state_one, state_e);
    state_abcd = _mm_shuffle_epi32(state_abcd, 0b0001_1011);
    _mm_storeu_si128(state.as_mut_ptr().cast(), state_abcd);
    state[4] = u32::from_ne_bytes(_mm_extract_epi32(state_e, 3).to_ne_bytes());
  }
}
