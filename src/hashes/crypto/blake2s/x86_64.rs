//! Blake2s AVX2 and AVX-512VL compression for x86_64.
//!
//! Each row of the 4x4 `u32` working matrix lives in one `__m128i`. The ten
//! rounds are unrolled, so every message index is a constant and each message
//! vector comes from four direct loads of the block.
//!
//! The diagonal step keeps row `b` in place and rotates rows `a`, `c`, and `d`
//! instead. Row `b` is the last row a G step writes and the first row the next
//! step reads, so a shuffle of `b` would add its latency to every half round.
//! Row `a` is final five dependent operations before `b`, which hides its
//! shuffle. Lane `i` of the diagonal step therefore computes diagonal
//! `G_{(i + 3) mod 4}`, and its message vectors are rotated to match.
//!
//! The AVX2 kernel rotates with shifts and byte shuffles; the AVX-512VL kernel
//! uses `VPRORD`. The block loop keeps the chaining value in registers.
//!
//! # Safety
//!
//! The AVX2 entry points require AVX2. The AVX-512VL entry points require
//! AVX-512F and AVX-512VL, which imply AVX2.

use core::arch::x86_64::*;

use super::kernels::{IV, SIGMA, split_counter};

/// Message construction and row rotations for one instruction set.
trait Rotate {
  /// Message words in the form `message` reads.
  type Message: Copy;

  /// Loads the message words of `block`.
  ///
  /// # Safety
  ///
  /// The caller must enable the instruction set of the implementation.
  unsafe fn load_message(block: &[u8; 64]) -> Self::Message;

  /// Builds the message vector whose lane `k` holds word `i_k` of `block`;
  /// `m` is `load_message(block)`.
  ///
  /// # Safety
  ///
  /// The caller must enable the instruction set of the implementation.
  unsafe fn message(block: &[u8; 64], m: &Self::Message, i0: u8, i1: u8, i2: u8, i3: u8) -> __m128i;

  /// Rotates each lane right by 16 bits.
  ///
  /// # Safety
  ///
  /// The caller must enable the instruction set of the implementation.
  unsafe fn ror16(x: __m128i) -> __m128i;
  /// Rotates each lane right by 12 bits.
  ///
  /// # Safety
  ///
  /// The caller must enable the instruction set of the implementation.
  unsafe fn ror12(x: __m128i) -> __m128i;
  /// Rotates each lane right by 8 bits.
  ///
  /// # Safety
  ///
  /// The caller must enable the instruction set of the implementation.
  unsafe fn ror8(x: __m128i) -> __m128i;
  /// Rotates each lane right by 7 bits.
  ///
  /// # Safety
  ///
  /// The caller must enable the instruction set of the implementation.
  unsafe fn ror7(x: __m128i) -> __m128i;
}

/// AVX2 rotations: byte shuffles for 16 and 8, shift/or otherwise.
struct Avx2;

impl Rotate for Avx2 {
  /// The words are read from the block where they are needed.
  type Message = ();

  #[inline(always)]
  unsafe fn load_message(_block: &[u8; 64]) {}

  #[inline(always)]
  unsafe fn message(block: &[u8; 64], (): &(), i0: u8, i1: u8, i2: u8, i3: u8) -> __m128i {
    // SAFETY: the caller enables AVX2.
    unsafe { _mm_setr_epi32(word(block, i0), word(block, i1), word(block, i2), word(block, i3)) }
  }

  #[inline(always)]
  unsafe fn ror16(x: __m128i) -> __m128i {
    // SAFETY: the caller enables AVX2.
    unsafe {
      let mask = _mm_setr_epi8(2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13);
      _mm_shuffle_epi8(x, mask)
    }
  }

  #[inline(always)]
  unsafe fn ror12(x: __m128i) -> __m128i {
    // SAFETY: the caller enables AVX2.
    unsafe { _mm_or_si128(_mm_srli_epi32(x, 12), _mm_slli_epi32(x, 20)) }
  }

  #[inline(always)]
  unsafe fn ror8(x: __m128i) -> __m128i {
    // SAFETY: the caller enables AVX2.
    unsafe {
      let mask = _mm_setr_epi8(1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12);
      _mm_shuffle_epi8(x, mask)
    }
  }

  #[inline(always)]
  unsafe fn ror7(x: __m128i) -> __m128i {
    // SAFETY: the caller enables AVX2.
    unsafe { _mm_or_si128(_mm_srli_epi32(x, 7), _mm_slli_epi32(x, 25)) }
  }
}

/// AVX-512VL rotations: `VPRORD` for every distance.
struct Avx512vl;

impl Rotate for Avx512vl {
  /// Words 0–3, 4–7, 8–11, and 12–15.
  type Message = [__m128i; 4];

  #[inline(always)]
  unsafe fn load_message(block: &[u8; 64]) -> [__m128i; 4] {
    let (rows, _) = block.as_chunks::<16>();
    // SAFETY: the caller enables AVX-512F and AVX-512VL; each row is 16
    // readable bytes.
    unsafe { core::array::from_fn(|row| _mm_loadu_si128(rows[row].as_ptr().cast())) }
  }

  #[inline(always)]
  unsafe fn message(_block: &[u8; 64], m: &[__m128i; 4], i0: u8, i1: u8, i2: u8, i3: u8) -> __m128i {
    // Each two-source permute selects from eight words: rows 0 and 1 hold
    // words 0–7 and rows 2 and 3 hold words 8–15. The blend takes lane `k`
    // from the second permute when word `i_k` is 8 or more.
    let high = |i: u8, bit: u8| if i >= 8 { bit } else { 0 };
    let mask = high(i0, 1) | high(i1, 2) | high(i2, 4) | high(i3, 8);
    // SAFETY: the caller enables AVX-512F and AVX-512VL.
    unsafe {
      let index = _mm_setr_epi32(
        i32::from(i0 & 7),
        i32::from(i1 & 7),
        i32::from(i2 & 7),
        i32::from(i3 & 7),
      );
      let low_words = _mm_permutex2var_epi32(m[0], index, m[1]);
      let high_words = _mm_permutex2var_epi32(m[2], index, m[3]);
      _mm_mask_blend_epi32(mask, low_words, high_words)
    }
  }

  #[inline(always)]
  unsafe fn ror16(x: __m128i) -> __m128i {
    // SAFETY: the caller enables AVX-512F and AVX-512VL.
    unsafe { _mm_ror_epi32(x, 16) }
  }

  #[inline(always)]
  unsafe fn ror12(x: __m128i) -> __m128i {
    // SAFETY: the caller enables AVX-512F and AVX-512VL.
    unsafe { _mm_ror_epi32(x, 12) }
  }

  #[inline(always)]
  unsafe fn ror8(x: __m128i) -> __m128i {
    // SAFETY: the caller enables AVX-512F and AVX-512VL.
    unsafe { _mm_ror_epi32(x, 8) }
  }

  #[inline(always)]
  unsafe fn ror7(x: __m128i) -> __m128i {
    // SAFETY: the caller enables AVX-512F and AVX-512VL.
    unsafe { _mm_ror_epi32(x, 7) }
  }
}

/// Returns message word `index` of `block`.
#[inline(always)]
fn word(block: &[u8; 64], index: u8) -> i32 {
  let (words, _) = block.as_chunks::<4>();
  i32::from_le_bytes(words[usize::from(index)])
}

/// Returns `x` unchanged, hidden from LLVM's reassociation.
///
/// Without it LLVM rewrites `(a + m) + b` as `(a + b) + m`, which puts the
/// message add on the dependency chain of every half round.
#[inline(always)]
fn opaque(mut x: __m128i) -> __m128i {
  // SAFETY: the template emits no instruction. Its one operand is the vector
  // register holding `x` on entry and exit; it touches no memory, stack, or
  // flags. SSE2, which provides the register class, is baseline on x86_64.
  unsafe { core::arch::asm!("/* {0} */", inout(xmm_reg) x, options(pure, nomem, nostack, preserves_flags)) };
  x
}

/// Applies G to all four lanes.
///
/// # Safety
///
/// The caller must enable AVX2 and the instruction set of `R`.
#[inline(always)]
unsafe fn g<R: Rotate>(a: &mut __m128i, b: &mut __m128i, c: &mut __m128i, d: &mut __m128i, mx: __m128i, my: __m128i) {
  // SAFETY: the caller enables AVX2 and the instruction set of `R`.
  unsafe {
    // The message word is added first: `a` is ready before `b`.
    *a = _mm_add_epi32(opaque(_mm_add_epi32(*a, mx)), *b);
    *d = R::ror16(_mm_xor_si128(*d, *a));
    *c = _mm_add_epi32(*c, *d);
    *b = R::ror12(_mm_xor_si128(*b, *c));
    *a = _mm_add_epi32(opaque(_mm_add_epi32(*a, my)), *b);
    *d = R::ror8(_mm_xor_si128(*d, *a));
    *c = _mm_add_epi32(*c, *d);
    *b = R::ror7(_mm_xor_si128(*b, *c));
  }
}

/// Applies one round: the column step, then the diagonal step.
///
/// # Safety
///
/// The caller must enable AVX2 and the instruction set of `R`.
#[inline(always)]
unsafe fn round<R: Rotate>(
  a: &mut __m128i,
  b: &mut __m128i,
  c: &mut __m128i,
  d: &mut __m128i,
  block: &[u8; 64],
  m: &R::Message,
  s: &[u8; 16],
) {
  // SAFETY: the caller enables AVX2 and the instruction set of `R`.
  unsafe {
    // Column step: lane i computes G_i.
    let mx = R::message(block, m, s[0], s[2], s[4], s[6]);
    let my = R::message(block, m, s[1], s[3], s[5], s[7]);
    g::<R>(a, b, c, d, mx, my);

    // Diagonal step: lane i keeps b[i] and computes G_{(i+3) mod 4}, so it
    // needs a[i-1], c[i+1], and d[i+2].
    *a = _mm_shuffle_epi32(*a, 0x93);
    *c = _mm_shuffle_epi32(*c, 0x39);
    *d = _mm_shuffle_epi32(*d, 0x4E);
    let mx = R::message(block, m, s[14], s[8], s[10], s[12]);
    let my = R::message(block, m, s[15], s[9], s[11], s[13]);
    g::<R>(a, b, c, d, mx, my);
    *a = _mm_shuffle_epi32(*a, 0x39);
    *c = _mm_shuffle_epi32(*c, 0x93);
    *d = _mm_shuffle_epi32(*d, 0x4E);
  }
}

/// Compresses one block into the chaining value rows `(h0, h1)`.
///
/// # Safety
///
/// The caller must enable AVX2 and the instruction set of `R`.
#[inline(always)]
unsafe fn compress_rows<R: Rotate>(
  h0: __m128i,
  h1: __m128i,
  block: &[u8; 64],
  t0: u32,
  t1: u32,
  f0: u32,
) -> (__m128i, __m128i) {
  // SAFETY: the caller enables AVX2 and the instruction set of `R`.
  unsafe {
    let m = R::load_message(block);
    let mut a = h0;
    let mut b = h1;
    let mut c = _mm_setr_epi32(
      IV[0].cast_signed(),
      IV[1].cast_signed(),
      IV[2].cast_signed(),
      IV[3].cast_signed(),
    );
    let mut d = _mm_setr_epi32(
      (IV[4] ^ t0).cast_signed(),
      (IV[5] ^ t1).cast_signed(),
      (IV[6] ^ f0).cast_signed(),
      IV[7].cast_signed(),
    );

    round::<R>(&mut a, &mut b, &mut c, &mut d, block, &m, &SIGMA[0]);
    round::<R>(&mut a, &mut b, &mut c, &mut d, block, &m, &SIGMA[1]);
    round::<R>(&mut a, &mut b, &mut c, &mut d, block, &m, &SIGMA[2]);
    round::<R>(&mut a, &mut b, &mut c, &mut d, block, &m, &SIGMA[3]);
    round::<R>(&mut a, &mut b, &mut c, &mut d, block, &m, &SIGMA[4]);
    round::<R>(&mut a, &mut b, &mut c, &mut d, block, &m, &SIGMA[5]);
    round::<R>(&mut a, &mut b, &mut c, &mut d, block, &m, &SIGMA[6]);
    round::<R>(&mut a, &mut b, &mut c, &mut d, block, &m, &SIGMA[7]);
    round::<R>(&mut a, &mut b, &mut c, &mut d, block, &m, &SIGMA[8]);
    round::<R>(&mut a, &mut b, &mut c, &mut d, block, &m, &SIGMA[9]);

    (
      _mm_xor_si128(h0, _mm_xor_si128(a, c)),
      _mm_xor_si128(h1, _mm_xor_si128(b, d)),
    )
  }
}

/// Compresses one block into `h`.
///
/// # Safety
///
/// The caller must enable AVX2 and the instruction set of `R`.
#[inline(always)]
unsafe fn compress_impl<R: Rotate>(h: &mut [u32; 8], block: &[u8; 64], t: u64, last: bool) {
  let (t0, t1) = split_counter(t);
  let f0 = u32::from(last).wrapping_neg();
  // SAFETY: the caller enables AVX2 and the instruction set of `R`. `h` is
  // eight `u32` words, so both 16-byte unaligned accesses are in bounds.
  unsafe {
    let h0 = _mm_loadu_si128(h.as_ptr().cast());
    let h1 = _mm_loadu_si128(h.as_ptr().add(4).cast());
    let (h0, h1) = compress_rows::<R>(h0, h1, block, t0, t1, f0);
    _mm_storeu_si128(h.as_mut_ptr().cast(), h0);
    _mm_storeu_si128(h.as_mut_ptr().add(4).cast(), h1);
  }
}

/// Compresses whole non-final blocks into `h`, advancing `t` by 64 per block.
///
/// # Safety
///
/// The caller must enable AVX2 and the instruction set of `R`.
#[inline(always)]
unsafe fn compress_blocks_impl<R: Rotate>(h: &mut [u32; 8], blocks: &[u8], t: &mut u64) {
  let (chunks, remainder) = blocks.as_chunks::<64>();
  debug_assert!(remainder.is_empty());
  // SAFETY: the caller enables AVX2 and the instruction set of `R`. `h` is
  // eight `u32` words, so both 16-byte unaligned accesses are in bounds.
  unsafe {
    let mut h0 = _mm_loadu_si128(h.as_ptr().cast());
    let mut h1 = _mm_loadu_si128(h.as_ptr().add(4).cast());
    for block in chunks {
      *t = t.strict_add(64);
      let (t0, t1) = split_counter(*t);
      (h0, h1) = compress_rows::<R>(h0, h1, block, t0, t1, 0);
    }
    _mm_storeu_si128(h.as_mut_ptr().cast(), h0);
    _mm_storeu_si128(h.as_mut_ptr().add(4).cast(), h1);
  }
}

/// Blake2s AVX2 compress of one block.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn compress_avx2(h: &mut [u32; 8], block: &[u8; 64], t: u64, last: bool) {
  // SAFETY: this function enables AVX2.
  unsafe { compress_impl::<Avx2>(h, block, t, last) }
}

/// Blake2s AVX2 compress of whole non-final blocks.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn compress_blocks_avx2(h: &mut [u32; 8], blocks: &[u8], t: &mut u64) {
  // SAFETY: this function enables AVX2.
  unsafe { compress_blocks_impl::<Avx2>(h, blocks, t) }
}

/// Blake2s AVX-512VL compress of one block.
///
/// # Safety
///
/// Caller must ensure AVX-512F and AVX-512VL are available.
#[target_feature(enable = "avx512f,avx512vl")]
pub(super) unsafe fn compress_avx512vl(h: &mut [u32; 8], block: &[u8; 64], t: u64, last: bool) {
  // SAFETY: this function enables AVX-512F and AVX-512VL, which imply AVX2.
  unsafe { compress_impl::<Avx512vl>(h, block, t, last) }
}

/// Blake2s AVX-512VL compress of whole non-final blocks.
///
/// # Safety
///
/// Caller must ensure AVX-512F and AVX-512VL are available.
#[target_feature(enable = "avx512f,avx512vl")]
pub(super) unsafe fn compress_blocks_avx512vl(h: &mut [u32; 8], blocks: &[u8], t: &mut u64) {
  // SAFETY: this function enables AVX-512F and AVX-512VL, which imply AVX2.
  unsafe { compress_blocks_impl::<Avx512vl>(h, blocks, t) }
}
