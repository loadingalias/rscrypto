//! Blake2b AVX2 and AVX-512VL compression for x86_64.
//!
//! Each row of the 4x4 `u64` working matrix lives in one `__m256i`. The twelve
//! rounds are unrolled, so every message index is a constant and each message
//! vector comes from four direct loads of the block.
//!
//! The diagonal step keeps row `b` in place and rotates rows `a`, `c`, and `d`
//! instead. Row `b` is the last row a G step writes and the first row the next
//! step reads, so a permute of `b` would add its latency to every half round.
//! Row `a` is final five dependent operations before `b`, which hides its
//! permute. Lane `i` of the diagonal step therefore computes diagonal
//! `G_{(i + 3) mod 4}`, and its message vectors are rotated to match.
//!
//! The AVX2 kernel rotates with byte shuffles and shifts; the AVX-512VL kernel
//! uses `VPRORQ`. The block loop keeps the chaining value in registers.
//!
//! # Safety
//!
//! The AVX2 entry points require AVX2. The AVX-512VL entry points require
//! AVX-512F and AVX-512VL, which imply AVX2.

use core::arch::x86_64::*;

use super::kernels::{Blake2bCounter, IV, SIGMA, split_counter};

/// Message construction and row rotations for one instruction set.
trait Rotate {
  /// Message words in the form `message` reads.
  type Message: Copy;

  /// Loads the message words of `block`.
  ///
  /// # Safety
  ///
  /// The caller must enable the instruction set of the implementation.
  unsafe fn load_message(block: &[u8; 128]) -> Self::Message;

  /// Builds the message vector whose lane `k` holds word `i_k` of `block`;
  /// `m` is `load_message(block)`.
  ///
  /// # Safety
  ///
  /// The caller must enable the instruction set of the implementation.
  unsafe fn message(block: &[u8; 128], m: &Self::Message, i0: u8, i1: u8, i2: u8, i3: u8) -> __m256i;

  /// Rotates each lane right by 32 bits.
  ///
  /// # Safety
  ///
  /// The caller must enable the instruction set of the implementation.
  unsafe fn ror32(x: __m256i) -> __m256i;
  /// Rotates each lane right by 24 bits.
  ///
  /// # Safety
  ///
  /// The caller must enable the instruction set of the implementation.
  unsafe fn ror24(x: __m256i) -> __m256i;
  /// Rotates each lane right by 16 bits.
  ///
  /// # Safety
  ///
  /// The caller must enable the instruction set of the implementation.
  unsafe fn ror16(x: __m256i) -> __m256i;
  /// Rotates each lane right by 63 bits.
  ///
  /// # Safety
  ///
  /// The caller must enable the instruction set of the implementation.
  unsafe fn ror63(x: __m256i) -> __m256i;
}

/// AVX2 rotations: dword and byte shuffles, and add/shift for 63.
struct Avx2;

impl Rotate for Avx2 {
  /// The words are read from the block where they are needed.
  type Message = ();

  #[inline(always)]
  unsafe fn load_message(_block: &[u8; 128]) {}

  #[inline(always)]
  unsafe fn message(block: &[u8; 128], (): &(), i0: u8, i1: u8, i2: u8, i3: u8) -> __m256i {
    // SAFETY: the caller enables AVX2.
    unsafe { _mm256_setr_epi64x(word(block, i0), word(block, i1), word(block, i2), word(block, i3)) }
  }

  #[inline(always)]
  unsafe fn ror32(x: __m256i) -> __m256i {
    // SAFETY: the caller enables AVX2.
    unsafe { _mm256_shuffle_epi32(x, 0xB1) }
  }

  #[inline(always)]
  unsafe fn ror24(x: __m256i) -> __m256i {
    // SAFETY: the caller enables AVX2.
    unsafe {
      let mask = _mm256_setr_epi8(
        3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10, 3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10,
      );
      _mm256_shuffle_epi8(x, mask)
    }
  }

  #[inline(always)]
  unsafe fn ror16(x: __m256i) -> __m256i {
    // SAFETY: the caller enables AVX2.
    unsafe {
      let mask = _mm256_setr_epi8(
        2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9, 2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9,
      );
      _mm256_shuffle_epi8(x, mask)
    }
  }

  #[inline(always)]
  unsafe fn ror63(x: __m256i) -> __m256i {
    // SAFETY: the caller enables AVX2. Rotating right by 63 is rotating left
    // by 1: (x + x) ^ (x >> 63).
    unsafe { _mm256_xor_si256(_mm256_srli_epi64(x, 63), _mm256_add_epi64(x, x)) }
  }
}

/// AVX-512VL rotations: `VPRORQ` for every distance.
struct Avx512vl;

impl Rotate for Avx512vl {
  /// Words 0–3, 4–7, 8–11, and 12–15.
  type Message = [__m256i; 4];

  #[inline(always)]
  unsafe fn load_message(block: &[u8; 128]) -> [__m256i; 4] {
    let (rows, _) = block.as_chunks::<32>();
    // SAFETY: the caller enables AVX-512F and AVX-512VL; each row is 32
    // readable bytes.
    unsafe { core::array::from_fn(|row| _mm256_loadu_si256(rows[row].as_ptr().cast())) }
  }

  #[inline(always)]
  unsafe fn message(_block: &[u8; 128], m: &[__m256i; 4], i0: u8, i1: u8, i2: u8, i3: u8) -> __m256i {
    // Each two-source permute selects from eight words: rows 0 and 1 hold
    // words 0–7 and rows 2 and 3 hold words 8–15. The blend takes lane `k`
    // from the second permute when word `i_k` is 8 or more.
    let high = |i: u8, bit: u8| if i >= 8 { bit } else { 0 };
    let mask = high(i0, 1) | high(i1, 2) | high(i2, 4) | high(i3, 8);
    // SAFETY: the caller enables AVX-512F and AVX-512VL.
    unsafe {
      let index = _mm256_setr_epi64x(
        i64::from(i0 & 7),
        i64::from(i1 & 7),
        i64::from(i2 & 7),
        i64::from(i3 & 7),
      );
      let low_words = _mm256_permutex2var_epi64(m[0], index, m[1]);
      let high_words = _mm256_permutex2var_epi64(m[2], index, m[3]);
      _mm256_mask_blend_epi64(mask, low_words, high_words)
    }
  }

  #[inline(always)]
  unsafe fn ror32(x: __m256i) -> __m256i {
    // SAFETY: the caller enables AVX-512F and AVX-512VL.
    unsafe { _mm256_ror_epi64(x, 32) }
  }

  #[inline(always)]
  unsafe fn ror24(x: __m256i) -> __m256i {
    // SAFETY: the caller enables AVX-512F and AVX-512VL.
    unsafe { _mm256_ror_epi64(x, 24) }
  }

  #[inline(always)]
  unsafe fn ror16(x: __m256i) -> __m256i {
    // SAFETY: the caller enables AVX-512F and AVX-512VL.
    unsafe { _mm256_ror_epi64(x, 16) }
  }

  #[inline(always)]
  unsafe fn ror63(x: __m256i) -> __m256i {
    // SAFETY: the caller enables AVX-512F and AVX-512VL.
    unsafe { _mm256_ror_epi64(x, 63) }
  }
}

/// Returns message word `index` of `block`.
#[inline(always)]
fn word(block: &[u8; 128], index: u8) -> i64 {
  let (words, _) = block.as_chunks::<8>();
  i64::from_le_bytes(words[usize::from(index)])
}

/// Returns `x` unchanged, hidden from LLVM's reassociation.
///
/// Without it LLVM rewrites `(a + m) + b` as `(a + b) + m`, which puts the
/// message add on the dependency chain of every half round.
///
/// # Safety
///
/// The caller must enable AVX2.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn opaque(mut x: __m256i) -> __m256i {
  // SAFETY: the template emits no instruction. Its one operand is the vector
  // register holding `x` on entry and exit; it touches no memory, stack, or
  // flags.
  unsafe { core::arch::asm!("/* {0} */", inout(ymm_reg) x, options(pure, nomem, nostack, preserves_flags)) };
  x
}

/// Applies G to all four lanes.
///
/// # Safety
///
/// The caller must enable AVX2 and the instruction set of `R`.
#[inline(always)]
unsafe fn g<R: Rotate>(a: &mut __m256i, b: &mut __m256i, c: &mut __m256i, d: &mut __m256i, mx: __m256i, my: __m256i) {
  // SAFETY: the caller enables AVX2 and the instruction set of `R`.
  unsafe {
    // The message word is added first: `a` is ready before `b`.
    *a = _mm256_add_epi64(opaque(_mm256_add_epi64(*a, mx)), *b);
    *d = R::ror32(_mm256_xor_si256(*d, *a));
    *c = _mm256_add_epi64(*c, *d);
    *b = R::ror24(_mm256_xor_si256(*b, *c));
    *a = _mm256_add_epi64(opaque(_mm256_add_epi64(*a, my)), *b);
    *d = R::ror16(_mm256_xor_si256(*d, *a));
    *c = _mm256_add_epi64(*c, *d);
    *b = R::ror63(_mm256_xor_si256(*b, *c));
  }
}

/// Applies one round: the column step, then the diagonal step.
///
/// # Safety
///
/// The caller must enable AVX2 and the instruction set of `R`.
#[inline(always)]
unsafe fn round<R: Rotate>(
  a: &mut __m256i,
  b: &mut __m256i,
  c: &mut __m256i,
  d: &mut __m256i,
  block: &[u8; 128],
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
    *a = _mm256_permute4x64_epi64(*a, 0x93);
    *c = _mm256_permute4x64_epi64(*c, 0x39);
    *d = _mm256_permute4x64_epi64(*d, 0x4E);
    let mx = R::message(block, m, s[14], s[8], s[10], s[12]);
    let my = R::message(block, m, s[15], s[9], s[11], s[13]);
    g::<R>(a, b, c, d, mx, my);
    *a = _mm256_permute4x64_epi64(*a, 0x39);
    *c = _mm256_permute4x64_epi64(*c, 0x93);
    *d = _mm256_permute4x64_epi64(*d, 0x4E);
  }
}

/// Compresses one block into the chaining value rows `(h0, h1)`.
///
/// # Safety
///
/// The caller must enable AVX2 and the instruction set of `R`.
#[inline(always)]
unsafe fn compress_rows<R: Rotate>(
  h0: __m256i,
  h1: __m256i,
  block: &[u8; 128],
  t0: u64,
  t1: u64,
  f0: u64,
) -> (__m256i, __m256i) {
  // SAFETY: the caller enables AVX2 and the instruction set of `R`.
  unsafe {
    let m = R::load_message(block);
    let mut a = h0;
    let mut b = h1;
    let mut c = _mm256_setr_epi64x(
      IV[0].cast_signed(),
      IV[1].cast_signed(),
      IV[2].cast_signed(),
      IV[3].cast_signed(),
    );
    let mut d = _mm256_setr_epi64x(
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
    round::<R>(&mut a, &mut b, &mut c, &mut d, block, &m, &SIGMA[0]);
    round::<R>(&mut a, &mut b, &mut c, &mut d, block, &m, &SIGMA[1]);

    (
      _mm256_xor_si256(h0, _mm256_xor_si256(a, c)),
      _mm256_xor_si256(h1, _mm256_xor_si256(b, d)),
    )
  }
}

/// Compresses one block into `h`.
///
/// # Safety
///
/// The caller must enable AVX2 and the instruction set of `R`.
#[inline(always)]
unsafe fn compress_impl<R: Rotate>(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
  let (t0, t1) = split_counter(t);
  let f0 = u64::from(last).wrapping_neg();
  // SAFETY: the caller enables AVX2 and the instruction set of `R`. `h` is
  // eight `u64` words, so both 32-byte unaligned accesses are in bounds.
  unsafe {
    let h0 = _mm256_loadu_si256(h.as_ptr().cast());
    let h1 = _mm256_loadu_si256(h.as_ptr().add(4).cast());
    let (h0, h1) = compress_rows::<R>(h0, h1, block, t0, t1, f0);
    _mm256_storeu_si256(h.as_mut_ptr().cast(), h0);
    _mm256_storeu_si256(h.as_mut_ptr().add(4).cast(), h1);
  }
}

/// Compresses whole non-final blocks into `h`, advancing `t` by 128 per block.
///
/// # Safety
///
/// The caller must enable AVX2 and the instruction set of `R`.
#[inline(always)]
unsafe fn compress_blocks_impl<R: Rotate>(h: &mut [u64; 8], blocks: &[u8], t: &mut Blake2bCounter) {
  let (chunks, remainder) = blocks.as_chunks::<128>();
  debug_assert!(remainder.is_empty());
  // SAFETY: the caller enables AVX2 and the instruction set of `R`. `h` is
  // eight `u64` words, so both 32-byte unaligned accesses are in bounds.
  unsafe {
    let mut h0 = _mm256_loadu_si256(h.as_ptr().cast());
    let mut h1 = _mm256_loadu_si256(h.as_ptr().add(4).cast());
    for block in chunks {
      t.add_len(128);
      let (t0, t1) = t.words();
      (h0, h1) = compress_rows::<R>(h0, h1, block, t0, t1, 0);
    }
    _mm256_storeu_si256(h.as_mut_ptr().cast(), h0);
    _mm256_storeu_si256(h.as_mut_ptr().add(4).cast(), h1);
  }
}

/// Blake2b AVX2 compress of one block.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn compress_avx2(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
  // SAFETY: this function enables AVX2.
  unsafe { compress_impl::<Avx2>(h, block, t, last) }
}

/// Blake2b AVX2 compress of whole non-final blocks.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn compress_blocks_avx2(h: &mut [u64; 8], blocks: &[u8], t: &mut Blake2bCounter) {
  // SAFETY: this function enables AVX2.
  unsafe { compress_blocks_impl::<Avx2>(h, blocks, t) }
}

/// Blake2b AVX-512VL compress of one block.
///
/// # Safety
///
/// Caller must ensure AVX-512F and AVX-512VL are available.
#[target_feature(enable = "avx512f,avx512vl")]
pub(super) unsafe fn compress_avx512vl(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
  // SAFETY: this function enables AVX-512F and AVX-512VL, which imply AVX2.
  unsafe { compress_impl::<Avx512vl>(h, block, t, last) }
}

/// Blake2b AVX-512VL compress of whole non-final blocks.
///
/// # Safety
///
/// Caller must ensure AVX-512F and AVX-512VL are available.
#[target_feature(enable = "avx512f,avx512vl")]
pub(super) unsafe fn compress_blocks_avx512vl(h: &mut [u64; 8], blocks: &[u8], t: &mut Blake2bCounter) {
  // SAFETY: this function enables AVX-512F and AVX-512VL, which imply AVX2.
  unsafe { compress_blocks_impl::<Avx512vl>(h, blocks, t) }
}
