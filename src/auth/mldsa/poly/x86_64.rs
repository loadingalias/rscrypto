//! Eight-lane canonical ML-DSA arithmetic for Linux AVX2.
//!
//! This is the portable Montgomery arithmetic applied independently to lanes.
//! Roots and polynomial layout remain unchanged. All shuffles and addresses
//! depend on public transform stages, never coefficient values.

use core::arch::x86_64::*;

use super::{INV_N, N, NEG_Q_INVERSE, Poly, Q, R2, ROOTS, montgomery};

/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn reduce(x: __m256i) -> __m256i {
  // x < 2q < 2^31, so signed comparison implements the unsigned bound.
  let mask = _mm256_cmpgt_epi32(x, _mm256_set1_epi32(Q.strict_sub(1).cast_signed()));
  _mm256_sub_epi32(x, _mm256_and_si256(mask, _mm256_set1_epi32(Q.cast_signed())))
}

/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn multiply(a: __m256i, b: __m256i) -> __m256i {
  // Inputs are below 2q. As in super::montgomery, t + mq < 2^56.
  // Low-word multiplication intentionally wraps modulo 2^32.
  let q = _mm256_set1_epi32(Q.cast_signed());
  let m = _mm256_mullo_epi32(_mm256_mullo_epi32(a, b), _mm256_set1_epi32(NEG_Q_INVERSE.cast_signed()));
  let even = _mm256_add_epi64(_mm256_mul_epu32(a, b), _mm256_mul_epu32(m, q));
  let odd = _mm256_add_epi64(
    _mm256_mul_epu32(_mm256_srli_epi64::<32>(a), _mm256_srli_epi64::<32>(b)),
    _mm256_mul_epu32(_mm256_srli_epi64::<32>(m), q),
  );
  reduce(_mm256_blend_epi32::<0xaa>(_mm256_srli_epi64::<32>(even), odd))
}

/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn difference(a: __m256i, b: __m256i) -> __m256i {
  _mm256_sub_epi32(_mm256_add_epi32(a, _mm256_set1_epi32(Q.cast_signed())), b)
}

/// # Safety
/// The caller must establish AVX2. All coefficients must be below q.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn ntt(poly: &mut Poly) {
  const FIRST_FACTOR: u32 = montgomery(R2, ROOTS[1]);
  let (left, right) = poly.0.split_at_mut(N / 2);
  for (a, b) in left.as_chunks_mut::<8>().0.iter_mut().zip(right.as_chunks_mut::<8>().0) {
    // SAFETY: Disjoint initialized eight-u32 arrays. Unaligned loads/stores
    // require no extra alignment; both accesses cover exactly their array.
    unsafe {
      let x = multiply(
        _mm256_loadu_si256(a.as_ptr().cast()),
        _mm256_set1_epi32(R2.cast_signed()),
      );
      let y = multiply(
        _mm256_loadu_si256(b.as_ptr().cast()),
        _mm256_set1_epi32(FIRST_FACTOR.cast_signed()),
      );
      _mm256_storeu_si256(a.as_mut_ptr().cast(), reduce(_mm256_add_epi32(x, y)));
      _mm256_storeu_si256(b.as_mut_ptr().cast(), reduce(difference(x, y)));
    }
  }
  let mut root = 2usize;
  let mut width = N / 4;
  while width >= 8 {
    for block in poly.0.chunks_exact_mut(width.strict_mul(2)) {
      let zeta = _mm256_set1_epi32(ROOTS[root].cast_signed());
      root = root.strict_add(1);
      let (left, right) = block.split_at_mut(width);
      for (a, b) in left.as_chunks_mut::<8>().0.iter_mut().zip(right.as_chunks_mut::<8>().0) {
        // SAFETY: Each reference covers exactly eight initialized lanes, and
        // the mutable chunks are disjoint. AVX2 is established by the caller.
        unsafe {
          let x = _mm256_loadu_si256(a.as_ptr().cast());
          let y = multiply(_mm256_loadu_si256(b.as_ptr().cast()), zeta);
          _mm256_storeu_si256(a.as_mut_ptr().cast(), reduce(_mm256_add_epi32(x, y)));
          _mm256_storeu_si256(b.as_mut_ptr().cast(), reduce(difference(x, y)));
        }
      }
    }
    width >>= 1;
  }
  // Widths four, two, and one stay in one vector. Duplicate each butterfly's
  // inputs, then retain sums in the left lanes and differences in the right.
  for (index, block) in poly.0.as_chunks_mut::<8>().0.iter_mut().enumerate() {
    let r2 = 64usize.strict_add(index.strict_mul(2));
    let r1 = 128usize.strict_add(index.strict_mul(4));
    // SAFETY: One initialized eight-u32 array, uniquely borrowed, with an
    // unaligned load and store covering exactly its 32 bytes. No pointer escapes.
    unsafe {
      let v = _mm256_loadu_si256(block.as_ptr().cast());
      let x = _mm256_permute2x128_si256::<0x00>(v, v);
      let y = multiply(
        _mm256_permute2x128_si256::<0x11>(v, v),
        _mm256_set1_epi32(ROOTS[32usize.strict_add(index)].cast_signed()),
      );
      let v = _mm256_blend_epi32::<0xf0>(reduce(_mm256_add_epi32(x, y)), reduce(difference(x, y)));
      let z2 = _mm256_setr_epi32(
        ROOTS[r2].cast_signed(),
        ROOTS[r2].cast_signed(),
        ROOTS[r2].cast_signed(),
        ROOTS[r2].cast_signed(),
        ROOTS[r2.strict_add(1)].cast_signed(),
        ROOTS[r2.strict_add(1)].cast_signed(),
        ROOTS[r2.strict_add(1)].cast_signed(),
        ROOTS[r2.strict_add(1)].cast_signed(),
      );
      let x = _mm256_shuffle_epi32::<0x44>(v);
      let y = multiply(_mm256_shuffle_epi32::<0xee>(v), z2);
      let v = _mm256_blend_epi32::<0xcc>(reduce(_mm256_add_epi32(x, y)), reduce(difference(x, y)));
      let z1 = _mm256_setr_epi32(
        ROOTS[r1].cast_signed(),
        ROOTS[r1].cast_signed(),
        ROOTS[r1.strict_add(1)].cast_signed(),
        ROOTS[r1.strict_add(1)].cast_signed(),
        ROOTS[r1.strict_add(2)].cast_signed(),
        ROOTS[r1.strict_add(2)].cast_signed(),
        ROOTS[r1.strict_add(3)].cast_signed(),
        ROOTS[r1.strict_add(3)].cast_signed(),
      );
      let x = _mm256_shuffle_epi32::<0xa0>(v);
      let y = multiply(_mm256_shuffle_epi32::<0xf5>(v), z1);
      _mm256_storeu_si256(
        block.as_mut_ptr().cast(),
        _mm256_blend_epi32::<0xaa>(reduce(_mm256_add_epi32(x, y)), reduce(difference(x, y))),
      );
    }
  }
}

/// # Safety
/// The caller must establish AVX2. All coefficients must be below q.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn inverse_ntt(poly: &mut Poly) {
  for (index, block) in poly.0.as_chunks_mut::<8>().0.iter_mut().enumerate() {
    let r1 = 255usize.strict_sub(index.strict_mul(4));
    let r2 = 127usize.strict_sub(index.strict_mul(2));
    let z1 = _mm256_setr_epi32(
      Q.strict_sub(ROOTS[r1]).cast_signed(),
      Q.strict_sub(ROOTS[r1]).cast_signed(),
      Q.strict_sub(ROOTS[r1.strict_sub(1)]).cast_signed(),
      Q.strict_sub(ROOTS[r1.strict_sub(1)]).cast_signed(),
      Q.strict_sub(ROOTS[r1.strict_sub(2)]).cast_signed(),
      Q.strict_sub(ROOTS[r1.strict_sub(2)]).cast_signed(),
      Q.strict_sub(ROOTS[r1.strict_sub(3)]).cast_signed(),
      Q.strict_sub(ROOTS[r1.strict_sub(3)]).cast_signed(),
    );
    let z2 = _mm256_setr_epi32(
      Q.strict_sub(ROOTS[r2]).cast_signed(),
      Q.strict_sub(ROOTS[r2]).cast_signed(),
      Q.strict_sub(ROOTS[r2]).cast_signed(),
      Q.strict_sub(ROOTS[r2]).cast_signed(),
      Q.strict_sub(ROOTS[r2.strict_sub(1)]).cast_signed(),
      Q.strict_sub(ROOTS[r2.strict_sub(1)]).cast_signed(),
      Q.strict_sub(ROOTS[r2.strict_sub(1)]).cast_signed(),
      Q.strict_sub(ROOTS[r2.strict_sub(1)]).cast_signed(),
    );
    // SAFETY: The load/store covers exactly the uniquely borrowed eight-lane
    // block; alignment is unrestricted. Root indices are public and in 32..256.
    unsafe {
      let v = _mm256_loadu_si256(block.as_ptr().cast());
      let x = _mm256_shuffle_epi32::<0xa0>(v);
      let y = _mm256_shuffle_epi32::<0xf5>(v);
      let v = _mm256_blend_epi32::<0xaa>(reduce(_mm256_add_epi32(x, y)), multiply(difference(x, y), z1));
      let x = _mm256_shuffle_epi32::<0x44>(v);
      let y = _mm256_shuffle_epi32::<0xee>(v);
      let v = _mm256_blend_epi32::<0xcc>(reduce(_mm256_add_epi32(x, y)), multiply(difference(x, y), z2));
      let x = _mm256_permute2x128_si256::<0x00>(v, v);
      let y = _mm256_permute2x128_si256::<0x11>(v, v);
      let z4 = _mm256_set1_epi32(Q.strict_sub(ROOTS[63usize.strict_sub(index)]).cast_signed());
      _mm256_storeu_si256(
        block.as_mut_ptr().cast(),
        _mm256_blend_epi32::<0xf0>(reduce(_mm256_add_epi32(x, y)), multiply(difference(x, y), z4)),
      );
    }
  }
  let mut root = N / 8;
  let mut width = 8usize;
  while width < N / 2 {
    for block in poly.0.chunks_exact_mut(width.strict_mul(2)) {
      root = root.strict_sub(1);
      let zeta = _mm256_set1_epi32(Q.strict_sub(ROOTS[root]).cast_signed());
      let (left, right) = block.split_at_mut(width);
      for (a, b) in left.as_chunks_mut::<8>().0.iter_mut().zip(right.as_chunks_mut::<8>().0) {
        // SAFETY: Disjoint initialized eight-lane chunks; unaligned accesses
        // stay inside them. The caller establishes AVX2.
        unsafe {
          let x = _mm256_loadu_si256(a.as_ptr().cast());
          let y = _mm256_loadu_si256(b.as_ptr().cast());
          _mm256_storeu_si256(a.as_mut_ptr().cast(), reduce(_mm256_add_epi32(x, y)));
          _mm256_storeu_si256(b.as_mut_ptr().cast(), multiply(difference(x, y), zeta));
        }
      }
    }
    width = width.strict_mul(2);
  }
  const LAST_FACTOR: u32 = montgomery(Q.strict_sub(ROOTS[1]), INV_N);
  let (left, right) = poly.0.split_at_mut(N / 2);
  for (a, b) in left.as_chunks_mut::<8>().0.iter_mut().zip(right.as_chunks_mut::<8>().0) {
    // SAFETY: Fixed-size disjoint arrays cover each load/store, with no pointer
    // escape. AVX2 was established before entry.
    unsafe {
      let x = _mm256_loadu_si256(a.as_ptr().cast());
      let y = _mm256_loadu_si256(b.as_ptr().cast());
      _mm256_storeu_si256(
        a.as_mut_ptr().cast(),
        multiply(_mm256_add_epi32(x, y), _mm256_set1_epi32(INV_N.cast_signed())),
      );
      _mm256_storeu_si256(
        b.as_mut_ptr().cast(),
        multiply(difference(x, y), _mm256_set1_epi32(LAST_FACTOR.cast_signed())),
      );
    }
  }
}

/// # Safety
/// The caller must establish AVX2. All coefficients must be below q.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn accumulate_product(out: &mut [u32; N], a: &[u32; N], b: &[u32; N]) {
  for ((out, a), b) in out
    .as_chunks_mut::<8>()
    .0
    .iter_mut()
    .zip(a.as_chunks::<8>().0)
    .zip(b.as_chunks::<8>().0)
  {
    // SAFETY: Eight initialized u32 lanes per array. Output cannot alias either
    // input. Unaligned accesses cover exactly each array; N has no vector tail.
    unsafe {
      let product = multiply(
        _mm256_loadu_si256(a.as_ptr().cast()),
        _mm256_loadu_si256(b.as_ptr().cast()),
      );
      let sum = reduce(_mm256_add_epi32(_mm256_loadu_si256(out.as_ptr().cast()), product));
      _mm256_storeu_si256(out.as_mut_ptr().cast(), sum);
    }
  }
}
