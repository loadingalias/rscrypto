//! Sixteen-lane FIPS 204 transforms using the AVX2 backend's signed Montgomery arithmetic.
//! The four narrow stages stay within one vector; every permutation and
//! multiplier index depends only on the public stage and coefficient position.

use core::arch::x86_64::*;

use super::{FIRST_TWIDDLE, FORWARD, INV_N_TWIDDLE, INVERSE, LAST_TWIDDLE, N, Poly, Q, QINV, R2_TWIDDLE, Twiddle};

const LANES: usize = 16;
const POSITIONS: [i32; LANES] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
type LaneTable = ([[i32; LANES]; 16], [[i32; LANES]; 16]);

const FORWARD_8: LaneTable = lane_table(false, 8);
const FORWARD_4: LaneTable = lane_table(false, 4);
const FORWARD_2: LaneTable = lane_table(false, 2);
const FORWARD_1: LaneTable = lane_table(false, 1);
const INVERSE_8: LaneTable = lane_table(true, 8);
const INVERSE_4: LaneTable = lane_table(true, 4);
const INVERSE_2: LaneTable = lane_table(true, 2);
const INVERSE_1: LaneTable = lane_table(true, 1);

const fn lane_table(inverse: bool, width: usize) -> LaneTable {
  let table = if inverse { INVERSE } else { FORWARD };
  let mut values = [[0; LANES]; 16];
  let mut twisted = [[0; LANES]; 16];
  let mut vector = 0usize;
  while vector < 16 {
    let mut lane = 0usize;
    while lane < LANES {
      let pair_width = width.strict_mul(2);
      let group = vector
        .strict_mul(LANES.strict_div(pair_width))
        .strict_add(lane.strict_div(pair_width));
      let root = if inverse {
        256usize.strict_div(width).strict_sub(1).strict_sub(group)
      } else {
        128usize.strict_div(width).strict_add(group)
      };
      values[vector][lane] = table[root].value;
      twisted[vector][lane] = table[root].twisted;
      lane = lane.strict_add(1);
    }
    vector = vector.strict_add(1);
  }
  (values, twisted)
}

const fn indices(width: usize, high: bool) -> [i32; LANES] {
  let mut out = [0; LANES];
  let mut i = 0;
  while i < LANES {
    let index = if high { i | width } else { i & !width };
    out[i] = POSITIONS[index];
    i = i.strict_add(1);
  }
  out
}

const fn high_mask(width: usize) -> u16 {
  let mut mask = 0u16;
  let mut i = 0usize;
  while i < LANES {
    if i & width != 0 {
      mask |= 1u16 << i;
    }
    i = i.strict_add(1);
  }
  mask
}

/// # Safety
/// Calling outside an AVX-512F function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn load(chunk: &[u32; LANES]) -> __m512i {
  // SAFETY: The reference provides exactly sixteen initialized u32 lanes; unaligned
  // access imposes no alignment requirement beyond that of the array.
  unsafe { _mm512_loadu_si512(chunk.as_ptr().cast()) }
}

/// # Safety
/// Calling outside an AVX-512F function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn store(chunk: &mut [u32; LANES], value: __m512i) {
  // SAFETY: The unique array reference covers exactly sixteen writable lanes. Every
  // lane is written, so there is no masked tail or partial initialization.
  unsafe { _mm512_storeu_si512(chunk.as_mut_ptr().cast(), value) };
}

/// Signed Montgomery multiplication with the AVX2 kernel's centered multiplier.
/// Inputs bounded by `16q` and multipliers in `(-q/2, q/2]` give exact signed
/// 64-bit products; the divisible difference divided by 2^32 lies in `(-q, q)`.
///
/// # Safety
/// Calling outside an AVX-512F function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn multiply(a: __m512i, values: __m512i, twisted: __m512i) -> __m512i {
  let q = _mm512_set1_epi32(Q.cast_signed());
  let even = _mm512_sub_epi64(
    _mm512_mul_epi32(a, values),
    _mm512_mul_epi32(_mm512_mul_epi32(a, twisted), q),
  );
  let odd_a = _mm512_srli_epi64::<32>(a);
  let odd = _mm512_sub_epi64(
    _mm512_mul_epi32(odd_a, _mm512_srli_epi64::<32>(values)),
    _mm512_mul_epi32(_mm512_mul_epi32(odd_a, _mm512_srli_epi64::<32>(twisted)), q),
  );
  _mm512_mask_blend_epi32(0xaaaa, _mm512_srli_epi64::<32>(even), odd)
}

/// # Safety
/// Calling outside an AVX-512F function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn broadcast(a: __m512i, multiplier: Twiddle) -> __m512i {
  multiply(
    a,
    _mm512_set1_epi32(multiplier.value),
    _mm512_set1_epi32(multiplier.twisted),
  )
}

/// The same bounded signed reduction as the AVX2 backend.
///
/// # Safety
/// Calling outside an AVX-512F function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn reduce32(x: __m512i) -> __m512i {
  let t = _mm512_srai_epi32::<23>(_mm512_add_epi32(x, _mm512_set1_epi32(1 << 22)));
  let tq = _mm512_add_epi32(
    _mm512_sub_epi32(_mm512_slli_epi32::<23>(t), _mm512_slli_epi32::<13>(t)),
    t,
  );
  _mm512_sub_epi32(x, tq)
}

/// Canonicalize lanes whose magnitude is below `9q`.
///
/// # Safety
/// Calling outside an AVX-512F function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn canonical(x: __m512i) -> __m512i {
  let x = reduce32(x);
  _mm512_add_epi32(
    x,
    _mm512_and_si512(_mm512_srai_epi32::<31>(x), _mm512_set1_epi32(Q.cast_signed())),
  )
}

/// One fixed-width in-vector butterfly stage.
///
/// # Safety
/// Calling outside an AVX-512F function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn narrow<const WIDTH: usize, const INVERSE_STAGE: bool>(v: __m512i, vector: usize, table: &LaneTable) -> __m512i {
  const { assert!(WIDTH == 1 || WIDTH == 2 || WIDTH == 4 || WIDTH == 8) };
  let lower = const { indices(WIDTH, false) };
  let upper = const { indices(WIDTH, true) };
  // SAFETY: These arrays contain sixteen initialized i32 lanes. The lane-table
  // row is chosen by the public vector index, always below sixteen.
  let (lower, upper, values, twisted) = unsafe {
    (
      _mm512_loadu_si512(lower.as_ptr().cast()),
      _mm512_loadu_si512(upper.as_ptr().cast()),
      _mm512_loadu_si512(table.0[vector].as_ptr().cast()),
      _mm512_loadu_si512(table.1[vector].as_ptr().cast()),
    )
  };
  let a = _mm512_permutexvar_epi32(lower, v);
  let b = _mm512_permutexvar_epi32(upper, v);
  if INVERSE_STAGE {
    _mm512_mask_blend_epi32(
      const { high_mask(WIDTH) },
      _mm512_add_epi32(a, b),
      multiply(_mm512_sub_epi32(a, b), values, twisted),
    )
  } else {
    let b = multiply(b, values, twisted);
    _mm512_mask_blend_epi32(
      const { high_mask(WIDTH) },
      _mm512_add_epi32(a, b),
      _mm512_sub_epi32(a, b),
    )
  }
}

/// # Safety
/// Calling outside an AVX-512F function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn wide_forward<const WIDTH: usize>(chunks: &mut [[u32; LANES]]) {
  const { assert!(WIDTH == 16 || WIDTH == 32 || WIDTH == 64) };
  let mut root = const { 128usize.strict_div(WIDTH) };
  for group in chunks.chunks_exact_mut(const { WIDTH.strict_div(8) }) {
    let (a, b) = group.split_at_mut(const { WIDTH.strict_div(LANES) });
    for (a, b) in a.iter_mut().zip(b) {
      let x = load(a);
      let y = broadcast(load(b), FORWARD[root]);
      store(a, _mm512_add_epi32(x, y));
      store(b, _mm512_sub_epi32(x, y));
    }
    root = root.strict_add(1);
  }
}

/// # Safety
/// Calling outside an AVX-512F function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn wide_inverse<const WIDTH: usize>(chunks: &mut [[u32; LANES]]) {
  const { assert!(WIDTH == 16 || WIDTH == 32 || WIDTH == 64) };
  let mut root = const { 256usize.strict_div(WIDTH).strict_sub(1) };
  for group in chunks.chunks_exact_mut(const { WIDTH.strict_div(8) }) {
    let (a, b) = group.split_at_mut(const { WIDTH.strict_div(LANES) });
    for (a, b) in a.iter_mut().zip(b) {
      let x = load(a);
      let y = load(b);
      store(a, _mm512_add_epi32(x, y));
      store(b, broadcast(_mm512_sub_epi32(x, y), INVERSE[root]));
    }
    root = root.strict_sub(1);
  }
}

/// FIPS 204 forward transform, including Montgomery conversion.
///
/// Every signed product is in `(-q, q)`, so the eight stages leave magnitudes
/// below `9q`. The final reduction restores canonical Montgomery residues.
///
/// # Safety
/// The caller must establish AVX-512F and canonical input coefficients below q.
#[target_feature(enable = "avx512f")]
pub(in crate::auth::mldsa::poly) unsafe fn ntt(poly: &mut Poly) {
  let chunks = poly.0.as_chunks_mut::<LANES>().0;
  let (a, b) = chunks.split_at_mut(8);
  for (a, b) in a.iter_mut().zip(b) {
    let x = broadcast(load(a), R2_TWIDDLE);
    let y = broadcast(load(b), FIRST_TWIDDLE);
    store(a, _mm512_add_epi32(x, y));
    store(b, _mm512_sub_epi32(x, y));
  }
  wide_forward::<64>(chunks);
  wide_forward::<32>(chunks);
  wide_forward::<16>(chunks);
  for (vector, chunk) in chunks.iter_mut().enumerate() {
    let x = narrow::<8, false>(load(chunk), vector, &FORWARD_8);
    let x = narrow::<4, false>(x, vector, &FORWARD_4);
    let x = narrow::<2, false>(x, vector, &FORWARD_2);
    store(chunk, canonical(narrow::<1, false>(x, vector, &FORWARD_1)));
  }
}

/// FIPS 204 inverse transform, including normalization and Montgomery removal.
///
/// Four initial sum stages stay below `16q`; reducing then bounds the next
/// three sums by `4.2q` and the final sum by `8.4q`. The scaled last butterfly
/// leaves `(-q, q)` before canonicalization, as in the AVX2 backend.
///
/// # Safety
/// The caller must establish AVX-512F and canonical input coefficients below q.
#[target_feature(enable = "avx512f")]
pub(in crate::auth::mldsa::poly) unsafe fn inverse_ntt(poly: &mut Poly) {
  let chunks = poly.0.as_chunks_mut::<LANES>().0;
  for (vector, chunk) in chunks.iter_mut().enumerate() {
    let x = narrow::<1, true>(load(chunk), vector, &INVERSE_1);
    let x = narrow::<2, true>(x, vector, &INVERSE_2);
    let x = narrow::<4, true>(x, vector, &INVERSE_4);
    store(chunk, reduce32(narrow::<8, true>(x, vector, &INVERSE_8)));
  }
  wide_inverse::<16>(chunks);
  wide_inverse::<32>(chunks);
  wide_inverse::<64>(chunks);
  let (a, b) = chunks.split_at_mut(8);
  for (a, b) in a.iter_mut().zip(b) {
    let x = load(a);
    let y = load(b);
    store(a, canonical(broadcast(_mm512_add_epi32(x, y), INV_N_TWIDDLE)));
    store(b, canonical(broadcast(_mm512_sub_epi32(x, y), LAST_TWIDDLE)));
  }
}

/// Montgomery-reduce eight even and eight odd 64-bit product sums.
///
/// Each sum is at most `7(q-1)^2 < 2^49`. With signed low word
/// `m = sum·q⁻¹ mod 2^32`, `(sum-mq)/2^32` lies in `(-q,q)`: the positive
/// `sum/2^32` term is below `115000`, while `|mq/2^32| <= q/2`.
/// One masked addition of q therefore gives the canonical residue.
///
/// # Safety
/// Calling outside an AVX-512F function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn reduce_products(even: __m512i, odd: __m512i) -> __m512i {
  let q = _mm512_set1_epi32(Q.cast_signed());
  let qinv = _mm512_set1_epi32(QINV);
  let even_m = _mm512_mul_epi32(even, qinv);
  let odd_m = _mm512_mul_epi32(odd, qinv);
  let even = _mm512_srli_epi64::<32>(_mm512_sub_epi64(even, _mm512_mul_epi32(even_m, q)));
  let odd = _mm512_sub_epi64(odd, _mm512_mul_epi32(odd_m, q));
  let value = _mm512_mask_blend_epi32(0xaaaa, even, odd);
  _mm512_add_epi32(value, _mm512_and_si512(_mm512_srai_epi32::<31>(value), q))
}

/// Canonical pointwise Montgomery product, equal to the portable authority.
///
/// # Safety
/// The caller must establish AVX-512F, canonical operands below q, and disjoint output.
#[target_feature(enable = "avx512f")]
pub(in crate::auth::mldsa::poly) unsafe fn product(out: &mut [u32; N], a: &[u32; N], b: &[u32; N]) {
  for ((out, a), b) in out
    .as_chunks_mut::<LANES>()
    .0
    .iter_mut()
    .zip(a.as_chunks::<LANES>().0)
    .zip(b.as_chunks::<LANES>().0)
  {
    let a = load(a);
    let b = load(b);
    store(
      out,
      reduce_products(
        _mm512_mul_epi32(a, b),
        _mm512_mul_epi32(_mm512_srli_epi64::<32>(a), _mm512_srli_epi64::<32>(b)),
      ),
    );
  }
}

/// Accumulate an entire FIPS 204 matrix row before one Montgomery reduction.
///
/// # Safety
/// The caller must establish AVX-512F, canonical operands below q, and disjoint output.
#[target_feature(enable = "avx512f")]
pub(in crate::auth::mldsa::poly) unsafe fn row_product<const L: usize>(
  out: &mut [u32; N],
  a: &[[u32; N]; L],
  b: &[Poly; L],
) {
  const { assert!(L <= 7) };
  for (vector, out) in out.as_chunks_mut::<LANES>().0.iter_mut().enumerate() {
    let mut even = _mm512_setzero_si512();
    let mut odd = _mm512_setzero_si512();
    for (a, b) in a.iter().zip(b) {
      let a = load(&a.as_chunks::<LANES>().0[vector]);
      let b = load(&b.0.as_chunks::<LANES>().0[vector]);
      even = _mm512_add_epi64(even, _mm512_mul_epi32(a, b));
      odd = _mm512_add_epi64(
        odd,
        _mm512_mul_epi32(_mm512_srli_epi64::<32>(a), _mm512_srli_epi64::<32>(b)),
      );
    }
    store(out, reduce_products(even, odd));
  }
}
