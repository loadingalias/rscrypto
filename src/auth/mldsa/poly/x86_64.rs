//! Eight-lane ML-DSA arithmetic and challenge scans for Linux AVX2.
//!
//! Products and elementwise operations apply the portable canonical Montgomery arithmetic to
//! lanes. The transforms use signed Montgomery multiplication with lazily reduced lanes and
//! reduce to the same canonical residues as the portable transforms. Roots and polynomial layout
//! remain unchanged. All shuffles and addresses depend on public transform stages, never
//! coefficient values.

use core::arch::x86_64::*;

use super::{FIRST_FACTOR, INV_N, LAST_FACTOR, N, NEG_Q_INVERSE, Poly, Q, R2, ROOTS};
use crate::auth::mldsa::sampling::{CHALLENGE_BYTES, CHALLENGE_WINDOWS};

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

/// `q⁻¹ mod 2^32`, as a signed lane.
const QINV: i32 = 58_728_449;

/// A signed Montgomery multiplier: the centered residue `value` in `(-q/2, q/2]` and
/// `twisted = value·q⁻¹ mod 2^32`.
#[derive(Clone, Copy)]
struct Twiddle {
  value: i32,
  twisted: i32,
}

const fn twiddle(x: u32) -> Twiddle {
  let value = if x > Q / 2 {
    x.cast_signed().wrapping_sub(Q.cast_signed())
  } else {
    x.cast_signed()
  };
  Twiddle {
    value,
    twisted: value.wrapping_mul(QINV),
  }
}

/// Forward multipliers `ROOTS[k]` and inverse multipliers `q - ROOTS[k]`.
const FORWARD: [Twiddle; N] = twiddles(false);
const INVERSE: [Twiddle; N] = twiddles(true);

const fn twiddles(inverse: bool) -> [Twiddle; N] {
  let mut table = [Twiddle { value: 0, twisted: 0 }; N];
  let mut k = 1usize;
  while k < N {
    table[k] = twiddle(if inverse { Q.strict_sub(ROOTS[k]) } else { ROOTS[k] });
    k = k.strict_add(1);
  }
  table
}

/// Per-vector lane multipliers for the in-vector width-2 and width-1 stages, as
/// `(values, twisted)` rows of eight lanes, one row per eight-coefficient vector.
type LaneTable = ([[i32; 8]; N / 8], [[i32; 8]; N / 8]);

/// Width 2: lanes 0-3 use root `first(j)`, lanes 4-7 use `first(j) + step`.
/// Width 1: lane pair `2i, 2i + 1` uses root `first(j) + i·step` for `i` in `0..4`.
const fn lane_table(inverse: bool, width_one: bool) -> LaneTable {
  let table = if inverse { INVERSE } else { FORWARD };
  let mut values = [[0i32; 8]; N / 8];
  let mut twisted = [[0i32; 8]; N / 8];
  let mut j = 0usize;
  while j < N / 8 {
    let mut lane = 0usize;
    while lane < 8 {
      let offset = if width_one { lane / 2 } else { lane / 4 };
      let root = match (inverse, width_one) {
        (false, false) => 64usize.strict_add(j.strict_mul(2)).strict_add(offset),
        (false, true) => 128usize.strict_add(j.strict_mul(4)).strict_add(offset),
        (true, false) => 127usize.strict_sub(j.strict_mul(2)).strict_sub(offset),
        (true, true) => 255usize.strict_sub(j.strict_mul(4)).strict_sub(offset),
      };
      values[j][lane] = table[root].value;
      twisted[j][lane] = table[root].twisted;
      lane = lane.strict_add(1);
    }
    j = j.strict_add(1);
  }
  (values, twisted)
}

const FORWARD_WIDTH_2: LaneTable = lane_table(false, false);
const FORWARD_WIDTH_1: LaneTable = lane_table(false, true);
const INVERSE_WIDTH_2: LaneTable = lane_table(true, false);
const INVERSE_WIDTH_1: LaneTable = lane_table(true, true);
const R2_TWIDDLE: Twiddle = twiddle(R2);
const FIRST_TWIDDLE: Twiddle = twiddle(FIRST_FACTOR);
const INV_N_TWIDDLE: Twiddle = twiddle(INV_N);
const LAST_TWIDDLE: Twiddle = twiddle(LAST_FACTOR);

/// Signed Montgomery multiplication `a·value·2^-32 mod q` for one multiplier in every lane.
///
/// With `m = a·twisted mod 2^32`, `a·value - m·q` is an exact multiple of 2^32. For any `i32`
/// lane and `|value| <= q/2` the quotient lies in `(-q, q)`.
///
/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn mont_mul_broadcast(a: __m256i, multiplier: Twiddle) -> __m256i {
  let value = _mm256_set1_epi32(multiplier.value);
  let twisted = _mm256_set1_epi32(multiplier.twisted);
  mont_mul_lanes_shifted(a, value, twisted, value, twisted)
}

/// [`mont_mul_broadcast`] with a multiplier per lane.
///
/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn mont_mul_lanes(a: __m256i, value: __m256i, twisted: __m256i) -> __m256i {
  mont_mul_lanes_shifted(
    a,
    value,
    twisted,
    _mm256_srli_epi64::<32>(value),
    _mm256_srli_epi64::<32>(twisted),
  )
}

/// Even lanes multiply by `value`/`twisted`; odd lanes, moved down, by `value_odd`/`twisted_odd`.
///
/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn mont_mul_lanes_shifted(
  a: __m256i,
  value: __m256i,
  twisted: __m256i,
  value_odd: __m256i,
  twisted_odd: __m256i,
) -> __m256i {
  let q = _mm256_set1_epi32(Q.cast_signed());
  let a_odd = _mm256_srli_epi64::<32>(a);
  let even = _mm256_sub_epi64(
    _mm256_mul_epi32(a, value),
    _mm256_mul_epi32(_mm256_mul_epi32(a, twisted), q),
  );
  let odd = _mm256_sub_epi64(
    _mm256_mul_epi32(a_odd, value_odd),
    _mm256_mul_epi32(_mm256_mul_epi32(a_odd, twisted_odd), q),
  );
  _mm256_blend_epi32::<0xaa>(_mm256_srli_epi64::<32>(even), odd)
}

/// Signed reduction of lanes in `(-2^31 + 2^22, 2^31 - 2^22)` to `(-0.52·q, 0.52·q)` for
/// lanes below `16·q` in magnitude: `x - ⌊(x + 2^22) / 2^23⌋·q`, with `t·q` as shifts because
/// `q = 2^23 - 2^13 + 1`.
///
/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn reduce32(x: __m256i) -> __m256i {
  let t = _mm256_srai_epi32::<23>(_mm256_add_epi32(x, _mm256_set1_epi32(1 << 22)));
  let tq = _mm256_add_epi32(
    _mm256_sub_epi32(_mm256_slli_epi32::<23>(t), _mm256_slli_epi32::<13>(t)),
    t,
  );
  _mm256_sub_epi32(x, tq)
}

/// The canonical residue in `[0, q)` of a lane below `9·q` in magnitude.
///
/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn canonical(x: __m256i) -> __m256i {
  // reduce32 leaves (-q, q); a negative lane gains q.
  let r = reduce32(x);
  _mm256_add_epi32(
    r,
    _mm256_and_si256(_mm256_srai_epi32::<31>(r), _mm256_set1_epi32(Q.cast_signed())),
  )
}

/// Cooley–Tukey butterfly `(a, b) -> (a + t, a - t)` with `t = b·ζ`, without reduction.
///
/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn forward_butterfly(a: &mut __m256i, b: &mut __m256i, zeta: Twiddle) {
  let t = mont_mul_broadcast(*b, zeta);
  *b = _mm256_sub_epi32(*a, t);
  *a = _mm256_add_epi32(*a, t);
}

/// Gentleman–Sande butterfly `(a, b) -> (a + b, (a - b)·ζ)`, without reducing the sum.
///
/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn inverse_butterfly(a: &mut __m256i, b: &mut __m256i, zeta: Twiddle) {
  let difference = _mm256_sub_epi32(*a, *b);
  *a = _mm256_add_epi32(*a, *b);
  *b = mont_mul_broadcast(difference, zeta);
}

/// Load row `j` of a lane table.
///
/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn lane_row(table: &LaneTable, j: usize) -> (__m256i, __m256i) {
  // SAFETY: Each row is eight initialized i32 values; the unaligned loads read exactly them.
  unsafe {
    (
      _mm256_loadu_si256(table.0[j].as_ptr().cast()),
      _mm256_loadu_si256(table.1[j].as_ptr().cast()),
    )
  }
}

/// The width-4, width-2 and width-1 forward stages inside vector `j`.
///
/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn forward_in_vector(v: __m256i, j: usize) -> __m256i {
  let x = _mm256_permute2x128_si256::<0x00>(v, v);
  let y = mont_mul_broadcast(_mm256_permute2x128_si256::<0x11>(v, v), FORWARD[32usize.strict_add(j)]);
  let v = _mm256_blend_epi32::<0xf0>(_mm256_add_epi32(x, y), _mm256_sub_epi32(x, y));
  let (value, twisted) = lane_row(&FORWARD_WIDTH_2, j);
  let x = _mm256_shuffle_epi32::<0x44>(v);
  let y = mont_mul_lanes(_mm256_shuffle_epi32::<0xee>(v), value, twisted);
  let v = _mm256_blend_epi32::<0xcc>(_mm256_add_epi32(x, y), _mm256_sub_epi32(x, y));
  let (value, twisted) = lane_row(&FORWARD_WIDTH_1, j);
  let x = _mm256_shuffle_epi32::<0xa0>(v);
  let y = mont_mul_lanes(_mm256_shuffle_epi32::<0xf5>(v), value, twisted);
  _mm256_blend_epi32::<0xaa>(_mm256_add_epi32(x, y), _mm256_sub_epi32(x, y))
}

/// The width-1, width-2 and width-4 inverse stages inside vector `j`.
///
/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn inverse_in_vector(v: __m256i, j: usize) -> __m256i {
  let (value, twisted) = lane_row(&INVERSE_WIDTH_1, j);
  let x = _mm256_shuffle_epi32::<0xa0>(v);
  let y = _mm256_shuffle_epi32::<0xf5>(v);
  let v = _mm256_blend_epi32::<0xaa>(
    _mm256_add_epi32(x, y),
    mont_mul_lanes(_mm256_sub_epi32(x, y), value, twisted),
  );
  let (value, twisted) = lane_row(&INVERSE_WIDTH_2, j);
  let x = _mm256_shuffle_epi32::<0x44>(v);
  let y = _mm256_shuffle_epi32::<0xee>(v);
  let v = _mm256_blend_epi32::<0xcc>(
    _mm256_add_epi32(x, y),
    mont_mul_lanes(_mm256_sub_epi32(x, y), value, twisted),
  );
  let x = _mm256_permute2x128_si256::<0x00>(v, v);
  let y = _mm256_permute2x128_si256::<0x11>(v, v);
  _mm256_blend_epi32::<0xf0>(
    _mm256_add_epi32(x, y),
    mont_mul_broadcast(_mm256_sub_epi32(x, y), INVERSE[63usize.strict_sub(j)]),
  )
}

/// Load vector `j` of the polynomial.
///
/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn load_vector(chunks: &[[u32; 8]], j: usize) -> __m256i {
  // SAFETY: `chunks[j]` is eight initialized u32 lanes; the unaligned load reads exactly them.
  unsafe { _mm256_loadu_si256(chunks[j].as_ptr().cast()) }
}

/// Store vector `j` of the polynomial.
///
/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn store_vector(chunks: &mut [[u32; 8]], j: usize, value: __m256i) {
  // SAFETY: `chunks[j]` is eight uniquely borrowed u32 lanes; the unaligned store writes exactly them.
  unsafe { _mm256_storeu_si256(chunks[j].as_mut_ptr().cast(), value) }
}

/// Load four vectors of the polynomial.
///
/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn load_four(chunks: &[[u32; 8]], indices: [usize; 4]) -> [__m256i; 4] {
  let mut out = [_mm256_setzero_si256(); 4];
  for (lanes, j) in out.iter_mut().zip(indices) {
    *lanes = load_vector(chunks, j);
  }
  out
}

/// Store four vectors of the polynomial.
///
/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn store_four(chunks: &mut [[u32; 8]], indices: [usize; 4], values: [__m256i; 4]) {
  for (j, lanes) in indices.into_iter().zip(values) {
    store_vector(chunks, j, lanes);
  }
}

/// FIPS 204 Algorithm 41 with conversion to Montgomery form, equal to the portable transform.
///
/// Lanes are signed and lazily reduced: each Montgomery product lies in `(-q, q)` and each
/// stage adds at most `q`, so after the eight stages every lane is below `9·q` and one final
/// reduction returns canonical residues. Three passes each hold four coefficient vectors, so the
/// working set stays in registers: widths 128 and 64, then 32 and 16, then 8 across vector pairs
/// with 4, 2 and 1 inside each vector.
///
/// # Safety
/// The caller must establish AVX2. All coefficients must be below q.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn ntt(poly: &mut Poly) {
  let chunks = poly.0.as_chunks_mut::<8>().0;
  for v in 0..8usize {
    let indices = [v, v.strict_add(8), v.strict_add(16), v.strict_add(24)];
    let mut x = load_four(chunks, indices);
    let [x0, x1, x2, x3] = &mut x;
    // Width 128, fused with the Montgomery conversion: M(a, R2) and M(b, M(R2, ζ1)).
    for (a, b) in [(&mut *x0, &mut *x2), (&mut *x1, &mut *x3)] {
      let left = mont_mul_broadcast(*a, R2_TWIDDLE);
      let right = mont_mul_broadcast(*b, FIRST_TWIDDLE);
      *a = _mm256_add_epi32(left, right);
      *b = _mm256_sub_epi32(left, right);
    }
    forward_butterfly(x0, x1, FORWARD[2]);
    forward_butterfly(x2, x3, FORWARD[3]);
    store_four(chunks, indices, x);
  }
  for g in 0..4usize {
    for b in 0..2usize {
      let first = g.strict_mul(8).strict_add(b);
      let indices = [first, first.strict_add(2), first.strict_add(4), first.strict_add(6)];
      let mut x = load_four(chunks, indices);
      let [x0, x1, x2, x3] = &mut x;
      forward_butterfly(x0, x2, FORWARD[4usize.strict_add(g)]);
      forward_butterfly(x1, x3, FORWARD[4usize.strict_add(g)]);
      forward_butterfly(x0, x1, FORWARD[8usize.strict_add(g.strict_mul(2))]);
      forward_butterfly(x2, x3, FORWARD[9usize.strict_add(g.strict_mul(2))]);
      store_four(chunks, indices, x);
    }
  }
  for m in 0..16usize {
    let j = m.strict_mul(2);
    let mut a = load_vector(chunks, j);
    let mut b = load_vector(chunks, j.strict_add(1));
    forward_butterfly(&mut a, &mut b, FORWARD[16usize.strict_add(m)]);
    store_vector(chunks, j, canonical(forward_in_vector(a, j)));
    store_vector(
      chunks,
      j.strict_add(1),
      canonical(forward_in_vector(b, j.strict_add(1))),
    );
  }
}

/// FIPS 204 Algorithm 42 with conversion out of Montgomery form, equal to the portable
/// transform.
///
/// The passes mirror [`ntt`]. Sums double at each stage, so after width 8 (below `16·q`) every
/// lane is reduced to below `0.52·q`; the remaining stages stay below `8.4·q`, and the last stage
/// multiplies both outputs by `n⁻¹`-scaled constants, leaving `(-q, q)` before the final
/// reduction.
///
/// # Safety
/// The caller must establish AVX2. All coefficients must be below q.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn inverse_ntt(poly: &mut Poly) {
  let chunks = poly.0.as_chunks_mut::<8>().0;
  for m in 0..16usize {
    let j = m.strict_mul(2);
    let mut a = inverse_in_vector(load_vector(chunks, j), j);
    let mut b = inverse_in_vector(load_vector(chunks, j.strict_add(1)), j.strict_add(1));
    inverse_butterfly(&mut a, &mut b, INVERSE[31usize.strict_sub(m)]);
    store_vector(chunks, j, reduce32(a));
    store_vector(chunks, j.strict_add(1), reduce32(b));
  }
  for g in 0..4usize {
    for b in 0..2usize {
      let first = g.strict_mul(8).strict_add(b);
      let indices = [first, first.strict_add(2), first.strict_add(4), first.strict_add(6)];
      let mut x = load_four(chunks, indices);
      let [x0, x1, x2, x3] = &mut x;
      inverse_butterfly(x0, x1, INVERSE[15usize.strict_sub(g.strict_mul(2))]);
      inverse_butterfly(x2, x3, INVERSE[14usize.strict_sub(g.strict_mul(2))]);
      inverse_butterfly(x0, x2, INVERSE[7usize.strict_sub(g)]);
      inverse_butterfly(x1, x3, INVERSE[7usize.strict_sub(g)]);
      store_four(chunks, indices, x);
    }
  }
  for v in 0..8usize {
    let indices = [v, v.strict_add(8), v.strict_add(16), v.strict_add(24)];
    let mut x = load_four(chunks, indices);
    let [x0, x1, x2, x3] = &mut x;
    inverse_butterfly(x0, x1, INVERSE[3]);
    inverse_butterfly(x2, x3, INVERSE[2]);
    // Width 128, fused with normalization: M(a + b, n⁻¹) and M(a - b, M(q - ζ1, n⁻¹)).
    for (a, b) in [(&mut *x0, &mut *x2), (&mut *x1, &mut *x3)] {
      let difference = _mm256_sub_epi32(*a, *b);
      *a = canonical(mont_mul_broadcast(_mm256_add_epi32(*a, *b), INV_N_TWIDDLE));
      *b = canonical(mont_mul_broadcast(difference, LAST_TWIDDLE));
    }
    store_four(chunks, indices, x);
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

/// Fixed-work search for the first eligible challenge byte.
///
/// Returns the minimum of `position << 8 | byte` over positions at or after
/// `offset` whose byte is at most `limit`, or `u32::MAX` when none qualifies,
/// as `sampling::first_challenge_candidate` does. Every window is read, and
/// neither address nor control flow depends on a byte, offset or limit.
///
/// # Safety
/// The caller must establish AVX2.
#[target_feature(enable = "avx2")]
pub(in crate::auth::mldsa) unsafe fn challenge_first_candidate(
  bytes: &[u8; CHALLENGE_BYTES],
  offset: u32,
  limit: u32,
) -> u32 {
  // Offsets are at most 221 and limits at most 255. Masking first makes each
  // conversion provably in range, so no check branches on the secret offset.
  let offset = _mm256_set1_epi16(
    u16::try_from(offset & 0xffff)
      .expect("masked offsets fit u16")
      .cast_signed(),
  );
  let limit = _mm256_set1_epi16(
    u16::try_from(limit & 0xffff)
      .expect("masked limits fit u16")
      .cast_signed(),
  );
  let lanes = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  let none = _mm256_set1_epi16(-1);
  let mut first = none;
  for start in CHALLENGE_WINDOWS {
    let window: &[u8; 16] = bytes[start..start.strict_add(16)]
      .try_into()
      .expect("challenge windows have sixteen bytes");
    // SAFETY: The unaligned load reads exactly the sixteen initialized bytes of `window`.
    let candidates = _mm256_cvtepu8_epi16(unsafe { _mm_loadu_si128(window.as_ptr().cast()) });
    let start = i16::try_from(start).expect("challenge positions fit i16");
    let positions = _mm256_add_epi16(_mm256_set1_epi16(start), lanes);
    // Unsigned comparisons as min/max equalities: position >= offset, byte <= limit.
    let after_offset = _mm256_cmpeq_epi16(_mm256_max_epu16(positions, offset), positions);
    let in_range = _mm256_cmpeq_epi16(_mm256_min_epu16(candidates, limit), candidates);
    let packed = _mm256_or_si256(_mm256_slli_epi16::<8>(positions), candidates);
    let eligible = _mm256_and_si256(after_offset, in_range);
    first = _mm256_min_epu16(first, _mm256_or_si256(packed, _mm256_andnot_si256(eligible, none)));
  }
  let first = _mm_min_epu16(_mm256_castsi256_si128(first), _mm256_extracti128_si256::<1>(first));
  let first = _mm_cvtsi128_si32(_mm_minpos_epu16(first)).cast_unsigned() & 0xffff;
  // An eligible packed value is at most 220 << 8 | 255, so 0xffff means none.
  let none = (first ^ 0xffff).wrapping_sub(1) >> 31;
  first | super::opaque_mask(0u32.wrapping_sub(none))
}

/// Write `sign` at coefficient `j` and return the coefficient it replaces.
///
/// Every coefficient is read and written; the index comparison selects lanes
/// without a secret-dependent address, as the portable placement scan does.
///
/// # Safety
/// The caller must establish AVX2.
#[target_feature(enable = "avx2")]
pub(in crate::auth::mldsa) unsafe fn challenge_place(coefficients: &mut [u32; N], j: u32, sign: u32) -> u32 {
  let target = _mm256_set1_epi32(j.cast_signed());
  let sign = _mm256_set1_epi32(sign.cast_signed());
  let mut index = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  let mut previous = _mm256_setzero_si256();
  for chunk in coefficients.as_chunks_mut::<8>().0 {
    // SAFETY: Each unaligned load and store covers exactly one uniquely borrowed
    // eight-lane chunk. N has no vector tail. The caller establishes AVX2.
    unsafe {
      let value = _mm256_loadu_si256(chunk.as_ptr().cast());
      let hit = _mm256_cmpeq_epi32(index, target);
      previous = _mm256_or_si256(previous, _mm256_and_si256(value, hit));
      _mm256_storeu_si256(chunk.as_mut_ptr().cast(), _mm256_blendv_epi8(value, sign, hit));
    }
    index = _mm256_add_epi32(index, _mm256_set1_epi32(8));
  }
  let previous = _mm_or_si128(
    _mm256_castsi256_si128(previous),
    _mm256_extracti128_si256::<1>(previous),
  );
  let previous = _mm_or_si128(previous, _mm_shuffle_epi32::<0x4e>(previous));
  let previous = _mm_or_si128(previous, _mm_shuffle_epi32::<0xb1>(previous));
  _mm_cvtsi128_si32(previous).cast_unsigned()
}

/// Lane-wise canonical addition: inputs below q, output below q.
///
/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn add_canonical(a: __m256i, b: __m256i) -> __m256i {
  // a + b < 2q; when it is below q, subtracting q wraps above 2^31 and the unsigned minimum keeps it.
  let sum = _mm256_add_epi32(a, b);
  _mm256_min_epu32(sum, _mm256_sub_epi32(sum, _mm256_set1_epi32(Q.cast_signed())))
}

/// Lane-wise canonical subtraction: inputs below q, output below q.
///
/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn sub_canonical(a: __m256i, b: __m256i) -> __m256i {
  let q = _mm256_set1_epi32(Q.cast_signed());
  let difference = _mm256_sub_epi32(_mm256_add_epi32(a, q), b);
  _mm256_min_epu32(difference, _mm256_sub_epi32(difference, q))
}

/// Lane-wise FIPS 204 HighBits of canonical coefficients.
///
/// `⌊(x + γ2 - 1) / (2·γ2)⌋` is computed as a shift and a multiply-shift that are exact for
/// every canonical input (`super::tests` checks all q of them); the quotient that equals
/// `(q - 1) / (2·γ2)` maps to zero as in Algorithm 36.
///
/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn high_bits_lanes(x: __m256i, gamma2: u32) -> __m256i {
  let rounded = _mm256_add_epi32(x, _mm256_set1_epi32(gamma2.strict_sub(1).cast_signed()));
  let (quotient, modulus) = if gamma2 == 95_232 {
    let y = _mm256_srli_epi32::<11>(rounded);
    (
      _mm256_srli_epi32::<19>(_mm256_mullo_epi32(y, _mm256_set1_epi32(5_638))),
      44,
    )
  } else {
    let y = _mm256_srli_epi32::<9>(rounded);
    (
      _mm256_srli_epi32::<25>(_mm256_mullo_epi32(y, _mm256_set1_epi32(32_801))),
      16,
    )
  };
  _mm256_andnot_si256(_mm256_cmpeq_epi32(quotient, _mm256_set1_epi32(modulus)), quotient)
}

/// `a[i] = a[i] + b[i] mod q` for canonical coefficients.
///
/// # Safety
/// The caller must establish AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn add_assign(a: &mut [u32; N], b: &[u32; N]) {
  for (a, b) in a.as_chunks_mut::<8>().0.iter_mut().zip(b.as_chunks::<8>().0) {
    // SAFETY: Each chunk is eight initialized u32 lanes; unaligned accesses cover exactly them.
    unsafe {
      let sum = add_canonical(
        _mm256_loadu_si256(a.as_ptr().cast()),
        _mm256_loadu_si256(b.as_ptr().cast()),
      );
      _mm256_storeu_si256(a.as_mut_ptr().cast(), sum);
    }
  }
}

/// `a[i] = a[i] - b[i] mod q` for canonical coefficients.
///
/// # Safety
/// The caller must establish AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn sub_assign(a: &mut [u32; N], b: &[u32; N]) {
  for (a, b) in a.as_chunks_mut::<8>().0.iter_mut().zip(b.as_chunks::<8>().0) {
    // SAFETY: Each chunk is eight initialized u32 lanes; unaligned accesses cover exactly them.
    unsafe {
      let difference = sub_canonical(
        _mm256_loadu_si256(a.as_ptr().cast()),
        _mm256_loadu_si256(b.as_ptr().cast()),
      );
      _mm256_storeu_si256(a.as_mut_ptr().cast(), difference);
    }
  }
}

/// One when any centered coefficient has magnitude at least `bound` (`bound < q/2`), else zero.
///
/// A canonical `x` has centered magnitude at least `bound` exactly when
/// `bound <= x <= q - bound`, that is when `x - bound` (wrapping) is at most `q - 2·bound`.
/// Every lane is tested; only the aggregate leaves.
///
/// # Safety
/// The caller must establish AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn exceeds_bound(a: &[u32; N], bound: u32) -> u32 {
  let offset = _mm256_set1_epi32(bound.cast_signed());
  let limit = _mm256_set1_epi32(Q.strict_sub(bound.strict_mul(2)).cast_signed());
  let mut exceeded = _mm256_setzero_si256();
  for chunk in a.as_chunks::<8>().0 {
    // SAFETY: `chunk` is eight initialized u32 lanes; the unaligned load reads exactly them.
    let x = unsafe { _mm256_loadu_si256(chunk.as_ptr().cast()) };
    let shifted = _mm256_sub_epi32(x, offset);
    // Lanes where `shifted <= limit` (unsigned) are all ones.
    exceeded = _mm256_or_si256(exceeded, _mm256_cmpeq_epi32(_mm256_min_epu32(shifted, limit), shifted));
  }
  u32::from(_mm256_testz_si256(exceeded, exceeded) == 0)
}

/// `out[i] = HighBits(a[i])` for canonical coefficients.
///
/// # Safety
/// The caller must establish AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn high_bits_into(a: &[u32; N], gamma2: u32, out: &mut [u32; N]) {
  for (a, out) in a.as_chunks::<8>().0.iter().zip(out.as_chunks_mut::<8>().0) {
    // SAFETY: Each chunk is eight initialized u32 lanes; unaligned accesses cover exactly them.
    unsafe {
      let high = high_bits_lanes(_mm256_loadu_si256(a.as_ptr().cast()), gamma2);
      _mm256_storeu_si256(out.as_mut_ptr().cast(), high);
    }
  }
}

/// `out[i] = LowBits(a[i]) mod q`, the canonical low part of Decompose.
///
/// # Safety
/// The caller must establish AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn low_bits_into(a: &[u32; N], gamma2: u32, out: &mut [u32; N]) {
  let alpha = _mm256_set1_epi32(gamma2.strict_mul(2).cast_signed());
  for (a, out) in a.as_chunks::<8>().0.iter().zip(out.as_chunks_mut::<8>().0) {
    // SAFETY: Each chunk is eight initialized u32 lanes; unaligned accesses cover exactly them.
    unsafe {
      let x = _mm256_loadu_si256(a.as_ptr().cast());
      // HighBits is at most 15 (γ2 = 261888) or 43 (γ2 = 95232), so high·2γ2 is below q.
      let scaled = _mm256_mullo_epi32(high_bits_lanes(x, gamma2), alpha);
      _mm256_storeu_si256(out.as_mut_ptr().cast(), sub_canonical(x, scaled));
    }
  }
}

/// Replace `r[i]` with the hint bit `HighBits(r[i]) != HighBits(r[i] + t[i])` and return their
/// count.
///
/// # Safety
/// The caller must establish AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn make_hints(r: &mut [u32; N], t: &[u32; N], gamma2: u32) -> usize {
  let one = _mm256_set1_epi32(1);
  let mut count = _mm256_setzero_si256();
  for (r, t) in r.as_chunks_mut::<8>().0.iter_mut().zip(t.as_chunks::<8>().0) {
    // SAFETY: Each chunk is eight initialized u32 lanes; unaligned accesses cover exactly them.
    unsafe {
      let x = _mm256_loadu_si256(r.as_ptr().cast());
      let shifted = add_canonical(x, _mm256_loadu_si256(t.as_ptr().cast()));
      let same = _mm256_cmpeq_epi32(high_bits_lanes(x, gamma2), high_bits_lanes(shifted, gamma2));
      let hint = _mm256_andnot_si256(same, one);
      count = _mm256_add_epi32(count, hint);
      _mm256_storeu_si256(r.as_mut_ptr().cast(), hint);
    }
  }
  let mut lanes = [0u32; 8];
  // SAFETY: `lanes` is eight writable u32 values; the unaligned store writes exactly them.
  unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), count) };
  // Each lane counts at most N / 8 hints.
  let total = lanes.iter().fold(0u32, |sum, &lane| sum.strict_add(lane));
  usize::try_from(total).expect("at most N hints")
}
