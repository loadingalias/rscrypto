//! AVX2 parallel point arithmetic for Ed25519.
//!
//! Extended Edwards point operations using HWCD'08 parallel formulas that
//! compute all 4 coordinates (X, Y, Z, T) simultaneously via vectorized
//! field arithmetic.
//!
//! # Hamburg scaling trick
//!
//! The cached point format absorbs the curve constant `d = -d1/d2` (where
//! `d1 = 121665`, `d2 = 121666`) into the precomputed values. This
//! eliminates a separate multiply by `2d` during addition, at the cost of
//! scaling all output coordinates by `d2²` — which cancels in projective
//! coordinates.

#[cfg(target_arch = "x86_64")]
use super::{
  field::FieldElement,
  field_avx2::{FieldElement2625x4, Lanes, Shuffle},
  field_ifma::FieldElement51x4,
  point::{CachedPoint, ExtendedPoint},
  scalar,
};

/// Hamburg constants for the curve `d = -d1/d2`.
const D1: u64 = 121_665;
const D2: u64 = 121_666;

/// Extended Edwards point in AVX2 vectorized form.
///
/// Lanes: `(X, Y, Z, T)` as `(A, B, C, D)`.
#[derive(Clone, Copy)]
#[cfg(target_arch = "x86_64")]
pub(crate) struct ExtendedPointAvx2(pub(crate) FieldElement2625x4);

/// Cached point for efficient addition (Hamburg-scaled projective format).
///
/// Lanes: `(d2·(Y−X), d2·(Y+X), 2·d2·Z, −2·d1·T)` as `(A, B, C, D)`.
///
/// The `d2` scaling factor cancels in projective coordinates after the
/// uniform multiply.
#[derive(Clone, Copy)]
#[cfg(target_arch = "x86_64")]
pub(crate) struct CachedPointAvx2(pub(crate) FieldElement2625x4);

#[cfg(target_arch = "x86_64")]
impl ExtendedPointAvx2 {
  /// Pack a scalar extended point into AVX2 vectorized form.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn from_extended(p: &ExtendedPoint) -> Self {
    let (x, y, z, t) = p.components();
    Self(FieldElement2625x4::new(x, y, z, t))
  }

  /// Unpack back to a scalar extended point.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn to_extended(self) -> ExtendedPoint {
    let [x, y, z, t] = self.0.split();
    ExtendedPoint::from_raw(x, y, z, t)
  }

  /// Convert to Hamburg-scaled cached format for addition.
  ///
  /// Computes: `(d2·(Y−X), d2·(Y+X), 2·d2·Z, −2·d1·T)`.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn to_cached(self) -> CachedPointAvx2 {
    // Step 1: Compute (Y-X, Y+X) in lanes A,B; keep Z,T in C,D.
    let ds = self.0.diff_sum(); // (Y-X, Y+X, T-Z, Z+T)
    let prepared = self.0.blend(&ds, Lanes::AB); // (Y-X, Y+X, Z, T)

    // Step 2: Multiply by Hamburg constants (d2, d2, 2·d2, 2·d1).
    let constants = hamburg_constants();
    let scaled = prepared.mul(&constants);

    // Step 3: Negate lane D → −2·d1·T.
    let negated = scaled.negate_lazy();
    CachedPointAvx2(scaled.blend(&negated, Lanes::D))
  }

  /// Add a cached point to this extended point.
  ///
  /// Uses HWCD'08 parallel addition: 2 uniform vectorized multiplies.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn add_cached(&self, other: &CachedPointAvx2) -> Self {
    // Step 1: Prepare self as (Y-X, Y+X, Z, T) by blending diff_sum into A,B.
    let ds = self.0.diff_sum();
    let tmp = self.0.blend(&ds, Lanes::AB); // (Y1-X1, Y1+X1, Z1, T1)

    // Step 2: Uniform multiply with cached point.
    // (Y1-X1)·d2·(Y2-X2), (Y1+X1)·d2·(Y2+X2), Z1·2·d2·Z2, T1·(−2·d1·T2)
    let product = tmp.mul(&other.0);

    // Step 3: Swap C↔D to align for diff_sum.
    // After swap: lane C has the T-product, lane D has the Z-product.
    let swapped = product.shuffle(Shuffle::ABDC);

    // Step 4: diff_sum computes (e, h, f, g) (up to Hamburg scaling).
    let ehfg = swapped.diff_sum();

    // Step 5: Shuffle into final multiply operands.
    let t0 = ehfg.shuffle(Shuffle::ADDA); // (e, g, g, e)
    let t1 = ehfg.shuffle(Shuffle::CBCB); // (f, h, f, h)

    // Step 6: Uniform multiply → (e·f, g·h, g·f, e·h) = (X3, Y3, Z3, T3).
    Self(t0.mul(&t1))
  }

  /// Double this point using dedicated HWCD'08 parallel doubling.
  ///
  /// Uses 1 uniform vectorized square + 1 uniform vectorized multiply.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn double(&self) -> Self {
    // Step 1: Build (X, Y, Z, X+Y) for squaring.
    let ab = self.0.shuffle(Shuffle::ABAB); // (X, Y, X, Y)
    let ba = ab.shuffle(Shuffle::BADC); // (Y, X, Y, X)
    let xy_sum = ab.add(&ba); // (X+Y, Y+X, X+Y, Y+X)
    let prepared = self.0.blend(&xy_sum, Lanes::D); // (X, Y, Z, X+Y)

    // Step 2: Square with D-lane negation.
    // Result: (X², Y², Z², −(X+Y)²) = (S1, S2, S3, −S4).
    let sq = prepared.square_and_negate_d();

    // Step 3: Compute (S5, S6, S8, S9) via non-uniform adds.
    //   S5 = S1 + S2          (= −h)
    //   S6 = S1 − S2          (= −g)
    //   S8 = S1 − S2 + 2·S3  (= −f)
    //   S9 = S1 + S2 − S4    (= −e)
    // Double-negations cancel in the final multiply.

    let zero = FieldElement2625x4::zero();
    let s1 = sq.shuffle(Shuffle::AAAA); // (S1, S1, S1, S1)
    let s2 = sq.shuffle(Shuffle::BBBB); // (S2, S2, S2, S2)

    // Build the target vector incrementally:
    let sq_doubled = sq.add(&sq); // (2S1, 2S2, 2S3, −2S4)
    let mut tmp = zero.blend(&sq_doubled, Lanes::C); // (0, 0, 2S3, 0)
    tmp = tmp.blend(&sq, Lanes::D); // (0, 0, 2S3, −S4)
    tmp = tmp.add(&s1); // (S1, S1, S1+2S3, S1−S4)

    let s2_in_ad = zero.blend(&s2, Lanes::AD); // (S2, 0, 0, S2)
    tmp = tmp.add(&s2_in_ad); // (S1+S2, S1, S1+2S3, S1+S2−S4)

    let neg_s2 = s2.negate_lazy();
    let neg_s2_in_bc = zero.blend(&neg_s2, Lanes::BC); // (0, −S2, −S2, 0)
    tmp = tmp.add(&neg_s2_in_bc); // (S1+S2, S1−S2, S1−S2+2S3, S1+S2−S4) = (S5, S6, S8, S9)

    // Step 4: Shuffle into final multiply operands.
    let t0 = tmp.shuffle(Shuffle::CACA); // (S8, S5, S8, S5)
    let t1 = tmp.shuffle(Shuffle::DBBD); // (S9, S6, S6, S9)

    // Step 5: Uniform multiply → (S8·S9, S5·S6, S8·S6, S5·S9) = (X3, Y3, Z3, T3).
    Self(t0.mul(&t1))
  }
}

#[cfg(target_arch = "x86_64")]
impl CachedPointAvx2 {
  /// Negate the cached point (for subtraction).
  ///
  /// Swaps `(Y−X)` and `(Y+X)` and negates `T`, implementing curve
  /// negation `(X:Y:Z:T) → (−X:Y:Z:−T)`.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn neg(&self) -> Self {
    let swapped = self.0.shuffle(Shuffle::BACD); // swap A↔B, keep C,D
    let negated = swapped.negate_lazy();
    Self(swapped.blend(&negated, Lanes::D)) // negate D only
  }
}

/// Hamburg scaling constants for projective cached: `(d2, d2, 2·d2, 2·d1)`.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn hamburg_constants() -> FieldElement2625x4 {
  let d2_fe = FieldElement::from_small(D2);
  let d2_fe_2 = FieldElement::from_small(D2.wrapping_mul(2));
  let d1_fe_2 = FieldElement::from_small(D1.wrapping_mul(2));
  FieldElement2625x4::new(&d2_fe, &d2_fe, &d2_fe_2, &d1_fe_2)
}

/// Hamburg scaling constants for affine cached: `(d2, d2, 2·d2, d2)`.
///
/// The D-lane uses `d2` (not `2·d1`) because the affine table's `t2d`
/// field already encodes the `d` constant, so `d2 · t2d = −2·d1·T`.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn hamburg_affine_constants() -> FieldElement2625x4 {
  let d2_fe = FieldElement::from_small(D2);
  let d2_fe_2 = FieldElement::from_small(D2.wrapping_mul(2));
  FieldElement2625x4::new(&d2_fe, &d2_fe, &d2_fe_2, &d2_fe)
}

// ---------------------------------------------------------------------------
// Scalar multiplication
// ---------------------------------------------------------------------------

/// Convert an affine `CachedPoint` (from the basepoint table) to
/// Hamburg-scaled `CachedPointAvx2` format.
///
/// The affine table stores `(Y+X, Y−X, t2d)` where `t2d = 2d·T` and
/// `Z = 1`. Since `d = −d1/d2`, we have `t2d = (−2·d1/d2)·T`, so
/// `d2·t2d = −2·d1·T` — exactly the D-lane value the Hamburg format needs.
///
/// We pack `(Y−X, Y+X, 1, t2d)` and multiply by `(d2, d2, 2·d2, d2)`:
/// - A = d2·(Y−X), B = d2·(Y+X), C = 2·d2, D = −2·d1·T.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn cached_from_affine(cp: &CachedPoint, constants: &FieldElement2625x4) -> CachedPointAvx2 {
  let (y_plus_x, y_minus_x, t2d) = cp.components();
  let packed = FieldElement2625x4::new(y_minus_x, y_plus_x, &FieldElement::ONE, t2d);
  CachedPointAvx2(packed.mul(constants))
}

/// Add a signed digit from an affine cached table (basepoint table).
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn add_signed_cached_avx2(
  acc: ExtendedPointAvx2,
  table: &[CachedPoint; 8],
  digit: i8,
  affine_k: &FieldElement2625x4,
) -> ExtendedPointAvx2 {
  let index = usize::from(digit.unsigned_abs()).wrapping_sub(1);
  let Some(point) = table.get(index) else {
    return acc;
  };

  let cached = cached_from_affine(point, affine_k);
  if digit > 0 {
    acc.add_cached(&cached)
  } else {
    acc.add_cached(&cached.neg())
  }
}

/// Add a signed digit from a runtime CachedPointAvx2 table.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn add_signed_runtime_cached_avx2(
  acc: ExtendedPointAvx2,
  table: &[CachedPointAvx2; 8],
  digit: i8,
) -> ExtendedPointAvx2 {
  let index = usize::from(digit.unsigned_abs()).wrapping_sub(1);
  let Some(point) = table.get(index) else {
    return acc;
  };

  if digit > 0 {
    acc.add_cached(point)
  } else {
    acc.add_cached(&point.neg())
  }
}

/// Build a runtime table of `[1P, 2P, ..., 8P]` in Hamburg-scaled cached format.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn cached_multiples_avx2(point: &ExtendedPointAvx2) -> [CachedPointAvx2; 8] {
  let mut acc = *point;
  let point_cached = point.to_cached();
  let first = acc.to_cached();

  // Build [1P, 2P, ..., 8P] — initialize from first entry then accumulate.
  let mut out = [first; 8];
  for entry in out.iter_mut().skip(1) {
    acc = acc.add_cached(&point_cached);
    *entry = acc.to_cached();
  }

  out
}

/// Variable-base signed radix-16 scalar multiplication using AVX2 point ops.
///
/// Builds a runtime cached table and scans the scalar in signed radix-16 digits,
/// using AVX2 parallel doubling and addition throughout.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn scalar_mul_vartime_avx2(point: &ExtendedPoint, scalar_bytes: &[u8; 32]) -> ExtendedPoint {
  let digits = scalar::as_radix_16(scalar_bytes);
  let avx_point = ExtendedPointAvx2::from_extended(point);
  let table = cached_multiples_avx2(&avx_point);

  let mut acc = ExtendedPointAvx2::from_extended(&ExtendedPoint::identity());

  for digit in digits.iter().rev().copied() {
    acc = acc.double().double().double().double();
    if digit != 0 {
      acc = add_signed_runtime_cached_avx2(acc, &table, digit);
    }
  }

  acc.to_extended()
}

/// Fixed-base scalar multiplication for the Ed25519 basepoint using AVX2.
///
/// Uses the static radix-16 precomputed table, converting each affine
/// cached entry to Hamburg format on the fly.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn scalar_mul_basepoint_avx2(scalar_bytes: &[u8; 32]) -> ExtendedPoint {
  use super::point::BASEPOINT_RADIX16_TABLE;

  let digits = scalar::as_radix_16(scalar_bytes);
  let affine_k = hamburg_affine_constants();
  let mut acc = ExtendedPointAvx2::from_extended(&ExtendedPoint::identity());

  for (position, digit) in digits.iter().copied().enumerate() {
    if digit != 0
      && let Some(table) = BASEPOINT_RADIX16_TABLE.get(position)
    {
      acc = add_signed_cached_avx2(acc, table, digit, &affine_k);
    }
  }

  acc.to_extended()
}

/// Straus/Shamir interleaved double-scalar multiply: `[s]B + [h]A`.
///
/// Scans both scalars in lockstep over 64 signed radix-16 digits
/// (high → low), sharing all 256 doublings. Each digit position adds
/// from the flat basepoint table for `s` and from a runtime AVX2 cached
/// table for `h`.
///
/// Variable-time: branches on scalar nibble values. Safe for verification
/// where both `s` and `h` are public (derived from the message/signature).
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn straus_basepoint_vartime_avx2(s: &[u8; 32], h: &[u8; 32], a: &ExtendedPoint) -> ExtendedPoint {
  use super::point::BASEPOINT_RADIX16_TABLE;

  let s_digits = scalar::as_radix_16(s);
  let h_digits = scalar::as_radix_16(h);
  let affine_k = hamburg_affine_constants();

  let avx_a = ExtendedPointAvx2::from_extended(a);
  let a_table = cached_multiples_avx2(&avx_a);

  let mut acc = ExtendedPointAvx2::from_extended(&ExtendedPoint::identity());

  for (&sd, &hd) in s_digits.iter().zip(h_digits.iter()).rev() {
    acc = acc.double().double().double().double();
    if sd != 0 {
      acc = add_signed_cached_avx2(acc, &BASEPOINT_RADIX16_TABLE[0], sd, &affine_k);
    }
    if hd != 0 {
      acc = add_signed_runtime_cached_avx2(acc, &a_table, hd);
    }
  }

  acc.to_extended()
}

// ===========================================================================
// AVX-512 IFMA point operations (radix-2^51 via FieldElement51x4)
// ===========================================================================

/// Extended Edwards point in IFMA vectorized form.
///
/// Lanes: `(X, Y, Z, T)` as `(A, B, C, D)`.
#[derive(Clone, Copy)]
#[cfg(target_arch = "x86_64")]
pub(crate) struct ExtendedPointIfma(pub(crate) FieldElement51x4);

/// Cached point for efficient addition (Hamburg-scaled, IFMA format).
#[derive(Clone, Copy)]
#[cfg(target_arch = "x86_64")]
pub(crate) struct CachedPointIfma(pub(crate) FieldElement51x4);

#[cfg(target_arch = "x86_64")]
impl ExtendedPointIfma {
  /// Pack a scalar extended point into IFMA vectorized form.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn from_extended(p: &ExtendedPoint) -> Self {
    let (x, y, z, t) = p.components();
    Self(FieldElement51x4::new(x, y, z, t))
  }

  /// Unpack back to a scalar extended point.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn to_extended(self) -> ExtendedPoint {
    let [x, y, z, t] = self.0.split();
    ExtendedPoint::from_raw(x, y, z, t)
  }

  /// Convert to Hamburg-scaled cached format for addition.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX-512 IFMA + VL are available.
  #[inline]
  #[target_feature(enable = "avx2,avx512ifma,avx512vl")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn to_cached(self) -> CachedPointIfma {
    let ds = self.0.diff_sum();
    let prepared = self.0.blend(&ds, Lanes::AB).reduce();
    let constants = hamburg_constants_ifma();
    let scaled = prepared.mul(&constants);
    let negated = scaled.negate_lazy();
    CachedPointIfma(scaled.blend(&negated, Lanes::D))
  }

  /// Add a cached point to this extended point.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX-512 IFMA + VL are available.
  #[inline]
  #[target_feature(enable = "avx2,avx512ifma,avx512vl")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn add_cached(&self, other: &CachedPointIfma) -> Self {
    let ds = self.0.diff_sum();
    let tmp = self.0.blend(&ds, Lanes::AB).reduce();
    let product = tmp.mul(&other.0);
    let swapped = product.shuffle(Shuffle::ABDC);
    let ehfg = swapped.diff_sum().reduce();
    let t0 = ehfg.shuffle(Shuffle::ADDA);
    let t1 = ehfg.shuffle(Shuffle::CBCB);
    Self(t0.mul(&t1))
  }

  /// Double this point using HWCD'08 parallel doubling.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX-512 IFMA + VL are available.
  #[inline]
  #[target_feature(enable = "avx2,avx512ifma,avx512vl")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn double(&self) -> Self {
    let ab = self.0.shuffle(Shuffle::ABAB);
    let ba = ab.shuffle(Shuffle::BADC);
    let xy_sum = ab.add(&ba);
    let prepared = self.0.blend(&xy_sum, Lanes::D).reduce();
    let sq = prepared.square_and_negate_d();

    let zero = FieldElement51x4::zero();
    let s1 = sq.shuffle(Shuffle::AAAA);
    let s2 = sq.shuffle(Shuffle::BBBB);

    let sq_doubled = sq.add(&sq);
    let mut tmp = zero.blend(&sq_doubled, Lanes::C);
    tmp = tmp.blend(&sq, Lanes::D);
    tmp = tmp.add(&s1);
    let s2_in_ad = zero.blend(&s2, Lanes::AD);
    tmp = tmp.add(&s2_in_ad);
    let neg_s2 = s2.negate_lazy();
    let neg_s2_in_bc = zero.blend(&neg_s2, Lanes::BC);
    tmp = tmp.add(&neg_s2_in_bc).reduce();

    let t0 = tmp.shuffle(Shuffle::CACA);
    let t1 = tmp.shuffle(Shuffle::DBBD);
    Self(t0.mul(&t1))
  }
}

#[cfg(target_arch = "x86_64")]
impl CachedPointIfma {
  /// Negate the cached point (for subtraction).
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn neg(&self) -> Self {
    let swapped = self.0.shuffle(Shuffle::BACD);
    let negated = swapped.negate_lazy();
    Self(swapped.blend(&negated, Lanes::D))
  }
}

/// Hamburg scaling constants for projective cached (IFMA): `(d2, d2, 2·d2, 2·d1)`.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn hamburg_constants_ifma() -> FieldElement51x4 {
  let d2_fe = FieldElement::from_small(D2);
  let d2_fe_2 = FieldElement::from_small(D2.wrapping_mul(2));
  let d1_fe_2 = FieldElement::from_small(D1.wrapping_mul(2));
  FieldElement51x4::new(&d2_fe, &d2_fe, &d2_fe_2, &d1_fe_2)
}

/// Hamburg scaling constants for affine cached (IFMA): `(d2, d2, 2·d2, d2)`.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn hamburg_affine_constants_ifma() -> FieldElement51x4 {
  let d2_fe = FieldElement::from_small(D2);
  let d2_fe_2 = FieldElement::from_small(D2.wrapping_mul(2));
  FieldElement51x4::new(&d2_fe, &d2_fe, &d2_fe_2, &d2_fe)
}

/// Convert an affine `CachedPoint` to Hamburg-scaled IFMA cached format.
///
/// # Safety
///
/// Caller must ensure AVX-512 IFMA + VL are available.
#[inline]
#[target_feature(enable = "avx2,avx512ifma,avx512vl")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn cached_from_affine_ifma(cp: &CachedPoint, constants: &FieldElement51x4) -> CachedPointIfma {
  let (y_plus_x, y_minus_x, t2d) = cp.components();
  let packed = FieldElement51x4::new(y_minus_x, y_plus_x, &FieldElement::ONE, t2d);
  CachedPointIfma(packed.mul(constants))
}

/// Add a signed digit from an affine cached table (IFMA).
///
/// # Safety
///
/// Caller must ensure AVX-512 IFMA + VL are available.
#[inline]
#[target_feature(enable = "avx2,avx512ifma,avx512vl")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn add_signed_cached_ifma(
  acc: ExtendedPointIfma,
  table: &[CachedPoint; 8],
  digit: i8,
  affine_k: &FieldElement51x4,
) -> ExtendedPointIfma {
  let index = usize::from(digit.unsigned_abs()).wrapping_sub(1);
  let Some(point) = table.get(index) else {
    return acc;
  };
  let cached = cached_from_affine_ifma(point, affine_k);
  if digit > 0 {
    acc.add_cached(&cached)
  } else {
    acc.add_cached(&cached.neg())
  }
}

/// Add a signed digit from a runtime CachedPointIfma table.
///
/// # Safety
///
/// Caller must ensure AVX-512 IFMA + VL are available.
#[inline]
#[target_feature(enable = "avx2,avx512ifma,avx512vl")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn add_signed_runtime_cached_ifma(
  acc: ExtendedPointIfma,
  table: &[CachedPointIfma; 8],
  digit: i8,
) -> ExtendedPointIfma {
  let index = usize::from(digit.unsigned_abs()).wrapping_sub(1);
  let Some(point) = table.get(index) else {
    return acc;
  };
  if digit > 0 {
    acc.add_cached(point)
  } else {
    acc.add_cached(&point.neg())
  }
}

/// Build a runtime table of `[1P, 2P, ..., 8P]` in IFMA Hamburg-scaled cached format.
///
/// # Safety
///
/// Caller must ensure AVX-512 IFMA + VL are available.
#[target_feature(enable = "avx2,avx512ifma,avx512vl")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn cached_multiples_ifma(point: &ExtendedPointIfma) -> [CachedPointIfma; 8] {
  let mut acc = *point;
  let point_cached = point.to_cached();
  let first = acc.to_cached();

  let mut out = [first; 8];
  for entry in out.iter_mut().skip(1) {
    acc = acc.add_cached(&point_cached);
    *entry = acc.to_cached();
  }
  out
}

/// Variable-base scalar multiplication using AVX-512 IFMA.
///
/// # Safety
///
/// Caller must ensure AVX-512 IFMA + VL are available.
#[target_feature(enable = "avx2,avx512ifma,avx512vl")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn scalar_mul_vartime_ifma(point: &ExtendedPoint, scalar_bytes: &[u8; 32]) -> ExtendedPoint {
  let digits = scalar::as_radix_16(scalar_bytes);
  let ifma_point = ExtendedPointIfma::from_extended(point);
  let table = cached_multiples_ifma(&ifma_point);
  let mut acc = ExtendedPointIfma::from_extended(&ExtendedPoint::identity());
  for digit in digits.iter().rev().copied() {
    acc = acc.double().double().double().double();
    if digit != 0 {
      acc = add_signed_runtime_cached_ifma(acc, &table, digit);
    }
  }
  acc.to_extended()
}

/// Fixed-base scalar multiplication for the Ed25519 basepoint using IFMA.
///
/// # Safety
///
/// Caller must ensure AVX-512 IFMA + VL are available.
#[target_feature(enable = "avx2,avx512ifma,avx512vl")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn scalar_mul_basepoint_ifma(scalar_bytes: &[u8; 32]) -> ExtendedPoint {
  use super::point::BASEPOINT_RADIX16_TABLE;

  let digits = scalar::as_radix_16(scalar_bytes);
  let affine_k = hamburg_affine_constants_ifma();
  let mut acc = ExtendedPointIfma::from_extended(&ExtendedPoint::identity());

  for (position, digit) in digits.iter().copied().enumerate() {
    if digit != 0
      && let Some(table) = BASEPOINT_RADIX16_TABLE.get(position)
    {
      acc = add_signed_cached_ifma(acc, table, digit, &affine_k);
    }
  }

  acc.to_extended()
}

/// Straus/Shamir interleaved double-scalar multiply using IFMA.
///
/// # Safety
///
/// Caller must ensure AVX-512 IFMA + VL are available.
#[target_feature(enable = "avx2,avx512ifma,avx512vl")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn straus_basepoint_vartime_ifma(s: &[u8; 32], h: &[u8; 32], a: &ExtendedPoint) -> ExtendedPoint {
  use super::point::BASEPOINT_RADIX16_TABLE;

  let s_digits = scalar::as_radix_16(s);
  let h_digits = scalar::as_radix_16(h);
  let affine_k = hamburg_affine_constants_ifma();

  let ifma_a = ExtendedPointIfma::from_extended(a);
  let a_table = cached_multiples_ifma(&ifma_a);

  let mut acc = ExtendedPointIfma::from_extended(&ExtendedPoint::identity());

  for (&sd, &hd) in s_digits.iter().zip(h_digits.iter()).rev() {
    acc = acc.double().double().double().double();
    if sd != 0 {
      acc = add_signed_cached_ifma(acc, &BASEPOINT_RADIX16_TABLE[0], sd, &affine_k);
    }
    if hd != 0 {
      acc = add_signed_runtime_cached_ifma(acc, &a_table, hd);
    }
  }

  acc.to_extended()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
  use super::*;
  use crate::auth::ed25519::point::ExtendedPoint;

  fn basepoint() -> ExtendedPoint {
    ExtendedPoint::basepoint()
  }

  fn decode_hex_32(hex: &str) -> [u8; 32] {
    let bytes = hex.as_bytes();
    let mut out = [0u8; 32];
    for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(2)) {
      *dst = hex_val(chunk[0]) << 4 | hex_val(chunk[1]);
    }
    out
  }

  fn hex_val(b: u8) -> u8 {
    match b {
      b'0'..=b'9' => b - b'0',
      b'a'..=b'f' => b - b'a' + 10,
      _ => panic!("invalid hex"),
    }
  }

  #[test]
  fn pack_unpack_roundtrip() {
    if !is_x86_feature_detected!("avx2") {
      return;
    }
    let bp = basepoint();

    // SAFETY: AVX2 availability checked by the runtime guard above.
    unsafe {
      let avx = ExtendedPointAvx2::from_extended(&bp);
      let back = avx.to_extended();

      assert!(
        bp.equals_projective(&back),
        "pack/unpack roundtrip should preserve point"
      );
    }
  }

  #[test]
  fn double_matches_scalar() {
    if !is_x86_feature_detected!("avx2") {
      return;
    }
    let bp = basepoint();
    let scalar_doubled = bp.double();

    // SAFETY: AVX2 availability checked by the runtime guard above.
    unsafe {
      let avx = ExtendedPointAvx2::from_extended(&bp);
      let avx_doubled = avx.double();
      let result = avx_doubled.to_extended();

      assert!(
        scalar_doubled.equals_projective(&result),
        "AVX2 double should match scalar double"
      );
    }
  }

  #[test]
  fn double_chain_matches_scalar() {
    if !is_x86_feature_detected!("avx2") {
      return;
    }
    let bp = basepoint();
    // 8B = cofactor mul
    let scalar_8b = bp.double().double().double();

    // SAFETY: AVX2 availability checked by the runtime guard above.
    unsafe {
      let avx = ExtendedPointAvx2::from_extended(&bp);
      let avx_8b = avx.double().double().double();
      let result = avx_8b.to_extended();

      assert!(
        scalar_8b.equals_projective(&result),
        "AVX2 triple-double (8B) should match scalar"
      );
    }
  }

  #[test]
  fn add_cached_matches_scalar_add() {
    if !is_x86_feature_detected!("avx2") {
      return;
    }
    let bp = basepoint();
    let bp2 = bp.double();
    // Scalar: B + 2B = 3B
    let scalar_3b = bp.add(&bp2);

    // SAFETY: AVX2 availability checked by the runtime guard above.
    unsafe {
      let avx_bp = ExtendedPointAvx2::from_extended(&bp);
      let avx_bp2 = ExtendedPointAvx2::from_extended(&bp2);
      let cached_bp2 = avx_bp2.to_cached();
      let avx_3b = avx_bp.add_cached(&cached_bp2);
      let result = avx_3b.to_extended();

      assert!(
        scalar_3b.equals_projective(&result),
        "AVX2 add should match scalar add (B + 2B = 3B)"
      );
    }
  }

  #[test]
  fn add_cached_neg_is_subtraction() {
    if !is_x86_feature_detected!("avx2") {
      return;
    }
    let bp = basepoint();
    let bp2 = bp.double();

    // SAFETY: AVX2 availability checked by the runtime guard above.
    unsafe {
      // 2B + (−B) should equal B
      let avx_bp2 = ExtendedPointAvx2::from_extended(&bp2);
      let avx_bp = ExtendedPointAvx2::from_extended(&bp);
      let cached_bp_neg = avx_bp.to_cached().neg();
      let result = avx_bp2.add_cached(&cached_bp_neg).to_extended();

      assert!(bp.equals_projective(&result), "2B + (−B) should equal B");
    }
  }

  #[test]
  fn add_then_double_matches_scalar() {
    if !is_x86_feature_detected!("avx2") {
      return;
    }
    let bp = basepoint();
    // Scalar: 2*(B + B) = 2*(2B) = 4B
    let scalar_4b = bp.add(&bp).double();

    // SAFETY: AVX2 availability checked by the runtime guard above.
    unsafe {
      let avx_bp = ExtendedPointAvx2::from_extended(&bp);
      let cached_bp = avx_bp.to_cached();
      let avx_2b = avx_bp.add_cached(&cached_bp);
      let avx_4b = avx_2b.double();
      let result = avx_4b.to_extended();

      assert!(
        scalar_4b.equals_projective(&result),
        "AVX2 add+double should match scalar (4B)"
      );
    }
  }

  #[test]
  fn identity_addition() {
    if !is_x86_feature_detected!("avx2") {
      return;
    }
    let bp = basepoint();
    let identity = ExtendedPoint::identity();

    // SAFETY: AVX2 availability checked by the runtime guard above.
    unsafe {
      let avx_bp = ExtendedPointAvx2::from_extended(&bp);
      let avx_id = ExtendedPointAvx2::from_extended(&identity);
      let cached_id = avx_id.to_cached();
      let result = avx_bp.add_cached(&cached_id).to_extended();

      assert!(bp.equals_projective(&result), "B + identity should equal B");
    }
  }

  // -----------------------------------------------------------------------
  // Phase 3: Scalar multiplication tests
  // -----------------------------------------------------------------------

  #[test]
  fn scalar_mul_vartime_matches_scalar() {
    if !is_x86_feature_detected!("avx2") {
      return;
    }
    let bp = basepoint();
    let mut scalar = [0u8; 32];
    scalar[0] = 42;

    let scalar_result = bp.scalar_mul_vartime(&scalar);

    // SAFETY: AVX2 availability checked by the runtime guard above.
    unsafe {
      let avx_result = scalar_mul_vartime_avx2(&bp, &scalar);
      assert!(
        scalar_result.equals_projective(&avx_result),
        "AVX2 vartime scalar mul should match scalar"
      );
    }
  }

  #[test]
  fn scalar_mul_basepoint_matches_scalar() {
    if !is_x86_feature_detected!("avx2") {
      return;
    }
    let mut scalar = [0u8; 32];
    scalar[0] = 1;
    let scalar_result = ExtendedPoint::scalar_mul_basepoint(&scalar);

    // SAFETY: AVX2 availability checked by the runtime guard above.
    unsafe {
      let avx_result = scalar_mul_basepoint_avx2(&scalar);
      assert!(
        scalar_result.equals_projective(&avx_result),
        "AVX2 basepoint mul [1]B should match scalar"
      );
    }
  }

  #[test]
  fn scalar_mul_basepoint_rfc8032_vector1() {
    if !is_x86_feature_detected!("avx2") {
      return;
    }
    use crate::auth::ed25519::{Ed25519SecretKey, hash::ExpandedSecret};

    let secret = Ed25519SecretKey::from_bytes(decode_hex_32(
      "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
    ));
    let expanded = ExpandedSecret::from_secret_key(&secret);
    let expected = decode_hex_32("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");

    // SAFETY: AVX2 availability checked by the runtime guard above.
    unsafe {
      let avx_pub = scalar_mul_basepoint_avx2(expanded.scalar_bytes());
      assert_eq!(
        avx_pub.to_bytes(),
        Some(expected),
        "AVX2 basepoint mul should match RFC 8032 vector 1"
      );
    }
  }

  #[test]
  fn straus_matches_scalar() {
    if !is_x86_feature_detected!("avx2") {
      return;
    }
    let bp = basepoint();
    let a = bp.double().double(); // 4B as the variable-base point

    let mut s = [0u8; 32];
    s[0] = 7;
    let mut h = [0u8; 32];
    h[0] = 13;

    let scalar_result = ExtendedPoint::straus_basepoint_vartime(&s, &h, &a);

    // SAFETY: AVX2 availability checked by the runtime guard above.
    unsafe {
      let avx_result = straus_basepoint_vartime_avx2(&s, &h, &a);
      assert!(
        scalar_result.equals_projective(&avx_result),
        "AVX2 Straus should match scalar"
      );
    }
  }

  #[test]
  fn straus_matches_scalar_large_scalars() {
    if !is_x86_feature_detected!("avx2") {
      return;
    }
    // Reproduce the actual verify path: full 256-bit scalars, realistic point.
    use crate::{
      auth::ed25519::{Ed25519Keypair, Ed25519SecretKey, hash::ExpandedSecret},
      hashes::crypto::Sha512,
      traits::Digest,
    };

    let secret = Ed25519SecretKey::from_bytes([13u8; 32]);
    let keypair = Ed25519Keypair::from_secret_key(secret);
    let public = keypair.public_key();
    let sig = keypair.sign(b"test message for straus");

    // Extract R, s, compute challenge h — same as verify()
    let sig_bytes = sig.as_bytes();
    let mut r_bytes = [0u8; 32];
    let mut s_bytes = [0u8; 32];
    r_bytes.copy_from_slice(&sig_bytes[..32]);
    s_bytes.copy_from_slice(&sig_bytes[32..]);

    let r_point = ExtendedPoint::from_bytes(&r_bytes).unwrap();
    let a_point = ExtendedPoint::from_bytes(public.as_bytes()).unwrap();
    let s_scalar = crate::auth::ed25519::scalar::from_canonical_bytes(&s_bytes).unwrap();

    let mut hasher = Sha512::new();
    hasher.update(&r_bytes);
    hasher.update(public.as_bytes());
    hasher.update(b"test message for straus");
    let challenge = crate::auth::ed25519::scalar::reduce_bytes_mod_order(&hasher.finalize());
    let neg_challenge = crate::auth::ed25519::scalar::negate_mod(&challenge);
    let neg_challenge_bytes = crate::auth::ed25519::scalar::to_bytes(&neg_challenge);
    let s_canonical = crate::auth::ed25519::scalar::to_bytes(&s_scalar);

    let scalar_result = ExtendedPoint::straus_basepoint_vartime(&s_canonical, &neg_challenge_bytes, &a_point);

    // SAFETY: AVX2 availability checked by the runtime guard above.
    unsafe {
      let avx_result = straus_basepoint_vartime_avx2(&s_canonical, &neg_challenge_bytes, &a_point);
      assert!(
        scalar_result.equals_projective(&avx_result),
        "AVX2 Straus with large scalars should match scalar"
      );

      // Also verify the full equation: [s]B + [-h]A == [8]R (after cofactor)
      let combined = avx_result.mul_by_cofactor();
      let expected = r_point.mul_by_cofactor();
      assert!(
        combined.equals_projective(&expected),
        "AVX2 Straus verify equation should hold"
      );
    }
  }

  // -----------------------------------------------------------------------
  // Phase 2: Point operation tests
  // -----------------------------------------------------------------------

  #[test]
  fn double_of_identity_is_identity() {
    if !is_x86_feature_detected!("avx2") {
      return;
    }
    let identity = ExtendedPoint::identity();

    // SAFETY: AVX2 availability checked by the runtime guard above.
    unsafe {
      let avx_id = ExtendedPointAvx2::from_extended(&identity);
      let result = avx_id.double().to_extended();

      assert!(
        identity.equals_projective(&result),
        "double(identity) should be identity"
      );
    }
  }

  // -----------------------------------------------------------------------
  // IFMA point operation tests
  // -----------------------------------------------------------------------

  #[test]
  fn ifma_pack_unpack_roundtrip() {
    if !is_x86_feature_detected!("avx512ifma") {
      return;
    }
    let bp = basepoint();

    // SAFETY: AVX-512 IFMA availability checked above.
    unsafe {
      let ifma = ExtendedPointIfma::from_extended(&bp);
      let back = ifma.to_extended();
      assert!(bp.equals_projective(&back), "IFMA pack/unpack roundtrip");
    }
  }

  #[test]
  fn ifma_double_matches_scalar() {
    if !is_x86_feature_detected!("avx512ifma") {
      return;
    }
    let bp = basepoint();
    let scalar_doubled = bp.double();

    // SAFETY: AVX-512 IFMA availability checked above.
    unsafe {
      let ifma = ExtendedPointIfma::from_extended(&bp);
      let result = ifma.double().to_extended();
      assert!(
        scalar_doubled.equals_projective(&result),
        "IFMA double should match scalar"
      );
    }
  }

  #[test]
  fn ifma_double_chain_matches_scalar() {
    if !is_x86_feature_detected!("avx512ifma") {
      return;
    }
    let bp = basepoint();
    let scalar_8b = bp.double().double().double();

    // SAFETY: AVX-512 IFMA availability checked above.
    unsafe {
      let ifma = ExtendedPointIfma::from_extended(&bp);
      let result = ifma.double().double().double().to_extended();
      assert!(
        scalar_8b.equals_projective(&result),
        "IFMA triple-double (8B) should match scalar"
      );
    }
  }

  #[test]
  fn ifma_add_cached_matches_scalar() {
    if !is_x86_feature_detected!("avx512ifma") {
      return;
    }
    let bp = basepoint();
    let bp2 = bp.double();
    let scalar_3b = bp.add(&bp2);

    // SAFETY: AVX-512 IFMA availability checked above.
    unsafe {
      let ifma_bp = ExtendedPointIfma::from_extended(&bp);
      let ifma_bp2 = ExtendedPointIfma::from_extended(&bp2);
      let cached = ifma_bp2.to_cached();
      let result = ifma_bp.add_cached(&cached).to_extended();
      assert!(
        scalar_3b.equals_projective(&result),
        "IFMA add should match scalar (B + 2B = 3B)"
      );
    }
  }

  #[test]
  fn ifma_add_then_double_matches_scalar() {
    if !is_x86_feature_detected!("avx512ifma") {
      return;
    }
    let bp = basepoint();
    let scalar_4b = bp.add(&bp).double();

    // SAFETY: AVX-512 IFMA availability checked above.
    unsafe {
      let ifma_bp = ExtendedPointIfma::from_extended(&bp);
      let cached = ifma_bp.to_cached();
      let ifma_2b = ifma_bp.add_cached(&cached);
      let result = ifma_2b.double().to_extended();
      assert!(
        scalar_4b.equals_projective(&result),
        "IFMA add+double should match scalar (4B)"
      );
    }
  }

  #[test]
  fn ifma_neg_cached_is_subtraction() {
    if !is_x86_feature_detected!("avx512ifma") {
      return;
    }
    let bp = basepoint();
    let bp2 = bp.double();

    // SAFETY: AVX-512 IFMA availability checked above.
    unsafe {
      let ifma_bp2 = ExtendedPointIfma::from_extended(&bp2);
      let ifma_bp = ExtendedPointIfma::from_extended(&bp);
      let neg_cached = ifma_bp.to_cached().neg();
      let result = ifma_bp2.add_cached(&neg_cached).to_extended();
      assert!(bp.equals_projective(&result), "IFMA 2B + (-B) should equal B");
    }
  }

  #[test]
  fn ifma_scalar_mul_vartime_matches_scalar() {
    if !is_x86_feature_detected!("avx512ifma") {
      return;
    }
    let bp = basepoint();
    let mut scalar = [0u8; 32];
    scalar[0] = 42;
    let scalar_result = bp.scalar_mul_vartime(&scalar);

    // SAFETY: AVX-512 IFMA availability checked above.
    unsafe {
      let result = scalar_mul_vartime_ifma(&bp, &scalar);
      assert!(
        scalar_result.equals_projective(&result),
        "IFMA vartime scalar mul should match scalar"
      );
    }
  }

  #[test]
  fn ifma_scalar_mul_basepoint_matches_scalar() {
    if !is_x86_feature_detected!("avx512ifma") {
      return;
    }
    let mut scalar = [0u8; 32];
    scalar[0] = 1;
    let scalar_result = ExtendedPoint::scalar_mul_basepoint(&scalar);

    // SAFETY: AVX-512 IFMA availability checked above.
    unsafe {
      let result = scalar_mul_basepoint_ifma(&scalar);
      assert!(
        scalar_result.equals_projective(&result),
        "IFMA basepoint mul [1]B should match scalar"
      );
    }
  }
}
