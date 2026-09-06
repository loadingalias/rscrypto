//! Portable P-256 arithmetic and encoding authority shared by ECDSA and ECDH.

#[cfg(feature = "p256-ecdh")]
use core::cmp::Ordering;

#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
use crate::traits::ct;

#[cfg(feature = "p256-ecdh")]
const FIELD_BYTES: usize = 32;
#[cfg(feature = "p256-ecdh")]
const SEC1_BYTES: usize = 65;
#[cfg(test)]
const COMB_WINDOW_BITS: usize = 4;
#[cfg(test)]
const COMB_WINDOW_SIZE: usize = 1usize << COMB_WINDOW_BITS;
#[cfg(test)]
const COMB_WINDOW_ROWS: usize = 256 / COMB_WINDOW_BITS;
#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
const SIGNED_WINDOW_SIZE: usize = 16;
#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
const SIGNED_WINDOW_DIGITS: usize = 52;
#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
const SIGNED_FIXED_BASE_COLUMNS: usize = 4;
#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
const SIGNED_FIXED_BASE_ROWS: usize = 13;

const FIELD_MODULUS: Uint = Uint([
  0xffff_ffff_ffff_ffff,
  0x0000_0000_ffff_ffff,
  0x0000_0000_0000_0000,
  0xffff_ffff_0000_0001,
]);
#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
const FIELD_MODULUS_MINUS_TWO: Uint = Uint([
  0xffff_ffff_ffff_fffd,
  0x0000_0000_ffff_ffff,
  0x0000_0000_0000_0000,
  0xffff_ffff_0000_0001,
]);
#[cfg(feature = "p256-ecdh")]
const SCALAR_MODULUS: Uint = Uint([
  0xf3b9_cac2_fc63_2551,
  0xbce6_faad_a717_9e84,
  0xffff_ffff_ffff_ffff,
  0xffff_ffff_0000_0000,
]);
#[cfg(any(
  feature = "p256-ecdh",
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
const CURVE_B_MONTGOMERY: Uint = Uint([
  0xd89c_df62_29c4_bddf,
  0xacf0_05cd_7884_3090,
  0xe5a2_20ab_f721_2ed6,
  0xdc30_061d_0487_4834,
]);
#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
const GENERATOR_X: Uint = Uint([
  0xf4a1_3945_d898_c296,
  0x7703_7d81_2deb_33a0,
  0xf8bc_e6e5_63a4_40f2,
  0x6b17_d1f2_e12c_4247,
]);
#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
const GENERATOR_Y: Uint = Uint([
  0xcbb6_4068_37bf_51f5,
  0x2bce_3357_6b31_5ece,
  0x8ee7_eb4a_7c0f_9e16,
  0x4fe3_42e2_fe1a_7f9b,
]);
const FIELD_R2: Uint = Uint([
  0x0000_0000_0000_0003,
  0xffff_fffb_ffff_ffff,
  0xffff_ffff_ffff_fffe,
  0x0000_0004_ffff_fffd,
]);

#[derive(Clone, Copy, PartialEq, Eq)]
struct Uint([u64; 4]);

impl Uint {
  #[cfg(any(
    test,
    feature = "portable-only",
    miri,
    not(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    ))
  ))]
  const ZERO: Self = Self([0; 4]);
  const ONE: Self = Self([1, 0, 0, 0]);

  #[cfg(feature = "p256-ecdh")]
  fn from_be_slice(bytes: &[u8]) -> Option<Self> {
    if bytes.len() != FIELD_BYTES {
      return None;
    }
    let mut limbs = [0u64; 4];
    for (limb, chunk) in limbs.iter_mut().zip(bytes.rchunks_exact(8)) {
      let mut word = [0u8; 8];
      word.copy_from_slice(chunk);
      *limb = u64::from_be_bytes(word);
    }
    Some(Self(limbs))
  }

  #[cfg(feature = "p256-ecdh")]
  fn write_be(self, out: &mut [u8; FIELD_BYTES]) {
    for (chunk, limb) in out.rchunks_exact_mut(8).zip(self.0) {
      chunk.copy_from_slice(&limb.to_be_bytes());
    }
  }

  #[cfg(feature = "p256-ecdh")]
  fn cmp(&self, other: &Self) -> Ordering {
    for (&left, &right) in self.0.iter().zip(other.0.iter()).rev() {
      if left < right {
        return Ordering::Less;
      }
      if left > right {
        return Ordering::Greater;
      }
    }
    Ordering::Equal
  }

  fn add_raw(self, rhs: Self) -> (Self, u64) {
    let mut out = [0u64; 4];
    let mut carry = 0u64;
    for ((dst, left), right) in out.iter_mut().zip(self.0).zip(rhs.0) {
      (*dst, carry) = adc_limb(left, right, carry);
    }
    (Self(out), carry)
  }

  fn sub_raw(self, rhs: Self) -> (Self, u64) {
    let mut out = [0u64; 4];
    let mut borrow = 0u64;
    for ((dst, left), right) in out.iter_mut().zip(self.0).zip(rhs.0) {
      (*dst, borrow) = sbb_limb(left, right, borrow);
    }
    (Self(out), borrow)
  }

  fn add_mod(self, rhs: Self) -> Self {
    let (sum, carry) = self.add_raw(rhs);
    let (reduced, borrow) = sum.sub_raw(FIELD_MODULUS);
    Self::select(sum, reduced, mask_nonzero(carry) | mask_zero(borrow))
  }

  fn sub_mod(self, rhs: Self) -> Self {
    let (difference, borrow) = self.sub_raw(rhs);
    let added = difference.add_raw(FIELD_MODULUS).0;
    Self::select(difference, added, mask_nonzero(borrow))
  }

  #[cfg(any(
    test,
    feature = "portable-only",
    miri,
    not(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    ))
  ))]
  fn bit(self, bit: usize) -> bool {
    let limb = bit / 64;
    let shift = bit % 64;
    self.0.get(limb).copied().unwrap_or(0) & (1u64 << shift) != 0
  }

  #[cfg(any(
    test,
    feature = "portable-only",
    miri,
    not(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    ))
  ))]
  fn bit_mask(self, bit: usize) -> u64 {
    let limb = bit / 64;
    let shift = bit % 64;
    mask_nonzero((self.0.get(limb).copied().unwrap_or(0) >> shift) & 1)
  }

  fn zero_mask(self) -> u64 {
    mask_zero(self.0.into_iter().fold(0u64, |acc, limb| acc | limb))
  }

  fn select(left: Self, right: Self, mask: u64) -> Self {
    #[cfg(target_arch = "s390x")]
    let mask = core::hint::black_box(mask);
    let mut out = [0u64; 4];
    for ((dst, left), right) in out.iter_mut().zip(left.0).zip(right.0) {
      *dst = left ^ (mask & (left ^ right));
    }
    Self(out)
  }

  #[cfg(any(
    test,
    feature = "portable-only",
    miri,
    not(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    ))
  ))]
  fn zeroize_no_fence(&mut self) {
    ct::zeroize_words_no_fence(&mut self.0);
  }
}

#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
struct Scalar(Uint);

#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
impl Scalar {
  #[cfg(feature = "p256-ecdh")]
  fn from_bytes(bytes: &[u8; FIELD_BYTES]) -> Self {
    Self(Uint::from_be_slice(bytes).unwrap_or(Uint::ZERO))
  }

  fn signed_radix_32(&self) -> [u8; SIGNED_WINDOW_DIGITS] {
    let mut digits = [0u8; SIGNED_WINDOW_DIGITS];
    let mut carry = 0u32;
    for (index, digit) in digits.iter_mut().enumerate() {
      let mut value = 0u32;
      for offset in 0..5 {
        let bit = index.strict_mul(5).strict_add(offset);
        let bit = (self.0.bit_mask(bit) & 1).to_le_bytes()[0];
        value |= u32::from(bit) << offset;
      }
      value = value.strict_add(carry);
      let negative = 16u32.wrapping_sub(value) >> 31;
      *digit = value.wrapping_sub(negative << 5).to_le_bytes()[0];
      carry = negative;
    }
    digits
  }
}

#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
impl Drop for Scalar {
  fn drop(&mut self) {
    ct::zeroize_words(&mut self.0.0);
  }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct FieldElement(Uint);

impl FieldElement {
  fn from_uint(value: Uint) -> Self {
    Self(montgomery_mul(value, FIELD_R2))
  }

  const fn from_montgomery(value: Uint) -> Self {
    Self(value)
  }

  fn to_uint(self) -> Uint {
    montgomery_mul(self.0, Uint::ONE)
  }

  #[cfg(any(
    test,
    feature = "portable-only",
    miri,
    not(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    ))
  ))]
  const fn zero() -> Self {
    Self::from_montgomery(Uint::ZERO)
  }

  #[cfg(any(
    test,
    feature = "portable-only",
    miri,
    not(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    ))
  ))]
  fn one() -> Self {
    Self::from_uint(Uint::ONE)
  }

  fn add(self, rhs: Self) -> Self {
    Self::from_montgomery(self.0.add_mod(rhs.0))
  }

  fn sub(self, rhs: Self) -> Self {
    Self::from_montgomery(self.0.sub_mod(rhs.0))
  }

  #[cfg(any(
    test,
    feature = "portable-only",
    miri,
    not(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    ))
  ))]
  fn mul(self, rhs: Self) -> Self {
    Self::from_montgomery(montgomery_mul(self.0, rhs.0))
  }

  #[cfg(any(
    test,
    feature = "portable-only",
    miri,
    not(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    ))
  ))]
  fn square(self) -> Self {
    self.mul(self)
  }

  fn double(self) -> Self {
    self.add(self)
  }

  fn triple(self) -> Self {
    self.double().add(self)
  }

  #[cfg(any(
    test,
    feature = "portable-only",
    miri,
    not(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    ))
  ))]
  fn invert(self) -> Self {
    let mut acc = Self::one();
    for bit in (0..256).rev() {
      acc = acc.square();
      // FIELD_MODULUS_MINUS_TWO is public and fixed for every operation.
      if FIELD_MODULUS_MINUS_TWO.bit(bit) {
        acc = acc.mul(self);
      }
    }
    acc
  }

  #[cfg(any(
    test,
    feature = "portable-only",
    miri,
    not(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    ))
  ))]
  fn select(left: Self, right: Self, mask: u64) -> Self {
    Self::from_montgomery(Uint::select(left.0, right.0, mask))
  }
}

#[derive(Clone, Copy)]
struct Affine {
  x: FieldElement,
  y: FieldElement,
}

impl Affine {
  #[cfg(any(
    test,
    feature = "portable-only",
    miri,
    not(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    ))
  ))]
  fn generator() -> Self {
    Self {
      x: FieldElement::from_uint(GENERATOR_X),
      y: FieldElement::from_uint(GENERATOR_Y),
    }
  }

  #[cfg(all(
    feature = "p256-ecdh",
    any(
      test,
      feature = "portable-only",
      miri,
      not(any(
        all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
        all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
      ))
    )
  ))]
  fn is_on_curve(self) -> bool {
    let lhs = self.y.square();
    let rhs = self
      .x
      .square()
      .mul(self.x)
      .sub(self.x.triple())
      .add(FieldElement::from_montgomery(CURVE_B_MONTGOMERY));
    lhs == rhs
  }

  #[cfg(feature = "p256-ecdh")]
  fn encode_sec1(self) -> [u8; SEC1_BYTES] {
    let mut bytes = [0u8; SEC1_BYTES];
    bytes[0] = 0x04;
    let (x, y) = bytes[1..].split_at_mut(FIELD_BYTES);
    let mut x_bytes = [0u8; FIELD_BYTES];
    let mut y_bytes = [0u8; FIELD_BYTES];
    self.x.to_uint().write_be(&mut x_bytes);
    self.y.to_uint().write_be(&mut y_bytes);
    x.copy_from_slice(&x_bytes);
    y.copy_from_slice(&y_bytes);
    bytes
  }

  #[cfg(test)]
  fn select(table: &[Self; COMB_WINDOW_SIZE], digit: usize) -> Self {
    let mut selected = table[0];
    for (index, &candidate) in table.iter().enumerate() {
      let mask = mask_equal_usize(digit, index);
      selected.x = FieldElement::select(selected.x, candidate.x, mask);
      selected.y = FieldElement::select(selected.y, candidate.y, mask);
    }
    selected
  }

  #[cfg(any(
    test,
    feature = "portable-only",
    miri,
    not(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    ))
  ))]
  fn select_signed(table: &[Self; SIGNED_WINDOW_SIZE], digit: u8) -> (Self, u64) {
    let sign = 0u8.wrapping_sub(digit >> 7);
    let magnitude = usize::from((digit ^ sign).wrapping_sub(sign));
    let mut selected = table[0];
    for (index, &candidate) in table.iter().enumerate() {
      let mask = mask_equal_usize(magnitude, index.strict_add(1));
      selected.x = FieldElement::select(selected.x, candidate.x, mask);
      selected.y = FieldElement::select(selected.y, candidate.y, mask);
    }
    let negated_y = FieldElement::zero().sub(selected.y);
    selected.y = FieldElement::select(selected.y, negated_y, 0u64.wrapping_sub(u64::from(sign & 1)));
    (selected, mask_equal_usize(magnitude, 0))
  }
}

#[derive(Clone, Copy)]
#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
struct Projective {
  x: FieldElement,
  y: FieldElement,
  z: FieldElement,
  infinity_mask: u64,
}

#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
impl Projective {
  fn infinity() -> Self {
    Self {
      x: FieldElement::zero(),
      y: FieldElement::one(),
      z: FieldElement::zero(),
      infinity_mask: u64::MAX,
    }
  }

  fn from_affine(point: Affine) -> Self {
    Self {
      x: point.x,
      y: point.y,
      z: FieldElement::one(),
      infinity_mask: 0,
    }
  }

  fn select(left: Self, right: Self, mask: u64) -> Self {
    Self {
      x: FieldElement::select(left.x, right.x, mask),
      y: FieldElement::select(left.y, right.y, mask),
      z: FieldElement::select(left.z, right.z, mask),
      infinity_mask: left.infinity_mask ^ (mask & (left.infinity_mask ^ right.infinity_mask)),
    }
  }

  #[cfg(feature = "p256-ecdh")]
  fn select_signed(table: &[Self; SIGNED_WINDOW_SIZE], digit: u8) -> Self {
    let sign = 0u8.wrapping_sub(digit >> 7);
    let magnitude = usize::from((digit ^ sign).wrapping_sub(sign));
    let mut selected = Self::infinity();
    for (index, &candidate) in table.iter().enumerate() {
      selected = Self::select(selected, candidate, mask_equal_usize(magnitude, index.strict_add(1)));
    }
    let negated = selected.negate();
    Self::select(selected, negated, 0u64.wrapping_sub(u64::from(sign & 1)))
  }

  #[cfg(feature = "p256-ecdh")]
  fn negate(self) -> Self {
    Self {
      x: self.x,
      y: FieldElement::zero().sub(self.y),
      z: self.z,
      infinity_mask: self.infinity_mask,
    }
  }

  fn infinity_mask(self) -> u64 {
    self.infinity_mask
  }

  fn double(self) -> Self {
    let s = self.y.mul(self.z).double();
    let w = self.x.sub(self.z).mul(self.x.add(self.z)).triple();
    let r = self.y.mul(s);
    let ss = s.square();
    let rr = r.square();
    let b = self.x.mul(r).double();
    let h = w.square().sub(b).sub(b);
    let z = s.mul(ss);
    let x = s.mul(h);
    let y = w.mul(b.sub(h)).sub(rr.double());
    let result = Self {
      x,
      y,
      z,
      infinity_mask: 0,
    };
    Self::select(result, Self::infinity(), self.infinity_mask())
  }

  fn add_mixed(self, rhs: Affine, rhs_infinity_mask: u64) -> Self {
    // Complete Renes-Costello-Batina addition for a = -3, specialized for
    // affine `rhs` (ePrint 2015/1060, algorithm 4 with Z2 = 1).
    let curve_b = FieldElement::from_montgomery(CURVE_B_MONTGOMERY);
    let x1x2 = self.x.mul(rhs.x);
    let y1y2 = self.y.mul(rhs.y);
    let c = self.x.add(self.y).mul(rhs.x.add(rhs.y)).sub(x1x2).sub(y1y2);
    let d = rhs.y.mul(self.z).add(self.y);
    let e = rhs.x.mul(self.z).add(self.x);
    let f = e.sub(curve_b.mul(self.z)).triple();
    let g = y1y2.sub(f);
    let h = y1y2.add(f);
    let i = self.z.triple();
    let j = curve_b.mul(e).sub(x1x2).sub(i).triple();
    let k = x1x2.triple().sub(i);
    let l = d.mul(j);
    let m = k.mul(j);
    let n = k.mul(c);
    let y = h.mul(g).add(m);
    let x = h.mul(c).sub(l);
    let z = g.mul(d).add(n);
    let added = Self {
      x,
      y,
      z,
      infinity_mask: z.0.zero_mask(),
    };
    Self::select(added, self, rhs_infinity_mask)
  }

  #[cfg(feature = "p256-ecdh")]
  fn add(self, rhs: Self) -> Self {
    let curve_b = FieldElement::from_montgomery(CURVE_B_MONTGOMERY);
    let x1x2 = self.x.mul(rhs.x);
    let y1y2 = self.y.mul(rhs.y);
    let z1z2 = self.z.mul(rhs.z);
    let c = self.x.add(self.y).mul(rhs.x.add(rhs.y)).sub(x1x2).sub(y1y2);
    let d = self.y.add(self.z).mul(rhs.y.add(rhs.z)).sub(y1y2).sub(z1z2);
    let e = self.x.add(self.z).mul(rhs.x.add(rhs.z)).sub(x1x2).sub(z1z2);
    let f = e.sub(curve_b.mul(z1z2)).triple();
    let g = y1y2.sub(f);
    let h = y1y2.add(f);
    let i = z1z2.triple();
    let j = curve_b.mul(e).sub(x1x2).sub(i).triple();
    let k = x1x2.triple().sub(i);
    let l = d.mul(j);
    let m = k.mul(j);
    let n = k.mul(c);
    let y = h.mul(g).add(m);
    let x = h.mul(c).sub(l);
    let z = g.mul(d).add(n);
    Self {
      x,
      y,
      z,
      infinity_mask: z.0.zero_mask(),
    }
  }

  fn to_affine(self) -> Affine {
    let inverse_z = self.z.invert();
    Affine {
      x: self.x.mul(inverse_z),
      y: self.y.mul(inverse_z),
    }
  }
}

#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
struct SecretProjective(Projective);

#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
impl SecretProjective {
  fn add_mixed(&self, rhs: Affine, rhs_infinity_mask: u64) -> Self {
    Self(self.0.add_mixed(rhs, rhs_infinity_mask))
  }

  #[cfg(feature = "p256-ecdh")]
  fn add(&self, rhs: Projective) -> Self {
    Self(self.0.add(rhs))
  }

  fn to_affine(&self) -> Affine {
    self.0.to_affine()
  }
}

#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
impl Drop for SecretProjective {
  fn drop(&mut self) {
    self.0.x.0.zeroize_no_fence();
    self.0.y.0.zeroize_no_fence();
    self.0.z.0.zeroize_no_fence();
    self.0.infinity_mask = 0;
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

#[cfg(any(
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
fn scalar_mul_generator_affine(scalar: &Scalar) -> Affine {
  scalar_mul_generator_portable(scalar).to_affine()
}

/// Multiply the P-256 generator by a validated nonzero scalar.
///
/// This is the neutral P-256 substrate boundary shared by ECDSA and ECDH. The
/// result contains little-endian canonical affine `x || y` limbs.
#[cfg(all(
  feature = "ecdsa-p256",
  any(feature = "portable-only", miri),
  any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  )
))]
pub(super) fn scalar_mul_generator_words(scalar: &[u64; 4]) -> [u64; 8] {
  let scalar = Scalar(Uint(*scalar));
  let point = scalar_mul_generator_affine(&scalar);
  let x = point.x.to_uint();
  let y = point.y.to_uint();
  let mut output = [0u64; 8];
  output[..4].copy_from_slice(&x.0);
  output[4..].copy_from_slice(&y.0);
  output
}

#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
fn scalar_mul_generator_portable(scalar: &Scalar) -> SecretProjective {
  let digits = scalar.signed_radix_32();
  let (first, first_infinity) = Affine::select_signed(&generator_signed_table(0), digits[SIGNED_FIXED_BASE_ROWS - 1]);
  let mut acc = SecretProjective(Projective::select(
    Projective::from_affine(first),
    Projective::infinity(),
    first_infinity,
  ));
  for column in 1..SIGNED_FIXED_BASE_COLUMNS {
    let digit = digits[SIGNED_FIXED_BASE_ROWS
      .strict_mul(column)
      .strict_add(SIGNED_FIXED_BASE_ROWS - 1)];
    let (selected, infinity) = Affine::select_signed(&generator_signed_table(column), digit);
    acc = acc.add_mixed(selected, infinity);
  }
  for row in (0..SIGNED_FIXED_BASE_ROWS - 1).rev() {
    for _ in 0..5 {
      acc = SecretProjective(acc.0.double());
    }
    for column in 0..SIGNED_FIXED_BASE_COLUMNS {
      let digit = digits[SIGNED_FIXED_BASE_ROWS.strict_mul(column).strict_add(row)];
      let (selected, infinity) = Affine::select_signed(&generator_signed_table(column), digit);
      acc = acc.add_mixed(selected, infinity);
    }
  }
  acc
}

#[cfg(test)]
fn scalar_mul_generator_comb_reference(scalar: &Scalar) -> SecretProjective {
  let table = generator_comb_table();
  let mut acc = SecretProjective(Projective::infinity());
  for row in (0..COMB_WINDOW_ROWS).rev() {
    acc = SecretProjective(acc.0.double());
    let mut digit = 0usize;
    for offset in 0..COMB_WINDOW_BITS {
      let bit = (scalar.0.bit_mask(row.strict_add(COMB_WINDOW_ROWS.strict_mul(offset))) & 1).to_le_bytes()[0];
      digit |= usize::from(bit) << offset;
    }
    acc = acc.add_mixed(Affine::select(&table, digit), mask_equal_usize(digit, 0));
  }
  acc
}

#[cfg(feature = "p256-ecdh")]
#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
fn scalar_mul_public_table(scalar: &Scalar, table: &[Projective; SIGNED_WINDOW_SIZE]) -> SecretProjective {
  scalar_mul_table(scalar, table)
}

#[cfg(feature = "p256-ecdh")]
#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
fn scalar_mul_table(scalar: &Scalar, table: &[Projective; SIGNED_WINDOW_SIZE]) -> SecretProjective {
  let digits = scalar.signed_radix_32();
  let mut acc = SecretProjective(Projective::select_signed(table, digits[SIGNED_WINDOW_DIGITS - 1]));
  for row in (0..SIGNED_WINDOW_DIGITS - 1).rev() {
    for _ in 0..5 {
      acc = SecretProjective(acc.0.double());
    }
    acc = acc.add(Projective::select_signed(table, digits[row]));
  }
  acc
}

#[cfg(feature = "p256-ecdh")]
#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
fn precompute_public_table(point: Affine) -> [Projective; SIGNED_WINDOW_SIZE] {
  let base = Projective::from_affine(point);
  let mut table = [base; SIGNED_WINDOW_SIZE];
  for index in 1usize..8 {
    let doubled = index.strict_mul(2).strict_sub(1);
    table[doubled] = table[index.strict_sub(1)].double();
    table[doubled.strict_add(1)] = table[doubled].add(base);
  }
  table[SIGNED_WINDOW_SIZE - 1] = table[7].double();
  table
}

#[cfg(feature = "p256-ecdh")]
#[derive(Clone, Copy)]
pub(super) struct PublicPoint(Affine);

#[cfg(feature = "p256-ecdh")]
impl PublicPoint {
  #[cfg(all(
    not(feature = "portable-only"),
    not(miri),
    any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    )
  ))]
  pub(super) fn from_affine_words(words: [u64; 8]) -> Self {
    let [x0, x1, x2, x3, y0, y1, y2, y3] = words;
    Self(Affine {
      x: FieldElement::from_uint(Uint([x0, x1, x2, x3])),
      y: FieldElement::from_uint(Uint([y0, y1, y2, y3])),
    })
  }

  #[cfg(all(
    not(feature = "portable-only"),
    not(miri),
    any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    )
  ))]
  pub(super) fn from_montgomery_curve_terms(
    x: [u64; 4],
    y: [u64; 4],
    y_squared: [u64; 4],
    x_cubed: [u64; 4],
  ) -> Option<Self> {
    let x = FieldElement::from_montgomery(Uint(x));
    let y = FieldElement::from_montgomery(Uint(y));
    let lhs = FieldElement::from_montgomery(Uint(y_squared));
    let rhs = FieldElement::from_montgomery(Uint(x_cubed))
      .sub(x.triple())
      .add(FieldElement::from_montgomery(CURVE_B_MONTGOMERY));
    (lhs == rhs).then_some(Self(Affine { x, y }))
  }

  #[cfg(all(
    not(feature = "portable-only"),
    not(miri),
    any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    )
  ))]
  pub(super) fn to_affine_words(self) -> [u64; 8] {
    let x = self.0.x.to_uint();
    let y = self.0.y.to_uint();
    let mut words = [0u64; 8];
    words[..4].copy_from_slice(&x.0);
    words[4..].copy_from_slice(&y.0);
    words
  }

  #[cfg(any(
    test,
    feature = "portable-only",
    miri,
    not(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    ))
  ))]
  pub(super) fn from_sec1_bytes(bytes: &[u8]) -> Option<Self> {
    let [x0, x1, x2, x3, y0, y1, y2, y3] = parse_sec1_words(bytes)?;
    let point = Affine {
      x: FieldElement::from_uint(Uint([x0, x1, x2, x3])),
      y: FieldElement::from_uint(Uint([y0, y1, y2, y3])),
    };
    point.is_on_curve().then_some(Self(point))
  }

  pub(super) fn to_sec1_bytes(self) -> [u8; SEC1_BYTES] {
    self.0.encode_sec1()
  }
}

#[cfg(feature = "p256-ecdh")]
pub(super) fn parse_sec1_words(bytes: &[u8]) -> Option<[u64; 8]> {
  if bytes.len() != SEC1_BYTES || bytes.first().copied() != Some(0x04) {
    return None;
  }
  let coordinates = bytes.get(1..)?;
  let (x_bytes, y_bytes) = coordinates.split_at(FIELD_BYTES);
  let x = Uint::from_be_slice(x_bytes)?;
  let y = Uint::from_be_slice(y_bytes)?;
  if x.cmp(&FIELD_MODULUS).is_ge() || y.cmp(&FIELD_MODULUS).is_ge() {
    return None;
  }
  Some([x.0[0], x.0[1], x.0[2], x.0[3], y.0[0], y.0[1], y.0[2], y.0[3]])
}

#[cfg(all(
  test,
  feature = "p256-ecdh",
  not(feature = "portable-only"),
  not(miri),
  any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  )
))]
pub(super) fn curve_terms_for_test(words: &[u64; 8]) -> ([u64; 4], [u64; 4], [u64; 4], [u64; 4]) {
  let x = FieldElement::from_uint(Uint(words[..4].try_into().expect("four x-coordinate limbs")));
  let y = FieldElement::from_uint(Uint(words[4..].try_into().expect("four y-coordinate limbs")));
  (x.0.0, y.0.0, y.square().0.0, x.square().mul(x).0.0)
}

#[cfg(feature = "p256-ecdh")]
pub(super) fn scalar_is_canonical_nonzero(bytes: &[u8; FIELD_BYTES]) -> bool {
  let Some(candidate) = Uint::from_be_slice(bytes) else {
    return false;
  };
  candidate.zero_mask() == 0 && candidate.cmp(&SCALAR_MODULUS).is_lt()
}

#[cfg(feature = "p256-ecdh")]
#[cfg(any(
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
pub(super) fn public_key_from_scalar(bytes: &[u8; FIELD_BYTES]) -> PublicPoint {
  let scalar = Scalar::from_bytes(bytes);
  PublicPoint(scalar_mul_generator_affine(&scalar))
}

#[cfg(feature = "p256-ecdh")]
#[cfg(any(
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
pub(super) fn agree(bytes: &[u8; FIELD_BYTES], public: PublicPoint) -> [u8; FIELD_BYTES] {
  let table = precompute_public_table(public.0);
  let scalar = Scalar::from_bytes(bytes);
  let point = scalar_mul_public_table(&scalar, &table).to_affine();
  let mut shared = [0u8; FIELD_BYTES];
  point.x.to_uint().write_be(&mut shared);
  shared
}

/// Return the production P-256 window-table selection as Montgomery limbs.
#[cfg(all(
  feature = "p256-ecdh",
  feature = "diag",
  any(
    feature = "portable-only",
    miri,
    not(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    ))
  )
))]
pub(super) fn diag_select_window_limb_digest(digit: u8) -> [u64; 8] {
  let (selected, _) = Affine::select_signed(&generator_signed_table(0), digit);
  let mut output = [0u64; 8];
  output[..4].copy_from_slice(&selected.x.0.0);
  output[4..].copy_from_slice(&selected.y.0.0);
  output
}

#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
fn generator_signed_table(column: usize) -> [Affine; SIGNED_WINDOW_SIZE] {
  let mut table = [Affine::generator(); SIGNED_WINDOW_SIZE];
  for ((point, x), y) in table
    .iter_mut()
    .zip(GENERATOR_SIGNED_X[column])
    .zip(GENERATOR_SIGNED_Y[column])
  {
    *point = Affine {
      x: FieldElement::from_montgomery(x),
      y: FieldElement::from_montgomery(y),
    };
  }
  table
}

#[cfg(test)]
fn generator_comb_table() -> [Affine; COMB_WINDOW_SIZE] {
  let mut table = [Affine::generator(); COMB_WINDOW_SIZE];
  for ((point, x), y) in table.iter_mut().zip(GENERATOR_COMB_X).zip(GENERATOR_COMB_Y) {
    *point = Affine {
      x: FieldElement::from_montgomery(x),
      y: FieldElement::from_montgomery(y),
    };
  }
  table
}

#[inline(always)]
fn mask_nonzero(value: u64) -> u64 {
  0u64.wrapping_sub((value | value.wrapping_neg()) >> 63)
}

#[inline(always)]
fn mask_zero(value: u64) -> u64 {
  !mask_nonzero(value)
}

#[inline(always)]
#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
fn mask_equal_usize(left: usize, right: usize) -> u64 {
  mask_zero((left ^ right) as u64)
}

#[cfg(target_arch = "s390x")]
#[inline(always)]
fn low_u32(value: u64) -> u32 {
  let [b0, b1, b2, b3, _, _, _, _] = value.to_le_bytes();
  u32::from_le_bytes([b0, b1, b2, b3])
}

#[cfg(target_arch = "s390x")]
#[inline(always)]
fn low_u32_signed(value: i64) -> u32 {
  let [b0, b1, b2, b3, _, _, _, _] = value.to_le_bytes();
  u32::from_le_bytes([b0, b1, b2, b3])
}

#[inline(always)]
fn adc_limb(left: u64, right: u64, carry: u64) -> (u64, u64) {
  #[cfg(target_arch = "s390x")]
  {
    let low = u64::from(low_u32(left))
      .strict_add(u64::from(low_u32(right)))
      .strict_add(carry);
    let high = (left >> 32).strict_add(right >> 32).strict_add(low >> 32);
    (u64::from(low_u32(low)) | (u64::from(low_u32(high)) << 32), high >> 32)
  }

  #[cfg(not(target_arch = "s390x"))]
  {
    let (sum, carry0) = left.overflowing_add(right);
    let (sum, carry1) = sum.overflowing_add(carry);
    (sum, u64::from(carry0 | carry1))
  }
}

#[inline(always)]
fn sbb_limb(left: u64, right: u64, borrow: u64) -> (u64, u64) {
  #[cfg(target_arch = "s390x")]
  {
    let low = i64::from(low_u32(left))
      .strict_sub(i64::from(low_u32(right)))
      .strict_sub(i64::from(low_u32(borrow)));
    let low_borrow = (low >> 63) & 1;
    let high = i64::from(low_u32(left >> 32))
      .strict_sub(i64::from(low_u32(right >> 32)))
      .strict_sub(low_borrow);
    (
      u64::from(low_u32_signed(low)) | (u64::from(low_u32_signed(high)) << 32),
      u64::from(low_u32_signed((high >> 63) & 1)),
    )
  }

  #[cfg(not(target_arch = "s390x"))]
  {
    let (difference, borrow0) = left.overflowing_sub(right);
    let (difference, borrow1) = difference.overflowing_sub(borrow);
    (difference, u64::from(borrow0 | borrow1))
  }
}

#[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
#[inline(never)]
fn ct_mul_u64_wide(left: u64, right: u64) -> (u64, u64) {
  let mut product_low = 0u64;
  let mut product_high = 0u64;
  let mut multiplicand_low = left;
  let mut multiplicand_high = 0u64;
  let mut multiplier = right;
  for _ in 0..(u64::BITS / 4) {
    // These targets lower ordinary wide multiplication through instructions or
    // helpers whose latency is operand-dependent. Keep the radix-16 authority
    // fixed-work and prevent LLVM from recognizing it as an ordinary product.
    let digit = core::hint::black_box(multiplier & 0xf);
    for bit in 0..4 {
      let mask = 0u64.wrapping_sub((digit >> bit) & 1);
      let (next_low, carry) = adc_limb(product_low, multiplicand_low & mask, 0);
      let (next_high, _) = adc_limb(product_high, multiplicand_high & mask, carry);
      product_low = next_low;
      product_high = next_high;
      multiplicand_high = (multiplicand_high << 1) | (multiplicand_low >> 63);
      multiplicand_low <<= 1;
    }
    multiplier >>= 4;
  }
  (product_low, product_high)
}

#[inline(always)]
fn mul_u64_wide(left: u64, right: u64) -> (u64, u64) {
  #[cfg(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
  {
    ct_mul_u64_wide(left, right)
  }

  #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x")))]
  {
    split_u128(u128::from(left).strict_mul(u128::from(right)))
  }
}

#[inline(always)]
#[cfg(any(
  test,
  not(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))
))]
fn split_u128(value: u128) -> (u64, u64) {
  let [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15] = value.to_le_bytes();
  (
    u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7]),
    u64::from_le_bytes([b8, b9, b10, b11, b12, b13, b14, b15]),
  )
}

#[inline(always)]
fn mac_limb(acc: u64, left: u64, right: u64, carry: u64) -> (u64, u64) {
  let (product_low, product_high) = mul_u64_wide(left, right);
  let (result, carry0) = adc_limb(product_low, acc, 0);
  let (result, carry1) = adc_limb(result, carry, 0);
  let (high, overflow0) = adc_limb(product_high, carry0, 0);
  let (high, overflow1) = adc_limb(high, carry1, 0);
  debug_assert_eq!(overflow0 | overflow1, 0);
  (result, high)
}

fn montgomery_mul(left: Uint, right: Uint) -> Uint {
  let (w0, carry) = mac_limb(0, left.0[0], right.0[0], 0);
  let (w1, carry) = mac_limb(0, left.0[0], right.0[1], carry);
  let (w2, carry) = mac_limb(0, left.0[0], right.0[2], carry);
  let (w3, w4) = mac_limb(0, left.0[0], right.0[3], carry);
  let (w1, carry) = mac_limb(w1, left.0[1], right.0[0], 0);
  let (w2, carry) = mac_limb(w2, left.0[1], right.0[1], carry);
  let (w3, carry) = mac_limb(w3, left.0[1], right.0[2], carry);
  let (w4, w5) = mac_limb(w4, left.0[1], right.0[3], carry);
  let (w2, carry) = mac_limb(w2, left.0[2], right.0[0], 0);
  let (w3, carry) = mac_limb(w3, left.0[2], right.0[1], carry);
  let (w4, carry) = mac_limb(w4, left.0[2], right.0[2], carry);
  let (w5, w6) = mac_limb(w5, left.0[2], right.0[3], carry);
  let (w3, carry) = mac_limb(w3, left.0[3], right.0[0], 0);
  let (w4, carry) = mac_limb(w4, left.0[3], right.0[1], carry);
  let (w5, carry) = mac_limb(w5, left.0[3], right.0[2], carry);
  let (w6, w7) = mac_limb(w6, left.0[3], right.0[3], carry);
  montgomery_reduce([w0, w1, w2, w3, w4, w5, w6, w7])
}

fn montgomery_reduce(limbs: [u64; 8]) -> Uint {
  let [r0, r1, r2, r3, r4, r5, r6, r7] = limbs;
  let p = FIELD_MODULUS.0;
  let (r1, carry) = mac_limb(r1, r0, p[1], r0);
  let (r2, carry) = adc_limb(r2, 0, carry);
  let (r3, carry) = mac_limb(r3, r0, p[3], carry);
  let (r4, carry2) = adc_limb(r4, 0, carry);
  let (r2, carry) = mac_limb(r2, r1, p[1], r1);
  let (r3, carry) = adc_limb(r3, 0, carry);
  let (r4, carry) = mac_limb(r4, r1, p[3], carry);
  let (r5, carry2) = adc_limb(r5, carry2, carry);
  let (r3, carry) = mac_limb(r3, r2, p[1], r2);
  let (r4, carry) = adc_limb(r4, 0, carry);
  let (r5, carry) = mac_limb(r5, r2, p[3], carry);
  let (r6, carry2) = adc_limb(r6, carry2, carry);
  let (r4, carry) = mac_limb(r4, r3, p[1], r3);
  let (r5, carry) = adc_limb(r5, 0, carry);
  let (r6, carry) = mac_limb(r6, r3, p[3], carry);
  let (r7, r8) = adc_limb(r7, carry2, carry);
  subtract_modulus_once([r4, r5, r6, r7, r8])
}

fn subtract_modulus_once(limbs: [u64; 5]) -> Uint {
  let p = FIELD_MODULUS.0;
  let (w0, borrow) = sbb_limb(limbs[0], p[0], 0);
  let (w1, borrow) = sbb_limb(limbs[1], p[1], borrow);
  let (w2, borrow) = sbb_limb(limbs[2], p[2], borrow);
  let (w3, borrow) = sbb_limb(limbs[3], p[3], borrow);
  let (_, borrow) = sbb_limb(limbs[4], 0, borrow);
  let reduced = Uint([w0, w1, w2, w3]);
  let (s0, carry) = adc_limb(w0, p[0], 0);
  let (s1, carry) = adc_limb(w1, p[1], carry);
  let (s2, carry) = adc_limb(w2, p[2], carry);
  let (s3, _) = adc_limb(w3, p[3], carry);
  Uint::select(reduced, Uint([s0, s1, s2, s3]), mask_nonzero(borrow))
}

#[cfg(test)]
const GENERATOR_COMB_X: [Uint; COMB_WINDOW_SIZE] = [
  Uint([
    0x79e7_30d4_18a9_143c,
    0x75ba_95fc_5fed_b601,
    0x79fb_732b_7762_2510,
    0x1890_5f76_a537_55c6,
  ]),
  Uint([
    0x79e7_30d4_18a9_143c,
    0x75ba_95fc_5fed_b601,
    0x79fb_732b_7762_2510,
    0x1890_5f76_a537_55c6,
  ]),
  Uint([
    0x4f92_2fc5_16a0_d2bb,
    0x0d5c_c16c_1a62_3499,
    0x9241_cf3a_57c6_2c8b,
    0x2f5e_6961_fd1b_667f,
  ]),
  Uint([
    0x9e56_6847_e137_bbbc,
    0xe434_469e_8a6a_0bec,
    0xb1c4_2761_79d7_3463,
    0x5abe_0285_133d_0015,
  ]),
  Uint([
    0x62a8_c244_bfe2_0925,
    0x91c1_9ac3_8fdc_e867,
    0x5a96_a5d5_dd38_7063,
    0x61d5_87d4_21d3_24f6,
  ]),
  Uint([
    0x1c89_1f2b_2cb1_9ffd,
    0x01ba_8d5b_b192_3c23,
    0xb6d0_3d67_8ac5_ca8e,
    0x586e_b04c_1f13_bedc,
  ]),
  Uint([
    0x6257_7734_d2b5_33d5,
    0x673b_8af6_a1bd_ddc0,
    0x577e_7c9a_a79e_c293,
    0xbb6d_e651_c3b2_66b1,
  ]),
  Uint([
    0xbd6a_38e1_1ae5_aa1c,
    0xb8b7_652b_49e7_3658,
    0x0b13_0014_ee5f_87ed,
    0x9d0f_27b2_aeeb_ffcd,
  ]),
  Uint([
    0x56f8_410e_f4f8_b16a,
    0x9724_1afe_c47b_266a,
    0x0a40_6b8e_6d9c_87c1,
    0x803f_3e02_cd42_ab1b,
  ]),
  Uint([
    0x846a_56f2_c379_ab34,
    0xa8ee_068b_841d_f8d1,
    0x2031_4459_176c_68ef,
    0xf1af_32d5_915f_1f30,
  ]),
  Uint([
    0xed93_e225_d5be_5a2b,
    0x6fe7_9983_5934_f3c6,
    0x4314_0926_2262_6ffc,
    0x50bb_b4d9_7990_216a,
  ]),
  Uint([
    0xfc68_b5c5_9b39_1593,
    0xc385_f5a2_5982_70fc,
    0x7144_f3aa_d19a_dcbb,
    0xdd55_8999_83fb_ae0c,
  ]),
  Uint([
    0x5fe1_4bfe_80ec_21fe,
    0xf6ce_116a_c255_be82,
    0x98bc_5a07_2f4a_5d67,
    0xfad2_7148_db7e_63af,
  ]),
  Uint([
    0x1e9e_cc49_a56c_0dd7,
    0xa5cf_fcd8_4608_6c74,
    0x8f7a_1408_f505_aece,
    0xb37b_85c0_bef0_c47e,
  ]),
  Uint([
    0x0a1c_7294_95c8_f8be,
    0x2961_c480_3bf3_62bf,
    0x9e41_8403_df63_d4ac,
    0xc109_f9cb_91ec_e900,
  ]),
  Uint([
    0x0d5a_e356_4291_3074,
    0x5549_1b27_48a5_42b1,
    0x469c_a665_b310_732a,
    0x2959_1d52_5f1a_4cc1,
  ]),
];

#[cfg(test)]
const GENERATOR_COMB_Y: [Uint; COMB_WINDOW_SIZE] = [
  Uint([
    0xddf2_5357_ce95_560a,
    0x8b4a_b8e4_ba19_e45c,
    0xd2e8_8688_dd21_f325,
    0x8571_ff18_2588_5d85,
  ]),
  Uint([
    0xddf2_5357_ce95_560a,
    0x8b4a_b8e4_ba19_e45c,
    0xd2e8_8688_dd21_f325,
    0x8571_ff18_2588_5d85,
  ]),
  Uint([
    0x5c15_c70b_f5a0_1797,
    0x3d20_b44d_6095_6192,
    0x0491_1b37_071f_db52,
    0xf648_f916_8d6f_0f7b,
  ]),
  Uint([
    0x92aa_837c_c04c_7dab,
    0x573d_9f4c_4326_0c07,
    0x0c93_1562_78e6_cc37,
    0x94bb_725b_6b6f_7383,
  ]),
  Uint([
    0xe876_73a2_a371_73ea,
    0x2384_8008_5377_8b65,
    0x10f8_441e_05ba_b43e,
    0xfa11_fe12_4621_efbe,
  ]),
  Uint([
    0x0c35_c6e5_27e8_ed09,
    0x1e81_a33c_1819_ede2,
    0x278f_d6c0_56c6_52fa,
    0x19d5_ac08_7086_4f11,
  ]),
  Uint([
    0xe7e9_303a_b652_59b3,
    0xd6a0_afd3_d03a_7480,
    0xc5ac_83d1_9b3c_fc27,
    0x60b4_619a_5d18_b99b,
  ]),
  Uint([
    0xca92_4631_7a73_0a55,
    0x9c95_5b2f_ddbb_c83a,
    0x07c1_dfe0_ac01_9a71,
    0x244a_566d_356e_c48d,
  ]),
  Uint([
    0x7f03_09a8_04db_ec69,
    0xa83b_85f7_3bba_d05f,
    0xc609_7273_ad8e_197f,
    0xc097_440e_5067_adc1,
  ]),
  Uint([
    0x99c3_7531_5d75_bd50,
    0x837c_ffba_f72f_67bc,
    0x0613_a418_48d7_723f,
    0x23d0_f130_e2d4_1c8b,
  ]),
  Uint([
    0x3781_91c6_e57e_c63e,
    0x6542_2c40_181d_cdb2,
    0x41a8_099b_0236_e0f6,
    0x2b10_0118_01fe_49c3,
  ]),
  Uint([
    0x93b8_8b8e_74b8_2ff4,
    0xd2e0_3c40_71e7_34c9,
    0x9a7a_9eaf_43c0_322a,
    0xe6e4_c551_149d_6041,
  ]),
  Uint([
    0x90c0_b6ac_29ab_05b3,
    0x37a9_a83c_4e25_1ae6,
    0x0a7d_c875_c2aa_de7d,
    0x7738_7de3_9f0e_1a84,
  ]),
  Uint([
    0x3596_b6e4_cc0e_6a8f,
    0xfd6d_4bbf_6b38_8f23,
    0xaba4_53fa_c39c_ef4e,
    0x9c13_5ac8_f9f6_28d5,
  ]),
  Uint([
    0xc2d0_95d0_5894_5705,
    0xb908_3d96_ddeb_85c0,
    0x8469_2b8d_7a40_449b,
    0x9bc3_344f_2eee_1ee1,
  ]),
  Uint([
    0xe76f_5b6b_b84f_983f,
    0xbe7e_ef41_9f5f_84e1,
    0x1200_d496_80ba_a189,
    0x6376_551f_18ef_332c,
  ]),
];

#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
const GENERATOR_SIGNED_X: [[Uint; SIGNED_WINDOW_SIZE]; SIGNED_FIXED_BASE_COLUMNS] = [
  [
    Uint([
      0x79e7_30d4_18a9_143c,
      0x75ba_95fc_5fed_b601,
      0x79fb_732b_7762_2510,
      0x1890_5f76_a537_55c6,
    ]),
    Uint([
      0x8500_46d4_10dd_d64d,
      0xaa6a_e3c1_a433_827d,
      0x7322_0503_8d14_90d9,
      0xf6bb_32e4_3dcf_3a3b,
    ]),
    Uint([
      0xffac_3f90_4eeb_c127,
      0xb027_f84a_087d_81fb,
      0x66ad_77dd_87cb_bc98,
      0x2693_6a3f_b6ff_747e,
    ]),
    Uint([
      0x74b0_b50d_4691_8dcc,
      0x4650_a6ed_c623_c173,
      0x0cda_acac_e810_0af2,
      0x5773_62f5_41b0_176b,
    ]),
    Uint([
      0xbe1b_8aae_c45c_61f5,
      0x90ec_649a_94b9_537d,
      0x941c_b5aa_d076_c20c,
      0xc907_9605_8905_23c8,
    ]),
    Uint([
      0x4039_4737_3e77_664a,
      0x55ae_744f_346c_ee3e,
      0xd50a_961a_5b17_a3ad,
      0x1307_4b59_5421_3673,
    ]),
    Uint([
      0x0746_354e_a017_3b4f,
      0x2bd2_0213_d23c_00f7,
      0xf43e_aab5_0c23_bb08,
      0x13ba_5119_c312_3e03,
    ]),
    Uint([
      0x27f1_4cd1_9499_a78f,
      0x462a_b5c5_6f9b_3455,
      0x8f90_f02a_f02c_fc6b,
      0xb763_891e_b265_230d,
    ]),
    Uint([
      0x75c9_6e8f_264e_20e8,
      0xabe6_bfed_59a7_a841,
      0x2cc0_9c04_44c8_eb00,
      0xe05b_3080_f0c4_e16b,
    ]),
    Uint([
      0x2c18_dbd1_9dc2_1ec8,
      0x98f9_868a_0fcf_8139,
      0x737d_2cd6_4825_0b49,
      0xcc61_c947_24b3_428f,
    ]),
    Uint([
      0xea7d_260a_6245_e404,
      0x9de4_0795_6e7f_dfe0,
      0x1ff3_a415_8dac_1ab5,
      0x3e70_90f1_649c_9073,
    ]),
    Uint([
      0x04b7_1aa7_d269_7768,
      0xabde_def5_ca34_5a33,
      0x2409_d29d_ee37_385e,
      0x4ee1_df77_cb83_e156,
    ]),
    Uint([
      0xccc4_2563_4b2e_d709,
      0x0e35_6769_856f_d30d,
      0xbcbc_d43f_559e_9811,
      0x7384_77ac_5395_b759,
    ]),
    Uint([
      0xa242_a35b_b0cf_664a,
      0x126e_48f7_7f97_07e3,
      0x1717_bf54_c683_2660,
      0xfaae_7332_fd12_c72e,
    ]),
    Uint([
      0x72bc_d8b7_bc60_055b,
      0x03cc_23ee_56e2_7e4b,
      0xee33_7424_e481_9370,
      0xe2aa_0e43_0ad3_da09,
    ]),
    Uint([
      0x808b_0b65_0bc6_fb80,
      0x5882_e075_3ffe_2e6b,
      0xd5ef_2f7c_2c83_f549,
      0x54d6_3c80_9103_b723,
    ]),
  ],
  [
    Uint([
      0x027c_c8b8_fac6_1d9a,
      0x7d25_e062_e3c6_fe8a,
      0xe088_05bf_e5bf_f503,
      0x1327_1e6c_6ff6_32f7,
    ]),
    Uint([
      0x9ad5_462b_b4d8_bc50,
      0x181c_0b16_a919_5770,
      0xebd4_fe1c_7841_2a68,
      0xae03_41bc_c0df_f48c,
    ]),
    Uint([
      0x3286_5719_a8af_d30b,
      0x8679_8328_8a82_6dce,
      0xdf04_e891_c4a8_fbe0,
      0xbb6b_6e1b_ebf5_6ad3,
    ]),
    Uint([
      0x0a50_b12e_523b_8bf6,
      0x8009_eb5b_8f91_0c1b,
      0xf535_af82_4a16_7588,
      0x0f83_5f9c_fb2a_2abd,
    ]),
    Uint([
      0x87a7_ebd1_e0a1_b12a,
      0x1e4e_f88d_770b_a95f,
      0x8c33_345c_dc2a_e9cb,
      0xcecf_1276_01cc_8403,
    ]),
    Uint([
      0xb822_2605_2b7c_e542,
      0xe6d4_ce99_7472_bde1,
      0x53e1_6ebe_09d2_f4da,
      0x180f_f42e_53b9_2b2e,
    ]),
    Uint([
      0xb956_970e_2fdd_23cc,
      0xb802_88bc_5682_e971,
      0xe6e6_d91e_9ae8_6ebc,
      0x0564_c83f_8c9f_1939,
    ]),
    Uint([
      0x58af_2010_f5b3_43bc,
      0x0f2e_400a_f2f1_42fe,
      0x3483_bfde_a85f_4bdf,
      0xf0b1_d093_03bf_eaa9,
    ]),
    Uint([
      0x4ed7_1457_6be5_f7de,
      0xd930_06f8_c226_3c9e,
      0xe073_694c_caca_cb36,
      0x2ff7_a5b4_3ae1_18ab,
    ]),
    Uint([
      0x137a_4fb4_86df_2a61,
      0xa1ed_9c07_ecf7_b4a2,
      0xb2e4_60e2_7bd0_42ff,
      0xb7f5_e2fa_5f62_f5ec,
    ]),
    Uint([
      0x6793_0af2_31f6_3950,
      0xa777_97c1_14ca_a2c9,
      0x526e_80ee_27ac_7e62,
      0xe1e6_e626_58b2_8aec,
    ]),
    Uint([
      0xec2f_ccaa_ddce_3345,
      0x2a68_11b7_012a_4350,
      0x9676_0ff1_ac59_8bdc,
      0x054d_652a_d1bf_4128,
    ]),
    Uint([
      0x1778_5b77_99eb_6df0,
      0x26c3_cc51_7386_b779,
      0x345e_d988_6417_a48e,
      0xe990_b4e4_07d6_ef31,
    ]),
    Uint([
      0x19e6_125d_ec3f_1dec,
      0x07b1_f040_9111_78da,
      0xd93e_deda_904a_6738,
      0x5518_7a5a_0beb_edcd,
    ]),
    Uint([
      0x3806_b69b_9222_2f1f,
      0x5a24_59ca_6cf7_ae70,
      0x6789_f69c_a852_17ee,
      0x5f23_2b5e_e3dc_85ac,
    ]),
    Uint([
      0xb674_481b_7bfe_7178,
      0x4e1d_ebae_6540_5868,
      0x061b_2821_c48c_867d,
      0x69c1_5b35_513b_30ea,
    ]),
  ],
  [
    Uint([
      0xb448_0f04_41c2_3fa3,
      0xb471_2eb0_c198_9a2e,
      0x3ccb_ba0f_93a2_9ca7,
      0x6e20_5c14_d619_428c,
    ]),
    Uint([
      0xe3b2_2c6b_c4fe_3c39,
      0xba4a_8153_6c7b_ebdf,
      0xf23a_b6b7_2569_3459,
      0x53bc_3770_1492_2b11,
    ]),
    Uint([
      0x5066_efb6_d979_0ed6,
      0xa77a_0cbc_a6aa_793b,
      0x1a91_5f3c_223e_042e,
      0x1c5d_ef04_69c5_874b,
    ]),
    Uint([
      0x6a70_91c2_e48f_b889,
      0x2688_2c13_7b8a_9d06,
      0xa249_8663_1b82_a0e2,
      0x844e_d736_3518_152d,
    ]),
    Uint([
      0x16ea_b6a2_0d64_5fd6,
      0x632c_bd8d_f61d_3148,
      0xcc1b_f7cf_6207_9ae9,
      0x257e_e5c7_f33e_ccbb,
    ]),
    Uint([
      0xf01d_095d_c838_5050,
      0x0d54_a5d5_df4b_441c,
      0x2a37_ccb4_0927_706a,
      0xdf00_8f54_45d7_eb7e,
    ]),
    Uint([
      0x1554_d46d_a670_ff1d,
      0x2483_3d88_cb97_a1cc,
      0x8fa6_ab3c_ded9_7493,
      0x215e_0371_8992_6498,
    ]),
    Uint([
      0x82ee_061d_5a74_be50,
      0xe417_81c4_dea1_6ff5,
      0xe0b0_c81e_99bf_c8a2,
      0x624f_4d69_0b54_7e2d,
    ]),
    Uint([
      0xc431_a238_013f_f83b,
      0x7c00_18b2_fad6_9d08,
      0x99ae_b52a_4c95_89ea,
      0x121f_41ab_9b1c_f19f,
    ]),
    Uint([
      0x775c_bfa8_6d51_8ffb,
      0xdece_e1f6_930f_124b,
      0x9a40_2804_f5e8_1d0f,
      0x0e82_25c5_2a0e_eb2f,
    ]),
    Uint([
      0xa98f_42fa_3d84_3d53,
      0x3377_7cc6_13ef_927a,
      0xc440_cdbe_cb84_ca74,
      0x8c22_f963_1dc7_c5dd,
    ]),
    Uint([
      0x3815_1e27_4d55_9d96,
      0x4f18_c0d3_b8db_6c01,
      0x49a3_aa83_6f99_21af,
      0xdbea_b27b_8c04_6029,
    ]),
    Uint([
      0x896d_5723_37e4_40d7,
      0x685c_5fd9_ade2_3f68,
      0xb5b1_a26d_c2c6_4918,
      0xb939_0e30_dad6_580c,
    ]),
    Uint([
      0x733b_64d3_9de4_0ca3,
      0x1d4b_6d6f_d2f3_857e,
      0xbe2b_e8e9_b2ed_92f7,
      0x64ca_7047_b77d_a248,
    ]),
    Uint([
      0x7872_e34b_3390_ff23,
      0x968c_e4ab_de7d_18ef,
      0x9b4a_745e_627f_e7b1,
      0x9607_b0a0_caff_3e2a,
    ]),
    Uint([
      0x715c_9f97_3112_795f,
      0xe824_4437_984e_6ee1,
      0x55cb_4858_ecb6_6bcd,
      0x7c13_6735_abaf_fbee,
    ]),
  ],
  [
    Uint([
      0xc492_ec64_4cd8_f64c,
      0x58a2_d790_279d_7b51,
      0x0ced_1fc5_1fc7_5256,
      0x3e65_8aed_8f43_3017,
    ]),
    Uint([
      0x8303_604f_692a_c542,
      0xf079_ffe1_227b_91d3,
      0x19f6_3e63_15aa_f9bd,
      0xf99e_e565_f1f3_44fb,
    ]),
    Uint([
      0xc035_f697_960e_b8c7,
      0xf159_9f2c_e2de_04d3,
      0x8924_50f8_d2ad_9228,
      0x7d48_129b_b829_c1ab,
    ]),
    Uint([
      0x972b_3f8f_81a1_b3be,
      0x4f3c_e145_ce27_64a0,
      0xe2d0_f1cc_28c4_f5f7,
      0xdeee_0c0d_c7f3_985b,
    ]),
    Uint([
      0xa691_398a_4a9e_b3f0,
      0x56c1_dbff_3b99_a48f,
      0x9a87_e1b9_1b4b_5b32,
      0xad63_9614_5378_b5fe,
    ]),
    Uint([
      0xca6c_0937_b1b7_6ba6,
      0x1a2e_ab85_4d20_26dc,
      0xb171_5e15_19d9_ae0a,
      0xf1ad_9199_bac4_a026,
    ]),
    Uint([
      0xc832_7149_a8c2_5ff6,
      0x29bf_2556_782e_6569,
      0x9012_f5c6_cd68_fc38,
      0x3e67_e8bd_3b98_2ad5,
    ]),
    Uint([
      0xefb2_6a75_3f73_f449,
      0x1d1c_94f8_8d44_fc79,
      0x49f0_fbc5_3bc0_dc4d,
      0xb747_ea0b_3698_a0d0,
    ]),
    Uint([
      0xe5d2_7171_f6bd_f1bf,
      0x0b77_b876_facb_0d8f,
      0xda95_471d_8496_a31b,
      0x46a5_0dbb_3f16_b103,
    ]),
    Uint([
      0xc38e_438f_0876_fd4e,
      0x45f0_c307_83d2_f383,
      0x203c_c2ec_b109_34cb,
      0x6a8f_2439_2c9d_46ee,
    ]),
    Uint([
      0xe6ec_9809_3a69_fc01,
      0x7e20_fecb_faa9_dfc2,
      0x5cfd_bb07_f56f_2a55,
      0xb1cd_6868_0bbd_bfdf,
    ]),
    Uint([
      0x1c0a_8e44_fc94_dea3,
      0x34c8_cdbf_dad6_a0b0,
      0x919c_3840_0411_3cef,
      0xfd32_fba4_1549_0ffa,
    ]),
    Uint([
      0x5463_7e41_8299_7cc1,
      0x08c5_a96c_e372_0c9c,
      0x78bc_e01c_11de_5d45,
      0x49d6_23e5_0dfd_d75a,
    ]),
    Uint([
      0x2b1d_402a_ba16_f73b,
      0x2fb3_1014_8cf9_b9fc,
      0x2d51_e60e_446e_f7bf,
      0xc731_021b_b91e_1745,
    ]),
    Uint([
      0x2389_9fe8_6625_95c2,
      0x495d_6727_11a8_0773,
      0x86c9_71d2_b0d1_d43b,
      0xb518_637c_93b7_a65f,
    ]),
    Uint([
      0x6e30_251a_cb45_2fdb,
      0x31ee_6965_50f3_0650,
      0xb0b3_e508_9335_48d9,
      0xb894_9a4f_f4b0_ef5b,
    ]),
  ],
];

#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
const GENERATOR_SIGNED_Y: [[Uint; SIGNED_WINDOW_SIZE]; SIGNED_FIXED_BASE_COLUMNS] = [
  [
    Uint([
      0xddf2_5357_ce95_560a,
      0x8b4a_b8e4_ba19_e45c,
      0xd2e8_8688_dd21_f325,
      0x8571_ff18_2588_5d85,
    ]),
    Uint([
      0x2f36_48d3_61be_e1a5,
      0x152c_d7cb_eb23_6ff8,
      0x19a8_fb0e_9204_2dbe,
      0x78c5_7751_0a5b_8a3b,
    ]),
    Uint([
      0xb04c_5c1f_c983_a7eb,
      0x583e_47ad_0861_fe1a,
      0x7882_0831_1a2e_e98e,
      0xd5f0_6a29_e587_cc07,
    ]),
    Uint([
      0x2d96_f24c_e4cb_aba6,
      0x1762_8471_fad6_f447,
      0x6b6c_36de_e5dd_d22e,
      0x84b1_4c39_4c5a_b863,
    ]),
    Uint([
      0xeb30_9b4a_e7ba_4f10,
      0x73c5_68ef_e5eb_882b,
      0x3540_a987_7e7a_1f68,
      0x73a0_76bb_2dd1_e916,
    ]),
    Uint([
      0x93d3_6220_d377_e44b,
      0x299c_2b53_adff_14b5,
      0xf424_d44c_ef63_9f11,
      0xa4c9_916d_4a07_f75f,
    ]),
    Uint([
      0x2847_d030_3f5b_9d4d,
      0x6742_f2f2_5da6_7bdd,
      0xef93_3bdc_77c9_4195,
      0xeaed_d915_6e24_0867,
    ]),
    Uint([
      0xf59d_a3a9_532d_4977,
      0x21e3_327d_cf9e_ba15,
      0x123c_7b84_be60_bbf0,
      0x56ec_12f2_7706_df76,
    ]),
    Uint([
      0x1eb7_777a_a45f_3314,
      0x56af_7bed_ce5d_45e3,
      0x2b6e_019a_88b1_2f1a,
      0x0866_59cd_fd83_5f9b,
    ]),
    Uint([
      0x0c2b_4078_80dd_9e76,
      0xc43a_8991_383f_be08,
      0x5f7d_2d65_779b_e5d2,
      0x7871_9a54_eb3b_4ab5,
    ]),
    Uint([
      0x1a76_8561_2b94_4e88,
      0x250f_939e_e57f_61c8,
      0x0c0d_aa89_1ead_643d,
      0x6893_0023_e125_b88e,
    ]),
    Uint([
      0x0cac_12d9_1cbb_5b43,
      0x170e_d2f6_ca89_5637,
      0x2822_8cfa_8ade_6d66,
      0x7ff5_7c95_5323_8aca,
    ]),
    Uint([
      0x3575_2b90_c00e_e17f,
      0x6874_8390_742e_d2e3,
      0x7cd0_6422_bd1f_5bc1,
      0xfbc0_8769_c9e7_b797,
    ]),
    Uint([
      0x27b5_2db7_995d_586b,
      0xbe29_569e_8322_37c2,
      0xe8e4_193e_2a65_e7db,
      0x1527_06dc_2eaa_1bbb,
    ]),
    Uint([
      0x40b8_524f_6383_c45d,
      0xd766_3554_42a4_1b25,
      0x64ef_a6de_778a_4797,
      0x2042_170a_7079_adf4,
    ]),
    Uint([
      0xf2f1_1bd6_52a2_3f9b,
      0x3670_c319_4b0b_6587,
      0x55c4_623b_b158_0e9e,
      0x64ed_f7b2_01ef_e220,
    ]),
  ],
  [
    Uint([
      0x55dc_a6c0_232f_76a5,
      0x8957_c32d_701e_f426,
      0xee72_8bcb_a10a_5178,
      0x5ea6_0411_b62c_5173,
    ]),
    Uint([
      0xb6bc_45cf_7003_e866,
      0xf11a_6dea_8a24_a41b,
      0x5407_151a_d04c_24c2,
      0x62c9_d27d_da5b_7b68,
    ]),
    Uint([
      0x0a69_5b11_471f_1ff0,
      0xd76c_3389_be15_baf0,
      0x018e_db95_be96_c43e,
      0xf2be_aaf4_9079_4158,
    ]),
    Uint([
      0xf59b_2931_2afc_eb62,
      0xc797_df2a_169d_383f,
      0xeb3f_5fb0_66ac_02b0,
      0x029d_4c6f_daa2_d0ca,
    ]),
    Uint([
      0x687c_012e_1b39_b80f,
      0xfd90_d0ad_35c3_3ba4,
      0xa3ef_5a67_5c96_61c2,
      0x368f_c88e_e017_429e,
    ]),
    Uint([
      0xc59b_cc02_2c34_a1c6,
      0x3803_d6f9_422c_46c2,
      0x18af_f74f_5c14_a8a2,
      0x55ae_bf80_10a0_8b28,
    ]),
    Uint([
      0x5519_32a2_3956_0368,
      0xe893_752b_049c_28e2,
      0x0b03_cee5_a6a1_58c3,
      0xe12d_656b_0496_4263,
    ]),
    Uint([
      0x2ea0_1b95_c708_1603,
      0xe943_e4c9_3dba_1097,
      0x47be_92ad_b438_f3a6,
      0x00bb_7742_e5bf_6636,
    ]),
    Uint([
      0x3cce_53f1_cd87_1236,
      0xf156_a39d_c2aa_6d52,
      0x9cc5_f271_b198_d76d,
      0xbc61_5b6f_8138_3d39,
    ]),
    Uint([
      0x7aa6_ec6b_cc24_23b7,
      0x75ce_0a7f_ba63_eea7,
      0x67a4_5fb1_f250_a6e1,
      0x93bc_919c_e53c_dc9f,
    ]),
    Uint([
      0x6361_78b0_b3c9_fef0,
      0xaf77_52e0_6d5f_90be,
      0x94ec_af18_eece_51cf,
      0x2864_d0ed_ca80_6e1f,
    ]),
    Uint([
      0x0a11_51d4_92a2_1005,
      0xad7f_3971_3311_0fdf,
      0x8c95_928c_1960_100f,
      0x6c91_c825_7bf0_3362,
    ]),
    Uint([
      0x0f45_6b7e_2586_abba,
      0x239c_a6a5_59c9_6e9a,
      0xe327_459c_e2eb_4206,
      0x3a4c_3313_a002_b90a,
    ]),
    Uint([
      0xf7d0_4722_eb32_9d41,
      0xf449_099e_f170_b391,
      0xfd31_7a69_ca99_f828,
      0x50c3_db2b_34a4_976d,
    ]),
    Uint([
      0x660e_3ec5_48e9_e516,
      0x124b_4e47_3197_eb31,
      0x10a0_cb13_aafc_ca23,
      0x7bd6_3ba4_8213_224f,
    ]),
    Uint([
      0x3b4a_1666_3687_1088,
      0xe5e2_9f5d_1220_b1ff,
      0x4b82_bb35_233d_9f4d,
      0x4e07_6333_18cd_c675,
    ]),
  ],
  [
    Uint([
      0x90db_7957_b364_1686,
      0x0432_691d_45ac_8b4e,
      0x07a7_59ac_f64e_0350,
      0x0514_d89c_9c97_2517,
    ]),
    Uint([
      0x4645_c8ab_5afc_60db,
      0xaa02_2355_20b9_f2a3,
      0x52a2_954c_ce0f_c507,
      0x8c27_31bb_7ce1_c2e7,
    ]),
    Uint([
      0x0e83_0078_73b6_c1da,
      0x55cf_85d2_fcd8_557a,
      0x0f7c_7c76_0460_f3b1,
      0x8705_2acb_46e5_8063,
    ]),
    Uint([
      0x282f_476f_d86e_27c7,
      0xa04e_daca_04af_efdc,
      0x8b25_6ebc_6119_e34d,
      0x56a4_13e9_0787_d78b,
    ]),
    Uint([
      0xbf6b_34a8_1680_ac73,
      0xaa08_4e88_72c7_7aa0,
      0x7b5a_864e_05a0_a1d1,
      0x0641_f6db_359a_1b16,
    ]),
    Uint([
      0x74eb_34f3_5bf7_16c7,
      0x57a6_5b58_641b_d6ca,
      0xef34_5e48_35e6_fa02,
      0x191f_913b_8834_2a09,
    ]),
    Uint([
      0x549b_d592_e56d_74ff,
      0x58a8_caf5_43b5_e1ec,
      0x3c60_87a3_23e9_3cb9,
      0x8b05_4987_5648_b83c,
    ]),
    Uint([
      0x3a83_545d_bdcc_9ae4,
      0x2573_dbb6_409b_1e8e,
      0x4829_60c4_a6c9_3539,
      0xf010_59ad_5ae1_8798,
    ]),
    Uint([
      0x0cfb_bcba_ef0f_5958,
      0x8deb_3aeb_7be8_fbdc,
      0x12b9_5408_1f15_aa31,
      0x5acc_09b3_4c0c_06fd,
    ]),
    Uint([
      0x884a_5d39_fee9_e867,
      0x9540_428f_fb50_5454,
      0xb2bf_2e20_107a_70d1,
      0xd991_7c3b_a010_b2aa,
    ]),
    Uint([
      0x4bc8_2b70_c8d9_4708,
      0x7e0b_43fc_c814_364f,
      0x286d_4e24_86f5_9b7e,
      0x1abc_895e_4d6b_f4c4,
    ]),
    Uint([
      0x242b_9eaa_7040_bf3b,
      0x39c4_79e5_1614_b091,
      0x338e_de2b_0e4b_af5d,
      0x5bb1_92b7_f0a5_3945,
    ]),
    Uint([
      0x8791_1c4e_7dee_5b9b,
      0xb90c_5053_deb0_4f6e,
      0x37b9_42a1_8f06_5aa6,
      0x34ac_df2a_1ca0_928d,
    ]),
    Uint([
      0xc65d_ae9b_8da9_9315,
      0x9c14_5175_0fc6_98a4,
      0x8a29_6b94_ff95_8c27,
      0x3868_4e08_4395_0097,
    ]),
    Uint([
      0x1b05_818e_eb40_e3a5,
      0x6ac6_2204_c0fa_8d7a,
      0xb5b9_0585_71ed_4809,
      0xb243_2ef0_f7cb_65f2,
    ]),
    Uint([
      0x5466_1595_5dbe_c38e,
      0x51c0_782c_388a_d153,
      0x9ba4_c53a_c6e0_952f,
      0x27e6_782a_1b21_dfa8,
    ]),
  ],
  [
    Uint([
      0x0b61_942e_05da_59eb,
      0xba3d_60a3_0ddc_3722,
      0x7c31_1cd1_742e_7f87,
      0x6473_ffee_f6b0_1b6e,
    ]),
    Uint([
      0x8a1d_661f_d621_9199,
      0x8c88_3bc6_d48c_e41c,
      0x1065_118f_3c74_d904,
      0x7138_89ee_0faf_8b1b,
    ]),
    Uint([
      0x24d7_85e1_3a50_afc9,
      0x2745_ba27_63a9_6ee0,
      0x9565_3401_3bfb_6d7b,
      0x5362_0267_1bad_2a42,
    ]),
    Uint([
      0x7df4_adc0_d39e_25c3,
      0x4061_9820_c467_a080,
      0x440e_bc93_61cf_5a58,
      0x5277_29a6_422a_d600,
    ]),
    Uint([
      0x437a_243e_c26b_5302,
      0x0275_878c_3ccb_4c10,
      0x0e81_e4a2_1de0_7015,
      0x0c62_65c9_850d_f3c0,
    ]),
    Uint([
      0x35b3_dfb8_07ea_7b0e,
      0xedf5_496f_3ed9_eb89,
      0x8932_e5ff_2d6d_08ab,
      0xf314_874e_25bd_2731,
    ]),
    Uint([
      0x5e3a_7538_6ecd_ca88,
      0xf297_eaa6_c175_3a04,
      0x1012_1e54_05db_3256,
      0xab96_97d4_f085_1055,
    ]),
    Uint([
      0x5218_c3fe_228d_291e,
      0x35b8_04b5_43c1_29d6,
      0xfac8_59b8_d1ac_c516,
      0x6c10_697d_95d6_e668,
    ]),
    Uint([
      0x2a4f_3f97_7b86_5bff,
      0x8481_95e6_6b1c_198c,
      0x491a_d088_2170_2ea6,
      0x3f20_b437_4903_5228,
    ]),
    Uint([
      0xf16b_431b_65cc_de7b,
      0x41e2_cd18_27e7_6a6f,
      0xb9c8_cf8f_4e34_84d7,
      0x6442_6efd_8315_244a,
    ]),
    Uint([
      0x247b_4995_986e_b9ed,
      0x7478_5bf5_3dd0_955e,
      0x88f7_4f61_c0c7_a201,
      0x8861_a15b_5d01_a80d,
    ]),
    Uint([
      0x58d1_90f6_795d_cfb7,
      0xfef0_1b03_8358_8baf,
      0x9e6d_1d63_ca1f_c1c0,
      0x5317_3f96_f0a4_1ac9,
    ]),
    Uint([
      0x8c72_a468_0fb2_a3ac,
      0xcc53_bbff_319c_25af,
      0x198e_ba79_78a9_2421,
      0xcd61_f28b_a3bd_ecf3,
    ]),
    Uint([
      0x9d3b_4724_4fee_99d4,
      0x4bca_48b6_fac5_c1ea,
      0x70f5_f514_bbea_9af7,
      0x751f_55a5_974c_283a,
    ]),
    Uint([
      0x30e4_53ba_d98c_99ce,
      0xba6e_0d4a_14d3_9f5b,
      0xf7db_02a6_431c_e415,
      0xcd90_9c7c_f6e1_d823,
    ]),
    Uint([
      0x208b_8326_3c88_f3bd,
      0xab14_7c30_db1d_9989,
      0xed65_15fd_44d4_df03,
      0x17a1_2f75_e72e_b0c5,
    ]),
  ],
];

#[cfg(all(test, feature = "p256-ecdh"))]
mod tests {
  #[cfg(all(
    not(feature = "portable-only"),
    not(miri),
    any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    )
  ))]
  fn affine_words(point: super::Affine) -> [u64; 8] {
    let x = point.x.to_uint();
    let y = point.y.to_uint();
    let mut words = [0u64; 8];
    words[..4].copy_from_slice(&x.0);
    words[4..].copy_from_slice(&y.0);
    words
  }

  #[test]
  fn target_shaped_wide_multiply_matches_u128() {
    let edges = [0, 1, u64::MAX / 2, u64::MAX - 1, u64::MAX];
    for left in edges {
      for right in edges {
        let product = u128::from(left).strict_mul(u128::from(right));
        assert_eq!(super::ct_mul_u64_wide(left, right), super::split_u128(product));
      }
    }
  }

  #[test]
  fn fixed_base_signed_window_matches_comb_and_arbitrary_point_window() {
    let table = super::precompute_public_table(super::Affine::generator());
    for bytes in [[0x11; 32], [0x42; 32], [0x7f; 32], [0xa5; 32]] {
      let scalar = super::Scalar::from_bytes(&bytes);
      let fixed = super::scalar_mul_generator_portable(&scalar).to_affine().encode_sec1();
      let comb = super::scalar_mul_generator_comb_reference(&scalar)
        .to_affine()
        .encode_sec1();
      let arbitrary = super::scalar_mul_public_table(&scalar, &table)
        .to_affine()
        .encode_sec1();
      assert_eq!(fixed, comb);
      assert_eq!(fixed, arbitrary);
    }
  }

  #[cfg(all(
    not(feature = "portable-only"),
    not(miri),
    any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    )
  ))]
  #[test]
  fn native_scalar_multiplication_matches_portable_authority() {
    let mut scalar_one = [0u8; 32];
    scalar_one[31] = 1;
    let scalar_order_minus_one = [
      0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xbc, 0xe6, 0xfa,
      0xad, 0xa7, 0x17, 0x9e, 0x84, 0xf3, 0xb9, 0xca, 0xc2, 0xfc, 0x63, 0x25, 0x50,
    ];
    let cases = [
      scalar_one,
      [0x11; 32],
      [0x42; 32],
      [0x7f; 32],
      [0xa5; 32],
      scalar_order_minus_one,
    ];

    for scalar_bytes in cases {
      let scalar = super::Scalar::from_bytes(&scalar_bytes);
      let portable = super::scalar_mul_generator_portable(&scalar).to_affine();
      let native = super::super::p256_core::scalar_mul_generator_words(&scalar.0.0);
      assert_eq!(native, affine_words(portable), "fixed-base scalar {scalar_bytes:02x?}");

      for peer_bytes in cases {
        let peer_scalar = super::Scalar::from_bytes(&peer_bytes);
        let peer = super::scalar_mul_generator_portable(&peer_scalar).to_affine();
        let table = super::precompute_public_table(peer);
        let portable = super::scalar_mul_public_table(&scalar, &table).to_affine();
        let native = super::super::p256_core::scalar_mul_words(&scalar.0.0, &affine_words(peer));
        assert_eq!(
          native,
          affine_words(portable),
          "arbitrary-point scalars {scalar_bytes:02x?} * {peer_bytes:02x?}"
        );
      }
    }
  }

  #[cfg(all(
    not(feature = "portable-only"),
    not(miri),
    any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    )
  ))]
  #[test]
  fn native_public_point_validation_matches_portable_authority() {
    let mut cases = [[0u8; 65]; 8];
    cases[0] = super::Affine::generator().encode_sec1();
    for (case, scalar_bytes) in cases[1..5]
      .iter_mut()
      .zip([[0x11; 32], [0x42; 32], [0x7f; 32], [0xa5; 32]])
    {
      let scalar = super::Scalar::from_bytes(&scalar_bytes);
      *case = super::scalar_mul_generator_portable(&scalar).to_affine().encode_sec1();
    }
    cases[5][0] = 0x04;
    cases[5][64] = 1;
    cases[6][0] = 0x04;
    cases[6][32] = 1;
    cases[7].fill(0xff);
    cases[7][0] = 0x04;

    for bytes in cases {
      let portable = super::PublicPoint::from_sec1_bytes(&bytes);
      let native = super::super::p256_core::public_point_from_sec1(&bytes);
      assert_eq!(native.is_some(), portable.is_some(), "SEC1 point {bytes:02x?}");
      if let (Some(native), Some(portable)) = (native, portable) {
        assert_eq!(native.to_sec1_bytes(), portable.to_sec1_bytes());
      }
    }
  }
}
