//! Portable P-256 arithmetic and encoding authority shared by ECDSA and ECDH.

#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
use super::ecdsa_generator_tables::{
  P256_SIGNING_COMB_WIDTH, P256_SIGNING_GENERATOR_COMB_SHIFT_13_X, P256_SIGNING_GENERATOR_COMB_SHIFT_13_Y,
  P256_SIGNING_GENERATOR_COMB_SHIFT_25_X, P256_SIGNING_GENERATOR_COMB_SHIFT_25_Y, P256_SIGNING_GENERATOR_COMB_X,
  P256_SIGNING_GENERATOR_COMB_Y,
};

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
const FIXED_BASE_COMB_ROWS: usize = 37;
#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
const _: () = assert!(FIXED_BASE_COMB_ROWS * P256_SIGNING_COMB_WIDTH == 259);

const FIELD_MODULUS: Uint = Uint([
  0xffff_ffff_ffff_ffff,
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
#[cfg(test)]
const GENERATOR_X: Uint = Uint([
  0xf4a1_3945_d898_c296,
  0x7703_7d81_2deb_33a0,
  0xf8bc_e6e5_63a4_40f2,
  0x6b17_d1f2_e12c_4247,
]);
#[cfg(test)]
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
#[cfg(any(
  test,
  feature = "portable-only",
  miri,
  not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  ))
))]
const FIELD_ONE_MONTGOMERY: Uint = Uint([
  0x0000_0000_0000_0001,
  0xffff_ffff_0000_0000,
  0xffff_ffff_ffff_ffff,
  0x0000_0000_ffff_fffe,
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
  #[cfg(test)]
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
    let mask = 0u64.wrapping_sub(borrow);
    difference
      .add_raw(Self([mask, FIELD_MODULUS.0[1] & mask, 0, FIELD_MODULUS.0[3] & mask]))
      .0
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
    #[cfg(any(
      target_arch = "riscv32",
      target_arch = "riscv64",
      target_arch = "s390x",
      target_arch = "x86_64"
    ))]
    // SECURITY: Keep the mask opaque so the tested LLVM builds retain bitwise
    // selection instead of branching on a secret digit. Binary CT evidence is
    // still required; black_box is not a language-level constant-time guarantee.
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
    let [r0, r1, r2, r3] = self.0.0;
    montgomery_reduce([r0, r1, r2, r3, 0, 0, 0, 0])
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
  const fn one() -> Self {
    Self::from_montgomery(FIELD_ONE_MONTGOMERY)
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
    Self::from_montgomery(montgomery_square(self.0))
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
  #[inline(always)]
  fn square_repeated<const N: usize>(mut self) -> Self {
    for _ in 0..N {
      self = self.square();
    }
    self
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
    // Build compact all-one exponents, then assemble
    // p - 2 = 2^256 - 2^224 + 2^192 + 2^96 - 3. This fixed chain uses
    // 255 squarings and 12 general multiplications.
    let x3 = self.square().mul(self);
    let x7 = x3.square().mul(self);
    let x63 = x7.square_repeated::<3>().mul(x7);
    let x4095 = x63.square_repeated::<6>().mul(x63);
    let x32767 = x4095.square_repeated::<3>().mul(x7);
    let x65535 = x32767.square().mul(self);
    let x2_32_minus_1 = x65535.square_repeated::<16>().mul(x65535);
    let x2_47_minus_2_15 = x2_32_minus_1.square_repeated::<15>();
    let x2_47_minus_1 = x2_47_minus_2_15.mul(x32767);
    let tail = x2_47_minus_2_15
      .square_repeated::<17>()
      .mul(self)
      .square_repeated::<143>()
      .mul(x2_47_minus_1)
      .square_repeated::<47>();
    x2_47_minus_1.mul(tail).square_repeated::<2>().mul(self)
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
  #[cfg(test)]
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
  fn select_generator_comb<const SHIFT: usize>(digit: usize) -> (Self, u64) {
    assert!(SHIFT == 0 || SHIFT == 13 || SHIFT == 25);
    let table_x = if SHIFT == 0 {
      &P256_SIGNING_GENERATOR_COMB_X
    } else if SHIFT == 13 {
      &P256_SIGNING_GENERATOR_COMB_SHIFT_13_X
    } else {
      &P256_SIGNING_GENERATOR_COMB_SHIFT_25_X
    };
    let table_y = if SHIFT == 0 {
      &P256_SIGNING_GENERATOR_COMB_Y
    } else if SHIFT == 13 {
      &P256_SIGNING_GENERATOR_COMB_SHIFT_13_Y
    } else {
      &P256_SIGNING_GENERATOR_COMB_SHIFT_25_Y
    };

    #[cfg(target_arch = "riscv64")]
    {
      let mut x = [0u64; 4];
      let mut y = [0u64; 4];
      let mut index = 0;
      while index < table_x.len() {
        let candidate_x = table_x[index];
        let candidate_y = table_y[index];
        // SECURITY: Keep the equality mask opaque so LLVM retains the full
        // table scan instead of loading from a secret-derived address.
        let mask = core::hint::black_box(mask_equal_usize(digit, index));
        x[0] |= candidate_x.0[0] & mask;
        x[1] |= candidate_x.0[1] & mask;
        x[2] |= candidate_x.0[2] & mask;
        x[3] |= candidate_x.0[3] & mask;
        y[0] |= candidate_y.0[0] & mask;
        y[1] |= candidate_y.0[1] & mask;
        y[2] |= candidate_y.0[2] & mask;
        y[3] |= candidate_y.0[3] & mask;
        index = index.strict_add(1);
      }
      (
        Self {
          x: FieldElement::from_montgomery(Uint(x)),
          y: FieldElement::from_montgomery(Uint(y)),
        },
        mask_equal_usize(digit, 0),
      )
    }

    #[cfg(not(target_arch = "riscv64"))]
    {
      let mut x = Uint(table_x[0].0);
      let mut y = Uint(table_y[0].0);
      for (index, (&candidate_x, &candidate_y)) in table_x.iter().zip(table_y.iter()).enumerate() {
        let mask = mask_equal_usize(digit, index);
        x = Uint::select(x, Uint(candidate_x.0), mask);
        y = Uint::select(y, Uint(candidate_y.0), mask);
      }
      (
        Self {
          x: FieldElement::from_montgomery(x),
          y: FieldElement::from_montgomery(y),
        },
        mask_equal_usize(digit, 0),
      )
    }
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

  /// Doubles a point whose coordinates use the Jacobian convention.
  fn double_jacobian(self) -> Self {
    let delta = self.z.square();
    let gamma = self.y.square();
    let beta = self.x.mul(gamma);
    let alpha = self.x.sub(delta).mul(self.x.add(delta)).triple();
    let x = alpha.square().sub(beta.double().double().double());
    let z = self.y.add(self.z).square().sub(gamma).sub(delta);
    let y = alpha
      .mul(beta.double().double().sub(x))
      .sub(gamma.square().double().double().double());
    let doubled = Self {
      x,
      y,
      z,
      infinity_mask: 0,
    };
    Self::select(doubled, Self::infinity(), self.infinity_mask() | self.y.0.zero_mask())
  }

  /// Adds an affine point to a Jacobian point when finite operands are
  /// known to be neither equal nor opposite.
  fn add_mixed_nonexceptional_jacobian(self, rhs: Affine, rhs_infinity_mask: u64) -> Self {
    let z1z1 = self.z.square();
    let u2 = rhs.x.mul(z1z1);
    let s2 = rhs.y.mul(self.z).mul(z1z1);
    let h = u2.sub(self.x);
    let r = s2.sub(self.y).double();
    let hh = h.square();
    let i = hh.double().double();
    let j = h.mul(i);
    let v = self.x.mul(i);
    let x = r.square().sub(j).sub(v.double());
    let y = r.mul(v.sub(x)).sub(self.y.mul(j).double());
    let z = self.z.add(h).square().sub(z1z1).sub(hh);
    let added = Self {
      x,
      y,
      z,
      infinity_mask: 0,
    };
    let with_self_infinity = Self::select(added, Self::from_affine(rhs), self.infinity_mask());
    Self::select(with_self_infinity, self, rhs_infinity_mask)
  }

  #[cfg(test)]
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
  #[cfg(test)]
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

  fn to_affine_jacobian(&self) -> Affine {
    let inverse_z = self.0.z.invert();
    let inverse_z_squared = inverse_z.square();
    Affine {
      x: self.0.x.mul(inverse_z_squared),
      y: self.0.y.mul(inverse_z_squared).mul(inverse_z),
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
  scalar_mul_generator_portable(scalar).to_affine_jacobian()
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
    all(target_arch = "x86_64", target_os = "linux")
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
  #[inline(always)]
  fn digit(scalar: &Scalar, row: usize) -> usize {
    let mut digit = 0usize;
    for column in 0..P256_SIGNING_COMB_WIDTH {
      let bit = row.strict_add(column.strict_mul(FIXED_BASE_COMB_ROWS));
      digit |= usize::from((scalar.0.bit_mask(bit) & 1).to_le_bytes()[0]) << column;
    }
    digit
  }

  const MIDDLE_ROW: usize = 13;
  const HIGH_ROW: usize = 25;
  const LOW_ROWS: usize = MIDDLE_ROW;

  let top_row = LOW_ROWS.strict_sub(1);
  let (selected, infinity) = Affine::select_generator_comb::<0>(digit(scalar, top_row));
  let mut acc = Projective::from_affine(selected);
  acc.infinity_mask = infinity;

  for row in (0..top_row).rev() {
    acc = acc.double_jacobian();
    let (selected, infinity) = Affine::select_generator_comb::<0>(digit(scalar, row));
    acc = acc.add_mixed_nonexceptional_jacobian(selected, infinity);

    let middle_row = row.strict_add(MIDDLE_ROW);
    let (selected, infinity) = Affine::select_generator_comb::<13>(digit(scalar, middle_row));
    acc = acc.add_mixed_nonexceptional_jacobian(selected, infinity);

    let high_row = row.strict_add(HIGH_ROW);
    let (selected, infinity) = Affine::select_generator_comb::<25>(digit(scalar, high_row));

    // At each addition, the accumulator and table point represent disjoint
    // scalar-bit positions: the low table contributes offset zero modulo 37,
    // and the other tables contribute offsets 13 and 25. Every partial sum is
    // a subset of the canonical scalar shifted right by `row`, so it remains
    // below the group order. Finite operands are therefore neither equal nor
    // opposite.
    acc = acc.add_mixed_nonexceptional_jacobian(selected, infinity);
  }
  SecretProjective(acc)
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
  all(rscrypto_internal, feature = "diag"),
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
  let (selected, _) = Affine::select_generator_comb::<0>(usize::from(digit));
  let mut output = [0u64; 8];
  output[..4].copy_from_slice(&selected.x.0.0);
  output[4..].copy_from_slice(&selected.y.0.0);
  output
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

#[cfg(any(test, target_arch = "riscv32", target_arch = "s390x"))]
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

#[cfg(any(test, target_arch = "riscv64"))]
#[derive(Clone, Copy)]
struct Riscv64MulLimb {
  normalized: u64,
  padding_mask: u64,
  shift_low: u64,
  shift_high: u64,
}

#[cfg(any(test, target_arch = "riscv64"))]
impl Riscv64MulLimb {
  #[inline(always)]
  fn new(value: u64) -> Self {
    const HIGH_BIT: u64 = 1 << 63;
    let padding_mask = 0u64.wrapping_sub((value >> 63) ^ 1);
    Self {
      normalized: core::hint::black_box(value | HIGH_BIT),
      padding_mask,
      shift_low: value << 63,
      shift_high: value >> 1,
    }
  }
}

#[cfg(any(test, target_arch = "riscv64"))]
#[inline(always)]
fn ct_mul_riscv64_limbs(left: Riscv64MulLimb, right: Riscv64MulLimb) -> (u64, u64) {
  const HIGH_BIT: u64 = 1 << 63;

  let product = u128::from(left.normalized).strict_mul(u128::from(right.normalized));
  let (product_low, product_high) = split_u128(product);

  // Subtract the two conditional 2^63 cross terms and their 2^126
  // intersection to recover the original product.
  let (product_low, borrow) = sbb_limb(product_low, right.shift_low & left.padding_mask, 0);
  let (product_high, borrow) = sbb_limb(product_high, right.shift_high & left.padding_mask, borrow);
  debug_assert_eq!(borrow, 0);
  let (product_low, borrow) = sbb_limb(product_low, left.shift_low & right.padding_mask, 0);
  let (product_high, borrow) = sbb_limb(product_high, left.shift_high & right.padding_mask, borrow);
  debug_assert_eq!(borrow, 0);
  let intersection = (HIGH_BIT >> 1) & left.padding_mask & right.padding_mask;
  let (product_high, borrow) = sbb_limb(product_high, intersection, 0);
  debug_assert_eq!(borrow, 0);

  (product_low, product_high)
}

#[cfg(test)]
#[inline(never)]
fn ct_mul_u64_wide_riscv64(left: u64, right: u64) -> (u64, u64) {
  ct_mul_riscv64_limbs(Riscv64MulLimb::new(left), Riscv64MulLimb::new(right))
}

#[inline(always)]
#[cfg(any(test, not(target_arch = "riscv64")))]
fn mul_u64_wide(left: u64, right: u64) -> (u64, u64) {
  #[cfg(target_arch = "riscv64")]
  {
    ct_mul_u64_wide_riscv64(left, right)
  }

  #[cfg(any(target_arch = "riscv32", target_arch = "s390x"))]
  {
    ct_mul_u64_wide(left, right)
  }

  #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x")))]
  {
    split_u128(u128::from(left).strict_mul(u128::from(right)))
  }
}

#[inline(always)]
#[cfg(any(test, not(any(target_arch = "riscv32", target_arch = "s390x"))))]
fn split_u128(value: u128) -> (u64, u64) {
  let [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15] = value.to_le_bytes();
  (
    u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7]),
    u64::from_le_bytes([b8, b9, b10, b11, b12, b13, b14, b15]),
  )
}

#[inline(always)]
fn mac_wide(acc: u64, product_low: u64, product_high: u64, carry: u64) -> (u64, u64) {
  let (result, carry0) = adc_limb(product_low, acc, 0);
  let (result, carry1) = adc_limb(result, carry, 0);
  let (high, overflow0) = adc_limb(product_high, carry0, 0);
  let (high, overflow1) = adc_limb(high, carry1, 0);
  debug_assert_eq!(overflow0 | overflow1, 0);
  (result, high)
}

#[inline(always)]
#[cfg(any(test, not(target_arch = "riscv64")))]
fn mac_limb(acc: u64, left: u64, right: u64, carry: u64) -> (u64, u64) {
  let (product_low, product_high) = mul_u64_wide(left, right);
  mac_wide(acc, product_low, product_high, carry)
}

#[cfg(any(test, target_arch = "riscv64"))]
#[inline(always)]
fn mac_riscv64_limb(acc: u64, left: Riscv64MulLimb, right: Riscv64MulLimb, carry: u64) -> (u64, u64) {
  let (product_low, product_high) = ct_mul_riscv64_limbs(left, right);
  mac_wide(acc, product_low, product_high, carry)
}

#[cfg(any(test, target_arch = "riscv64"))]
#[inline(always)]
fn mul_riscv64_limb(left: Riscv64MulLimb, right: Riscv64MulLimb) -> (u64, u64) {
  ct_mul_riscv64_limbs(left, right)
}

#[inline(always)]
fn mac_p256_modulus_limb_1_plus_value(acc: u64, value: u64) -> (u64, u64) {
  // value * (2^32 - 1) + value = value * 2^32.
  let (low, carry) = adc_limb(acc, value << 32, 0);
  (low, (value >> 32).strict_add(carry))
}

#[inline(always)]
fn mac_p256_modulus_limb_3(acc: u64, value: u64, carry: u64) -> (u64, u64) {
  // FIELD_MODULUS[3] = 2^64 - 2^32 + 1.
  let (product_low, borrow) = sbb_limb(0, value << 32, 0);
  let product_high = value.wrapping_sub(value >> 32).wrapping_sub(borrow);
  let (product_low, product_carry) = adc_limb(product_low, value, 0);
  let product_high = product_high.wrapping_add(product_carry);
  mac_wide(acc, product_low, product_high, carry)
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
struct WideAccumulator {
  low: u64,
  high: u64,
  overflow: u64,
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
impl WideAccumulator {
  #[inline(always)]
  fn add(&mut self, low: u64, high: u64) {
    let (low, carry) = adc_limb(self.low, low, 0);
    let (high, carry) = adc_limb(self.high, high, carry);
    self.low = low;
    self.high = high;
    self.overflow = self.overflow.strict_add(carry);
  }

  #[inline(always)]
  fn add_double(&mut self, low: u64, high: u64) {
    self.add(low, high);
    self.add(low, high);
  }

  #[inline(always)]
  fn take_limb(&mut self) -> u64 {
    let limb = self.low;
    self.low = self.high;
    self.high = self.overflow;
    self.overflow = 0;
    limb
  }
}

#[inline(always)]
fn multiply_256_wide<T: Copy>(left: [T; 4], right: [T; 4], mac: impl Fn(u64, T, T, u64) -> (u64, u64)) -> [u64; 8] {
  let (w0, carry) = mac(0, left[0], right[0], 0);
  let (w1, carry) = mac(0, left[0], right[1], carry);
  let (w2, carry) = mac(0, left[0], right[2], carry);
  let (w3, w4) = mac(0, left[0], right[3], carry);
  let (w1, carry) = mac(w1, left[1], right[0], 0);
  let (w2, carry) = mac(w2, left[1], right[1], carry);
  let (w3, carry) = mac(w3, left[1], right[2], carry);
  let (w4, w5) = mac(w4, left[1], right[3], carry);
  let (w2, carry) = mac(w2, left[2], right[0], 0);
  let (w3, carry) = mac(w3, left[2], right[1], carry);
  let (w4, carry) = mac(w4, left[2], right[2], carry);
  let (w5, w6) = mac(w5, left[2], right[3], carry);
  let (w3, carry) = mac(w3, left[3], right[0], 0);
  let (w4, carry) = mac(w4, left[3], right[1], carry);
  let (w5, carry) = mac(w5, left[3], right[2], carry);
  let (w6, w7) = mac(w6, left[3], right[3], carry);
  [w0, w1, w2, w3, w4, w5, w6, w7]
}

#[cfg(any(test, target_arch = "riscv64"))]
#[inline(always)]
fn riscv64_mul_limbs(value: Uint) -> [Riscv64MulLimb; 4] {
  [
    Riscv64MulLimb::new(value.0[0]),
    Riscv64MulLimb::new(value.0[1]),
    Riscv64MulLimb::new(value.0[2]),
    Riscv64MulLimb::new(value.0[3]),
  ]
}

#[cfg(any(test, target_arch = "riscv64"))]
#[inline(never)]
fn montgomery_mul_riscv64(left: Uint, right: Uint) -> Uint {
  montgomery_reduce(multiply_256_wide(
    riscv64_mul_limbs(left),
    riscv64_mul_limbs(right),
    mac_riscv64_limb,
  ))
}

#[cfg(any(test, target_arch = "riscv64"))]
#[inline(never)]
fn montgomery_square_riscv64(value: Uint) -> Uint {
  montgomery_reduce(square_256_wide(riscv64_mul_limbs(value), mul_riscv64_limb))
}

fn montgomery_mul(left: Uint, right: Uint) -> Uint {
  #[cfg(target_arch = "riscv64")]
  {
    montgomery_mul_riscv64(left, right)
  }

  #[cfg(not(target_arch = "riscv64"))]
  {
    montgomery_reduce(multiply_256_wide(left.0, right.0, mac_limb))
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
#[inline(always)]
fn square_256_wide<T: Copy>(value: [T; 4], mul: impl Fn(T, T) -> (u64, u64)) -> [u64; 8] {
  let mut acc = WideAccumulator {
    low: 0,
    high: 0,
    overflow: 0,
  };

  let (p00_low, p00_high) = mul(value[0], value[0]);
  acc.add(p00_low, p00_high);
  let w0 = acc.take_limb();

  let (p01_low, p01_high) = mul(value[0], value[1]);
  acc.add_double(p01_low, p01_high);
  let w1 = acc.take_limb();

  let (p02_low, p02_high) = mul(value[0], value[2]);
  let (p11_low, p11_high) = mul(value[1], value[1]);
  acc.add_double(p02_low, p02_high);
  acc.add(p11_low, p11_high);
  let w2 = acc.take_limb();

  let (p03_low, p03_high) = mul(value[0], value[3]);
  let (p12_low, p12_high) = mul(value[1], value[2]);
  acc.add_double(p03_low, p03_high);
  acc.add_double(p12_low, p12_high);
  let w3 = acc.take_limb();

  let (p13_low, p13_high) = mul(value[1], value[3]);
  let (p22_low, p22_high) = mul(value[2], value[2]);
  acc.add_double(p13_low, p13_high);
  acc.add(p22_low, p22_high);
  let w4 = acc.take_limb();

  let (p23_low, p23_high) = mul(value[2], value[3]);
  acc.add_double(p23_low, p23_high);
  let w5 = acc.take_limb();

  let (p33_low, p33_high) = mul(value[3], value[3]);
  acc.add(p33_low, p33_high);
  let w6 = acc.take_limb();
  let w7 = acc.take_limb();
  debug_assert_eq!(acc.low | acc.high | acc.overflow, 0);

  [w0, w1, w2, w3, w4, w5, w6, w7]
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
fn montgomery_square(value: Uint) -> Uint {
  #[cfg(target_arch = "riscv64")]
  {
    montgomery_square_riscv64(value)
  }

  #[cfg(not(target_arch = "riscv64"))]
  {
    montgomery_reduce(square_256_wide(value.0, mul_u64_wide))
  }
}

fn montgomery_reduce(limbs: [u64; 8]) -> Uint {
  let [r0, r1, r2, r3, r4, r5, r6, r7] = limbs;
  let (r1, carry) = mac_p256_modulus_limb_1_plus_value(r1, r0);
  let (r2, carry) = adc_limb(r2, 0, carry);
  let (r3, carry) = mac_p256_modulus_limb_3(r3, r0, carry);
  let (r4, carry2) = adc_limb(r4, 0, carry);
  let (r2, carry) = mac_p256_modulus_limb_1_plus_value(r2, r1);
  let (r3, carry) = adc_limb(r3, 0, carry);
  let (r4, carry) = mac_p256_modulus_limb_3(r4, r1, carry);
  let (r5, carry2) = adc_limb(r5, carry2, carry);
  let (r3, carry) = mac_p256_modulus_limb_1_plus_value(r3, r2);
  let (r4, carry) = adc_limb(r4, 0, carry);
  let (r5, carry) = mac_p256_modulus_limb_3(r5, r2, carry);
  let (r6, carry2) = adc_limb(r6, carry2, carry);
  let (r4, carry) = mac_p256_modulus_limb_1_plus_value(r4, r3);
  let (r5, carry) = adc_limb(r5, 0, carry);
  let (r6, carry) = mac_p256_modulus_limb_3(r6, r3, carry);
  let (r7, r8) = adc_limb(r7, carry2, carry);
  subtract_modulus_once([r4, r5, r6, r7, r8])
}

fn subtract_modulus_once(limbs: [u64; 5]) -> Uint {
  let p = FIELD_MODULUS.0;
  let (w0, borrow) = sbb_limb(limbs[0], p[0], 0);
  let (w1, borrow) = sbb_limb(limbs[1], p[1], borrow);
  let (w2, borrow) = sbb_limb(limbs[2], p[2], borrow);
  let (w3, borrow) = sbb_limb(limbs[3], p[3], borrow);
  // Montgomery inputs are below p^2, so reduction is below 2p and the high
  // limb is at most one. A nonzero high limb always authorizes subtraction.
  debug_assert!(limbs[4] <= 1);
  let borrow = borrow & (limbs[4] ^ 1);
  let mask = 0u64.wrapping_sub(borrow);
  let (w0, carry) = adc_limb(w0, mask, 0);
  let (w1, carry) = adc_limb(w1, p[1] & mask, carry);
  let (w2, carry) = adc_limb(w2, 0, carry);
  let (w3, _) = adc_limb(w3, p[3] & mask, carry);
  Uint([w0, w1, w2, w3])
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
    let edges = [
      0,
      1,
      0x5555_5555_5555_5555,
      0x7fff_ffff_ffff_ffff,
      0x8000_0000_0000_0000,
      0xaaaa_aaaa_aaaa_aaaa,
      u64::MAX - 1,
      u64::MAX,
    ];
    for left in edges {
      for right in edges {
        let product = u128::from(left).strict_mul(u128::from(right));
        assert_eq!(super::ct_mul_u64_wide(left, right), super::split_u128(product));
        assert_eq!(super::ct_mul_u64_wide_riscv64(left, right), super::split_u128(product));
      }
    }

    let field_edges = [
      super::Uint::ZERO,
      super::Uint::ONE,
      super::GENERATOR_X,
      super::Uint([0xffff_ffff_ffff_fffe, 0x0000_0000_ffff_ffff, 0, 0xffff_ffff_0000_0001]),
    ];
    for left in field_edges {
      assert!(super::montgomery_square_riscv64(left) == super::montgomery_square(left));
      for right in field_edges {
        assert!(super::montgomery_mul_riscv64(left, right) == super::montgomery_mul(left, right));
      }
    }
  }

  #[test]
  fn field_inversion_is_multiplicative_inverse() {
    assert!(super::FieldElement::zero().invert() == super::FieldElement::zero());
    for value in [
      super::Uint::ONE,
      super::GENERATOR_X,
      super::GENERATOR_Y,
      super::Uint([0xffff_ffff_ffff_fffe, 0x0000_0000_ffff_ffff, 0, 0xffff_ffff_0000_0001]),
    ] {
      let value = super::FieldElement::from_uint(value);
      assert!(value.mul(value.invert()) == super::FieldElement::one());
    }
  }

  #[test]
  fn fixed_base_comb_matches_reference_and_arbitrary_point_window() {
    let table = super::precompute_public_table(super::Affine::generator());
    for bytes in [[0x11; 32], [0x42; 32], [0x7f; 32], [0xa5; 32]] {
      let scalar = super::Scalar::from_bytes(&bytes);
      let fixed = super::scalar_mul_generator_portable(&scalar)
        .to_affine_jacobian()
        .encode_sec1();
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

  #[test]
  fn split_generator_comb_tables_match_rustcrypto() {
    use p256::elliptic_curve::sec1::ToSec1Point as _;

    fn assert_table<const SHIFT: usize>() {
      for digit in 1..(1usize << super::P256_SIGNING_COMB_WIDTH) {
        let mut scalar = [0u8; 32];
        for column in 0..super::P256_SIGNING_COMB_WIDTH {
          if digit & (1 << column) != 0 {
            let bit = SHIFT.strict_add(column.strict_mul(super::FIXED_BASE_COMB_ROWS));
            scalar[31usize.strict_sub(bit / 8)] |= 1 << (bit % 8);
          }
        }
        let expected = p256::SecretKey::from_slice(&scalar)
          .expect("comb digit is a canonical nonzero P-256 scalar")
          .public_key()
          .to_sec1_point(false);
        let (selected, infinity) = super::Affine::select_generator_comb::<SHIFT>(digit);
        assert_eq!(infinity, 0);
        assert_eq!(selected.encode_sec1().as_slice(), expected.as_bytes());
      }
    }

    assert_table::<0>();
    assert_table::<13>();
    assert_table::<25>();
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
      let portable = super::scalar_mul_generator_portable(&scalar).to_affine_jacobian();
      let native = super::super::p256_core::scalar_mul_generator_words(&scalar.0.0);
      assert_eq!(native, affine_words(portable), "fixed-base scalar {scalar_bytes:02x?}");

      for peer_bytes in cases {
        let peer_scalar = super::Scalar::from_bytes(&peer_bytes);
        let peer = super::scalar_mul_generator_portable(&peer_scalar).to_affine_jacobian();
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
      *case = super::scalar_mul_generator_portable(&scalar)
        .to_affine_jacobian()
        .encode_sec1();
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
