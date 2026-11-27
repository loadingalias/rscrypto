//! Multiprecision and Montgomery arithmetic.

use super::*;

impl RsaPublicKey {
  /// Allocate scratch for repeated public operations, with this key's `R² mod n` precomputed.
  #[inline]
  #[must_use]
  pub fn public_scratch(&self) -> RsaPublicScratch {
    RsaPublicScratch::with_montgomery_r2(self)
  }

  /// Apply the RSA public operation to a fixed-width representative.
  ///
  /// This is the raw `m = s^e mod n` primitive used by signature verification
  /// padding code. It does not parse or check PKCS #1 v1.5/PSS padding.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPublicOpError`] if `input` or `out` is not exactly the
  /// modulus length, or if `input >= n`.
  #[must_use = "RSA public-operation failure must be checked; a dropped Result silently accepts an invalid \
                representative"]
  pub fn public_operation(&self, input: &[u8], out: &mut [u8]) -> Result<(), RsaPublicOpError> {
    let mut scratch = RsaPublicScratch::new(self);
    self.public_operation_with_scratch(input, out, &mut scratch)
  }

  /// Apply the RSA public operation using caller-owned scratch space.
  ///
  /// Reusing scratch avoids allocation in steady-state verification loops.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPublicOpError`] if lengths are invalid, the representative is
  /// out of range, or `scratch` was allocated for a different modulus width.
  #[must_use = "RSA public-operation failure must be checked; a dropped Result silently accepts an invalid \
                representative"]
  pub fn public_operation_with_scratch(
    &self,
    input: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), RsaPublicOpError> {
    let result = self.public_operation_by_division(input, out, scratch);
    clear_output_on_error(result, out)
  }
}

impl RsaPublicScratch {
  /// Allocate scratch space for `key`.
  ///
  /// Construction only allocates and serves any key with the same modulus width.
  /// [`RsaPublicKey::public_scratch`] also precomputes that key's `R² mod n` for reuse.
  #[must_use]
  pub fn new(key: &RsaPublicKey) -> Self {
    let limbs = key.modulus.limbs.len();
    let bytes = key.modulus.bytes.len();

    // Unbound, an imported key enters the Montgomery domain by long division. `public_scratch`
    // and private keys bind a key's `R² mod n` in the `montgomery` slots; the all-zero modulus
    // slot binds no key, because every RSA modulus is odd.
    Self {
      montgomery: vec![0u64; limbs.strict_mul(2)].into_boxed_slice(),
      limbs: vec![0u64; limbs.strict_mul(6).strict_add(2)].into_boxed_slice(),
      bytes: vec![0u8; bytes.strict_mul(3)].into_boxed_slice(),
      limb_count: limbs,
      byte_count: bytes,
    }
  }

  pub(super) fn arithmetic_scratch(&mut self) -> RsaPublicArithmeticScratch<'_> {
    split_limb_scratch(&mut self.limbs, self.limb_count, &mut self.montgomery)
  }

  pub(super) fn split_all(&mut self) -> (RsaPublicArithmeticScratch<'_>, &mut [u8], &mut [u8], &mut [u8]) {
    let arithmetic_scratch = split_limb_scratch(&mut self.limbs, self.limb_count, &mut self.montgomery);
    let (encoded, db, db_mask) = split_byte_scratch(&mut self.bytes, self.byte_count);
    (arithmetic_scratch, encoded, db, db_mask)
  }
}

impl fmt::Debug for RsaPublicScratch {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("RsaPublicScratch")
      .field("limbs", &self.limb_count)
      .finish_non_exhaustive()
  }
}

fn split_limb_scratch<'a>(
  limbs: &'a mut [u64],
  limb_count: usize,
  montgomery: &'a mut [u64],
) -> RsaPublicArithmeticScratch<'a> {
  let (modulus, r2) = montgomery.split_at_mut(limb_count);
  let t_len = limb_count.strict_mul(2).strict_add(2);
  let (t, rest) = limbs.split_at_mut(t_len);
  let (x, rest) = rest.split_at_mut(limb_count);
  let (tmp, rest) = rest.split_at_mut(limb_count);
  let (base, rest) = rest.split_at_mut(limb_count);
  let (acc, _) = rest.split_at_mut(limb_count);
  RsaPublicArithmeticScratch {
    modulus,
    r2,
    t,
    x,
    tmp,
    base,
    acc,
  }
}

fn split_byte_scratch(bytes: &mut [u8], byte_count: usize) -> (&mut [u8], &mut [u8], &mut [u8]) {
  let (encoded, rest) = bytes.split_at_mut(byte_count);
  let (db, db_mask) = rest.split_at_mut(byte_count);
  (encoded, db, db_mask)
}

impl Drop for RsaPublicModulus {
  fn drop(&mut self) {
    ct::zeroize(&mut self.bytes);
    ct::zeroize_words(&mut self.limbs);
    #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
    ct::zeroize_words(&mut self.n0_256);
    if let Some(r2) = &mut self.r2 {
      ct::zeroize_words(r2);
    }
    ct::zeroize_words(core::slice::from_mut(&mut self.bits));
    ct::zeroize_words(core::slice::from_mut(&mut self.n0));
  }
}

impl PartialEq for RsaPublicModulus {
  fn eq(&self, other: &Self) -> bool {
    self.bits == other.bits && self.bytes == other.bytes
  }
}

impl Eq for RsaPublicModulus {}

impl Hash for RsaPublicModulus {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.bytes.hash(state);
    self.bits.hash(state);
  }
}

macro_rules! public_exponent_schedule {
  ($exponent:expr, $square:expr, $multiply:expr $(,)?) => {
    match $exponent {
      3 => {
        $square;
        $multiply;
      }
      17 => {
        for _ in 0..4 {
          $square;
        }
        $multiply;
      }
      65_537 => {
        for _ in 0..16 {
          $square;
        }
        $multiply;
      }
      other => {
        let top_bit = 63usize.strict_sub(other.leading_zeros() as usize);
        for bit in (0..top_bit).rev() {
          $square;
          if (other >> bit) & 1 == 1 {
            $multiply;
          }
        }
      }
    }
  };
}

impl RsaPublicModulus {
  pub(super) fn new(bytes: &[u8], bits: usize) -> Self {
    Self::new_inner(bytes, bits, false)
  }

  pub(super) fn new_with_montgomery_r2(bytes: &[u8], bits: usize) -> Self {
    Self::new_inner(bytes, bits, true)
  }

  fn new_inner(bytes: &[u8], bits: usize, precompute_r2: bool) -> Self {
    let limbs = limbs_from_be(bytes);
    let n0 = montgomery_n0(limbs.first().copied().unwrap_or(1));
    #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
    let n0_256 = if rsa_aarch64_mont::supports(limbs.len()) {
      rsa_aarch64_mont::n0_256(&limbs, n0)
    } else {
      [0; 4]
    };
    let mut modulus = Self {
      bytes: Box::from(bytes),
      limbs: limbs.into_boxed_slice(),
      r2: None,
      bits,
      n0,
      #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
      n0_256,
    };
    if precompute_r2 {
      modulus.r2 = Some(public_montgomery_r2(&modulus));
    }
    modulus
  }

  pub(super) fn montgomery_r2(&self) -> Option<&[u64]> {
    self.r2.as_deref()
  }

  /// Write `R² mod n` into `r2`, copying the precomputed value when this modulus has one.
  fn montgomery_r2_into(&self, r2: &mut [u64], tmp: &mut [u64], t: &mut [u64]) {
    match self.montgomery_r2() {
      Some(precomputed) => r2.copy_from_slice(precomputed),
      None => public_montgomery_r2_into(r2, self, tmp, t),
    }
  }

  pub(super) fn public_operation(
    &self,
    exponent: RsaPublicExponent,
    input: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), RsaPublicOpError> {
    self.public_operation_with_arithmetic_scratch(exponent, input, out, scratch.arithmetic_scratch())
  }

  fn public_operation_with_arithmetic_scratch(
    &self,
    exponent: RsaPublicExponent,
    input: &[u8],
    out: &mut [u8],
    mut scratch: RsaPublicArithmeticScratch<'_>,
  ) -> Result<(), RsaPublicOpError> {
    self.bind_public_arithmetic_scratch(&mut scratch)?;
    let RsaPublicArithmeticScratch {
      modulus: _,
      r2,
      t,
      x,
      tmp,
      base,
      acc,
    } = scratch;
    let limbs = self.limbs.len();
    if input.len() != self.bytes.len() || out.len() != self.bytes.len() {
      return Err(RsaPublicOpError::InvalidLength);
    }
    if x.len() != limbs
      || base.len() != limbs
      || acc.len() != limbs
      || tmp.len() != limbs
      || t.len() != limbs.strict_mul(2).strict_add(2)
    {
      return Err(RsaPublicOpError::InvalidScratch);
    }

    limbs_from_be_into(input, x);
    if cmp_limbs(x, &self.limbs) != core::cmp::Ordering::Less {
      return Err(RsaPublicOpError::RepresentativeOutOfRange);
    }

    if use_public_cios_montgomery(self) {
      #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
      if exponent.as_u64() == 65_537
        && rsa_aarch64_mont::supports(limbs)
        && t.len() >= rsa_aarch64_mont::scratch_words(limbs)
      {
        rsa_aarch64_mont::mul(base, x, r2, &self.limbs, &self.n0_256, t);
        copy_limbs(acc, base);
        for _ in 0..16 {
          rsa_aarch64_mont::square_in_place(acc, &self.limbs, &self.n0_256, t);
        }
        rsa_aarch64_mont::mul_in_place_left(acc, base, &self.limbs, &self.n0_256, t);
        mont_reduce_cios(base, acc, self, t);
        limbs_to_be(base, out);
        return Ok(());
      }

      mont_mul_cios(base, x, r2, self, t);
      copy_limbs(acc, base);

      public_exponent_schedule!(
        exponent.as_u64(),
        mont_square_cios_in_place(acc, tmp, self, t),
        mont_mul_cios_in_place_left(acc, base, tmp, self, t),
      );

      mont_reduce_cios(base, acc, self, t);
    } else {
      mont_mul(base, x, r2, self, t);
      copy_limbs(acc, base);

      public_exponent_schedule!(
        exponent.as_u64(),
        mont_square_in_place(acc, tmp, self, t),
        mont_mul_in_place_left(acc, base, tmp, self, t),
      );

      mont_reduce(base, acc, self, t);
    }

    limbs_to_be(base, out);
    Ok(())
  }

  fn bind_public_arithmetic_scratch(
    &self,
    scratch: &mut RsaPublicArithmeticScratch<'_>,
  ) -> Result<(), RsaPublicOpError> {
    let limbs = self.limbs.len();
    if scratch.modulus.len() != limbs || scratch.r2.len() != limbs {
      return Err(RsaPublicOpError::InvalidScratch);
    }
    if scratch.modulus != self.limbs.as_ref() {
      self.montgomery_r2_into(scratch.r2, scratch.tmp, scratch.t);
      scratch.modulus.copy_from_slice(&self.limbs);
    }
    Ok(())
  }
}

pub(super) fn limbs_from_be(bytes: &[u8]) -> Vec<u64> {
  let limbs = bytes.len().strict_add(7) / 8;
  let mut out = vec![0u64; limbs];
  limbs_from_be_into(bytes, &mut out);
  out
}

pub(super) fn limbs_from_be_into(bytes: &[u8], out: &mut [u64]) {
  let full_limbs = bytes.len() / 8;
  let leading = bytes.len() % 8;
  let needed_limbs = bytes.len().div_ceil(8);
  if out.len() != needed_limbs {
    out.fill(0);
  }

  for (limb_index, slot) in out.iter_mut().enumerate().take(full_limbs) {
    let start = bytes.len().strict_sub((limb_index.strict_add(1)).strict_mul(8));
    let mut limb = [0u8; 8];
    limb.copy_from_slice(&bytes[start..start.strict_add(8)]);
    *slot = u64::from_be_bytes(limb);
  }

  if leading != 0 {
    let mut limb = 0u64;
    for &byte in &bytes[..leading] {
      limb = (limb << 8) | u64::from(byte);
    }
    out[full_limbs] = limb;
  }
}

pub(super) fn limbs_to_be(limbs: &[u64], out: &mut [u8]) {
  let full_limbs = out.len() / 8;
  let leading = out.len() % 8;

  for (limb_index, limb) in limbs.iter().copied().enumerate().take(full_limbs) {
    let start = out.len().strict_sub((limb_index.strict_add(1)).strict_mul(8));
    out[start..start.strict_add(8)].copy_from_slice(&limb.to_be_bytes());
  }

  if leading != 0 {
    let limb = limbs[full_limbs].to_be_bytes();
    out[..leading].copy_from_slice(&limb[8usize.strict_sub(leading)..]);
  }
}

pub(super) fn copy_limbs(dst: &mut [u64], src: &[u64]) {
  for (d, s) in dst.iter_mut().zip(src.iter().copied()) {
    *d = s;
  }
}

pub(super) fn is_zero_unsigned_be(bytes: &[u8]) -> bool {
  let mut acc = 0u8;
  for &byte in bytes {
    acc |= byte;
  }
  acc == 0
}

pub(super) fn is_canonical_positive_unsigned_be(bytes: &[u8]) -> bool {
  matches!(bytes.first(), Some(&first) if first != 0)
}

#[cfg(feature = "getrandom")]
pub(super) fn unsigned_be_cmp(left: &[u8], right: &[u8]) -> core::cmp::Ordering {
  match left.len().cmp(&right.len()) {
    core::cmp::Ordering::Equal => left.cmp(right),
    ordering => ordering,
  }
}

fn ct_lt_u8(left: u8, right: u8) -> u8 {
  ((u16::from(left).wrapping_sub(u16::from(right)) >> 8) & 1) as u8
}

pub(super) fn ct_unsigned_be_lt_public_shape(left: &[u8], right: &[u8]) -> bool {
  match left.len().cmp(&right.len()) {
    core::cmp::Ordering::Less => true,
    core::cmp::Ordering::Greater => false,
    core::cmp::Ordering::Equal => ct_unsigned_be_lt_same_len(left, right) == 1,
  }
}

/// Return 1 when `left < right`, else 0, for big-endian values of one public length.
///
/// Every byte is compared, and the result is an arithmetic bit rather than a branch condition.
pub(super) fn ct_unsigned_be_lt_same_len(left: &[u8], right: &[u8]) -> u8 {
  debug_assert_eq!(left.len(), right.len());
  let mut lt = 0u8;
  let mut gt = 0u8;
  for (&left_byte, &right_byte) in left.iter().zip(right) {
    let undecided = (lt | gt) ^ 1;
    lt |= undecided & ct_lt_u8(left_byte, right_byte);
    gt |= undecided & ct_lt_u8(right_byte, left_byte);
  }
  lt
}

pub(super) fn private_component_modulus(bytes: &[u8]) -> Result<RsaPublicModulus, RsaPrivateOpError> {
  if bytes.is_empty() {
    return Err(RsaPrivateOpError::RepresentativeOutOfRange);
  }
  let limbs = limbs_from_be(bytes);
  let n0 = montgomery_n0(limbs.first().copied().unwrap_or(1));
  let r2 = private_montgomery_r2(bytes).map_err(|_| RsaPrivateOpError::RepresentativeOutOfRange)?;
  // A fixed sequence of multiplications and borrow chains: no branch depends on the prime.
  #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
  let n0_256 = if rsa_aarch64_mont::supports(limbs.len()) {
    rsa_aarch64_mont::n0_256(&limbs, n0)
  } else {
    [0; 4]
  };
  Ok(RsaPublicModulus {
    bytes: Box::from(bytes),
    limbs: limbs.into_boxed_slice(),
    r2: Some(r2),
    bits: bytes.len().strict_mul(8),
    n0,
    #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
    n0_256,
  })
}

pub(super) fn left_pad_be(src: &[u8], out: &mut [u8]) -> Result<(), RsaPrivateOpError> {
  if src.len() > out.len() {
    return Err(RsaPrivateOpError::InvalidLength);
  }
  out.fill(0);
  let offset = out.len().strict_sub(src.len());
  let Some(dst) = out.get_mut(offset..) else {
    return Err(RsaPrivateOpError::InvalidLength);
  };
  dst.copy_from_slice(src);
  Ok(())
}

pub(super) fn private_sub_mod_unsigned_be_to_fixed(
  left: &[u8],
  right: &[u8],
  modulus: &[u8],
  out: &mut [u8],
) -> Result<(), RsaPrivateOpError> {
  if left.len() != right.len() || left.len() != modulus.len() || out.len() != modulus.len() {
    return Err(RsaPrivateOpError::InvalidLength);
  }

  let mut borrow = 0u16;
  for index in 0..out.len() {
    let src = out.len().strict_sub(index).strict_sub(1);
    let subtrahend = u16::from(right[src]).strict_add(borrow);
    let difference = u16::from(left[src]).wrapping_sub(subtrahend);
    out[src] = difference.to_le_bytes()[0];
    borrow = (difference >> 15) & 1;
  }

  let mask = 0u16.wrapping_sub(borrow);
  let mut carry = 0u16;
  for index in 0..out.len() {
    let dst = out.len().strict_sub(index).strict_sub(1);
    let addend = u16::from(modulus[dst]) & mask;
    let sum = u16::from(out[dst]).strict_add(addend).strict_add(carry);
    out[dst] = sum.to_le_bytes()[0];
    carry = sum >> 8;
  }

  Ok(())
}

#[cfg(feature = "getrandom")]
pub(super) fn private_sub_unsigned_be_to_len(
  left: &[u8],
  right: &[u8],
  len: usize,
) -> Result<SecretBigEndianBuffer, RsaPrivateOpError> {
  if left.len() > len.strict_add(1) || right.len() > len || unsigned_be_cmp(left, right) == core::cmp::Ordering::Less {
    return Err(RsaPrivateOpError::InvalidLength);
  }

  let mut out = vec![0u8; len.strict_add(1)];
  let mut borrow = 0i16;
  for index in 0..out.len() {
    let left_byte = left
      .len()
      .checked_sub(index.strict_add(1))
      .and_then(|src| left.get(src))
      .copied()
      .unwrap_or(0);
    let right_byte = right
      .len()
      .checked_sub(index.strict_add(1))
      .and_then(|src| right.get(src))
      .copied()
      .unwrap_or(0);
    let mut difference = i16::from(left_byte)
      .strict_sub(i16::from(right_byte))
      .strict_sub(borrow);
    if difference < 0 {
      difference = difference.strict_add(256);
      borrow = 1;
    } else {
      borrow = 0;
    }
    let dst = out.len().strict_sub(index).strict_sub(1);
    out[dst] = difference.to_le_bytes()[0];
  }

  if borrow != 0 {
    return Err(RsaPrivateOpError::InvalidLength);
  }
  Ok(private_import_canonical_unsigned_be(out))
}

pub(super) fn private_valid_blinding_factor(factor: &[u8], modulus: &[u8]) -> bool {
  if factor.len() != modulus.len() {
    return false;
  }
  let mut nonzero = 0u8;
  let mut borrow = 0u16;
  for index in (0..factor.len()).rev() {
    nonzero |= factor[index];
    let (difference, first_borrow) = u16::from(factor[index]).overflowing_sub(u16::from(modulus[index]));
    let (_, carry_borrow) = difference.overflowing_sub(borrow);
    borrow = u16::from(first_borrow) | u16::from(carry_borrow);
  }
  ct_nonzero_u64(u64::from(nonzero)) & u64::from(borrow) == 1
}

pub(super) fn mod_mul_representatives(
  modulus: &RsaPublicModulus,
  left: &[u8],
  right: &[u8],
  out: &mut [u8],
) -> Result<(), RsaPrivateOpError> {
  let bytes = modulus.bytes.len();
  let limbs = modulus.limbs.len();
  if left.len() != bytes || right.len() != bytes || out.len() != bytes {
    return Err(RsaPrivateOpError::InvalidLength);
  }
  let r2 = modulus
    .montgomery_r2()
    .ok_or(RsaPrivateOpError::RepresentativeOutOfRange)?;

  let mut t = SecretLimbs::zeroed(limbs.strict_mul(2).strict_add(2));
  let mut left_limbs = SecretLimbs::zeroed(limbs);
  let mut right_limbs = SecretLimbs::zeroed(limbs);
  let mut left_mont = SecretLimbs::zeroed(limbs);
  let mut right_mont = SecretLimbs::zeroed(limbs);
  let mut product_mont = SecretLimbs::zeroed(limbs);
  let mut product = SecretLimbs::zeroed(limbs);

  limbs_from_be_into(left, left_limbs.as_mut_slice());
  limbs_from_be_into(right, right_limbs.as_mut_slice());
  if !ct_limbs_lt(left_limbs.as_slice(), &modulus.limbs) || !ct_limbs_lt(right_limbs.as_slice(), &modulus.limbs) {
    return Err(RsaPrivateOpError::RepresentativeOutOfRange);
  }

  private_mont_mul(
    left_mont.as_mut_slice(),
    left_limbs.as_slice(),
    r2,
    modulus,
    t.as_mut_slice(),
  );
  private_mont_mul(
    right_mont.as_mut_slice(),
    right_limbs.as_slice(),
    r2,
    modulus,
    t.as_mut_slice(),
  );
  private_mont_mul(
    product_mont.as_mut_slice(),
    left_mont.as_slice(),
    right_mont.as_slice(),
    modulus,
    t.as_mut_slice(),
  );
  private_mont_reduce(
    product.as_mut_slice(),
    product_mont.as_slice(),
    modulus,
    t.as_mut_slice(),
  );
  limbs_to_be(product.as_slice(), out);
  Ok(())
}

pub(super) fn mod_mul_representatives_with_scratch(
  modulus: &RsaPublicModulus,
  left: &[u8],
  right: &[u8],
  out: &mut [u8],
  scratch: &mut RsaPrivateMulScratch,
) -> Result<(), RsaPrivateOpError> {
  let bytes = modulus.bytes.len();
  let limbs = modulus.limbs.len();
  if left.len() != bytes || right.len() != bytes || out.len() != bytes {
    return Err(RsaPrivateOpError::InvalidLength);
  }
  let r2 = modulus
    .montgomery_r2()
    .ok_or(RsaPrivateOpError::RepresentativeOutOfRange)?;
  scratch.ensure_limb_count(limbs)?;

  let t_len = limbs.strict_mul(2).strict_add(2);
  let t = scratch
    .t
    .as_mut_slice()
    .get_mut(..t_len)
    .ok_or(RsaPrivateOpError::InvalidScratch)?;
  let left_limbs = scratch
    .left_limbs
    .as_mut_slice()
    .get_mut(..limbs)
    .ok_or(RsaPrivateOpError::InvalidScratch)?;
  let right_limbs = scratch
    .right_limbs
    .as_mut_slice()
    .get_mut(..limbs)
    .ok_or(RsaPrivateOpError::InvalidScratch)?;
  let left_mont = scratch
    .left_mont
    .as_mut_slice()
    .get_mut(..limbs)
    .ok_or(RsaPrivateOpError::InvalidScratch)?;
  let right_mont = scratch
    .right_mont
    .as_mut_slice()
    .get_mut(..limbs)
    .ok_or(RsaPrivateOpError::InvalidScratch)?;
  let product_mont = scratch
    .product_mont
    .as_mut_slice()
    .get_mut(..limbs)
    .ok_or(RsaPrivateOpError::InvalidScratch)?;
  let product = scratch
    .product
    .as_mut_slice()
    .get_mut(..limbs)
    .ok_or(RsaPrivateOpError::InvalidScratch)?;

  limbs_from_be_into(left, left_limbs);
  limbs_from_be_into(right, right_limbs);
  if !ct_limbs_lt(left_limbs, &modulus.limbs) || !ct_limbs_lt(right_limbs, &modulus.limbs) {
    return Err(RsaPrivateOpError::RepresentativeOutOfRange);
  }

  private_mont_mul(left_mont, left_limbs, r2, modulus, t);
  private_mont_mul(right_mont, right_limbs, r2, modulus, t);
  private_mont_mul(product_mont, left_mont, right_mont, modulus, t);
  private_mont_reduce(product, product_mont, modulus, t);
  limbs_to_be(product, out);
  Ok(())
}

/// `out = left · right · R⁻¹ mod n` for representatives below `n`, with `R = 2^(64·limbs)`: one
/// Montgomery multiplication, without conversion into or out of the Montgomery domain.
pub(super) fn mont_mul_representatives_with_scratch(
  modulus: &RsaPublicModulus,
  left: &[u8],
  right: &[u8],
  out: &mut [u8],
  scratch: &mut RsaPrivateMulScratch,
) -> Result<(), RsaPrivateOpError> {
  let bytes = modulus.bytes.len();
  let limbs = modulus.limbs.len();
  if left.len() != bytes || right.len() != bytes || out.len() != bytes {
    return Err(RsaPrivateOpError::InvalidLength);
  }
  scratch.ensure_limb_count(limbs)?;

  let t_len = limbs.strict_mul(2).strict_add(2);
  let t = scratch
    .t
    .as_mut_slice()
    .get_mut(..t_len)
    .ok_or(RsaPrivateOpError::InvalidScratch)?;
  let left_limbs = scratch
    .left_limbs
    .as_mut_slice()
    .get_mut(..limbs)
    .ok_or(RsaPrivateOpError::InvalidScratch)?;
  let right_limbs = scratch
    .right_limbs
    .as_mut_slice()
    .get_mut(..limbs)
    .ok_or(RsaPrivateOpError::InvalidScratch)?;
  let product = scratch
    .product
    .as_mut_slice()
    .get_mut(..limbs)
    .ok_or(RsaPrivateOpError::InvalidScratch)?;

  limbs_from_be_into(left, left_limbs);
  limbs_from_be_into(right, right_limbs);
  if !ct_limbs_lt(left_limbs, &modulus.limbs) || !ct_limbs_lt(right_limbs, &modulus.limbs) {
    return Err(RsaPrivateOpError::RepresentativeOutOfRange);
  }

  private_mont_mul(product, left_limbs, right_limbs, modulus, t);
  limbs_to_be(product, out);
  Ok(())
}

pub(super) fn private_mont_mul(out: &mut [u64], a: &[u64], b: &[u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  if use_cios_montgomery(modulus) {
    mont_mul_cios(out, a, b, modulus, t);
  } else {
    mont_mul(out, a, b, modulus, t);
  }
}

/// Square `value` in place with the same width selection as `private_mont_mul`.
///
/// The square kernels compute exactly `value² · R⁻¹ mod n`; `tmp` is scratch.
pub(super) fn private_mont_square_in_place(
  value: &mut [u64],
  tmp: &mut [u64],
  modulus: &RsaPublicModulus,
  t: &mut [u64],
) {
  if use_cios_montgomery(modulus) {
    mont_square_cios_in_place(value, tmp, modulus, t);
  } else {
    mont_square_in_place(value, tmp, modulus, t);
  }
}

pub(super) fn private_mont_reduce(out: &mut [u64], value: &[u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  if use_cios_montgomery(modulus) {
    mont_reduce_cios(out, value, modulus, t);
  } else {
    mont_reduce(out, value, modulus, t);
  }
}

#[inline(always)]
pub(super) fn split_u128(value: u128) -> (u64, u64) {
  (
    u64::try_from(value & u128::from(u64::MAX)).expect("masked low half of u128 must fit u64"),
    u64::try_from(value >> u64::BITS).expect("shifted high half of u128 must fit u64"),
  )
}

pub(super) fn private_product_add_unsigned_be_to_fixed(
  left: &[u8],
  right: &[u8],
  addend: &[u8],
  out: &mut [u8],
  scratch: &mut RsaPrivateMulScratch,
) -> Result<(), RsaPrivateOpError> {
  if addend.len() > out.len() {
    return Err(RsaPrivateOpError::InvalidLength);
  }

  let left_limb_count = left.len().strict_add(7) / 8;
  let right_limb_count = right.len().strict_add(7) / 8;
  let out_limb_count = out.len().strict_add(7) / 8;
  if left_limb_count.strict_add(right_limb_count) > out_limb_count {
    return Err(RsaPrivateOpError::RepresentativeOutOfRange);
  }
  scratch.ensure_limb_count(out_limb_count)?;

  let left_limbs = scratch
    .left_limbs
    .as_mut_slice()
    .get_mut(..left_limb_count)
    .ok_or(RsaPrivateOpError::InvalidScratch)?;
  let right_limbs = scratch
    .right_limbs
    .as_mut_slice()
    .get_mut(..right_limb_count)
    .ok_or(RsaPrivateOpError::InvalidScratch)?;
  let product = scratch
    .product
    .as_mut_slice()
    .get_mut(..out_limb_count)
    .ok_or(RsaPrivateOpError::InvalidScratch)?;

  limbs_from_be_into(left, left_limbs);
  limbs_from_be_into(right, right_limbs);
  product.fill(0);

  for (left_index, &left_limb) in left_limbs.iter().enumerate() {
    let mut carry = 0u64;
    for (right_index, &right_limb) in right_limbs.iter().enumerate() {
      let index = left_index.strict_add(right_index);
      let limb = product
        .get_mut(index)
        .ok_or(RsaPrivateOpError::RepresentativeOutOfRange)?;
      let acc = u128::from(*limb)
        .strict_add(u128::from(left_limb).strict_mul(u128::from(right_limb)))
        .strict_add(u128::from(carry));
      let (acc_low, acc_high) = split_u128(acc);
      *limb = acc_low;
      carry = acc_high;
    }

    let index = left_index.strict_add(right_limbs.len());
    let carry_limbs = product
      .get_mut(index..)
      .ok_or(RsaPrivateOpError::RepresentativeOutOfRange)?;
    for limb in carry_limbs {
      let acc = u128::from(*limb).strict_add(u128::from(carry));
      let (acc_low, acc_high) = split_u128(acc);
      *limb = acc_low;
      carry = acc_high;
    }
    if carry != 0 {
      return Err(RsaPrivateOpError::RepresentativeOutOfRange);
    }
  }

  limbs_to_be(product, out);

  let mut carry = 0u16;
  for index in 0..out.len() {
    let dst = out.len().strict_sub(index).strict_sub(1);
    let add_byte = addend
      .len()
      .checked_sub(index.strict_add(1))
      .and_then(|src| addend.get(src))
      .copied()
      .unwrap_or(0);
    let sum = u16::from(out[dst]).strict_add(u16::from(add_byte)).strict_add(carry);
    out[dst] = sum.to_le_bytes()[0];
    carry = sum >> 8;
  }

  if carry == 0 {
    Ok(())
  } else {
    Err(RsaPrivateOpError::RepresentativeOutOfRange)
  }
}

/// Reduce `value` modulo `modulus` into the fixed-width `out`, given
/// `value < modulus · 2^(8·cofactor_len)`.
///
/// When that bound is at most `modulus · R` for `R = 2^(64·L)` and `value` fits in `2·L` limbs,
/// split `value = high·R + low`. The bound gives `high < modulus`, so one Montgomery reduction
/// and one modular addition give `(low·R⁻¹ + high) mod modulus = value·R⁻¹ mod modulus`, and a
/// Montgomery multiplication by `R² mod modulus` removes `R⁻¹`. Other shapes use the bit-serial
/// reduction. The choice depends only on public lengths.
pub(super) fn private_reduce_unsigned_be_mod_to_fixed(
  value: &[u8],
  cofactor_len: usize,
  modulus: &RsaPublicModulus,
  out: &mut [u8],
  scratch: &mut RsaPrivateExponentScratch,
) -> Result<(), RsaPrivateOpError> {
  let limbs = modulus.limbs.len();
  let limb_bytes = limbs.strict_mul(8);
  let max_value_len = limb_bytes.strict_mul(2);
  if cofactor_len > limb_bytes || value.len() > max_value_len {
    return private_import_unsigned_be_mod_to_fixed(value, modulus, out, scratch);
  }
  if out.len() != modulus.bytes.len() {
    return Err(RsaPrivateOpError::InvalidLength);
  }
  let r2 = modulus
    .montgomery_r2()
    .ok_or(RsaPrivateOpError::RepresentativeOutOfRange)?;
  let RsaPrivateExponentWorkspace {
    t,
    representative: low,
    base: high,
    reduced,
    ..
  } = scratch.workspace(limbs)?;

  // A value no wider than one limb width has no high part.
  let (high_bytes, low_bytes) = value.split_at(value.len().saturating_sub(limb_bytes));
  limbs_from_be_into(low_bytes, low);
  limbs_from_be_into(high_bytes, high);
  private_mont_reduce(reduced, low, modulus, t);
  add_mod_in_place(reduced, high, &modulus.limbs);
  private_mont_mul(low, reduced, r2, modulus, t);

  limbs_to_be(low, out);
  Ok(())
}

pub(super) fn add_bit_mod_in_place(value: &mut [u64], modulus: &[u64], bit: u8) {
  debug_assert_eq!(value.len(), modulus.len());

  let mut carry = u64::from(bit & 1);
  for limb in value.iter_mut() {
    let (sum, overflow) = limb.overflowing_add(carry);
    *limb = sum;
    carry = u64::from(overflow);
  }
  subtract_modulus_if_needed(value, modulus, carry);
}

pub(super) fn cmp_limbs(a: &[u64], b: &[u64]) -> core::cmp::Ordering {
  debug_assert_eq!(a.len(), b.len());
  for index in (0..a.len()).rev() {
    match a[index].cmp(&b[index]) {
      core::cmp::Ordering::Equal => {}
      other => return other,
    }
  }
  core::cmp::Ordering::Equal
}

pub(super) fn ct_limbs_lt(left: &[u64], right: &[u64]) -> bool {
  debug_assert_eq!(left.len(), right.len());
  let mut borrow = 0u64;
  for (&left_limb, &right_limb) in left.iter().zip(right) {
    let (difference, limb_borrow) = left_limb.overflowing_sub(right_limb);
    let (_, carry_borrow) = difference.overflowing_sub(borrow);
    borrow = u64::from(limb_borrow) | u64::from(carry_borrow);
  }
  borrow == 1
}

fn limb_bit_len(limbs: &[u64]) -> usize {
  for index in (0..limbs.len()).rev() {
    let limb = limbs[index];
    if limb != 0 {
      return index
        .strict_mul(64)
        .strict_add(64usize.strict_sub(limb.leading_zeros() as usize));
    }
  }
  0
}

fn public_montgomery_r2(modulus: &RsaPublicModulus) -> Box<[u64]> {
  let limbs = modulus.limbs.len();
  let mut r2 = vec![0u64; limbs];
  let mut tmp = vec![0u64; limbs];
  let mut t = vec![0u64; limbs.strict_mul(2).strict_add(2)];
  public_montgomery_r2_into(&mut r2, modulus, &mut tmp, &mut t);
  r2.into_boxed_slice()
}

/// Write `R² mod n` for `R = 2^(64·L)` and the odd `L`-limb public modulus `n`.
///
/// Doubling from `2^(bits(n) - 1)` reaches `2^(65·L) mod n`, the Montgomery form of `2^L`.
/// A Montgomery squaring maps the Montgomery form of `2^k` to that of `2^(2·k)`, so six
/// squarings reach the Montgomery form of `2^(64·L) = R`, which is `R² mod n`. For a modulus
/// that fills its top limb, this takes `L + 1` doublings instead of `64·L + 1`.
fn public_montgomery_r2_into(r2: &mut [u64], modulus: &RsaPublicModulus, tmp: &mut [u64], t: &mut [u64]) {
  let limbs = modulus.limbs.len();
  debug_assert_eq!(r2.len(), limbs);
  debug_assert_eq!(modulus.limbs.first().copied().unwrap_or(0) & 1, 1);
  pow2_mod_into(r2, limbs.strict_mul(65), &modulus.limbs);
  for _ in 0..6 {
    if use_public_cios_montgomery(modulus) {
      mont_square_cios_in_place(r2, tmp, modulus, t);
    } else {
      mont_square_in_place(r2, tmp, modulus, t);
    }
  }
}

fn private_montgomery_r2(modulus: &[u8]) -> Result<Box<[u64]>, RsaKeyError> {
  let limb_count = modulus.len().strict_add(7) / 8;
  let mut power = vec![0u8; limb_count.strict_mul(16).strict_add(1)];
  let Some(first) = power.get_mut(0) else {
    return Err(RsaKeyError::InvalidModulus);
  };
  *first = 1;
  let mut reduced = vec![0u8; modulus.len()];
  private_import_unsigned_be_mod_to_len(&power, modulus, &mut reduced)?;
  ct::zeroize(&mut power);
  let limbs = limbs_from_be(&reduced).into_boxed_slice();
  ct::zeroize(&mut reduced);
  Ok(limbs)
}

pub(super) fn ct_nonzero_u64(value: u64) -> u64 {
  (value | value.wrapping_neg()) >> 63
}

pub(super) fn sub_modulus_in_place(value: &mut [u64], modulus: &[u64]) -> u64 {
  debug_assert_eq!(value.len(), modulus.len());
  let mut borrow = 0u64;
  for index in 0..value.len() {
    let (tmp, b1) = value[index].overflowing_sub(modulus[index]);
    let (tmp, b2) = tmp.overflowing_sub(borrow);
    value[index] = tmp;
    borrow = u64::from(b1) | u64::from(b2);
  }
  borrow
}

fn add_modulus_masked(value: &mut [u64], modulus: &[u64], choice: u64) {
  debug_assert_eq!(value.len(), modulus.len());
  // SECURITY: Keep the complete mask opaque to discourage LLVM from splitting
  // restoration by secret carry or borrow. Target-specific binary and timing evidence is
  // still required for this optimization barrier.
  let mask = core::hint::black_box(0u64.wrapping_sub(choice & 1));
  let mut carry = 0u64;
  for index in 0..value.len() {
    let addend = modulus[index] & mask;
    let (sum, overflow) = value[index].overflowing_add(addend);
    let (sum, carry_overflow) = sum.overflowing_add(carry);
    value[index] = sum;
    carry = u64::from(overflow) | u64::from(carry_overflow);
  }
}

pub(super) fn subtract_modulus_if_needed(value: &mut [u64], modulus: &[u64], extra: u64) {
  debug_assert_eq!(value.len(), modulus.len());
  let borrow = sub_modulus_in_place(value, modulus);
  let restore = borrow & (ct_nonzero_u64(extra) ^ 1);
  add_modulus_masked(value, modulus, restore);
}

fn add_mod_in_place(value: &mut [u64], addend: &[u64], modulus: &[u64]) {
  debug_assert_eq!(value.len(), addend.len());
  debug_assert_eq!(value.len(), modulus.len());

  let mut carry = 0u64;
  for index in 0..value.len() {
    let (sum, overflow) = value[index].overflowing_add(addend[index]);
    let (sum, carry_overflow) = sum.overflowing_add(carry);
    value[index] = sum;
    carry = u64::from(overflow) | u64::from(carry_overflow);
  }

  subtract_modulus_if_needed(value, modulus, carry);
}

pub(super) fn double_mod_in_place(value: &mut [u64], modulus: &[u64]) {
  debug_assert_eq!(value.len(), modulus.len());
  let mut carry = 0u64;
  for limb in value.iter_mut() {
    let next = *limb >> 63;
    *limb = (*limb << 1) | carry;
    carry = next;
  }
  subtract_modulus_if_needed(value, modulus, carry);
}

fn pow2_mod_into(out: &mut [u64], bits: usize, modulus: &[u64]) {
  out.fill(0);
  let modulus_bits = limb_bit_len(modulus);
  let direct_bits = core::cmp::min(bits, modulus_bits.saturating_sub(1));
  out[direct_bits / 64] = 1u64 << (direct_bits % 64);
  for _ in direct_bits..bits {
    double_mod_in_place(out, modulus);
  }
}

#[must_use]
fn montgomery_n0(n0: u64) -> u64 {
  debug_assert_eq!(n0 & 1, 1);
  let mut inv = 1u64;
  for _ in 0..6 {
    inv = inv.wrapping_mul(2u64.wrapping_sub(n0.wrapping_mul(inv)));
  }
  inv.wrapping_neg()
}

pub(super) fn mont_square_in_place(value: &mut [u64], tmp: &mut [u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  let _ = tmp;
  mont_square_product(value, modulus, t);
}

pub(super) fn mont_mul_in_place_left(
  left: &mut [u64],
  right: &[u64],
  tmp: &mut [u64],
  modulus: &RsaPublicModulus,
  t: &mut [u64],
) {
  copy_limbs(tmp, left);
  mont_mul(left, tmp, right, modulus, t);
}

pub(super) fn mont_square_cios_in_place(value: &mut [u64], tmp: &mut [u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
  if value.len() == modulus.limbs.len()
    && rsa_aarch64_mont::supports(modulus.limbs.len())
    && t.len() >= rsa_aarch64_mont::scratch_words(modulus.limbs.len())
  {
    rsa_aarch64_mont::square_in_place(value, &modulus.limbs, &modulus.n0_256, t);
    return;
  }

  #[cfg(all(target_arch = "x86_64", not(feature = "portable-only"), not(miri)))]
  if value.len() == modulus.limbs.len()
    && rsa_x86_64_mont::supports(modulus.limbs.len())
    && t.len() >= rsa_x86_64_mont::scratch_words(modulus.limbs.len())
  {
    rsa_x86_64_mont::square_in_place(value, &modulus.limbs, modulus.n0, t);
    return;
  }

  if t.len() >= modulus.limbs.len().strict_mul(2) {
    mont_square_portable_in_place(value, modulus, t);
    return;
  }

  copy_limbs(tmp, value);
  mont_mul_cios(value, tmp, tmp, modulus, t);
}

pub(super) fn mont_mul_cios_in_place_left(
  left: &mut [u64],
  right: &[u64],
  tmp: &mut [u64],
  modulus: &RsaPublicModulus,
  t: &mut [u64],
) {
  #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
  if left.len() == modulus.limbs.len()
    && right.len() == modulus.limbs.len()
    && rsa_aarch64_mont::supports(modulus.limbs.len())
    && t.len() >= rsa_aarch64_mont::scratch_words(modulus.limbs.len())
  {
    rsa_aarch64_mont::mul_in_place_left(left, right, &modulus.limbs, &modulus.n0_256, t);
    return;
  }

  #[cfg(all(target_arch = "x86_64", not(feature = "portable-only"), not(miri)))]
  if left.len() == modulus.limbs.len()
    && right.len() == modulus.limbs.len()
    && rsa_x86_64_mont::supports(modulus.limbs.len())
    && t.len() >= rsa_x86_64_mont::scratch_words(modulus.limbs.len())
  {
    rsa_x86_64_mont::mul_in_place_left(left, right, &modulus.limbs, modulus.n0, t);
    return;
  }

  copy_limbs(tmp, left);
  mont_mul_cios(left, tmp, right, modulus, t);
}

#[inline]
#[must_use]
fn use_cios_montgomery(modulus: &RsaPublicModulus) -> bool {
  let limbs = modulus.limbs.len();
  limbs <= 64 || limbs == 128
}

#[inline]
#[must_use]
pub(super) fn use_public_cios_montgomery(modulus: &RsaPublicModulus) -> bool {
  modulus.limbs.len() <= 128
}

pub(super) fn mont_mul_cios(out: &mut [u64], a: &[u64], b: &[u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  let n = modulus.limbs.len();
  debug_assert_eq!(out.len(), n);
  debug_assert_eq!(a.len(), n);
  debug_assert_eq!(b.len(), n);
  debug_assert!(t.len() >= n.strict_add(2));

  #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
  if rsa_aarch64_mont::supports(n) && t.len() >= rsa_aarch64_mont::scratch_words(n) {
    rsa_aarch64_mont::mul(out, a, b, &modulus.limbs, &modulus.n0_256, t);
    return;
  }

  #[cfg(all(target_arch = "x86_64", not(feature = "portable-only"), not(miri)))]
  if rsa_x86_64_mont::supports(n) && t.len() >= rsa_x86_64_mont::scratch_words(n) {
    rsa_x86_64_mont::mul(out, a, b, &modulus.limbs, modulus.n0, t);
    return;
  }

  mont_mul_cios_portable(out, a, b, modulus, t);
}

/// `acc + a·b + carry` as `(low, high)`.
///
/// The sum is at most `(2^64 - 1)^2 + 2·(2^64 - 1) = 2^128 - 1`, so the additions never wrap;
/// wrapping operations avoid an overflow branch on secret carries.
#[inline(always)]
pub(super) fn mont_mac(acc: u64, a: u64, b: u64, carry: u64) -> (u64, u64) {
  #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only"), not(miri)))]
  {
    let low: u64;
    let high: u64;
    // SAFETY: These baseline POWER64 integer instructions compute the same full-width sum
    // proved above. Both additions consume their carry through XER before the next ADDC
    // replaces it; the final high word cannot overflow. All operands and the XER clobber are
    // declared. The block has no memory accesses, stack, calls or branches.
    unsafe {
      core::arch::asm!(
        "mulld {low}, {a}, {b}",
        "mulhdu {high}, {a}, {b}",
        "addc {low}, {low}, {acc}",
        "addze {high}, {high}",
        "addc {low}, {low}, {carry}",
        "addze {high}, {high}",
        a = in(reg) a,
        b = in(reg) b,
        acc = in(reg) acc,
        carry = in(reg) carry,
        low = out(reg) low,
        high = out(reg) high,
        out("xer") _,
        options(pure, nomem, nostack),
      );
    }
    (low, high)
  }
  #[cfg(not(all(target_arch = "powerpc64", not(feature = "portable-only"), not(miri))))]
  split_u128(
    u128::from(a)
      .strict_mul(u128::from(b))
      .wrapping_add(u128::from(acc))
      .wrapping_add(u128::from(carry)),
  )
}

/// Montgomery multiplication by CIOS: each row adds `a·b_i`, then adds `q·n` and shifts down one
/// limb in the same pass.
pub(super) fn mont_mul_cios_portable(out: &mut [u64], a: &[u64], b: &[u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  #[cfg(any(target_arch = "powerpc64", target_arch = "s390x", target_arch = "riscv64"))]
  if modulus.limbs.len() == 16 {
    return mont_mul_cios_portable_at_width::<16>(out, a, b, modulus, t);
  }
  mont_mul_cios_portable_at_width::<0>(out, a, b, modulus, t);
}

fn mont_mul_cios_portable_at_width<const WIDTH: usize>(
  out: &mut [u64],
  a: &[u64],
  b: &[u64],
  modulus: &RsaPublicModulus,
  t: &mut [u64],
) {
  let n = if WIDTH == 0 { modulus.limbs.len() } else { WIDTH };
  debug_assert_eq!(out.len(), n);
  debug_assert_eq!(a.len(), n);
  debug_assert_eq!(b.len(), n);
  debug_assert!(t.len() >= n.strict_add(2));

  let a = &a[..n];
  let limbs = &modulus.limbs[..n];
  let t = &mut t[..n.strict_add(2)];
  t.fill(0);

  for &b_limb in &b[..n] {
    let mut carry = 0u64;
    for (t_limb, &a_limb) in t.iter_mut().zip(a) {
      (*t_limb, carry) = mont_mac(*t_limb, a_limb, b_limb, carry);
    }
    let (sum, overflow) = t[n].overflowing_add(carry);
    t[n] = sum;
    t[n.strict_add(1)] = u64::from(overflow);

    // `t + q·n` clears limb 0; write each sum one limb lower.
    let q = t[0].wrapping_mul(modulus.n0);
    let (_, mut carry) = mont_mac(t[0], q, limbs[0], 0);
    for index in 1..n {
      (t[index.strict_sub(1)], carry) = mont_mac(t[index], q, limbs[index], carry);
    }
    let (sum, overflow) = t[n].overflowing_add(carry);
    t[n.strict_sub(1)] = sum;
    // The running value stays below 2n, so its top limb is 0 or 1 and this cannot wrap.
    t[n] = t[n.strict_add(1)].wrapping_add(u64::from(overflow));
  }

  out.copy_from_slice(&t[..n]);
  subtract_modulus_if_needed(out, limbs, t[n]);
}

/// Montgomery squaring: [`square_wide_portable`] then [`mont_reduce_wide_portable`]. `t` needs
/// at least `2n` limbs.
fn mont_square_portable_in_place(value: &mut [u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  let n = modulus.limbs.len();
  debug_assert_eq!(value.len(), n);
  let wide = &mut t[..n.strict_mul(2)];
  square_wide_portable(wide, value);
  mont_reduce_wide_portable(value, wide, modulus);
}

/// `value²` into `out` (`2n` limbs): the `n(n - 1)/2` cross products once, doubled, then the `n`
/// diagonal squares.
fn square_wide_portable(out: &mut [u64], value: &[u64]) {
  #[cfg(any(target_arch = "powerpc64", target_arch = "s390x", target_arch = "riscv64"))]
  if value.len() == 16 {
    return square_wide_portable_at_width::<16>(out, value);
  }
  square_wide_portable_at_width::<0>(out, value);
}

fn square_wide_portable_at_width<const WIDTH: usize>(out: &mut [u64], value: &[u64]) {
  let n = if WIDTH == 0 { value.len() } else { WIDTH };
  debug_assert_eq!(out.len(), n.strict_mul(2));
  #[cfg(not(target_arch = "s390x"))]
  let value = &value[..n];
  #[cfg(not(target_arch = "s390x"))]
  let out = &mut out[..n.strict_mul(2)];
  out.fill(0);

  // Row i adds `v_i·v_j` for j > i at limbs 2i + 1 through i + n - 1; its carry starts limb i + n.
  for (row, &value_row) in value.iter().enumerate() {
    let next = row.strict_add(1);
    let mut carry = 0u64;
    for (limb, &value_column) in out[row.strict_add(next)..].iter_mut().zip(&value[next..]) {
      (*limb, carry) = mont_mac(*limb, value_row, value_column, carry);
    }
    out[row.strict_add(n)] = carry;
  }

  // The cross sum is below 2^(128n - 1), so doubling it fits.
  let mut high_bit = 0u64;
  for limb in out.iter_mut() {
    let next = *limb >> 63;
    *limb = (*limb << 1) | high_bit;
    high_bit = next;
  }

  let mut carry = 0u64;
  for ([low, high], &limb) in out.as_chunks_mut::<2>().0.iter_mut().zip(value) {
    // `low + limb² + carry` fits 128 bits, so its high limb is the square's high limb plus the low
    // limb's carry, and adding `high` to it leaves a 0/1 carry. Both sums stay in 128-bit
    // arithmetic, as in `mont_mac`: a carry added as a separate `bool` compiled to a branch on
    // RISC-V (Constant-Time 39835996527).
    let (sum_low, square_high) = mont_mac(*low, limb, limb, carry);
    let (sum_high, carry_out) = split_u128(u128::from(*high).wrapping_add(u128::from(square_high)));
    *low = sum_low;
    *high = sum_high;
    carry = carry_out;
  }
}

/// Montgomery-reduce a `2n`-limb value below `n·R` into `out`.
///
/// Row i clears limb i by adding `q·n·2^(64i)`; its carry out of limb i + n waits in `top` for the
/// next row instead of propagating through the upper limbs.
fn mont_reduce_wide_portable(out: &mut [u64], wide: &mut [u64], modulus: &RsaPublicModulus) {
  #[cfg(any(target_arch = "powerpc64", target_arch = "s390x", target_arch = "riscv64"))]
  if modulus.limbs.len() == 16 {
    return mont_reduce_wide_portable_at_width::<16>(out, wide, modulus);
  }
  mont_reduce_wide_portable_at_width::<0>(out, wide, modulus);
}

fn mont_reduce_wide_portable_at_width<const WIDTH: usize>(
  out: &mut [u64],
  wide: &mut [u64],
  modulus: &RsaPublicModulus,
) {
  let n = if WIDTH == 0 { modulus.limbs.len() } else { WIDTH };
  debug_assert_eq!(out.len(), n);
  debug_assert_eq!(wide.len(), n.strict_mul(2));
  let limbs = &modulus.limbs[..n];

  let mut top = 0u64;
  for row in 0..n {
    let q = wide[row].wrapping_mul(modulus.n0);
    let mut carry = 0u64;
    let end = row.strict_add(n);
    for (limb, &modulus_limb) in wide[row..end].iter_mut().zip(limbs) {
      (*limb, carry) = mont_mac(*limb, q, modulus_limb, carry);
    }
    // limb + carry + top is below 2^65, so at most one of the two additions carries.
    let (sum, overflow0) = wide[end].overflowing_add(carry);
    let (sum, overflow1) = sum.overflowing_add(top);
    wide[end] = sum;
    top = u64::from(overflow0 | overflow1);
  }

  out.copy_from_slice(&wide[n..]);
  subtract_modulus_if_needed(out, limbs, top);
}

fn mont_reduce_cios(out: &mut [u64], value: &[u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  let n = modulus.limbs.len();
  debug_assert_eq!(out.len(), n);
  debug_assert_eq!(value.len(), n);
  debug_assert!(t.len() >= n.strict_add(2));

  #[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "macos"),
    not(feature = "portable-only"),
    not(miri)
  ))]
  if n == 32 && t.len() >= 66 {
    rsa_aarch64_asm::mont_reduce_cios_32(out, value, &modulus.limbs, modulus.n0, t);
    return;
  }

  #[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "macos"),
    not(feature = "portable-only"),
    not(miri)
  ))]
  if matches!(n, 48 | 64 | 128) && t.len() >= rsa_aarch64_asm::bignum_mont_scratch_words(n) {
    rsa_aarch64_asm::mont_reduce_cios_words(out, value, &modulus.limbs, modulus.n0, n, t);
    return;
  }

  mont_reduce_cios_portable(out, value, modulus, t);
}

pub(super) fn mont_reduce_cios_portable(out: &mut [u64], value: &[u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  let n = modulus.limbs.len();
  debug_assert_eq!(out.len(), n);
  debug_assert_eq!(value.len(), n);
  debug_assert!(t.len() >= n.strict_add(2));

  let limbs = &modulus.limbs[..n];
  let t = &mut t[..n.strict_add(1)];
  t[..n].copy_from_slice(value);
  t[n] = 0;

  for _ in 0..n {
    // `t + q·n` clears limb 0; write each sum one limb lower.
    let q = t[0].wrapping_mul(modulus.n0);
    let (_, mut carry) = mont_mac(t[0], q, limbs[0], 0);
    for index in 1..n {
      (t[index.strict_sub(1)], carry) = mont_mac(t[index], q, limbs[index], carry);
    }
    let (sum, overflow) = t[n].overflowing_add(carry);
    t[n.strict_sub(1)] = sum;
    t[n] = u64::from(overflow);
  }

  out.copy_from_slice(&t[..n]);
  subtract_modulus_if_needed(out, limbs, t[n]);
}

/// Montgomery squaring for widths without a CIOS kernel: [`square_wide_portable`] then
/// [`mont_reduce_wide_portable`].
fn mont_square_product(out: &mut [u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  let n = modulus.limbs.len();
  debug_assert_eq!(out.len(), n);
  debug_assert!(t.len() >= n.strict_mul(2).strict_add(2));

  let wide = &mut t[..n.strict_mul(2)];
  square_wide_portable(wide, out);
  mont_reduce_wide_portable(out, wide, modulus);
}

/// `a·b·R⁻¹ mod n` for widths without a CIOS kernel: the full product by operand scanning, then
/// [`mont_reduce_wide_portable`].
pub(super) fn mont_mul(out: &mut [u64], a: &[u64], b: &[u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  let n = modulus.limbs.len();
  debug_assert_eq!(out.len(), n);
  debug_assert_eq!(a.len(), n);
  debug_assert_eq!(b.len(), n);
  debug_assert!(t.len() >= n.strict_mul(2).strict_add(2));

  let wide = &mut t[..n.strict_mul(2)];
  wide.fill(0);
  for (row, &b_limb) in b[..n].iter().enumerate() {
    let mut carry = 0u64;
    for (limb, &a_limb) in wide[row..].iter_mut().zip(&a[..n]) {
      (*limb, carry) = mont_mac(*limb, a_limb, b_limb, carry);
    }
    // Earlier rows end below limb `row + n`.
    wide[row.strict_add(n)] = carry;
  }
  mont_reduce_wide_portable(out, wide, modulus);
}

/// `value·R⁻¹ mod n` for widths without a CIOS kernel.
pub(super) fn mont_reduce(out: &mut [u64], value: &[u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  let n = modulus.limbs.len();
  debug_assert_eq!(out.len(), n);
  debug_assert_eq!(value.len(), n);
  debug_assert!(t.len() >= n.strict_mul(2).strict_add(2));

  let wide = &mut t[..n.strict_mul(2)];
  let (low, high) = wide.split_at_mut(n);
  low.copy_from_slice(value);
  high.fill(0);
  mont_reduce_wide_portable(out, wide, modulus);
}

#[cfg(test)]
mod tests;
