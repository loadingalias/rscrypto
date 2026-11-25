#[inline(always)]
fn mask_nonzero(value: u64) -> u64 {
  0u64.wrapping_sub((value | value.wrapping_neg()) >> 63)
}

#[inline(always)]
fn mask_zero(value: u64) -> u64 {
  !mask_nonzero(value)
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
    // Split limbs so s390x carry extraction stays arithmetic.
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
