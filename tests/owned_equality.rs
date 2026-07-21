fn assert_owned_equality<const N: usize, T>(make: impl Fn([u8; N]) -> T, compare: impl Fn(&T, &T) -> bool) {
  let bytes = [0x5a; N];
  assert!(compare(&make(bytes), &make(bytes)));

  for index in [0, N / 2, N.strict_sub(1)] {
    let mut different = bytes;
    different[index] ^= 1;
    assert!(!compare(&make(bytes), &make(different)));
  }
}

#[test]
#[cfg(feature = "aes-gcm")]
fn aes128_key_owns_16_byte_equality() {
  assert_owned_equality(rscrypto::Aes128GcmKey::from_bytes, |left, right| {
    left.ct_eq(right).declassify()
  });
}

#[test]
#[cfg(feature = "x25519")]
fn x25519_secret_key_owns_32_byte_equality() {
  assert_owned_equality(rscrypto::X25519SecretKey::from_bytes, |left, right| {
    left.ct_eq(right).declassify()
  });
}

#[test]
#[cfg(feature = "hmac")]
fn hmac_sha384_tag_owns_48_byte_equality() {
  assert_owned_equality(rscrypto::HmacSha384Tag::from_bytes, |left, right| {
    left.ct_eq(right).declassify()
  });
}

#[test]
#[cfg(feature = "hmac")]
fn hmac_sha512_tag_owns_64_byte_equality() {
  assert_owned_equality(rscrypto::HmacSha512Tag::from_bytes, |left, right| {
    left.ct_eq(right).declassify()
  });
}
