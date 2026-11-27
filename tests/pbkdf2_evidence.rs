#![cfg(feature = "pbkdf2")]

use rscrypto::{Pbkdf2Sha256, Pbkdf2Sha512};

#[test]
fn sha256_primitive_verifies_the_derived_key() {
  let password = [0x37; 32];
  let mut expected = [0; 32];
  pbkdf2::pbkdf2_hmac::<sha2::Sha256>(&password, b"salt", 1, &mut expected);

  Pbkdf2Sha256::verify_password_primitive(&password, b"salt", 1, &expected)
    .expect("the independent PBKDF2 output verifies");
  assert!(Pbkdf2Sha256::new(&password).verify(b"salt", 1, &expected).is_err());
  for index in 0..expected.len() {
    let mut wrong = expected;
    wrong[index] ^= 1;
    assert!(Pbkdf2Sha256::verify_password_primitive(&password, b"salt", 1, &wrong).is_err());
  }
}

#[test]
fn sha512_primitive_verifies_the_derived_key() {
  let password = [0x93; 64];
  let mut expected = [0; 64];
  pbkdf2::pbkdf2_hmac::<sha2::Sha512>(&password, b"salt", 1, &mut expected);

  Pbkdf2Sha512::verify_password_primitive(&password, b"salt", 1, &expected)
    .expect("the independent PBKDF2 output verifies");
  assert!(Pbkdf2Sha512::new(&password).verify(b"salt", 1, &expected).is_err());
  for index in 0..expected.len() {
    let mut wrong = expected;
    wrong[index] ^= 1;
    assert!(Pbkdf2Sha512::verify_password_primitive(&password, b"salt", 1, &wrong).is_err());
  }
}
