#![cfg(all(rscrypto_internal, feature = "diag", feature = "pbkdf2"))]

use rscrypto::{Pbkdf2Sha256, Pbkdf2Sha512, auth};

#[test]
fn sha256_proof_hook_verifies_the_derived_key() {
  let password = [0x37; 32];
  let mut expected = [0; 32];
  pbkdf2::pbkdf2_hmac::<sha2::Sha256>(&password, b"salt", 1, &mut expected);

  assert!(auth::diag_pbkdf2_sha256_verify_portable(&password, &expected));
  assert!(Pbkdf2Sha256::new(&password).verify(b"salt", 1, &expected).is_err());
  for index in 0..expected.len() {
    let mut wrong = expected;
    wrong[index] ^= 1;
    assert!(!auth::diag_pbkdf2_sha256_verify_portable(&password, &wrong));
  }
}

#[test]
fn sha512_proof_hook_verifies_the_derived_key() {
  let password = [0x93; 64];
  let mut expected = [0; 64];
  pbkdf2::pbkdf2_hmac::<sha2::Sha512>(&password, b"salt", 1, &mut expected);

  assert!(auth::diag_pbkdf2_sha512_verify_portable(&password, &expected));
  assert!(Pbkdf2Sha512::new(&password).verify(b"salt", 1, &expected).is_err());
  for index in 0..expected.len() {
    let mut wrong = expected;
    wrong[index] ^= 1;
    assert!(!auth::diag_pbkdf2_sha512_verify_portable(&password, &wrong));
  }
}
