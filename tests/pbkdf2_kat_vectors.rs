//! PBKDF2-HMAC-SHA256 / SHA-512 known-answer vectors.
//!
//! RFC 6070 only covers PBKDF2-HMAC-SHA1. rscrypto ships SHA-256 and SHA-512
//! variants, so this file pins them against widely-published test vectors
//! that predate any one vendor:
//!
//! - SHA-256: IETF draft-josefsson-scrypt-kdf-01 §10 / Stack Overflow's canonical
//!   PBKDF2-HMAC-SHA256 vectors (also reproduced by RustCrypto `pbkdf2` crate tests).
//! - SHA-512: RFC 7914 companion values (and the RustCrypto test fixtures).
//!
//! The differential suite in `tests/pbkdf2_differential.rs` already pins
//! rscrypto against the RustCrypto `pbkdf2` crate across many parameter
//! triples; this file adds a frozen, reviewer-friendly KAT surface that
//! does not depend on any external oracle.
#![cfg(feature = "pbkdf2")]

use rscrypto::{Pbkdf2Sha256, Pbkdf2Sha512};

fn hex_to_bytes(s: &str) -> Vec<u8> {
  let s: String = s.chars().filter(|c| !c.is_ascii_whitespace()).collect();
  (0..s.len())
    .step_by(2)
    .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
    .collect()
}

// ─── SHA-256 ────────────────────────────────────────────────────────────────

#[test]
fn pbkdf2_sha256_kat_c1_dk32() {
  // P="password" S="salt" c=1 dk_len=32
  let expected = hex_to_bytes("120fb6cffcf8b32c43e7225256c4f837 a86548c92ccc35480805987cb70be17b");
  let mut out = [0u8; 32];
  Pbkdf2Sha256::derive_key(b"password", b"salt", 1, &mut out).unwrap();
  assert_eq!(out.as_slice(), expected.as_slice());
}

#[test]
fn pbkdf2_sha256_kat_c2_dk32() {
  // P="password" S="salt" c=2 dk_len=32
  let expected = hex_to_bytes("ae4d0c95af6b46d32d0adff928f06dd0 2a303f8ef3c251dfd6e2d85a95474c43");
  let mut out = [0u8; 32];
  Pbkdf2Sha256::derive_key(b"password", b"salt", 2, &mut out).unwrap();
  assert_eq!(out.as_slice(), expected.as_slice());
}

#[cfg(not(miri))]
#[test]
fn pbkdf2_sha256_kat_c4096_dk32() {
  // P="password" S="salt" c=4096 dk_len=32
  let expected = hex_to_bytes("c5e478d59288c841aa530db6845c4c8d 962893a001ce4e11a4963873aa98134a");
  let mut out = [0u8; 32];
  Pbkdf2Sha256::derive_key(b"password", b"salt", 4096, &mut out).unwrap();
  assert_eq!(out.as_slice(), expected.as_slice());
}

#[cfg(not(miri))]
#[test]
fn pbkdf2_sha256_kat_long_inputs_c4096_dk40() {
  // P="passwordPASSWORDpassword" S="saltSALTsaltSALTsaltSALTsaltSALTsalt" c=4096 dk_len=40
  let expected = hex_to_bytes("348c89dbcbd32b2f32d814b8116e84cf 2b17347ebc1800181c4e2a1fb8dd53e1 c635518c7dac47e9");
  let mut out = [0u8; 40];
  Pbkdf2Sha256::derive_key(
    b"passwordPASSWORDpassword",
    b"saltSALTsaltSALTsaltSALTsaltSALTsalt",
    4096,
    &mut out,
  )
  .unwrap();
  assert_eq!(out.as_slice(), expected.as_slice());
}

#[test]
fn pbkdf2_sha256_kat_embedded_nul_c4096_dk16() {
  // P="pass\0word" S="sa\0lt" c=4096 dk_len=16 — guards against C-string
  // NUL-termination bugs in the HMAC key/salt path.
  let expected = hex_to_bytes("89b69d0516f829893c696226650a8687");
  let mut out = [0u8; 16];
  Pbkdf2Sha256::derive_key(b"pass\x00word", b"sa\x00lt", 4096, &mut out).unwrap();
  assert_eq!(out.as_slice(), expected.as_slice());
}

// ─── SHA-512 ────────────────────────────────────────────────────────────────

#[test]
fn pbkdf2_sha512_kat_c1_dk64() {
  // P="password" S="salt" c=1 dk_len=64
  let expected = hex_to_bytes(
    "867f70cf1ade02cff3752599a3a53dc4 af34c7a669815ae5d513554e1c8cf252c02d470a285a0501bad999bfe943c08f \
     050235d7d68b1da55e63f73b60a57fce",
  );
  let mut out = [0u8; 64];
  Pbkdf2Sha512::derive_key(b"password", b"salt", 1, &mut out).unwrap();
  assert_eq!(out.as_slice(), expected.as_slice());
}

#[test]
fn pbkdf2_sha512_kat_c2_dk64() {
  // P="password" S="salt" c=2 dk_len=64
  let expected = hex_to_bytes(
    "e1d9c16aa681708a45f5c7c4e215ceb6 6e011a2e9f0040713f18aefdb866d53cf76cab2868a39b9f7840edce4fef5a82 \
     be67335c77a6068e04112754f27ccf4e",
  );
  let mut out = [0u8; 64];
  Pbkdf2Sha512::derive_key(b"password", b"salt", 2, &mut out).unwrap();
  assert_eq!(out.as_slice(), expected.as_slice());
}

#[cfg(not(miri))]
#[test]
fn pbkdf2_sha512_kat_c4096_dk64() {
  // P="password" S="salt" c=4096 dk_len=64
  let expected = hex_to_bytes(
    "d197b1b33db0143e018b12f3d1d1479e 6cdebdcc97c5c0f87f6902e072f457b5143f30602641b3d55cd335988cb36b84 \
     376060ecd532e039b742a239434af2d5",
  );
  let mut out = [0u8; 64];
  Pbkdf2Sha512::derive_key(b"password", b"salt", 4096, &mut out).unwrap();
  assert_eq!(out.as_slice(), expected.as_slice());
}

// ─── Streaming / one-shot equivalence ───────────────────────────────────────

#[test]
fn pbkdf2_sha256_state_reuse_matches_oneshot() {
  let state = Pbkdf2Sha256::new(b"password");
  let mut from_state = [0u8; 32];
  state.derive(b"salt", 100, &mut from_state).unwrap();

  let mut from_oneshot = [0u8; 32];
  Pbkdf2Sha256::derive_key(b"password", b"salt", 100, &mut from_oneshot).unwrap();

  assert_eq!(from_state, from_oneshot);

  // The cached state must be reusable for a second derivation.
  let mut from_state_again = [0u8; 32];
  state.derive(b"salt2", 50, &mut from_state_again).unwrap();

  let mut from_oneshot_again = [0u8; 32];
  Pbkdf2Sha256::derive_key(b"password", b"salt2", 50, &mut from_oneshot_again).unwrap();

  assert_eq!(from_state_again, from_oneshot_again);
}

#[test]
fn pbkdf2_sha512_state_reuse_matches_oneshot() {
  let state = Pbkdf2Sha512::new(b"password");
  let mut from_state = [0u8; 64];
  state.derive(b"salt", 100, &mut from_state).unwrap();

  let mut from_oneshot = [0u8; 64];
  Pbkdf2Sha512::derive_key(b"password", b"salt", 100, &mut from_oneshot).unwrap();

  assert_eq!(from_state, from_oneshot);
}
