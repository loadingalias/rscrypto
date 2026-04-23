//! Integration oracle tests for Ascon-AEAD128.
//!
//! Validates rscrypto's Ascon-AEAD128 against the `ascon-aead` crate across
//! multiple input sizes and boundary conditions.
//!
//! The unit tests in `src/aead/ascon128.rs` already use the oracle for a few
//! fixed vectors. This file extends coverage with systematic size sweeps
//! and forgery rejection from the integration-test perspective.
//!
//! # Coverage
//!
//! 1. Encrypt/decrypt round-trip matches `ascon-aead` oracle
//! 2. Empty plaintext, empty AAD, both empty
//! 3. Ascon rate (8-byte) boundary sizes
//! 4. Large multi-block inputs
//! 5. Tag forgery / ciphertext / AAD tampering rejection

#![cfg(feature = "aead")]

use ascon_aead::{
  AsconAead128 as Oracle,
  aead::{AeadInPlace, KeyInit, generic_array::GenericArray},
};
use rscrypto::{AsconAead128, AsconAead128Key, AsconAead128Tag, aead::Nonce128};

fn assert_matches_oracle(key_bytes: &[u8; 16], nonce_bytes: &[u8; 16], aad: &[u8], plaintext: &[u8]) {
  let key = AsconAead128Key::from_bytes(*key_bytes);
  let nonce = Nonce128::from_bytes(*nonce_bytes);
  let cipher = AsconAead128::new(&key);

  let oracle = Oracle::new(GenericArray::from_slice(key_bytes));
  let oracle_nonce = GenericArray::from_slice(nonce_bytes);

  // Encrypt with rscrypto.
  let mut ours = plaintext.to_vec();
  let tag = cipher.encrypt_in_place(&nonce, aad, &mut ours).unwrap();

  // Encrypt with oracle.
  let mut oracle_buf = plaintext.to_vec();
  let oracle_tag = oracle
    .encrypt_in_place_detached(oracle_nonce, aad, &mut oracle_buf)
    .unwrap();

  assert_eq!(ours, oracle_buf, "ciphertext mismatch (len={})", plaintext.len());
  assert_eq!(
    tag.as_bytes(),
    oracle_tag.as_slice(),
    "tag mismatch (len={})",
    plaintext.len()
  );

  // Decrypt with rscrypto.
  cipher.decrypt_in_place(&nonce, aad, &mut ours, &tag).unwrap();
  assert_eq!(ours, plaintext, "decrypt round-trip failed (len={})", plaintext.len());
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Oracle Agreement
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn ascon_aead128_matches_oracle() {
  let key = [0x42u8; 16];
  let nonce = [0x24u8; 16];
  let aad = b"rscrypto-ascon-aead-oracle";
  let plaintext = b"lightweight AEAD from NIST SP 800-232";

  assert_matches_oracle(&key, &nonce, aad, plaintext);
}

#[test]
fn ascon_aead128_oracle_empty_plaintext() {
  assert_matches_oracle(&[0xAA; 16], &[0xBB; 16], b"aad-only", b"");
}

#[test]
fn ascon_aead128_oracle_empty_aad() {
  assert_matches_oracle(&[0xCC; 16], &[0xDD; 16], b"", b"no associated data");
}

#[test]
fn ascon_aead128_oracle_both_empty() {
  assert_matches_oracle(&[0xEE; 16], &[0xFF; 16], b"", b"");
}

#[test]
fn ascon_aead128_oracle_rate_boundary_sizes() {
  let key = [0x55u8; 16];
  let nonce = [0x66u8; 16];
  let aad = b"boundary";

  // Ascon-AEAD128 rate = 16 bytes (128 bits).
  for size in [1, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33, 48, 64, 128, 256] {
    let plaintext = vec![0xABu8; size];
    assert_matches_oracle(&key, &nonce, aad, &plaintext);
  }
}

#[test]
fn ascon_aead128_oracle_large_input() {
  let key = [0x77u8; 16];
  let nonce = [0x88u8; 16];
  let plaintext: Vec<u8> = (0..8192).map(|i| (i & 0xFF) as u8).collect();
  assert_matches_oracle(&key, &nonce, b"large", &plaintext);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Forgery Rejection
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn ascon_aead128_rejects_modified_tag() {
  let key = AsconAead128Key::from_bytes([0x11; 16]);
  let nonce = Nonce128::from_bytes([0x22; 16]);
  let cipher = AsconAead128::new(&key);

  let mut buffer = *b"forgery-check";
  let mut tag = cipher.encrypt_in_place(&nonce, b"aad", &mut buffer).unwrap().to_bytes();
  tag[0] ^= 1;

  assert!(
    cipher
      .decrypt_in_place(&nonce, b"aad", &mut buffer, &AsconAead128Tag::from_bytes(tag))
      .is_err()
  );
}

#[test]
fn ascon_aead128_rejects_modified_ciphertext() {
  let key = AsconAead128Key::from_bytes([0x33; 16]);
  let nonce = Nonce128::from_bytes([0x44; 16]);
  let cipher = AsconAead128::new(&key);

  let mut buffer = *b"tamper-detect";
  let tag = cipher.encrypt_in_place(&nonce, b"", &mut buffer).unwrap();
  buffer[0] ^= 1;

  assert!(cipher.decrypt_in_place(&nonce, b"", &mut buffer, &tag).is_err());
}

#[test]
fn ascon_aead128_rejects_wrong_aad() {
  let key = AsconAead128Key::from_bytes([0x55; 16]);
  let nonce = Nonce128::from_bytes([0x66; 16]);
  let cipher = AsconAead128::new(&key);

  let mut buffer = *b"aad-mismatch";
  let tag = cipher.encrypt_in_place(&nonce, b"correct", &mut buffer).unwrap();

  assert!(cipher.decrypt_in_place(&nonce, b"wrong", &mut buffer, &tag).is_err());
}
