//! Integration oracle tests for AEGIS-256.
//!
//! Validates rscrypto's AEGIS-256 against the `aegis` crate (C reference
//! backend by default) across multiple input sizes and boundary conditions.
//!
//! # Coverage
//!
//! 1. Encrypt/decrypt round-trip matches `aegis` crate oracle
//! 2. Empty plaintext, empty AAD, both empty
//! 3. Varied sizes including non-block-aligned
//! 4. Large multi-block inputs
//! 5. Tag forgery / ciphertext / AAD tampering rejection

#![cfg(feature = "aead")]

use aegis::aegis256::Aegis256 as Oracle;
use rscrypto::{Aegis256, Aegis256Key, Aegis256Tag, aead::Nonce256};

fn assert_matches_oracle(key_bytes: &[u8; 32], nonce_bytes: &[u8; 32], aad: &[u8], plaintext: &[u8]) {
  let key = Aegis256Key::from_bytes(*key_bytes);
  let nonce = Nonce256::from_bytes(*nonce_bytes);
  let cipher = Aegis256::new(&key);

  // Encrypt with rscrypto.
  let mut ours = plaintext.to_vec();
  let tag = cipher.encrypt_in_place(&nonce, aad, &mut ours).unwrap();

  // Encrypt with oracle (consumes self, returns (Vec<u8>, [u8; 16])).
  let (oracle_ct, oracle_tag) = Oracle::<16>::new(key_bytes, nonce_bytes).encrypt(plaintext, aad);

  assert_eq!(ours, oracle_ct, "ciphertext mismatch (len={})", plaintext.len());
  assert_eq!(tag.as_bytes(), &oracle_tag, "tag mismatch (len={})", plaintext.len());

  // Decrypt with rscrypto.
  cipher.decrypt_in_place(&nonce, aad, &mut ours, &tag).unwrap();
  assert_eq!(ours, plaintext, "decrypt round-trip failed (len={})", plaintext.len());

  // Cross-decrypt: oracle decrypts rscrypto's ciphertext.
  let oracle_pt = Oracle::<16>::new(key_bytes, nonce_bytes)
    .decrypt(&oracle_ct, &oracle_tag, aad)
    .unwrap();
  assert_eq!(
    oracle_pt,
    plaintext,
    "oracle cross-decrypt failed (len={})",
    plaintext.len()
  );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Oracle Agreement
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn aegis256_matches_oracle() {
  let key = [0x42u8; 32];
  let nonce = [0x24u8; 32];
  let aad = b"rscrypto-aegis256-oracle";
  let plaintext = b"AEGIS-256 high-performance AEAD";

  assert_matches_oracle(&key, &nonce, aad, plaintext);
}

#[test]
fn aegis256_oracle_empty_plaintext() {
  assert_matches_oracle(&[0xAA; 32], &[0xBB; 32], b"aad-only", b"");
}

#[test]
fn aegis256_oracle_empty_aad() {
  assert_matches_oracle(&[0xCC; 32], &[0xDD; 32], b"", b"no associated data");
}

#[test]
fn aegis256_oracle_both_empty() {
  assert_matches_oracle(&[0xEE; 32], &[0xFF; 32], b"", b"");
}

#[test]
fn aegis256_oracle_varied_sizes() {
  let key = [0x55u8; 32];
  let nonce = [0x66u8; 32];
  let aad = b"sizes";

  // AEGIS-256 operates on 16-byte blocks internally.
  for size in [1, 7, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 128, 256] {
    let plaintext = vec![0xABu8; size];
    assert_matches_oracle(&key, &nonce, aad, &plaintext);
  }
}

#[test]
fn aegis256_oracle_large_input() {
  let key = [0x77u8; 32];
  let nonce = [0x88u8; 32];
  let plaintext: Vec<u8> = (0..8192).map(|i| (i & 0xFF) as u8).collect();
  assert_matches_oracle(&key, &nonce, b"large", &plaintext);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Forgery Rejection
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn aegis256_rejects_modified_tag() {
  let key = Aegis256Key::from_bytes([0x11; 32]);
  let nonce = Nonce256::from_bytes([0x22; 32]);
  let cipher = Aegis256::new(&key);

  let mut buffer = *b"forgery-check";
  let mut tag = cipher.encrypt_in_place(&nonce, b"aad", &mut buffer).unwrap().to_bytes();
  tag[0] ^= 1;

  assert!(
    cipher
      .decrypt_in_place(&nonce, b"aad", &mut buffer, &Aegis256Tag::from_bytes(tag))
      .is_err()
  );
}

#[test]
fn aegis256_rejects_modified_ciphertext() {
  let key = Aegis256Key::from_bytes([0x33; 32]);
  let nonce = Nonce256::from_bytes([0x44; 32]);
  let cipher = Aegis256::new(&key);

  let mut buffer = *b"tamper-detect";
  let tag = cipher.encrypt_in_place(&nonce, b"", &mut buffer).unwrap();
  buffer[0] ^= 1;

  assert!(cipher.decrypt_in_place(&nonce, b"", &mut buffer, &tag).is_err());
}

#[test]
fn aegis256_rejects_wrong_aad() {
  let key = Aegis256Key::from_bytes([0x55; 32]);
  let nonce = Nonce256::from_bytes([0x66; 32]);
  let cipher = Aegis256::new(&key);

  let mut buffer = *b"aad-mismatch";
  let tag = cipher.encrypt_in_place(&nonce, b"correct", &mut buffer).unwrap();

  assert!(cipher.decrypt_in_place(&nonce, b"wrong", &mut buffer, &tag).is_err());
}
