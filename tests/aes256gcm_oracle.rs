//! Integration oracle tests for AES-256-GCM.
//!
//! Validates rscrypto's AES-256-GCM against the RustCrypto `aes-gcm` crate
//! across multiple input sizes, AAD patterns, and boundary conditions.
//!
//! # Coverage
//!
//! 1. Encrypt/decrypt round-trip matches RustCrypto oracle
//! 2. Empty plaintext with AAD
//! 3. Empty AAD with plaintext
//! 4. Large multi-block inputs
//! 5. Tag forgery rejection

#![cfg(feature = "aead")]

use aes_gcm::{
  Aes256Gcm as Oracle, KeyInit,
  aead::{AeadInPlace, generic_array::GenericArray},
};
use rscrypto::{Aes256Gcm, Aes256GcmKey, Aes256GcmTag, aead::Nonce96};

fn deterministic_bytes(seed: u8, len: usize) -> Vec<u8> {
  let mut out = Vec::with_capacity(len);
  let mut x = seed as u32;
  for _ in 0..len {
    x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    out.push((x >> 24) as u8);
  }
  out
}

fn assert_matches_oracle(key_bytes: &[u8; 32], nonce_bytes: &[u8; 12], aad: &[u8], plaintext: &[u8]) {
  let key = Aes256GcmKey::from_bytes(*key_bytes);
  let nonce = Nonce96::from_bytes(*nonce_bytes);
  let cipher = Aes256Gcm::new(&key);

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

// Oracle Agreement

#[test]
fn aes256gcm_matches_rustcrypto_oracle() {
  let key = [0x42u8; 32];
  let nonce = [0x24u8; 12];
  let aad = b"rscrypto-aes-gcm-oracle";
  let plaintext = b"portable baseline first, SIMD later";

  assert_matches_oracle(&key, &nonce, aad, plaintext);
}

#[test]
fn aes256gcm_oracle_empty_plaintext() {
  assert_matches_oracle(&[0xAA; 32], &[0xBB; 12], b"aad-only", b"");
}

#[test]
fn aes256gcm_oracle_empty_aad() {
  assert_matches_oracle(&[0xCC; 32], &[0xDD; 12], b"", b"no associated data");
}

#[test]
fn aes256gcm_oracle_both_empty() {
  assert_matches_oracle(&[0xEE; 32], &[0xFF; 12], b"", b"");
}

#[test]
fn aes256gcm_oracle_block_boundary_sizes() {
  let key = [0x55u8; 32];
  let nonce = [0x66u8; 12];
  let aad = b"boundary";

  // AES block = 16 bytes. Test at, below, and above boundaries.
  for size in [1, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 128, 256] {
    let plaintext = vec![0xABu8; size];
    assert_matches_oracle(&key, &nonce, aad, &plaintext);
  }
}

#[test]
fn aes256gcm_oracle_large_input() {
  let key = [0x77u8; 32];
  let nonce = [0x88u8; 12];
  // 8 KiB — exercises multi-block GHASH and CTR paths.
  let plaintext: Vec<u8> = (0..8192).map(|i| (i & 0xFF) as u8).collect();
  assert_matches_oracle(&key, &nonce, b"large", &plaintext);
}

#[test]
fn aes256gcm_oracle_aad_size_sweep() {
  let key = [0x99u8; 32];
  let nonce = [0xAAu8; 12];
  let plaintext = b"fixed-plaintext";

  // Sweeps cover the wide-GHASH 4-block boundary (64-byte chunks) and the
  // partial-tail seam at +/-1 around 16-byte block boundaries.
  for aad_len in [0, 1, 15, 16, 17, 32, 33, 47, 48, 49, 64, 65, 80, 81, 128, 1024] {
    let aad: Vec<u8> = (0..aad_len).map(|i| (i & 0xFF) as u8).collect();
    assert_matches_oracle(&key, &nonce, &aad, plaintext);
  }
}

#[test]
fn aes256gcm_oracle_all_short_lengths() {
  let key = [0xA8u8; 32];

  for size in 0..=255usize {
    let mut nonce = [0x28u8; 12];
    nonce[8..12].copy_from_slice(&(size as u32).to_be_bytes());
    let aad = if size % 3 == 0 {
      Vec::new()
    } else {
      deterministic_bytes(0x52 ^ size as u8, size % 97)
    };
    let plaintext = deterministic_bytes(0x91 ^ size as u8, size);
    assert_matches_oracle(&key, &nonce, &aad, &plaintext);
  }
}

#[test]
fn aes256gcm_oracle_large_aligned_and_unaligned_lengths() {
  let key = [0x6Du8; 32];
  let aad_cases = [Vec::new(), deterministic_bytes(0xBC, 257)];

  for (case_idx, &size) in [4096usize, 4097, 8191, 8192, 8193, 16_384, 16_385].iter().enumerate() {
    let mut nonce = [0x81u8; 12];
    nonce[4..8].copy_from_slice(&(case_idx as u32).to_be_bytes());
    nonce[8..12].copy_from_slice(&(size as u32).to_be_bytes());
    let plaintext = deterministic_bytes(0x43 ^ case_idx as u8, size);
    for aad in &aad_cases {
      assert_matches_oracle(&key, &nonce, aad, &plaintext);
    }
  }
}

// Forgery Rejection

#[test]
fn aes256gcm_rejects_modified_tag() {
  let key = Aes256GcmKey::from_bytes([0x11; 32]);
  let nonce = Nonce96::from_bytes([0x22; 12]);
  let cipher = Aes256Gcm::new(&key);

  let mut buffer = *b"forgery-check";
  let mut tag = cipher.encrypt_in_place(&nonce, b"aad", &mut buffer).unwrap().to_bytes();
  tag[0] ^= 1;

  assert!(
    cipher
      .decrypt_in_place(&nonce, b"aad", &mut buffer, &Aes256GcmTag::from_bytes(tag))
      .is_err()
  );
}

#[test]
fn aes256gcm_rejects_modified_ciphertext() {
  let key = Aes256GcmKey::from_bytes([0x33; 32]);
  let nonce = Nonce96::from_bytes([0x44; 12]);
  let cipher = Aes256Gcm::new(&key);

  let mut buffer = *b"tamper-detect";
  let tag = cipher.encrypt_in_place(&nonce, b"", &mut buffer).unwrap();
  buffer[0] ^= 1;

  assert!(cipher.decrypt_in_place(&nonce, b"", &mut buffer, &tag).is_err());
}

#[test]
fn aes256gcm_rejects_wrong_aad() {
  let key = Aes256GcmKey::from_bytes([0x55; 32]);
  let nonce = Nonce96::from_bytes([0x66; 12]);
  let cipher = Aes256Gcm::new(&key);

  let mut buffer = *b"aad-mismatch";
  let tag = cipher.encrypt_in_place(&nonce, b"correct", &mut buffer).unwrap();

  assert!(cipher.decrypt_in_place(&nonce, b"wrong", &mut buffer, &tag).is_err());
}
