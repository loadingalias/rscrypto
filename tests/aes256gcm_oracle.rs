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
  Aes256Gcm as Oracle,
  aead::{AeadInOut, KeyInit, array::Array},
};
use rscrypto::{
  Aes256Gcm, Aes256GcmKey, Aes256GcmTag,
  aead::{Nonce96, expert::AeadWithNonce},
};

fn deterministic_bytes(seed: u8, len: usize) -> Vec<u8> {
  let mut out = Vec::with_capacity(len);
  let mut x = u32::from(seed);
  for _ in 0..len {
    x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    out.push(x.to_be_bytes()[0]);
  }
  out
}

fn assert_matches_oracle(key_bytes: &[u8; 32], nonce_bytes: &[u8; 12], aad: &[u8], plaintext: &[u8]) {
  let key = Aes256GcmKey::from_bytes(*key_bytes);
  let nonce = Nonce96::from_bytes(*nonce_bytes);
  let cipher = Aes256Gcm::new(&key);

  let oracle = Oracle::new(&Array(*key_bytes));
  let oracle_nonce = Array(*nonce_bytes);

  // Encrypt with rscrypto.
  let mut ours = plaintext.to_vec();
  let tag = cipher
    .encrypt_in_place(&nonce, aad, &mut ours)
    .expect("rscrypto must seal valid AES-256-GCM oracle input");

  // Encrypt with oracle.
  let mut oracle_buf = plaintext.to_vec();
  let oracle_tag = oracle
    .encrypt_inout_detached(&oracle_nonce, aad, oracle_buf.as_mut_slice().into())
    .expect("RustCrypto must seal valid AES-256-GCM oracle input");

  assert_eq!(ours, oracle_buf, "ciphertext mismatch (len={})", plaintext.len());
  assert_eq!(
    tag.as_bytes(),
    oracle_tag.as_slice(),
    "tag mismatch (len={})",
    plaintext.len()
  );

  // Decrypt with rscrypto.
  cipher
    .decrypt_in_place(&nonce, aad, &mut ours, &tag)
    .expect("fresh AES-256-GCM ciphertext must authenticate");
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
  let plaintext: Vec<u8> = (0usize..8192).map(|i| i.to_le_bytes()[0]).collect();
  assert_matches_oracle(&key, &nonce, b"large", &plaintext);
}

#[test]
fn aes256gcm_oracle_aad_size_sweep() {
  let key = [0x99u8; 32];
  let nonce = [0xAAu8; 12];
  let plaintext = b"fixed-plaintext";

  // Sweeps cover the wide-GHASH 4-block boundary (64-byte chunks) and the
  // partial-tail seam at +/-1 around 16-byte block boundaries.
  for aad_len in [0usize, 1, 15, 16, 17, 32, 33, 47, 48, 49, 64, 65, 80, 81, 128, 1024] {
    let aad: Vec<u8> = (0..aad_len).map(|i| i.to_le_bytes()[0]).collect();
    assert_matches_oracle(&key, &nonce, &aad, plaintext);
  }
}

#[test]
fn aes256gcm_oracle_all_short_lengths() {
  let key = [0xA8u8; 32];

  for size in 0..=255usize {
    let mut nonce = [0x28u8; 12];
    let size_word = u32::try_from(size).expect("short-length case must fit in a 32-bit nonce field");
    let size_byte = size.to_le_bytes()[0];
    nonce[8..12].copy_from_slice(&size_word.to_be_bytes());
    let aad = if size % 3 == 0 {
      Vec::new()
    } else {
      deterministic_bytes(0x52 ^ size_byte, size % 97)
    };
    let plaintext = deterministic_bytes(0x91 ^ size_byte, size);
    assert_matches_oracle(&key, &nonce, &aad, &plaintext);
  }
}

#[test]
fn aes256gcm_oracle_large_aligned_and_unaligned_lengths() {
  let key = [0x6Du8; 32];
  let aad_cases = [Vec::new(), deterministic_bytes(0xBC, 257)];

  for (case_idx, &size) in [4096usize, 4097, 8191, 8192, 8193, 16_384, 16_385].iter().enumerate() {
    let mut nonce = [0x81u8; 12];
    let case_word = u32::try_from(case_idx).expect("GCM oracle case index must fit in a 32-bit nonce field");
    let size_word = u32::try_from(size).expect("GCM oracle size must fit in a 32-bit nonce field");
    nonce[4..8].copy_from_slice(&case_word.to_be_bytes());
    nonce[8..12].copy_from_slice(&size_word.to_be_bytes());
    let plaintext = deterministic_bytes(0x43 ^ case_idx.to_le_bytes()[0], size);
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
  let mut tag = cipher
    .encrypt_in_place(&nonce, b"aad", &mut buffer)
    .expect("AES-256-GCM tag-forgery fixture must seal")
    .to_bytes();
  tag[0] ^= 1;

  cipher
    .decrypt_in_place(&nonce, b"aad", &mut buffer, &Aes256GcmTag::from_bytes(tag))
    .expect_err("AES-256-GCM must reject a modified tag");
}

#[test]
fn aes256gcm_rejects_modified_ciphertext() {
  let key = Aes256GcmKey::from_bytes([0x33; 32]);
  let nonce = Nonce96::from_bytes([0x44; 12]);
  let cipher = Aes256Gcm::new(&key);

  let mut buffer = *b"tamper-detect";
  let tag = cipher
    .encrypt_in_place(&nonce, b"", &mut buffer)
    .expect("AES-256-GCM ciphertext-tampering fixture must seal");
  buffer[0] ^= 1;

  cipher
    .decrypt_in_place(&nonce, b"", &mut buffer, &tag)
    .expect_err("AES-256-GCM must reject modified ciphertext");
}

#[test]
fn aes256gcm_rejects_wrong_aad() {
  let key = Aes256GcmKey::from_bytes([0x55; 32]);
  let nonce = Nonce96::from_bytes([0x66; 12]);
  let cipher = Aes256Gcm::new(&key);

  let mut buffer = *b"aad-mismatch";
  let tag = cipher
    .encrypt_in_place(&nonce, b"correct", &mut buffer)
    .expect("AES-256-GCM AAD-mismatch fixture must seal");

  cipher
    .decrypt_in_place(&nonce, b"wrong", &mut buffer, &tag)
    .expect_err("AES-256-GCM must reject incorrect associated data");
}
