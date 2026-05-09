#![cfg(all(
  feature = "aead",
  not(feature = "portable-only"),
  target_arch = "aarch64",
  target_os = "macos"
))]

use aes_gcm::{
  Aes128Gcm as Aes128Oracle, Aes256Gcm as Aes256Oracle, KeyInit,
  aead::{AeadInPlace, generic_array::GenericArray},
};
use rscrypto::{Aes128Gcm, Aes128GcmKey, Aes256Gcm, Aes256GcmKey, aead::Nonce96};

fn deterministic_bytes(seed: u8, len: usize) -> Vec<u8> {
  let mut out = Vec::with_capacity(len);
  let mut x = seed as u32;
  for _ in 0..len {
    x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    out.push((x >> 24) as u8);
  }
  out
}

fn assert_aes128_matches_oracle(len: usize, aad: &[u8]) {
  let key_bytes = [0x31u8; 16];
  let mut nonce_bytes = [0x42u8; 12];
  nonce_bytes[4..8].copy_from_slice(&(aad.len() as u32).to_be_bytes());
  nonce_bytes[8..12].copy_from_slice(&(len as u32).to_be_bytes());
  let plaintext = deterministic_bytes(0x80 ^ len as u8, len);

  let cipher = Aes128Gcm::new(&Aes128GcmKey::from_bytes(key_bytes));
  let nonce = Nonce96::from_bytes(nonce_bytes);
  let mut ours = plaintext.clone();
  let tag = cipher.encrypt_in_place(&nonce, aad, &mut ours).unwrap();

  let oracle = Aes128Oracle::new(GenericArray::from_slice(&key_bytes));
  let mut expected = plaintext.clone();
  let expected_tag = oracle
    .encrypt_in_place_detached(GenericArray::from_slice(&nonce_bytes), aad, &mut expected)
    .unwrap();

  assert_eq!(ours, expected, "AES-128-GCM ciphertext mismatch at len {len}");
  assert_eq!(
    tag.as_bytes(),
    expected_tag.as_slice(),
    "AES-128-GCM tag mismatch at len {len}"
  );

  cipher.decrypt_in_place(&nonce, aad, &mut ours, &tag).unwrap();
  assert_eq!(ours, plaintext, "AES-128-GCM open mismatch at len {len}");
}

fn assert_aes256_matches_oracle(len: usize, aad: &[u8]) {
  let key_bytes = [0x53u8; 32];
  let mut nonce_bytes = [0x64u8; 12];
  nonce_bytes[4..8].copy_from_slice(&(aad.len() as u32).to_be_bytes());
  nonce_bytes[8..12].copy_from_slice(&(len as u32).to_be_bytes());
  let plaintext = deterministic_bytes(0xA0 ^ len as u8, len);

  let cipher = Aes256Gcm::new(&Aes256GcmKey::from_bytes(key_bytes));
  let nonce = Nonce96::from_bytes(nonce_bytes);
  let mut ours = plaintext.clone();
  let tag = cipher.encrypt_in_place(&nonce, aad, &mut ours).unwrap();

  let oracle = Aes256Oracle::new(GenericArray::from_slice(&key_bytes));
  let mut expected = plaintext.clone();
  let expected_tag = oracle
    .encrypt_in_place_detached(GenericArray::from_slice(&nonce_bytes), aad, &mut expected)
    .unwrap();

  assert_eq!(ours, expected, "AES-256-GCM ciphertext mismatch at len {len}");
  assert_eq!(
    tag.as_bytes(),
    expected_tag.as_slice(),
    "AES-256-GCM tag mismatch at len {len}"
  );

  cipher.decrypt_in_place(&nonce, aad, &mut ours, &tag).unwrap();
  assert_eq!(ours, plaintext, "AES-256-GCM open mismatch at len {len}");
}

#[test]
fn aes_gcm_aarch64_asm_matches_oracle_for_all_lengths_to_4096() {
  let aad_cases = [Vec::new(), deterministic_bytes(0x21, 257)];

  for len in 0..=4096 {
    for aad in &aad_cases {
      assert_aes128_matches_oracle(len, aad);
      assert_aes256_matches_oracle(len, aad);
    }
  }
}

#[test]
fn aes_gcm_aarch64_asm_matches_oracle_for_sustained_lengths() {
  let aad_cases = [Vec::new(), deterministic_bytes(0x37, 333)];

  for len in [64 * 1024, 256 * 1024, 1024 * 1024] {
    for aad in &aad_cases {
      assert_aes128_matches_oracle(len, aad);
      assert_aes256_matches_oracle(len, aad);
    }
  }
}
