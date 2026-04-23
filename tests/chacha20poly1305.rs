#![cfg(feature = "aead")]

use chacha20poly1305::{
  ChaCha20Poly1305 as Oracle, KeyInit,
  aead::{Aead as _, AeadInPlace, Key, Payload, generic_array::GenericArray},
};
use rscrypto::{ChaCha20Poly1305, ChaCha20Poly1305Key, ChaCha20Poly1305Tag, aead::Nonce96};

mod common;
use common::decode_hex_vec as decode_hex;

fn pattern_bytes(len: usize, seed: u8) -> Vec<u8> {
  let mut out = vec![0u8; len];
  for (index, byte) in out.iter_mut().enumerate() {
    *byte = seed
      .wrapping_add((index as u8).wrapping_mul(17))
      .wrapping_add((index as u8).rotate_left(1));
  }
  out
}

#[test]
fn chacha20poly1305_matches_rfc_8439_vector() {
  let plaintext = decode_hex(
    "4c616469657320616e642047656e746c656d656e206f662074686520636c617373206f66202739393a204966204920636f756c64206f6666657220796f75206f6e6c79206f6e652074697020666f7220746865206675747572652c2073756e73637265656e20776f756c642062652069742e",
  );
  let aad = decode_hex("50515253c0c1c2c3c4c5c6c7");
  let key = decode_hex("808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f");
  let nonce = decode_hex("070000004041424344454647");
  let expected_ciphertext = decode_hex(
    "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d63dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b3692ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc3ff4def08e4b7a9de576d26586cec64b6116",
  );
  let expected_tag = decode_hex("1ae10b594f09e26a7e902ecbd0600691");

  let key = ChaCha20Poly1305Key::from_bytes(key.try_into().unwrap());
  let nonce = Nonce96::from_bytes(nonce.try_into().unwrap());
  let cipher = ChaCha20Poly1305::new(&key);

  let mut sealed = vec![0u8; plaintext.len() + ChaCha20Poly1305::TAG_SIZE];
  cipher.encrypt(&nonce, &aad, &plaintext, &mut sealed).unwrap();

  assert_eq!(&sealed[..plaintext.len()], expected_ciphertext.as_slice());
  assert_eq!(&sealed[plaintext.len()..], expected_tag.as_slice());

  let mut opened = vec![0u8; plaintext.len()];
  cipher.decrypt(&nonce, &aad, &sealed, &mut opened).unwrap();
  assert_eq!(opened, plaintext);
}

#[test]
fn chacha20poly1305_matches_rustcrypto_oracle() {
  let key_bytes = [0x42u8; ChaCha20Poly1305::KEY_SIZE];
  let nonce_bytes = [0x24u8; Nonce96::LENGTH];
  let aad = b"rscrypto-chacha-aead";
  let plaintext = b"portable baseline first, SIMD later";

  let key = ChaCha20Poly1305Key::from_bytes(key_bytes);
  let nonce = Nonce96::from_bytes(nonce_bytes);
  let cipher = ChaCha20Poly1305::new(&key);

  let oracle = Oracle::new(Key::<Oracle>::from_slice(&key_bytes));
  let oracle_nonce = GenericArray::clone_from_slice(&nonce_bytes);

  let mut ours = plaintext.to_vec();
  let tag = cipher.encrypt_in_place(&nonce, aad, &mut ours).unwrap();

  let mut oracle_buffer = plaintext.to_vec();
  let oracle_tag = oracle
    .encrypt_in_place_detached(&oracle_nonce, aad, &mut oracle_buffer)
    .unwrap();

  assert_eq!(ours, oracle_buffer);
  assert_eq!(tag.as_bytes(), oracle_tag.as_slice());

  let typed_tag = ChaCha20Poly1305Tag::from_bytes(tag.to_bytes());
  cipher.decrypt_in_place(&nonce, aad, &mut ours, &typed_tag).unwrap();
  assert_eq!(ours, plaintext);
}

#[test]
fn chacha20poly1305_rejects_modified_tag() {
  let key = ChaCha20Poly1305Key::from_bytes([0x11; ChaCha20Poly1305::KEY_SIZE]);
  let nonce = Nonce96::from_bytes([0x22; Nonce96::LENGTH]);
  let cipher = ChaCha20Poly1305::new(&key);

  let mut buffer = *b"forgery-check";
  let mut tag = cipher.encrypt_in_place(&nonce, b"aad", &mut buffer).unwrap().to_bytes();
  tag[0] ^= 1;

  assert!(
    cipher
      .decrypt_in_place(&nonce, b"aad", &mut buffer, &ChaCha20Poly1305Tag::from_bytes(tag))
      .is_err()
  );
}

#[test]
fn chacha20poly1305_rejects_wrong_tag_length() {
  assert!(ChaCha20Poly1305::tag_from_slice(&[0u8; 0]).is_err());
  assert!(ChaCha20Poly1305::tag_from_slice(&[0u8; 15]).is_err());
  assert!(ChaCha20Poly1305::tag_from_slice(&[0u8; 17]).is_err());
  assert!(ChaCha20Poly1305::tag_from_slice(&[0u8; 16]).is_ok());
}

#[test]
fn chacha20poly1305_boundary_and_large_inputs_match_oracle() {
  const PLAINTEXT_LENS: &[usize] = &[
    0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1023, 1024, 4095, 4096, 16_383, 16_384,
  ];
  const AAD_LENS: &[usize] = &[0, 1, 15, 16, 17, 31, 32, 33, 255, 256];

  let key_bytes = [0x42u8; ChaCha20Poly1305::KEY_SIZE];
  let nonce_bytes = [0x24u8; Nonce96::LENGTH];
  let key = ChaCha20Poly1305Key::from_bytes(key_bytes);
  let nonce = Nonce96::from_bytes(nonce_bytes);
  let cipher = ChaCha20Poly1305::new(&key);

  let oracle = Oracle::new(Key::<Oracle>::from_slice(&key_bytes));
  let oracle_nonce = GenericArray::clone_from_slice(&nonce_bytes);

  for &plaintext_len in PLAINTEXT_LENS {
    let plaintext = pattern_bytes(plaintext_len, 0x31);

    for &aad_len in AAD_LENS {
      let aad = pattern_bytes(aad_len, 0x9b);

      let mut combined = vec![0u8; plaintext_len + ChaCha20Poly1305::TAG_SIZE];
      cipher.encrypt(&nonce, &aad, &plaintext, &mut combined).unwrap();

      let oracle_combined = oracle
        .encrypt(
          &oracle_nonce,
          Payload {
            msg: &plaintext,
            aad: &aad,
          },
        )
        .unwrap();
      assert_eq!(
        combined, oracle_combined,
        "combined ciphertext mismatch pt_len={plaintext_len} aad_len={aad_len}"
      );

      let mut opened = vec![0u8; plaintext_len];
      cipher.decrypt(&nonce, &aad, &oracle_combined, &mut opened).unwrap();
      assert_eq!(
        opened, plaintext,
        "combined decrypt mismatch pt_len={plaintext_len} aad_len={aad_len}"
      );

      let oracle_opened = oracle
        .decrypt(
          &oracle_nonce,
          Payload {
            msg: &combined,
            aad: &aad,
          },
        )
        .unwrap();
      assert_eq!(
        oracle_opened, plaintext,
        "oracle decrypt mismatch pt_len={plaintext_len} aad_len={aad_len}"
      );

      let mut detached = plaintext.clone();
      let tag = cipher.encrypt_in_place(&nonce, &aad, &mut detached).unwrap();
      assert_eq!(
        detached,
        oracle_combined[..plaintext_len],
        "detached ciphertext mismatch pt_len={plaintext_len} aad_len={aad_len}"
      );
      assert_eq!(
        tag.as_bytes(),
        &oracle_combined[plaintext_len..],
        "detached tag mismatch pt_len={plaintext_len} aad_len={aad_len}"
      );

      let typed_tag = ChaCha20Poly1305Tag::from_bytes(tag.to_bytes());
      cipher
        .decrypt_in_place(&nonce, &aad, &mut detached, &typed_tag)
        .unwrap();
      assert_eq!(
        detached, plaintext,
        "detached decrypt mismatch pt_len={plaintext_len} aad_len={aad_len}"
      );
    }
  }
}
