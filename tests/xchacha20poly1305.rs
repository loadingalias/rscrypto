#![cfg(feature = "aead")]

use chacha20poly1305::{
  KeyInit, XChaCha20Poly1305 as Oracle,
  aead::{Aead as _, AeadInOut, Payload, array::Array},
};
use rscrypto::{
  XChaCha20Poly1305, XChaCha20Poly1305Key, XChaCha20Poly1305Tag,
  aead::{Nonce192, expert::AeadWithNonce},
};

mod common;
use common::decode_hex_vec as decode_hex;

fn pattern_bytes(len: usize, seed: u8) -> Vec<u8> {
  let mut out = vec![0u8; len];
  for (index, byte) in out.iter_mut().enumerate() {
    let index = index.to_le_bytes()[0];
    *byte = seed
      .wrapping_add(index.wrapping_mul(19))
      .wrapping_add(index.rotate_left(2));
  }
  out
}

#[test]
fn xchacha20poly1305_matches_draft_vector() {
  let plaintext = decode_hex(
    "4c616469657320616e642047656e746c656d656e206f662074686520636c617373206f66202739393a204966204920636f756c64206f6666657220796f75206f6e6c79206f6e652074697020666f7220746865206675747572652c2073756e73637265656e20776f756c642062652069742e",
  );
  let aad = decode_hex("50515253c0c1c2c3c4c5c6c7");
  let key = decode_hex("808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f");
  let nonce = decode_hex("404142434445464748494a4b4c4d4e4f5051525354555657");
  let expected_ciphertext = decode_hex(
    "bd6d179d3e83d43b9576579493c0e939572a1700252bfaccbed2902c21396cbb731c7f1b0b4aa6440bf3a82f4eda7e39ae64c6708c54c216cb96b72e1213b4522f8c9ba40db5d945b11b69b982c1bb9e3f3fac2bc369488f76b2383565d3fff921f9664c97637da9768812f615c68b13b52e",
  );
  let expected_tag = decode_hex("c0875924c1c7987947deafd8780acf49");

  let key = XChaCha20Poly1305Key::from_bytes(key.try_into().expect("draft XChaCha20-Poly1305 key must be 32 bytes"));
  let nonce = Nonce192::from_bytes(
    nonce
      .try_into()
      .expect("draft XChaCha20-Poly1305 nonce must be 24 bytes"),
  );
  let cipher = XChaCha20Poly1305::new(&key);

  let mut sealed = vec![0u8; plaintext.len() + XChaCha20Poly1305::TAG_SIZE];
  cipher
    .encrypt(&nonce, &aad, &plaintext, &mut sealed)
    .expect("draft XChaCha20-Poly1305 seal buffer must fit plaintext and tag");

  assert_eq!(&sealed[..plaintext.len()], expected_ciphertext.as_slice());
  assert_eq!(&sealed[plaintext.len()..], expected_tag.as_slice());

  let mut opened = vec![0u8; plaintext.len()];
  cipher
    .decrypt(&nonce, &aad, &sealed, &mut opened)
    .expect("draft XChaCha20-Poly1305 ciphertext and tag must authenticate");
  assert_eq!(opened, plaintext);
}

#[test]
fn xchacha20poly1305_matches_rustcrypto_oracle() {
  let key_bytes = [0x42u8; XChaCha20Poly1305::KEY_SIZE];
  let nonce_bytes = [0x24u8; Nonce192::LENGTH];
  let aad = b"rscrypto-xchacha-aead";
  let plaintext = b"portable baseline first, SIMD later";

  let key = XChaCha20Poly1305Key::from_bytes(key_bytes);
  let nonce = Nonce192::from_bytes(nonce_bytes);
  let cipher = XChaCha20Poly1305::new(&key);

  let oracle = Oracle::new(&Array(key_bytes));
  let oracle_nonce = Array(nonce_bytes);

  let mut ours = plaintext.to_vec();
  let tag = cipher
    .encrypt_in_place(&nonce, aad, &mut ours)
    .expect("rscrypto XChaCha20-Poly1305 oracle input must seal");

  let mut oracle_buffer = plaintext.to_vec();
  let oracle_tag = oracle
    .encrypt_inout_detached(&oracle_nonce, aad, oracle_buffer.as_mut_slice().into())
    .expect("RustCrypto XChaCha20-Poly1305 oracle input must seal");

  assert_eq!(ours, oracle_buffer);
  assert_eq!(tag.as_bytes(), oracle_tag.as_slice());

  let typed_tag = XChaCha20Poly1305Tag::from_bytes(tag.to_bytes());
  cipher
    .decrypt_in_place(&nonce, aad, &mut ours, &typed_tag)
    .expect("fresh rscrypto XChaCha20-Poly1305 ciphertext must authenticate");
  assert_eq!(ours, plaintext);
}

#[test]
fn xchacha20poly1305_rejects_modified_tag() {
  let key = XChaCha20Poly1305Key::from_bytes([0x11; XChaCha20Poly1305::KEY_SIZE]);
  let nonce = Nonce192::from_bytes([0x22; Nonce192::LENGTH]);
  let cipher = XChaCha20Poly1305::new(&key);

  let mut buffer = *b"forgery-check";
  let mut tag = cipher
    .encrypt_in_place(&nonce, b"aad", &mut buffer)
    .expect("XChaCha20-Poly1305 forgery fixture must seal")
    .to_bytes();
  tag[0] ^= 1;

  cipher
    .decrypt_in_place(&nonce, b"aad", &mut buffer, &XChaCha20Poly1305Tag::from_bytes(tag))
    .expect_err("modified XChaCha20-Poly1305 tag must fail authentication");
}

#[test]
fn xchacha20poly1305_rejects_wrong_tag_length() {
  XChaCha20Poly1305::tag_from_slice(&[0u8; 0]).expect_err("empty XChaCha20-Poly1305 tag must be rejected");
  XChaCha20Poly1305::tag_from_slice(&[0u8; 15]).expect_err("short XChaCha20-Poly1305 tag must be rejected");
  XChaCha20Poly1305::tag_from_slice(&[0u8; 17]).expect_err("long XChaCha20-Poly1305 tag must be rejected");
  let _tag = XChaCha20Poly1305::tag_from_slice(&[0u8; 16]).expect("16-byte XChaCha20-Poly1305 tag must be accepted");
}

#[test]
fn xchacha20poly1305_boundary_and_large_inputs_match_oracle() {
  const PLAINTEXT_LENS: &[usize] = &[
    0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1023, 1024, 4095, 4096, 16_383, 16_384,
  ];
  const AAD_LENS: &[usize] = &[0, 1, 15, 16, 17, 31, 32, 33, 255, 256];

  let key_bytes = [0x42u8; XChaCha20Poly1305::KEY_SIZE];
  let nonce_bytes = [0x24u8; Nonce192::LENGTH];
  let key = XChaCha20Poly1305Key::from_bytes(key_bytes);
  let nonce = Nonce192::from_bytes(nonce_bytes);
  let cipher = XChaCha20Poly1305::new(&key);

  let oracle = Oracle::new(&Array(key_bytes));
  let oracle_nonce = Array(nonce_bytes);

  for &plaintext_len in PLAINTEXT_LENS {
    let plaintext = pattern_bytes(plaintext_len, 0x27);

    for &aad_len in AAD_LENS {
      let aad = pattern_bytes(aad_len, 0xc4);

      let mut combined = vec![0u8; plaintext_len + XChaCha20Poly1305::TAG_SIZE];
      cipher
        .encrypt(&nonce, &aad, &plaintext, &mut combined)
        .expect("rscrypto combined XChaCha20-Poly1305 oracle input must seal");

      let oracle_combined = oracle
        .encrypt(
          &oracle_nonce,
          Payload {
            msg: &plaintext,
            aad: &aad,
          },
        )
        .expect("RustCrypto combined XChaCha20-Poly1305 oracle input must seal");
      assert_eq!(
        combined, oracle_combined,
        "combined ciphertext mismatch pt_len={plaintext_len} aad_len={aad_len}"
      );

      let mut opened = vec![0u8; plaintext_len];
      cipher
        .decrypt(&nonce, &aad, &oracle_combined, &mut opened)
        .expect("RustCrypto XChaCha20-Poly1305 ciphertext must open in rscrypto");
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
        .expect("rscrypto XChaCha20-Poly1305 ciphertext must open in RustCrypto");
      assert_eq!(
        oracle_opened, plaintext,
        "oracle decrypt mismatch pt_len={plaintext_len} aad_len={aad_len}"
      );

      let mut detached = plaintext.clone();
      let tag = cipher
        .encrypt_in_place(&nonce, &aad, &mut detached)
        .expect("rscrypto detached XChaCha20-Poly1305 oracle input must seal");
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

      let typed_tag = XChaCha20Poly1305Tag::from_bytes(tag.to_bytes());
      cipher
        .decrypt_in_place(&nonce, &aad, &mut detached, &typed_tag)
        .expect("fresh detached XChaCha20-Poly1305 ciphertext must authenticate");
      assert_eq!(
        detached, plaintext,
        "detached decrypt mismatch pt_len={plaintext_len} aad_len={aad_len}"
      );
    }
  }
}
