#![cfg(feature = "aead")]

use chacha20poly1305::{
  KeyInit, XChaCha20Poly1305 as Oracle,
  aead::{AeadInPlace, Key, generic_array::GenericArray},
};
use rscrypto::{XChaCha20Poly1305, XChaCha20Poly1305Key, XChaCha20Poly1305Tag, aead::Nonce192};

mod common;
use common::decode_hex_vec as decode_hex;

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

  let key = XChaCha20Poly1305Key::from_bytes(key.try_into().unwrap());
  let nonce = Nonce192::from_bytes(nonce.try_into().unwrap());
  let cipher = XChaCha20Poly1305::new(&key);

  let mut sealed = vec![0u8; plaintext.len() + XChaCha20Poly1305::TAG_SIZE];
  cipher.encrypt(&nonce, &aad, &plaintext, &mut sealed).unwrap();

  assert_eq!(&sealed[..plaintext.len()], expected_ciphertext.as_slice());
  assert_eq!(&sealed[plaintext.len()..], expected_tag.as_slice());

  let mut opened = vec![0u8; plaintext.len()];
  cipher.decrypt(&nonce, &aad, &sealed, &mut opened).unwrap();
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

  let oracle = Oracle::new(Key::<Oracle>::from_slice(&key_bytes));
  let oracle_nonce = GenericArray::clone_from_slice(&nonce_bytes);

  let mut ours = plaintext.to_vec();
  let tag = cipher.encrypt_in_place(&nonce, aad, &mut ours);

  let mut oracle_buffer = plaintext.to_vec();
  let oracle_tag = oracle
    .encrypt_in_place_detached(&oracle_nonce, aad, &mut oracle_buffer)
    .unwrap();

  assert_eq!(ours, oracle_buffer);
  assert_eq!(tag.as_bytes(), oracle_tag.as_slice());

  let typed_tag = XChaCha20Poly1305Tag::from_bytes(tag.to_bytes());
  cipher.decrypt_in_place(&nonce, aad, &mut ours, &typed_tag).unwrap();
  assert_eq!(ours, plaintext);
}

#[test]
fn xchacha20poly1305_rejects_modified_tag() {
  let key = XChaCha20Poly1305Key::from_bytes([0x11; XChaCha20Poly1305::KEY_SIZE]);
  let nonce = Nonce192::from_bytes([0x22; Nonce192::LENGTH]);
  let cipher = XChaCha20Poly1305::new(&key);

  let mut buffer = *b"forgery-check";
  let mut tag = cipher.encrypt_in_place(&nonce, b"aad", &mut buffer).to_bytes();
  tag[0] ^= 1;

  assert!(
    cipher
      .decrypt_in_place(&nonce, b"aad", &mut buffer, &XChaCha20Poly1305Tag::from_bytes(tag))
      .is_err()
  );
}
