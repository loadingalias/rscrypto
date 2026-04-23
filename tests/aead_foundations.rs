#[cfg(feature = "aead")]
use rscrypto::{
  Aead, ChaCha20Poly1305, ChaCha20Poly1305Key,
  aead::{AeadBufferError, Nonce96, OpenError},
};

#[cfg(feature = "aead")]
fn fixture_cipher() -> ChaCha20Poly1305 {
  ChaCha20Poly1305::new(&ChaCha20Poly1305Key::from_bytes([0x11; ChaCha20Poly1305::KEY_SIZE]))
}

#[test]
#[cfg(feature = "aead")]
fn nonce96_round_trips() {
  let nonce = Nonce96::from_bytes([0xA5; Nonce96::LENGTH]);

  assert_eq!(nonce.to_bytes(), [0xA5; Nonce96::LENGTH]);
  assert_eq!(nonce.as_bytes(), &[0xA5; Nonce96::LENGTH]);
}

#[test]
#[cfg(feature = "aead")]
fn aead_encrypt_and_decrypt_helpers_round_trip() {
  let nonce = Nonce96::from_bytes([0x22; Nonce96::LENGTH]);
  let aad = b"header";
  let plaintext = *b"hello world!";
  let aead = fixture_cipher();

  let mut sealed = [0u8; 12 + ChaCha20Poly1305::TAG_SIZE];
  aead.encrypt(&nonce, aad, &plaintext, &mut sealed).unwrap();

  let mut opened = [0u8; 12];
  aead.decrypt(&nonce, aad, &sealed, &mut opened).unwrap();

  assert_eq!(opened, plaintext);
}

#[test]
#[cfg(feature = "aead")]
fn detached_aliases_match_core_behavior() {
  let nonce = Nonce96::from_bytes([0x55; Nonce96::LENGTH]);
  let aad = b"aad";
  let aead = fixture_cipher();

  let mut left = *b"detached";
  let mut right = left;

  let tag_left = aead.encrypt_in_place(&nonce, aad, &mut left).unwrap();
  let tag_right = aead.encrypt_in_place_detached(&nonce, aad, &mut right).unwrap();

  assert_eq!(left, right);
  assert_eq!(tag_left, tag_right);

  aead
    .decrypt_in_place_detached(&nonce, aad, &mut right, &tag_right)
    .unwrap();
  assert_eq!(right, *b"detached");
}

#[test]
#[cfg(feature = "aead")]
fn aead_open_reports_buffer_and_verification_failures() {
  let nonce = Nonce96::from_bytes([0x88; Nonce96::LENGTH]);
  let aead = fixture_cipher();

  let mut sealed = [0u8; 4 + ChaCha20Poly1305::TAG_SIZE];
  aead.encrypt(&nonce, b"aad", b"data", &mut sealed).unwrap();

  let mut short_out = [0u8; 3];
  assert_eq!(
    aead.decrypt(&nonce, b"aad", &sealed, &mut short_out),
    Err(OpenError::buffer())
  );

  sealed[0] ^= 1;
  let mut opened = [0u8; 4];
  assert_eq!(
    aead.decrypt(&nonce, b"aad", &sealed, &mut opened),
    Err(OpenError::verification())
  );
}

#[test]
#[cfg(feature = "aead")]
fn aead_length_helpers_reject_invalid_sizes() {
  assert_eq!(ChaCha20Poly1305::ciphertext_len(5).unwrap(), 21);
  assert_eq!(ChaCha20Poly1305::plaintext_len(21).unwrap(), 5);
  assert_eq!(ChaCha20Poly1305::plaintext_len(15), Err(AeadBufferError::new()));
}
