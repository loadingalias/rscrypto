#[cfg(all(feature = "aead", feature = "getrandom"))]
use rscrypto::aead::{RandomSealError, SealError};
#[cfg(feature = "aead")]
use rscrypto::{
  Aead, Aegis256, Aegis256Key, Aegis256Tag, Aes128Gcm, Aes128GcmKey, Aes128GcmSiv, Aes128GcmSivKey, Aes128GcmSivTag,
  Aes128GcmTag, Aes256Gcm, Aes256GcmKey, Aes256GcmSiv, Aes256GcmSivKey, Aes256GcmSivTag, Aes256GcmTag, AsconAead128,
  AsconAead128Key, AsconAead128Tag, ChaCha20Poly1305, ChaCha20Poly1305Key, ChaCha20Poly1305Tag, XChaCha20Poly1305,
  XChaCha20Poly1305Key, XChaCha20Poly1305Tag,
  aead::{AeadBufferError, Nonce96, Nonce128, Nonce192, Nonce256, OpenError},
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
#[cfg(all(feature = "aead", feature = "getrandom"))]
fn aead_seal_random_issues_fresh_nonces_and_round_trips() {
  let aad = b"header";
  let aead = fixture_cipher();

  let mut sealed_a = [0u8; 12 + ChaCha20Poly1305::TAG_SIZE];
  let nonce_a = aead.seal_random(aad, b"hello world!", &mut sealed_a).unwrap();

  let mut sealed_b = [0u8; 12 + ChaCha20Poly1305::TAG_SIZE];
  let nonce_b = aead.seal_random(aad, b"hello world!", &mut sealed_b).unwrap();

  assert_ne!(nonce_a, nonce_b, "successive random seals must not reuse a nonce");

  let mut opened = [0u8; 12];
  aead.decrypt(&nonce_a, aad, &sealed_a, &mut opened).unwrap();
  assert_eq!(&opened, b"hello world!");
}

#[test]
#[cfg(all(feature = "aead", feature = "getrandom"))]
fn aead_seal_random_in_place_returns_nonce_and_tag_for_open() {
  let aad = b"header";
  let aead = fixture_cipher();
  let mut buffer = *b"detached";

  let (nonce, tag) = aead.seal_random_in_place(aad, &mut buffer).unwrap();
  assert_ne!(&buffer, b"detached");

  aead.decrypt_in_place(&nonce, aad, &mut buffer, &tag).unwrap();
  assert_eq!(&buffer, b"detached");
}

#[test]
#[cfg(all(feature = "aead", feature = "getrandom"))]
fn aead_seal_random_maps_buffer_errors_without_mutating_output() {
  let aead = fixture_cipher();
  let mut out = [0xA5; 3];

  let err = aead.seal_random(b"", b"data", &mut out).unwrap_err();

  assert_eq!(err, RandomSealError::seal(SealError::buffer()));
  assert_eq!(
    out, [0xA5; 3],
    "combined random seal must preserve encrypt() buffer-shape failure behavior"
  );
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
  let mut opened = [0xA5; 4];
  assert_eq!(
    aead.decrypt(&nonce, b"aad", &sealed, &mut opened),
    Err(OpenError::verification())
  );
  assert_eq!(
    opened, [0u8; 4],
    "combined decrypt must clear output on verification failure"
  );
}

#[test]
#[cfg(feature = "aead")]
fn all_aeads_clear_in_place_buffer_on_verification_failure() {
  macro_rules! assert_clears {
    ($name:literal, $cipher:expr, $nonce:expr, $tag:ty) => {{
      let cipher = $cipher;
      let nonce = $nonce;
      let mut buffer = *b"failed-open-clear";
      let tag = cipher.encrypt_in_place(&nonce, b"aad", &mut buffer).unwrap();
      let mut bad_tag = tag.to_bytes();
      bad_tag[0] ^= 0x80;
      let bad_tag = <$tag>::from_bytes(bad_tag);

      assert_eq!(
        cipher.decrypt_in_place(&nonce, b"aad", &mut buffer, &bad_tag),
        Err(OpenError::verification()),
        "{} must reject a forged tag",
        $name
      );
      assert!(
        buffer.iter().all(|&byte| byte == 0),
        "{} must clear the caller buffer on failed open",
        $name
      );
    }};
  }

  assert_clears!(
    "ChaCha20-Poly1305",
    ChaCha20Poly1305::new(&ChaCha20Poly1305Key::from_bytes([0x11; ChaCha20Poly1305::KEY_SIZE])),
    Nonce96::from_bytes([0x21; Nonce96::LENGTH]),
    ChaCha20Poly1305Tag
  );
  assert_clears!(
    "XChaCha20-Poly1305",
    XChaCha20Poly1305::new(&XChaCha20Poly1305Key::from_bytes([0x12; XChaCha20Poly1305::KEY_SIZE])),
    Nonce192::from_bytes([0x22; Nonce192::LENGTH]),
    XChaCha20Poly1305Tag
  );
  assert_clears!(
    "AES-128-GCM",
    Aes128Gcm::new(&Aes128GcmKey::from_bytes([0x13; Aes128Gcm::KEY_SIZE])),
    Nonce96::from_bytes([0x23; Nonce96::LENGTH]),
    Aes128GcmTag
  );
  assert_clears!(
    "AES-256-GCM",
    Aes256Gcm::new(&Aes256GcmKey::from_bytes([0x14; Aes256Gcm::KEY_SIZE])),
    Nonce96::from_bytes([0x24; Nonce96::LENGTH]),
    Aes256GcmTag
  );
  assert_clears!(
    "AES-128-GCM-SIV",
    Aes128GcmSiv::new(&Aes128GcmSivKey::from_bytes([0x15; Aes128GcmSiv::KEY_SIZE])),
    Nonce96::from_bytes([0x25; Nonce96::LENGTH]),
    Aes128GcmSivTag
  );
  assert_clears!(
    "AES-256-GCM-SIV",
    Aes256GcmSiv::new(&Aes256GcmSivKey::from_bytes([0x16; Aes256GcmSiv::KEY_SIZE])),
    Nonce96::from_bytes([0x26; Nonce96::LENGTH]),
    Aes256GcmSivTag
  );
  assert_clears!(
    "AEGIS-256",
    Aegis256::new(&Aegis256Key::from_bytes([0x17; Aegis256::KEY_SIZE])),
    Nonce256::from_bytes([0x27; Nonce256::LENGTH]),
    Aegis256Tag
  );
  assert_clears!(
    "Ascon-AEAD128",
    AsconAead128::new(&AsconAead128Key::from_bytes([0x18; AsconAead128::KEY_SIZE])),
    Nonce128::from_bytes([0x28; Nonce128::LENGTH]),
    AsconAead128Tag
  );
}

#[test]
#[cfg(feature = "aead")]
fn aead_length_helpers_reject_invalid_sizes() {
  assert_eq!(ChaCha20Poly1305::ciphertext_len(5).unwrap(), 21);
  assert_eq!(ChaCha20Poly1305::plaintext_len(21).unwrap(), 5);
  assert_eq!(ChaCha20Poly1305::plaintext_len(15), Err(AeadBufferError::new()));
}
