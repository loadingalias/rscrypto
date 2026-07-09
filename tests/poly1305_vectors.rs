#![cfg(feature = "poly1305")]

use rscrypto::{Poly1305, Poly1305OneTimeKey, Poly1305Tag};

mod common;
use common::decode_hex_array;

#[test]
fn poly1305_matches_rfc_8439_section_2_5_2() {
  let key = Poly1305OneTimeKey::from_bytes(decode_hex_array(
    "85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b",
  ));
  let message = b"Cryptographic Forum Research Group";
  let expected = Poly1305Tag::from_bytes(decode_hex_array("a8061dc1305136c6c22b8baf0c0127a9"));

  let tag = Poly1305::authenticate_once(key, message);
  assert_eq!(tag, expected);
}

#[test]
fn poly1305_streaming_matches_oneshot() {
  let key = Poly1305OneTimeKey::from_bytes([0x42; Poly1305OneTimeKey::LENGTH]);
  let expected = Poly1305::authenticate_once(Poly1305OneTimeKey::from_bytes([0x42; 32]), b"abcdef");

  let mut poly1305 = Poly1305::new(key);
  poly1305.update(b"a");
  poly1305.update(b"bcde");
  poly1305.update(b"f");

  assert_eq!(poly1305.finalize(), expected);
}

#[test]
fn poly1305_verify_rejects_corrupted_tag() {
  let key = Poly1305OneTimeKey::from_bytes([0x24; Poly1305OneTimeKey::LENGTH]);
  let message = b"message";
  let tag = Poly1305::authenticate_once(Poly1305OneTimeKey::from_bytes([0x24; 32]), message);
  let mut corrupted = tag.to_bytes();
  corrupted[7] ^= 0x80;
  let corrupted = Poly1305Tag::from_bytes(corrupted);

  assert!(Poly1305::verify_once(key, message, &corrupted).is_err());
}

#[test]
fn poly1305_try_generate_with_uses_fallible_fill() {
  let key = Poly1305OneTimeKey::try_generate_with(|out| {
    out.fill(0x11);
    Ok::<(), ()>(())
  })
  .unwrap();

  assert_eq!(key.as_bytes(), &[0x11; Poly1305OneTimeKey::LENGTH]);
  assert!(Poly1305OneTimeKey::try_generate_with(|_| Err("rng")).is_err());
}
