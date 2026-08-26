#![cfg(feature = "websocket-sha1")]

use proptest::prelude::*;
use rscrypto::hashes::legacy::WebSocketAcceptDigest;
use sha1::{Digest as _, Sha1};

const WEBSOCKET_GUID: &[u8] = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

fn oracle(sec_websocket_key: &[u8]) -> [u8; 20] {
  let mut sha1 = Sha1::new();
  sha1.update(sec_websocket_key);
  sha1.update(WEBSOCKET_GUID);
  let output = sha1.finalize();
  let mut digest = [0u8; 20];
  digest.copy_from_slice(&output);
  digest
}

#[test]
fn rfc_6455_example_matches_exact_digest() {
  let digest = WebSocketAcceptDigest::compute(b"dGhlIHNhbXBsZSBub25jZQ==");
  assert_eq!(
    digest.as_ref(),
    [
      0xb3, 0x7a, 0x4f, 0x2c, 0xc0, 0x62, 0x4f, 0x16, 0x90, 0xf6, 0x46, 0x06, 0xcf, 0x38, 0x59, 0x45, 0xb2, 0xbe, 0xc4,
      0xea,
    ]
  );
}

#[test]
fn field_value_is_hashed_byte_for_byte() {
  let original = WebSocketAcceptDigest::compute(b"example-key");
  let leading_space = WebSocketAcceptDigest::compute(b" example-key");
  let trailing_space = WebSocketAcceptDigest::compute(b"example-key ");

  assert_ne!(original, leading_space);
  assert_ne!(original, trailing_space);
  assert_eq!(leading_space.as_ref(), oracle(b" example-key"));
  assert_eq!(trailing_space.as_ref(), oracle(b"example-key "));
}

#[test]
fn padding_and_block_boundaries_match_independent_oracle() {
  for len in [0usize, 1, 19, 20, 27, 28, 55, 56, 63, 64, 127, 128] {
    let key = vec![0xa5; len];
    assert_eq!(
      WebSocketAcceptDigest::compute(&key).as_ref(),
      oracle(&key),
      "WebSocket accept digest mismatch for key length {len}"
    );
  }
}

proptest! {
  #[test]
  fn generated_field_values_match_independent_oracle(
    key in proptest::collection::vec(any::<u8>(), 0..4096),
  ) {
    let ours = WebSocketAcceptDigest::compute(&key);
    prop_assert_eq!(ours.as_ref(), oracle(&key));
  }
}
