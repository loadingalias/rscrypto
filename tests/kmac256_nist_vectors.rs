#![cfg(feature = "kmac")]

use rscrypto::Kmac256;

mod common;
use common::decode_hex_vec as decode_hex;

fn ascending_bytes(len: usize) -> Vec<u8> {
  (0..len).map(|byte| byte as u8).collect()
}

fn key_bytes() -> Vec<u8> {
  (0x40u8..=0x5f).collect()
}

#[test]
fn kmac256_nist_sample_4_matches() {
  let key = key_bytes();
  let data = ascending_bytes(4);
  let expected = decode_hex(concat!(
    "20c570c31346f703c9ac36c61c03cb64c3970d0cfc787e9b79599d273a68d2f7",
    "f69d4cc3de9d104a351689f27cf6f5951f0103f33f4f24871024d9c27773a8dd"
  ));

  let mut actual = vec![0u8; expected.len()];
  Kmac256::mac_into(&key, b"My Tagged Application", &data, &mut actual);
  assert_eq!(actual, expected, "kmac256 sample 4 one-shot mismatch");
  assert!(Kmac256::verify_tag(&key, b"My Tagged Application", &data, &expected).is_ok());

  let mut streaming = Kmac256::new(&key, b"My Tagged Application");
  streaming.update(&data[..1]);
  streaming.update(&data[1..]);
  let mut streaming_out = vec![0u8; expected.len()];
  streaming.finalize_into(&mut streaming_out);
  assert_eq!(streaming_out, expected, "kmac256 sample 4 streaming mismatch");
}

#[test]
fn kmac256_nist_sample_5_matches() {
  let key = key_bytes();
  let data = ascending_bytes(200);
  let expected = decode_hex(concat!(
    "75358cf39e41494e949707927cee0af20a3ff553904c86b08f21cc414bcfd691",
    "589d27cf5e15369cbbff8b9a4c2eb17800855d0235ff635da82533ec6b759b69"
  ));

  let mut actual = vec![0u8; expected.len()];
  Kmac256::mac_into(&key, b"", &data, &mut actual);
  assert_eq!(actual, expected, "kmac256 sample 5 one-shot mismatch");
  assert!(Kmac256::verify_tag(&key, b"", &data, &expected).is_ok());

  let mut streaming = Kmac256::new(&key, b"");
  streaming.update(&data[..136]);
  streaming.update(&data[136..]);
  let mut streaming_out = vec![0u8; expected.len()];
  streaming.finalize_into(&mut streaming_out);
  assert_eq!(streaming_out, expected, "kmac256 sample 5 streaming mismatch");
}

#[test]
fn kmac256_nist_sample_6_matches() {
  let key = key_bytes();
  let data = ascending_bytes(200);
  let expected = decode_hex(concat!(
    "b58618f71f92e1d56c1b8c55ddd7cd188b97b4ca4d99831eb2699a837da2e4d9",
    "70fbacfde50033aea585f1a2708510c32d07880801bd182898fe476876fc8965"
  ));

  let mut actual = vec![0u8; expected.len()];
  Kmac256::mac_into(&key, b"My Tagged Application", &data, &mut actual);
  assert_eq!(actual, expected, "kmac256 sample 6 one-shot mismatch");
  assert!(Kmac256::verify_tag(&key, b"My Tagged Application", &data, &expected).is_ok());

  let mut streaming = Kmac256::new(&key, b"My Tagged Application");
  streaming.update(&data[..136]);
  streaming.update(&data[136..]);
  let mut streaming_out = vec![0u8; expected.len()];
  streaming.finalize_into(&mut streaming_out);
  assert_eq!(streaming_out, expected, "kmac256 sample 6 streaming mismatch");
}
