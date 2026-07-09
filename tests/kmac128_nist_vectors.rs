#![cfg(feature = "kmac")]

use rscrypto::Kmac128;

mod common;
use common::decode_hex_vec as decode_hex;

fn ascending_bytes(len: usize) -> Vec<u8> {
  (0..len).map(|byte| byte as u8).collect()
}

fn key_bytes() -> Vec<u8> {
  (0x40u8..=0x5f).collect()
}

#[test]
fn kmac128_nist_sample_1_matches() {
  let key = key_bytes();
  let data = ascending_bytes(4);
  let expected = decode_hex("e5780b0d3ea6f7d3a429c5706aa43a00fadbd7d49628839e3187243f456ee14e");

  let mut actual = vec![0u8; expected.len()];
  Kmac128::mac_into(&key, b"", &data, &mut actual);
  assert_eq!(actual, expected, "kmac128 sample 1 one-shot mismatch");
  assert!(Kmac128::verify_tag(&key, b"", &data, &expected).is_ok());
}

#[test]
fn kmac128_nist_sample_2_matches() {
  let key = key_bytes();
  let data = ascending_bytes(4);
  let expected = decode_hex("3b1fba963cd8b0b59e8c1a6d71888b7143651af8ba0a7070c0979e2811324aa5");

  let mut actual = vec![0u8; expected.len()];
  Kmac128::mac_into(&key, b"My Tagged Application", &data, &mut actual);
  assert_eq!(actual, expected, "kmac128 sample 2 one-shot mismatch");
  assert!(Kmac128::verify_tag(&key, b"My Tagged Application", &data, &expected).is_ok());
}

#[test]
fn kmac128_nist_sample_3_matches() {
  let key = key_bytes();
  let data = ascending_bytes(200);
  let expected = decode_hex("1f5b4e6cca02209e0dcb5ca635b89a15e271ecc760071dfd805faa38f9729230");

  let mut actual = vec![0u8; expected.len()];
  Kmac128::mac_into(&key, b"My Tagged Application", &data, &mut actual);
  assert_eq!(actual, expected, "kmac128 sample 3 one-shot mismatch");
  assert!(Kmac128::verify_tag(&key, b"My Tagged Application", &data, &expected).is_ok());

  let mut streaming = Kmac128::new(&key, b"My Tagged Application");
  streaming.update(&data[..168]);
  streaming.update(&data[168..]);
  let mut streaming_out = vec![0u8; expected.len()];
  streaming.finalize_into(&mut streaming_out);
  assert_eq!(streaming_out, expected, "kmac128 sample 3 streaming mismatch");
}

#[test]
fn kmac128_verify_rejects_empty_and_corrupted_tags() {
  let key = key_bytes();
  let data = ascending_bytes(200);
  let expected = Kmac128::mac_array::<32>(&key, b"My Tagged Application", &data);

  assert!(
    Kmac128::verify_tag(&key, b"My Tagged Application", &data, &[]).is_err(),
    "KMAC128 must reject an empty expected tag"
  );

  for index in [0, expected.len() / 2, expected.len() - 1] {
    let mut corrupted = expected;
    corrupted[index] ^= 0x80;
    assert!(
      Kmac128::verify_tag(&key, b"My Tagged Application", &data, &corrupted).is_err(),
      "KMAC128 accepted a tag corrupted at byte {index}"
    );
  }
}
