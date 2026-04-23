#![cfg(feature = "hashes")]

use rscrypto::{Cshake256, traits::Xof as _};

mod common;
use common::decode_hex_vec as decode_hex;

fn ascending_bytes(len: usize) -> Vec<u8> {
  (0..len).map(|byte| byte as u8).collect()
}

#[test]
fn cshake256_nist_sample_3_matches() {
  let data = ascending_bytes(4);
  let expected = decode_hex(concat!(
    "d008828e2b80ac9d2218ffee1d070c48b8e4c87bff32c9699d5b6896eee0edd1",
    "64020e2be0560858d9c00c037e34a96937c561a74c412bb4c746469527281c8c"
  ));

  let mut actual = vec![0u8; expected.len()];
  Cshake256::hash_into(b"", b"Email Signature", &data, &mut actual);
  assert_eq!(actual, expected, "cshake256 sample 3 one-shot mismatch");

  let mut streaming = Cshake256::new(b"", b"Email Signature");
  streaming.update(&data[..1]);
  streaming.update(&data[1..]);
  let mut streaming_out = vec![0u8; expected.len()];
  streaming.finalize_xof().squeeze(&mut streaming_out);
  assert_eq!(streaming_out, expected, "cshake256 sample 3 streaming mismatch");
}

#[test]
fn cshake256_nist_sample_4_matches() {
  let data = ascending_bytes(200);
  let expected = decode_hex(concat!(
    "07dc27b11e51fbac75bc7b3c1d983e8b4b85fb1defaf218912ac864302730917",
    "27f42b17ed1df63e8ec118f04b23633c1dfb1574c8fb55cb45da8e25afb092bb"
  ));

  let mut actual = vec![0u8; expected.len()];
  Cshake256::hash_into(b"", b"Email Signature", &data, &mut actual);
  assert_eq!(actual, expected, "cshake256 sample 4 one-shot mismatch");

  let mut streaming = Cshake256::new(b"", b"Email Signature");
  streaming.update(&data[..136]);
  streaming.update(&data[136..]);
  let mut streaming_out = vec![0u8; expected.len()];
  streaming.finalize_xof().squeeze(&mut streaming_out);
  assert_eq!(streaming_out, expected, "cshake256 sample 4 streaming mismatch");
}
