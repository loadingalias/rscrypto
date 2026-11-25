#![cfg(feature = "kmac")]

use proptest::prelude::*;
use rscrypto::Kmac128 as Kmac;

mod common;

fn oracle(key: &[u8], customization: &[u8], data: &[u8], out: &mut [u8]) {
  use tiny_keccak::Hasher as _;

  let mut kmac = tiny_keccak::Kmac::v128(key, customization);
  kmac.update(data);
  kmac.finalize(out);
}

#[test]
fn kmac128_exact_rate_key_bytepad_matches_openssl() {
  // OpenSSL 3.6.3 KMAC-128, empty customization/message, 32-byte output.
  let key = [0x42; 163];
  let expected = common::decode_hex_vec("6c9c526b592fdabce43190f544cf5ae6671d223d268eb5a206283df289346fc8");
  let mut actual = [0u8; 32];
  Kmac::mac_into(&key, b"", b"", &mut actual);
  assert_eq!(actual.as_slice(), expected);
}

const RATE: usize = 168;

include!("common/kmac_differential.rs");
