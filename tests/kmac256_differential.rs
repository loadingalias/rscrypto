#![cfg(feature = "kmac")]

use proptest::prelude::*;
use rscrypto::Kmac256 as Kmac;

mod common;

fn oracle(key: &[u8], customization: &[u8], data: &[u8], out: &mut [u8]) {
  use tiny_keccak::Hasher as _;

  let mut kmac = tiny_keccak::Kmac::v256(key, customization);
  kmac.update(data);
  kmac.finalize(out);
}

#[test]
fn kmac256_exact_rate_key_bytepad_matches_openssl() {
  // OpenSSL 3.6.3 KMAC-256, empty customization/message, 64-byte output.
  let key = [0x42; 131];
  let expected = common::decode_hex_vec(
    "875ea09c011f7ab1f6238aeac8bc0f88951a567be7447cd23a6c6187a086c94a64202d2c1f46ab1ddfdff61d173eba49fcf3039a70d088c908b46f3c3693a6d9",
  );
  let mut actual = [0u8; 64];
  Kmac::mac_into(&key, b"", b"", &mut actual);
  assert_eq!(actual.as_slice(), expected);
}

const RATE: usize = 136;

include!("common/kmac_differential.rs");
