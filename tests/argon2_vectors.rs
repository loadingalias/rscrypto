//! RFC 9106 Appendix A known-answer vectors for Argon2d / i / id.
//!
//! These vectors use the full parameter set (password, salt, secret `K`,
//! associated data `X`) to exercise every field of `H0`. They match the
//! values published in RFC 9106 §A.1–A.3.

#![cfg(all(feature = "argon2", not(miri)))]

use rscrypto::{Argon2Params, Argon2Version, Argon2d, Argon2i, Argon2id};

fn canonical_params() -> Argon2Params {
  Argon2Params::new()
    .memory_cost_kib(32)
    .time_cost(3)
    .parallelism(4)
    .output_len(32)
    .version(Argon2Version::V0x13)
    .secret(&[0x03u8; 8])
    .associated_data(&[0x04u8; 12])
    .build()
    .expect("RFC 9106 canonical parameters are valid")
}

const RFC_PASSWORD: &[u8] = &[0x01u8; 32];
const RFC_SALT: &[u8] = &[0x02u8; 16];

#[test]
fn rfc9106_appendix_a1_argon2d() {
  let expected: [u8; 32] = [
    0x51, 0x2b, 0x39, 0x1b, 0x6f, 0x11, 0x62, 0x97, 0x53, 0x71, 0xd3, 0x09, 0x19, 0x73, 0x42, 0x94, 0xf8, 0x68, 0xe3,
    0xbe, 0x39, 0x84, 0xf3, 0xc1, 0xa1, 0x3a, 0x4d, 0xb9, 0xfa, 0xbe, 0x4a, 0xcb,
  ];
  let mut out = [0u8; 32];
  Argon2d::hash(&canonical_params(), RFC_PASSWORD, RFC_SALT, &mut out).unwrap();
  assert_eq!(out, expected);
}

#[test]
fn rfc9106_appendix_a2_argon2i() {
  let expected: [u8; 32] = [
    0xc8, 0x14, 0xd9, 0xd1, 0xdc, 0x7f, 0x37, 0xaa, 0x13, 0xf0, 0xd7, 0x7f, 0x24, 0x94, 0xbd, 0xa1, 0xc8, 0xde, 0x6b,
    0x01, 0x6d, 0xd3, 0x88, 0xd2, 0x99, 0x52, 0xa4, 0xc4, 0x67, 0x2b, 0x6c, 0xe8,
  ];
  let mut out = [0u8; 32];
  Argon2i::hash(&canonical_params(), RFC_PASSWORD, RFC_SALT, &mut out).unwrap();
  assert_eq!(out, expected);
}

#[test]
fn rfc9106_appendix_a3_argon2id() {
  let expected: [u8; 32] = [
    0x0d, 0x64, 0x0d, 0xf5, 0x8d, 0x78, 0x76, 0x6c, 0x08, 0xc0, 0x37, 0xa3, 0x4a, 0x8b, 0x53, 0xc9, 0xd0, 0x1e, 0xf0,
    0x45, 0x2d, 0x75, 0xb6, 0x5e, 0xb5, 0x25, 0x20, 0xe9, 0x6b, 0x01, 0xe6, 0x59,
  ];
  let mut out = [0u8; 32];
  Argon2id::hash(&canonical_params(), RFC_PASSWORD, RFC_SALT, &mut out).unwrap();
  assert_eq!(out, expected);
}

#[test]
fn all_three_variants_produce_distinct_output() {
  // Independent sanity check — with identical inputs, the three variants
  // must produce three different 32-byte tags (reference-indexing modes
  // diverge).
  let params = canonical_params();
  let d = Argon2d::hash_array::<32>(&params, RFC_PASSWORD, RFC_SALT).unwrap();
  let i = Argon2i::hash_array::<32>(&params, RFC_PASSWORD, RFC_SALT).unwrap();
  let id = Argon2id::hash_array::<32>(&params, RFC_PASSWORD, RFC_SALT).unwrap();
  assert_ne!(d, i);
  assert_ne!(d, id);
  assert_ne!(i, id);
}
