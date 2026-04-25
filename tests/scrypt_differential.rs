//! scrypt differential tests against the RustCrypto `scrypt` crate.
//!
//! RFC 7914 KAT coverage lives in `tests/scrypt_vectors.rs`. This file
//! exercises a spread of cost parameters with proptest-generated
//! passwords and salts, plus a deterministic byte-flip-at-every-position
//! test that pins the verify contract.

#![cfg(all(feature = "scrypt", not(miri)))]

use proptest::{prelude::*, test_runner::Config as ProptestConfig};
use rscrypto::{Scrypt, ScryptParams};

fn oracle_scrypt(password: &[u8], salt: &[u8], log_n: u8, r: u32, p: u32, out_len: usize) -> Vec<u8> {
  let params = scrypt::Params::new(log_n, r, p, out_len).unwrap();
  let mut out = vec![0u8; out_len];
  scrypt::scrypt(password, salt, &params, &mut out).unwrap();
  out
}

fn rs_hash(password: &[u8], salt: &[u8], log_n: u8, r: u32, p: u32, out_len: usize) -> Vec<u8> {
  let params = ScryptParams::new()
    .log_n(log_n)
    .r(r)
    .p(p)
    .output_len(out_len as u32)
    .build()
    .unwrap();
  let mut out = vec![0u8; out_len];
  Scrypt::hash(&params, password, salt, &mut out).unwrap();
  out
}

proptest! {
  #![proptest_config(ProptestConfig::with_cases(16))]

  #[test]
  fn scrypt_matches_oracle_small_params(
    password in proptest::collection::vec(any::<u8>(), 0..64),
    salt in proptest::collection::vec(any::<u8>(), 0..32),
    log_n in 4u8..=7,
    r in 1u32..=4,
    p in 1u32..=2,
    // The `scrypt 0.11` oracle rejects short `out_len` (< 10) and imposes
    // an upper bound tied to `(log_n, r)`. We exercise the PBKDF2
    // single-byte tail in `scrypt_short_dklen_is_prefix_of_wide` below and
    // cover the common dkLen choices here.
    out_len in proptest::sample::select(vec![16usize, 32, 48, 64]),
  ) {
    let actual = rs_hash(&password, &salt, log_n, r, p, out_len);
    let expected = oracle_scrypt(&password, &salt, log_n, r, p, out_len);
    prop_assert_eq!(actual, expected);
  }

  #[test]
  fn scrypt_verify_accepts_correct(
    password in proptest::collection::vec(any::<u8>(), 1..48),
    salt in proptest::collection::vec(any::<u8>(), 8..32),
    log_n in 4u8..=6,
    r in 1u32..=2,
    p in 1u32..=2,
  ) {
    let params = ScryptParams::new()
      .log_n(log_n)
      .r(r)
      .p(p)
      .output_len(32)
      .build()
      .unwrap();
    let mut hash = [0u8; 32];
    Scrypt::hash(&params, &password, &salt, &mut hash).unwrap();
    prop_assert!(Scrypt::verify(&params, &password, &salt, &hash).is_ok());
  }
}

#[test]
fn scrypt_minimal_log_n_matches_oracle() {
  // `log_n = 1` (N = 2) is the smallest value `ScryptParams::validate`
  // accepts. In the pair-unrolled ROMix it gives `pairs = 1`, so each loop
  // executes exactly one pair — the boundary case for the pair-unroll.
  // Any off-by-one in the ping-pong would surface here first.
  for r in 1u32..=3 {
    for p in 1u32..=2 {
      let actual = rs_hash(b"password", b"salty-salty-salt", 1, r, p, 32);
      let expected = oracle_scrypt(b"password", b"salty-salty-salt", 1, r, p, 32);
      assert_eq!(actual, expected, "log_n=1 r={r} p={p} mismatch");
    }
  }
}

#[test]
fn scrypt_short_dklen_is_prefix_of_wide() {
  // PBKDF2-HMAC-SHA256's output is a concatenation of blocks; asking for
  // `N` bytes yields the first `N` bytes of the block stream. Therefore
  // `Scrypt::hash(.., out_len=1)` must equal the first byte of
  // `Scrypt::hash(.., out_len=64)` for the same inputs. Exercises the
  // `out_len=1` PBKDF2 single-byte tail that the oracle rejects.
  let params_short = rscrypto::ScryptParams::new()
    .log_n(6)
    .r(2)
    .p(1)
    .output_len(1)
    .build()
    .unwrap();
  let params_wide = rscrypto::ScryptParams::new()
    .log_n(6)
    .r(2)
    .p(1)
    .output_len(64)
    .build()
    .unwrap();
  let mut short_out = [0u8; 1];
  let mut wide_out = [0u8; 64];
  Scrypt::hash(&params_short, b"pw", b"salty-salty-salt", &mut short_out).unwrap();
  Scrypt::hash(&params_wide, b"pw", b"salty-salty-salt", &mut wide_out).unwrap();
  assert_eq!(short_out[0], wide_out[0]);
}

#[test]
fn scrypt_verify_rejects_byte_flip_at_every_position() {
  let params = ScryptParams::new().log_n(6).r(2).p(1).output_len(32).build().unwrap();
  let password = b"correct horse battery staple";
  let salt = b"random-salt-1234";
  let mut hash = [0u8; 32];
  Scrypt::hash(&params, password, salt, &mut hash).unwrap();

  for pos in 0..hash.len() {
    let mut tampered = hash;
    tampered[pos] ^= 0x01;
    assert!(
      Scrypt::verify(&params, password, salt, &tampered).is_err(),
      "verify must reject flip at byte {pos}",
    );
  }
}

#[cfg(feature = "phc-strings")]
mod phc_roundtrip {
  use super::*;

  proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    fn scrypt_phc_roundtrip_verifies(
      password in proptest::collection::vec(any::<u8>(), 1..48),
      salt in proptest::collection::vec(any::<u8>(), 8..32),
      log_n in 4u8..=6,
      r in 1u32..=2,
      p in 1u32..=2,
    ) {
      let params = ScryptParams::new()
        .log_n(log_n)
        .r(r)
        .p(p)
        .output_len(32)
        .build()
        .unwrap();
      let encoded = Scrypt::hash_string_with_salt(&params, &password, &salt).unwrap();
      prop_assert!(Scrypt::verify_string(&password, &encoded).is_ok());
    }
  }
}
