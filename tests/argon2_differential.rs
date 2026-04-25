//! Argon2 differential tests against the RustCrypto `argon2` crate.
//!
//! RFC 9106 KAT coverage lives in `tests/argon2_vectors.rs`. This file
//! exercises the d / i / id variants under a spread of cost parameters with
//! proptest-generated passwords and salts.

#![cfg(all(feature = "argon2", not(miri)))]

use proptest::{prelude::*, test_runner::Config as ProptestConfig};
use rscrypto::{Argon2Params, Argon2Version, Argon2d, Argon2i, Argon2id};

/// Number of proptest cases. Argon2 hashing is expensive (one case ≈ 1-100ms
/// at the cost knobs we sweep), so we cap fast iterative dev runs at 16 and
/// open up to 256 in CI release builds for broader parameter coverage. The
/// gate is `cfg!(debug_assertions)`: cargo nextest defaults to debug, while
/// the bench/CI lane runs `--release`. Override with the `PROPTEST_CASES`
/// env var when needed.
const CASES_DEBUG: u32 = 16;
const CASES_RELEASE: u32 = 256;
const fn proptest_cases() -> u32 {
  if cfg!(debug_assertions) {
    CASES_DEBUG
  } else {
    CASES_RELEASE
  }
}

fn oracle_hash(
  algo: argon2::Algorithm,
  password: &[u8],
  salt: &[u8],
  m_kib: u32,
  t: u32,
  p: u32,
  out_len: usize,
) -> Vec<u8> {
  let params = argon2::Params::new(m_kib, t, p, Some(out_len)).unwrap();
  let ctx = argon2::Argon2::new(algo, argon2::Version::V0x13, params);
  let mut out = vec![0u8; out_len];
  ctx.hash_password_into(password, salt, &mut out).unwrap();
  out
}

fn rs_hash_id(password: &[u8], salt: &[u8], m_kib: u32, t: u32, p: u32, out_len: usize) -> Vec<u8> {
  let params = Argon2Params::new()
    .memory_cost_kib(m_kib)
    .time_cost(t)
    .parallelism(p)
    .output_len(out_len as u32)
    .version(Argon2Version::V0x13)
    .build()
    .unwrap();
  let mut out = vec![0u8; out_len];
  Argon2id::hash(&params, password, salt, &mut out).unwrap();
  out
}

fn rs_hash_d(password: &[u8], salt: &[u8], m_kib: u32, t: u32, p: u32, out_len: usize) -> Vec<u8> {
  let params = Argon2Params::new()
    .memory_cost_kib(m_kib)
    .time_cost(t)
    .parallelism(p)
    .output_len(out_len as u32)
    .version(Argon2Version::V0x13)
    .build()
    .unwrap();
  let mut out = vec![0u8; out_len];
  Argon2d::hash(&params, password, salt, &mut out).unwrap();
  out
}

fn rs_hash_i(password: &[u8], salt: &[u8], m_kib: u32, t: u32, p: u32, out_len: usize) -> Vec<u8> {
  let params = Argon2Params::new()
    .memory_cost_kib(m_kib)
    .time_cost(t)
    .parallelism(p)
    .output_len(out_len as u32)
    .version(Argon2Version::V0x13)
    .build()
    .unwrap();
  let mut out = vec![0u8; out_len];
  Argon2i::hash(&params, password, salt, &mut out).unwrap();
  out
}

proptest! {
  #![proptest_config(ProptestConfig::with_cases(proptest_cases()))]

  #[test]
  fn argon2id_matches_oracle(
    password in proptest::collection::vec(any::<u8>(), 0..64),
    salt in proptest::collection::vec(any::<u8>(), 8..32),
    m in proptest::sample::select(vec![8u32, 16, 32, 64]),
    t in 1u32..=3,
    p in 1u32..=2,
    out_len in proptest::sample::select(vec![16usize, 32, 48]),
  ) {
    prop_assume!(m >= 8 * p);
    let actual = rs_hash_id(&password, &salt, m, t, p, out_len);
    let expected = oracle_hash(argon2::Algorithm::Argon2id, &password, &salt, m, t, p, out_len);
    prop_assert_eq!(actual, expected);
  }

  #[test]
  fn argon2d_matches_oracle(
    password in proptest::collection::vec(any::<u8>(), 0..64),
    salt in proptest::collection::vec(any::<u8>(), 8..32),
    m in proptest::sample::select(vec![8u32, 16, 32, 64]),
    t in 1u32..=3,
    p in 1u32..=2,
    out_len in proptest::sample::select(vec![16usize, 32, 48]),
  ) {
    prop_assume!(m >= 8 * p);
    let actual = rs_hash_d(&password, &salt, m, t, p, out_len);
    let expected = oracle_hash(argon2::Algorithm::Argon2d, &password, &salt, m, t, p, out_len);
    prop_assert_eq!(actual, expected);
  }

  #[test]
  fn argon2i_matches_oracle(
    password in proptest::collection::vec(any::<u8>(), 0..64),
    salt in proptest::collection::vec(any::<u8>(), 8..32),
    m in proptest::sample::select(vec![8u32, 16, 32, 64]),
    t in 1u32..=3,
    p in 1u32..=2,
    out_len in proptest::sample::select(vec![16usize, 32, 48]),
  ) {
    prop_assume!(m >= 8 * p);
    let actual = rs_hash_i(&password, &salt, m, t, p, out_len);
    let expected = oracle_hash(argon2::Algorithm::Argon2i, &password, &salt, m, t, p, out_len);
    prop_assert_eq!(actual, expected);
  }

  #[test]
  fn argon2id_verify_accepts_correct(
    password in proptest::collection::vec(any::<u8>(), 1..64),
    salt in proptest::collection::vec(any::<u8>(), 8..32),
    m in proptest::sample::select(vec![8u32, 16, 32]),
    t in 1u32..=2,
    p in 1u32..=2,
  ) {
    prop_assume!(m >= 8 * p);
    let params = Argon2Params::new()
      .memory_cost_kib(m)
      .time_cost(t)
      .parallelism(p)
      .output_len(32)
      .build()
      .unwrap();
    let mut hash = [0u8; 32];
    Argon2id::hash(&params, &password, &salt, &mut hash).unwrap();
    prop_assert!(Argon2id::verify(&params, &password, &salt, &hash).is_ok());
  }
}

/// Forced p ≥ 4 differential. The proptest sweep above caps p at 2 so it
/// runs in reasonable time across hundreds of cases; this fixed-config test
/// exercises the higher-lane parallel-fill path (with `parallel` enabled, it
/// takes the rayon `fill_slice_parallel` route; without, it sequentially
/// iterates 4 lanes — both must match the RustCrypto oracle).
#[test]
fn argon2id_p4_matches_oracle() {
  // m_kib must be ≥ 8·p, so p=4 implies m ≥ 32 KiB.
  let m_kib = 32u32;
  let t = 2u32;
  let p = 4u32;
  let out_len = 32usize;
  let password = b"a sufficiently long password for argon2 hashing tests";
  let salt = b"argon2-p4-fixed-salt";

  let actual = rs_hash_id(password, salt, m_kib, t, p, out_len);
  let expected = oracle_hash(argon2::Algorithm::Argon2id, password, salt, m_kib, t, p, out_len);
  assert_eq!(actual, expected, "Argon2id at p=4 must byte-match RustCrypto oracle");
}

/// Forced p = 8 differential — exercises the maximum-realistic lane count
/// most deployments would use. RFC 9106 supports up to 2^24-1 lanes; we cap
/// at 8 because beyond that the per-lane segment shrinks below useful scope.
#[test]
fn argon2id_p8_matches_oracle() {
  let m_kib = 64u32; // ≥ 8·8 = 64 KiB
  let t = 1u32;
  let p = 8u32;
  let out_len = 32usize;
  let password = b"argon2id p=8 test password";
  let salt = b"argon2id-p8-salt";

  let actual = rs_hash_id(password, salt, m_kib, t, p, out_len);
  let expected = oracle_hash(argon2::Algorithm::Argon2id, password, salt, m_kib, t, p, out_len);
  assert_eq!(actual, expected, "Argon2id at p=8 must byte-match RustCrypto oracle");
}

/// Length-mismatch input must reject. Together with
/// `argon2id_verify_rejects_byte_flip_at_every_position` (byte mismatch),
/// this confirms `verify` rejects every wrong-length tag without panicking
/// or accepting. The wall-clock parity (length-mismatch takes the same
/// hash-dominated cost as a byte-mismatch) is enforced by the implementation
/// — see the doc comment on `verify` in `src/auth/argon2/mod.rs`.
#[test]
fn argon2id_verify_rejects_length_mismatch() {
  let params = Argon2Params::new()
    .memory_cost_kib(8)
    .time_cost(1)
    .parallelism(1)
    .output_len(32)
    .build()
    .unwrap();
  let password = b"pw";
  let salt = b"saltsalt";
  let mut hash = [0u8; 32];
  Argon2id::hash(&params, password, salt, &mut hash).unwrap();

  // Sanity: correct length verifies.
  assert!(Argon2id::verify(&params, password, salt, &hash).is_ok());

  // Various wrong lengths: shorter, longer, empty.
  let too_short: &[u8] = &hash[..16];
  let too_long: &[u8] = &[hash.as_ref(), &[0u8; 8]].concat()[..40];
  let empty: &[u8] = &[];

  assert!(Argon2id::verify(&params, password, salt, too_short).is_err());
  assert!(Argon2id::verify(&params, password, salt, too_long).is_err());
  assert!(Argon2id::verify(&params, password, salt, empty).is_err());

  // And a length one byte off in either direction — the boundary cases.
  let off_minus_one: &[u8] = &hash[..31];
  let off_plus_one: &[u8] = &[hash.as_ref(), &[0u8; 1]].concat();
  assert!(Argon2id::verify(&params, password, salt, off_minus_one).is_err());
  assert!(Argon2id::verify(&params, password, salt, off_plus_one).is_err());
}

#[test]
fn argon2id_verify_rejects_byte_flip_at_every_position() {
  let params = Argon2Params::new()
    .memory_cost_kib(32)
    .time_cost(2)
    .parallelism(1)
    .output_len(32)
    .build()
    .unwrap();
  let password = b"correct horse battery staple";
  let salt = b"random-salt-1234";
  let mut hash = [0u8; 32];
  Argon2id::hash(&params, password, salt, &mut hash).unwrap();

  for pos in 0..hash.len() {
    let mut tampered = hash;
    tampered[pos] ^= 0x01;
    assert!(
      Argon2id::verify(&params, password, salt, &tampered).is_err(),
      "verify must reject flip at byte {pos}"
    );
  }
}

#[cfg(feature = "phc-strings")]
mod phc_roundtrip {
  use super::*;

  proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    fn argon2id_phc_roundtrip_verifies(
      password in proptest::collection::vec(any::<u8>(), 1..48),
      salt in proptest::collection::vec(any::<u8>(), 8..32),
      m in proptest::sample::select(vec![8u32, 16, 32]),
      t in 1u32..=2,
      p in 1u32..=2,
    ) {
      prop_assume!(m >= 8 * p);
      let params = Argon2Params::new()
        .memory_cost_kib(m)
        .time_cost(t)
        .parallelism(p)
        .output_len(32)
        .build()
        .unwrap();
      let encoded = Argon2id::hash_string_with_salt(&params, &password, &salt).unwrap();
      prop_assert!(Argon2id::verify_string(&password, &encoded).is_ok());
    }
  }
}
