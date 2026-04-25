//! Argon2 lane-parallelism correctness tests.
//!
//! Gated on `feature = "parallel"`: these stress the rayon-driven slice
//! driver at higher lane counts than the baseline differential suite,
//! and verify that the parallel path is byte-for-byte deterministic and
//! agrees with the RustCrypto `argon2` oracle.
//!
//! The RFC 9106 Appendix A KAT suite (`tests/argon2_vectors.rs`) and the
//! lower-lane differential suite (`tests/argon2_differential.rs`) already
//! exercise the parallel path under `--features parallel` because they
//! use `p ≥ 2`; this file adds the explicit higher-lane oracles plus a
//! determinism check that would catch any data-race-induced corruption.
#![cfg(all(feature = "argon2", feature = "parallel", not(miri)))]

use rscrypto::{Argon2Params, Argon2Version, Argon2d, Argon2i, Argon2id};

fn rs_params(m_kib: u32, t: u32, p: u32, out_len: u32) -> Argon2Params {
  Argon2Params::new()
    .memory_cost_kib(m_kib)
    .time_cost(t)
    .parallelism(p)
    .output_len(out_len)
    .version(Argon2Version::V0x13)
    .build()
    .unwrap()
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

const PASSWORD: &[u8] = b"phase3-parallel-correctness-test";
const SALT: &[u8] = b"rscrypto-parallel-salt-bytes!!";

// ─── High-lane-count oracle parity ─────────────────────────────────────────

/// At p = 8 the rayon driver spawns 8 concurrent tasks per slice. Across
/// 4 slices × `time_cost` passes that is 8·4·t fill_segment calls per
/// hash; any data-race or aliasing bug in the parallel path almost
/// certainly perturbs a block, which would change the final tag.
#[test]
fn argon2id_p8_matches_oracle() {
  let m = 256u32; // m >= 8*p
  let t = 2u32;
  let p = 8u32;
  let out_len = 32usize;

  let params = rs_params(m, t, p, out_len as u32);
  let mut actual = vec![0u8; out_len];
  Argon2id::hash(&params, PASSWORD, SALT, &mut actual).unwrap();

  let expected = oracle_hash(argon2::Algorithm::Argon2id, PASSWORD, SALT, m, t, p, out_len);
  assert_eq!(actual, expected, "argon2id p=8 mismatch vs RustCrypto oracle");
}

#[test]
fn argon2d_p8_matches_oracle() {
  let m = 256u32;
  let t = 2u32;
  let p = 8u32;
  let out_len = 32usize;

  let params = rs_params(m, t, p, out_len as u32);
  let mut actual = vec![0u8; out_len];
  Argon2d::hash(&params, PASSWORD, SALT, &mut actual).unwrap();

  let expected = oracle_hash(argon2::Algorithm::Argon2d, PASSWORD, SALT, m, t, p, out_len);
  assert_eq!(actual, expected, "argon2d p=8 mismatch vs RustCrypto oracle");
}

#[test]
fn argon2i_p8_matches_oracle() {
  let m = 256u32;
  let t = 2u32;
  let p = 8u32;
  let out_len = 32usize;

  let params = rs_params(m, t, p, out_len as u32);
  let mut actual = vec![0u8; out_len];
  Argon2i::hash(&params, PASSWORD, SALT, &mut actual).unwrap();

  let expected = oracle_hash(argon2::Algorithm::Argon2i, PASSWORD, SALT, m, t, p, out_len);
  assert_eq!(actual, expected, "argon2i p=8 mismatch vs RustCrypto oracle");
}

#[test]
fn argon2id_p16_matches_oracle() {
  let m = 512u32; // m >= 8*p; keep small enough for a fast unit test
  let t = 2u32;
  let p = 16u32;
  let out_len = 32usize;

  let params = rs_params(m, t, p, out_len as u32);
  let mut actual = vec![0u8; out_len];
  Argon2id::hash(&params, PASSWORD, SALT, &mut actual).unwrap();

  let expected = oracle_hash(argon2::Algorithm::Argon2id, PASSWORD, SALT, m, t, p, out_len);
  assert_eq!(actual, expected, "argon2id p=16 mismatch vs RustCrypto oracle");
}

// ─── Determinism under parallel scheduling ─────────────────────────────────

/// Hash the same inputs many times in a row at high parallelism. Rayon's
/// scheduler may interleave the lane tasks differently across runs; the
/// fill order must be irrelevant to the output, so every iteration must
/// produce the same tag. Any non-determinism would indicate a data race.
#[test]
fn argon2id_parallel_is_deterministic() {
  let params = rs_params(256, 2, 8, 32);

  let mut reference = [0u8; 32];
  Argon2id::hash(&params, PASSWORD, SALT, &mut reference).unwrap();

  for trial in 0..16 {
    let mut out = [0u8; 32];
    Argon2id::hash(&params, PASSWORD, SALT, &mut out).unwrap();
    assert_eq!(out, reference, "non-deterministic output on trial {trial}");
  }
}

#[test]
fn argon2d_parallel_is_deterministic() {
  let params = rs_params(256, 2, 8, 32);

  let mut reference = [0u8; 32];
  Argon2d::hash(&params, PASSWORD, SALT, &mut reference).unwrap();

  for trial in 0..16 {
    let mut out = [0u8; 32];
    Argon2d::hash(&params, PASSWORD, SALT, &mut out).unwrap();
    assert_eq!(out, reference, "non-deterministic output on trial {trial}");
  }
}

// ─── p = 1 fast path under parallel feature ────────────────────────────────

/// With `parallel` on but `p = 1`, the dispatcher must skip rayon and run
/// the sequential path. This test verifies the fast-path produces the
/// same output as the oracle (and, by transitivity, the same output as
/// the parallel path would on a single lane).
#[test]
fn argon2id_p1_fast_path_matches_oracle() {
  let m = 32u32;
  let t = 3u32;
  let p = 1u32;
  let out_len = 32usize;

  let params = rs_params(m, t, p, out_len as u32);
  let mut actual = vec![0u8; out_len];
  Argon2id::hash(&params, PASSWORD, SALT, &mut actual).unwrap();

  let expected = oracle_hash(argon2::Algorithm::Argon2id, PASSWORD, SALT, m, t, p, out_len);
  assert_eq!(actual, expected, "argon2id p=1 fast-path mismatch vs oracle");
}

// ─── Verify still works under parallel hashing ─────────────────────────────

#[test]
fn argon2id_parallel_verify_round_trip() {
  let params = rs_params(256, 2, 8, 32);
  let mut hash = [0u8; 32];
  Argon2id::hash(&params, PASSWORD, SALT, &mut hash).unwrap();

  assert!(Argon2id::verify(&params, PASSWORD, SALT, &hash).is_ok());
  assert!(Argon2id::verify(&params, b"wrong-password-zzzzzzzzzzzzzzzzzz", SALT, &hash).is_err());
}
