//! Integration oracle tests for Ascon hash and XOF.
//!
//! Validates rscrypto's Ascon-Hash256 and Ascon-XOF128 against the `ascon-hash`
//! crate (RustCrypto `digest`-based API) across multiple input sizes.
//!
//! The existing `ascon_differential.rs` tests only verify self-consistency
//! (streaming == oneshot). This file adds cross-implementation validation.
//!
//! # Coverage
//!
//! 1. Ascon-Hash256 matches `ascon-hash` oracle (oneshot + streaming)
//! 2. Ascon-XOF128 matches `ascon-hash` oracle (multiple squeeze lengths)
//! 3. Empty input, single byte, block boundaries, large inputs

#![cfg(feature = "hashes")]

use ascon_hash::{AsconHash256 as OracleHash, AsconXof128 as OracleXof};
use digest::{Digest as OracleDigest, ExtendableOutput, Update as OracleUpdate, XofReader as OracleXofReader};
use rscrypto::{
  hashes::crypto::{AsconHash256, AsconXof},
  traits::{Digest as _, Xof as _},
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Ascon-Hash256 Oracle
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn assert_hash_matches_oracle(input: &[u8]) {
  let ours = AsconHash256::digest(input);

  let mut oracle_h = OracleHash::new();
  OracleUpdate::update(&mut oracle_h, input);
  let oracle: [u8; 32] = OracleDigest::finalize(oracle_h).into();

  assert_eq!(ours, oracle, "Ascon-Hash256 mismatch (len={})", input.len());
}

#[test]
fn ascon_hash256_oracle_empty() {
  assert_hash_matches_oracle(b"");
}

#[test]
fn ascon_hash256_oracle_short() {
  assert_hash_matches_oracle(b"a");
  assert_hash_matches_oracle(b"abc");
  assert_hash_matches_oracle(b"rscrypto");
}

#[test]
fn ascon_hash256_oracle_rate_boundaries() {
  // Ascon hash rate = 8 bytes.
  for size in [1, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33, 64, 128] {
    let input = vec![0xABu8; size];
    assert_hash_matches_oracle(&input);
  }
}

#[test]
fn ascon_hash256_oracle_large() {
  let input: Vec<u8> = (0..8192).map(|i| (i & 0xFF) as u8).collect();
  assert_hash_matches_oracle(&input);
}

#[test]
fn ascon_hash256_oracle_streaming() {
  let data = b"streaming oracle comparison across chunk boundaries";

  // rscrypto streaming.
  let mut ours = AsconHash256::new();
  ours.update(&data[..10]);
  ours.update(&data[10..]);
  let ours_digest = ours.finalize();

  // Oracle streaming.
  let mut oracle = OracleHash::new();
  OracleUpdate::update(&mut oracle, &data[..10]);
  OracleUpdate::update(&mut oracle, &data[10..]);
  let oracle_digest: [u8; 32] = OracleDigest::finalize(oracle).into();

  assert_eq!(ours_digest, oracle_digest);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Ascon-XOF128 Oracle
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn assert_xof_matches_oracle(input: &[u8], squeeze_len: usize) {
  // rscrypto XOF.
  let mut ours_reader = AsconXof::xof(input);
  let mut ours = vec![0u8; squeeze_len];
  ours_reader.squeeze(&mut ours);

  // Oracle XOF.
  let mut oracle_xof = OracleXof::default();
  OracleUpdate::update(&mut oracle_xof, input);
  let mut oracle_reader = ExtendableOutput::finalize_xof(oracle_xof);
  let mut oracle = vec![0u8; squeeze_len];
  OracleXofReader::read(&mut oracle_reader, &mut oracle);

  assert_eq!(
    ours,
    oracle,
    "Ascon-XOF128 mismatch (input_len={}, squeeze_len={})",
    input.len(),
    squeeze_len
  );
}

#[test]
fn ascon_xof128_oracle_empty_input() {
  assert_xof_matches_oracle(b"", 32);
  assert_xof_matches_oracle(b"", 64);
  assert_xof_matches_oracle(b"", 128);
}

#[test]
fn ascon_xof128_oracle_short() {
  assert_xof_matches_oracle(b"abc", 32);
  assert_xof_matches_oracle(b"rscrypto", 64);
}

#[test]
fn ascon_xof128_oracle_rate_boundaries() {
  for size in [1, 7, 8, 9, 15, 16, 17, 32, 64] {
    let input = vec![0xCDu8; size];
    assert_xof_matches_oracle(&input, 32);
    assert_xof_matches_oracle(&input, 64);
  }
}

#[test]
fn ascon_xof128_oracle_large_squeeze() {
  // Large squeeze output to test multi-block XOF.
  assert_xof_matches_oracle(b"large squeeze test", 1024);
}

#[test]
fn ascon_xof128_oracle_large_input() {
  let input: Vec<u8> = (0..4096).map(|i| (i & 0xFF) as u8).collect();
  assert_xof_matches_oracle(&input, 64);
}

#[test]
fn ascon_xof128_oracle_streaming_multi_squeeze() {
  let data = b"multi-squeeze streaming comparison";

  // rscrypto: stream input, squeeze in two parts.
  let mut h = AsconXof::new();
  h.update(&data[..15]);
  h.update(&data[15..]);
  let mut reader = h.finalize_xof();
  let mut ours_a = [0u8; 32];
  let mut ours_b = [0u8; 32];
  reader.squeeze(&mut ours_a);
  reader.squeeze(&mut ours_b);

  // Oracle: same pattern.
  let mut oracle_h = OracleXof::default();
  OracleUpdate::update(&mut oracle_h, &data[..15]);
  OracleUpdate::update(&mut oracle_h, &data[15..]);
  let mut oracle_reader = ExtendableOutput::finalize_xof(oracle_h);
  let mut oracle_a = [0u8; 32];
  let mut oracle_b = [0u8; 32];
  OracleXofReader::read(&mut oracle_reader, &mut oracle_a);
  OracleXofReader::read(&mut oracle_reader, &mut oracle_b);

  assert_eq!(ours_a, oracle_a, "first squeeze mismatch");
  assert_eq!(ours_b, oracle_b, "second squeeze mismatch");
}
