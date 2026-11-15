//! Ed25519 verification decides the speccheck edge cases by the strict policy.
//!
//! The twelve cases of `testdata/ed25519/speccheck` separate verifiers by
//! their checks on small-order and non-canonical points, on non-canonical S,
//! and by the cofactored or cofactorless equation. Wycheproof has no such
//! case. Each host runs its production backend; the portable run checks the
//! portable backend.

#![cfg(feature = "ed25519")]

use rscrypto::{Ed25519PublicKey, Ed25519Signature};
use serde_json::Value;

mod common;
use common::decode_hex_vec;

const CASES: &str = include_str!("../testdata/ed25519/speccheck/cases.json");

/// The verdict of each case under the strict policy: reject a small-order or
/// non-canonical A or R and an S of at least L, and check `[S]B = R + [h]A`
/// without the cofactor. Only case 3, with mixed-order A and R that satisfy
/// both equations, verifies.
const STRICT: [bool; 12] = [
  false, false, false, true, false, false, false, false, false, false, false, false,
];

#[test]
fn speccheck_cases_follow_the_strict_policy() {
  let cases: Vec<Value> = serde_json::from_str(CASES).expect("the pinned speccheck JSON parses");
  assert_eq!(cases.len(), STRICT.len(), "speccheck has twelve cases");
  for (index, (case, expected)) in cases.iter().zip(STRICT).enumerate() {
    let field = |name: &str| decode_hex_vec(case[name].as_str().expect(name));
    let message = field("message");
    let public: [u8; 32] = field("pub_key").try_into().expect("32-byte public key");
    let signature: [u8; 64] = field("signature").try_into().expect("64-byte signature");

    let verdict = Ed25519PublicKey::from_bytes(public)
      .verify(&message, &Ed25519Signature::from_bytes(signature))
      .is_ok();
    assert_eq!(verdict, expected, "speccheck case {index}");

    let dalek_strict = ed25519_dalek::VerifyingKey::from_bytes(&public)
      .and_then(|key| key.verify_strict(&message, &ed25519_dalek::Signature::from_bytes(&signature)))
      .is_ok();
    assert_eq!(
      dalek_strict, expected,
      "ed25519-dalek verify_strict, speccheck case {index}"
    );
  }
}
