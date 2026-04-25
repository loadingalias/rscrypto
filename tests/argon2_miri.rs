//! Miri-friendly Argon2 sanity test.
//!
//! The full proptest differential and RFC 9106 KAT suites in
//! `tests/argon2_differential.rs` and `tests/argon2_vectors.rs` are gated
//! `#![cfg(not(miri))]` because Argon2 hashing under Miri is too slow to be
//! useful. This file exercises the *smallest possible* Argon2 configuration
//! (`m=8 KiB, t=1, p=1, out=4 bytes`) so the portable kernel runs end-to-end
//! under Miri's Stacked-Borrows interpreter — catching memory-safety
//! regressions in `MatrixView`, the address-block stream, and the `H'`
//! variable-length expansion that the larger lanes exercise too quickly to
//! scrutinise.
//!
//! Coverage:
//! - Argon2id with `p=1` (sequential fill path, no rayon).
//! - Argon2id with `p=2` is also exercised when `parallel` is OFF — that forces
//!   `fill_slice_sequential`, which still uses `MatrixView` raw-pointer access. `MatrixView::block`
//!   / `block_mut` carry the unsafe weight.
//!
//! When `parallel` is ON, the rayon-driven fill path is unreachable from
//! Miri (Miri does not support cross-thread synchronisation primitives well
//! enough for rayon's pool); skipping that path is acceptable here because
//! `tests/argon2_differential.rs` covers it on the non-Miri lane and the
//! sequential `MatrixView` exercise still stresses the same raw-pointer
//! invariants.

#![cfg(feature = "argon2")]

use rscrypto::{Argon2Params, Argon2Version, Argon2id};

/// Single deterministic hash with the cheapest valid Argon2id config. Goal
/// is exclusively memory-safety scrutiny under Miri; the actual hash output
/// is irrelevant — we only care that hashing completes without UB.
#[test]
fn argon2id_minimal_p1_no_ub() {
  let params = Argon2Params::new()
    .memory_cost_kib(8) // RFC 9106 lower bound
    .time_cost(1)
    .parallelism(1)
    .output_len(4)
    .version(Argon2Version::V0x13)
    .build()
    .expect("minimal params are valid");

  let mut out = [0u8; 4];
  Argon2id::hash(&params, b"pw", b"abcdefgh", &mut out).expect("Argon2id hash must succeed");

  // Sanity: the output is non-zero (Argon2id can't produce all-zero from
  // a valid password+salt with overwhelming probability — if this fails
  // there's a much bigger bug).
  assert_ne!(out, [0u8; 4], "Argon2id output must not be all-zero");
}

/// `p=2, parallelism=2` exercises `MatrixView::block` / `block_mut` even
/// without rayon — the sequential fill path uses the same raw-pointer view
/// to hand out per-lane mutable borrows.
#[test]
fn argon2id_minimal_p2_no_ub() {
  let params = Argon2Params::new()
    .memory_cost_kib(16) // ≥ 8·p = 16
    .time_cost(1)
    .parallelism(2)
    .output_len(4)
    .version(Argon2Version::V0x13)
    .build()
    .expect("p=2 params are valid");

  let mut out = [0u8; 4];
  Argon2id::hash(&params, b"pw", b"abcdefgh", &mut out).expect("p=2 hash must succeed");

  assert_ne!(out, [0u8; 4]);
}

/// Verify path — exercises the `H'` expansion + constant-time tag compare.
#[test]
fn argon2id_minimal_verify_no_ub() {
  let params = Argon2Params::new()
    .memory_cost_kib(8)
    .time_cost(1)
    .parallelism(1)
    .output_len(4)
    .build()
    .expect("minimal params are valid");

  let mut hash = [0u8; 4];
  Argon2id::hash(&params, b"correct", b"saltsalt", &mut hash).unwrap();

  assert!(Argon2id::verify(&params, b"correct", b"saltsalt", &hash).is_ok());
  assert!(Argon2id::verify(&params, b"wrong", b"saltsalt", &hash).is_err());
}
