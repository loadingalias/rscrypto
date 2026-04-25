//! X25519 differential oracle vs `x25519-dalek`.
//!
//! Complementary to `tests/x25519_vectors.rs`:
//!
//! - `tests/x25519_vectors.rs` — RFC 7748 KAT vectors plus a deterministic 32-seed differential.
//!   Hand-curated.
//! - this file — proptest-driven sweep across arbitrary 32-byte scalars and peer keys, plus
//!   commutativity properties that don't appear in the vector-table file.
//!
//! Goal: catch any divergence between the rscrypto Edwards-backed Montgomery
//! ladder and the dalek reference implementation across a much broader input
//! space than hand-written cases. Per the audit's M-15 finding, the
//! correctness oracle here is `x25519-dalek` itself — both crates are stable
//! and any byte-level disagreement is a real bug in one of them.

#![cfg(all(feature = "x25519", not(miri)))]

use proptest::prelude::*;
use rscrypto::{X25519Error, X25519PublicKey, X25519SecretKey};
use x25519_dalek::{PublicKey as DalekPublicKey, StaticSecret as DalekStaticSecret};

const PROPTEST_CASES: u32 = if cfg!(debug_assertions) { 64 } else { 256 };

fn arb_32_bytes() -> impl Strategy<Value = [u8; 32]> {
  proptest::array::uniform32(any::<u8>())
}

proptest::proptest! {
  #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

  /// Public key derivation must be byte-identical between rscrypto and dalek.
  #[test]
  fn public_key_byte_identical_with_dalek(secret_bytes in arb_32_bytes()) {
    let ours_pk = X25519SecretKey::from_bytes(secret_bytes).public_key();
    let dalek_pk = DalekPublicKey::from(&DalekStaticSecret::from(secret_bytes));
    prop_assert_eq!(ours_pk.to_bytes(), dalek_pk.to_bytes());
  }

  /// Diffie-Hellman shared secret must agree on every non-low-order pair.
  /// Low-order outputs (all-zero shared secret) yield `X25519Error` here and
  /// the all-zero array on the dalek side; this property tests both arms.
  #[test]
  fn shared_secret_matches_dalek(secret_bytes in arb_32_bytes(), peer_bytes in arb_32_bytes()) {
    let ours_secret = X25519SecretKey::from_bytes(secret_bytes);
    let ours_peer = X25519PublicKey::from_bytes(peer_bytes);

    let dalek_secret = DalekStaticSecret::from(secret_bytes);
    let dalek_peer = DalekPublicKey::from(peer_bytes);
    let dalek_shared = dalek_secret.diffie_hellman(&dalek_peer).to_bytes();

    let ours_result = ours_secret.diffie_hellman(&ours_peer);

    if dalek_shared == [0u8; 32] {
      prop_assert_eq!(ours_result, Err(X25519Error::new()),
        "dalek emitted all-zero shared secret but rscrypto did not reject");
    } else {
      let ours = ours_result.expect("rscrypto rejected non-low-order point that dalek accepted");
      prop_assert_eq!(*ours.as_bytes(), dalek_shared);
    }
  }

  /// DH commutativity: alice·B(bob) == bob·B(alice). Tests the Montgomery
  /// ladder symmetry across the full input space, not just the hand-curated
  /// seeds in tests/x25519_vectors.rs.
  #[test]
  fn dh_commutativity(alice_bytes in arb_32_bytes(), bob_bytes in arb_32_bytes()) {
    let alice = X25519SecretKey::from_bytes(alice_bytes);
    let bob = X25519SecretKey::from_bytes(bob_bytes);

    let alice_pub = alice.public_key();
    let bob_pub = bob.public_key();

    let alice_view = alice.diffie_hellman(&bob_pub);
    let bob_view = bob.diffie_hellman(&alice_pub);

    match (alice_view, bob_view) {
      (Ok(a), Ok(b)) => prop_assert_eq!(a.as_bytes(), b.as_bytes()),
      (Err(_), Err(_)) => {} // Both correctly identified a low-order pair.
      (Ok(_), Err(_)) | (Err(_), Ok(_)) => {
        prop_assert!(false, "asymmetric low-order rejection in DH commutativity");
      }
    }
  }

  /// Cross-implementation DH: rscrypto secret + dalek peer must match
  /// dalek secret + rscrypto peer when both pairs derive from the same
  /// scalar bytes. Tests inter-operability with downstream dalek consumers.
  #[test]
  fn cross_impl_dh_matches(alice_bytes in arb_32_bytes(), bob_bytes in arb_32_bytes()) {
    let ours_alice = X25519SecretKey::from_bytes(alice_bytes);
    let dalek_bob = DalekStaticSecret::from(bob_bytes);
    let dalek_bob_pub = DalekPublicKey::from(&dalek_bob);
    let ours_bob_pub = X25519PublicKey::from_bytes(dalek_bob_pub.to_bytes());

    let ours_view = ours_alice.diffie_hellman(&ours_bob_pub);
    let dalek_alice_pub = DalekPublicKey::from(&DalekStaticSecret::from(alice_bytes));
    let dalek_view = dalek_bob.diffie_hellman(&dalek_alice_pub).to_bytes();

    if dalek_view == [0u8; 32] {
      prop_assert_eq!(ours_view, Err(X25519Error::new()));
    } else {
      let ours = ours_view.expect("rscrypto rejected what dalek accepted");
      prop_assert_eq!(*ours.as_bytes(), dalek_view);
    }
  }
}

/// Static differential: every basepoint scalar multiplication on hand-picked
/// edge values must match dalek byte-for-byte. Catches regressions in the
/// scalar clamping path that proptests might miss with low probability.
#[test]
fn basepoint_edge_scalars_match_dalek() {
  let edges: &[[u8; 32]] = &[
    [0u8; 32],
    [0xFFu8; 32],
    {
      let mut s = [0u8; 32];
      s[0] = 1;
      s
    },
    {
      let mut s = [0xFFu8; 32];
      s[0] = 0xF8; // clamp-target low byte
      s[31] = 0x7F; // clamp-target high byte
      s
    },
    // Order-related edge: the curve order's least significant bytes.
    [
      0xED, 0xD3, 0xF5, 0x5C, 0x1A, 0x63, 0x12, 0x58, 0xD6, 0x9C, 0xF7, 0xA2, 0xDE, 0xF9, 0xDE, 0x14, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0x10,
    ],
  ];

  for (i, scalar) in edges.iter().enumerate() {
    let ours = X25519SecretKey::from_bytes(*scalar).public_key();
    let dalek = DalekPublicKey::from(&DalekStaticSecret::from(*scalar));
    assert_eq!(ours.to_bytes(), dalek.to_bytes(), "basepoint scalar #{i} mismatch");
  }
}
