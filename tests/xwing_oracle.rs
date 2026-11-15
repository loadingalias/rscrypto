//! X-Wing against RustCrypto `x-wing`, an independent implementation of
//! draft-connolly-cfrg-xwing-kem on RustCrypto ML-KEM, x25519-dalek, and
//! RustCrypto SHA-3.
//!
//! Its `DecapsulationKey` accepts an all-zero X25519 output.
//! `DecapsulationKeyRejectNonContrib` rejects it, as rscrypto does, so it is
//! the decapsulation oracle here.
#![cfg(feature = "x-wing")]

use proptest::{prelude::*, test_runner::Config as ProptestConfig};
use rscrypto::{Kem, XWing, XWingCiphertext, XWingDecapsulationKey, XWingError};
use x_wing::{Decapsulator as _, KeyExport as _, TryDecapsulate as _};

mod common;
use common::decode_hex_vec;

const PROPTEST_CASES: u32 = if cfg!(debug_assertions) { 32 } else { 96 };

fn filled(bytes: &[u8]) -> impl FnMut(&mut [u8]) -> Result<(), XWingError> + '_ {
  |out| {
    out.copy_from_slice(bytes);
    Ok(())
  }
}

fn bytes<const N: usize>() -> impl Strategy<Value = [u8; N]> {
  proptest::collection::vec(any::<u8>(), N).prop_map(|bytes| bytes.try_into().expect("strategy length"))
}

/// Decapsulate with both implementations: equal secrets, or both reject.
fn assert_decapsulation_matches(seed: [u8; 32], ciphertext: [u8; XWing::CIPHERTEXT_SIZE]) {
  let ours = XWing::decapsulate(
    &XWingDecapsulationKey::from_bytes(seed),
    &XWingCiphertext::from_bytes(ciphertext),
  );
  let theirs = x_wing::DecapsulationKeyRejectNonContrib::from(seed).try_decapsulate(&ciphertext.into());
  match (ours, theirs) {
    (Ok(ours), Ok(theirs)) => assert_eq!(ours.as_bytes().as_slice(), theirs.as_slice()),
    (ours, theirs) => {
      assert_eq!(ours.err(), Some(XWingError::InvalidCiphertext));
      assert_eq!(
        theirs.ok().map(|_| ()),
        None,
        "x-wing accepted a ciphertext that rscrypto rejected"
      );
    }
  }
}

proptest! {
  #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

  /// Key generation and deterministic encapsulation give the oracle's bytes,
  /// and a ciphertext with one changed bit decapsulates as in the oracle:
  /// ML-KEM implicit rejection, or a different X25519 input.
  #[test]
  fn matches_x_wing(
    seed in bytes::<32>(),
    randomness in bytes::<64>(),
    bit in 0..XWing::CIPHERTEXT_SIZE * 8,
  ) {
    let (encapsulation_key, _) = XWing::generate_keypair(filled(&seed)).expect("key generation");
    let oracle = x_wing::DecapsulationKey::from(seed);
    let oracle_key = oracle.encapsulation_key().to_bytes();
    prop_assert_eq!(encapsulation_key.as_bytes().as_slice(), oracle_key.as_slice());

    let (ciphertext, shared_secret) =
      XWing::encapsulate(&encapsulation_key, filled(&randomness)).expect("encapsulation");
    let (oracle_ciphertext, oracle_secret) =
      oracle.encapsulation_key().encapsulate_deterministic(&randomness.into());
    prop_assert_eq!(ciphertext.as_bytes().as_slice(), oracle_ciphertext.as_slice());
    prop_assert_eq!(shared_secret.as_bytes().as_slice(), oracle_secret.as_slice());

    assert_decapsulation_matches(seed, ciphertext.to_bytes());
    let mut modified = ciphertext.to_bytes();
    modified[bit / 8] ^= 1 << (bit % 8);
    assert_decapsulation_matches(seed, modified);
  }

  /// Arbitrary ciphertexts decapsulate as in the oracle. Half of them set bit
  /// 255 of the X25519 part, which RFC 7748 ignores.
  #[test]
  fn arbitrary_ciphertexts_match_x_wing(seed in bytes::<32>(), ciphertext in bytes::<1120>()) {
    assert_decapsulation_matches(seed, ciphertext);
  }
}

/// The oracle also rejects each small-order X25519 part, which confirms the
/// pinned list that `tests/xwing.rs` checks.
#[test]
fn small_order_ciphertexts_match_x_wing() {
  let seed = [0x42; 32];
  let (encapsulation_key, _) = XWing::generate_keypair(filled(&seed)).expect("key generation");
  let (ciphertext, _) = XWing::encapsulate(&encapsulation_key, filled(&[0x24; 64])).expect("encapsulation");
  for line in include_str!("../testdata/xwing/low-order-points.txt").lines() {
    let mut modified = ciphertext.to_bytes();
    modified[XWing::CIPHERTEXT_SIZE - 32..].copy_from_slice(&decode_hex_vec(line));
    assert_decapsulation_matches(seed, modified);
  }
}
