#![cfg(feature = "p256-ecdh")]

use p256::elliptic_curve::sec1::ToSec1Point as _;
use proptest::{prelude::*, test_runner::Config as ProptestConfig};
use rscrypto::{P256EphemeralSecret, P256PublicKey};

const PROPERTY_CASES: u32 = 96;

fn canonical_scalar(mut bytes: [u8; 32]) -> [u8; 32] {
  bytes[0] &= 0x7f;
  bytes[31] |= 1;
  bytes
}

fn secret(bytes: [u8; 32]) -> P256EphemeralSecret {
  P256EphemeralSecret::try_generate_with(|candidate| {
    candidate.copy_from_slice(&bytes);
    Ok::<(), core::convert::Infallible>(())
  })
  .expect("normalized property scalar must be canonical and nonzero")
}

proptest! {
  #![proptest_config(ProptestConfig::with_cases(PROPERTY_CASES))]

  #[test]
  fn public_derivation_and_agreement_match_independent_implementations(
    left in proptest::array::uniform32(any::<u8>()),
    right in proptest::array::uniform32(any::<u8>()),
  ) {
    let left = canonical_scalar(left);
    let right = canonical_scalar(right);
    let rustcrypto_left = p256::SecretKey::from_slice(&left).expect("normalized RustCrypto scalar");
    let rustcrypto_right = p256::SecretKey::from_slice(&right).expect("normalized RustCrypto scalar");

    let ours_public = secret(left).public_key();
    let rustcrypto_public = rustcrypto_left.public_key().to_sec1_point(false);
    prop_assert_eq!(
      ours_public.as_sec1_bytes(),
      rustcrypto_public.as_bytes(),
    );

    let peer = secret(right).public_key();
    let ours_shared = secret(left).diffie_hellman(&peer);
    let rustcrypto_shared = p256::ecdh::diffie_hellman(
      rustcrypto_left.to_nonzero_scalar(),
      rustcrypto_right.public_key().as_affine(),
    );
    prop_assert_eq!(
      ours_shared.as_bytes().as_slice(),
      rustcrypto_shared.raw_secret_bytes().as_slice(),
    );

    let mut crrl_left = left;
    crrl_left.reverse();
    let crrl_scalar = crrl::p256::Scalar::decode(&crrl_left).expect("normalized CRRL scalar");
    let crrl_peer = crrl::p256::Point::decode(peer.as_sec1_bytes()).expect("generated CRRL peer");
    prop_assert_eq!(
      &core::ops::Mul::mul(crrl_peer, crrl_scalar).encode_uncompressed()[1..33],
      ours_shared.as_bytes(),
    );
  }

  #[test]
  fn canonical_sec1_parser_matches_rustcrypto(bytes in proptest::collection::vec(any::<u8>(), 0..96)) {
    let rustcrypto = p256::PublicKey::from_sec1_bytes(&bytes);
    let expected = bytes.len() == P256PublicKey::SEC1_LENGTH
      && bytes.first() == Some(&0x04)
      && rustcrypto.is_ok();
    prop_assert_eq!(P256PublicKey::from_sec1_bytes(&bytes).is_ok(), expected);
  }
}
