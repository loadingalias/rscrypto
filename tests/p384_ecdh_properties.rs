#![cfg(feature = "p384-ecdh")]

use p384::elliptic_curve::sec1::ToSec1Point as _;
use proptest::{prelude::*, test_runner::Config as ProptestConfig};
use rscrypto::{P384EphemeralSecret, P384PublicKey};

const PROPERTY_CASES: u32 = 96;

fn canonical_scalar(mut bytes: [u8; 48]) -> [u8; 48] {
  bytes[0] &= 0x7f;
  bytes[47] |= 1;
  bytes
}

fn secret(bytes: [u8; 48]) -> P384EphemeralSecret {
  P384EphemeralSecret::try_generate_with(|candidate| {
    candidate.copy_from_slice(&bytes);
    Ok::<(), core::convert::Infallible>(())
  })
  .expect("normalized property scalar must be canonical and nonzero")
}

fn scalar_strategy() -> impl Strategy<Value = [u8; 48]> {
  proptest::collection::vec(any::<u8>(), 48).prop_map(|bytes| {
    let mut scalar = [0u8; 48];
    scalar.copy_from_slice(&bytes);
    canonical_scalar(scalar)
  })
}

proptest! {
  #![proptest_config(ProptestConfig::with_cases(PROPERTY_CASES))]

  #[test]
  fn public_derivation_and_agreement_match_rustcrypto(left in scalar_strategy(), right in scalar_strategy()) {
    let rustcrypto_left = p384::SecretKey::from_slice(&left).expect("normalized RustCrypto scalar");
    let rustcrypto_right = p384::SecretKey::from_slice(&right).expect("normalized RustCrypto scalar");

    let ours_public = secret(left).public_key();
    let rustcrypto_public = rustcrypto_left.public_key().to_sec1_point(false);
    prop_assert_eq!(
      ours_public.as_sec1_bytes().as_slice(),
      rustcrypto_public.as_bytes(),
    );

    let peer = secret(right).public_key();
    let ours_shared = secret(left).diffie_hellman(&peer);
    let rustcrypto_shared = p384::ecdh::diffie_hellman(
      rustcrypto_left.to_nonzero_scalar(),
      rustcrypto_right.public_key().as_affine(),
    );
    prop_assert_eq!(
      ours_shared.as_bytes().as_slice(),
      rustcrypto_shared.raw_secret_bytes().as_slice(),
    );
  }

  #[test]
  fn canonical_sec1_parser_matches_rustcrypto(bytes in proptest::collection::vec(any::<u8>(), 0..144)) {
    let rustcrypto = p384::PublicKey::from_sec1_bytes(&bytes);
    let expected = bytes.len() == P384PublicKey::SEC1_LENGTH
      && bytes.first() == Some(&0x04)
      && rustcrypto.is_ok();
    prop_assert_eq!(P384PublicKey::from_sec1_bytes(&bytes).is_ok(), expected);
  }
}
