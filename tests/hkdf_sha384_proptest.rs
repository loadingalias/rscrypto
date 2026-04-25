#![cfg(feature = "hkdf")]

use proptest::prelude::*;
use rscrypto::HkdfSha384;

proptest! {
  #[test]
  fn hkdf_sha384_matches_oracle(
    salt in proptest::collection::vec(any::<u8>(), 0..128),
    ikm in proptest::collection::vec(any::<u8>(), 1..256),
    info in proptest::collection::vec(any::<u8>(), 0..128),
    out_len in 1usize..=255,
  ) {
    use hkdf::Hkdf;
    use sha2::Sha384;

    let mut ours = vec![0u8; out_len];
    let our_result = HkdfSha384::new(&salt, &ikm).expand(&info, &mut ours);

    let oracle = Hkdf::<Sha384>::new(Some(&salt), &ikm);
    let mut expected = vec![0u8; out_len];
    let oracle_result = oracle.expand(&info, &mut expected);

    match (our_result, oracle_result) {
      (Ok(()), Ok(())) => prop_assert_eq!(ours, expected),
      (Err(_), Err(_)) => {} // both reject — consistent
      _ => prop_assert!(false, "expand agreement: ours={our_result:?} oracle={oracle_result:?}"),
    }
  }
}
