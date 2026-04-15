#![cfg(feature = "hmac")]

use proptest::prelude::*;
use rscrypto::{HmacSha256, Mac as _};

proptest! {
  #[test]
  fn hmac_sha256_matches_oracle(
    key in proptest::collection::vec(any::<u8>(), 0..256),
    data in proptest::collection::vec(any::<u8>(), 0..4096),
  ) {
    use hmac::{Hmac, KeyInit as _, Mac as _};
    use sha2::Sha256;

    let ours = HmacSha256::mac(&key, &data);

    let mut oracle = Hmac::<Sha256>::new_from_slice(&key).unwrap();
    oracle.update(&data);
    let expected = oracle.finalize().into_bytes();

    prop_assert_eq!(&ours[..], expected.as_slice());
  }

  #[test]
  fn hmac_sha256_streaming_matches_oneshot(
    key in proptest::collection::vec(any::<u8>(), 0..256),
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    chunk_size in 1usize..=256,
  ) {
    let oneshot = HmacSha256::mac(&key, &data);

    let mut mac = HmacSha256::new(&key);
    for chunk in data.chunks(chunk_size) {
      mac.update(chunk);
    }
    let streaming = mac.finalize();

    prop_assert_eq!(streaming, oneshot);
  }
}
