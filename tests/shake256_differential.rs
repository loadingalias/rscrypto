#![cfg(feature = "sha3")]

use proptest::prelude::*;
use rscrypto::{hashes::crypto::Shake256, traits::Xof as _};

fn shake256_ref(data: &[u8], out: &mut [u8]) {
  use tiny_keccak::{Hasher as _, Xof as _};
  let mut h = tiny_keccak::Shake::v256();
  h.update(data);
  h.squeeze(out);
}

proptest! {
  #[test]
  fn shake256_one_shot_matches_tiny_keccak(
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..2048,
  ) {
    let mut expected = vec![0u8; out_len];
    shake256_ref(&data, &mut expected);

    let mut actual = vec![0u8; out_len];
    Shake256::xof(&data).squeeze(&mut actual);

    prop_assert_eq!(actual, expected);
  }

  #[test]
  fn shake256_streaming_matches_tiny_keccak(
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..2048,
  ) {
    let mut expected = vec![0u8; out_len];
    shake256_ref(&data, &mut expected);

    let mut h = Shake256::new();
    let mut i = 0usize;
    while i < data.len() {
      let step = usize::from(data[i]).strict_rem(97).strict_add(1);
      let end = core::cmp::min(data.len(), i.strict_add(step));
      h.update(&data[i..end]);
      i = end;
    }

    let mut xof = h.finalize_xof();
    let mut actual = vec![0u8; out_len];
    xof.squeeze(&mut actual);

    prop_assert_eq!(actual, expected);
  }

  #[test]
  fn shake256_multi_squeeze_matches_tiny_keccak(
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..2048,
    split in any::<usize>(),
  ) {
    let split = split.strict_rem(out_len.strict_add(1));

    let mut expected = vec![0u8; out_len];
    {
      use tiny_keccak::{Hasher as _, Xof as _};
      let mut h = tiny_keccak::Shake::v256();
      h.update(&data);
      h.squeeze(&mut expected[..split]);
      h.squeeze(&mut expected[split..]);
    }

    let mut actual = vec![0u8; out_len];
    {
      let mut h = Shake256::new();
      h.update(&data);
      let mut xof = h.finalize_xof();
      xof.squeeze(&mut actual[..split]);
      xof.squeeze(&mut actual[split..]);
    }

    prop_assert_eq!(actual, expected);
  }
}

#[test]
fn shake256_squeeze_lane_and_rate_boundaries() {
  // Every lane offset and rate boundary exercises partial/full/partial copies.
  const RATE: usize = 136;
  let data = [0xa5; 193];
  let mut expected = vec![0u8; RATE.strict_mul(2).strict_add(1)];
  shake256_ref(&data, &mut expected);
  for prefix in 0..=RATE {
    for length in [0, 1, 7, 8, 9, 15, 16, RATE.strict_sub(1), RATE, RATE.strict_add(1)] {
      let mut reader = Shake256::xof(&data);
      let mut actual = vec![0u8; prefix.strict_add(length)];
      reader.squeeze(&mut actual[..prefix]);
      reader.squeeze(&mut actual[prefix..]);
      assert_eq!(actual, expected[..actual.len()], "prefix={prefix}, length={length}");
    }
  }
}
