#![cfg(feature = "hashes")]

use proptest::prelude::*;
use rscrypto::{hashes::crypto::Shake128, traits::Xof as _};

fn shake128_ref(data: &[u8], out: &mut [u8]) {
  use tiny_keccak::{Hasher as _, Xof as _};
  let mut h = tiny_keccak::Shake::v128();
  h.update(data);
  h.squeeze(out);
}

proptest! {
  #[test]
  fn shake128_one_shot_matches_tiny_keccak(
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..2048,
  ) {
    let mut expected = vec![0u8; out_len];
    shake128_ref(&data, &mut expected);

    let mut actual = vec![0u8; out_len];
    Shake128::xof(&data).squeeze(&mut actual);

    prop_assert_eq!(actual, expected);
  }

  #[test]
  fn shake128_streaming_matches_tiny_keccak(
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..2048,
  ) {
    let mut expected = vec![0u8; out_len];
    shake128_ref(&data, &mut expected);

    let mut h = Shake128::new();
    let mut i = 0usize;
    while i < data.len() {
      let step = (data[i] as usize % 97) + 1;
      let end = core::cmp::min(data.len(), i + step);
      h.update(&data[i..end]);
      i = end;
    }

    let mut xof = h.finalize_xof();
    let mut actual = vec![0u8; out_len];
    xof.squeeze(&mut actual);

    prop_assert_eq!(actual, expected);
  }

  #[test]
  fn shake128_multi_squeeze_matches_tiny_keccak(
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..2048,
    split in any::<usize>(),
  ) {
    let split = split % (out_len + 1);

    let mut expected = vec![0u8; out_len];
    {
      use tiny_keccak::{Hasher as _, Xof as _};
      let mut h = tiny_keccak::Shake::v128();
      h.update(&data);
      h.squeeze(&mut expected[..split]);
      h.squeeze(&mut expected[split..]);
    }

    let mut actual = vec![0u8; out_len];
    {
      let mut h = Shake128::new();
      h.update(&data);
      let mut xof = h.finalize_xof();
      xof.squeeze(&mut actual[..split]);
      xof.squeeze(&mut actual[split..]);
    }

    prop_assert_eq!(actual, expected);
  }
}
