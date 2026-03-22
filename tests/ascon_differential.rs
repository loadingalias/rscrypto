#![cfg(feature = "hashes")]

use proptest::prelude::*;
use rscrypto::{
  hashes::crypto::{AsconHash256, AsconXof},
  traits::{Digest as _, Xof as _},
};

fn ascon_hash256_streaming(data: &[u8]) -> [u8; 32] {
  let mut hasher = AsconHash256::new();
  let mut i = 0usize;
  while i < data.len() {
    let step = (data[i] as usize % 97) + 1;
    let end = core::cmp::min(data.len(), i + step);
    hasher.update(&data[i..end]);
    i = end;
  }
  hasher.finalize()
}

fn ascon_xof_streaming(data: &[u8], out: &mut [u8]) {
  let split_data = data.len() / 2;
  let split_out = out.len() / 2;

  let mut hasher = AsconXof::new();
  hasher.update(&data[..split_data]);
  hasher.update(&data[split_data..]);
  let mut xof = hasher.finalize_xof();
  xof.squeeze(&mut out[..split_out]);
  xof.squeeze(&mut out[split_out..]);
}

proptest! {
  #[test]
  fn ascon_hash256_one_shot_matches_streaming(data in proptest::collection::vec(any::<u8>(), 0..8192)) {
    prop_assert_eq!(AsconHash256::digest(&data), ascon_hash256_streaming(&data));
  }

  #[test]
  fn ascon_xof_one_shot_matches_streaming(
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..2048,
  ) {
    let mut expected = vec![0u8; out_len];
    ascon_xof_streaming(&data, &mut expected);

    let mut actual = vec![0u8; out_len];
    AsconXof::xof(&data).squeeze(&mut actual);

    prop_assert_eq!(actual, expected);
  }

  #[test]
  fn ascon_xof_streaming_and_multi_squeeze_matches_oracle(
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..2048,
    split_data in any::<usize>(),
    split_out in any::<usize>(),
  ) {
    let split_data = split_data % (data.len() + 1);
    let split_out = split_out % (out_len + 1);

    let mut expected = vec![0u8; out_len];
    {
      let mut h = AsconXof::new();
      h.update(&data[..split_data]);
      h.update(&data[split_data..]);
      let mut xof = h.finalize_xof();
      xof.squeeze(&mut expected[..split_out]);
      xof.squeeze(&mut expected[split_out..]);
    }

    let mut actual = vec![0u8; out_len];
    {
      let mut h = AsconXof::new();
      h.update(&data[..split_data]);
      h.update(&data[split_data..]);
      let mut xof = h.finalize_xof();
      xof.squeeze(&mut actual[..split_out]);
      xof.squeeze(&mut actual[split_out..]);
    }

    prop_assert_eq!(actual, expected);
  }
}
