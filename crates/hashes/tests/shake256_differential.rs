use hashes::crypto::Shake256;
use proptest::prelude::*;
use traits::Xof as _;

fn shake256_ref(data: &[u8], out: &mut [u8]) {
  use sha3::digest::{ExtendableOutput, Update, XofReader};
  let mut h = sha3::Shake256::default();
  h.update(data);
  let mut reader = h.finalize_xof();
  reader.read(out);
}

proptest! {
  #[test]
  fn shake256_one_shot_matches_sha3_crate(
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..2048,
  ) {
    let mut expected = vec![0u8; out_len];
    shake256_ref(&data, &mut expected);

    let mut actual = vec![0u8; out_len];
    Shake256::hash_into(&data, &mut actual);

    prop_assert_eq!(actual, expected);
  }

  #[test]
  fn shake256_streaming_matches_sha3_crate(
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..2048,
  ) {
    let mut expected = vec![0u8; out_len];
    shake256_ref(&data, &mut expected);

    let mut h = Shake256::new();
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
  fn shake256_multi_squeeze_matches_sha3_crate(
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..2048,
    split in any::<usize>(),
  ) {
    let split = split % (out_len + 1);

    let mut expected = vec![0u8; out_len];
    {
      use sha3::digest::{ExtendableOutput, Update, XofReader};
      let mut h = sha3::Shake256::default();
      h.update(&data);
      let mut reader = h.finalize_xof();
      reader.read(&mut expected[..split]);
      reader.read(&mut expected[split..]);
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
