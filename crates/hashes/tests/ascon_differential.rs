use hashes::crypto::{AsconHash256, AsconXof128};
use proptest::prelude::*;
use traits::{Digest as _, Xof as _};

fn ascon_hash256_ref(data: &[u8]) -> [u8; 32] {
  use ascon_hash256::Digest as _;
  let out = ascon_hash256::AsconHash256::digest(data);
  let mut bytes = [0u8; 32];
  bytes.copy_from_slice(&out);
  bytes
}

fn ascon_xof128_ref(data: &[u8], out: &mut [u8]) {
  use ascon_hash256::digest::{ExtendableOutput, Update, XofReader};
  let mut h = ascon_hash256::AsconXof128::default();
  h.update(data);
  let mut reader = h.finalize_xof();
  reader.read(out);
}

proptest! {
  #[test]
  fn ascon_hash256_one_shot_matches_oracle(data in proptest::collection::vec(any::<u8>(), 0..8192)) {
    prop_assert_eq!(AsconHash256::digest(&data), ascon_hash256_ref(&data));
  }

  #[test]
  fn ascon_hash256_streaming_matches_oracle(data in proptest::collection::vec(any::<u8>(), 0..8192)) {
    let expected = ascon_hash256_ref(&data);

    let mut h = AsconHash256::new();
    let mut i = 0usize;
    while i < data.len() {
      let step = (data[i] as usize % 97) + 1;
      let end = core::cmp::min(data.len(), i + step);
      h.update(&data[i..end]);
      i = end;
    }

    prop_assert_eq!(h.finalize(), expected);
  }

  #[test]
  fn ascon_xof128_one_shot_matches_oracle(
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..2048,
  ) {
    let mut expected = vec![0u8; out_len];
    ascon_xof128_ref(&data, &mut expected);

    let mut actual = vec![0u8; out_len];
    AsconXof128::hash_into(&data, &mut actual);

    prop_assert_eq!(actual, expected);
  }

  #[test]
  fn ascon_xof128_streaming_and_multi_squeeze_matches_oracle(
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..2048,
    split_data in any::<usize>(),
    split_out in any::<usize>(),
  ) {
    let split_data = split_data % (data.len() + 1);
    let split_out = split_out % (out_len + 1);

    let mut expected = vec![0u8; out_len];
    {
      use ascon_hash256::digest::{ExtendableOutput, Update, XofReader};
      let mut h = ascon_hash256::AsconXof128::default();
      h.update(&data[..split_data]);
      h.update(&data[split_data..]);
      let mut reader = h.finalize_xof();
      reader.read(&mut expected[..split_out]);
      reader.read(&mut expected[split_out..]);
    }

    let mut actual = vec![0u8; out_len];
    {
      let mut h = AsconXof128::new();
      h.update(&data[..split_data]);
      h.update(&data[split_data..]);
      let mut xof = h.finalize_xof();
      xof.squeeze(&mut actual[..split_out]);
      xof.squeeze(&mut actual[split_out..]);
    }

    prop_assert_eq!(actual, expected);
  }
}
