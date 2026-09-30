#![cfg(feature = "blake3")]

mod support;

use rscrypto::{Digest, hashes::crypto::Blake3, traits::Xof as _};
use support::vector_blob::BlobIterator;

fn update_input_pattern(hasher: &mut Blake3, len: usize) {
  let mut remaining = len;
  let mut offset = 0usize;
  let mut buf = [0u8; 1024];
  while remaining != 0 {
    let take = core::cmp::min(remaining, buf.len());
    for (i, b) in buf[..take].iter_mut().enumerate() {
      let value = offset.strict_add(i).strict_rem(251);
      *b = u8::try_from(value).expect("BLAKE3 fixture byte must fit in u8");
    }
    hasher.update(&buf[..take]);
    offset = offset.strict_add(take);
    remaining = remaining.strict_sub(take);
  }
}

fn input_pattern(len: usize) -> Vec<u8> {
  (0..len)
    .map(|i| u8::try_from(i.strict_rem(251)).expect("BLAKE3 fixture byte must fit in u8"))
    .collect()
}

// Official vector for the empty input, evaluated at compile time.
const EMPTY_DIGEST: [u8; 32] = Blake3::digest_const(&[]);
const EMPTY_EXPECTED: [u8; 32] = [
  0xaf, 0x13, 0x49, 0xb9, 0xf5, 0xf9, 0xa1, 0xa6, 0xa0, 0x40, 0x4d, 0xea, 0x36, 0xdc, 0xc9, 0x49, 0x9b, 0xcb, 0x25,
  0xc9, 0xad, 0xc1, 0x12, 0xb7, 0xcc, 0x9a, 0x93, 0xca, 0xe4, 0x1f, 0x32, 0x62,
];
const _: () = {
  let mut i = 0;
  while i < EMPTY_DIGEST.len() {
    assert!(EMPTY_DIGEST[i] == EMPTY_EXPECTED[i]);
    i += 1;
  }
};

fn decode_u64_le(bytes: &[u8]) -> u64 {
  let arr: [u8; 8] = bytes.try_into().expect("expected 8-byte little-endian u64");
  u64::from_le_bytes(arr)
}

#[test]
fn blake3_official_test_vectors() {
  let blb = include_bytes!("../testdata/blake3/test_vectors.blb");
  for (i, row) in BlobIterator::<6>::new(blb)
    .expect("blake3 vector corpus must parse")
    .enumerate()
  {
    let [
      key_bytes,
      context_bytes,
      input_len_bytes,
      hash_xof,
      keyed_hash_xof,
      derive_key_xof,
    ] = row.expect("BLAKE3 vector row must decode");

    assert_eq!(key_bytes.len(), 32, "blake3 key length mismatch at case {i}");
    let mut key = [0u8; 32];
    key.copy_from_slice(key_bytes);

    let context = core::str::from_utf8(context_bytes).expect("blake3 context_string is valid UTF-8");
    let input_len =
      usize::try_from(decode_u64_le(input_len_bytes)).expect("BLAKE3 vector input length must fit in usize");

    // Hash mode
    {
      let mut h = Blake3::new();
      update_input_pattern(&mut h, input_len);
      assert_eq!(&h.finalize()[..], &hash_xof[..32], "hash digest case {i}");
      if input_len <= 1024 {
        assert_eq!(
          &Blake3::digest_const(&input_pattern(input_len))[..],
          &hash_xof[..32],
          "const digest case {i}"
        );
      }

      let mut xof = h.finalize_xof();
      let mut out = vec![0u8; hash_xof.len()];
      xof.squeeze(&mut out);
      assert_eq!(&out[..], hash_xof, "hash xof case {i}");
    }

    // Keyed hash mode
    {
      let mut h = Blake3::new_keyed(&key);
      update_input_pattern(&mut h, input_len);
      assert_eq!(&h.finalize()[..], &keyed_hash_xof[..32], "keyed digest case {i}");

      let mut xof = h.finalize_xof();
      let mut out = vec![0u8; keyed_hash_xof.len()];
      xof.squeeze(&mut out);
      assert_eq!(&out[..], keyed_hash_xof, "keyed xof case {i}");
    }

    // Derive key mode
    {
      let mut h = Blake3::new_derive_key(context);
      update_input_pattern(&mut h, input_len);
      assert_eq!(&h.finalize()[..], &derive_key_xof[..32], "derive digest case {i}");

      let mut xof = h.finalize_xof();
      let mut out = vec![0u8; derive_key_xof.len()];
      xof.squeeze(&mut out);
      assert_eq!(&out[..], derive_key_xof, "derive xof case {i}");
    }
  }
}
