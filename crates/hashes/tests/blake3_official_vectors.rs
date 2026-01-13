use digest::dev::blobby::Blob6Iterator;
use hashes::{Digest, crypto::Blake3};
use traits::Xof as _;

fn update_input_pattern(hasher: &mut Blake3, len: usize) {
  let mut remaining = len;
  let mut offset = 0usize;
  let mut buf = [0u8; 1024];
  while remaining != 0 {
    let take = core::cmp::min(remaining, buf.len());
    for (i, b) in buf[..take].iter_mut().enumerate() {
      *b = ((offset + i) % 251) as u8;
    }
    hasher.update(&buf[..take]);
    offset += take;
    remaining -= take;
  }
}

fn decode_u64_le(bytes: &[u8]) -> u64 {
  let arr: [u8; 8] = bytes.try_into().expect("expected 8-byte little-endian u64");
  u64::from_le_bytes(arr)
}

#[test]
fn blake3_official_test_vectors() {
  let blb = include_bytes!("../testdata/blake3/test_vectors.blb");

  for (i, row) in Blob6Iterator::new(blb).unwrap().enumerate() {
    let [
      key_bytes,
      context_bytes,
      input_len_bytes,
      hash_xof,
      keyed_hash_xof,
      derive_key_xof,
    ] = row.unwrap();

    assert_eq!(key_bytes.len(), 32, "blake3 key length mismatch at case {i}");
    let mut key = [0u8; 32];
    key.copy_from_slice(key_bytes);

    let context = core::str::from_utf8(context_bytes).expect("blake3 context_string is valid UTF-8");
    let input_len = decode_u64_le(input_len_bytes) as usize;

    // Hash mode
    {
      let mut h = Blake3::new();
      update_input_pattern(&mut h, input_len);
      assert_eq!(&h.finalize()[..], &hash_xof[..32], "hash digest case {i}");

      let mut xof = h.finalize_xof();
      let mut out = vec![0u8; hash_xof.len()];
      xof.squeeze(&mut out);
      assert_eq!(out, hash_xof, "hash xof case {i}");
    }

    // Keyed hash mode
    {
      let mut h = Blake3::new_keyed(&key);
      update_input_pattern(&mut h, input_len);
      assert_eq!(&h.finalize()[..], &keyed_hash_xof[..32], "keyed digest case {i}");

      let mut xof = h.finalize_xof();
      let mut out = vec![0u8; keyed_hash_xof.len()];
      xof.squeeze(&mut out);
      assert_eq!(out, keyed_hash_xof, "keyed xof case {i}");
    }

    // Derive key mode
    {
      let mut h = Blake3::new_derive_key(context);
      update_input_pattern(&mut h, input_len);
      assert_eq!(&h.finalize()[..], &derive_key_xof[..32], "derive digest case {i}");

      let mut xof = h.finalize_xof();
      let mut out = vec![0u8; derive_key_xof.len()];
      xof.squeeze(&mut out);
      assert_eq!(out, derive_key_xof, "derive xof case {i}");
    }
  }
}
