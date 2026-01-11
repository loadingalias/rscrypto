use hashes::crypto::Blake3;
use serde::Deserialize;
use traits::{Digest as _, Xof as _};

#[derive(Deserialize)]
struct Vectors {
  key: String,
  context_string: String,
  cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
  input_len: usize,
  hash: String,
  keyed_hash: String,
  derive_key: String,
}

fn decode_hex(hex: &str) -> Vec<u8> {
  assert_eq!(hex.len() % 2, 0);
  let mut out = vec![0u8; hex.len() / 2];
  for (i, chunk) in hex.as_bytes().chunks_exact(2).enumerate() {
    let hi = (chunk[0] as char).to_digit(16).unwrap() as u8;
    let lo = (chunk[1] as char).to_digit(16).unwrap() as u8;
    out[i] = (hi << 4) | lo;
  }
  out
}

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

#[test]
fn blake3_official_test_vectors() {
  let json = include_str!("../testdata/blake3/test_vectors.json");
  let vectors: Vectors = serde_json::from_str(json).unwrap();

  let key_bytes = vectors.key.as_bytes();
  assert_eq!(key_bytes.len(), 32);
  let mut key = [0u8; 32];
  key.copy_from_slice(key_bytes);

  for (i, case) in vectors.cases.iter().enumerate() {
    let expected_hash_xof = decode_hex(&case.hash);
    let expected_keyed_xof = decode_hex(&case.keyed_hash);
    let expected_derive_xof = decode_hex(&case.derive_key);

    // Hash mode
    {
      let mut h = Blake3::new();
      update_input_pattern(&mut h, case.input_len);
      assert_eq!(&h.finalize()[..], &expected_hash_xof[..32], "hash digest case {i}");

      let mut xof = h.finalize_xof();
      let mut out = vec![0u8; expected_hash_xof.len()];
      xof.squeeze(&mut out);
      assert_eq!(out, expected_hash_xof, "hash xof case {i}");
    }

    // Keyed hash mode
    {
      let mut h = Blake3::new_keyed(&key);
      update_input_pattern(&mut h, case.input_len);
      assert_eq!(&h.finalize()[..], &expected_keyed_xof[..32], "keyed digest case {i}");

      let mut xof = h.finalize_xof();
      let mut out = vec![0u8; expected_keyed_xof.len()];
      xof.squeeze(&mut out);
      assert_eq!(out, expected_keyed_xof, "keyed xof case {i}");
    }

    // Derive key mode
    {
      let mut h = Blake3::new_derive_key(&vectors.context_string);
      update_input_pattern(&mut h, case.input_len);
      assert_eq!(&h.finalize()[..], &expected_derive_xof[..32], "derive digest case {i}");

      let mut xof = h.finalize_xof();
      let mut out = vec![0u8; expected_derive_xof.len()];
      xof.squeeze(&mut out);
      assert_eq!(out, expected_derive_xof, "derive xof case {i}");
    }
  }
}
