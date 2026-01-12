#![no_main]

use hashes::crypto::Blake3;
use libfuzzer_sys::fuzz_target;
use traits::{Digest as _, Xof as _};

fn parse_u16_le(input: &[u8]) -> u16 {
  match input.len() {
    0 => 0,
    1 => input[0] as u16,
    _ => u16::from_le_bytes([input[0], input[1]]),
  }
}

fuzz_target!(|data: &[u8]| {
  let ours = Blake3::digest(data);
  let expected = *blake3::hash(data).as_bytes();
  assert_eq!(ours, expected);

  // Keyed hash mode
  {
    let key_bytes = data.get(..32).unwrap_or(data);
    let mut key = [0u8; 32];
    key[..key_bytes.len()].copy_from_slice(key_bytes);

    let ours = {
      let mut h = Blake3::new_keyed(&key);
      h.update(data);
      h.finalize()
    };
    let expected = *blake3::keyed_hash(&key, data).as_bytes();
    assert_eq!(ours, expected);
  }

  // Derive-key mode (use ASCII context to guarantee valid UTF-8 `&str`).
  {
    let ctx_len = (data.get(0).copied().unwrap_or(0) as usize) % 65;
    let ctx_src = data.get(32..).unwrap_or(&[]);
    let ctx_len = core::cmp::min(ctx_len, ctx_src.len());
    let mut context = alloc::string::String::with_capacity(ctx_len);
    for &b in &ctx_src[..ctx_len] {
      context.push((b'a' + (b % 26)) as char);
    }

    let ours = {
      let mut h = Blake3::new_derive_key(&context);
      h.update(data);
      h.finalize()
    };
    let expected = {
      let mut h = blake3::Hasher::new_derive_key(&context);
      h.update(data);
      *h.finalize().as_bytes()
    };
    assert_eq!(ours, expected);
  }

  // Also validate XOF against the official crate, with multi-squeeze.
  let out_len = (parse_u16_le(data) as usize) % 2049;
  let split = if out_len == 0 {
    0usize
  } else {
    data.get(2).copied().unwrap_or(0) as usize % (out_len + 1)
  };

  let mut ours_xof = vec![0u8; out_len];
  {
    let mut xof = Blake3::new();
    xof.update(data);
    let mut reader = xof.finalize_xof();
    reader.squeeze(&mut ours_xof[..split]);
    reader.squeeze(&mut ours_xof[split..]);
  }

  let mut expected_xof = vec![0u8; out_len];
  {
    let mut hasher = blake3::Hasher::new();
    hasher.update(data);
    let mut out = hasher.finalize_xof();
    out.fill(&mut expected_xof[..split]);
    out.fill(&mut expected_xof[split..]);
  }
  assert_eq!(ours_xof, expected_xof);
});

extern crate alloc;
