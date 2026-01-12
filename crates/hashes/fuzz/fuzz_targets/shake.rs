#![no_main]

use hashes::crypto::{Shake128, Shake256};
use libfuzzer_sys::fuzz_target;
use traits::Xof as _;

fn parse_u16_le(input: &[u8]) -> u16 {
  match input.len() {
    0 => 0,
    1 => input[0] as u16,
    _ => u16::from_le_bytes([input[0], input[1]]),
  }
}

fuzz_target!(|input: &[u8]| {
  // Layout:
  // - 2 bytes: out_len (capped)
  // - 1 byte: split_out (mod out_len+1)
  // - rest: data
  let out_len = (parse_u16_le(input) as usize) % 2049;
  let split_out = if out_len == 0 {
    0usize
  } else {
    input.get(2).copied().unwrap_or(0) as usize % (out_len + 1)
  };
  let data = input.get(3..).unwrap_or(&[]);

  // SHAKE128
  {
    let mut ours = vec![0u8; out_len];
    Shake128::hash_into(data, &mut ours);

    // Streaming + multi-squeeze
    let split_data = if data.is_empty() { 0 } else { (data[0] as usize) % (data.len() + 1) };
    let (a, b) = data.split_at(split_data);
    let mut h = Shake128::new();
    h.update(a);
    h.update(b);
    let mut xof = h.finalize_xof();
    let mut streamed = vec![0u8; out_len];
    xof.squeeze(&mut streamed[..split_out]);
    xof.squeeze(&mut streamed[split_out..]);
    assert_eq!(ours, streamed);

    use sha3::digest::{ExtendableOutput, Update, XofReader};
    let mut hh = sha3::Shake128::default();
    hh.update(data);
    let mut reader = hh.finalize_xof();
    let mut expected = vec![0u8; out_len];
    reader.read(&mut expected);
    assert_eq!(ours, expected);
  }

  // SHAKE256
  {
    let mut ours = vec![0u8; out_len];
    Shake256::hash_into(data, &mut ours);

    // Streaming + multi-squeeze
    let split_data = if data.is_empty() { 0 } else { (data[0] as usize) % (data.len() + 1) };
    let (a, b) = data.split_at(split_data);
    let mut h = Shake256::new();
    h.update(a);
    h.update(b);
    let mut xof = h.finalize_xof();
    let mut streamed = vec![0u8; out_len];
    xof.squeeze(&mut streamed[..split_out]);
    xof.squeeze(&mut streamed[split_out..]);
    assert_eq!(ours, streamed);

    use sha3::digest::{ExtendableOutput, Update, XofReader};
    let mut hh = sha3::Shake256::default();
    hh.update(data);
    let mut reader = hh.finalize_xof();
    let mut expected = vec![0u8; out_len];
    reader.read(&mut expected);
    assert_eq!(ours, expected);
  }
});

