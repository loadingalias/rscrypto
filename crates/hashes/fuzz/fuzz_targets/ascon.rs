#![no_main]

use hashes::crypto::{AsconHash256, AsconXof128};
use libfuzzer_sys::fuzz_target;
use traits::{Digest as _, Xof as _};

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

  // Hash256
  {
    let ours = AsconHash256::digest(data);

    // streaming split
    let split_data = if data.is_empty() { 0 } else { (data[0] as usize) % (data.len() + 1) };
    let (a, b) = data.split_at(split_data);
    let mut h = AsconHash256::new();
    h.update(a);
    h.update(b);
    assert_eq!(ours, h.finalize());

    use ascon_hash256::Digest as _;
    let ref_out = ascon_hash256::AsconHash256::digest(data);
    let mut expected = [0u8; 32];
    expected.copy_from_slice(&ref_out);
    assert_eq!(ours, expected);
  }

  // XOF128
  {
    let mut ours = vec![0u8; out_len];
    AsconXof128::hash_into(data, &mut ours);

    // streaming + multi-squeeze
    let split_data = if data.is_empty() { 0 } else { (data[0] as usize) % (data.len() + 1) };
    let (a, b) = data.split_at(split_data);
    let mut h = AsconXof128::new();
    h.update(a);
    h.update(b);
    let mut xof = h.finalize_xof();
    let mut streamed = vec![0u8; out_len];
    xof.squeeze(&mut streamed[..split_out]);
    xof.squeeze(&mut streamed[split_out..]);
    assert_eq!(ours, streamed);

    use ascon_hash256::digest::{ExtendableOutput, Update, XofReader};
    let mut hh = ascon_hash256::AsconXof128::default();
    hh.update(data);
    let mut reader = hh.finalize_xof();
    let mut expected = vec![0u8; out_len];
    reader.read(&mut expected);
    assert_eq!(ours, expected);
  }
});

