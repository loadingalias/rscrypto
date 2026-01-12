#![no_main]

use hashes::crypto::{Kmac128, Kmac256};
use libfuzzer_sys::fuzz_target;
use traits::Xof as _;

fn parse_u16_le(input: &[u8]) -> u16 {
  match input.len() {
    0 => 0,
    1 => input[0] as u16,
    _ => u16::from_le_bytes([input[0], input[1]]),
  }
}

fn split_len(max: usize, b: u8) -> usize {
  if max == 0 { 0 } else { (b as usize) % (max + 1) }
}

fuzz_target!(|input: &[u8]| {
  // Layout:
  // - 1 byte key_len (<= 64)
  // - 1 byte custom_len (<= 64)
  // - 2 bytes out_len (<= 512)
  // - rest: key || custom || data
  let key_len = input.get(0).copied().unwrap_or(0) as usize % 65;
  let custom_len = input.get(1).copied().unwrap_or(0) as usize % 65;
  let out_len = (parse_u16_le(input.get(2..).unwrap_or(&[])) as usize) % 513;
  let rest = input.get(4..).unwrap_or(&[]);

  let key_len = core::cmp::min(key_len, rest.len());
  let key = &rest[..key_len];
  let rest = &rest[key_len..];

  let custom_len = core::cmp::min(custom_len, rest.len());
  let customization = &rest[..custom_len];
  let data = &rest[custom_len..];

  let split_data = if data.is_empty() { 0 } else { (data[0] as usize) % (data.len() + 1) };
  let (a, b) = data.split_at(split_data);
  let split_out = split_len(out_len, data.get(1).copied().unwrap_or(0));

  // KMAC128 fixed-length output (L = out_len)
  {
    let mut ours = vec![0u8; out_len];
    let mut h = Kmac128::new(key, customization);
    h.update(a);
    h.update(b);
    h.finalize_into(&mut ours);

    use tiny_keccak::{Hasher, Kmac};
    let mut expected = vec![0u8; out_len];
    let mut hh = Kmac::v128(key, customization);
    hh.update(data);
    hh.finalize(&mut expected);
    assert_eq!(ours, expected);
  }

  // KMAC128 XOF (L = 0) + multi-squeeze
  {
    let mut ours = vec![0u8; out_len];
    let mut h = Kmac128::new(key, customization);
    h.update(a);
    h.update(b);
    let mut xof = h.finalize_xof();
    xof.squeeze(&mut ours[..split_out]);
    xof.squeeze(&mut ours[split_out..]);

    use tiny_keccak::{Hasher, IntoXof, Kmac as KmacRef, Xof};
    let mut expected = vec![0u8; out_len];
    let mut hh = KmacRef::v128(key, customization);
    hh.update(data);
    let mut reader = hh.into_xof();
    reader.squeeze(&mut expected[..split_out]);
    reader.squeeze(&mut expected[split_out..]);
    assert_eq!(ours, expected);
  }

  // KMAC256 fixed-length output (L = out_len)
  {
    let mut ours = vec![0u8; out_len];
    let mut h = Kmac256::new(key, customization);
    h.update(a);
    h.update(b);
    h.finalize_into(&mut ours);

    use tiny_keccak::{Hasher, Kmac};
    let mut expected = vec![0u8; out_len];
    let mut hh = Kmac::v256(key, customization);
    hh.update(data);
    hh.finalize(&mut expected);
    assert_eq!(ours, expected);
  }

  // KMAC256 XOF (L = 0) + multi-squeeze
  {
    let mut ours = vec![0u8; out_len];
    let mut h = Kmac256::new(key, customization);
    h.update(a);
    h.update(b);
    let mut xof = h.finalize_xof();
    xof.squeeze(&mut ours[..split_out]);
    xof.squeeze(&mut ours[split_out..]);

    use tiny_keccak::{Hasher, IntoXof, Kmac as KmacRef, Xof};
    let mut expected = vec![0u8; out_len];
    let mut hh = KmacRef::v256(key, customization);
    hh.update(data);
    let mut reader = hh.into_xof();
    reader.squeeze(&mut expected[..split_out]);
    reader.squeeze(&mut expected[split_out..]);
    assert_eq!(ours, expected);
  }
});

