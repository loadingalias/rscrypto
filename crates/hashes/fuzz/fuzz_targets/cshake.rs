#![no_main]

use hashes::crypto::{CShake128, CShake256};
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
  // - 1 byte fn_len
  // - 1 byte custom_len
  // - 2 bytes out_len (capped)
  // - remaining: fn || custom || data
  let fn_len = input.get(0).copied().unwrap_or(0) as usize % 65;
  let custom_len = input.get(1).copied().unwrap_or(0) as usize % 65;
  let out_len = (parse_u16_le(input.get(2..).unwrap_or(&[])) as usize) % 1025;
  let rest = input.get(4..).unwrap_or(&[]);

  let fn_len = core::cmp::min(fn_len, rest.len());
  let function_name = &rest[..fn_len];
  let rest = &rest[fn_len..];

  let custom_len = core::cmp::min(custom_len, rest.len());
  let customization = &rest[..custom_len];
  let data = &rest[custom_len..];

  let split_data = if data.is_empty() { 0 } else { (data[0] as usize) % (data.len() + 1) };
  let (a, b) = data.split_at(split_data);
  let split_out = split_len(out_len, data.get(1).copied().unwrap_or(0));

  // cSHAKE128
  {
    let mut ours = vec![0u8; out_len];
    CShake128::hash_into(function_name, customization, data, &mut ours);

    let mut h = CShake128::new_with_function_name(function_name, customization);
    h.update(a);
    h.update(b);
    let mut xof = h.finalize_xof();
    let mut streamed = vec![0u8; out_len];
    xof.squeeze(&mut streamed[..split_out]);
    xof.squeeze(&mut streamed[split_out..]);
    assert_eq!(ours, streamed);

    use sha3::digest::{ExtendableOutput, Update, XofReader};
    let core = if function_name.is_empty() {
      sha3::CShake128Core::new(customization)
    } else {
      sha3::CShake128Core::new_with_function_name(function_name, customization)
    };
    let mut hh = sha3::CShake128::from_core(core);
    hh.update(data);
    let mut reader = hh.finalize_xof();
    let mut expected = vec![0u8; out_len];
    reader.read(&mut expected);
    assert_eq!(ours, expected);
  }

  // cSHAKE256
  {
    let mut ours = vec![0u8; out_len];
    CShake256::hash_into(function_name, customization, data, &mut ours);

    let mut h = CShake256::new_with_function_name(function_name, customization);
    h.update(a);
    h.update(b);
    let mut xof = h.finalize_xof();
    let mut streamed = vec![0u8; out_len];
    xof.squeeze(&mut streamed[..split_out]);
    xof.squeeze(&mut streamed[split_out..]);
    assert_eq!(ours, streamed);

    use sha3::digest::{ExtendableOutput, Update, XofReader};
    let core = if function_name.is_empty() {
      sha3::CShake256Core::new(customization)
    } else {
      sha3::CShake256Core::new_with_function_name(function_name, customization)
    };
    let mut hh = sha3::CShake256::from_core(core);
    hh.update(data);
    let mut reader = hh.finalize_xof();
    let mut expected = vec![0u8; out_len];
    reader.read(&mut expected);
    assert_eq!(ours, expected);
  }
});

