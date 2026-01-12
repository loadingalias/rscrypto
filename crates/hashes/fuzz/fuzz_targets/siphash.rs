#![no_main]

use hashes::fast::{SipHash13, SipHash24};
use libfuzzer_sys::fuzz_target;
use traits::FastHash as _;

fuzz_target!(|input: &[u8]| {
  let key_bytes_len = core::cmp::min(16, input.len());
  let (key_bytes, data) = input.split_at(key_bytes_len);

  let mut key = [0u64; 2];
  for i in 0..2 {
    let start = i * 8;
    if start >= key_bytes.len() {
      break;
    }
    let end = core::cmp::min(start + 8, key_bytes.len());
    let mut tmp = [0u8; 8];
    tmp[..end - start].copy_from_slice(&key_bytes[start..end]);
    key[i] = u64::from_le_bytes(tmp);
  }

  let ours13 = SipHash13::hash_with_seed(key, data);
  let ours24 = SipHash24::hash_with_seed(key, data);

  use core::hash::Hasher as _;
  let mut h13 = siphasher::sip::SipHasher13::new_with_keys(key[0], key[1]);
  h13.write(data);
  let exp13 = h13.finish();

  let mut h24 = siphasher::sip::SipHasher24::new_with_keys(key[0], key[1]);
  h24.write(data);
  let exp24 = h24.finish();

  assert_eq!(ours13, exp13);
  assert_eq!(ours24, exp24);
});
