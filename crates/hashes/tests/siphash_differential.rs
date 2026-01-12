use hashes::fast::{SipHash13, SipHash24};
use proptest::prelude::*;
use traits::FastHash as _;

fn siphasher13_ref(key: [u64; 2], data: &[u8]) -> u64 {
  use core::hash::Hasher as _;
  let mut h = siphasher::sip::SipHasher13::new_with_keys(key[0], key[1]);
  h.write(data);
  h.finish()
}

fn siphasher24_ref(key: [u64; 2], data: &[u8]) -> u64 {
  use core::hash::Hasher as _;
  let mut h = siphasher::sip::SipHasher24::new_with_keys(key[0], key[1]);
  h.write(data);
  h.finish()
}

proptest! {
  #[test]
  fn siphash13_matches_siphasher(key in any::<[u64; 2]>(), data in proptest::collection::vec(any::<u8>(), 0..4096)) {
    let ours = SipHash13::hash_with_seed(key, &data);
    let expected = siphasher13_ref(key, &data);
    prop_assert_eq!(ours, expected);
  }

  #[test]
  fn siphash24_matches_siphasher(key in any::<[u64; 2]>(), data in proptest::collection::vec(any::<u8>(), 0..4096)) {
    let ours = SipHash24::hash_with_seed(key, &data);
    let expected = siphasher24_ref(key, &data);
    prop_assert_eq!(ours, expected);
  }
}
