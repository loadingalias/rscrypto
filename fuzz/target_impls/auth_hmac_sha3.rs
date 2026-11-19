use rscrypto::{HmacSha3_224, HmacSha3_256, HmacSha3_384, HmacSha3_512};
use rscrypto_fuzz::{
  FuzzInput, assert_mac_against_oracle, assert_mac_reset, assert_mac_streaming, some_or_return, split_at_ratio,
};

macro_rules! check {
  ($ours:ty, $digest:ty, $key:expr, $message:expr, $split:expr) => {{
    assert_mac_streaming::<$ours>($key, $message, $split);
    assert_mac_reset::<$ours>($key, $message);
    let tag = <$ours>::mac($key, $message);
    <$ours>::verify_tag($key, $message, &tag).expect("verify must accept the correct tag");
    // Differential: rscrypto ↔ RustCrypto SimpleHmac over sha3
    assert_mac_against_oracle::<$ours>($key, $message, &tag, |key, message| {
      use hmac::{KeyInit as _, Mac as _};
      let mut oracle = hmac::SimpleHmac::<$digest>::new_from_slice(key).expect("HMAC accepts keys of every length");
      oracle.update(message);
      oracle.finalize().into_bytes().to_vec()
    });
  }};
}

pub(super) fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let split: u8 = some_or_return!(input.byte());
  let key_split: u8 = some_or_return!(input.byte());
  let (key, message) = split_at_ratio(input.rest(), key_split);

  check!(HmacSha3_224, sha3::Sha3_224, key, message, split);
  check!(HmacSha3_256, sha3::Sha3_256, key, message, split);
  check!(HmacSha3_384, sha3::Sha3_384, key, message, split);
  check!(HmacSha3_512, sha3::Sha3_512, key, message, split);
}
