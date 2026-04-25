use libfuzzer_sys::fuzz_target;
use rscrypto::HmacSha256;
use rscrypto_fuzz::{
    FuzzInput, assert_mac_against_oracle, assert_mac_reset, assert_mac_streaming,
    some_or_return, split_at_ratio,
};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let split: u8 = some_or_return!(input.byte());
    let key_split: u8 = some_or_return!(input.byte());
    let rest = input.rest();

    // Variable-length key is important for HMAC's longer-than-block-size path.
    let (key, message) = split_at_ratio(rest, key_split);

    assert_mac_streaming::<HmacSha256>(key, message, split);
    assert_mac_reset::<HmacSha256>(key, message);

    let tag = HmacSha256::mac(key, message);
    HmacSha256::verify_tag(key, message, &tag).expect("verify must accept correct tag");

    assert_mac_against_oracle::<HmacSha256>(key, message, &tag, |key, msg| {
        use hmac::{Hmac, KeyInit, Mac as _};
        let mut oracle = <Hmac<sha2::Sha256> as KeyInit>::new_from_slice(key).unwrap();
        oracle.update(msg);
        oracle.finalize().into_bytes().to_vec()
    });
});
