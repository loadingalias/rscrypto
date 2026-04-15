
use libfuzzer_sys::fuzz_target;
use rscrypto::{HmacSha512, Mac};
use rscrypto_fuzz::{FuzzInput, assert_mac_streaming, split_at_ratio, some_or_return};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let split: u8 = some_or_return!(input.byte());
    let key_split: u8 = some_or_return!(input.byte());
    let rest = input.rest();

    let (key, message) = split_at_ratio(rest, key_split);

    assert_mac_streaming::<HmacSha512>(key, message, split);

    let tag = HmacSha512::mac(key, message);
    HmacSha512::verify_tag(key, message, &tag).expect("hmac-sha512 verify must accept correct tag");

    {
        let mut mac = HmacSha512::new(key);
        mac.update(message);
        let first = mac.finalize();
        mac.reset();
        mac.update(message);
        let second = mac.finalize();
        assert_eq!(first, second, "hmac-sha512 changed after reset");
    }

    {
        use hmac::{Hmac, KeyInit, Mac as _};
        type OracleHmac512 = Hmac<sha2::Sha512>;

        let mut oracle = <OracleHmac512 as KeyInit>::new_from_slice(key).unwrap();
        oracle.update(message);
        let oracle_tag = oracle.finalize().into_bytes();

        assert_eq!(&tag[..], oracle_tag.as_slice(), "hmac-sha512 oracle mismatch");
    }
});
