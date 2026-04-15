
use libfuzzer_sys::fuzz_target;
use rscrypto::{HmacSha256, Mac};
use rscrypto_fuzz::{FuzzInput, assert_mac_streaming, split_at_ratio, some_or_return};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let split: u8 = some_or_return!(input.byte());
    let key_split: u8 = some_or_return!(input.byte());
    let rest = input.rest();

    // Split rest into (key, message) — variable-length key is important
    let (key, message) = split_at_ratio(rest, key_split);

    // Property: streaming equivalence
    assert_mac_streaming::<HmacSha256>(key, message, split);

    // Property: verify succeeds for correct tag
    let tag = HmacSha256::mac(key, message);
    HmacSha256::verify_tag(key, message, &tag).expect("verify must accept correct tag");

    // Property: reset restores keyed initial state
    {
        let mut m = HmacSha256::new(key);
        m.update(message);
        let first = m.finalize();
        m.reset();
        m.update(message);
        let second = m.finalize();
        assert_eq!(first, second, "hmac: changed after reset");
    }

    // Differential: rscrypto ↔ hmac crate
    {
        use hmac::{Hmac, Mac as _, KeyInit};
        type OracleHmac = Hmac<sha2::Sha256>;

        let mut oracle = <OracleHmac as KeyInit>::new_from_slice(key).unwrap();
        oracle.update(message);
        let oracle_tag = oracle.finalize().into_bytes();

        assert_eq!(&tag[..], oracle_tag.as_slice(), "hmac oracle mismatch");
    }
});
