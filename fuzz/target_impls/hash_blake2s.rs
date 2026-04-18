
use blake2::{Blake2s128 as OracleBlake2s128, Blake2s256 as OracleBlake2s256, Blake2sMac, Digest as _};
use digest::typenum::{U16, U32};
use hmac::{Mac as _, digest::KeyInit};
use libfuzzer_sys::fuzz_target;
use rscrypto::{Blake2s128, Blake2s256, Digest};
use rscrypto_fuzz::{FuzzInput, assert_digest_chunked, assert_digest_reset, some_or_return, split_at_ratio};

type OracleBlake2sMac128 = Blake2sMac<U16>;
type OracleBlake2sMac256 = Blake2sMac<U32>;

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let split: u8 = some_or_return!(input.byte());
    let key_ratio: u8 = some_or_return!(input.byte());
    let data = input.rest();

    assert_digest_chunked::<Blake2s128>(data, split);
    assert_digest_chunked::<Blake2s256>(data, split);
    assert_digest_reset::<Blake2s128>(data);
    assert_digest_reset::<Blake2s256>(data);

    let ours_128 = Blake2s128::digest(data);
    let oracle_128 = OracleBlake2s128::digest(data);
    assert_eq!(&ours_128[..], oracle_128.as_slice(), "blake2s128 mismatch");

    let ours_256 = Blake2s256::digest(data);
    let oracle_256 = OracleBlake2s256::digest(data);
    assert_eq!(&ours_256[..], oracle_256.as_slice(), "blake2s256 mismatch");

    if !data.is_empty() {
        let split_idx = split_at_ratio(data, key_ratio).0.len();
        let key_len = split_idx.clamp(1, 32);
        let (key, msg) = data.split_at(key_len);
        let (msg_a, msg_b) = split_at_ratio(msg, split);

        let mut ours_128_stream = Blake2s128::new_keyed(key);
        ours_128_stream.update(msg_a);
        ours_128_stream.update(msg_b);
        let ours_128_keyed = ours_128_stream.finalize();

        let mut oracle_128_mac = OracleBlake2sMac128::new_from_slice(key).unwrap();
        oracle_128_mac.update(msg);
        let oracle_128_keyed = oracle_128_mac.finalize().into_bytes();
        assert_eq!(&ours_128_keyed[..], &oracle_128_keyed[..], "blake2s128 keyed mismatch");

        let mut ours_256_stream = Blake2s256::new_keyed(key);
        ours_256_stream.update(msg_a);
        ours_256_stream.update(msg_b);
        let ours_256_keyed = ours_256_stream.finalize();

        let mut oracle_256_mac = OracleBlake2sMac256::new_from_slice(key).unwrap();
        oracle_256_mac.update(msg);
        let oracle_256_keyed = oracle_256_mac.finalize().into_bytes();
        assert_eq!(&ours_256_keyed[..], &oracle_256_keyed[..], "blake2s256 keyed mismatch");
    }
});
