#![no_main]

use libfuzzer_sys::fuzz_target;
use rscrypto::HkdfSha384;
use rscrypto_fuzz::{FuzzInput, split_at_ratio, some_or_return};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let salt_split: u8 = some_or_return!(input.byte());
    let ikm_split: u8 = some_or_return!(input.byte());
    let out_len_bytes: [u8; 2] = some_or_return!(input.bytes());
    let rest = input.rest();

    let (salt, remainder) = split_at_ratio(rest, salt_split);
    let (ikm, info) = split_at_ratio(remainder, ikm_split);

    let out_len = usize::from(u16::from_le_bytes(out_len_bytes))
        % HkdfSha384::MAX_OUTPUT_SIZE.strict_add(33);

    let hk = HkdfSha384::new(salt, ikm);
    let mut okm = vec![0u8; out_len];
    let ours_expand = hk.expand(info, &mut okm);

    let mut okm2 = vec![0u8; out_len];
    let ours_derive = HkdfSha384::derive(salt, ikm, info, &mut okm2);
    assert_eq!(
        ours_expand.is_ok(),
        ours_derive.is_ok(),
        "hkdf-sha384 expand vs derive result mismatch"
    );

    let hk2 = HkdfSha384::extract(salt, ikm);
    assert_eq!(hk.prk(), hk2.prk(), "hkdf-sha384 extract vs new PRK mismatch");

    let oracle = hkdf::Hkdf::<sha2::Sha384>::new(Some(salt), ikm);
    let mut oracle_okm = vec![0u8; out_len];
    let oracle_expand = oracle.expand(info, &mut oracle_okm);
    assert_eq!(
        ours_expand.is_ok(),
        oracle_expand.is_ok(),
        "hkdf-sha384 oracle result mismatch"
    );

    if ours_expand.is_ok() {
        assert_eq!(okm, okm2, "hkdf-sha384 expand vs derive mismatch");
        assert_eq!(okm, oracle_okm, "hkdf-sha384 oracle mismatch");
    }
});
