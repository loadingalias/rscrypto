#![no_main]

use libfuzzer_sys::fuzz_target;
use rscrypto::Kmac256;
use rscrypto_fuzz::{FuzzInput, split_at_ratio, some_or_return};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let split: u8 = some_or_return!(input.byte());
    let key_split: u8 = some_or_return!(input.byte());
    let out_len_byte: u8 = some_or_return!(input.byte());
    let rest = input.rest();

    let (key, remainder) = split_at_ratio(rest, key_split);
    let (custom, message) = split_at_ratio(remainder, split);
    let out_len = (out_len_byte as usize % 128).strict_add(1);

    // Property: streaming equivalence
    let mut expected = vec![0u8; out_len];
    Kmac256::mac_into(key, custom, message, &mut expected);

    let (a, b) = split_at_ratio(message, split.wrapping_add(37));
    let mut kmac = Kmac256::new(key, custom);
    kmac.update(a);
    kmac.update(b);
    let mut got = vec![0u8; out_len];
    kmac.finalize_into(&mut got);
    assert_eq!(expected, got, "streaming kmac mismatch");

    // Property: reset restores initial state
    kmac.reset();
    kmac.update(message);
    let mut reset_out = vec![0u8; out_len];
    kmac.finalize_into(&mut reset_out);
    assert_eq!(expected, reset_out, "kmac changed after reset");

    // Property: verify accepts correct tag
    Kmac256::verify(key, custom, message, &expected).expect("verify must accept correct tag");

    // Differential: rscrypto ↔ tiny-keccak
    {
        use tiny_keccak::{Hasher, Kmac as OracleKmac};

        let mut oracle = OracleKmac::v256(key, custom);
        oracle.update(message);
        let mut oracle_out = vec![0u8; out_len];
        oracle.finalize(&mut oracle_out);
        assert_eq!(expected, oracle_out, "tiny-keccak kmac oracle mismatch");
    }
});
