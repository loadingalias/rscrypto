#![no_main]

use libfuzzer_sys::fuzz_target;
use rscrypto::{Cshake256, Xof};
use rscrypto_fuzz::{FuzzInput, split_at_ratio, some_or_return};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let name_split: u8 = some_or_return!(input.byte());
    let custom_split: u8 = some_or_return!(input.byte());
    let message_split: u8 = some_or_return!(input.byte());
    let out_len_byte: u8 = some_or_return!(input.byte());
    let squeeze_split: u8 = some_or_return!(input.byte());
    let rest = input.rest();

    let (name, remainder) = split_at_ratio(rest, name_split);
    let (custom, message) = split_at_ratio(remainder, custom_split);
    let out_len = out_len_byte as usize;
    let split_out = if out_len == 0 {
        0
    } else {
        out_len.strict_mul(squeeze_split as usize) / 255
    };

    // Differential: rscrypto ↔ tiny-keccak
    let mut expected = vec![0u8; out_len];
    use tiny_keccak::{CShake, Hasher, Xof as _};

    let mut oracle = CShake::v256(name, custom);
    oracle.update(message);
    oracle.squeeze(&mut expected[..split_out]);
    oracle.squeeze(&mut expected[split_out..]);

    let (a, b) = split_at_ratio(message, message_split);
    let mut h2 = Cshake256::new(name, custom);
    h2.update(a);
    h2.update(b);
    h2.update(&[]);
    let mut got = vec![0u8; out_len];
    let mut reader = h2.finalize_xof();
    reader.squeeze(&mut got[..split_out]);
    reader.squeeze(&mut got[split_out..]);
    reader.squeeze(&mut []);
    assert_eq!(expected, got, "cshake256 streaming mismatch");

    // Property: reset restores the initial cSHAKE prefix state.
    {
        h2.reset();
        h2.update(message);
        let mut reset = vec![0u8; out_len];
        h2.finalize_xof().squeeze(&mut reset);
        assert_eq!(expected, reset, "cshake256 reset mismatch");
    }

    // Property: one-shot helper matches the oracle too.
    {
        let mut oneshot = vec![0u8; out_len];
        Cshake256::hash_into(name, custom, message, &mut oneshot);
        assert_eq!(expected, oneshot, "cshake256 one-shot mismatch");
    }
});
