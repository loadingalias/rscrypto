#![no_main]

use libfuzzer_sys::fuzz_target;
use rscrypto::{Blake3, Digest};
use rscrypto_fuzz::{FuzzInput, assert_xof_prefix, some_or_return};

// Blake3 derive_key context must be a valid &str. We use a fixed context
// and let the fuzzer vary the key material — the context is just a domain
// separator, not the interesting input.
const CONTEXT: &str = "rscrypto fuzz 2026-04-12 derive_key test context";

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let split: u8 = some_or_return!(input.byte());
    let data = input.rest();

    // Property: chunk-split equivalence (derive_key mode)
    let expected = Blake3::derive_key(CONTEXT, data);
    let (a, b) = rscrypto_fuzz::split_at_ratio(data, split);
    let mut h = Blake3::new_derive_key(CONTEXT);
    h.update(a);
    h.update(b);
    let got = h.finalize();
    assert_eq!(expected, got, "derive_key blake3: chunk-split mismatch");

    // Property: reset preserves derive_key mode
    let mut h2 = Blake3::new_derive_key(CONTEXT);
    h2.update(data);
    let first = h2.finalize();
    h2.reset();
    h2.update(data);
    let second = h2.finalize();
    assert_eq!(first, second, "derive_key blake3: reset changed result");

    // Property: XOF prefix consistency (derive_key mode)
    {
        let mut h = Blake3::new_derive_key(CONTEXT);
        h.update(data);
        assert_xof_prefix!(h.finalize_xof(), 32, 128, "derive_key blake3 xof prefix mismatch");
    }

    // Differential: rscrypto ↔ blake3 crate (derive_key mode)
    {
        let oracle = blake3::derive_key(CONTEXT, data);
        assert_eq!(expected, oracle, "derive_key blake3 oracle mismatch");

        // XOF differential
        let mut h = Blake3::new_derive_key(CONTEXT);
        h.update(data);
        let mut reader = h.finalize_xof();
        let mut ours = [0u8; 64];
        rscrypto::Xof::squeeze(&mut reader, &mut ours);

        let mut oh = blake3::Hasher::new_derive_key(CONTEXT);
        oh.update(data);
        let mut or = oh.finalize_xof();
        let mut theirs = [0u8; 64];
        or.fill(&mut theirs);
        assert_eq!(ours, theirs, "derive_key blake3 xof oracle mismatch");
    }
});
