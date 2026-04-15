
use libfuzzer_sys::fuzz_target;
use rscrypto::{Blake3, Digest};
use rscrypto_fuzz::{FuzzInput, assert_xof_prefix, some_or_return};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let key: [u8; 32] = some_or_return!(input.bytes());
    let split: u8 = some_or_return!(input.byte());
    let data = input.rest();

    // Property: chunk-split equivalence (keyed mode)
    let expected = Blake3::keyed_digest(&key, data);
    let (a, b) = rscrypto_fuzz::split_at_ratio(data, split);
    let mut h = Blake3::new_keyed(&key);
    h.update(a);
    h.update(b);
    let got = h.finalize();
    assert_eq!(expected, got, "keyed blake3: chunk-split mismatch");

    // Property: reset preserves keyed mode
    let mut h2 = Blake3::new_keyed(&key);
    h2.update(data);
    let first = h2.finalize();
    h2.reset();
    h2.update(data);
    let second = h2.finalize();
    assert_eq!(first, second, "keyed blake3: reset changed result");

    // Property: XOF prefix consistency (keyed mode)
    {
        let mut h = Blake3::new_keyed(&key);
        h.update(data);
        assert_xof_prefix!(h.finalize_xof(), 32, 128, "keyed blake3 xof prefix mismatch");
    }

    // Differential: rscrypto ↔ blake3 crate (keyed mode)
    {
        let oracle = blake3::keyed_hash(&key, data);
        assert_eq!(&expected[..], oracle.as_bytes(), "keyed blake3 oracle mismatch");

        // XOF differential
        let mut h = Blake3::new_keyed(&key);
        h.update(data);
        let mut reader = h.finalize_xof();
        let mut ours = [0u8; 64];
        rscrypto::Xof::squeeze(&mut reader, &mut ours);

        let mut oh = blake3::Hasher::new_keyed(&key);
        oh.update(data);
        let mut or = oh.finalize_xof();
        let mut theirs = [0u8; 64];
        or.fill(&mut theirs);
        assert_eq!(ours, theirs, "keyed blake3 xof oracle mismatch");
    }
});
