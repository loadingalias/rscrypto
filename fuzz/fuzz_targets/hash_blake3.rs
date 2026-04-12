#![no_main]

use libfuzzer_sys::fuzz_target;
use rscrypto::{Blake3, Digest};
use rscrypto_fuzz::{FuzzInput, assert_digest_chunked, assert_digest_reset, some_or_return};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let split: u8 = some_or_return!(input.byte());
    let out_len_byte: u8 = some_or_return!(input.byte());
    let squeeze_split: u8 = some_or_return!(input.byte());
    let data = input.rest();
    let out_len = (out_len_byte as usize % 256).strict_add(1);
    let split_out = out_len.strict_mul(squeeze_split as usize) / 256;

    // Property: chunk-split equivalence
    assert_digest_chunked::<Blake3>(data, split);

    // Property: reset correctness
    assert_digest_reset::<Blake3>(data);

    // Differential: rscrypto ↔ blake3 crate
    {
        let ours = Blake3::digest(data);
        let oracle = blake3::hash(data);
        assert_eq!(&ours[..], oracle.as_bytes(), "blake3 hash mismatch");

        let mut h = Blake3::new();
        h.update(data);
        let mut reader = h.finalize_xof();
        let mut ours_xof = vec![0u8; out_len];
        rscrypto::Xof::squeeze(&mut reader, &mut ours_xof[..split_out]);
        rscrypto::Xof::squeeze(&mut reader, &mut ours_xof[split_out..]);

        let mut oracle_hasher = blake3::Hasher::new();
        oracle_hasher.update(data);
        let mut oracle_reader = oracle_hasher.finalize_xof();
        let mut oracle_xof = vec![0u8; out_len];
        oracle_reader.fill(&mut oracle_xof);

        assert_eq!(ours_xof, oracle_xof, "blake3 xof mismatch");

        let mut oneshot = Blake3::xof(data);
        let mut oneshot_xof = vec![0u8; out_len];
        rscrypto::Xof::squeeze(&mut oneshot, &mut oneshot_xof);
        assert_eq!(oneshot_xof, oracle_xof, "blake3 oneshot xof mismatch");
    }
});
