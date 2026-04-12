#![no_main]

use libfuzzer_sys::fuzz_target;
use rscrypto::{Digest, Sha3_224, Sha3_256, Sha3_384, Sha3_512, Shake128, Shake256, Xof};
use rscrypto_fuzz::{FuzzInput, assert_digest_chunked, assert_digest_reset, some_or_return};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let split: u8 = some_or_return!(input.byte());
    let out_len_byte: u8 = some_or_return!(input.byte());
    let squeeze_split: u8 = some_or_return!(input.byte());
    let data = input.rest();
    let out_len = (out_len_byte as usize % 256).strict_add(1);
    let split_out = out_len.strict_mul(squeeze_split as usize) / 256;

    // Property: chunk-split equivalence for all SHA-3 variants
    assert_digest_chunked::<Sha3_224>(data, split);
    assert_digest_chunked::<Sha3_256>(data, split);
    assert_digest_chunked::<Sha3_384>(data, split);
    assert_digest_chunked::<Sha3_512>(data, split);

    // Property: reset correctness
    assert_digest_reset::<Sha3_256>(data);

    // Differential: rscrypto ↔ sha3 crate
    {
        use digest::Digest as _;

        let ours = Sha3_256::digest(data);
        let oracle = sha3::Sha3_256::digest(data);
        assert_eq!(&ours[..], oracle.as_slice(), "sha3-256 mismatch");

        let ours = Sha3_224::digest(data);
        let oracle = sha3::Sha3_224::digest(data);
        assert_eq!(&ours[..], oracle.as_slice(), "sha3-224 mismatch");

        let ours = Sha3_384::digest(data);
        let oracle = sha3::Sha3_384::digest(data);
        assert_eq!(&ours[..], oracle.as_slice(), "sha3-384 mismatch");

        let ours = Sha3_512::digest(data);
        let oracle = sha3::Sha3_512::digest(data);
        assert_eq!(&ours[..], oracle.as_slice(), "sha3-512 mismatch");
    }

    // Differential: rscrypto SHAKE ↔ sha3 crate
    {
        use sha3::digest::{ExtendableOutput, Update, XofReader};

        let mut oracle = sha3::Shake256::default();
        oracle.update(data);
        let mut oracle_reader = oracle.finalize_xof();
        let mut expected = vec![0u8; out_len];
        oracle_reader.read(&mut expected[..split_out]);
        oracle_reader.read(&mut expected[split_out..]);

        let mut h = Shake256::new();
        h.update(data);
        let mut reader = h.finalize_xof();
        let mut actual = vec![0u8; out_len];
        reader.squeeze(&mut actual[..split_out]);
        reader.squeeze(&mut actual[split_out..]);
        assert_eq!(actual, expected, "shake256 streaming xof mismatch");

        let mut oneshot = Shake256::xof(data);
        let mut oneshot_out = vec![0u8; out_len];
        oneshot.squeeze(&mut oneshot_out);
        assert_eq!(oneshot_out, expected, "shake256 oneshot xof mismatch");
    }

    {
        use sha3::digest::{ExtendableOutput, Update, XofReader};

        let mut oracle = sha3::Shake128::default();
        oracle.update(data);
        let mut oracle_reader = oracle.finalize_xof();
        let mut expected = vec![0u8; out_len];
        oracle_reader.read(&mut expected[..split_out]);
        oracle_reader.read(&mut expected[split_out..]);

        let mut h = Shake128::new();
        h.update(data);
        let mut reader = h.finalize_xof();
        let mut actual = vec![0u8; out_len];
        reader.squeeze(&mut actual[..split_out]);
        reader.squeeze(&mut actual[split_out..]);
        assert_eq!(actual, expected, "shake128 streaming xof mismatch");

        let mut oneshot = Shake128::xof(data);
        let mut oneshot_out = vec![0u8; out_len];
        oneshot.squeeze(&mut oneshot_out);
        assert_eq!(oneshot_out, expected, "shake128 oneshot xof mismatch");
    }
});
