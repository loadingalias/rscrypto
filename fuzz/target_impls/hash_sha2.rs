
use libfuzzer_sys::fuzz_target;
use rscrypto::{Sha224, Sha256, Sha384, Sha512, Sha512_256};
use rscrypto_fuzz::{FuzzInput, assert_digest_chunked, assert_digest_reset, some_or_return};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let split: u8 = some_or_return!(input.byte());
    let data = input.rest();

    // Property: chunk-split equivalence for all SHA-2 variants
    assert_digest_chunked::<Sha256>(data, split);
    assert_digest_chunked::<Sha224>(data, split);
    assert_digest_chunked::<Sha384>(data, split);
    assert_digest_chunked::<Sha512>(data, split);
    assert_digest_chunked::<Sha512_256>(data, split);

    // Property: reset correctness for all SHA-2 variants
    assert_digest_reset::<Sha256>(data);
    assert_digest_reset::<Sha224>(data);
    assert_digest_reset::<Sha384>(data);
    assert_digest_reset::<Sha512>(data);
    assert_digest_reset::<Sha512_256>(data);

    // Differential: rscrypto ↔ sha2 crate
    {
        use digest::Digest as _;

        let ours_256 = Sha256::digest(data);
        let oracle_256 = sha2::Sha256::digest(data);
        assert_eq!(&ours_256[..], oracle_256.as_slice(), "sha256 mismatch");

        let ours_224 = Sha224::digest(data);
        let oracle_224 = sha2::Sha224::digest(data);
        assert_eq!(&ours_224[..], oracle_224.as_slice(), "sha224 mismatch");

        let ours_384 = Sha384::digest(data);
        let oracle_384 = sha2::Sha384::digest(data);
        assert_eq!(&ours_384[..], oracle_384.as_slice(), "sha384 mismatch");

        let ours_512 = Sha512::digest(data);
        let oracle_512 = sha2::Sha512::digest(data);
        assert_eq!(&ours_512[..], oracle_512.as_slice(), "sha512 mismatch");

        let ours_512_256 = Sha512_256::digest(data);
        let oracle_512_256 = sha2::Sha512_256::digest(data);
        assert_eq!(
            &ours_512_256[..],
            oracle_512_256.as_slice(),
            "sha512/256 mismatch"
        );
    }
});
