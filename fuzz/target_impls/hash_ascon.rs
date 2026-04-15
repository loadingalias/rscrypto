
use libfuzzer_sys::fuzz_target;
use rscrypto::{AsconHash256, AsconXof, Digest, Xof};
use rscrypto_fuzz::{
    FuzzInput, assert_digest_chunked, assert_digest_reset, some_or_return, split_at_ratio,
};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let split: u8 = some_or_return!(input.byte());
    let out_len_byte: u8 = some_or_return!(input.byte());
    let squeeze_split: u8 = some_or_return!(input.byte());
    let data = input.rest();
    let out_len = out_len_byte as usize;
    let split_out = if out_len == 0 {
        0
    } else {
        out_len.strict_mul(squeeze_split as usize) / 255
    };

    // Property: chunk-split equivalence
    assert_digest_chunked::<AsconHash256>(data, split);

    // Property: reset correctness
    assert_digest_reset::<AsconHash256>(data);

    // Property: Ascon-XOF one-shot vs streaming, including split squeezes.
    {
        let mut expected = vec![0u8; out_len];
        AsconXof::hash_into(data, &mut expected);

        let (a, b) = split_at_ratio(data, split.wrapping_add(29));
        let mut h = AsconXof::new();
        h.update(a);
        h.update(b);
        h.update(&[]);
        let mut reader = h.finalize_xof();

        let mut got = vec![0u8; out_len];
        reader.squeeze(&mut got[..split_out]);
        reader.squeeze(&mut got[split_out..]);
        reader.squeeze(&mut []);
        assert_eq!(expected, got, "ascon xof streaming mismatch");

        h.reset();
        h.update(data);
        let mut reset = vec![0u8; out_len];
        h.finalize_xof().squeeze(&mut reset);
        assert_eq!(expected, reset, "ascon xof reset mismatch");
    }

    // Differential: rscrypto ↔ ascon-hash crate
    {
        use digest::Digest as _;

        let ours = AsconHash256::digest(data);
        let oracle = ascon_hash::AsconHash256::digest(data);
        assert_eq!(&ours[..], oracle.as_slice(), "ascon hash mismatch");
    }
});
