use libfuzzer_sys::fuzz_target;
use rscrypto::{Scrypt, ScryptParams};
use rscrypto_fuzz::{FuzzInput, some_or_return, split_at_ratio};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let pw_salt_split: u8 = some_or_return!(input.byte());
    let out_len_byte: u8 = some_or_return!(input.byte());
    let log_n_byte: u8 = some_or_return!(input.byte());
    let r_byte: u8 = some_or_return!(input.byte());
    let rest = input.rest();

    // Fuzzer-friendly cost: log_n ∈ [1, 6] (N ≤ 64), r ∈ [1, 4], p = 1.
    // Keeps each iteration ≤ ~1ms while still exercising every code path
    // including the log_n=1 pair-unroll boundary.
    let log_n = 1u8.strict_add(log_n_byte % 6);
    let r = 1u32.strict_add(u32::from(r_byte) % 4);
    let out_len = 1u32.strict_add(u32::from(out_len_byte) % 64);

    let params = ScryptParams::new()
        .log_n(log_n)
        .r(r)
        .p(1)
        .output_len(out_len)
        .build()
        .expect("params must be valid for fuzzer ranges");

    let (password, salt) = split_at_ratio(rest, pw_salt_split);

    // Property: hash succeeds for valid inputs.
    let mut actual = vec![0u8; out_len as usize];
    Scrypt::hash(&params, password, salt, &mut actual).expect("hash within fuzz ranges");

    // Property: verify accepts the hash it just produced.
    Scrypt::verify(&params, password, salt, &actual)
        .expect("verify must accept the hash it just produced");

    // Property: a single-bit flip in the hash is always rejected.
    if !actual.is_empty() {
        let mut tampered = actual.clone();
        tampered[0] ^= 1;
        assert!(
            Scrypt::verify(&params, password, salt, &tampered).is_err(),
            "verify must reject single-bit flip"
        );
    }

    // Differential: rscrypto ↔ RustCrypto scrypt oracle.
    //
    // The oracle rejects `len ∉ [10, 64]` per its `Params::new` contract,
    // while rscrypto accepts any `output_len ≥ MIN_OUTPUT_LEN = 1`. Skip
    // the differential below the oracle's window — coverage of the small
    // output range is provided by the rscrypto-only properties above.
    if (10..=64).contains(&out_len) {
        let oracle_params = scrypt::Params::new(log_n, r, 1, out_len as usize)
            .expect("oracle params");
        let mut expected = vec![0u8; out_len as usize];
        scrypt::scrypt(password, salt, &oracle_params, &mut expected).expect("oracle scrypt");
        assert_eq!(actual, expected, "scrypt oracle mismatch");
    }
});
