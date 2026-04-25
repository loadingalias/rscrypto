use libfuzzer_sys::fuzz_target;
use rscrypto::{Argon2Params, Argon2Version, Argon2id};
use rscrypto_fuzz::{FuzzInput, pad_salt_to, some_or_return, split_at_ratio};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let pw_salt_split: u8 = some_or_return!(input.byte());
    let out_len_byte: u8 = some_or_return!(input.byte());
    let m_byte: u8 = some_or_return!(input.byte());
    let t_byte: u8 = some_or_return!(input.byte());
    let rest = input.rest();

    // Scale fuzzer bytes into Miri/CI-friendly cost parameters.
    // p = 1 keeps single-lane path; m_kib ∈ [8, 23] and t ∈ [1, 4] stay small
    // enough that thousands of iterations complete in seconds per fuzz job.
    let m_kib = 8u32.strict_add(u32::from(m_byte) % 16);
    let t = 1u32.strict_add(u32::from(t_byte) % 4);
    let out_len = 4u32.strict_add(u32::from(out_len_byte) % 29); // 4..=32 per RFC 9106 §3.1

    let params = Argon2Params::new()
        .memory_cost_kib(m_kib)
        .time_cost(t)
        .parallelism(1)
        .output_len(out_len)
        .version(Argon2Version::V0x13)
        .build()
        .expect("params must be valid for fuzzer ranges");

    let (password, salt_material) = split_at_ratio(rest, pw_salt_split);
    let salt_buf = pad_salt_to::<16>(salt_material, pw_salt_split);

    // Property: hash succeeds for valid inputs within the fuzz-constrained range.
    let mut actual = vec![0u8; out_len as usize];
    Argon2id::hash(&params, password, &salt_buf, &mut actual).expect("hash within fuzz ranges");

    // Property: verify accepts the hash it just produced.
    Argon2id::verify(&params, password, &salt_buf, &actual)
        .expect("verify must accept the hash it just produced");

    // Property: a single-bit flip in the hash is always rejected.
    if !actual.is_empty() {
        let mut tampered = actual.clone();
        tampered[0] ^= 1;
        assert!(
            Argon2id::verify(&params, password, &salt_buf, &tampered).is_err(),
            "verify must reject single-bit flip"
        );
    }

    // Differential: rscrypto ↔ RustCrypto argon2 oracle.
    {
        let oracle_params = argon2::Params::new(m_kib, t, 1, Some(out_len as usize))
            .expect("oracle params");
        let ctx = argon2::Argon2::new(argon2::Algorithm::Argon2id, argon2::Version::V0x13, oracle_params);
        let mut expected = vec![0u8; out_len as usize];
        ctx.hash_password_into(password, &salt_buf, &mut expected)
            .expect("oracle hash");
        assert_eq!(actual, expected, "argon2id oracle mismatch");
    }
});
