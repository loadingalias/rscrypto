use libfuzzer_sys::fuzz_target;
use rscrypto::{Argon2Params, Argon2Version, Argon2i};
use rscrypto_fuzz::{FuzzInput, pad_salt_to, some_or_return, split_at_ratio};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let pw_salt_split: u8 = some_or_return!(input.byte());
    let out_len_byte: u8 = some_or_return!(input.byte());
    let m_byte: u8 = some_or_return!(input.byte());
    let t_byte: u8 = some_or_return!(input.byte());
    let rest = input.rest();

    let m_kib = 8u32.strict_add(u32::from(m_byte) % 16);
    let t = 1u32.strict_add(u32::from(t_byte) % 4);
    let out_len = 4u32.strict_add(u32::from(out_len_byte) % 29);

    let params = Argon2Params::new()
        .memory_cost_kib(m_kib)
        .time_cost(t)
        .parallelism(1)
        .output_len(out_len)
        .version(Argon2Version::V0x13)
        .build()
        .expect("params must be valid for fuzzer ranges");

    // See `auth_argon2id.rs` for cost-parameter rationale.
    let (password, salt_material) = split_at_ratio(rest, pw_salt_split);
    let salt_buf = pad_salt_to::<16>(salt_material, pw_salt_split);

    let mut actual = vec![0u8; out_len as usize];
    Argon2i::hash(&params, password, &salt_buf, &mut actual).expect("hash within fuzz ranges");

    Argon2i::verify(&params, password, &salt_buf, &actual)
        .expect("verify must accept the hash it just produced");

    if !actual.is_empty() {
        let mut tampered = actual.clone();
        tampered[0] ^= 1;
        assert!(
            Argon2i::verify(&params, password, &salt_buf, &tampered).is_err(),
            "verify must reject single-bit flip"
        );
    }

    {
        let oracle_params = argon2::Params::new(m_kib, t, 1, Some(out_len as usize))
            .expect("oracle params");
        let ctx = argon2::Argon2::new(argon2::Algorithm::Argon2i, argon2::Version::V0x13, oracle_params);
        let mut expected = vec![0u8; out_len as usize];
        ctx.hash_password_into(password, &salt_buf, &mut expected)
            .expect("oracle hash");
        assert_eq!(actual, expected, "argon2i oracle mismatch");
    }
});
