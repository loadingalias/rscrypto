// PHC-string fuzzer. Two complementary properties:
//
// 1. No-panic on adversarial input. `decode_string` for every supported
//    algorithm must return a `Result`, never panic, regardless of how
//    structurally invalid the candidate string is.
// 2. Roundtrip on freshly-emitted strings. A variant byte selects
//    Argon2id/2d/2i/Scrypt; verify_string must accept the fresh emission,
//    and any single-byte XOR mutation must produce verify_string-Err.
//
// `verify_string` is intentionally NOT called on attacker-controlled
// candidates: doing so re-executes the KDF with attacker-chosen cost
// parameters, collapsing fuzz throughput. KDF panic-surface is fuzzed
// directly by `auth_argon2{id,d,i}.rs` and `auth_scrypt.rs`.

use libfuzzer_sys::fuzz_target;
use rscrypto::{
    Argon2Params, Argon2Version, Argon2d, Argon2i, Argon2id, Scrypt, ScryptParams,
};
use rscrypto_fuzz::{FuzzInput, pad_salt_to, some_or_return, split_at_ratio};

fn no_panic_on_decode(candidate: &str) {
    // `_` discards both Ok and Err — the assertion is "did not panic".
    // `decode_string` is the parser surface; running it for all four
    // hashers exposes algorithm-tag, version, parameter, and base64
    // edge cases without paying a KDF cost.
    let _ = Argon2id::decode_string(candidate);
    let _ = Argon2d::decode_string(candidate);
    let _ = Argon2i::decode_string(candidate);
    let _ = Scrypt::decode_string(candidate);
}

fn argon2_params(out_len: u32) -> Argon2Params {
    Argon2Params::new()
        .memory_cost_kib(8)
        .time_cost(1)
        .parallelism(1)
        .output_len(out_len)
        .version(Argon2Version::V0x13)
        .build()
        .expect("argon2 fuzz params must validate")
}

fn scrypt_params(out_len: u32) -> ScryptParams {
    ScryptParams::new()
        .log_n(1)
        .r(1)
        .p(1)
        .output_len(out_len)
        .build()
        .expect("scrypt fuzz params must validate")
}

fn assert_roundtrip(encoded: &str, password: &[u8], variant: u8) {
    match variant % 4 {
        0 => Argon2id::verify_string(password, encoded)
            .expect("argon2id verify_string must accept its own emission"),
        1 => Argon2d::verify_string(password, encoded)
            .expect("argon2d verify_string must accept its own emission"),
        2 => Argon2i::verify_string(password, encoded)
            .expect("argon2i verify_string must accept its own emission"),
        _ => Scrypt::verify_string(password, encoded)
            .expect("scrypt verify_string must accept its own emission"),
    }
}

fn assert_byte_flip_rejected(encoded: &str, password: &[u8], variant: u8, flip_byte: u8) {
    if encoded.is_empty() {
        return;
    }
    // PHC strings are pure ASCII (`$,/=+0-9A-Za-z`) — XOR with 0x01 keeps
    // every byte in the ASCII range, so the mutation always lands in valid
    // UTF-8 and the verify_string surface is fully exercised. This matches
    // the single-bit-flip idiom used in every other rscrypto fuzz target.
    let mut bytes = encoded.as_bytes().to_vec();
    let idx = (flip_byte as usize) % bytes.len();
    bytes[idx] ^= 0x01;
    let mutated = core::str::from_utf8(&bytes)
        .expect("ASCII PHC string XOR-1 must remain valid UTF-8");

    let result = match variant % 4 {
        0 => Argon2id::verify_string(password, mutated),
        1 => Argon2d::verify_string(password, mutated),
        2 => Argon2i::verify_string(password, mutated),
        _ => Scrypt::verify_string(password, mutated),
    };
    assert!(
        result.is_err(),
        "verify_string must reject single-bit XOR mutation"
    );
}

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let variant: u8 = some_or_return!(input.byte());
    let pw_salt_split: u8 = some_or_return!(input.byte());
    let out_len_byte: u8 = some_or_return!(input.byte());
    let flip_byte: u8 = some_or_return!(input.byte());
    let rest = input.rest();

    // Output length 4..=32 — RFC 9106 §3.1 minimum is 4; scrypt accepts ≥ 1
    // but the unified range keeps the cross-variant property surface flat.
    let out_len = 4u32.strict_add(u32::from(out_len_byte) % 29);
    let (password, salt_material) = split_at_ratio(rest, pw_salt_split);
    let salt = pad_salt_to::<16>(salt_material, pw_salt_split);

    // ── Property 1 — no panic on adversarial input ──────────────────────
    //
    // `from_utf8_lossy` preserves byte information through replacement-
    // character substitution (U+FFFD), so non-UTF-8 fuzzer bytes still
    // produce a structurally rich candidate string rather than collapsing
    // to empty.
    let candidate = String::from_utf8_lossy(rest);
    no_panic_on_decode(&candidate);

    // ── Property 2 — roundtrip and tamper-rejection ─────────────────────
    let encoded = match variant % 4 {
        0 => Argon2id::hash_string_with_salt(&argon2_params(out_len), password, &salt)
            .expect("argon2id hash_string_with_salt must succeed for fuzz params"),
        1 => Argon2d::hash_string_with_salt(&argon2_params(out_len), password, &salt)
            .expect("argon2d hash_string_with_salt must succeed for fuzz params"),
        2 => Argon2i::hash_string_with_salt(&argon2_params(out_len), password, &salt)
            .expect("argon2i hash_string_with_salt must succeed for fuzz params"),
        _ => Scrypt::hash_string_with_salt(&scrypt_params(out_len), password, &salt)
            .expect("scrypt hash_string_with_salt must succeed for fuzz params"),
    };

    assert_roundtrip(&encoded, password, variant);
    assert_byte_flip_rejected(&encoded, password, variant, flip_byte);
    no_panic_on_decode(&encoded);
});
