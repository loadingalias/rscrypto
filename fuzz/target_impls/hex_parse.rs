
use core::{
    fmt::{Debug, UpperHex},
    str::FromStr,
};

use libfuzzer_sys::fuzz_target;
use rscrypto::{
    aead::Nonce96,
    Aes256GcmKey, Aes256GcmTag, Ed25519PublicKey, Ed25519SecretKey, Ed25519Signature, InvalidHexError,
    X25519PublicKey, X25519SecretKey, X25519SharedSecret,
};
use rscrypto_fuzz::{FuzzInput, some_or_return};

fn ascii_candidate(bytes: &[u8], len: usize) -> String {
    if len == 0 {
        return String::new();
    }

    if bytes.is_empty() {
        return "~".repeat(len);
    }

    bytes.iter()
        .copied()
        .cycle()
        .take(len)
        .map(|byte| char::from(32 + (byte % 95)))
        .collect()
}

fn exercise_public_parse<T>(value: T, candidate: &str)
where
    T: Copy + Eq + Debug + core::fmt::Display + UpperHex + FromStr<Err = InvalidHexError>,
{
    let lower = value.to_string();
    let upper = format!("{value:X}");
    let short = &lower[..lower.len().strict_sub(1)];

    assert_eq!(lower.parse::<T>().unwrap(), value, "public lower parse mismatch");
    assert_eq!(upper.parse::<T>().unwrap(), value, "public upper parse mismatch");
    assert_eq!(
        short.parse::<T>(),
        Err(InvalidHexError::InvalidLength),
        "public invalid length mismatch"
    );

    if let Ok(parsed) = candidate.parse::<T>() {
        assert_eq!(parsed.to_string().parse::<T>().unwrap(), parsed);
        assert_eq!(format!("{parsed:X}").parse::<T>().unwrap(), parsed);
    }
}

fn exercise_secret_parse<T>(
    value: T,
    lower: String,
    upper: String,
    candidate: &str,
)
where
    T: Eq + Debug + FromStr<Err = InvalidHexError>,
{
    let short = &lower[..lower.len().strict_sub(1)];

    assert_eq!(lower.parse::<T>().unwrap(), value, "secret lower parse mismatch");
    assert_eq!(upper.parse::<T>().unwrap(), value, "secret upper parse mismatch");
    assert_eq!(
        short.parse::<T>(),
        Err(InvalidHexError::InvalidLength),
        "secret invalid length mismatch"
    );

    if let Ok(parsed) = candidate.parse::<T>() {
        let rendered = format!("{parsed:?}");
        assert!(!rendered.is_empty(), "secret debug output must exist");
    }
}

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let nonce_bytes: [u8; 12] = some_or_return!(input.bytes());
    let tag_bytes: [u8; 16] = some_or_return!(input.bytes());
    let key_bytes: [u8; 32] = some_or_return!(input.bytes());
    let public_bytes: [u8; 32] = some_or_return!(input.bytes());
    let secret_bytes: [u8; 32] = some_or_return!(input.bytes());
    let signature_bytes: [u8; 64] = some_or_return!(input.bytes());
    let candidate_bytes = input.rest();

    let nonce_candidate = ascii_candidate(candidate_bytes, Nonce96::LENGTH.strict_mul(2));
    let tag_candidate = ascii_candidate(candidate_bytes, Aes256GcmTag::LENGTH.strict_mul(2));
    let key_candidate = ascii_candidate(candidate_bytes, Aes256GcmKey::LENGTH.strict_mul(2));
    let public_candidate = ascii_candidate(candidate_bytes, Ed25519PublicKey::LENGTH.strict_mul(2));
    let secret_candidate = ascii_candidate(candidate_bytes, Ed25519SecretKey::LENGTH.strict_mul(2));
    let signature_candidate = ascii_candidate(candidate_bytes, Ed25519Signature::LENGTH.strict_mul(2));
    let x25519_public_candidate = ascii_candidate(candidate_bytes, X25519PublicKey::LENGTH.strict_mul(2));
    let x25519_secret_candidate = ascii_candidate(candidate_bytes, X25519SecretKey::LENGTH.strict_mul(2));
    let x25519_shared_candidate = ascii_candidate(candidate_bytes, X25519SharedSecret::LENGTH.strict_mul(2));

    let nonce = Nonce96::from_bytes(nonce_bytes);
    exercise_public_parse(nonce, &nonce_candidate);

    let tag = Aes256GcmTag::from_bytes(tag_bytes);
    exercise_public_parse(tag, &tag_candidate);

    let key = Aes256GcmKey::from_bytes(key_bytes);
    exercise_secret_parse(
        key.clone(),
        format!("{}", key.display_secret()),
        format!("{}", key.display_secret()).to_uppercase(),
        &key_candidate,
    );

    let public = Ed25519PublicKey::from_bytes(public_bytes);
    exercise_public_parse(public, &public_candidate);

    let secret = Ed25519SecretKey::from_bytes(secret_bytes);
    exercise_secret_parse(
        secret.clone(),
        format!("{}", secret.display_secret()),
        format!("{}", secret.display_secret()).to_uppercase(),
        &secret_candidate,
    );

    let signature = Ed25519Signature::from_bytes(signature_bytes);
    exercise_public_parse(signature, &signature_candidate);

    let x25519_public = X25519PublicKey::from_bytes(public_bytes);
    exercise_public_parse(x25519_public, &x25519_public_candidate);

    let x25519_secret = X25519SecretKey::from_bytes(secret_bytes);
    exercise_secret_parse(
        x25519_secret.clone(),
        format!("{}", x25519_secret.display_secret()),
        format!("{}", x25519_secret.display_secret()).to_uppercase(),
        &x25519_secret_candidate,
    );

    let x25519_shared = X25519SharedSecret::from_bytes(key_bytes);
    exercise_secret_parse(
        x25519_shared.clone(),
        format!("{}", x25519_shared.display_secret()),
        format!("{}", x25519_shared.display_secret()).to_uppercase(),
        &x25519_shared_candidate,
    );
});
