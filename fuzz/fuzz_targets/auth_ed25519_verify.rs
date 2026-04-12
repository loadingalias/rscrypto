#![no_main]

use libfuzzer_sys::fuzz_target;
use rscrypto::{Ed25519PublicKey, Ed25519Signature};
use rscrypto_fuzz::{FuzzInput, some_or_return};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let public_bytes: [u8; 32] = some_or_return!(input.bytes());
    let signature_bytes: [u8; 64] = some_or_return!(input.bytes());
    let message = input.rest();

    let public = Ed25519PublicKey::from_bytes(public_bytes);
    let signature = Ed25519Signature::from_bytes(signature_bytes);
    let ours_ok = public.verify(message, &signature).is_ok();

    let oracle_signature = ed25519_dalek::Signature::from_bytes(&signature_bytes);
    let oracle_ok = ed25519_dalek::VerifyingKey::from_bytes(&public_bytes)
        .and_then(|verifying_key| verifying_key.verify_strict(message, &oracle_signature))
        .is_ok();

    assert_eq!(ours_ok, oracle_ok, "ed25519 strict verify mismatch");
});
