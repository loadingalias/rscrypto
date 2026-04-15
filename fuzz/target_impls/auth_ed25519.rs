
use libfuzzer_sys::fuzz_target;
use rscrypto::{Ed25519SecretKey, Ed25519Signature};
use rscrypto_fuzz::{FuzzInput, some_or_return};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let key_bytes: [u8; 32] = some_or_return!(input.bytes());
    let message = input.rest();

    let secret = Ed25519SecretKey::from_bytes(key_bytes);
    let public = secret.public_key();
    let sig = secret.sign(message);

    // Property: sign → verify roundtrip
    public
        .verify(message, &sig)
        .expect("roundtrip: verify must succeed");

    // Property: random signature must be rejected
    let bad_sig = Ed25519Signature::from_bytes([0xAB; 64]);
    assert!(
        public.verify(message, &bad_sig).is_err(),
        "accepted garbage signature"
    );

    // Property: wrong message must be rejected
    if !message.is_empty() {
        let mut wrong = message.to_vec();
        wrong[0] ^= 1;
        assert!(
            public.verify(&wrong, &sig).is_err(),
            "accepted signature for wrong message"
        );
    }

    // Differential: rscrypto ↔ ed25519-dalek
    {
        use ed25519_dalek::{Signer, Verifier};

        let dalek_sk = ed25519_dalek::SigningKey::from_bytes(&key_bytes);
        let dalek_vk = dalek_sk.verifying_key();

        // Public keys must match
        assert_eq!(
            public.to_bytes(),
            dalek_vk.to_bytes(),
            "public key mismatch"
        );

        // Signatures must match (Ed25519 is deterministic)
        let dalek_sig = dalek_sk.sign(message);
        assert_eq!(
            sig.to_bytes(),
            dalek_sig.to_bytes(),
            "signature mismatch"
        );

        // Cross-verify: our sig with their verifier
        let our_as_dalek = ed25519_dalek::Signature::from_bytes(&sig.to_bytes());
        dalek_vk
            .verify(message, &our_as_dalek)
            .expect("dalek rejected our signature");

        // Cross-verify: their sig with our verifier
        let theirs_as_ours = Ed25519Signature::from_bytes(dalek_sig.to_bytes());
        public
            .verify(message, &theirs_as_ours)
            .expect("we rejected dalek signature");
    }
});
