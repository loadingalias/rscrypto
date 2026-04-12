#![no_main]

use libfuzzer_sys::fuzz_target;
use rscrypto::{XChaCha20Poly1305, XChaCha20Poly1305Key, aead::Nonce192};
use rscrypto_fuzz::{FuzzInput, assert_aead_forgery, assert_aead_roundtrip, some_or_return};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let key_bytes: [u8; 32] = some_or_return!(input.bytes());
    let nonce_bytes: [u8; 24] = some_or_return!(input.bytes());
    let control: u8 = some_or_return!(input.byte());
    let (aad, plaintext) = some_or_return!(input.split_rest());

    let cipher = XChaCha20Poly1305::new(&XChaCha20Poly1305Key::from_bytes(key_bytes));
    let nonce = Nonce192::from_bytes(nonce_bytes);

    assert_aead_roundtrip(&cipher, &nonce, aad, plaintext);
    assert_aead_forgery(&cipher, &nonce, aad, plaintext, control);

    // Differential: rscrypto ↔ chacha20poly1305 crate (XChaCha variant)
    {
        use chacha20poly1305::aead::{Aead as _, KeyInit, Payload};
        let oracle = chacha20poly1305::XChaCha20Poly1305::new_from_slice(&key_bytes).unwrap();
        let on = chacha20poly1305::XNonce::from_slice(&nonce_bytes);

        let mut ct = plaintext.to_vec();
        let tag = cipher.encrypt_in_place(&nonce, aad, &mut ct);
        let mut combined = ct.clone();
        combined.extend_from_slice(tag.as_ref());
        let pt = oracle.decrypt(on, Payload { msg: &combined, aad }).unwrap();
        assert_eq!(pt, plaintext, "oracle failed to decrypt our ciphertext");

        let oct = oracle.encrypt(on, Payload { msg: plaintext, aad }).unwrap();
        let (body, otag) = oct.split_at(oct.len().strict_sub(16));
        let mut buf = body.to_vec();
        cipher
            .decrypt_in_place(
                &nonce,
                aad,
                &mut buf,
                &XChaCha20Poly1305::tag_from_slice(otag).unwrap(),
            )
            .expect("we failed to decrypt oracle ciphertext");
        assert_eq!(buf, plaintext, "decrypt mismatch on oracle ciphertext");
    }
});
