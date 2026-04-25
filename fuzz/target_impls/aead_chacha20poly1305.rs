use libfuzzer_sys::fuzz_target;
use rscrypto::{ChaCha20Poly1305, ChaCha20Poly1305Key, aead::Nonce96};
use rscrypto_fuzz::{
    FuzzInput, assert_aead_against_oracle, assert_aead_forgery, assert_aead_roundtrip,
    some_or_return,
};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let key_bytes: [u8; 32] = some_or_return!(input.bytes());
    let nonce_bytes: [u8; 12] = some_or_return!(input.bytes());
    let control: u8 = some_or_return!(input.byte());
    let (aad, plaintext) = some_or_return!(input.split_rest());

    let cipher = ChaCha20Poly1305::new(&ChaCha20Poly1305Key::from_bytes(key_bytes));
    let nonce = Nonce96::from_bytes(nonce_bytes);

    assert_aead_roundtrip(&cipher, &nonce, aad, plaintext);
    assert_aead_forgery(&cipher, &nonce, aad, plaintext, control);

    // Differential: rscrypto ↔ chacha20poly1305 crate.
    use chacha20poly1305::aead::{Aead as _, KeyInit, Payload};
    let oracle = chacha20poly1305::ChaCha20Poly1305::new_from_slice(&key_bytes).unwrap();
    let on = chacha20poly1305::Nonce::from_slice(&nonce_bytes);
    assert_aead_against_oracle(
        &cipher,
        &nonce,
        aad,
        plaintext,
        |pt, aad| oracle.encrypt(on, Payload { msg: pt, aad }).unwrap(),
        |ct, aad| oracle.decrypt(on, Payload { msg: ct, aad }).unwrap(),
    );
});
