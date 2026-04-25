use libfuzzer_sys::fuzz_target;
use rscrypto::{Aes256GcmSiv, Aes256GcmSivKey, aead::Nonce96};
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

    let cipher = Aes256GcmSiv::new(&Aes256GcmSivKey::from_bytes(key_bytes));
    let nonce = Nonce96::from_bytes(nonce_bytes);

    assert_aead_roundtrip(&cipher, &nonce, aad, plaintext);
    assert_aead_forgery(&cipher, &nonce, aad, plaintext, control);

    // Differential: rscrypto ↔ aes-gcm-siv crate.
    use aes_gcm_siv::aead::{Aead as _, KeyInit, Payload};
    let oracle = aes_gcm_siv::Aes256GcmSiv::new_from_slice(&key_bytes).unwrap();
    let on = aes_gcm_siv::Nonce::from_slice(&nonce_bytes);
    assert_aead_against_oracle(
        &cipher,
        &nonce,
        aad,
        plaintext,
        |pt, aad| oracle.encrypt(on, Payload { msg: pt, aad }).unwrap(),
        |ct, aad| oracle.decrypt(on, Payload { msg: ct, aad }).unwrap(),
    );
});
