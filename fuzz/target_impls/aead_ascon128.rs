use libfuzzer_sys::fuzz_target;
use rscrypto::{AsconAead128, AsconAead128Key, aead::Nonce128};
use rscrypto_fuzz::{
    FuzzInput, assert_aead_against_oracle, assert_aead_forgery, assert_aead_roundtrip,
    some_or_return,
};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let key_bytes: [u8; 16] = some_or_return!(input.bytes());
    let nonce_bytes: [u8; 16] = some_or_return!(input.bytes());
    let control: u8 = some_or_return!(input.byte());
    let (aad, plaintext) = some_or_return!(input.split_rest());

    let cipher = AsconAead128::new(&AsconAead128Key::from_bytes(key_bytes));
    let nonce = Nonce128::from_bytes(nonce_bytes);

    assert_aead_roundtrip(&cipher, &nonce, aad, plaintext);
    assert_aead_forgery(&cipher, &nonce, aad, plaintext, control);

    // Differential: rscrypto ↔ ascon-aead crate.
    use ascon_aead::aead::{Aead as _, KeyInit, Payload, generic_array::GenericArray};
    let oracle = ascon_aead::AsconAead128::new_from_slice(&key_bytes).unwrap();
    let on = GenericArray::from_slice(&nonce_bytes);
    assert_aead_against_oracle(
        &cipher,
        &nonce,
        aad,
        plaintext,
        |pt, aad| oracle.encrypt(on, Payload { msg: pt, aad }).unwrap(),
        |ct, aad| oracle.decrypt(on, Payload { msg: ct, aad }).unwrap(),
    );
});
