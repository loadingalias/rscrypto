use rscrypto::{Aes128Gcm, Aes128GcmKey, aead::Nonce96};
use rscrypto_fuzz::{
  FuzzInput, assert_aead_against_oracle, assert_aead_forgery, assert_aead_roundtrip, some_or_return,
};

pub fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let key_bytes: [u8; 16] = some_or_return!(input.bytes());
  let nonce_bytes: [u8; 12] = some_or_return!(input.bytes());
  let control: u8 = some_or_return!(input.byte());
  let (aad, plaintext) = some_or_return!(input.split_rest());

  let cipher = Aes128Gcm::new(&Aes128GcmKey::from_bytes(key_bytes));
  let nonce = Nonce96::from_bytes(nonce_bytes);

  assert_aead_roundtrip(&cipher, &nonce, aad, plaintext);
  assert_aead_forgery(&cipher, &nonce, aad, plaintext, control);

  // Differential: rscrypto ↔ aes-gcm crate.
  use aes_gcm::aead::{Aead as _, KeyInit, Payload};
  let oracle = aes_gcm::Aes128Gcm::new_from_slice(&key_bytes).unwrap();
  let on = aes_gcm::Nonce::from(nonce_bytes);
  assert_aead_against_oracle(
    &cipher,
    &nonce,
    aad,
    plaintext,
    |pt, aad| oracle.encrypt(&on, Payload { msg: pt, aad }).unwrap(),
    |ct, aad| oracle.decrypt(&on, Payload { msg: ct, aad }).unwrap(),
  );
}
