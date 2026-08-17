use rscrypto::{Aes128GcmSiv, Aes128GcmSivKey, aead::Nonce96};
use rscrypto_fuzz::{
  FuzzInput, assert_aead_against_oracle, assert_aead_forgery, assert_aead_roundtrip, some_or_return,
};

pub(super) fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let key_bytes: [u8; 16] = some_or_return!(input.bytes());
  let nonce_bytes: [u8; 12] = some_or_return!(input.bytes());
  let control: u8 = some_or_return!(input.byte());
  let (aad, plaintext) = some_or_return!(input.split_rest());

  let cipher = Aes128GcmSiv::new(&Aes128GcmSivKey::from_bytes(key_bytes));
  let nonce = Nonce96::from_bytes(nonce_bytes);

  assert_aead_roundtrip(&cipher, &nonce, aad, plaintext);
  assert_aead_forgery(&cipher, &nonce, aad, plaintext, control);

  // Differential: rscrypto ↔ aes-gcm-siv crate.
  use aes_gcm_siv::aead::{Aead as _, KeyInit, Payload};
  let oracle = aes_gcm_siv::Aes128GcmSiv::new_from_slice(&key_bytes).expect("AES-128-GCM-SIV accepts a 16-byte key");
  let on = aes_gcm_siv::Nonce::from(nonce_bytes);
  assert_aead_against_oracle(
    &cipher,
    &nonce,
    aad,
    plaintext,
    |pt, aad| {
      oracle
        .encrypt(&on, Payload { msg: pt, aad })
        .expect("oracle encryption accepts the fuzz input")
    },
    |ct, aad| {
      oracle
        .decrypt(&on, Payload { msg: ct, aad })
        .expect("oracle must accept the equivalent rscrypto ciphertext")
    },
  );
}
