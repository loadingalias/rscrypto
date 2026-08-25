use aes_siv::{KeyInit as _, siv::Aes128Siv};
use rscrypto::{AesSivCmac256, AesSivCmac256Key, AesSivCmac256Nonce, aead::OpenError};
use rscrypto_fuzz::{FuzzInput, some_or_return, split_at_ratio};

pub(super) fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let key_bytes: [u8; 32] = some_or_return!(input.bytes());
  let nonce_control = some_or_return!(input.byte());
  let aad_control = some_or_return!(input.byte());
  let corruption_control = some_or_return!(input.byte());
  let rest = input.rest();
  if rest.is_empty() {
    return;
  }

  let nonce_bound = rest.len().min(64);
  let nonce_len = usize::from(nonce_control).rem_euclid(nonce_bound).strict_add(1);
  let (nonce_bytes, payload) = rest.split_at(nonce_len);
  let (aad, plaintext) = split_at_ratio(payload, aad_control);

  let key = AesSivCmac256Key::from_bytes(key_bytes);
  let cipher = AesSivCmac256::new(&key);
  let nonce = AesSivCmac256Nonce::try_from(nonce_bytes).expect("constructed nonce is non-empty");

  let mut combined = vec![0u8; plaintext.len().strict_add(AesSivCmac256::TAG_SIZE)];
  cipher
    .seal(nonce, aad, plaintext, &mut combined)
    .expect("fuzz output has the exact required length");

  let mut opened = vec![0xA5; plaintext.len()];
  cipher
    .open(nonce, aad, &combined, &mut opened)
    .expect("fresh ciphertext must authenticate");
  assert_eq!(opened, plaintext, "roundtrip plaintext mismatch");

  let mut oracle = Aes128Siv::new((&key_bytes).into());
  let expected = oracle
    .encrypt([aad, nonce_bytes], plaintext)
    .expect("two-header oracle input is within its component bound");
  assert_eq!(combined, expected, "rscrypto diverged from the RustCrypto oracle");

  let index = usize::from(corruption_control).rem_euclid(combined.len());
  combined[index] ^= 1;
  opened.fill(0xA5);
  assert_eq!(
    cipher.open(nonce, aad, &combined, &mut opened),
    Err(OpenError::verification()),
    "corrupted ciphertext must be rejected opaquely"
  );
  assert!(
    opened.iter().all(|&byte| byte == 0),
    "failed open must clear the complete output"
  );
}
