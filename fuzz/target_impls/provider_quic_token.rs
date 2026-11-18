use rscrypto_rustls::quic::AddressTokenKey;
use rustls::crypto::{GetRandomFailed, SecureRandom};

/// Fixture entropy: every request is filled with one byte.
#[derive(Debug)]
struct Fixed(u8);

impl SecureRandom for Fixed {
  fn fill(&self, output: &mut [u8]) -> Result<(), GetRandomFailed> {
    output.fill(self.0);
    Ok(())
  }
}

/// Open arbitrary address tokens, then check authentication failures on real
/// seals.
pub(super) fn run(data: &[u8]) {
  if data.len() > 4144 {
    return;
  }
  const AAD: &[u8] = b"fuzz address context";
  let key = AddressTokenKey::generate(&Fixed(0x42)).expect("fixed fixture key");
  let mut bytes = data.to_vec();
  let _opened = key.open(&mut bytes, AAD);
  if data.len() <= 4096 {
    let mut token = data.to_vec();
    key.seal(&Fixed(0x13), &mut token, AAD).expect("bounded seal");
    let nonce = &token[token.len().strict_sub(16)..];
    assert_eq!(nonce, [0x13; 16], "the token ends with its entropy-drawn nonce");
    let mut bad = token.clone();
    bad[0] ^= 1;
    key.open(&mut bad, AAD).expect_err("tampered token");
    let mut other = token.clone();
    key.open(&mut other, b"other context").expect_err("wrong context");
    assert_eq!(key.open(&mut token, AAD).expect("valid token"), data);
  }
}
