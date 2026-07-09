//! Run with: `cargo run --example aead_seal_open --features chacha20poly1305,getrandom`

use rscrypto::{Aead, ChaCha20Poly1305, ChaCha20Poly1305Key};

fn main() -> Result<(), Box<dyn core::error::Error>> {
  let key = ChaCha20Poly1305Key::try_random()?;
  let cipher = ChaCha20Poly1305::new(&key);

  let aad = b"tenant=alpha;record=42";
  let plaintext = b"confidential payload";
  let (nonce, sealed) = cipher.seal_random_to_vec(aad, plaintext)?;
  let opened = cipher.decrypt_to_vec(&nonce, aad, &sealed)?;

  assert_eq!(opened, plaintext);
  println!("ChaCha20-Poly1305 sealed {} bytes", plaintext.len());
  Ok(())
}
