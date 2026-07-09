//! Run with: `cargo run --example signatures --features ed25519,ecdsa-p256,getrandom`

use rscrypto::{EcdsaP256Keypair, Ed25519Keypair};

fn main() -> Result<(), Box<dyn core::error::Error>> {
  let message = b"message to authenticate";

  let ed25519 = Ed25519Keypair::try_generate()?;
  let ed25519_signature = ed25519.sign(message);
  ed25519.public_key().verify(message, &ed25519_signature)?;

  let ecdsa = EcdsaP256Keypair::try_generate()?;
  let ecdsa_signature = ecdsa.try_sign(message)?;
  ecdsa.public_key().verify(message, &ecdsa_signature)?;

  println!("Ed25519 and ECDSA P-256 signatures verified");
  Ok(())
}
