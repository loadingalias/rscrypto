use rscrypto::Ed25519Keypair;

fn main() -> Result<(), Box<dyn core::error::Error>> {
  let message = b"message to authenticate";
  let keypair = Ed25519Keypair::try_generate()?;
  let signature = keypair.sign(message);

  keypair.public_key().verify(message, &signature)?;

  println!("Ed25519 signature verified");
  Ok(())
}
