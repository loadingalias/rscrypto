//! Run with: `cargo run --example mlkem_encapsulation --features ml-kem,getrandom`

use rscrypto::{Kem, MlKem768};

fn main() -> Result<(), Box<dyn core::error::Error>> {
  let (encapsulation_key, decapsulation_key) = MlKem768::try_generate_keypair()?;
  let (ciphertext, shared_secret) = MlKem768::try_encapsulate(&encapsulation_key)?;
  let decapsulated = MlKem768::decapsulate(&decapsulation_key, &ciphertext)?;

  assert_eq!(shared_secret, decapsulated);
  println!(
    "ML-KEM-768 encapsulated {} shared-secret bytes",
    shared_secret.as_bytes().len()
  );
  Ok(())
}
