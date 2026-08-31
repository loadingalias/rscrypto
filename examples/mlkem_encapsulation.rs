use rscrypto::{Kem, MlKem768};

fn main() -> Result<(), Box<dyn core::error::Error>> {
  let (encapsulation_key, decapsulation_key) = MlKem768::try_generate_keypair()?;
  let (ciphertext, shared_secret) = MlKem768::try_encapsulate(&encapsulation_key)?;
  let decapsulated = MlKem768::decapsulate(&decapsulation_key, &ciphertext)?;

  if !shared_secret.ct_eq(&decapsulated).declassify() {
    return Err(std::io::Error::other("ML-KEM shared secrets differ").into());
  }
  println!(
    "ML-KEM-768 encapsulated {} shared-secret bytes",
    shared_secret.as_bytes().len()
  );
  Ok(())
}
