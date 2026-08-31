use rscrypto::X25519SecretKey;

fn main() -> Result<(), Box<dyn core::error::Error>> {
  let alice = X25519SecretKey::try_generate()?;
  let bob = X25519SecretKey::try_generate()?;

  let alice_shared = alice.diffie_hellman(&bob.public_key())?;
  let bob_shared = bob.diffie_hellman(&alice.public_key())?;

  if !alice_shared.ct_eq(&bob_shared).declassify() {
    return Err(std::io::Error::other("X25519 shared secrets differ").into());
  }

  // A protocol must bind this raw shared secret to its transcript with a KDF.
  println!("X25519 key agreement succeeded");
  Ok(())
}
