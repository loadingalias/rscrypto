use rscrypto::P256EphemeralSecret;

fn main() -> Result<(), Box<dyn core::error::Error>> {
  let alice = P256EphemeralSecret::try_generate()?;
  let bob = P256EphemeralSecret::try_generate()?;
  let alice_public = alice.public_key();
  let bob_public = bob.public_key();

  let alice_shared = alice.diffie_hellman(&bob_public);
  let bob_shared = bob.diffie_hellman(&alice_public);
  if !alice_shared.ct_eq(&bob_shared).declassify() {
    return Err(std::io::Error::other("P-256 shared secrets differ").into());
  }

  // The 32-byte raw x-coordinate is KDF input, not a uniformly random
  // application key. Bind it to the protocol transcript in the caller's KDF.
  // ECDH is unauthenticated: the protocol must also authenticate both public
  // keys or the transcript that contains them.
  println!("P-256 key agreement succeeded");
  Ok(())
}
