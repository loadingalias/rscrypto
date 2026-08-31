//! Hash and verify a password with the bounded Argon2id policy.

use rscrypto::Argon2idPassword;

fn main() -> Result<(), Box<dyn core::error::Error>> {
  let password = b"correct horse battery staple";

  let policy = Argon2idPassword::default();
  let record = policy.hash_password(password)?;

  policy.verify_password(password, &record)?;
  if policy.verify_password(b"wrong password", &record).is_ok() {
    return Err(std::io::Error::other("Argon2id accepted the wrong password").into());
  }

  println!("{record}");
  Ok(())
}
