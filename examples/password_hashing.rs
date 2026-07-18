//! Password hashing with Argon2id and scrypt.
//!
//! Run with:
//!
//! ```text
//! cargo run --example password_hashing --features password-hashing,getrandom
//! ```

use rscrypto::{Argon2idPassword, ScryptPassword};

fn main() -> Result<(), Box<dyn core::error::Error>> {
  let password = b"correct horse battery staple";

  let argon2 = Argon2idPassword::default();
  let argon2_phc = argon2.hash_password(password)?;
  assert!(argon2.verify_password(password, &argon2_phc).is_ok());
  assert!(argon2.verify_password(b"wrong password", &argon2_phc).is_err());

  let scrypt = ScryptPassword::default();
  let scrypt_phc = scrypt.hash_password(password)?;
  assert!(scrypt.verify_password(password, &scrypt_phc).is_ok());
  assert!(scrypt.verify_password(b"wrong password", &scrypt_phc).is_err());

  println!("{argon2_phc}");
  println!("{scrypt_phc}");

  Ok(())
}
