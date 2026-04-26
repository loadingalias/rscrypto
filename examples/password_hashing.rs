//! Password hashing with Argon2id and scrypt.
//!
//! Run with:
//!
//! ```text
//! cargo run --example password_hashing --features password-hashing,getrandom
//! ```

use rscrypto::{Argon2Params, Argon2VerifyPolicy, Argon2id, Scrypt, ScryptParams, ScryptVerifyPolicy};

fn main() -> Result<(), Box<dyn core::error::Error>> {
  let password = b"correct horse battery staple";

  let argon2 = Argon2Params::new().build()?;
  let argon2_phc = Argon2id::hash_string(&argon2, password)?;
  assert!(Argon2id::verify_string_with_policy(password, &argon2_phc, &Argon2VerifyPolicy::default()).is_ok());
  assert!(Argon2id::verify_string_with_policy(b"wrong password", &argon2_phc, &Argon2VerifyPolicy::default()).is_err());

  let scrypt = ScryptParams::new().build()?;
  let scrypt_phc = Scrypt::hash_string(&scrypt, password)?;
  assert!(Scrypt::verify_string_with_policy(password, &scrypt_phc, &ScryptVerifyPolicy::default()).is_ok());
  assert!(Scrypt::verify_string_with_policy(b"wrong password", &scrypt_phc, &ScryptVerifyPolicy::default()).is_err());

  println!("{argon2_phc}");
  println!("{scrypt_phc}");

  Ok(())
}
