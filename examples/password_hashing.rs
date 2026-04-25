//! Password hashing with Argon2id and scrypt.
//!
//! Run with:
//!
//! ```text
//! cargo run --example password_hashing --features password-hashing
//! ```

use rscrypto::{Argon2Params, Argon2id, Scrypt, ScryptParams};

fn main() -> Result<(), Box<dyn core::error::Error>> {
  let password = b"correct horse battery staple";
  let salt = b"random-salt-1234";

  let argon2 = Argon2Params::new()
    .memory_cost_kib(32)
    .time_cost(1)
    .parallelism(1)
    .output_len(32)
    .build()?;
  let argon2_phc = Argon2id::hash_string_with_salt(&argon2, password, salt)?;
  assert!(Argon2id::verify_string(password, &argon2_phc).is_ok());
  assert!(Argon2id::verify_string(b"wrong password", &argon2_phc).is_err());

  let scrypt = ScryptParams::new().log_n(4).r(1).p(1).output_len(32).build()?;
  let scrypt_phc = Scrypt::hash_string_with_salt(&scrypt, password, salt)?;
  assert!(Scrypt::verify_string(password, &scrypt_phc).is_ok());
  assert!(Scrypt::verify_string(b"wrong password", &scrypt_phc).is_err());

  println!("{argon2_phc}");
  println!("{scrypt_phc}");

  Ok(())
}
