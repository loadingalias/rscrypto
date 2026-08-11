//! Password-record generation with a caller-owned entropy authority.

#![cfg(all(feature = "phc-strings", any(feature = "argon2", feature = "scrypt")))]

use core::cell::Cell;

use rscrypto::{PasswordHashError, PasswordStatus};

const PASSWORD: &[u8] = b"correct horse battery staple";
const SALT: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const SALT_B64: &str = "AAECAwQFBgcICQoLDA0ODw";

#[cfg(feature = "argon2")]
#[test]
fn argon2_hashes_with_one_caller_entropy_fill() {
  use rscrypto::{Argon2Params, Argon2idPassword};

  let passwords = Argon2idPassword::new(Argon2Params::new(32, 2, 1).unwrap()).unwrap();
  let fills = Cell::new(0usize);
  let record = passwords
    .hash_password_with(PASSWORD, |salt| {
      fills.set(fills.get().strict_add(1));
      assert_eq!(salt.len(), SALT.len());
      salt.copy_from_slice(&SALT);
      Ok::<(), &'static str>(())
    })
    .unwrap();

  assert_eq!(fills.get(), 1);
  assert_eq!(record.split('$').nth(4), Some(SALT_B64));
  assert_eq!(
    passwords.verify_password(PASSWORD, &record),
    Ok(PasswordStatus::Current)
  );
}

#[cfg(feature = "scrypt")]
#[test]
fn scrypt_hashes_with_one_caller_entropy_fill() {
  use rscrypto::{ScryptParams, ScryptPassword};

  let passwords = ScryptPassword::new(ScryptParams::new(4, 1, 1).unwrap()).unwrap();
  let fills = Cell::new(0usize);
  let record = passwords
    .hash_password_with(PASSWORD, |salt| {
      fills.set(fills.get().strict_add(1));
      assert_eq!(salt.len(), SALT.len());
      salt.copy_from_slice(&SALT);
      Ok::<(), &'static str>(())
    })
    .unwrap();

  assert_eq!(fills.get(), 1);
  assert_eq!(record.split('$').nth(3), Some(SALT_B64));
  assert_eq!(
    passwords.verify_password(PASSWORD, &record),
    Ok(PasswordStatus::Current)
  );
}

#[cfg(feature = "argon2")]
#[test]
fn argon2_preserves_entropy_errors_without_hashing() {
  use rscrypto::{Argon2Error, Argon2Params, Argon2idPassword};

  let passwords = Argon2idPassword::new(Argon2Params::new(32, 2, 1).unwrap()).unwrap();
  let error = passwords
    .hash_password_with(PASSWORD, |_| Err("entropy unavailable"))
    .unwrap_err();

  assert_eq!(
    error,
    PasswordHashError::<_, Argon2Error>::Entropy("entropy unavailable")
  );
}

#[cfg(feature = "scrypt")]
#[test]
fn scrypt_preserves_entropy_errors_without_hashing() {
  use rscrypto::{ScryptError, ScryptParams, ScryptPassword};

  let passwords = ScryptPassword::new(ScryptParams::new(4, 1, 1).unwrap()).unwrap();
  let error = passwords
    .hash_password_with(PASSWORD, |_| Err("entropy unavailable"))
    .unwrap_err();

  assert_eq!(
    error,
    PasswordHashError::<_, ScryptError>::Entropy("entropy unavailable")
  );
}
