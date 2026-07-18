// Hostile PHC input reaches only the bounded public password verifiers.
//
// The verification profiles are deliberately tiny so a fuzzer-generated
// canonical record remains cheap while malformed or over-budget inputs
// exercise the parser and approval boundary at full throughput.

use rscrypto::{
  Argon2Params, Argon2idPassword, ScryptParams, ScryptPassword,
};

pub fn run(data: &[u8]) {
  let split = data.len() / 2;
  let (password, encoded_bytes) = data.split_at(split);
  let encoded = String::from_utf8_lossy(encoded_bytes);

  let argon2 = Argon2idPassword::new(
    Argon2Params::new(8, 1, 1).expect("fixed Argon2 fuzz profile is valid"),
  )
  .expect("fixed Argon2 fuzz profile fits the target");
  let scrypt = ScryptPassword::new(
    ScryptParams::new(1, 1, 1).expect("fixed scrypt fuzz profile is valid"),
  )
  .expect("fixed scrypt fuzz profile fits the target");

  let _ = argon2.verify_password(password, &encoded);
  let _ = scrypt.verify_password(password, &encoded);
}
