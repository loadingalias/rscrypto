// Hostile PHC input reaches only the bounded public password verifiers.
//
// The verification profiles are deliberately tiny so a fuzzer-generated
// canonical record remains cheap while malformed or over-budget inputs
// exercise the parser and approval boundary at full throughput.

use rscrypto::{Argon2Block, Argon2Context, Argon2Params, Argon2idPassword, ScryptBlock, ScryptParams, ScryptPassword};

pub(super) fn run(data: &[u8]) {
  let split = data.len() / 2;
  let (password, encoded_bytes) = data.split_at(split);
  let encoded = String::from_utf8_lossy(encoded_bytes);

  let argon2 = Argon2idPassword::new(Argon2Params::new(8, 1, 1).expect("fixed Argon2 fuzz profile is valid"))
    .expect("fixed Argon2 fuzz profile fits the target");
  let scrypt = ScryptPassword::new(ScryptParams::new(1, 1, 1).expect("fixed scrypt fuzz profile is valid"))
    .expect("fixed scrypt fuzz profile fits the target");

  let mut argon2_memory = [const { Argon2Block::ZERO }; 8];
  let mut scrypt_memory = [const { ScryptBlock::ZERO }; 10];
  assert_eq!(
    argon2.verify_password(password, &encoded),
    argon2.verify_password_with_memory(password, &encoded, &mut argon2_memory),
  );
  let context = Argon2Context::new(password, encoded_bytes);
  assert_eq!(
    argon2.verify_password_with_context(password, &encoded, context),
    argon2.verify_password_with_context_and_memory(password, &encoded, context, &mut argon2_memory),
  );
  assert_eq!(
    scrypt.verify_password(password, &encoded),
    scrypt.verify_password_with_memory(password, &encoded, &mut scrypt_memory),
  );
}
