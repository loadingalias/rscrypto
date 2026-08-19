use rscrypto::{Pbkdf2Sha256, Pbkdf2Sha512};
use rscrypto_fuzz::{FuzzInput, some_or_return, split_at_ratio};

pub(super) fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let split: u8 = some_or_return!(input.byte());
  let out_len_byte: u8 = some_or_return!(input.byte());
  let iterations_bytes: [u8; 2] = some_or_return!(input.bytes());
  let data = input.rest();

  let (password, salt) = split_at_ratio(data, split);
  let out_len = usize::from(out_len_byte).rem_euclid(96).strict_add(1);
  let iterations = u32::from(u16::from_le_bytes(iterations_bytes))
    .rem_euclid(64)
    .strict_add(1);

  let mut ours_256 = vec![0u8; out_len];
  let mut ours_256_state = vec![0u8; out_len];
  Pbkdf2Sha256::derive_key_primitive(password, salt, iterations, &mut ours_256)
    .expect("the bounded PBKDF2-SHA-256 request is valid");
  Pbkdf2Sha256::new(password)
    .derive(salt, iterations, &mut ours_256_state)
    .expect("the bounded PBKDF2-SHA-256 request is valid");
  assert_eq!(ours_256, ours_256_state, "pbkdf2-sha256 state reuse mismatch");
  Pbkdf2Sha256::verify_password_primitive(password, salt, iterations, &ours_256)
    .expect("PBKDF2-SHA-256 must verify its own output");
  let mut wrong_256 = ours_256.clone();
  wrong_256[0] ^= 1;
  let _verification_error = Pbkdf2Sha256::verify_password_primitive(password, salt, iterations, &wrong_256)
    .expect_err("PBKDF2-SHA-256 must reject a corrupted output");

  let mut oracle_256 = vec![0u8; out_len];
  pbkdf2::pbkdf2_hmac::<sha2::Sha256>(password, salt, iterations, &mut oracle_256);
  assert_eq!(ours_256, oracle_256, "pbkdf2-sha256 oracle mismatch");

  let mut ours_512 = vec![0u8; out_len];
  let mut ours_512_state = vec![0u8; out_len];
  Pbkdf2Sha512::derive_key_primitive(password, salt, iterations, &mut ours_512)
    .expect("the bounded PBKDF2-SHA-512 request is valid");
  Pbkdf2Sha512::new(password)
    .derive(salt, iterations, &mut ours_512_state)
    .expect("the bounded PBKDF2-SHA-512 request is valid");
  assert_eq!(ours_512, ours_512_state, "pbkdf2-sha512 state reuse mismatch");
  Pbkdf2Sha512::verify_password_primitive(password, salt, iterations, &ours_512)
    .expect("PBKDF2-SHA-512 must verify its own output");
  let mut wrong_512 = ours_512.clone();
  wrong_512[0] ^= 1;
  let _verification_error = Pbkdf2Sha512::verify_password_primitive(password, salt, iterations, &wrong_512)
    .expect_err("PBKDF2-SHA-512 must reject a corrupted output");

  let mut oracle_512 = vec![0u8; out_len];
  pbkdf2::pbkdf2_hmac::<sha2::Sha512>(password, salt, iterations, &mut oracle_512);
  assert_eq!(ours_512, oracle_512, "pbkdf2-sha512 oracle mismatch");
}
