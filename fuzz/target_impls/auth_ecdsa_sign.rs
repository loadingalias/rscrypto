use p256::ecdsa::{Signature as P256OracleSignature, VerifyingKey as P256OracleVerifyingKey, signature::Verifier as _};
use p384::ecdsa::{Signature as P384OracleSignature, VerifyingKey as P384OracleVerifyingKey};
use rscrypto::{EcdsaP256Keypair, EcdsaP256SecretKey, EcdsaP384Keypair, EcdsaP384SecretKey};
use rscrypto_fuzz::{FuzzInput, some_or_return};

pub fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let selector = some_or_return!(input.byte());

  if selector & 1 == 0 {
    run_p256(&mut input);
  } else {
    run_p384(&mut input);
  }
}

fn run_p256(input: &mut FuzzInput<'_>) {
  let secret_bytes: [u8; EcdsaP256SecretKey::LENGTH] = some_or_return!(input.bytes());
  let secret = some_or_return!(EcdsaP256SecretKey::from_bytes(secret_bytes).ok());
  let keypair = EcdsaP256Keypair::from_secret_key(secret);
  let message = input.rest();
  let public = keypair.public_key();
  let signature = some_or_return!(keypair.try_sign(message).ok());
  let oracle_public = P256OracleVerifyingKey::from_sec1_bytes(&public.to_sec1_bytes()).expect("derived P-256 public key");
  let oracle_signature = P256OracleSignature::from_slice(signature.as_bytes()).expect("derived P-256 signature");

  assert!(public.verify(message, &signature).is_ok());
  assert!(oracle_public.verify(message, &oracle_signature).is_ok());

  let mut tampered = message.to_vec();
  tampered.push(0x80);
  assert!(public.verify(&tampered, &signature).is_err());
}

fn run_p384(input: &mut FuzzInput<'_>) {
  let secret_bytes: [u8; EcdsaP384SecretKey::LENGTH] = some_or_return!(input.bytes());
  let secret = some_or_return!(EcdsaP384SecretKey::from_bytes(secret_bytes).ok());
  let keypair = EcdsaP384Keypair::from_secret_key(secret);
  let message = input.rest();
  let public = keypair.public_key();
  let signature = some_or_return!(keypair.try_sign(message).ok());
  let oracle_public = P384OracleVerifyingKey::from_sec1_bytes(&public.to_sec1_bytes()).expect("derived P-384 public key");
  let oracle_signature = P384OracleSignature::from_slice(signature.as_bytes()).expect("derived P-384 signature");

  assert!(public.verify(message, &signature).is_ok());
  assert!(oracle_public.verify(message, &oracle_signature).is_ok());

  let mut tampered = message.to_vec();
  tampered.push(0x80);
  assert!(public.verify(&tampered, &signature).is_err());
}
