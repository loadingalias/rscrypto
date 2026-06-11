use p256::ecdsa::{Signature as P256OracleSignature, VerifyingKey as P256OracleVerifyingKey, signature::Verifier as _};
use p384::ecdsa::{Signature as P384OracleSignature, VerifyingKey as P384OracleVerifyingKey};
use rscrypto::{
  EcdsaP256PublicKey, EcdsaP256Signature, EcdsaP384PublicKey, EcdsaP384Signature, EcdsaError,
};
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
  let public_bytes: [u8; EcdsaP256PublicKey::SEC1_LENGTH] = some_or_return!(input.bytes());
  let signature_bytes: [u8; EcdsaP256Signature::LENGTH] = some_or_return!(input.bytes());
  let (parser_material, message) = some_or_return!(input.split_rest());

  let _ = EcdsaP256PublicKey::from_spki_der(parser_material);
  let _ = EcdsaP256Signature::from_der(parser_material);

  let ours_ok = match (
    EcdsaP256PublicKey::from_sec1_bytes(&public_bytes),
    EcdsaP256Signature::from_bytes(signature_bytes),
  ) {
    (Ok(public), Ok(signature)) => public.verify(message, &signature).is_ok(),
    (Err(EcdsaError::InvalidPublicKey), _) | (_, Err(EcdsaError::InvalidSignature)) => false,
    (Err(_), _) | (_, Err(_)) => false,
  };

  let oracle_ok = match (
    P256OracleVerifyingKey::from_sec1_bytes(&public_bytes),
    P256OracleSignature::from_slice(&signature_bytes),
  ) {
    (Ok(public), Ok(signature)) => public.verify(message, &signature).is_ok(),
    _ => false,
  };

  assert_eq!(ours_ok, oracle_ok, "P-256 ECDSA verify mismatch");
}

fn run_p384(input: &mut FuzzInput<'_>) {
  let public_bytes: [u8; EcdsaP384PublicKey::SEC1_LENGTH] = some_or_return!(input.bytes());
  let signature_bytes: [u8; EcdsaP384Signature::LENGTH] = some_or_return!(input.bytes());
  let (parser_material, message) = some_or_return!(input.split_rest());

  let _ = EcdsaP384PublicKey::from_spki_der(parser_material);
  let _ = EcdsaP384Signature::from_der(parser_material);

  let ours_ok = match (
    EcdsaP384PublicKey::from_sec1_bytes(&public_bytes),
    EcdsaP384Signature::from_bytes(signature_bytes),
  ) {
    (Ok(public), Ok(signature)) => public.verify(message, &signature).is_ok(),
    (Err(EcdsaError::InvalidPublicKey), _) | (_, Err(EcdsaError::InvalidSignature)) => false,
    (Err(_), _) | (_, Err(_)) => false,
  };

  let oracle_ok = match (
    P384OracleVerifyingKey::from_sec1_bytes(&public_bytes),
    P384OracleSignature::from_slice(&signature_bytes),
  ) {
    (Ok(public), Ok(signature)) => public.verify(message, &signature).is_ok(),
    _ => false,
  };

  assert_eq!(ours_ok, oracle_ok, "P-384 ECDSA verify mismatch");
}
