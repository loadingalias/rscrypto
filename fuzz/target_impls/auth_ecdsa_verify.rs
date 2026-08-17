use p256::ecdsa::{
  Signature as P256OracleSignature, SigningKey as P256OracleSigningKey, VerifyingKey as P256OracleVerifyingKey,
  signature::Verifier as P256Verifier,
};
use p384::ecdsa::{
  Signature as P384OracleSignature, SigningKey as P384OracleSigningKey, VerifyingKey as P384OracleVerifyingKey,
  signature::Verifier as P384Verifier,
};
use rscrypto::{EcdsaError, EcdsaP256PublicKey, EcdsaP256Signature, EcdsaP384PublicKey, EcdsaP384Signature};
use rscrypto_fuzz::{FuzzInput, some_or_return};

fn array_from_slice<const N: usize>(bytes: &[u8]) -> [u8; N] {
  bytes.try_into().expect("oracle ECDSA signature length must match")
}

pub(super) fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let selector = some_or_return!(input.byte());
  let derive_public = selector & 2 != 0;

  if selector & 1 == 0 {
    run_p256(&mut input, derive_public);
  } else {
    run_p384(&mut input, derive_public);
  }
}

fn run_p256(input: &mut FuzzInput<'_>, derive_public: bool) {
  let public_bytes: [u8; EcdsaP256PublicKey::SEC1_LENGTH] = if derive_public {
    let secret_bytes: [u8; 32] = some_or_return!(input.bytes());
    let signing_key = some_or_return!(P256OracleSigningKey::from_slice(&secret_bytes).ok());
    let verifying_key = signing_key.verifying_key();
    let spki = p256::pkcs8::EncodePublicKey::to_public_key_der(verifying_key)
      .expect("oracle P-256 public key must encode as SPKI");
    let ours = EcdsaP256PublicKey::from_spki_der(spki.as_bytes()).expect("rscrypto must parse oracle P-256 SPKI");
    assert_eq!(
      ours.to_sec1_bytes().as_slice(),
      verifying_key.to_sec1_bytes().as_ref(),
      "P-256 SPKI round-trip mismatch"
    );
    some_or_return!(verifying_key.to_sec1_bytes().as_ref().try_into().ok())
  } else {
    some_or_return!(input.bytes())
  };
  let signature_bytes: [u8; EcdsaP256Signature::LENGTH] = some_or_return!(input.bytes());
  let (parser_material, message) = some_or_return!(input.split_rest());

  let ours_spki = EcdsaP256PublicKey::from_spki_der(parser_material)
    .ok()
    .map(|public| public.to_sec1_bytes());
  let oracle_spki = <P256OracleVerifyingKey as p256::pkcs8::DecodePublicKey>::from_public_key_der(parser_material)
    .ok()
    .map(|public| public.to_sec1_bytes());
  if let Some(ours) = ours_spki {
    assert_eq!(
      oracle_spki.as_deref(),
      Some(ours.as_slice()),
      "P-256 SPKI parser accepted a key rejected by the oracle"
    );
  }

  let ours_der = EcdsaP256Signature::from_der(parser_material)
    .ok()
    .map(EcdsaP256Signature::to_bytes);
  let oracle_der = P256OracleSignature::from_der(parser_material)
    .ok()
    .map(|signature| array_from_slice(signature.to_bytes().as_ref()));
  assert_eq!(ours_der, oracle_der, "P-256 ECDSA DER parser mismatch");

  let ours_public = EcdsaP256PublicKey::from_sec1_bytes(&public_bytes);
  let oracle_public = P256OracleVerifyingKey::from_sec1_bytes(&public_bytes);
  assert_eq!(
    ours_public.is_ok(),
    oracle_public.is_ok(),
    "P-256 uncompressed SEC1 parser mismatch"
  );

  let ours_signature = EcdsaP256Signature::from_bytes(signature_bytes);
  let oracle_signature = P256OracleSignature::from_slice(&signature_bytes);
  assert_eq!(
    ours_signature.as_ref().ok().map(|signature| signature.to_bytes()),
    oracle_signature
      .as_ref()
      .ok()
      .map(|signature| array_from_slice(signature.to_bytes().as_ref())),
    "P-256 raw signature parser mismatch"
  );
  if let Ok(signature) = &oracle_signature {
    let der = signature.to_der();
    assert_eq!(
      EcdsaP256Signature::from_der(der.as_bytes())
        .ok()
        .map(EcdsaP256Signature::to_bytes),
      Some(array_from_slice(signature.to_bytes().as_ref())),
      "P-256 DER signature round-trip mismatch"
    );
  }

  let ours_ok = match (ours_public, ours_signature) {
    (Ok(public), Ok(signature)) => public.verify(message, &signature).is_ok(),
    (Err(EcdsaError::InvalidPublicKey), _) | (_, Err(EcdsaError::InvalidSignature)) => false,
    (Err(_), _) | (_, Err(_)) => false,
  };

  let oracle_ok = match (oracle_public, oracle_signature) {
    (Ok(public), Ok(signature)) => P256Verifier::verify(&public, message, &signature).is_ok(),
    _ => false,
  };

  assert_eq!(ours_ok, oracle_ok, "P-256 ECDSA verify mismatch");
}

fn run_p384(input: &mut FuzzInput<'_>, derive_public: bool) {
  let public_bytes: [u8; EcdsaP384PublicKey::SEC1_LENGTH] = if derive_public {
    let secret_bytes: [u8; 48] = some_or_return!(input.bytes());
    let signing_key = some_or_return!(P384OracleSigningKey::from_slice(&secret_bytes).ok());
    let verifying_key = signing_key.verifying_key();
    let public_key = p384::PublicKey::from_sec1_bytes(verifying_key.to_sec1_bytes().as_ref())
      .expect("oracle P-384 verifying key must convert to a public key");
    let spki = p384::pkcs8::EncodePublicKey::to_public_key_der(&public_key)
      .expect("oracle P-384 public key must encode as SPKI");
    let ours = EcdsaP384PublicKey::from_spki_der(spki.as_bytes()).expect("rscrypto must parse oracle P-384 SPKI");
    assert_eq!(
      ours.to_sec1_bytes().as_slice(),
      verifying_key.to_sec1_bytes().as_ref(),
      "P-384 SPKI round-trip mismatch"
    );
    some_or_return!(verifying_key.to_sec1_bytes().as_ref().try_into().ok())
  } else {
    some_or_return!(input.bytes())
  };
  let signature_bytes: [u8; EcdsaP384Signature::LENGTH] = some_or_return!(input.bytes());
  let (parser_material, message) = some_or_return!(input.split_rest());

  let ours_spki = EcdsaP384PublicKey::from_spki_der(parser_material)
    .ok()
    .map(|public| public.to_sec1_bytes());
  let oracle_spki = <P384OracleVerifyingKey as p384::pkcs8::DecodePublicKey>::from_public_key_der(parser_material)
    .ok()
    .map(|public| public.to_sec1_bytes());
  if let Some(ours) = ours_spki {
    assert_eq!(
      oracle_spki.as_deref(),
      Some(ours.as_slice()),
      "P-384 SPKI parser accepted a key rejected by the oracle"
    );
  }

  let ours_der = EcdsaP384Signature::from_der(parser_material)
    .ok()
    .map(EcdsaP384Signature::to_bytes);
  let oracle_der = P384OracleSignature::from_der(parser_material)
    .ok()
    .map(|signature| array_from_slice(signature.to_bytes().as_ref()));
  assert_eq!(ours_der, oracle_der, "P-384 ECDSA DER parser mismatch");

  let ours_public = EcdsaP384PublicKey::from_sec1_bytes(&public_bytes);
  let oracle_public = P384OracleVerifyingKey::from_sec1_bytes(&public_bytes);
  assert_eq!(
    ours_public.is_ok(),
    oracle_public.is_ok(),
    "P-384 uncompressed SEC1 parser mismatch"
  );

  let ours_signature = EcdsaP384Signature::from_bytes(signature_bytes);
  let oracle_signature = P384OracleSignature::from_slice(&signature_bytes);
  assert_eq!(
    ours_signature.as_ref().ok().map(|signature| signature.to_bytes()),
    oracle_signature
      .as_ref()
      .ok()
      .map(|signature| array_from_slice(signature.to_bytes().as_ref())),
    "P-384 raw signature parser mismatch"
  );
  if let Ok(signature) = &oracle_signature {
    let der = signature.to_der();
    assert_eq!(
      EcdsaP384Signature::from_der(der.as_bytes())
        .ok()
        .map(EcdsaP384Signature::to_bytes),
      Some(array_from_slice(signature.to_bytes().as_ref())),
      "P-384 DER signature round-trip mismatch"
    );
  }

  let ours_ok = match (ours_public, ours_signature) {
    (Ok(public), Ok(signature)) => public.verify(message, &signature).is_ok(),
    (Err(EcdsaError::InvalidPublicKey), _) | (_, Err(EcdsaError::InvalidSignature)) => false,
    (Err(_), _) | (_, Err(_)) => false,
  };

  let oracle_ok = match (oracle_public, oracle_signature) {
    (Ok(public), Ok(signature)) => P384Verifier::verify(&public, message, &signature).is_ok(),
    _ => false,
  };

  assert_eq!(ours_ok, oracle_ok, "P-384 ECDSA verify mismatch");
}
