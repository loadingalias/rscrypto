use p256::ecdsa::{
  Signature as P256OracleSignature, VerifyingKey as P256OracleVerifyingKey, signature::Verifier as P256Verifier,
};
use p384::ecdsa::{
  Signature as P384OracleSignature, VerifyingKey as P384OracleVerifyingKey, signature::Verifier as P384Verifier,
};
use rscrypto::{
  EcdsaP256Keypair, EcdsaP256PublicKey, EcdsaP256SecretKey, EcdsaP256Signature, EcdsaP384Keypair, EcdsaP384PublicKey,
  EcdsaP384SecretKey, EcdsaP384Signature,
};
use rscrypto_fuzz::{BitMutation, FuzzInput, some_or_return};

pub(super) fn run(data: &[u8]) {
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
  let mutation = some_or_return!(input.bit_mutation());
  let message = input.rest();
  let public = keypair.public_key();
  let signature = some_or_return!(keypair.try_sign(message).ok());
  let oracle_public =
    P256OracleVerifyingKey::from_sec1_bytes(&public.to_sec1_bytes()).expect("derived P-256 public key");
  let oracle_signature = P256OracleSignature::from_slice(signature.as_bytes()).expect("derived P-256 signature");

  public
    .verify(message, &signature)
    .expect("P-256 public key must verify its own signature");
  P256Verifier::verify(&oracle_public, message, &oracle_signature)
    .expect("P-256 oracle must verify the equivalent signature");

  let mut der = [0; EcdsaP256Signature::DER_MAX_LENGTH];
  let encoded = signature.to_der_into(&mut der);
  assert_eq!(
    encoded,
    oracle_signature.to_der().as_bytes(),
    "P-256 DER signature must match the oracle encoding"
  );
  assert_eq!(
    EcdsaP256Signature::from_der(encoded),
    Ok(signature),
    "P-256 DER signature must round-trip"
  );
  exercise_p256_encodings(keypair.secret_key(), &public, &mutation, message);

  let mut tampered = message.to_vec();
  tampered.push(0x80);
  let _verification_error = public
    .verify(&tampered, &signature)
    .expect_err("P-256 verification must reject a tampered message");
}

fn run_p384(input: &mut FuzzInput<'_>) {
  let secret_bytes: [u8; EcdsaP384SecretKey::LENGTH] = some_or_return!(input.bytes());
  let secret = some_or_return!(EcdsaP384SecretKey::from_bytes(secret_bytes).ok());
  let keypair = EcdsaP384Keypair::from_secret_key(secret);
  let mutation = some_or_return!(input.bit_mutation());
  let message = input.rest();
  let public = keypair.public_key();
  let signature = some_or_return!(keypair.try_sign(message).ok());
  let oracle_public =
    P384OracleVerifyingKey::from_sec1_bytes(&public.to_sec1_bytes()).expect("derived P-384 public key");
  let oracle_signature = P384OracleSignature::from_slice(signature.as_bytes()).expect("derived P-384 signature");

  public
    .verify(message, &signature)
    .expect("P-384 public key must verify its own signature");
  P384Verifier::verify(&oracle_public, message, &oracle_signature)
    .expect("P-384 oracle must verify the equivalent signature");

  let mut der = [0; EcdsaP384Signature::DER_MAX_LENGTH];
  let encoded = signature.to_der_into(&mut der);
  assert_eq!(
    encoded,
    oracle_signature.to_der().as_bytes(),
    "P-384 DER signature must match the oracle encoding"
  );
  assert_eq!(
    EcdsaP384Signature::from_der(encoded),
    Ok(signature),
    "P-384 DER signature must round-trip"
  );
  exercise_p384_encodings(keypair.secret_key(), &public, &mutation, message);

  let mut tampered = message.to_vec();
  tampered.push(0x80);
  let _verification_error = public
    .verify(&tampered, &signature)
    .expect_err("P-384 verification must reject a tampered message");
}

/// Exports round-trip, and an accepted input of the export length is the
/// export layout: no other P-256 encoding has that length.
fn exercise_p256_encodings(
  secret: &EcdsaP256SecretKey,
  public: &EcdsaP256PublicKey,
  mutation: &BitMutation,
  message: &[u8],
) {
  let mut pkcs8 = [0; EcdsaP256SecretKey::PKCS8_DER_LENGTH];
  secret.to_pkcs8_der_into(&mut pkcs8);
  let imported = EcdsaP256SecretKey::from_pkcs8_der(&pkcs8).expect("P-256 PKCS #8 export must import");
  assert_eq!(imported.public_key().to_sec1_bytes(), public.to_sec1_bytes());
  assert_eq!(
    EcdsaP256PublicKey::from_spki_der(&public.to_spki_der()).map(|key| key.to_sec1_bytes()),
    Ok(public.to_sec1_bytes()),
    "P-256 SPKI export must import"
  );
  mutation.apply(&mut pkcs8);
  for candidate in [pkcs8.as_slice(), message] {
    if let Ok(parsed) = EcdsaP256SecretKey::from_pkcs8_der(candidate)
      && candidate.len() == EcdsaP256SecretKey::PKCS8_DER_LENGTH
    {
      let mut encoded = [0; EcdsaP256SecretKey::PKCS8_DER_LENGTH];
      parsed.to_pkcs8_der_into(&mut encoded);
      assert_eq!(
        encoded.as_slice(),
        candidate,
        "accepted P-256 PKCS #8 must be canonical"
      );
    }
    // Arbitrary byte lengths and contents must not panic at a public parser.
    let _sec1 = EcdsaP256SecretKey::from_sec1_der(candidate);
    let _spki = EcdsaP256PublicKey::from_spki_der(candidate);
    let _signature = EcdsaP256Signature::from_der(candidate);
  }
}

/// Exports round-trip, and an accepted input of the export length is the
/// export layout: no other P-384 encoding has that length.
fn exercise_p384_encodings(
  secret: &EcdsaP384SecretKey,
  public: &EcdsaP384PublicKey,
  mutation: &BitMutation,
  message: &[u8],
) {
  let mut pkcs8 = [0; EcdsaP384SecretKey::PKCS8_DER_LENGTH];
  secret.to_pkcs8_der_into(&mut pkcs8);
  let imported = EcdsaP384SecretKey::from_pkcs8_der(&pkcs8).expect("P-384 PKCS #8 export must import");
  assert_eq!(imported.public_key().to_sec1_bytes(), public.to_sec1_bytes());
  assert_eq!(
    EcdsaP384PublicKey::from_spki_der(&public.to_spki_der()).map(|key| key.to_sec1_bytes()),
    Ok(public.to_sec1_bytes()),
    "P-384 SPKI export must import"
  );
  mutation.apply(&mut pkcs8);
  for candidate in [pkcs8.as_slice(), message] {
    if let Ok(parsed) = EcdsaP384SecretKey::from_pkcs8_der(candidate)
      && candidate.len() == EcdsaP384SecretKey::PKCS8_DER_LENGTH
    {
      let mut encoded = [0; EcdsaP384SecretKey::PKCS8_DER_LENGTH];
      parsed.to_pkcs8_der_into(&mut encoded);
      assert_eq!(
        encoded.as_slice(),
        candidate,
        "accepted P-384 PKCS #8 must be canonical"
      );
    }
    // Arbitrary byte lengths and contents must not panic at a public parser.
    let _sec1 = EcdsaP384SecretKey::from_sec1_der(candidate);
    let _spki = EcdsaP384PublicKey::from_spki_der(candidate);
    let _signature = EcdsaP384Signature::from_der(candidate);
  }
}
