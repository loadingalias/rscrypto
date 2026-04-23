#![cfg(feature = "ed25519")]

use ed25519_dalek::{Signature as DalekSignature, Signer, SigningKey, VerifyingKey};
use rscrypto::{Ed25519Keypair, Ed25519PublicKey, Ed25519SecretKey, Ed25519Signature};

fn patterned_message(len: usize, mul: u8, add: u8) -> Vec<u8> {
  (0..len)
    .map(|i| (i as u8).wrapping_mul(mul).wrapping_add(add))
    .collect()
}

#[test]
fn ed25519_matches_dalek_for_deterministic_cases() {
  let cases = [
    ([0u8; 32], Vec::new()),
    ([0xFFu8; 32], vec![0x72]),
    ([0x42u8; 32], patterned_message(16, 17, 3)),
    ([0x11u8; 32], patterned_message(64, 31, 7)),
    ([0xA5u8; 32], patterned_message(255, 13, 1)),
    ([0x5Au8; 32], patterned_message(1023, 29, 9)),
  ];

  for (secret_bytes, message) in cases {
    let signing_key = SigningKey::from_bytes(&secret_bytes);
    let verifying_key: VerifyingKey = signing_key.verifying_key();

    let secret = Ed25519SecretKey::from_bytes(secret_bytes);
    let keypair = Ed25519Keypair::from_secret_key(secret);
    let public = keypair.public_key();
    let ours = keypair.sign(&message);
    let oracle: DalekSignature = signing_key.sign(&message);

    assert_eq!(public.to_bytes(), verifying_key.to_bytes());
    assert_eq!(ours.to_bytes(), oracle.to_bytes());
    assert!(public.verify(&message, &ours).is_ok());

    let oracle_public = Ed25519PublicKey::from_bytes(verifying_key.to_bytes());
    let oracle_signature = Ed25519Signature::from_bytes(oracle.to_bytes());
    assert!(oracle_public.verify(&message, &oracle_signature).is_ok());
    assert!(verifying_key.verify_strict(&message, &oracle).is_ok());
  }
}

#[test]
fn ed25519_and_dalek_agree_on_rejection_for_tampered_signature() {
  let secret_bytes = [0x33u8; 32];
  let message = patterned_message(128, 19, 5);

  let signing_key = SigningKey::from_bytes(&secret_bytes);
  let verifying_key = signing_key.verifying_key();
  let secret = Ed25519SecretKey::from_bytes(secret_bytes);
  let public = secret.public_key();
  let mut signature = secret.sign(&message).to_bytes();
  signature[17] ^= 0x80;

  let ours = Ed25519Signature::from_bytes(signature);
  let oracle = DalekSignature::from_bytes(&signature);

  assert!(public.verify(&message, &ours).is_err());
  assert!(verifying_key.verify_strict(&message, &oracle).is_err());
}
