use rscrypto::{Ed25519PublicKey, Ed25519SecretKey, Ed25519Signature};
use rscrypto_fuzz::{FuzzInput, some_or_return};

pub(super) fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let key_bytes: [u8; 32] = some_or_return!(input.bytes());
  let mutation = some_or_return!(input.bit_mutation());
  let message = input.rest();

  let secret = Ed25519SecretKey::from_bytes(key_bytes);
  let public = secret.public_key();
  let sig = secret.sign(message);

  // Property: sign → verify roundtrip
  public.verify(message, &sig).expect("roundtrip: verify must succeed");

  // Property: random signature must be rejected
  let bad_sig = Ed25519Signature::from_bytes([0xAB; 64]);
  assert!(public.verify(message, &bad_sig).is_err(), "accepted garbage signature");

  // Property: wrong message must be rejected
  if !message.is_empty() {
    let mut wrong = message.to_vec();
    wrong[0] ^= 1;
    assert!(
      public.verify(&wrong, &sig).is_err(),
      "accepted signature for wrong message"
    );
  }

  // Exports round-trip, and an accepted input of the export length is the
  // version 1 layout: a version 2 encoding is longer.
  let mut pkcs8 = [0; Ed25519SecretKey::PKCS8_DER_LENGTH];
  secret.to_pkcs8_der_into(&mut pkcs8);
  assert_eq!(
    Ed25519SecretKey::from_pkcs8_der(&pkcs8).map(|imported| *imported.as_bytes()),
    Ok(key_bytes),
    "PKCS #8 export must import"
  );
  assert_eq!(
    Ed25519PublicKey::from_spki_der(&public.to_spki_der()).map(|imported| imported.to_bytes()),
    Ok(public.to_bytes()),
    "SPKI export must import"
  );
  mutation.apply(&mut pkcs8);
  for candidate in [pkcs8.as_slice(), message] {
    if let Ok(parsed) = Ed25519SecretKey::from_pkcs8_der(candidate)
      && candidate.len() == Ed25519SecretKey::PKCS8_DER_LENGTH
    {
      let mut encoded = [0; Ed25519SecretKey::PKCS8_DER_LENGTH];
      parsed.to_pkcs8_der_into(&mut encoded);
      assert_eq!(encoded.as_slice(), candidate, "accepted PKCS #8 must be canonical");
    }
    // Arbitrary byte lengths and contents must not panic at a public parser.
    let _spki = Ed25519PublicKey::from_spki_der(candidate);
  }

  // Differential: rscrypto ↔ ed25519-dalek
  {
    use ed25519_dalek::{Signer, Verifier};

    let dalek_sk = ed25519_dalek::SigningKey::from_bytes(&key_bytes);
    let dalek_vk = dalek_sk.verifying_key();

    // Public keys must match
    assert_eq!(public.to_bytes(), dalek_vk.to_bytes(), "public key mismatch");

    // Signatures must match (Ed25519 is deterministic)
    let dalek_sig = dalek_sk.sign(message);
    assert_eq!(sig.to_bytes(), dalek_sig.to_bytes(), "signature mismatch");

    // Cross-verify: our sig with their verifier
    let our_as_dalek = ed25519_dalek::Signature::from_bytes(&sig.to_bytes());
    dalek_vk
      .verify(message, &our_as_dalek)
      .expect("dalek rejected our signature");

    // Cross-verify: their sig with our verifier
    let theirs_as_ours = Ed25519Signature::from_bytes(dalek_sig.to_bytes());
    public
      .verify(message, &theirs_as_ours)
      .expect("we rejected dalek signature");
  }
}
