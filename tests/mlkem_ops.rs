#![cfg(feature = "ml-kem")]

use rscrypto::{Kem, MlKem768, MlKem768Ciphertext, MlKem768DecapsulationKey, MlKem768EncapsulationKey, MlKemError};

fn deterministic_bytes<const N: usize>(offset: u8) -> [u8; N] {
  let mut out = [0u8; N];
  for (i, byte) in out.iter_mut().enumerate() {
    *byte = offset.wrapping_add(i as u8);
  }
  out
}

#[test]
fn mlkem768_kem_round_trip_with_deterministic_randomness() {
  let key_random = deterministic_bytes::<{ MlKem768::KEY_GENERATION_RANDOM_SIZE }>(0x10);
  let encaps_random = deterministic_bytes::<{ MlKem768::ENCAPSULATION_RANDOM_SIZE }>(0xa0);

  let (ek, dk) = MlKem768::generate_keypair(|out| {
    assert_eq!(out.len(), key_random.len());
    out.copy_from_slice(&key_random);
    Ok::<(), MlKemError>(())
  })
  .unwrap();

  ek.validate().unwrap();
  dk.validate().unwrap();

  let (ciphertext, encapsulated) = MlKem768::encapsulate(&ek, |out| {
    assert_eq!(out.len(), encaps_random.len());
    out.copy_from_slice(&encaps_random);
    Ok::<(), MlKemError>(())
  })
  .unwrap();

  let decapsulated = MlKem768::decapsulate(&dk, &ciphertext).unwrap();
  assert_eq!(encapsulated, decapsulated);
}

#[test]
fn mlkem768_encapsulation_rejects_noncanonical_public_key_before_randomness() {
  let bad_key = MlKem768EncapsulationKey::from_bytes([0xff; MlKem768::ENCAPSULATION_KEY_SIZE]);
  let mut random_called = false;

  let err = MlKem768::encapsulate(&bad_key, |out| {
    random_called = true;
    out.fill(0);
    Ok::<(), MlKemError>(())
  })
  .unwrap_err();

  assert_eq!(err, MlKemError::InvalidEncapsulationKey);
  assert!(!random_called);
}

#[test]
fn mlkem768_slice_parsers_reject_wrong_lengths() {
  assert_eq!(
    MlKem768EncapsulationKey::try_from_slice(&[0u8; MlKem768::ENCAPSULATION_KEY_SIZE - 1]).unwrap_err(),
    MlKemError::InvalidEncapsulationKey
  );
  assert_eq!(
    MlKem768DecapsulationKey::try_from_slice(&[0u8; MlKem768::DECAPSULATION_KEY_SIZE - 1]).unwrap_err(),
    MlKemError::InvalidDecapsulationKey
  );
  assert_eq!(
    MlKem768Ciphertext::try_from_slice(&[0u8; MlKem768::CIPHERTEXT_SIZE - 1]).unwrap_err(),
    MlKemError::InvalidCiphertext
  );

  let ciphertext = MlKem768Ciphertext::try_from_slice(&[0u8; MlKem768::CIPHERTEXT_SIZE]).unwrap();
  ciphertext.validate().unwrap();
}

#[test]
fn mlkem768_decapsulation_rejects_decapsulation_key_hash_mismatch() {
  let key_random = deterministic_bytes::<{ MlKem768::KEY_GENERATION_RANDOM_SIZE }>(0x20);
  let encaps_random = deterministic_bytes::<{ MlKem768::ENCAPSULATION_RANDOM_SIZE }>(0xb0);

  let (ek, dk) = MlKem768::generate_keypair(|out| {
    out.copy_from_slice(&key_random);
    Ok::<(), MlKemError>(())
  })
  .unwrap();
  let (ciphertext, _) = MlKem768::encapsulate(&ek, |out| {
    out.copy_from_slice(&encaps_random);
    Ok::<(), MlKemError>(())
  })
  .unwrap();

  let mut bad_dk = dk.expose_secret().expose();
  let hash_start = MlKem768::DECAPSULATION_KEY_SIZE - 64;
  bad_dk[hash_start] ^= 0x01;
  let bad_dk = MlKem768DecapsulationKey::from_bytes(bad_dk);

  let err = MlKem768::decapsulate(&bad_dk, &ciphertext).unwrap_err();
  assert_eq!(err, MlKemError::InvalidDecapsulationKey);
}

#[test]
fn mlkem768_decapsulation_uses_implicit_rejection_for_modified_ciphertext() {
  let key_random = deterministic_bytes::<{ MlKem768::KEY_GENERATION_RANDOM_SIZE }>(0x30);
  let encaps_random = deterministic_bytes::<{ MlKem768::ENCAPSULATION_RANDOM_SIZE }>(0xc0);

  let (ek, dk) = MlKem768::generate_keypair(|out| {
    out.copy_from_slice(&key_random);
    Ok::<(), MlKemError>(())
  })
  .unwrap();
  let (ciphertext, encapsulated) = MlKem768::encapsulate(&ek, |out| {
    out.copy_from_slice(&encaps_random);
    Ok::<(), MlKemError>(())
  })
  .unwrap();

  let mut modified = ciphertext.to_bytes();
  modified[0] ^= 0x01;
  let modified = MlKem768Ciphertext::from_bytes(modified);

  let rejected = MlKem768::decapsulate(&dk, &modified).unwrap();
  assert_ne!(encapsulated, rejected);
}
