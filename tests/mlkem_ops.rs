#![cfg(feature = "ml-kem")]

use rscrypto::{
  Kem, MlKem512, MlKem512Ciphertext, MlKem512DecapsulationKey, MlKem512EncapsulationKey,
  MlKem512PreparedDecapsulationKey, MlKem512PreparedEncapsulationKey, MlKem768, MlKem768Ciphertext,
  MlKem768DecapsulationKey, MlKem768EncapsulationKey, MlKem768PreparedDecapsulationKey,
  MlKem768PreparedEncapsulationKey, MlKem1024, MlKem1024Ciphertext, MlKem1024DecapsulationKey,
  MlKem1024EncapsulationKey, MlKem1024PreparedDecapsulationKey, MlKem1024PreparedEncapsulationKey, MlKemError,
};

fn deterministic_bytes<const N: usize>(offset: u8) -> [u8; N] {
  let mut out = [0u8; N];
  for (i, byte) in out.iter_mut().enumerate() {
    *byte = offset.wrapping_add(i as u8);
  }
  out
}

macro_rules! mlkem_profile_tests {
  (
    $round_trip:ident,
    $rejects_noncanonical_public_key:ident,
    $prepared_keys_match_validating_api:ident,
    $prepared_keys_reject_invalid_material:ident,
    $slice_parsers_reject_wrong_lengths:ident,
    $rejects_decapsulation_key_hash_mismatch:ident,
    $uses_implicit_rejection:ident,
    $profile:ty,
    $ciphertext:ty,
    $decapsulation_key:ty,
    $encapsulation_key:ty,
    $prepared_decapsulation_key:ty,
    $prepared_encapsulation_key:ty
  ) => {
    #[test]
    fn $round_trip() {
      let key_random = deterministic_bytes::<{ <$profile>::KEY_GENERATION_RANDOM_SIZE }>(0x10);
      let encaps_random = deterministic_bytes::<{ <$profile>::ENCAPSULATION_RANDOM_SIZE }>(0xa0);

      let (ek, dk) = <$profile>::generate_keypair(|out| {
        assert_eq!(out.len(), key_random.len());
        out.copy_from_slice(&key_random);
        Ok::<(), MlKemError>(())
      })
      .unwrap();

      ek.validate().unwrap();
      dk.validate().unwrap();

      let (ciphertext, encapsulated) = <$profile>::encapsulate(&ek, |out| {
        assert_eq!(out.len(), encaps_random.len());
        out.copy_from_slice(&encaps_random);
        Ok::<(), MlKemError>(())
      })
      .unwrap();

      let decapsulated = <$profile>::decapsulate(&dk, &ciphertext).unwrap();
      assert_eq!(encapsulated, decapsulated);
    }

    #[test]
    fn $rejects_noncanonical_public_key() {
      let bad_key = <$encapsulation_key>::from_bytes([0xff; <$profile>::ENCAPSULATION_KEY_SIZE]);
      let mut random_called = false;

      let err = <$profile>::encapsulate(&bad_key, |out| {
        random_called = true;
        out.fill(0);
        Ok::<(), MlKemError>(())
      })
      .unwrap_err();

      assert_eq!(err, MlKemError::InvalidEncapsulationKey);
      assert!(!random_called);
    }

    #[test]
    fn $prepared_keys_match_validating_api() {
      let key_random = deterministic_bytes::<{ <$profile>::KEY_GENERATION_RANDOM_SIZE }>(0x18);
      let encaps_random = deterministic_bytes::<{ <$profile>::ENCAPSULATION_RANDOM_SIZE }>(0xa8);

      let (ek, dk) = <$profile>::generate_keypair(|out| {
        out.copy_from_slice(&key_random);
        Ok::<(), MlKemError>(())
      })
      .unwrap();

      let prepared_ek = ek.prepare().unwrap();
      let prepared_dk = dk.prepare().unwrap();
      let profile_prepared_ek = <$profile>::prepare_encapsulation_key(&ek).unwrap();
      let profile_prepared_dk = <$profile>::prepare_decapsulation_key(&dk).unwrap();
      assert_eq!(prepared_ek, profile_prepared_ek);
      assert_eq!(prepared_dk, profile_prepared_dk);

      let (validating_ciphertext, validating_shared) = <$profile>::encapsulate(&ek, |out| {
        out.copy_from_slice(&encaps_random);
        Ok::<(), MlKemError>(())
      })
      .unwrap();
      let (prepared_ciphertext, prepared_shared) = prepared_ek
        .encapsulate(|out| {
          out.copy_from_slice(&encaps_random);
          Ok::<(), MlKemError>(())
        })
        .unwrap();

      assert_eq!(validating_ciphertext, prepared_ciphertext);
      assert_eq!(validating_shared, prepared_shared);
      assert_eq!(
        prepared_dk.decapsulate(&prepared_ciphertext).unwrap(),
        validating_shared
      );
      assert_eq!(
        <$profile>::decapsulate_prepared(&prepared_dk, &prepared_ciphertext).unwrap(),
        <$profile>::decapsulate(&dk, &prepared_ciphertext).unwrap()
      );
    }

    #[test]
    fn $prepared_keys_reject_invalid_material() {
      let bad_ek = <$encapsulation_key>::from_bytes([0xff; <$profile>::ENCAPSULATION_KEY_SIZE]);
      assert_eq!(bad_ek.prepare().unwrap_err(), MlKemError::InvalidEncapsulationKey);
      assert_eq!(
        <$prepared_encapsulation_key>::try_from_slice(&[0u8; <$profile>::ENCAPSULATION_KEY_SIZE - 1]).unwrap_err(),
        MlKemError::InvalidEncapsulationKey
      );

      let key_random = deterministic_bytes::<{ <$profile>::KEY_GENERATION_RANDOM_SIZE }>(0x28);
      let (_, dk) = <$profile>::generate_keypair(|out| {
        out.copy_from_slice(&key_random);
        Ok::<(), MlKemError>(())
      })
      .unwrap();
      let mut bad_dk = dk.expose_secret().expose();
      let hash_start = <$profile>::DECAPSULATION_KEY_SIZE - 64;
      bad_dk[hash_start] ^= 0x01;
      let bad_dk = <$decapsulation_key>::from_bytes(bad_dk);

      assert_eq!(bad_dk.prepare().unwrap_err(), MlKemError::InvalidDecapsulationKey);
      assert_eq!(
        <$prepared_decapsulation_key>::try_from_slice(&[0u8; <$profile>::DECAPSULATION_KEY_SIZE - 1]).unwrap_err(),
        MlKemError::InvalidDecapsulationKey
      );
    }

    #[test]
    fn $slice_parsers_reject_wrong_lengths() {
      assert_eq!(
        <$encapsulation_key>::try_from_slice(&[0u8; <$profile>::ENCAPSULATION_KEY_SIZE - 1]).unwrap_err(),
        MlKemError::InvalidEncapsulationKey
      );
      assert_eq!(
        <$decapsulation_key>::try_from_slice(&[0u8; <$profile>::DECAPSULATION_KEY_SIZE - 1]).unwrap_err(),
        MlKemError::InvalidDecapsulationKey
      );
      assert_eq!(
        <$ciphertext>::try_from_slice(&[0u8; <$profile>::CIPHERTEXT_SIZE - 1]).unwrap_err(),
        MlKemError::InvalidCiphertext
      );

      let ciphertext = <$ciphertext>::try_from_slice(&[0u8; <$profile>::CIPHERTEXT_SIZE]).unwrap();
      ciphertext.validate().unwrap();
    }

    #[test]
    fn $rejects_decapsulation_key_hash_mismatch() {
      let key_random = deterministic_bytes::<{ <$profile>::KEY_GENERATION_RANDOM_SIZE }>(0x20);
      let encaps_random = deterministic_bytes::<{ <$profile>::ENCAPSULATION_RANDOM_SIZE }>(0xb0);

      let (ek, dk) = <$profile>::generate_keypair(|out| {
        out.copy_from_slice(&key_random);
        Ok::<(), MlKemError>(())
      })
      .unwrap();
      let (ciphertext, _) = <$profile>::encapsulate(&ek, |out| {
        out.copy_from_slice(&encaps_random);
        Ok::<(), MlKemError>(())
      })
      .unwrap();

      let mut bad_dk = dk.expose_secret().expose();
      let hash_start = <$profile>::DECAPSULATION_KEY_SIZE - 64;
      bad_dk[hash_start] ^= 0x01;
      let bad_dk = <$decapsulation_key>::from_bytes(bad_dk);

      let err = <$profile>::decapsulate(&bad_dk, &ciphertext).unwrap_err();
      assert_eq!(err, MlKemError::InvalidDecapsulationKey);
    }

    #[test]
    fn $uses_implicit_rejection() {
      let key_random = deterministic_bytes::<{ <$profile>::KEY_GENERATION_RANDOM_SIZE }>(0x30);
      let encaps_random = deterministic_bytes::<{ <$profile>::ENCAPSULATION_RANDOM_SIZE }>(0xc0);

      let (ek, dk) = <$profile>::generate_keypair(|out| {
        out.copy_from_slice(&key_random);
        Ok::<(), MlKemError>(())
      })
      .unwrap();
      let (ciphertext, encapsulated) = <$profile>::encapsulate(&ek, |out| {
        out.copy_from_slice(&encaps_random);
        Ok::<(), MlKemError>(())
      })
      .unwrap();

      let mut modified = ciphertext.to_bytes();
      modified[0] ^= 0x01;
      let modified = <$ciphertext>::from_bytes(modified);

      let rejected = <$profile>::decapsulate(&dk, &modified).unwrap();
      assert_ne!(encapsulated, rejected);
    }
  };
}

mlkem_profile_tests!(
  mlkem512_kem_round_trip_with_deterministic_randomness,
  mlkem512_encapsulation_rejects_noncanonical_public_key_before_randomness,
  mlkem512_prepared_keys_match_validating_api,
  mlkem512_prepared_keys_reject_invalid_material,
  mlkem512_slice_parsers_reject_wrong_lengths,
  mlkem512_decapsulation_rejects_decapsulation_key_hash_mismatch,
  mlkem512_decapsulation_uses_implicit_rejection_for_modified_ciphertext,
  MlKem512,
  MlKem512Ciphertext,
  MlKem512DecapsulationKey,
  MlKem512EncapsulationKey,
  MlKem512PreparedDecapsulationKey,
  MlKem512PreparedEncapsulationKey
);

mlkem_profile_tests!(
  mlkem768_kem_round_trip_with_deterministic_randomness,
  mlkem768_encapsulation_rejects_noncanonical_public_key_before_randomness,
  mlkem768_prepared_keys_match_validating_api,
  mlkem768_prepared_keys_reject_invalid_material,
  mlkem768_slice_parsers_reject_wrong_lengths,
  mlkem768_decapsulation_rejects_decapsulation_key_hash_mismatch,
  mlkem768_decapsulation_uses_implicit_rejection_for_modified_ciphertext,
  MlKem768,
  MlKem768Ciphertext,
  MlKem768DecapsulationKey,
  MlKem768EncapsulationKey,
  MlKem768PreparedDecapsulationKey,
  MlKem768PreparedEncapsulationKey
);

mlkem_profile_tests!(
  mlkem1024_kem_round_trip_with_deterministic_randomness,
  mlkem1024_encapsulation_rejects_noncanonical_public_key_before_randomness,
  mlkem1024_prepared_keys_match_validating_api,
  mlkem1024_prepared_keys_reject_invalid_material,
  mlkem1024_slice_parsers_reject_wrong_lengths,
  mlkem1024_decapsulation_rejects_decapsulation_key_hash_mismatch,
  mlkem1024_decapsulation_uses_implicit_rejection_for_modified_ciphertext,
  MlKem1024,
  MlKem1024Ciphertext,
  MlKem1024DecapsulationKey,
  MlKem1024EncapsulationKey,
  MlKem1024PreparedDecapsulationKey,
  MlKem1024PreparedEncapsulationKey
);
