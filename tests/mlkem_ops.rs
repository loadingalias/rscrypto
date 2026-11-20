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
    let i = u8::try_from(i).expect("ML-KEM deterministic entropy index must fit in u8");
    *byte = offset.wrapping_add(i);
  }
  out
}

macro_rules! mlkem_profile_tests {
  (
    $round_trip:ident,
    $import_rejects_invalid_keys:ident,
    $prepared_keys_match_direct_api:ident,
    $slice_parsers_reject_wrong_lengths:ident,
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

      // Exported keys pass the FIPS 203 input checks when imported again.
      <$encapsulation_key>::try_from_slice(ek.as_bytes()).unwrap();
      <$decapsulation_key>::try_from_slice(dk.as_bytes()).unwrap();

      let (ciphertext, encapsulated) = <$profile>::encapsulate(&ek, |out| {
        assert_eq!(out.len(), encaps_random.len());
        out.copy_from_slice(&encaps_random);
        Ok::<(), MlKemError>(())
      })
      .unwrap();

      let decapsulated = <$profile>::decapsulate(&dk, &ciphertext).unwrap();
      assert!(encapsulated.ct_eq(&decapsulated).declassify());
    }

    /// No invalid key reaches encapsulation or decapsulation: each constructor
    /// runs the FIPS 203 check that the key type then guarantees.
    #[test]
    fn $import_rejects_invalid_keys() {
      // Every coefficient of an all-0xff key is above q, so the modulus check fails.
      let noncanonical = [0xff; <$profile>::ENCAPSULATION_KEY_SIZE];
      assert_eq!(
        <$encapsulation_key>::try_from_slice(&noncanonical).unwrap_err(),
        MlKemError::InvalidEncapsulationKey
      );
      assert_eq!(
        <$prepared_encapsulation_key>::try_from_slice(&noncanonical).unwrap_err(),
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
      assert_eq!(
        <$decapsulation_key>::try_from_slice(&bad_dk).unwrap_err(),
        MlKemError::InvalidDecapsulationKey
      );
      assert_eq!(
        <$prepared_decapsulation_key>::try_from_slice(&bad_dk).unwrap_err(),
        MlKemError::InvalidDecapsulationKey
      );

      assert_eq!(
        <$prepared_encapsulation_key>::try_from_slice(&[0u8; <$profile>::ENCAPSULATION_KEY_SIZE - 1]).unwrap_err(),
        MlKemError::InvalidEncapsulationKey
      );
      assert_eq!(
        <$prepared_decapsulation_key>::try_from_slice(&[0u8; <$profile>::DECAPSULATION_KEY_SIZE - 1]).unwrap_err(),
        MlKemError::InvalidDecapsulationKey
      );
    }

    #[test]
    fn $prepared_keys_match_direct_api() {
      let key_random = deterministic_bytes::<{ <$profile>::KEY_GENERATION_RANDOM_SIZE }>(0x18);
      let encaps_random = deterministic_bytes::<{ <$profile>::ENCAPSULATION_RANDOM_SIZE }>(0xa8);

      let (ek, dk) = <$profile>::generate_keypair(|out| {
        out.copy_from_slice(&key_random);
        Ok::<(), MlKemError>(())
      })
      .unwrap();

      let prepared_ek = ek.prepare();
      let prepared_dk = dk.prepare();
      assert_eq!(prepared_ek, <$profile>::prepare_encapsulation_key(&ek));
      assert!(
        prepared_dk
          .ct_eq(&<$profile>::prepare_decapsulation_key(&dk))
          .declassify()
      );
      // The fused raw import prepares the same keys.
      assert_eq!(
        prepared_ek,
        <$prepared_encapsulation_key>::try_from_slice(ek.as_bytes()).unwrap()
      );
      assert!(
        prepared_dk
          .ct_eq(&<$prepared_decapsulation_key>::try_from_slice(dk.as_bytes()).unwrap())
          .declassify()
      );

      let (direct_ciphertext, direct_shared) = <$profile>::encapsulate(&ek, |out| {
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

      assert_eq!(direct_ciphertext, prepared_ciphertext);
      assert!(direct_shared.ct_eq(&prepared_shared).declassify());

      let prepared_decapsulated = prepared_dk.decapsulate(&prepared_ciphertext);
      assert!(prepared_decapsulated.ct_eq(&direct_shared).declassify());

      let profile_prepared_shared = <$profile>::decapsulate_prepared(&prepared_dk, &prepared_ciphertext);
      let direct_decapsulated = <$profile>::decapsulate(&dk, &prepared_ciphertext).unwrap();
      assert!(profile_prepared_shared.ct_eq(&direct_decapsulated).declassify());
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
      <$ciphertext>::try_from_slice(&[0u8; <$profile>::CIPHERTEXT_SIZE]).unwrap();
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
      assert!(!encapsulated.ct_eq(&rejected).declassify());
    }
  };
}

mlkem_profile_tests!(
  mlkem512_kem_round_trip_with_deterministic_randomness,
  mlkem512_import_rejects_invalid_keys,
  mlkem512_prepared_keys_match_direct_api,
  mlkem512_slice_parsers_reject_wrong_lengths,
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
  mlkem768_import_rejects_invalid_keys,
  mlkem768_prepared_keys_match_direct_api,
  mlkem768_slice_parsers_reject_wrong_lengths,
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
  mlkem1024_import_rejects_invalid_keys,
  mlkem1024_prepared_keys_match_direct_api,
  mlkem1024_slice_parsers_reject_wrong_lengths,
  mlkem1024_decapsulation_uses_implicit_rejection_for_modified_ciphertext,
  MlKem1024,
  MlKem1024Ciphertext,
  MlKem1024DecapsulationKey,
  MlKem1024EncapsulationKey,
  MlKem1024PreparedDecapsulationKey,
  MlKem1024PreparedEncapsulationKey
);
