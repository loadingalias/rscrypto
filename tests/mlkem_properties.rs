#![cfg(feature = "ml-kem")]

use fips203::{
  ml_kem_768 as fips_mlkem768,
  traits::{Decaps as _, Encaps as _, KeyGen as _, SerDes as _},
};
use proptest::{prelude::*, test_runner::Config as ProptestConfig};
use rscrypto::{Kem, MlKem768, MlKem768Ciphertext, MlKem768DecapsulationKey, MlKem768EncapsulationKey, MlKemError};

const PROPTEST_CASES: u32 = if cfg!(debug_assertions) { 32 } else { 96 };

fn arbitrary_bytes_32() -> impl Strategy<Value = [u8; 32]> {
  proptest::array::uniform32(any::<u8>())
}

fn key_random(d: &[u8; 32], z: &[u8; 32]) -> [u8; MlKem768::KEY_GENERATION_RANDOM_SIZE] {
  let mut random = [0u8; MlKem768::KEY_GENERATION_RANDOM_SIZE];
  random[..32].copy_from_slice(d);
  random[32..].copy_from_slice(z);
  random
}

proptest! {
  #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

  #[test]
  fn mlkem768_matches_fips203_for_arbitrary_seeds(
    d in arbitrary_bytes_32(),
    z in arbitrary_bytes_32(),
    m in arbitrary_bytes_32(),
  ) {
    let random = key_random(&d, &z);
    let (ek, dk) = MlKem768::generate_keypair(|out| {
      out.copy_from_slice(&random);
      Ok::<(), MlKemError>(())
    })
    .unwrap();
    let (fips_ek, fips_dk) = fips_mlkem768::KG::keygen_from_seed(d, z);

    prop_assert_eq!(*ek.as_bytes(), fips_ek.clone().into_bytes());
    prop_assert_eq!(*dk.expose_secret().as_bytes(), fips_dk.clone().into_bytes());

    let (ciphertext, shared_secret) = MlKem768::encapsulate(&ek, |out| {
      out.copy_from_slice(&m);
      Ok::<(), MlKemError>(())
    })
    .unwrap();
    let (fips_shared_secret, fips_ciphertext) = fips_ek.encaps_from_seed(&m);

    prop_assert_eq!(*ciphertext.as_bytes(), fips_ciphertext.clone().into_bytes());
    prop_assert_eq!(*shared_secret.expose_secret().as_bytes(), fips_shared_secret.into_bytes());

    let decapsulated = MlKem768::decapsulate(&dk, &ciphertext).unwrap();
    let fips_decapsulated = fips_dk.try_decaps(&fips_ciphertext).unwrap();
    prop_assert_eq!(*decapsulated.expose_secret().as_bytes(), fips_decapsulated.into_bytes());
  }

  #[test]
  fn mlkem768_modified_ciphertexts_use_implicit_rejection(
    d in arbitrary_bytes_32(),
    z in arbitrary_bytes_32(),
    m in arbitrary_bytes_32(),
    byte_idx in 0usize..MlKem768::CIPHERTEXT_SIZE,
    bit_idx in 0u8..8,
  ) {
    let random = key_random(&d, &z);
    let (ek, dk) = MlKem768::generate_keypair(|out| {
      out.copy_from_slice(&random);
      Ok::<(), MlKemError>(())
    })
    .unwrap();
    let (ciphertext, encapsulated) = MlKem768::encapsulate(&ek, |out| {
      out.copy_from_slice(&m);
      Ok::<(), MlKemError>(())
    })
    .unwrap();

    let mut modified = ciphertext.to_bytes();
    modified[byte_idx] ^= 1u8 << bit_idx;
    let rejected = MlKem768::decapsulate(&dk, &MlKem768Ciphertext::from_bytes(modified)).unwrap();

    prop_assert_ne!(encapsulated, rejected);
  }

  #[test]
  fn mlkem768_slice_parsers_handle_arbitrary_lengths(bytes in prop::collection::vec(any::<u8>(), 0..=2600)) {
    if bytes.len() != MlKem768::ENCAPSULATION_KEY_SIZE {
      prop_assert_eq!(
        MlKem768EncapsulationKey::try_from_slice(&bytes).unwrap_err(),
        MlKemError::InvalidEncapsulationKey
      );
    } else {
      let _ = MlKem768EncapsulationKey::try_from_slice(&bytes);
    }

    if bytes.len() != MlKem768::DECAPSULATION_KEY_SIZE {
      prop_assert_eq!(
        MlKem768DecapsulationKey::try_from_slice(&bytes).unwrap_err(),
        MlKemError::InvalidDecapsulationKey
      );
    } else {
      let _ = MlKem768DecapsulationKey::try_from_slice(&bytes);
    }

    if bytes.len() != MlKem768::CIPHERTEXT_SIZE {
      prop_assert_eq!(
        MlKem768Ciphertext::try_from_slice(&bytes).unwrap_err(),
        MlKemError::InvalidCiphertext
      );
    } else {
      MlKem768Ciphertext::try_from_slice(&bytes).unwrap().validate().unwrap();
    }
  }
}
