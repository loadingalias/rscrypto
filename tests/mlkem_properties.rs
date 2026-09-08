#![cfg(feature = "ml-kem")]

use fips203::{
  ml_kem_512 as fips_mlkem512, ml_kem_768 as fips_mlkem768, ml_kem_1024 as fips_mlkem1024,
  traits::{Decaps as _, Encaps as _, KeyGen as _, SerDes as _},
};
use proptest::{prelude::*, test_runner::Config as ProptestConfig};
use rscrypto::{
  Kem, MlKem512, MlKem512Ciphertext, MlKem512DecapsulationKey, MlKem512EncapsulationKey, MlKem768, MlKem768Ciphertext,
  MlKem768DecapsulationKey, MlKem768EncapsulationKey, MlKem1024, MlKem1024Ciphertext, MlKem1024DecapsulationKey,
  MlKem1024EncapsulationKey, MlKemError,
};

const PROPTEST_CASES: u32 = if cfg!(debug_assertions) { 32 } else { 96 };

fn arbitrary_bytes_32() -> impl Strategy<Value = [u8; 32]> {
  proptest::array::uniform32(any::<u8>())
}

fn key_random<const N: usize>(d: &[u8; 32], z: &[u8; 32]) -> [u8; N] {
  let mut random = [0u8; N];
  random[..32].copy_from_slice(d);
  random[32..].copy_from_slice(z);
  random
}

#[test]
fn mlkem512_matches_fips203_for_reduced_feature_seed() {
  let d = [
    249, 206, 215, 37, 228, 105, 120, 238, 82, 21, 50, 99, 184, 68, 205, 166, 255, 59, 174, 206, 253, 125, 87, 13, 254,
    16, 123, 248, 146, 130, 47, 191,
  ];
  let z = [
    105, 76, 117, 153, 16, 21, 249, 206, 157, 192, 254, 141, 117, 121, 220, 189, 227, 149, 254, 63, 23, 252, 51, 113,
    212, 103, 7, 205, 195, 26, 35, 106,
  ];
  let m = [
    140, 117, 20, 218, 228, 30, 170, 42, 115, 49, 83, 151, 0, 35, 162, 240, 143, 132, 166, 48, 23, 183, 210, 64, 65,
    202, 86, 235, 26, 5, 223, 188,
  ];

  let random = key_random::<{ MlKem512::KEY_GENERATION_RANDOM_SIZE }>(&d, &z);
  let (ek, dk) = MlKem512::generate_keypair(|out| {
    out.copy_from_slice(&random);
    Ok::<(), MlKemError>(())
  })
  .expect("deterministic ML-KEM-512 key generation must succeed");
  let (fips_ek, fips_dk) = fips_mlkem512::KG::keygen_from_seed(d, z);

  assert_eq!(*ek.as_bytes(), fips_ek.clone().into_bytes());
  assert_eq!(*dk.expose_secret().as_bytes(), fips_dk.clone().into_bytes());

  let (ciphertext, shared_secret) = MlKem512::encapsulate(&ek, |out| {
    out.copy_from_slice(&m);
    Ok::<(), MlKemError>(())
  })
  .expect("deterministic ML-KEM-512 encapsulation must succeed");
  let (fips_shared_secret, fips_ciphertext) = fips_ek.encaps_from_seed(&m);

  assert_eq!(*ciphertext.as_bytes(), fips_ciphertext.clone().into_bytes());
  assert_eq!(
    *shared_secret.expose_secret().as_bytes(),
    fips_shared_secret.into_bytes()
  );

  let decapsulated = MlKem512::decapsulate(&dk, &ciphertext).expect("generated ML-KEM-512 ciphertext must decapsulate");
  let fips_decapsulated = fips_dk
    .try_decaps(&fips_ciphertext)
    .expect("FIPS 203 ML-KEM-512 ciphertext must decapsulate");
  assert_eq!(*decapsulated.expose_secret().as_bytes(), fips_decapsulated.into_bytes());
}

macro_rules! mlkem_profile_properties {
  (
    $matches_fips203:ident,
    $implicit_rejection:ident,
    $slice_parsers:ident,
    $profile:ty,
    $ciphertext:ty,
    $decapsulation_key:ty,
    $encapsulation_key:ty,
    $fips:ident,
    $name:literal
  ) => {
    proptest! {
      #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

      #[test]
      fn $matches_fips203(
        d in arbitrary_bytes_32(),
        z in arbitrary_bytes_32(),
        m in arbitrary_bytes_32(),
      ) {
        let random = key_random::<{ <$profile>::KEY_GENERATION_RANDOM_SIZE }>(&d, &z);
        let (ek, dk) = <$profile>::generate_keypair(|out| {
          out.copy_from_slice(&random);
          Ok::<(), MlKemError>(())
        })
        .expect("deterministic ML-KEM key generation must succeed");
        let (fips_ek, fips_dk) = $fips::KG::keygen_from_seed(d, z);

        prop_assert_eq!(*ek.as_bytes(), fips_ek.clone().into_bytes());
        prop_assert_eq!(*dk.expose_secret().as_bytes(), fips_dk.clone().into_bytes());

        let (ciphertext, shared_secret) = <$profile>::encapsulate(&ek, |out| {
          out.copy_from_slice(&m);
          Ok::<(), MlKemError>(())
        })
        .expect("deterministic ML-KEM encapsulation must succeed");
        let (fips_shared_secret, fips_ciphertext) = fips_ek.encaps_from_seed(&m);

        prop_assert_eq!(*ciphertext.as_bytes(), fips_ciphertext.clone().into_bytes());
        prop_assert_eq!(*shared_secret.expose_secret().as_bytes(), fips_shared_secret.into_bytes());

        let decapsulated = <$profile>::decapsulate(&dk, &ciphertext)
          .expect("generated ML-KEM ciphertext must decapsulate");
        let fips_decapsulated = fips_dk
          .try_decaps(&fips_ciphertext)
          .expect("FIPS 203 ML-KEM ciphertext must decapsulate");
        prop_assert_eq!(*decapsulated.expose_secret().as_bytes(), fips_decapsulated.into_bytes());
      }

      #[test]
      fn $implicit_rejection(
        d in arbitrary_bytes_32(),
        z in arbitrary_bytes_32(),
        m in arbitrary_bytes_32(),
        byte_idx in 0usize..<$profile>::CIPHERTEXT_SIZE,
        bit_idx in 0u8..8,
      ) {
        let random = key_random::<{ <$profile>::KEY_GENERATION_RANDOM_SIZE }>(&d, &z);
        let (ek, dk) = <$profile>::generate_keypair(|out| {
          out.copy_from_slice(&random);
          Ok::<(), MlKemError>(())
        })
        .expect("deterministic ML-KEM key generation must succeed");
        let (ciphertext, encapsulated) = <$profile>::encapsulate(&ek, |out| {
          out.copy_from_slice(&m);
          Ok::<(), MlKemError>(())
        })
        .expect("deterministic ML-KEM encapsulation must succeed");

        let mut modified = ciphertext.to_bytes();
        modified[byte_idx] ^= 1u8 << bit_idx;
        let rejected = <$profile>::decapsulate(&dk, &<$ciphertext>::from_bytes(modified))
          .expect("modified ML-KEM ciphertext must use implicit rejection");

        prop_assert!(
          !encapsulated.ct_eq(&rejected).declassify(),
          "{name} modified ciphertext returned original shared secret",
          name = $name
        );
      }

      #[test]
      fn $slice_parsers(
        d in arbitrary_bytes_32(),
        z in arbitrary_bytes_32(),
        bytes in prop::collection::vec(any::<u8>(), <$profile>::CIPHERTEXT_SIZE),
      ) {
        for len in [0, <$profile>::ENCAPSULATION_KEY_SIZE - 1, <$profile>::ENCAPSULATION_KEY_SIZE + 1] {
          prop_assert_eq!(
            <$encapsulation_key>::try_from_slice(&vec![0; len]).expect_err("wrong-length encapsulation key"),
            MlKemError::InvalidEncapsulationKey
          );
        }
        for len in [0, <$profile>::DECAPSULATION_KEY_SIZE - 1, <$profile>::DECAPSULATION_KEY_SIZE + 1] {
          prop_assert_eq!(
            <$decapsulation_key>::try_from_slice(&vec![0; len]).expect_err("wrong-length decapsulation key"),
            MlKemError::InvalidDecapsulationKey
          );
        }
        for len in [0, <$profile>::CIPHERTEXT_SIZE - 1, <$profile>::CIPHERTEXT_SIZE + 1] {
          prop_assert_eq!(
            <$ciphertext>::try_from_slice(&vec![0; len]).expect_err("wrong-length ciphertext"),
            MlKemError::InvalidCiphertext
          );
        }

        let (ek, dk) = $fips::KG::keygen_from_seed(d, z);
        let mut ek_bytes = ek.into_bytes();
        let mut dk_bytes = dk.into_bytes();
        let parsed_ek = <$encapsulation_key>::try_from_slice(&ek_bytes)
          .expect("independently generated encapsulation key must parse");
        prop_assert_eq!(parsed_ek.as_ref(), ek_bytes.as_slice());
        let parsed_dk = <$decapsulation_key>::try_from_slice(&dk_bytes)
          .expect("independently generated decapsulation key must parse");
        prop_assert_eq!(parsed_dk.as_ref(), dk_bytes.as_slice());

        // Encode the first 12-bit coefficient as 4095, outside the canonical range 0..3329.
        ek_bytes[0] = 0xff;
        ek_bytes[1] |= 0x0f;
        prop_assert_eq!(
          <$encapsulation_key>::try_from_slice(&ek_bytes).expect_err("noncanonical coefficient"),
          MlKemError::InvalidEncapsulationKey
        );
        // The final 64 bytes are H(ek) followed by z; corrupt only the stored hash.
        dk_bytes[<$profile>::DECAPSULATION_KEY_SIZE - 64] ^= 1;
        prop_assert_eq!(
          <$decapsulation_key>::try_from_slice(&dk_bytes).expect_err("corrupted encapsulation-key hash"),
          MlKemError::InvalidDecapsulationKey
        );

        let ciphertext = <$ciphertext>::try_from_slice(&bytes)
          .expect("every exact-size ciphertext encoding must parse");
        prop_assert_eq!(ciphertext.as_ref(), bytes.as_slice());
      }
    }
  };
}

mlkem_profile_properties!(
  mlkem512_matches_fips203_for_arbitrary_seeds,
  mlkem512_modified_ciphertexts_use_implicit_rejection,
  mlkem512_slice_parsers_check_boundaries_and_exact_size,
  MlKem512,
  MlKem512Ciphertext,
  MlKem512DecapsulationKey,
  MlKem512EncapsulationKey,
  fips_mlkem512,
  "ML-KEM-512"
);

mlkem_profile_properties!(
  mlkem768_matches_fips203_for_arbitrary_seeds,
  mlkem768_modified_ciphertexts_use_implicit_rejection,
  mlkem768_slice_parsers_check_boundaries_and_exact_size,
  MlKem768,
  MlKem768Ciphertext,
  MlKem768DecapsulationKey,
  MlKem768EncapsulationKey,
  fips_mlkem768,
  "ML-KEM-768"
);

mlkem_profile_properties!(
  mlkem1024_matches_fips203_for_arbitrary_seeds,
  mlkem1024_modified_ciphertexts_use_implicit_rejection,
  mlkem1024_slice_parsers_check_boundaries_and_exact_size,
  MlKem1024,
  MlKem1024Ciphertext,
  MlKem1024DecapsulationKey,
  MlKem1024EncapsulationKey,
  fips_mlkem1024,
  "ML-KEM-1024"
);
