#![cfg(feature = "ml-kem")]

use rscrypto::{
  MlKem512, MlKem512Ciphertext, MlKem512DecapsulationKey, MlKem512EncapsulationKey, MlKem512Seed, MlKem512SharedSecret,
  MlKem768, MlKem768Ciphertext, MlKem768DecapsulationKey, MlKem768Seed, MlKem768SharedSecret, MlKem1024,
  MlKem1024Ciphertext, MlKem1024DecapsulationKey, MlKem1024EncapsulationKey, MlKem1024Seed, MlKem1024SharedSecret,
};

#[test]
fn mlkem_profile_sizes_match_fips203_table3() {
  assert_eq!(MlKem512::ENCAPSULATION_KEY_SIZE, 800);
  assert_eq!(MlKem512::DECAPSULATION_KEY_SIZE, 1632);
  assert_eq!(MlKem512::CIPHERTEXT_SIZE, 768);
  assert_eq!(MlKem512::SHARED_SECRET_SIZE, 32);

  assert_eq!(MlKem768::ENCAPSULATION_KEY_SIZE, 1184);
  assert_eq!(MlKem768::DECAPSULATION_KEY_SIZE, 2400);
  assert_eq!(MlKem768::CIPHERTEXT_SIZE, 1088);
  assert_eq!(MlKem768::SHARED_SECRET_SIZE, 32);

  assert_eq!(MlKem1024::ENCAPSULATION_KEY_SIZE, 1568);
  assert_eq!(MlKem1024::DECAPSULATION_KEY_SIZE, 3168);
  assert_eq!(MlKem1024::CIPHERTEXT_SIZE, 1568);
  assert_eq!(MlKem1024::SHARED_SECRET_SIZE, 32);
}

#[test]
fn mlkem_profile_randomness_and_security_strengths_match_fips203_table2() {
  assert_eq!(MlKem512::KEY_GENERATION_RANDOM_SIZE, 64);
  assert_eq!(MlKem768::KEY_GENERATION_RANDOM_SIZE, 64);
  assert_eq!(MlKem1024::KEY_GENERATION_RANDOM_SIZE, 64);
  assert_eq!(MlKem512::ENCAPSULATION_RANDOM_SIZE, 32);
  assert_eq!(MlKem768::ENCAPSULATION_RANDOM_SIZE, 32);
  assert_eq!(MlKem1024::ENCAPSULATION_RANDOM_SIZE, 32);

  assert_eq!(MlKem512::SECURITY_CATEGORY, 1);
  assert_eq!(MlKem768::SECURITY_CATEGORY, 3);
  assert_eq!(MlKem1024::SECURITY_CATEGORY, 5);
  assert_eq!(MlKem512::REQUIRED_RBG_STRENGTH_BITS, 128);
  assert_eq!(MlKem768::REQUIRED_RBG_STRENGTH_BITS, 192);
  assert_eq!(MlKem1024::REQUIRED_RBG_STRENGTH_BITS, 256);
}

#[test]
fn mlkem_public_values_roundtrip_all_bytes() {
  // A key is valid by construction, so it comes from a seed; its bytes import again.
  let (ek512, _) = MlKem512Seed::from_bytes([0x51; 64]).keypair();
  let ct512 = MlKem512Ciphertext::from_bytes([0x52; MlKem512Ciphertext::LENGTH]);
  assert!(MlKem512EncapsulationKey::try_from_slice(&ek512.to_bytes()).expect("a generated key imports") == ek512);
  assert_eq!(ct512.to_bytes(), [0x52; MlKem512Ciphertext::LENGTH]);
  assert!(ek512 != MlKem512Seed::from_bytes([0x53; 64]).keypair().0);
  assert!(ct512 != MlKem512Ciphertext::from_bytes([0x53; MlKem512Ciphertext::LENGTH]));

  let (ek768, _) = MlKem768Seed::from_bytes([0x61; 64]).keypair();
  let ct768 = MlKem768Ciphertext::from_bytes([0x62; MlKem768Ciphertext::LENGTH]);
  assert_eq!(ek768.to_bytes(), *ek768.as_bytes());
  assert_eq!(ct768.to_bytes(), [0x62; MlKem768Ciphertext::LENGTH]);

  let (ek1024, _) = MlKem1024Seed::from_bytes([0x71; 64]).keypair();
  let ct1024 = MlKem1024Ciphertext::from_bytes([0x72; MlKem1024Ciphertext::LENGTH]);
  assert!(MlKem1024EncapsulationKey::try_from_slice(&ek1024.to_bytes()).expect("a generated key imports") == ek1024);
  assert_eq!(ct1024.to_bytes(), [0x72; MlKem1024Ciphertext::LENGTH]);
}

#[test]
fn mlkem_secret_values_redact_debug_and_require_explicit_extraction() {
  let (_, dk512) = MlKem512Seed::from_bytes([0xa1; 64]).keypair();
  let ss512 = MlKem512SharedSecret::from_bytes([0xa2; MlKem512SharedSecret::LENGTH]);
  assert_eq!(format!("{dk512:?}"), "MlKem512DecapsulationKey(****)");
  assert_eq!(format!("{ss512:?}"), "MlKem512SharedSecret(****)");
  assert_eq!(dk512.expose_secret().as_bytes(), dk512.as_bytes());
  MlKem512DecapsulationKey::try_from_slice(dk512.as_bytes()).expect("a generated key imports");
  assert!(ss512.expose_secret().as_bytes().iter().all(|&byte| byte == 0xa2));

  let (_, dk768) = MlKem768Seed::from_bytes([0xb1; 64]).keypair();
  let ss768 = MlKem768SharedSecret::from_bytes([0xb2; MlKem768SharedSecret::LENGTH]);
  assert_eq!(format!("{dk768:?}"), "MlKem768DecapsulationKey(****)");
  assert_eq!(format!("{ss768:?}"), "MlKem768SharedSecret(****)");
  assert!(
    dk768
      .ct_eq(&MlKem768DecapsulationKey::try_from_slice(dk768.as_bytes()).expect("a generated key imports"))
      .declassify()
  );
  assert!(
    !ss768
      .ct_eq(&MlKem768SharedSecret::from_bytes([0xb3; MlKem768SharedSecret::LENGTH]))
      .declassify()
  );

  let (_, dk1024) = MlKem1024Seed::from_bytes([0xc1; 64]).keypair();
  let ss1024 = MlKem1024SharedSecret::from_bytes([0xc2; MlKem1024SharedSecret::LENGTH]);
  assert_eq!(format!("{dk1024:?}"), "MlKem1024DecapsulationKey(****)");
  assert_eq!(format!("{ss1024:?}"), "MlKem1024SharedSecret(****)");
  assert_eq!(
    dk1024.display_secret().to_string().len(),
    MlKem1024DecapsulationKey::LENGTH * 2
  );
  assert_eq!(
    ss1024.display_secret().to_string().len(),
    MlKem1024SharedSecret::LENGTH * 2
  );
}
