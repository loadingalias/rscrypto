#![cfg(feature = "ml-kem")]

use rscrypto::{
  Kem, MlKem512, MlKem512Ciphertext, MlKem512DecapsulationKey, MlKem512EncapsulationKey, MlKem768, MlKem768Ciphertext,
  MlKem768DecapsulationKey, MlKem768EncapsulationKey, MlKem1024, MlKem1024Ciphertext, MlKem1024DecapsulationKey,
  MlKem1024EncapsulationKey, MlKemError,
};
use tiny_keccak::{Hasher as _, Shake, Xof as _};

mod common;
use common::decode_hex_vec;

// C2SP CCTV ML-KEM vectors; provenance in testdata/mlkem/cctv/README.md.
macro_rules! cctv {
  ($name:literal) => {
    include_str!(concat!("../testdata/mlkem/cctv/", $name))
  };
}

/// Read `name = hex` lines into their values, in file order.
fn fields(text: &str, name: &str) -> Vec<Vec<u8>> {
  text
    .lines()
    .filter_map(|line| line.split_once(" = "))
    .filter(|(key, _)| *key == name)
    .map(|(_, value)| decode_hex_vec(value))
    .collect()
}

macro_rules! cctv_suite {
  (
    $profile:ty, $ek:ty, $dk:ty, $ct:ty,
    modulus = $modulus:literal, strcmp = $strcmp:literal, modulus_count = $modulus_count:literal $(,)?
  ) => {
    /// Every unreduced coefficient value in every position must fail the
    /// FIPS 203 section 7.2 modulus check.
    #[test]
    fn modulus_check_rejects_every_unreduced_coefficient() {
      let mut count = 0usize;
      for (line_number, line) in cctv!($modulus).lines().enumerate() {
        let key = decode_hex_vec(line);
        assert_eq!(key.len(), <$profile>::ENCAPSULATION_KEY_SIZE, "line {line_number}");
        assert!(
          <$ek>::try_from_slice(&key).is_err(),
          "line {line_number}: unreduced encapsulation key accepted"
        );
        count = count.strict_add(1);
      }
      assert_eq!(count, $modulus_count);
    }

    /// The re-encrypted ciphertext differs only after a zero byte, so a
    /// `strcmp`-style comparison would accept it; implicit rejection must fire.
    #[test]
    fn implicit_rejection_compares_every_ciphertext_byte() {
      let text = cctv!($strcmp);
      let keys = fields(text, "dk");
      let ciphertexts = fields(text, "c");
      let shared = fields(text, "K");
      assert!(!keys.is_empty());
      assert_eq!(keys.len(), ciphertexts.len());
      assert_eq!(keys.len(), shared.len());
      for ((key, ciphertext), expected) in keys.iter().zip(&ciphertexts).zip(&shared) {
        let key = <$dk>::try_from_slice(key).expect("CCTV decapsulation key must import");
        let ciphertext = <$ct>::try_from_slice(ciphertext).expect("CCTV ciphertext must import");
        let actual = <$profile>::decapsulate(&key, &ciphertext).expect("decapsulation must succeed");
        assert!(actual.expose_secret().as_bytes() == expected.as_slice());
      }
    }
  };
}

mod mlkem512 {
  use super::*;
  cctv_suite!(
    MlKem512,
    MlKem512EncapsulationKey,
    MlKem512DecapsulationKey,
    MlKem512Ciphertext,
    modulus = "modulus-ML-KEM-512.txt",
    strcmp = "strcmp-ML-KEM-512.txt",
    modulus_count = 775,
  );
}

mod mlkem768 {
  use super::*;
  cctv_suite!(
    MlKem768,
    MlKem768EncapsulationKey,
    MlKem768DecapsulationKey,
    MlKem768Ciphertext,
    modulus = "modulus-ML-KEM-768.txt",
    strcmp = "strcmp-ML-KEM-768.txt",
    modulus_count = 780,
  );
}

mod mlkem1024 {
  use super::*;
  cctv_suite!(
    MlKem1024,
    MlKem1024EncapsulationKey,
    MlKem1024DecapsulationKey,
    MlKem1024Ciphertext,
    modulus = "modulus-ML-KEM-1024.txt",
    strcmp = "strcmp-ML-KEM-1024.txt",
    modulus_count = 1040,
  );
}

/// Accumulated ML-KEM-768 vectors from the Go standard library's final
/// FIPS 203 implementation (`crypto/mlkem` `TestAccumulated`).
///
/// One SHAKE128 stream with empty input supplies, per iteration, the 64-byte
/// `d || z` seed, the 32-byte encapsulation message, and a random ciphertext.
/// A second SHAKE128 absorbs the encapsulation key, ciphertext, shared secret,
/// and the implicit-rejection secret for the random ciphertext.
fn accumulated_mlkem768(iterations: usize) -> [u8; 32] {
  let mut source = Shake::v128();
  let mut accumulator = Shake::v128();
  for _ in 0..iterations {
    let mut seed = [0u8; MlKem768::KEY_GENERATION_RANDOM_SIZE];
    source.squeeze(&mut seed);
    let (ek, dk) = MlKem768::generate_keypair(|out| {
      out.copy_from_slice(&seed);
      Ok::<(), MlKemError>(())
    })
    .expect("key generation");
    accumulator.update(ek.as_bytes());

    let mut message = [0u8; MlKem768::ENCAPSULATION_RANDOM_SIZE];
    source.squeeze(&mut message);
    let (ciphertext, shared) = MlKem768::encapsulate(&ek, |out| {
      out.copy_from_slice(&message);
      Ok::<(), MlKemError>(())
    })
    .expect("encapsulation");
    accumulator.update(ciphertext.as_bytes());
    accumulator.update(shared.expose_secret().as_bytes());

    let decapsulated = MlKem768::decapsulate(&dk, &ciphertext).expect("decapsulation");
    assert!(decapsulated.expose_secret().as_bytes() == shared.expose_secret().as_bytes());

    let mut random = [0u8; MlKem768::CIPHERTEXT_SIZE];
    source.squeeze(&mut random);
    let random = MlKem768Ciphertext::try_from_slice(&random).expect("any ciphertext bytes are well-formed");
    let rejected = MlKem768::decapsulate(&dk, &random).expect("decapsulation");
    accumulator.update(rejected.expose_secret().as_bytes());
  }
  let mut digest = [0u8; 32];
  accumulator.finalize(&mut digest);
  digest
}

#[test]
fn mlkem768_matches_go_accumulated_10k() {
  let expected = decode_hex_vec("8a518cc63da366322a8e7a818c7a0d63483cb3528d34a4cf42f35d5ad73f22fc");
  assert!(accumulated_mlkem768(10_000).as_slice() == expected.as_slice());
}

#[test]
#[ignore = "one million ML-KEM-768 round trips; run explicitly for release evidence"]
fn mlkem768_matches_go_accumulated_1m() {
  let expected = decode_hex_vec("424bf8f0e8ae99b78d788a6e2e8e9cdaf9773fc0c08a6f433507cb559edfd0f0");
  assert!(accumulated_mlkem768(1_000_000).as_slice() == expected.as_slice());
}
