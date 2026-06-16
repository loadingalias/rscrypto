use rscrypto::{Kem as _, MlKem768, MlKem768Ciphertext, MlKem768DecapsulationKey, MlKem768EncapsulationKey, MlKemError};
use rscrypto_fuzz::{FuzzInput, some_or_return};

pub fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let key_random: [u8; MlKem768::KEY_GENERATION_RANDOM_SIZE] = some_or_return!(input.bytes());
  let encaps_random: [u8; MlKem768::ENCAPSULATION_RANDOM_SIZE] = some_or_return!(input.bytes());

  let (ek, dk) = MlKem768::generate_keypair(|out| {
    out.copy_from_slice(&key_random);
    Ok::<(), MlKemError>(())
  })
  .expect("fixed-size ML-KEM keygen randomness must be accepted");
  let (ciphertext, encapsulated) = MlKem768::encapsulate(&ek, |out| {
    out.copy_from_slice(&encaps_random);
    Ok::<(), MlKemError>(())
  })
  .expect("generated ML-KEM encapsulation key must be valid");

  let decapsulated = MlKem768::decapsulate(&dk, &ciphertext).expect("generated ML-KEM decapsulation input must be valid");
  assert_eq!(encapsulated, decapsulated, "ML-KEM round trip mismatch");

  let parse_material = input.rest();
  let _ = MlKem768EncapsulationKey::try_from_slice(parse_material);
  let _ = MlKem768DecapsulationKey::try_from_slice(parse_material);
  let _ = MlKem768Ciphertext::try_from_slice(parse_material);

  let Some(byte_idx) = input.byte() else {
    return;
  };
  let Some(bit_idx) = input.byte() else {
    return;
  };

  let mut modified = ciphertext.to_bytes();
  modified[byte_idx as usize % MlKem768::CIPHERTEXT_SIZE] ^= 1u8 << (bit_idx & 7);
  let rejected = MlKem768::decapsulate(&dk, &MlKem768Ciphertext::from_bytes(modified))
    .expect("ML-KEM implicit rejection returns a shared secret");
  assert_ne!(encapsulated, rejected, "ML-KEM modified ciphertext accepted original secret");
}
