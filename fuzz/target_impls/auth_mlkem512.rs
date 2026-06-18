use rscrypto::{Kem as _, MlKem512, MlKem512Ciphertext, MlKem512DecapsulationKey, MlKem512EncapsulationKey, MlKemError};
use rscrypto_fuzz::{FuzzInput, some_or_return};

pub fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let key_random: [u8; MlKem512::KEY_GENERATION_RANDOM_SIZE] = some_or_return!(input.bytes());
  let encaps_random: [u8; MlKem512::ENCAPSULATION_RANDOM_SIZE] = some_or_return!(input.bytes());

  let (ek, dk) = MlKem512::generate_keypair(|out| {
    out.copy_from_slice(&key_random);
    Ok::<(), MlKemError>(())
  })
  .expect("fixed-size ML-KEM keygen randomness must be accepted");
  let (ciphertext, encapsulated) = MlKem512::encapsulate(&ek, |out| {
    out.copy_from_slice(&encaps_random);
    Ok::<(), MlKemError>(())
  })
  .expect("generated ML-KEM encapsulation key must be valid");

  let decapsulated = MlKem512::decapsulate(&dk, &ciphertext).expect("generated ML-KEM decapsulation input must be valid");
  assert_eq!(encapsulated, decapsulated, "ML-KEM-512 round trip mismatch");

  let parse_material = input.rest();
  let _ = MlKem512EncapsulationKey::try_from_slice(parse_material);
  let _ = MlKem512DecapsulationKey::try_from_slice(parse_material);
  let _ = MlKem512Ciphertext::try_from_slice(parse_material);

  let Some(byte_idx) = input.byte() else {
    return;
  };
  let Some(bit_idx) = input.byte() else {
    return;
  };

  let mut modified = ciphertext.to_bytes();
  modified[byte_idx as usize % MlKem512::CIPHERTEXT_SIZE] ^= 1u8 << (bit_idx & 7);
  let rejected = MlKem512::decapsulate(&dk, &MlKem512Ciphertext::from_bytes(modified))
    .expect("ML-KEM implicit rejection returns a shared secret");
  assert_ne!(encapsulated, rejected, "ML-KEM-512 modified ciphertext accepted original secret");
}
