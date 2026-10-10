use rscrypto::{
  Kem as _, MlKem512, MlKem512Ciphertext, MlKem512DecapsulationKey, MlKem512EncapsulationKey, MlKem512Seed, MlKemError,
};
use rscrypto_fuzz::{FuzzInput, some_or_return};

pub(super) fn run(data: &[u8]) {
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

  let decapsulated =
    MlKem512::decapsulate(&dk, &ciphertext).expect("generated ML-KEM decapsulation input must be valid");
  assert!(
    encapsulated.ct_eq(&decapsulated).declassify(),
    "ML-KEM-512 round trip mismatch"
  );

  // Each fixed-length PKIX form has one encoding per key: a generated encoding
  // round-trips, and an accepted single-bit change must re-encode to itself.
  let seed = MlKem512Seed::from_bytes(key_random);
  let mut spki = ek.to_spki_der();
  let mut seed_der = [0; MlKem512Seed::PKCS8_DER_LENGTH];
  seed.to_pkcs8_der_into(&mut seed_der);
  let mut expanded_der = [0; MlKem512DecapsulationKey::PKCS8_DER_LENGTH];
  dk.to_pkcs8_der_into(&mut expanded_der);
  assert_eq!(
    MlKem512EncapsulationKey::from_spki_der(&spki)
      .expect("SPKI round trip")
      .as_bytes(),
    ek.as_bytes()
  );
  for der in [seed_der.as_slice(), expanded_der.as_slice()] {
    let imported = MlKem512DecapsulationKey::from_pkcs8_der(der).expect("PKCS #8 round trip");
    assert_eq!(imported.as_bytes(), dk.as_bytes());
  }
  flip_bit(&mut spki, &encaps_random);
  flip_bit(&mut seed_der, &encaps_random);
  flip_bit(&mut expanded_der, &encaps_random);

  let parse_material = input.rest();
  if let Ok(parsed) = MlKem512EncapsulationKey::from_spki_der(&spki) {
    assert_eq!(parsed.to_spki_der(), spki, "accepted SPKI must be canonical");
  }
  for candidate in [seed_der.as_slice(), expanded_der.as_slice(), parse_material] {
    let parsed_seed = MlKem512Seed::from_pkcs8_der(candidate);
    if let Ok(parsed_seed) = &parsed_seed
      && candidate.len() == MlKem512Seed::PKCS8_DER_LENGTH
    {
      let mut encoded = [0; MlKem512Seed::PKCS8_DER_LENGTH];
      parsed_seed.to_pkcs8_der_into(&mut encoded);
      assert_eq!(encoded.as_slice(), candidate, "accepted seed form must be canonical");
    }
    if let Ok(parsed) = MlKem512DecapsulationKey::from_pkcs8_der(candidate) {
      if candidate.len() == MlKem512DecapsulationKey::PKCS8_DER_LENGTH {
        let mut encoded = [0; MlKem512DecapsulationKey::PKCS8_DER_LENGTH];
        parsed.to_pkcs8_der_into(&mut encoded);
        assert_eq!(
          encoded.as_slice(),
          candidate,
          "accepted expanded form must be canonical"
        );
      }
      if let Ok(parsed_seed) = parsed_seed {
        assert_eq!(
          parsed_seed.keypair().1.as_bytes(),
          parsed.as_bytes(),
          "both imports name one key"
        );
      }
    }
  }
  let _spki_result = MlKem512EncapsulationKey::from_spki_der(parse_material);
  let _encapsulation_key_result = MlKem512EncapsulationKey::try_from_slice(parse_material);
  let _decapsulation_key_result = MlKem512DecapsulationKey::try_from_slice(parse_material);
  let _ciphertext_result = MlKem512Ciphertext::try_from_slice(parse_material);

  let mutation = some_or_return!(input.bit_mutation());
  let mut modified = ciphertext.to_bytes();
  mutation.apply(&mut modified);
  let rejected = MlKem512::decapsulate(&dk, &MlKem512Ciphertext::from_bytes(modified))
    .expect("ML-KEM implicit rejection returns a shared secret");
  assert!(
    !encapsulated.ct_eq(&rejected).declassify(),
    "ML-KEM-512 modified ciphertext accepted original secret"
  );
}

/// Flip one bit of `bytes`, chosen by `selector`.
fn flip_bit(bytes: &mut [u8], selector: &[u8; MlKem512::ENCAPSULATION_RANDOM_SIZE]) {
  let Some(position) = usize::from(u16::from_le_bytes([selector[0], selector[1]])).checked_rem(bytes.len()) else {
    return;
  };
  if let Some(byte) = bytes.get_mut(position) {
    *byte ^= 1u8.rotate_left(u32::from(selector[2]));
  }
}
