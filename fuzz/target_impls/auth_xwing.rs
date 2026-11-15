use rscrypto::{
  Kem as _, XWing, XWingCiphertext, XWingDecapsulationKey, XWingEncapsulationKey, XWingError,
  XWingPreparedDecapsulationKey,
};
use rscrypto_fuzz::{FuzzInput, some_or_return};
use x_wing::{Decapsulator as _, KeyExport as _, TryDecapsulate as _};

/// X-Wing against RustCrypto `x-wing`. Its `DecapsulationKeyRejectNonContrib`
/// rejects an all-zero X25519 output, as rscrypto does.
pub(super) fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let seed: [u8; XWing::DECAPSULATION_KEY_SIZE] = some_or_return!(input.bytes());
  let randomness: [u8; XWing::ENCAPSULATION_RANDOM_SIZE] = some_or_return!(input.bytes());
  let fill = |out: &mut [u8]| {
    out.copy_from_slice(&randomness);
    Ok::<(), XWingError>(())
  };

  let decapsulation_key = XWingDecapsulationKey::from_bytes(seed);
  let prepared = decapsulation_key.prepare();
  let oracle = x_wing::DecapsulationKeyRejectNonContrib::from(seed);
  let oracle_key = oracle.encapsulation_key();
  let encapsulation_key = prepared.encapsulation_key();
  assert_eq!(
    encapsulation_key.as_bytes().as_slice(),
    oracle_key.to_bytes().as_slice(),
    "X-Wing encapsulation key mismatch"
  );

  let (ciphertext, shared_secret) =
    XWing::encapsulate(&encapsulation_key, fill).expect("a derived X-Wing key must be valid");
  let (oracle_ciphertext, oracle_secret) = oracle_key.encapsulate_deterministic(&randomness.into());
  assert_eq!(ciphertext.as_bytes().as_slice(), oracle_ciphertext.as_slice());
  assert_eq!(shared_secret.as_bytes().as_slice(), oracle_secret.as_slice());
  let decapsulated = prepared.decapsulate(&ciphertext).expect("X-Wing round trip");
  assert!(
    shared_secret.ct_eq(&decapsulated).declassify(),
    "X-Wing round trip mismatch"
  );

  // A changed ciphertext gets ML-KEM implicit rejection or, for a small-order
  // X25519 part, an error: the same outcome on both paths and in the oracle.
  let mutation = some_or_return!(input.bit_mutation());
  let mut modified = ciphertext.to_bytes();
  mutation.apply(&mut modified);
  decapsulates_as_oracle(&decapsulation_key, &prepared, &oracle, modified);

  let parse_material = input.rest();
  if let Ok(candidate) = <[u8; XWing::CIPHERTEXT_SIZE]>::try_from(parse_material) {
    decapsulates_as_oracle(&decapsulation_key, &prepared, &oracle, candidate);
  }
  let parsed = XWingEncapsulationKey::try_from_slice(parse_material);
  let oracle_parsed = x_wing::EncapsulationKey::try_from(parse_material);
  assert_eq!(
    parsed.is_ok(),
    oracle_parsed.is_ok(),
    "X-Wing encapsulation key validation mismatch"
  );
  if let (Ok(parsed), Ok(oracle_parsed)) = (parsed, oracle_parsed) {
    // The oracle encapsulates to a small-order X25519 part, which rscrypto rejects.
    if let Ok((ciphertext, shared_secret)) = XWing::encapsulate(&parsed, fill) {
      let (oracle_ciphertext, oracle_secret) = oracle_parsed.encapsulate_deterministic(&randomness.into());
      assert_eq!(ciphertext.as_bytes().as_slice(), oracle_ciphertext.as_slice());
      assert_eq!(shared_secret.as_bytes().as_slice(), oracle_secret.as_slice());
    }
  }
  let _decapsulation_key_result = XWingDecapsulationKey::try_from_slice(parse_material);
  let _ciphertext_result = XWingCiphertext::try_from_slice(parse_material);
}

fn decapsulates_as_oracle(
  decapsulation_key: &XWingDecapsulationKey,
  prepared: &XWingPreparedDecapsulationKey,
  oracle: &x_wing::DecapsulationKeyRejectNonContrib,
  ciphertext: [u8; XWing::CIPHERTEXT_SIZE],
) {
  let typed = XWingCiphertext::from_bytes(ciphertext);
  let compact = XWing::decapsulate(decapsulation_key, &typed);
  let reused = prepared.decapsulate(&typed);
  let expected = oracle.try_decapsulate(&ciphertext.into());
  match expected {
    Ok(expected) => {
      let compact = compact.expect("the oracle accepted this X-Wing ciphertext");
      assert_eq!(
        compact.as_bytes().as_slice(),
        expected.as_slice(),
        "X-Wing decapsulation mismatch"
      );
      let reused = reused.expect("the oracle accepted this X-Wing ciphertext");
      assert_eq!(
        reused.as_bytes(),
        compact.as_bytes(),
        "X-Wing prepared decapsulation mismatch"
      );
    }
    Err(_) => {
      assert_eq!(compact.err(), Some(XWingError::InvalidCiphertext));
      assert_eq!(reused.err(), Some(XWingError::InvalidCiphertext));
    }
  }
}
