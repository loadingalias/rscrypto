use rscrypto::{RsaPublicKey, RsaPublicKeyPolicy};
use rscrypto_fuzz::{FuzzInput, some_or_return};

pub fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let mode = some_or_return!(input.byte());
  let der = input.rest();

  let parsed = match mode % 4 {
    0 => RsaPublicKey::from_pkcs1_der(der),
    1 => RsaPublicKey::from_spki_der(der),
    2 => RsaPublicKey::from_pkcs1_der_with_policy(der, &RsaPublicKeyPolicy::modern_verification()),
    _ => RsaPublicKey::from_spki_der_with_policy(
      der,
      &RsaPublicKeyPolicy::legacy_verification().allow_legacy_small_exponents(),
    ),
  };

  if let Ok(key) = parsed {
    let mut representative = key.modulus().to_vec();
    for byte in representative.iter_mut().rev() {
      if *byte != 0 {
        *byte = byte.strict_sub(1);
        break;
      }
      *byte = 0xff;
    }

    let mut out = vec![0u8; key.modulus().len()];
    let mut scratch = key.public_scratch();
    key
      .public_operation_with_scratch(&representative, &mut out, &mut scratch)
      .expect("modulus - 1 representative must be accepted");
  }
}
