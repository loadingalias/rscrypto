use rscrypto::{RsaPkcs1v15Profile, RsaPssProfile, RsaPublicKey};
use rscrypto_fuzz::{FuzzInput, some_or_return, split_at_ratio};

const RSA3072_SPKI: &[u8] = include_bytes!("../../benches/rsa_fixtures/rsa3072_spki.der");
const RSA3072_PSS_SHA256: &[u8] = include_bytes!("../../benches/rsa_fixtures/rsa3072_pss_sha256.sig");
const RSA3072_PKCS1V15_SHA256: &[u8] = include_bytes!("../../benches/rsa_fixtures/rsa3072_pkcs1v15_sha256.sig");
const MESSAGE_PSS: &[u8] = b"rscrypto RSA-PSS verification fixture";
const MESSAGE_PKCS1V15: &[u8] = b"rscrypto RSA-PKCS1-v1_5 verification fixture";

fn signature_candidate(material: &[u8], len: usize) -> Vec<u8> {
  let mut out = vec![0u8; len];
  if material.is_empty() {
    return out;
  }

  for (index, byte) in out.iter_mut().enumerate() {
    *byte = material[index % material.len()];
  }
  out
}

pub fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let mode = some_or_return!(input.byte());
  let split = some_or_return!(input.byte());
  let (signature_material, message) = split_at_ratio(input.rest(), split);

  let key = RsaPublicKey::from_spki_der(RSA3072_SPKI).expect("fuzz RSA fixture must parse");
  let mut scratch = key.public_scratch();

  match mode % 6 {
    0 => {
      key
        .verify_pss_with_scratch(RsaPssProfile::Sha256, MESSAGE_PSS, RSA3072_PSS_SHA256, &mut scratch)
        .expect("valid RSA-PSS fixture must verify");
    }
    1 => {
      key
        .verify_pkcs1v15_with_scratch(
          RsaPkcs1v15Profile::Sha256,
          MESSAGE_PKCS1V15,
          RSA3072_PKCS1V15_SHA256,
          &mut scratch,
        )
        .expect("valid RSA-PKCS1-v1_5 fixture must verify");
    }
    2 => {
      let signature = signature_candidate(signature_material, key.modulus().len());
      let _ = key.verify_pss_with_scratch(RsaPssProfile::Sha256, message, &signature, &mut scratch);
    }
    3 => {
      let signature = signature_candidate(signature_material, key.modulus().len());
      let _ = key.verify_pkcs1v15_with_scratch(RsaPkcs1v15Profile::Sha256, message, &signature, &mut scratch);
    }
    4 => {
      let _ = key.verify_pss_with_scratch(RsaPssProfile::Sha384, message, signature_material, &mut scratch);
    }
    _ => {
      let _ = key.verify_pkcs1v15_with_scratch(RsaPkcs1v15Profile::Sha512, message, signature_material, &mut scratch);
    }
  }
}
