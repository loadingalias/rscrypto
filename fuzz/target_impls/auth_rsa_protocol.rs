use rscrypto::{RsaPublicKey, RsaSignatureProfile, RsaX509PublicKey};
use rscrypto_fuzz::{FuzzInput, some_or_return, split_at_ratio};

const RSA3072_SPKI: &[u8] = include_bytes!("../../benches/rsa_fixtures/rsa3072_spki.der");
const RSA3072_PSS_SHA256: &[u8] = include_bytes!("../../benches/rsa_fixtures/rsa3072_pss_sha256.sig");
const RSA3072_PKCS1V15_SHA256: &[u8] = include_bytes!("../../benches/rsa_fixtures/rsa3072_pkcs1v15_sha256.sig");
const MESSAGE_PSS: &[u8] = b"rscrypto RSA-PSS verification fixture";
const MESSAGE_PKCS1V15: &[u8] = b"rscrypto RSA-PKCS1-v1_5 verification fixture";

const X509_PSS_SHA256_ALGORITHM: &[u8] = &[
  0x30, 0x41, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0a, 0x30, 0x34, 0xa0, 0x0f, 0x30, 0x0d,
  0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, 0x05, 0x00, 0xa1, 0x1c, 0x30, 0x1a, 0x06, 0x09,
  0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x08, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03,
  0x04, 0x02, 0x01, 0x05, 0x00, 0xa2, 0x03, 0x02, 0x01, 0x20,
];
const X509_SHA256_WITH_RSA_ENCRYPTION: &[u8] = &[
  0x30, 0x0d, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0b, 0x05, 0x00,
];
const X509_SHA256_WITH_RSA_ENCRYPTION_MISSING_NULL: &[u8] = &[
  0x30, 0x0b, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0b,
];
const X509_SHA1_WITH_RSA_ENCRYPTION: &[u8] = &[
  0x30, 0x0d, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x05, 0x05, 0x00,
];
const X509_PSS_DEFAULT_SHA1_ALGORITHM: &[u8] = &[
  0x30, 0x0d, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0a, 0x30, 0x00,
];
const X509_MALFORMED_INDEFINITE_SEQUENCE: &[u8] = &[0x30, 0x80, 0x00, 0x00];

const JWT_ALGS: [&str; 16] = [
  "PS256", "PS384", "PS512", "RS256", "RS384", "RS512", "none", "HS256", "ES256", "EdDSA", "ps256", "", "PS1",
  "RS1", "PS256\0", "RS256 ",
];
const COSE_ALGORITHMS: [i64; 16] = [
  -37,
  -38,
  -39,
  -257,
  -258,
  -259,
  -65535,
  -7,
  0,
  1,
  37,
  i64::MAX,
  -65_536,
  i64::MIN,
  -1,
  i64::MAX - 1,
];
const TLS_SCHEMES: [u16; 19] = [
  0x0804, 0x0805, 0x0806, 0x0809, 0x080a, 0x080b, 0x0401, 0x0501, 0x0601, 0x0101, 0x0201, 0x0420,
  0x0520, 0x0620, 0x0301, 0x0203, 0x0403, 0, 0xffff,
];
const X509_ALGORITHMS: [&[u8]; 6] = [
  X509_PSS_SHA256_ALGORITHM,
  X509_SHA256_WITH_RSA_ENCRYPTION,
  X509_SHA256_WITH_RSA_ENCRYPTION_MISSING_NULL,
  X509_SHA1_WITH_RSA_ENCRYPTION,
  X509_PSS_DEFAULT_SHA1_ALGORITHM,
  X509_MALFORMED_INDEFINITE_SEQUENCE,
];

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

fn selected_signature<'a>(material: &'a [u8], full_width: &'a [u8], selector: u8) -> &'a [u8] {
  if selector & 0x80 == 0 { full_width } else { material }
}

#[inline]
fn select<T>(items: &[T], selector: u8) -> &T {
  &items[(selector as usize) % items.len()]
}

pub fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let mode = some_or_return!(input.byte());
  let selector = some_or_return!(input.byte());
  let split = some_or_return!(input.byte());
  let (signature_material, message) = split_at_ratio(input.rest(), split);

  let key = RsaPublicKey::from_spki_der(RSA3072_SPKI).expect("fuzz RSA fixture must parse");
  let x509_key = RsaX509PublicKey::from_spki_der(RSA3072_SPKI).expect("fuzz RSA X.509 fixture must parse");
  let full_width_signature = signature_candidate(signature_material, key.modulus().len());
  let signature = selected_signature(signature_material, &full_width_signature, selector);
  let mut scratch = key.public_scratch();

  match mode % 14 {
    0 => {
      key
        .verify_jwt_alg_with_scratch("PS256", MESSAGE_PSS, RSA3072_PSS_SHA256, &mut scratch)
        .expect("valid JWT PS256 fixture must verify");
    }
    1 => {
      key
        .verify_cose_algorithm_id_with_scratch(-37, MESSAGE_PSS, RSA3072_PSS_SHA256, &mut scratch)
        .expect("valid COSE PS256 fixture must verify");
    }
    2 => {
      x509_key
        .verify_signature_from_x509_algorithm_der_with_scratch(
          X509_PSS_SHA256_ALGORITHM,
          MESSAGE_PSS,
          RSA3072_PSS_SHA256,
          &mut scratch,
        )
        .expect("valid X.509 RSASSA-PSS fixture must verify");
    }
    3 => {
      x509_key
        .verify_tls13_signature_scheme_with_scratch(0x0804, MESSAGE_PSS, RSA3072_PSS_SHA256, &mut scratch)
        .expect("valid TLS 1.3 rsa_pss_rsae_sha256 fixture must verify");
    }
    4 => {
      x509_key
        .verify_tls_certificate_signature_scheme_with_scratch(
          0x0401,
          MESSAGE_PKCS1V15,
          RSA3072_PKCS1V15_SHA256,
          &mut scratch,
        )
        .expect("valid TLS certificate rsa_pkcs1_sha256 fixture must verify");
    }
    5 => {
      let _ = key.verify_jwt_alg_with_scratch(select(&JWT_ALGS, selector), message, signature, &mut scratch);
    }
    6 => {
      let _ = key.verify_cose_algorithm_id_with_scratch(
        *select(&COSE_ALGORITHMS, selector),
        message,
        signature,
        &mut scratch,
      );
    }
    7 => {
      let _ = x509_key.verify_signature_from_x509_algorithm_der_with_scratch(
        select(&X509_ALGORITHMS, selector),
        message,
        signature,
        &mut scratch,
      );
    }
    8 => {
      let _ =
        x509_key.verify_signature_from_x509_algorithm_der_with_scratch(message, MESSAGE_PSS, signature, &mut scratch);
    }
    9 => {
      let scheme = if selector & 1 == 0 {
        *select(&TLS_SCHEMES, selector)
      } else {
        u16::from_be_bytes([selector, split])
      };
      let _ = x509_key.verify_tls13_signature_scheme_with_scratch(scheme, message, signature, &mut scratch);
    }
    10 => {
      let scheme = if selector & 1 == 0 {
        *select(&TLS_SCHEMES, selector)
      } else {
        u16::from_be_bytes([selector, split])
      };
      let _ = x509_key.verify_tls_certificate_signature_scheme_with_scratch(scheme, message, signature, &mut scratch);
    }
    11 => {
      let _ = RsaSignatureProfile::from_x509_signature_algorithm_der(message);
    }
    12 => {
      let _ = RsaSignatureProfile::from_jwt_alg(select(&JWT_ALGS, selector));
      let _ = RsaSignatureProfile::from_cose_algorithm_id(*select(&COSE_ALGORITHMS, selector));
      let _ = RsaSignatureProfile::from_tls13_signature_scheme(u16::from_be_bytes([selector, split]));
      let _ = RsaSignatureProfile::from_tls_certificate_signature_scheme(u16::from_be_bytes([selector, split]));
    }
    _ => {
      let _ = RsaX509PublicKey::from_spki_der(message);
    }
  }
}
