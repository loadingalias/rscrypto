#![cfg(feature = "rsa")]

use rscrypto::{
  RsaPkcs1v15Profile, RsaPssProfile, RsaPublicKey, RsaSignatureProfile, RsaX509PublicKey, VerificationError,
};

const RSA3072_SPKI: &[u8] = include_bytes!("../benches/rsa_fixtures/rsa3072_spki.der");
const RSA3072_PSS_SHA256: &[u8] = include_bytes!("../benches/rsa_fixtures/rsa3072_pss_sha256.sig");
const RSA3072_PKCS1V15_SHA256: &[u8] = include_bytes!("../benches/rsa_fixtures/rsa3072_pkcs1v15_sha256.sig");

const PSS_MESSAGE: &[u8] = b"rscrypto RSA-PSS verification fixture";
const PKCS1V15_MESSAGE: &[u8] = b"rscrypto RSA-PKCS1-v1_5 verification fixture";

fn assert_rejects(result: Result<(), VerificationError>) {
  assert_eq!(result, Err(VerificationError::new()));
}

#[test]
fn rsa_signatures_reject_padding_profile_confusion() {
  let key = RsaPublicKey::from_spki_der(RSA3072_SPKI).expect("RSA-3072 SPKI fixture must parse");
  let pss = RsaSignatureProfile::pss(RsaPssProfile::Sha256);
  let pkcs1v15 = RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256);

  key
    .verify_signature(pss, PSS_MESSAGE, RSA3072_PSS_SHA256)
    .expect("RSA-PSS fixture signature must verify with its matching profile");
  key
    .verify_signature(pkcs1v15, PKCS1V15_MESSAGE, RSA3072_PKCS1V15_SHA256)
    .expect("PKCS#1 v1.5 fixture signature must verify with its matching profile");

  assert_rejects(key.verify_signature(pkcs1v15, PSS_MESSAGE, RSA3072_PSS_SHA256));
  assert_rejects(key.verify_signature(pss, PKCS1V15_MESSAGE, RSA3072_PKCS1V15_SHA256));
  assert_rejects(key.verify_pss(RsaPssProfile::Sha256, PSS_MESSAGE, RSA3072_PKCS1V15_SHA256));
  assert_rejects(key.verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, PKCS1V15_MESSAGE, RSA3072_PSS_SHA256));
}

#[test]
fn rsa_protocol_scheme_helpers_reject_signature_profile_confusion() {
  let key = RsaX509PublicKey::from_spki_der(RSA3072_SPKI).expect("RSA-3072 X.509 SPKI fixture must parse");
  let raw_key = RsaPublicKey::from_spki_der(RSA3072_SPKI).expect("RSA-3072 SPKI fixture must parse");

  key
    .verify_tls13_signature_scheme(0x0804, PSS_MESSAGE, RSA3072_PSS_SHA256)
    .expect("TLS 1.3 rsa_pss_rsae_sha256 fixture must verify");
  key
    .verify_tls_certificate_signature_scheme(0x0401, PKCS1V15_MESSAGE, RSA3072_PKCS1V15_SHA256)
    .expect("TLS rsa_pkcs1_sha256 certificate fixture must verify");

  assert_rejects(key.verify_tls13_signature_scheme(0x0804, PKCS1V15_MESSAGE, RSA3072_PKCS1V15_SHA256));
  assert_rejects(key.verify_tls_certificate_signature_scheme(0x0401, PSS_MESSAGE, RSA3072_PSS_SHA256));
  assert_rejects(raw_key.verify_cose_algorithm_id(-37, PKCS1V15_MESSAGE, RSA3072_PKCS1V15_SHA256));
  assert_rejects(raw_key.verify_cose_algorithm_id(-257, PSS_MESSAGE, RSA3072_PSS_SHA256));
}
