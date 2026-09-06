#![cfg(any(feature = "ecdsa-p256", feature = "ecdsa-p384"))]

#[cfg(feature = "ecdsa-p256")]
use p256::ecdsa::{Signature as P256OracleSignature, SigningKey as P256OracleSigningKey};
#[cfg(feature = "ecdsa-p384")]
use p384::ecdsa::{Signature as P384OracleSignature, SigningKey as P384OracleSigningKey};
#[cfg(feature = "ecdsa-p256")]
use rscrypto::{EcdsaP256PublicKey, EcdsaP256SecretKey, EcdsaP256Signature};
#[cfg(feature = "ecdsa-p384")]
use rscrypto::{EcdsaP384PublicKey, EcdsaP384SecretKey, EcdsaP384Signature};

const ID_EC_PUBLIC_KEY_OID: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01];
#[cfg(feature = "ecdsa-p256")]
const SECP256R1_OID: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07];
#[cfg(feature = "ecdsa-p384")]
const SECP384R1_OID: &[u8] = &[0x2b, 0x81, 0x04, 0x00, 0x22];

fn array_from_slice<const N: usize>(slice: &[u8]) -> [u8; N] {
  let mut out = [0u8; N];
  out.copy_from_slice(slice);
  out
}

fn der_len(len: usize) -> Vec<u8> {
  if len < 128 {
    return vec![u8::try_from(len).expect("short DER length must fit in one byte")];
  }
  let bytes = len.to_be_bytes();
  let first = bytes
    .iter()
    .position(|&byte| byte != 0)
    .expect("long DER length must contain a nonzero byte");
  let body = &bytes[first..];
  let mut out = Vec::with_capacity(body.len().strict_add(1));
  let body_len = u8::try_from(body.len()).expect("DER length-of-length must fit in one byte");
  out.push(0x80 | body_len);
  out.extend_from_slice(body);
  out
}

fn tlv(tag: u8, value: &[u8]) -> Vec<u8> {
  let capacity = value.len().strict_add(5);
  let mut out = Vec::with_capacity(capacity);
  out.push(tag);
  out.extend_from_slice(&der_len(value.len()));
  out.extend_from_slice(value);
  out
}

fn spki_der(curve_oid: &[u8], sec1: &[u8]) -> Vec<u8> {
  let mut algorithm = Vec::new();
  algorithm.extend_from_slice(&tlv(0x06, ID_EC_PUBLIC_KEY_OID));
  algorithm.extend_from_slice(&tlv(0x06, curve_oid));

  let mut bit_string = Vec::with_capacity(sec1.len().strict_add(1));
  bit_string.push(0);
  bit_string.extend_from_slice(sec1);

  let mut spki = Vec::new();
  spki.extend_from_slice(&tlv(0x30, &algorithm));
  spki.extend_from_slice(&tlv(0x03, &bit_string));
  tlv(0x30, &spki)
}

#[cfg(feature = "ecdsa-p256")]
#[test]
fn p256_verify_accepts_rustcrypto_raw_and_der_signatures() {
  let secret = [0x11u8; 32];
  let message = b"rscrypto p-256 oracle verification";
  let signing_key = P256OracleSigningKey::from_slice(&secret).expect("P-256 oracle secret must parse");
  let sec1 = EcdsaP256SecretKey::from_bytes(secret)
    .expect("P-256 rscrypto secret must parse")
    .public_key()
    .to_sec1_bytes();
  let public =
    EcdsaP256PublicKey::from_spki_der(&spki_der(SECP256R1_OID, sec1.as_slice())).expect("P-256 SPKI must parse");

  let oracle_signature: P256OracleSignature = p256::ecdsa::signature::Signer::sign(&signing_key, message);
  let raw = EcdsaP256Signature::from_bytes(array_from_slice(oracle_signature.to_bytes().as_ref()))
    .expect("P-256 raw signature must parse");
  let der = EcdsaP256Signature::from_der(oracle_signature.to_der().as_bytes()).expect("P-256 DER signature must parse");

  public
    .verify(message, &raw)
    .expect("rscrypto must verify the RustCrypto P-256 raw signature");
  public
    .verify(message, &der)
    .expect("rscrypto must verify the RustCrypto P-256 DER signature");
}

#[cfg(feature = "ecdsa-p256")]
#[test]
fn p256_verify_rejects_tampered_rustcrypto_signature() {
  let secret = [0x23u8; 32];
  let message = b"rscrypto p-256 tamper";
  let signing_key = P256OracleSigningKey::from_slice(&secret).expect("P-256 oracle secret must parse");
  let sec1 = EcdsaP256SecretKey::from_bytes(secret)
    .expect("P-256 rscrypto secret must parse")
    .public_key()
    .to_sec1_bytes();
  let public =
    EcdsaP256PublicKey::from_spki_der(&spki_der(SECP256R1_OID, sec1.as_slice())).expect("P-256 SPKI must parse");

  let oracle_signature: P256OracleSignature = p256::ecdsa::signature::Signer::sign(&signing_key, message);
  let mut bytes: [u8; EcdsaP256Signature::LENGTH] = array_from_slice(oracle_signature.to_bytes().as_ref());
  bytes[17] ^= 0x40;
  let tampered = EcdsaP256Signature::from_bytes(bytes).expect("tampered P-256 signature scalar shape must parse");

  public
    .verify(message, &tampered)
    .expect_err("rscrypto must reject a tampered P-256 signature");
  let original = EcdsaP256Signature::from_bytes(array_from_slice(oracle_signature.to_bytes().as_ref()))
    .expect("original P-256 signature must parse");
  public
    .verify(b"wrong message", &original)
    .expect_err("rscrypto must reject a P-256 signature for the wrong message");
}

#[cfg(feature = "ecdsa-p256")]
#[test]
fn p256_sign_derives_oracle_public_key_and_signature_verifies_with_rustcrypto() {
  let secret = [0x5du8; 32];
  let message = b"rscrypto p-256 signing oracle";
  let signing_key = P256OracleSigningKey::from_slice(&secret).expect("P-256 oracle secret must parse");
  let verifying_key = signing_key.verifying_key();
  let rs_secret = EcdsaP256SecretKey::from_bytes(secret).expect("P-256 rscrypto secret must parse");
  let rs_public = rs_secret.public_key();
  let rs_signature = rs_secret
    .try_sign(message)
    .expect("P-256 rscrypto signing must succeed");
  let oracle_signature =
    P256OracleSignature::from_slice(rs_signature.as_bytes()).expect("P-256 oracle signature must parse");

  rs_public
    .verify(message, &rs_signature)
    .expect("rscrypto must verify its P-256 signature");
  p256::ecdsa::signature::Verifier::verify(verifying_key, message, &oracle_signature)
    .expect("RustCrypto must verify rscrypto P-256 signature");
}

#[cfg(feature = "ecdsa-p256")]
#[test]
fn p256_blinded_sign_matches_deterministic_signature_and_rustcrypto_oracle() {
  let secret = [0x7bu8; 32];
  let message = b"rscrypto p-256 blinded signing oracle";
  let rs_secret = EcdsaP256SecretKey::from_bytes(secret).expect("P-256 rscrypto secret must parse");
  let rs_public = rs_secret.public_key();
  let oracle_public = P256OracleSigningKey::from_slice(&secret)
    .expect("P-256 oracle secret must parse")
    .verifying_key()
    .to_owned();

  let deterministic = rs_secret
    .try_sign(message)
    .expect("P-256 deterministic signing must succeed");
  let blinded = rs_secret
    .try_sign_blinded_with(message, |blind| {
      blind.fill(0xa6);
      Ok::<(), core::convert::Infallible>(())
    })
    .expect("P-256 blinded signing must succeed");
  let oracle_signature =
    P256OracleSignature::from_slice(blinded.as_bytes()).expect("P-256 oracle signature must parse");

  assert_eq!(deterministic, blinded);
  rs_public
    .verify(message, &blinded)
    .expect("rscrypto must verify its blinded P-256 signature");
  p256::ecdsa::signature::Verifier::verify(&oracle_public, message, &oracle_signature)
    .expect("RustCrypto must verify rscrypto blinded P-256 signature");
}

#[cfg(feature = "ecdsa-p384")]
#[test]
fn p384_verify_accepts_rustcrypto_raw_and_der_signatures() {
  let secret = [0x31u8; 48];
  let message = b"rscrypto p-384 oracle verification";
  let signing_key = P384OracleSigningKey::from_slice(&secret).expect("P-384 oracle secret must parse");
  let sec1 = EcdsaP384SecretKey::from_bytes(secret)
    .expect("P-384 rscrypto secret must parse")
    .public_key()
    .to_sec1_bytes();
  let public =
    EcdsaP384PublicKey::from_spki_der(&spki_der(SECP384R1_OID, sec1.as_slice())).expect("P-384 SPKI must parse");

  let oracle_signature: P384OracleSignature = p384::ecdsa::signature::Signer::sign(&signing_key, message);
  let raw = EcdsaP384Signature::from_bytes(array_from_slice(oracle_signature.to_bytes().as_ref()))
    .expect("P-384 raw signature must parse");
  let der = EcdsaP384Signature::from_der(oracle_signature.to_der().as_bytes()).expect("P-384 DER signature must parse");

  public
    .verify(message, &raw)
    .expect("rscrypto must verify the RustCrypto P-384 raw signature");
  public
    .verify(message, &der)
    .expect("rscrypto must verify the RustCrypto P-384 DER signature");
}

#[cfg(feature = "ecdsa-p384")]
#[test]
fn p384_verify_rejects_tampered_rustcrypto_signature() {
  let secret = [0x44u8; 48];
  let message = b"rscrypto p-384 tamper";
  let signing_key = P384OracleSigningKey::from_slice(&secret).expect("P-384 oracle secret must parse");
  let sec1 = EcdsaP384SecretKey::from_bytes(secret)
    .expect("P-384 rscrypto secret must parse")
    .public_key()
    .to_sec1_bytes();
  let public =
    EcdsaP384PublicKey::from_spki_der(&spki_der(SECP384R1_OID, sec1.as_slice())).expect("P-384 SPKI must parse");

  let oracle_signature: P384OracleSignature = p384::ecdsa::signature::Signer::sign(&signing_key, message);
  let mut bytes: [u8; EcdsaP384Signature::LENGTH] = array_from_slice(oracle_signature.to_bytes().as_ref());
  bytes[29] ^= 0x10;
  let tampered = EcdsaP384Signature::from_bytes(bytes).expect("tampered P-384 signature scalar shape must parse");

  public
    .verify(message, &tampered)
    .expect_err("rscrypto must reject a tampered P-384 signature");
  let original = EcdsaP384Signature::from_bytes(array_from_slice(oracle_signature.to_bytes().as_ref()))
    .expect("original P-384 signature must parse");
  public
    .verify(b"wrong message", &original)
    .expect_err("rscrypto must reject a P-384 signature for the wrong message");
}

#[cfg(feature = "ecdsa-p384")]
#[test]
fn p384_sign_derives_oracle_public_key_and_signature_verifies_with_rustcrypto() {
  let secret = [0x6eu8; 48];
  let message = b"rscrypto p-384 signing oracle";
  let signing_key = P384OracleSigningKey::from_slice(&secret).expect("P-384 oracle secret must parse");
  let verifying_key = signing_key.verifying_key();
  let rs_secret = EcdsaP384SecretKey::from_bytes(secret).expect("P-384 rscrypto secret must parse");
  let rs_public = rs_secret.public_key();
  let rs_signature = rs_secret
    .try_sign(message)
    .expect("P-384 rscrypto signing must succeed");
  let oracle_signature =
    P384OracleSignature::from_slice(rs_signature.as_bytes()).expect("P-384 oracle signature must parse");

  rs_public
    .verify(message, &rs_signature)
    .expect("rscrypto must verify its P-384 signature");
  p384::ecdsa::signature::Verifier::verify(verifying_key, message, &oracle_signature)
    .expect("RustCrypto must verify rscrypto P-384 signature");
}

#[cfg(feature = "ecdsa-p384")]
#[test]
fn p384_blinded_sign_matches_deterministic_signature_and_rustcrypto_oracle() {
  let secret = [0x52u8; 48];
  let message = b"rscrypto p-384 blinded signing oracle";
  let rs_secret = EcdsaP384SecretKey::from_bytes(secret).expect("P-384 rscrypto secret must parse");
  let rs_public = rs_secret.public_key();
  let oracle_public = P384OracleSigningKey::from_slice(&secret)
    .expect("P-384 oracle secret must parse")
    .verifying_key()
    .to_owned();

  let deterministic = rs_secret
    .try_sign(message)
    .expect("P-384 deterministic signing must succeed");
  let blinded = rs_secret
    .try_sign_blinded_with(message, |blind| {
      blind.fill(0xc3);
      Ok::<(), core::convert::Infallible>(())
    })
    .expect("P-384 blinded signing must succeed");
  let oracle_signature =
    P384OracleSignature::from_slice(blinded.as_bytes()).expect("P-384 oracle signature must parse");

  assert_eq!(deterministic, blinded);
  rs_public
    .verify(message, &blinded)
    .expect("rscrypto must verify its blinded P-384 signature");
  p384::ecdsa::signature::Verifier::verify(&oracle_public, message, &oracle_signature)
    .expect("RustCrypto must verify rscrypto blinded P-384 signature");
}
