#![cfg(all(
  feature = "sha2",
  feature = "hmac",
  feature = "hkdf",
  feature = "pbkdf2",
  feature = "aes-gcm",
  feature = "chacha20poly1305",
  feature = "ed25519",
  feature = "rsa",
))]

use core::num::NonZeroU32;

use ring::{aead as ring_aead, digest as ring_digest, hkdf as ring_hkdf, hmac as ring_hmac, pbkdf2 as ring_pbkdf2};
use rscrypto::{
  Aes256Gcm, Aes256GcmKey, ChaCha20Poly1305, ChaCha20Poly1305Key, Ed25519SecretKey, HkdfSha256, HmacSha256,
  Pbkdf2Sha256, RsaPkcs1v15Profile, RsaPssProfile, RsaPublicKey, Sha256, aead::Nonce96,
};

const DATA: &[u8] = b"ring migration equivalence data";
const KEY_32: [u8; 32] = [0x42; 32];
const NONCE_12: [u8; 12] = [0x31; 12];
const AAD: &[u8] = b"ring migration aad";

const RSA3072_SPKI: &[u8] = include_bytes!("../benches/rsa_fixtures/rsa3072_spki.der");
const RSA3072_PSS_SHA256: &[u8] = include_bytes!("../benches/rsa_fixtures/rsa3072_pss_sha256.sig");
const RSA3072_PKCS1V15_SHA256: &[u8] = include_bytes!("../benches/rsa_fixtures/rsa3072_pkcs1v15_sha256.sig");
const MESSAGE_PSS: &[u8] = b"rscrypto RSA-PSS verification fixture";
const MESSAGE_PKCS1V15: &[u8] = b"rscrypto RSA-PKCS1-v1_5 verification fixture";

struct RingHkdfLen(usize);

impl ring_hkdf::KeyType for RingHkdfLen {
  fn len(&self) -> usize {
    self.0
  }
}

#[test]
fn test_ring_digest_hmac_hkdf_and_pbkdf2_migration_examples_are_byte_equivalent() {
  let ring_digest = ring_digest::digest(&ring_digest::SHA256, DATA);
  assert_eq!(Sha256::digest(DATA).as_slice(), ring_digest.as_ref());

  let ring_hmac_key = ring_hmac::Key::new(ring_hmac::HMAC_SHA256, &KEY_32);
  let ring_hmac = ring_hmac::sign(&ring_hmac_key, DATA);
  assert_eq!(HmacSha256::mac(&KEY_32, DATA).as_slice(), ring_hmac.as_ref());

  let salt = b"ring migration salt";
  let ikm = b"ring migration input key material";
  let info = b"ring migration context";
  let mut ring_okm = [0u8; 42];
  ring_hkdf::Salt::new(ring_hkdf::HKDF_SHA256, salt)
    .extract(ikm)
    .expand(&[info], RingHkdfLen(ring_okm.len()))
    .unwrap()
    .fill(&mut ring_okm)
    .unwrap();

  let mut ours_okm = [0u8; 42];
  HkdfSha256::new(salt, ikm).expand(info, &mut ours_okm).unwrap();
  assert_eq!(ours_okm, ring_okm);

  let iterations = NonZeroU32::new(600_000).unwrap();
  let mut ring_pbkdf2 = [0u8; 32];
  ring_pbkdf2::derive(
    ring_pbkdf2::PBKDF2_HMAC_SHA256,
    iterations,
    salt,
    b"ring migration password",
    &mut ring_pbkdf2,
  );

  let mut ours_pbkdf2 = [0u8; 32];
  Pbkdf2Sha256::derive_key(b"ring migration password", salt, iterations.get(), &mut ours_pbkdf2).unwrap();
  assert_eq!(ours_pbkdf2, ring_pbkdf2);
}

#[test]
fn test_ring_aead_migration_examples_are_byte_equivalent() {
  let ring_aes = ring_aead_seal(&ring_aead::AES_256_GCM, &KEY_32, DATA);
  let aes = Aes256Gcm::new(&Aes256GcmKey::from_bytes(KEY_32));
  let nonce = Nonce96::from_bytes(NONCE_12);
  let mut ours_aes = vec![0u8; DATA.len() + 16];
  aes.encrypt(&nonce, AAD, DATA, &mut ours_aes).unwrap();
  assert_eq!(ours_aes, ring_aes);

  let ring_chacha = ring_aead_seal(&ring_aead::CHACHA20_POLY1305, &KEY_32, DATA);
  let chacha = ChaCha20Poly1305::new(&ChaCha20Poly1305Key::from_bytes(KEY_32));
  let mut ours_chacha = vec![0u8; DATA.len() + 16];
  chacha.encrypt(&nonce, AAD, DATA, &mut ours_chacha).unwrap();
  assert_eq!(ours_chacha, ring_chacha);
}

#[test]
fn test_ring_ed25519_and_rsa_verify_migration_examples_are_compatible() {
  use ring::signature::KeyPair as _;

  let seed = [0x13; 32];
  let ring_ed25519 = ring::signature::Ed25519KeyPair::from_seed_unchecked(&seed).unwrap();
  let ours_ed25519 = Ed25519SecretKey::from_bytes(seed);
  let ours_public = ours_ed25519.public_key();
  let ours_signature = ours_ed25519.sign(DATA);

  assert_eq!(ours_public.as_bytes(), ring_ed25519.public_key().as_ref());
  assert_eq!(ours_signature.as_bytes(), ring_ed25519.sign(DATA).as_ref());

  ring::signature::UnparsedPublicKey::new(&ring::signature::ED25519, ours_public.as_bytes())
    .verify(DATA, ours_signature.as_bytes())
    .unwrap();
  ours_public.verify(DATA, &ours_signature).unwrap();

  let ours = RsaPublicKey::from_spki_der(RSA3072_SPKI).unwrap();
  let pkcs1 = ours.to_pkcs1_der();
  ours
    .verify_pss(RsaPssProfile::Sha256, MESSAGE_PSS, RSA3072_PSS_SHA256)
    .unwrap();
  ours
    .verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, MESSAGE_PKCS1V15, RSA3072_PKCS1V15_SHA256)
    .unwrap();

  ring::signature::UnparsedPublicKey::new(&ring::signature::RSA_PSS_2048_8192_SHA256, &pkcs1)
    .verify(MESSAGE_PSS, RSA3072_PSS_SHA256)
    .unwrap();
  ring::signature::UnparsedPublicKey::new(&ring::signature::RSA_PKCS1_2048_8192_SHA256, &pkcs1)
    .verify(MESSAGE_PKCS1V15, RSA3072_PKCS1V15_SHA256)
    .unwrap();
}

fn ring_aead_seal(algorithm: &'static ring_aead::Algorithm, key_bytes: &[u8], plaintext: &[u8]) -> Vec<u8> {
  let key = ring_aead::LessSafeKey::new(ring_aead::UnboundKey::new(algorithm, key_bytes).unwrap());
  let mut out = plaintext.to_vec();
  key
    .seal_in_place_append_tag(
      ring_aead::Nonce::assume_unique_for_key(NONCE_12),
      ring_aead::Aad::from(AAD),
      &mut out,
    )
    .unwrap();
  out
}
