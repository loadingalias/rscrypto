#![cfg(all(
  any(unix, windows),
  not(target_arch = "wasm32"),
  not(any(target_arch = "s390x", target_arch = "powerpc64"))
))]
#![cfg(all(
  feature = "sha2",
  feature = "hmac",
  feature = "hkdf",
  feature = "pbkdf2",
  feature = "aes-gcm",
  feature = "chacha20poly1305",
  feature = "ed25519",
  feature = "x25519",
  feature = "rsa",
))]

use core::num::NonZeroU32;

use aws_lc_rs::{
  aead as aws_aead, agreement as aws_agreement, digest as aws_digest, hkdf as aws_hkdf, hmac as aws_hmac,
  pbkdf2 as aws_pbkdf2, signature as aws_signature,
};
use rscrypto::{
  Aes256Gcm, Aes256GcmKey, ChaCha20Poly1305, ChaCha20Poly1305Key, Ed25519SecretKey, HkdfSha256, HmacSha256,
  Pbkdf2Sha256, RsaPkcs1v15Profile, RsaPssProfile, RsaPublicKey, Sha256, X25519PublicKey, X25519SecretKey,
  aead::{Nonce96, expert::AeadWithNonce},
};

const DATA: &[u8] = b"migration equivalence data";
const KEY_32: [u8; 32] = [0x42; 32];
const NONCE_12: [u8; 12] = [0x24; 12];
const AAD: &[u8] = b"migration aad";

const RSA3072_SPKI: &[u8] = include_bytes!("../benches/rsa_fixtures/rsa3072_spki.der");
const RSA3072_PSS_SHA256: &[u8] = include_bytes!("../benches/rsa_fixtures/rsa3072_pss_sha256.sig");
const RSA3072_PKCS1V15_SHA256: &[u8] = include_bytes!("../benches/rsa_fixtures/rsa3072_pkcs1v15_sha256.sig");
const MESSAGE_PSS: &[u8] = b"rscrypto RSA-PSS verification fixture";
const MESSAGE_PKCS1V15: &[u8] = b"rscrypto RSA-PKCS1-v1_5 verification fixture";

struct AwsHkdfLen(usize);

impl aws_hkdf::KeyType for AwsHkdfLen {
  fn len(&self) -> usize {
    self.0
  }
}

#[test]
fn test_aws_lc_rs_digest_hmac_hkdf_and_pbkdf2_migration_examples_are_byte_equivalent() {
  let aws_digest = aws_digest::digest(&aws_digest::SHA256, DATA);
  assert_eq!(Sha256::digest(DATA).as_slice(), aws_digest.as_ref());

  let aws_hmac_key = aws_hmac::Key::new(aws_hmac::HMAC_SHA256, &KEY_32);
  let aws_hmac = aws_hmac::sign(&aws_hmac_key, DATA);
  assert_eq!(HmacSha256::mac(&KEY_32, DATA).as_slice(), aws_hmac.as_ref());

  let salt = b"migration salt!!";
  let ikm = b"migration input key material";
  let info = b"migration context";
  let mut aws_okm = [0u8; 42];
  aws_hkdf::Salt::new(aws_hkdf::HKDF_SHA256, salt)
    .extract(ikm)
    .expand(&[info], AwsHkdfLen(aws_okm.len()))
    .expect("42-byte AWS-LC HKDF expansion must be valid")
    .fill(&mut aws_okm)
    .expect("AWS-LC HKDF output buffer must match the requested length");

  let mut ours_okm = [0u8; 42];
  HkdfSha256::new(salt, ikm)
    .expand(info, &mut ours_okm)
    .expect("42-byte rscrypto HKDF expansion must be valid");
  assert_eq!(ours_okm, aws_okm);

  let iterations = NonZeroU32::new(600_000).expect("PBKDF2 iteration count must be nonzero");
  let mut aws_pbkdf2 = [0u8; 32];
  aws_pbkdf2::derive(
    aws_pbkdf2::PBKDF2_HMAC_SHA256,
    iterations,
    salt,
    b"migration password",
    &mut aws_pbkdf2,
  );

  let mut ours_pbkdf2 = [0u8; 32];
  Pbkdf2Sha256::derive_key(b"migration password", salt, iterations.get(), &mut ours_pbkdf2)
    .expect("rscrypto PBKDF2 parameters must be valid");
  assert_eq!(ours_pbkdf2, aws_pbkdf2);
}

#[test]
fn test_aws_lc_rs_aead_migration_examples_are_byte_equivalent() {
  let aws_aes = aws_aead_seal(&aws_aead::AES_256_GCM, &KEY_32, DATA);
  let aes = Aes256Gcm::new(&Aes256GcmKey::from_bytes(KEY_32));
  let nonce = Nonce96::from_bytes(NONCE_12);
  let mut ours_aes = vec![0u8; DATA.len() + 16];
  aes
    .encrypt(&nonce, AAD, DATA, &mut ours_aes)
    .expect("rscrypto AES-256-GCM seal buffer must fit plaintext and tag");
  assert_eq!(ours_aes, aws_aes);

  let mut opened = vec![0u8; DATA.len()];
  aes
    .decrypt(&nonce, AAD, &ours_aes, &mut opened)
    .expect("freshly sealed rscrypto AES-256-GCM ciphertext must authenticate");
  assert_eq!(opened, DATA);

  let aws_chacha = aws_aead_seal(&aws_aead::CHACHA20_POLY1305, &KEY_32, DATA);
  let chacha = ChaCha20Poly1305::new(&ChaCha20Poly1305Key::from_bytes(KEY_32));
  let mut ours_chacha = vec![0u8; DATA.len() + 16];
  chacha
    .encrypt(&nonce, AAD, DATA, &mut ours_chacha)
    .expect("rscrypto ChaCha20-Poly1305 seal buffer must fit plaintext and tag");
  assert_eq!(ours_chacha, aws_chacha);

  let mut opened = vec![0u8; DATA.len()];
  chacha
    .decrypt(&nonce, AAD, &ours_chacha, &mut opened)
    .expect("freshly sealed rscrypto ChaCha20-Poly1305 ciphertext must authenticate");
  assert_eq!(opened, DATA);
}

#[test]
fn test_aws_lc_rs_ed25519_and_x25519_migration_examples_are_byte_equivalent() {
  use aws_lc_rs::signature::KeyPair as _;

  let seed = [0x13; 32];
  let aws_ed25519 =
    aws_signature::Ed25519KeyPair::from_seed_unchecked(&seed).expect("32-byte AWS-LC Ed25519 seed must be valid");
  let ours_ed25519 = Ed25519SecretKey::from_bytes(seed);
  let ours_public = ours_ed25519.public_key();
  let ours_signature = ours_ed25519.sign(DATA);

  assert_eq!(ours_public.as_bytes(), aws_ed25519.public_key().as_ref());
  assert_eq!(ours_signature.as_bytes(), aws_ed25519.sign(DATA).as_ref());

  aws_signature::UnparsedPublicKey::new(&aws_signature::ED25519, ours_public.as_bytes())
    .verify(DATA, ours_signature.as_bytes())
    .expect("AWS-LC must verify the rscrypto Ed25519 signature");
  ours_public
    .verify(DATA, &ours_signature)
    .expect("rscrypto must verify its Ed25519 signature");

  let alice_bytes = [0x18; 32];
  let bob_bytes = [0x34; 32];
  let ours_alice = X25519SecretKey::from_bytes(alice_bytes);
  let ours_bob_public = X25519SecretKey::from_bytes(bob_bytes).public_key();
  let ours_shared = ours_alice
    .diffie_hellman(&ours_bob_public)
    .expect("fixed X25519 keys must produce a nonzero shared secret");

  let aws_alice = aws_agreement::PrivateKey::from_private_key(&aws_agreement::X25519, &alice_bytes)
    .expect("32-byte AWS-LC Alice X25519 key must be valid");
  let aws_bob = aws_agreement::PrivateKey::from_private_key(&aws_agreement::X25519, &bob_bytes)
    .expect("32-byte AWS-LC Bob X25519 key must be valid");
  let aws_bob_public = aws_bob
    .compute_public_key()
    .expect("AWS-LC must derive Bob's X25519 public key");
  let mut aws_bob_public_bytes = [0u8; 32];
  aws_bob_public_bytes.copy_from_slice(aws_bob_public.as_ref());
  assert_eq!(ours_bob_public.as_bytes(), &aws_bob_public_bytes);

  let peer = X25519PublicKey::from_bytes(aws_bob_public_bytes);
  assert_eq!(
    ours_alice
      .diffie_hellman(&peer)
      .expect("AWS-LC X25519 public key must produce a nonzero rscrypto shared secret")
      .as_bytes(),
    ours_shared.as_bytes()
  );

  let aws_peer = aws_agreement::UnparsedPublicKey::new(&aws_agreement::X25519, aws_bob_public_bytes);
  let aws_shared = aws_agreement::agree(&aws_alice, aws_peer, (), |bytes| {
    let mut out = [0u8; 32];
    out.copy_from_slice(bytes);
    Ok::<[u8; 32], ()>(out)
  })
  .expect("AWS-LC X25519 agreement must accept the fixed key pair");
  assert_eq!(ours_shared.as_bytes(), &aws_shared);
}

#[test]
fn test_aws_lc_rs_rsa_verify_migration_examples_accept_the_same_fixtures() {
  let ours = RsaPublicKey::from_spki_der(RSA3072_SPKI).expect("embedded RSA-3072 SPKI fixture must parse");
  ours
    .verify_pss(RsaPssProfile::Sha256, MESSAGE_PSS, RSA3072_PSS_SHA256)
    .expect("rscrypto must verify the RSA-PSS fixture");
  ours
    .verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, MESSAGE_PKCS1V15, RSA3072_PKCS1V15_SHA256)
    .expect("rscrypto must verify the RSA-PKCS1-v1_5 fixture");

  aws_signature::UnparsedPublicKey::new(&aws_signature::RSA_PSS_2048_8192_SHA256, RSA3072_SPKI)
    .verify(MESSAGE_PSS, RSA3072_PSS_SHA256)
    .expect("AWS-LC must verify the RSA-PSS fixture");
  aws_signature::UnparsedPublicKey::new(&aws_signature::RSA_PKCS1_2048_8192_SHA256, RSA3072_SPKI)
    .verify(MESSAGE_PKCS1V15, RSA3072_PKCS1V15_SHA256)
    .expect("AWS-LC must verify the RSA-PKCS1-v1_5 fixture");
}

fn aws_aead_seal(algorithm: &'static aws_aead::Algorithm, key_bytes: &[u8], plaintext: &[u8]) -> Vec<u8> {
  let key = aws_aead::LessSafeKey::new(
    aws_aead::UnboundKey::new(algorithm, key_bytes).expect("AWS-LC AEAD key length must match the algorithm"),
  );
  let mut out = plaintext.to_vec();
  key
    .seal_in_place_append_tag(
      aws_aead::Nonce::assume_unique_for_key(NONCE_12),
      aws_aead::Aad::from(AAD),
      &mut out,
    )
    .expect("AWS-LC AEAD sealing must accept the fixed nonce and buffer");
  out
}
