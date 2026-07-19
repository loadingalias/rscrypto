#![allow(unused_imports)]

#[cfg(feature = "aead")]
use rscrypto::Aead;
#[cfg(any(feature = "hmac", feature = "hmac-sha3", feature = "kmac"))]
use rscrypto::Mac;
#[cfg(all(feature = "aead", feature = "diag"))]
use rscrypto::aead::introspect::{
  DispatchInfo as AeadDispatchInfo, aegis256_backend, aes256gcm_backend, aes256gcmsiv_backend, ascon_aead128_backend,
  chacha20poly1305_backend, xchacha20poly1305_backend,
};
#[cfg(feature = "aead")]
use rscrypto::aead::{
  AeadBufferError, Aegis256, Aegis256Key, Aegis256Tag, ChaCha20Poly1305, ChaCha20Poly1305Key, ChaCha20Poly1305Tag,
  Nonce96, Nonce128, Nonce192, Nonce256, OpenError, SealError, XChaCha20Poly1305, XChaCha20Poly1305Key,
  XChaCha20Poly1305Tag,
};
#[cfg(feature = "hkdf")]
use rscrypto::auth::HkdfOutputLengthError;
#[cfg(all(feature = "checksums", feature = "alloc"))]
use rscrypto::checksum::buffered::BufferedCrc32C;
#[cfg(feature = "checksums")]
use rscrypto::checksum::config::{
  Crc16Config, Crc16Force, Crc24Config, Crc24Force, Crc32Config, Crc32Force, Crc64Config, Crc64Force,
};
#[cfg(all(feature = "checksums", feature = "diag"))]
use rscrypto::checksum::introspect::{DispatchInfo, KernelIntrospect, is_hardware_accelerated, kernel_for};
#[cfg(feature = "checksums")]
use rscrypto::checksum::{Crc32Castagnoli, Crc32Ieee, Crc64Xz};
#[cfg(feature = "hashes")]
use rscrypto::hashes::fast::{RapidHash64, RapidHashFast64, RapidHashFast128, Xxh3_64};
#[cfg(all(feature = "hashes", feature = "diag"))]
use rscrypto::hashes::introspect::{
  DispatchInfo as HashDispatchInfo, KernelIntrospect as HashKernelIntrospect, kernel_for as hash_kernel_for,
};
#[cfg(all(feature = "hashes", feature = "std"))]
use rscrypto::hashes::{DigestReader, DigestWriter};
#[cfg(feature = "hashes")]
use rscrypto::{
  AsconCxof128, AsconCxof128Reader, AsconHash256, AsconXof, AsconXofReader, Blake3, Blake3XofReader, Cshake128,
  Cshake128XofReader, Cshake256, Cshake256XofReader, Digest, FastHash, RapidHash, RapidHash128, Sha3_224, Sha3_256,
  Sha3_384, Sha3_512, Sha224, Sha256, Sha384, Sha512, Sha512_256, Shake128, Shake128XofReader, Shake256,
  Shake256XofReader, Xof, Xxh3, Xxh3_128,
};
#[cfg(feature = "checksums")]
use rscrypto::{Checksum, ChecksumCombine, Crc16Ccitt, Crc16Ibm, Crc24OpenPgp, Crc32, Crc32C, Crc64, Crc64Nvme};
#[cfg(feature = "ed25519")]
use rscrypto::{Ed25519Keypair, Ed25519PublicKey, Ed25519SecretKey, Ed25519Signature};
#[cfg(feature = "hkdf")]
use rscrypto::{HkdfSha256, HkdfSha384, HkdfSha512};
#[cfg(feature = "hmac-sha3")]
use rscrypto::{
  HmacSha3_224, HmacSha3_224Tag, HmacSha3_256, HmacSha3_256Tag, HmacSha3_384, HmacSha3_384Tag, HmacSha3_512,
  HmacSha3_512Tag,
};
#[cfg(feature = "hmac")]
use rscrypto::{HmacSha256, HmacSha256Tag, HmacSha384, HmacSha384Tag, HmacSha512, HmacSha512Tag};
#[cfg(feature = "kmac")]
use rscrypto::{Kmac128, Kmac256};
#[cfg(feature = "ml-kem")]
use rscrypto::{
  MlKem512, MlKem512Ciphertext, MlKem512DecapsulationKey, MlKem512EncapsulationKey, MlKem512PreparedDecapsulationKey,
  MlKem512PreparedEncapsulationKey, MlKem512SharedSecret, MlKem768, MlKem768Ciphertext, MlKem768DecapsulationKey,
  MlKem768EncapsulationKey, MlKem768PreparedDecapsulationKey, MlKem768PreparedEncapsulationKey, MlKem768SharedSecret,
  MlKem1024, MlKem1024Ciphertext, MlKem1024DecapsulationKey, MlKem1024EncapsulationKey,
  MlKem1024PreparedDecapsulationKey, MlKem1024PreparedEncapsulationKey, MlKem1024SharedSecret,
};
#[cfg(feature = "poly1305")]
use rscrypto::{Poly1305, Poly1305OneTimeKey, Poly1305Tag};
#[cfg(feature = "rsa")]
use rscrypto::{
  RsaEncryptionError, RsaJwtAlgorithm, RsaJwtVerifier, RsaKeyError, RsaKeyGenerationError, RsaOaepProfile,
  RsaPkcs1v15Profile, RsaPrivateKey, RsaPrivateKeyParts, RsaPrivateOpError, RsaPrivateScratch,
  RsaProtocolAlgorithmError, RsaPssProfile, RsaPublicExponent, RsaPublicExponentPolicy, RsaPublicKey,
  RsaPublicKeyPolicy, RsaPublicOpError, RsaPublicScratch, RsaSignatureProfile, RsaTlsSignatureSchemes,
  RsaX509PublicKey, RsaX509PublicKeyAlgorithm,
};
use rscrypto::{VerificationError, ct};
#[cfg(feature = "x25519")]
use rscrypto::{X25519Error, X25519PublicKey, X25519SecretKey, X25519SharedSecret};

#[cfg(feature = "rsa")]
fn fill_rsa_random_with(byte: u8) -> impl FnMut(&mut [u8]) -> Result<(), RsaEncryptionError> {
  move |out| {
    out.fill(byte);
    Ok(())
  }
}

#[cfg(feature = "rsa")]
fn fill_rsa_random_from(bytes: &[u8]) -> impl FnMut(&mut [u8]) -> Result<(), RsaEncryptionError> + '_ {
  let mut offset = 0usize;
  move |out| {
    let end = offset.checked_add(out.len()).ok_or(RsaEncryptionError::InvalidLength)?;
    let Some(random) = bytes.get(offset..end) else {
      return Err(RsaEncryptionError::InvalidLength);
    };
    out.copy_from_slice(random);
    offset = end;
    Ok(())
  }
}

#[test]
fn root_surface_core_exports_compile() {
  let _ = VerificationError::new();
  assert!(ct::constant_time_eq(b"ok", b"ok"));
}

#[test]
#[cfg(feature = "aead")]
fn root_surface_aead_exports_compile() {
  let nonce96 = Nonce96::from_bytes([0x11; Nonce96::LENGTH]);
  let nonce128 = Nonce128::from_bytes([0x22; Nonce128::LENGTH]);
  let nonce192 = Nonce192::from_bytes([0x33; Nonce192::LENGTH]);

  assert_eq!(nonce96.as_bytes().len(), Nonce96::LENGTH);
  assert_eq!(nonce128.as_bytes().len(), Nonce128::LENGTH);
  assert_eq!(nonce192.as_bytes().len(), Nonce192::LENGTH);

  let nonce256 = Nonce256::from_bytes([0x44; Nonce256::LENGTH]);
  assert_eq!(nonce256.as_bytes().len(), Nonce256::LENGTH);

  let _ = AeadBufferError::new();
  let _ = SealError::buffer();
  let _ = SealError::too_large();
  let _ = OpenError::buffer();
  let _ = OpenError::too_large();
  let _ = OpenError::verification();

  #[cfg(feature = "diag")]
  {
    let _ = AeadDispatchInfo::current();
    let _ = aes256gcm_backend();
    let _ = aes256gcmsiv_backend();
    let _ = chacha20poly1305_backend();
    let _ = xchacha20poly1305_backend();
    let _ = aegis256_backend();
    let _ = ascon_aead128_backend();
  }

  fn assert_aead_trait<T: Aead>() {}

  #[derive(Clone)]
  struct Marker;

  impl Aead for Marker {
    const KEY_SIZE: usize = 32;
    const NONCE_SIZE: usize = Nonce96::LENGTH;
    const TAG_SIZE: usize = 16;

    type Key = [u8; 32];
    type Nonce = Nonce96;
    type Tag = [u8; 16];

    fn new(_key: &Self::Key) -> Self {
      Self
    }

    fn tag_from_slice(bytes: &[u8]) -> Result<Self::Tag, AeadBufferError> {
      if bytes.len() != Self::TAG_SIZE {
        return Err(AeadBufferError::new());
      }

      let mut tag = [0u8; Self::TAG_SIZE];
      tag.copy_from_slice(bytes);
      Ok(tag)
    }

    fn encrypt_in_place(&self, _nonce: &Self::Nonce, _aad: &[u8], _buffer: &mut [u8]) -> Result<Self::Tag, SealError> {
      Ok([0u8; Self::TAG_SIZE])
    }

    fn decrypt_in_place(
      &self,
      _nonce: &Self::Nonce,
      _aad: &[u8],
      _buffer: &mut [u8],
      _tag: &Self::Tag,
    ) -> Result<(), OpenError> {
      Ok(())
    }
  }

  assert_aead_trait::<Marker>();

  let key = XChaCha20Poly1305Key::from_bytes([0x44; XChaCha20Poly1305::KEY_SIZE]);
  let cipher = XChaCha20Poly1305::new(&key);
  let mut sealed = [0u8; 20];
  cipher.encrypt(&nonce192, b"aad", b"test", &mut sealed).unwrap();
  let _ = XChaCha20Poly1305Tag::from_bytes([0u8; XChaCha20Poly1305Tag::LENGTH]);

  let key = ChaCha20Poly1305Key::from_bytes([0x55; ChaCha20Poly1305::KEY_SIZE]);
  let cipher = ChaCha20Poly1305::new(&key);
  let mut sealed = [0u8; 20];
  cipher.encrypt(&nonce96, b"aad", b"test", &mut sealed).unwrap();
  let _ = ChaCha20Poly1305Tag::from_bytes([0u8; ChaCha20Poly1305Tag::LENGTH]);

  let key = Aegis256Key::from_bytes([0x66; Aegis256::KEY_SIZE]);
  let cipher = Aegis256::new(&key);
  let mut sealed = [0u8; 20];
  cipher.encrypt(&nonce256, b"aad", b"test", &mut sealed).unwrap();
  let _ = Aegis256Tag::from_bytes([0u8; Aegis256Tag::LENGTH]);
}

#[test]
#[cfg(feature = "hmac")]
fn root_surface_mac_exports_compile() {
  let key = b"root-surface-key";
  let data = b"root-surface-data";

  let tag = HmacSha256::mac(key, data);
  let tag384 = HmacSha384::mac(key, data);
  let tag512 = HmacSha512::mac(key, data);
  let _ = HmacSha256Tag::from_bytes(tag.to_bytes());
  let _ = HmacSha384Tag::from_bytes(tag384.to_bytes());
  let _ = HmacSha512Tag::from_bytes(tag512.to_bytes());

  let mut mac = HmacSha256::new(key);
  mac.update(data);
  assert_eq!(tag, mac.finalize());
  assert!(mac.verify(&tag).is_ok());

  let mut mac384 = HmacSha384::new(key);
  mac384.update(data);
  assert_eq!(tag384, mac384.finalize());
  assert!(mac384.verify(&tag384).is_ok());

  let mut mac512 = HmacSha512::new(key);
  mac512.update(data);
  assert_eq!(tag512, mac512.finalize());
  assert!(mac512.verify(&tag512).is_ok());
}

#[test]
#[cfg(feature = "hmac-sha3")]
fn root_surface_hmac_sha3_exports_compile() {
  let key = b"root-surface-key";
  let data = b"root-surface-data";

  let tag224 = HmacSha3_224::mac(key, data);
  let tag256 = HmacSha3_256::mac(key, data);
  let tag384 = HmacSha3_384::mac(key, data);
  let tag512 = HmacSha3_512::mac(key, data);
  let _ = HmacSha3_224Tag::from_bytes(tag224.to_bytes());
  let _ = HmacSha3_256Tag::from_bytes(tag256.to_bytes());
  let _ = HmacSha3_384Tag::from_bytes(tag384.to_bytes());
  let _ = HmacSha3_512Tag::from_bytes(tag512.to_bytes());

  assert!(HmacSha3_224::verify_tag(key, data, &tag224).is_ok());
  assert!(HmacSha3_256::verify_tag(key, data, &tag256).is_ok());
  assert!(HmacSha3_384::verify_tag(key, data, &tag384).is_ok());
  assert!(HmacSha3_512::verify_tag(key, data, &tag512).is_ok());
}

#[test]
#[cfg(feature = "hkdf")]
fn root_surface_kdf_exports_compile() {
  let key = b"root-surface-key";

  let mut out = [0u8; 32];
  let hkdf = HkdfSha256::new(b"salt", key);
  hkdf.expand(b"info", &mut out).unwrap();
  assert_eq!(out, HkdfSha256::derive_array::<32>(b"salt", key, b"info").unwrap());

  let mut out384 = [0u8; 48];
  let hkdf384 = HkdfSha384::new(b"salt", key);
  hkdf384.expand(b"info", &mut out384).unwrap();
  assert_eq!(out384, HkdfSha384::derive_array::<48>(b"salt", key, b"info").unwrap());

  let mut out512 = [0u8; 64];
  let hkdf512 = HkdfSha512::new(b"salt", key);
  hkdf512.expand(b"info", &mut out512).unwrap();
  assert_eq!(out512, HkdfSha512::derive_array::<64>(b"salt", key, b"info").unwrap());
  let _ = HkdfOutputLengthError::new();
}

#[test]
#[cfg(feature = "kmac")]
fn root_surface_kmac_exports_compile() {
  let key = b"root-surface-key";
  let data = b"root-surface-data";
  let mut out128 = [0u8; 32];
  let mut kmac128 = Kmac128::new(key, b"svc=v1");
  kmac128.update(data);
  kmac128.finalize_into(&mut out128);
  assert!(Kmac128::verify_tag(key, b"svc=v1", data, &out128).is_ok());

  let mut out = [0u8; 32];
  let mut kmac = Kmac256::new(key, b"svc=v1");
  kmac.update(data);
  kmac.finalize_into(&mut out);
  assert!(Kmac256::verify_tag(key, b"svc=v1", data, &out).is_ok());
}

#[test]
#[cfg(feature = "poly1305")]
fn root_surface_poly1305_exports_compile() {
  let key = Poly1305OneTimeKey::from_bytes([0x33; Poly1305OneTimeKey::LENGTH]);
  let tag = Poly1305::authenticate_once(key, b"root-surface-poly1305");
  let _ = Poly1305Tag::from_bytes(tag.to_bytes());

  let key = Poly1305OneTimeKey::from_bytes([0x33; Poly1305OneTimeKey::LENGTH]);
  assert!(Poly1305::verify_once(key, b"root-surface-poly1305", &tag).is_ok());
}

#[test]
#[cfg(feature = "ml-kem")]
fn root_surface_mlkem_exports_compile() {
  let ek512 = MlKem512EncapsulationKey::from_bytes([0x11; MlKem512EncapsulationKey::LENGTH]);
  let dk512 = MlKem512DecapsulationKey::from_bytes([0x12; MlKem512DecapsulationKey::LENGTH]);
  let ct512 = MlKem512Ciphertext::from_bytes([0x13; MlKem512Ciphertext::LENGTH]);
  let ss512 = MlKem512SharedSecret::from_bytes([0x14; MlKem512SharedSecret::LENGTH]);
  assert_eq!(ek512.as_bytes().len(), MlKem512::ENCAPSULATION_KEY_SIZE);
  assert_eq!(dk512.as_bytes().len(), MlKem512::DECAPSULATION_KEY_SIZE);
  assert_eq!(ct512.as_bytes().len(), MlKem512::CIPHERTEXT_SIZE);
  assert_eq!(ss512.as_bytes().len(), MlKem512::SHARED_SECRET_SIZE);
  assert_eq!(
    MlKem512PreparedEncapsulationKey::LENGTH,
    MlKem512::ENCAPSULATION_KEY_SIZE
  );
  assert_eq!(
    MlKem512PreparedDecapsulationKey::LENGTH,
    MlKem512::DECAPSULATION_KEY_SIZE
  );

  let ek768 = MlKem768EncapsulationKey::from_bytes([0x21; MlKem768EncapsulationKey::LENGTH]);
  let dk768 = MlKem768DecapsulationKey::from_bytes([0x22; MlKem768DecapsulationKey::LENGTH]);
  let ct768 = MlKem768Ciphertext::from_bytes([0x23; MlKem768Ciphertext::LENGTH]);
  let ss768 = MlKem768SharedSecret::from_bytes([0x24; MlKem768SharedSecret::LENGTH]);
  assert_eq!(ek768.as_bytes().len(), MlKem768::ENCAPSULATION_KEY_SIZE);
  assert_eq!(dk768.as_bytes().len(), MlKem768::DECAPSULATION_KEY_SIZE);
  assert_eq!(ct768.as_bytes().len(), MlKem768::CIPHERTEXT_SIZE);
  assert_eq!(ss768.as_bytes().len(), MlKem768::SHARED_SECRET_SIZE);
  assert_eq!(
    MlKem768PreparedEncapsulationKey::LENGTH,
    MlKem768::ENCAPSULATION_KEY_SIZE
  );
  assert_eq!(
    MlKem768PreparedDecapsulationKey::LENGTH,
    MlKem768::DECAPSULATION_KEY_SIZE
  );

  let ek1024 = MlKem1024EncapsulationKey::from_bytes([0x31; MlKem1024EncapsulationKey::LENGTH]);
  let dk1024 = MlKem1024DecapsulationKey::from_bytes([0x32; MlKem1024DecapsulationKey::LENGTH]);
  let ct1024 = MlKem1024Ciphertext::from_bytes([0x33; MlKem1024Ciphertext::LENGTH]);
  let ss1024 = MlKem1024SharedSecret::from_bytes([0x34; MlKem1024SharedSecret::LENGTH]);
  assert_eq!(ek1024.as_bytes().len(), MlKem1024::ENCAPSULATION_KEY_SIZE);
  assert_eq!(dk1024.as_bytes().len(), MlKem1024::DECAPSULATION_KEY_SIZE);
  assert_eq!(ct1024.as_bytes().len(), MlKem1024::CIPHERTEXT_SIZE);
  assert_eq!(ss1024.as_bytes().len(), MlKem1024::SHARED_SECRET_SIZE);
  assert_eq!(
    MlKem1024PreparedEncapsulationKey::LENGTH,
    MlKem1024::ENCAPSULATION_KEY_SIZE
  );
  assert_eq!(
    MlKem1024PreparedDecapsulationKey::LENGTH,
    MlKem1024::DECAPSULATION_KEY_SIZE
  );
}

#[test]
#[cfg(feature = "ed25519")]
fn root_surface_signature_exports_compile() {
  let secret = Ed25519SecretKey::from_bytes([7u8; Ed25519SecretKey::LENGTH]);
  let keypair = Ed25519Keypair::from_secret_key(secret.duplicate_secret());
  let public = keypair.public_key();
  let signature = keypair.sign(b"root-surface-ed25519");

  assert_eq!(secret.as_bytes().len(), 32);
  assert_eq!(public.as_bytes().len(), 32);
  assert_eq!(signature.as_bytes().len(), 64);
  assert!(public.verify(b"root-surface-ed25519", &signature).is_ok());
}

#[test]
#[cfg(feature = "rsa")]
fn root_surface_rsa_exports_compile() {
  let policy = RsaPublicKeyPolicy::legacy_verification().allow_legacy_small_exponents();
  assert_eq!(policy.min_modulus_bits(), 2048);
  assert_eq!(policy.max_modulus_bits(), 8192);
  let _ = RsaPublicExponentPolicy::Common65537;
  let _ = RsaKeyError::InvalidModulus;
  let _ = RsaKeyGenerationError::InvalidModulusBits;
  let _ = RsaEncryptionError::InvalidLength;
  let _ = RsaPrivateOpError::InvalidLength;
  let _ = RsaPublicOpError::RepresentativeOutOfRange;
  let _ = RsaProtocolAlgorithmError::UnsupportedAlgorithm;
  let _ = RsaOaepProfile::Sha256;
  let _ = RsaPssProfile::Sha256;
  let _ = RsaPkcs1v15Profile::Sha256;
  let _: Option<RsaPublicExponent> = None;
  let _: Option<RsaPublicKey> = None;
  let _: Option<RsaPublicScratch> = None;
  let _: Option<RsaJwtVerifier<'static>> = None;
  let _: Option<RsaPrivateKey> = None;
  let _: Option<RsaPrivateKeyParts<'static>> = None;
  let _: Option<RsaPrivateScratch> = None;
  let _: Option<RsaX509PublicKey> = None;
  assert_eq!(
    RsaSignatureProfile::pss(RsaPssProfile::Sha256).pss_parts(),
    Some((RsaPssProfile::Sha256, 32))
  );
  assert_eq!(
    RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha384).pkcs1v15_profile(),
    Some(RsaPkcs1v15Profile::Sha384)
  );
  assert_eq!(
    RsaSignatureProfile::from_tls13_signature_scheme(0x0804).unwrap(),
    RsaSignatureProfile::pss(RsaPssProfile::Sha256)
  );
  assert!(
    RsaX509PublicKeyAlgorithm::RsaPss
      .permits_signature_profile(RsaSignatureProfile::pss(RsaPssProfile::Sha256))
      .is_ok()
  );
  assert!(
    RsaX509PublicKeyAlgorithm::RsaEncryption
      .signature_profile_from_tls13_signature_scheme(0x0804)
      .is_ok()
  );
  let advertised = RsaX509PublicKeyAlgorithm::RsaEncryption.advertised_tls13_signature_schemes();
  assert_eq!(advertised.len(), 3);
  assert!(advertised.contains(0x0804));
  let _ = RsaTlsSignatureSchemes::MAX_LEN;
}

#[test]
#[cfg(all(feature = "rsa", feature = "getrandom"))]
fn root_surface_rsa_generated_key_end_to_end() {
  const X509_SHA256_WITH_RSA_ENCRYPTION: &[u8] = &[
    0x30, 0x0d, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0b, 0x05, 0x00,
  ];
  const X509_PSS_SHA256_ALGORITHM: &[u8] = &[
    0x30, 0x41, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0a, 0x30, 0x34, 0xa0, 0x0f, 0x30, 0x0d,
    0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, 0x05, 0x00, 0xa1, 0x1c, 0x30, 0x1a, 0x06, 0x09,
    0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x08, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03,
    0x04, 0x02, 0x01, 0x05, 0x00, 0xa2, 0x03, 0x02, 0x01, 0x20,
  ];

  let policy = RsaPublicKeyPolicy::legacy_verification();
  let key = RsaPrivateKey::generate_with_policy(2048, &policy).unwrap();
  let public_key = key.public_key();
  let x509_key = RsaX509PublicKey::from_spki_der_with_policy(&public_key.to_spki_der(), &policy).unwrap();
  let message = b"root-surface-rsa-generated-key";
  let mut private_scratch = key.private_scratch();
  let mut public_scratch = public_key.public_scratch();
  let mut x509_scratch = x509_key.public_key().public_scratch();

  let pkcs1_der = key.to_pkcs1_der();
  let pkcs8_der = key.to_pkcs8_der();
  assert_eq!(format!("{pkcs1_der:?}"), "SecretVec(****)");
  assert_eq!(
    RsaPrivateKey::from_pkcs1_der_with_policy(&pkcs1_der, &policy)
      .unwrap()
      .public_key(),
    public_key
  );
  assert_eq!(
    RsaPrivateKey::from_pkcs8_der_with_policy(&pkcs8_der, &policy)
      .unwrap()
      .public_key(),
    public_key
  );
  assert_eq!(
    RsaPublicKey::from_pkcs1_der_with_policy(&public_key.to_pkcs1_der(), &policy).unwrap(),
    *public_key
  );
  assert_eq!(
    RsaPublicKey::from_spki_der_with_policy(&public_key.to_spki_der(), &policy).unwrap(),
    *public_key
  );

  let mut unprotected_pkcs1 = key.to_pkcs1_der().into_unprotected_vec();
  assert_eq!(
    RsaPrivateKey::from_pkcs1_der_with_policy(&unprotected_pkcs1, &policy)
      .unwrap()
      .public_key(),
    public_key
  );
  rscrypto::traits::ct::zeroize(&mut unprotected_pkcs1);

  let mut signature = vec![0u8; key.signature_len()];
  for (pkcs1v15_profile, pss_profile) in [
    (RsaPkcs1v15Profile::Sha256, RsaPssProfile::Sha256),
    (RsaPkcs1v15Profile::Sha384, RsaPssProfile::Sha384),
    (RsaPkcs1v15Profile::Sha512, RsaPssProfile::Sha512),
  ] {
    let pkcs1v15_profile = RsaSignatureProfile::pkcs1v15(pkcs1v15_profile);
    key.sign_signature(pkcs1v15_profile, message, &mut signature).unwrap();
    public_key
      .verify_signature(pkcs1v15_profile, message, &signature)
      .unwrap();
    public_key
      .verify_signature_with_scratch(pkcs1v15_profile, message, &signature, &mut public_scratch)
      .unwrap();
    key
      .sign_signature_with_scratch(pkcs1v15_profile, message, &mut signature, &mut private_scratch)
      .unwrap();
    public_key
      .verify_signature(pkcs1v15_profile, message, &signature)
      .unwrap();
    public_key
      .verify_signature_with_scratch(pkcs1v15_profile, message, &signature, &mut public_scratch)
      .unwrap();

    let pss_profile = RsaSignatureProfile::pss(pss_profile);
    key.sign_signature(pss_profile, message, &mut signature).unwrap();
    public_key.verify_signature(pss_profile, message, &signature).unwrap();
    public_key
      .verify_signature_with_scratch(pss_profile, message, &signature, &mut public_scratch)
      .unwrap();
    key
      .sign_signature_with_scratch(pss_profile, message, &mut signature, &mut private_scratch)
      .unwrap();
    public_key.verify_signature(pss_profile, message, &signature).unwrap();
    public_key
      .verify_signature_with_scratch(pss_profile, message, &signature, &mut public_scratch)
      .unwrap();
  }
  let explicit_pss_profile = RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha384, 24);
  key
    .sign_signature_with_scratch(explicit_pss_profile, message, &mut signature, &mut private_scratch)
    .unwrap();
  public_key
    .verify_signature(explicit_pss_profile, message, &signature)
    .unwrap();
  public_key
    .verify_signature_with_scratch(explicit_pss_profile, message, &signature, &mut public_scratch)
    .unwrap();

  key
    .sign_x509_signature_algorithm_der(X509_SHA256_WITH_RSA_ENCRYPTION, message, &mut signature)
    .unwrap();
  x509_key
    .verify_signature_from_x509_algorithm_der(X509_SHA256_WITH_RSA_ENCRYPTION, message, &signature)
    .unwrap();
  x509_key
    .verify_signature_from_x509_algorithm_der_with_scratch(
      X509_SHA256_WITH_RSA_ENCRYPTION,
      message,
      &signature,
      &mut x509_scratch,
    )
    .unwrap();
  key
    .sign_x509_signature_algorithm_der_with_scratch(
      X509_PSS_SHA256_ALGORITHM,
      message,
      &mut signature,
      &mut private_scratch,
    )
    .unwrap();
  x509_key
    .verify_signature_from_x509_algorithm_der(X509_PSS_SHA256_ALGORITHM, message, &signature)
    .unwrap();
  x509_key
    .verify_signature_from_x509_algorithm_der_with_scratch(
      X509_PSS_SHA256_ALGORITHM,
      message,
      &signature,
      &mut x509_scratch,
    )
    .unwrap();

  let pss_sha256 = RsaSignatureProfile::pss(RsaPssProfile::Sha256);
  let pkcs1v15_sha256 = RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256);

  key
    .sign_tls13_signature_scheme(0x0804, message, &mut signature)
    .unwrap();
  x509_key
    .verify_expected_tls13_signature_scheme(0x0804, 0x0804, pss_sha256, message, &signature)
    .unwrap();
  x509_key
    .verify_expected_tls13_signature_scheme_with_scratch(
      0x0804,
      0x0804,
      pss_sha256,
      message,
      &signature,
      &mut x509_scratch,
    )
    .unwrap();
  key
    .sign_tls13_signature_scheme_with_scratch(0x0804, message, &mut signature, &mut private_scratch)
    .unwrap();
  x509_key
    .verify_expected_tls13_signature_scheme(0x0804, 0x0804, pss_sha256, message, &signature)
    .unwrap();
  x509_key
    .verify_expected_tls13_signature_scheme_with_scratch(
      0x0804,
      0x0804,
      pss_sha256,
      message,
      &signature,
      &mut x509_scratch,
    )
    .unwrap();

  key
    .sign_tls_certificate_signature_scheme(0x0401, message, &mut signature)
    .unwrap();
  x509_key
    .verify_expected_tls_certificate_signature_scheme(0x0401, 0x0401, pkcs1v15_sha256, message, &signature)
    .unwrap();
  x509_key
    .verify_expected_tls_certificate_signature_scheme_with_scratch(
      0x0401,
      0x0401,
      pkcs1v15_sha256,
      message,
      &signature,
      &mut x509_scratch,
    )
    .unwrap();
  key
    .sign_tls_certificate_signature_scheme_with_scratch(0x0401, message, &mut signature, &mut private_scratch)
    .unwrap();
  x509_key
    .verify_expected_tls_certificate_signature_scheme(0x0401, 0x0401, pkcs1v15_sha256, message, &signature)
    .unwrap();
  x509_key
    .verify_expected_tls_certificate_signature_scheme_with_scratch(
      0x0401,
      0x0401,
      pkcs1v15_sha256,
      message,
      &signature,
      &mut x509_scratch,
    )
    .unwrap();

  key
    .jwt_signer(RsaJwtAlgorithm::Ps256)
    .try_sign_into(message, &mut signature)
    .unwrap();
  let verifier = public_key.jwt_verifier(RsaJwtAlgorithm::Ps256);
  verifier.verify("PS256", message, &signature).unwrap();
  verifier
    .verify_with_scratch("PS256", message, &signature, &mut public_scratch)
    .unwrap();
  key
    .sign_signature_with_scratch(
      RsaJwtAlgorithm::Rs256.signature_profile(),
      message,
      &mut signature,
      &mut private_scratch,
    )
    .unwrap();
  let verifier = public_key.jwt_verifier(RsaJwtAlgorithm::Rs256);
  verifier.verify("RS256", message, &signature).unwrap();
  verifier
    .verify_with_scratch("RS256", message, &signature, &mut public_scratch)
    .unwrap();

  key.sign_cose_algorithm_id(-37, message, &mut signature).unwrap();
  public_key
    .verify_expected_cose_algorithm_id(-37, -37, pss_sha256, message, &signature)
    .unwrap();
  public_key
    .verify_expected_cose_algorithm_id_with_scratch(-37, -37, pss_sha256, message, &signature, &mut public_scratch)
    .unwrap();
  key
    .sign_cose_algorithm_id_with_scratch(-257, message, &mut signature, &mut private_scratch)
    .unwrap();
  public_key
    .verify_expected_cose_algorithm_id(-257, -257, pkcs1v15_sha256, message, &signature)
    .unwrap();
  public_key
    .verify_expected_cose_algorithm_id_with_scratch(
      -257,
      -257,
      pkcs1v15_sha256,
      message,
      &signature,
      &mut public_scratch,
    )
    .unwrap();

  let label = b"root-surface-rsa-label";
  let plaintext = b"root-surface-rsa-oaep";
  let mut ciphertext = vec![0u8; key.signature_len()];
  let mut decrypted = vec![0u8; key.signature_len()];
  for oaep_profile in [RsaOaepProfile::Sha256, RsaOaepProfile::Sha384, RsaOaepProfile::Sha512] {
    public_key
      .encrypt_oaep(oaep_profile, label, plaintext, &mut ciphertext)
      .unwrap();
    public_key
      .encrypt_oaep_with_scratch(oaep_profile, label, plaintext, &mut ciphertext, &mut public_scratch)
      .unwrap();
    let decrypted_len = key
      .decrypt_oaep(oaep_profile, label, &ciphertext, &mut decrypted)
      .unwrap();
    assert_eq!(&decrypted[..decrypted_len], plaintext);
    let decrypted_len = key
      .decrypt_oaep_with_scratch(oaep_profile, label, &ciphertext, &mut decrypted, &mut private_scratch)
      .unwrap();
    assert_eq!(&decrypted[..decrypted_len], plaintext);
  }

  let legacy_plaintext = b"root-surface-rsaes-pkcs1v15";
  public_key.encrypt_pkcs1v15(legacy_plaintext, &mut ciphertext).unwrap();
  public_key
    .encrypt_pkcs1v15_with_scratch(legacy_plaintext, &mut ciphertext, &mut public_scratch)
    .unwrap();
  let decrypted_len = key.decrypt_pkcs1v15(&ciphertext, &mut decrypted).unwrap();
  assert_eq!(&decrypted[..decrypted_len], legacy_plaintext);
  let decrypted_len = key
    .decrypt_pkcs1v15_with_scratch(&ciphertext, &mut decrypted, &mut private_scratch)
    .unwrap();
  assert_eq!(&decrypted[..decrypted_len], legacy_plaintext);
}

#[test]
#[cfg(all(feature = "rsa", feature = "getrandom"))]
fn root_surface_rsa_default_generated_key_end_to_end() {
  let key = RsaPrivateKey::generate(3072).unwrap();
  assert_eq!(key.public_key().modulus_bits(), 3072);
  let public_key = key.public_key();
  let message = b"root-surface-rsa-default-generated-key";
  let mut private_scratch = key.private_scratch();
  let mut public_scratch = public_key.public_scratch();

  let pkcs1_der = key.to_pkcs1_der();
  let pkcs8_der = key.to_pkcs8_der();
  assert_eq!(
    RsaPrivateKey::from_pkcs1_der(&pkcs1_der).unwrap().public_key(),
    public_key
  );
  assert_eq!(
    RsaPrivateKey::from_pkcs8_der(&pkcs8_der).unwrap().public_key(),
    public_key
  );
  assert_eq!(
    RsaPublicKey::from_pkcs1_der(&public_key.to_pkcs1_der()).unwrap(),
    *public_key
  );
  assert_eq!(
    RsaPublicKey::from_spki_der(&public_key.to_spki_der()).unwrap(),
    *public_key
  );

  let mut signature = vec![0u8; key.signature_len()];
  let pkcs1v15_profile = RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256);
  key
    .sign_signature_with_scratch(pkcs1v15_profile, message, &mut signature, &mut private_scratch)
    .unwrap();
  public_key
    .verify_signature_with_scratch(pkcs1v15_profile, message, &signature, &mut public_scratch)
    .unwrap();

  let pss_profile = RsaSignatureProfile::pss(RsaPssProfile::Sha256);
  key
    .sign_signature_with_scratch(pss_profile, message, &mut signature, &mut private_scratch)
    .unwrap();
  public_key
    .verify_signature_with_scratch(pss_profile, message, &signature, &mut public_scratch)
    .unwrap();

  let label = b"root-surface-rsa-default-label";
  let plaintext = b"root-surface-rsa-default-oaep";
  let seed = [0x37; 32];
  let mut ciphertext = vec![0u8; key.signature_len()];
  let mut decrypted = vec![0u8; key.signature_len()];
  public_key
    .encrypt_oaep_with_random_fill_and_scratch(
      RsaOaepProfile::Sha256,
      label,
      plaintext,
      &mut ciphertext,
      &mut public_scratch,
      fill_rsa_random_from(&seed),
    )
    .unwrap();
  let decrypted_len = key
    .decrypt_oaep_with_scratch(
      RsaOaepProfile::Sha256,
      label,
      &ciphertext,
      &mut decrypted,
      &mut private_scratch,
    )
    .unwrap();
  assert_eq!(&decrypted[..decrypted_len], plaintext);

  let legacy_plaintext = b"root-surface-rsa-default-rsaes-pkcs1v15";
  public_key
    .encrypt_pkcs1v15_with_random_fill_and_scratch(
      legacy_plaintext,
      &mut ciphertext,
      &mut public_scratch,
      fill_rsa_random_with(0x5b),
    )
    .unwrap();
  let decrypted_len = key
    .decrypt_pkcs1v15_with_scratch(&ciphertext, &mut decrypted, &mut private_scratch)
    .unwrap();
  assert_eq!(&decrypted[..decrypted_len], legacy_plaintext);
}

#[test]
#[cfg(feature = "x25519")]
fn root_surface_key_exchange_exports_compile() {
  let alice = X25519SecretKey::from_bytes([11u8; X25519SecretKey::LENGTH]);
  let bob = X25519SecretKey::from_bytes([13u8; X25519SecretKey::LENGTH]);
  let alice_public: X25519PublicKey = (&alice).into();
  let bob_public: X25519PublicKey = (&bob).into();
  let alice_shared = alice.diffie_hellman(&bob_public).unwrap();
  let bob_shared = X25519SharedSecret::diffie_hellman(&bob, &alice_public).unwrap();

  assert_eq!(alice_public.as_bytes().len(), 32);
  assert_eq!(alice_shared.as_bytes().len(), 32);
  assert_eq!(alice_shared, bob_shared);
  let _ = X25519Error::new();
}

#[test]
#[cfg(feature = "checksums")]
fn root_surface_checksum_exports_compile() {
  let data = b"root-surface";
  let (left, right) = data.split_at(4);

  let oneshot = Crc32C::checksum(data);

  let mut streaming = Crc32C::new();
  streaming.update(left);
  streaming.update(right);

  assert_eq!(oneshot, streaming.finalize());
  assert_eq!(
    oneshot,
    Crc32C::combine(Crc32C::checksum(left), Crc32C::checksum(right), right.len())
  );

  assert_eq!(Crc32Ieee::checksum(data), Crc32::checksum(data));
  assert_eq!(Crc32Castagnoli::checksum(data), Crc32C::checksum(data));
  assert_eq!(Crc64Xz::checksum(data), Crc64::checksum(data));
}

#[test]
#[cfg(all(feature = "checksums", feature = "alloc"))]
fn buffered_checksum_constructors_compile() {
  let data = b"root-surface";

  let mut buffered = rscrypto::Crc32C::buffered();
  buffered.update(data);

  assert_eq!(buffered.finalize(), rscrypto::Crc32C::checksum(data));

  let mut explicit = BufferedCrc32C::new();
  explicit.update(data);
  assert_eq!(explicit.finalize(), rscrypto::Crc32C::checksum(data));
}

#[test]
#[cfg(feature = "hashes")]
fn root_surface_hash_exports_compile() {
  let data = b"root-surface";

  let oneshot = Blake3::digest(data);

  let mut streaming = Blake3::new();
  streaming.update(data);
  assert_eq!(oneshot, streaming.finalize());

  let mut xof = Blake3::xof(data);
  let mut out = [0u8; 16];
  xof.squeeze(&mut out);

  let mut shake = Shake256::xof(data);
  shake.squeeze(&mut out);

  let mut ascon = AsconXof::xof(data);
  ascon.squeeze(&mut out);

  let mut cshake = Cshake256::xof(b"", b"ctx=v1", data);
  cshake.squeeze(&mut out);

  let mut cshake128 = Cshake128::xof(b"", b"ctx=v1", data);
  cshake128.squeeze(&mut out);
  let _: Option<Cshake128XofReader> = None;
  let mut cxof = AsconCxof128::xof(b"ctx=v1", data).unwrap();
  cxof.squeeze(&mut out);

  assert_eq!(Xxh3::hash(data), Xxh3_64::hash(data));
  assert_eq!(RapidHash::hash(data), RapidHash64::hash(data));
  let _ = RapidHashFast64::hash(data);
  let _ = RapidHashFast128::hash(data);
  assert_ne!(Xxh3::hash(data), Xxh3::hash_with_seed(7, data));
}

#[test]
#[cfg(all(feature = "hashes", feature = "std"))]
fn digest_reader_writer_round_trip() {
  use std::io::{Cursor, Read, Write};

  let data = b"hello digest reader writer";
  let expected = Sha256::digest(data);

  // DigestReader: read data through and verify digest matches.
  let mut reader = DigestReader::<_, Sha256>::new(Cursor::new(data.to_vec()));
  let mut sink = Vec::new();
  std::io::copy(&mut reader, &mut sink).unwrap();
  assert_eq!(reader.digest(), expected);

  // DigestWriter: write data through and verify digest matches.
  let mut writer = DigestWriter::<_, Sha256>::new(Vec::new());
  writer.write_all(data).unwrap();
  let (out, digest) = writer.into_parts();
  assert_eq!(&out, data);
  assert_eq!(digest, expected);
}

#[test]
#[cfg(all(feature = "checksums", feature = "diag"))]
fn advanced_checksum_modules_compile() {
  fn assert_kernel_introspect<T: KernelIntrospect>() {}

  let _: Crc32Config = Crc32::config();
  let _ = Crc32Force::Auto;
  let _ = DispatchInfo::current();
  let _ = kernel_for::<Crc32>(64);
  let _ = is_hardware_accelerated();
  let _ = rscrypto::platform::describe();
  assert_kernel_introspect::<Crc32>();
}

#[test]
#[cfg(all(feature = "hashes", feature = "diag"))]
fn advanced_hash_modules_compile() {
  fn assert_hash_kernel_introspect<T: HashKernelIntrospect>() {}

  let _ = HashDispatchInfo::current();
  let _ = hash_kernel_for::<Sha256>(64);
  let _ = hash_kernel_for::<Shake256>(64);
  let _ = hash_kernel_for::<Blake3>(64);
  let _ = hash_kernel_for::<AsconHash256>(64);
  let _ = hash_kernel_for::<AsconXof>(64);
  let _ = hash_kernel_for::<AsconCxof128>(64);
  let _ = hash_kernel_for::<Xxh3>(64);
  let _ = hash_kernel_for::<RapidHash128>(64);

  assert_hash_kernel_introspect::<Sha224>();
  assert_hash_kernel_introspect::<Sha256>();
  assert_hash_kernel_introspect::<Sha384>();
  assert_hash_kernel_introspect::<Sha512>();
  assert_hash_kernel_introspect::<Sha512_256>();
  assert_hash_kernel_introspect::<Sha3_224>();
  assert_hash_kernel_introspect::<Sha3_256>();
  assert_hash_kernel_introspect::<Sha3_384>();
  assert_hash_kernel_introspect::<Sha3_512>();
  assert_hash_kernel_introspect::<Shake128>();
  assert_hash_kernel_introspect::<Shake256>();
  assert_hash_kernel_introspect::<Cshake128>();
  assert_hash_kernel_introspect::<Cshake256>();
  assert_hash_kernel_introspect::<AsconCxof128>();
  assert_hash_kernel_introspect::<Blake3>();
  assert_hash_kernel_introspect::<AsconHash256>();
  assert_hash_kernel_introspect::<AsconXof>();
  assert_hash_kernel_introspect::<Xxh3>();
  assert_hash_kernel_introspect::<Xxh3_128>();
  assert_hash_kernel_introspect::<RapidHash>();
  assert_hash_kernel_introspect::<RapidHash128>();
}
