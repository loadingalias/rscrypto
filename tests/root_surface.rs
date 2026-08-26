#[cfg(feature = "aead")]
use rscrypto::Aead;
#[cfg(feature = "hmac")]
use rscrypto::Mac;
#[cfg(all(feature = "rsa", feature = "getrandom"))]
use rscrypto::RsaJwtAlgorithm;
#[cfg(all(feature = "hashes", any(feature = "std", feature = "diag")))]
use rscrypto::Sha256;
#[cfg(feature = "aead")]
use rscrypto::aead::expert::AeadWithNonce;
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
#[cfg(all(feature = "checksums", feature = "diag"))]
use rscrypto::checksum::config::{
  Crc16Config, Crc16Force, Crc24Config, Crc24Force, Crc32Config, Crc32Force, Crc64Config, Crc64Force,
};
#[cfg(all(feature = "checksums", feature = "diag"))]
use rscrypto::checksum::introspect::{DispatchInfo, KernelIntrospect, is_hardware_accelerated, kernel_for};
#[cfg(feature = "checksums")]
use rscrypto::checksum::{Crc32Castagnoli, Crc32Ieee, Crc64Xz};
#[cfg(feature = "hashes")]
use rscrypto::hashes::fast::Xxh3_64;
#[cfg(all(feature = "hashes", feature = "diag"))]
use rscrypto::hashes::introspect::{
  DispatchInfo as HashDispatchInfo, KernelIntrospect as HashKernelIntrospect, kernel_for as hash_kernel_for,
};
#[cfg(feature = "websocket-sha1")]
use rscrypto::hashes::legacy::WebSocketAcceptDigest;
#[cfg(all(feature = "hashes", feature = "std"))]
use rscrypto::hashes::{DigestReader, DigestWriter};
#[cfg(feature = "hashes")]
use rscrypto::{
  AsconCxof128, AsconCxof128Reader, AsconXof, AsconXofReader, Blake3, Blake3XofReader, Cshake128, Cshake128XofReader,
  Cshake256, Cshake256XofReader, Digest, FastHash, RapidHash64, RapidHasher, RapidRandomState, RapidSeededState,
  RapidStreamHasher, Shake128, Shake128XofReader, Shake256, Shake256XofReader, Xof, Xxh3,
};
#[cfg(all(feature = "hashes", feature = "diag"))]
use rscrypto::{AsconHash256, Sha3_224, Sha3_256, Sha3_384, Sha3_512, Sha224, Sha384, Sha512, Sha512_256, Xxh3_128};
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
  RsaBlindingPair, RsaEncryptionError, RsaJwtVerifier, RsaKeyError, RsaKeyGenerationError, RsaOaepProfile,
  RsaPkcs1v15Profile, RsaPrivateKey, RsaPrivateKeyParts, RsaPrivateOpError, RsaPrivateScratch,
  RsaProtocolAlgorithmError, RsaPssProfile, RsaPublicExponent, RsaPublicExponentPolicy, RsaPublicKey,
  RsaPublicKeyPolicy, RsaPublicOpError, RsaPublicScratch, RsaSignatureProfile, RsaTlsSignatureSchemes,
  RsaX509PublicKey, RsaX509PublicKeyAlgorithm,
};
use rscrypto::{VerificationError, ct};
#[cfg(feature = "x25519")]
use rscrypto::{X25519Error, X25519PublicKey, X25519SecretKey, X25519SharedSecret};

#[cfg(all(feature = "rsa", feature = "getrandom"))]
fn fill_rsa_random_with(byte: u8) -> impl FnMut(&mut [u8]) -> Result<(), RsaEncryptionError> {
  move |out| {
    out.fill(byte);
    Ok(())
  }
}

#[cfg(all(feature = "rsa", feature = "getrandom"))]
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
  let _error = VerificationError::new();
  let mut secret = [0x5a; 8];
  ct::zeroize(&mut secret);
  assert_eq!(secret, [0; 8]);
}

#[test]
#[cfg(feature = "websocket-sha1")]
fn websocket_accept_digest_stays_on_legacy_module_surface() {
  let digest = WebSocketAcceptDigest::compute(b"dGhlIHNhbXBsZSBub25jZQ==");
  assert_eq!(digest.as_ref().len(), 20);
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

  let _buffer_error = AeadBufferError::new();
  let _seal_buffer_error = SealError::buffer();
  let _seal_length_error = SealError::too_large();
  let _open_buffer_error = OpenError::buffer();
  let _open_length_error = OpenError::too_large();
  let _verification_error = OpenError::verification();

  #[cfg(feature = "diag")]
  {
    let _dispatch = AeadDispatchInfo::current();
    let _aes256gcm_backend = aes256gcm_backend();
    let _aes256gcmsiv_backend = aes256gcmsiv_backend();
    let _chacha20poly1305_backend = chacha20poly1305_backend();
    let _xchacha20poly1305_backend = xchacha20poly1305_backend();
    let _aegis256_backend = aegis256_backend();
    let _ascon_aead128_backend = ascon_aead128_backend();
  }

  fn assert_aead_trait<T: Aead>() {}
  assert_aead_trait::<ChaCha20Poly1305>();

  let key = XChaCha20Poly1305Key::from_bytes([0x44; XChaCha20Poly1305::KEY_SIZE]);
  let cipher = XChaCha20Poly1305::new(&key);
  let mut sealed = [0u8; 20];
  cipher
    .encrypt(&nonce192, b"aad", b"test", &mut sealed)
    .expect("valid XChaCha20-Poly1305 input must encrypt");
  let _tag = XChaCha20Poly1305Tag::from_bytes([0u8; XChaCha20Poly1305Tag::LENGTH]);

  let key = ChaCha20Poly1305Key::from_bytes([0x55; ChaCha20Poly1305::KEY_SIZE]);
  let cipher = ChaCha20Poly1305::new(&key);
  let mut sealed = [0u8; 20];
  cipher
    .encrypt(&nonce96, b"aad", b"test", &mut sealed)
    .expect("valid ChaCha20-Poly1305 input must encrypt");
  let _tag = ChaCha20Poly1305Tag::from_bytes([0u8; ChaCha20Poly1305Tag::LENGTH]);

  let key = Aegis256Key::from_bytes([0x66; Aegis256::KEY_SIZE]);
  let cipher = Aegis256::new(&key);
  let mut sealed = [0u8; 20];
  cipher
    .encrypt(&nonce256, b"aad", b"test", &mut sealed)
    .expect("valid AEGIS-256 input must encrypt");
  let _tag = Aegis256Tag::from_bytes([0u8; Aegis256Tag::LENGTH]);
}

#[test]
#[cfg(feature = "hmac")]
fn root_surface_mac_exports_compile() {
  let key = b"root-surface-key";
  let data = b"root-surface-data";

  let tag = HmacSha256::mac(key, data);
  let tag384 = HmacSha384::mac(key, data);
  let tag512 = HmacSha512::mac(key, data);
  let _tag256 = HmacSha256Tag::from_bytes(tag.to_bytes());
  let _tag384 = HmacSha384Tag::from_bytes(tag384.to_bytes());
  let _tag512 = HmacSha512Tag::from_bytes(tag512.to_bytes());

  let mut mac = HmacSha256::new(key);
  mac.update(data);
  assert!(tag.ct_eq(&mac.finalize()).declassify());
  mac.verify(&tag).expect("matching HMAC-SHA-256 tag must verify");
  let prefix = *tag
    .as_bytes()
    .first_chunk::<8>()
    .expect("an HMAC-SHA-256 tag must contain an eight-byte prefix");
  HmacSha256::verify_truncated_tag_64(key, data, &prefix).expect("matching truncated HMAC-SHA-256 tag must verify");

  let mut mac384 = HmacSha384::new(key);
  mac384.update(data);
  assert!(tag384.ct_eq(&mac384.finalize()).declassify());
  mac384.verify(&tag384).expect("matching HMAC-SHA-384 tag must verify");

  let mut mac512 = HmacSha512::new(key);
  mac512.update(data);
  assert!(tag512.ct_eq(&mac512.finalize()).declassify());
  mac512.verify(&tag512).expect("matching HMAC-SHA-512 tag must verify");
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
  let _tag224 = HmacSha3_224Tag::from_bytes(tag224.to_bytes());
  let _tag256 = HmacSha3_256Tag::from_bytes(tag256.to_bytes());
  let _tag384 = HmacSha3_384Tag::from_bytes(tag384.to_bytes());
  let _tag512 = HmacSha3_512Tag::from_bytes(tag512.to_bytes());

  HmacSha3_224::verify_tag(key, data, &tag224).expect("matching HMAC-SHA3-224 tag must verify");
  HmacSha3_256::verify_tag(key, data, &tag256).expect("matching HMAC-SHA3-256 tag must verify");
  HmacSha3_384::verify_tag(key, data, &tag384).expect("matching HMAC-SHA3-384 tag must verify");
  HmacSha3_512::verify_tag(key, data, &tag512).expect("matching HMAC-SHA3-512 tag must verify");
}

#[test]
#[cfg(feature = "hkdf")]
fn root_surface_kdf_exports_compile() {
  let key = b"root-surface-key";

  let mut out = [0u8; 32];
  let hkdf = HkdfSha256::new(b"salt", key);
  hkdf
    .expand(b"info", &mut out)
    .expect("32-byte HKDF-SHA-256 output must fit");
  assert_eq!(
    out,
    HkdfSha256::derive_array::<32>(b"salt", key, b"info").expect("32-byte HKDF-SHA-256 output must fit")
  );

  let mut out384 = [0u8; 48];
  let hkdf384 = HkdfSha384::new(b"salt", key);
  hkdf384
    .expand(b"info", &mut out384)
    .expect("48-byte HKDF-SHA-384 output must fit");
  assert_eq!(
    out384,
    HkdfSha384::derive_array::<48>(b"salt", key, b"info").expect("48-byte HKDF-SHA-384 output must fit")
  );

  let mut out512 = [0u8; 64];
  let hkdf512 = HkdfSha512::new(b"salt", key);
  hkdf512
    .expand(b"info", &mut out512)
    .expect("64-byte HKDF-SHA-512 output must fit");
  assert_eq!(
    out512,
    HkdfSha512::derive_array::<64>(b"salt", key, b"info").expect("64-byte HKDF-SHA-512 output must fit")
  );
  let _length_error = HkdfOutputLengthError::new();
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
  Kmac128::verify_tag(key, b"svc=v1", data, &out128).expect("matching KMAC128 tag must verify");

  let mut out = [0u8; 32];
  let mut kmac = Kmac256::new(key, b"svc=v1");
  kmac.update(data);
  kmac.finalize_into(&mut out);
  Kmac256::verify_tag(key, b"svc=v1", data, &out).expect("matching KMAC256 tag must verify");
}

#[test]
#[cfg(feature = "poly1305")]
fn root_surface_poly1305_exports_compile() {
  let key = Poly1305OneTimeKey::from_bytes([0x33; Poly1305OneTimeKey::LENGTH]);
  let tag = Poly1305::authenticate_once(key, b"root-surface-poly1305");
  let _tag = Poly1305Tag::from_bytes(tag.to_bytes());

  let key = Poly1305OneTimeKey::from_bytes([0x33; Poly1305OneTimeKey::LENGTH]);
  Poly1305::verify_once(key, b"root-surface-poly1305", &tag).expect("matching Poly1305 tag must verify");
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
  let public: Ed25519PublicKey = keypair.public_key();
  let signature: Ed25519Signature = keypair.sign(b"root-surface-ed25519");

  assert_eq!(secret.as_bytes().len(), 32);
  assert_eq!(public.as_bytes().len(), 32);
  assert_eq!(signature.as_bytes().len(), 64);
  public
    .verify(b"root-surface-ed25519", &signature)
    .expect("matching Ed25519 signature must verify");
}

#[test]
#[cfg(feature = "rsa")]
fn root_surface_rsa_exports_compile() {
  let policy = RsaPublicKeyPolicy::legacy_verification().allow_legacy_small_exponents();
  assert_eq!(policy.min_modulus_bits(), 2048);
  assert_eq!(policy.max_modulus_bits(), 8192);
  let _exponent_policy = RsaPublicExponentPolicy::Common65537;
  let _key_error = RsaKeyError::InvalidModulus;
  let _generation_error = RsaKeyGenerationError::InvalidModulusBits;
  let _encryption_error = RsaEncryptionError::InvalidLength;
  let _private_op_error = RsaPrivateOpError::InvalidLength;
  let _public_op_error = RsaPublicOpError::RepresentativeOutOfRange;
  let _protocol_error = RsaProtocolAlgorithmError::UnsupportedAlgorithm;
  let _oaep_profile = RsaOaepProfile::Sha256;
  let _pss_profile = RsaPssProfile::Sha256;
  let _pkcs1v15_profile = RsaPkcs1v15Profile::Sha256;
  let _public_exponent: Option<RsaPublicExponent> = None;
  let _public_key: Option<RsaPublicKey> = None;
  let _public_scratch: Option<RsaPublicScratch> = None;
  let _jwt_verifier: Option<RsaJwtVerifier<'static>> = None;
  let _private_key: Option<RsaPrivateKey> = None;
  let _private_key_parts: Option<RsaPrivateKeyParts<'static>> = None;
  let _blinding_pair = RsaBlindingPair::new(&[], &[]);
  let _private_scratch: Option<RsaPrivateScratch> = None;
  let _x509_public_key: Option<RsaX509PublicKey> = None;
  assert_eq!(
    RsaSignatureProfile::pss(RsaPssProfile::Sha256).pss_parts(),
    Some((RsaPssProfile::Sha256, 32))
  );
  assert_eq!(
    RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha384).pkcs1v15_profile(),
    Some(RsaPkcs1v15Profile::Sha384)
  );
  assert_eq!(
    RsaSignatureProfile::from_tls13_signature_scheme(0x0804)
      .expect("TLS 1.3 rsa_pss_rsae_sha256 must map to an RSA signature profile"),
    RsaSignatureProfile::pss(RsaPssProfile::Sha256)
  );
  RsaX509PublicKeyAlgorithm::RsaPss
    .permits_signature_profile(RsaSignatureProfile::pss(RsaPssProfile::Sha256))
    .expect("an RSA-PSS key must permit the SHA-256 PSS profile");
  RsaX509PublicKeyAlgorithm::RsaEncryption
    .signature_profile_from_tls13_signature_scheme(0x0804)
    .expect("an rsaEncryption key must accept TLS 1.3 rsa_pss_rsae_sha256");
  let advertised = RsaX509PublicKeyAlgorithm::RsaEncryption.advertised_tls13_signature_schemes();
  assert_eq!(advertised.len(), 3);
  assert!(advertised.contains(0x0804));
  let _maximum_scheme_count = RsaTlsSignatureSchemes::MAX_LEN;
  fn consume_result<T, E>(_: Result<T, E>) {}
  let _caller_random_signing_surface = |key: &RsaPrivateKey, out: &mut [u8], scratch: &mut RsaPrivateScratch| {
    let message = b"root-surface-caller-random-rsa";
    let profile = RsaSignatureProfile::pss(RsaPssProfile::Sha256);
    consume_result(key.sign_signature_with_random_fill(profile, message, out, |_| Ok::<(), ()>(())));
    consume_result(
      key.sign_signature_with_random_fill_and_scratch(profile, message, out, scratch, |_| Ok::<(), ()>(())),
    );
    consume_result(key.sign_pss_with_random_fill(RsaPssProfile::Sha256, message, out, |_| Ok::<(), ()>(())));
    consume_result(
      key.sign_pss_with_random_fill_and_scratch(RsaPssProfile::Sha256, message, out, scratch, |_| Ok::<(), ()>(())),
    );
    consume_result(key.sign_tls13_signature_scheme_with_random_fill(0x0804, message, out, |_| Ok::<(), ()>(())));
    consume_result(
      key.sign_tls13_signature_scheme_with_random_fill_and_scratch(0x0804, message, out, scratch, |_| Ok::<(), ()>(())),
    );
    consume_result(
      key.sign_tls_certificate_signature_scheme_with_random_fill(0x0401, message, out, |_| Ok::<(), ()>(())),
    );
    consume_result(key.sign_tls_certificate_signature_scheme_with_random_fill_and_scratch(
      0x0401,
      message,
      out,
      scratch,
      |_| Ok::<(), ()>(()),
    ));
  };
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
  let key =
    RsaPrivateKey::generate_with_policy(2048, &policy).expect("the supported 2048-bit policy must generate an RSA key");
  let public_key = key.public_key();
  let x509_key = RsaX509PublicKey::from_spki_der_with_policy(&public_key.to_spki_der(), &policy)
    .expect("the generated public key must round-trip through X.509 SPKI");
  let message = b"root-surface-rsa-generated-key";
  let mut private_scratch = key.private_scratch();
  let mut public_scratch = public_key.public_scratch();
  let mut x509_scratch = x509_key.public_key().public_scratch();

  let pkcs1_der = key.to_pkcs1_der();
  let pkcs8_der = key.to_pkcs8_der();
  assert_eq!(format!("{pkcs1_der:?}"), "SecretVec(****)");
  assert_eq!(
    RsaPrivateKey::from_pkcs1_der_with_policy(&pkcs1_der, &policy)
      .expect("the generated private key must round-trip through PKCS#1 DER")
      .public_key(),
    public_key
  );
  assert_eq!(
    RsaPrivateKey::from_pkcs8_der_with_policy(&pkcs8_der, &policy)
      .expect("the generated private key must round-trip through PKCS#8 DER")
      .public_key(),
    public_key
  );
  assert_eq!(
    RsaPublicKey::from_pkcs1_der_with_policy(&public_key.to_pkcs1_der(), &policy)
      .expect("the generated public key must round-trip through PKCS#1 DER"),
    *public_key
  );
  assert_eq!(
    RsaPublicKey::from_spki_der_with_policy(&public_key.to_spki_der(), &policy)
      .expect("the generated public key must round-trip through SPKI DER"),
    *public_key
  );

  let mut unprotected_pkcs1 = key.to_pkcs1_der().into_unprotected_vec();
  assert_eq!(
    RsaPrivateKey::from_pkcs1_der_with_policy(&unprotected_pkcs1, &policy)
      .expect("the unprotected PKCS#1 bytes must parse before zeroization")
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
    key
      .sign_signature(pkcs1v15_profile, message, &mut signature)
      .expect("the generated key must produce a PKCS#1 v1.5 signature");
    public_key
      .verify_signature(pkcs1v15_profile, message, &signature)
      .expect("the generated public key must verify its PKCS#1 v1.5 signature");
    public_key
      .verify_signature_with_scratch(pkcs1v15_profile, message, &signature, &mut public_scratch)
      .expect("scratch-backed PKCS#1 v1.5 verification must accept the matching signature");
    key
      .sign_signature_with_scratch(pkcs1v15_profile, message, &mut signature, &mut private_scratch)
      .expect("scratch-backed PKCS#1 v1.5 signing must succeed");
    public_key
      .verify_signature(pkcs1v15_profile, message, &signature)
      .expect("the generated public key must verify the scratch-backed PKCS#1 v1.5 signature");
    public_key
      .verify_signature_with_scratch(pkcs1v15_profile, message, &signature, &mut public_scratch)
      .expect("scratch-backed verification must accept the scratch-backed PKCS#1 v1.5 signature");

    let pss_profile = RsaSignatureProfile::pss(pss_profile);
    key
      .sign_signature(pss_profile, message, &mut signature)
      .expect("the generated key must produce an RSA-PSS signature");
    public_key
      .verify_signature(pss_profile, message, &signature)
      .expect("the generated public key must verify its RSA-PSS signature");
    public_key
      .verify_signature_with_scratch(pss_profile, message, &signature, &mut public_scratch)
      .expect("scratch-backed RSA-PSS verification must accept the matching signature");
    key
      .sign_signature_with_scratch(pss_profile, message, &mut signature, &mut private_scratch)
      .expect("scratch-backed RSA-PSS signing must succeed");
    public_key
      .verify_signature(pss_profile, message, &signature)
      .expect("the generated public key must verify the scratch-backed RSA-PSS signature");
    public_key
      .verify_signature_with_scratch(pss_profile, message, &signature, &mut public_scratch)
      .expect("scratch-backed verification must accept the scratch-backed RSA-PSS signature");
  }
  let explicit_pss_profile = RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha384, 24);
  key
    .sign_signature_with_scratch(explicit_pss_profile, message, &mut signature, &mut private_scratch)
    .expect("RSA-PSS signing with an explicit salt length must succeed");
  public_key
    .verify_signature(explicit_pss_profile, message, &signature)
    .expect("RSA-PSS verification must honor the explicit salt length");
  public_key
    .verify_signature_with_scratch(explicit_pss_profile, message, &signature, &mut public_scratch)
    .expect("scratch-backed RSA-PSS verification must honor the explicit salt length");

  key
    .sign_x509_signature_algorithm_der(X509_SHA256_WITH_RSA_ENCRYPTION, message, &mut signature)
    .expect("the PKCS#1 v1.5 X.509 algorithm identifier must be accepted for signing");
  x509_key
    .verify_signature_from_x509_algorithm_der(X509_SHA256_WITH_RSA_ENCRYPTION, message, &signature)
    .expect("the X.509 key must verify the matching PKCS#1 v1.5 signature");
  x509_key
    .verify_signature_from_x509_algorithm_der_with_scratch(
      X509_SHA256_WITH_RSA_ENCRYPTION,
      message,
      &signature,
      &mut x509_scratch,
    )
    .expect("scratch-backed X.509 verification must accept the matching PKCS#1 v1.5 signature");
  key
    .sign_x509_signature_algorithm_der_with_scratch(
      X509_PSS_SHA256_ALGORITHM,
      message,
      &mut signature,
      &mut private_scratch,
    )
    .expect("the RSA-PSS X.509 algorithm identifier must be accepted for scratch-backed signing");
  x509_key
    .verify_signature_from_x509_algorithm_der(X509_PSS_SHA256_ALGORITHM, message, &signature)
    .expect("the X.509 key must verify the matching RSA-PSS signature");
  x509_key
    .verify_signature_from_x509_algorithm_der_with_scratch(
      X509_PSS_SHA256_ALGORITHM,
      message,
      &signature,
      &mut x509_scratch,
    )
    .expect("scratch-backed X.509 verification must accept the matching RSA-PSS signature");

  let pss_sha256 = RsaSignatureProfile::pss(RsaPssProfile::Sha256);
  let pkcs1v15_sha256 = RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256);

  key
    .sign_tls13_signature_scheme(0x0804, message, &mut signature)
    .expect("TLS 1.3 rsa_pss_rsae_sha256 signing must succeed");
  x509_key
    .verify_expected_tls13_signature_scheme(0x0804, 0x0804, pss_sha256, message, &signature)
    .expect("the matching TLS 1.3 RSA-PSS signature must verify");
  x509_key
    .verify_expected_tls13_signature_scheme_with_scratch(
      0x0804,
      0x0804,
      pss_sha256,
      message,
      &signature,
      &mut x509_scratch,
    )
    .expect("scratch-backed TLS 1.3 RSA-PSS verification must accept the matching signature");
  key
    .sign_tls13_signature_scheme_with_scratch(0x0804, message, &mut signature, &mut private_scratch)
    .expect("scratch-backed TLS 1.3 RSA-PSS signing must succeed");
  x509_key
    .verify_expected_tls13_signature_scheme(0x0804, 0x0804, pss_sha256, message, &signature)
    .expect("the scratch-backed TLS 1.3 RSA-PSS signature must verify");
  x509_key
    .verify_expected_tls13_signature_scheme_with_scratch(
      0x0804,
      0x0804,
      pss_sha256,
      message,
      &signature,
      &mut x509_scratch,
    )
    .expect("scratch-backed verification must accept the scratch-backed TLS 1.3 RSA-PSS signature");

  key
    .sign_tls_certificate_signature_scheme(0x0401, message, &mut signature)
    .expect("TLS certificate rsa_pkcs1_sha256 signing must succeed");
  x509_key
    .verify_expected_tls_certificate_signature_scheme(0x0401, 0x0401, pkcs1v15_sha256, message, &signature)
    .expect("the matching TLS certificate PKCS#1 v1.5 signature must verify");
  x509_key
    .verify_expected_tls_certificate_signature_scheme_with_scratch(
      0x0401,
      0x0401,
      pkcs1v15_sha256,
      message,
      &signature,
      &mut x509_scratch,
    )
    .expect("scratch-backed TLS certificate verification must accept the matching signature");
  key
    .sign_tls_certificate_signature_scheme_with_scratch(0x0401, message, &mut signature, &mut private_scratch)
    .expect("scratch-backed TLS certificate PKCS#1 v1.5 signing must succeed");
  x509_key
    .verify_expected_tls_certificate_signature_scheme(0x0401, 0x0401, pkcs1v15_sha256, message, &signature)
    .expect("the scratch-backed TLS certificate signature must verify");
  x509_key
    .verify_expected_tls_certificate_signature_scheme_with_scratch(
      0x0401,
      0x0401,
      pkcs1v15_sha256,
      message,
      &signature,
      &mut x509_scratch,
    )
    .expect("scratch-backed verification must accept the scratch-backed TLS certificate signature");

  key
    .jwt_signer(RsaJwtAlgorithm::Ps256)
    .try_sign_into(message, &mut signature)
    .expect("PS256 JWT signing must succeed");
  let verifier = public_key.jwt_verifier(RsaJwtAlgorithm::Ps256);
  verifier
    .verify("PS256", message, &signature)
    .expect("the matching PS256 JWT signature must verify");
  verifier
    .verify_with_scratch("PS256", message, &signature, &mut public_scratch)
    .expect("scratch-backed PS256 JWT verification must accept the matching signature");
  key
    .sign_signature_with_scratch(
      RsaJwtAlgorithm::Rs256.signature_profile(),
      message,
      &mut signature,
      &mut private_scratch,
    )
    .expect("scratch-backed RS256 signing must succeed");
  let verifier = public_key.jwt_verifier(RsaJwtAlgorithm::Rs256);
  verifier
    .verify("RS256", message, &signature)
    .expect("the matching RS256 JWT signature must verify");
  verifier
    .verify_with_scratch("RS256", message, &signature, &mut public_scratch)
    .expect("scratch-backed RS256 JWT verification must accept the matching signature");

  key
    .sign_cose_algorithm_id(-37, message, &mut signature)
    .expect("COSE PS256 signing must succeed");
  public_key
    .verify_expected_cose_algorithm_id(-37, -37, pss_sha256, message, &signature)
    .expect("the matching COSE PS256 signature must verify");
  public_key
    .verify_expected_cose_algorithm_id_with_scratch(-37, -37, pss_sha256, message, &signature, &mut public_scratch)
    .expect("scratch-backed COSE PS256 verification must accept the matching signature");
  key
    .sign_cose_algorithm_id_with_scratch(-257, message, &mut signature, &mut private_scratch)
    .expect("scratch-backed COSE RS256 signing must succeed");
  public_key
    .verify_expected_cose_algorithm_id(-257, -257, pkcs1v15_sha256, message, &signature)
    .expect("the matching COSE RS256 signature must verify");
  public_key
    .verify_expected_cose_algorithm_id_with_scratch(
      -257,
      -257,
      pkcs1v15_sha256,
      message,
      &signature,
      &mut public_scratch,
    )
    .expect("scratch-backed COSE RS256 verification must accept the matching signature");

  let label = b"root-surface-rsa-label";
  let plaintext = b"root-surface-rsa-oaep";
  let mut ciphertext = vec![0u8; key.signature_len()];
  let mut decrypted = vec![0u8; key.signature_len()];
  for oaep_profile in [RsaOaepProfile::Sha256, RsaOaepProfile::Sha384, RsaOaepProfile::Sha512] {
    public_key
      .encrypt_oaep(oaep_profile, label, plaintext, &mut ciphertext)
      .expect("OAEP encryption of the bounded plaintext must succeed");
    public_key
      .encrypt_oaep_with_scratch(oaep_profile, label, plaintext, &mut ciphertext, &mut public_scratch)
      .expect("scratch-backed OAEP encryption of the bounded plaintext must succeed");
    let decrypted_len = key
      .decrypt_oaep(oaep_profile, label, &ciphertext, &mut decrypted)
      .expect("OAEP decryption with the matching key and label must succeed");
    assert_eq!(&decrypted[..decrypted_len], plaintext);
    let decrypted_len = key
      .decrypt_oaep_with_scratch(oaep_profile, label, &ciphertext, &mut decrypted, &mut private_scratch)
      .expect("scratch-backed OAEP decryption with the matching key and label must succeed");
    assert_eq!(&decrypted[..decrypted_len], plaintext);
  }

  let legacy_plaintext = b"root-surface-rsaes-pkcs1v15";
  public_key
    .encrypt_pkcs1v15(legacy_plaintext, &mut ciphertext)
    .expect("PKCS#1 v1.5 encryption of the bounded plaintext must succeed");
  public_key
    .encrypt_pkcs1v15_with_scratch(legacy_plaintext, &mut ciphertext, &mut public_scratch)
    .expect("scratch-backed PKCS#1 v1.5 encryption of the bounded plaintext must succeed");
  let decrypted_len = key
    .decrypt_pkcs1v15(&ciphertext, &mut decrypted)
    .expect("PKCS#1 v1.5 decryption with the matching key must succeed");
  assert_eq!(&decrypted[..decrypted_len], legacy_plaintext);
  let decrypted_len = key
    .decrypt_pkcs1v15_with_scratch(&ciphertext, &mut decrypted, &mut private_scratch)
    .expect("scratch-backed PKCS#1 v1.5 decryption with the matching key must succeed");
  assert_eq!(&decrypted[..decrypted_len], legacy_plaintext);
}

#[test]
#[cfg(all(feature = "rsa", feature = "getrandom"))]
fn root_surface_rsa_default_generated_key_end_to_end() {
  let key = RsaPrivateKey::generate(3072).expect("the default policy must generate a 3072-bit RSA key");
  assert_eq!(key.public_key().modulus_bits(), 3072);
  let public_key = key.public_key();
  let message = b"root-surface-rsa-default-generated-key";
  let mut private_scratch = key.private_scratch();
  let mut public_scratch = public_key.public_scratch();

  let pkcs1_der = key.to_pkcs1_der();
  let pkcs8_der = key.to_pkcs8_der();
  assert_eq!(
    RsaPrivateKey::from_pkcs1_der(&pkcs1_der)
      .expect("the default-policy private key must round-trip through PKCS#1 DER")
      .public_key(),
    public_key
  );
  assert_eq!(
    RsaPrivateKey::from_pkcs8_der(&pkcs8_der)
      .expect("the default-policy private key must round-trip through PKCS#8 DER")
      .public_key(),
    public_key
  );
  assert_eq!(
    RsaPublicKey::from_pkcs1_der(&public_key.to_pkcs1_der())
      .expect("the default-policy public key must round-trip through PKCS#1 DER"),
    *public_key
  );
  assert_eq!(
    RsaPublicKey::from_spki_der(&public_key.to_spki_der())
      .expect("the default-policy public key must round-trip through SPKI DER"),
    *public_key
  );

  let mut signature = vec![0u8; key.signature_len()];
  let pkcs1v15_profile = RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256);
  key
    .sign_signature_with_scratch(pkcs1v15_profile, message, &mut signature, &mut private_scratch)
    .expect("default-policy scratch-backed PKCS#1 v1.5 signing must succeed");
  public_key
    .verify_signature_with_scratch(pkcs1v15_profile, message, &signature, &mut public_scratch)
    .expect("default-policy scratch-backed PKCS#1 v1.5 verification must succeed");

  let pss_profile = RsaSignatureProfile::pss(RsaPssProfile::Sha256);
  key
    .sign_signature_with_scratch(pss_profile, message, &mut signature, &mut private_scratch)
    .expect("default-policy scratch-backed RSA-PSS signing must succeed");
  public_key
    .verify_signature_with_scratch(pss_profile, message, &signature, &mut public_scratch)
    .expect("default-policy scratch-backed RSA-PSS verification must succeed");

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
    .expect("deterministic scratch-backed OAEP encryption must succeed");
  let decrypted_len = key
    .decrypt_oaep_with_scratch(
      RsaOaepProfile::Sha256,
      label,
      &ciphertext,
      &mut decrypted,
      &mut private_scratch,
    )
    .expect("scratch-backed OAEP decryption of the deterministic ciphertext must succeed");
  assert_eq!(&decrypted[..decrypted_len], plaintext);

  let legacy_plaintext = b"root-surface-rsa-default-rsaes-pkcs1v15";
  public_key
    .encrypt_pkcs1v15_with_random_fill_and_scratch(
      legacy_plaintext,
      &mut ciphertext,
      &mut public_scratch,
      fill_rsa_random_with(0x5b),
    )
    .expect("deterministic scratch-backed PKCS#1 v1.5 encryption must succeed");
  let decrypted_len = key
    .decrypt_pkcs1v15_with_scratch(&ciphertext, &mut decrypted, &mut private_scratch)
    .expect("scratch-backed PKCS#1 v1.5 decryption of the deterministic ciphertext must succeed");
  assert_eq!(&decrypted[..decrypted_len], legacy_plaintext);
}

#[test]
#[cfg(feature = "x25519")]
fn root_surface_key_exchange_exports_compile() {
  let alice = X25519SecretKey::from_bytes([11u8; X25519SecretKey::LENGTH]);
  let bob = X25519SecretKey::from_bytes([13u8; X25519SecretKey::LENGTH]);
  let alice_public: X25519PublicKey = (&alice).into();
  let bob_public: X25519PublicKey = (&bob).into();
  let alice_shared = alice
    .diffie_hellman(&bob_public)
    .expect("the fixed Alice and Bob X25519 keys must produce a shared secret");
  let bob_shared = X25519SharedSecret::diffie_hellman(&bob, &alice_public)
    .expect("the fixed Bob and Alice X25519 keys must produce a shared secret");

  assert_eq!(alice_public.as_bytes().len(), 32);
  assert_eq!(alice_shared.as_bytes().len(), 32);
  assert!(alice_shared.ct_eq(&bob_shared).declassify());
  let _error = X25519Error::new();
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
  let _crc16_ccitt = Crc16Ccitt::checksum(data);
  let _crc16_ibm = Crc16Ibm::checksum(data);
  let _crc24_openpgp = Crc24OpenPgp::checksum(data);
  let _crc64_nvme = Crc64Nvme::checksum(data);
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
  use core::hash::{BuildHasher, Hasher};

  let data = b"root-surface";

  let oneshot = Blake3::digest(data);

  let mut streaming = Blake3::new();
  streaming.update(data);
  assert_eq!(oneshot, streaming.finalize());

  let mut xof: Blake3XofReader = Blake3::xof(data);
  let mut out = [0u8; 16];
  xof.squeeze(&mut out);

  let mut shake128: Shake128XofReader = Shake128::xof(data);
  shake128.squeeze(&mut out);

  let mut shake: Shake256XofReader = Shake256::xof(data);
  shake.squeeze(&mut out);

  let mut ascon: AsconXofReader = AsconXof::xof(data);
  ascon.squeeze(&mut out);

  let mut cshake: Cshake256XofReader = Cshake256::xof(b"", b"ctx=v1", data);
  cshake.squeeze(&mut out);

  let mut cshake128: Cshake128XofReader = Cshake128::xof(b"", b"ctx=v1", data);
  cshake128.squeeze(&mut out);
  let mut cxof: AsconCxof128Reader =
    AsconCxof128::xof(b"ctx=v1", data).expect("the short Ascon-CXOF customization must be accepted");
  cxof.squeeze(&mut out);

  assert_eq!(Xxh3::hash(data), Xxh3_64::hash(data));
  let _rapid_hash = RapidHash64::hash(data);
  let deterministic = RapidSeededState::new(42);
  let mut collection: RapidHasher = deterministic.build_hasher();
  collection.write(data);
  let _collection_hash = collection.finish();

  let random = RapidRandomState::try_new_with(|seed| {
    seed.copy_from_slice(&42u64.to_le_bytes());
    Ok::<_, ()>(())
  })
  .expect("the deterministic seed filler must initialize RapidHash state");
  let _randomized_hash = random.hash_one(data);

  let mut stream = RapidStreamHasher::new();
  stream.write(data);
  assert_eq!(stream.finish(), RapidHash64::hash(data));
  assert_ne!(Xxh3::hash(data), Xxh3::hash_with_seed(7, data));
}

#[test]
#[cfg(all(feature = "hashes", feature = "std"))]
fn digest_reader_writer_round_trip() {
  use std::io::Write;

  let data = b"hello digest reader writer";
  let expected = Sha256::digest(data);

  // DigestReader: read data through and verify digest matches.
  let mut reader = DigestReader::<_, Sha256>::new(data.as_slice());
  let mut sink = Vec::new();
  std::io::copy(&mut reader, &mut sink).expect("copying from an in-memory digest reader must succeed");
  assert_eq!(reader.digest(), expected);

  // DigestWriter: write data through and verify digest matches.
  let mut writer = DigestWriter::<_, Sha256>::new(Vec::new());
  writer
    .write_all(data)
    .expect("writing to an in-memory digest writer must succeed");
  let (out, digest) = writer.into_parts();
  assert_eq!(&out, data);
  assert_eq!(digest, expected);
}

#[test]
#[cfg(all(feature = "checksums", feature = "diag"))]
fn advanced_checksum_modules_compile() {
  fn assert_kernel_introspect<T: KernelIntrospect>() {}

  let _crc16_config: Crc16Config = Crc16Ccitt::config();
  let _crc24_config: Crc24Config = Crc24OpenPgp::config();
  let _crc32_config: Crc32Config = Crc32::config();
  let _crc64_config: Crc64Config = Crc64::config();
  let _crc16_force = Crc16Force::Auto;
  let _crc24_force = Crc24Force::Auto;
  let _crc32_force = Crc32Force::Auto;
  let _crc64_force = Crc64Force::Auto;
  let _dispatch = DispatchInfo::current();
  let _kernel = kernel_for::<Crc32>(64);
  let _accelerated = is_hardware_accelerated();
  let _platform = rscrypto::platform::describe();
  assert_kernel_introspect::<Crc32>();
}

#[test]
#[cfg(all(feature = "hashes", feature = "diag"))]
fn advanced_hash_modules_compile() {
  fn assert_hash_kernel_introspect<T: HashKernelIntrospect>() {}

  let _dispatch = HashDispatchInfo::current();
  let _sha256_kernel = hash_kernel_for::<Sha256>(64);
  let _shake256_kernel = hash_kernel_for::<Shake256>(64);
  let _blake3_kernel = hash_kernel_for::<Blake3>(64);
  let _ascon_hash_kernel = hash_kernel_for::<AsconHash256>(64);
  let _ascon_xof_kernel = hash_kernel_for::<AsconXof>(64);
  let _ascon_cxof_kernel = hash_kernel_for::<AsconCxof128>(64);
  let _xxh3_kernel = hash_kernel_for::<Xxh3>(64);

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
}
