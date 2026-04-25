#![allow(unused_imports)]

#[cfg(all(feature = "checksums", feature = "std"))]
use std::io::{Cursor, Read, Write};

#[cfg(feature = "kmac")]
use rscrypto::Kmac256;
use rscrypto::VerificationError;
#[cfg(feature = "aead")]
use rscrypto::{
  Aead, Aegis256, Aegis256Key, Aes256Gcm, Aes256GcmKey, Aes256GcmSiv, Aes256GcmSivKey, AsconAead128, AsconAead128Key,
  ChaCha20Poly1305, ChaCha20Poly1305Key, XChaCha20Poly1305, XChaCha20Poly1305Key,
  aead::{AeadBufferError, Nonce96, Nonce128, Nonce192, Nonce256, OpenError, SealError},
};
#[cfg(feature = "hashes")]
use rscrypto::{
  AsconCxof128, AsconXof, Blake3, Cshake256, Digest, Sha3_224, Sha3_256, Sha3_384, Sha3_512, Sha224, Sha256, Sha384,
  Sha512, Sha512_256, Shake128, Shake256, Xof,
};
#[cfg(feature = "ed25519")]
use rscrypto::{Ed25519Keypair, Ed25519PublicKey, Ed25519SecretKey};
#[cfg(feature = "hkdf")]
use rscrypto::{HkdfSha256, HkdfSha384, auth::HkdfOutputLengthError};
#[cfg(feature = "hmac")]
use rscrypto::{HmacSha256, HmacSha384, HmacSha512, Mac};
#[cfg(feature = "x25519")]
use rscrypto::{X25519Error, X25519PublicKey, X25519SecretKey};

#[cfg(feature = "hashes")]
fn assert_digest_api<D>()
where
  D: Digest,
  D::Output: PartialEq + core::fmt::Debug,
{
  let mut h = D::new();
  h.update(b"abc");
  let expected = h.finalize();
  h.reset();
  h.update(b"abc");
  assert_eq!(h.finalize(), expected);
}

#[cfg(feature = "hashes")]
fn squeeze_32(mut reader: impl Xof) -> [u8; 32] {
  let mut out = [0u8; 32];
  reader.squeeze(&mut out);
  out
}

#[cfg(feature = "hmac")]
fn assert_mac_api<M>()
where
  M: Mac,
  M::Tag: PartialEq + core::fmt::Debug,
{
  let key = b"api-consistency-key";
  let mut mac = M::new(key);
  mac.update(b"abc");
  let expected = mac.finalize();
  mac.reset();
  mac.update(b"abc");
  assert_eq!(mac.finalize(), expected);
  assert!(mac.verify(&expected).is_ok());
}

#[cfg(feature = "aead")]
fn assert_aead_api<A>(key: A::Key, nonce: A::Nonce)
where
  A: Aead,
{
  let aead = A::new(&key);
  let plaintext = b"abc";

  let mut sealed = [0u8; 19];
  aead.encrypt(&nonce, b"aad", plaintext, &mut sealed).unwrap();

  let mut opened = [0u8; 3];
  aead.decrypt(&nonce, b"aad", &sealed, &mut opened).unwrap();
  assert_eq!(&opened, plaintext);

  let mut detached = *b"abc";
  let tag = aead.encrypt_in_place_detached(&nonce, b"aad", &mut detached).unwrap();
  aead.decrypt_in_place(&nonce, b"aad", &mut detached, &tag).unwrap();
  assert_eq!(&detached, plaintext);
}

#[test]
#[cfg(feature = "hashes")]
fn all_digests_follow_new_update_finalize_reset() {
  assert_digest_api::<Sha224>();
  assert_digest_api::<Sha256>();
  assert_digest_api::<Sha384>();
  assert_digest_api::<Sha512>();
  assert_digest_api::<Sha512_256>();
  assert_digest_api::<Sha3_224>();
  assert_digest_api::<Sha3_256>();
  assert_digest_api::<Sha3_384>();
  assert_digest_api::<Sha3_512>();
  assert_digest_api::<Blake3>();
}

#[test]
#[cfg(feature = "hashes")]
fn all_xofs_follow_new_update_finalize_xof_and_xof() {
  macro_rules! assert_xof_api {
    ($ty:ty) => {{
      let data = b"abc";
      let mut h = <$ty>::new();
      h.update(data);
      let streaming = squeeze_32(h.clone().finalize_xof());
      h.reset();
      let oneshot = squeeze_32(<$ty>::xof(data));
      assert_eq!(streaming, oneshot);
    }};
  }

  assert_xof_api!(Shake128);
  assert_xof_api!(Shake256);
  assert_xof_api!(Blake3);
  assert_xof_api!(AsconXof);

  let data = b"abc";
  let mut cshake = Cshake256::new(b"", b"ctx=v1");
  cshake.update(data);
  let streaming = squeeze_32(cshake.finalize_xof());
  cshake.reset();
  cshake.update(data);
  assert_eq!(streaming, squeeze_32(cshake.finalize_xof()));

  let mut cxof = AsconCxof128::new(b"ctx=v1").unwrap();
  cxof.update(data);
  let streaming = squeeze_32(cxof.finalize_xof());
  cxof.reset();
  cxof.update(data);
  assert_eq!(streaming, squeeze_32(cxof.finalize_xof()));
}

#[test]
#[cfg(feature = "hmac")]
fn all_macs_follow_new_update_finalize_reset() {
  assert_mac_api::<HmacSha256>();
  assert_mac_api::<HmacSha384>();
  assert_mac_api::<HmacSha512>();
}

#[test]
#[cfg(feature = "kmac")]
fn kmac_follows_new_update_finalize_into_reset_and_verify() {
  let mut kmac = Kmac256::new(b"api-consistency-key", b"ctx=v1");
  kmac.update(b"abc");
  let expected = Kmac256::mac_array::<32>(b"api-consistency-key", b"ctx=v1", b"abc");
  let mut actual = [0u8; 32];
  kmac.finalize_into(&mut actual);
  assert_eq!(actual, expected);
  kmac.reset();
  kmac.update(b"abc");
  kmac.finalize_into(&mut actual);
  assert_eq!(actual, expected);
  assert!(kmac.verify(&expected).is_ok());
}

#[test]
#[cfg(feature = "hkdf")]
fn hkdfs_follow_new_expand_and_derive_array_conventions() {
  let hkdf256 = HkdfSha256::new(b"salt", b"ikm");
  let hkdf384 = HkdfSha384::new(b"salt", b"ikm");

  let okm256 = hkdf256.expand_array::<32>(b"info").unwrap();
  let okm384 = hkdf384.expand_array::<48>(b"info").unwrap();

  assert_eq!(
    okm256,
    HkdfSha256::derive_array::<32>(b"salt", b"ikm", b"info").unwrap()
  );
  assert_eq!(
    okm384,
    HkdfSha384::derive_array::<48>(b"salt", b"ikm", b"info").unwrap()
  );
  assert_eq!(
    HkdfSha256::derive_array::<32>(b"salt", b"ikm", b"info").unwrap(),
    okm256
  );
  assert_eq!(
    HkdfSha384::derive_array::<48>(b"salt", b"ikm", b"info").unwrap(),
    okm384
  );
  assert_eq!(hkdf256.prk().len(), 32);
  assert_eq!(hkdf384.prk().len(), 48);

  let mut oversized = vec![0u8; HkdfSha256::MAX_OUTPUT_SIZE + 1];
  assert_eq!(
    HkdfSha256::derive(b"salt", b"ikm", b"info", &mut oversized),
    Err(HkdfOutputLengthError::new())
  );
}

#[test]
#[cfg(feature = "ed25519")]
fn ed25519_types_follow_byte_roundtrip_and_verify_conventions() {
  let secret = Ed25519SecretKey::from_bytes([0x24; Ed25519SecretKey::LENGTH]);
  let keypair = Ed25519Keypair::from_secret_key(secret.clone());
  let public = Ed25519PublicKey::from_bytes(keypair.public_key().to_bytes());
  let signature = keypair.sign(b"api-consistency-ed25519");

  assert_eq!(*secret.expose_secret().as_bytes(), *secret.as_bytes());
  assert_eq!(public.to_bytes(), *public.as_bytes());
  assert_eq!(signature.to_bytes(), *signature.as_bytes());
  assert!(public.verify(b"api-consistency-ed25519", &signature).is_ok());
}

#[test]
#[cfg(feature = "x25519")]
fn x25519_types_follow_byte_roundtrip_conventions() {
  let secret = X25519SecretKey::from_bytes([0x42; X25519SecretKey::LENGTH]);
  let public = X25519PublicKey::from_bytes(secret.public_key().to_bytes());
  let shared = secret.diffie_hellman(&public).unwrap();

  assert_eq!(*secret.expose_secret().as_bytes(), *secret.as_bytes());
  assert_eq!(public.to_bytes(), *public.as_bytes());
  assert_eq!(*shared.expose_secret().as_bytes(), *shared.as_bytes());
}

#[test]
fn verification_error_follows_new_default_and_display_conventions() {
  assert_eq!(VerificationError::new(), VerificationError::default());
  assert_eq!(VerificationError::new().to_string(), "verification failed");
}

#[test]
#[cfg(feature = "hkdf")]
fn hkdf_error_follows_new_default_and_display_conventions() {
  assert_eq!(HkdfOutputLengthError::new(), HkdfOutputLengthError);
  assert_eq!(
    HkdfOutputLengthError::new().to_string(),
    "requested HKDF output exceeds the algorithm maximum"
  );
}

#[test]
#[cfg(feature = "x25519")]
fn x25519_error_follows_new_default_and_display_conventions() {
  assert_eq!(X25519Error::new(), X25519Error);
  assert_eq!(X25519Error::new().to_string(), "x25519 shared secret is all-zero");
}

#[test]
#[cfg(feature = "aead")]
fn aead_errors_follow_new_default_and_display_conventions() {
  assert_eq!(AeadBufferError::new(), AeadBufferError);
  assert_eq!(AeadBufferError::new().to_string(), "buffer length mismatch");
  assert_eq!(SealError::default(), SealError::buffer());
  assert_eq!(
    SealError::too_large().to_string(),
    "input exceeds the algorithm maximum length"
  );
  assert_eq!(OpenError::default(), OpenError::buffer());
  assert_eq!(OpenError::buffer().to_string(), "buffer length mismatch");
  assert_eq!(
    OpenError::too_large().to_string(),
    "input exceeds the algorithm maximum length"
  );
  assert_eq!(OpenError::verification().to_string(), "verification failed");
}

#[test]
#[cfg(feature = "aead")]
fn all_aeads_follow_new_encrypt_decrypt_and_detached_aliases() {
  assert_aead_api::<Aes256Gcm>(
    Aes256GcmKey::from_bytes([1u8; Aes256Gcm::KEY_SIZE]),
    Nonce96::from_bytes([2u8; Nonce96::LENGTH]),
  );
  assert_aead_api::<Aes256GcmSiv>(
    Aes256GcmSivKey::from_bytes([3u8; Aes256GcmSiv::KEY_SIZE]),
    Nonce96::from_bytes([4u8; Nonce96::LENGTH]),
  );
  assert_aead_api::<ChaCha20Poly1305>(
    ChaCha20Poly1305Key::from_bytes([5u8; ChaCha20Poly1305::KEY_SIZE]),
    Nonce96::from_bytes([6u8; Nonce96::LENGTH]),
  );
  assert_aead_api::<XChaCha20Poly1305>(
    XChaCha20Poly1305Key::from_bytes([7u8; XChaCha20Poly1305::KEY_SIZE]),
    Nonce192::from_bytes([8u8; Nonce192::LENGTH]),
  );
  assert_aead_api::<AsconAead128>(
    AsconAead128Key::from_bytes([9u8; AsconAead128::KEY_SIZE]),
    Nonce128::from_bytes([10u8; Nonce128::LENGTH]),
  );
  assert_aead_api::<Aegis256>(
    Aegis256Key::from_bytes([11u8; Aegis256::KEY_SIZE]),
    Nonce256::from_bytes([12u8; Nonce256::LENGTH]),
  );
}

#[test]
#[cfg(all(feature = "checksums", feature = "std"))]
fn checksum_adapters_use_checksum() -> std::io::Result<()> {
  use rscrypto::{Checksum as _, Crc32C};

  let mut reader = Crc32C::reader(Cursor::new(b"abc".to_vec()));
  std::io::copy(&mut reader, &mut std::io::sink())?;
  assert_eq!(reader.checksum(), Crc32C::checksum(b"abc"));

  let mut writer = Crc32C::writer(Vec::new());
  writer.write_all(b"abc")?;
  assert_eq!(writer.checksum(), Crc32C::checksum(b"abc"));

  Ok(())
}
