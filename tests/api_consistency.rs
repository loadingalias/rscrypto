#![allow(unused_imports)]

#[cfg(all(feature = "checksums", feature = "std"))]
use std::io::{Cursor, Read, Write};

#[cfg(any(feature = "ecdsa-p256", feature = "ecdsa-p384"))]
use rscrypto::EcdsaKeyGenerationError;
#[cfg(any(feature = "hmac", feature = "hmac-sha3"))]
use rscrypto::Mac;
#[cfg(feature = "aead")]
use rscrypto::{
  Aead, Aegis256, Aegis256Key, Aes128Gcm, Aes128GcmKey, Aes128GcmSiv, Aes128GcmSivKey, Aes256Gcm, Aes256GcmKey,
  Aes256GcmSiv, Aes256GcmSivKey, AsconAead128, AsconAead128Key, ChaCha20Poly1305, ChaCha20Poly1305Key,
  XChaCha20Poly1305, XChaCha20Poly1305Key,
  aead::{AeadBufferError, Nonce96, Nonce128, Nonce192, Nonce256, OpenError, SealError},
};
#[cfg(feature = "hashes")]
use rscrypto::{
  AsconCxof128, AsconXof, Blake3, Cshake128, Cshake256, Digest, Sha3_224, Sha3_256, Sha3_384, Sha3_512, Sha224, Sha256,
  Sha384, Sha512, Sha512_256, Shake128, Shake256, Xof,
};
#[cfg(feature = "ecdsa-p256")]
use rscrypto::{EcdsaP256Keypair, EcdsaP256PublicKey, EcdsaP256SecretKey};
#[cfg(feature = "ecdsa-p384")]
use rscrypto::{EcdsaP384Keypair, EcdsaP384PublicKey, EcdsaP384SecretKey};
#[cfg(feature = "ed25519")]
use rscrypto::{Ed25519Keypair, Ed25519PublicKey, Ed25519SecretKey};
#[cfg(feature = "hkdf")]
use rscrypto::{HkdfSha256, HkdfSha384, HkdfSha512, auth::HkdfOutputLengthError};
#[cfg(feature = "hmac-sha3")]
use rscrypto::{HmacSha3_224, HmacSha3_256, HmacSha3_384, HmacSha3_512};
#[cfg(feature = "hmac")]
use rscrypto::{HmacSha256, HmacSha384, HmacSha512};
use rscrypto::{Kem, TrySigner, TrySignerInto, VerificationError, Verifier};
#[cfg(feature = "kmac")]
use rscrypto::{Kmac128, Kmac256};
#[cfg(feature = "poly1305")]
use rscrypto::{Poly1305, Poly1305OneTimeKey, Poly1305Tag};
#[cfg(feature = "rsa")]
use rscrypto::{RsaPkcs1v15Profile, RsaPssProfile, RsaPublicKey, RsaSignatureProfile};
#[cfg(feature = "x25519")]
use rscrypto::{X25519Error, X25519PublicKey, X25519SecretKey};

#[derive(Clone)]
struct ToyKem;

impl Kem for ToyKem {
  const ENCAPSULATION_KEY_SIZE: usize = 3;
  const DECAPSULATION_KEY_SIZE: usize = 4;
  const CIPHERTEXT_SIZE: usize = 5;
  const SHARED_SECRET_SIZE: usize = 2;

  type EncapsulationKey = [u8; Self::ENCAPSULATION_KEY_SIZE];
  type DecapsulationKey = [u8; Self::DECAPSULATION_KEY_SIZE];
  type Ciphertext = [u8; Self::CIPHERTEXT_SIZE];
  type SharedSecret = [u8; Self::SHARED_SECRET_SIZE];
  type KeyGenerationError = ();
  type EncapsulationError = ();
  type DecapsulationError = ();

  fn generate_keypair(
    mut fill_random: impl FnMut(&mut [u8]) -> Result<(), Self::KeyGenerationError>,
  ) -> Result<(Self::EncapsulationKey, Self::DecapsulationKey), Self::KeyGenerationError> {
    let mut seed = [0u8; 4];
    fill_random(&mut seed)?;
    Ok(([seed[0], seed[1], seed[2]], seed))
  }

  fn encapsulate(
    encapsulation_key: &Self::EncapsulationKey,
    mut fill_random: impl FnMut(&mut [u8]) -> Result<(), Self::EncapsulationError>,
  ) -> Result<(Self::Ciphertext, Self::SharedSecret), Self::EncapsulationError> {
    let mut nonce = [0u8; 2];
    fill_random(&mut nonce)?;
    Ok((
      [
        encapsulation_key[0],
        encapsulation_key[1],
        encapsulation_key[2],
        nonce[0],
        nonce[1],
      ],
      nonce,
    ))
  }

  fn decapsulate(
    decapsulation_key: &Self::DecapsulationKey,
    ciphertext: &Self::Ciphertext,
  ) -> Result<Self::SharedSecret, Self::DecapsulationError> {
    Ok([decapsulation_key[3], ciphertext[4]])
  }
}

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

#[cfg(any(feature = "hmac", feature = "hmac-sha3"))]
fn assert_mac_api<M: Mac>() {
  let key = b"api-consistency-key";
  let mut mac = M::new(key);
  mac.update(b"abc");
  let expected = mac.finalize();
  mac.reset();
  mac.update(b"abc");
  assert!(mac.verify(&expected).is_ok());
}

#[cfg(any(feature = "hmac", feature = "hmac-sha3"))]
struct NonCloneMac {
  key: u8,
  state: u8,
}

#[cfg(any(feature = "hmac", feature = "hmac-sha3"))]
impl Mac for NonCloneMac {
  const TAG_SIZE: usize = 1;
  type Tag = [u8; Self::TAG_SIZE];

  fn new(key: &[u8]) -> Self {
    Self {
      key: key.first().copied().unwrap_or(0),
      state: 0,
    }
  }

  fn update(&mut self, data: &[u8]) {
    self.state = data.iter().fold(self.state, |state, byte| state.wrapping_add(*byte));
  }

  fn finalize(&self) -> Self::Tag {
    [self.state ^ self.key]
  }

  fn reset(&mut self) {
    self.state = 0;
  }

  fn verify(&self, expected: &Self::Tag) -> Result<(), VerificationError> {
    if self.finalize() == *expected {
      Ok(())
    } else {
      Err(VerificationError::new())
    }
  }
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

  #[cfg(feature = "alloc")]
  {
    let sealed_vec = aead.encrypt_to_vec(&nonce, b"aad", plaintext).unwrap();
    assert_eq!(sealed_vec.as_slice(), sealed);
    let opened_vec = aead.decrypt_to_vec(&nonce, b"aad", &sealed_vec).unwrap();
    assert_eq!(opened_vec.as_slice(), plaintext);

    let mut tampered = sealed_vec;
    tampered[0] ^= 1;
    assert!(aead.decrypt_to_vec(&nonce, b"aad", &tampered).is_err());
  }

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
  let mut cshake128 = Cshake128::new(b"", b"ctx=v1");
  cshake128.update(data);
  let streaming128 = squeeze_32(cshake128.finalize_xof());
  cshake128.reset();
  cshake128.update(data);
  assert_eq!(streaming128, squeeze_32(cshake128.finalize_xof()));

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
#[cfg(any(feature = "hmac", feature = "hmac-sha3"))]
fn mac_trait_accepts_non_clone_implementations() {
  assert_mac_api::<NonCloneMac>();
}

#[test]
#[cfg(feature = "hmac-sha3")]
fn hmac_sha3_macs_follow_new_update_finalize_reset() {
  assert_mac_api::<HmacSha3_224>();
  assert_mac_api::<HmacSha3_256>();
  assert_mac_api::<HmacSha3_384>();
  assert_mac_api::<HmacSha3_512>();
}

#[test]
#[cfg(feature = "kmac")]
fn kmac_follows_new_update_finalize_into_reset_and_verify() {
  let mut kmac128 = Kmac128::new(b"api-consistency-key", b"ctx=v1");
  kmac128.update(b"abc");
  let expected128 = Kmac128::mac_array::<32>(b"api-consistency-key", b"ctx=v1", b"abc");
  let mut actual128 = [0u8; 32];
  kmac128.finalize_into(&mut actual128);
  assert_eq!(actual128, expected128);
  kmac128.reset();
  kmac128.update(b"abc");
  kmac128.finalize_into(&mut actual128);
  assert_eq!(actual128, expected128);
  assert!(kmac128.verify(&expected128).is_ok());

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
  let hkdf512 = HkdfSha512::new(b"salt", b"ikm");

  let okm256 = hkdf256.expand_array::<32>(b"info").unwrap();
  let okm384 = hkdf384.expand_array::<48>(b"info").unwrap();
  let okm512 = hkdf512.expand_array::<64>(b"info").unwrap();

  assert_eq!(
    okm256,
    HkdfSha256::derive_array::<32>(b"salt", b"ikm", b"info").unwrap()
  );
  assert_eq!(
    okm384,
    HkdfSha384::derive_array::<48>(b"salt", b"ikm", b"info").unwrap()
  );
  assert_eq!(
    okm512,
    HkdfSha512::derive_array::<64>(b"salt", b"ikm", b"info").unwrap()
  );
  assert_eq!(
    HkdfSha256::derive_array::<32>(b"salt", b"ikm", b"info").unwrap(),
    okm256
  );
  assert_eq!(
    HkdfSha384::derive_array::<48>(b"salt", b"ikm", b"info").unwrap(),
    okm384
  );
  assert_eq!(
    HkdfSha512::derive_array::<64>(b"salt", b"ikm", b"info").unwrap(),
    okm512
  );
  assert_eq!(hkdf256.prk().len(), 32);
  assert_eq!(hkdf384.prk().len(), 48);
  assert_eq!(hkdf512.prk().len(), 64);

  let mut oversized = vec![0u8; HkdfSha256::MAX_OUTPUT_SIZE + 1];
  assert_eq!(
    HkdfSha256::derive(b"salt", b"ikm", b"info", &mut oversized),
    Err(HkdfOutputLengthError::new())
  );
}

#[test]
#[cfg(feature = "poly1305")]
fn poly1305_consumes_one_time_key_and_verifies() {
  let key = Poly1305OneTimeKey::from_bytes([0x42; Poly1305OneTimeKey::LENGTH]);
  let tag = Poly1305::authenticate_once(key, b"api-consistency-poly1305");
  assert_eq!(tag.as_bytes().len(), Poly1305Tag::LENGTH);

  let key = Poly1305OneTimeKey::from_bytes([0x42; Poly1305OneTimeKey::LENGTH]);
  assert!(Poly1305::verify_once(key, b"api-consistency-poly1305", &tag).is_ok());
}

#[test]
#[cfg(feature = "ed25519")]
fn ed25519_types_follow_byte_roundtrip_and_verify_conventions() {
  let secret = Ed25519SecretKey::try_generate_with(|out| {
    out.fill(0x24);
    Ok::<(), ()>(())
  })
  .unwrap();
  let keypair = Ed25519Keypair::from_secret_key(secret.duplicate_secret());
  let public = Ed25519PublicKey::from_bytes(keypair.public_key().to_bytes());
  let signature = TrySigner::try_sign(&keypair, b"api-consistency-ed25519").unwrap();

  assert_eq!(*secret.expose_secret().as_bytes(), *secret.as_bytes());
  assert_eq!(secret.duplicate_secret().as_bytes(), secret.as_bytes());
  assert_eq!(public.to_bytes(), *public.as_bytes());
  assert_eq!(signature.to_bytes(), *signature.as_bytes());
  assert!(public.verify(b"api-consistency-ed25519", &signature).is_ok());
  assert!(Verifier::verify(&public, b"api-consistency-ed25519", &signature).is_ok());
}

#[test]
#[cfg(feature = "x25519")]
fn x25519_types_follow_byte_roundtrip_conventions() {
  let secret = X25519SecretKey::try_generate_with(|out| {
    out.fill(0x42);
    Ok::<(), ()>(())
  })
  .unwrap();
  let public = X25519PublicKey::from_bytes(secret.public_key().to_bytes());
  let shared = secret.diffie_hellman(&public).unwrap();

  assert_eq!(*secret.expose_secret().as_bytes(), *secret.as_bytes());
  assert_eq!(public.to_bytes(), *public.as_bytes());
  assert_eq!(*shared.expose_secret().as_bytes(), *shared.as_bytes());
}

#[test]
#[cfg(feature = "ecdsa-p256")]
fn ecdsa_p256_keygen_and_native_signature_traits_are_consistent() {
  let keypair = EcdsaP256Keypair::try_generate_with(|out| {
    out.fill(1);
    Ok::<(), ()>(())
  })
  .unwrap();
  let public = EcdsaP256PublicKey::from_sec1_bytes(&keypair.public_key().to_sec1_bytes()).unwrap();
  let signature = TrySigner::try_sign(&keypair, b"api-consistency-ecdsa-p256").unwrap();

  assert!(Verifier::verify(&public, b"api-consistency-ecdsa-p256", &signature).is_ok());

  let rejected = EcdsaP256SecretKey::try_generate_with(|out| {
    out.fill(0);
    Ok::<(), ()>(())
  })
  .unwrap_err();
  assert_eq!(rejected, EcdsaKeyGenerationError::InvalidSecretKey);

  let rng_error = EcdsaP256SecretKey::try_generate_with(|_| Err("rng")).unwrap_err();
  assert_eq!(rng_error, EcdsaKeyGenerationError::Random("rng"));
}

#[test]
#[cfg(feature = "ecdsa-p384")]
fn ecdsa_p384_keygen_and_native_signature_traits_are_consistent() {
  let keypair = EcdsaP384Keypair::try_generate_with(|out| {
    out.fill(1);
    Ok::<(), ()>(())
  })
  .unwrap();
  let public = EcdsaP384PublicKey::from_sec1_bytes(&keypair.public_key().to_sec1_bytes()).unwrap();
  let signature = TrySigner::try_sign(&keypair, b"api-consistency-ecdsa-p384").unwrap();

  assert!(Verifier::verify(&public, b"api-consistency-ecdsa-p384", &signature).is_ok());
}

#[test]
#[cfg(feature = "rsa")]
fn rsa_signature_verifier_requires_a_bound_profile() {
  let key = RsaPublicKey::from_spki_der(include_bytes!("../benches/rsa_fixtures/rsa3072_spki.der")).unwrap();
  let message = b"rscrypto RSA-PSS verification fixture";
  let pss_signature = include_bytes!("../benches/rsa_fixtures/rsa3072_pss_sha256.sig");
  let pss_verifier = key.verifier(RsaSignatureProfile::pss(RsaPssProfile::Sha256));

  assert!(pss_verifier.verify(message, pss_signature).is_ok());
  assert!(Verifier::verify(&pss_verifier, message, pss_signature.as_slice()).is_ok());

  let wrong_profile = key.verifier(RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256));
  assert!(wrong_profile.verify(message, pss_signature).is_err());
}

#[test]
#[cfg(all(feature = "rsa", feature = "getrandom"))]
fn rsa_signature_signer_shape_is_profile_bound() {
  fn assert_signer<T>()
  where
    T: TrySigner<Signature = Vec<u8>, Error = rscrypto::RsaPrivateOpError>
      + TrySignerInto<Error = rscrypto::RsaPrivateOpError>,
  {
  }

  assert_signer::<rscrypto::RsaSignatureSigner<'static>>();
}

#[test]
fn verification_error_follows_new_default_and_display_conventions() {
  assert_eq!(VerificationError::new(), VerificationError::default());
  assert_eq!(VerificationError::new().to_string(), "verification failed");
}

#[test]
fn kem_trait_uses_typed_key_ciphertext_and_secret_outputs() {
  assert_eq!(ToyKem::ENCAPSULATION_KEY_SIZE, 3);
  assert_eq!(ToyKem::DECAPSULATION_KEY_SIZE, 4);
  assert_eq!(ToyKem::CIPHERTEXT_SIZE, 5);
  assert_eq!(ToyKem::SHARED_SECRET_SIZE, 2);

  let (encapsulation_key, decapsulation_key) = ToyKem::generate_keypair(|out| {
    out.copy_from_slice(&[1, 2, 3, 4]);
    Ok(())
  })
  .unwrap();
  assert_eq!(encapsulation_key.as_ref(), &[1, 2, 3]);
  assert_eq!(decapsulation_key.as_ref(), &[1, 2, 3, 4]);

  let (ciphertext, encapsulated_secret) = ToyKem::encapsulate(&encapsulation_key, |out| {
    out.copy_from_slice(&[8, 9]);
    Ok(())
  })
  .unwrap();
  assert_eq!(ciphertext.as_ref(), &[1, 2, 3, 8, 9]);
  assert_eq!(encapsulated_secret.as_ref(), &[8, 9]);

  let decapsulated_secret = ToyKem::decapsulate(&decapsulation_key, &ciphertext).unwrap();
  assert_eq!(decapsulated_secret.as_ref(), &[4, 9]);
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
  assert_aead_api::<Aes128Gcm>(
    Aes128GcmKey::from_bytes([0u8; Aes128Gcm::KEY_SIZE]),
    Nonce96::from_bytes([1u8; Nonce96::LENGTH]),
  );
  assert_aead_api::<Aes256Gcm>(
    Aes256GcmKey::from_bytes([1u8; Aes256Gcm::KEY_SIZE]),
    Nonce96::from_bytes([2u8; Nonce96::LENGTH]),
  );
  assert_aead_api::<Aes128GcmSiv>(
    Aes128GcmSivKey::from_bytes([2u8; Aes128GcmSiv::KEY_SIZE]),
    Nonce96::from_bytes([3u8; Nonce96::LENGTH]),
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
#[cfg(all(feature = "aead", feature = "getrandom", feature = "alloc"))]
fn aead_random_to_vec_seals_and_opens() {
  let cipher = ChaCha20Poly1305::new(&ChaCha20Poly1305Key::from_bytes([0x33; ChaCha20Poly1305::KEY_SIZE]));
  let (nonce, sealed) = cipher.seal_random_to_vec(b"aad", b"plaintext").unwrap();
  let opened = cipher.decrypt_to_vec(&nonce, b"aad", &sealed).unwrap();

  assert_eq!(opened, b"plaintext");
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
