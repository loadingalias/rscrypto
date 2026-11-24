//! HPKE for Encrypted Client Hello (ECH). The `ech` feature provides the suite from RFC 9849 section 9.
//!
//! Pass the server's encoded ECH configuration list and your supported suites to
//! [`rustls::client::EchConfig::new`]. Set the resulting configuration on the client with `with_ech`.

use alloc::{alloc::Global, boxed::Box, vec::Vec};
use core::fmt;

use rscrypto::{
  HpkeError, HpkeReceiver, HpkeSender, HpkeSuite as _, HpkeX25519HkdfSha256Aes128Gcm as Suite, HpkeX25519PrivateKey,
  X25519PublicKey, X25519SecretKey,
};
use rustls::Error;
use rustls::crypto::SecureRandom;
use rustls::crypto::hpke::{
  EncapsulatedSecret, Hpke, HpkeOpener, HpkePrivateKey, HpkePublicKey, HpkeSealer, HpkeSuite, HpkeSymmetricCipherSuite,
};

/// DHKEM(X25519, HKDF-SHA256) with HKDF-SHA256 and AES-128-GCM, using operating-system randomness.
#[cfg(feature = "getrandom")]
pub static X25519_HKDF_SHA256_AES_128_GCM: &dyn Hpke = &X25519HkdfSha256Aes128Gcm::new(crate::random::SECURE_RANDOM);

/// The ECH suites available with operating-system randomness.
#[cfg(feature = "getrandom")]
pub static ALL_SUPPORTED_SUITES: &[&dyn Hpke] = &[X25519_HKDF_SHA256_AES_128_GCM];

/// The RFC 9849 X25519 suite with a caller-supplied entropy source.
pub struct X25519HkdfSha256Aes128Gcm {
  random: &'static dyn SecureRandom,
}

impl fmt::Debug for X25519HkdfSha256Aes128Gcm {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("X25519HkdfSha256Aes128Gcm")
  }
}

impl X25519HkdfSha256Aes128Gcm {
  /// Use `random` for all sender encapsulations and recipient key generation.
  /// A connection-scoped replay stream can supply sender entropy. Shared recipient keys need a separate source.
  #[must_use]
  pub const fn new(random: &'static dyn SecureRandom) -> Self {
    Self { random }
  }

  fn sender(&self, info: &[u8], key: &HpkePublicKey) -> Result<(EncapsulatedSecret, Sealer), Error> {
    let key = X25519PublicKey::from_bytes(key.0.as_slice().try_into().map_err(|_| Error::EncryptError)?);
    let (enc, sender) = HpkeSender::<Suite>::setup_in(
      &key,
      info,
      |out| self.random.fill(out).map_err(|_| HpkeError::RandomGenerationFailed),
      Global,
    )
    .map_err(|error| match error {
      HpkeError::RandomGenerationFailed => Error::FailedToGetRandomBytes,
      _ => Error::EncryptError,
    })?;
    Ok((EncapsulatedSecret(enc.as_bytes().to_vec()), Sealer(sender)))
  }
}

impl Hpke for X25519HkdfSha256Aes128Gcm {
  fn suite(&self) -> HpkeSuite {
    HpkeSuite {
      kem: Suite::KEM_ID.into(),
      sym: HpkeSymmetricCipherSuite {
        kdf_id: Suite::KDF_ID.into(),
        aead_id: Suite::AEAD_ID.into(),
      },
    }
  }

  fn seal(
    &self,
    info: &[u8],
    aad: &[u8],
    plaintext: &[u8],
    key: &HpkePublicKey,
  ) -> Result<(EncapsulatedSecret, Vec<u8>), Error> {
    let (enc, mut sender) = self.sender(info, key)?;
    Ok((enc, sender.seal(aad, plaintext)?))
  }

  fn setup_sealer(&self, info: &[u8], key: &HpkePublicKey) -> Result<(EncapsulatedSecret, Box<dyn HpkeSealer>), Error> {
    let (enc, sender) = self.sender(info, key)?;
    Ok((enc, Box::new(sender)))
  }

  fn open(
    &self,
    enc: &EncapsulatedSecret,
    info: &[u8],
    aad: &[u8],
    ciphertext: &[u8],
    key: &HpkePrivateKey,
  ) -> Result<Vec<u8>, Error> {
    Opener::new(enc, info, key)?.open(aad, ciphertext)
  }

  fn setup_opener(
    &self,
    enc: &EncapsulatedSecret,
    info: &[u8],
    key: &HpkePrivateKey,
  ) -> Result<Box<dyn HpkeOpener>, Error> {
    Ok(Box::new(Opener::new(enc, info, key)?))
  }

  fn generate_key_pair(&self) -> Result<(HpkePublicKey, HpkePrivateKey), Error> {
    let key = X25519SecretKey::try_generate_with(|out| crate::owner_fill(self.random, out))?;
    Ok((
      HpkePublicKey(key.public_key().as_bytes().to_vec()),
      key.as_bytes().to_vec().into(),
    ))
  }
}

// The core schedules the keys in their final allocation. Boxing either rustls adapter moves only a pointer.
#[derive(Debug)]
struct Sealer(Box<HpkeSender<Suite>>);

impl HpkeSealer for Sealer {
  fn seal(&mut self, aad: &[u8], plaintext: &[u8]) -> Result<Vec<u8>, Error> {
    self.0.seal_to_vec(aad, plaintext).map_err(|_| Error::EncryptError)
  }
}

#[derive(Debug)]
struct Opener(Box<HpkeReceiver<Suite>>);

impl Opener {
  fn new(enc: &EncapsulatedSecret, info: &[u8], key: &HpkePrivateKey) -> Result<Self, Error> {
    let enc = X25519PublicKey::from_bytes(enc.0.as_slice().try_into().map_err(|_| Error::DecryptError)?);
    let key = X25519SecretKey::from_bytes(key.secret_bytes().try_into().map_err(|_| Error::DecryptError)?);
    HpkeReceiver::<Suite>::setup_in(&enc, &HpkeX25519PrivateKey::new(key), info, Global)
      .map(Self)
      .map_err(|_| Error::DecryptError)
  }
}

impl HpkeOpener for Opener {
  fn open(&mut self, aad: &[u8], ciphertext: &[u8]) -> Result<Vec<u8>, Error> {
    self.0.open_to_vec(aad, ciphertext).map_err(|_| Error::DecryptError)
  }
}
