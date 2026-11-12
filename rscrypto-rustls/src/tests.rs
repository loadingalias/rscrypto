use rustls::crypto::{GetRandomFailed, SecureRandom};

/// Caller entropy for tests: an independent source behind this crate's
/// caller-entropy constructors, so the suite runs without `getrandom`.
#[derive(Debug)]
pub(crate) struct Random;

impl SecureRandom for Random {
  fn fill(&self, output: &mut [u8]) -> Result<(), GetRandomFailed> {
    rustls_ring::DEFAULT_TLS13_PROVIDER.secure_random.fill(output)
  }
}

pub(crate) static RANDOM: Random = Random;

/// Fills every request with one byte, so an independent oracle can rebuild the drawn keys.
#[cfg(all(feature = "aes-gcm", feature = "hkdf"))]
#[derive(Debug)]
pub(crate) struct Fixed(pub(crate) u8);

#[cfg(all(feature = "aes-gcm", feature = "hkdf"))]
impl SecureRandom for Fixed {
  fn fill(&self, output: &mut [u8]) -> Result<(), GetRandomFailed> {
    output.fill(self.0);
    Ok(())
  }
}

/// The key provider under test: [`crate::sign::KEY_PROVIDER`] with `getrandom`,
/// otherwise the same loader over [`RANDOM`].
#[cfg(any(
  feature = "ml-dsa",
  all(
    feature = "hkdf",
    any(feature = "aes-gcm", feature = "chacha20poly1305"),
    any(feature = "x25519", feature = "p256-ecdh", feature = "p384-ecdh"),
    any(feature = "ecdsa-p256", feature = "ecdsa-p384", feature = "ed25519", feature = "rsa")
  )
))]
pub(crate) fn key_provider() -> &'static dyn rustls::crypto::KeyProvider {
  #[cfg(feature = "getrandom")]
  {
    crate::sign::KEY_PROVIDER
  }
  #[cfg(not(feature = "getrandom"))]
  {
    static KEYS: crate::sign::Provider = crate::sign::Provider::new(&RANDOM);
    &KEYS
  }
}

#[cfg(all(feature = "hkdf", feature = "aes-gcm", feature = "chacha20poly1305"))]
mod symmetric;
#[cfg(all(feature = "hkdf", feature = "aes-gcm"))]
mod vectors;
#[cfg(all(feature = "hkdf", feature = "aes-gcm"))]
pub(crate) use vectors::{fields, hex};

#[cfg(all(
  feature = "x25519",
  feature = "p256-ecdh",
  feature = "p384-ecdh",
  feature = "ml-kem",
  feature = "ecdsa-p256",
  feature = "ecdsa-p384",
  feature = "ed25519",
  feature = "rsa",
  feature = "hkdf",
  any(feature = "aes-gcm", feature = "chacha20poly1305")
))]
mod authentication;

#[cfg(all(
  feature = "hkdf",
  any(feature = "aes-gcm", feature = "chacha20poly1305"),
  any(feature = "x25519", feature = "p256-ecdh", feature = "p384-ecdh"),
  any(
    feature = "ecdsa-p256",
    feature = "ecdsa-p384",
    feature = "ed25519",
    feature = "rsa",
    feature = "ml-dsa"
  )
))]
pub(crate) mod tls;

#[cfg(all(
  feature = "hkdf",
  any(feature = "aes-gcm", feature = "chacha20poly1305"),
  any(feature = "x25519", feature = "p256-ecdh", feature = "p384-ecdh"),
  any(feature = "ecdsa-p256", feature = "ecdsa-p384", feature = "ed25519", feature = "rsa")
))]
mod client;

#[cfg(all(
  feature = "chacha20-rng",
  feature = "hkdf",
  feature = "aes-gcm",
  feature = "x25519",
  feature = "ed25519"
))]
mod replay;

#[cfg(all(
  feature = "hkdf",
  any(feature = "aes-gcm", feature = "chacha20poly1305"),
  any(feature = "x25519", feature = "p256-ecdh", feature = "p384-ecdh"),
  any(
    feature = "ecdsa-p256",
    feature = "ecdsa-p384",
    feature = "ed25519",
    feature = "rsa",
    feature = "ml-dsa"
  )
))]
mod profiles;

#[cfg(feature = "ml-dsa")]
mod mldsa;

#[cfg(all(
  feature = "mldsa65-ed25519",
  feature = "mldsa65-ecdsa-p256",
  feature = "ml-dsa",
  feature = "ed25519",
  feature = "ecdsa-p256",
  feature = "hkdf",
  feature = "aes-gcm",
  feature = "x25519",
  feature = "p256-ecdh"
))]
mod composite;

#[cfg(all(
  feature = "hkdf",
  feature = "aes-gcm",
  feature = "chacha20poly1305",
  feature = "x25519",
  feature = "p256-ecdh",
  feature = "p384-ecdh",
  feature = "ecdsa-p256",
  feature = "ecdsa-p384",
  feature = "ed25519",
  feature = "rsa"
))]
mod tls12;

#[cfg(all(
  feature = "hkdf",
  any(feature = "aes-gcm", feature = "chacha20poly1305"),
  any(feature = "x25519", feature = "p256-ecdh", feature = "p384-ecdh"),
  any(
    feature = "ecdsa-p256",
    feature = "ecdsa-p384",
    feature = "ed25519",
    feature = "rsa",
    feature = "ml-dsa"
  )
))]
mod anchors;
