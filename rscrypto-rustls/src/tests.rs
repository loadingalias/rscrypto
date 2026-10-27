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
#[cfg(any(
  feature = "quinn",
  all(feature = "hkdf", feature = "aes-gcm", feature = "chacha20poly1305")
))]
mod vectors;
#[cfg(any(
  feature = "quinn",
  all(feature = "hkdf", feature = "aes-gcm", feature = "chacha20poly1305")
))]
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
