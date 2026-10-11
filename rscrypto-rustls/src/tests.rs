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
  feature = "getrandom",
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
  feature = "getrandom",
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
  feature = "getrandom",
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

#[cfg(all(feature = "ml-dsa", feature = "getrandom"))]
mod mldsa;
