//! Key-encapsulation mechanism traits.
//!
//! KEMs establish a shared secret over a public channel through three
//! operations: key generation, encapsulation, and decapsulation. The trait
//! keeps those operations static and typed so implementations can remain
//! `no_std`, allocation-free, and dispatch-free.

use core::fmt::Debug;

/// Key-encapsulation mechanism with fixed-size typed inputs and outputs.
///
/// Probabilistic operations receive randomness from a caller-supplied fill
/// closure. This avoids baking an operating-system RNG dependency into the
/// trait while still giving concrete implementations one explicit place to
/// request fresh random bytes.
pub trait Kem {
  /// Encapsulation key size in bytes.
  const ENCAPSULATION_KEY_SIZE: usize;

  /// Decapsulation key size in bytes.
  const DECAPSULATION_KEY_SIZE: usize;

  /// Ciphertext size in bytes.
  const CIPHERTEXT_SIZE: usize;

  /// Shared-secret size in bytes.
  const SHARED_SECRET_SIZE: usize;

  /// Public key used by the encapsulating party.
  type EncapsulationKey: Clone + Eq + Debug + AsRef<[u8]>;

  /// Private key used by the decapsulating party.
  type DecapsulationKey: Eq + Debug + AsRef<[u8]>;

  /// Public ciphertext sent by the encapsulating party.
  type Ciphertext: Clone + Eq + Debug + AsRef<[u8]>;

  /// Secret established by encapsulation or decapsulation.
  type SharedSecret: Eq + Debug + AsRef<[u8]>;

  /// Error returned by key generation.
  type KeyGenerationError;

  /// Error returned by encapsulation.
  type EncapsulationError;

  /// Error returned by decapsulation.
  type DecapsulationError;

  /// Generate an encapsulation/decapsulation key pair.
  ///
  /// Implementations decide how many bytes are requested from `fill_random`.
  /// The closure must provide fresh cryptographic randomness and return an
  /// error if it cannot completely fill the requested buffer.
  fn generate_keypair(
    fill_random: impl FnMut(&mut [u8]) -> Result<(), Self::KeyGenerationError>,
  ) -> Result<(Self::EncapsulationKey, Self::DecapsulationKey), Self::KeyGenerationError>;

  /// Encapsulate to `encapsulation_key`.
  ///
  /// Implementations decide how many bytes are requested from `fill_random`.
  /// The returned ciphertext is public; the returned shared secret is secret
  /// key material and must be handled accordingly.
  #[must_use = "KEM encapsulation output must be used; dropping it loses the established shared secret"]
  fn encapsulate(
    encapsulation_key: &Self::EncapsulationKey,
    fill_random: impl FnMut(&mut [u8]) -> Result<(), Self::EncapsulationError>,
  ) -> Result<(Self::Ciphertext, Self::SharedSecret), Self::EncapsulationError>;

  /// Decapsulate `ciphertext` with `decapsulation_key`.
  ///
  /// Concrete KEMs must preserve their scheme's failure behavior. For ML-KEM,
  /// malformed input handling and implicit rejection must not reveal which
  /// component failed.
  #[must_use = "KEM decapsulation output must be checked and used; dropping the Result can ignore input failure"]
  fn decapsulate(
    decapsulation_key: &Self::DecapsulationKey,
    ciphertext: &Self::Ciphertext,
  ) -> Result<Self::SharedSecret, Self::DecapsulationError>;
}
