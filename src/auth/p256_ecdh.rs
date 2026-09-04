//! Ephemeral P-256 Diffie-Hellman key agreement (NIST SP 800-56A).
//!
//! Peer keys use canonical uncompressed SEC1 encoding. Parsing validates the
//! complete public point before any private scalar arithmetic is reachable.
//! ECDH does not authenticate either party. Callers must authenticate the
//! exchanged public keys or a transcript that binds them, then feed the raw
//! shared x-coordinate into a protocol-specific KDF.

use core::{
  fmt,
  hash::{Hash, Hasher},
};

use super::p256_portable::{self, PublicPoint};
use crate::{SecretBytes, traits::ct};

const FIELD_BYTES: usize = 32;
const SEC1_BYTES: usize = 65;
const SCALAR_SAMPLING_ATTEMPTS: usize = 32;

#[cfg(all(
  not(feature = "portable-only"),
  not(miri),
  any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  )
))]
struct ZeroizingNativeWords<const N: usize>([u64; N]);

#[cfg(all(
  not(feature = "portable-only"),
  not(miri),
  any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  )
))]
impl<const N: usize> Drop for ZeroizingNativeWords<N> {
  fn drop(&mut self) {
    ct::zeroize_words(&mut self.0);
  }
}

#[cfg(all(
  not(feature = "portable-only"),
  not(miri),
  any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  )
))]
fn native_words<const N: usize>(bytes: &[u8]) -> [u64; N] {
  debug_assert_eq!(bytes.len(), N.strict_mul(8));
  let mut words = [0u64; N];
  for (word, chunk) in words.iter_mut().zip(bytes.rchunks_exact(8)) {
    let mut limb = [0u8; 8];
    limb.copy_from_slice(chunk);
    *word = u64::from_be_bytes(limb);
  }
  words
}

/// Error returned by bounded ephemeral P-256 key generation.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum P256KeyGenerationError<E> {
  /// The caller-provided entropy source failed.
  Random(E),
  /// No valid scalar was sampled within the fixed retry budget.
  ScalarSamplingExhausted,
}

impl<E> fmt::Debug for P256KeyGenerationError<E> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Random(_) => f.write_str("Random(..)"),
      Self::ScalarSamplingExhausted => f.write_str("ScalarSamplingExhausted"),
    }
  }
}

impl<E> fmt::Display for P256KeyGenerationError<E> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Random(_) => f.write_str("P-256 key-generation random source failed"),
      Self::ScalarSamplingExhausted => f.write_str("P-256 key-generation scalar rejection limit reached"),
    }
  }
}

impl<E> core::error::Error for P256KeyGenerationError<E> where E: core::error::Error + 'static {}

/// Error returned when a P-256 public key is not canonical uncompressed SEC1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct P256PublicKeyError;

impl fmt::Display for P256PublicKeyError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("invalid canonical uncompressed P-256 public key")
  }
}

impl core::error::Error for P256PublicKeyError {}

/// One-use P-256 secret scalar for ephemeral key agreement.
pub struct P256EphemeralSecret([u8; FIELD_BYTES]);

impl P256EphemeralSecret {
  /// Secret scalar length in bytes.
  pub const LENGTH: usize = FIELD_BYTES;

  /// Fill and rejection-sample an ephemeral scalar with a bounded retry count.
  ///
  /// The callback is invoked at most 32 times. It must overwrite the complete
  /// zero-initialized buffer or return an error. Entropy acquisition and
  /// candidate rejection are outside the constant-time public-derivation and
  /// agreement claims. Partially filled bytes and rejected candidates are
  /// cleared before the callback can run again or the method returns.
  pub fn try_generate_with<E>(
    mut fill: impl FnMut(&mut [u8; Self::LENGTH]) -> Result<(), E>,
  ) -> Result<Self, P256KeyGenerationError<E>> {
    let mut candidate = Self([0u8; FIELD_BYTES]);
    for _ in 0..SCALAR_SAMPLING_ATTEMPTS {
      fill(&mut candidate.0).map_err(P256KeyGenerationError::Random)?;
      if p256_portable::scalar_is_canonical_nonzero(&candidate.0) {
        return Ok(candidate);
      }
      ct::zeroize(&mut candidate.0);
    }
    Err(P256KeyGenerationError::ScalarSamplingExhausted)
  }

  /// Generate an ephemeral scalar from the platform entropy source.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  pub fn try_generate() -> Result<Self, P256KeyGenerationError<getrandom::Error>> {
    Self::try_generate_with(|candidate| getrandom::fill(candidate))
  }

  /// Derive the matching canonical uncompressed public key.
  #[must_use]
  pub fn public_key(&self) -> P256PublicKey {
    #[cfg(all(
      not(feature = "portable-only"),
      not(miri),
      any(
        all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
        all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
      )
    ))]
    {
      let scalar = ZeroizingNativeWords(native_words(&self.0));
      let words = super::p256_core::scalar_mul_generator_words(&scalar.0);
      P256PublicKey::from_point(PublicPoint::from_affine_words(words))
    }

    #[cfg(not(all(
      not(feature = "portable-only"),
      not(miri),
      any(
        all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
        all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
      )
    )))]
    {
      P256PublicKey::from_point(p256_portable::public_key_from_scalar(&self.0))
    }
  }

  /// Consume this ephemeral scalar and derive the peer agreement value.
  ///
  /// The result is the fixed-width big-endian x-coordinate specified by the
  /// ECC CDH primitive. `public` has already passed complete SEC1 and curve
  /// validation. Agreement does not authenticate `public`; the caller must
  /// authenticate the peer key or a transcript that binds it and derive
  /// application keys with a protocol-specific KDF.
  #[must_use]
  pub fn diffie_hellman(self, public: &P256PublicKey) -> P256SharedSecret {
    #[cfg(all(
      not(feature = "portable-only"),
      not(miri),
      any(
        all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
        all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
      )
    ))]
    {
      let scalar = ZeroizingNativeWords(native_words(&self.0));
      let point = public.point.to_affine_words();
      let output = ZeroizingNativeWords(super::p256_core::scalar_mul_words(&scalar.0, &point));
      let mut shared = [0u8; FIELD_BYTES];
      for (chunk, word) in shared.rchunks_exact_mut(8).zip(output.0[..4].iter().copied()) {
        chunk.copy_from_slice(&word.to_be_bytes());
      }
      P256SharedSecret(shared)
    }

    #[cfg(not(all(
      not(feature = "portable-only"),
      not(miri),
      any(
        all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
        all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
      )
    )))]
    {
      P256SharedSecret(p256_portable::agree(&self.0, public.point))
    }
  }
}

impl fmt::Debug for P256EphemeralSecret {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("P256EphemeralSecret(****)")
  }
}

impl Drop for P256EphemeralSecret {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

/// Validated canonical uncompressed SEC1 P-256 public key.
#[derive(Clone)]
pub struct P256PublicKey {
  bytes: [u8; SEC1_BYTES],
  point: PublicPoint,
}

impl P256PublicKey {
  /// Canonical uncompressed SEC1 length in bytes.
  pub const SEC1_LENGTH: usize = SEC1_BYTES;

  /// Parse and validate a canonical uncompressed SEC1 P-256 public key.
  ///
  /// Compressed keys, infinity, non-canonical coordinates, and off-curve
  /// points are rejected before private scalar arithmetic can begin.
  pub fn from_sec1_bytes(bytes: &[u8]) -> Result<Self, P256PublicKeyError> {
    let point = super::p256_core::public_point_from_sec1(bytes).ok_or(P256PublicKeyError)?;
    let mut canonical = [0u8; SEC1_BYTES];
    canonical.copy_from_slice(bytes);
    Ok(Self {
      bytes: canonical,
      point,
    })
  }

  /// Return canonical uncompressed SEC1 bytes.
  #[must_use]
  pub const fn to_sec1_bytes(&self) -> [u8; Self::SEC1_LENGTH] {
    self.bytes
  }

  /// Borrow canonical uncompressed SEC1 bytes.
  #[must_use]
  pub const fn as_sec1_bytes(&self) -> &[u8; Self::SEC1_LENGTH] {
    &self.bytes
  }

  fn from_point(point: PublicPoint) -> Self {
    Self {
      bytes: point.to_sec1_bytes(),
      point,
    }
  }
}

impl PartialEq for P256PublicKey {
  fn eq(&self, other: &Self) -> bool {
    self.bytes == other.bytes
  }
}

impl Eq for P256PublicKey {}

impl Hash for P256PublicKey {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.bytes.hash(state);
  }
}

impl AsRef<[u8]> for P256PublicKey {
  fn as_ref(&self) -> &[u8] {
    &self.bytes
  }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl serde::Serialize for P256PublicKey {
  fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_bytes(&self.bytes)
  }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<'de> serde::Deserialize<'de> for P256PublicKey {
  fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    struct PublicKeyVisitor;

    impl<'de> serde::de::Visitor<'de> for PublicKeyVisitor {
      type Value = P256PublicKey;

      fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("65 canonical uncompressed SEC1 P-256 public-key bytes")
      }

      fn visit_bytes<E: serde::de::Error>(self, bytes: &[u8]) -> Result<Self::Value, E> {
        P256PublicKey::from_sec1_bytes(bytes).map_err(|_| E::custom("invalid canonical uncompressed P-256 public key"))
      }

      fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
        let mut bytes = [0u8; P256PublicKey::SEC1_LENGTH];
        for (index, byte) in bytes.iter_mut().enumerate() {
          *byte = sequence
            .next_element()?
            .ok_or_else(|| serde::de::Error::invalid_length(index, &self))?;
        }
        if sequence.next_element::<serde::de::IgnoredAny>()?.is_some() {
          return Err(serde::de::Error::invalid_length(P256PublicKey::SEC1_LENGTH + 1, &self));
        }
        P256PublicKey::from_sec1_bytes(&bytes)
          .map_err(|_| serde::de::Error::custom("invalid canonical uncompressed P-256 public key"))
      }
    }

    deserializer.deserialize_bytes(PublicKeyVisitor)
  }
}

impl fmt::Debug for P256PublicKey {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "P256PublicKey(")?;
    crate::hex::fmt_hex_lower(&self.bytes, f)?;
    write!(f, ")")
  }
}

/// Fixed-width P-256 ECC CDH shared secret.
pub struct P256SharedSecret([u8; FIELD_BYTES]);

impl P256SharedSecret {
  /// Shared-secret length in bytes.
  pub const LENGTH: usize = FIELD_BYTES;

  /// Compare two shared secrets without exposing a branchable boolean.
  pub fn ct_eq(&self, other: &Self) -> ct::CtDecision {
    ct::fixed_eq(&self.0, &other.0)
  }

  /// Borrow the fixed-width shared-secret bytes.
  #[must_use]
  pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
    &self.0
  }

  /// Explicitly copy the shared secret into a zeroizing owner.
  #[must_use]
  pub fn expose_secret(&self) -> SecretBytes<{ Self::LENGTH }> {
    SecretBytes::new(self.0)
  }
}

impl fmt::Debug for P256SharedSecret {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("P256SharedSecret(****)")
  }
}

impl Drop for P256SharedSecret {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

/// Return the production P-256 secret-window selection as Montgomery limbs.
#[cfg(all(
  feature = "diag",
  any(
    feature = "portable-only",
    miri,
    not(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    ))
  )
))]
#[doc(hidden)]
pub fn diag_p256_ecdh_select_window_limb_digest(digit: u8) -> [u64; 8] {
  p256_portable::diag_select_window_limb_digest(digit)
}

/// Exercise P-256 ECDH candidate cleanup on success and partial-fill failure.
#[cfg(feature = "diag")]
#[doc(hidden)]
#[unsafe(no_mangle)]
#[inline(never)]
pub(crate) fn diag_zeroize_p256_ecdh_generation(value: u8, fail: bool) -> u8 {
  let result = P256EphemeralSecret::try_generate_with(|candidate| {
    candidate[..16].fill(core::hint::black_box(value));
    if core::hint::black_box(fail) {
      return Err(());
    }
    candidate[16..].fill(core::hint::black_box(value));
    Ok(())
  });
  result.map_or(0, |secret| core::hint::black_box(secret.0[0]))
}

/// Exercise P-256 ECDH scalar, projective-state, and shared-secret cleanup.
#[cfg(feature = "diag")]
#[doc(hidden)]
#[unsafe(no_mangle)]
#[inline(never)]
pub(crate) fn diag_zeroize_p256_ecdh_agreement(secret: [u8; 32]) -> u8 {
  let Ok(secret) = P256EphemeralSecret::try_generate_with(|candidate| {
    candidate.copy_from_slice(core::hint::black_box(&secret));
    Ok::<(), core::convert::Infallible>(())
  }) else {
    return 0;
  };
  let Ok(peer) = P256EphemeralSecret::try_generate_with(|candidate| {
    candidate.fill(0x24);
    Ok::<(), core::convert::Infallible>(())
  }) else {
    return 0;
  };
  let shared = secret.diffie_hellman(&peer.public_key());
  core::hint::black_box(shared.as_bytes()[0])
}

#[cfg(test)]
mod tests {
  use super::{P256EphemeralSecret, P256KeyGenerationError, P256PublicKey};

  const GENERATOR_X_BYTES: [u8; 32] = [
    0x6b, 0x17, 0xd1, 0xf2, 0xe1, 0x2c, 0x42, 0x47, 0xf8, 0xbc, 0xe6, 0xe5, 0x63, 0xa4, 0x40, 0xf2, 0x77, 0x03, 0x7d,
    0x81, 0x2d, 0xeb, 0x33, 0xa0, 0xf4, 0xa1, 0x39, 0x45, 0xd8, 0x98, 0xc2, 0x96,
  ];
  const GENERATOR_Y_BYTES: [u8; 32] = [
    0x4f, 0xe3, 0x42, 0xe2, 0xfe, 0x1a, 0x7f, 0x9b, 0x8e, 0xe7, 0xeb, 0x4a, 0x7c, 0x0f, 0x9e, 0x16, 0x2b, 0xce, 0x33,
    0x57, 0x6b, 0x31, 0x5e, 0xce, 0xcb, 0xb6, 0x40, 0x68, 0x37, 0xbf, 0x51, 0xf5,
  ];

  fn generated(bytes: [u8; 32]) -> P256EphemeralSecret {
    P256EphemeralSecret::try_generate_with(|out| {
      out.copy_from_slice(&bytes);
      Ok::<(), core::convert::Infallible>(())
    })
    .expect("valid P-256 scalar must generate")
  }

  #[test]
  fn scalar_one_derives_the_p256_generator() {
    let mut scalar = [0u8; 32];
    scalar[31] = 1;
    let public = generated(scalar).public_key();
    assert_eq!(public.as_sec1_bytes()[0], 0x04);
    assert_eq!(&public.as_sec1_bytes()[1..33], &GENERATOR_X_BYTES);
    assert_eq!(&public.as_sec1_bytes()[33..], &GENERATOR_Y_BYTES);
  }

  #[test]
  fn agreement_is_symmetric_and_fixed_width() {
    let mut alice_bytes = [0u8; 32];
    alice_bytes[31] = 7;
    let mut bob_bytes = [0u8; 32];
    bob_bytes[31] = 9;
    let alice = generated(alice_bytes);
    let bob = generated(bob_bytes);
    let alice_public = alice.public_key();
    let bob_public = bob.public_key();
    let alice_shared = alice.diffie_hellman(&bob_public);
    let bob_shared = bob.diffie_hellman(&alice_public);
    assert!(alice_shared.ct_eq(&bob_shared).declassify());
    assert_eq!(alice_shared.as_bytes().len(), 32);
  }

  #[test]
  fn parsing_rejects_every_public_shape_class() {
    assert_eq!(P256PublicKey::from_sec1_bytes(&[]), Err(super::P256PublicKeyError));
    assert_eq!(
      P256PublicKey::from_sec1_bytes(&[0x04; 64]),
      Err(super::P256PublicKeyError)
    );
    assert_eq!(
      P256PublicKey::from_sec1_bytes(&[0x04; 66]),
      Err(super::P256PublicKeyError)
    );
    let mut compressed = [0u8; 33];
    compressed[0] = 0x02;
    assert_eq!(
      P256PublicKey::from_sec1_bytes(&compressed),
      Err(super::P256PublicKeyError)
    );
    assert_eq!(
      P256PublicKey::from_sec1_bytes(&[0u8; 65]),
      Err(super::P256PublicKeyError)
    );
    assert_eq!(
      P256PublicKey::from_sec1_bytes(&[0x04; 65]),
      Err(super::P256PublicKeyError)
    );
  }

  #[test]
  fn generation_propagates_partial_fill_failure_and_bounds_rejection() {
    let expected = 17u8;
    let error = P256EphemeralSecret::try_generate_with(|out| {
      out[..4].fill(0xaa);
      Err(expected)
    })
    .expect_err("failing entropy callback must be returned");
    assert_eq!(error, P256KeyGenerationError::Random(expected));

    let mut calls = 0usize;
    let error = P256EphemeralSecret::try_generate_with(|out| {
      calls = calls.strict_add(1);
      out.fill(0);
      Ok::<(), core::convert::Infallible>(())
    })
    .expect_err("zero is never a valid P-256 scalar");
    assert_eq!(error, P256KeyGenerationError::ScalarSamplingExhausted);
    assert_eq!(calls, 32);

    let mut calls = 0usize;
    let secret = P256EphemeralSecret::try_generate_with(|out| {
      calls = calls.strict_add(1);
      if calls == 1 {
        out.fill(0xff);
      } else {
        assert_eq!(*out, [0u8; 32], "each rejection-sampling attempt starts cleared");
        out[31] = 1;
      }
      Ok::<(), core::convert::Infallible>(())
    })
    .expect("second candidate is scalar one");
    assert_eq!(calls, 2);
    assert_eq!(secret.public_key().as_sec1_bytes()[1..33], GENERATOR_X_BYTES);
  }

  #[cfg(miri)]
  #[test]
  fn miri_uses_portable_p256_ecdh_path() {
    let alice = generated([0x42; 32]);
    let peer = generated([0x24; 32]).public_key();
    let shared = alice.diffie_hellman(&peer);
    assert_eq!(shared.as_bytes().len(), P256EphemeralSecret::LENGTH);
  }
}
