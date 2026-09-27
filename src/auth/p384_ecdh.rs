//! Ephemeral P-384 Diffie-Hellman key agreement (NIST SP 800-56A).
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

use super::p384_portable::{self, FIELD_BYTES, PublicPoint, SEC1_BYTES};
use crate::{SecretBytes, traits::ct};

const SCALAR_SAMPLING_ATTEMPTS: usize = 32;

/// Error returned by bounded ephemeral P-384 key generation.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum P384KeyGenerationError<E> {
  /// The caller-provided entropy source failed.
  Random(E),
  /// No valid scalar was sampled within the fixed retry budget.
  ScalarSamplingExhausted,
}

impl<E> fmt::Debug for P384KeyGenerationError<E> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Random(_) => f.write_str("Random(..)"),
      Self::ScalarSamplingExhausted => f.write_str("ScalarSamplingExhausted"),
    }
  }
}

impl<E> fmt::Display for P384KeyGenerationError<E> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Random(_) => f.write_str("P-384 key-generation random source failed"),
      Self::ScalarSamplingExhausted => f.write_str("P-384 key-generation scalar rejection limit reached"),
    }
  }
}

impl<E> core::error::Error for P384KeyGenerationError<E> where E: core::error::Error + 'static {}

/// Error returned when a P-384 public key is not canonical uncompressed SEC1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct P384PublicKeyError;

impl fmt::Display for P384PublicKeyError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("invalid canonical uncompressed P-384 public key")
  }
}

impl core::error::Error for P384PublicKeyError {}

/// One-use P-384 secret scalar for ephemeral key agreement.
pub struct P384EphemeralSecret([u8; FIELD_BYTES]);

impl P384EphemeralSecret {
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
  ) -> Result<Self, P384KeyGenerationError<E>> {
    let mut candidate = Self([0u8; FIELD_BYTES]);
    for _ in 0..SCALAR_SAMPLING_ATTEMPTS {
      fill(&mut candidate.0).map_err(P384KeyGenerationError::Random)?;
      if p384_portable::scalar_is_canonical_nonzero(&candidate.0) {
        return Ok(candidate);
      }
      ct::zeroize(&mut candidate.0);
    }
    Err(P384KeyGenerationError::ScalarSamplingExhausted)
  }

  /// Generate an ephemeral scalar from the platform entropy source.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  pub fn try_generate() -> Result<Self, P384KeyGenerationError<getrandom::Error>> {
    Self::try_generate_with(|candidate| getrandom::fill(candidate))
  }

  /// Derive the matching canonical uncompressed public key.
  #[must_use]
  pub fn public_key(&self) -> P384PublicKey {
    P384PublicKey::from_point(p384_portable::public_key_from_scalar(&self.0))
  }

  /// Consume this ephemeral scalar and derive the peer agreement value.
  ///
  /// The result is the fixed-width big-endian x-coordinate specified by the
  /// ECC CDH primitive. `public` has already passed complete SEC1 and curve
  /// validation. Agreement does not authenticate `public`; the caller must
  /// authenticate the peer key or a transcript that binds it and derive
  /// application keys with a protocol-specific KDF.
  #[must_use]
  pub fn diffie_hellman(self, public: &P384PublicKey) -> P384SharedSecret {
    let mut shared = P384SharedSecret([0u8; FIELD_BYTES]);
    p384_portable::agree(&self.0, public.point, &mut shared.0);
    shared
  }
}

impl fmt::Debug for P384EphemeralSecret {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("P384EphemeralSecret(****)")
  }
}

impl Drop for P384EphemeralSecret {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

/// Validated canonical uncompressed SEC1 P-384 public key.
#[derive(Clone)]
pub struct P384PublicKey {
  bytes: [u8; SEC1_BYTES],
  point: PublicPoint,
}

impl P384PublicKey {
  /// Canonical uncompressed SEC1 length in bytes.
  pub const SEC1_LENGTH: usize = SEC1_BYTES;

  /// Parse and validate a canonical uncompressed SEC1 P-384 public key.
  ///
  /// Compressed keys, infinity, non-canonical coordinates, and off-curve
  /// points are rejected before private scalar arithmetic can begin.
  pub fn from_sec1_bytes(bytes: &[u8]) -> Result<Self, P384PublicKeyError> {
    let point = PublicPoint::from_sec1_bytes(bytes).ok_or(P384PublicKeyError)?;
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

impl PartialEq for P384PublicKey {
  fn eq(&self, other: &Self) -> bool {
    self.bytes == other.bytes
  }
}

impl Eq for P384PublicKey {}

impl Hash for P384PublicKey {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.bytes.hash(state);
  }
}

impl AsRef<[u8]> for P384PublicKey {
  fn as_ref(&self) -> &[u8] {
    &self.bytes
  }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl serde::Serialize for P384PublicKey {
  fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_bytes(&self.bytes)
  }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<'de> serde::Deserialize<'de> for P384PublicKey {
  fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    struct PublicKeyVisitor;

    impl<'de> serde::de::Visitor<'de> for PublicKeyVisitor {
      type Value = P384PublicKey;

      fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("97 canonical uncompressed SEC1 P-384 public-key bytes")
      }

      fn visit_bytes<E: serde::de::Error>(self, bytes: &[u8]) -> Result<Self::Value, E> {
        P384PublicKey::from_sec1_bytes(bytes).map_err(|_| E::custom("invalid canonical uncompressed P-384 public key"))
      }

      fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
        let mut bytes = [0u8; P384PublicKey::SEC1_LENGTH];
        for (index, byte) in bytes.iter_mut().enumerate() {
          *byte = sequence
            .next_element()?
            .ok_or_else(|| serde::de::Error::invalid_length(index, &self))?;
        }
        if sequence.next_element::<serde::de::IgnoredAny>()?.is_some() {
          return Err(serde::de::Error::invalid_length(P384PublicKey::SEC1_LENGTH + 1, &self));
        }
        P384PublicKey::from_sec1_bytes(&bytes)
          .map_err(|_| serde::de::Error::custom("invalid canonical uncompressed P-384 public key"))
      }
    }

    deserializer.deserialize_bytes(PublicKeyVisitor)
  }
}

impl fmt::Debug for P384PublicKey {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "P384PublicKey(")?;
    crate::hex::fmt_hex_lower(&self.bytes, f)?;
    write!(f, ")")
  }
}

/// Fixed-width P-384 ECC CDH shared secret.
pub struct P384SharedSecret([u8; FIELD_BYTES]);

impl P384SharedSecret {
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

impl fmt::Debug for P384SharedSecret {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("P384SharedSecret(****)")
  }
}

impl Drop for P384SharedSecret {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

/// Return the production P-384 fixed-base comb selection as Montgomery limbs.
#[cfg(all(rscrypto_internal, feature = "diag"))]
#[doc(hidden)]
pub fn diag_p384_ecdh_select_generator_limb_digest(digit: u8) -> [u64; 12] {
  p384_portable::diag_select_generator_limb_digest(digit)
}

/// Return the production P-384 signed-window selection as Montgomery limbs.
#[cfg(all(rscrypto_internal, feature = "diag"))]
#[doc(hidden)]
pub fn diag_p384_ecdh_select_window_limb_digest(digit: u8) -> [u64; 30] {
  p384_portable::diag_select_window_limb_digest(digit)
}

/// Exercise P-384 ECDH candidate cleanup on success and partial-fill failure.
#[cfg(all(rscrypto_internal, feature = "diag"))]
#[doc(hidden)]
#[unsafe(no_mangle)]
#[inline(never)]
pub(crate) fn diag_zeroize_p384_ecdh_generation(value: u8, fail: bool) -> u8 {
  let result = P384EphemeralSecret::try_generate_with(|candidate| {
    candidate[..24].fill(core::hint::black_box(value & 0x7f));
    if core::hint::black_box(fail) {
      return Err(());
    }
    candidate[24..].fill(core::hint::black_box(value));
    Ok(())
  });
  result.map_or(0, |secret| core::hint::black_box(secret.0[0]))
}

/// Exercise P-384 ECDH scalar, projective-state, and shared-secret cleanup.
#[cfg(all(rscrypto_internal, feature = "diag"))]
#[doc(hidden)]
#[unsafe(no_mangle)]
#[inline(never)]
pub(crate) fn diag_zeroize_p384_ecdh_agreement(secret: [u8; 48]) -> u8 {
  let Ok(secret) = P384EphemeralSecret::try_generate_with(|candidate| {
    candidate.copy_from_slice(core::hint::black_box(&secret));
    Ok::<(), core::convert::Infallible>(())
  }) else {
    return 0;
  };
  let Ok(peer) = P384EphemeralSecret::try_generate_with(|candidate| {
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
  use super::{P384EphemeralSecret, P384KeyGenerationError, P384PublicKey, P384PublicKeyError};

  const GENERATOR_X_BYTES: [u8; 48] = [
    0xaa, 0x87, 0xca, 0x22, 0xbe, 0x8b, 0x05, 0x37, 0x8e, 0xb1, 0xc7, 0x1e, 0xf3, 0x20, 0xad, 0x74, 0x6e, 0x1d, 0x3b,
    0x62, 0x8b, 0xa7, 0x9b, 0x98, 0x59, 0xf7, 0x41, 0xe0, 0x82, 0x54, 0x2a, 0x38, 0x55, 0x02, 0xf2, 0x5d, 0xbf, 0x55,
    0x29, 0x6c, 0x3a, 0x54, 0x5e, 0x38, 0x72, 0x76, 0x0a, 0xb7,
  ];
  #[cfg(miri)]
  const GENERATOR_Y_BYTES: [u8; 48] = [
    0x36, 0x17, 0xde, 0x4a, 0x96, 0x26, 0x2c, 0x6f, 0x5d, 0x9e, 0x98, 0xbf, 0x92, 0x92, 0xdc, 0x29, 0xf8, 0xf4, 0x1d,
    0xbd, 0x28, 0x9a, 0x14, 0x7c, 0xe9, 0xda, 0x31, 0x13, 0xb5, 0xf0, 0xb8, 0xc0, 0x0a, 0x60, 0xb1, 0xce, 0x1d, 0x7e,
    0x81, 0x9d, 0x7a, 0x43, 0x1d, 0x7c, 0x90, 0xea, 0x0e, 0x5f,
  ];

  fn generated(bytes: [u8; 48]) -> P384EphemeralSecret {
    P384EphemeralSecret::try_generate_with(|out| {
      out.copy_from_slice(&bytes);
      Ok::<(), core::convert::Infallible>(())
    })
    .expect("valid P-384 scalar must generate")
  }

  #[test]
  fn scalar_one_derives_the_p384_generator() {
    let mut scalar = [0u8; 48];
    scalar[47] = 1;
    let public = generated(scalar).public_key();
    assert_eq!(public.as_sec1_bytes()[0], 0x04);
    assert_eq!(&public.as_sec1_bytes()[1..49], &GENERATOR_X_BYTES);
    assert_eq!(P384PublicKey::from_sec1_bytes(public.as_sec1_bytes()), Ok(public));
  }

  #[test]
  fn agreement_is_symmetric_and_fixed_width() {
    let alice = generated([0x07; 48]);
    let bob = generated([0x09; 48]);
    let alice_public = alice.public_key();
    let bob_public = bob.public_key();
    let alice_shared = alice.diffie_hellman(&bob_public);
    let bob_shared = bob.diffie_hellman(&alice_public);
    assert!(alice_shared.ct_eq(&bob_shared).declassify());
    assert_eq!(alice_shared.as_bytes().len(), 48);
  }

  #[test]
  fn parsing_rejects_every_public_shape_class() {
    assert_eq!(P384PublicKey::from_sec1_bytes(&[]), Err(P384PublicKeyError));
    assert_eq!(P384PublicKey::from_sec1_bytes(&[0x04; 96]), Err(P384PublicKeyError));
    assert_eq!(P384PublicKey::from_sec1_bytes(&[0x04; 98]), Err(P384PublicKeyError));
    let mut compressed = [0u8; 49];
    compressed[0] = 0x02;
    assert_eq!(P384PublicKey::from_sec1_bytes(&compressed), Err(P384PublicKeyError));
    assert_eq!(P384PublicKey::from_sec1_bytes(&[0u8; 97]), Err(P384PublicKeyError));
    assert_eq!(P384PublicKey::from_sec1_bytes(&[0x04; 97]), Err(P384PublicKeyError));
    let mut non_canonical = [0xffu8; 97];
    non_canonical[0] = 0x04;
    assert_eq!(P384PublicKey::from_sec1_bytes(&non_canonical), Err(P384PublicKeyError));
  }

  #[test]
  fn generation_propagates_partial_fill_failure_and_bounds_rejection() {
    let expected = 17u8;
    let error = P384EphemeralSecret::try_generate_with(|out| {
      out[..4].fill(0xaa);
      Err(expected)
    })
    .expect_err("failing entropy callback must be returned");
    assert_eq!(error, P384KeyGenerationError::Random(expected));

    let mut calls = 0usize;
    let error = P384EphemeralSecret::try_generate_with(|out| {
      calls = calls.strict_add(1);
      out.fill(0);
      Ok::<(), core::convert::Infallible>(())
    })
    .expect_err("zero is never a valid P-384 scalar");
    assert_eq!(error, P384KeyGenerationError::ScalarSamplingExhausted);
    assert_eq!(calls, 32);

    let mut calls = 0usize;
    let secret = P384EphemeralSecret::try_generate_with(|out| {
      calls = calls.strict_add(1);
      if calls == 1 {
        out.fill(0xff);
      } else {
        assert_eq!(*out, [0u8; 48], "each rejection-sampling attempt starts cleared");
        out[47] = 1;
      }
      Ok::<(), core::convert::Infallible>(())
    })
    .expect("second candidate is scalar one");
    assert_eq!(calls, 2);
    assert_eq!(secret.public_key().as_sec1_bytes()[1..49], GENERATOR_X_BYTES);
  }

  #[cfg(miri)]
  #[test]
  fn miri_uses_portable_p384_ecdh_path() {
    let mut peer = [0u8; 97];
    peer[0] = 0x04;
    peer[1..49].copy_from_slice(&GENERATOR_X_BYTES);
    peer[49..].copy_from_slice(&GENERATOR_Y_BYTES);
    let peer = P384PublicKey::from_sec1_bytes(&peer).expect("generator must parse");
    let shared = generated([0x42; 48]).diffie_hellman(&peer);
    assert_eq!(shared.as_bytes().len(), P384EphemeralSecret::LENGTH);
  }
}
