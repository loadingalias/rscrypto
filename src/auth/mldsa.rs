//! ML-DSA signatures (FIPS 204), with original portable arithmetic.
//!
//! All messages and contexts use byte strings. A context is at most 255 bytes.
//! Hedged signing takes caller-provided cryptographic randomness; deterministic
//! signing is an explicit, separate operation. The `ml-dsa` leaf needs neither
//! allocation nor OS entropy. See [`MlDsaPrehash`] for HashML-DSA.
//!
//! Public keys have RFC 9881 SubjectPublicKeyInfo import and export. Private
//! keys have RFC 9881 PKCS #8 import and export: a seed owner such as
//! [`MlDsa44Seed`] keeps the recommended seed form, and a secret key exports
//! the expanded form.
//!
//! Target qualification is ongoing; no whole-operation constant-time claim
//! is made. Signing needs tens of KiB of stack. Prepared keys keep up to
//! 79 KiB of polynomials in caller-owned storage, which may live on the stack,
//! in a static, or on the heap. Core-only availability does not establish
//! suitability for a constrained stack.
//!
//! ```
//! use rscrypto::{MlDsa44, MlDsa44PreparedSecretKeyStorage, MlDsaError};
//! // Fixed seeds are suitable for reproducible examples, not production keys.
//! let (public, secret) = MlDsa44::keypair_from_seed(&[7; 32])?;
//! let signature = secret.sign_deterministic(b"release manifest", b"example")?;
//! public.verify_with_context(b"release manifest", b"example", &signature)?;
//!
//! // Repeated signing: expand the key once into caller-owned storage.
//! let mut storage = MlDsa44PreparedSecretKeyStorage::new();
//! let prepared = secret.prepare(&mut storage)?;
//! assert_eq!(prepared.sign_deterministic(b"release manifest", b"example")?, signature);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod encoding;
mod pkcs8;
mod poly;
mod portable;
mod sampling;
#[cfg(feature = "serde")]
mod serde_impl;
mod spki;
#[cfg(test)]
mod tests;

use crate::secret::ZeroizingBytes;
use crate::traits::ct::{self, DataIndependentTiming};
use crate::{SecretBytes, VerificationError, Verifier};
#[cfg(feature = "alloc")]
use alloc::boxed::Box;
#[cfg(feature = "alloc")]
use core::alloc::Allocator;
use core::fmt;

/// ML-DSA key generation, preparation, or signing failure.
///
/// Key import uses [`MlDsaKeyError`]; verification uses opaque [`VerificationError`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlDsaError {
  /// Signing or preparation found an invalid stored secret-key encoding.
  ///
  /// Every constructor validates the key, so this arises only if key memory
  /// changes after validation. The operation clears its secret state first.
  InvalidSecretKey,
  /// The signature has the wrong length, noncanonical hints, or an invalid response norm.
  InvalidSignature,
  /// The context exceeds the FIPS 204 limit of 255 bytes.
  ContextTooLong,
  /// The prehash length does not match its declared algorithm.
  InvalidPrehashLength,
  /// The random source failed, including after partially filling its buffer.
  RandomGenerationFailed,
  /// A FIPS 204 bounded sampler or signing loop exhausted its limit.
  RejectionLimit,
}

impl fmt::Display for MlDsaError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(match self {
      Self::InvalidSecretKey => "invalid ML-DSA secret key",
      Self::InvalidSignature => "invalid ML-DSA signature",
      Self::ContextTooLong => "ML-DSA context exceeds 255 bytes",
      Self::InvalidPrehashLength => "ML-DSA prehash length does not match its algorithm",
      Self::RandomGenerationFailed => "ML-DSA random generation failed",
      Self::RejectionLimit => "ML-DSA rejection limit reached",
    })
  }
}

impl core::error::Error for MlDsaError {}

/// ML-DSA key import failure.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum MlDsaKeyError {
  /// DER input was malformed or non-canonical.
  MalformedDer,
  /// The algorithm identifier names another algorithm or ML-DSA parameter set.
  UnsupportedAlgorithm,
  /// The DER is a well-formed key of this algorithm in a form this import
  /// does not accept: PKCS #8 attributes, or an expanded-only private key
  /// where a seed is required.
  UnsupportedEncoding,
  /// The public-key encoding has the wrong length.
  InvalidPublicKey,
  /// The secret-key encoding is malformed or its redundant fields disagree.
  InvalidSecretKey,
  /// Key expansion exhausted a FIPS 204 Appendix C sampling bound. For an
  /// expanded key the outcome depends only on its public seed; expanding a
  /// private seed also samples the secret vectors.
  RejectionLimit,
}

impl fmt::Display for MlDsaKeyError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(match self {
      Self::MalformedDer => "malformed ML-DSA DER",
      Self::UnsupportedAlgorithm => "unsupported ML-DSA key algorithm",
      Self::UnsupportedEncoding => "unsupported ML-DSA key encoding",
      Self::InvalidPublicKey => "invalid ML-DSA public key",
      Self::InvalidSecretKey => "invalid ML-DSA secret key",
      Self::RejectionLimit => "ML-DSA key expansion reached its rejection limit",
    })
  }
}

impl core::error::Error for MlDsaKeyError {}

/// Key import from a seed: generation fails only by exhausting a FIPS 204
/// sampling bound.
fn seed_expansion_error(error: MlDsaError) -> MlDsaKeyError {
  debug_assert_eq!(error, MlDsaError::RejectionLimit);
  MlDsaKeyError::RejectionLimit
}

/// Standard hash identifiers for HashML-DSA.
///
/// Choosing a prehash limits the signature's collision security to that of the
/// hash. SHA-224 and SHA-512/224 provide 112 bits; the 256-bit digests and SHAKE128
/// provide 128; 384-bit digests provide 192; SHA-512, SHA3-512, and SHAKE256 provide
/// 256. Select a profile with enough strength for the application and parameter set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlDsaPrehashAlgorithm {
  /// SHA-224, 28 bytes.
  Sha224,
  /// SHA-256, 32 bytes.
  Sha256,
  /// SHA-384, 48 bytes.
  Sha384,
  /// SHA-512, 64 bytes.
  Sha512,
  /// SHA-512/224, 28 bytes.
  Sha512_224,
  /// SHA-512/256, 32 bytes.
  Sha512_256,
  /// SHA3-224, 28 bytes.
  Sha3_224,
  /// SHA3-256, 32 bytes.
  Sha3_256,
  /// SHA3-384, 48 bytes.
  Sha3_384,
  /// SHA3-512, 64 bytes.
  Sha3_512,
  /// SHAKE128 with a 32-byte output.
  Shake128,
  /// SHAKE256 with a 64-byte output.
  Shake256,
}

impl MlDsaPrehashAlgorithm {
  /// Required digest length, in bytes.
  #[must_use]
  pub const fn digest_len(self) -> usize {
    match self {
      Self::Sha224 | Self::Sha512_224 | Self::Sha3_224 => 28,
      Self::Sha256 | Self::Sha512_256 | Self::Sha3_256 | Self::Shake128 => 32,
      Self::Sha384 | Self::Sha3_384 => 48,
      Self::Sha512 | Self::Sha3_512 | Self::Shake256 => 64,
    }
  }

  const fn oid(self) -> [u8; 11] {
    let suffix = match self {
      Self::Sha256 => 1,
      Self::Sha384 => 2,
      Self::Sha512 => 3,
      Self::Sha224 => 4,
      Self::Sha512_224 => 5,
      Self::Sha512_256 => 6,
      Self::Sha3_224 => 7,
      Self::Sha3_256 => 8,
      Self::Sha3_384 => 9,
      Self::Sha3_512 => 10,
      Self::Shake128 => 11,
      Self::Shake256 => 12,
    };
    [0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, suffix]
  }
}

/// Borrowed, length-checked digest for HashML-DSA.
///
/// The caller computes the digest of the original message using `algorithm`.
/// Construction checks its length, not its provenance. The algorithm identifier
/// and digest are both bound into the signature using FIPS 204 domain separation.
#[derive(Clone, Copy)]
pub struct MlDsaPrehash<'a> {
  algorithm: MlDsaPrehashAlgorithm,
  digest: &'a [u8],
}

impl<'a> MlDsaPrehash<'a> {
  /// Bind a digest to its algorithm. Returns an error for an incorrect length.
  pub fn new(algorithm: MlDsaPrehashAlgorithm, digest: &'a [u8]) -> Result<Self, MlDsaError> {
    if digest.len() != algorithm.digest_len() {
      return Err(MlDsaError::InvalidPrehashLength);
    }
    Ok(Self { algorithm, digest })
  }
}

impl fmt::Debug for MlDsaPrehash<'_> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("MlDsaPrehash")
      .field("algorithm", &self.algorithm)
      .finish_non_exhaustive()
  }
}

#[derive(Clone, Copy)]
struct Parameters {
  k: usize,
  l: usize,
  eta: u32,
  eta_bits: usize,
  tau: usize,
  beta: u32,
  gamma1: u32,
  gamma2: u32,
  z_bits: usize,
  w1_bits: usize,
  omega: usize,
  challenge_bytes: usize,
}

impl Parameters {
  const fn signature_len(self) -> usize {
    self
      .challenge_bytes
      .strict_add(self.l.strict_mul(self.z_bits.strict_mul(32)))
      .strict_add(self.omega)
      .strict_add(self.k)
  }
}

const P44: Parameters = Parameters {
  k: 4,
  l: 4,
  eta: 2,
  eta_bits: 3,
  tau: 39,
  beta: 78,
  gamma1: 1 << 17,
  gamma2: 95_232,
  z_bits: 18,
  w1_bits: 6,
  omega: 80,
  challenge_bytes: 32,
};
const P65: Parameters = Parameters {
  k: 6,
  l: 5,
  eta: 4,
  eta_bits: 4,
  tau: 49,
  beta: 196,
  gamma1: 1 << 19,
  gamma2: 261_888,
  z_bits: 20,
  w1_bits: 4,
  omega: 55,
  challenge_bytes: 48,
};
const P87: Parameters = Parameters {
  k: 8,
  l: 7,
  eta: 2,
  eta_bits: 3,
  tau: 60,
  beta: 120,
  gamma1: 1 << 19,
  gamma2: 261_888,
  z_bits: 20,
  w1_bits: 4,
  omega: 75,
  challenge_bytes: 64,
};

fn representative(
  tr: &[u8],
  message: &[u8],
  context: &[u8],
  prehash: Option<MlDsaPrehash<'_>>,
  mu: &mut [u8; 64],
) -> Result<(), MlDsaError> {
  let context_len = u8::try_from(context.len()).map_err(|_| MlDsaError::ContextTooLong)?;
  match prehash {
    Some(prehash) => sampling::hash(
      &[tr, &[1, context_len], context, &prehash.algorithm.oid(), prehash.digest],
      mu,
    ),
    None => sampling::hash(&[tr, &[0, context_len], context, message], mu),
  }
  Ok(())
}

macro_rules! signing_methods {
  ($signature:ident) => {
    /// Pure deterministic ML-DSA. Contexts longer than 255 bytes are rejected.
    /// Sampler exhaustion returns no signature and clears intermediate state.
    pub fn sign_deterministic(&self, message: &[u8], context: &[u8]) -> Result<$signature, MlDsaError> {
      let mut mu = ZeroizingBytes::zeroed();
      representative(
        &self.encoded_secret()[64..128],
        message,
        context,
        None,
        mu.as_mut_array(),
      )?;
      self.sign_representative(mu.as_array(), &[0; 32])
    }

    /// Pure hedged ML-DSA using 32 bytes from `fill_random`.
    /// Validate context before requesting entropy. Errors leave the key unchanged.
    pub fn sign_with(
      &self,
      message: &[u8],
      context: &[u8],
      mut fill_random: impl FnMut(&mut [u8]) -> Result<(), MlDsaError>,
    ) -> Result<$signature, MlDsaError> {
      let mut mu = ZeroizingBytes::zeroed();
      representative(
        &self.encoded_secret()[64..128],
        message,
        context,
        None,
        mu.as_mut_array(),
      )?;
      let mut random = ZeroizingBytes::zeroed();
      fill_random(random.as_mut_array())?;
      self.sign_representative(mu.as_array(), random.as_array())
    }

    /// Deterministic HashML-DSA over an explicitly identified prehash.
    pub fn sign_prehash_deterministic(
      &self,
      prehash: MlDsaPrehash<'_>,
      context: &[u8],
    ) -> Result<$signature, MlDsaError> {
      let mut mu = ZeroizingBytes::zeroed();
      representative(
        &self.encoded_secret()[64..128],
        &[],
        context,
        Some(prehash),
        mu.as_mut_array(),
      )?;
      self.sign_representative(mu.as_array(), &[0; 32])
    }

    /// Hedged HashML-DSA using caller-provided cryptographic randomness.
    pub fn sign_prehash_with(
      &self,
      prehash: MlDsaPrehash<'_>,
      context: &[u8],
      mut fill_random: impl FnMut(&mut [u8]) -> Result<(), MlDsaError>,
    ) -> Result<$signature, MlDsaError> {
      let mut mu = ZeroizingBytes::zeroed();
      representative(
        &self.encoded_secret()[64..128],
        &[],
        context,
        Some(prehash),
        mu.as_mut_array(),
      )?;
      let mut random = ZeroizingBytes::zeroed();
      fill_random(random.as_mut_array())?;
      self.sign_representative(mu.as_array(), random.as_array())
    }

    /// Pure hedged ML-DSA using OS entropy.
    #[cfg(feature = "getrandom")]
    pub fn try_sign(&self, message: &[u8], context: &[u8]) -> Result<$signature, MlDsaError> {
      self.sign_with(message, context, |out| {
        getrandom::fill(out).map_err(|_| MlDsaError::RandomGenerationFailed)
      })
    }
  };
}

macro_rules! verification_methods {
  ($signature:ident) => {
    /// Verify pure ML-DSA with an empty context.
    pub fn verify(&self, message: &[u8], signature: &$signature) -> Result<(), VerificationError> {
      self.verify_with_context(message, &[], signature)
    }

    /// Verify pure ML-DSA. Every invalid signature or context returns an opaque error.
    pub fn verify_with_context(
      &self,
      message: &[u8],
      context: &[u8],
      signature: &$signature,
    ) -> Result<(), VerificationError> {
      self.verify_input(message, context, None, signature)
    }

    /// Verify HashML-DSA over a caller-computed, algorithm-bound digest.
    pub fn verify_prehash(
      &self,
      prehash: MlDsaPrehash<'_>,
      context: &[u8],
      signature: &$signature,
    ) -> Result<(), VerificationError> {
      self.verify_input(&[], context, Some(prehash), signature)
    }
  };
}

macro_rules! parameter_set {
  ($profile:ident, $public:ident, $secret:ident, $seed:ident, $signature:ident, $prepared_secret:ident, $prepared_public:ident, $secret_storage:ident, $public_storage:ident, $p:ident, $k:literal, $l:literal, $pk:literal, $sk:literal, $sig:literal, $arc:literal) => {
    /// FIPS 204 parameter set with typed key-generation outputs.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct $profile;

    impl $profile {
      /// Deterministically expand a 32-byte secret seed into a key pair.
      ///
      /// Use a fresh cryptographically random seed in production. The caller owns
      /// cleanup of the borrowed seed. Internal sampler exhaustion returns an error.
      pub fn keypair_from_seed(seed: &[u8; 32]) -> Result<($public, $secret), MlDsaError> {
        let mut public = $public([0; $pk]);
        let mut bytes = ZeroizingBytes::<$sk>::zeroed();
        portable::keygen::<$k, $l>(seed, $p, &mut public.0, bytes.as_mut_array())?;
        let secret = $secret {
          bytes: ZeroizingBytes::new(*bytes.as_array()),
          public: public.clone(),
        };
        Ok((public, secret))
      }

      /// Generate keys using 32 bytes from `fill_random`.
      /// The callback must fill the entire buffer with fresh cryptographic randomness.
      /// Failure clears even a partially filled buffer and returns no key material.
      pub fn generate_keypair(
        mut fill_random: impl FnMut(&mut [u8]) -> Result<(), MlDsaError>,
      ) -> Result<($public, $secret), MlDsaError> {
        let mut seed = ZeroizingBytes::zeroed();
        fill_random(seed.as_mut_array())?;
        Self::keypair_from_seed(seed.as_array())
      }

      /// Generate a key pair using OS entropy.
      #[cfg(feature = "getrandom")]
      pub fn try_generate_keypair() -> Result<($public, $secret), MlDsaError> {
        Self::generate_keypair(|out| getrandom::fill(out).map_err(|_| MlDsaError::RandomGenerationFailed))
      }

      /// Like [`Self::keypair_from_seed`], with the secret key in memory from `alloc`.
      ///
      /// The key is generated directly into its allocation, so moving the box
      /// moves only a pointer and no by-value copy of the key is left behind.
      /// The box clears the key on drop; failure clears it before returning.
      /// Allocation failure is handled as by [`Box::new_in`].
      #[cfg(feature = "alloc")]
      pub fn keypair_from_seed_in<A: Allocator>(
        seed: &[u8; 32],
        alloc: A,
      ) -> Result<($public, Box<$secret, A>), MlDsaError> {
        let mut secret = Box::new_in($secret::zeroed(), alloc);
        portable::keygen::<$k, $l>(seed, $p, &mut secret.public.0, secret.bytes.as_mut_array())?;
        Ok((secret.public.clone(), secret))
      }

      /// Like [`Self::generate_keypair`], with the secret key in memory from `alloc`.
      /// See [`Self::keypair_from_seed_in`].
      #[cfg(feature = "alloc")]
      pub fn generate_keypair_in<A: Allocator>(
        mut fill_random: impl FnMut(&mut [u8]) -> Result<(), MlDsaError>,
        alloc: A,
      ) -> Result<($public, Box<$secret, A>), MlDsaError> {
        let mut seed = ZeroizingBytes::zeroed();
        fill_random(seed.as_mut_array())?;
        Self::keypair_from_seed_in(seed.as_array(), alloc)
      }

      /// Like [`Self::try_generate_keypair`], with the secret key in memory from `alloc`.
      #[cfg(all(feature = "alloc", feature = "getrandom"))]
      pub fn try_generate_keypair_in<A: Allocator>(alloc: A) -> Result<($public, Box<$secret, A>), MlDsaError> {
        Self::generate_keypair_in(
          |out| getrandom::fill(out).map_err(|_| MlDsaError::RandomGenerationFailed),
          alloc,
        )
      }
    }

    /// Canonical FIPS 204 encoded public key.
    #[derive(Clone, Eq, PartialEq)]
    pub struct $public([u8; $pk]);

    impl $public {
      /// Encoded public-key length in bytes.
      pub const LENGTH: usize = $pk;

      /// Construct from a fixed-width encoding. All bit patterns are admissible.
      #[must_use]
      pub const fn from_bytes(bytes: [u8; $pk]) -> Self {
        Self(bytes)
      }

      /// Parse a public key, rejecting an incorrect length.
      pub fn try_from_slice(bytes: &[u8]) -> Result<Self, MlDsaKeyError> {
        let array = bytes.try_into().map_err(|_| MlDsaKeyError::InvalidPublicKey)?;
        Ok(Self(array))
      }

      /// DER-encoded RFC 9881 SubjectPublicKeyInfo length in bytes.
      pub const SPKI_DER_LENGTH: usize = spki::HEADER_LENGTH.strict_add($pk);

      const SPKI_HEADER: [u8; spki::HEADER_LENGTH] = spki::header($arc, $pk);

      /// Parse an RFC 9881 SubjectPublicKeyInfo for this parameter set.
      ///
      /// Accepts only the unique DER encoding: this parameter set's algorithm
      /// identifier with absent parameters, a BIT STRING with no unused bits,
      /// the exact public-key length, and no trailing input.
      ///
      /// # Errors
      ///
      /// Returns [`MlDsaKeyError::UnsupportedAlgorithm`] for a well-formed key of
      /// another algorithm or ML-DSA parameter set, including HashML-DSA;
      /// [`MlDsaKeyError::InvalidPublicKey`] for a wrong key length; and
      /// [`MlDsaKeyError::MalformedDer`] for any other encoding. No heap allocation.
      pub fn from_spki_der(der: &[u8]) -> Result<Self, MlDsaKeyError> {
        Self::try_from_slice(spki::decode(der, &Self::SPKI_HEADER, $pk)?)
      }

      /// Encode the RFC 9881 SubjectPublicKeyInfo DER. No heap allocation.
      #[must_use]
      pub const fn to_spki_der(&self) -> [u8; Self::SPKI_DER_LENGTH] {
        spki::encode(&Self::SPKI_HEADER, &self.0)
      }

      /// Prepare the matrix and transformed public key in caller-owned storage
      /// for repeated verification.
      ///
      /// The returned handle borrows this key and `storage`; the storage can be
      /// reused after every handle is dropped. Ordinary verification expands
      /// rows on demand instead. No heap allocation.
      pub fn prepare<'a>(&'a self, storage: &'a mut $public_storage) -> Result<$prepared_public<'a>, MlDsaError> {
        storage.state.prepare(&self.0)?;
        Ok($prepared_public {
          key: self,
          state: &storage.state,
        })
      }

      /// Borrow the canonical encoding.
      #[must_use]
      pub const fn as_bytes(&self) -> &[u8; $pk] {
        &self.0
      }

      /// Copy the public encoding.
      #[must_use]
      pub const fn to_bytes(&self) -> [u8; $pk] {
        self.0
      }

      verification_methods!($signature);

      fn verify_input(
        &self,
        message: &[u8],
        context: &[u8],
        prehash: Option<MlDsaPrehash<'_>>,
        signature: &$signature,
      ) -> Result<(), VerificationError> {
        let mut tr = [0; 64];
        sampling::hash(&[&self.0], &mut tr);
        let mut mu = ZeroizingBytes::zeroed();
        representative(&tr, message, context, prehash, mu.as_mut_array()).map_err(|_| VerificationError::new())?;
        portable::verify::<$k, $l>(&self.0, mu.as_array(), &signature.0, $p).map_err(|_| VerificationError::new())
      }
    }

    impl AsRef<[u8]> for $public {
      fn as_ref(&self) -> &[u8] {
        &self.0
      }
    }
    impl fmt::Debug for $public {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple(stringify!($public)).field(&self.0.as_slice()).finish()
      }
    }
    impl Verifier<$signature> for $public {
      fn verify(&self, message: &[u8], signature: &$signature) -> Result<(), VerificationError> {
        self.verify(message, signature)
      }
    }

    /// Validated expanded secret key. Not `Clone` or `Copy`; owned bytes are zeroized.
    pub struct $secret {
      bytes: ZeroizingBytes<$sk>,
      public: $public,
    }

    impl $secret {
      /// FIPS 204 expanded secret-key encoding length.
      pub const LENGTH: usize = $sk;

      /// Import and validate an expanded secret key, including t0 and the public-key hash.
      /// The caller retains responsibility for clearing the borrowed input.
      pub fn try_from_slice(input: &[u8]) -> Result<Self, MlDsaKeyError> {
        if input.len() != $sk {
          return Err(MlDsaKeyError::InvalidSecretKey);
        }
        let mut bytes = ZeroizingBytes::zeroed();
        bytes.as_mut_array().copy_from_slice(input);
        let mut public = $public([0; $pk]);
        portable::validate_secret::<$k, $l>(bytes.as_array(), $p, &mut public.0)?;
        Ok(Self {
          bytes: ZeroizingBytes::new(*bytes.as_array()),
          public,
        })
      }

      /// Like [`Self::try_from_slice`], with the key imported into memory from `alloc`.
      ///
      /// The key is copied directly into its allocation, so moving the box moves
      /// only a pointer. The box clears the key on drop; a rejected key is
      /// cleared before returning. Allocation failure is handled as by
      /// [`Box::new_in`]. The caller retains responsibility for clearing `input`.
      #[cfg(feature = "alloc")]
      pub fn try_from_slice_in<A: Allocator>(input: &[u8], alloc: A) -> Result<Box<Self, A>, MlDsaKeyError> {
        if input.len() != $sk {
          return Err(MlDsaKeyError::InvalidSecretKey);
        }
        let mut secret = Box::new_in(Self::zeroed(), alloc);
        secret.bytes.as_mut_array().copy_from_slice(input);
        portable::validate_secret::<$k, $l>(secret.bytes.as_array(), $p, &mut secret.public.0)?;
        Ok(secret)
      }

      /// Zero-filled owner that an allocation is filled from; never returned.
      #[cfg(feature = "alloc")]
      const fn zeroed() -> Self {
        Self {
          bytes: ZeroizingBytes::zeroed(),
          public: $public([0; $pk]),
        }
      }

      /// DER length of this key as an RFC 9881 expanded-form PKCS #8 private key.
      pub const PKCS8_DER_LENGTH: usize = pkcs8::EXPANDED_HEADER_LENGTH.strict_add($sk);

      const PKCS8_HEADER: [u8; pkcs8::EXPANDED_HEADER_LENGTH] = pkcs8::expanded_header($arc, $sk);

      /// Import an RFC 9881 private key from RFC 5958 OneAsymmetricKey
      /// (PKCS #8) DER for this parameter set.
      ///
      /// Accepts the seed, expanded, and both forms, in a version 1 container
      /// or a version 2 container with a public key. Expanded keys are
      /// validated as by [`Self::try_from_slice`]. A seed is expanded and not
      /// retained. In the both form, the expanded key must equal the seed's;
      /// a version 2 public key must belong to the key. The caller retains
      /// responsibility for clearing `der`. No heap allocation.
      #[doc = concat!("Use [`", stringify!($seed), "::from_pkcs8_der`] to keep the seed.")]
      ///
      /// # Errors
      ///
      /// Returns [`MlDsaKeyError::MalformedDer`] for malformed or non-canonical
      /// DER, including a version that disagrees with the public-key field;
      /// [`MlDsaKeyError::UnsupportedAlgorithm`] for another algorithm or
      /// parameter set; [`MlDsaKeyError::UnsupportedEncoding`] for attributes;
      /// [`MlDsaKeyError::InvalidSecretKey`] for a wrong-length or invalid key,
      /// or redundant fields that disagree; [`MlDsaKeyError::InvalidPublicKey`]
      /// for a wrong-length public key; and [`MlDsaKeyError::RejectionLimit`]
      /// if expansion exhausts its sampling bound.
      pub fn from_pkcs8_der(der: &[u8]) -> Result<Self, MlDsaKeyError> {
        let _dit = DataIndependentTiming::enter();
        let decoded = pkcs8::decode::<$sk, $pk>(der, $arc)?;
        let (secret, expanded) = match decoded.private_key {
          pkcs8::PrivateKey::Seed(seed) => (Self::from_seed(seed)?, None),
          pkcs8::PrivateKey::Expanded(expanded) => (Self::try_from_slice(expanded)?, None),
          pkcs8::PrivateKey::Both { seed, expanded } => (Self::from_seed(seed)?, Some(expanded)),
        };
        secret.check_redundant(expanded, decoded.public_key)?;
        Ok(secret)
      }

      /// Like [`Self::from_pkcs8_der`], with the key in memory from `alloc`.
      ///
      /// The key is written directly into its allocation, as by
      /// [`Self::try_from_slice_in`]. The box clears the key on drop; a
      /// rejected key is cleared before returning. Allocation failure is
      /// handled as by [`Box::new_in`].
      #[cfg(feature = "alloc")]
      pub fn from_pkcs8_der_in<A: Allocator>(der: &[u8], alloc: A) -> Result<Box<Self, A>, MlDsaKeyError> {
        let _dit = DataIndependentTiming::enter();
        let decoded = pkcs8::decode::<$sk, $pk>(der, $arc)?;
        let (secret, expanded) = match decoded.private_key {
          pkcs8::PrivateKey::Seed(seed) => (Self::from_seed_in(seed, alloc)?, None),
          pkcs8::PrivateKey::Expanded(expanded) => (Self::try_from_slice_in(expanded, alloc)?, None),
          pkcs8::PrivateKey::Both { seed, expanded } => (Self::from_seed_in(seed, alloc)?, Some(expanded)),
        };
        secret.check_redundant(expanded, decoded.public_key)?;
        Ok(secret)
      }

      /// Write this key as an RFC 9881 expanded-form private key in version 1
      /// OneAsymmetricKey (PKCS #8) DER. No heap allocation.
      ///
      /// `out` then holds the secret key, and the caller owns its cleanup.
      /// This key keeps no seed, so it cannot write the recommended seed form.
      #[doc = concat!("[`", stringify!($seed), "::to_pkcs8_der_into`] writes it.")]
      pub fn to_pkcs8_der_into(&self, out: &mut [u8; Self::PKCS8_DER_LENGTH]) {
        pkcs8::write(&Self::PKCS8_HEADER, self.bytes.as_array(), out);
      }

      fn from_seed(seed: &[u8; 32]) -> Result<Self, MlDsaKeyError> {
        $profile::keypair_from_seed(seed)
          .map(|(_, secret)| secret)
          .map_err(seed_expansion_error)
      }

      #[cfg(feature = "alloc")]
      fn from_seed_in<A: Allocator>(seed: &[u8; 32], alloc: A) -> Result<Box<Self, A>, MlDsaKeyError> {
        $profile::keypair_from_seed_in(seed, alloc)
          .map(|(_, secret)| secret)
          .map_err(seed_expansion_error)
      }

      /// Reject imported redundant fields that disagree with this key: an
      /// expanded key that is not its seed's (RFC 9881 section 8.2), or a
      /// public key that is not its own.
      fn check_redundant(&self, expanded: Option<&[u8; $sk]>, public: Option<&[u8; $pk]>) -> Result<(), MlDsaKeyError> {
        if let Some(expanded) = expanded
          && !ct::fixed_eq(self.bytes.as_array(), expanded).declassify()
        {
          return Err(MlDsaKeyError::InvalidSecretKey);
        }
        if public.is_some_and(|public| *public != self.public.0) {
          return Err(MlDsaKeyError::InvalidSecretKey);
        }
        Ok(())
      }

      /// Prepare secret polynomials and the public matrix in caller-owned
      /// storage for repeated signing.
      ///
      /// The returned handle borrows this key and `storage`. It clears the
      /// transformed secrets when dropped; a failed preparation clears them
      /// before returning. The storage can be reused after the handle is
      /// dropped. No heap allocation. See the module's resource contract.
      pub fn prepare<'a>(&'a self, storage: &'a mut $secret_storage) -> Result<$prepared_secret<'a>, MlDsaError> {
        let prepared = $prepared_secret { key: self, storage };
        if !prepared.storage.state.decode(self.bytes.as_array(), $p) {
          return Err(MlDsaError::InvalidSecretKey);
        }
        prepared.storage.matrix.expand_into(&self.bytes.as_array()[..32])?;
        Ok(prepared)
      }

      /// Borrow the validated public key.
      #[must_use]
      pub const fn public_key(&self) -> &$public {
        &self.public
      }

      /// Explicitly export the expanded key into a zeroizing owner.
      #[must_use]
      pub fn expose_secret(&self) -> SecretBytes<$sk> {
        SecretBytes::new(*self.bytes.as_array())
      }

      signing_methods!($signature);

      fn encoded_secret(&self) -> &[u8; $sk] {
        self.bytes.as_array()
      }

      fn sign_representative(&self, mu: &[u8; 64], random: &[u8; 32]) -> Result<$signature, MlDsaError> {
        let mut output = ZeroizingBytes::<$sig>::zeroed();
        portable::sign::<$k, $l>(self.bytes.as_array(), mu, random, $p, output.as_mut_array())?;
        // Only the accepted signature crosses from the secret candidate owner.
        Ok($signature(*output.as_array()))
      }
    }

    impl fmt::Debug for $secret {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(concat!(stringify!($secret), "(****)"))
      }
    }

    /// FIPS 204 key-generation seed ξ: the RFC 9881 recommended private-key form.
    ///
    /// The seed determines the key pair, so protect it like the secret key.
    /// Not `Clone` or `Copy`; zeroized on drop; `Debug` is redacted.
    pub struct $seed(ZeroizingBytes<32>);

    impl $seed {
      /// Seed length in bytes.
      pub const LENGTH: usize = pkcs8::SEED_LENGTH;

      /// DER length of the RFC 9881 seed-form PKCS #8 private key.
      pub const PKCS8_DER_LENGTH: usize = pkcs8::SEED_HEADER_LENGTH.strict_add(pkcs8::SEED_LENGTH);

      const PKCS8_HEADER: [u8; pkcs8::SEED_HEADER_LENGTH] = pkcs8::seed_header($arc);

      /// Wrap a seed. The caller retains responsibility for clearing `bytes`.
      #[must_use]
      pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(ZeroizingBytes::new(bytes))
      }

      /// Generate a seed using 32 bytes from `fill_random`.
      /// The callback must fill the entire buffer with fresh cryptographic randomness.
      /// Failure clears even a partially filled buffer and returns no seed.
      pub fn generate(mut fill_random: impl FnMut(&mut [u8]) -> Result<(), MlDsaError>) -> Result<Self, MlDsaError> {
        let mut seed = Self(ZeroizingBytes::zeroed());
        fill_random(seed.0.as_mut_array())?;
        Ok(seed)
      }

      /// Generate a seed using OS entropy.
      #[cfg(feature = "getrandom")]
      pub fn try_generate() -> Result<Self, MlDsaError> {
        Self::generate(|out| getrandom::fill(out).map_err(|_| MlDsaError::RandomGenerationFailed))
      }

      #[doc = concat!("Expand the seed into its key pair, as by [`", stringify!($profile), "::keypair_from_seed`].")]
      pub fn keypair(&self) -> Result<($public, $secret), MlDsaError> {
        $profile::keypair_from_seed(self.0.as_array())
      }

      #[doc = concat!("Like [`Self::keypair`], with the secret key in memory from `alloc`, as by [`", stringify!($profile), "::keypair_from_seed_in`].")]
      #[cfg(feature = "alloc")]
      pub fn keypair_in<A: Allocator>(&self, alloc: A) -> Result<($public, Box<$secret, A>), MlDsaError> {
        $profile::keypair_from_seed_in(self.0.as_array(), alloc)
      }

      /// Explicitly export the seed into a zeroizing owner.
      #[must_use]
      pub fn expose_secret(&self) -> SecretBytes<32> {
        SecretBytes::new(*self.0.as_array())
      }

      /// Import the seed from an RFC 9881 private key in RFC 5958
      /// OneAsymmetricKey (PKCS #8) DER for this parameter set.
      ///
      /// Accepts the seed and both forms, in a version 1 container or a
      /// version 2 container with a public key. When the both form or a public
      /// key is present, the seed is expanded once to check them. The caller
      /// retains responsibility for clearing `der`. No heap allocation.
      ///
      /// # Errors
      ///
      #[doc = concat!("As [`", stringify!($secret), "::from_pkcs8_der`], and [`MlDsaKeyError::UnsupportedEncoding`]")]
      /// for an expanded-only key: expansion cannot recover a discarded seed.
      pub fn from_pkcs8_der(der: &[u8]) -> Result<Self, MlDsaKeyError> {
        let _dit = DataIndependentTiming::enter();
        let decoded = pkcs8::decode::<$sk, $pk>(der, $arc)?;
        let (seed, expanded) = match decoded.private_key {
          pkcs8::PrivateKey::Seed(seed) => (seed, None),
          pkcs8::PrivateKey::Both { seed, expanded } => (seed, Some(expanded)),
          pkcs8::PrivateKey::Expanded(_) => return Err(MlDsaKeyError::UnsupportedEncoding),
        };
        let owner = Self(ZeroizingBytes::new(*seed));
        if expanded.is_some() || decoded.public_key.is_some() {
          $secret::from_seed(owner.0.as_array())?.check_redundant(expanded, decoded.public_key)?;
        }
        Ok(owner)
      }

      /// Write the seed as an RFC 9881 seed-form private key in version 1
      /// OneAsymmetricKey (PKCS #8) DER. No heap allocation.
      ///
      /// `out` then holds the seed, and the caller owns its cleanup.
      pub fn to_pkcs8_der_into(&self, out: &mut [u8; Self::PKCS8_DER_LENGTH]) {
        pkcs8::write(&Self::PKCS8_HEADER, self.0.as_array(), out);
      }
    }

    impl fmt::Debug for $seed {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(concat!(stringify!($seed), "(****)"))
      }
    }

    /// Caller-owned storage for a prepared signing key: transformed secret
    /// polynomials and the expanded public matrix.
    ///
    /// Create it where it should live (stack, static, or heap), then pass it to
    /// the secret key's `prepare`. The prepared handle clears the secret
    /// polynomials when dropped, and the storage clears them again when it is
    /// dropped. Not `Clone` or `Copy`.
    pub struct $secret_storage {
      state: portable::SigningState<$k, $l>,
      matrix: portable::Matrix<$k, $l>,
    }

    impl $secret_storage {
      /// Empty storage; the size is fixed by the parameter set.
      #[must_use]
      pub const fn new() -> Self {
        Self {
          state: portable::SigningState::zero(),
          matrix: portable::Matrix::zero(),
        }
      }
    }

    impl Default for $secret_storage {
      fn default() -> Self {
        Self::new()
      }
    }

    impl fmt::Debug for $secret_storage {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(concat!(stringify!($secret_storage), "(****)"))
      }
    }

    /// Prepared signing key over caller-owned storage.
    ///
    /// Borrows the secret key and its storage, and clears the transformed
    /// secrets when dropped. Not `Clone` or `Copy`.
    pub struct $prepared_secret<'a> {
      key: &'a $secret,
      storage: &'a mut $secret_storage,
    }

    impl Drop for $prepared_secret<'_> {
      fn drop(&mut self) {
        self.storage.state.clear();
      }
    }

    impl $prepared_secret<'_> {
      signing_methods!($signature);

      fn encoded_secret(&self) -> &[u8; $sk] {
        self.key.bytes.as_array()
      }

      fn sign_representative(&self, mu: &[u8; 64], random: &[u8; 32]) -> Result<$signature, MlDsaError> {
        let mut output = ZeroizingBytes::<$sig>::zeroed();
        portable::sign_with_state(
          self.encoded_secret(),
          mu,
          random,
          $p,
          output.as_mut_array(),
          &self.storage.state,
          Some(&self.storage.matrix),
        )?;
        Ok($signature(*output.as_array()))
      }
    }

    impl fmt::Debug for $prepared_secret<'_> {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(concat!(stringify!($prepared_secret), "(****)"))
      }
    }

    /// Caller-owned storage for a prepared public key: the expanded matrix,
    /// transformed key, and public-key hash. All contents are public.
    pub struct $public_storage {
      state: portable::VerifyingState<$k, $l>,
    }

    impl $public_storage {
      /// Empty storage; the size is fixed by the parameter set.
      #[must_use]
      pub const fn new() -> Self {
        Self {
          state: portable::VerifyingState::zero(),
        }
      }
    }

    impl Default for $public_storage {
      fn default() -> Self {
        Self::new()
      }
    }

    impl fmt::Debug for $public_storage {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(stringify!($public_storage)).finish_non_exhaustive()
      }
    }

    /// Prepared verification key over caller-owned storage.
    ///
    /// Borrows the public key and its storage. Copies share the same storage.
    #[derive(Clone, Copy)]
    pub struct $prepared_public<'a> {
      key: &'a $public,
      state: &'a portable::VerifyingState<$k, $l>,
    }

    impl $prepared_public<'_> {
      verification_methods!($signature);

      fn verify_input(
        &self,
        message: &[u8],
        context: &[u8],
        prehash: Option<MlDsaPrehash<'_>>,
        signature: &$signature,
      ) -> Result<(), VerificationError> {
        let mut mu = ZeroizingBytes::zeroed();
        representative(&self.state.tr, message, context, prehash, mu.as_mut_array())
          .map_err(|_| VerificationError::new())?;
        portable::verify_with_state(&self.key.0, mu.as_array(), &signature.0, $p, Some(self.state))
          .map_err(|_| VerificationError::new())
      }
    }

    impl fmt::Debug for $prepared_public<'_> {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple(stringify!($prepared_public)).field(self.key).finish()
      }
    }

    impl Verifier<$signature> for $prepared_public<'_> {
      fn verify(&self, message: &[u8], signature: &$signature) -> Result<(), VerificationError> {
        self.verify(message, signature)
      }
    }

    /// Canonical encoded ML-DSA signature with a valid response norm.
    #[derive(Clone, Eq, PartialEq)]
    pub struct $signature([u8; $sig]);

    impl $signature {
      /// Encoded signature length.
      pub const LENGTH: usize = $sig;

      /// Parse and check length, response norm, ordered hints, and zero padding.
      /// This is structural validation; authenticate with the public key's verifier.
      pub fn try_from_slice(bytes: &[u8]) -> Result<Self, MlDsaError> {
        if !encoding::valid_signature(bytes, $p) {
          return Err(MlDsaError::InvalidSignature);
        }
        let array = bytes.try_into().map_err(|_| MlDsaError::InvalidSignature)?;
        Ok(Self(array))
      }

      /// Borrow the canonical encoding.
      #[must_use]
      pub const fn as_bytes(&self) -> &[u8; $sig] {
        &self.0
      }

      /// Copy the public signature encoding.
      #[must_use]
      pub const fn to_bytes(&self) -> [u8; $sig] {
        self.0
      }
    }

    impl AsRef<[u8]> for $signature {
      fn as_ref(&self) -> &[u8] {
        &self.0
      }
    }
    impl fmt::Debug for $signature {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple(stringify!($signature)).field(&self.0.as_slice()).finish()
      }
    }
  };
}

parameter_set!(
  MlDsa44,
  MlDsa44PublicKey,
  MlDsa44SecretKey,
  MlDsa44Seed,
  MlDsa44Signature,
  MlDsa44PreparedSecretKey,
  MlDsa44PreparedPublicKey,
  MlDsa44PreparedSecretKeyStorage,
  MlDsa44PreparedPublicKeyStorage,
  P44,
  4,
  4,
  1312,
  2560,
  2420,
  17
);
parameter_set!(
  MlDsa65,
  MlDsa65PublicKey,
  MlDsa65SecretKey,
  MlDsa65Seed,
  MlDsa65Signature,
  MlDsa65PreparedSecretKey,
  MlDsa65PreparedPublicKey,
  MlDsa65PreparedSecretKeyStorage,
  MlDsa65PreparedPublicKeyStorage,
  P65,
  6,
  5,
  1952,
  4032,
  3309,
  18
);
parameter_set!(
  MlDsa87,
  MlDsa87PublicKey,
  MlDsa87SecretKey,
  MlDsa87Seed,
  MlDsa87Signature,
  MlDsa87PreparedSecretKey,
  MlDsa87PreparedPublicKey,
  MlDsa87PreparedSecretKeyStorage,
  MlDsa87PreparedPublicKeyStorage,
  P87,
  8,
  7,
  2592,
  4896,
  4627,
  19
);

#[cfg(all(rscrypto_internal, feature = "diag"))]
mod diagnostics;
#[cfg(all(rscrypto_internal, feature = "diag"))]
pub use diagnostics::{
  diag_mldsa_accumulate, diag_mldsa_mask, diag_mldsa_montgomery, diag_mldsa_montgomery_batch, diag_mldsa_norm,
  diag_mldsa_ntt, diag_mldsa_prepare44, diag_mldsa_prepare65, diag_mldsa_prepare87, diag_mldsa_product,
  diag_mldsa_rounding,
};

/// Diagnostic execution of the production inverse NTT on canonical coefficients.
///
/// Available only to internal evidence builds. Inputs must be below 8,380,417.
#[cfg(all(rscrypto_internal, feature = "diag"))]
#[must_use]
pub fn diag_mldsa_inverse_ntt(input: &[u32; 256]) -> u32 {
  let _dit = crate::traits::ct::DataIndependentTiming::enter();
  let mut value = poly::Poly::zero();
  value.0.copy_from_slice(input);
  value.inverse_ntt();
  core::hint::black_box(&value.0).iter().fold(0, |digest, x| digest ^ x)
}

/// Diagnostic execution of the production scalar inverse NTT, bypassing dispatch.
///
/// Available only to internal evidence builds. Inputs must be below 8,380,417.
#[cfg(all(rscrypto_internal, feature = "diag"))]
#[must_use]
pub fn diag_mldsa_inverse_ntt_portable(input: &[u32; 256]) -> u32 {
  let _dit = crate::traits::ct::DataIndependentTiming::enter();
  let mut value = poly::Poly::zero();
  value.0.copy_from_slice(input);
  value.inverse_ntt_scalar::<false>();
  core::hint::black_box(&value.0).iter().fold(0, |digest, x| digest ^ x)
}

/// Diagnostic execution of the production secret-noise sampler.
///
/// Available only to internal evidence builds. `eta` must be two or four.
#[cfg(all(rscrypto_internal, feature = "diag"))]
#[must_use]
pub fn diag_mldsa_noise(seed: &[u8; 64], eta: u32) -> (bool, u32) {
  let _dit = crate::traits::ct::DataIndependentTiming::enter();
  let mut value = poly::Poly::zero();
  let valid = sampling::noise(seed, 0, eta, &mut value).is_ok();
  (
    valid,
    core::hint::black_box(&value.0).iter().fold(0, |digest, x| digest ^ x),
  )
}

/// Diagnostic execution of the production unpublished-challenge sampler.
///
/// Available only to internal evidence builds. Use standard seed lengths and tau.
#[cfg(all(rscrypto_internal, feature = "diag"))]
#[must_use]
pub fn diag_mldsa_challenge(seed: &[u8], tau: usize) -> (bool, u32) {
  let _dit = crate::traits::ct::DataIndependentTiming::enter();
  let mut value = poly::Poly::zero();
  let valid = sampling::challenge(seed, tau, &mut value).is_ok();
  (
    valid,
    core::hint::black_box(&value.0).iter().fold(0, |digest, x| digest ^ x),
  )
}
