//! ML-DSA signatures (FIPS 204), with original portable arithmetic.
//!
//! All messages and contexts use byte strings. A context is at most 255 bytes.
//! Hedged signing takes caller-provided cryptographic randomness; deterministic
//! signing is an explicit, separate operation. The `ml-dsa` leaf needs neither
//! allocation nor OS entropy. See [`MlDsaPrehash`] for HashML-DSA.
//!
//! Target qualification is ongoing; no whole-operation constant-time claim
//! is made. Signing needs tens of KiB of stack. Prepared owners retain up to
//! 79 KiB of polynomial storage, and construction uses additional stack temporaries.
//! Core-only availability does not establish suitability for a constrained stack.
//!
//! ```
//! use rscrypto::{MlDsa44, MlDsaError};
//! // Fixed seeds are suitable for reproducible examples, not production keys.
//! let (public, secret) = MlDsa44::keypair_from_seed(&[7; 32])?;
//! let signature = secret.sign_deterministic(b"release manifest", b"example")?;
//! public.verify_with_context(b"release manifest", b"example", &signature)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod encoding;
mod poly;
mod portable;
mod sampling;
#[cfg(feature = "serde")]
mod serde_impl;
#[cfg(test)]
mod tests;

use crate::secret::ZeroizingBytes;
use crate::{SecretBytes, VerificationError, Verifier};
use core::fmt;

/// ML-DSA construction or signing failure. Verification uses opaque [`VerificationError`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlDsaError {
  /// The public-key encoding has the wrong length.
  InvalidPublicKey,
  /// The secret-key encoding is malformed or its redundant fields disagree.
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
      Self::InvalidPublicKey => "invalid ML-DSA public key",
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
  ($profile:ident, $public:ident, $secret:ident, $signature:ident, $prepared_secret:ident, $prepared_public:ident, $p:ident, $k:literal, $l:literal, $pk:literal, $sk:literal, $sig:literal) => {
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
      pub fn try_from_slice(bytes: &[u8]) -> Result<Self, MlDsaError> {
        let array = bytes.try_into().map_err(|_| MlDsaError::InvalidPublicKey)?;
        Ok(Self(array))
      }

      /// Prepare the matrix and transformed public key for repeated verification.
      /// This explicit owner uses more memory; ordinary verification expands rows on demand.
      pub fn prepare(&self) -> Result<$prepared_public<'_>, MlDsaError> {
        Ok($prepared_public {
          key: self,
          state: portable::VerifyingState::prepare(&self.0)?,
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
      pub fn try_from_slice(input: &[u8]) -> Result<Self, MlDsaError> {
        if input.len() != $sk {
          return Err(MlDsaError::InvalidSecretKey);
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

      /// Prepare secret polynomials and the public matrix for repeated signing.
      /// The returned owner borrows this key, owns and clears transformed secrets,
      /// and performs no heap allocation. See the module's resource contract.
      pub fn prepare(&self) -> Result<$prepared_secret<'_>, MlDsaError> {
        let mut prepared = $prepared_secret {
          key: self,
          state: portable::SigningState::zero(),
          matrix: portable::Matrix::zero(),
        };
        prepared.state.decode(self.bytes.as_array(), $p)?;
        prepared.matrix.expand_into(&self.bytes.as_array()[..32])?;
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

    /// Reusable signing state with an explicit borrowed key lifetime.
    /// Transformed secrets are zeroized on drop. Not `Clone` or `Copy`.
    pub struct $prepared_secret<'a> {
      key: &'a $secret,
      state: portable::SigningState<$k, $l>,
      matrix: portable::Matrix<$k, $l>,
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
          &self.state,
          Some(&self.matrix),
        )?;
        Ok($signature(*output.as_array()))
      }
    }

    impl fmt::Debug for $prepared_secret<'_> {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(concat!(stringify!($prepared_secret), "(****)"))
      }
    }

    /// Reusable public matrix and transformed key for verification.
    pub struct $prepared_public<'a> {
      key: &'a $public,
      state: portable::VerifyingState<$k, $l>,
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
        portable::verify_with_state(&self.key.0, mu.as_array(), &signature.0, $p, Some(&self.state))
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
  MlDsa44Signature,
  MlDsa44PreparedSecretKey,
  MlDsa44PreparedPublicKey,
  P44,
  4,
  4,
  1312,
  2560,
  2420
);
parameter_set!(
  MlDsa65,
  MlDsa65PublicKey,
  MlDsa65SecretKey,
  MlDsa65Signature,
  MlDsa65PreparedSecretKey,
  MlDsa65PreparedPublicKey,
  P65,
  6,
  5,
  1952,
  4032,
  3309
);
parameter_set!(
  MlDsa87,
  MlDsa87PublicKey,
  MlDsa87SecretKey,
  MlDsa87Signature,
  MlDsa87PreparedSecretKey,
  MlDsa87PreparedPublicKey,
  P87,
  8,
  7,
  2592,
  4896,
  4627
);

/// Diagnostic execution of the production inverse NTT on canonical coefficients.
///
/// Available only to internal evidence builds. Inputs must be below 8,380,417.
#[cfg(all(rscrypto_internal, feature = "diag"))]
#[must_use]
pub fn diag_mldsa_inverse_ntt(input: &[u32; 256]) -> u32 {
  let mut value = poly::Poly::zero();
  value.0.copy_from_slice(input);
  value.inverse_ntt();
  core::hint::black_box(&value.0).iter().fold(0, |digest, x| digest ^ x)
}

/// Diagnostic execution of the production secret-noise sampler.
///
/// Available only to internal evidence builds. `eta` must be two or four.
#[cfg(all(rscrypto_internal, feature = "diag"))]
#[must_use]
pub fn diag_mldsa_noise(seed: &[u8; 64], eta: u32) -> (bool, u32) {
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
  let mut value = poly::Poly::zero();
  let valid = sampling::challenge(seed, tau, &mut value).is_ok();
  (
    valid,
    core::hint::black_box(&value.0).iter().fold(0, |digest, x| digest ^ x),
  )
}
