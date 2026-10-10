//! SLH-DSA stateless hash-based signatures (FIPS 205), with an original
//! portable implementation of all 12 parameter sets.
//!
//! Each set has a pure profile, such as [`SlhDsaSha2_128s`], and an RFC 9909
//! HashSLH-DSA profile, such as [`HashSlhDsaSha2_128sWithSha256`], that signs
//! a digest of the message under that RFC's fixed hash pairing. Pure and
//! HashSLH-DSA keys are distinct types: a key of one never signs or verifies
//! the other, as RFC 9909 Section 8 requires.
//!
//! Messages and contexts are byte strings; a context is at most 255 bytes,
//! and the standard protocols use the empty context. Hedged signing takes
//! n bytes of caller-provided cryptographic randomness; deterministic signing
//! is a separate operation. FIPS 205 Section 9.1 recommends using each key
//! for only one of the two variants. Public keys have RFC 9909
//! SubjectPublicKeyInfo import and export, and secret keys RFC 9909 PKCS #8.
//!
//! Signatures are written into a caller-provided buffer of
//! `SIGNATURE_LENGTH` bytes, 7,856 to 49,856 depending on the set; verification
//! takes the signature as a byte slice and rejects any other length. Key
//! generation, signing, and verification allocate nothing and need no OS
//! entropy. Their work is fixed by the parameter set, apart from hashing the
//! message. The small (`s`) sets compute larger trees, so they sign much more
//! slowly than the fast (`f`) sets. Importing a secret key costs one key
//! generation, because it recomputes the public root as FIPS 205 Section 3.1
//! specifies.
//!
//! Target qualification is ongoing; no constant-time or stack-bound claim is
//! made yet.
//!
//! ```
//! use rscrypto::{SlhDsaError, SlhDsaSha2_128f};
//! // A fixed callback is suitable for reproducible examples, not production keys.
//! let (public, secret) = SlhDsaSha2_128f::generate_keypair(|seed| {
//!   seed.fill(7);
//!   Ok(())
//! })?;
//! let mut signature = [0; SlhDsaSha2_128f::SIGNATURE_LENGTH];
//! secret.sign_deterministic(b"release manifest", b"", &mut signature)?;
//! public.verify(b"release manifest", &signature)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod address;
mod hash;
mod params;
mod scheme;
#[cfg(test)]
mod tests;

use self::hash::Suite;
use self::params::Params;
use crate::backend::{
  der::MalformedDer,
  pkix::{self, KeyError},
};
use crate::hashes::crypto::{Sha256, Sha512, Shake128, Shake256};
use crate::secret::ZeroizingBytes;
use crate::traits::ct::DataIndependentTiming;
use crate::{SecretBytes, VerificationError, Verifier};
use core::fmt;

/// SLH-DSA key generation or signing failure.
///
/// Key import uses [`SlhDsaKeyError`]; verification uses opaque
/// [`VerificationError`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SlhDsaError {
  /// The context exceeds the FIPS 205 limit of 255 bytes.
  ContextTooLong,
  /// The random source failed, including after partially filling its buffer.
  RandomGenerationFailed,
}

impl fmt::Display for SlhDsaError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(match self {
      Self::ContextTooLong => "SLH-DSA context exceeds 255 bytes",
      Self::RandomGenerationFailed => "SLH-DSA random generation failed",
    })
  }
}

impl core::error::Error for SlhDsaError {}

/// SLH-DSA key import failure.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum SlhDsaKeyError {
  /// DER input was malformed or non-canonical.
  MalformedDer,
  /// The algorithm identifier names another algorithm, SLH-DSA parameter
  /// set, or pure or HashSLH-DSA profile.
  UnsupportedAlgorithm,
  /// The DER is a well-formed key of this algorithm in a form this import
  /// does not accept: PKCS #8 attributes.
  UnsupportedEncoding,
  /// The public-key encoding has the wrong length.
  InvalidPublicKey,
  /// The secret key has the wrong length, its public root does not match
  /// its seeds, or a PKCS #8 public key is not its own.
  InvalidSecretKey,
}

impl fmt::Display for SlhDsaKeyError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(match self {
      Self::MalformedDer => "malformed SLH-DSA DER",
      Self::UnsupportedAlgorithm => "unsupported SLH-DSA key algorithm",
      Self::UnsupportedEncoding => "unsupported SLH-DSA key encoding",
      Self::InvalidPublicKey => "invalid SLH-DSA public key",
      Self::InvalidSecretKey => "invalid SLH-DSA secret key",
    })
  }
}

impl core::error::Error for SlhDsaKeyError {}

impl MalformedDer for SlhDsaKeyError {
  const MALFORMED_DER: Self = Self::MalformedDer;
}

impl KeyError for SlhDsaKeyError {
  const UNSUPPORTED_ALGORITHM: Self = Self::UnsupportedAlgorithm;
  const UNSUPPORTED_ENCODING: Self = Self::UnsupportedEncoding;
  const INVALID_PUBLIC_KEY: Self = Self::InvalidPublicKey;
  const INVALID_SECRET_KEY: Self = Self::InvalidSecretKey;
}

/// Domain separator of M' for pure signing (FIPS 205 Algorithm 22).
const PURE_SEPARATOR: u8 = 0;
/// Domain separator of M' for HashSLH-DSA (FIPS 205 Algorithm 23).
const PREHASH_SEPARATOR: u8 = 1;

/// An RFC 9909 HashSLH-DSA pre-hash function.
#[derive(Clone, Copy)]
enum Prehash {
  Sha256,
  Sha512,
  /// SHAKE128 with a 256-bit output.
  Shake128,
  /// SHAKE256 with a 512-bit output.
  Shake256,
}

impl Prehash {
  /// DER encoding of the function's OID, with tag and length (FIPS 205
  /// Algorithm 23, lines 10-19).
  const fn oid(self) -> [u8; 11] {
    let arc = match self {
      Self::Sha256 => 0x01,
      Self::Sha512 => 0x03,
      Self::Shake128 => 0x0b,
      Self::Shake256 => 0x0c,
    };
    [0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, arc]
  }

  /// PH_M of `message` into `out`, whose length is the function's output.
  fn digest(self, message: &[u8], out: &mut [u8]) {
    match self {
      Self::Sha256 => out.copy_from_slice(&Sha256::digest(message)),
      Self::Sha512 => out.copy_from_slice(&Sha512::digest(message)),
      Self::Shake128 => Shake128::hash_into(message, out),
      Self::Shake256 => Shake256::hash_into(message, out),
    }
  }
}

/// The prefix of M', `toByte(separator, 1) || toByte(|ctx|, 1)`, or `None`
/// for a context longer than 255 bytes.
fn domain(separator: u8, context: &[u8]) -> Option<[u8; 2]> {
  u8::try_from(context.len()).ok().map(|len| [separator, len])
}

/// The two n-byte halves of a public key, PK.seed and PK.root.
fn halves<const N: usize>(public: &[u8]) -> (&[u8; N], &[u8; N]) {
  let (chunks, _) = public.as_chunks::<N>();
  (&chunks[0], &chunks[1])
}

/// Sign M' = `separator || |context| || context || body` (FIPS 205 Algorithms
/// 22 and 23) into `signature`, with `opt_rand` filled by `randomize`.
///
/// The context is checked before `randomize` runs. Any failure zero-fills
/// `signature`; none can occur once signing starts.
fn sign_message<S: Suite<N>, const N: usize>(
  p: &Params,
  key: &scheme::PrivateKey<'_, N>,
  separator: u8,
  context: &[u8],
  body: [&[u8]; 2],
  randomize: impl FnOnce(&mut [u8; N]) -> Result<(), SlhDsaError>,
  signature: &mut [u8],
) -> Result<(), SlhDsaError> {
  let _dit = DataIndependentTiming::enter();
  let result = domain(separator, context)
    .ok_or(SlhDsaError::ContextTooLong)
    .and_then(|prefix| {
      let mut opt_rand = ZeroizingBytes::<N>::zeroed();
      randomize(opt_rand.as_mut_array())?;
      let (nodes, _) = signature.as_chunks_mut::<N>();
      scheme::sign::<S, N>(
        p,
        key,
        &[&prefix, context, body[0], body[1]],
        opt_rand.as_array(),
        nodes,
      );
      Ok(())
    });
  if result.is_err() {
    signature.fill(0);
  }
  result
}

/// Verify a signature on M' (FIPS 205 Algorithms 24 and 25).
fn verify_message<S: Suite<N>, const N: usize>(
  p: &Params,
  public: &[u8],
  separator: u8,
  context: &[u8],
  body: [&[u8]; 2],
  signature: &[u8],
) -> Result<(), VerificationError> {
  let Some(prefix) = domain(separator, context) else {
    return Err(VerificationError::new());
  };
  let (pk_seed, pk_root) = halves::<N>(public);
  if scheme::verify::<S, N>(p, pk_seed, pk_root, &[&prefix, context, body[0], body[1]], signature) {
    Ok(())
  } else {
    Err(VerificationError::new())
  }
}

#[cfg(feature = "getrandom")]
fn os_random(out: &mut [u8]) -> Result<(), SlhDsaError> {
  getrandom::fill(out).map_err(|_| SlhDsaError::RandomGenerationFailed)
}

/// Signing methods of a pure secret key.
macro_rules! pure_signing {
  ($suite:ty, $p:path, $n:literal, $sig:literal) => {
    /// Deterministic pure SLH-DSA of `message` under `context` into
    /// `signature`.
    ///
    /// Uses PK.seed in place of fresh randomness (FIPS 205 Section 9.2). A
    /// context over 255 bytes returns [`SlhDsaError::ContextTooLong`] and
    /// zero-fills `signature`.
    pub fn sign_deterministic(
      &self,
      message: &[u8],
      context: &[u8],
      signature: &mut [u8; $sig],
    ) -> Result<(), SlhDsaError> {
      let pk_seed = self.public.halves().0;
      sign_message::<$suite, $n>(
        &$p,
        &self.parts(),
        PURE_SEPARATOR,
        context,
        [message, &[]],
        |opt_rand| {
          opt_rand.copy_from_slice(pk_seed);
          Ok(())
        },
        signature,
      )
    }

    /// Hedged pure SLH-DSA of `message` under `context` into `signature`,
    /// using n bytes from `fill_random`.
    ///
    /// The context is checked before entropy is requested. Any error,
    /// including one from `fill_random`, zero-fills `signature` and leaves
    /// the key unchanged.
    pub fn sign_with(
      &self,
      message: &[u8],
      context: &[u8],
      mut fill_random: impl FnMut(&mut [u8]) -> Result<(), SlhDsaError>,
      signature: &mut [u8; $sig],
    ) -> Result<(), SlhDsaError> {
      sign_message::<$suite, $n>(
        &$p,
        &self.parts(),
        PURE_SEPARATOR,
        context,
        [message, &[]],
        |opt_rand| fill_random(opt_rand),
        signature,
      )
    }

    /// Hedged pure SLH-DSA using OS entropy, as by [`Self::sign_with`].
    #[cfg(feature = "getrandom")]
    pub fn try_sign(&self, message: &[u8], context: &[u8], signature: &mut [u8; $sig]) -> Result<(), SlhDsaError> {
      self.sign_with(message, context, os_random, signature)
    }
  };
}

/// Verification methods of a pure public key.
macro_rules! pure_verification {
  ($suite:ty, $p:path, $n:literal) => {
    /// Verify pure SLH-DSA with an empty context.
    pub fn verify(&self, message: &[u8], signature: &[u8]) -> Result<(), VerificationError> {
      self.verify_with_context(message, &[], signature)
    }

    /// Verify pure SLH-DSA under `context`. A wrong-length signature, a
    /// context over 255 bytes, and every invalid signature return the same
    /// opaque error.
    pub fn verify_with_context(
      &self,
      message: &[u8],
      context: &[u8],
      signature: &[u8],
    ) -> Result<(), VerificationError> {
      verify_message::<$suite, $n>(&$p, &self.0, PURE_SEPARATOR, context, [message, &[]], signature)
    }
  };
}

/// Signing methods of a HashSLH-DSA secret key.
macro_rules! prehash_signing {
  ($suite:ty, $p:path, $n:literal, $sig:literal, $prehash:ident, $digest:literal) => {
    /// The RFC 9909 pre-hash function of this profile.
    const PREHASH: Prehash = Prehash::$prehash;

    /// Deterministic HashSLH-DSA of `message` under `context` into
    /// `signature`; this computes PH_M of `message`.
    ///
    /// Uses PK.seed in place of fresh randomness (FIPS 205 Section 9.2). A
    /// context over 255 bytes returns [`SlhDsaError::ContextTooLong`] and
    /// zero-fills `signature`.
    pub fn sign_deterministic(
      &self,
      message: &[u8],
      context: &[u8],
      signature: &mut [u8; $sig],
    ) -> Result<(), SlhDsaError> {
      let mut digest = [0; $digest];
      Self::PREHASH.digest(message, &mut digest);
      self.sign_prehash_deterministic(&digest, context, signature)
    }

    /// Hedged HashSLH-DSA of `message` under `context` into `signature`,
    /// using n bytes from `fill_random`; this computes PH_M of `message`.
    ///
    /// The context is checked before entropy is requested. Any error,
    /// including one from `fill_random`, zero-fills `signature` and leaves
    /// the key unchanged.
    pub fn sign_with(
      &self,
      message: &[u8],
      context: &[u8],
      fill_random: impl FnMut(&mut [u8]) -> Result<(), SlhDsaError>,
      signature: &mut [u8; $sig],
    ) -> Result<(), SlhDsaError> {
      let mut digest = [0; $digest];
      Self::PREHASH.digest(message, &mut digest);
      self.sign_prehash_with(&digest, context, fill_random, signature)
    }

    /// Hedged HashSLH-DSA using OS entropy, as by [`Self::sign_with`].
    #[cfg(feature = "getrandom")]
    pub fn try_sign(&self, message: &[u8], context: &[u8], signature: &mut [u8; $sig]) -> Result<(), SlhDsaError> {
      self.sign_with(message, context, os_random, signature)
    }

    /// Deterministic HashSLH-DSA of a caller-computed PH_M.
    ///
    /// `digest` must be this profile's pre-hash of the message; its
    /// provenance is not checked. Failure behaves as by
    /// [`Self::sign_deterministic`].
    pub fn sign_prehash_deterministic(
      &self,
      digest: &[u8; $digest],
      context: &[u8],
      signature: &mut [u8; $sig],
    ) -> Result<(), SlhDsaError> {
      let pk_seed = self.public.halves().0;
      sign_message::<$suite, $n>(
        &$p,
        &self.parts(),
        PREHASH_SEPARATOR,
        context,
        [&Self::PREHASH.oid(), digest],
        |opt_rand| {
          opt_rand.copy_from_slice(pk_seed);
          Ok(())
        },
        signature,
      )
    }

    /// Hedged HashSLH-DSA of a caller-computed PH_M, using n bytes from
    /// `fill_random`. Failure behaves as by [`Self::sign_with`].
    pub fn sign_prehash_with(
      &self,
      digest: &[u8; $digest],
      context: &[u8],
      mut fill_random: impl FnMut(&mut [u8]) -> Result<(), SlhDsaError>,
      signature: &mut [u8; $sig],
    ) -> Result<(), SlhDsaError> {
      sign_message::<$suite, $n>(
        &$p,
        &self.parts(),
        PREHASH_SEPARATOR,
        context,
        [&Self::PREHASH.oid(), digest],
        |opt_rand| fill_random(opt_rand),
        signature,
      )
    }
  };
}

/// Verification methods of a HashSLH-DSA public key.
macro_rules! prehash_verification {
  ($suite:ty, $p:path, $n:literal, $prehash:ident, $digest:literal) => {
    /// The RFC 9909 pre-hash function of this profile.
    const PREHASH: Prehash = Prehash::$prehash;

    /// Verify HashSLH-DSA with an empty context; this computes PH_M of
    /// `message`.
    pub fn verify(&self, message: &[u8], signature: &[u8]) -> Result<(), VerificationError> {
      self.verify_with_context(message, &[], signature)
    }

    /// Verify HashSLH-DSA under `context`; this computes PH_M of `message`.
    /// Every invalid signature or context returns the same opaque error.
    pub fn verify_with_context(
      &self,
      message: &[u8],
      context: &[u8],
      signature: &[u8],
    ) -> Result<(), VerificationError> {
      let mut digest = [0; $digest];
      Self::PREHASH.digest(message, &mut digest);
      self.verify_prehash(&digest, context, signature)
    }

    /// Verify HashSLH-DSA over a caller-computed PH_M.
    pub fn verify_prehash(
      &self,
      digest: &[u8; $digest],
      context: &[u8],
      signature: &[u8],
    ) -> Result<(), VerificationError> {
      verify_message::<$suite, $n>(
        &$p,
        &self.0,
        PREHASH_SEPARATOR,
        context,
        [&Self::PREHASH.oid(), digest],
        signature,
      )
    }
  };
}

/// One profile with its public and secret key types.
macro_rules! profile {
  (
    profile: $profile:ident,
    public: $public:ident,
    secret: $secret:ident,
    name: $name:expr,
    params: $p:path,
    suite: $suite:ty,
    n: $n:literal,
    pk: $pk:literal,
    sk: $sk:literal,
    seeds: $seeds:literal,
    sig: $sig:literal,
    arc: $arc:literal,
    signing: { $($signing:tt)* },
    verification: { $($verification:tt)* },
  ) => {
    #[doc = concat!($name, " profile with typed key-generation outputs.")]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct $profile;

    impl $profile {
      #[doc = concat!("RFC 9909 identifier: 2.16.840.1.101.3.4.3.", stringify!($arc), ".")]
      const ALGORITHM: pkix::Algorithm = pkix::Algorithm { family: 3, arc: $arc };

      /// Signature length in bytes.
      pub const SIGNATURE_LENGTH: usize = $sig;

      /// Generate a key pair from 3n bytes requested once from `fill_random`,
      /// read as SK.seed, SK.prf, and PK.seed (FIPS 205 Algorithm 21).
      ///
      /// The callback must fill the entire buffer with fresh cryptographic
      /// randomness from a generator of at least 8n bits of security
      /// strength. Failure clears even a partially filled buffer and returns
      /// no key material.
      pub fn generate_keypair(
        mut fill_random: impl FnMut(&mut [u8]) -> Result<(), SlhDsaError>,
      ) -> Result<($public, $secret), SlhDsaError> {
        let _dit = DataIndependentTiming::enter();
        let mut seeds = ZeroizingBytes::<$seeds>::zeroed();
        fill_random(seeds.as_mut_array())?;
        let (fields, _) = seeds.as_array().as_chunks::<$n>();
        let mut secret = $secret::zeroed();
        secret.seed.as_mut_array().copy_from_slice(&fields[0]);
        secret.prf.as_mut_array().copy_from_slice(&fields[1]);
        secret.public.0[..$n].copy_from_slice(&fields[2]);
        let mut root = [0; $n];
        secret.root(&mut root);
        secret.public.0[$n..].copy_from_slice(&root);
        Ok((secret.public.clone(), secret))
      }

      /// Generate a key pair using OS entropy.
      #[cfg(feature = "getrandom")]
      pub fn try_generate_keypair() -> Result<($public, $secret), SlhDsaError> {
        Self::generate_keypair(os_random)
      }
    }

    #[doc = concat!($name, " public key: PK.seed followed by PK.root.")]
    #[derive(Clone, Eq, PartialEq)]
    pub struct $public([u8; $pk]);

    impl $public {
      /// Encoded public-key length in bytes.
      pub const LENGTH: usize = $pk;

      /// DER-encoded RFC 9909 SubjectPublicKeyInfo length in bytes.
      pub const SPKI_DER_LENGTH: usize = pkix::spki_header_length($pk).strict_add($pk);

      const SPKI_HEADER: [u8; pkix::spki_header_length($pk)] = pkix::spki_header($profile::ALGORITHM, $pk);

      /// Construct from a fixed-width encoding. All bit patterns are admissible.
      #[must_use]
      pub const fn from_bytes(bytes: [u8; $pk]) -> Self {
        Self(bytes)
      }

      /// Parse a public key, rejecting an incorrect length.
      pub fn try_from_slice(bytes: &[u8]) -> Result<Self, SlhDsaKeyError> {
        let array = bytes.try_into().map_err(|_| SlhDsaKeyError::InvalidPublicKey)?;
        Ok(Self(array))
      }

      /// Parse an RFC 9909 SubjectPublicKeyInfo for this profile.
      ///
      /// Accepts only the unique DER encoding: this profile's algorithm
      /// identifier with absent parameters, a BIT STRING with no unused bits,
      /// the exact public-key length, and no trailing input.
      ///
      /// # Errors
      ///
      /// Returns [`SlhDsaKeyError::UnsupportedAlgorithm`] for a well-formed
      /// key of another algorithm, parameter set, or pure or HashSLH-DSA
      /// profile; [`SlhDsaKeyError::InvalidPublicKey`] for a wrong key length;
      /// and [`SlhDsaKeyError::MalformedDer`] for any other encoding. No heap
      /// allocation.
      pub fn from_spki_der(der: &[u8]) -> Result<Self, SlhDsaKeyError> {
        pkix::decode_spki(der, &Self::SPKI_HEADER).map(|key| Self(*key))
      }

      /// Encode the RFC 9909 SubjectPublicKeyInfo DER. No heap allocation.
      #[must_use]
      pub const fn to_spki_der(&self) -> [u8; Self::SPKI_DER_LENGTH] {
        pkix::concat(&Self::SPKI_HEADER, &self.0)
      }

      /// Borrow the encoding.
      #[must_use]
      pub const fn as_bytes(&self) -> &[u8; $pk] {
        &self.0
      }

      /// Copy the encoding.
      #[must_use]
      pub const fn to_bytes(&self) -> [u8; $pk] {
        self.0
      }

      fn halves(&self) -> (&[u8; $n], &[u8; $n]) {
        halves::<$n>(&self.0)
      }

      $($verification)*
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

    impl Verifier<[u8]> for $public {
      fn verify(&self, message: &[u8], signature: &[u8]) -> Result<(), VerificationError> {
        self.verify(message, signature)
      }
    }

    #[doc = concat!($name, " secret key with its public key.")]
    ///
    /// Not `Clone` or `Copy`; SK.seed and SK.prf are zeroized on drop, and
    /// `Debug` is redacted.
    pub struct $secret {
      seed: ZeroizingBytes<$n>,
      prf: ZeroizingBytes<$n>,
      public: $public,
    }

    impl $secret {
      /// Encoded secret-key length in bytes: SK.seed, SK.prf, PK.seed, and
      /// PK.root.
      pub const LENGTH: usize = $sk;

      /// DER length of this key as a version 1 RFC 9909 PKCS #8 private key.
      pub const PKCS8_DER_LENGTH: usize = pkix::raw_pkcs8_header_length($sk).strict_add($sk);

      const PKCS8_HEADER: [u8; pkix::raw_pkcs8_header_length($sk)] =
        pkix::raw_pkcs8_header($profile::ALGORITHM, $sk);

      /// Import a secret key and check that its PK.root belongs to its SK.seed
      /// and PK.seed (FIPS 205 Section 3.1).
      ///
      /// The check costs one key generation. The caller retains
      /// responsibility for clearing `input`; a rejected key is cleared
      /// before returning.
      pub fn try_from_slice(input: &[u8]) -> Result<Self, SlhDsaKeyError> {
        let _dit = DataIndependentTiming::enter();
        if input.len() != $sk {
          return Err(SlhDsaKeyError::InvalidSecretKey);
        }
        let (fields, _) = input.as_chunks::<$n>();
        let mut secret = Self::zeroed();
        secret.seed.as_mut_array().copy_from_slice(&fields[0]);
        secret.prf.as_mut_array().copy_from_slice(&fields[1]);
        secret.public.0[..$n].copy_from_slice(&fields[2]);
        secret.public.0[$n..].copy_from_slice(&fields[3]);
        let mut root = [0; $n];
        secret.root(&mut root);
        if root != fields[3] {
          return Err(SlhDsaKeyError::InvalidSecretKey);
        }
        Ok(secret)
      }

      /// Import an RFC 9909 private key from RFC 5958 OneAsymmetricKey
      /// (PKCS #8) DER for this profile.
      ///
      /// Accepts a version 1 container, or a version 2 container whose public
      /// key equals the key's own. The key is checked as by
      /// [`Self::try_from_slice`]. The caller retains responsibility for
      /// clearing `der`. No heap allocation.
      ///
      /// # Errors
      ///
      /// Returns [`SlhDsaKeyError::MalformedDer`] for malformed or
      /// non-canonical DER, including a version that disagrees with the
      /// public-key field; [`SlhDsaKeyError::UnsupportedAlgorithm`] for
      /// another algorithm, parameter set, or profile;
      /// [`SlhDsaKeyError::UnsupportedEncoding`] for attributes;
      /// [`SlhDsaKeyError::InvalidSecretKey`] for a wrong-length or
      /// inconsistent key or a foreign public key; and
      /// [`SlhDsaKeyError::InvalidPublicKey`] for a wrong-length public key.
      pub fn from_pkcs8_der(der: &[u8]) -> Result<Self, SlhDsaKeyError> {
        let _dit = DataIndependentTiming::enter();
        let (key, public) = pkix::decode_pkcs8_raw::<SlhDsaKeyError, $sk, $pk>(der, $profile::ALGORITHM)?;
        let secret = Self::try_from_slice(key)?;
        if public.is_some_and(|public| *public != secret.public.0) {
          return Err(SlhDsaKeyError::InvalidSecretKey);
        }
        Ok(secret)
      }

      /// Write this key as a version 1 RFC 9909 PKCS #8 private key. No heap
      /// allocation.
      ///
      /// `out` then holds the secret key, and the caller owns its cleanup.
      pub fn to_pkcs8_der_into(&self, out: &mut [u8; Self::PKCS8_DER_LENGTH]) {
        let (header, key) = out.split_at_mut(Self::PKCS8_HEADER.len());
        header.copy_from_slice(&Self::PKCS8_HEADER);
        self.write_encoding(key);
      }

      /// Borrow the public key.
      #[must_use]
      pub const fn public_key(&self) -> &$public {
        &self.public
      }

      /// Explicitly export the encoded secret key into a zeroizing owner.
      #[must_use]
      pub fn expose_secret(&self) -> SecretBytes<$sk> {
        match SecretBytes::try_fill_with(|out| {
          self.write_encoding(out);
          Ok::<(), core::convert::Infallible>(())
        }) {
          Ok(secret) => secret,
          Err(never) => match never {},
        }
      }

      $($signing)*

      /// Zero-filled owner that construction fills in place; never returned.
      const fn zeroed() -> Self {
        Self {
          seed: ZeroizingBytes::zeroed(),
          prf: ZeroizingBytes::zeroed(),
          public: $public([0; $pk]),
        }
      }

      /// PK.root regenerated from SK.seed and PK.seed (FIPS 205 Algorithm 18).
      fn root(&self, out: &mut [u8; $n]) {
        scheme::keygen::<$suite, $n>(&$p, self.seed.as_array(), self.public.halves().0, out);
      }

      fn parts(&self) -> scheme::PrivateKey<'_, $n> {
        let (pk_seed, pk_root) = self.public.halves();
        scheme::PrivateKey {
          sk_seed: self.seed.as_array(),
          sk_prf: self.prf.as_array(),
          pk_seed,
          pk_root,
        }
      }

      /// Write SK.seed, SK.prf, PK.seed, and PK.root to `out`.
      fn write_encoding(&self, out: &mut [u8]) {
        let (seed, rest) = out.split_at_mut($n);
        let (prf, public) = rest.split_at_mut($n);
        seed.copy_from_slice(self.seed.as_array());
        prf.copy_from_slice(self.prf.as_array());
        public.copy_from_slice(&self.public.0);
      }
    }

    impl fmt::Debug for $secret {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(concat!(stringify!($secret), "(****)"))
      }
    }
  };
}

/// One FIPS 205 parameter set: its pure profile and its RFC 9909 HashSLH-DSA
/// profile.
macro_rules! parameter_set {
  (
    $name:literal,
    pure: ($profile:ident, $public:ident, $secret:ident, $arc:literal),
    prehash: ($hash_profile:ident, $hash_public:ident, $hash_secret:ident, $hash_arc:literal, $hash_name:literal, $prehash:ident, $digest:literal),
    params: $p:path,
    suite: $suite:ty,
    n: $n:literal,
    pk: $pk:literal,
    sk: $sk:literal,
    seeds: $seeds:literal,
    sig: $sig:literal $(,)?
  ) => {
    profile! {
      profile: $profile,
      public: $public,
      secret: $secret,
      name: concat!("Pure ", $name),
      params: $p,
      suite: $suite,
      n: $n,
      pk: $pk,
      sk: $sk,
      seeds: $seeds,
      sig: $sig,
      arc: $arc,
      signing: { pure_signing!($suite, $p, $n, $sig); },
      verification: { pure_verification!($suite, $p, $n); },
    }

    profile! {
      profile: $hash_profile,
      public: $hash_public,
      secret: $hash_secret,
      name: concat!("HashSLH-DSA ", $name, " with ", $hash_name),
      params: $p,
      suite: $suite,
      n: $n,
      pk: $pk,
      sk: $sk,
      seeds: $seeds,
      sig: $sig,
      arc: $hash_arc,
      signing: { prehash_signing!($suite, $p, $n, $sig, $prehash, $digest); },
      verification: { prehash_verification!($suite, $p, $n, $prehash, $digest); },
    }
  };
}

parameter_set!(
  "SLH-DSA-SHA2-128s",
  pure: (SlhDsaSha2_128s, SlhDsaSha2_128sPublicKey, SlhDsaSha2_128sSecretKey, 20),
  prehash: (
    HashSlhDsaSha2_128sWithSha256,
    HashSlhDsaSha2_128sWithSha256PublicKey,
    HashSlhDsaSha2_128sWithSha256SecretKey,
    35,
    "SHA-256",
    Sha256,
    32
  ),
  params: params::P128S,
  suite: hash::Sha2Category1<16>,
  n: 16,
  pk: 32,
  sk: 64,
  seeds: 48,
  sig: 7856,
);
parameter_set!(
  "SLH-DSA-SHA2-128f",
  pure: (SlhDsaSha2_128f, SlhDsaSha2_128fPublicKey, SlhDsaSha2_128fSecretKey, 21),
  prehash: (
    HashSlhDsaSha2_128fWithSha256,
    HashSlhDsaSha2_128fWithSha256PublicKey,
    HashSlhDsaSha2_128fWithSha256SecretKey,
    36,
    "SHA-256",
    Sha256,
    32
  ),
  params: params::P128F,
  suite: hash::Sha2Category1<16>,
  n: 16,
  pk: 32,
  sk: 64,
  seeds: 48,
  sig: 17088,
);
parameter_set!(
  "SLH-DSA-SHA2-192s",
  pure: (SlhDsaSha2_192s, SlhDsaSha2_192sPublicKey, SlhDsaSha2_192sSecretKey, 22),
  prehash: (
    HashSlhDsaSha2_192sWithSha512,
    HashSlhDsaSha2_192sWithSha512PublicKey,
    HashSlhDsaSha2_192sWithSha512SecretKey,
    37,
    "SHA-512",
    Sha512,
    64
  ),
  params: params::P192S,
  suite: hash::Sha2Category3And5<24>,
  n: 24,
  pk: 48,
  sk: 96,
  seeds: 72,
  sig: 16224,
);
parameter_set!(
  "SLH-DSA-SHA2-192f",
  pure: (SlhDsaSha2_192f, SlhDsaSha2_192fPublicKey, SlhDsaSha2_192fSecretKey, 23),
  prehash: (
    HashSlhDsaSha2_192fWithSha512,
    HashSlhDsaSha2_192fWithSha512PublicKey,
    HashSlhDsaSha2_192fWithSha512SecretKey,
    38,
    "SHA-512",
    Sha512,
    64
  ),
  params: params::P192F,
  suite: hash::Sha2Category3And5<24>,
  n: 24,
  pk: 48,
  sk: 96,
  seeds: 72,
  sig: 35664,
);
parameter_set!(
  "SLH-DSA-SHA2-256s",
  pure: (SlhDsaSha2_256s, SlhDsaSha2_256sPublicKey, SlhDsaSha2_256sSecretKey, 24),
  prehash: (
    HashSlhDsaSha2_256sWithSha512,
    HashSlhDsaSha2_256sWithSha512PublicKey,
    HashSlhDsaSha2_256sWithSha512SecretKey,
    39,
    "SHA-512",
    Sha512,
    64
  ),
  params: params::P256S,
  suite: hash::Sha2Category3And5<32>,
  n: 32,
  pk: 64,
  sk: 128,
  seeds: 96,
  sig: 29792,
);
parameter_set!(
  "SLH-DSA-SHA2-256f",
  pure: (SlhDsaSha2_256f, SlhDsaSha2_256fPublicKey, SlhDsaSha2_256fSecretKey, 25),
  prehash: (
    HashSlhDsaSha2_256fWithSha512,
    HashSlhDsaSha2_256fWithSha512PublicKey,
    HashSlhDsaSha2_256fWithSha512SecretKey,
    40,
    "SHA-512",
    Sha512,
    64
  ),
  params: params::P256F,
  suite: hash::Sha2Category3And5<32>,
  n: 32,
  pk: 64,
  sk: 128,
  seeds: 96,
  sig: 49856,
);
parameter_set!(
  "SLH-DSA-SHAKE-128s",
  pure: (SlhDsaShake128s, SlhDsaShake128sPublicKey, SlhDsaShake128sSecretKey, 26),
  prehash: (
    HashSlhDsaShake128sWithShake128,
    HashSlhDsaShake128sWithShake128PublicKey,
    HashSlhDsaShake128sWithShake128SecretKey,
    41,
    "SHAKE128",
    Shake128,
    32
  ),
  params: params::P128S,
  suite: hash::Shake<16>,
  n: 16,
  pk: 32,
  sk: 64,
  seeds: 48,
  sig: 7856,
);
parameter_set!(
  "SLH-DSA-SHAKE-128f",
  pure: (SlhDsaShake128f, SlhDsaShake128fPublicKey, SlhDsaShake128fSecretKey, 27),
  prehash: (
    HashSlhDsaShake128fWithShake128,
    HashSlhDsaShake128fWithShake128PublicKey,
    HashSlhDsaShake128fWithShake128SecretKey,
    42,
    "SHAKE128",
    Shake128,
    32
  ),
  params: params::P128F,
  suite: hash::Shake<16>,
  n: 16,
  pk: 32,
  sk: 64,
  seeds: 48,
  sig: 17088,
);
parameter_set!(
  "SLH-DSA-SHAKE-192s",
  pure: (SlhDsaShake192s, SlhDsaShake192sPublicKey, SlhDsaShake192sSecretKey, 28),
  prehash: (
    HashSlhDsaShake192sWithShake256,
    HashSlhDsaShake192sWithShake256PublicKey,
    HashSlhDsaShake192sWithShake256SecretKey,
    43,
    "SHAKE256",
    Shake256,
    64
  ),
  params: params::P192S,
  suite: hash::Shake<24>,
  n: 24,
  pk: 48,
  sk: 96,
  seeds: 72,
  sig: 16224,
);
parameter_set!(
  "SLH-DSA-SHAKE-192f",
  pure: (SlhDsaShake192f, SlhDsaShake192fPublicKey, SlhDsaShake192fSecretKey, 29),
  prehash: (
    HashSlhDsaShake192fWithShake256,
    HashSlhDsaShake192fWithShake256PublicKey,
    HashSlhDsaShake192fWithShake256SecretKey,
    44,
    "SHAKE256",
    Shake256,
    64
  ),
  params: params::P192F,
  suite: hash::Shake<24>,
  n: 24,
  pk: 48,
  sk: 96,
  seeds: 72,
  sig: 35664,
);
parameter_set!(
  "SLH-DSA-SHAKE-256s",
  pure: (SlhDsaShake256s, SlhDsaShake256sPublicKey, SlhDsaShake256sSecretKey, 30),
  prehash: (
    HashSlhDsaShake256sWithShake256,
    HashSlhDsaShake256sWithShake256PublicKey,
    HashSlhDsaShake256sWithShake256SecretKey,
    45,
    "SHAKE256",
    Shake256,
    64
  ),
  params: params::P256S,
  suite: hash::Shake<32>,
  n: 32,
  pk: 64,
  sk: 128,
  seeds: 96,
  sig: 29792,
);
parameter_set!(
  "SLH-DSA-SHAKE-256f",
  pure: (SlhDsaShake256f, SlhDsaShake256fPublicKey, SlhDsaShake256fSecretKey, 31),
  prehash: (
    HashSlhDsaShake256fWithShake256,
    HashSlhDsaShake256fWithShake256PublicKey,
    HashSlhDsaShake256fWithShake256SecretKey,
    46,
    "SHAKE256",
    Shake256,
    64
  ),
  params: params::P256F,
  suite: hash::Shake<32>,
  n: 32,
  pk: 64,
  sk: 128,
  seeds: 96,
  sig: 49856,
);
