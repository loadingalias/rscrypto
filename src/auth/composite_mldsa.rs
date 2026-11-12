//! Composite ML-DSA signatures: ML-DSA-65 with Ed25519 or ECDSA P-256.
//!
//! This module implements two algorithms of
//! [`draft-ietf-lamps-pq-composite-sigs-19`](https://datatracker.ietf.org/doc/draft-ietf-lamps-pq-composite-sigs/19/),
//! which the RFC Editor is publishing as RFC 10073:
//!
//! | Type | Algorithm | OID | Feature |
//! | --- | --- | --- | --- |
//! | [`MlDsa65Ed25519`] | `id-MLDSA65-Ed25519-SHA512` | 1.3.6.1.5.5.7.6.48 | `mldsa65-ed25519` |
//! | [`MlDsa65EcdsaP256`] | `id-MLDSA65-ECDSA-P256-SHA512` | 1.3.6.1.5.5.7.6.45 | `mldsa65-ecdsa-p256` |
//!
//! A composite signs the message representative
//! `M' = Prefix || Label || len(ctx) || ctx || SHA-512(M)` with pure ML-DSA-65,
//! whose FIPS 204 context is the algorithm's label, and with the traditional
//! algorithm: Ed25519, or ECDSA P-256 with SHA-256. A signature is valid only
//! if both component signatures verify. The application context `ctx` is at
//! most 255 bytes; the methods without a context use the empty one.
//!
//! The components are rscrypto's ML-DSA-65, Ed25519, and ECDSA P-256
//! implementations; this module adds only the combiner and the encodings, and
//! no arithmetic. Secret keys keep the cleanup and timing properties of their
//! component owners: no whole-operation constant-time claim is made, because
//! ML-DSA signing has none.
//!
//! Encodings follow section 4 of the draft. A public key is the ML-DSA-65
//! public key followed by the Ed25519 key or the uncompressed P-256 point. A
//! private key is the 32-byte ML-DSA-65 seed followed by the Ed25519 seed or
//! an RFC 5915 `ECPrivateKey` that names the curve and omits the public key.
//! A signature is the ML-DSA-65 signature followed by the Ed25519 signature or
//! a DER `Ecdsa-Sig-Value`. SubjectPublicKeyInfo and PKCS #8 carry these raw
//! encodings with the composite OID and absent parameters.
//!
//! The draft forbids reusing component keys outside the composite, so keys
//! come only from fresh generation or from composite encodings; no
//! constructor takes existing component keys. ECDSA signatures are
//! malleable, so [`MlDsa65EcdsaP256`] is not strongly unforgeable.
//!
//! Signing needs tens of KiB of stack, like ML-DSA-65. With `alloc`,
//! `prepare_in` expands the ML-DSA key once into allocated storage for
//! repeated signing.
//!
//! ```
//! # #[cfg(feature = "mldsa65-ed25519")] {
//! use rscrypto::{MlDsa65Ed25519, MlDsa65Ed25519PublicKey};
//! // A fixed filler keeps the example reproducible; use fresh randomness
//! // for real keys.
//! let fill = |out: &mut [u8]| {
//!   out.fill(7);
//!   Ok(())
//! };
//! let (public, secret) = MlDsa65Ed25519::generate_keypair(fill)?;
//! let signature = secret.sign_deterministic(b"release manifest", b"example")?;
//! public.verify_with_context(b"release manifest", b"example", &signature)?;
//!
//! let spki = public.to_spki_der();
//! assert_eq!(MlDsa65Ed25519PublicKey::from_spki_der(&spki)?, public);
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#[cfg(feature = "alloc")]
use alloc::boxed::Box;
#[cfg(feature = "alloc")]
use core::alloc::Allocator;
use core::fmt;
use core::ops::DerefMut;

use crate::backend::{
  der::MalformedDer,
  pkix::{self, KeyError},
};
use crate::hashes::crypto::Sha512;
use crate::secret::ZeroizingBytes;
use crate::traits::ct::DataIndependentTiming;
use crate::{
  MlDsa65PreparedSecretKey, MlDsa65PreparedSecretKeyStorage, MlDsa65PublicKey, MlDsa65SecretKey, MlDsa65Signature,
  MlDsaError, SecretBytes, VerificationError, Verifier,
};

/// Composite signing or key-generation failure.
///
/// Key import uses [`CompositeMlDsaKeyError`]; verification uses opaque
/// [`VerificationError`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CompositeMlDsaError {
  /// The application context exceeds 255 bytes.
  ContextTooLong,
  /// The random source failed, including after partially filling its buffer.
  RandomGenerationFailed,
  /// A component key generation exhausted its bounded sampling or retry budget.
  KeyGenerationFailed,
  /// Signing or preparation failed in a component; the error does not say which.
  ///
  /// Every component failure is either cryptographically negligible or means
  /// that key memory changed after validation.
  SigningFailed,
  /// The signature has the wrong length or an invalid component encoding.
  InvalidSignature,
}

impl fmt::Display for CompositeMlDsaError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(match self {
      Self::ContextTooLong => "composite ML-DSA context exceeds 255 bytes",
      Self::RandomGenerationFailed => "composite ML-DSA random generation failed",
      Self::KeyGenerationFailed => "composite ML-DSA key generation failed",
      Self::SigningFailed => "composite ML-DSA signing failed",
      Self::InvalidSignature => "invalid composite ML-DSA signature",
    })
  }
}

impl core::error::Error for CompositeMlDsaError {}

/// Composite key import failure.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum CompositeMlDsaKeyError {
  /// DER input was malformed or non-canonical.
  MalformedDer,
  /// The algorithm identifier names another algorithm.
  UnsupportedAlgorithm,
  /// The input is a well-formed key of this algorithm in a form this import
  /// does not accept: PKCS #8 attributes, or an `ECPrivateKey` that omits the
  /// curve or carries a public key.
  UnsupportedEncoding,
  /// The public key has the wrong length or an invalid component.
  InvalidPublicKey,
  /// The private key has the wrong length, an invalid component, or a
  /// redundant public key that is not its own.
  InvalidSecretKey,
  /// Expanding the ML-DSA seed exhausted a FIPS 204 sampling bound.
  RejectionLimit,
}

impl fmt::Display for CompositeMlDsaKeyError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(match self {
      Self::MalformedDer => "malformed composite ML-DSA DER",
      Self::UnsupportedAlgorithm => "unsupported composite ML-DSA key algorithm",
      Self::UnsupportedEncoding => "unsupported composite ML-DSA key encoding",
      Self::InvalidPublicKey => "invalid composite ML-DSA public key",
      Self::InvalidSecretKey => "invalid composite ML-DSA secret key",
      Self::RejectionLimit => "composite ML-DSA key expansion reached its rejection limit",
    })
  }
}

impl core::error::Error for CompositeMlDsaKeyError {}

impl MalformedDer for CompositeMlDsaKeyError {
  const MALFORMED_DER: Self = Self::MalformedDer;
}

impl KeyError for CompositeMlDsaKeyError {
  const UNSUPPORTED_ALGORITHM: Self = Self::UnsupportedAlgorithm;
  const UNSUPPORTED_ENCODING: Self = Self::UnsupportedEncoding;
  const INVALID_PUBLIC_KEY: Self = Self::InvalidPublicKey;
  const INVALID_SECRET_KEY: Self = Self::InvalidSecretKey;
}

/// Section 2.2 prefix: the ASCII bytes of `CompositeAlgorithmSignatures2025`.
const PREFIX: &[u8; 32] = b"CompositeAlgorithmSignatures2025";
/// Byte length of a SHA-512 prehash.
const PREHASH_LENGTH: usize = 64;
/// The longest label of this module's algorithms.
const MAX_LABEL_LENGTH: usize = 33;
/// The longest message representative: the prefix, the longest label, the
/// context length octet, a 255-byte context, and the prehash.
const MAX_REPRESENTATIVE_LENGTH: usize = PREFIX.len() + MAX_LABEL_LENGTH + 1 + 255 + PREHASH_LENGTH;
/// ML-DSA-65 key-generation seed length.
const ML_DSA_SEED_LENGTH: usize = 32;
/// Bytes of one AlgorithmIdentifier element with an 8-byte OID and absent
/// parameters.
const IDENTIFIER_LENGTH: usize = 12;

/// The AlgorithmIdentifier element for OID 1.3.6.1.5.5.7.6.`arc`, with
/// absent parameters.
const fn identifier(arc: u8) -> [u8; IDENTIFIER_LENGTH] {
  [0x30, 0x0a, 0x06, 0x08, 0x2b, 0x06, 0x01, 0x05, 0x05, 0x07, 0x06, arc]
}

/// The message representative `M'` of section 3.2, in a stack buffer. It
/// holds only public data: the label, the context, and the message digest.
struct Representative {
  bytes: [u8; MAX_REPRESENTATIVE_LENGTH],
  len: usize,
}

impl Representative {
  fn new(label: &[u8], context: &[u8], prehash: &[u8; PREHASH_LENGTH]) -> Result<Self, CompositeMlDsaError> {
    debug_assert!(label.len() <= MAX_LABEL_LENGTH);
    let context_len = u8::try_from(context.len()).map_err(|_| CompositeMlDsaError::ContextTooLong)?;
    let mut bytes = [0; MAX_REPRESENTATIVE_LENGTH];
    let mut len = 0usize;
    for part in [&PREFIX[..], label, &[context_len], context, prehash] {
      let end = len.strict_add(part.len());
      bytes[len..end].copy_from_slice(part);
      len = end;
    }
    Ok(Self { bytes, len })
  }

  fn as_slice(&self) -> &[u8] {
    &self.bytes[..self.len]
  }
}

fn prehash(message: &[u8]) -> [u8; PREHASH_LENGTH] {
  Sha512::digest(message)
}

fn seed_expansion_error(error: MlDsaError) -> CompositeMlDsaKeyError {
  debug_assert_eq!(error, MlDsaError::RejectionLimit);
  CompositeMlDsaKeyError::RejectionLimit
}

/// ML-DSA-65 key material in a composite secret key: the seed that the
/// private-key encoding carries and the expanded key that signing uses.
struct MlDsaPart {
  seed: ZeroizingBytes<ML_DSA_SEED_LENGTH>,
  secret: MlDsa65SecretKey,
}

impl MlDsaPart {
  const fn zeroed() -> Self {
    Self {
      seed: ZeroizingBytes::zeroed(),
      secret: MlDsa65SecretKey::zeroed(),
    }
  }

  /// Store `seed` and expand it in place.
  fn import_into(&mut self, seed: &[u8; ML_DSA_SEED_LENGTH]) -> Result<(), CompositeMlDsaKeyError> {
    self.seed.as_mut_array().copy_from_slice(seed);
    self
      .secret
      .expand_seed_into(self.seed.as_array())
      .map_err(seed_expansion_error)
  }

  /// Fill the seed from `fill_random` and expand it in place.
  fn generate_into(
    &mut self,
    fill_random: &mut impl FnMut(&mut [u8]) -> Result<(), CompositeMlDsaError>,
  ) -> Result<(), CompositeMlDsaError> {
    fill_random(self.seed.as_mut_array())?;
    self
      .secret
      .expand_seed_into(self.seed.as_array())
      .map_err(|_| CompositeMlDsaError::KeyGenerationFailed)
  }
}

/// Sign `M'` with ML-DSA-65 and the label as its FIPS 204 context. `random`
/// is the hedging randomness, or `None` for deterministic signing.
fn sign_ml_dsa(
  sign_deterministic: impl FnOnce(&[u8], &[u8]) -> Result<MlDsa65Signature, MlDsaError>,
  sign_with: impl FnOnce(&[u8], &[u8], &[u8; 32]) -> Result<MlDsa65Signature, MlDsaError>,
  representative: &Representative,
  label: &[u8],
  random: Option<&[u8; 32]>,
) -> Result<MlDsa65Signature, CompositeMlDsaError> {
  match random {
    None => sign_deterministic(representative.as_slice(), label),
    Some(random) => sign_with(representative.as_slice(), label, random),
  }
  .map_err(|_| CompositeMlDsaError::SigningFailed)
}

macro_rules! composite {
  (
    profile: $profile:ident,
    public: $public:ident,
    secret: $secret:ident,
    signature: $signature:ident,
    prepared: $prepared:ident,
    name: $name:literal,
    label: $label:literal,
    arc: $arc:literal,
    trad_public_length: $trad_pk:expr,
    trad_secret_length: $trad_sk:expr,
    random_length: $random_len:expr,
  ) => {
    #[doc = concat!("`", $name, "` from draft-ietf-lamps-pq-composite-sigs-19 (RFC-to-be 10073).")]
    ///
    /// See the [module documentation](crate::auth::composite_mldsa) for the construction and encodings.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct $profile;

    impl $profile {
      #[doc = concat!("Dotted object identifier of `", $name, "`.")]
      pub const OID: &'static str = concat!("1.3.6.1.5.5.7.6.", stringify!($arc));

      /// Signature label: the ASCII bytes that bind a signature to this
      /// algorithm, which is also the ML-DSA context.
      pub const LABEL: &'static [u8] = $label;

      const IDENTIFIER: [u8; IDENTIFIER_LENGTH] = identifier($arc);

      /// DER contents of [`Self::OID`].
      const OID_CONTENTS: [u8; 8] = {
        let [_, _, _, _, a, b, c, d, e, f, g, h] = Self::IDENTIFIER;
        [a, b, c, d, e, f, g, h]
      };

      /// Generate a fresh key pair using randomness from `fill_random`.
      ///
      /// Both component keys are generated here, as the draft requires. The
      /// callback must fill each buffer completely with fresh cryptographic
      /// randomness. Its first error is returned unchanged, and no key
      /// material is returned; a partially filled buffer is cleared.
      ///
      /// # Errors
      ///
      /// Returns the callback's error, or
      /// [`CompositeMlDsaError::KeyGenerationFailed`] if a component exhausts
      /// its bounded sampling or retry budget.
      pub fn generate_keypair(
        mut fill_random: impl FnMut(&mut [u8]) -> Result<(), CompositeMlDsaError>,
      ) -> Result<($public, $secret), CompositeMlDsaError> {
        let mut secret = $secret::zeroed();
        secret.generate_into(&mut fill_random)?;
        Ok((secret.public_key(), secret))
      }

      /// Generate a fresh key pair using OS entropy.
      #[cfg(feature = "getrandom")]
      pub fn try_generate_keypair() -> Result<($public, $secret), CompositeMlDsaError> {
        Self::generate_keypair(|out| getrandom::fill(out).map_err(|_| CompositeMlDsaError::RandomGenerationFailed))
      }

      /// Like [`Self::generate_keypair`], with the secret key in memory from `alloc`.
      ///
      /// Both component keys are generated directly into the allocation, so
      /// moving the box moves only a pointer and no by-value copy of the key
      /// is left behind. The box clears the key on drop; failure clears it
      /// before returning. Allocation failure is handled as by [`Box::new_in`].
      #[cfg(feature = "alloc")]
      pub fn generate_keypair_in<A: Allocator>(
        mut fill_random: impl FnMut(&mut [u8]) -> Result<(), CompositeMlDsaError>,
        alloc: A,
      ) -> Result<($public, Box<$secret, A>), CompositeMlDsaError> {
        let mut secret = Box::new_in($secret::zeroed(), alloc);
        secret.generate_into(&mut fill_random)?;
        Ok((secret.public_key(), secret))
      }

      /// Like [`Self::try_generate_keypair`], with the secret key in memory from `alloc`.
      #[cfg(all(feature = "alloc", feature = "getrandom"))]
      pub fn try_generate_keypair_in<A: Allocator>(
        alloc: A,
      ) -> Result<($public, Box<$secret, A>), CompositeMlDsaError> {
        Self::generate_keypair_in(
          |out| getrandom::fill(out).map_err(|_| CompositeMlDsaError::RandomGenerationFailed),
          alloc,
        )
      }
    }

    #[doc = concat!("`", $name, "` public key: the ML-DSA-65 public key and the traditional public key.")]
    #[derive(Clone, Eq, PartialEq)]
    pub struct $public {
      ml_dsa: MlDsa65PublicKey,
      trad: [u8; $trad_pk],
    }

    impl $public {
      /// Encoded public-key length in bytes.
      pub const LENGTH: usize = MlDsa65PublicKey::LENGTH + $trad_pk;

      /// DER-encoded SubjectPublicKeyInfo length in bytes.
      pub const SPKI_DER_LENGTH: usize =
        pkix::spki_header_length_for(IDENTIFIER_LENGTH, Self::LENGTH).strict_add(Self::LENGTH);

      const SPKI_HEADER: [u8; pkix::spki_header_length_for(IDENTIFIER_LENGTH, Self::LENGTH)] =
        pkix::spki_header_for(&$profile::IDENTIFIER, Self::LENGTH);

      /// Parse the raw composite public-key encoding.
      ///
      /// # Errors
      ///
      /// Returns [`CompositeMlDsaKeyError::InvalidPublicKey`] for a wrong
      /// length or an invalid traditional component.
      pub fn try_from_slice(bytes: &[u8]) -> Result<Self, CompositeMlDsaKeyError> {
        let bytes: &[u8; Self::LENGTH] = bytes.try_into().map_err(|_| CompositeMlDsaKeyError::InvalidPublicKey)?;
        Self::from_bytes(bytes)
      }

      /// Parse the fixed-length raw composite public-key encoding.
      ///
      /// # Errors
      ///
      /// Returns [`CompositeMlDsaKeyError::InvalidPublicKey`] for an invalid
      /// traditional component.
      pub fn from_bytes(bytes: &[u8; Self::LENGTH]) -> Result<Self, CompositeMlDsaKeyError> {
        let (ml_dsa, trad) = bytes
          .split_first_chunk::<{ MlDsa65PublicKey::LENGTH }>()
          .expect("fixed length");
        let trad: [u8; $trad_pk] = trad.try_into().expect("fixed length");
        if !Self::traditional_is_valid(&trad) {
          return Err(CompositeMlDsaKeyError::InvalidPublicKey);
        }
        Ok(Self {
          ml_dsa: MlDsa65PublicKey::from_bytes(*ml_dsa),
          trad,
        })
      }

      /// Copy the raw composite public-key encoding.
      #[must_use]
      pub fn to_bytes(&self) -> [u8; Self::LENGTH] {
        pkix::concat(self.ml_dsa.as_bytes(), &self.trad)
      }

      /// Parse a SubjectPublicKeyInfo for this algorithm.
      ///
      /// Accepts only the unique DER encoding: this algorithm's identifier
      /// with absent parameters, a BIT STRING with no unused bits, the exact
      /// key length, and no trailing input. No heap allocation.
      ///
      /// # Errors
      ///
      /// Returns [`CompositeMlDsaKeyError::UnsupportedAlgorithm`] for a
      /// well-formed key of another algorithm,
      /// [`CompositeMlDsaKeyError::InvalidPublicKey`] for a wrong key length or
      /// an invalid component, and [`CompositeMlDsaKeyError::MalformedDer`]
      /// for any other encoding.
      pub fn from_spki_der(der: &[u8]) -> Result<Self, CompositeMlDsaKeyError> {
        Self::from_bytes(pkix::decode_spki(der, &Self::SPKI_HEADER)?)
      }

      /// Encode the SubjectPublicKeyInfo DER. No heap allocation.
      #[must_use]
      pub fn to_spki_der(&self) -> [u8; Self::SPKI_DER_LENGTH] {
        pkix::concat(&Self::SPKI_HEADER, &self.to_bytes())
      }

      /// Verify a signature with the empty application context.
      pub fn verify(&self, message: &[u8], signature: &$signature) -> Result<(), VerificationError> {
        self.verify_with_context(message, &[], signature)
      }

      /// Verify a signature over `message` with an application context.
      ///
      /// Succeeds only if both component signatures verify. Every invalid
      /// signature or context returns an opaque error.
      pub fn verify_with_context(
        &self,
        message: &[u8],
        context: &[u8],
        signature: &$signature,
      ) -> Result<(), VerificationError> {
        self.verify_prehash(&prehash(message), context, signature)
      }

      /// Verify a signature over a caller-computed SHA-512 digest of the
      /// message, as the draft's section 10.5 externalizes the prehash.
      pub fn verify_prehash(
        &self,
        prehash: &[u8; 64],
        context: &[u8],
        signature: &$signature,
      ) -> Result<(), VerificationError> {
        let representative =
          Representative::new($profile::LABEL, context, prehash).map_err(|_| VerificationError::new())?;
        let ml_dsa = signature.ml_dsa()?;
        self
          .ml_dsa
          .verify_with_context(representative.as_slice(), $profile::LABEL, &ml_dsa)?;
        self.verify_traditional(representative.as_slice(), signature)
      }
    }

    impl fmt::Debug for $public {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple(stringify!($public))
          .field(&self.to_bytes().as_slice())
          .finish()
      }
    }

    impl Verifier<$signature> for $public {
      fn verify(&self, message: &[u8], signature: &$signature) -> Result<(), VerificationError> {
        self.verify(message, signature)
      }
    }

    #[doc = concat!("`", $name, "` secret key.")]
    ///
    /// Holds the ML-DSA-65 seed and expanded key and the traditional secret
    /// key in their component owners, which clear them on drop. Not `Clone`
    /// or `Copy`; `Debug` is redacted.
    pub struct $secret {
      ml_dsa: MlDsaPart,
      trad: Traditional,
    }

    impl $secret {
      /// Raw composite private-key length in bytes: the ML-DSA-65 seed and
      /// the traditional private key.
      pub const LENGTH: usize = ML_DSA_SEED_LENGTH + $trad_sk;

      /// DER length of this key as a version 1 PKCS #8 private key.
      pub const PKCS8_DER_LENGTH: usize =
        pkix::raw_pkcs8_header_length_for(IDENTIFIER_LENGTH, Self::LENGTH).strict_add(Self::LENGTH);

      const PKCS8_HEADER: [u8; pkix::raw_pkcs8_header_length_for(IDENTIFIER_LENGTH, Self::LENGTH)] =
        pkix::raw_pkcs8_header_for(&$profile::IDENTIFIER, Self::LENGTH);

      /// Zero-filled owner that the constructors fill in place; never returned unfilled.
      const fn zeroed() -> Self {
        Self {
          ml_dsa: MlDsaPart::zeroed(),
          trad: Traditional::zeroed(),
        }
      }

      /// Import the raw encoding into this owner in place: the traditional
      /// key first, then the ML-DSA-65 seed.
      fn import_into(&mut self, input: &[u8; Self::LENGTH]) -> Result<(), CompositeMlDsaKeyError> {
        let _dit = DataIndependentTiming::enter();
        let (seed, trad) = input.split_first_chunk::<ML_DSA_SEED_LENGTH>().expect("fixed length");
        self.trad.import_into(trad.try_into().expect("fixed length"))?;
        self.ml_dsa.import_into(seed)
      }

      /// Generate into this owner in place: the ML-DSA-65 key first, then the
      /// traditional key.
      fn generate_into(
        &mut self,
        fill_random: &mut impl FnMut(&mut [u8]) -> Result<(), CompositeMlDsaError>,
      ) -> Result<(), CompositeMlDsaError> {
        let _dit = DataIndependentTiming::enter();
        self.ml_dsa.generate_into(fill_random)?;
        self.trad.generate_into(fill_random)
      }

      /// Reject a carried public key that is not this key's own.
      fn check_public(&self, public: Option<&[u8; $public::LENGTH]>) -> Result<(), CompositeMlDsaKeyError> {
        if public.is_some_and(|public| *public != self.public_key().to_bytes()) {
          return Err(CompositeMlDsaKeyError::InvalidSecretKey);
        }
        Ok(())
      }

      /// Import the raw composite private-key encoding.
      ///
      /// The ML-DSA-65 seed is expanded and the traditional key is validated.
      /// The caller retains responsibility for clearing `input`. No heap
      /// allocation.
      ///
      /// # Errors
      ///
      /// Returns [`CompositeMlDsaKeyError::InvalidSecretKey`] for a wrong
      /// length or an invalid traditional key, and
      /// [`CompositeMlDsaKeyError::RejectionLimit`] if seed expansion
      /// exhausts its sampling bound.
      pub fn try_from_slice(input: &[u8]) -> Result<Self, CompositeMlDsaKeyError> {
        let input: &[u8; Self::LENGTH] = input.try_into().map_err(|_| CompositeMlDsaKeyError::InvalidSecretKey)?;
        Self::from_bytes(input)
      }

      /// Import the fixed-length raw composite private-key encoding, as by
      /// [`Self::try_from_slice`].
      pub fn from_bytes(input: &[u8; Self::LENGTH]) -> Result<Self, CompositeMlDsaKeyError> {
        let mut secret = Self::zeroed();
        secret.import_into(input)?;
        Ok(secret)
      }

      /// Like [`Self::try_from_slice`], with the key imported into memory from `alloc`.
      ///
      /// The components are written directly into the allocation, so moving
      /// the box moves only a pointer. The box clears the key on drop; a
      /// rejected key is cleared before returning. A wrong length is rejected
      /// before allocating. Allocation failure is handled as by
      /// [`Box::new_in`]. The caller retains responsibility for clearing `input`.
      #[cfg(feature = "alloc")]
      pub fn try_from_slice_in<A: Allocator>(input: &[u8], alloc: A) -> Result<Box<Self, A>, CompositeMlDsaKeyError> {
        let input: &[u8; Self::LENGTH] = input.try_into().map_err(|_| CompositeMlDsaKeyError::InvalidSecretKey)?;
        let mut secret = Box::new_in(Self::zeroed(), alloc);
        secret.import_into(input)?;
        Ok(secret)
      }

      /// Import a private key from RFC 5958 OneAsymmetricKey (PKCS #8) DER.
      ///
      /// The algorithm identifier must name this algorithm with absent
      /// parameters, and `privateKey` must hold the raw composite encoding.
      /// Accepts a version 1 container, or a version 2 container whose public
      /// key belongs to the key. The caller retains responsibility for
      /// clearing `der`. No heap allocation.
      ///
      /// # Errors
      ///
      /// Returns [`CompositeMlDsaKeyError::MalformedDer`] for malformed or
      /// non-canonical DER, including a version that disagrees with the
      /// public-key field; [`CompositeMlDsaKeyError::UnsupportedAlgorithm`]
      /// for another algorithm; [`CompositeMlDsaKeyError::UnsupportedEncoding`]
      /// for attributes; [`CompositeMlDsaKeyError::InvalidPublicKey`] for a
      /// wrong-length public key; and the errors of [`Self::from_bytes`],
      /// with [`CompositeMlDsaKeyError::InvalidSecretKey`] for a public key
      /// that is not its own.
      pub fn from_pkcs8_der(der: &[u8]) -> Result<Self, CompositeMlDsaKeyError> {
        let _dit = DataIndependentTiming::enter();
        let (key, public) = pkix::decode_pkcs8_raw_oid::<CompositeMlDsaKeyError, { Self::LENGTH }, { $public::LENGTH }>(
          der,
          &$profile::OID_CONTENTS,
        )?;
        let secret = Self::from_bytes(key)?;
        secret.check_public(public)?;
        Ok(secret)
      }

      /// Like [`Self::from_pkcs8_der`], with the key in memory from `alloc`.
      ///
      /// The DER is decoded first, so malformed DER allocates nothing. The key
      /// is then written directly into its allocation, as by
      /// [`Self::try_from_slice_in`]. The box clears the key on drop; a
      /// rejected key is cleared before returning. Allocation failure is
      /// handled as by [`Box::new_in`].
      #[cfg(feature = "alloc")]
      pub fn from_pkcs8_der_in<A: Allocator>(der: &[u8], alloc: A) -> Result<Box<Self, A>, CompositeMlDsaKeyError> {
        let _dit = DataIndependentTiming::enter();
        let (key, public) = pkix::decode_pkcs8_raw_oid::<CompositeMlDsaKeyError, { Self::LENGTH }, { $public::LENGTH }>(
          der,
          &$profile::OID_CONTENTS,
        )?;
        let mut secret = Box::new_in(Self::zeroed(), alloc);
        secret.import_into(key)?;
        secret.check_public(public)?;
        Ok(secret)
      }

      /// Write this key as a version 1 PKCS #8 private key. No heap
      /// allocation.
      ///
      /// `out` then holds the secret key, and the caller owns its cleanup.
      pub fn to_pkcs8_der_into(&self, out: &mut [u8; Self::PKCS8_DER_LENGTH]) {
        let raw = self.expose_secret();
        pkix::write(&Self::PKCS8_HEADER, raw.as_bytes(), out);
      }

      /// Explicitly export the raw composite private-key encoding into a
      /// zeroizing owner.
      #[must_use]
      pub fn expose_secret(&self) -> SecretBytes<{ Self::LENGTH }> {
        let mut raw = ZeroizingBytes::<{ Self::LENGTH }>::zeroed();
        let (seed, trad) = raw.as_mut_array().split_at_mut(ML_DSA_SEED_LENGTH);
        seed.copy_from_slice(self.ml_dsa.seed.as_array());
        self.export_traditional(trad.try_into().expect("fixed length"));
        SecretBytes::new(*raw.as_array())
      }

      /// The matching composite public key.
      #[must_use]
      pub fn public_key(&self) -> $public {
        $public {
          ml_dsa: self.ml_dsa.secret.public_key().clone(),
          trad: self.trad.public_bytes(),
        }
      }

      /// Expand the ML-DSA-65 key into caller-owned storage for repeated
      /// signing.
      ///
      /// The returned key borrows `storage`; it clears the prepared secrets
      /// when dropped, and a failed preparation clears them before returning.
      /// No heap allocation.
      ///
      /// # Errors
      ///
      /// Returns [`CompositeMlDsaError::SigningFailed`] if preparation fails.
      pub fn prepare<'a>(
        &self,
        storage: &'a mut MlDsa65PreparedSecretKeyStorage,
      ) -> Result<$prepared<&'a mut MlDsa65PreparedSecretKeyStorage>, CompositeMlDsaError> {
        Ok($prepared {
          ml_dsa: self
            .ml_dsa
            .secret
            .prepare(storage)
            .map_err(|_| CompositeMlDsaError::SigningFailed)?,
          trad: self.trad.duplicate(),
        })
      }

      /// Like [`Self::prepare`], with storage allocated from `alloc`.
      ///
      /// The prepared key owns its storage and a copy of the traditional
      /// key, so it can be kept independently of this key. Dropping it clears
      /// the secrets before the allocation is freed. Allocation failure is
      /// handled as by [`Box::new_in`].
      #[cfg(feature = "alloc")]
      pub fn prepare_in<A: Allocator>(
        &self,
        alloc: A,
      ) -> Result<$prepared<Box<MlDsa65PreparedSecretKeyStorage, A>>, CompositeMlDsaError> {
        Ok($prepared {
          ml_dsa: self
            .ml_dsa
            .secret
            .prepare_in(alloc)
            .map_err(|_| CompositeMlDsaError::SigningFailed)?,
          trad: self.trad.duplicate(),
        })
      }

      composite_signing_methods!($profile, $signature, $random_len);

      fn sign_components(
        &self,
        representative: &Representative,
        random: Option<&[u8; $random_len]>,
      ) -> Result<$signature, CompositeMlDsaError> {
        let ml_dsa = sign_ml_dsa(
          |message, context| self.ml_dsa.secret.sign_deterministic(message, context),
          |message, context, random| {
            self.ml_dsa.secret.sign_with(message, context, |out| {
              out.copy_from_slice(random);
              Ok(())
            })
          },
          representative,
          $profile::LABEL,
          random.map(|random| random.first_chunk::<32>().expect("ML-DSA randomness")),
        )?;
        $signature::from_components(&ml_dsa, &self.trad, representative, random)
      }
    }

    impl fmt::Debug for $secret {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(concat!(stringify!($secret), "(****)"))
      }
    }

    #[doc = concat!("`", $name, "` signing key prepared for repeated signing, in ML-DSA-65 storage `S`.")]
    ///
    /// `S` is caller-owned storage borrowed from `prepare`, or a box
    /// allocated by `prepare_in`. Signatures equal those of the secret key it
    /// was prepared from. The prepared ML-DSA secrets are cleared on drop.
    /// Not `Clone` or `Copy`.
    pub struct $prepared<S: DerefMut<Target = MlDsa65PreparedSecretKeyStorage>> {
      ml_dsa: MlDsa65PreparedSecretKey<S>,
      trad: Traditional,
    }

    impl<S: DerefMut<Target = MlDsa65PreparedSecretKeyStorage>> $prepared<S> {
      composite_signing_methods!($profile, $signature, $random_len);

      fn sign_components(
        &self,
        representative: &Representative,
        random: Option<&[u8; $random_len]>,
      ) -> Result<$signature, CompositeMlDsaError> {
        let ml_dsa = sign_ml_dsa(
          |message, context| self.ml_dsa.sign_deterministic(message, context),
          |message, context, random| {
            self.ml_dsa.sign_with(message, context, |out| {
              out.copy_from_slice(random);
              Ok(())
            })
          },
          representative,
          $profile::LABEL,
          random.map(|random| random.first_chunk::<32>().expect("ML-DSA randomness")),
        )?;
        $signature::from_components(&ml_dsa, &self.trad, representative, random)
      }
    }

    impl<S: DerefMut<Target = MlDsa65PreparedSecretKeyStorage>> fmt::Debug for $prepared<S> {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(concat!(stringify!($prepared), "(****)"))
      }
    }
  };
}

macro_rules! composite_signing_methods {
  ($profile:ident, $signature:ident, $random_len:expr) => {
    /// Deterministic signing with an application context of at most 255
    /// bytes.
    ///
    /// ML-DSA-65 signs deterministically, and the traditional component
    /// signs as it always does without caller randomness.
    ///
    /// # Errors
    ///
    /// Returns [`CompositeMlDsaError::ContextTooLong`] or
    /// [`CompositeMlDsaError::SigningFailed`].
    pub fn sign_deterministic(&self, message: &[u8], context: &[u8]) -> Result<$signature, CompositeMlDsaError> {
      self.sign_prehash_deterministic(&prehash(message), context)
    }

    /// Hedged signing with randomness from `fill_random`.
    ///
    /// The context is checked before randomness is requested, and all
    /// randomness is requested before any secret computation. The callback's
    /// first error is returned unchanged and leaves the key usable.
    ///
    /// # Errors
    ///
    /// Returns [`CompositeMlDsaError::ContextTooLong`], the callback's error,
    /// or [`CompositeMlDsaError::SigningFailed`].
    pub fn sign_with(
      &self,
      message: &[u8],
      context: &[u8],
      fill_random: impl FnMut(&mut [u8]) -> Result<(), CompositeMlDsaError>,
    ) -> Result<$signature, CompositeMlDsaError> {
      self.sign_prehash_with(&prehash(message), context, fill_random)
    }

    /// Hedged signing with OS entropy.
    #[cfg(feature = "getrandom")]
    pub fn try_sign(&self, message: &[u8], context: &[u8]) -> Result<$signature, CompositeMlDsaError> {
      self.sign_with(message, context, |out| {
        getrandom::fill(out).map_err(|_| CompositeMlDsaError::RandomGenerationFailed)
      })
    }

    /// Deterministic signing over a caller-computed SHA-512 digest of the
    /// message, as the draft's section 10.5 externalizes the prehash.
    ///
    /// The result equals [`Self::sign_deterministic`] over the message.
    pub fn sign_prehash_deterministic(
      &self,
      prehash: &[u8; 64],
      context: &[u8],
    ) -> Result<$signature, CompositeMlDsaError> {
      let representative = Representative::new($profile::LABEL, context, prehash)?;
      let _dit = DataIndependentTiming::enter();
      self.sign_components(&representative, None)
    }

    /// Hedged signing over a caller-computed SHA-512 digest of the message,
    /// as by [`Self::sign_with`].
    pub fn sign_prehash_with(
      &self,
      prehash: &[u8; 64],
      context: &[u8],
      mut fill_random: impl FnMut(&mut [u8]) -> Result<(), CompositeMlDsaError>,
    ) -> Result<$signature, CompositeMlDsaError> {
      let representative = Representative::new($profile::LABEL, context, prehash)?;
      let mut random = ZeroizingBytes::<{ $random_len }>::zeroed();
      fill_random(random.as_mut_array())?;
      let _dit = DataIndependentTiming::enter();
      self.sign_components(&representative, Some(random.as_array()))
    }
  };
}

#[cfg(feature = "mldsa65-ed25519")]
pub use ed25519_composite::{
  MlDsa65Ed25519, MlDsa65Ed25519PreparedSecretKey, MlDsa65Ed25519PublicKey, MlDsa65Ed25519SecretKey,
  MlDsa65Ed25519Signature,
};

#[cfg(feature = "mldsa65-ed25519")]
mod ed25519_composite {
  use super::*;
  use crate::{Ed25519Keypair, Ed25519PublicKey, Ed25519SecretKey, Ed25519Signature};

  /// The Ed25519 component of a secret key.
  struct Traditional(Ed25519Keypair);

  impl Traditional {
    const fn zeroed() -> Self {
      Self(Ed25519Keypair::zeroed())
    }

    /// Every 32-byte string is an Ed25519 secret key.
    fn import_into(&mut self, seed: &[u8; Ed25519SecretKey::LENGTH]) -> Result<(), CompositeMlDsaKeyError> {
      self.0.import_into(seed);
      Ok(())
    }

    fn generate_into(
      &mut self,
      fill_random: &mut impl FnMut(&mut [u8]) -> Result<(), CompositeMlDsaError>,
    ) -> Result<(), CompositeMlDsaError> {
      self.0.generate_into(fill_random)
    }

    fn public_bytes(&self) -> [u8; Ed25519PublicKey::LENGTH] {
      self.0.public_key().to_bytes()
    }

    fn duplicate(&self) -> Self {
      Self(self.0.duplicate_secret())
    }
  }

  composite! {
    profile: MlDsa65Ed25519,
    public: MlDsa65Ed25519PublicKey,
    secret: MlDsa65Ed25519SecretKey,
    signature: MlDsa65Ed25519Signature,
    prepared: MlDsa65Ed25519PreparedSecretKey,
    name: "id-MLDSA65-Ed25519-SHA512",
    label: b"COMPSIG-MLDSA65-Ed25519-SHA512",
    arc: 48,
    trad_public_length: Ed25519PublicKey::LENGTH,
    trad_secret_length: Ed25519SecretKey::LENGTH,
    random_length: 32,
  }

  impl MlDsa65Ed25519PublicKey {
    /// Every 32-byte string is an admissible Ed25519 key encoding; one that
    /// is not a valid point fails verification, as for [`Ed25519PublicKey`].
    fn traditional_is_valid(_: &[u8; Ed25519PublicKey::LENGTH]) -> bool {
      true
    }

    fn verify_traditional(
      &self,
      representative: &[u8],
      signature: &MlDsa65Ed25519Signature,
    ) -> Result<(), VerificationError> {
      Ed25519PublicKey::from_bytes(self.trad).verify(representative, &signature.ed25519())
    }
  }

  impl MlDsa65Ed25519SecretKey {
    fn export_traditional(&self, out: &mut [u8; Ed25519SecretKey::LENGTH]) {
      out.copy_from_slice(self.trad.0.secret_key().as_bytes());
    }
  }

  /// `id-MLDSA65-Ed25519-SHA512` signature: the ML-DSA-65 signature and the
  /// Ed25519 signature.
  #[derive(Clone, Eq, PartialEq)]
  pub struct MlDsa65Ed25519Signature([u8; Self::LENGTH]);

  impl MlDsa65Ed25519Signature {
    /// Encoded signature length in bytes.
    pub const LENGTH: usize = MlDsa65Signature::LENGTH + Ed25519Signature::LENGTH;

    /// Parse a signature, checking its length and the ML-DSA-65 component's
    /// structure. This is structural validation; authenticate with the
    /// public key's verifier.
    ///
    /// # Errors
    ///
    /// Returns [`CompositeMlDsaError::InvalidSignature`].
    pub fn try_from_slice(bytes: &[u8]) -> Result<Self, CompositeMlDsaError> {
      let bytes: [u8; Self::LENGTH] = bytes.try_into().map_err(|_| CompositeMlDsaError::InvalidSignature)?;
      MlDsa65Signature::try_from_slice(&bytes[..MlDsa65Signature::LENGTH])
        .map_err(|_| CompositeMlDsaError::InvalidSignature)?;
      Ok(Self(bytes))
    }

    /// Borrow the encoding.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
      &self.0
    }

    /// Copy the encoding.
    #[must_use]
    pub const fn to_bytes(&self) -> [u8; Self::LENGTH] {
      self.0
    }

    fn ml_dsa(&self) -> Result<MlDsa65Signature, VerificationError> {
      MlDsa65Signature::try_from_slice(&self.0[..MlDsa65Signature::LENGTH]).map_err(|_| VerificationError::new())
    }

    fn ed25519(&self) -> Ed25519Signature {
      Ed25519Signature::from_bytes(self.0[MlDsa65Signature::LENGTH..].try_into().expect("fixed length"))
    }

    fn from_components(
      ml_dsa: &MlDsa65Signature,
      trad: &Traditional,
      representative: &Representative,
      _random: Option<&[u8; 32]>,
    ) -> Result<Self, CompositeMlDsaError> {
      let ed25519 = trad.0.sign(representative.as_slice());
      Ok(Self(pkix::concat(ml_dsa.as_bytes(), ed25519.as_bytes())))
    }
  }

  impl AsRef<[u8]> for MlDsa65Ed25519Signature {
    fn as_ref(&self) -> &[u8] {
      &self.0
    }
  }

  impl fmt::Debug for MlDsa65Ed25519Signature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      f.debug_tuple("MlDsa65Ed25519Signature")
        .field(&self.0.as_slice())
        .finish()
    }
  }
}

#[cfg(feature = "mldsa65-ecdsa-p256")]
pub use ecdsa_p256_composite::{
  MlDsa65EcdsaP256, MlDsa65EcdsaP256PreparedSecretKey, MlDsa65EcdsaP256PublicKey, MlDsa65EcdsaP256SecretKey,
  MlDsa65EcdsaP256Signature,
};

#[cfg(feature = "mldsa65-ecdsa-p256")]
mod ecdsa_p256_composite {
  use super::*;
  use crate::{EcdsaError, EcdsaP256PublicKey, EcdsaP256SecretKey, EcdsaP256Signature};

  /// RFC 5915 `ECPrivateKey` bytes before the P-256 scalar: version 1 and the
  /// 32-byte private key header.
  const EC_PRIVATE_KEY_PREFIX: [u8; 7] = [
    0x30, 0x31, // ECPrivateKey, 49 bytes
    0x02, 0x01, 0x01, // ecPrivkeyVer1
    0x04, 0x20, // privateKey OCTET STRING, 32 bytes
  ];
  /// `ECPrivateKey` bytes after the scalar: the `prime256v1` named curve and
  /// no public key, as section 4 of the draft requires.
  const EC_PRIVATE_KEY_SUFFIX: [u8; 12] = [
    0xa0, 0x0a, // parameters [0], 10 bytes
    0x06, 0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07, // prime256v1
  ];
  /// DER contents of the `prime256v1` OID.
  const SECP256R1_OID: [u8; 8] = [0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07];
  /// Encoded `ECPrivateKey` length.
  const EC_PRIVATE_KEY_LENGTH: usize =
    EC_PRIVATE_KEY_PREFIX.len() + EcdsaP256SecretKey::LENGTH + EC_PRIVATE_KEY_SUFFIX.len();

  /// The ECDSA P-256 component of a secret key, with its public point.
  struct Traditional {
    secret: EcdsaP256SecretKey,
    public: [u8; EcdsaP256PublicKey::SEC1_LENGTH],
  }

  impl Traditional {
    const fn zeroed() -> Self {
      Self {
        secret: EcdsaP256SecretKey::zeroed(),
        public: [0; EcdsaP256PublicKey::SEC1_LENGTH],
      }
    }

    /// Accept exactly the `ECPrivateKey` form of section 4; classify any
    /// other input with the general RFC 5915 parser.
    fn import_into(&mut self, input: &[u8; EC_PRIVATE_KEY_LENGTH]) -> Result<(), CompositeMlDsaKeyError> {
      let (prefix, rest) = input
        .split_first_chunk::<{ EC_PRIVATE_KEY_PREFIX.len() }>()
        .expect("fixed length");
      let (scalar, suffix) = rest
        .split_first_chunk::<{ EcdsaP256SecretKey::LENGTH }>()
        .expect("fixed length");
      if *prefix != EC_PRIVATE_KEY_PREFIX || *suffix != EC_PRIVATE_KEY_SUFFIX {
        return Err(
          match pkix::decode_ec_private_key::<
            CompositeMlDsaKeyError,
            { EcdsaP256SecretKey::LENGTH },
            { EcdsaP256PublicKey::SEC1_LENGTH },
          >(input, &SECP256R1_OID)
          {
            Ok(_) => CompositeMlDsaKeyError::UnsupportedEncoding,
            Err(error) => error,
          },
        );
      }
      self
        .secret
        .import_into(scalar)
        .map_err(|_| CompositeMlDsaKeyError::InvalidSecretKey)?;
      self.derive_public();
      Ok(())
    }

    fn generate_into(
      &mut self,
      fill_random: &mut impl FnMut(&mut [u8]) -> Result<(), CompositeMlDsaError>,
    ) -> Result<(), CompositeMlDsaError> {
      self.secret.generate_into(fill_random).map_err(|error| match error {
        crate::EcdsaKeyGenerationError::Random(error) => error,
        _ => CompositeMlDsaError::KeyGenerationFailed,
      })?;
      self.derive_public();
      Ok(())
    }

    fn derive_public(&mut self) {
      self.public = self.secret.public_key().to_sec1_bytes();
    }

    fn public_bytes(&self) -> [u8; EcdsaP256PublicKey::SEC1_LENGTH] {
      self.public
    }

    fn duplicate(&self) -> Self {
      Self {
        secret: self.secret.duplicate_secret(),
        public: self.public,
      }
    }
  }

  composite! {
    profile: MlDsa65EcdsaP256,
    public: MlDsa65EcdsaP256PublicKey,
    secret: MlDsa65EcdsaP256SecretKey,
    signature: MlDsa65EcdsaP256Signature,
    prepared: MlDsa65EcdsaP256PreparedSecretKey,
    name: "id-MLDSA65-ECDSA-P256-SHA512",
    label: b"COMPSIG-MLDSA65-ECDSA-P256-SHA512",
    arc: 45,
    trad_public_length: EcdsaP256PublicKey::SEC1_LENGTH,
    trad_secret_length: EC_PRIVATE_KEY_LENGTH,
    random_length: 96,
  }

  impl MlDsa65EcdsaP256PublicKey {
    /// An uncompressed SEC1 point on the curve.
    fn traditional_is_valid(point: &[u8; EcdsaP256PublicKey::SEC1_LENGTH]) -> bool {
      EcdsaP256PublicKey::is_valid_sec1(point)
    }

    fn verify_traditional(
      &self,
      representative: &[u8],
      signature: &MlDsa65EcdsaP256Signature,
    ) -> Result<(), VerificationError> {
      EcdsaP256PublicKey::verify_sec1(&self.trad, representative, &signature.ecdsa()?)
    }
  }

  impl MlDsa65EcdsaP256SecretKey {
    fn export_traditional(&self, out: &mut [u8; EC_PRIVATE_KEY_LENGTH]) {
      let (prefix, rest) = out.split_at_mut(EC_PRIVATE_KEY_PREFIX.len());
      let (scalar, suffix) = rest.split_at_mut(EcdsaP256SecretKey::LENGTH);
      prefix.copy_from_slice(&EC_PRIVATE_KEY_PREFIX);
      scalar.copy_from_slice(self.trad.secret.as_bytes());
      suffix.copy_from_slice(&EC_PRIVATE_KEY_SUFFIX);
    }
  }

  /// `id-MLDSA65-ECDSA-P256-SHA512` signature: the ML-DSA-65 signature and a
  /// DER `Ecdsa-Sig-Value`, whose length varies.
  #[derive(Clone, Eq, PartialEq)]
  pub struct MlDsa65EcdsaP256Signature {
    bytes: [u8; Self::MAX_LENGTH],
    len: usize,
  }

  impl MlDsa65EcdsaP256Signature {
    /// Longest encoded signature in bytes.
    pub const MAX_LENGTH: usize = MlDsa65Signature::LENGTH + EcdsaP256Signature::DER_MAX_LENGTH;

    /// Parse a signature, checking the ML-DSA-65 component's structure and
    /// that the rest is one canonical DER `Ecdsa-Sig-Value` with scalars in
    /// range. This is structural validation; authenticate with the public
    /// key's verifier.
    ///
    /// # Errors
    ///
    /// Returns [`CompositeMlDsaError::InvalidSignature`].
    pub fn try_from_slice(bytes: &[u8]) -> Result<Self, CompositeMlDsaError> {
      if bytes.len() > Self::MAX_LENGTH {
        return Err(CompositeMlDsaError::InvalidSignature);
      }
      let (ml_dsa, ecdsa) = bytes
        .split_at_checked(MlDsa65Signature::LENGTH)
        .ok_or(CompositeMlDsaError::InvalidSignature)?;
      MlDsa65Signature::try_from_slice(ml_dsa).map_err(|_| CompositeMlDsaError::InvalidSignature)?;
      let parsed = EcdsaP256Signature::from_der(ecdsa).map_err(|_| CompositeMlDsaError::InvalidSignature)?;
      // Only the unique DER encoding of the scalars is accepted.
      let mut canonical = [0; EcdsaP256Signature::DER_MAX_LENGTH];
      if parsed.to_der_into(&mut canonical) != ecdsa {
        return Err(CompositeMlDsaError::InvalidSignature);
      }
      let mut out = [0; Self::MAX_LENGTH];
      out[..bytes.len()].copy_from_slice(bytes);
      Ok(Self {
        bytes: out,
        len: bytes.len(),
      })
    }

    /// Borrow the encoding.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
      &self.bytes[..self.len]
    }

    fn ml_dsa(&self) -> Result<MlDsa65Signature, VerificationError> {
      MlDsa65Signature::try_from_slice(&self.bytes[..MlDsa65Signature::LENGTH]).map_err(|_| VerificationError::new())
    }

    fn ecdsa(&self) -> Result<EcdsaP256Signature, VerificationError> {
      EcdsaP256Signature::from_der(&self.bytes[MlDsa65Signature::LENGTH..self.len])
        .map_err(|_| VerificationError::new())
    }

    fn from_components(
      ml_dsa: &MlDsa65Signature,
      trad: &Traditional,
      representative: &Representative,
      random: Option<&[u8; 96]>,
    ) -> Result<Self, CompositeMlDsaError> {
      let ecdsa = match random {
        None => trad.secret.try_sign(representative.as_slice()),
        Some(random) => trad
          .secret
          .try_sign_blinded_with(representative.as_slice(), |blind| {
            blind.copy_from_slice(&random[32..]);
            Ok::<(), EcdsaError>(())
          })
          .map_err(|_| EcdsaError::SigningFailure),
      }
      .map_err(|_| CompositeMlDsaError::SigningFailed)?;
      let mut der = [0; EcdsaP256Signature::DER_MAX_LENGTH];
      let der = ecdsa.to_der_into(&mut der);
      let mut bytes = [0; Self::MAX_LENGTH];
      bytes[..MlDsa65Signature::LENGTH].copy_from_slice(ml_dsa.as_bytes());
      let len = MlDsa65Signature::LENGTH.strict_add(der.len());
      bytes[MlDsa65Signature::LENGTH..len].copy_from_slice(der);
      Ok(Self { bytes, len })
    }
  }

  impl AsRef<[u8]> for MlDsa65EcdsaP256Signature {
    fn as_ref(&self) -> &[u8] {
      self.as_bytes()
    }
  }

  impl fmt::Debug for MlDsa65EcdsaP256Signature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      f.debug_tuple("MlDsa65EcdsaP256Signature")
        .field(&self.as_bytes())
        .finish()
    }
  }
}
