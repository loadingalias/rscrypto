//! RSA public-key parsing and typed public-key representation.
//!
//! This module starts RSA support at the trust boundary: strict DER parsing of
//! public keys into a key type that later verifier code can consume without
//! re-validating shape, length, parity, or exponent policy.
//!
//! # Quick Start
//!
//! Verify signatures with a typed profile. Reuse [`RsaPublicScratch`] when a key
//! verifies more than one signature; reused verification performs no heap
//! allocation.
//!
//! ```rust
//! use rscrypto::{RsaPssProfile, RsaPublicKey, RsaSignatureProfile};
//!
//! let public_key = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/rsa/fixtures/rsa3072_spki.der"));
//! let signature = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/rsa/fixtures/rsa3072_pss_sha256.sig"));
//! let message = b"rscrypto RSA-PSS verification fixture";
//!
//! let key = RsaPublicKey::from_spki_der(public_key).map_err(|_| "RSA public key parses")?;
//!
//! key
//!   .verify_pss(RsaPssProfile::Sha256, message, signature)
//!   .map_err(|_| "PSS signature verifies")?;
//!
//! let mut scratch = key.public_scratch();
//! key
//!   .verify_signature_with_scratch(
//!     RsaSignatureProfile::pss(RsaPssProfile::Sha256),
//!     message,
//!     signature,
//!     &mut scratch,
//!   )
//!   .map_err(|_| "PSS signature verifies with reused scratch")?;
//!
//! # Ok::<(), &'static str>(())
//! ```
//!
//! RSA signing, OAEP encryption/decryption, legacy RSAES-PKCS1-v1_5
//! encryption, private-key import, and private-key generation are available
//! through [`RsaPrivateKey`] and [`RsaPublicKey`]. Legacy RSAES-PKCS1-v1_5
//! decryption, with implicit rejection, is the explicit opt-in
//! [`expert::RsaPkcs1v15Decrypt`]. Private operations
//! are blinded and fault-checked. Callers can provide signing entropy with the
//! `rsa` feature alone; OS-backed convenience APIs and key generation require
//! the `getrandom` feature.

mod arithmetic;
mod exponentiation;
mod implicit_rejection;
mod inverse;
mod keys;
mod oaep;
mod pkcs1v15;
mod private;
mod protocol;
mod pss;
#[cfg(test)]
mod tests;

use arithmetic::*;
use exponentiation::*;
use inverse::*;
use keys::*;
use private::*;

use alloc::{boxed::Box, vec, vec::Vec};

use core::{
  fmt,
  hash::{Hash, Hasher},
};

use crate::{
  SecretVec,
  backend::{
    der::{self, MalformedDer, TAG_BIT_STRING, TAG_OBJECT_IDENTIFIER, TAG_SEQUENCE},
    mgf1::mgf1,
  },
  hashes::crypto::{Sha256, Sha384, Sha512},
  platform::ProcessOrigin,
  traits::{Digest, VerificationError, ct},
};

#[cfg(all(
  target_arch = "aarch64",
  any(target_os = "macos", target_os = "linux"),
  not(feature = "portable-only"),
  not(miri)
))]
#[path = "rsa_aarch64_asm.rs"]
mod rsa_aarch64_asm;

#[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
#[path = "rsa_aarch64_mont.rs"]
mod rsa_aarch64_mont;

// Production selects the paired engine on Apple cores. The portable library test includes its Linux BINSEC entries.
#[cfg(all(
  target_arch = "aarch64",
  not(miri),
  any(
    all(target_vendor = "apple", not(feature = "portable-only")),
    all(test, feature = "portable-only")
  )
))]
#[path = "rsa_aarch64_pair.rs"]
mod rsa_aarch64_pair;

#[cfg(all(target_arch = "x86_64", not(feature = "portable-only"), not(miri)))]
#[path = "rsa_x86_64_mont.rs"]
mod rsa_x86_64_mont;

#[cfg(all(
  target_arch = "x86_64",
  target_feature = "sse2",
  not(feature = "portable-only"),
  not(miri)
))]
#[path = "rsa_x86_64_ifma.rs"]
mod rsa_x86_64_ifma;

/// RSAES-PKCS1-v1_5 padding bytes: `0x00 0x02`, at least eight PS bytes, and `0x00` (RFC 8017 7.2.1).
const RSAES_PKCS1V15_OVERHEAD: usize = 11;

const TAG_INTEGER: u8 = 0x02;

const TAG_OCTET_STRING: u8 = 0x04;

const TAG_NULL: u8 = 0x05;

const TAG_CONTEXT_0: u8 = 0xa0;

const TAG_CONTEXT_1: u8 = 0xa1;

const TAG_CONTEXT_2: u8 = 0xa2;

const TAG_CONTEXT_3: u8 = 0xa3;

const RSA_ENCRYPTION_OID: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01];

const SHA1_WITH_RSA_ENCRYPTION_OID: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x05];

const ID_RSASSA_PSS_OID: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0a];

const SHA256_WITH_RSA_ENCRYPTION_OID: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0b];

const SHA384_WITH_RSA_ENCRYPTION_OID: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0c];

const SHA512_WITH_RSA_ENCRYPTION_OID: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0d];

const ID_MGF1_OID: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x08];

const ID_SHA1_OID: &[u8] = &[0x2b, 0x0e, 0x03, 0x02, 0x1a];

const ID_SHA256_OID: &[u8] = &[0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01];

const ID_SHA384_OID: &[u8] = &[0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x02];

const ID_SHA512_OID: &[u8] = &[0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x03];

const SHA256_DIGEST_INFO_PREFIX: &[u8] = &[
  0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, 0x05, 0x00, 0x04, 0x20,
];

const SHA384_DIGEST_INFO_PREFIX: &[u8] = &[
  0x30, 0x41, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x02, 0x05, 0x00, 0x04, 0x30,
];

const SHA512_DIGEST_INFO_PREFIX: &[u8] = &[
  0x30, 0x51, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x03, 0x05, 0x00, 0x04, 0x40,
];

const MIN_RSA_MODULUS_BITS: usize = 2048;

const PRIVATE_FIXED_WINDOW_TABLE_ENTRIES: usize = 16;

#[cfg(feature = "getrandom")]
const RSA_KEYGEN_PUBLIC_EXPONENT: u64 = 65_537;

#[cfg(feature = "getrandom")]
const RSA_KEYGEN_MILLER_RABIN_ROUNDS: usize = 32;

const RSA_IMPORT_MILLER_RABIN_BASES: [u16; 32] = [
  2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109,
  113, 127, 131,
];

#[cfg(feature = "getrandom")]
const RSA_KEYGEN_PAIR_ATTEMPTS: usize = 64;

#[cfg(feature = "getrandom")]
const RSA_KEYGEN_MIN_PRIME_DISTANCE_SECURITY_MARGIN_BITS: usize = 100;

#[cfg(feature = "getrandom")]
const RSA_KEYGEN_DRBG_ENTROPY_BYTES: usize = 32;

const HMAC_SHA256_OUT_BYTES: usize = 32;

const HMAC_SHA256_BLOCK_BYTES: usize = 64;

/// draft-irtf-cfrg-rsa-guidance-10 section 7.2: 128 two-octet candidate lengths.
const IMPLICIT_REJECTION_CANDIDATE_BYTES: usize = 256;

#[cfg(feature = "getrandom")]
const RSA_KEYGEN_DRBG_NONCE_BYTES: usize = 16;

#[cfg(feature = "getrandom")]
const RSA_KEYGEN_DRBG_OUT_BYTES: usize = 32;

#[cfg(feature = "getrandom")]
const RSA_KEYGEN_DRBG_KEY_BYTES: usize = 32;

#[cfg(feature = "getrandom")]
const RSA_KEYGEN_DRBG_PERSONALIZATION: &[u8] = b"rscrypto RSA FIPS 186-5 A.1.3 HMAC_DRBG";

#[cfg(feature = "getrandom")]
const RSA_KEYGEN_SQRT2_HALF_TOP64: [u8; 8] = [0xb5, 0x04, 0xf3, 0x33, 0xf9, 0xde, 0x64, 0x84];

const RSA_SMALL_PRIMES: &[u16] = &[
  3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
  127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241,
  251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383,
  389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523,
  541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673,
  677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829,
  839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997,
];

/// Errors returned when parsing or validating an RSA public key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RsaKeyError {
  /// DER was structurally invalid or non-canonical.
  MalformedDer,
  /// The key algorithm identifier is not RSA `rsaEncryption`.
  UnsupportedAlgorithm,
  /// The RSA modulus is zero, even, too small, too large, or otherwise invalid.
  InvalidModulus,
  /// The public exponent is zero, even, too small, too large, or disallowed by policy.
  InvalidPublicExponent,
}

impl fmt::Display for RsaKeyError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(match self {
      Self::MalformedDer => "malformed DER",
      Self::UnsupportedAlgorithm => "unsupported RSA public-key algorithm",
      Self::InvalidModulus => "invalid RSA modulus",
      Self::InvalidPublicExponent => "invalid RSA public exponent",
    })
  }
}

impl core::error::Error for RsaKeyError {}

/// Errors returned by the low-level RSA public operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RsaPublicOpError {
  /// Input and output representatives must be exactly the modulus length.
  InvalidLength,
  /// The input representative must be strictly smaller than the RSA modulus.
  RepresentativeOutOfRange,
  /// Scratch storage was created for a different modulus width.
  InvalidScratch,
}

impl fmt::Display for RsaPublicOpError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(match self {
      Self::InvalidLength => "invalid RSA representative length",
      Self::RepresentativeOutOfRange => "RSA representative out of range",
      Self::InvalidScratch => "invalid RSA scratch space",
    })
  }
}

impl core::error::Error for RsaPublicOpError {}

/// Errors returned by RSA public-key encryption.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RsaEncryptionError {
  /// Input, output, or seed buffers have an invalid length.
  InvalidLength,
  /// The message is too long for this key and encryption profile.
  MessageTooLong,
  /// The caller-provided or platform entropy source was unavailable.
  EntropyUnavailable,
  /// The underlying RSA public operation failed.
  PublicOperationFailed,
}

impl fmt::Display for RsaEncryptionError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(match self {
      Self::InvalidLength => "invalid RSA encryption length",
      Self::MessageTooLong => "RSA message too long for key",
      Self::EntropyUnavailable => "RSA entropy source unavailable",
      Self::PublicOperationFailed => "RSA public encryption operation failed",
    })
  }
}

impl core::error::Error for RsaEncryptionError {}

/// Errors returned by RSA private-key generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RsaKeyGenerationError {
  /// Requested modulus length is outside the key-generation policy.
  ///
  /// `RsaPrivateKey::generate` accepts even lengths from 3072 through 8192
  /// bits. To mint an RSA-2048 compatibility key, call
  /// `RsaPrivateKey::generate_with_policy` with
  /// [`RsaPublicKeyPolicy::modulus_2048_to_8192`]. Odd lengths and lengths
  /// below 2048 bits are always rejected.
  InvalidModulusBits,
  /// The platform entropy source was unavailable.
  EntropyUnavailable,
  /// Prime search exhausted its bounded retry budget.
  PrimeSearchFailed,
  /// Internal big-integer arithmetic failed while deriving key components.
  ArithmeticFailure,
}

impl fmt::Display for RsaKeyGenerationError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(match self {
      Self::InvalidModulusBits => "invalid RSA key-generation modulus length",
      Self::EntropyUnavailable => "RSA key-generation entropy source unavailable",
      Self::PrimeSearchFailed => "RSA key-generation prime search failed",
      Self::ArithmeticFailure => "RSA key-generation arithmetic failed",
    })
  }
}

impl core::error::Error for RsaKeyGenerationError {}

/// RSA private-key generation contract exposed by this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RsaKeyGenerationContract {
  /// Probable-prime generation following FIPS 186-5 Appendix A.1.3 in code.
  ///
  /// The implementation uses an internal HMAC_DRBG seeded from `getrandom`,
  /// applies the FIPS A.1.1/A.1.3 prime constraints, and derives `d` modulo
  /// `lcm(p - 1, q - 1)`. This is an algorithmic conformance statement only;
  /// this crate is not a CMVP-validated FIPS 140-3 cryptographic module.
  Fips1865A13ProbablePrime,
}

/// Errors returned by RSA private-key operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RsaPrivateOpError {
  /// Input, output, salt, or blinding-factor buffers have an invalid length.
  InvalidLength,
  /// Scratch storage was created for a different modulus width, or a signing
  /// workspace was created for another key.
  InvalidScratch,
  /// The key is too small for the requested encoded message.
  MessageTooLong,
  /// A representative was not strictly smaller than the RSA modulus.
  RepresentativeOutOfRange,
  /// The supplied blinding factor and inverse are invalid for this modulus.
  InvalidBlindingFactor,
  /// The caller-provided or platform entropy source was unavailable.
  EntropyUnavailable,
  /// RSAES-OAEP decoding failed.
  ///
  /// RSAES-PKCS1-v1_5 decryption never returns this variant: it uses implicit
  /// rejection ([`expert::RsaPkcs1v15Decrypt`]).
  DecryptionFailed,
  /// The public-operation fault check did not recover the encoded message.
  FaultCheckFailed,
  /// The requested protocol algorithm is not an accepted RSA signing profile.
  UnsupportedAlgorithm,
}

impl fmt::Display for RsaPrivateOpError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(match self {
      Self::InvalidLength => "invalid RSA private-operation length",
      Self::InvalidScratch => "invalid RSA private-operation scratch space",
      Self::MessageTooLong => "RSA message too long for key",
      Self::RepresentativeOutOfRange => "RSA private-operation representative out of range",
      Self::InvalidBlindingFactor => "invalid RSA blinding factor",
      Self::EntropyUnavailable => "RSA entropy source unavailable",
      Self::DecryptionFailed => "RSA decryption failed",
      Self::FaultCheckFailed => "RSA private-operation fault check failed",
      Self::UnsupportedAlgorithm => "unsupported RSA private-operation algorithm",
    })
  }
}

impl core::error::Error for RsaPrivateOpError {}

/// Supported RSASSA-PSS hash/MGF1 verification profiles.
///
/// Each profile uses the same SHA-2 digest for the message hash and MGF1. The
/// default verification methods enforce salt length equal to the digest output
/// length, which is the TLS 1.3 profile. Call the explicit salt-length
/// verification methods when an adapter has parsed RSASSA-PSS parameters from a
/// standards-defined algorithm identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RsaPssProfile {
  /// RSASSA-PSS with SHA-256, MGF1-SHA-256, and a 32-byte salt.
  Sha256,
  /// RSASSA-PSS with SHA-384, MGF1-SHA-384, and a 48-byte salt.
  Sha384,
  /// RSASSA-PSS with SHA-512, MGF1-SHA-512, and a 64-byte salt.
  Sha512,
}

impl RsaPssProfile {
  /// Digest and salt length for this profile.
  #[inline]
  #[must_use]
  pub const fn digest_len(self) -> usize {
    match self {
      Self::Sha256 => 32,
      Self::Sha384 => 48,
      Self::Sha512 => 64,
    }
  }
}

/// Supported RSASSA-PKCS1-v1_5 verification profiles.
///
/// This intentionally supports SHA-2 only. SHA-1 verification belongs behind a
/// future explicit legacy certificate-policy boundary, not in the primitive
/// default surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RsaPkcs1v15Profile {
  /// RSASSA-PKCS1-v1_5 with SHA-256.
  Sha256,
  /// RSASSA-PKCS1-v1_5 with SHA-384.
  Sha384,
  /// RSASSA-PKCS1-v1_5 with SHA-512.
  Sha512,
}

/// Supported RSAES-OAEP encryption/decryption profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RsaOaepProfile {
  /// RSAES-OAEP with SHA-256 for both the label hash and MGF1.
  Sha256,
  /// RSAES-OAEP with SHA-384 for both the label hash and MGF1.
  Sha384,
  /// RSAES-OAEP with SHA-512 for both the label hash and MGF1.
  Sha512,
}

impl RsaOaepProfile {
  /// Digest and OAEP seed length for this profile.
  #[inline]
  #[must_use]
  pub const fn digest_len(self) -> usize {
    match self {
      Self::Sha256 => 32,
      Self::Sha384 => 48,
      Self::Sha512 => 64,
    }
  }
}

/// Error returned when a protocol algorithm identifier has no supported RSA profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RsaProtocolAlgorithmError {
  /// The algorithm identifier is not valid DER for the expected protocol field.
  MalformedAlgorithmIdentifier,
  /// The identifier is unknown, non-RSA, ambiguous, or intentionally unsupported.
  UnsupportedAlgorithm,
}

impl RsaProtocolAlgorithmError {
  /// Return `true` when the protocol algorithm identifier was malformed.
  ///
  /// Providers can use this to keep diagnostics structured without matching
  /// every enum variant or parsing the display string.
  #[inline]
  #[must_use]
  pub const fn is_malformed_algorithm_identifier(self) -> bool {
    match self {
      Self::MalformedAlgorithmIdentifier => true,
      Self::UnsupportedAlgorithm => false,
    }
  }

  /// Return `true` when the protocol algorithm is unsupported by RSA verification.
  ///
  /// This includes unknown, non-RSA, ambiguous, SHA-1, and deliberately disabled
  /// legacy identifiers.
  #[inline]
  #[must_use]
  pub const fn is_unsupported_algorithm(self) -> bool {
    match self {
      Self::MalformedAlgorithmIdentifier => false,
      Self::UnsupportedAlgorithm => true,
    }
  }
}

impl fmt::Display for RsaProtocolAlgorithmError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(match self {
      Self::MalformedAlgorithmIdentifier => "malformed RSA protocol algorithm identifier",
      Self::UnsupportedAlgorithm => "unsupported RSA protocol algorithm",
    })
  }
}

impl core::error::Error for RsaProtocolAlgorithmError {}

/// RSA algorithms supported by the JWT/JWS policy boundary.
///
/// This type is intentionally not constructible from a string. Applications
/// choose one variant from trusted verifier configuration, then pass the
/// peer-controlled JOSE `alg` value only to [`RsaJwtVerifier::verify`] for an
/// exact match. A token therefore cannot select or widen the verifier's
/// cryptographic policy.
///
/// Runtime strings cannot construct policy:
///
/// ```compile_fail
/// use rscrypto::RsaJwtAlgorithm;
///
/// let algorithm: RsaJwtAlgorithm = "PS256".parse().unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RsaJwtAlgorithm {
  /// RSASSA-PKCS1-v1_5 with SHA-256 (`RS256`).
  Rs256,
  /// RSASSA-PKCS1-v1_5 with SHA-384 (`RS384`).
  Rs384,
  /// RSASSA-PKCS1-v1_5 with SHA-512 (`RS512`).
  Rs512,
  /// RSASSA-PSS with SHA-256 and a 32-byte salt (`PS256`).
  Ps256,
  /// RSASSA-PSS with SHA-384 and a 48-byte salt (`PS384`).
  Ps384,
  /// RSASSA-PSS with SHA-512 and a 64-byte salt (`PS512`).
  Ps512,
}

/// Typed RSA signature profile.
///
/// This is the primitive-layer selector adapters should map to after they have
/// parsed protocol-specific identifiers such as TLS `SignatureScheme`, COSE
/// algorithm IDs, or X.509 algorithm parameters. JWT/JWS uses the narrower
/// [`RsaJwtAlgorithm`] and [`RsaJwtVerifier`] boundary so a token cannot select
/// this profile. Unsupported protocol algorithms, SHA-1 profiles, and ambiguous
/// parameter encodings should fail before constructing this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RsaSignatureProfile {
  /// RSASSA-PSS with a typed hash/MGF1 profile and explicit salt length.
  Pss {
    /// SHA-2 hash/MGF1 profile.
    profile: RsaPssProfile,
    /// Required PSS salt length in bytes.
    salt_len: usize,
  },
  /// RSASSA-PKCS1-v1_5 with an exact SHA-2 `DigestInfo`.
  Pkcs1v15(RsaPkcs1v15Profile),
}

/// Public exponent policy for RSA public-key parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RsaPublicExponentPolicy {
  /// Accept only the common `65537` exponent.
  Common65537,
  /// Accept `3`, `17`, and `65537`.
  ///
  /// This exists for verifying and importing deployed keys. Keep signing keys
  /// at `65537` unless a protocol adapter has a concrete compatibility reason
  /// to loosen it.
  LegacySmallFermat,
  /// Accept any odd exponent at least `3` that fits in `u64`.
  ///
  /// This is for standards vectors and compatibility imports. It is not the
  /// default because new RSA material should keep using `65537`.
  LegacyOdd,
}

impl RsaPublicExponentPolicy {
  #[inline]
  #[must_use]
  const fn accepts(self, value: u64) -> bool {
    match self {
      Self::Common65537 => value == 65_537,
      Self::LegacySmallFermat => value == 3 || value == 17 || value == 65_537,
      Self::LegacyOdd => value >= 3 && value & 1 == 1,
    }
  }
}

/// RSA key validation policy (moduli and public exponents) for any key role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RsaPublicKeyPolicy {
  min_modulus_bits: usize,
  max_modulus_bits: usize,
  exponent_policy: RsaPublicExponentPolicy,
}

impl Default for RsaPublicKeyPolicy {
  #[inline]
  fn default() -> Self {
    Self::modulus_3072_to_8192()
  }
}

impl RsaPublicKeyPolicy {
  /// RSA-2048 through RSA-8192, exponent `65537`.
  pub const MODULUS_2048_TO_8192: Self = Self {
    min_modulus_bits: MIN_RSA_MODULUS_BITS,
    max_modulus_bits: 8192,
    exponent_policy: RsaPublicExponentPolicy::Common65537,
  };

  /// RSA-3072 through RSA-8192, exponent `65537`.
  pub const MODULUS_3072_TO_8192: Self = Self {
    min_modulus_bits: 3072,
    max_modulus_bits: 8192,
    exponent_policy: RsaPublicExponentPolicy::Common65537,
  };

  /// Accept RSA-2048 through RSA-8192 with exponent `65537`.
  ///
  /// RSA-2048 gives about 112-bit security (NIST SP 800-57 Part 1). Use it for
  /// deployed RSA-2048 material, such as RSA-2048 certificates and their keys.
  #[inline]
  #[must_use]
  pub const fn modulus_2048_to_8192() -> Self {
    Self::MODULUS_2048_TO_8192
  }

  /// Accept RSA-3072 through RSA-8192 with exponent `65537`.
  ///
  /// RSA-3072 gives about 128-bit security (NIST SP 800-57 Part 1). This is
  /// the default policy for parsers and key generation.
  #[inline]
  #[must_use]
  pub const fn modulus_3072_to_8192() -> Self {
    Self::MODULUS_3072_TO_8192
  }

  /// Return this policy with a different minimum modulus length.
  #[inline]
  #[must_use]
  pub const fn with_min_modulus_bits(mut self, bits: usize) -> Self {
    self.min_modulus_bits = if bits < MIN_RSA_MODULUS_BITS {
      MIN_RSA_MODULUS_BITS
    } else {
      bits
    };
    self
  }

  /// Return this policy with a different maximum modulus length.
  #[inline]
  #[must_use]
  pub const fn with_max_modulus_bits(mut self, bits: usize) -> Self {
    self.max_modulus_bits = bits;
    self
  }

  /// Return this policy while accepting legacy exponents `3` and `17`.
  #[inline]
  #[must_use]
  pub const fn allow_legacy_small_exponents(mut self) -> Self {
    self.exponent_policy = RsaPublicExponentPolicy::LegacySmallFermat;
    self
  }

  /// Return this policy while accepting any odd public exponent that fits in `u64`.
  ///
  /// This exists for standards-vector coverage and compatibility verification
  /// of unusual deployed keys. It should not be used for newly minted RSA keys.
  #[inline]
  #[must_use]
  pub const fn allow_legacy_odd_exponents(mut self) -> Self {
    self.exponent_policy = RsaPublicExponentPolicy::LegacyOdd;
    self
  }

  /// Minimum accepted modulus bit length.
  #[inline]
  #[must_use]
  pub const fn min_modulus_bits(&self) -> usize {
    self.min_modulus_bits
  }

  /// Maximum accepted modulus bit length.
  #[inline]
  #[must_use]
  pub const fn max_modulus_bits(&self) -> usize {
    self.max_modulus_bits
  }
}

/// Validated RSA public exponent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RsaPublicExponent(u64);

impl RsaPublicExponent {
  /// Return the exponent as `u64`.
  #[inline]
  #[must_use]
  pub const fn as_u64(self) -> u64 {
    self.0
  }
}

/// Validated RSA public key.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct RsaPublicKey {
  modulus: RsaPublicModulus,
  exponent: RsaPublicExponent,
}

/// Validated RSA private key.
///
/// Imports require two conventional half-modulus-width factors that pass
/// trial division and a 32-base Miller-Rabin probable-prime screen. The
/// screening detects weak factor shapes and compositeness evidence; it is not
/// a mathematical proof of primality.
///
/// Private operations use caller-provided entropy, OS-backed entropy through
/// the `getrandom` feature, or an explicit caller-supplied blinding factor and
/// modular inverse for deterministic tests and constrained integrations.
pub struct RsaPrivateKey {
  components: RsaPrivateKeyComponents,
}

/// Caller-owned scratch for RSA private operations.
///
/// Reusing this scratch avoids top-level private-operation buffer allocation
/// for caller-random and deterministic blinding APIs. Scratch is bound to a
/// modulus width, not a specific key; using it with a different width returns
/// [`RsaPrivateOpError::InvalidScratch`].
pub struct RsaPrivateScratch {
  encoded: SecretBigEndianBuffer,
  salt: SecretBigEndianBuffer,
  blinding_factor: SecretBigEndianBuffer,
  blinding_inverse: SecretBigEndianBuffer,
  inverse_mask: SecretBigEndianBuffer,
  blinding_power: SecretBigEndianBuffer,
  blinded: SecretBigEndianBuffer,
  blinded_private_result: SecretBigEndianBuffer,
  checked: SecretBigEndianBuffer,
  one: SecretBigEndianBuffer,
  public_scratch: RsaPublicScratch,
  mul_scratch: RsaPrivateMulScratch,
  inverse_scratch: RsaPrivateInverseScratch,
  exponent_scratch: RsaPrivateExponentScratch,
}

/// RSA signer bound to a concrete signature profile.
#[derive(Clone, Copy)]
pub struct RsaSignatureSigner<'a> {
  key: &'a RsaPrivateKey,
  profile: RsaSignatureProfile,
}

/// RSA verifier bound to a concrete signature profile.
#[derive(Clone, Copy)]
pub struct RsaSignatureVerifier<'a> {
  key: &'a RsaPublicKey,
  profile: RsaSignatureProfile,
}

/// RSA JWT/JWS verifier bound to one key and one locally selected algorithm.
///
/// A JOSE provider must parse the protected header, reject duplicate members
/// and malformed serialization, and pass the single decoded `alg` string to
/// [`Self::verify`]. This verifier checks that string against its fixed policy;
/// it never maps peer metadata to a signature profile.
#[derive(Clone, Copy)]
pub struct RsaJwtVerifier<'a> {
  key: &'a RsaPublicKey,
  algorithm: RsaJwtAlgorithm,
}

/// Borrowed RSA private-key CRT components.
///
/// These fields contain private key material. Keep values canonical unsigned
/// big-endian, without leading zero padding. Both factors must have the
/// conventional half-modulus bit length and pass the private-import
/// probable-prime screen.
#[derive(Clone, Copy)]
pub struct RsaPrivateKeyParts<'a> {
  /// RSA modulus `n`.
  pub modulus: &'a [u8],
  /// RSA public exponent `e`.
  pub public_exponent: u64,
  /// RSA private exponent `d`.
  pub private_exponent: &'a [u8],
  /// First prime factor `p`.
  pub prime_p: &'a [u8],
  /// Second prime factor `q`.
  pub prime_q: &'a [u8],
  /// CRT exponent `d mod (p - 1)`.
  pub exponent_p: &'a [u8],
  /// CRT exponent `d mod (q - 1)`.
  pub exponent_q: &'a [u8],
  /// CRT coefficient `q^-1 mod p`.
  pub coefficient: &'a [u8],
}

impl fmt::Debug for RsaPrivateKeyParts<'_> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("RsaPrivateKeyParts")
      .field("modulus_bits", &unsigned_be_bit_len(self.modulus))
      .field("public_exponent", &self.public_exponent)
      .field("private_exponent", &"****")
      .field("prime_p", &"****")
      .field("prime_q", &"****")
      .field("exponent_p", &"****")
      .field("exponent_q", &"****")
      .field("coefficient", &"****")
      .finish()
  }
}

/// Expert RSA capabilities that normal application code does not need.
pub mod expert;

impl expert::RsaPkcs1v15Decrypt for RsaPrivateKey {
  #[cfg(feature = "getrandom")]
  fn decrypt_pkcs1v15(&self, ciphertext: &[u8], out: &mut [u8]) -> Result<usize, RsaPrivateOpError> {
    crate::traits::ct::with_data_independent_timing(|| {
      self.reject_short_decryption_output(RSAES_PKCS1V15_OVERHEAD, out)?;
      let result = self.components.random_blinding_factor().and_then(|blinding| {
        self.components.decrypt_pkcs1v15_with_blinding_factor(
          ciphertext,
          RsaBlindingPair::trusted(blinding.factor(), blinding.inverse()),
          out,
        )
      });
      clear_output_on_error(result, out)
    })
  }

  #[cfg(feature = "getrandom")]
  fn decrypt_pkcs1v15_with_scratch(
    &self,
    ciphertext: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<usize, RsaPrivateOpError> {
    crate::traits::ct::with_data_independent_timing(|| {
      self.reject_short_decryption_output(RSAES_PKCS1V15_OVERHEAD, out)?;
      let result = self
        .components
        .random_blinding_factor_into_scratch(scratch)
        .and_then(|()| {
          self
            .components
            .decrypt_pkcs1v15_with_stored_blinding_and_scratch(ciphertext, out, scratch)
        });
      scratch.clear();
      clear_decryption_output_on_error(result, out)
    })
  }

  fn decrypt_pkcs1v15_with_blinding_factor(
    &self,
    ciphertext: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
  ) -> Result<usize, RsaPrivateOpError> {
    crate::traits::ct::with_data_independent_timing(|| {
      self.reject_short_decryption_output(RSAES_PKCS1V15_OVERHEAD, out)?;
      let result = self
        .components
        .decrypt_pkcs1v15_with_blinding_factor(ciphertext, blinding, out);
      clear_decryption_output_on_error(result, out)
    })
  }

  fn decrypt_pkcs1v15_with_blinding_factor_and_scratch(
    &self,
    ciphertext: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<usize, RsaPrivateOpError> {
    crate::traits::ct::with_data_independent_timing(|| {
      self.reject_short_decryption_output(RSAES_PKCS1V15_OVERHEAD, out)?;
      let result = self
        .components
        .decrypt_pkcs1v15_with_blinding_factor_and_scratch(ciphertext, blinding, out, scratch);
      clear_decryption_output_on_error(result, out)
    })
  }
}

impl fmt::Debug for RsaPrivateKey {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("RsaPrivateKey")
      .field("public", self.public_key())
      .field("private_components", &"****")
      .finish_non_exhaustive()
  }
}

// Private-key component storage and import validation. Import validation may
// branch on private input while rejecting malformed key material; it is kept
// separate from private-operation exponentiation so validation arithmetic does
// not become signing, decryption, blinding, CRT recombination, or key
// generation arithmetic by accident.
struct RsaPrivateKeyComponents {
  public: RsaPublicKey,
  private_exponent: SecretBigEndianInteger,
  prime_p: SecretBigEndianInteger,
  prime_q: SecretBigEndianInteger,
  prime_p_modulus: RsaPublicModulus,
  prime_q_modulus: RsaPublicModulus,
  exponent_p: SecretBigEndianInteger,
  exponent_q: SecretBigEndianInteger,
  coefficient: SecretBigEndianInteger,
}

impl fmt::Debug for RsaPrivateKeyComponents {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("RsaPrivateKeyComponents")
      .field("modulus_bits", &self.public.modulus_bits())
      .field("public_exponent", &self.public.public_exponent().as_u64())
      .field("private_exponent", &"****")
      .field("prime_p", &"****")
      .field("prime_q", &"****")
      .field("prime_p_modulus", &"****")
      .field("prime_q_modulus", &"****")
      .field("exponent_p", &"****")
      .field("exponent_q", &"****")
      .field("coefficient", &"****")
      .finish()
  }
}

/// Signatures in one blinding epoch of [`RsaSigningWorkspace`].
const RSA_SIGNING_WORKSPACE_EPOCH_USES: u8 = 32;

/// Caller-owned state for repeated RSA signing with one key: private-operation
/// scratch and a blinding pair that later signatures reuse.
///
/// Every RSA private operation multiplies its input by `rᵉ mod n` for a
/// blinding factor `r`, then multiplies the result by `r⁻¹ mod n`. Without a
/// workspace, each signature draws a new `r` from its entropy source, inverts
/// it, and computes `rᵉ`. With a workspace, one draw starts an epoch of up to
/// 32 signatures. The first signature uses `r`. Each later signature squares
/// both cached values first, so the `k`-th signature of an epoch uses
/// `r^(2^(k-1))`. The squared values stay consistent: `(r²)ᵉ` and `(r²)⁻¹`.
/// Reuse needs no entropy, no modular inverse, and no `rᵉ` computation. AWS-LC
/// also squares its cached blinding pair between refreshes and draws a new one
/// every 32 uses.
///
/// Create a workspace with [`RsaPrivateKey::signing_workspace`], and pass it to
/// [`RsaPrivateKey::sign_signature_with_random_fill_and_workspace`] or, with
/// `getrandom`, `RsaPrivateKey::sign_signature_with_workspace`. A signature
/// through the workspace equals the signature of the uncached API for the same
/// key, message, and salt. A workspace belongs to the key that created it:
/// signing with another key returns [`RsaPrivateOpError::InvalidScratch`].
///
/// # Refresh
///
/// A signature draws a new factor from its blinding source, through the same
/// checked path as the uncached APIs, when any of these holds:
///
/// - the workspace is new, or its 32 uses are spent;
/// - the previous operation through the workspace failed. Every error clears
///   the pair;
/// - the [process origin](crate::platform::ProcessOrigin) of the pair is not
///   current, because a process fork copied the workspace. A forked child
///   therefore does not continue its parent's blinding sequence.
///
/// The schedule depends only on the public use count and the process origin.
///
/// # Security
///
/// - The factors within one epoch are correlated: each is the square of the
///   previous one. Their secrecy rests on the epoch's random `r`. Use the
///   uncached APIs if every signature needs an independent factor.
/// - The workspace keeps `rᵉ mod n` and `r⁻¹ mod n` between signatures. It is
///   secret state: it is not `Clone`, its `Debug` output shows only the width
///   and the remaining use count, and drop zeroizes both values. Signing clears
///   the scratch before it returns.
/// - After a fork, the new factor is fresh only when the blinding source is
///   fresh. A source that the fork also copied, such as a `ChaCha20Rng`, gives
///   the child its parent's next factor; refuse such a source in the child.
///   Fork detection has the targets and limits of
///   [`ProcessOrigin`]. Do not use one
///   workspace in two copies of a process image, such as a restored
///   virtual-machine snapshot.
/// - Signing borrows the workspace mutably. Give each thread its own workspace.
pub struct RsaSigningWorkspace {
  scratch: RsaPrivateScratch,
  /// Public modulus of the key that owns this workspace.
  modulus: Box<[u8]>,
  /// Public exponent of that key; the cached power is raised to it.
  public_exponent: u64,
  /// `rᵉ mod n` for the factor `r` of the latest signature.
  blinding_power: SecretBigEndianBuffer,
  /// `r⁻¹ mod n` for the same factor.
  blinding_inverse: SecretBigEndianBuffer,
  /// Signatures left in the epoch. Zero means that the next signature refreshes.
  remaining: u8,
  /// The process that drew the epoch's factor.
  origin: ProcessOrigin,
}

struct SecretBigEndianInteger {
  bytes: Box<[u8]>,
}

impl SecretBigEndianInteger {
  fn new(bytes: &[u8]) -> Result<Self, RsaKeyError> {
    if is_zero_unsigned_be(bytes) {
      return Err(RsaKeyError::InvalidModulus);
    }
    Ok(Self {
      bytes: Box::from(bytes),
    })
  }

  #[cfg(feature = "getrandom")]
  fn from_vec(mut bytes: Vec<u8>) -> Result<Self, RsaKeyError> {
    if is_zero_unsigned_be(&bytes) {
      ct::zeroize(&mut bytes);
      return Err(RsaKeyError::InvalidModulus);
    }
    Ok(Self {
      bytes: bytes.into_boxed_slice(),
    })
  }

  fn new_fixed_width(bytes: &[u8], width: usize) -> Result<Self, RsaKeyError> {
    if is_zero_unsigned_be(bytes) || bytes.len() > width {
      return Err(RsaKeyError::InvalidModulus);
    }
    let mut fixed = vec![0u8; width];
    let offset = width.strict_sub(bytes.len());
    let Some(dst) = fixed.get_mut(offset..) else {
      return Err(RsaKeyError::InvalidModulus);
    };
    dst.copy_from_slice(bytes);
    Ok(Self {
      bytes: fixed.into_boxed_slice(),
    })
  }

  #[inline]
  fn as_bytes(&self) -> &[u8] {
    &self.bytes
  }
}

impl fmt::Debug for SecretBigEndianInteger {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("SecretBigEndianInteger(****)")
  }
}

impl Drop for SecretBigEndianInteger {
  fn drop(&mut self) {
    ct::zeroize(&mut self.bytes);
  }
}

struct SecretBigEndianBuffer {
  bytes: Vec<u8>,
}

impl SecretBigEndianBuffer {
  #[cfg(feature = "getrandom")]
  fn new(bytes: Vec<u8>) -> Self {
    Self { bytes }
  }

  fn zeroed(len: usize) -> Self {
    Self { bytes: vec![0u8; len] }
  }

  #[inline]
  fn as_slice(&self) -> &[u8] {
    &self.bytes
  }

  #[inline]
  fn as_mut_slice(&mut self) -> &mut [u8] {
    &mut self.bytes
  }

  #[cfg(feature = "getrandom")]
  fn into_vec(mut self) -> Vec<u8> {
    core::mem::take(&mut self.bytes)
  }
}

impl Drop for SecretBigEndianBuffer {
  fn drop(&mut self) {
    ct::zeroize(&mut self.bytes);
  }
}

#[cfg(feature = "getrandom")]
struct RsaBlindingFactor {
  factor: SecretBigEndianBuffer,
  inverse: SecretBigEndianBuffer,
}

/// Borrowed inputs for a caller-blinded RSA private operation.
///
/// The factor and inverse must be fixed-width, modulus-sized representatives
/// satisfying `factor * inverse == 1 mod n`. Construction only borrows the
/// inputs; each private operation validates them against its key before secret
/// exponentiation. This type does not own, copy, or zeroize the borrowed bytes.
#[derive(Clone, Copy)]
pub struct RsaBlindingPair<'a> {
  factor: &'a [u8],
  inverse: &'a [u8],
  validate: bool,
}

struct SecretLimbs {
  limbs: Vec<u64>,
}

impl SecretLimbs {
  fn from_be(bytes: &[u8]) -> Self {
    Self {
      limbs: limbs_from_be(bytes),
    }
  }

  fn zeroed(len: usize) -> Self {
    Self { limbs: vec![0; len] }
  }

  #[cfg(feature = "getrandom")]
  fn from_limbs(limbs: &[u64]) -> Self {
    Self { limbs: limbs.to_vec() }
  }

  #[inline]
  fn as_slice(&self) -> &[u64] {
    &self.limbs
  }

  #[inline]
  fn as_mut_slice(&mut self) -> &mut [u64] {
    &mut self.limbs
  }
}

impl Drop for SecretLimbs {
  fn drop(&mut self) {
    ct::zeroize_words(&mut self.limbs);
  }
}

#[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
struct SecretI31 {
  words: Vec<u32>,
}

#[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
impl SecretI31 {
  fn zeroed(len: usize) -> Self {
    Self { words: vec![0; len] }
  }

  #[inline]
  fn as_mut_slice(&mut self) -> &mut [u32] {
    &mut self.words
  }
}

#[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
impl Drop for SecretI31 {
  fn drop(&mut self) {
    ct::zeroize_words(&mut self.words);
  }
}

/// Blinding-inverse workspace for the selected inverse.
///
/// RISC-V and IBM Z use the 31-bit batched inverse, whose multiply is selected per target in
/// `private_i31_mul_u64_low`. Other targets use 62-bit batched divsteps.
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
type RsaPrivateInverseScratch = RsaPrivateI31InverseScratch;

#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x")))]
type RsaPrivateInverseScratch = RsaPrivateS62InverseScratch;

#[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
struct RsaPrivateI31InverseScratch {
  a: SecretI31,
  b: SecretI31,
  u: SecretI31,
  v: SecretI31,
  modulus: SecretI31,
  bit_len: usize,
}

#[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
struct RsaPrivateI31InverseWorkspace<'a> {
  a: &'a mut [u32],
  b: &'a mut [u32],
  u: &'a mut [u32],
  v: &'a mut [u32],
  modulus: &'a mut [u32],
}

#[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
impl RsaPrivateI31InverseScratch {
  fn new(bit_len: usize) -> Self {
    let word_count = private_i31_word_count(bit_len);
    Self {
      a: SecretI31::zeroed(word_count),
      b: SecretI31::zeroed(word_count),
      u: SecretI31::zeroed(word_count),
      v: SecretI31::zeroed(word_count),
      modulus: SecretI31::zeroed(word_count),
      bit_len,
    }
  }

  fn workspace(&mut self, bit_len: usize) -> Result<RsaPrivateI31InverseWorkspace<'_>, RsaPrivateOpError> {
    if bit_len != self.bit_len {
      return Err(RsaPrivateOpError::InvalidScratch);
    }
    let word_count = private_i31_word_count(bit_len);
    Ok(RsaPrivateI31InverseWorkspace {
      a: private_i31_scratch_prefix(&mut self.a, word_count)?,
      b: private_i31_scratch_prefix(&mut self.b, word_count)?,
      u: private_i31_scratch_prefix(&mut self.u, word_count)?,
      v: private_i31_scratch_prefix(&mut self.v, word_count)?,
      modulus: private_i31_scratch_prefix(&mut self.modulus, word_count)?,
    })
  }

  fn clear(&mut self) {
    ct::zeroize_words(self.a.as_mut_slice());
    ct::zeroize_words(self.b.as_mut_slice());
    ct::zeroize_words(self.u.as_mut_slice());
    ct::zeroize_words(self.v.as_mut_slice());
    ct::zeroize_words(self.modulus.as_mut_slice());
  }
}

#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x")))]
struct RsaPrivateS62InverseScratch {
  f: SecretLimbs,
  g: SecretLimbs,
  d: SecretLimbs,
  e: SecretLimbs,
  modulus: SecretLimbs,
  bit_len: usize,
}

#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x")))]
struct RsaPrivateS62InverseWorkspace<'a> {
  f: &'a mut [u64],
  g: &'a mut [u64],
  d: &'a mut [u64],
  e: &'a mut [u64],
  modulus: &'a mut [u64],
}

#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x")))]
impl RsaPrivateS62InverseScratch {
  fn new(bit_len: usize) -> Self {
    let limb_count = private_s62_limb_count(bit_len);
    Self {
      f: SecretLimbs::zeroed(limb_count),
      g: SecretLimbs::zeroed(limb_count),
      d: SecretLimbs::zeroed(limb_count),
      e: SecretLimbs::zeroed(limb_count),
      modulus: SecretLimbs::zeroed(limb_count),
      bit_len,
    }
  }

  fn workspace(&mut self, bit_len: usize) -> Result<RsaPrivateS62InverseWorkspace<'_>, RsaPrivateOpError> {
    if bit_len != self.bit_len {
      return Err(RsaPrivateOpError::InvalidScratch);
    }
    let limb_count = private_s62_limb_count(bit_len);
    Ok(RsaPrivateS62InverseWorkspace {
      f: private_scratch_prefix(&mut self.f, limb_count)?,
      g: private_scratch_prefix(&mut self.g, limb_count)?,
      d: private_scratch_prefix(&mut self.d, limb_count)?,
      e: private_scratch_prefix(&mut self.e, limb_count)?,
      modulus: private_scratch_prefix(&mut self.modulus, limb_count)?,
    })
  }

  fn clear(&mut self) {
    ct::zeroize_words(self.f.as_mut_slice());
    ct::zeroize_words(self.g.as_mut_slice());
    ct::zeroize_words(self.d.as_mut_slice());
    ct::zeroize_words(self.e.as_mut_slice());
    ct::zeroize_words(self.modulus.as_mut_slice());
  }
}

struct RsaPrivateMulScratch {
  t: SecretLimbs,
  left_limbs: SecretLimbs,
  right_limbs: SecretLimbs,
  left_mont: SecretLimbs,
  right_mont: SecretLimbs,
  product_mont: SecretLimbs,
  product: SecretLimbs,
  max_limb_count: usize,
}

impl RsaPrivateMulScratch {
  fn new(max_limb_count: usize) -> Self {
    Self {
      t: SecretLimbs::zeroed(max_limb_count.strict_mul(2).strict_add(2)),
      left_limbs: SecretLimbs::zeroed(max_limb_count),
      right_limbs: SecretLimbs::zeroed(max_limb_count),
      left_mont: SecretLimbs::zeroed(max_limb_count),
      right_mont: SecretLimbs::zeroed(max_limb_count),
      product_mont: SecretLimbs::zeroed(max_limb_count),
      product: SecretLimbs::zeroed(max_limb_count),
      max_limb_count,
    }
  }

  fn ensure_limb_count(&self, limb_count: usize) -> Result<(), RsaPrivateOpError> {
    if limb_count <= self.max_limb_count
      && self.t.as_slice().len() >= limb_count.strict_mul(2).strict_add(2)
      && self.left_limbs.as_slice().len() >= limb_count
      && self.right_limbs.as_slice().len() >= limb_count
      && self.left_mont.as_slice().len() >= limb_count
      && self.right_mont.as_slice().len() >= limb_count
      && self.product_mont.as_slice().len() >= limb_count
      && self.product.as_slice().len() >= limb_count
    {
      Ok(())
    } else {
      Err(RsaPrivateOpError::InvalidScratch)
    }
  }

  fn clear(&mut self) {
    ct::zeroize_words(self.t.as_mut_slice());
    ct::zeroize_words(self.left_limbs.as_mut_slice());
    ct::zeroize_words(self.right_limbs.as_mut_slice());
    ct::zeroize_words(self.left_mont.as_mut_slice());
    ct::zeroize_words(self.right_mont.as_mut_slice());
    ct::zeroize_words(self.product_mont.as_mut_slice());
    ct::zeroize_words(self.product.as_mut_slice());
  }
}

struct RsaPrivateExponentScratch {
  t: SecretLimbs,
  representative: SecretLimbs,
  one: SecretLimbs,
  base: SecretLimbs,
  acc: SecretLimbs,
  squared: SecretLimbs,
  multiplied: SecretLimbs,
  selected: SecretLimbs,
  reduced: SecretLimbs,
  table: SecretLimbs,
  // Radix-2^52 window table and digit buffers for the IFMA exponentiation; empty without it.
  ifma: SecretLimbs,
  // Radix-2^28 paired window table and buffers for the Apple NEON CRT pair; empty without it.
  pair: SecretLimbs,
  max_limb_count: usize,
}

struct RsaPrivateExponentWorkspace<'a> {
  t: &'a mut [u64],
  representative: &'a mut [u64],
  one: &'a mut [u64],
  base: &'a mut [u64],
  acc: &'a mut [u64],
  squared: &'a mut [u64],
  multiplied: &'a mut [u64],
  selected: &'a mut [u64],
  reduced: &'a mut [u64],
  table: &'a mut [u64],
  ifma: &'a mut [u64],
  #[cfg(all(
    target_arch = "aarch64",
    target_vendor = "apple",
    not(feature = "portable-only"),
    not(miri)
  ))]
  pair: &'a mut [u64],
}

impl RsaPrivateExponentScratch {
  fn new(max_limb_count: usize) -> Self {
    Self {
      t: SecretLimbs::zeroed(max_limb_count.strict_mul(2).strict_add(2)),
      representative: SecretLimbs::zeroed(max_limb_count),
      one: SecretLimbs::zeroed(max_limb_count),
      base: SecretLimbs::zeroed(max_limb_count),
      acc: SecretLimbs::zeroed(max_limb_count),
      squared: SecretLimbs::zeroed(max_limb_count),
      multiplied: SecretLimbs::zeroed(max_limb_count),
      selected: SecretLimbs::zeroed(max_limb_count),
      reduced: SecretLimbs::zeroed(max_limb_count),
      table: SecretLimbs::zeroed(max_limb_count.strict_mul(PRIVATE_FIXED_WINDOW_TABLE_ENTRIES)),
      ifma: SecretLimbs::zeroed(private_ifma_workspace_words(max_limb_count)),
      pair: SecretLimbs::zeroed(private_pair_workspace_words(max_limb_count)),
      max_limb_count,
    }
  }

  fn ensure_limb_count(&self, limb_count: usize) -> Result<(), RsaPrivateOpError> {
    if limb_count <= self.max_limb_count
      && self.t.as_slice().len() >= limb_count.strict_mul(2).strict_add(2)
      && self.representative.as_slice().len() >= limb_count
      && self.one.as_slice().len() >= limb_count
      && self.base.as_slice().len() >= limb_count
      && self.acc.as_slice().len() >= limb_count
      && self.squared.as_slice().len() >= limb_count
      && self.multiplied.as_slice().len() >= limb_count
      && self.selected.as_slice().len() >= limb_count
      && self.reduced.as_slice().len() >= limb_count
      && self.table.as_slice().len() >= limb_count.strict_mul(PRIVATE_FIXED_WINDOW_TABLE_ENTRIES)
    {
      Ok(())
    } else {
      Err(RsaPrivateOpError::InvalidScratch)
    }
  }

  fn workspace(&mut self, limb_count: usize) -> Result<RsaPrivateExponentWorkspace<'_>, RsaPrivateOpError> {
    self.ensure_limb_count(limb_count)?;
    let t_len = limb_count.strict_mul(2).strict_add(2);
    let table_len = limb_count.strict_mul(PRIVATE_FIXED_WINDOW_TABLE_ENTRIES);
    Ok(RsaPrivateExponentWorkspace {
      t: private_scratch_prefix(&mut self.t, t_len)?,
      representative: private_scratch_prefix(&mut self.representative, limb_count)?,
      one: private_scratch_prefix(&mut self.one, limb_count)?,
      base: private_scratch_prefix(&mut self.base, limb_count)?,
      acc: private_scratch_prefix(&mut self.acc, limb_count)?,
      squared: private_scratch_prefix(&mut self.squared, limb_count)?,
      multiplied: private_scratch_prefix(&mut self.multiplied, limb_count)?,
      selected: private_scratch_prefix(&mut self.selected, limb_count)?,
      reduced: private_scratch_prefix(&mut self.reduced, limb_count)?,
      table: private_scratch_prefix(&mut self.table, table_len)?,
      ifma: self.ifma.as_mut_slice(),
      #[cfg(all(
        target_arch = "aarch64",
        target_vendor = "apple",
        not(feature = "portable-only"),
        not(miri)
      ))]
      pair: self.pair.as_mut_slice(),
    })
  }

  fn workspace_with_table_entries<const ENTRIES: usize>(
    &mut self,
    limb_count: usize,
  ) -> Result<RsaPrivateExponentWorkspace<'_>, RsaPrivateOpError> {
    self.ensure_limb_count(limb_count)?;
    let t_len = limb_count.strict_mul(2).strict_add(2);
    let table_len = limb_count.strict_mul(ENTRIES);
    Ok(RsaPrivateExponentWorkspace {
      t: private_scratch_prefix(&mut self.t, t_len)?,
      representative: private_scratch_prefix(&mut self.representative, limb_count)?,
      one: private_scratch_prefix(&mut self.one, limb_count)?,
      base: private_scratch_prefix(&mut self.base, limb_count)?,
      acc: private_scratch_prefix(&mut self.acc, limb_count)?,
      squared: private_scratch_prefix(&mut self.squared, limb_count)?,
      multiplied: private_scratch_prefix(&mut self.multiplied, limb_count)?,
      selected: private_scratch_prefix(&mut self.selected, limb_count)?,
      reduced: private_scratch_prefix(&mut self.reduced, limb_count)?,
      table: private_scratch_prefix(&mut self.table, table_len)?,
      ifma: self.ifma.as_mut_slice(),
      #[cfg(all(
        target_arch = "aarch64",
        target_vendor = "apple",
        not(feature = "portable-only"),
        not(miri)
      ))]
      pair: self.pair.as_mut_slice(),
    })
  }

  fn clear(&mut self) {
    ct::zeroize_words(self.t.as_mut_slice());
    ct::zeroize_words(self.representative.as_mut_slice());
    ct::zeroize_words(self.one.as_mut_slice());
    ct::zeroize_words(self.base.as_mut_slice());
    ct::zeroize_words(self.acc.as_mut_slice());
    ct::zeroize_words(self.squared.as_mut_slice());
    ct::zeroize_words(self.multiplied.as_mut_slice());
    ct::zeroize_words(self.selected.as_mut_slice());
    ct::zeroize_words(self.reduced.as_mut_slice());
    ct::zeroize_words(self.table.as_mut_slice());
    ct::zeroize_words(self.ifma.as_mut_slice());
    ct::zeroize_words(self.pair.as_mut_slice());
  }
}

/// X.509 RSA public-key algorithm constraints.
///
/// `rsaEncryption` keys are unconstrained RSA public keys. `id-RSASSA-PSS`
/// keys are restricted to PSS signatures, and may additionally constrain the
/// hash/MGF1 profile and minimum salt length through their SPKI parameters.
/// RFC 4055 section 3.3 requires signature parameters to match restricted key
/// parameters except that signature `saltLength` may be greater than or equal
/// to the key parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RsaX509PublicKeyAlgorithm {
  /// X.509 `rsaEncryption` with `NULL` parameters.
  RsaEncryption,
  /// X.509 `id-RSASSA-PSS` with absent parameters: PSS-only, no parameter restriction.
  RsaPss,
  /// X.509 `id-RSASSA-PSS` with explicit supported parameter restrictions.
  RsaPssRestricted {
    /// Required SHA-2 hash/MGF1 profile.
    profile: RsaPssProfile,
    /// Minimum accepted PSS salt length from the key parameters.
    minimum_salt_len: usize,
  },
}

const RSA_TLS_SIGNATURE_SCHEME_CAPACITY: usize = 6;

/// Fixed-capacity TLS RSA `SignatureScheme` advertisement list.
///
/// Providers can use this to advertise only schemes the current RSA key
/// algorithm can verify. The list is allocation-free and preserves insertion
/// order; unused slots are not exposed by [`Self::iter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RsaTlsSignatureSchemes {
  schemes: [u16; RSA_TLS_SIGNATURE_SCHEME_CAPACITY],
  len: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RsaTlsPublicKeyAlgorithm {
  RsaEncryption,
  RsaPss,
}

/// RSA public key paired with its X.509 SPKI algorithm constraints.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct RsaX509PublicKey {
  key: RsaPublicKey,
  algorithm: RsaX509PublicKeyAlgorithm,
}

impl fmt::Debug for RsaPublicKey {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("RsaPublicKey")
      .field("modulus_bits", &self.modulus.bits)
      .field("public_exponent", &self.exponent.as_u64())
      .finish_non_exhaustive()
  }
}

/// Caller-owned scratch for RSA public operations.
pub struct RsaPublicScratch {
  montgomery: Box<[u64]>,
  limbs: Box<[u64]>,
  bytes: Box<[u8]>,
  limb_count: usize,
  byte_count: usize,
}

struct RsaPublicArithmeticScratch<'a> {
  modulus: &'a mut [u64],
  r2: &'a mut [u64],
  t: &'a mut [u64],
  x: &'a mut [u64],
  tmp: &'a mut [u64],
  base: &'a mut [u64],
  acc: &'a mut [u64],
}

#[derive(Clone)]
struct RsaPublicModulus {
  bytes: Box<[u8]>,
  limbs: Box<[u64]>,
  r2: Option<Box<[u64]>>,
  bits: usize,
  n0: u64,
  /// `-limbs⁻¹ mod 2^256`, the four-limb Montgomery constant of the AArch64 kernels.
  #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
  n0_256: [u64; 4],
}

// DER AlgorithmIdentifier for rsaEncryption with NULL parameters.
const RSA_ENCRYPTION_ALGORITHM: &[u8] = &[0x30, 13, 6, 9, 0x2a, 0x86, 0x48, 0x86, 0xf7, 13, 1, 1, 1, 5, 0];

struct RsaPrivateKeyDerComponents<'a> {
  modulus: &'a [u8],
  public_exponent: RsaPublicExponent,
  private_exponent: &'a [u8],
  prime_p: &'a [u8],
  prime_q: &'a [u8],
  exponent_p: &'a [u8],
  exponent_q: &'a [u8],
  coefficient: &'a [u8],
}

#[cfg(feature = "getrandom")]
struct RsaKeygenDrbg {
  key: [u8; RSA_KEYGEN_DRBG_KEY_BYTES],
  value: [u8; RSA_KEYGEN_DRBG_OUT_BYTES],
}

/// HMAC-SHA256 over the concatenation of `data`, shared by the key-generation
/// DRBG and PKCS#1 v1.5 implicit rejection without the `hmac` feature.
fn hmac_sha256_parts<'a>(
  key: &[u8; HMAC_SHA256_OUT_BYTES],
  data: impl IntoIterator<Item = &'a [u8]>,
) -> [u8; HMAC_SHA256_OUT_BYTES] {
  let mut ipad = [0x36u8; HMAC_SHA256_BLOCK_BYTES];
  let mut opad = [0x5cu8; HMAC_SHA256_BLOCK_BYTES];
  for ((ipad_byte, opad_byte), key_byte) in ipad.iter_mut().zip(opad.iter_mut()).zip(key) {
    *ipad_byte ^= key_byte;
    *opad_byte ^= key_byte;
  }

  let mut inner = Sha256::new();
  inner.update(&ipad);
  for part in data {
    inner.update(part);
  }
  let mut inner_digest = inner.finalize_secret();

  let mut outer = Sha256::new();
  outer.update(&opad);
  outer.update(inner_digest.as_ref());
  let tag = outer.finalize_secret();

  ct::zeroize(&mut ipad);
  ct::zeroize(&mut opad);
  ct::zeroize(&mut inner_digest);
  tag
}

struct ParsedX509CertificateSignature<'a> {
  tbs_certificate_der: &'a [u8],
  signature_algorithm_der: &'a [u8],
  signature: &'a [u8],
}

const PKCS1V15_NONZERO_PADDING_MAX_RETRIES: usize = 128;

fn clear_decryption_output_on_error(
  result: Result<usize, RsaPrivateOpError>,
  out: &mut [u8],
) -> Result<usize, RsaPrivateOpError> {
  clear_output_on_error(result, out)
}

fn clear_output_on_error<T, E>(result: Result<T, E>, out: &mut [u8]) -> Result<T, E> {
  if result.is_err() {
    ct::zeroize(out);
  }
  result
}

fn ct_select_usize(zero: usize, one: usize, choice: u8) -> usize {
  // SECURITY: Keep the secret-derived mask opaque so LLVM cannot lower the
  // selection to a branch on targets without a conditional move, such as RISC-V.
  let mask = core::hint::black_box(0usize.wrapping_sub(usize::from(choice & 1)));
  (zero & !mask) | (one & mask)
}

#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x")))]
const PRIVATE_S62_MASK: u64 = u64::MAX >> 2;

#[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
const PRIVATE_I31_MASK: u32 = 0x7fff_ffff;

/// One CRT component's exponentiation: modulus, exponent, input and output.
type RsaCrtExponentiation<'a> = (&'a RsaPublicModulus, &'a [u8], &'a [u8], &'a mut [u8]);

struct RsaPrivateWindowWorkspace<'a> {
  t: &'a mut [u64],
  acc: &'a mut [u64],
  squared: &'a mut [u64],
  multiplied: &'a mut [u64],
  selected: &'a mut [u64],
}

type DerReader<'a> = der::DerReader<'a, RsaKeyError>;

#[path = "rsa_public_division.rs"]
mod rsa_public_division;

#[cfg(all(test, not(miri), feature = "portable-only", feature = "rsa"))]
mod binsec {
  use crate::binsec::*;

  #[unsafe(no_mangle)]
  #[inline(never)]
  pub(super) extern "C" fn ct_binsec_rsa_private_component_validation_32() -> ! {
    // SAFETY: These pointers reference fixed harness globals with static storage.
    let component = unsafe { array_from_global(&raw const CT_BINSEC_RSA_COMPONENT_32) };
    // SAFETY: These pointers reference fixed harness globals with static storage.
    let upper = unsafe { array_from_global(&raw const CT_BINSEC_RSA_UPPER_32) };
    // SAFETY: These pointers reference fixed harness globals with static storage.
    let other = unsafe { array_from_global(&raw const CT_BINSEC_RSA_OTHER_32) };
    let ok = super::component_test::check(&component, &upper, &other);
    ct_binsec_done(ok)
  }
}

#[cfg(all(
  test,
  feature = "full",
  feature = "parallel",
  feature = "getrandom",
  any(unix, windows),
  not(miri)
))]
pub(crate) mod ct_tests;

#[cfg(all(
  test,
  any(
    feature = "portable-only",
    all(
      feature = "full",
      feature = "parallel",
      feature = "getrandom",
      any(unix, windows),
      not(miri)
    )
  )
))]
mod component_test {
  use super::*;
  #[inline(always)]
  /// Evaluate the fixed-width RSA private-component checks used by diagnostic harnesses.
  ///
  /// Returns 1 when `component` is canonical (nonzero first byte), below `upper_bound`, distinct
  /// from `other`, and odd, else 0. Each check is an arithmetic bit, and the bits pass through
  /// `black_box` before the conjunction: without conditional moves, as on IBM Z's baseline CPU,
  /// LLVM otherwise lowers the conjunction to branches on the secret checks.
  pub(crate) fn check(component: &[u8; 32], upper_bound: &[u8; 32], other: &[u8; 32]) -> u8 {
    let canonical = core::hint::black_box(ct_nonzero_u8(component[0]));
    let less_than_bound = core::hint::black_box(ct_unsigned_be_lt_same_len(component, upper_bound));
    let distinct = core::hint::black_box(u8::from(!ct_slices_eq_public_shape(component, other)));
    let odd = component[31] & 1;
    canonical & less_than_bound & distinct & odd
  }
  /// Return 1 when `value` is nonzero, else 0, without a data-dependent comparison.
  pub(super) fn ct_nonzero_u8(value: u8) -> u8 {
    ((u16::from(value).wrapping_neg() >> 8) & 1) as u8
  }
}
