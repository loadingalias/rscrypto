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
//! let public_key = include_bytes!("../../benches/rsa_fixtures/rsa3072_spki.der");
//! let signature = include_bytes!("../../benches/rsa_fixtures/rsa3072_pss_sha256.sig");
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
//! encryption/decryption, private-key import, and private-key generation are
//! available through [`RsaPrivateKey`] and [`RsaPublicKey`]. Private operations
//! are blinded and fault-checked; RNG-backed APIs require the `getrandom`
//! feature.

use alloc::{boxed::Box, vec, vec::Vec};
use core::{
  fmt,
  hash::{Hash, Hasher},
};

use crate::{
  hashes::crypto::{Sha256, Sha384, Sha512},
  traits::{Digest, VerificationError, ct},
};

#[cfg(all(
  target_arch = "aarch64",
  target_os = "macos",
  not(feature = "portable-only"),
  not(miri)
))]
#[path = "rsa_aarch64_asm.rs"]
mod rsa_aarch64_asm;

#[cfg(all(
  target_arch = "x86_64",
  target_os = "linux",
  not(feature = "portable-only"),
  not(miri)
))]
#[path = "rsa_x86_64_asm.rs"]
mod rsa_x86_64_asm;

const TAG_SEQUENCE: u8 = 0x30;
const TAG_INTEGER: u8 = 0x02;
const TAG_BIT_STRING: u8 = 0x03;
const TAG_OCTET_STRING: u8 = 0x04;
const TAG_NULL: u8 = 0x05;
const TAG_OBJECT_IDENTIFIER: u8 = 0x06;
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
#[cfg(feature = "getrandom")]
const RSA_KEYGEN_PAIR_ATTEMPTS: usize = 64;
#[cfg(feature = "getrandom")]
const RSA_KEYGEN_MIN_PRIME_DISTANCE_SECURITY_MARGIN_BITS: usize = 100;
#[cfg(feature = "getrandom")]
const RSA_KEYGEN_DRBG_ENTROPY_BYTES: usize = 32;
#[cfg(feature = "getrandom")]
const RSA_KEYGEN_DRBG_NONCE_BYTES: usize = 16;
#[cfg(feature = "getrandom")]
const RSA_KEYGEN_DRBG_OUT_BYTES: usize = 32;
#[cfg(feature = "getrandom")]
const RSA_KEYGEN_DRBG_KEY_BYTES: usize = 32;
#[cfg(feature = "getrandom")]
const RSA_KEYGEN_DRBG_HMAC_BLOCK_BYTES: usize = 64;
#[cfg(feature = "getrandom")]
const RSA_KEYGEN_DRBG_PERSONALIZATION: &[u8] = b"rscrypto RSA FIPS 186-5 A.1.3 HMAC_DRBG";
#[cfg(feature = "getrandom")]
const RSA_KEYGEN_SQRT2_HALF_TOP64: [u8; 8] = [0xb5, 0x04, 0xf3, 0x33, 0xf9, 0xde, 0x64, 0x84];
#[cfg(feature = "getrandom")]
const RSA_KEYGEN_SMALL_PRIMES: &[u16] = &[
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
  /// The platform entropy source was unavailable.
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
  /// Requested modulus length is outside this implementation's RSA policy.
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
  /// Scratch storage was created for a different modulus width.
  InvalidScratch,
  /// The key is too small for the requested encoded message.
  MessageTooLong,
  /// A representative was not strictly smaller than the RSA modulus.
  RepresentativeOutOfRange,
  /// The supplied blinding factor and inverse are invalid for this modulus.
  InvalidBlindingFactor,
  /// The platform entropy source was unavailable.
  EntropyUnavailable,
  /// RSA decryption padding was invalid.
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

/// Typed RSA signature profile.
///
/// This is the primitive-layer selector adapters should map to after they have
/// parsed protocol-specific identifiers such as TLS `SignatureScheme`, JWT
/// `alg`, COSE algorithm IDs, or X.509 algorithm parameters. Unsupported
/// protocol algorithms, SHA-1 profiles, and ambiguous parameter encodings
/// should fail before constructing this type.
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

impl RsaSignatureProfile {
  /// Return a TLS 1.3-style PSS profile with salt length equal to digest length.
  #[inline]
  #[must_use]
  pub const fn pss(profile: RsaPssProfile) -> Self {
    Self::Pss {
      profile,
      salt_len: profile.digest_len(),
    }
  }

  /// Return a PSS profile with an explicitly parsed salt length.
  #[inline]
  #[must_use]
  pub const fn pss_with_salt_len(profile: RsaPssProfile, salt_len: usize) -> Self {
    Self::Pss { profile, salt_len }
  }

  /// Return a PKCS#1 v1.5 signature profile.
  #[inline]
  #[must_use]
  pub const fn pkcs1v15(profile: RsaPkcs1v15Profile) -> Self {
    Self::Pkcs1v15(profile)
  }

  /// Return the PSS profile and salt length when this is an RSASSA-PSS profile.
  #[inline]
  #[must_use]
  pub const fn pss_parts(self) -> Option<(RsaPssProfile, usize)> {
    match self {
      Self::Pss { profile, salt_len } => Some((profile, salt_len)),
      Self::Pkcs1v15(_) => None,
    }
  }

  /// Return the PKCS#1 v1.5 profile when this is an RSASSA-PKCS1-v1_5 profile.
  #[inline]
  #[must_use]
  pub const fn pkcs1v15_profile(self) -> Option<RsaPkcs1v15Profile> {
    match self {
      Self::Pss { .. } => None,
      Self::Pkcs1v15(profile) => Some(profile),
    }
  }

  /// Map a TLS 1.3 `SignatureScheme` for `CertificateVerify`.
  ///
  /// TLS 1.3 RSA handshake signatures are RSASSA-PSS only. Legacy
  /// RSASSA-PKCS1-v1_5 scheme IDs intentionally return
  /// [`RsaProtocolAlgorithmError::UnsupportedAlgorithm`] here, even though
  /// they may still be valid for certificate-chain signatures.
  ///
  /// # Errors
  ///
  /// Returns [`RsaProtocolAlgorithmError::UnsupportedAlgorithm`] for unknown,
  /// non-RSA, SHA-1, or PKCS#1 v1.5 TLS scheme IDs.
  #[inline]
  pub const fn from_tls13_signature_scheme(scheme: u16) -> Result<Self, RsaProtocolAlgorithmError> {
    match scheme {
      0x0804 | 0x0809 => Ok(Self::pss(RsaPssProfile::Sha256)),
      0x0805 | 0x080a => Ok(Self::pss(RsaPssProfile::Sha384)),
      0x0806 | 0x080b => Ok(Self::pss(RsaPssProfile::Sha512)),
      _ => Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm),
    }
  }

  /// Map a TLS certificate signature scheme ID to an RSA signature profile.
  ///
  /// This includes the SHA-2 PKCS#1 v1.5 identifiers used for legacy
  /// certificate-chain compatibility, plus the TLS 1.3 RSASSA-PSS identifiers.
  /// SHA-1 and MD5/SHA-1 identifiers are always rejected.
  ///
  /// # Errors
  ///
  /// Returns [`RsaProtocolAlgorithmError::UnsupportedAlgorithm`] for unknown,
  /// non-RSA, SHA-1, or MD5/SHA-1 TLS scheme IDs.
  #[inline]
  pub const fn from_tls_certificate_signature_scheme(scheme: u16) -> Result<Self, RsaProtocolAlgorithmError> {
    match scheme {
      0x0401 => Ok(Self::pkcs1v15(RsaPkcs1v15Profile::Sha256)),
      0x0501 => Ok(Self::pkcs1v15(RsaPkcs1v15Profile::Sha384)),
      0x0601 => Ok(Self::pkcs1v15(RsaPkcs1v15Profile::Sha512)),
      _ => Self::from_tls13_signature_scheme(scheme),
    }
  }

  /// Map a JWT `alg` value to an RSA signature profile.
  ///
  /// Only the explicit RSA SHA-2 algorithms from JOSE are accepted. `none`,
  /// HMAC, ECDSA, EdDSA, SHA-1, and unknown values fail closed.
  ///
  /// # Errors
  ///
  /// Returns [`RsaProtocolAlgorithmError::UnsupportedAlgorithm`] for any value
  /// other than `RS256`, `RS384`, `RS512`, `PS256`, `PS384`, or `PS512`.
  #[inline]
  pub fn from_jwt_alg(alg: &str) -> Result<Self, RsaProtocolAlgorithmError> {
    match alg {
      "RS256" => Ok(Self::pkcs1v15(RsaPkcs1v15Profile::Sha256)),
      "RS384" => Ok(Self::pkcs1v15(RsaPkcs1v15Profile::Sha384)),
      "RS512" => Ok(Self::pkcs1v15(RsaPkcs1v15Profile::Sha512)),
      "PS256" => Ok(Self::pss(RsaPssProfile::Sha256)),
      "PS384" => Ok(Self::pss(RsaPssProfile::Sha384)),
      "PS512" => Ok(Self::pss(RsaPssProfile::Sha512)),
      _ => Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm),
    }
  }

  /// Map a COSE algorithm ID to an RSA signature profile.
  ///
  /// Accepts the explicit COSE RSASSA-PKCS1-v1_5 SHA-2 IDs `-257`, `-258`,
  /// and `-259`, plus the RSASSA-PSS SHA-2 IDs `-37`, `-38`, and `-39`.
  /// SHA-1 and non-RSA IDs fail closed.
  ///
  /// # Errors
  ///
  /// Returns [`RsaProtocolAlgorithmError::UnsupportedAlgorithm`] for unknown,
  /// non-RSA, or SHA-1 COSE algorithm IDs.
  #[inline]
  pub const fn from_cose_algorithm_id(algorithm: i64) -> Result<Self, RsaProtocolAlgorithmError> {
    match algorithm {
      -257 => Ok(Self::pkcs1v15(RsaPkcs1v15Profile::Sha256)),
      -258 => Ok(Self::pkcs1v15(RsaPkcs1v15Profile::Sha384)),
      -259 => Ok(Self::pkcs1v15(RsaPkcs1v15Profile::Sha512)),
      -37 => Ok(Self::pss(RsaPssProfile::Sha256)),
      -38 => Ok(Self::pss(RsaPssProfile::Sha384)),
      -39 => Ok(Self::pss(RsaPssProfile::Sha512)),
      _ => Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm),
    }
  }

  /// Map an X.509 signature `AlgorithmIdentifier` DER value to an RSA profile.
  ///
  /// SHA-2 `sha*WithRSAEncryption` identifiers may carry absent or exact
  /// `NULL` parameters. RSASSA-PSS identifiers must carry explicit RFC 4055
  /// parameters with a supported SHA-2 hash, MGF1 using the same hash, absent
  /// or explicit salt length, and absent or explicit trailer field `1`.
  /// Missing salt length maps to the RFC 4055 default of 20 bytes. SHA-1
  /// defaults are rejected.
  ///
  /// # Errors
  ///
  /// Returns [`RsaProtocolAlgorithmError::MalformedAlgorithmIdentifier`] for
  /// non-canonical DER and [`RsaProtocolAlgorithmError::UnsupportedAlgorithm`]
  /// for unknown, non-RSA, SHA-1, or otherwise unsupported parameters.
  #[inline]
  pub fn from_x509_signature_algorithm_der(der: &[u8]) -> Result<Self, RsaProtocolAlgorithmError> {
    parse_x509_signature_algorithm(der)
  }
}

#[cfg(feature = "diag")]
impl RsaPkcs1v15Profile {
  fn digest_info_prefix(self) -> &'static [u8] {
    match self {
      Self::Sha256 => SHA256_DIGEST_INFO_PREFIX,
      Self::Sha384 => SHA384_DIGEST_INFO_PREFIX,
      Self::Sha512 => SHA512_DIGEST_INFO_PREFIX,
    }
  }
}

#[cfg(feature = "diag")]
fn diag_verify_pss_encoded(
  profile: RsaPssProfile,
  message: &[u8],
  encoded: &[u8],
  em_bits: usize,
  db: &mut [u8],
  db_mask: &mut [u8],
) -> Result<(), VerificationError> {
  match profile {
    RsaPssProfile::Sha256 => {
      verify_pss_encoded_with_scratch::<Sha256>(message, encoded, em_bits, profile.digest_len(), db, db_mask)
    }
    RsaPssProfile::Sha384 => {
      verify_pss_encoded_with_scratch::<Sha384>(message, encoded, em_bits, profile.digest_len(), db, db_mask)
    }
    RsaPssProfile::Sha512 => {
      verify_pss_encoded_with_scratch::<Sha512>(message, encoded, em_bits, profile.digest_len(), db, db_mask)
    }
  }
}

/// Verify a pre-exponentiated RSASSA-PSS encoded message.
///
/// Diagnostic-only helper for component benchmarking and padding parser
/// validation. Normal callers should use [`RsaPublicKey::verify_pss`].
///
/// # Errors
///
/// Returns an opaque [`VerificationError`] if the encoded message is invalid.
#[cfg(feature = "diag")]
#[doc(hidden)]
#[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
pub fn diag_rsa_verify_pss_encoded(
  profile: RsaPssProfile,
  message: &[u8],
  encoded: &[u8],
  em_bits: usize,
) -> Result<(), VerificationError> {
  let mut db = vec![0u8; encoded.len()];
  let mut db_mask = vec![0u8; encoded.len()];
  diag_verify_pss_encoded(profile, message, encoded, em_bits, &mut db, &mut db_mask)
}

/// Verify a pre-exponentiated RSASSA-PSS encoded message with caller scratch.
///
/// Diagnostic-only helper for allocation-free component benchmarking.
///
/// # Errors
///
/// Returns an opaque [`VerificationError`] if the encoded message is invalid.
#[cfg(feature = "diag")]
#[doc(hidden)]
#[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
pub fn diag_rsa_verify_pss_encoded_with_scratch(
  profile: RsaPssProfile,
  message: &[u8],
  encoded: &[u8],
  em_bits: usize,
  db: &mut [u8],
  db_mask: &mut [u8],
) -> Result<(), VerificationError> {
  diag_verify_pss_encoded(profile, message, encoded, em_bits, db, db_mask)
}

/// Verify a pre-exponentiated RSASSA-PKCS1-v1_5 encoded message.
///
/// Diagnostic-only helper for component benchmarking and padding parser
/// validation. Normal callers should use [`RsaPublicKey::verify_pkcs1v15`].
///
/// # Errors
///
/// Returns an opaque [`VerificationError`] if the encoded message is invalid.
#[cfg(feature = "diag")]
#[doc(hidden)]
#[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
pub fn diag_rsa_verify_pkcs1v15_encoded(
  profile: RsaPkcs1v15Profile,
  message: &[u8],
  encoded: &[u8],
) -> Result<(), VerificationError> {
  match profile {
    RsaPkcs1v15Profile::Sha256 => verify_pkcs1v15_encoded::<Sha256>(message, encoded, profile.digest_info_prefix()),
    RsaPkcs1v15Profile::Sha384 => verify_pkcs1v15_encoded::<Sha384>(message, encoded, profile.digest_info_prefix()),
    RsaPkcs1v15Profile::Sha512 => verify_pkcs1v15_encoded::<Sha512>(message, encoded, profile.digest_info_prefix()),
  }
}

/// Apply the RSA public operation with a simple bit-serial modular multiplier.
///
/// Diagnostic-only baseline for arithmetic benchmarking. This intentionally
/// avoids Montgomery multiplication so the production path can be measured
/// against a straightforward alternative without exposing a generic big-int API.
///
/// # Errors
///
/// Returns [`RsaPublicOpError`] if `input` or `out` is not exactly the modulus
/// length, or if `input >= n`.
#[cfg(feature = "diag")]
#[doc(hidden)]
pub fn diag_rsa_public_operation_bitserial(
  key: &RsaPublicKey,
  input: &[u8],
  out: &mut [u8],
) -> Result<(), RsaPublicOpError> {
  let result = key.modulus.public_operation_bitserial(key.exponent, input, out);
  clear_output_on_error(result, out)
}

/// Apply the RSA public operation with product-then-reduce Montgomery multiplication.
///
/// Diagnostic-only benchmark baseline for threshold selection. This forces the
/// original product-then-reduce Montgomery path even for modulus sizes where
/// production uses CIOS.
///
/// # Errors
///
/// Returns [`RsaPublicOpError`] if `input` or `out` is not exactly the modulus
/// length, if `input >= n`, or if `scratch` was allocated for another key size.
#[cfg(feature = "diag")]
#[doc(hidden)]
pub fn diag_rsa_public_operation_product(
  key: &RsaPublicKey,
  input: &[u8],
  out: &mut [u8],
  scratch: &mut RsaPublicScratch,
) -> Result<(), RsaPublicOpError> {
  let result = key.modulus.public_operation_product(key.exponent, input, out, scratch);
  clear_output_on_error(result, out)
}

/// Apply the RSA public operation using Comba product accumulation plus REDC.
///
/// Diagnostic-only benchmark candidate for multiplication strategy. Normal
/// callers should use [`RsaPublicKey::public_operation`].
///
/// # Errors
///
/// Returns [`RsaPublicOpError`] if `input` or `out` is not exactly the modulus
/// length, if `input >= n`, or if `scratch` was allocated for another key size.
#[cfg(feature = "diag")]
#[doc(hidden)]
pub fn diag_rsa_public_operation_comba_product(
  key: &RsaPublicKey,
  input: &[u8],
  out: &mut [u8],
  scratch: &mut RsaPublicScratch,
) -> Result<(), RsaPublicOpError> {
  let result = key
    .modulus
    .public_operation_comba_product(key.exponent, input, out, scratch);
  clear_output_on_error(result, out)
}

/// Apply the RSA public operation with CIOS Montgomery multiplication.
///
/// Diagnostic-only benchmark candidate for comparing the production
/// product-then-reduce Montgomery path against coarsely integrated operand
/// scanning. Normal callers should use [`RsaPublicKey::public_operation`].
///
/// # Errors
///
/// Returns [`RsaPublicOpError`] if `input` or `out` is not exactly the modulus
/// length, if `input >= n`, or if `scratch` was allocated for another key size.
#[cfg(feature = "diag")]
#[doc(hidden)]
pub fn diag_rsa_public_operation_cios(
  key: &RsaPublicKey,
  input: &[u8],
  out: &mut [u8],
  scratch: &mut RsaPublicScratch,
) -> Result<(), RsaPublicOpError> {
  let result = key.modulus.public_operation_cios(key.exponent, input, out, scratch);
  clear_output_on_error(result, out)
}

/// Apply the RSA public operation with the generic square-and-multiply exponent loop.
///
/// Diagnostic-only benchmark baseline for exponentiation strategy. This forces
/// the generic public-exponent loop even when production uses a specialized
/// path for common Fermat exponents.
#[cfg(feature = "diag")]
#[doc(hidden)]
#[inline]
pub fn diag_rsa_public_operation_generic_exponent(
  key: &RsaPublicKey,
  input: &[u8],
  out: &mut [u8],
  scratch: &mut RsaPublicScratch,
) -> Result<(), RsaPublicOpError> {
  let result = key
    .modulus
    .public_operation_generic_exponent(key.exponent, input, out, scratch);
  clear_output_on_error(result, out)
}

/// Apply the RSA public operation with a width-2 sliding-window exponent loop.
///
/// This diagnostic helper exists only to measure whether public-exponent
/// windowing can beat the simpler square-and-multiply path. Normal callers
/// should use [`RsaPublicKey::public_operation`].
#[cfg(feature = "diag")]
#[doc(hidden)]
pub fn diag_rsa_public_operation_window2_exponent(
  key: &RsaPublicKey,
  input: &[u8],
  out: &mut [u8],
  scratch: &mut RsaPublicScratch,
) -> Result<(), RsaPublicOpError> {
  let result = key
    .modulus
    .public_operation_window2_exponent(key.exponent, input, out, scratch);
  clear_output_on_error(result, out)
}

/// Parse and validate SPKI public-key DER without constructing Montgomery state.
///
/// Diagnostic-only helper for separating DER/key validation cost from
/// `RsaPublicKey` import precompute in benchmarks. Normal callers should use
/// [`RsaPublicKey::from_spki_der`].
///
/// # Errors
///
/// Returns [`RsaKeyError`] if parsing or validation fails.
#[cfg(feature = "diag")]
#[doc(hidden)]
pub fn diag_rsa_validate_spki_public_key_der(
  der: &[u8],
  policy: &RsaPublicKeyPolicy,
) -> Result<(usize, usize, RsaPublicExponent), RsaKeyError> {
  let (algorithm, public_key_der) = parse_spki_der(der)?;
  parse_rsa_algorithm_identifier(algorithm)?;
  let (modulus, modulus_bits, exponent) = parse_pkcs1_public_key_der_parts(public_key_der, policy)?;
  Ok((modulus.len(), modulus_bits, exponent))
}

/// Precompute Montgomery `R^2 mod n` for a canonical public modulus.
///
/// Diagnostic-only helper for component benchmarking. The return value is a
/// checksum of the computed limbs so benchmarks cannot erase the work while the
/// internal limb representation stays private.
///
/// # Errors
///
/// Returns [`RsaKeyError`] if the modulus violates the default public-key policy.
#[cfg(feature = "diag")]
#[doc(hidden)]
pub fn diag_rsa_precompute_public_montgomery_r2(modulus: &[u8]) -> Result<u64, RsaKeyError> {
  let policy = RsaPublicKeyPolicy::default();
  validate_modulus(modulus, &policy)?;
  let limbs = limbs_from_be(modulus);
  let r2 = public_montgomery_r2(&limbs);
  Ok(limb_checksum(&r2))
}

/// Derive a blinding-factor inverse for RSA side-channel diagnostics.
///
/// This helper is intentionally available only with `diag,getrandom`; normal
/// callers should use the signing and decryption APIs, which generate and clear
/// blinding material internally.
#[cfg(all(feature = "diag", feature = "getrandom"))]
#[cfg_attr(docsrs, doc(cfg(all(feature = "diag", feature = "getrandom"))))]
#[doc(hidden)]
pub fn diag_rsa_blinding_factor_inverse(
  key: &RsaPrivateKey,
  factor: &[u8],
  out: &mut [u8],
) -> Result<(), RsaPrivateOpError> {
  key.components.blinding_factor_inverse(factor, out)
}

/// Public exponent policy for RSA public-key parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RsaPublicExponentPolicy {
  /// Accept only the common `65537` exponent.
  Common65537,
  /// Accept `3`, `17`, and `65537`.
  ///
  /// This exists for legacy verification/import paths. Keep new signing-key
  /// policy at `65537` unless a protocol adapter has a concrete compatibility
  /// reason to loosen it.
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

/// RSA public-key validation policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RsaPublicKeyPolicy {
  min_modulus_bits: usize,
  max_modulus_bits: usize,
  exponent_policy: RsaPublicExponentPolicy,
}

impl Default for RsaPublicKeyPolicy {
  #[inline]
  fn default() -> Self {
    Self::legacy_verification()
  }
}

impl RsaPublicKeyPolicy {
  /// Compatibility verification policy: RSA-2048 through RSA-8192, exponent `65537`.
  pub const LEGACY_VERIFICATION: Self = Self {
    min_modulus_bits: MIN_RSA_MODULUS_BITS,
    max_modulus_bits: 8192,
    exponent_policy: RsaPublicExponentPolicy::Common65537,
  };

  /// Modern verification policy: RSA-3072 through RSA-8192, exponent `65537`.
  pub const MODERN_VERIFICATION: Self = Self {
    min_modulus_bits: 3072,
    max_modulus_bits: 8192,
    exponent_policy: RsaPublicExponentPolicy::Common65537,
  };

  /// Return the default legacy verification policy.
  #[inline]
  #[must_use]
  pub const fn legacy_verification() -> Self {
    Self::LEGACY_VERIFICATION
  }

  /// Return a modern RSA verification policy for newly minted material.
  #[inline]
  #[must_use]
  pub const fn modern_verification() -> Self {
    Self::MODERN_VERIFICATION
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
/// Private operations use either OS-backed blinding through the `getrandom`
/// feature or an explicit caller-supplied blinding factor and modular inverse
/// for deterministic tests and constrained integrations.
pub struct RsaPrivateKey {
  components: RsaPrivateKeyComponents,
}

/// Caller-owned scratch for RSA private operations.
///
/// Reusing this scratch avoids top-level private-operation buffer allocation
/// for deterministic blinding APIs. Scratch is bound to a modulus width, not a
/// specific key; using it with a different width returns
/// [`RsaPrivateOpError::InvalidScratch`].
pub struct RsaPrivateScratch {
  encoded: SecretBigEndianBuffer,
  salt: SecretBigEndianBuffer,
  blinding_factor: SecretBigEndianBuffer,
  blinding_inverse: SecretBigEndianBuffer,
  blinding_power: SecretBigEndianBuffer,
  blinded: SecretBigEndianBuffer,
  blinded_private_result: SecretBigEndianBuffer,
  checked: SecretBigEndianBuffer,
  one: SecretBigEndianBuffer,
  public_scratch: RsaPublicScratch,
  mul_scratch: RsaPrivateMulScratch,
  exponent_p_scratch: RsaPrivateExponentScratch,
  exponent_q_scratch: RsaPrivateExponentScratch,
}

/// Borrowed RSA private-key CRT components.
///
/// These fields contain private key material. Keep values canonical unsigned
/// big-endian, without leading zero padding.
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

impl RsaPrivateKey {
  /// Key-generation contract used by [`Self::generate`] and
  /// [`Self::generate_with_policy`].
  ///
  /// The contract is intentionally explicit: generated keys use probable-prime
  /// RSA generation following FIPS 186-5 Appendix A.1.3 in code. This is not a
  /// CMVP/FIPS 140-3 validation claim.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  pub const GENERATION_CONTRACT: RsaKeyGenerationContract = RsaKeyGenerationContract::Fips1865A13ProbablePrime;

  /// Generate a new RSA private key with public exponent `65537`.
  ///
  /// New key material uses the modern RSA policy: 3072 through 8192 modulus
  /// bits. Use imported legacy RSA-2048 keys only for compatibility.
  ///
  /// Generation follows [`Self::GENERATION_CONTRACT`]: FIPS 186-5 Appendix
  /// A.1.3 probable-prime RSA generation in code. This is not a CMVP/FIPS 140-3
  /// validation claim.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyGenerationError`] if the requested size is outside policy,
  /// entropy is unavailable, or bounded prime search fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  pub fn generate(modulus_bits: usize) -> Result<Self, RsaKeyGenerationError> {
    Self::generate_with_policy(modulus_bits, &RsaPublicKeyPolicy::modern_verification())
  }

  /// Generate a new RSA private key with an explicit modulus policy.
  ///
  /// The generated exponent is always `65537`; the policy only controls
  /// accepted modulus size. This is mainly for tests and deployments that must
  /// mint RSA-2048 compatibility keys.
  ///
  /// Generation follows [`Self::GENERATION_CONTRACT`]: FIPS 186-5 Appendix
  /// A.1.3 probable-prime RSA generation in code. This is not a CMVP/FIPS 140-3
  /// validation claim.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyGenerationError`] if the requested size is outside policy,
  /// entropy is unavailable, or bounded prime search fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  pub fn generate_with_policy(modulus_bits: usize, policy: &RsaPublicKeyPolicy) -> Result<Self, RsaKeyGenerationError> {
    generate_rsa_private_key(modulus_bits, policy).map(|components| Self { components })
  }

  /// Parse a PKCS #1 `RSAPrivateKey` DER object with the default policy.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if the DER is malformed or the key components are
  /// inconsistent with RSA private-key policy.
  pub fn from_pkcs1_der(der: &[u8]) -> Result<Self, RsaKeyError> {
    Self::from_pkcs1_der_with_policy(der, &RsaPublicKeyPolicy::default())
  }

  /// Parse a PKCS #1 `RSAPrivateKey` DER object with an explicit public-key policy.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if parsing or validation fails.
  pub fn from_pkcs1_der_with_policy(der: &[u8], policy: &RsaPublicKeyPolicy) -> Result<Self, RsaKeyError> {
    parse_pkcs1_private_key_der_with_policy(der, policy).map(|components| Self { components })
  }

  /// Parse a PKCS #8 `PrivateKeyInfo` DER object with the default policy.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if the DER is malformed, not an RSA private key,
  /// or the embedded PKCS #1 private key is invalid.
  pub fn from_pkcs8_der(der: &[u8]) -> Result<Self, RsaKeyError> {
    Self::from_pkcs8_der_with_policy(der, &RsaPublicKeyPolicy::default())
  }

  /// Parse a PKCS #8 `PrivateKeyInfo` DER object with an explicit public-key policy.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if parsing or validation fails.
  pub fn from_pkcs8_der_with_policy(der: &[u8], policy: &RsaPublicKeyPolicy) -> Result<Self, RsaKeyError> {
    parse_pkcs8_private_key_der_with_policy(der, policy).map(|components| Self { components })
  }

  /// Build an RSA private key from canonical unsigned big-endian CRT components.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if the components violate RSA private-key policy
  /// or fail consistency checks.
  pub fn from_components(parts: RsaPrivateKeyParts<'_>) -> Result<Self, RsaKeyError> {
    Self::from_components_with_policy(parts, &RsaPublicKeyPolicy::default())
  }

  /// Build an RSA private key from CRT components with an explicit public-key policy.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if the components violate `policy` or fail
  /// consistency checks.
  pub fn from_components_with_policy(
    parts: RsaPrivateKeyParts<'_>,
    policy: &RsaPublicKeyPolicy,
  ) -> Result<Self, RsaKeyError> {
    let public_exponent = parse_public_exponent(&parts.public_exponent.to_be_bytes(), policy)?;
    let components = RsaPrivateKeyDerComponents {
      modulus: parts.modulus,
      public_exponent,
      private_exponent: parts.private_exponent,
      prime_p: parts.prime_p,
      prime_q: parts.prime_q,
      exponent_p: parts.exponent_p,
      exponent_q: parts.exponent_q,
      coefficient: parts.coefficient,
    };
    private_key_components_from_parts(&components, policy).map(|components| Self { components })
  }

  /// Borrow the validated RSA public key corresponding to this private key.
  #[inline]
  #[must_use]
  pub fn public_key(&self) -> &RsaPublicKey {
    self.components.public_key()
  }

  /// Encode this private key as a PKCS #1 `RSAPrivateKey` DER object.
  ///
  /// The returned bytes contain private key material. Callers that persist or
  /// log this buffer are responsible for protecting it.
  #[must_use]
  pub fn to_pkcs1_der(&self) -> Vec<u8> {
    self.components.to_pkcs1_der()
  }

  /// Encode this private key as a PKCS #8 `PrivateKeyInfo` DER object.
  ///
  /// The returned bytes contain private key material. Callers that persist or
  /// log this buffer are responsible for protecting it.
  #[must_use]
  pub fn to_pkcs8_der(&self) -> Vec<u8> {
    let mut pkcs1 = self.to_pkcs1_der();
    let der = der_sequence_from_parts(&[
      der_integer_unsigned(&[0]).as_slice(),
      der_rsa_encryption_algorithm_identifier().as_slice(),
      der_tlv(TAG_OCTET_STRING, &pkcs1).as_slice(),
    ]);
    ct::zeroize(&mut pkcs1);
    der
  }

  /// Return the fixed signature length for this key.
  #[inline]
  #[must_use]
  pub fn signature_len(&self) -> usize {
    self.public_key().modulus().len()
  }

  /// Allocate reusable scratch space for deterministic private operations.
  ///
  /// Use this with the `*_with_blinding_factor_and_scratch` methods when a
  /// caller supplies validated blinding material and wants steady-state signing
  /// or decryption without top-level temporary buffer allocation.
  #[inline]
  #[must_use]
  pub fn private_scratch(&self) -> RsaPrivateScratch {
    RsaPrivateScratch::new(self)
  }

  /// Sign a message using a typed RSA signature profile and OS-backed randomness.
  ///
  /// PKCS#1 v1.5 profiles use deterministic EMSA-PKCS1-v1_5 encoding. PSS
  /// profiles generate a random salt with the profile's explicit salt length.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, the key is too
  /// small for the selected profile, or the post-signing public fault check
  /// fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_signature(
    &self,
    profile: RsaSignatureProfile,
    message: &[u8],
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    match profile {
      RsaSignatureProfile::Pss { profile, salt_len } => self.sign_pss_with_salt_len(profile, salt_len, message, out),
      RsaSignatureProfile::Pkcs1v15(profile) => self.sign_pkcs1v15(profile, message, out),
    }
  }

  /// Sign a message using a typed RSA signature profile, OS-backed randomness,
  /// and caller-owned private-operation scratch.
  ///
  /// Reusing scratch avoids top-level private-operation temporary allocation
  /// after setup. PKCS#1 v1.5 profiles use deterministic EMSA-PKCS1-v1_5
  /// encoding. PSS profiles generate a random salt with the profile's explicit
  /// salt length.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, `scratch` was
  /// allocated for a different modulus width, the key is too small for the
  /// selected profile, or the post-signing public fault check fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_signature_with_scratch(
    &self,
    profile: RsaSignatureProfile,
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    match profile {
      RsaSignatureProfile::Pss { profile, salt_len } => {
        self.sign_pss_with_salt_len_and_scratch(profile, salt_len, message, out, scratch)
      }
      RsaSignatureProfile::Pkcs1v15(profile) => self.sign_pkcs1v15_with_scratch(profile, message, out, scratch),
    }
  }

  /// Sign using an X.509 signature `AlgorithmIdentifier` DER value.
  ///
  /// Primitive helper only: this does not build or validate certificates. It
  /// accepts the SHA-2 `sha*WithRSAEncryption` identifiers and supported
  /// RSASSA-PSS parameter sets.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError::UnsupportedAlgorithm`] if `der` is malformed
  /// or not an accepted RSA SHA-2 signing algorithm. Other errors match
  /// [`Self::sign_signature`].
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_x509_signature_algorithm_der(
    &self,
    der: &[u8],
    message: &[u8],
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    let result = RsaSignatureProfile::from_x509_signature_algorithm_der(der)
      .map_err(|_| RsaPrivateOpError::UnsupportedAlgorithm)
      .and_then(|profile| self.sign_signature(profile, message, out));
    clear_output_on_error(result, out)
  }

  /// Sign using an X.509 signature `AlgorithmIdentifier` DER value and caller-owned scratch.
  ///
  /// Primitive helper only: this does not build or validate certificates. It
  /// accepts the SHA-2 `sha*WithRSAEncryption` identifiers and supported
  /// RSASSA-PSS parameter sets.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError::UnsupportedAlgorithm`] if `der` is malformed
  /// or not an accepted RSA SHA-2 signing algorithm. Other errors match
  /// [`Self::sign_signature_with_scratch`].
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_x509_signature_algorithm_der_with_scratch(
    &self,
    der: &[u8],
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    let result = RsaSignatureProfile::from_x509_signature_algorithm_der(der)
      .map_err(|_| RsaPrivateOpError::UnsupportedAlgorithm)
      .and_then(|profile| self.sign_signature_with_scratch(profile, message, out, scratch));
    clear_output_on_error(result, out)
  }

  /// Sign a TLS 1.3 `CertificateVerify` message using a parsed signature scheme.
  ///
  /// Primitive helper only: this does not construct the TLS transcript message
  /// or enforce certificate-chain policy. TLS 1.3 RSA handshake signatures are
  /// RSASSA-PSS only; legacy PKCS#1 v1.5 scheme IDs are rejected here.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError::UnsupportedAlgorithm`] if `scheme` is not an
  /// accepted TLS 1.3 RSA signing scheme. Other errors match
  /// [`Self::sign_signature`].
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_tls13_signature_scheme(
    &self,
    scheme: u16,
    message: &[u8],
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    let result = RsaSignatureProfile::from_tls13_signature_scheme(scheme)
      .map_err(|_| RsaPrivateOpError::UnsupportedAlgorithm)
      .and_then(|profile| self.sign_signature(profile, message, out));
    clear_output_on_error(result, out)
  }

  /// Sign a TLS 1.3 `CertificateVerify` message using a parsed signature scheme and caller-owned
  /// scratch.
  ///
  /// Primitive helper only: this does not construct the TLS transcript message
  /// or enforce certificate-chain policy. TLS 1.3 RSA handshake signatures are
  /// RSASSA-PSS only; legacy PKCS#1 v1.5 scheme IDs are rejected here.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError::UnsupportedAlgorithm`] if `scheme` is not an
  /// accepted TLS 1.3 RSA signing scheme. Other errors match
  /// [`Self::sign_signature_with_scratch`].
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_tls13_signature_scheme_with_scratch(
    &self,
    scheme: u16,
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    let result = RsaSignatureProfile::from_tls13_signature_scheme(scheme)
      .map_err(|_| RsaPrivateOpError::UnsupportedAlgorithm)
      .and_then(|profile| self.sign_signature_with_scratch(profile, message, out, scratch));
    clear_output_on_error(result, out)
  }

  /// Sign using a TLS certificate signature scheme ID.
  ///
  /// Primitive helper only: this does not build or validate certificates. It
  /// accepts the SHA-2 PKCS#1 v1.5 certificate-signature schemes and the TLS
  /// 1.3 RSA-PSS schemes.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError::UnsupportedAlgorithm`] if `scheme` is not an
  /// accepted RSA certificate signing scheme. Other errors match
  /// [`Self::sign_signature`].
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_tls_certificate_signature_scheme(
    &self,
    scheme: u16,
    message: &[u8],
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    let result = RsaSignatureProfile::from_tls_certificate_signature_scheme(scheme)
      .map_err(|_| RsaPrivateOpError::UnsupportedAlgorithm)
      .and_then(|profile| self.sign_signature(profile, message, out));
    clear_output_on_error(result, out)
  }

  /// Sign using a TLS certificate signature scheme ID and caller-owned scratch.
  ///
  /// Primitive helper only: this does not build or validate certificates. It
  /// accepts the SHA-2 PKCS#1 v1.5 certificate-signature schemes and the TLS
  /// 1.3 RSA-PSS schemes.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError::UnsupportedAlgorithm`] if `scheme` is not an
  /// accepted RSA certificate signing scheme. Other errors match
  /// [`Self::sign_signature_with_scratch`].
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_tls_certificate_signature_scheme_with_scratch(
    &self,
    scheme: u16,
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    let result = RsaSignatureProfile::from_tls_certificate_signature_scheme(scheme)
      .map_err(|_| RsaPrivateOpError::UnsupportedAlgorithm)
      .and_then(|profile| self.sign_signature_with_scratch(profile, message, out, scratch));
    clear_output_on_error(result, out)
  }

  /// Sign a JWT/JWS signing input using an already-parsed JOSE `alg`.
  ///
  /// Primitive helper only: this is not a JWT, JWS, JOSE, or JSON provider
  /// integration.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError::UnsupportedAlgorithm`] if `alg` is not an
  /// accepted RSA SHA-2 JOSE algorithm. Other errors match
  /// [`Self::sign_signature`].
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_jwt_alg(&self, alg: &str, message: &[u8], out: &mut [u8]) -> Result<(), RsaPrivateOpError> {
    let result = RsaSignatureProfile::from_jwt_alg(alg)
      .map_err(|_| RsaPrivateOpError::UnsupportedAlgorithm)
      .and_then(|profile| self.sign_signature(profile, message, out));
    clear_output_on_error(result, out)
  }

  /// Sign a JWT/JWS signing input using an already-parsed JOSE `alg` and caller-owned scratch.
  ///
  /// Primitive helper only: this is not a JWT, JWS, JOSE, or JSON provider
  /// integration.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError::UnsupportedAlgorithm`] if `alg` is not an
  /// accepted RSA SHA-2 JOSE algorithm. Other errors match
  /// [`Self::sign_signature_with_scratch`].
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_jwt_alg_with_scratch(
    &self,
    alg: &str,
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    let result = RsaSignatureProfile::from_jwt_alg(alg)
      .map_err(|_| RsaPrivateOpError::UnsupportedAlgorithm)
      .and_then(|profile| self.sign_signature_with_scratch(profile, message, out, scratch));
    clear_output_on_error(result, out)
  }

  /// Sign a COSE Sig_structure using an already-parsed COSE algorithm ID.
  ///
  /// Primitive helper only: this is not a COSE, CBOR, or CWT provider
  /// integration.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError::UnsupportedAlgorithm`] if `algorithm` is not
  /// an accepted RSA SHA-2 COSE algorithm. Other errors match
  /// [`Self::sign_signature`].
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_cose_algorithm_id(
    &self,
    algorithm: i64,
    message: &[u8],
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    let result = RsaSignatureProfile::from_cose_algorithm_id(algorithm)
      .map_err(|_| RsaPrivateOpError::UnsupportedAlgorithm)
      .and_then(|profile| self.sign_signature(profile, message, out));
    clear_output_on_error(result, out)
  }

  /// Sign a COSE Sig_structure using an already-parsed COSE algorithm ID and caller-owned scratch.
  ///
  /// Primitive helper only: this is not a COSE, CBOR, or CWT provider
  /// integration.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError::UnsupportedAlgorithm`] if `algorithm` is not
  /// an accepted RSA SHA-2 COSE algorithm. Other errors match
  /// [`Self::sign_signature_with_scratch`].
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_cose_algorithm_id_with_scratch(
    &self,
    algorithm: i64,
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    let result = RsaSignatureProfile::from_cose_algorithm_id(algorithm)
      .map_err(|_| RsaPrivateOpError::UnsupportedAlgorithm)
      .and_then(|profile| self.sign_signature_with_scratch(profile, message, out, scratch));
    clear_output_on_error(result, out)
  }

  /// Sign a message using RSASSA-PKCS1-v1_5 and OS-backed blinding.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, the key is too
  /// small for the selected profile, or the post-signing public fault check
  /// fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_pkcs1v15(
    &self,
    profile: RsaPkcs1v15Profile,
    message: &[u8],
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    let result = self.components.random_blinding_factor().and_then(|blinding| {
      self.components.sign_pkcs1v15_with_blinding_factor(
        profile,
        message,
        RsaBlindingPair::trusted(blinding.factor(), blinding.inverse()),
        out,
      )
    });
    clear_output_on_error(result, out)
  }

  /// Sign a message using RSASSA-PKCS1-v1_5, OS-backed blinding, and caller-owned scratch.
  ///
  /// Reusing scratch avoids top-level private-operation temporary allocation
  /// after setup.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, `scratch` was
  /// allocated for a different modulus width, the key is too small for the
  /// selected profile, or the post-signing public fault check fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_pkcs1v15_with_scratch(
    &self,
    profile: RsaPkcs1v15Profile,
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    let result = self
      .components
      .random_blinding_factor_into_scratch(scratch)
      .and_then(|()| {
        self
          .components
          .sign_pkcs1v15_with_stored_blinding_and_scratch(profile, message, out, scratch)
      });
    scratch.clear();
    clear_output_on_error(result, out)
  }

  /// Sign a message using RSASSA-PSS and OS-backed random salt/blinding.
  ///
  /// The generated salt length is the selected profile's digest length.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, the key is too
  /// small for the selected profile, or the post-signing public fault check
  /// fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_pss(&self, profile: RsaPssProfile, message: &[u8], out: &mut [u8]) -> Result<(), RsaPrivateOpError> {
    self.sign_pss_with_salt_len(profile, profile.digest_len(), message, out)
  }

  /// Sign a message using RSASSA-PSS, OS-backed random salt/blinding, and caller-owned scratch.
  ///
  /// The generated salt length is the selected profile's digest length.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, `scratch` was
  /// allocated for a different modulus width, the key is too small for the
  /// selected profile, or the post-signing public fault check fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_pss_with_scratch(
    &self,
    profile: RsaPssProfile,
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    self.sign_pss_with_salt_len_and_scratch(profile, profile.digest_len(), message, out, scratch)
  }

  /// Sign a message using RSASSA-PSS with an explicit random salt length.
  ///
  /// Use this after parsing protocol parameters that permit a salt length other
  /// than the digest length. TLS 1.3-style signatures should prefer
  /// [`Self::sign_pss`] or [`Self::sign_signature`].
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, the key is too
  /// small for the requested salt length, or the post-signing public fault
  /// check fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_pss_with_salt_len(
    &self,
    profile: RsaPssProfile,
    salt_len: usize,
    message: &[u8],
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    let result = if !self.public_key().pss_salt_len_is_possible(profile, salt_len) {
      Err(RsaPrivateOpError::MessageTooLong)
    } else {
      let mut salt = vec![0u8; salt_len];
      let result = getrandom::fill(&mut salt)
        .map_err(|_| RsaPrivateOpError::EntropyUnavailable)
        .and_then(|()| self.components.random_blinding_factor())
        .and_then(|blinding| {
          self.components.sign_pss_with_salt_and_blinding_factor(
            profile,
            message,
            &salt,
            RsaBlindingPair::trusted(blinding.factor(), blinding.inverse()),
            out,
          )
        });
      ct::zeroize(&mut salt);
      result
    };
    clear_output_on_error(result, out)
  }

  /// Sign a message using RSASSA-PSS with an explicit random salt length and caller-owned scratch.
  ///
  /// Use this after parsing protocol parameters that permit a salt length other
  /// than the digest length. TLS 1.3-style signatures should prefer
  /// [`Self::sign_pss_with_scratch`] or [`Self::sign_signature_with_scratch`].
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, `scratch` was
  /// allocated for a different modulus width, the key is too small for the
  /// requested salt length, or the post-signing public fault check fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_pss_with_salt_len_and_scratch(
    &self,
    profile: RsaPssProfile,
    salt_len: usize,
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    let result = if !self.public_key().pss_salt_len_is_possible(profile, salt_len) {
      Err(RsaPrivateOpError::MessageTooLong)
    } else {
      scratch.ensure_len(self.public_key().modulus().len()).and_then(|()| {
        let salt = scratch
          .salt
          .as_mut_slice()
          .get_mut(..salt_len)
          .ok_or(RsaPrivateOpError::MessageTooLong)?;
        getrandom::fill(salt)
          .map_err(|_| RsaPrivateOpError::EntropyUnavailable)
          .and_then(|()| self.components.random_blinding_factor_into_scratch(scratch))
          .and_then(|()| {
            self
              .components
              .sign_pss_with_stored_salt_and_blinding_and_scratch(profile, message, salt_len, out, scratch)
          })
      })
    };
    scratch.clear();
    clear_output_on_error(result, out)
  }

  /// Decrypt an RSAES-OAEP ciphertext using OS-backed blinding.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, the ciphertext is
  /// malformed, or OAEP decoding fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA decryption failure must be checked; a dropped Result silently discards plaintext"]
  pub fn decrypt_oaep(
    &self,
    profile: RsaOaepProfile,
    label: &[u8],
    ciphertext: &[u8],
    out: &mut [u8],
  ) -> Result<usize, RsaPrivateOpError> {
    let result = self.components.random_blinding_factor().and_then(|blinding| {
      self.components.decrypt_oaep_with_blinding_factor(
        profile,
        label,
        ciphertext,
        RsaBlindingPair::trusted(blinding.factor(), blinding.inverse()),
        out,
      )
    });
    clear_output_on_error(result, out)
  }

  /// Decrypt an RSAES-OAEP ciphertext using OS-backed blinding and caller-owned scratch.
  ///
  /// Reusing scratch avoids top-level private-operation temporary allocation
  /// after setup.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, `scratch` was
  /// allocated for a different modulus width, the ciphertext is malformed, or
  /// OAEP decoding fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA decryption failure must be checked; a dropped Result silently discards plaintext"]
  pub fn decrypt_oaep_with_scratch(
    &self,
    profile: RsaOaepProfile,
    label: &[u8],
    ciphertext: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<usize, RsaPrivateOpError> {
    let result = self
      .components
      .random_blinding_factor_into_scratch(scratch)
      .and_then(|()| {
        self
          .components
          .decrypt_oaep_with_stored_blinding_and_scratch(profile, label, ciphertext, out, scratch)
      });
    scratch.clear();
    clear_decryption_output_on_error(result, out)
  }

  /// Decrypt an RSAES-PKCS1-v1_5 ciphertext using OS-backed blinding.
  ///
  /// Prefer OAEP for new protocols. This legacy primitive returns
  /// [`RsaPrivateOpError::DecryptionFailed`] for PKCS#1 v1.5 padding failures.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, the ciphertext is
  /// malformed, or PKCS#1 v1.5 decoding fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA decryption failure must be checked; a dropped Result silently discards plaintext"]
  pub fn decrypt_pkcs1v15(&self, ciphertext: &[u8], out: &mut [u8]) -> Result<usize, RsaPrivateOpError> {
    let result = self.components.random_blinding_factor().and_then(|blinding| {
      self.components.decrypt_pkcs1v15_with_blinding_factor(
        ciphertext,
        RsaBlindingPair::trusted(blinding.factor(), blinding.inverse()),
        out,
      )
    });
    clear_output_on_error(result, out)
  }

  /// Decrypt an RSAES-PKCS1-v1_5 ciphertext using OS-backed blinding and caller-owned scratch.
  ///
  /// Prefer OAEP for new protocols. Reusing scratch avoids top-level
  /// private-operation temporary allocation after setup.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, `scratch` was
  /// allocated for a different modulus width, the ciphertext is malformed, or
  /// PKCS#1 v1.5 decoding fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA decryption failure must be checked; a dropped Result silently discards plaintext"]
  pub fn decrypt_pkcs1v15_with_scratch(
    &self,
    ciphertext: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<usize, RsaPrivateOpError> {
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
  }

  /// Sign a message using RSASSA-PKCS1-v1_5 with a caller-supplied blinding factor.
  ///
  /// `blinding_factor` and `blinding_factor_inverse` must be fixed-width
  /// modulus-sized representatives satisfying
  /// `blinding_factor * blinding_factor_inverse == 1 mod n`.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if lengths are invalid, the blinding pair is
  /// invalid, the key is too small, or the post-signing public fault check fails.
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_pkcs1v15_with_blinding_factor(
    &self,
    profile: RsaPkcs1v15Profile,
    message: &[u8],
    blinding_factor: &[u8],
    blinding_factor_inverse: &[u8],
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    let result = self.components.sign_pkcs1v15_with_blinding_factor(
      profile,
      message,
      RsaBlindingPair::caller_supplied(blinding_factor, blinding_factor_inverse),
      out,
    );
    clear_output_on_error(result, out)
  }

  /// Sign using RSASSA-PKCS1-v1_5 with caller-supplied blinding and scratch.
  ///
  /// Reusing scratch avoids top-level private-operation temporary allocation
  /// after setup. The blinding-factor requirements are the same as
  /// [`Self::sign_pkcs1v15_with_blinding_factor`].
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if lengths are invalid, `scratch` was
  /// allocated for a different modulus width, the blinding pair is invalid, the
  /// key is too small, or the post-signing public fault check fails.
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_pkcs1v15_with_blinding_factor_and_scratch(
    &self,
    profile: RsaPkcs1v15Profile,
    message: &[u8],
    blinding_factor: &[u8],
    blinding_factor_inverse: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    let result = self.components.sign_pkcs1v15_with_blinding_factor_and_scratch(
      profile,
      message,
      RsaBlindingPair::caller_supplied(blinding_factor, blinding_factor_inverse),
      out,
      scratch,
    );
    clear_output_on_error(result, out)
  }

  /// Sign a message using RSASSA-PSS with explicit salt and caller-supplied blinding.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if lengths are invalid, the blinding pair is
  /// invalid, the key is too small for the salt/profile, or the post-signing
  /// public fault check fails.
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_pss_with_salt_and_blinding_factor(
    &self,
    profile: RsaPssProfile,
    message: &[u8],
    salt: &[u8],
    blinding_factor: &[u8],
    blinding_factor_inverse: &[u8],
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    let result = self.components.sign_pss_with_salt_and_blinding_factor(
      profile,
      message,
      salt,
      RsaBlindingPair::caller_supplied(blinding_factor, blinding_factor_inverse),
      out,
    );
    clear_output_on_error(result, out)
  }

  /// Sign using RSASSA-PSS with explicit salt, caller blinding, and scratch.
  ///
  /// Reusing scratch avoids top-level private-operation temporary allocation
  /// after setup. The salt and blinding-factor requirements are the same as
  /// [`Self::sign_pss_with_salt_and_blinding_factor`].
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if lengths are invalid, `scratch` was
  /// allocated for a different modulus width, the blinding pair is invalid, the
  /// key is too small for the salt/profile, or the post-signing public fault
  /// check fails.
  #[allow(clippy::too_many_arguments)]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_pss_with_salt_and_blinding_factor_and_scratch(
    &self,
    profile: RsaPssProfile,
    message: &[u8],
    salt: &[u8],
    blinding_factor: &[u8],
    blinding_factor_inverse: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    let result = self.components.sign_pss_with_salt_and_blinding_factor_and_scratch(
      profile,
      message,
      salt,
      RsaBlindingPair::caller_supplied(blinding_factor, blinding_factor_inverse),
      out,
      scratch,
    );
    clear_output_on_error(result, out)
  }

  /// Decrypt an RSAES-OAEP ciphertext with caller-supplied blinding.
  ///
  /// The decrypted plaintext is written to `out`, and the plaintext length is
  /// returned. The label must match the label used by the encrypting side.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if lengths are invalid, the blinding pair is
  /// invalid, the ciphertext representative is out of range, or OAEP decoding
  /// fails.
  #[must_use = "RSA decryption failure must be checked; a dropped Result silently discards plaintext"]
  pub fn decrypt_oaep_with_blinding_factor(
    &self,
    profile: RsaOaepProfile,
    label: &[u8],
    ciphertext: &[u8],
    blinding_factor: &[u8],
    blinding_factor_inverse: &[u8],
    out: &mut [u8],
  ) -> Result<usize, RsaPrivateOpError> {
    let result = self.components.decrypt_oaep_with_blinding_factor(
      profile,
      label,
      ciphertext,
      RsaBlindingPair::caller_supplied(blinding_factor, blinding_factor_inverse),
      out,
    );
    clear_decryption_output_on_error(result, out)
  }

  /// Decrypt RSAES-PKCS1-v1_5 with caller-supplied blinding.
  ///
  /// Prefer OAEP for new protocols. This legacy primitive keeps padding
  /// failures opaque but does not by itself make a protocol Bleichenbacher-safe.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if lengths are invalid, the blinding pair is
  /// invalid, the ciphertext representative is out of range, or PKCS#1 v1.5
  /// decoding fails.
  #[must_use = "RSA decryption failure must be checked; a dropped Result silently discards plaintext"]
  pub fn decrypt_pkcs1v15_with_blinding_factor(
    &self,
    ciphertext: &[u8],
    blinding_factor: &[u8],
    blinding_factor_inverse: &[u8],
    out: &mut [u8],
  ) -> Result<usize, RsaPrivateOpError> {
    let result = self.components.decrypt_pkcs1v15_with_blinding_factor(
      ciphertext,
      RsaBlindingPair::caller_supplied(blinding_factor, blinding_factor_inverse),
      out,
    );
    clear_decryption_output_on_error(result, out)
  }

  /// Decrypt RSAES-OAEP with caller-supplied blinding and scratch.
  ///
  /// Reusing scratch avoids top-level private-operation temporary allocation
  /// after setup. The blinding-factor requirements are the same as
  /// [`Self::decrypt_oaep_with_blinding_factor`].
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if lengths are invalid, `scratch` was
  /// allocated for a different modulus width, the blinding pair is invalid, the
  /// ciphertext representative is out of range, or OAEP decoding fails.
  #[allow(clippy::too_many_arguments)]
  #[must_use = "RSA decryption failure must be checked; a dropped Result silently discards plaintext"]
  pub fn decrypt_oaep_with_blinding_factor_and_scratch(
    &self,
    profile: RsaOaepProfile,
    label: &[u8],
    ciphertext: &[u8],
    blinding_factor: &[u8],
    blinding_factor_inverse: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<usize, RsaPrivateOpError> {
    let result = self.components.decrypt_oaep_with_blinding_factor_and_scratch(
      profile,
      label,
      ciphertext,
      RsaBlindingPair::caller_supplied(blinding_factor, blinding_factor_inverse),
      out,
      scratch,
    );
    clear_decryption_output_on_error(result, out)
  }

  /// Decrypt RSAES-PKCS1-v1_5 with caller-supplied blinding and scratch.
  ///
  /// Reusing scratch avoids top-level private-operation temporary allocation
  /// after setup. The blinding-factor requirements are the same as
  /// [`Self::decrypt_pkcs1v15_with_blinding_factor`].
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if lengths are invalid, `scratch` was
  /// allocated for a different modulus width, the blinding pair is invalid, the
  /// ciphertext representative is out of range, or PKCS#1 v1.5 decoding fails.
  #[must_use = "RSA decryption failure must be checked; a dropped Result silently discards plaintext"]
  pub fn decrypt_pkcs1v15_with_blinding_factor_and_scratch(
    &self,
    ciphertext: &[u8],
    blinding_factor: &[u8],
    blinding_factor_inverse: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<usize, RsaPrivateOpError> {
    let result = self.components.decrypt_pkcs1v15_with_blinding_factor_and_scratch(
      ciphertext,
      RsaBlindingPair::caller_supplied(blinding_factor, blinding_factor_inverse),
      out,
      scratch,
    );
    clear_decryption_output_on_error(result, out)
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

impl RsaPrivateScratch {
  /// Allocate scratch space for `key`.
  #[must_use]
  pub fn new(key: &RsaPrivateKey) -> Self {
    let len = key.signature_len();
    let mut one = SecretBigEndianBuffer::zeroed(len);
    if let Some(last) = one.as_mut_slice().last_mut() {
      *last = 1;
    }
    Self {
      encoded: SecretBigEndianBuffer::zeroed(len),
      salt: SecretBigEndianBuffer::zeroed(len),
      blinding_factor: SecretBigEndianBuffer::zeroed(len),
      blinding_inverse: SecretBigEndianBuffer::zeroed(len),
      blinding_power: SecretBigEndianBuffer::zeroed(len),
      blinded: SecretBigEndianBuffer::zeroed(len),
      blinded_private_result: SecretBigEndianBuffer::zeroed(len),
      checked: SecretBigEndianBuffer::zeroed(len),
      one,
      public_scratch: key.public_key().public_scratch(),
      mul_scratch: RsaPrivateMulScratch::new(key.components.public.modulus.limbs.len()),
      exponent_p_scratch: RsaPrivateExponentScratch::new(key.components.prime_p_modulus.limbs.len()),
      exponent_q_scratch: RsaPrivateExponentScratch::new(key.components.prime_q_modulus.limbs.len()),
    }
  }

  fn ensure_len(&self, len: usize) -> Result<(), RsaPrivateOpError> {
    if self.encoded.as_slice().len() == len
      && self.salt.as_slice().len() == len
      && self.blinding_factor.as_slice().len() == len
      && self.blinding_inverse.as_slice().len() == len
      && self.blinding_power.as_slice().len() == len
      && self.blinded.as_slice().len() == len
      && self.blinded_private_result.as_slice().len() == len
      && self.checked.as_slice().len() == len
      && self.one.as_slice().len() == len
    {
      Ok(())
    } else {
      Err(RsaPrivateOpError::InvalidScratch)
    }
  }

  fn set_one(&mut self) -> Result<(), RsaPrivateOpError> {
    ct::zeroize(self.one.as_mut_slice());
    let Some(last) = self.one.as_mut_slice().last_mut() else {
      return Err(RsaPrivateOpError::InvalidScratch);
    };
    *last = 1;
    Ok(())
  }

  fn clear(&mut self) {
    ct::zeroize(self.encoded.as_mut_slice());
    ct::zeroize(self.salt.as_mut_slice());
    ct::zeroize(self.blinding_factor.as_mut_slice());
    ct::zeroize(self.blinding_inverse.as_mut_slice());
    ct::zeroize(self.blinding_power.as_mut_slice());
    ct::zeroize(self.blinded.as_mut_slice());
    ct::zeroize(self.blinded_private_result.as_mut_slice());
    ct::zeroize(self.checked.as_mut_slice());
    ct::zeroize(self.one.as_mut_slice());
    ct::zeroize_words(&mut self.public_scratch.limbs[..]);
    ct::zeroize(&mut self.public_scratch.bytes[..]);
    self.mul_scratch.clear();
    self.exponent_p_scratch.clear();
    self.exponent_q_scratch.clear();
  }
}

impl fmt::Debug for RsaPrivateScratch {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("RsaPrivateScratch")
      .field("bytes", &self.encoded.as_slice().len())
      .finish_non_exhaustive()
  }
}

impl Drop for RsaPrivateScratch {
  fn drop(&mut self) {
    self.clear();
  }
}

// Private-key component storage and import validation. Import validation may
// branch on private input while rejecting malformed key material; it is kept
// separate from private-operation exponentiation so validation arithmetic does
// not become signing, decryption, blinding, CRT recombination, or key
// generation arithmetic by accident.
#[allow(dead_code)]
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

#[allow(dead_code)]
impl RsaPrivateKeyComponents {
  #[inline]
  fn public_key(&self) -> &RsaPublicKey {
    &self.public
  }

  fn to_pkcs1_der(&self) -> Vec<u8> {
    der_sequence_from_parts(&[
      der_integer_unsigned(&[0]).as_slice(),
      der_integer_unsigned(self.public.modulus()).as_slice(),
      der_integer_unsigned(&self.public.public_exponent().as_u64().to_be_bytes()).as_slice(),
      der_integer_unsigned(self.private_exponent.as_bytes()).as_slice(),
      der_integer_unsigned(self.prime_p.as_bytes()).as_slice(),
      der_integer_unsigned(self.prime_q.as_bytes()).as_slice(),
      der_integer_unsigned(self.exponent_p.as_bytes()).as_slice(),
      der_integer_unsigned(self.exponent_q.as_bytes()).as_slice(),
      der_integer_unsigned(self.coefficient.as_bytes()).as_slice(),
    ])
  }

  #[cfg(feature = "getrandom")]
  fn random_blinding_factor(&self) -> Result<RsaBlindingFactor, RsaPrivateOpError> {
    let len = self.public.modulus().len();
    for _ in 0..128 {
      let mut factor = vec![0u8; len];
      if getrandom::fill(&mut factor).is_err() {
        ct::zeroize(&mut factor);
        return Err(RsaPrivateOpError::EntropyUnavailable);
      }
      if is_zero_unsigned_be(&factor) || unsigned_be_cmp(&factor, self.public.modulus()) != core::cmp::Ordering::Less {
        ct::zeroize(&mut factor);
        continue;
      }

      let mut inverse = vec![0u8; len];
      match self.blinding_factor_inverse(&factor, &mut inverse) {
        Ok(()) => {
          let mut check = vec![0u8; len];
          if mod_mul_representatives(&self.public.modulus, &factor, &inverse, &mut check).is_ok()
            && check.last() == Some(&1)
          {
            let is_one = check
              .get(..check.len().strict_sub(1))
              .is_some_and(|prefix| prefix.iter().all(|&byte| byte == 0));
            if is_one {
              return Ok(RsaBlindingFactor {
                factor: SecretBigEndianBuffer::new(factor),
                inverse: SecretBigEndianBuffer::new(inverse),
              });
            }
          }
          ct::zeroize(&mut factor);
          ct::zeroize(&mut inverse);
        }
        Err(_) => {
          ct::zeroize(&mut factor);
          ct::zeroize(&mut inverse);
        }
      }
    }
    Err(RsaPrivateOpError::InvalidBlindingFactor)
  }

  #[cfg(feature = "getrandom")]
  fn random_blinding_factor_into_scratch(&self, scratch: &mut RsaPrivateScratch) -> Result<(), RsaPrivateOpError> {
    let len = self.public.modulus().len();
    scratch.ensure_len(len)?;
    for _ in 0..128 {
      if getrandom::fill(scratch.blinding_factor.as_mut_slice()).is_err() {
        ct::zeroize(scratch.blinding_factor.as_mut_slice());
        return Err(RsaPrivateOpError::EntropyUnavailable);
      }
      if is_zero_unsigned_be(scratch.blinding_factor.as_slice())
        || unsigned_be_cmp(scratch.blinding_factor.as_slice(), self.public.modulus()) != core::cmp::Ordering::Less
      {
        ct::zeroize(scratch.blinding_factor.as_mut_slice());
        continue;
      }

      scratch.blinding_inverse.as_mut_slice().fill(0);
      match self.blinding_factor_inverse_into_scratch(scratch) {
        Ok(()) => {
          mod_mul_representatives_with_scratch(
            &self.public.modulus,
            scratch.blinding_factor.as_slice(),
            scratch.blinding_inverse.as_slice(),
            scratch.checked.as_mut_slice(),
            &mut scratch.mul_scratch,
          )?;
          if scratch.checked.as_slice().last() == Some(&1) {
            let is_one = scratch
              .checked
              .as_slice()
              .get(..scratch.checked.as_slice().len().strict_sub(1))
              .is_some_and(|prefix| prefix.iter().all(|&byte| byte == 0));
            if is_one {
              return Ok(());
            }
          }
          ct::zeroize(scratch.blinding_factor.as_mut_slice());
          ct::zeroize(scratch.blinding_inverse.as_mut_slice());
          ct::zeroize(scratch.checked.as_mut_slice());
        }
        Err(_) => {
          ct::zeroize(scratch.blinding_factor.as_mut_slice());
          ct::zeroize(scratch.blinding_inverse.as_mut_slice());
          ct::zeroize(scratch.checked.as_mut_slice());
        }
      }
    }
    Err(RsaPrivateOpError::InvalidBlindingFactor)
  }

  #[cfg(feature = "getrandom")]
  fn blinding_factor_inverse(&self, factor: &[u8], out: &mut [u8]) -> Result<(), RsaPrivateOpError> {
    let n_len = self.public.modulus().len();
    if factor.len() != n_len || out.len() != n_len {
      return Err(RsaPrivateOpError::InvalidLength);
    }

    let prime_p = self.prime_p.as_bytes();
    let prime_q = self.prime_q.as_bytes();
    let modulus_p = &self.prime_p_modulus;
    let modulus_q = &self.prime_q_modulus;

    let factor_p = private_import_unsigned_be_mod(factor, prime_p);
    let factor_q = private_import_unsigned_be_mod(factor, prime_q);
    if is_zero_unsigned_be(factor_p.as_slice()) || is_zero_unsigned_be(factor_q.as_slice()) {
      return Err(RsaPrivateOpError::InvalidBlindingFactor);
    }

    let p_minus_one =
      private_import_decrement_unsigned_be(prime_p).map_err(|_| RsaPrivateOpError::InvalidBlindingFactor)?;
    let p_minus_two = private_import_decrement_unsigned_be(p_minus_one.as_slice())
      .map_err(|_| RsaPrivateOpError::InvalidBlindingFactor)?;
    let q_minus_one =
      private_import_decrement_unsigned_be(prime_q).map_err(|_| RsaPrivateOpError::InvalidBlindingFactor)?;
    let q_minus_two = private_import_decrement_unsigned_be(q_minus_one.as_slice())
      .map_err(|_| RsaPrivateOpError::InvalidBlindingFactor)?;

    let mut factor_p_fixed = SecretBigEndianBuffer::zeroed(prime_p.len());
    let mut factor_q_fixed = SecretBigEndianBuffer::zeroed(prime_q.len());
    left_pad_be(factor_p.as_slice(), factor_p_fixed.as_mut_slice())?;
    left_pad_be(factor_q.as_slice(), factor_q_fixed.as_mut_slice())?;

    let mut inverse_p = SecretBigEndianBuffer::zeroed(prime_p.len());
    let mut inverse_q = SecretBigEndianBuffer::zeroed(prime_q.len());
    private_exponentiate_representative(
      modulus_p,
      p_minus_two.as_slice(),
      factor_p_fixed.as_slice(),
      inverse_p.as_mut_slice(),
    )?;
    private_exponentiate_representative(
      modulus_q,
      q_minus_two.as_slice(),
      factor_q_fixed.as_slice(),
      inverse_q.as_mut_slice(),
    )?;

    let inverse_q_mod_p = private_import_unsigned_be_mod(inverse_q.as_slice(), prime_p);
    let mut inverse_q_mod_p_fixed = SecretBigEndianBuffer::zeroed(prime_p.len());
    left_pad_be(inverse_q_mod_p.as_slice(), inverse_q_mod_p_fixed.as_mut_slice())?;
    let difference = private_sub_mod_unsigned_be(inverse_p.as_slice(), inverse_q_mod_p_fixed.as_slice(), prime_p)?;
    let mut difference_fixed = SecretBigEndianBuffer::zeroed(prime_p.len());
    left_pad_be(difference.as_slice(), difference_fixed.as_mut_slice())?;

    let mut coefficient = SecretBigEndianBuffer::zeroed(prime_p.len());
    left_pad_be(self.coefficient.as_bytes(), coefficient.as_mut_slice())?;
    let mut h = SecretBigEndianBuffer::zeroed(prime_p.len());
    mod_mul_representatives(
      modulus_p,
      coefficient.as_slice(),
      difference_fixed.as_slice(),
      h.as_mut_slice(),
    )?;

    let q_times_h =
      private_import_product_unsigned_be(prime_q, h.as_slice()).ok_or(RsaPrivateOpError::RepresentativeOutOfRange)?;
    let recombined = private_add_unsigned_be_to_len(q_times_h.as_slice(), inverse_q.as_slice(), n_len)?;
    left_pad_be(recombined.as_slice(), out)
  }

  #[cfg(feature = "getrandom")]
  fn blinding_factor_inverse_into_scratch(&self, scratch: &mut RsaPrivateScratch) -> Result<(), RsaPrivateOpError> {
    let n_len = self.public.modulus().len();
    scratch.ensure_len(n_len)?;
    if scratch.blinding_factor.as_slice().len() != n_len || scratch.blinding_inverse.as_slice().len() != n_len {
      return Err(RsaPrivateOpError::InvalidLength);
    }

    let prime_p = self.prime_p.as_bytes();
    let prime_q = self.prime_q.as_bytes();
    let modulus_p = &self.prime_p_modulus;
    let modulus_q = &self.prime_q_modulus;

    {
      let factor_p = scratch
        .blinding_power
        .as_mut_slice()
        .get_mut(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?;
      private_import_unsigned_be_mod_to_fixed(
        scratch.blinding_factor.as_slice(),
        modulus_p,
        factor_p,
        &mut scratch.exponent_p_scratch,
      )?;
      if is_zero_unsigned_be(factor_p) {
        return Err(RsaPrivateOpError::InvalidBlindingFactor);
      }
    }
    {
      let factor_q = scratch
        .checked
        .as_mut_slice()
        .get_mut(..prime_q.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?;
      private_import_unsigned_be_mod_to_fixed(
        scratch.blinding_factor.as_slice(),
        modulus_q,
        factor_q,
        &mut scratch.exponent_q_scratch,
      )?;
      if is_zero_unsigned_be(factor_q) {
        return Err(RsaPrivateOpError::InvalidBlindingFactor);
      }
    }

    private_sub_small_unsigned_be_to_fixed(
      prime_p,
      2,
      scratch
        .blinded
        .as_mut_slice()
        .get_mut(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
    )?;
    private_sub_small_unsigned_be_to_fixed(
      prime_q,
      2,
      scratch
        .encoded
        .as_mut_slice()
        .get_mut(..prime_q.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
    )?;

    private_exponentiate_representative_with_scratch(
      modulus_p,
      scratch
        .blinded
        .as_slice()
        .get(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      scratch
        .blinding_power
        .as_slice()
        .get(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      scratch
        .blinded_private_result
        .as_mut_slice()
        .get_mut(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      &mut scratch.exponent_p_scratch,
    )?;
    private_exponentiate_representative_with_scratch(
      modulus_q,
      scratch
        .encoded
        .as_slice()
        .get(..prime_q.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      scratch
        .checked
        .as_slice()
        .get(..prime_q.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      scratch
        .blinded
        .as_mut_slice()
        .get_mut(..prime_q.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      &mut scratch.exponent_q_scratch,
    )?;

    private_import_unsigned_be_mod_to_fixed(
      scratch
        .blinded
        .as_slice()
        .get(..prime_q.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      modulus_p,
      scratch
        .blinding_power
        .as_mut_slice()
        .get_mut(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      &mut scratch.exponent_p_scratch,
    )?;
    private_sub_mod_unsigned_be_to_fixed(
      scratch
        .blinded_private_result
        .as_slice()
        .get(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      scratch
        .blinding_power
        .as_slice()
        .get(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      prime_p,
      scratch
        .checked
        .as_mut_slice()
        .get_mut(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
    )?;

    left_pad_be(
      self.coefficient.as_bytes(),
      scratch
        .blinding_power
        .as_mut_slice()
        .get_mut(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
    )?;
    mod_mul_representatives_with_scratch(
      modulus_p,
      scratch
        .blinding_power
        .as_slice()
        .get(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      scratch
        .checked
        .as_slice()
        .get(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      scratch
        .blinded_private_result
        .as_mut_slice()
        .get_mut(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      &mut scratch.mul_scratch,
    )?;

    private_product_add_unsigned_be_to_fixed(
      prime_q,
      scratch
        .blinded_private_result
        .as_slice()
        .get(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      scratch
        .blinded
        .as_slice()
        .get(..prime_q.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      scratch.blinding_inverse.as_mut_slice(),
      &mut scratch.mul_scratch,
    )
  }

  fn sign_pkcs1v15_with_blinding_factor(
    &self,
    profile: RsaPkcs1v15Profile,
    message: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    let mut encoded = vec![0u8; self.public.modulus().len()];
    let result = match profile {
      RsaPkcs1v15Profile::Sha256 => encode_pkcs1v15::<Sha256>(message, SHA256_DIGEST_INFO_PREFIX, &mut encoded),
      RsaPkcs1v15Profile::Sha384 => encode_pkcs1v15::<Sha384>(message, SHA384_DIGEST_INFO_PREFIX, &mut encoded),
      RsaPkcs1v15Profile::Sha512 => encode_pkcs1v15::<Sha512>(message, SHA512_DIGEST_INFO_PREFIX, &mut encoded),
    }
    .and_then(|()| self.sign_encoded_message_with_blinding_factor(&encoded, blinding, out));
    ct::zeroize(&mut encoded);
    result
  }

  fn sign_pkcs1v15_with_blinding_factor_and_scratch(
    &self,
    profile: RsaPkcs1v15Profile,
    message: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    scratch.ensure_len(self.public.modulus().len())?;
    let result = match profile {
      RsaPkcs1v15Profile::Sha256 => {
        encode_pkcs1v15::<Sha256>(message, SHA256_DIGEST_INFO_PREFIX, scratch.encoded.as_mut_slice())
      }
      RsaPkcs1v15Profile::Sha384 => {
        encode_pkcs1v15::<Sha384>(message, SHA384_DIGEST_INFO_PREFIX, scratch.encoded.as_mut_slice())
      }
      RsaPkcs1v15Profile::Sha512 => {
        encode_pkcs1v15::<Sha512>(message, SHA512_DIGEST_INFO_PREFIX, scratch.encoded.as_mut_slice())
      }
    }
    .and_then(|()| self.sign_encoded_message_with_blinding_factor_and_scratch(blinding, out, scratch));
    scratch.clear();
    result
  }

  fn sign_pkcs1v15_with_stored_blinding_and_scratch(
    &self,
    profile: RsaPkcs1v15Profile,
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    scratch.ensure_len(self.public.modulus().len())?;
    match profile {
      RsaPkcs1v15Profile::Sha256 => {
        encode_pkcs1v15::<Sha256>(message, SHA256_DIGEST_INFO_PREFIX, scratch.encoded.as_mut_slice())
      }
      RsaPkcs1v15Profile::Sha384 => {
        encode_pkcs1v15::<Sha384>(message, SHA384_DIGEST_INFO_PREFIX, scratch.encoded.as_mut_slice())
      }
      RsaPkcs1v15Profile::Sha512 => {
        encode_pkcs1v15::<Sha512>(message, SHA512_DIGEST_INFO_PREFIX, scratch.encoded.as_mut_slice())
      }
    }
    .and_then(|()| self.sign_encoded_message_with_stored_blinding_and_scratch(out, scratch))
  }

  fn sign_pss_with_salt_and_blinding_factor(
    &self,
    profile: RsaPssProfile,
    message: &[u8],
    salt: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    let em_bits = self.public.modulus_bits().strict_sub(1);
    let em_len = em_bits.strict_add(7) / 8;
    let mut encoded = vec![0u8; self.public.modulus().len()];
    let leading = encoded.len().strict_sub(em_len);
    let Some(encoded_message) = encoded.get_mut(leading..) else {
      ct::zeroize(&mut encoded);
      return Err(RsaPrivateOpError::MessageTooLong);
    };
    let result = match profile {
      RsaPssProfile::Sha256 => encode_pss::<Sha256>(message, salt, em_bits, encoded_message),
      RsaPssProfile::Sha384 => encode_pss::<Sha384>(message, salt, em_bits, encoded_message),
      RsaPssProfile::Sha512 => encode_pss::<Sha512>(message, salt, em_bits, encoded_message),
    }
    .and_then(|()| self.sign_encoded_message_with_blinding_factor(&encoded, blinding, out));
    ct::zeroize(&mut encoded);
    result
  }

  fn sign_pss_with_salt_and_blinding_factor_and_scratch(
    &self,
    profile: RsaPssProfile,
    message: &[u8],
    salt: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    scratch.ensure_len(self.public.modulus().len())?;
    let em_bits = self.public.modulus_bits().strict_sub(1);
    let em_len = em_bits.strict_add(7) / 8;
    let leading = scratch.encoded.as_slice().len().strict_sub(em_len);
    let Some(encoded_message) = scratch.encoded.as_mut_slice().get_mut(leading..) else {
      scratch.clear();
      return Err(RsaPrivateOpError::MessageTooLong);
    };
    let result = match profile {
      RsaPssProfile::Sha256 => encode_pss_with_mask::<Sha256>(
        message,
        salt,
        em_bits,
        encoded_message,
        scratch.blinding_power.as_mut_slice(),
      ),
      RsaPssProfile::Sha384 => encode_pss_with_mask::<Sha384>(
        message,
        salt,
        em_bits,
        encoded_message,
        scratch.blinding_power.as_mut_slice(),
      ),
      RsaPssProfile::Sha512 => encode_pss_with_mask::<Sha512>(
        message,
        salt,
        em_bits,
        encoded_message,
        scratch.blinding_power.as_mut_slice(),
      ),
    }
    .and_then(|()| self.sign_encoded_message_with_blinding_factor_and_scratch(blinding, out, scratch));
    scratch.clear();
    result
  }

  fn sign_pss_with_stored_salt_and_blinding_and_scratch(
    &self,
    profile: RsaPssProfile,
    message: &[u8],
    salt_len: usize,
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    scratch.ensure_len(self.public.modulus().len())?;
    let em_bits = self.public.modulus_bits().strict_sub(1);
    let em_len = em_bits.strict_add(7) / 8;
    let leading = scratch.encoded.as_slice().len().strict_sub(em_len);
    {
      let salt = scratch
        .salt
        .as_slice()
        .get(..salt_len)
        .ok_or(RsaPrivateOpError::MessageTooLong)?;
      let encoded_message = scratch
        .encoded
        .as_mut_slice()
        .get_mut(leading..)
        .ok_or(RsaPrivateOpError::MessageTooLong)?;
      match profile {
        RsaPssProfile::Sha256 => encode_pss_with_mask::<Sha256>(
          message,
          salt,
          em_bits,
          encoded_message,
          scratch.blinding_power.as_mut_slice(),
        ),
        RsaPssProfile::Sha384 => encode_pss_with_mask::<Sha384>(
          message,
          salt,
          em_bits,
          encoded_message,
          scratch.blinding_power.as_mut_slice(),
        ),
        RsaPssProfile::Sha512 => encode_pss_with_mask::<Sha512>(
          message,
          salt,
          em_bits,
          encoded_message,
          scratch.blinding_power.as_mut_slice(),
        ),
      }
    }
    .and_then(|()| self.sign_encoded_message_with_stored_blinding_and_scratch(out, scratch))
  }

  fn sign_encoded_message_with_blinding_factor(
    &self,
    encoded: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    let len = self.public.modulus().len();
    if encoded.len() != len || out.len() != len {
      return Err(RsaPrivateOpError::InvalidLength);
    }

    self.private_operation_with_blinding_factor(encoded, blinding, out)
  }

  fn sign_encoded_message_with_blinding_factor_and_scratch(
    &self,
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    let len = self.public.modulus().len();
    if out.len() != len {
      return Err(RsaPrivateOpError::InvalidLength);
    }

    self.private_operation_with_blinding_factor_and_scratch(blinding, out, scratch)
  }

  fn sign_encoded_message_with_stored_blinding_and_scratch(
    &self,
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    let len = self.public.modulus().len();
    if out.len() != len {
      return Err(RsaPrivateOpError::InvalidLength);
    }

    self
      .private_operation_from_scratch_encoded_with_stored_blinding(scratch)
      .map(|()| {
        out.copy_from_slice(scratch.blinded_private_result.as_slice());
      })
  }

  fn decrypt_oaep_with_blinding_factor(
    &self,
    profile: RsaOaepProfile,
    label: &[u8],
    ciphertext: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
  ) -> Result<usize, RsaPrivateOpError> {
    let mut encoded = vec![0u8; self.public.modulus().len()];
    let result = self
      .private_operation_with_blinding_factor(ciphertext, blinding, &mut encoded)
      .and_then(|()| match profile {
        RsaOaepProfile::Sha256 => decode_oaep::<Sha256>(label, &mut encoded, out),
        RsaOaepProfile::Sha384 => decode_oaep::<Sha384>(label, &mut encoded, out),
        RsaOaepProfile::Sha512 => decode_oaep::<Sha512>(label, &mut encoded, out),
      });
    ct::zeroize(&mut encoded);
    clear_decryption_output_on_error(result, out)
  }

  fn decrypt_pkcs1v15_with_blinding_factor(
    &self,
    ciphertext: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
  ) -> Result<usize, RsaPrivateOpError> {
    let mut encoded = vec![0u8; self.public.modulus().len()];
    let result = self
      .private_operation_with_blinding_factor(ciphertext, blinding, &mut encoded)
      .and_then(|()| decode_pkcs1v15_encryption(&encoded, out));
    ct::zeroize(&mut encoded);
    clear_decryption_output_on_error(result, out)
  }

  fn decrypt_oaep_with_stored_blinding_and_scratch(
    &self,
    profile: RsaOaepProfile,
    label: &[u8],
    ciphertext: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<usize, RsaPrivateOpError> {
    let result = scratch.ensure_len(self.public.modulus().len()).and_then(|()| {
      if ciphertext.len() != scratch.encoded.as_slice().len() {
        Err(RsaPrivateOpError::InvalidLength)
      } else {
        scratch.encoded.as_mut_slice().copy_from_slice(ciphertext);
        self
          .private_operation_from_scratch_encoded_with_stored_blinding(scratch)
          .map(|()| {
            scratch
              .encoded
              .as_mut_slice()
              .copy_from_slice(scratch.blinded_private_result.as_slice())
          })
          .and_then(|()| match profile {
            RsaOaepProfile::Sha256 => decode_oaep_with_masks::<Sha256>(
              label,
              scratch.encoded.as_mut_slice(),
              out,
              scratch.blinding_power.as_mut_slice(),
              scratch.blinded.as_mut_slice(),
            ),
            RsaOaepProfile::Sha384 => decode_oaep_with_masks::<Sha384>(
              label,
              scratch.encoded.as_mut_slice(),
              out,
              scratch.blinding_power.as_mut_slice(),
              scratch.blinded.as_mut_slice(),
            ),
            RsaOaepProfile::Sha512 => decode_oaep_with_masks::<Sha512>(
              label,
              scratch.encoded.as_mut_slice(),
              out,
              scratch.blinding_power.as_mut_slice(),
              scratch.blinded.as_mut_slice(),
            ),
          })
      }
    });
    clear_decryption_output_on_error(result, out)
  }

  fn decrypt_pkcs1v15_with_stored_blinding_and_scratch(
    &self,
    ciphertext: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<usize, RsaPrivateOpError> {
    let result = scratch.ensure_len(self.public.modulus().len()).and_then(|()| {
      if ciphertext.len() != scratch.encoded.as_slice().len() {
        Err(RsaPrivateOpError::InvalidLength)
      } else {
        scratch.encoded.as_mut_slice().copy_from_slice(ciphertext);
        self
          .private_operation_from_scratch_encoded_with_stored_blinding(scratch)
          .and_then(|()| decode_pkcs1v15_encryption(scratch.blinded_private_result.as_slice(), out))
      }
    });
    clear_decryption_output_on_error(result, out)
  }

  fn decrypt_oaep_with_blinding_factor_and_scratch(
    &self,
    profile: RsaOaepProfile,
    label: &[u8],
    ciphertext: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<usize, RsaPrivateOpError> {
    let result = scratch.ensure_len(self.public.modulus().len()).and_then(|()| {
      if ciphertext.len() != scratch.encoded.as_slice().len() {
        Err(RsaPrivateOpError::InvalidLength)
      } else {
        scratch.encoded.as_mut_slice().copy_from_slice(ciphertext);
        self
          .private_operation_from_scratch_encoded(blinding, scratch)
          .map(|()| {
            scratch
              .encoded
              .as_mut_slice()
              .copy_from_slice(scratch.blinded_private_result.as_slice())
          })
          .and_then(|()| match profile {
            RsaOaepProfile::Sha256 => decode_oaep_with_masks::<Sha256>(
              label,
              scratch.encoded.as_mut_slice(),
              out,
              scratch.blinding_power.as_mut_slice(),
              scratch.blinded.as_mut_slice(),
            ),
            RsaOaepProfile::Sha384 => decode_oaep_with_masks::<Sha384>(
              label,
              scratch.encoded.as_mut_slice(),
              out,
              scratch.blinding_power.as_mut_slice(),
              scratch.blinded.as_mut_slice(),
            ),
            RsaOaepProfile::Sha512 => decode_oaep_with_masks::<Sha512>(
              label,
              scratch.encoded.as_mut_slice(),
              out,
              scratch.blinding_power.as_mut_slice(),
              scratch.blinded.as_mut_slice(),
            ),
          })
      }
    });
    scratch.clear();
    clear_decryption_output_on_error(result, out)
  }

  fn decrypt_pkcs1v15_with_blinding_factor_and_scratch(
    &self,
    ciphertext: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<usize, RsaPrivateOpError> {
    let result = scratch.ensure_len(self.public.modulus().len()).and_then(|()| {
      if ciphertext.len() != scratch.encoded.as_slice().len() {
        Err(RsaPrivateOpError::InvalidLength)
      } else {
        scratch.encoded.as_mut_slice().copy_from_slice(ciphertext);
        self
          .private_operation_from_scratch_encoded(blinding, scratch)
          .and_then(|()| decode_pkcs1v15_encryption(scratch.blinded_private_result.as_slice(), out))
      }
    });
    scratch.clear();
    clear_decryption_output_on_error(result, out)
  }

  fn private_operation_with_blinding_factor(
    &self,
    input: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    let len = self.public.modulus().len();
    if input.len() != len || out.len() != len || blinding.factor.len() != len || blinding.inverse.len() != len {
      return Err(RsaPrivateOpError::InvalidLength);
    }

    if blinding.validate {
      let mut one = vec![0u8; len];
      let Some(last) = one.last_mut() else {
        return Err(RsaPrivateOpError::InvalidLength);
      };
      *last = 1;
      let mut factor_check = vec![0u8; len];
      mod_mul_representatives(
        &self.public.modulus,
        blinding.factor,
        blinding.inverse,
        &mut factor_check,
      )?;
      if !ct::constant_time_eq(&factor_check, &one) {
        return Err(RsaPrivateOpError::InvalidBlindingFactor);
      }
    }

    let mut blinding_power = SecretBigEndianBuffer::zeroed(len);
    let mut scratch = self.public.public_scratch();
    self
      .public
      .public_operation_with_scratch(blinding.factor, blinding_power.as_mut_slice(), &mut scratch)
      .map_err(|_| RsaPrivateOpError::InvalidBlindingFactor)?;

    let mut blinded = SecretBigEndianBuffer::zeroed(len);
    mod_mul_representatives(
      &self.public.modulus,
      input,
      blinding_power.as_slice(),
      blinded.as_mut_slice(),
    )?;
    let mut blinded_signature = SecretBigEndianBuffer::zeroed(len);
    self.private_operation(blinded.as_slice(), blinded_signature.as_mut_slice())?;
    mod_mul_representatives(
      &self.public.modulus,
      blinded_signature.as_slice(),
      blinding.inverse,
      out,
    )?;

    let mut checked = vec![0u8; len];
    self
      .public
      .public_operation_with_scratch(out, &mut checked, &mut scratch)
      .map_err(|_| RsaPrivateOpError::FaultCheckFailed)?;
    if ct::constant_time_eq(&checked, input) {
      Ok(())
    } else {
      Err(RsaPrivateOpError::FaultCheckFailed)
    }
  }

  fn private_operation_with_blinding_factor_and_scratch(
    &self,
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    if out.len() != self.public.modulus().len() {
      return Err(RsaPrivateOpError::InvalidLength);
    }

    self
      .private_operation_from_scratch_encoded(blinding, scratch)
      .map(|()| {
        out.copy_from_slice(scratch.blinded_private_result.as_slice());
      })
  }

  fn private_operation_from_scratch_encoded(
    &self,
    blinding: RsaBlindingPair<'_>,
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    let len = self.public.modulus().len();
    if scratch.encoded.as_slice().len() != len || blinding.factor.len() != len || blinding.inverse.len() != len {
      return Err(RsaPrivateOpError::InvalidLength);
    }
    scratch.ensure_len(len)?;

    if blinding.validate {
      scratch.set_one()?;
      mod_mul_representatives_with_scratch(
        &self.public.modulus,
        blinding.factor,
        blinding.inverse,
        scratch.checked.as_mut_slice(),
        &mut scratch.mul_scratch,
      )?;
      if !ct::constant_time_eq(scratch.checked.as_slice(), scratch.one.as_slice()) {
        return Err(RsaPrivateOpError::InvalidBlindingFactor);
      }
    }

    self
      .public
      .public_operation_with_scratch(
        blinding.factor,
        scratch.blinding_power.as_mut_slice(),
        &mut scratch.public_scratch,
      )
      .map_err(|_| RsaPrivateOpError::InvalidBlindingFactor)?;

    mod_mul_representatives_with_scratch(
      &self.public.modulus,
      scratch.encoded.as_slice(),
      scratch.blinding_power.as_slice(),
      scratch.blinded.as_mut_slice(),
      &mut scratch.mul_scratch,
    )?;
    self.private_operation_crt_from_blinded_scratch(scratch)?;
    mod_mul_representatives_with_scratch(
      &self.public.modulus,
      scratch.blinded_private_result.as_slice(),
      blinding.inverse,
      scratch.blinded.as_mut_slice(),
      &mut scratch.mul_scratch,
    )?;

    self
      .public
      .public_operation_with_scratch(
        scratch.blinded.as_slice(),
        scratch.checked.as_mut_slice(),
        &mut scratch.public_scratch,
      )
      .map_err(|_| RsaPrivateOpError::FaultCheckFailed)?;
    if ct::constant_time_eq(scratch.checked.as_slice(), scratch.encoded.as_slice()) {
      scratch
        .blinded_private_result
        .as_mut_slice()
        .copy_from_slice(scratch.blinded.as_slice());
      Ok(())
    } else {
      Err(RsaPrivateOpError::FaultCheckFailed)
    }
  }

  fn private_operation_from_scratch_encoded_with_stored_blinding(
    &self,
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    let len = self.public.modulus().len();
    if scratch.encoded.as_slice().len() != len
      || scratch.blinding_factor.as_slice().len() != len
      || scratch.blinding_inverse.as_slice().len() != len
    {
      return Err(RsaPrivateOpError::InvalidLength);
    }
    scratch.ensure_len(len)?;

    self
      .public
      .public_operation_with_scratch(
        scratch.blinding_factor.as_slice(),
        scratch.blinding_power.as_mut_slice(),
        &mut scratch.public_scratch,
      )
      .map_err(|_| RsaPrivateOpError::InvalidBlindingFactor)?;

    mod_mul_representatives_with_scratch(
      &self.public.modulus,
      scratch.encoded.as_slice(),
      scratch.blinding_power.as_slice(),
      scratch.blinded.as_mut_slice(),
      &mut scratch.mul_scratch,
    )?;
    self.private_operation_crt_from_blinded_scratch(scratch)?;
    mod_mul_representatives_with_scratch(
      &self.public.modulus,
      scratch.blinded_private_result.as_slice(),
      scratch.blinding_inverse.as_slice(),
      scratch.blinded.as_mut_slice(),
      &mut scratch.mul_scratch,
    )?;

    self
      .public
      .public_operation_with_scratch(
        scratch.blinded.as_slice(),
        scratch.checked.as_mut_slice(),
        &mut scratch.public_scratch,
      )
      .map_err(|_| RsaPrivateOpError::FaultCheckFailed)?;
    if ct::constant_time_eq(scratch.checked.as_slice(), scratch.encoded.as_slice()) {
      scratch
        .blinded_private_result
        .as_mut_slice()
        .copy_from_slice(scratch.blinded.as_slice());
      Ok(())
    } else {
      Err(RsaPrivateOpError::FaultCheckFailed)
    }
  }

  fn private_operation(&self, input: &[u8], out: &mut [u8]) -> Result<(), RsaPrivateOpError> {
    self.private_operation_crt(input, out)
  }

  fn private_operation_crt(&self, input: &[u8], out: &mut [u8]) -> Result<(), RsaPrivateOpError> {
    let n_len = self.public.modulus().len();
    if input.len() != n_len || out.len() != n_len {
      return Err(RsaPrivateOpError::InvalidLength);
    }

    let prime_p = self.prime_p.as_bytes();
    let prime_q = self.prime_q.as_bytes();
    let modulus_p = &self.prime_p_modulus;
    let modulus_q = &self.prime_q_modulus;

    let input_p = private_import_unsigned_be_mod(input, prime_p);
    let input_q = private_import_unsigned_be_mod(input, prime_q);
    let mut representative_p = SecretBigEndianBuffer::zeroed(prime_p.len());
    let mut representative_q = SecretBigEndianBuffer::zeroed(prime_q.len());
    left_pad_be(input_p.as_slice(), representative_p.as_mut_slice())?;
    left_pad_be(input_q.as_slice(), representative_q.as_mut_slice())?;

    let mut m1 = SecretBigEndianBuffer::zeroed(prime_p.len());
    let mut m2 = SecretBigEndianBuffer::zeroed(prime_q.len());
    private_exponentiate_representative(
      modulus_p,
      self.exponent_p.as_bytes(),
      representative_p.as_slice(),
      m1.as_mut_slice(),
    )?;
    private_exponentiate_representative(
      modulus_q,
      self.exponent_q.as_bytes(),
      representative_q.as_slice(),
      m2.as_mut_slice(),
    )?;

    let m2_mod_p = private_import_unsigned_be_mod(m2.as_slice(), prime_p);
    let mut m2_mod_p_fixed = SecretBigEndianBuffer::zeroed(prime_p.len());
    left_pad_be(m2_mod_p.as_slice(), m2_mod_p_fixed.as_mut_slice())?;
    let difference = private_sub_mod_unsigned_be(m1.as_slice(), m2_mod_p_fixed.as_slice(), prime_p)?;
    let mut difference_fixed = SecretBigEndianBuffer::zeroed(prime_p.len());
    left_pad_be(difference.as_slice(), difference_fixed.as_mut_slice())?;

    let mut coefficient = SecretBigEndianBuffer::zeroed(prime_p.len());
    left_pad_be(self.coefficient.as_bytes(), coefficient.as_mut_slice())?;
    let mut h = SecretBigEndianBuffer::zeroed(prime_p.len());
    mod_mul_representatives(
      modulus_p,
      coefficient.as_slice(),
      difference_fixed.as_slice(),
      h.as_mut_slice(),
    )?;

    let q_times_h =
      private_import_product_unsigned_be(prime_q, h.as_slice()).ok_or(RsaPrivateOpError::RepresentativeOutOfRange)?;
    let recombined = private_add_unsigned_be_to_len(q_times_h.as_slice(), m2.as_slice(), n_len)?;
    left_pad_be(recombined.as_slice(), out)?;
    Ok(())
  }

  fn private_operation_crt_from_blinded_scratch(
    &self,
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    let n_len = self.public.modulus().len();
    if scratch.blinded.as_slice().len() != n_len || scratch.blinded_private_result.as_slice().len() != n_len {
      return Err(RsaPrivateOpError::InvalidLength);
    }

    let prime_p = self.prime_p.as_bytes();
    let prime_q = self.prime_q.as_bytes();
    let modulus_p = &self.prime_p_modulus;
    let modulus_q = &self.prime_q_modulus;
    scratch.exponent_p_scratch.ensure_limb_count(modulus_p.limbs.len())?;
    scratch.exponent_q_scratch.ensure_limb_count(modulus_q.limbs.len())?;

    {
      let representative_p = scratch
        .blinding_power
        .as_mut_slice()
        .get_mut(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?;
      private_import_unsigned_be_mod_to_fixed(
        scratch.blinded.as_slice(),
        modulus_p,
        representative_p,
        &mut scratch.exponent_p_scratch,
      )?;
    }
    {
      let representative_q = scratch
        .checked
        .as_mut_slice()
        .get_mut(..prime_q.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?;
      private_import_unsigned_be_mod_to_fixed(
        scratch.blinded.as_slice(),
        modulus_q,
        representative_q,
        &mut scratch.exponent_q_scratch,
      )?;
    }

    private_exponentiate_representative_with_scratch(
      modulus_p,
      self.exponent_p.as_bytes(),
      scratch
        .blinding_power
        .as_slice()
        .get(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      scratch
        .blinded_private_result
        .as_mut_slice()
        .get_mut(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      &mut scratch.exponent_p_scratch,
    )?;
    private_exponentiate_representative_with_scratch(
      modulus_q,
      self.exponent_q.as_bytes(),
      scratch
        .checked
        .as_slice()
        .get(..prime_q.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      scratch
        .blinded
        .as_mut_slice()
        .get_mut(..prime_q.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      &mut scratch.exponent_q_scratch,
    )?;

    private_import_unsigned_be_mod_to_fixed(
      scratch
        .blinded
        .as_slice()
        .get(..prime_q.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      modulus_p,
      scratch
        .blinding_power
        .as_mut_slice()
        .get_mut(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      &mut scratch.exponent_p_scratch,
    )?;
    private_sub_mod_unsigned_be_to_fixed(
      scratch
        .blinded_private_result
        .as_slice()
        .get(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      scratch
        .blinding_power
        .as_slice()
        .get(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      prime_p,
      scratch
        .checked
        .as_mut_slice()
        .get_mut(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
    )?;

    left_pad_be(
      self.coefficient.as_bytes(),
      scratch
        .blinding_power
        .as_mut_slice()
        .get_mut(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
    )?;
    mod_mul_representatives_with_scratch(
      modulus_p,
      scratch
        .blinding_power
        .as_slice()
        .get(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      scratch
        .checked
        .as_slice()
        .get(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      scratch
        .blinded_private_result
        .as_mut_slice()
        .get_mut(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      &mut scratch.mul_scratch,
    )?;

    private_product_add_unsigned_be_to_fixed(
      prime_q,
      scratch
        .blinded_private_result
        .as_slice()
        .get(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      scratch
        .blinded
        .as_slice()
        .get(..prime_q.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      scratch.checked.as_mut_slice(),
      &mut scratch.mul_scratch,
    )?;
    scratch
      .blinded_private_result
      .as_mut_slice()
      .copy_from_slice(scratch.checked.as_slice());
    Ok(())
  }
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

#[allow(dead_code)]
struct SecretBigEndianInteger {
  bytes: Box<[u8]>,
}

#[allow(dead_code)]
impl SecretBigEndianInteger {
  fn new(bytes: &[u8]) -> Result<Self, RsaKeyError> {
    if is_zero_unsigned_be(bytes) {
      return Err(RsaKeyError::InvalidModulus);
    }
    Ok(Self {
      bytes: Box::from(bytes),
    })
  }

  fn from_vec(mut bytes: Vec<u8>) -> Result<Self, RsaKeyError> {
    if is_zero_unsigned_be(&bytes) {
      ct::zeroize(&mut bytes);
      return Err(RsaKeyError::InvalidModulus);
    }
    Ok(Self {
      bytes: bytes.into_boxed_slice(),
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

#[cfg(feature = "getrandom")]
impl RsaBlindingFactor {
  #[inline]
  fn factor(&self) -> &[u8] {
    self.factor.as_slice()
  }

  #[inline]
  fn inverse(&self) -> &[u8] {
    self.inverse.as_slice()
  }
}

struct RsaBlindingPair<'a> {
  factor: &'a [u8],
  inverse: &'a [u8],
  validate: bool,
}

impl<'a> RsaBlindingPair<'a> {
  #[cfg(feature = "getrandom")]
  #[inline]
  const fn trusted(factor: &'a [u8], inverse: &'a [u8]) -> Self {
    Self {
      factor,
      inverse,
      validate: false,
    }
  }

  #[inline]
  const fn caller_supplied(factor: &'a [u8], inverse: &'a [u8]) -> Self {
    Self {
      factor,
      inverse,
      validate: true,
    }
  }
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
  limb_count: usize,
}

impl RsaPrivateExponentScratch {
  fn new(limb_count: usize) -> Self {
    Self {
      t: SecretLimbs::zeroed(limb_count.strict_mul(2).strict_add(2)),
      representative: SecretLimbs::zeroed(limb_count),
      one: SecretLimbs::zeroed(limb_count),
      base: SecretLimbs::zeroed(limb_count),
      acc: SecretLimbs::zeroed(limb_count),
      squared: SecretLimbs::zeroed(limb_count),
      multiplied: SecretLimbs::zeroed(limb_count),
      selected: SecretLimbs::zeroed(limb_count),
      reduced: SecretLimbs::zeroed(limb_count),
      table: SecretLimbs::zeroed(limb_count.strict_mul(PRIVATE_FIXED_WINDOW_TABLE_ENTRIES)),
      limb_count,
    }
  }

  fn ensure_limb_count(&self, limb_count: usize) -> Result<(), RsaPrivateOpError> {
    if self.limb_count == limb_count
      && self.t.as_slice().len() == limb_count.strict_mul(2).strict_add(2)
      && self.representative.as_slice().len() == limb_count
      && self.one.as_slice().len() == limb_count
      && self.base.as_slice().len() == limb_count
      && self.acc.as_slice().len() == limb_count
      && self.squared.as_slice().len() == limb_count
      && self.multiplied.as_slice().len() == limb_count
      && self.selected.as_slice().len() == limb_count
      && self.reduced.as_slice().len() == limb_count
      && self.table.as_slice().len() == limb_count.strict_mul(PRIVATE_FIXED_WINDOW_TABLE_ENTRIES)
    {
      Ok(())
    } else {
      Err(RsaPrivateOpError::InvalidScratch)
    }
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

impl RsaTlsSignatureSchemes {
  /// Maximum number of RSA TLS schemes a single key algorithm can advertise.
  pub const MAX_LEN: usize = RSA_TLS_SIGNATURE_SCHEME_CAPACITY;

  const EMPTY: Self = Self {
    schemes: [0; RSA_TLS_SIGNATURE_SCHEME_CAPACITY],
    len: 0,
  };

  #[inline]
  const fn new(schemes: [u16; RSA_TLS_SIGNATURE_SCHEME_CAPACITY], len: u8) -> Self {
    Self { schemes, len }
  }

  /// Number of advertised schemes.
  #[inline]
  #[must_use]
  pub const fn len(&self) -> usize {
    self.len as usize
  }

  /// Return `true` when no schemes are advertised.
  #[inline]
  #[must_use]
  pub const fn is_empty(&self) -> bool {
    self.len == 0
  }

  /// Iterate over the advertised scheme IDs.
  #[inline]
  pub fn iter(&self) -> impl Iterator<Item = u16> + '_ {
    self.schemes.iter().copied().take(self.len())
  }

  /// Return `true` if this list advertises `scheme`.
  #[inline]
  #[must_use]
  pub fn contains(&self, scheme: u16) -> bool {
    self.iter().any(|advertised| advertised == scheme)
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RsaTlsPublicKeyAlgorithm {
  RsaEncryption,
  RsaPss,
}

impl RsaX509PublicKeyAlgorithm {
  /// Check whether this key algorithm permits a signature profile.
  ///
  /// # Errors
  ///
  /// Returns [`RsaProtocolAlgorithmError::UnsupportedAlgorithm`] when the
  /// signature profile conflicts with X.509 key-algorithm constraints.
  #[inline]
  pub fn permits_signature_profile(self, profile: RsaSignatureProfile) -> Result<(), RsaProtocolAlgorithmError> {
    match self {
      Self::RsaEncryption => Ok(()),
      Self::RsaPss => match profile {
        RsaSignatureProfile::Pss { .. } => Ok(()),
        RsaSignatureProfile::Pkcs1v15(_) => Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm),
      },
      Self::RsaPssRestricted {
        profile: required,
        minimum_salt_len,
      } => match profile {
        RsaSignatureProfile::Pss { profile, salt_len } if profile == required && salt_len >= minimum_salt_len => Ok(()),
        RsaSignatureProfile::Pss { .. } | RsaSignatureProfile::Pkcs1v15(_) => {
          Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
        }
      },
    }
  }

  /// Map and constrain a TLS 1.3 `CertificateVerify` RSA `SignatureScheme`.
  ///
  /// `rsa_pss_rsae_*` schemes require an X.509 `rsaEncryption` public-key
  /// algorithm. `rsa_pss_pss_*` schemes require an X.509 `id-RSASSA-PSS`
  /// public-key algorithm. PKCS#1 v1.5, SHA-1, and RFC 9963 legacy
  /// client-certificate code points are rejected by the primitive default.
  ///
  /// # Errors
  ///
  /// Returns [`RsaProtocolAlgorithmError::UnsupportedAlgorithm`] for unknown,
  /// non-RSA, SHA-1, PKCS#1 v1.5, RFC 9963 legacy, or
  /// key-algorithm-confused scheme IDs.
  #[inline]
  pub fn signature_profile_from_tls13_signature_scheme(
    self,
    scheme: u16,
  ) -> Result<RsaSignatureProfile, RsaProtocolAlgorithmError> {
    let (profile, required_key_algorithm) = tls13_signature_scheme_profile_and_key_algorithm(scheme)?;
    self.permits_tls_key_algorithm(required_key_algorithm, profile)?;
    Ok(profile)
  }

  /// Map and constrain a TLS certificate RSA `SignatureScheme`.
  ///
  /// This includes legacy SHA-2 PKCS#1 v1.5 certificate-chain signatures for
  /// `rsaEncryption` keys, plus TLS 1.3 RSA-PSS RSAE/PSS schemes with their
  /// required X.509 public-key algorithm. RFC 9963 client `CertificateVerify`
  /// legacy code points are intentionally not certificate-chain schemes.
  ///
  /// # Errors
  ///
  /// Returns [`RsaProtocolAlgorithmError::UnsupportedAlgorithm`] for unknown,
  /// non-RSA, SHA-1, or key-algorithm-confused scheme IDs.
  #[inline]
  pub fn signature_profile_from_tls_certificate_signature_scheme(
    self,
    scheme: u16,
  ) -> Result<RsaSignatureProfile, RsaProtocolAlgorithmError> {
    let (profile, required_key_algorithm) = tls_certificate_signature_scheme_profile_and_key_algorithm(scheme)?;
    self.permits_tls_key_algorithm(required_key_algorithm, profile)?;
    Ok(profile)
  }

  /// Return TLS 1.3 RSA `CertificateVerify` schemes this key algorithm may advertise.
  ///
  /// Providers should use this list rather than advertising all RSA schemes:
  /// `rsa_pss_rsae_*` is valid only for `rsaEncryption` keys and
  /// `rsa_pss_pss_*` is valid only for `id-RSASSA-PSS` keys.
  #[inline]
  #[must_use]
  pub const fn advertised_tls13_signature_schemes(self) -> RsaTlsSignatureSchemes {
    match self {
      Self::RsaEncryption => RsaTlsSignatureSchemes::new([0x0804, 0x0805, 0x0806, 0, 0, 0], 3),
      Self::RsaPss => RsaTlsSignatureSchemes::new([0x0809, 0x080a, 0x080b, 0, 0, 0], 3),
      Self::RsaPssRestricted {
        profile,
        minimum_salt_len,
      } => advertised_restricted_pss_tls13_signature_scheme(profile, minimum_salt_len),
    }
  }

  /// Return TLS certificate RSA signature schemes this key algorithm may advertise.
  ///
  /// `rsaEncryption` keys may advertise SHA-2 PKCS#1 v1.5 compatibility
  /// schemes and RSAE-PSS schemes. PSS keys advertise only PSS-PSS schemes
  /// permitted by their SPKI parameters.
  #[inline]
  #[must_use]
  pub const fn advertised_tls_certificate_signature_schemes(self) -> RsaTlsSignatureSchemes {
    match self {
      Self::RsaEncryption => RsaTlsSignatureSchemes::new([0x0804, 0x0805, 0x0806, 0x0401, 0x0501, 0x0601], 6),
      Self::RsaPss => RsaTlsSignatureSchemes::new([0x0809, 0x080a, 0x080b, 0, 0, 0], 3),
      Self::RsaPssRestricted {
        profile,
        minimum_salt_len,
      } => advertised_restricted_pss_tls13_signature_scheme(profile, minimum_salt_len),
    }
  }

  fn permits_tls_key_algorithm(
    self,
    required: RsaTlsPublicKeyAlgorithm,
    profile: RsaSignatureProfile,
  ) -> Result<(), RsaProtocolAlgorithmError> {
    match required {
      RsaTlsPublicKeyAlgorithm::RsaEncryption => match self {
        Self::RsaEncryption => self.permits_signature_profile(profile),
        Self::RsaPss | Self::RsaPssRestricted { .. } => Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm),
      },
      RsaTlsPublicKeyAlgorithm::RsaPss => match self {
        Self::RsaEncryption => Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm),
        Self::RsaPss | Self::RsaPssRestricted { .. } => self.permits_signature_profile(profile),
      },
    }
  }
}

/// RSA public key paired with its X.509 SPKI algorithm constraints.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct RsaX509PublicKey {
  key: RsaPublicKey,
  algorithm: RsaX509PublicKeyAlgorithm,
}

impl RsaX509PublicKey {
  /// Parse an X.509 RSA `SubjectPublicKeyInfo` DER object with the default policy.
  ///
  /// Unlike [`RsaPublicKey::from_spki_der`], this accepts both `rsaEncryption`
  /// and `id-RSASSA-PSS` SPKI algorithms while preserving PSS key constraints.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if the DER is malformed, the algorithm is not a
  /// supported RSA public-key algorithm, or the embedded key violates policy.
  pub fn from_spki_der(der: &[u8]) -> Result<Self, RsaKeyError> {
    Self::from_spki_der_with_policy(der, &RsaPublicKeyPolicy::default())
  }

  /// Parse an X.509 RSA `SubjectPublicKeyInfo` DER object with an explicit policy.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if parsing or validation fails.
  pub fn from_spki_der_with_policy(der: &[u8], policy: &RsaPublicKeyPolicy) -> Result<Self, RsaKeyError> {
    let (algorithm_der, public_key_der) = parse_spki_der(der)?;
    let algorithm = parse_x509_public_key_algorithm_identifier(algorithm_der)?;
    let key = RsaPublicKey::from_pkcs1_der_with_policy(public_key_der, policy)?;
    Ok(Self { key, algorithm })
  }

  /// Return the validated RSA public key.
  #[inline]
  #[must_use]
  pub const fn public_key(&self) -> &RsaPublicKey {
    &self.key
  }

  /// Return the parsed X.509 public-key algorithm constraints.
  #[inline]
  #[must_use]
  pub const fn key_algorithm(&self) -> RsaX509PublicKeyAlgorithm {
    self.algorithm
  }

  /// Parse and constrain an X.509 signature `AlgorithmIdentifier`.
  ///
  /// # Errors
  ///
  /// Returns [`RsaProtocolAlgorithmError`] if the signature algorithm is
  /// malformed, unsupported, or conflicts with the SPKI key algorithm.
  #[inline]
  pub fn signature_profile_from_x509_algorithm_der(
    &self,
    der: &[u8],
  ) -> Result<RsaSignatureProfile, RsaProtocolAlgorithmError> {
    let profile = RsaSignatureProfile::from_x509_signature_algorithm_der(der)?;
    self.algorithm.permits_signature_profile(profile)?;
    Ok(profile)
  }

  /// Map and constrain a TLS 1.3 `CertificateVerify` RSA `SignatureScheme`.
  ///
  /// # Errors
  ///
  /// Returns [`RsaProtocolAlgorithmError`] if the scheme is unsupported or
  /// conflicts with this key's X.509 public-key algorithm.
  #[inline]
  pub fn signature_profile_from_tls13_signature_scheme(
    &self,
    scheme: u16,
  ) -> Result<RsaSignatureProfile, RsaProtocolAlgorithmError> {
    self.algorithm.signature_profile_from_tls13_signature_scheme(scheme)
  }

  /// Map and constrain a TLS certificate RSA `SignatureScheme`.
  ///
  /// # Errors
  ///
  /// Returns [`RsaProtocolAlgorithmError`] if the scheme is unsupported or
  /// conflicts with this key's X.509 public-key algorithm.
  #[inline]
  pub fn signature_profile_from_tls_certificate_signature_scheme(
    &self,
    scheme: u16,
  ) -> Result<RsaSignatureProfile, RsaProtocolAlgorithmError> {
    self
      .algorithm
      .signature_profile_from_tls_certificate_signature_scheme(scheme)
  }

  /// Verify a signature using an X.509 signature `AlgorithmIdentifier`.
  ///
  /// Primitive helper only: this is not a WebPKI, TLS, JWT, COSE, or
  /// certificate-chain provider integration.
  ///
  /// Algorithm mismatch and unsupported algorithm choices collapse to the same
  /// opaque verification error as invalid signatures.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if the algorithm is malformed, unsupported,
  /// conflicts with this key's SPKI constraints, or the signature is invalid.
  #[inline]
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_signature_from_x509_algorithm_der(
    &self,
    signature_algorithm_der: &[u8],
    message: &[u8],
    signature: &[u8],
  ) -> Result<(), VerificationError> {
    let profile = self
      .signature_profile_from_x509_algorithm_der(signature_algorithm_der)
      .map_err(|_| VerificationError::new())?;
    if !self.key.signature_profile_is_possible(profile) {
      return Err(VerificationError::new());
    }
    let mut scratch = self.key.public_scratch();
    self
      .key
      .verify_signature_with_scratch(profile, message, signature, &mut scratch)
  }

  /// Verify a signature using an X.509 signature `AlgorithmIdentifier` and caller-owned scratch.
  ///
  /// Primitive helper only: this is not a WebPKI, TLS, JWT, COSE, or
  /// certificate-chain provider integration.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if the algorithm is malformed, unsupported,
  /// conflicts with this key's SPKI constraints, or the signature is invalid.
  #[inline]
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_signature_from_x509_algorithm_der_with_scratch(
    &self,
    signature_algorithm_der: &[u8],
    message: &[u8],
    signature: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    let profile = self
      .signature_profile_from_x509_algorithm_der(signature_algorithm_der)
      .map_err(|_| VerificationError::new())?;
    self
      .key
      .verify_signature_with_scratch(profile, message, signature, scratch)
  }

  /// Verify the signature on an X.509 certificate DER object.
  ///
  /// Primitive helper only: this is not a WebPKI or certificate-chain provider
  /// integration.
  ///
  /// This helper verifies only the certificate signature. It does not build or
  /// validate a certificate chain, names, validity periods, key usage, or other
  /// WebPKI policy. The caller supplies the issuer public key represented by
  /// `self`.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if the certificate DER is malformed, the
  /// TBSCertificate and outer signature algorithms differ, the signature
  /// algorithm is unsupported or conflicts with this issuer key, or the
  /// signature is invalid.
  #[inline]
  #[must_use = "certificate signature verification must be checked; a dropped Result silently accepts a forged \
                certificate"]
  pub fn verify_x509_certificate_signature_der(&self, certificate_der: &[u8]) -> Result<(), VerificationError> {
    let parsed = parse_x509_certificate_signature(certificate_der).map_err(|_| VerificationError::new())?;
    let profile = self
      .signature_profile_from_x509_algorithm_der(parsed.signature_algorithm_der)
      .map_err(|_| VerificationError::new())?;
    if !self.key.signature_profile_is_possible(profile) {
      return Err(VerificationError::new());
    }
    let mut scratch = self.key.public_scratch();
    self
      .key
      .verify_signature_with_scratch(profile, parsed.tbs_certificate_der, parsed.signature, &mut scratch)
  }

  /// Verify the signature on an X.509 certificate DER object using caller-owned scratch.
  ///
  /// Primitive helper only: this is not a WebPKI or certificate-chain provider
  /// integration.
  ///
  /// This helper verifies only the certificate signature. It does not build or
  /// validate a certificate chain, names, validity periods, key usage, or other
  /// WebPKI policy. The caller supplies the issuer public key represented by
  /// `self`.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if the certificate DER is malformed, the
  /// TBSCertificate and outer signature algorithms differ, the signature
  /// algorithm is unsupported or conflicts with this issuer key, or the
  /// signature is invalid.
  #[inline]
  #[must_use = "certificate signature verification must be checked; a dropped Result silently accepts a forged \
                certificate"]
  pub fn verify_x509_certificate_signature_der_with_scratch(
    &self,
    certificate_der: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    let parsed = parse_x509_certificate_signature(certificate_der).map_err(|_| VerificationError::new())?;
    self.verify_signature_from_x509_algorithm_der_with_scratch(
      parsed.signature_algorithm_der,
      parsed.tbs_certificate_der,
      parsed.signature,
      scratch,
    )
  }

  /// Verify a TLS 1.3 `CertificateVerify` RSA signature.
  ///
  /// Primitive helper only: this is not a TLS provider integration. The caller
  /// must construct the exact TLS `CertificateVerify` signed message and
  /// enforce handshake, certificate-chain, and policy state.
  ///
  /// Scheme mismatch and unsupported algorithm choices collapse to the same
  /// opaque verification error as invalid signatures.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if the scheme is unsupported, conflicts
  /// with this key's X.509 public-key algorithm, or the signature is invalid.
  #[inline]
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_tls13_signature_scheme(
    &self,
    scheme: u16,
    message: &[u8],
    signature: &[u8],
  ) -> Result<(), VerificationError> {
    let profile = self
      .signature_profile_from_tls13_signature_scheme(scheme)
      .map_err(|_| VerificationError::new())?;
    if !self.key.signature_profile_is_possible(profile) {
      return Err(VerificationError::new());
    }
    let mut scratch = self.key.public_scratch();
    self
      .key
      .verify_signature_with_scratch(profile, message, signature, &mut scratch)
  }

  /// Verify a TLS 1.3 `CertificateVerify` RSA signature with caller-owned scratch.
  ///
  /// Primitive helper only: this is not a TLS provider integration. The caller
  /// must construct the exact TLS `CertificateVerify` signed message and
  /// enforce handshake, certificate-chain, and policy state.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if the scheme is unsupported, conflicts
  /// with this key's X.509 public-key algorithm, or the signature is invalid.
  #[inline]
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_tls13_signature_scheme_with_scratch(
    &self,
    scheme: u16,
    message: &[u8],
    signature: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    let profile = self
      .signature_profile_from_tls13_signature_scheme(scheme)
      .map_err(|_| VerificationError::new())?;
    self
      .key
      .verify_signature_with_scratch(profile, message, signature, scratch)
  }

  /// Verify a TLS certificate RSA signature scheme.
  ///
  /// Primitive helper only: this is not a TLS or WebPKI provider integration.
  /// The caller must construct the signed message and enforce handshake,
  /// certificate-chain, and policy state.
  ///
  /// This helper is for protocol surfaces that already mapped a certificate
  /// signature to a TLS `SignatureScheme`; raw X.509 certificate signatures
  /// should normally use [`Self::verify_signature_from_x509_algorithm_der`].
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if the scheme is unsupported, conflicts
  /// with this key's X.509 public-key algorithm, or the signature is invalid.
  #[inline]
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_tls_certificate_signature_scheme(
    &self,
    scheme: u16,
    message: &[u8],
    signature: &[u8],
  ) -> Result<(), VerificationError> {
    let profile = self
      .signature_profile_from_tls_certificate_signature_scheme(scheme)
      .map_err(|_| VerificationError::new())?;
    if !self.key.signature_profile_is_possible(profile) {
      return Err(VerificationError::new());
    }
    let mut scratch = self.key.public_scratch();
    self
      .key
      .verify_signature_with_scratch(profile, message, signature, &mut scratch)
  }

  /// Verify a TLS certificate RSA signature scheme with caller-owned scratch.
  ///
  /// Primitive helper only: this is not a TLS or WebPKI provider integration.
  /// The caller must construct the signed message and enforce handshake,
  /// certificate-chain, and policy state.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if the scheme is unsupported, conflicts
  /// with this key's X.509 public-key algorithm, or the signature is invalid.
  #[inline]
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_tls_certificate_signature_scheme_with_scratch(
    &self,
    scheme: u16,
    message: &[u8],
    signature: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    let profile = self
      .signature_profile_from_tls_certificate_signature_scheme(scheme)
      .map_err(|_| VerificationError::new())?;
    self
      .key
      .verify_signature_with_scratch(profile, message, signature, scratch)
  }
}

impl RsaPublicKey {
  /// Build an RSA public key from canonical unsigned big-endian components.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if the modulus or exponent violates the default
  /// public-key policy.
  pub fn from_modulus_exponent(modulus: &[u8], public_exponent: u64) -> Result<Self, RsaKeyError> {
    Self::from_modulus_exponent_with_policy(modulus, public_exponent, &RsaPublicKeyPolicy::default())
  }

  /// Build an RSA public key from components with an explicit policy.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if the modulus or exponent violates `policy`.
  pub fn from_modulus_exponent_with_policy(
    modulus: &[u8],
    public_exponent: u64,
    policy: &RsaPublicKeyPolicy,
  ) -> Result<Self, RsaKeyError> {
    if policy.min_modulus_bits > policy.max_modulus_bits {
      return Err(RsaKeyError::InvalidModulus);
    }

    let exponent = parse_public_exponent(&public_exponent.to_be_bytes(), policy)?;
    let modulus_bits = validate_modulus(modulus, policy)?;
    Ok(Self {
      modulus: RsaPublicModulus::new(modulus, modulus_bits),
      exponent,
    })
  }

  /// Parse an RSA `SubjectPublicKeyInfo` DER object with the default policy.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if the DER is malformed, the algorithm identifier
  /// is not `rsaEncryption` with NULL parameters, or the embedded key violates
  /// RSA public-key policy.
  pub fn from_spki_der(der: &[u8]) -> Result<Self, RsaKeyError> {
    Self::from_spki_der_with_policy(der, &RsaPublicKeyPolicy::default())
  }

  /// Parse an RSA `SubjectPublicKeyInfo` DER object with an explicit policy.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if parsing or validation fails.
  pub fn from_spki_der_with_policy(der: &[u8], policy: &RsaPublicKeyPolicy) -> Result<Self, RsaKeyError> {
    let (algorithm, public_key_der) = parse_spki_der(der)?;
    parse_rsa_algorithm_identifier(algorithm)?;
    Self::from_pkcs1_der_with_policy(public_key_der, policy)
  }

  /// Parse a PKCS #1 `RSAPublicKey` DER object with the default policy.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if the DER is malformed or the key violates RSA
  /// public-key policy.
  pub fn from_pkcs1_der(der: &[u8]) -> Result<Self, RsaKeyError> {
    Self::from_pkcs1_der_with_policy(der, &RsaPublicKeyPolicy::default())
  }

  /// Parse a PKCS #1 `RSAPublicKey` DER object with an explicit policy.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if parsing or validation fails.
  pub fn from_pkcs1_der_with_policy(der: &[u8], policy: &RsaPublicKeyPolicy) -> Result<Self, RsaKeyError> {
    let (modulus, modulus_bits, exponent) = parse_pkcs1_public_key_der_parts(der, policy)?;
    let modulus = RsaPublicModulus::new(modulus, modulus_bits);

    Ok(Self { modulus, exponent })
  }

  /// Borrow the canonical unsigned big-endian modulus bytes.
  #[inline]
  #[must_use]
  pub fn modulus(&self) -> &[u8] {
    &self.modulus.bytes
  }

  /// Return the modulus bit length.
  #[inline]
  #[must_use]
  pub const fn modulus_bits(&self) -> usize {
    self.modulus.bits
  }

  /// Return the validated public exponent.
  #[inline]
  #[must_use]
  pub const fn public_exponent(&self) -> RsaPublicExponent {
    self.exponent
  }

  /// Encode this public key as a PKCS #1 `RSAPublicKey` DER object.
  #[must_use]
  pub fn to_pkcs1_der(&self) -> Vec<u8> {
    der_sequence_from_parts(&[
      der_integer_unsigned(self.modulus()).as_slice(),
      der_integer_unsigned(&self.public_exponent().as_u64().to_be_bytes()).as_slice(),
    ])
  }

  /// Encode this public key as an X.509 `SubjectPublicKeyInfo` DER object.
  #[must_use]
  pub fn to_spki_der(&self) -> Vec<u8> {
    let pkcs1 = self.to_pkcs1_der();
    der_sequence_from_parts(&[
      der_rsa_encryption_algorithm_identifier().as_slice(),
      der_bit_string_zero_unused(&pkcs1).as_slice(),
    ])
  }

  /// Allocate scratch space for repeated RSA public operations with this key.
  #[inline]
  #[must_use]
  pub fn public_scratch(&self) -> RsaPublicScratch {
    RsaPublicScratch::new(self)
  }

  /// Apply the RSA public operation to a fixed-width representative.
  ///
  /// This is the raw `m = s^e mod n` primitive used by signature verification
  /// padding code. It does not parse or check PKCS #1 v1.5/PSS padding.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPublicOpError`] if `input` or `out` is not exactly the
  /// modulus length, or if `input >= n`.
  #[must_use = "RSA public-operation failure must be checked; a dropped Result silently accepts an invalid \
                representative"]
  pub fn public_operation(&self, input: &[u8], out: &mut [u8]) -> Result<(), RsaPublicOpError> {
    let mut scratch = self.public_scratch();
    self.public_operation_with_scratch(input, out, &mut scratch)
  }

  /// Apply the RSA public operation using caller-owned scratch space.
  ///
  /// Reusing scratch avoids allocation in steady-state verification loops.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPublicOpError`] if lengths are invalid, the representative is
  /// out of range, or `scratch` was allocated for a different modulus width.
  #[must_use = "RSA public-operation failure must be checked; a dropped Result silently accepts an invalid \
                representative"]
  pub fn public_operation_with_scratch(
    &self,
    input: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), RsaPublicOpError> {
    let result = self.modulus.public_operation(self.exponent, input, out, scratch);
    clear_output_on_error(result, out)
  }

  /// Encrypt a message using legacy RSAES-PKCS1-v1_5 and OS-backed randomness.
  ///
  /// Prefer OAEP for new protocols. `out` must be exactly
  /// [`Self::modulus`]`.len()` bytes.
  ///
  /// # Errors
  ///
  /// Returns [`RsaEncryptionError`] if entropy is unavailable, the message is
  /// too long for this key, `out` has the wrong length, or the RSA public
  /// operation fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA encryption failure must be checked; a dropped Result silently discards ciphertext"]
  pub fn encrypt_pkcs1v15(&self, message: &[u8], out: &mut [u8]) -> Result<(), RsaEncryptionError> {
    let mut scratch = self.public_scratch();
    self.encrypt_pkcs1v15_with_scratch(message, out, &mut scratch)
  }

  /// Encrypt using legacy RSAES-PKCS1-v1_5, OS-backed randomness, and scratch.
  ///
  /// Reusing scratch avoids heap allocation after setup. Prefer OAEP for new
  /// protocols.
  ///
  /// # Errors
  ///
  /// Returns [`RsaEncryptionError`] if entropy is unavailable, lengths are
  /// invalid, the message is too long, or the RSA public operation fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA encryption failure must be checked; a dropped Result silently discards ciphertext"]
  pub fn encrypt_pkcs1v15_with_scratch(
    &self,
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), RsaEncryptionError> {
    if out.len() != self.modulus().len()
      || scratch.limb_count != self.modulus.limbs.len()
      || scratch.byte_count != self.modulus().len()
    {
      return clear_output_on_error(Err(RsaEncryptionError::InvalidLength), out);
    }
    let ps_len = match pkcs1v15_encryption_padding_len(self.modulus().len(), message.len()) {
      Ok(ps_len) => ps_len,
      Err(error) => return clear_output_on_error(Err(error), out),
    };
    let (arithmetic_scratch, encoded, seed, _) = scratch.split_all();
    let result = match seed.get_mut(..ps_len) {
      Some(padding_seed) => fill_pkcs1v15_nonzero_padding(padding_seed)
        .and_then(|()| encode_pkcs1v15_encryption_with_seed(message, padding_seed, encoded))
        .and_then(|()| {
          self
            .modulus
            .public_operation_with_arithmetic_scratch(self.exponent, encoded, out, arithmetic_scratch)
            .map_err(|_| RsaEncryptionError::PublicOperationFailed)
        }),
      None => Err(RsaEncryptionError::InvalidLength),
    };
    ct::zeroize(encoded);
    ct::zeroize(seed);
    clear_output_on_error(result, out)
  }

  /// Encrypt a message using legacy RSAES-PKCS1-v1_5 and caller-supplied padding.
  ///
  /// This is primarily for deterministic tests and protocol harnesses. `seed`
  /// must contain exactly the non-zero padding-string bytes required for this
  /// key and message length.
  ///
  /// # Errors
  ///
  /// Returns [`RsaEncryptionError`] if lengths are invalid, the message is too
  /// long, any seed byte is zero, or the RSA public operation fails.
  #[must_use = "RSA encryption failure must be checked; a dropped Result silently discards ciphertext"]
  pub fn encrypt_pkcs1v15_with_seed(
    &self,
    message: &[u8],
    seed: &[u8],
    out: &mut [u8],
  ) -> Result<(), RsaEncryptionError> {
    let mut scratch = self.public_scratch();
    self.encrypt_pkcs1v15_with_seed_and_scratch(message, seed, out, &mut scratch)
  }

  /// Encrypt using legacy RSAES-PKCS1-v1_5, caller-supplied padding, and scratch.
  ///
  /// Reusing scratch avoids heap allocation after setup. `seed` must contain
  /// exactly the non-zero padding-string bytes required for this key and
  /// message length.
  ///
  /// # Errors
  ///
  /// Returns [`RsaEncryptionError`] if lengths are invalid, the message is too
  /// long, any seed byte is zero, or the RSA public operation fails.
  #[must_use = "RSA encryption failure must be checked; a dropped Result silently discards ciphertext"]
  pub fn encrypt_pkcs1v15_with_seed_and_scratch(
    &self,
    message: &[u8],
    seed: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), RsaEncryptionError> {
    if out.len() != self.modulus().len()
      || scratch.limb_count != self.modulus.limbs.len()
      || scratch.byte_count != self.modulus().len()
    {
      return clear_output_on_error(Err(RsaEncryptionError::InvalidLength), out);
    }

    let (arithmetic_scratch, encoded, _, _) = scratch.split_all();
    let result = encode_pkcs1v15_encryption_with_seed(message, seed, encoded).and_then(|()| {
      self
        .modulus
        .public_operation_with_arithmetic_scratch(self.exponent, encoded, out, arithmetic_scratch)
        .map_err(|_| RsaEncryptionError::PublicOperationFailed)
    });
    ct::zeroize(encoded);
    clear_output_on_error(result, out)
  }

  /// Encrypt a message using RSAES-OAEP and OS-backed randomness.
  ///
  /// `out` must be exactly [`Self::modulus`]`.len()` bytes. The label must
  /// match the label supplied to OAEP decryption.
  ///
  /// # Errors
  ///
  /// Returns [`RsaEncryptionError`] if entropy is unavailable, the message is
  /// too long for this key/profile, `out` has the wrong length, or the RSA
  /// public operation fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA encryption failure must be checked; a dropped Result silently discards ciphertext"]
  pub fn encrypt_oaep(
    &self,
    profile: RsaOaepProfile,
    label: &[u8],
    message: &[u8],
    out: &mut [u8],
  ) -> Result<(), RsaEncryptionError> {
    let mut scratch = self.public_scratch();
    self.encrypt_oaep_with_scratch(profile, label, message, out, &mut scratch)
  }

  /// Encrypt a message using RSAES-OAEP, OS-backed randomness, and caller-owned scratch.
  ///
  /// Reusing scratch avoids heap allocation after setup. The label must match
  /// the label supplied to OAEP decryption.
  ///
  /// # Errors
  ///
  /// Returns [`RsaEncryptionError`] if entropy is unavailable, the message is
  /// too long for this key/profile, `out` has the wrong length, `scratch` was
  /// allocated for a different modulus width, or the RSA public operation fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA encryption failure must be checked; a dropped Result silently discards ciphertext"]
  pub fn encrypt_oaep_with_scratch(
    &self,
    profile: RsaOaepProfile,
    label: &[u8],
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), RsaEncryptionError> {
    let mut seed = [0u8; Sha512::OUTPUT_SIZE];
    let seed_len = profile.digest_len();
    let Some(seed_bytes) = seed.get_mut(..seed_len) else {
      return clear_output_on_error(Err(RsaEncryptionError::InvalidLength), out);
    };
    let result = match getrandom::fill(seed_bytes) {
      Ok(()) => self.encrypt_oaep_with_seed_and_scratch(profile, label, message, seed_bytes, out, scratch),
      Err(_) => Err(RsaEncryptionError::EntropyUnavailable),
    };
    ct::zeroize(&mut seed);
    clear_output_on_error(result, out)
  }

  /// Encrypt a message using RSAES-OAEP and caller-supplied seed bytes.
  ///
  /// This is primarily for deterministic tests and protocol harnesses. Normal
  /// callers with the `getrandom` feature should use [`Self::encrypt_oaep`].
  ///
  /// # Errors
  ///
  /// Returns [`RsaEncryptionError`] if lengths are invalid, the message is too
  /// long for this key/profile, or the RSA public operation fails.
  #[must_use = "RSA encryption failure must be checked; a dropped Result silently discards ciphertext"]
  pub fn encrypt_oaep_with_seed(
    &self,
    profile: RsaOaepProfile,
    label: &[u8],
    message: &[u8],
    seed: &[u8],
    out: &mut [u8],
  ) -> Result<(), RsaEncryptionError> {
    let mut scratch = self.public_scratch();
    self.encrypt_oaep_with_seed_and_scratch(profile, label, message, seed, out, &mut scratch)
  }

  /// Encrypt a message using RSAES-OAEP, caller-supplied seed bytes, and scratch.
  ///
  /// This is primarily for deterministic tests and protocol harnesses. Reusing
  /// scratch avoids heap allocation after setup.
  ///
  /// # Errors
  ///
  /// Returns [`RsaEncryptionError`] if lengths are invalid, the message is too
  /// long for this key/profile, `scratch` was allocated for a different modulus
  /// width, or the RSA public operation fails.
  #[must_use = "RSA encryption failure must be checked; a dropped Result silently discards ciphertext"]
  pub fn encrypt_oaep_with_seed_and_scratch(
    &self,
    profile: RsaOaepProfile,
    label: &[u8],
    message: &[u8],
    seed: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), RsaEncryptionError> {
    if out.len() != self.modulus().len()
      || scratch.limb_count != self.modulus.limbs.len()
      || scratch.byte_count != self.modulus().len()
    {
      return clear_output_on_error(Err(RsaEncryptionError::InvalidLength), out);
    }

    let (arithmetic_scratch, encoded, seed_mask, db_mask) = scratch.split_all();
    let result = match profile {
      RsaOaepProfile::Sha256 => encode_oaep_with_masks::<Sha256>(label, message, seed, encoded, db_mask, seed_mask),
      RsaOaepProfile::Sha384 => encode_oaep_with_masks::<Sha384>(label, message, seed, encoded, db_mask, seed_mask),
      RsaOaepProfile::Sha512 => encode_oaep_with_masks::<Sha512>(label, message, seed, encoded, db_mask, seed_mask),
    }
    .and_then(|()| {
      self
        .modulus
        .public_operation_with_arithmetic_scratch(self.exponent, encoded, out, arithmetic_scratch)
        .map_err(|_| RsaEncryptionError::PublicOperationFailed)
    });
    ct::zeroize(encoded);
    ct::zeroize(seed_mask);
    ct::zeroize(db_mask);
    clear_output_on_error(result, out)
  }

  /// Verify an RSASSA-PSS signature under a typed SHA-2 profile.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] for every invalid signature, malformed PSS
  /// encoding, unsupported representative, or policy mismatch. The error is
  /// intentionally opaque; protocol adapters must not split it into finer
  /// externally visible failure classes.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_pss(&self, profile: RsaPssProfile, message: &[u8], signature: &[u8]) -> Result<(), VerificationError> {
    let mut scratch = self.public_scratch();
    self.verify_pss_with_scratch(profile, message, signature, &mut scratch)
  }

  /// Verify an RSASSA-PSS signature with an explicit salt length.
  ///
  /// Use this when a protocol adapter has parsed RSASSA-PSS parameters that
  /// specify a salt length. Prefer [`Self::verify_pss`] for TLS 1.3-style
  /// profiles where the salt length must equal the digest length.
  ///
  /// # Errors
  ///
  /// Returns an opaque [`VerificationError`] for any invalid signature or
  /// impossible salt-length/profile combination.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_pss_with_salt_len(
    &self,
    profile: RsaPssProfile,
    salt_len: usize,
    message: &[u8],
    signature: &[u8],
  ) -> Result<(), VerificationError> {
    if !self.pss_salt_len_is_possible(profile, salt_len) {
      return Err(VerificationError::new());
    }
    let mut scratch = self.public_scratch();
    self.verify_pss_with_salt_len_and_scratch(profile, salt_len, message, signature, &mut scratch)
  }

  /// Verify an RSASSA-PSS signature with caller-owned RSA scratch space.
  ///
  /// This avoids repeated RSA arithmetic scratch allocation in steady-state
  /// verification loops. PSS decoding still uses bounded temporary buffers
  /// sized by the modulus length.
  ///
  /// # Errors
  ///
  /// Returns an opaque [`VerificationError`] for any invalid signature.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_pss_with_scratch(
    &self,
    profile: RsaPssProfile,
    message: &[u8],
    signature: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    self.verify_pss_with_salt_len_and_scratch(profile, profile.digest_len(), message, signature, scratch)
  }

  /// Verify an RSASSA-PSS signature with explicit salt length and caller-owned scratch.
  ///
  /// # Errors
  ///
  /// Returns an opaque [`VerificationError`] for any invalid signature.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_pss_with_salt_len_and_scratch(
    &self,
    profile: RsaPssProfile,
    salt_len: usize,
    message: &[u8],
    signature: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    if !self.pss_salt_len_is_possible(profile, salt_len) {
      return Err(VerificationError::new());
    }
    let (arithmetic_scratch, encoded, db, db_mask) = scratch.split_all();
    self
      .modulus
      .public_operation_with_arithmetic_scratch(self.exponent, signature, encoded, arithmetic_scratch)
      .map_err(|_| VerificationError::new())?;

    let em_bits = self.modulus_bits().strict_sub(1);
    let em_len = em_bits.strict_add(7) / 8;
    let leading = encoded.len().strict_sub(em_len);
    let Some(prefix) = encoded.get(..leading) else {
      return Err(VerificationError::new());
    };
    if prefix.iter().any(|&byte| byte != 0) {
      return Err(VerificationError::new());
    }
    let Some(encoded) = encoded.get(leading..) else {
      return Err(VerificationError::new());
    };

    match profile {
      RsaPssProfile::Sha256 => {
        verify_pss_encoded_with_scratch::<Sha256>(message, encoded, em_bits, salt_len, db, db_mask)
      }
      RsaPssProfile::Sha384 => {
        verify_pss_encoded_with_scratch::<Sha384>(message, encoded, em_bits, salt_len, db, db_mask)
      }
      RsaPssProfile::Sha512 => {
        verify_pss_encoded_with_scratch::<Sha512>(message, encoded, em_bits, salt_len, db, db_mask)
      }
    }
  }

  /// Verify an RSA signature under a typed signature profile.
  ///
  /// This is the preferred primitive entry point for protocol adapters after
  /// they map external algorithm identifiers and parameters to
  /// [`RsaSignatureProfile`]. Algorithm mismatches and malformed signatures all
  /// return the same opaque [`VerificationError`].
  ///
  /// # Errors
  ///
  /// Returns an opaque [`VerificationError`] for any invalid signature,
  /// malformed encoded message, unsupported representative, or profile
  /// mismatch.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_signature(
    &self,
    profile: RsaSignatureProfile,
    message: &[u8],
    signature: &[u8],
  ) -> Result<(), VerificationError> {
    if !self.signature_profile_is_possible(profile) {
      return Err(VerificationError::new());
    }
    let mut scratch = self.public_scratch();
    self.verify_signature_with_scratch(profile, message, signature, &mut scratch)
  }

  /// Verify an RSA signature under a typed profile with caller-owned scratch.
  ///
  /// # Errors
  ///
  /// Returns an opaque [`VerificationError`] for any invalid signature.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_signature_with_scratch(
    &self,
    profile: RsaSignatureProfile,
    message: &[u8],
    signature: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    if !self.signature_profile_is_possible(profile) {
      return Err(VerificationError::new());
    }
    match profile {
      RsaSignatureProfile::Pss { profile, salt_len } => {
        self.verify_pss_with_salt_len_and_scratch(profile, salt_len, message, signature, scratch)
      }
      RsaSignatureProfile::Pkcs1v15(profile) => self.verify_pkcs1v15_with_scratch(profile, message, signature, scratch),
    }
  }

  /// Verify an RSA JWT/JWS signature using an already-parsed JOSE `alg`.
  ///
  /// Primitive helper only: this is not a JWT, JWS, JOSE, or JSON provider
  /// integration.
  ///
  /// `message` is the caller-constructed JWS Signing Input. This helper does
  /// not parse JWT claims, JWS compact serialization, or JSON; it only enforces
  /// the explicit RSA SHA-2 `alg` mapping before signature verification.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if `alg` is not an accepted RSA SHA-2 JOSE
  /// algorithm, or if the signature is invalid.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_jwt_alg(&self, alg: &str, message: &[u8], signature: &[u8]) -> Result<(), VerificationError> {
    let profile = RsaSignatureProfile::from_jwt_alg(alg).map_err(|_| VerificationError::new())?;
    if !self.signature_profile_is_possible(profile) {
      return Err(VerificationError::new());
    }
    let mut scratch = self.public_scratch();
    self.verify_signature_with_scratch(profile, message, signature, &mut scratch)
  }

  /// Verify an RSA JWT/JWS signature using caller-owned RSA scratch space.
  ///
  /// Primitive helper only: this is not a JWT, JWS, JOSE, or JSON provider
  /// integration.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if `alg` is unsupported or verification
  /// fails.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_jwt_alg_with_scratch(
    &self,
    alg: &str,
    message: &[u8],
    signature: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    let profile = RsaSignatureProfile::from_jwt_alg(alg).map_err(|_| VerificationError::new())?;
    self.verify_signature_with_scratch(profile, message, signature, scratch)
  }

  /// Verify an RSA COSE signature using an already-parsed COSE algorithm ID.
  ///
  /// Primitive helper only: this is not a COSE, CBOR, or CWT provider
  /// integration.
  ///
  /// `message` is the caller-constructed COSE Sig_structure bytes. This helper
  /// does not parse CBOR or COSE objects; it only enforces the explicit RSA
  /// SHA-2 algorithm mapping before signature verification.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if `algorithm` is not an accepted RSA SHA-2
  /// COSE algorithm, or if the signature is invalid.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_cose_algorithm_id(
    &self,
    algorithm: i64,
    message: &[u8],
    signature: &[u8],
  ) -> Result<(), VerificationError> {
    let profile = RsaSignatureProfile::from_cose_algorithm_id(algorithm).map_err(|_| VerificationError::new())?;
    if !self.signature_profile_is_possible(profile) {
      return Err(VerificationError::new());
    }
    let mut scratch = self.public_scratch();
    self.verify_signature_with_scratch(profile, message, signature, &mut scratch)
  }

  /// Verify an RSA COSE signature using caller-owned RSA scratch space.
  ///
  /// Primitive helper only: this is not a COSE, CBOR, or CWT provider
  /// integration.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if `algorithm` is unsupported or
  /// verification fails.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_cose_algorithm_id_with_scratch(
    &self,
    algorithm: i64,
    message: &[u8],
    signature: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    let profile = RsaSignatureProfile::from_cose_algorithm_id(algorithm).map_err(|_| VerificationError::new())?;
    self.verify_signature_with_scratch(profile, message, signature, scratch)
  }

  /// Verify an RSASSA-PKCS1-v1_5 signature under a typed SHA-2 profile.
  ///
  /// # Errors
  ///
  /// Returns an opaque [`VerificationError`] for every invalid signature or
  /// malformed encoded message. The `DigestInfo` must be exact DER for the
  /// selected SHA-2 profile; BER leniency, trailing data, short padding, and
  /// OID/hash mismatches are rejected.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_pkcs1v15(
    &self,
    profile: RsaPkcs1v15Profile,
    message: &[u8],
    signature: &[u8],
  ) -> Result<(), VerificationError> {
    let mut scratch = self.public_scratch();
    self.verify_pkcs1v15_with_scratch(profile, message, signature, &mut scratch)
  }

  /// Verify an RSASSA-PKCS1-v1_5 signature with caller-owned RSA scratch space.
  ///
  /// # Errors
  ///
  /// Returns an opaque [`VerificationError`] for every invalid signature.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_pkcs1v15_with_scratch(
    &self,
    profile: RsaPkcs1v15Profile,
    message: &[u8],
    signature: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    let (arithmetic_scratch, encoded, _, _) = scratch.split_all();
    self
      .modulus
      .public_operation_with_arithmetic_scratch(self.exponent, signature, encoded, arithmetic_scratch)
      .map_err(|_| VerificationError::new())?;

    match profile {
      RsaPkcs1v15Profile::Sha256 => verify_pkcs1v15_encoded::<Sha256>(message, encoded, SHA256_DIGEST_INFO_PREFIX),
      RsaPkcs1v15Profile::Sha384 => verify_pkcs1v15_encoded::<Sha384>(message, encoded, SHA384_DIGEST_INFO_PREFIX),
      RsaPkcs1v15Profile::Sha512 => verify_pkcs1v15_encoded::<Sha512>(message, encoded, SHA512_DIGEST_INFO_PREFIX),
    }
  }

  /// Return `true` when this key is large enough for a signature profile.
  ///
  /// This is a cheap profile feasibility check for protocol adapters before
  /// allocating scratch or attempting an RSA public operation. It does not check
  /// X.509 key-algorithm constraints, parse external objects, or validate a
  /// signature.
  #[inline]
  #[must_use]
  pub fn signature_profile_is_possible(&self, profile: RsaSignatureProfile) -> bool {
    match profile {
      RsaSignatureProfile::Pss { profile, salt_len } => self.pss_salt_len_is_possible(profile, salt_len),
      RsaSignatureProfile::Pkcs1v15(_) => true,
    }
  }

  /// Return `true` when this key is large enough for the PSS hash and salt length.
  #[inline]
  #[must_use]
  pub fn pss_salt_len_is_possible(&self, profile: RsaPssProfile, salt_len: usize) -> bool {
    let em_bits = self.modulus_bits().strict_sub(1);
    let em_len = em_bits.strict_add(7) / 8;
    profile
      .digest_len()
      .checked_add(salt_len)
      .and_then(|len| len.checked_add(2))
      .is_some_and(|minimum_len| em_len >= minimum_len)
  }
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
  r2: Box<[u64]>,
  limbs: Box<[u64]>,
  bytes: Box<[u8]>,
  limb_count: usize,
  byte_count: usize,
}

struct RsaPublicArithmeticScratch<'a> {
  r2: &'a [u64],
  t: &'a mut [u64],
  x: &'a mut [u64],
  tmp: &'a mut [u64],
  base: &'a mut [u64],
  acc: &'a mut [u64],
}

impl RsaPublicScratch {
  /// Allocate scratch space for `key`.
  #[must_use]
  pub fn new(key: &RsaPublicKey) -> Self {
    let limbs = key.modulus.limbs.len();
    let bytes = key.modulus.bytes.len();
    Self {
      r2: public_montgomery_r2(&key.modulus.limbs),
      limbs: vec![0u64; limbs.strict_mul(6).strict_add(2)].into_boxed_slice(),
      bytes: vec![0u8; bytes.strict_mul(3)].into_boxed_slice(),
      limb_count: limbs,
      byte_count: bytes,
    }
  }

  fn arithmetic_scratch(&mut self) -> RsaPublicArithmeticScratch<'_> {
    split_limb_scratch(&mut self.limbs, self.limb_count, &self.r2)
  }

  fn split_all(&mut self) -> (RsaPublicArithmeticScratch<'_>, &mut [u8], &mut [u8], &mut [u8]) {
    let arithmetic_scratch = split_limb_scratch(&mut self.limbs, self.limb_count, &self.r2);
    let (encoded, db, db_mask) = split_byte_scratch(&mut self.bytes, self.byte_count);
    (arithmetic_scratch, encoded, db, db_mask)
  }
}

impl fmt::Debug for RsaPublicScratch {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("RsaPublicScratch")
      .field("limbs", &self.limb_count)
      .finish_non_exhaustive()
  }
}

fn split_limb_scratch<'a>(limbs: &'a mut [u64], limb_count: usize, r2: &'a [u64]) -> RsaPublicArithmeticScratch<'a> {
  let t_len = limb_count.strict_mul(2).strict_add(2);
  let (t, rest) = limbs.split_at_mut(t_len);
  let (x, rest) = rest.split_at_mut(limb_count);
  let (tmp, rest) = rest.split_at_mut(limb_count);
  let (base, rest) = rest.split_at_mut(limb_count);
  let (acc, _) = rest.split_at_mut(limb_count);
  RsaPublicArithmeticScratch {
    r2,
    t,
    x,
    tmp,
    base,
    acc,
  }
}

fn split_byte_scratch(bytes: &mut [u8], byte_count: usize) -> (&mut [u8], &mut [u8], &mut [u8]) {
  let (encoded, rest) = bytes.split_at_mut(byte_count);
  let (db, db_mask) = rest.split_at_mut(byte_count);
  (encoded, db, db_mask)
}

#[derive(Clone)]
struct RsaPublicModulus {
  bytes: Box<[u8]>,
  limbs: Box<[u64]>,
  r2: Option<Box<[u64]>>,
  bits: usize,
  n0: u64,
}

impl Drop for RsaPublicModulus {
  fn drop(&mut self) {
    ct::zeroize(&mut self.bytes);
    ct::zeroize_words(&mut self.limbs);
    if let Some(r2) = &mut self.r2 {
      ct::zeroize_words(r2);
    }
    ct::zeroize_words(core::slice::from_mut(&mut self.bits));
    ct::zeroize_words(core::slice::from_mut(&mut self.n0));
  }
}

impl PartialEq for RsaPublicModulus {
  fn eq(&self, other: &Self) -> bool {
    self.bits == other.bits && self.bytes == other.bytes
  }
}

impl Eq for RsaPublicModulus {}

impl Hash for RsaPublicModulus {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.bytes.hash(state);
    self.bits.hash(state);
  }
}

impl RsaPublicModulus {
  fn new(bytes: &[u8], bits: usize) -> Self {
    Self::new_inner(bytes, bits, false)
  }

  fn new_with_montgomery_r2(bytes: &[u8], bits: usize) -> Self {
    Self::new_inner(bytes, bits, true)
  }

  fn new_inner(bytes: &[u8], bits: usize, precompute_r2: bool) -> Self {
    let limbs = limbs_from_be(bytes);
    let n0 = montgomery_n0(limbs.first().copied().unwrap_or(1));
    let r2 = precompute_r2.then(|| public_montgomery_r2(&limbs));
    Self {
      bytes: Box::from(bytes),
      limbs: limbs.into_boxed_slice(),
      r2,
      bits,
      n0,
    }
  }

  fn montgomery_r2(&self) -> &[u64] {
    match self.r2.as_deref() {
      Some(r2) => r2,
      None => unreachable!("private RSA modulus missing Montgomery R^2"),
    }
  }

  fn public_operation(
    &self,
    exponent: RsaPublicExponent,
    input: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), RsaPublicOpError> {
    self.public_operation_with_arithmetic_scratch(exponent, input, out, scratch.arithmetic_scratch())
  }

  fn public_operation_with_arithmetic_scratch(
    &self,
    exponent: RsaPublicExponent,
    input: &[u8],
    out: &mut [u8],
    scratch: RsaPublicArithmeticScratch<'_>,
  ) -> Result<(), RsaPublicOpError> {
    let RsaPublicArithmeticScratch {
      r2,
      t,
      x,
      tmp,
      base,
      acc,
    } = scratch;
    let limbs = self.limbs.len();
    if input.len() != self.bytes.len() || out.len() != self.bytes.len() {
      return Err(RsaPublicOpError::InvalidLength);
    }
    if x.len() != limbs
      || base.len() != limbs
      || acc.len() != limbs
      || tmp.len() != limbs
      || t.len() != limbs.strict_mul(2).strict_add(2)
    {
      return Err(RsaPublicOpError::InvalidScratch);
    }

    limbs_from_be_into(input, x);
    if cmp_limbs(x, &self.limbs) != core::cmp::Ordering::Less {
      return Err(RsaPublicOpError::RepresentativeOutOfRange);
    }

    if use_cios_montgomery(self) {
      #[cfg(all(
        target_arch = "aarch64",
        target_os = "macos",
        not(feature = "portable-only"),
        not(miri)
      ))]
      if exponent.as_u64() == 65_537
        && rsa_aarch64_asm::supports_bignum_mont_words(limbs)
        && t.len() >= rsa_aarch64_asm::bignum_mont_scratch_words(limbs)
        && limbs == 32
      {
        rsa_aarch64_asm::public_e65537_mont_words(base, x, r2, acc, &self.limbs, self.n0, limbs, t);
        limbs_to_be(base, out);
        return Ok(());
      }

      mont_mul_cios(base, x, r2, self, t);
      copy_limbs(acc, base);

      match exponent.as_u64() {
        3 => {
          mont_square_cios_in_place(acc, tmp, self, t);
          mont_mul_cios_in_place_left(acc, base, tmp, self, t);
        }
        17 => {
          for _ in 0..4 {
            mont_square_cios_in_place(acc, tmp, self, t);
          }
          mont_mul_cios_in_place_left(acc, base, tmp, self, t);
        }
        65_537 => {
          for _ in 0..16 {
            mont_square_cios_in_place(acc, tmp, self, t);
          }
          mont_mul_cios_in_place_left(acc, base, tmp, self, t);
        }
        exponent => {
          let top_bit = 63usize.strict_sub(exponent.leading_zeros() as usize);
          for bit in (0..top_bit).rev() {
            mont_square_cios_in_place(acc, tmp, self, t);
            if (exponent >> bit) & 1 == 1 {
              mont_mul_cios_in_place_left(acc, base, tmp, self, t);
            }
          }
        }
      }

      mont_reduce_cios(base, acc, self, t);
    } else {
      mont_mul(base, x, r2, self, t);
      copy_limbs(acc, base);

      match exponent.as_u64() {
        3 => {
          mont_square_in_place(acc, tmp, self, t);
          mont_mul_in_place_left(acc, base, tmp, self, t);
        }
        17 => {
          for _ in 0..4 {
            mont_square_in_place(acc, tmp, self, t);
          }
          mont_mul_in_place_left(acc, base, tmp, self, t);
        }
        65_537 => {
          for _ in 0..16 {
            mont_square_in_place(acc, tmp, self, t);
          }
          mont_mul_in_place_left(acc, base, tmp, self, t);
        }
        exponent => {
          let top_bit = 63usize.strict_sub(exponent.leading_zeros() as usize);
          for bit in (0..top_bit).rev() {
            mont_square_in_place(acc, tmp, self, t);
            if (exponent >> bit) & 1 == 1 {
              mont_mul_in_place_left(acc, base, tmp, self, t);
            }
          }
        }
      }

      mont_reduce(base, acc, self, t);
    }

    limbs_to_be(base, out);
    Ok(())
  }

  #[cfg(feature = "diag")]
  fn public_operation_bitserial(
    &self,
    exponent: RsaPublicExponent,
    input: &[u8],
    out: &mut [u8],
  ) -> Result<(), RsaPublicOpError> {
    let limbs = self.limbs.len();
    if input.len() != self.bytes.len() || out.len() != self.bytes.len() {
      return Err(RsaPublicOpError::InvalidLength);
    }

    let mut base = vec![0u64; limbs];
    limbs_from_be_into(input, &mut base);
    if cmp_limbs(&base, &self.limbs) != core::cmp::Ordering::Less {
      return Err(RsaPublicOpError::RepresentativeOutOfRange);
    }

    let mut acc = base.clone();
    let mut tmp = vec![0u64; limbs];
    let mut addend = vec![0u64; limbs];
    let top_bit = 63usize.strict_sub(exponent.as_u64().leading_zeros() as usize);
    for bit in (0..top_bit).rev() {
      mul_mod_bitserial(&mut tmp, &acc, &acc, &self.limbs, &mut addend);
      copy_limbs(&mut acc, &tmp);
      if (exponent.as_u64() >> bit) & 1 == 1 {
        mul_mod_bitserial(&mut tmp, &acc, &base, &self.limbs, &mut addend);
        copy_limbs(&mut acc, &tmp);
      }
    }

    limbs_to_be(&acc, out);
    Ok(())
  }

  #[cfg(feature = "diag")]
  fn public_operation_generic_exponent(
    &self,
    exponent: RsaPublicExponent,
    input: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), RsaPublicOpError> {
    let RsaPublicArithmeticScratch {
      r2,
      t,
      x,
      tmp,
      base,
      acc,
    } = scratch.arithmetic_scratch();
    let limbs = self.limbs.len();
    if input.len() != self.bytes.len() || out.len() != self.bytes.len() {
      return Err(RsaPublicOpError::InvalidLength);
    }
    if x.len() != limbs
      || base.len() != limbs
      || acc.len() != limbs
      || tmp.len() != limbs
      || t.len() != limbs.strict_mul(2).strict_add(2)
    {
      return Err(RsaPublicOpError::InvalidScratch);
    }

    limbs_from_be_into(input, x);
    if cmp_limbs(x, &self.limbs) != core::cmp::Ordering::Less {
      return Err(RsaPublicOpError::RepresentativeOutOfRange);
    }

    mont_mul_auto(base, x, r2, self, t);
    copy_limbs(acc, base);

    let exponent = exponent.as_u64();
    let top_bit = 63usize.strict_sub(exponent.leading_zeros() as usize);
    for bit in (0..top_bit).rev() {
      mont_square_auto_in_place(acc, tmp, self, t);
      if (exponent >> bit) & 1 == 1 {
        mont_mul_auto_in_place_left(acc, base, tmp, self, t);
      }
    }

    mont_reduce_auto(base, acc, self, t);
    limbs_to_be(base, out);
    Ok(())
  }

  #[cfg(feature = "diag")]
  fn public_operation_window2_exponent(
    &self,
    exponent: RsaPublicExponent,
    input: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), RsaPublicOpError> {
    let RsaPublicArithmeticScratch {
      r2,
      t,
      x,
      tmp,
      base,
      acc,
    } = scratch.arithmetic_scratch();
    let limbs = self.limbs.len();
    if input.len() != self.bytes.len() || out.len() != self.bytes.len() {
      return Err(RsaPublicOpError::InvalidLength);
    }
    if x.len() != limbs
      || base.len() != limbs
      || acc.len() != limbs
      || tmp.len() != limbs
      || t.len() != limbs.strict_mul(2).strict_add(2)
    {
      return Err(RsaPublicOpError::InvalidScratch);
    }

    limbs_from_be_into(input, x);
    if cmp_limbs(x, &self.limbs) != core::cmp::Ordering::Less {
      return Err(RsaPublicOpError::RepresentativeOutOfRange);
    }

    mont_mul_auto(base, x, r2, self, t);
    mont_mul_auto(tmp, base, base, self, t);
    mont_mul_auto(x, tmp, base, self, t);
    copy_limbs(acc, base);

    let exponent = exponent.as_u64();
    let top_bit = 63usize.strict_sub(exponent.leading_zeros() as usize);
    let mut bit = top_bit;
    while bit > 0 {
      let next_bit = bit.strict_sub(1);
      if ((exponent >> next_bit) & 1) == 0 {
        mont_square_auto_in_place(acc, tmp, self, t);
        bit = next_bit;
      } else if next_bit > 0 && ((exponent >> next_bit.strict_sub(1)) & 1) == 1 {
        mont_square_auto_in_place(acc, tmp, self, t);
        mont_square_auto_in_place(acc, tmp, self, t);
        mont_mul_auto_in_place_left(acc, x, tmp, self, t);
        bit = next_bit.strict_sub(1);
      } else {
        mont_square_auto_in_place(acc, tmp, self, t);
        mont_mul_auto_in_place_left(acc, base, tmp, self, t);
        bit = next_bit;
      }
    }

    mont_reduce_auto(base, acc, self, t);
    limbs_to_be(base, out);
    Ok(())
  }

  #[cfg(feature = "diag")]
  fn public_operation_product(
    &self,
    exponent: RsaPublicExponent,
    input: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), RsaPublicOpError> {
    let RsaPublicArithmeticScratch {
      r2,
      t,
      x,
      tmp,
      base,
      acc,
    } = scratch.arithmetic_scratch();
    let limbs = self.limbs.len();
    if input.len() != self.bytes.len() || out.len() != self.bytes.len() {
      return Err(RsaPublicOpError::InvalidLength);
    }
    if x.len() != limbs
      || base.len() != limbs
      || acc.len() != limbs
      || tmp.len() != limbs
      || t.len() != limbs.strict_mul(2).strict_add(2)
    {
      return Err(RsaPublicOpError::InvalidScratch);
    }

    limbs_from_be_into(input, x);
    if cmp_limbs(x, &self.limbs) != core::cmp::Ordering::Less {
      return Err(RsaPublicOpError::RepresentativeOutOfRange);
    }

    mont_mul(base, x, r2, self, t);
    copy_limbs(acc, base);

    match exponent.as_u64() {
      3 => {
        mont_square_in_place(acc, tmp, self, t);
        mont_mul_in_place_left(acc, base, tmp, self, t);
      }
      17 => {
        for _ in 0..4 {
          mont_square_in_place(acc, tmp, self, t);
        }
        mont_mul_in_place_left(acc, base, tmp, self, t);
      }
      65_537 => {
        for _ in 0..16 {
          mont_square_in_place(acc, tmp, self, t);
        }
        mont_mul_in_place_left(acc, base, tmp, self, t);
      }
      exponent => {
        let top_bit = 63usize.strict_sub(exponent.leading_zeros() as usize);
        for bit in (0..top_bit).rev() {
          mont_square_in_place(acc, tmp, self, t);
          if (exponent >> bit) & 1 == 1 {
            mont_mul_in_place_left(acc, base, tmp, self, t);
          }
        }
      }
    }

    mont_reduce(base, acc, self, t);
    limbs_to_be(base, out);
    Ok(())
  }

  #[cfg(feature = "diag")]
  fn public_operation_comba_product(
    &self,
    exponent: RsaPublicExponent,
    input: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), RsaPublicOpError> {
    let RsaPublicArithmeticScratch {
      r2,
      t,
      x,
      tmp,
      base,
      acc,
    } = scratch.arithmetic_scratch();
    let limbs = self.limbs.len();
    if input.len() != self.bytes.len() || out.len() != self.bytes.len() {
      return Err(RsaPublicOpError::InvalidLength);
    }
    if x.len() != limbs
      || base.len() != limbs
      || acc.len() != limbs
      || tmp.len() != limbs
      || t.len() != limbs.strict_mul(2).strict_add(2)
    {
      return Err(RsaPublicOpError::InvalidScratch);
    }

    limbs_from_be_into(input, x);
    if cmp_limbs(x, &self.limbs) != core::cmp::Ordering::Less {
      return Err(RsaPublicOpError::RepresentativeOutOfRange);
    }

    mont_mul_comba(base, x, r2, self, t);
    copy_limbs(acc, base);

    match exponent.as_u64() {
      3 => {
        mont_square_comba_in_place(acc, tmp, self, t);
        mont_mul_comba_in_place_left(acc, base, tmp, self, t);
      }
      17 => {
        for _ in 0..4 {
          mont_square_comba_in_place(acc, tmp, self, t);
        }
        mont_mul_comba_in_place_left(acc, base, tmp, self, t);
      }
      65_537 => {
        for _ in 0..16 {
          mont_square_comba_in_place(acc, tmp, self, t);
        }
        mont_mul_comba_in_place_left(acc, base, tmp, self, t);
      }
      exponent => {
        let top_bit = 63usize.strict_sub(exponent.leading_zeros() as usize);
        for bit in (0..top_bit).rev() {
          mont_square_comba_in_place(acc, tmp, self, t);
          if (exponent >> bit) & 1 == 1 {
            mont_mul_comba_in_place_left(acc, base, tmp, self, t);
          }
        }
      }
    }

    mont_reduce(base, acc, self, t);
    limbs_to_be(base, out);
    Ok(())
  }

  #[cfg(feature = "diag")]
  fn public_operation_cios(
    &self,
    exponent: RsaPublicExponent,
    input: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), RsaPublicOpError> {
    let RsaPublicArithmeticScratch {
      r2,
      t,
      x,
      tmp,
      base,
      acc,
    } = scratch.arithmetic_scratch();
    let limbs = self.limbs.len();
    if input.len() != self.bytes.len() || out.len() != self.bytes.len() {
      return Err(RsaPublicOpError::InvalidLength);
    }
    if x.len() != limbs
      || base.len() != limbs
      || acc.len() != limbs
      || tmp.len() != limbs
      || t.len() != limbs.strict_mul(2).strict_add(2)
    {
      return Err(RsaPublicOpError::InvalidScratch);
    }

    limbs_from_be_into(input, x);
    if cmp_limbs(x, &self.limbs) != core::cmp::Ordering::Less {
      return Err(RsaPublicOpError::RepresentativeOutOfRange);
    }

    mont_mul_cios(base, x, r2, self, t);
    copy_limbs(acc, base);

    match exponent.as_u64() {
      3 => {
        mont_square_cios_in_place(acc, tmp, self, t);
        mont_mul_cios_in_place_left(acc, base, tmp, self, t);
      }
      17 => {
        for _ in 0..4 {
          mont_square_cios_in_place(acc, tmp, self, t);
        }
        mont_mul_cios_in_place_left(acc, base, tmp, self, t);
      }
      65_537 => {
        for _ in 0..16 {
          mont_square_cios_in_place(acc, tmp, self, t);
        }
        mont_mul_cios_in_place_left(acc, base, tmp, self, t);
      }
      exponent => {
        let top_bit = 63usize.strict_sub(exponent.leading_zeros() as usize);
        for bit in (0..top_bit).rev() {
          mont_square_cios_in_place(acc, tmp, self, t);
          if (exponent >> bit) & 1 == 1 {
            mont_mul_cios_in_place_left(acc, base, tmp, self, t);
          }
        }
      }
    }

    mont_reduce_cios(base, acc, self, t);
    limbs_to_be(base, out);
    Ok(())
  }
}

fn der_rsa_encryption_algorithm_identifier() -> Vec<u8> {
  der_sequence_from_parts(&[
    der_tlv(TAG_OBJECT_IDENTIFIER, RSA_ENCRYPTION_OID).as_slice(),
    der_tlv(TAG_NULL, &[]).as_slice(),
  ])
}

fn der_sequence_from_parts(parts: &[&[u8]]) -> Vec<u8> {
  let len = parts.iter().fold(0usize, |acc, part| acc.strict_add(part.len()));
  let mut body = Vec::with_capacity(len);
  for part in parts {
    body.extend_from_slice(part);
  }
  der_tlv(TAG_SEQUENCE, &body)
}

fn der_integer_unsigned(value: &[u8]) -> Vec<u8> {
  let first_nonzero = value.iter().position(|&byte| byte != 0);
  let value = first_nonzero.and_then(|index| value.get(index..)).unwrap_or(&[0]);
  let needs_sign_padding = value.first().is_some_and(|byte| byte & 0x80 != 0);
  let mut encoded = Vec::with_capacity(value.len().strict_add(usize::from(needs_sign_padding)));
  if needs_sign_padding {
    encoded.push(0);
  }
  encoded.extend_from_slice(value);
  der_tlv(TAG_INTEGER, &encoded)
}

fn der_bit_string_zero_unused(value: &[u8]) -> Vec<u8> {
  let mut body = Vec::with_capacity(value.len().strict_add(1));
  body.push(0);
  body.extend_from_slice(value);
  der_tlv(TAG_BIT_STRING, &body)
}

fn der_tlv(tag: u8, value: &[u8]) -> Vec<u8> {
  let mut out = Vec::with_capacity(1usize.strict_add(der_len_len(value.len())).strict_add(value.len()));
  out.push(tag);
  der_push_len(value.len(), &mut out);
  out.extend_from_slice(value);
  out
}

fn der_len_len(len: usize) -> usize {
  if len < 128 {
    return 1;
  }
  let significant = core::mem::size_of::<usize>().strict_sub((len.leading_zeros() as usize) / 8);
  1usize.strict_add(significant)
}

#[allow(clippy::cast_possible_truncation)]
fn der_push_len(len: usize, out: &mut Vec<u8>) {
  if len < 128 {
    out.push(len as u8);
    return;
  }

  let mut started = false;
  let len_len = der_len_len(len).strict_sub(1);
  out.push(0x80 | (len_len as u8));
  for index in (0..core::mem::size_of::<usize>()).rev() {
    let byte = (len >> index.strict_mul(8)) as u8;
    if byte != 0 || started {
      out.push(byte);
      started = true;
    }
  }
}

fn parse_rsa_algorithm_identifier(der: &[u8]) -> Result<(), RsaKeyError> {
  let mut reader = DerReader::new(der);
  let oid = reader.read_primitive(TAG_OBJECT_IDENTIFIER)?;
  if oid != RSA_ENCRYPTION_OID {
    return Err(RsaKeyError::UnsupportedAlgorithm);
  }

  let null = reader.read_primitive(TAG_NULL)?;
  if !null.is_empty() {
    return Err(RsaKeyError::MalformedDer);
  }
  reader.finish()
}

#[allow(dead_code)]
fn parse_pkcs8_private_key_der_with_policy(
  der: &[u8],
  policy: &RsaPublicKeyPolicy,
) -> Result<RsaPrivateKeyComponents, RsaKeyError> {
  let components = parse_pkcs8_private_key_der_parts_with_policy(der, policy)?;
  private_key_components_from_parts(&components, policy)
}

fn parse_pkcs8_private_key_der_parts_with_policy<'a>(
  der: &'a [u8],
  policy: &RsaPublicKeyPolicy,
) -> Result<RsaPrivateKeyDerComponents<'a>, RsaKeyError> {
  let mut root = DerReader::new(der);
  let private_key_info = root.read_constructed(TAG_SEQUENCE)?;
  root.finish()?;

  let mut private_key_info = DerReader::new(private_key_info);
  read_zero_version(private_key_info.read_primitive(TAG_INTEGER)?)?;
  parse_rsa_algorithm_identifier(private_key_info.read_constructed(TAG_SEQUENCE)?)?;
  let private_key = private_key_info.read_primitive(TAG_OCTET_STRING)?;
  private_key_info.finish()?;

  parse_pkcs1_private_key_der_parts_with_policy(private_key, policy)
}

#[allow(dead_code)]
fn parse_pkcs1_private_key_der_with_policy(
  der: &[u8],
  policy: &RsaPublicKeyPolicy,
) -> Result<RsaPrivateKeyComponents, RsaKeyError> {
  let components = parse_pkcs1_private_key_der_parts_with_policy(der, policy)?;
  private_key_components_from_parts(&components, policy)
}

fn parse_pkcs1_private_key_der_parts_with_policy<'a>(
  der: &'a [u8],
  policy: &RsaPublicKeyPolicy,
) -> Result<RsaPrivateKeyDerComponents<'a>, RsaKeyError> {
  if policy.min_modulus_bits > policy.max_modulus_bits {
    return Err(RsaKeyError::InvalidModulus);
  }

  let mut root = DerReader::new(der);
  let private_key = root.read_constructed(TAG_SEQUENCE)?;
  root.finish()?;

  let mut private_key = DerReader::new(private_key);
  read_zero_version(private_key.read_primitive(TAG_INTEGER)?)?;
  let components = RsaPrivateKeyDerComponents {
    modulus: read_positive_integer(private_key.read_primitive(TAG_INTEGER)?)?,
    public_exponent: parse_public_exponent(read_positive_integer(private_key.read_primitive(TAG_INTEGER)?)?, policy)?,
    private_exponent: read_positive_integer(private_key.read_primitive(TAG_INTEGER)?)?,
    prime_p: read_positive_integer(private_key.read_primitive(TAG_INTEGER)?)?,
    prime_q: read_positive_integer(private_key.read_primitive(TAG_INTEGER)?)?,
    exponent_p: read_positive_integer(private_key.read_primitive(TAG_INTEGER)?)?,
    exponent_q: read_positive_integer(private_key.read_primitive(TAG_INTEGER)?)?,
    coefficient: read_positive_integer(private_key.read_primitive(TAG_INTEGER)?)?,
  };
  private_key.finish()?;

  Ok(components)
}

#[cfg(feature = "diag")]
#[doc(hidden)]
pub fn diag_rsa_validate_pkcs8_private_key_der(der: &[u8], policy: &RsaPublicKeyPolicy) -> Result<usize, RsaKeyError> {
  let components = parse_pkcs8_private_key_der_parts_with_policy(der, policy)?;
  validate_modulus(components.modulus, policy)?;
  validate_private_key_components(&components)?;
  Ok(components.modulus.len())
}

#[cfg(feature = "diag")]
#[doc(hidden)]
pub fn diag_rsa_validate_pkcs8_private_key_der_stage(
  der: &[u8],
  policy: &RsaPublicKeyPolicy,
  stage: u8,
) -> Result<usize, RsaKeyError> {
  let components = parse_pkcs8_private_key_der_parts_with_policy(der, policy)?;
  validate_modulus(components.modulus, policy)?;
  validate_private_key_components_through_stage(&components, stage)?;
  Ok(components.modulus.len())
}

#[cfg(feature = "diag")]
#[doc(hidden)]
pub fn diag_rsa_import_pkcs8_private_key_der_stage(
  der: &[u8],
  policy: &RsaPublicKeyPolicy,
  stage: u8,
) -> Result<usize, RsaKeyError> {
  let components = parse_pkcs8_private_key_der_parts_with_policy(der, policy)?;
  let modulus_bits = validate_modulus(components.modulus, policy)?;
  validate_private_key_components(&components)?;
  if stage == 50 {
    return Ok(components.modulus.len());
  }

  let public = RsaPublicKey {
    modulus: RsaPublicModulus::new_with_montgomery_r2(components.modulus, modulus_bits),
    exponent: components.public_exponent,
  };
  let mut observed = public.modulus.bytes.len();
  if stage == 51 {
    return Ok(core::hint::black_box(observed));
  }

  let prime_p_modulus = private_component_modulus(components.prime_p).map_err(|_| RsaKeyError::InvalidModulus)?;
  observed ^= prime_p_modulus.limbs.len();
  if stage == 52 {
    return Ok(core::hint::black_box(observed));
  }

  let prime_q_modulus = private_component_modulus(components.prime_q).map_err(|_| RsaKeyError::InvalidModulus)?;
  observed ^= prime_q_modulus.limbs.len();
  if stage == 53 {
    return Ok(core::hint::black_box(observed));
  }

  let private_exponent = SecretBigEndianInteger::new(components.private_exponent)?;
  let prime_p = SecretBigEndianInteger::new(components.prime_p)?;
  let prime_q = SecretBigEndianInteger::new(components.prime_q)?;
  let exponent_p = SecretBigEndianInteger::new(components.exponent_p)?;
  let exponent_q = SecretBigEndianInteger::new(components.exponent_q)?;
  let coefficient = SecretBigEndianInteger::new(components.coefficient)?;
  observed ^= private_exponent.bytes.len();
  observed ^= prime_p.bytes.len();
  observed ^= prime_q.bytes.len();
  observed ^= exponent_p.bytes.len();
  observed ^= exponent_q.bytes.len();
  observed ^= coefficient.bytes.len();
  Ok(core::hint::black_box(observed))
}

fn private_key_components_from_parts(
  components: &RsaPrivateKeyDerComponents<'_>,
  policy: &RsaPublicKeyPolicy,
) -> Result<RsaPrivateKeyComponents, RsaKeyError> {
  if policy.min_modulus_bits > policy.max_modulus_bits {
    return Err(RsaKeyError::InvalidModulus);
  }

  let modulus_bits = validate_modulus(components.modulus, policy)?;
  validate_private_key_components(components)?;
  let public = RsaPublicKey {
    modulus: RsaPublicModulus::new_with_montgomery_r2(components.modulus, modulus_bits),
    exponent: components.public_exponent,
  };
  let prime_p_modulus = private_component_modulus(components.prime_p).map_err(|_| RsaKeyError::InvalidModulus)?;
  let prime_q_modulus = private_component_modulus(components.prime_q).map_err(|_| RsaKeyError::InvalidModulus)?;

  Ok(RsaPrivateKeyComponents {
    public,
    private_exponent: SecretBigEndianInteger::new(components.private_exponent)?,
    prime_p: SecretBigEndianInteger::new(components.prime_p)?,
    prime_q: SecretBigEndianInteger::new(components.prime_q)?,
    prime_p_modulus,
    prime_q_modulus,
    exponent_p: SecretBigEndianInteger::new(components.exponent_p)?,
    exponent_q: SecretBigEndianInteger::new(components.exponent_q)?,
    coefficient: SecretBigEndianInteger::new(components.coefficient)?,
  })
}

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
#[doc(hidden)]
pub fn fuzz_rsa_import_der(format: u8, der: &[u8]) -> bool {
  let policy = if format & 0b10 == 0 {
    RsaPublicKeyPolicy::legacy_verification()
  } else {
    RsaPublicKeyPolicy::legacy_verification().allow_legacy_small_exponents()
  };

  let parsed = if format & 0b01 == 0 {
    parse_pkcs1_private_key_der_with_policy(der, &policy)
  } else {
    parse_pkcs8_private_key_der_with_policy(der, &policy)
  };

  let Ok(key) = parsed else {
    return false;
  };

  let public_key = key.public_key();
  let mut representative = public_key.modulus().to_vec();
  for byte in representative.iter_mut().rev() {
    if *byte != 0 {
      *byte = byte.strict_sub(1);
      break;
    }
    *byte = 0xff;
  }

  let mut out = vec![0u8; public_key.modulus().len()];
  let mut scratch = public_key.public_scratch();
  if public_key
    .public_operation_with_scratch(&representative, &mut out, &mut scratch)
    .is_err()
  {
    return false;
  }
  true
}

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

fn read_zero_version(bytes: &[u8]) -> Result<(), RsaKeyError> {
  let version = read_positive_integer(bytes)?;
  if version == [0] {
    Ok(())
  } else {
    Err(RsaKeyError::UnsupportedAlgorithm)
  }
}

fn validate_private_key_components(components: &RsaPrivateKeyDerComponents<'_>) -> Result<(), RsaKeyError> {
  validate_private_key_components_through_stage(components, u8::MAX)
}

fn validate_private_key_components_through_stage(
  components: &RsaPrivateKeyDerComponents<'_>,
  stage: u8,
) -> Result<(), RsaKeyError> {
  let RsaPrivateKeyDerComponents {
    modulus,
    public_exponent,
    private_exponent,
    prime_p,
    prime_q,
    exponent_p,
    exponent_q,
    coefficient,
  } = *components;

  if !is_canonical_positive_unsigned_be(private_exponent) || !ct_unsigned_be_lt_public_shape(private_exponent, modulus)
  {
    return Err(RsaKeyError::InvalidModulus);
  }
  validate_private_prime_factor(prime_p)?;
  validate_private_prime_factor(prime_q)?;
  if stage == 0 {
    return Ok(());
  }
  if ct_slices_eq_public_shape(prime_p, prime_q) || !product_matches_unsigned_be_fixed(prime_p, prime_q, modulus) {
    return Err(RsaKeyError::InvalidModulus);
  }
  if stage == 1 {
    return Ok(());
  }
  validate_private_crt_component(exponent_p, prime_p)?;
  validate_private_crt_component(exponent_q, prime_q)?;
  validate_private_crt_component(coefficient, prime_p)?;
  if stage == 2 {
    return Ok(());
  }

  let mut p_minus_one = vec![0u8; prime_p.len()];
  let mut q_minus_one = vec![0u8; prime_q.len()];
  private_import_decrement_unsigned_be_to_fixed(prime_p, &mut p_minus_one)?;
  private_import_decrement_unsigned_be_to_fixed(prime_q, &mut q_minus_one)?;
  if stage == 30 {
    return Ok(());
  }

  let mut d_mod_p_minus_one = vec![0u8; p_minus_one.len()];
  private_import_unsigned_be_mod_to_len(private_exponent, &p_minus_one, &mut d_mod_p_minus_one)?;
  if stage == 32 {
    return Ok(());
  }
  if !ct_eq_left_padded_unsigned_be(exponent_p, &d_mod_p_minus_one) {
    return Err(RsaKeyError::InvalidModulus);
  }
  if stage == 31 {
    return Ok(());
  }
  let mut d_mod_q_minus_one = vec![0u8; q_minus_one.len()];
  private_import_unsigned_be_mod_to_len(private_exponent, &q_minus_one, &mut d_mod_q_minus_one)?;
  if !ct_eq_left_padded_unsigned_be(exponent_q, &d_mod_q_minus_one) {
    return Err(RsaKeyError::InvalidModulus);
  }
  if stage == 3 {
    return Ok(());
  }
  let public_exponent = public_exponent.as_u64().to_be_bytes();
  let mut e_times_d = vec![0u8; public_exponent.len().strict_add(private_exponent.len())];
  private_import_product_unsigned_be_to_fixed(&public_exponent, private_exponent, &mut e_times_d)?;
  if stage == 40 {
    return Ok(());
  }
  let mut e_times_d_mod_p_minus_one = vec![0u8; p_minus_one.len()];
  private_import_unsigned_be_mod_to_len(&e_times_d, &p_minus_one, &mut e_times_d_mod_p_minus_one)?;
  if !ct_eq_left_padded_unsigned_be(&[1], &e_times_d_mod_p_minus_one) {
    return Err(RsaKeyError::InvalidModulus);
  }
  if stage == 41 {
    return Ok(());
  }
  let mut e_times_d_mod_q_minus_one = vec![0u8; q_minus_one.len()];
  private_import_unsigned_be_mod_to_len(&e_times_d, &q_minus_one, &mut e_times_d_mod_q_minus_one)?;
  if !ct_eq_left_padded_unsigned_be(&[1], &e_times_d_mod_q_minus_one) {
    return Err(RsaKeyError::InvalidModulus);
  }
  if stage == 42 {
    return Ok(());
  }
  if stage == 4 {
    return Ok(());
  }

  let mut q_times_coefficient = vec![0u8; prime_q.len().strict_add(coefficient.len())];
  private_import_product_unsigned_be_to_fixed(prime_q, coefficient, &mut q_times_coefficient)?;
  let mut q_times_coefficient_mod_p = vec![0u8; prime_p.len()];
  private_import_unsigned_be_mod_to_len(&q_times_coefficient, prime_p, &mut q_times_coefficient_mod_p)?;
  if !ct_eq_left_padded_unsigned_be(&[1], &q_times_coefficient_mod_p) {
    return Err(RsaKeyError::InvalidModulus);
  }

  ct::zeroize(&mut p_minus_one);
  ct::zeroize(&mut q_minus_one);
  ct::zeroize(&mut d_mod_p_minus_one);
  ct::zeroize(&mut d_mod_q_minus_one);
  ct::zeroize(&mut e_times_d);
  ct::zeroize(&mut e_times_d_mod_p_minus_one);
  ct::zeroize(&mut e_times_d_mod_q_minus_one);
  ct::zeroize(&mut q_times_coefficient);
  ct::zeroize(&mut q_times_coefficient_mod_p);

  Ok(())
}

fn validate_private_prime_factor(bytes: &[u8]) -> Result<(), RsaKeyError> {
  let Some(&last) = bytes.last() else {
    return Err(RsaKeyError::InvalidModulus);
  };
  if !is_canonical_positive_unsigned_be(bytes) || bytes == [1] || last & 1 == 0 {
    return Err(RsaKeyError::InvalidModulus);
  }
  Ok(())
}

fn validate_private_crt_component(component: &[u8], upper_bound: &[u8]) -> Result<(), RsaKeyError> {
  if !is_canonical_positive_unsigned_be(component) || !ct_unsigned_be_lt_public_shape(component, upper_bound) {
    return Err(RsaKeyError::InvalidModulus);
  }
  Ok(())
}

#[cfg(feature = "getrandom")]
struct RsaKeygenDrbg {
  key: [u8; RSA_KEYGEN_DRBG_KEY_BYTES],
  value: [u8; RSA_KEYGEN_DRBG_OUT_BYTES],
}

#[cfg(feature = "getrandom")]
impl RsaKeygenDrbg {
  fn from_os_entropy() -> Result<Self, RsaKeyGenerationError> {
    let mut seed = [0u8; RSA_KEYGEN_DRBG_ENTROPY_BYTES + RSA_KEYGEN_DRBG_NONCE_BYTES];
    if getrandom::fill(&mut seed).is_err() {
      ct::zeroize(&mut seed);
      return Err(RsaKeyGenerationError::EntropyUnavailable);
    }

    let drbg = Self::new(&seed, RSA_KEYGEN_DRBG_PERSONALIZATION);
    ct::zeroize(&mut seed);
    Ok(drbg)
  }

  fn new(entropy_and_nonce: &[u8], personalization: &[u8]) -> Self {
    let mut drbg = Self {
      key: [0u8; RSA_KEYGEN_DRBG_KEY_BYTES],
      value: [1u8; RSA_KEYGEN_DRBG_OUT_BYTES],
    };
    drbg.update(&[entropy_and_nonce, personalization]);
    drbg
  }

  fn fill(&mut self, out: &mut [u8]) {
    let mut offset = 0usize;
    while offset < out.len() {
      self.value = keygen_hmac_sha256(&self.key, &[&self.value]);
      let chunk_len = core::cmp::min(RSA_KEYGEN_DRBG_OUT_BYTES, out.len().strict_sub(offset));
      if let (Some(dst), Some(src)) = (
        out.get_mut(offset..offset.strict_add(chunk_len)),
        self.value.get(..chunk_len),
      ) {
        dst.copy_from_slice(src);
      }
      offset = offset.strict_add(chunk_len);
    }
    self.update(&[]);
  }

  fn update(&mut self, seed_material: &[&[u8]]) {
    self.key = keygen_drbg_update_key(&self.key, &self.value, 0, seed_material);
    self.value = keygen_hmac_sha256(&self.key, &[&self.value]);

    if seed_material.iter().all(|part| part.is_empty()) {
      return;
    }

    self.key = keygen_drbg_update_key(&self.key, &self.value, 1, seed_material);
    self.value = keygen_hmac_sha256(&self.key, &[&self.value]);
  }
}

#[cfg(feature = "getrandom")]
impl Drop for RsaKeygenDrbg {
  fn drop(&mut self) {
    ct::zeroize(&mut self.key);
    ct::zeroize(&mut self.value);
  }
}

#[cfg(feature = "getrandom")]
fn keygen_drbg_update_key(
  key: &[u8; RSA_KEYGEN_DRBG_KEY_BYTES],
  value: &[u8; RSA_KEYGEN_DRBG_OUT_BYTES],
  domain: u8,
  seed_material: &[&[u8]],
) -> [u8; RSA_KEYGEN_DRBG_OUT_BYTES] {
  let domain = [domain];
  let mut mac_parts = Vec::with_capacity(seed_material.len().strict_add(2));
  mac_parts.push(&value[..]);
  mac_parts.push(&domain[..]);
  mac_parts.extend_from_slice(seed_material);
  keygen_hmac_sha256(key, &mac_parts)
}

#[cfg(feature = "getrandom")]
fn keygen_hmac_sha256(key: &[u8], data: &[&[u8]]) -> [u8; RSA_KEYGEN_DRBG_OUT_BYTES] {
  let mut key_block = [0u8; RSA_KEYGEN_DRBG_HMAC_BLOCK_BYTES];
  if key.len() > RSA_KEYGEN_DRBG_HMAC_BLOCK_BYTES {
    let digest = Sha256::digest(key);
    if let Some(dst) = key_block.get_mut(..RSA_KEYGEN_DRBG_OUT_BYTES) {
      dst.copy_from_slice(digest.as_ref());
    }
  } else {
    if let Some(dst) = key_block.get_mut(..key.len()) {
      dst.copy_from_slice(key);
    }
  }

  let mut ipad = [0x36u8; RSA_KEYGEN_DRBG_HMAC_BLOCK_BYTES];
  let mut opad = [0x5cu8; RSA_KEYGEN_DRBG_HMAC_BLOCK_BYTES];
  for ((ipad_byte, opad_byte), key_byte) in ipad.iter_mut().zip(opad.iter_mut()).zip(key_block.iter().copied()) {
    *ipad_byte ^= key_byte;
    *opad_byte ^= key_byte;
  }

  let mut inner = Sha256::new();
  inner.update(&ipad);
  for part in data {
    inner.update(part);
  }
  let inner_digest = inner.finalize();

  let mut outer = Sha256::new();
  outer.update(&opad);
  outer.update(inner_digest.as_ref());
  let tag = outer.finalize();

  ct::zeroize(&mut key_block);
  ct::zeroize(&mut ipad);
  ct::zeroize(&mut opad);
  let mut out = [0u8; RSA_KEYGEN_DRBG_OUT_BYTES];
  out.copy_from_slice(tag.as_ref());
  out
}

#[cfg(feature = "getrandom")]
fn generate_rsa_private_key(
  modulus_bits: usize,
  policy: &RsaPublicKeyPolicy,
) -> Result<RsaPrivateKeyComponents, RsaKeyGenerationError> {
  if policy.min_modulus_bits > policy.max_modulus_bits
    || modulus_bits < policy.min_modulus_bits
    || modulus_bits > policy.max_modulus_bits
    || modulus_bits < MIN_RSA_MODULUS_BITS
    || !modulus_bits.is_multiple_of(2)
    || !policy.exponent_policy.accepts(RSA_KEYGEN_PUBLIC_EXPONENT)
    || !keygen_public_exponent_is_fips_valid(RSA_KEYGEN_PUBLIC_EXPONENT)
  {
    return Err(RsaKeyGenerationError::InvalidModulusBits);
  }

  let mut drbg = RsaKeygenDrbg::from_os_entropy()?;
  let prime_p_bits = modulus_bits / 2;
  let prime_q_bits = modulus_bits / 2;
  for _ in 0..RSA_KEYGEN_PAIR_ATTEMPTS {
    let prime_p = keygen_generate_prime(&mut drbg, prime_p_bits, modulus_bits, None, modulus_bits.strict_mul(5))?;
    let prime_q = match keygen_generate_prime(
      &mut drbg,
      prime_q_bits,
      modulus_bits,
      Some(prime_p.as_slice()),
      modulus_bits.strict_mul(10),
    ) {
      Ok(prime_q) => prime_q,
      Err(err) => {
        let mut prime_p = prime_p;
        ct::zeroize(&mut prime_p);
        return Err(err);
      }
    };
    if let Some(components) = keygen_build_private_key_from_primes(modulus_bits, policy, prime_p, prime_q)? {
      return Ok(components);
    }
  }

  Err(RsaKeyGenerationError::PrimeSearchFailed)
}

#[cfg(feature = "getrandom")]
const fn keygen_public_exponent_is_fips_valid(exponent: u64) -> bool {
  exponent > (1u64 << 16) && exponent % 2 == 1
}

#[cfg(feature = "getrandom")]
fn keygen_build_private_key_from_primes(
  modulus_bits: usize,
  policy: &RsaPublicKeyPolicy,
  prime_p: Vec<u8>,
  prime_q: Vec<u8>,
) -> Result<Option<RsaPrivateKeyComponents>, RsaKeyGenerationError> {
  let prime_p = SecretBigEndianBuffer::new(prime_p);
  let prime_q = SecretBigEndianBuffer::new(prime_q);

  if prime_p.as_slice() == prime_q.as_slice() {
    return Ok(None);
  }
  if keygen_conflicts_with_public_exponent(prime_p.as_slice())
    || keygen_conflicts_with_public_exponent(prime_q.as_slice())
  {
    return Ok(None);
  }
  if !keygen_prime_distance_is_sufficient(prime_p.as_slice(), prime_q.as_slice(), modulus_bits) {
    return Ok(None);
  }

  let modulus = private_import_product_unsigned_be(prime_p.as_slice(), prime_q.as_slice())
    .ok_or(RsaKeyGenerationError::ArithmeticFailure)?;
  if unsigned_be_bit_len(modulus.as_slice()) != modulus_bits {
    return Ok(None);
  }

  let p_minus_one =
    private_import_decrement_unsigned_be(prime_p.as_slice()).map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?;
  let q_minus_one =
    private_import_decrement_unsigned_be(prime_q.as_slice()).map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?;
  let lambda = keygen_lcm_unsigned_be(p_minus_one.as_slice(), q_minus_one.as_slice())?;
  let private_exponent = SecretBigEndianBuffer::new(keygen_inverse_small_mod_odd(
    RSA_KEYGEN_PUBLIC_EXPONENT,
    lambda.as_slice(),
  )?);
  if !keygen_private_exponent_is_large_enough(private_exponent.as_slice(), modulus_bits) {
    return Ok(None);
  }
  let exponent_p = private_import_unsigned_be_mod(private_exponent.as_slice(), p_minus_one.as_slice());
  let exponent_q = private_import_unsigned_be_mod(private_exponent.as_slice(), q_minus_one.as_slice());
  let coefficient = SecretBigEndianBuffer::new(keygen_prime_mod_inverse(prime_q.as_slice(), prime_p.as_slice())?);

  let public_exponent = RsaPublicExponent(RSA_KEYGEN_PUBLIC_EXPONENT);
  let generated = RsaPrivateKeyDerComponents {
    modulus: modulus.as_slice(),
    public_exponent,
    private_exponent: private_exponent.as_slice(),
    prime_p: prime_p.as_slice(),
    prime_q: prime_q.as_slice(),
    exponent_p: exponent_p.as_slice(),
    exponent_q: exponent_q.as_slice(),
    coefficient: coefficient.as_slice(),
  };
  validate_private_key_components(&generated).map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?;
  let checked_modulus_bits =
    validate_modulus(modulus.as_slice(), policy).map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?;
  let public = RsaPublicKey {
    modulus: RsaPublicModulus::new_with_montgomery_r2(modulus.as_slice(), checked_modulus_bits),
    exponent: public_exponent,
  };
  let prime_p_modulus =
    private_component_modulus(prime_p.as_slice()).map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?;
  let prime_q_modulus =
    private_component_modulus(prime_q.as_slice()).map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?;

  Ok(Some(RsaPrivateKeyComponents {
    public,
    private_exponent: SecretBigEndianInteger::from_vec(private_exponent.into_vec())
      .map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?,
    prime_p: SecretBigEndianInteger::from_vec(prime_p.into_vec())
      .map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?,
    prime_q: SecretBigEndianInteger::from_vec(prime_q.into_vec())
      .map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?,
    prime_p_modulus,
    prime_q_modulus,
    exponent_p: SecretBigEndianInteger::from_vec(exponent_p.into_vec())
      .map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?,
    exponent_q: SecretBigEndianInteger::from_vec(exponent_q.into_vec())
      .map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?,
    coefficient: SecretBigEndianInteger::from_vec(coefficient.into_vec())
      .map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?,
  }))
}

#[cfg(feature = "getrandom")]
fn keygen_generate_prime(
  drbg: &mut RsaKeygenDrbg,
  bits: usize,
  modulus_bits: usize,
  other_prime: Option<&[u8]>,
  tested_candidate_limit: usize,
) -> Result<Vec<u8>, RsaKeyGenerationError> {
  if bits < 2 {
    return Err(RsaKeyGenerationError::InvalidModulusBits);
  }

  let mut tested_candidates = 0usize;
  while tested_candidates < tested_candidate_limit {
    let candidate = keygen_random_odd_candidate(drbg, bits);
    if !keygen_probable_prime_meets_fips_lower_bound(&candidate, bits)
      || other_prime.is_some_and(|prime| !keygen_prime_distance_is_sufficient(&candidate, prime, modulus_bits))
    {
      let mut candidate = candidate;
      ct::zeroize(&mut candidate);
      continue;
    }

    tested_candidates = tested_candidates.strict_add(1);
    // Trial division rejects obvious composites before the expensive B.3 Miller-Rabin work.
    if keygen_has_small_prime_factor(&candidate) || keygen_conflicts_with_public_exponent(&candidate) {
      let mut candidate = candidate;
      ct::zeroize(&mut candidate);
      continue;
    }
    match keygen_is_probable_prime(drbg, &candidate) {
      Ok(true) => return Ok(candidate),
      Ok(false) => {
        let mut candidate = candidate;
        ct::zeroize(&mut candidate);
      }
      Err(err) => {
        let mut candidate = candidate;
        ct::zeroize(&mut candidate);
        return Err(err);
      }
    }
  }

  Err(RsaKeyGenerationError::PrimeSearchFailed)
}

#[cfg(feature = "getrandom")]
const fn keygen_miller_rabin_rounds(_candidate_bits: usize) -> usize {
  RSA_KEYGEN_MILLER_RABIN_ROUNDS
}

#[cfg(feature = "getrandom")]
fn keygen_random_odd_candidate(drbg: &mut RsaKeygenDrbg, bits: usize) -> Vec<u8> {
  let len = bits.strict_add(7) / 8;
  let mut candidate = vec![0u8; len];
  drbg.fill(&mut candidate);
  keygen_mask_unused_top_bits(&mut candidate, bits);
  if let Some(last) = candidate.last_mut() {
    *last |= 1;
  }
  candidate
}

#[cfg(all(feature = "getrandom", test))]
fn keygen_candidate_has_fixed_shape(candidate: &[u8], bits: usize) -> bool {
  unsigned_be_bit_len(candidate) == bits
    && candidate.last().is_some_and(|last| last & 1 == 1)
    && bits > 1
    && keygen_bit_is_set(candidate, bits.strict_sub(1))
}

#[cfg(feature = "getrandom")]
fn keygen_probable_prime_meets_fips_lower_bound(candidate: &[u8], bits: usize) -> bool {
  if bits.is_multiple_of(8)
    && let Some(prefix) = candidate.get(..RSA_KEYGEN_SQRT2_HALF_TOP64.len())
  {
    match prefix.cmp(&RSA_KEYGEN_SQRT2_HALF_TOP64) {
      core::cmp::Ordering::Greater => return true,
      core::cmp::Ordering::Less => return false,
      core::cmp::Ordering::Equal => {}
    }
  }

  private_import_product_unsigned_be(candidate, candidate)
    .as_ref()
    .is_some_and(|square| unsigned_be_bit_len(square.as_slice()) >= bits.strict_mul(2))
}

#[cfg(feature = "getrandom")]
fn keygen_mask_unused_top_bits(bytes: &mut [u8], bits: usize) {
  let used_top_bits = bits % 8;
  if used_top_bits == 0 {
    return;
  }
  if let Some(first) = bytes.first_mut() {
    *first &= (1u8 << used_top_bits) - 1;
  }
}

#[cfg(feature = "getrandom")]
fn keygen_set_bit(bytes: &mut [u8], bit: usize) {
  let byte_from_end = bit / 8;
  let Some(byte_index) = bytes.len().checked_sub(byte_from_end.strict_add(1)) else {
    return;
  };
  if let Some(byte) = bytes.get_mut(byte_index) {
    *byte |= 1u8 << (bit % 8);
  }
}

#[cfg(feature = "getrandom")]
fn keygen_bit_is_set(bytes: &[u8], bit: usize) -> bool {
  let byte_from_end = bit / 8;
  let Some(byte_index) = bytes.len().checked_sub(byte_from_end.strict_add(1)) else {
    return false;
  };
  bytes.get(byte_index).is_some_and(|byte| byte & (1u8 << (bit % 8)) != 0)
}

#[cfg(feature = "getrandom")]
fn keygen_has_small_prime_factor(candidate: &[u8]) -> bool {
  RSA_KEYGEN_SMALL_PRIMES
    .iter()
    .any(|&prime| unsigned_be_mod_u64(candidate, u64::from(prime)) == 0)
}

#[cfg(feature = "getrandom")]
fn keygen_conflicts_with_public_exponent(candidate: &[u8]) -> bool {
  unsigned_be_mod_u64(candidate, RSA_KEYGEN_PUBLIC_EXPONENT) == 1
}

#[cfg(feature = "getrandom")]
fn keygen_prime_distance_is_sufficient(prime_p: &[u8], prime_q: &[u8], modulus_bits: usize) -> bool {
  let Some(min_distance_bits) = (modulus_bits / 2).checked_sub(RSA_KEYGEN_MIN_PRIME_DISTANCE_SECURITY_MARGIN_BITS)
  else {
    return true;
  };

  let (larger, smaller) = match unsigned_be_cmp(prime_p, prime_q) {
    core::cmp::Ordering::Less => (prime_q, prime_p),
    core::cmp::Ordering::Equal => return false,
    core::cmp::Ordering::Greater => (prime_p, prime_q),
  };
  let Ok(distance) = private_sub_unsigned_be_to_len(larger, smaller, larger.len()) else {
    return false;
  };
  unsigned_be_bit_len(distance.as_slice()) > min_distance_bits
}

#[cfg(feature = "getrandom")]
fn keygen_is_probable_prime(drbg: &mut RsaKeygenDrbg, candidate: &[u8]) -> Result<bool, RsaKeyGenerationError> {
  let n_minus_one =
    private_import_decrement_unsigned_be(candidate).map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?;
  let mut n_minus_one_fixed = vec![0u8; candidate.len()];
  if let Err(err) = keygen_left_pad(n_minus_one.as_slice(), &mut n_minus_one_fixed) {
    ct::zeroize(&mut n_minus_one_fixed);
    return Err(err);
  }

  let mut odd_part = n_minus_one.as_slice().to_vec();
  let mut powers_of_two = 0usize;
  while odd_part.last().is_some_and(|byte| byte & 1 == 0) {
    keygen_shift_right_one(&mut odd_part);
    odd_part = keygen_canonical_vec(odd_part);
    powers_of_two = powers_of_two.strict_add(1);
  }

  let result = (|| {
    let modulus = private_component_modulus(candidate).map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?;
    for _ in 0..keygen_miller_rabin_rounds(unsigned_be_bit_len(candidate)) {
      let mut base = keygen_random_miller_rabin_base(drbg, candidate, &n_minus_one_fixed)?;
      let accepted = keygen_miller_rabin_accepts_base(&modulus, &odd_part, powers_of_two, &n_minus_one_fixed, &base);
      ct::zeroize(&mut base);
      if !accepted? {
        return Ok(false);
      }
    }
    Ok(true)
  })();

  ct::zeroize(&mut n_minus_one_fixed);
  ct::zeroize(&mut odd_part);
  result
}

#[cfg(feature = "getrandom")]
fn keygen_random_miller_rabin_base(
  drbg: &mut RsaKeygenDrbg,
  candidate: &[u8],
  n_minus_one_fixed: &[u8],
) -> Result<Vec<u8>, RsaKeyGenerationError> {
  let mut n_minus_two = n_minus_one_fixed.to_vec();
  keygen_sub_one_fixed(&mut n_minus_two)?;
  let mut two = vec![0u8; candidate.len()];
  if let Some(last) = two.last_mut() {
    *last = 2;
  }

  for _ in 0..128 {
    let mut base = vec![0u8; candidate.len()];
    drbg.fill(&mut base);
    keygen_mask_unused_top_bits(&mut base, unsigned_be_bit_len(candidate));
    if unsigned_be_cmp(&base, &two) != core::cmp::Ordering::Less
      && unsigned_be_cmp(&base, &n_minus_two) != core::cmp::Ordering::Greater
    {
      ct::zeroize(&mut n_minus_two);
      ct::zeroize(&mut two);
      return Ok(base);
    }
    ct::zeroize(&mut base);
  }

  ct::zeroize(&mut n_minus_two);
  ct::zeroize(&mut two);
  Err(RsaKeyGenerationError::PrimeSearchFailed)
}

#[cfg(feature = "getrandom")]
fn keygen_miller_rabin_accepts_base(
  modulus: &RsaPublicModulus,
  odd_part: &[u8],
  powers_of_two: usize,
  n_minus_one_fixed: &[u8],
  base: &[u8],
) -> Result<bool, RsaKeyGenerationError> {
  let mut x = vec![0u8; modulus.bytes.len()];
  if private_exponentiate_representative(modulus, odd_part, base, &mut x).is_err() {
    ct::zeroize(&mut x);
    return Err(RsaKeyGenerationError::ArithmeticFailure);
  }

  let mut accepted = keygen_is_one_fixed(&x) || ct::constant_time_eq(&x, n_minus_one_fixed);
  for _ in 1..powers_of_two {
    if accepted {
      break;
    }
    let mut squared = vec![0u8; modulus.bytes.len()];
    if mod_mul_representatives(modulus, &x, &x, &mut squared).is_err() {
      ct::zeroize(&mut squared);
      ct::zeroize(&mut x);
      return Err(RsaKeyGenerationError::ArithmeticFailure);
    }
    x.copy_from_slice(&squared);
    ct::zeroize(&mut squared);
    accepted = ct::constant_time_eq(&x, n_minus_one_fixed);
  }

  ct::zeroize(&mut x);
  Ok(accepted)
}

#[cfg(feature = "getrandom")]
fn keygen_prime_mod_inverse(value: &[u8], prime_modulus: &[u8]) -> Result<Vec<u8>, RsaKeyGenerationError> {
  let modulus = private_component_modulus(prime_modulus).map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?;
  let value_mod = private_import_unsigned_be_mod(value, prime_modulus);
  let mut value_fixed = vec![0u8; prime_modulus.len()];
  if let Err(err) = keygen_left_pad(value_mod.as_slice(), &mut value_fixed) {
    ct::zeroize(&mut value_fixed);
    return Err(err);
  }

  let p_minus_one =
    private_import_decrement_unsigned_be(prime_modulus).map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?;
  let p_minus_two = private_import_decrement_unsigned_be(p_minus_one.as_slice())
    .map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?;
  let mut inverse = vec![0u8; prime_modulus.len()];
  let result =
    if private_exponentiate_representative(&modulus, p_minus_two.as_slice(), &value_fixed, &mut inverse).is_ok() {
      Ok(keygen_canonical_vec(inverse))
    } else {
      ct::zeroize(&mut inverse);
      Err(RsaKeyGenerationError::ArithmeticFailure)
    };
  ct::zeroize(&mut value_fixed);
  result
}

#[cfg(feature = "getrandom")]
fn keygen_lcm_unsigned_be(left: &[u8], right: &[u8]) -> Result<SecretBigEndianBuffer, RsaKeyGenerationError> {
  let gcd = keygen_gcd_unsigned_be(left, right)?;
  let quotient = keygen_div_exact_unsigned_be(left, gcd.as_slice())?;
  private_import_product_unsigned_be(quotient.as_slice(), right).ok_or(RsaKeyGenerationError::ArithmeticFailure)
}

#[cfg(feature = "getrandom")]
fn keygen_gcd_unsigned_be(left: &[u8], right: &[u8]) -> Result<SecretBigEndianBuffer, RsaKeyGenerationError> {
  let mut a = keygen_canonical_vec(left.to_vec());
  let mut b = keygen_canonical_vec(right.to_vec());
  while !is_zero_unsigned_be(&b) {
    let remainder = private_import_unsigned_be_mod(&a, &b);
    ct::zeroize(&mut a);
    a = b;
    b = remainder.as_slice().to_vec();
  }
  ct::zeroize(&mut b);
  Ok(SecretBigEndianBuffer::new(a))
}

#[cfg(feature = "getrandom")]
fn keygen_div_exact_unsigned_be(
  dividend: &[u8],
  divisor: &[u8],
) -> Result<SecretBigEndianBuffer, RsaKeyGenerationError> {
  if is_zero_unsigned_be(divisor) {
    return Err(RsaKeyGenerationError::ArithmeticFailure);
  }
  if unsigned_be_cmp(dividend, divisor) == core::cmp::Ordering::Less {
    return Err(RsaKeyGenerationError::ArithmeticFailure);
  }

  let quotient_len = dividend.len();
  let mut quotient = vec![0u8; quotient_len];
  let mut remainder = vec![0u8; divisor.len().strict_add(1)];
  let dividend_bits = unsigned_be_bit_len(dividend);
  for bit_index in (0..dividend_bits).rev() {
    keygen_shift_left_one_fixed(&mut remainder);
    if keygen_bit_is_set(dividend, bit_index)
      && let Some(last) = remainder.last_mut()
    {
      *last |= 1;
    }
    let remainder_canonical = keygen_canonical_vec(remainder.clone());
    if unsigned_be_cmp(&remainder_canonical, divisor) != core::cmp::Ordering::Less {
      let divisor_fixed = keygen_left_pad_vec(divisor, remainder.len())?;
      let mut difference = vec![0u8; remainder.len()];
      private_sub_unsigned_be_to_fixed(&remainder, divisor_fixed.as_slice(), &mut difference)
        .map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?;
      remainder.copy_from_slice(&difference);
      ct::zeroize(&mut difference);
      keygen_set_bit(&mut quotient, bit_index);
    }
  }

  let remainder_canonical = keygen_canonical_vec(remainder.clone());
  ct::zeroize(&mut remainder);
  if !is_zero_unsigned_be(&remainder_canonical) {
    ct::zeroize(&mut quotient);
    return Err(RsaKeyGenerationError::ArithmeticFailure);
  }

  Ok(SecretBigEndianBuffer::new(keygen_canonical_vec(quotient)))
}

#[cfg(feature = "getrandom")]
fn keygen_left_pad_vec(src: &[u8], len: usize) -> Result<Vec<u8>, RsaKeyGenerationError> {
  if src.len() > len {
    return Err(RsaKeyGenerationError::ArithmeticFailure);
  }
  let mut out = vec![0u8; len];
  keygen_left_pad(src, &mut out)?;
  Ok(out)
}

#[cfg(feature = "getrandom")]
fn keygen_shift_left_one_fixed(bytes: &mut [u8]) {
  let mut carry = 0u8;
  for byte in bytes.iter_mut().rev() {
    let next_carry = *byte >> 7;
    *byte = (*byte << 1) | carry;
    carry = next_carry;
  }
}

#[cfg(feature = "getrandom")]
fn keygen_private_exponent_is_large_enough(private_exponent: &[u8], modulus_bits: usize) -> bool {
  let minimum_bit = modulus_bits / 2;
  let mut minimum = vec![0u8; minimum_bit.strict_add(8) / 8];
  keygen_set_bit(&mut minimum, minimum_bit);
  unsigned_be_cmp(private_exponent, &minimum) == core::cmp::Ordering::Greater
}

#[cfg(feature = "getrandom")]
fn keygen_inverse_small_mod_odd(exponent: u64, modulus: &[u8]) -> Result<Vec<u8>, RsaKeyGenerationError> {
  let modulus_mod_exponent = unsigned_be_mod_u64(modulus, exponent);
  if modulus_mod_exponent == 0 {
    return Err(RsaKeyGenerationError::ArithmeticFailure);
  }

  for k in 1..exponent {
    if (1u128 + u128::from(k).strict_mul(u128::from(modulus_mod_exponent))).is_multiple_of(u128::from(exponent)) {
      return keygen_mul_u64_add_one_div_u64(modulus, k, exponent);
    }
  }

  Err(RsaKeyGenerationError::ArithmeticFailure)
}

#[cfg(feature = "getrandom")]
fn keygen_mul_u64_add_one_div_u64(
  value: &[u8],
  multiplier: u64,
  divisor: u64,
) -> Result<Vec<u8>, RsaKeyGenerationError> {
  let mut product_rev = Vec::with_capacity(value.len().strict_add(8));
  let mut carry = 1u128;
  for &byte in value.iter().rev() {
    let acc = u128::from(byte).strict_mul(u128::from(multiplier)).strict_add(carry);
    product_rev.push(acc as u8);
    carry = acc >> 8;
  }
  while carry != 0 {
    product_rev.push(carry as u8);
    carry >>= 8;
  }
  product_rev.reverse();

  let mut quotient = Vec::with_capacity(product_rev.len());
  let mut remainder = 0u128;
  let divisor = u128::from(divisor);
  for &byte in &product_rev {
    let acc = (remainder << 8).strict_add(u128::from(byte));
    quotient.push((acc / divisor) as u8);
    remainder = acc % divisor;
  }
  ct::zeroize(&mut product_rev);
  if remainder != 0 {
    ct::zeroize(&mut quotient);
    return Err(RsaKeyGenerationError::ArithmeticFailure);
  }

  Ok(keygen_canonical_vec(quotient))
}

#[cfg(feature = "getrandom")]
fn keygen_shift_right_one(bytes: &mut [u8]) {
  let mut carry = 0u8;
  for byte in bytes {
    let next_carry = *byte & 1;
    *byte = (*byte >> 1) | (carry << 7);
    carry = next_carry;
  }
}

#[cfg(feature = "getrandom")]
fn keygen_sub_one_fixed(bytes: &mut [u8]) -> Result<(), RsaKeyGenerationError> {
  for byte in bytes.iter_mut().rev() {
    if *byte == 0 {
      *byte = 0xff;
    } else {
      *byte = byte.strict_sub(1);
      return Ok(());
    }
  }
  Err(RsaKeyGenerationError::ArithmeticFailure)
}

#[cfg(feature = "getrandom")]
fn keygen_is_one_fixed(bytes: &[u8]) -> bool {
  bytes.last() == Some(&1)
    && bytes
      .get(..bytes.len().saturating_sub(1))
      .is_some_and(|prefix| prefix.iter().all(|&byte| byte == 0))
}

#[cfg(feature = "getrandom")]
fn keygen_left_pad(src: &[u8], out: &mut [u8]) -> Result<(), RsaKeyGenerationError> {
  if src.len() > out.len() {
    return Err(RsaKeyGenerationError::ArithmeticFailure);
  }
  out.fill(0);
  let offset = out.len().strict_sub(src.len());
  let Some(dst) = out.get_mut(offset..) else {
    return Err(RsaKeyGenerationError::ArithmeticFailure);
  };
  dst.copy_from_slice(src);
  Ok(())
}

#[cfg(feature = "getrandom")]
fn keygen_canonical_vec(bytes: Vec<u8>) -> Vec<u8> {
  let canonical = private_import_canonical_unsigned_be(bytes);
  canonical.as_slice().to_vec()
}

#[cfg(feature = "getrandom")]
fn unsigned_be_bit_len(bytes: &[u8]) -> usize {
  let Some(first_nonzero) = bytes.iter().position(|&byte| byte != 0) else {
    return 0;
  };
  let significant = bytes.get(first_nonzero..).unwrap_or_default();
  let Some(&first) = significant.first() else {
    return 0;
  };
  significant
    .len()
    .strict_sub(1)
    .strict_mul(8)
    .strict_add(8usize.strict_sub(first.leading_zeros() as usize))
}

#[cfg(feature = "getrandom")]
fn unsigned_be_mod_u64(bytes: &[u8], modulus: u64) -> u64 {
  let mut remainder = 0u128;
  let modulus = u128::from(modulus);
  for &byte in bytes {
    remainder = ((remainder << 8).strict_add(u128::from(byte))) % modulus;
  }
  remainder as u64
}

const fn tls13_signature_scheme_profile_and_key_algorithm(
  scheme: u16,
) -> Result<(RsaSignatureProfile, RsaTlsPublicKeyAlgorithm), RsaProtocolAlgorithmError> {
  match scheme {
    0x0804 => Ok((
      RsaSignatureProfile::pss(RsaPssProfile::Sha256),
      RsaTlsPublicKeyAlgorithm::RsaEncryption,
    )),
    0x0805 => Ok((
      RsaSignatureProfile::pss(RsaPssProfile::Sha384),
      RsaTlsPublicKeyAlgorithm::RsaEncryption,
    )),
    0x0806 => Ok((
      RsaSignatureProfile::pss(RsaPssProfile::Sha512),
      RsaTlsPublicKeyAlgorithm::RsaEncryption,
    )),
    0x0809 => Ok((
      RsaSignatureProfile::pss(RsaPssProfile::Sha256),
      RsaTlsPublicKeyAlgorithm::RsaPss,
    )),
    0x080a => Ok((
      RsaSignatureProfile::pss(RsaPssProfile::Sha384),
      RsaTlsPublicKeyAlgorithm::RsaPss,
    )),
    0x080b => Ok((
      RsaSignatureProfile::pss(RsaPssProfile::Sha512),
      RsaTlsPublicKeyAlgorithm::RsaPss,
    )),
    _ => Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm),
  }
}

const fn tls_certificate_signature_scheme_profile_and_key_algorithm(
  scheme: u16,
) -> Result<(RsaSignatureProfile, RsaTlsPublicKeyAlgorithm), RsaProtocolAlgorithmError> {
  match scheme {
    0x0401 => Ok((
      RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256),
      RsaTlsPublicKeyAlgorithm::RsaEncryption,
    )),
    0x0501 => Ok((
      RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha384),
      RsaTlsPublicKeyAlgorithm::RsaEncryption,
    )),
    0x0601 => Ok((
      RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha512),
      RsaTlsPublicKeyAlgorithm::RsaEncryption,
    )),
    _ => tls13_signature_scheme_profile_and_key_algorithm(scheme),
  }
}

const fn advertised_restricted_pss_tls13_signature_scheme(
  profile: RsaPssProfile,
  minimum_salt_len: usize,
) -> RsaTlsSignatureSchemes {
  match profile {
    RsaPssProfile::Sha256 if minimum_salt_len <= RsaPssProfile::Sha256.digest_len() => {
      RsaTlsSignatureSchemes::new([0x0809, 0, 0, 0, 0, 0], 1)
    }
    RsaPssProfile::Sha384 if minimum_salt_len <= RsaPssProfile::Sha384.digest_len() => {
      RsaTlsSignatureSchemes::new([0x080a, 0, 0, 0, 0, 0], 1)
    }
    RsaPssProfile::Sha512 if minimum_salt_len <= RsaPssProfile::Sha512.digest_len() => {
      RsaTlsSignatureSchemes::new([0x080b, 0, 0, 0, 0, 0], 1)
    }
    RsaPssProfile::Sha256 | RsaPssProfile::Sha384 | RsaPssProfile::Sha512 => RsaTlsSignatureSchemes::EMPTY,
  }
}

fn parse_spki_der(der: &[u8]) -> Result<(&[u8], &[u8]), RsaKeyError> {
  let mut root = DerReader::new(der);
  let spki = root.read_constructed(TAG_SEQUENCE)?;
  root.finish()?;

  let mut spki = DerReader::new(spki);
  let algorithm = spki.read_constructed(TAG_SEQUENCE)?;
  let subject_public_key = spki.read_primitive(TAG_BIT_STRING)?;
  spki.finish()?;

  let (&unused_bits, public_key_der) = subject_public_key.split_first().ok_or(RsaKeyError::MalformedDer)?;
  if unused_bits != 0 || public_key_der.is_empty() {
    return Err(RsaKeyError::MalformedDer);
  }

  Ok((algorithm, public_key_der))
}

fn parse_pkcs1_public_key_der_parts<'a>(
  der: &'a [u8],
  policy: &RsaPublicKeyPolicy,
) -> Result<(&'a [u8], usize, RsaPublicExponent), RsaKeyError> {
  if policy.min_modulus_bits > policy.max_modulus_bits {
    return Err(RsaKeyError::InvalidModulus);
  }

  let mut root = DerReader::new(der);
  let public_key = root.read_constructed(TAG_SEQUENCE)?;
  root.finish()?;

  let mut public_key = DerReader::new(public_key);
  let modulus = read_positive_integer(public_key.read_primitive(TAG_INTEGER)?)?;
  let exponent = parse_public_exponent(read_positive_integer(public_key.read_primitive(TAG_INTEGER)?)?, policy)?;
  public_key.finish()?;

  let modulus_bits = validate_modulus(modulus, policy)?;
  Ok((modulus, modulus_bits, exponent))
}

fn parse_x509_public_key_algorithm_identifier(der: &[u8]) -> Result<RsaX509PublicKeyAlgorithm, RsaKeyError> {
  let mut reader = DerReader::new(der);
  let oid = reader.read_primitive(TAG_OBJECT_IDENTIFIER)?;
  match oid {
    RSA_ENCRYPTION_OID => {
      let null = reader.read_primitive(TAG_NULL)?;
      if !null.is_empty() {
        return Err(RsaKeyError::MalformedDer);
      }
      reader.finish()?;
      Ok(RsaX509PublicKeyAlgorithm::RsaEncryption)
    }
    ID_RSASSA_PSS_OID => {
      if reader.peek_byte().is_none() {
        reader.finish()?;
        return Ok(RsaX509PublicKeyAlgorithm::RsaPss);
      }
      let params = reader.read_constructed(TAG_SEQUENCE)?;
      reader.finish()?;
      let (profile, minimum_salt_len) = parse_x509_pss_parameters(params).map_err(protocol_algorithm_to_key_error)?;
      Ok(RsaX509PublicKeyAlgorithm::RsaPssRestricted {
        profile,
        minimum_salt_len,
      })
    }
    _ => Err(RsaKeyError::UnsupportedAlgorithm),
  }
}

fn parse_x509_signature_algorithm(der: &[u8]) -> Result<RsaSignatureProfile, RsaProtocolAlgorithmError> {
  let mut root = DerReader::new(der);
  let algorithm = protocol_der(root.read_constructed(TAG_SEQUENCE))?;
  protocol_der(root.finish())?;

  let mut reader = DerReader::new(algorithm);
  let oid = protocol_der(reader.read_primitive(TAG_OBJECT_IDENTIFIER))?;
  let profile = match oid {
    SHA256_WITH_RSA_ENCRYPTION_OID => parse_x509_pkcs1v15_algorithm(reader, RsaPkcs1v15Profile::Sha256)?,
    SHA384_WITH_RSA_ENCRYPTION_OID => parse_x509_pkcs1v15_algorithm(reader, RsaPkcs1v15Profile::Sha384)?,
    SHA512_WITH_RSA_ENCRYPTION_OID => parse_x509_pkcs1v15_algorithm(reader, RsaPkcs1v15Profile::Sha512)?,
    ID_RSASSA_PSS_OID => parse_x509_pss_algorithm(reader)?,
    SHA1_WITH_RSA_ENCRYPTION_OID => return Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm),
    _ => return Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm),
  };
  Ok(profile)
}

struct ParsedX509CertificateSignature<'a> {
  tbs_certificate_der: &'a [u8],
  signature_algorithm_der: &'a [u8],
  signature: &'a [u8],
}

fn parse_x509_certificate_signature(
  der: &[u8],
) -> Result<ParsedX509CertificateSignature<'_>, RsaProtocolAlgorithmError> {
  let mut root = DerReader::new(der);
  let certificate = protocol_der(root.read_constructed(TAG_SEQUENCE))?;
  protocol_der(root.finish())?;

  let mut certificate = DerReader::new(certificate);
  let tbs_certificate_der = protocol_der(certificate.read_tlv(TAG_SEQUENCE))?;
  let signature_algorithm_der = protocol_der(certificate.read_tlv(TAG_SEQUENCE))?;
  let signature_value = protocol_der(certificate.read_primitive(TAG_BIT_STRING))?;
  protocol_der(certificate.finish())?;

  let tbs_signature_algorithm_der = parse_tbs_certificate_signature_algorithm(tbs_certificate_der)?;
  if tbs_signature_algorithm_der != signature_algorithm_der {
    return Err(RsaProtocolAlgorithmError::MalformedAlgorithmIdentifier);
  }

  let (&unused_bits, signature) = signature_value
    .split_first()
    .ok_or(RsaProtocolAlgorithmError::MalformedAlgorithmIdentifier)?;
  if unused_bits != 0 || signature.is_empty() {
    return Err(RsaProtocolAlgorithmError::MalformedAlgorithmIdentifier);
  }

  Ok(ParsedX509CertificateSignature {
    tbs_certificate_der,
    signature_algorithm_der,
    signature,
  })
}

fn parse_tbs_certificate_signature_algorithm(tbs_certificate_der: &[u8]) -> Result<&[u8], RsaProtocolAlgorithmError> {
  let mut root = DerReader::new(tbs_certificate_der);
  let tbs_certificate = protocol_der(root.read_constructed(TAG_SEQUENCE))?;
  protocol_der(root.finish())?;

  let mut tbs_certificate = DerReader::new(tbs_certificate);
  if tbs_certificate.peek_byte() == Some(TAG_CONTEXT_0) {
    let _ = protocol_der(tbs_certificate.read_constructed(TAG_CONTEXT_0))?;
  }
  let _ = protocol_der(tbs_certificate.read_primitive(TAG_INTEGER))?;
  protocol_der(tbs_certificate.read_tlv(TAG_SEQUENCE))
}

fn parse_x509_pkcs1v15_algorithm(
  mut reader: DerReader<'_>,
  profile: RsaPkcs1v15Profile,
) -> Result<RsaSignatureProfile, RsaProtocolAlgorithmError> {
  if reader.peek_byte().is_some() {
    let null = protocol_der(reader.read_primitive(TAG_NULL))?;
    if !null.is_empty() {
      return Err(RsaProtocolAlgorithmError::MalformedAlgorithmIdentifier);
    }
  }
  protocol_der(reader.finish())?;
  Ok(RsaSignatureProfile::pkcs1v15(profile))
}

fn parse_x509_pss_algorithm(mut reader: DerReader<'_>) -> Result<RsaSignatureProfile, RsaProtocolAlgorithmError> {
  let params = protocol_der(reader.read_constructed(TAG_SEQUENCE))?;
  protocol_der(reader.finish())?;

  let (profile, salt_len) = parse_x509_pss_parameters(params)?;
  Ok(RsaSignatureProfile::pss_with_salt_len(profile, salt_len))
}

fn parse_x509_pss_parameters(params: &[u8]) -> Result<(RsaPssProfile, usize), RsaProtocolAlgorithmError> {
  let mut params = DerReader::new(params);
  let hash = if params.peek_byte() == Some(TAG_CONTEXT_0) {
    let field = protocol_der(params.read_constructed(TAG_CONTEXT_0))?;
    Some(parse_x509_hash_algorithm(field)?)
  } else {
    None
  };
  let mgf = if params.peek_byte() == Some(TAG_CONTEXT_1) {
    let field = protocol_der(params.read_constructed(TAG_CONTEXT_1))?;
    Some(parse_x509_mgf1_algorithm(field)?)
  } else {
    None
  };
  let salt_len = if params.peek_byte() == Some(TAG_CONTEXT_2) {
    let field = protocol_der(params.read_constructed(TAG_CONTEXT_2))?;
    Some(parse_x509_nonnegative_integer(field)?)
  } else {
    None
  };
  if params.peek_byte() == Some(TAG_CONTEXT_3) {
    let field = protocol_der(params.read_constructed(TAG_CONTEXT_3))?;
    if parse_x509_nonnegative_integer(field)? != 1 {
      return Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm);
    }
  }
  protocol_der(params.finish())?;

  let Some(hash) = hash else {
    return Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm);
  };
  let Some(mgf) = mgf else {
    return Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm);
  };
  let salt_len = salt_len.unwrap_or(20);
  if hash != mgf {
    return Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm);
  }
  Ok((hash, salt_len))
}

fn parse_x509_hash_algorithm(der: &[u8]) -> Result<RsaPssProfile, RsaProtocolAlgorithmError> {
  let mut reader = DerReader::new(der);
  let algorithm = protocol_der(reader.read_constructed(TAG_SEQUENCE))?;
  protocol_der(reader.finish())?;
  parse_x509_hash_algorithm_body(algorithm)
}

fn parse_x509_hash_algorithm_body(algorithm: &[u8]) -> Result<RsaPssProfile, RsaProtocolAlgorithmError> {
  let mut algorithm = DerReader::new(algorithm);
  let oid = protocol_der(algorithm.read_primitive(TAG_OBJECT_IDENTIFIER))?;
  let profile = match oid {
    ID_SHA256_OID => RsaPssProfile::Sha256,
    ID_SHA384_OID => RsaPssProfile::Sha384,
    ID_SHA512_OID => RsaPssProfile::Sha512,
    ID_SHA1_OID => return Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm),
    _ => return Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm),
  };
  if algorithm.peek_byte().is_some() {
    let null = protocol_der(algorithm.read_primitive(TAG_NULL))?;
    if !null.is_empty() {
      return Err(RsaProtocolAlgorithmError::MalformedAlgorithmIdentifier);
    }
  }
  protocol_der(algorithm.finish())?;
  Ok(profile)
}

fn parse_x509_mgf1_algorithm(der: &[u8]) -> Result<RsaPssProfile, RsaProtocolAlgorithmError> {
  let mut reader = DerReader::new(der);
  let algorithm = protocol_der(reader.read_constructed(TAG_SEQUENCE))?;
  protocol_der(reader.finish())?;

  let mut algorithm = DerReader::new(algorithm);
  let oid = protocol_der(algorithm.read_primitive(TAG_OBJECT_IDENTIFIER))?;
  if oid != ID_MGF1_OID {
    return Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm);
  }
  let hash = parse_x509_hash_algorithm_body(protocol_der(algorithm.read_constructed(TAG_SEQUENCE))?)?;
  protocol_der(algorithm.finish())?;
  Ok(hash)
}

fn parse_x509_nonnegative_integer(der: &[u8]) -> Result<usize, RsaProtocolAlgorithmError> {
  let mut reader = DerReader::new(der);
  let integer = protocol_der(reader.read_primitive(TAG_INTEGER))?;
  protocol_der(reader.finish())?;
  let integer = protocol_der(read_positive_integer(integer))?;
  let mut value = 0usize;
  for &byte in integer {
    value = value
      .checked_mul(256)
      .and_then(|value| value.checked_add(usize::from(byte)))
      .ok_or(RsaProtocolAlgorithmError::UnsupportedAlgorithm)?;
  }
  Ok(value)
}

fn protocol_der<T>(result: Result<T, RsaKeyError>) -> Result<T, RsaProtocolAlgorithmError> {
  result.map_err(|_| RsaProtocolAlgorithmError::MalformedAlgorithmIdentifier)
}

const fn protocol_algorithm_to_key_error(error: RsaProtocolAlgorithmError) -> RsaKeyError {
  match error {
    RsaProtocolAlgorithmError::MalformedAlgorithmIdentifier => RsaKeyError::MalformedDer,
    RsaProtocolAlgorithmError::UnsupportedAlgorithm => RsaKeyError::UnsupportedAlgorithm,
  }
}

fn read_positive_integer(bytes: &[u8]) -> Result<&[u8], RsaKeyError> {
  let (&first, rest) = bytes.split_first().ok_or(RsaKeyError::MalformedDer)?;

  if first == 0 {
    let Some((&next, _)) = rest.split_first() else {
      return Ok(bytes);
    };
    if next & 0x80 == 0 {
      return Err(RsaKeyError::MalformedDer);
    }
    return Ok(rest);
  }

  if first & 0x80 != 0 {
    return Err(RsaKeyError::MalformedDer);
  }

  Ok(bytes)
}

fn parse_public_exponent(bytes: &[u8], policy: &RsaPublicKeyPolicy) -> Result<RsaPublicExponent, RsaKeyError> {
  if bytes.len() > core::mem::size_of::<u64>() {
    return Err(RsaKeyError::InvalidPublicExponent);
  }

  let mut value = 0u64;
  for &byte in bytes {
    value = (value << 8) | u64::from(byte);
  }

  if value < 3 || value & 1 == 0 || !policy.exponent_policy.accepts(value) {
    return Err(RsaKeyError::InvalidPublicExponent);
  }

  Ok(RsaPublicExponent(value))
}

fn validate_modulus(modulus: &[u8], policy: &RsaPublicKeyPolicy) -> Result<usize, RsaKeyError> {
  let (&first, _) = modulus.split_first().ok_or(RsaKeyError::InvalidModulus)?;
  let Some(&last) = modulus.last() else {
    return Err(RsaKeyError::InvalidModulus);
  };

  if first == 0 || last & 1 == 0 {
    return Err(RsaKeyError::InvalidModulus);
  }

  let modulus_bits = modulus
    .len()
    .strict_sub(1)
    .strict_mul(8)
    .strict_add(8usize.strict_sub(first.leading_zeros() as usize));

  if modulus_bits < policy.min_modulus_bits || modulus_bits > policy.max_modulus_bits {
    return Err(RsaKeyError::InvalidModulus);
  }

  Ok(modulus_bits)
}

#[allow(clippy::indexing_slicing)]
fn encode_pkcs1v15<D>(message: &[u8], digest_info_prefix: &[u8], out: &mut [u8]) -> Result<(), RsaPrivateOpError>
where
  D: Digest,
{
  let digest = D::digest(message);
  let digest_info_len = digest_info_prefix.len().strict_add(D::OUTPUT_SIZE);
  let minimum_len = 11usize.strict_add(digest_info_len);
  if out.len() < minimum_len {
    return Err(RsaPrivateOpError::MessageTooLong);
  }

  let ps_len = out.len().strict_sub(digest_info_len).strict_sub(3);
  out[0] = 0;
  out[1] = 1;
  out[2..2usize.strict_add(ps_len)].fill(0xff);
  let separator = 2usize.strict_add(ps_len);
  out[separator] = 0;
  let digest_info = separator.strict_add(1);
  out[digest_info..digest_info.strict_add(digest_info_prefix.len())].copy_from_slice(digest_info_prefix);
  out[digest_info.strict_add(digest_info_prefix.len())..].copy_from_slice(digest.as_ref());
  Ok(())
}

fn pkcs1v15_encryption_padding_len(key_len: usize, message_len: usize) -> Result<usize, RsaEncryptionError> {
  key_len
    .checked_sub(message_len)
    .and_then(|len| len.checked_sub(3))
    .filter(|&ps_len| ps_len >= 8)
    .ok_or(RsaEncryptionError::MessageTooLong)
}

#[cfg(feature = "getrandom")]
fn fill_pkcs1v15_nonzero_padding(out: &mut [u8]) -> Result<(), RsaEncryptionError> {
  getrandom::fill(out).map_err(|_| RsaEncryptionError::EntropyUnavailable)?;
  for byte in out.iter_mut() {
    while *byte == 0 {
      getrandom::fill(core::slice::from_mut(byte)).map_err(|_| RsaEncryptionError::EntropyUnavailable)?;
    }
  }
  Ok(())
}

#[allow(clippy::indexing_slicing)]
fn encode_pkcs1v15_encryption_with_seed(message: &[u8], seed: &[u8], out: &mut [u8]) -> Result<(), RsaEncryptionError> {
  let ps_len = pkcs1v15_encryption_padding_len(out.len(), message.len())?;
  if seed.len() != ps_len {
    return Err(RsaEncryptionError::InvalidLength);
  }
  if seed.contains(&0) {
    return Err(RsaEncryptionError::InvalidLength);
  }

  out[0] = 0;
  out[1] = 2;
  out[2..2usize.strict_add(ps_len)].copy_from_slice(seed);
  let separator = 2usize.strict_add(ps_len);
  out[separator] = 0;
  out[separator.strict_add(1)..].copy_from_slice(message);
  Ok(())
}

fn decode_pkcs1v15_encryption(encoded: &[u8], out: &mut [u8]) -> Result<usize, RsaPrivateOpError> {
  if encoded.len() < 11 {
    return Err(RsaPrivateOpError::DecryptionFailed);
  }

  let mut bad = u8::from(encoded.first().copied() != Some(0));
  bad |= u8::from(encoded.get(1).copied() != Some(2));
  let mut seen_separator = 0u8;
  let mut separator = 0usize;

  for (index, &byte) in encoded.iter().enumerate().skip(2) {
    let before_separator = seen_separator ^ 1;
    let is_zero = u8::from(byte == 0);
    let separator_at_index = before_separator & is_zero;
    separator = ct_select_usize(separator, index, separator_at_index);
    seen_separator |= separator_at_index;
  }

  bad |= seen_separator ^ 1;
  bad |= u8::from(separator < 10);
  if bad != 0 {
    return Err(RsaPrivateOpError::DecryptionFailed);
  }

  let message = encoded
    .get(separator.strict_add(1)..)
    .ok_or(RsaPrivateOpError::DecryptionFailed)?;
  if out.len() < message.len() {
    return Err(RsaPrivateOpError::InvalidLength);
  }
  let out = out.get_mut(..message.len()).ok_or(RsaPrivateOpError::InvalidLength)?;
  out.copy_from_slice(message);
  Ok(message.len())
}

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

#[allow(clippy::indexing_slicing)]
fn encode_pss<D>(message: &[u8], salt: &[u8], em_bits: usize, out: &mut [u8]) -> Result<(), RsaPrivateOpError>
where
  D: Digest,
{
  let h_len = D::OUTPUT_SIZE;
  let db_len = out
    .len()
    .checked_sub(h_len)
    .and_then(|len| len.checked_sub(1))
    .ok_or(RsaPrivateOpError::MessageTooLong)?;
  let mut db_mask = vec![0u8; db_len];
  let result = encode_pss_with_mask::<D>(message, salt, em_bits, out, &mut db_mask);
  ct::zeroize(&mut db_mask);
  result
}

#[allow(clippy::indexing_slicing)]
fn encode_pss_with_mask<D>(
  message: &[u8],
  salt: &[u8],
  em_bits: usize,
  out: &mut [u8],
  db_mask: &mut [u8],
) -> Result<(), RsaPrivateOpError>
where
  D: Digest,
{
  let h_len = D::OUTPUT_SIZE;
  let em_len = out.len();
  let minimum_len = h_len
    .checked_add(salt.len())
    .and_then(|len| len.checked_add(2))
    .ok_or(RsaPrivateOpError::MessageTooLong)?;
  if em_len < minimum_len {
    return Err(RsaPrivateOpError::MessageTooLong);
  }

  let unused_bits = em_len
    .checked_mul(8)
    .and_then(|bits| bits.checked_sub(em_bits))
    .ok_or(RsaPrivateOpError::MessageTooLong)?;
  if unused_bits >= 8 {
    return Err(RsaPrivateOpError::MessageTooLong);
  }

  let m_hash = D::digest(message);
  let mut h = D::new();
  h.update(&[0u8; 8]);
  h.update(m_hash.as_ref());
  h.update(salt);
  let h = h.finalize();

  let db_len = em_len.strict_sub(h_len).strict_sub(1);
  let Some(db_mask) = db_mask.get_mut(..db_len) else {
    return Err(RsaPrivateOpError::InvalidLength);
  };
  let ps_len = db_len.strict_sub(salt.len()).strict_sub(1);
  let (db, h_and_trailer) = out.split_at_mut(db_len);
  db[..ps_len].fill(0);
  db[ps_len] = 1;
  db[ps_len.strict_add(1)..].copy_from_slice(salt);

  mgf1::<D>(h.as_ref(), db_mask);
  for (byte, mask) in db.iter_mut().zip(db_mask.iter().copied()) {
    *byte ^= mask;
  }
  if unused_bits > 0
    && let Some(first) = db.first_mut()
  {
    *first &= 0xffu8 >> unused_bits;
  }

  let (h_out, trailer) = h_and_trailer.split_at_mut(h_len);
  h_out.copy_from_slice(h.as_ref());
  trailer[0] = 0xbc;
  Ok(())
}

#[allow(clippy::indexing_slicing)]
fn encode_oaep_with_masks<D>(
  label: &[u8],
  message: &[u8],
  seed: &[u8],
  out: &mut [u8],
  db_mask: &mut [u8],
  seed_mask: &mut [u8],
) -> Result<(), RsaEncryptionError>
where
  D: Digest,
{
  let h_len = D::OUTPUT_SIZE;
  if seed.len() != h_len || out.len() < h_len.strict_mul(2).strict_add(2) {
    return Err(RsaEncryptionError::InvalidLength);
  }
  let db_len = out.len().strict_sub(h_len).strict_sub(1);
  let Some(db_mask) = db_mask.get_mut(..db_len) else {
    return Err(RsaEncryptionError::InvalidLength);
  };
  let Some(seed_mask) = seed_mask.get_mut(..h_len) else {
    return Err(RsaEncryptionError::InvalidLength);
  };
  let ps_len = out
    .len()
    .checked_sub(message.len())
    .and_then(|len| len.checked_sub(h_len.strict_mul(2)))
    .and_then(|len| len.checked_sub(2))
    .ok_or(RsaEncryptionError::MessageTooLong)?;

  let label_hash = D::digest(label);
  let (leading, rest) = out.split_at_mut(1);
  leading[0] = 0;
  let (masked_seed, masked_db) = rest.split_at_mut(h_len);
  masked_seed.copy_from_slice(seed);
  masked_db[..h_len].copy_from_slice(label_hash.as_ref());
  masked_db[h_len..h_len.strict_add(ps_len)].fill(0);
  masked_db[h_len.strict_add(ps_len)] = 1;
  masked_db[h_len.strict_add(ps_len).strict_add(1)..].copy_from_slice(message);

  mgf1::<D>(seed, db_mask);
  for (byte, mask) in masked_db.iter_mut().zip(db_mask.iter().copied()) {
    *byte ^= mask;
  }

  mgf1::<D>(masked_db, seed_mask);
  for (byte, mask) in masked_seed.iter_mut().zip(seed_mask.iter().copied()) {
    *byte ^= mask;
  }
  Ok(())
}

#[allow(clippy::indexing_slicing)]
fn decode_oaep<D>(label: &[u8], encoded: &mut [u8], out: &mut [u8]) -> Result<usize, RsaPrivateOpError>
where
  D: Digest,
{
  let h_len = D::OUTPUT_SIZE;
  let db_mask_len = encoded
    .len()
    .checked_sub(h_len)
    .and_then(|len| len.checked_sub(1))
    .ok_or(RsaPrivateOpError::DecryptionFailed)?;
  let mut seed_mask = vec![0u8; h_len];
  let mut db_mask = vec![0u8; db_mask_len];
  let result = decode_oaep_with_masks::<D>(label, encoded, out, &mut seed_mask, &mut db_mask);
  ct::zeroize(&mut seed_mask);
  ct::zeroize(&mut db_mask);
  result
}

#[allow(clippy::indexing_slicing)]
fn decode_oaep_with_masks<D>(
  label: &[u8],
  encoded: &mut [u8],
  out: &mut [u8],
  seed_mask: &mut [u8],
  db_mask: &mut [u8],
) -> Result<usize, RsaPrivateOpError>
where
  D: Digest,
{
  let h_len = D::OUTPUT_SIZE;
  if encoded.len() < h_len.strict_mul(2).strict_add(2) {
    return Err(RsaPrivateOpError::DecryptionFailed);
  }
  let db_len = encoded.len().strict_sub(h_len).strict_sub(1);
  let Some(seed_mask) = seed_mask.get_mut(..h_len) else {
    return Err(RsaPrivateOpError::InvalidLength);
  };
  let Some(db_mask) = db_mask.get_mut(..db_len) else {
    return Err(RsaPrivateOpError::InvalidLength);
  };

  let label_hash = D::digest(label);
  let (leading, rest) = encoded.split_at_mut(1);
  let (masked_seed, masked_db) = rest.split_at_mut(h_len);
  let mut bad = u8::from(leading.first().copied() != Some(0));

  mgf1::<D>(masked_db, seed_mask);
  for (byte, mask) in masked_seed.iter_mut().zip(seed_mask.iter().copied()) {
    *byte ^= mask;
  }

  mgf1::<D>(masked_seed, db_mask);
  for (byte, mask) in masked_db.iter_mut().zip(db_mask.iter().copied()) {
    *byte ^= mask;
  }

  let (decoded_label_hash, rest) = masked_db.split_at(h_len);
  bad |= u8::from(!ct::constant_time_eq(decoded_label_hash, label_hash.as_ref()));

  let mut seen_separator = 0u8;
  let mut separator = 0usize;
  for (index, &byte) in rest.iter().enumerate() {
    let before_separator = seen_separator ^ 1;
    let is_zero = u8::from(byte == 0);
    let is_one = u8::from(byte == 1);
    let separator_at_index = before_separator & is_one;
    let invalid_before_separator = before_separator & ((is_zero | is_one) ^ 1);
    bad |= invalid_before_separator;
    separator = ct_select_usize(separator, index, separator_at_index);
    seen_separator |= separator_at_index;
  }
  bad |= seen_separator ^ 1;

  if bad != 0 {
    return Err(RsaPrivateOpError::DecryptionFailed);
  }
  let message = &rest[separator.strict_add(1)..];
  if out.len() < message.len() {
    return Err(RsaPrivateOpError::InvalidLength);
  }
  out[..message.len()].copy_from_slice(message);
  Ok(message.len())
}

fn ct_select_usize(zero: usize, one: usize, choice: u8) -> usize {
  let mask = 0usize.wrapping_sub(usize::from(choice & 1));
  (zero & !mask) | (one & mask)
}

#[cfg(test)]
fn verify_pss_encoded<D>(message: &[u8], encoded: &[u8], em_bits: usize) -> Result<(), VerificationError>
where
  D: Digest,
{
  let mut db = vec![0u8; encoded.len()];
  let mut db_mask = vec![0u8; encoded.len()];
  verify_pss_encoded_with_scratch::<D>(message, encoded, em_bits, D::OUTPUT_SIZE, &mut db, &mut db_mask)
}

fn verify_pss_encoded_with_scratch<D>(
  message: &[u8],
  encoded: &[u8],
  em_bits: usize,
  salt_len: usize,
  db: &mut [u8],
  db_mask: &mut [u8],
) -> Result<(), VerificationError>
where
  D: Digest,
{
  let h_len = D::OUTPUT_SIZE;
  let em_len = encoded.len();
  let min_len = h_len
    .checked_add(salt_len)
    .and_then(|len| len.checked_add(2))
    .ok_or_else(VerificationError::new)?;
  if em_len < min_len {
    return Err(VerificationError::new());
  }
  if encoded.last().copied() != Some(0xbc) {
    return Err(VerificationError::new());
  }

  let db_len = em_len.strict_sub(h_len).strict_sub(1);
  let (masked_db, h_and_trailer) = encoded.split_at(db_len);
  let h = h_and_trailer.get(..h_len).ok_or_else(VerificationError::new)?;
  let db = db.get_mut(..db_len).ok_or_else(VerificationError::new)?;
  let db_mask = db_mask.get_mut(..db_len).ok_or_else(VerificationError::new)?;

  let unused_bits = em_len
    .checked_mul(8)
    .and_then(|bits| bits.checked_sub(em_bits))
    .ok_or_else(VerificationError::new)?;
  if unused_bits >= 8 {
    return Err(VerificationError::new());
  }
  if unused_bits > 0 {
    let mask = 0xffu8 << (8usize.strict_sub(unused_bits) as u32);
    if masked_db.first().copied().unwrap_or(0) & mask != 0 {
      return Err(VerificationError::new());
    }
  }

  db.copy_from_slice(masked_db);
  db_mask.fill(0);
  mgf1::<D>(h, db_mask);
  for (dst, mask) in db.iter_mut().zip(db_mask.iter().copied()) {
    *dst ^= mask;
  }
  if unused_bits > 0 {
    let mask = 0xffu8 >> unused_bits;
    if let Some(first) = db.first_mut() {
      *first &= mask;
    }
  }

  let ps_len = em_len.strict_sub(h_len).strict_sub(salt_len).strict_sub(2);
  if db.get(..ps_len).is_none_or(|ps| ps.iter().any(|&byte| byte != 0)) {
    return Err(VerificationError::new());
  }
  if db.get(ps_len).copied() != Some(0x01) {
    return Err(VerificationError::new());
  }
  let salt_start = ps_len.strict_add(1);
  let salt = db.get(salt_start..).ok_or_else(VerificationError::new)?;
  if salt.len() != salt_len {
    return Err(VerificationError::new());
  }

  let m_hash = D::digest(message);
  let mut verifier = D::new();
  verifier.update(&[0u8; 8]);
  verifier.update(m_hash.as_ref());
  verifier.update(salt);
  let expected_h = verifier.finalize();

  if ct::constant_time_eq(expected_h.as_ref(), h) {
    Ok(())
  } else {
    Err(VerificationError::new())
  }
}

fn verify_pkcs1v15_encoded<D>(
  message: &[u8],
  encoded: &[u8],
  digest_info_prefix: &[u8],
) -> Result<(), VerificationError>
where
  D: Digest,
{
  let digest = D::digest(message);
  let digest_info_len = digest_info_prefix.len().strict_add(D::OUTPUT_SIZE);
  if encoded.len() < 11usize.strict_add(digest_info_len) {
    return Err(VerificationError::new());
  }

  let separator_index = encoded.len().strict_sub(digest_info_len).strict_sub(1);
  let padding = encoded.get(2..separator_index).ok_or_else(VerificationError::new)?;
  let digest_info = encoded
    .get(separator_index.strict_add(1)..)
    .ok_or_else(VerificationError::new)?;
  let (prefix, value) = digest_info.split_at(digest_info_prefix.len());

  let mut valid = encoded.first().copied() == Some(0x00);
  valid &= encoded.get(1).copied() == Some(0x01);
  valid &= padding.len() >= 8;
  for &byte in padding {
    valid &= byte == 0xff;
  }
  valid &= encoded.get(separator_index).copied() == Some(0x00);
  valid &= ct::constant_time_eq(prefix, digest_info_prefix);
  valid &= ct::constant_time_eq(value, digest.as_ref());

  if valid { Ok(()) } else { Err(VerificationError::new()) }
}

fn mgf1<D>(seed: &[u8], out: &mut [u8])
where
  D: Digest,
{
  let mut counter = 0u32;
  let mut offset = 0usize;
  while offset < out.len() {
    let digest = D::digest_vectored(&[seed, &counter.to_be_bytes()]);
    let chunk_len = core::cmp::min(D::OUTPUT_SIZE, out.len().strict_sub(offset));
    if let Some(dst) = out.get_mut(offset..offset.strict_add(chunk_len)) {
      let src = digest.as_ref().get(..chunk_len).unwrap_or_default();
      dst.copy_from_slice(src);
    }
    offset = offset.strict_add(chunk_len);
    counter = counter.strict_add(1);
  }
}

#[allow(clippy::indexing_slicing)]
fn limbs_from_be(bytes: &[u8]) -> Vec<u64> {
  let limbs = bytes.len().strict_add(7) / 8;
  let mut out = vec![0u64; limbs];
  limbs_from_be_into(bytes, &mut out);
  out
}

#[allow(clippy::indexing_slicing)]
fn limbs_from_be_into(bytes: &[u8], out: &mut [u64]) {
  let full_limbs = bytes.len() / 8;
  let leading = bytes.len() % 8;
  let needed_limbs = full_limbs + usize::from(leading != 0);
  if out.len() != needed_limbs {
    out.fill(0);
  }

  for (limb_index, slot) in out.iter_mut().enumerate().take(full_limbs) {
    let start = bytes.len().strict_sub((limb_index.strict_add(1)).strict_mul(8));
    let mut limb = [0u8; 8];
    limb.copy_from_slice(&bytes[start..start.strict_add(8)]);
    *slot = u64::from_be_bytes(limb);
  }

  if leading != 0 {
    let mut limb = 0u64;
    for &byte in &bytes[..leading] {
      limb = (limb << 8) | u64::from(byte);
    }
    out[full_limbs] = limb;
  }
}

#[allow(clippy::indexing_slicing)]
fn limbs_to_be(limbs: &[u64], out: &mut [u8]) {
  let full_limbs = out.len() / 8;
  let leading = out.len() % 8;

  for (limb_index, limb) in limbs.iter().copied().enumerate().take(full_limbs) {
    let start = out.len().strict_sub((limb_index.strict_add(1)).strict_mul(8));
    out[start..start.strict_add(8)].copy_from_slice(&limb.to_be_bytes());
  }

  if leading != 0 {
    let limb = limbs[full_limbs].to_be_bytes();
    out[..leading].copy_from_slice(&limb[8usize.strict_sub(leading)..]);
  }
}

fn copy_limbs(dst: &mut [u64], src: &[u64]) {
  for (d, s) in dst.iter_mut().zip(src.iter().copied()) {
    *d = s;
  }
}

fn is_zero_unsigned_be(bytes: &[u8]) -> bool {
  let mut acc = 0u8;
  for &byte in bytes {
    acc |= byte;
  }
  acc == 0
}

fn is_canonical_positive_unsigned_be(bytes: &[u8]) -> bool {
  matches!(bytes.first(), Some(&first) if first != 0)
}

fn unsigned_be_cmp(left: &[u8], right: &[u8]) -> core::cmp::Ordering {
  match left.len().cmp(&right.len()) {
    core::cmp::Ordering::Equal => left.cmp(right),
    ordering => ordering,
  }
}

fn ct_lt_u8(left: u8, right: u8) -> u8 {
  ((u16::from(left).wrapping_sub(u16::from(right)) >> 8) & 1) as u8
}

fn ct_unsigned_be_lt_public_shape(left: &[u8], right: &[u8]) -> bool {
  match left.len().cmp(&right.len()) {
    core::cmp::Ordering::Less => true,
    core::cmp::Ordering::Greater => false,
    core::cmp::Ordering::Equal => {
      let mut lt = 0u8;
      let mut gt = 0u8;
      for (&left_byte, &right_byte) in left.iter().zip(right) {
        let undecided = (lt | gt) ^ 1;
        lt |= undecided & ct_lt_u8(left_byte, right_byte);
        gt |= undecided & ct_lt_u8(right_byte, left_byte);
      }
      lt == 1
    }
  }
}

fn ct_slices_eq_public_shape(left: &[u8], right: &[u8]) -> bool {
  if left.len() != right.len() {
    return false;
  }
  ct::constant_time_eq(left, right)
}

fn product_matches_unsigned_be_fixed(left: &[u8], right: &[u8], expected: &[u8]) -> bool {
  let mut product = vec![0u8; expected.len()];
  let matched = private_import_product_unsigned_be_to_fixed(left, right, &mut product).is_ok()
    && ct::constant_time_eq(&product, expected);
  ct::zeroize(&mut product);
  matched
}

fn ct_eq_left_padded_unsigned_be(value: &[u8], expected: &[u8]) -> bool {
  if value.len() > expected.len() {
    return false;
  }
  let mut padded = vec![0u8; expected.len()];
  let offset = expected.len().strict_sub(value.len());
  let Some(dst) = padded.get_mut(offset..) else {
    return false;
  };
  dst.copy_from_slice(value);
  let eq = ct::constant_time_eq(&padded, expected);
  ct::zeroize(&mut padded);
  eq
}

fn private_component_modulus(bytes: &[u8]) -> Result<RsaPublicModulus, RsaPrivateOpError> {
  if bytes.is_empty() {
    return Err(RsaPrivateOpError::RepresentativeOutOfRange);
  }
  let limbs = limbs_from_be(bytes);
  let n0 = montgomery_n0(limbs.first().copied().unwrap_or(1));
  let r2 = private_montgomery_r2(bytes).map_err(|_| RsaPrivateOpError::RepresentativeOutOfRange)?;
  Ok(RsaPublicModulus {
    bytes: Box::from(bytes),
    limbs: limbs.into_boxed_slice(),
    r2: Some(r2),
    bits: bytes.len().strict_mul(8),
    n0,
  })
}

fn left_pad_be(src: &[u8], out: &mut [u8]) -> Result<(), RsaPrivateOpError> {
  if src.len() > out.len() {
    return Err(RsaPrivateOpError::InvalidLength);
  }
  out.fill(0);
  let offset = out.len().strict_sub(src.len());
  let Some(dst) = out.get_mut(offset..) else {
    return Err(RsaPrivateOpError::InvalidLength);
  };
  dst.copy_from_slice(src);
  Ok(())
}

fn private_sub_mod_unsigned_be(
  left: &[u8],
  right: &[u8],
  modulus: &[u8],
) -> Result<SecretBigEndianBuffer, RsaPrivateOpError> {
  if left.len() != right.len() || left.len() != modulus.len() {
    return Err(RsaPrivateOpError::InvalidLength);
  }

  let difference = match unsigned_be_cmp(left, right) {
    core::cmp::Ordering::Less => {
      let plus_modulus = private_add_unsigned_be_to_len(left, modulus, modulus.len().strict_add(1))?;
      private_sub_unsigned_be_to_len(plus_modulus.as_slice(), right, modulus.len())?
    }
    core::cmp::Ordering::Equal => SecretBigEndianBuffer::new(vec![0]),
    core::cmp::Ordering::Greater => private_sub_unsigned_be_to_len(left, right, modulus.len())?,
  };
  Ok(difference)
}

#[allow(clippy::indexing_slicing)]
fn private_sub_mod_unsigned_be_to_fixed(
  left: &[u8],
  right: &[u8],
  modulus: &[u8],
  out: &mut [u8],
) -> Result<(), RsaPrivateOpError> {
  if left.len() != right.len() || left.len() != modulus.len() || out.len() != modulus.len() {
    return Err(RsaPrivateOpError::InvalidLength);
  }

  match unsigned_be_cmp(left, right) {
    core::cmp::Ordering::Equal => {
      out.fill(0);
      Ok(())
    }
    core::cmp::Ordering::Greater => private_sub_unsigned_be_to_fixed(left, right, out),
    core::cmp::Ordering::Less => private_sub_mod_less_unsigned_be_to_fixed(left, right, modulus, out),
  }
}

#[allow(clippy::indexing_slicing)]
fn private_sub_unsigned_be_to_fixed(left: &[u8], right: &[u8], out: &mut [u8]) -> Result<(), RsaPrivateOpError> {
  if left.len() != right.len() || left.len() != out.len() || unsigned_be_cmp(left, right) == core::cmp::Ordering::Less {
    return Err(RsaPrivateOpError::InvalidLength);
  }

  let mut borrow = 0i16;
  for index in 0..out.len() {
    let src = out.len().strict_sub(index).strict_sub(1);
    let mut difference = i16::from(left[src]) - i16::from(right[src]) - borrow;
    if difference < 0 {
      difference += 256;
      borrow = 1;
    } else {
      borrow = 0;
    }
    out[src] = difference as u8;
  }

  if borrow == 0 {
    Ok(())
  } else {
    Err(RsaPrivateOpError::InvalidLength)
  }
}

#[allow(clippy::indexing_slicing)]
fn private_sub_mod_less_unsigned_be_to_fixed(
  left: &[u8],
  right: &[u8],
  modulus: &[u8],
  out: &mut [u8],
) -> Result<(), RsaPrivateOpError> {
  if left.len() != right.len() || left.len() != modulus.len() || out.len() != modulus.len() {
    return Err(RsaPrivateOpError::InvalidLength);
  }

  let mut carry = 0u16;
  let mut borrow = 0i16;
  for index in 0..out.len() {
    let src = out.len().strict_sub(index).strict_sub(1);
    let sum = u16::from(left[src])
      .strict_add(u16::from(modulus[src]))
      .strict_add(carry);
    let sum_byte = (sum & 0xff) as i16;
    carry = sum >> 8;

    let mut difference = sum_byte - i16::from(right[src]) - borrow;
    if difference < 0 {
      difference += 256;
      borrow = 1;
    } else {
      borrow = 0;
    }
    out[src] = difference as u8;
  }

  if carry == borrow as u16 {
    Ok(())
  } else {
    Err(RsaPrivateOpError::InvalidLength)
  }
}

#[allow(clippy::indexing_slicing)]
fn private_add_unsigned_be_to_len(
  left: &[u8],
  right: &[u8],
  len: usize,
) -> Result<SecretBigEndianBuffer, RsaPrivateOpError> {
  if left.len() > len || right.len() > len {
    return Err(RsaPrivateOpError::InvalidLength);
  }

  let mut out = vec![0u8; len];
  let mut carry = 0u16;
  for index in 0..len {
    let left_byte = left
      .len()
      .checked_sub(index.strict_add(1))
      .and_then(|src| left.get(src))
      .copied()
      .unwrap_or(0);
    let right_byte = right
      .len()
      .checked_sub(index.strict_add(1))
      .and_then(|src| right.get(src))
      .copied()
      .unwrap_or(0);
    let sum = u16::from(left_byte).strict_add(u16::from(right_byte)).strict_add(carry);
    let dst = len.strict_sub(index).strict_sub(1);
    out[dst] = sum as u8;
    carry = sum >> 8;
  }

  if carry != 0 {
    return Err(RsaPrivateOpError::RepresentativeOutOfRange);
  }
  Ok(private_import_canonical_unsigned_be(out))
}

#[allow(clippy::indexing_slicing)]
fn private_sub_unsigned_be_to_len(
  left: &[u8],
  right: &[u8],
  len: usize,
) -> Result<SecretBigEndianBuffer, RsaPrivateOpError> {
  if left.len() > len.strict_add(1) || right.len() > len || unsigned_be_cmp(left, right) == core::cmp::Ordering::Less {
    return Err(RsaPrivateOpError::InvalidLength);
  }

  let mut out = vec![0u8; len.strict_add(1)];
  let mut borrow = 0i16;
  for index in 0..out.len() {
    let left_byte = left
      .len()
      .checked_sub(index.strict_add(1))
      .and_then(|src| left.get(src))
      .copied()
      .unwrap_or(0);
    let right_byte = right
      .len()
      .checked_sub(index.strict_add(1))
      .and_then(|src| right.get(src))
      .copied()
      .unwrap_or(0);
    let mut difference = i16::from(left_byte) - i16::from(right_byte) - borrow;
    if difference < 0 {
      difference += 256;
      borrow = 1;
    } else {
      borrow = 0;
    }
    let dst = out.len().strict_sub(index).strict_sub(1);
    out[dst] = difference as u8;
  }

  if borrow != 0 {
    return Err(RsaPrivateOpError::InvalidLength);
  }
  Ok(private_import_canonical_unsigned_be(out))
}

#[allow(clippy::indexing_slicing)]
fn private_exponentiate_representative(
  modulus: &RsaPublicModulus,
  exponent: &[u8],
  input: &[u8],
  out: &mut [u8],
) -> Result<(), RsaPrivateOpError> {
  let bytes = modulus.bytes.len();
  let limbs = modulus.limbs.len();
  if input.len() != bytes || out.len() != bytes || exponent.len() > bytes {
    return Err(RsaPrivateOpError::InvalidLength);
  }

  let mut t = SecretLimbs::zeroed(limbs.strict_mul(2).strict_add(2));
  let mut representative = SecretLimbs::zeroed(limbs);
  let mut one = SecretLimbs::zeroed(limbs);
  let mut base = SecretLimbs::zeroed(limbs);
  let mut acc = SecretLimbs::zeroed(limbs);
  let mut squared = SecretLimbs::zeroed(limbs);
  let mut multiplied = SecretLimbs::zeroed(limbs);
  let mut selected = SecretLimbs::zeroed(limbs);
  let mut reduced = SecretLimbs::zeroed(limbs);

  limbs_from_be_into(input, representative.as_mut_slice());
  if cmp_limbs(representative.as_slice(), &modulus.limbs) != core::cmp::Ordering::Less {
    return Err(RsaPrivateOpError::RepresentativeOutOfRange);
  }

  one.as_mut_slice()[0] = 1;
  private_mont_mul(
    base.as_mut_slice(),
    representative.as_slice(),
    modulus.montgomery_r2(),
    modulus,
    t.as_mut_slice(),
  );
  private_mont_mul(
    acc.as_mut_slice(),
    one.as_slice(),
    modulus.montgomery_r2(),
    modulus,
    t.as_mut_slice(),
  );

  let mut table = private_fixed_window_table(base.as_slice(), acc.as_slice(), modulus, t.as_mut_slice());
  let leading_zero_bytes = bytes.strict_sub(exponent.len());
  for index in 0..bytes {
    let exponent_byte = if index < leading_zero_bytes {
      0
    } else {
      exponent[index.strict_sub(leading_zero_bytes)]
    };
    private_exponentiate_window(
      &table,
      exponent_byte >> 4,
      modulus,
      t.as_mut_slice(),
      acc.as_mut_slice(),
      squared.as_mut_slice(),
      multiplied.as_mut_slice(),
      selected.as_mut_slice(),
    );
    private_exponentiate_window(
      &table,
      exponent_byte & 0x0f,
      modulus,
      t.as_mut_slice(),
      acc.as_mut_slice(),
      squared.as_mut_slice(),
      multiplied.as_mut_slice(),
      selected.as_mut_slice(),
    );
  }

  ct::zeroize_words(table.as_mut_slice());
  private_mont_reduce(reduced.as_mut_slice(), acc.as_slice(), modulus, t.as_mut_slice());
  limbs_to_be(reduced.as_slice(), out);
  Ok(())
}

#[allow(clippy::indexing_slicing)]
fn private_exponentiate_representative_with_scratch(
  modulus: &RsaPublicModulus,
  exponent: &[u8],
  input: &[u8],
  out: &mut [u8],
  scratch: &mut RsaPrivateExponentScratch,
) -> Result<(), RsaPrivateOpError> {
  let bytes = modulus.bytes.len();
  let limbs = modulus.limbs.len();
  if input.len() != bytes || out.len() != bytes || exponent.len() > bytes {
    return Err(RsaPrivateOpError::InvalidLength);
  }
  scratch.ensure_limb_count(limbs)?;

  limbs_from_be_into(input, scratch.representative.as_mut_slice());
  if cmp_limbs(scratch.representative.as_slice(), &modulus.limbs) != core::cmp::Ordering::Less {
    return Err(RsaPrivateOpError::RepresentativeOutOfRange);
  }

  scratch.one.as_mut_slice().fill(0);
  scratch.one.as_mut_slice()[0] = 1;
  private_mont_mul(
    scratch.base.as_mut_slice(),
    scratch.representative.as_slice(),
    modulus.montgomery_r2(),
    modulus,
    scratch.t.as_mut_slice(),
  );
  private_mont_mul(
    scratch.acc.as_mut_slice(),
    scratch.one.as_slice(),
    modulus.montgomery_r2(),
    modulus,
    scratch.t.as_mut_slice(),
  );

  private_fixed_window_table_into(
    scratch.table.as_mut_slice(),
    scratch.base.as_slice(),
    scratch.acc.as_slice(),
    modulus,
    scratch.t.as_mut_slice(),
  );
  let leading_zero_bytes = bytes.strict_sub(exponent.len());
  for index in 0..bytes {
    let exponent_byte = if index < leading_zero_bytes {
      0
    } else {
      exponent[index.strict_sub(leading_zero_bytes)]
    };
    private_exponentiate_window(
      scratch.table.as_slice(),
      exponent_byte >> 4,
      modulus,
      scratch.t.as_mut_slice(),
      scratch.acc.as_mut_slice(),
      scratch.squared.as_mut_slice(),
      scratch.multiplied.as_mut_slice(),
      scratch.selected.as_mut_slice(),
    );
    private_exponentiate_window(
      scratch.table.as_slice(),
      exponent_byte & 0x0f,
      modulus,
      scratch.t.as_mut_slice(),
      scratch.acc.as_mut_slice(),
      scratch.squared.as_mut_slice(),
      scratch.multiplied.as_mut_slice(),
      scratch.selected.as_mut_slice(),
    );
  }

  private_mont_reduce(
    scratch.reduced.as_mut_slice(),
    scratch.acc.as_slice(),
    modulus,
    scratch.t.as_mut_slice(),
  );
  limbs_to_be(scratch.reduced.as_slice(), out);
  Ok(())
}

#[allow(clippy::indexing_slicing)]
fn private_fixed_window_table(
  base: &[u64],
  one_montgomery: &[u64],
  modulus: &RsaPublicModulus,
  t: &mut [u64],
) -> Vec<u64> {
  let limbs = base.len();
  let mut table = vec![0u64; PRIVATE_FIXED_WINDOW_TABLE_ENTRIES.strict_mul(limbs)];
  private_fixed_window_table_into(&mut table, base, one_montgomery, modulus, t);
  table
}

#[allow(clippy::indexing_slicing)]
fn private_fixed_window_table_into(
  table: &mut [u64],
  base: &[u64],
  one_montgomery: &[u64],
  modulus: &RsaPublicModulus,
  t: &mut [u64],
) {
  let limbs = base.len();
  debug_assert_eq!(table.len(), PRIVATE_FIXED_WINDOW_TABLE_ENTRIES.strict_mul(limbs));
  table[..limbs].copy_from_slice(one_montgomery);
  table[limbs..limbs.strict_mul(2)].copy_from_slice(base);

  for index in 2..PRIVATE_FIXED_WINDOW_TABLE_ENTRIES {
    let previous_start = index.strict_sub(1).strict_mul(limbs);
    let current_start = index.strict_mul(limbs);
    let (previous, current_and_rest) = table.split_at_mut(current_start);
    let previous = &previous[previous_start..previous_start.strict_add(limbs)];
    let current = &mut current_and_rest[..limbs];
    private_mont_mul(current, previous, base, modulus, t);
  }
}

#[allow(clippy::too_many_arguments)]
fn private_exponentiate_window(
  table: &[u64],
  window: u8,
  modulus: &RsaPublicModulus,
  t: &mut [u64],
  acc: &mut [u64],
  squared: &mut [u64],
  multiplied: &mut [u64],
  selected: &mut [u64],
) {
  for _ in 0..4 {
    private_mont_mul(squared, acc, acc, modulus, t);
    acc.copy_from_slice(squared);
  }

  private_select_window_power(selected, table, window);
  private_mont_mul(multiplied, acc, selected, modulus, t);
  acc.copy_from_slice(multiplied);
}

#[allow(clippy::indexing_slicing)]
#[inline(always)]
fn private_select_window_power(out: &mut [u64], table: &[u64], window: u8) {
  let limbs = out.len();
  debug_assert_eq!(table.len(), limbs.strict_mul(PRIVATE_FIXED_WINDOW_TABLE_ENTRIES));
  out.fill(0);
  for index in 0..PRIVATE_FIXED_WINDOW_TABLE_ENTRIES as u8 {
    let start = usize::from(index).strict_mul(limbs);
    let entry = &table[start..start.strict_add(limbs)];
    let mask = core::hint::black_box(private_choice_eq_mask_u8(window, index));
    for (dst, limb) in out.iter_mut().zip(entry) {
      // SAFETY: Volatile table read is used as a compiler barrier because:
      // 1. `limb` is a valid shared reference to one fixed-window table limb.
      // 2. `u64` is `Copy`, so the volatile read does not create ownership aliasing.
      // 3. Every table limb is read unconditionally; the secret window affects only masks.
      let limb = unsafe { core::ptr::read_volatile(limb) };
      *dst = (*dst & !mask) | (limb & mask);
    }
  }
}

#[cfg(feature = "diag")]
#[inline(always)]
pub fn diag_rsa_private_select_window_power_4(table: &[u64; 64], window: u8) -> [u64; 4] {
  let mut out = [0u64; 4];
  private_select_window_power(&mut out, table, window);
  out
}

#[cfg(feature = "diag")]
#[inline(always)]
pub fn diag_rsa_private_component_validation_32(component: &[u8; 32], upper_bound: &[u8; 32], other: &[u8; 32]) -> u8 {
  let canonical = u8::from(is_canonical_positive_unsigned_be(component));
  let less_than_bound = u8::from(ct_unsigned_be_lt_public_shape(component, upper_bound));
  let distinct = u8::from(!ct_slices_eq_public_shape(component, other));
  let odd = component.last().copied().unwrap_or(0) & 1;
  canonical & less_than_bound & distinct & odd
}

fn private_choice_eq_mask_u8(left: u8, right: u8) -> u64 {
  let diff = u64::from(left ^ right);
  let is_zero = ((diff | diff.wrapping_neg()) >> 63) ^ 1;
  0u64.wrapping_sub(is_zero)
}

fn mod_mul_representatives(
  modulus: &RsaPublicModulus,
  left: &[u8],
  right: &[u8],
  out: &mut [u8],
) -> Result<(), RsaPrivateOpError> {
  let bytes = modulus.bytes.len();
  let limbs = modulus.limbs.len();
  if left.len() != bytes || right.len() != bytes || out.len() != bytes {
    return Err(RsaPrivateOpError::InvalidLength);
  }

  let mut t = SecretLimbs::zeroed(limbs.strict_mul(2).strict_add(2));
  let mut left_limbs = SecretLimbs::zeroed(limbs);
  let mut right_limbs = SecretLimbs::zeroed(limbs);
  let mut left_mont = SecretLimbs::zeroed(limbs);
  let mut right_mont = SecretLimbs::zeroed(limbs);
  let mut product_mont = SecretLimbs::zeroed(limbs);
  let mut product = SecretLimbs::zeroed(limbs);

  limbs_from_be_into(left, left_limbs.as_mut_slice());
  limbs_from_be_into(right, right_limbs.as_mut_slice());
  if cmp_limbs(left_limbs.as_slice(), &modulus.limbs) != core::cmp::Ordering::Less
    || cmp_limbs(right_limbs.as_slice(), &modulus.limbs) != core::cmp::Ordering::Less
  {
    return Err(RsaPrivateOpError::RepresentativeOutOfRange);
  }

  private_mont_mul(
    left_mont.as_mut_slice(),
    left_limbs.as_slice(),
    modulus.montgomery_r2(),
    modulus,
    t.as_mut_slice(),
  );
  private_mont_mul(
    right_mont.as_mut_slice(),
    right_limbs.as_slice(),
    modulus.montgomery_r2(),
    modulus,
    t.as_mut_slice(),
  );
  private_mont_mul(
    product_mont.as_mut_slice(),
    left_mont.as_slice(),
    right_mont.as_slice(),
    modulus,
    t.as_mut_slice(),
  );
  private_mont_reduce(
    product.as_mut_slice(),
    product_mont.as_slice(),
    modulus,
    t.as_mut_slice(),
  );
  limbs_to_be(product.as_slice(), out);
  Ok(())
}

fn mod_mul_representatives_with_scratch(
  modulus: &RsaPublicModulus,
  left: &[u8],
  right: &[u8],
  out: &mut [u8],
  scratch: &mut RsaPrivateMulScratch,
) -> Result<(), RsaPrivateOpError> {
  let bytes = modulus.bytes.len();
  let limbs = modulus.limbs.len();
  if left.len() != bytes || right.len() != bytes || out.len() != bytes {
    return Err(RsaPrivateOpError::InvalidLength);
  }
  scratch.ensure_limb_count(limbs)?;

  let t_len = limbs.strict_mul(2).strict_add(2);
  let t = scratch
    .t
    .as_mut_slice()
    .get_mut(..t_len)
    .ok_or(RsaPrivateOpError::InvalidScratch)?;
  let left_limbs = scratch
    .left_limbs
    .as_mut_slice()
    .get_mut(..limbs)
    .ok_or(RsaPrivateOpError::InvalidScratch)?;
  let right_limbs = scratch
    .right_limbs
    .as_mut_slice()
    .get_mut(..limbs)
    .ok_or(RsaPrivateOpError::InvalidScratch)?;
  let left_mont = scratch
    .left_mont
    .as_mut_slice()
    .get_mut(..limbs)
    .ok_or(RsaPrivateOpError::InvalidScratch)?;
  let right_mont = scratch
    .right_mont
    .as_mut_slice()
    .get_mut(..limbs)
    .ok_or(RsaPrivateOpError::InvalidScratch)?;
  let product_mont = scratch
    .product_mont
    .as_mut_slice()
    .get_mut(..limbs)
    .ok_or(RsaPrivateOpError::InvalidScratch)?;
  let product = scratch
    .product
    .as_mut_slice()
    .get_mut(..limbs)
    .ok_or(RsaPrivateOpError::InvalidScratch)?;

  limbs_from_be_into(left, left_limbs);
  limbs_from_be_into(right, right_limbs);
  if cmp_limbs(left_limbs, &modulus.limbs) != core::cmp::Ordering::Less
    || cmp_limbs(right_limbs, &modulus.limbs) != core::cmp::Ordering::Less
  {
    return Err(RsaPrivateOpError::RepresentativeOutOfRange);
  }

  private_mont_mul(left_mont, left_limbs, modulus.montgomery_r2(), modulus, t);
  private_mont_mul(right_mont, right_limbs, modulus.montgomery_r2(), modulus, t);
  private_mont_mul(product_mont, left_mont, right_mont, modulus, t);
  private_mont_reduce(product, product_mont, modulus, t);
  limbs_to_be(product, out);
  Ok(())
}

fn private_mont_mul(out: &mut [u64], a: &[u64], b: &[u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  if use_cios_montgomery(modulus) {
    mont_mul_cios(out, a, b, modulus, t);
  } else {
    mont_mul(out, a, b, modulus, t);
  }
}

fn private_mont_reduce(out: &mut [u64], value: &[u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  if use_cios_montgomery(modulus) {
    mont_reduce_cios(out, value, modulus, t);
  } else {
    mont_reduce(out, value, modulus, t);
  }
}

fn private_import_product_unsigned_be(left: &[u8], right: &[u8]) -> Option<SecretBigEndianBuffer> {
  let left = SecretLimbs::from_be(left);
  let right = SecretLimbs::from_be(right);
  let mut product = SecretLimbs::zeroed(left.as_slice().len().strict_add(right.as_slice().len()));

  for (left_index, &left_limb) in left.as_slice().iter().enumerate() {
    let mut carry = 0u128;
    for (right_index, &right_limb) in right.as_slice().iter().enumerate() {
      let index = left_index.strict_add(right_index);
      let limb = product.as_mut_slice().get_mut(index)?;
      let acc = u128::from(*limb)
        .strict_add(u128::from(left_limb).strict_mul(u128::from(right_limb)))
        .strict_add(carry);
      *limb = acc as u64;
      carry = acc >> 64;
    }

    let index = left_index.strict_add(right.as_slice().len());
    for limb in product.as_mut_slice().get_mut(index..)?.iter_mut() {
      let acc = u128::from(*limb).strict_add(carry);
      *limb = acc as u64;
      carry = acc >> 64;
    }
    if carry != 0 {
      return None;
    }
  }

  Some(private_import_limbs_to_canonical_be(product.as_slice()))
}

fn private_import_product_unsigned_be_to_fixed(left: &[u8], right: &[u8], out: &mut [u8]) -> Result<(), RsaKeyError> {
  let left_limb_count = left.len().strict_add(7) / 8;
  let right_limb_count = right.len().strict_add(7) / 8;
  let out_limb_count = out.len().strict_add(7) / 8;
  if left_limb_count.strict_add(right_limb_count) > out_limb_count {
    return Err(RsaKeyError::InvalidModulus);
  }

  let left = SecretLimbs::from_be(left);
  let right = SecretLimbs::from_be(right);
  let mut product = SecretLimbs::zeroed(out_limb_count);

  for (left_index, &left_limb) in left.as_slice().iter().enumerate() {
    let mut carry = 0u128;
    for (right_index, &right_limb) in right.as_slice().iter().enumerate() {
      let index = left_index.strict_add(right_index);
      let Some(limb) = product.as_mut_slice().get_mut(index) else {
        return Err(RsaKeyError::InvalidModulus);
      };
      let acc = u128::from(*limb)
        .strict_add(u128::from(left_limb).strict_mul(u128::from(right_limb)))
        .strict_add(carry);
      *limb = acc as u64;
      carry = acc >> 64;
    }

    let index = left_index.strict_add(right.as_slice().len());
    let Some(carry_limbs) = product.as_mut_slice().get_mut(index..) else {
      return Err(RsaKeyError::InvalidModulus);
    };
    for limb in carry_limbs {
      let acc = u128::from(*limb).strict_add(carry);
      *limb = acc as u64;
      carry = acc >> 64;
    }
    if carry != 0 {
      return Err(RsaKeyError::InvalidModulus);
    }
  }

  let mut full = vec![0u8; out_limb_count.strict_mul(8)];
  limbs_to_be(product.as_slice(), &mut full);
  let excess = full.len().strict_sub(out.len());
  let Some(prefix) = full.get(..excess) else {
    ct::zeroize(&mut full);
    return Err(RsaKeyError::InvalidModulus);
  };
  if !is_zero_unsigned_be(prefix) {
    ct::zeroize(&mut full);
    return Err(RsaKeyError::InvalidModulus);
  }
  let Some(suffix) = full.get(excess..) else {
    ct::zeroize(&mut full);
    return Err(RsaKeyError::InvalidModulus);
  };
  out.copy_from_slice(suffix);
  ct::zeroize(&mut full);
  Ok(())
}

#[allow(clippy::indexing_slicing)]
fn private_product_add_unsigned_be_to_fixed(
  left: &[u8],
  right: &[u8],
  addend: &[u8],
  out: &mut [u8],
  scratch: &mut RsaPrivateMulScratch,
) -> Result<(), RsaPrivateOpError> {
  if addend.len() > out.len() {
    return Err(RsaPrivateOpError::InvalidLength);
  }

  let left_limb_count = left.len().strict_add(7) / 8;
  let right_limb_count = right.len().strict_add(7) / 8;
  let out_limb_count = out.len().strict_add(7) / 8;
  if left_limb_count.strict_add(right_limb_count) > out_limb_count {
    return Err(RsaPrivateOpError::RepresentativeOutOfRange);
  }
  scratch.ensure_limb_count(out_limb_count)?;

  let left_limbs = scratch
    .left_limbs
    .as_mut_slice()
    .get_mut(..left_limb_count)
    .ok_or(RsaPrivateOpError::InvalidScratch)?;
  let right_limbs = scratch
    .right_limbs
    .as_mut_slice()
    .get_mut(..right_limb_count)
    .ok_or(RsaPrivateOpError::InvalidScratch)?;
  let product = scratch
    .product
    .as_mut_slice()
    .get_mut(..out_limb_count)
    .ok_or(RsaPrivateOpError::InvalidScratch)?;

  limbs_from_be_into(left, left_limbs);
  limbs_from_be_into(right, right_limbs);
  product.fill(0);

  for (left_index, &left_limb) in left_limbs.iter().enumerate() {
    let mut carry = 0u128;
    for (right_index, &right_limb) in right_limbs.iter().enumerate() {
      let index = left_index.strict_add(right_index);
      let limb = product
        .get_mut(index)
        .ok_or(RsaPrivateOpError::RepresentativeOutOfRange)?;
      let acc = u128::from(*limb)
        .strict_add(u128::from(left_limb).strict_mul(u128::from(right_limb)))
        .strict_add(carry);
      *limb = acc as u64;
      carry = acc >> 64;
    }

    let index = left_index.strict_add(right_limbs.len());
    let carry_limbs = product
      .get_mut(index..)
      .ok_or(RsaPrivateOpError::RepresentativeOutOfRange)?;
    for limb in carry_limbs {
      let acc = u128::from(*limb).strict_add(carry);
      *limb = acc as u64;
      carry = acc >> 64;
    }
    if carry != 0 {
      return Err(RsaPrivateOpError::RepresentativeOutOfRange);
    }
  }

  limbs_to_be(product, out);

  let mut carry = 0u16;
  for index in 0..out.len() {
    let dst = out.len().strict_sub(index).strict_sub(1);
    let add_byte = addend
      .len()
      .checked_sub(index.strict_add(1))
      .and_then(|src| addend.get(src))
      .copied()
      .unwrap_or(0);
    let sum = u16::from(out[dst]).strict_add(u16::from(add_byte)).strict_add(carry);
    out[dst] = sum as u8;
    carry = sum >> 8;
  }

  if carry == 0 {
    Ok(())
  } else {
    Err(RsaPrivateOpError::RepresentativeOutOfRange)
  }
}

#[cfg(feature = "getrandom")]
fn private_import_decrement_unsigned_be(bytes: &[u8]) -> Result<SecretBigEndianBuffer, RsaKeyError> {
  let mut out = vec![0u8; bytes.len()];
  private_import_decrement_unsigned_be_to_fixed(bytes, &mut out)?;
  Ok(private_import_canonical_unsigned_be(out))
}

fn private_import_decrement_unsigned_be_to_fixed(bytes: &[u8], out: &mut [u8]) -> Result<(), RsaKeyError> {
  if bytes.is_empty() || bytes.len() != out.len() || is_zero_unsigned_be(bytes) || bytes == [1] {
    return Err(RsaKeyError::InvalidModulus);
  }

  out.copy_from_slice(bytes);
  let mut borrow = 1u8;
  for byte in out.iter_mut().rev() {
    let (difference, overflow) = byte.overflowing_sub(borrow);
    *byte = difference;
    borrow = u8::from(overflow);
  }

  if borrow == 0 {
    Ok(())
  } else {
    Err(RsaKeyError::InvalidModulus)
  }
}

#[allow(clippy::indexing_slicing)]
#[cfg(feature = "getrandom")]
fn private_sub_small_unsigned_be_to_fixed(
  bytes: &[u8],
  decrement: u8,
  out: &mut [u8],
) -> Result<(), RsaPrivateOpError> {
  if bytes.is_empty() || bytes.len() != out.len() || decrement == 0 {
    return Err(RsaPrivateOpError::InvalidLength);
  }

  out.copy_from_slice(bytes);
  let mut borrow = u16::from(decrement);
  for byte in out.iter_mut().rev() {
    let low = (borrow & 0xff) as u8;
    let (difference, overflow) = byte.overflowing_sub(low);
    *byte = difference;
    borrow = (borrow >> 8).strict_add(u16::from(overflow));
  }

  if borrow == 0 {
    Ok(())
  } else {
    Err(RsaPrivateOpError::InvalidLength)
  }
}

fn private_import_unsigned_be_mod(value: &[u8], modulus: &[u8]) -> SecretBigEndianBuffer {
  if is_zero_unsigned_be(modulus) {
    return SecretBigEndianBuffer::new(vec![0]);
  }

  let modulus = SecretLimbs::from_be(modulus);
  let mut remainder = SecretLimbs::zeroed(modulus.as_slice().len());
  for &byte in value {
    for bit in (0..8).rev() {
      double_mod_in_place(remainder.as_mut_slice(), modulus.as_slice());
      add_bit_mod_in_place(remainder.as_mut_slice(), modulus.as_slice(), (byte >> bit) & 1);
    }
  }

  private_import_limbs_to_canonical_be(remainder.as_slice())
}

#[inline(never)]
fn private_import_unsigned_be_mod_to_len(value: &[u8], modulus: &[u8], out: &mut [u8]) -> Result<(), RsaKeyError> {
  if modulus.is_empty() || modulus.len() != out.len() || is_zero_unsigned_be(modulus) {
    return Err(RsaKeyError::InvalidModulus);
  }

  let modulus = SecretLimbs::from_be(modulus);
  let mut remainder = SecretLimbs::zeroed(modulus.as_slice().len());
  for &byte in value {
    for bit in (0..8).rev() {
      double_mod_in_place(remainder.as_mut_slice(), modulus.as_slice());
      add_bit_mod_in_place(remainder.as_mut_slice(), modulus.as_slice(), (byte >> bit) & 1);
    }
  }

  limbs_to_be(remainder.as_slice(), out);
  Ok(())
}

fn private_import_unsigned_be_mod_to_fixed(
  value: &[u8],
  modulus: &RsaPublicModulus,
  out: &mut [u8],
  scratch: &mut RsaPrivateExponentScratch,
) -> Result<(), RsaPrivateOpError> {
  if out.len() != modulus.bytes.len() {
    return Err(RsaPrivateOpError::InvalidLength);
  }
  scratch.ensure_limb_count(modulus.limbs.len())?;

  let remainder = scratch.representative.as_mut_slice();
  remainder.fill(0);
  for &byte in value {
    for bit in (0..8).rev() {
      double_mod_in_place(remainder, &modulus.limbs);
      add_bit_mod_in_place(remainder, &modulus.limbs, (byte >> bit) & 1);
    }
  }

  limbs_to_be(remainder, out);
  Ok(())
}

#[allow(clippy::indexing_slicing)]
fn add_bit_mod_in_place(value: &mut [u64], modulus: &[u64], bit: u8) {
  debug_assert_eq!(value.len(), modulus.len());

  let mut carry = u64::from(bit & 1);
  for limb in value.iter_mut() {
    let (sum, overflow) = limb.overflowing_add(carry);
    *limb = sum;
    carry = u64::from(overflow);
  }
  subtract_modulus_if_needed(value, modulus, carry);
}

fn private_import_limbs_to_canonical_be(limbs: &[u64]) -> SecretBigEndianBuffer {
  let mut bytes = vec![0u8; limbs.len().strict_mul(8)];
  limbs_to_be(limbs, &mut bytes);
  private_import_canonical_unsigned_be(bytes)
}

fn private_import_canonical_unsigned_be(mut bytes: Vec<u8>) -> SecretBigEndianBuffer {
  let first_nonzero = bytes.iter().position(|&byte| byte != 0);
  let canonical = match first_nonzero {
    Some(0) => core::mem::take(&mut bytes),
    Some(index) => bytes.get(index..).map_or_else(|| vec![0], <[u8]>::to_vec),
    None => vec![0],
  };
  ct::zeroize(&mut bytes);
  SecretBigEndianBuffer::new(canonical)
}

#[allow(clippy::indexing_slicing)]
fn cmp_limbs(a: &[u64], b: &[u64]) -> core::cmp::Ordering {
  debug_assert_eq!(a.len(), b.len());
  for index in (0..a.len()).rev() {
    match a[index].cmp(&b[index]) {
      core::cmp::Ordering::Equal => {}
      other => return other,
    }
  }
  core::cmp::Ordering::Equal
}

#[allow(clippy::indexing_slicing)]
fn limb_bit_len(limbs: &[u64]) -> usize {
  for index in (0..limbs.len()).rev() {
    let limb = limbs[index];
    if limb != 0 {
      return index
        .strict_mul(64)
        .strict_add(64usize.strict_sub(limb.leading_zeros() as usize));
    }
  }
  0
}

fn public_montgomery_r2(limbs: &[u64]) -> Box<[u64]> {
  let mut r2 = vec![0u64; limbs.len()];
  pow2_mod_into(&mut r2, limbs.len().strict_mul(128), limbs);
  r2.into_boxed_slice()
}

fn private_montgomery_r2(modulus: &[u8]) -> Result<Box<[u64]>, RsaKeyError> {
  let limb_count = modulus.len().strict_add(7) / 8;
  let mut power = vec![0u8; limb_count.strict_mul(16).strict_add(1)];
  let Some(first) = power.get_mut(0) else {
    return Err(RsaKeyError::InvalidModulus);
  };
  *first = 1;
  let mut reduced = vec![0u8; modulus.len()];
  private_import_unsigned_be_mod_to_len(&power, modulus, &mut reduced)?;
  ct::zeroize(&mut power);
  let limbs = limbs_from_be(&reduced).into_boxed_slice();
  ct::zeroize(&mut reduced);
  Ok(limbs)
}

#[cfg(feature = "diag")]
fn limb_checksum(limbs: &[u64]) -> u64 {
  limbs.iter().copied().fold(0u64, |acc, limb| acc.rotate_left(13) ^ limb)
}

fn ct_nonzero_u64(value: u64) -> u64 {
  (value | value.wrapping_neg()) >> 63
}

#[allow(clippy::indexing_slicing)]
fn sub_modulus_in_place(value: &mut [u64], modulus: &[u64]) -> u64 {
  debug_assert_eq!(value.len(), modulus.len());
  let mut borrow = 0u64;
  for index in 0..value.len() {
    let (tmp, b1) = value[index].overflowing_sub(modulus[index]);
    let (tmp, b2) = tmp.overflowing_sub(borrow);
    value[index] = tmp;
    borrow = u64::from(b1) | u64::from(b2);
  }
  borrow
}

#[allow(clippy::indexing_slicing)]
fn add_modulus_masked(value: &mut [u64], modulus: &[u64], choice: u64) {
  debug_assert_eq!(value.len(), modulus.len());
  let mask = 0u64.wrapping_sub(choice & 1);
  let mut carry = 0u64;
  for index in 0..value.len() {
    let addend = modulus[index] & mask;
    let (sum, overflow) = value[index].overflowing_add(addend);
    let (sum, carry_overflow) = sum.overflowing_add(carry);
    value[index] = sum;
    carry = u64::from(overflow) | u64::from(carry_overflow);
  }
}

#[allow(clippy::indexing_slicing)]
fn subtract_modulus_if_needed(value: &mut [u64], modulus: &[u64], extra: u64) {
  debug_assert_eq!(value.len(), modulus.len());
  let borrow = sub_modulus_in_place(value, modulus);
  let restore = borrow & (ct_nonzero_u64(extra) ^ 1);
  add_modulus_masked(value, modulus, restore);
}

#[cfg(feature = "diag")]
#[allow(clippy::indexing_slicing)]
fn add_mod_in_place(value: &mut [u64], addend: &[u64], modulus: &[u64]) {
  debug_assert_eq!(value.len(), addend.len());
  debug_assert_eq!(value.len(), modulus.len());

  let mut carry = 0u64;
  for index in 0..value.len() {
    let (sum, overflow) = value[index].overflowing_add(addend[index]);
    let (sum, carry_overflow) = sum.overflowing_add(carry);
    value[index] = sum;
    carry = u64::from(overflow) | u64::from(carry_overflow);
  }

  subtract_modulus_if_needed(value, modulus, carry);
}

#[allow(clippy::indexing_slicing)]
fn double_mod_in_place(value: &mut [u64], modulus: &[u64]) {
  debug_assert_eq!(value.len(), modulus.len());
  let mut carry = 0u64;
  for limb in value.iter_mut() {
    let next = *limb >> 63;
    *limb = (*limb << 1) | carry;
    carry = next;
  }
  subtract_modulus_if_needed(value, modulus, carry);
}

#[cfg(feature = "diag")]
fn mul_mod_bitserial(out: &mut [u64], a: &[u64], b: &[u64], modulus: &[u64], addend: &mut [u64]) {
  debug_assert_eq!(out.len(), a.len());
  debug_assert_eq!(out.len(), b.len());
  debug_assert_eq!(out.len(), modulus.len());
  debug_assert_eq!(out.len(), addend.len());

  out.fill(0);
  copy_limbs(addend, a);

  for &limb in b {
    for bit in 0..64 {
      if (limb >> bit) & 1 == 1 {
        add_mod_in_place(out, addend, modulus);
      }
      double_mod_in_place(addend, modulus);
    }
  }
}

#[allow(clippy::indexing_slicing)]
fn pow2_mod_into(out: &mut [u64], bits: usize, modulus: &[u64]) {
  out.fill(0);
  let modulus_bits = limb_bit_len(modulus);
  let direct_bits = core::cmp::min(bits, modulus_bits.saturating_sub(1));
  out[direct_bits / 64] = 1u64 << (direct_bits % 64);
  for _ in direct_bits..bits {
    double_mod_in_place(out, modulus);
  }
}

#[must_use]
fn montgomery_n0(n0: u64) -> u64 {
  debug_assert_eq!(n0 & 1, 1);
  let mut inv = 1u64;
  for _ in 0..6 {
    inv = inv.wrapping_mul(2u64.wrapping_sub(n0.wrapping_mul(inv)));
  }
  inv.wrapping_neg()
}

#[allow(clippy::indexing_slicing)]
fn mont_square_in_place(value: &mut [u64], tmp: &mut [u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  let _ = tmp;
  mont_square_product(value, modulus, t);
}

#[allow(clippy::indexing_slicing)]
fn mont_mul_in_place_left(left: &mut [u64], right: &[u64], tmp: &mut [u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  copy_limbs(tmp, left);
  mont_mul(left, tmp, right, modulus, t);
}

#[cfg(feature = "diag")]
#[allow(clippy::indexing_slicing)]
fn mont_square_comba_in_place(value: &mut [u64], tmp: &mut [u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  copy_limbs(tmp, value);
  mont_mul_comba(value, tmp, tmp, modulus, t);
}

#[cfg(feature = "diag")]
#[allow(clippy::indexing_slicing)]
fn mont_mul_comba_in_place_left(
  left: &mut [u64],
  right: &[u64],
  tmp: &mut [u64],
  modulus: &RsaPublicModulus,
  t: &mut [u64],
) {
  copy_limbs(tmp, left);
  mont_mul_comba(left, tmp, right, modulus, t);
}

#[allow(clippy::indexing_slicing)]
fn mont_square_cios_in_place(value: &mut [u64], tmp: &mut [u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  #[cfg(all(
    target_arch = "aarch64",
    target_os = "macos",
    not(feature = "portable-only"),
    not(miri)
  ))]
  if value.len() == modulus.limbs.len()
    && rsa_aarch64_asm::supports_bignum_mont_words(modulus.limbs.len())
    && t.len() >= rsa_aarch64_asm::bignum_mont_scratch_words(modulus.limbs.len())
  {
    rsa_aarch64_asm::mont_square_cios_words_in_place(value, &modulus.limbs, modulus.n0, modulus.limbs.len(), t);
    return;
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_os = "linux",
    not(feature = "portable-only"),
    not(miri)
  ))]
  if value.len() == modulus.limbs.len()
    && rsa_x86_64_asm::supports_bignum_mont_square_words(modulus.limbs.len())
    && t.len() >= rsa_x86_64_asm::bignum_mont_scratch_words(modulus.limbs.len())
  {
    rsa_x86_64_asm::mont_square_cios_words_in_place(value, &modulus.limbs, modulus.n0, modulus.limbs.len(), t);
    return;
  }

  copy_limbs(tmp, value);
  mont_mul_cios(value, tmp, tmp, modulus, t);
}

#[allow(clippy::indexing_slicing)]
fn mont_mul_cios_in_place_left(
  left: &mut [u64],
  right: &[u64],
  tmp: &mut [u64],
  modulus: &RsaPublicModulus,
  t: &mut [u64],
) {
  #[cfg(all(
    target_arch = "aarch64",
    target_os = "macos",
    not(feature = "portable-only"),
    not(miri)
  ))]
  if left.len() == modulus.limbs.len()
    && right.len() == modulus.limbs.len()
    && rsa_aarch64_asm::supports_bignum_mont_words(modulus.limbs.len())
    && t.len() >= rsa_aarch64_asm::bignum_mont_scratch_words(modulus.limbs.len())
  {
    rsa_aarch64_asm::mont_mul_cios_words_in_place_left(left, right, &modulus.limbs, modulus.n0, modulus.limbs.len(), t);
    return;
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_os = "linux",
    not(feature = "portable-only"),
    not(miri)
  ))]
  if left.len() == modulus.limbs.len()
    && right.len() == modulus.limbs.len()
    && rsa_x86_64_asm::supports_bignum_mont_words(modulus.limbs.len())
    && t.len() >= rsa_x86_64_asm::bignum_mont_scratch_words(modulus.limbs.len())
  {
    rsa_x86_64_asm::mont_mul_cios_words_in_place_left(left, right, &modulus.limbs, modulus.n0, modulus.limbs.len(), t);
    return;
  }

  copy_limbs(tmp, left);
  mont_mul_cios(left, tmp, right, modulus, t);
}

#[inline]
#[must_use]
fn use_cios_montgomery(modulus: &RsaPublicModulus) -> bool {
  let limbs = modulus.limbs.len();
  if limbs <= 64 {
    return true;
  }

  #[cfg(all(
    target_arch = "aarch64",
    target_os = "macos",
    not(feature = "portable-only"),
    not(miri)
  ))]
  if limbs == 128 && rsa_aarch64_asm::supports_bignum_mont_words(limbs) {
    return true;
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_os = "linux",
    not(feature = "portable-only"),
    not(miri)
  ))]
  if limbs == 128 && rsa_x86_64_asm::supports_bignum_mont_words(limbs) {
    return true;
  }

  false
}

#[allow(clippy::indexing_slicing)]
#[cfg(feature = "diag")]
fn mont_square_auto_in_place(value: &mut [u64], tmp: &mut [u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  if use_cios_montgomery(modulus) {
    mont_square_cios_in_place(value, tmp, modulus, t);
  } else {
    mont_square_in_place(value, tmp, modulus, t);
  }
}

#[allow(clippy::indexing_slicing)]
#[cfg(feature = "diag")]
fn mont_mul_auto_in_place_left(
  left: &mut [u64],
  right: &[u64],
  tmp: &mut [u64],
  modulus: &RsaPublicModulus,
  t: &mut [u64],
) {
  if use_cios_montgomery(modulus) {
    mont_mul_cios_in_place_left(left, right, tmp, modulus, t);
  } else {
    mont_mul_in_place_left(left, right, tmp, modulus, t);
  }
}

#[cfg(feature = "diag")]
fn mont_mul_auto(out: &mut [u64], a: &[u64], b: &[u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  if use_cios_montgomery(modulus) {
    mont_mul_cios(out, a, b, modulus, t);
  } else {
    mont_mul(out, a, b, modulus, t);
  }
}

#[cfg(feature = "diag")]
fn mont_reduce_auto(out: &mut [u64], value: &[u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  if use_cios_montgomery(modulus) {
    mont_reduce_cios(out, value, modulus, t);
  } else {
    mont_reduce(out, value, modulus, t);
  }
}

#[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
fn mont_mul_cios(out: &mut [u64], a: &[u64], b: &[u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  let n = modulus.limbs.len();
  debug_assert_eq!(out.len(), n);
  debug_assert_eq!(a.len(), n);
  debug_assert_eq!(b.len(), n);
  debug_assert!(t.len() >= n.strict_add(2));

  #[cfg(all(
    target_arch = "aarch64",
    target_os = "macos",
    not(feature = "portable-only"),
    not(miri)
  ))]
  if rsa_aarch64_asm::supports_bignum_mont_words(n) && t.len() >= rsa_aarch64_asm::bignum_mont_scratch_words(n) {
    rsa_aarch64_asm::mont_mul_cios_words(out, a, b, &modulus.limbs, modulus.n0, n, t);
    return;
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_os = "linux",
    not(feature = "portable-only"),
    not(miri)
  ))]
  if rsa_x86_64_asm::supports_bignum_mont_words(n) && t.len() >= rsa_x86_64_asm::bignum_mont_scratch_words(n) {
    rsa_x86_64_asm::mont_mul_cios_words(out, a, b, &modulus.limbs, modulus.n0, n, t);
    return;
  }

  t[..n.strict_add(2)].fill(0);

  for i in 0..n {
    let mut carry = 0u64;
    for j in 0..n {
      let acc = u128::from(t[j])
        .strict_add(u128::from(a[j]).strict_mul(u128::from(b[i])))
        .strict_add(u128::from(carry));
      t[j] = acc as u64;
      carry = (acc >> 64) as u64;
    }
    let (sum, overflow) = t[n].overflowing_add(carry);
    t[n] = sum;
    t[n.strict_add(1)] = u64::from(overflow);

    let q = t[0].wrapping_mul(modulus.n0);
    carry = 0;
    for j in 0..n {
      let acc = u128::from(t[j])
        .strict_add(u128::from(q).strict_mul(u128::from(modulus.limbs[j])))
        .strict_add(u128::from(carry));
      t[j] = acc as u64;
      carry = (acc >> 64) as u64;
    }
    let (sum, overflow) = t[n].overflowing_add(carry);
    t[n] = sum;
    t[n.strict_add(1)] = t[n.strict_add(1)].strict_add(u64::from(overflow));

    for j in 0..=n {
      t[j] = t[j.strict_add(1)];
    }
    t[n.strict_add(1)] = 0;
  }

  for (dst, src) in out.iter_mut().zip(t[..n].iter().copied()) {
    *dst = src;
  }

  subtract_modulus_if_needed(out, &modulus.limbs, t[n]);
}

#[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
fn mont_reduce_cios(out: &mut [u64], value: &[u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  let n = modulus.limbs.len();
  debug_assert_eq!(out.len(), n);
  debug_assert_eq!(value.len(), n);
  debug_assert!(t.len() >= n.strict_add(2));

  #[cfg(all(
    target_arch = "aarch64",
    target_os = "macos",
    not(feature = "portable-only"),
    not(miri)
  ))]
  if n == 32 && t.len() >= 66 {
    rsa_aarch64_asm::mont_reduce_cios_32(out, value, &modulus.limbs, modulus.n0, t);
    return;
  }

  #[cfg(all(
    target_arch = "aarch64",
    target_os = "macos",
    not(feature = "portable-only"),
    not(miri)
  ))]
  if matches!(n, 48 | 64) && t.len() >= rsa_aarch64_asm::bignum_mont_scratch_words(n) {
    rsa_aarch64_asm::mont_reduce_cios_words(out, value, &modulus.limbs, modulus.n0, n, t);
    return;
  }

  t[..n.strict_add(2)].fill(0);
  t[..n].copy_from_slice(value);

  for _ in 0..n {
    let q = t[0].wrapping_mul(modulus.n0);
    let mut carry = 0u64;
    for j in 0..n {
      let acc = u128::from(t[j])
        .strict_add(u128::from(q).strict_mul(u128::from(modulus.limbs[j])))
        .strict_add(u128::from(carry));
      t[j] = acc as u64;
      carry = (acc >> 64) as u64;
    }
    let (sum, overflow) = t[n].overflowing_add(carry);
    t[n] = sum;
    t[n.strict_add(1)] = t[n.strict_add(1)].strict_add(u64::from(overflow));

    for j in 0..=n {
      t[j] = t[j.strict_add(1)];
    }
    t[n.strict_add(1)] = 0;
  }

  for (dst, src) in out.iter_mut().zip(t[..n].iter().copied()) {
    *dst = src;
  }

  subtract_modulus_if_needed(out, &modulus.limbs, t[n]);
}

#[cfg(feature = "diag")]
#[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
fn mont_mul_comba(out: &mut [u64], a: &[u64], b: &[u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  let n = modulus.limbs.len();
  debug_assert_eq!(out.len(), n);
  debug_assert_eq!(a.len(), n);
  debug_assert_eq!(b.len(), n);
  debug_assert!(t.len() >= n.strict_mul(2).strict_add(2));

  comba_mul_into(t, a, b);

  for i in 0..n {
    let q = t[i].wrapping_mul(modulus.n0);
    let mut carry = 0u128;
    for j in 0..n {
      let index = i.strict_add(j);
      let acc = u128::from(q)
        .strict_mul(u128::from(modulus.limbs[j]))
        .strict_add(u128::from(t[index]))
        .strict_add(carry);
      t[index] = acc as u64;
      carry = acc >> 64;
    }
    add_carry(t, i.strict_add(n), carry as u64);
  }

  for (dst, src) in out.iter_mut().zip(t[n..n.strict_add(n)].iter().copied()) {
    *dst = src;
  }

  let extra = t[n.strict_add(n)..].iter().copied().fold(0u64, |acc, limb| acc | limb);
  subtract_modulus_if_needed(out, &modulus.limbs, extra);
}

#[cfg(feature = "diag")]
#[allow(clippy::indexing_slicing)]
fn comba_mul_into(out: &mut [u64], a: &[u64], b: &[u64]) {
  debug_assert_eq!(a.len(), b.len());
  let n = a.len();
  debug_assert!(out.len() >= n.strict_mul(2).strict_add(2));

  out.fill(0);
  let mut carry_lo = 0u64;
  let mut carry_hi = 0u64;
  let product_limbs = n.strict_mul(2);

  for column in 0..product_limbs.strict_sub(1) {
    let mut acc_lo = carry_lo;
    let mut acc_mid = carry_hi;
    let mut acc_hi = 0u64;
    let min_i = column.saturating_sub(n.strict_sub(1));
    let max_i = column.min(n.strict_sub(1));

    for i in min_i..=max_i {
      add_product_to_acc(&mut acc_lo, &mut acc_mid, &mut acc_hi, a[i], b[column.strict_sub(i)]);
    }

    out[column] = acc_lo;
    carry_lo = acc_mid;
    carry_hi = acc_hi;
  }

  out[product_limbs.strict_sub(1)] = carry_lo;
  out[product_limbs] = carry_hi;
}

fn add_product_to_acc(acc_lo: &mut u64, acc_mid: &mut u64, acc_hi: &mut u64, a: u64, b: u64) {
  let product = u128::from(a).strict_mul(u128::from(b));
  let product_lo = product as u64;
  let product_hi = (product >> 64) as u64;

  let (lo, lo_overflow) = acc_lo.overflowing_add(product_lo);
  *acc_lo = lo;
  let (mid, product_hi_overflow) = acc_mid.overflowing_add(product_hi);
  let (mid, lo_carry_overflow) = mid.overflowing_add(u64::from(lo_overflow));
  *acc_mid = mid;
  *acc_hi = acc_hi.strict_add(u64::from(product_hi_overflow).strict_add(u64::from(lo_carry_overflow)));
}

#[allow(clippy::indexing_slicing)]
fn square_into_wide_product(out: &mut [u64], value: &[u64]) {
  let n = value.len();
  debug_assert!(out.len() >= n.strict_mul(2).strict_add(2));

  out.fill(0);
  let mut carry_lo = 0u64;
  let mut carry_hi = 0u64;
  let product_limbs = n.strict_mul(2);

  let mut column = 0usize;
  while column < product_limbs.strict_sub(1) {
    let mut acc_lo = carry_lo;
    let mut acc_mid = carry_hi;
    let mut acc_hi = 0u64;
    let min_i = column.saturating_sub(n.strict_sub(1));
    let max_i = column.min(n.strict_sub(1));

    for i in min_i..=max_i {
      let j = column.strict_sub(i);
      if i > j {
        break;
      }
      add_product_to_acc(&mut acc_lo, &mut acc_mid, &mut acc_hi, value[i], value[j]);
      if i != j {
        add_product_to_acc(&mut acc_lo, &mut acc_mid, &mut acc_hi, value[i], value[j]);
      }
    }

    out[column] = acc_lo;
    carry_lo = acc_mid;
    carry_hi = acc_hi;
    column = column.strict_add(1);
  }

  out[product_limbs.strict_sub(1)] = carry_lo;
  out[product_limbs] = carry_hi;
}

#[allow(clippy::indexing_slicing)]
fn mont_square_product(out: &mut [u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  let n = modulus.limbs.len();
  debug_assert_eq!(out.len(), n);
  debug_assert!(t.len() >= n.strict_mul(2).strict_add(2));

  square_into_wide_product(t, out);

  for i in 0..n {
    let q = t[i].wrapping_mul(modulus.n0);
    let mut carry = 0u128;
    for j in 0..n {
      let index = i.strict_add(j);
      let acc = u128::from(q) * u128::from(modulus.limbs[j]) + u128::from(t[index]) + carry;
      t[index] = acc as u64;
      carry = acc >> 64;
    }
    add_carry(t, i.strict_add(n), carry as u64);
  }

  for (dst, src) in out.iter_mut().zip(t[n..n.strict_add(n)].iter().copied()) {
    *dst = src;
  }

  let extra = t[n.strict_add(n)..].iter().copied().fold(0u64, |acc, limb| acc | limb);
  subtract_modulus_if_needed(out, &modulus.limbs, extra);
}

#[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
fn mont_mul(out: &mut [u64], a: &[u64], b: &[u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  let n = modulus.limbs.len();
  debug_assert_eq!(out.len(), n);
  debug_assert_eq!(a.len(), n);
  debug_assert_eq!(b.len(), n);
  debug_assert!(t.len() >= n.strict_mul(2).strict_add(2));

  t.fill(0);

  for i in 0..n {
    let mut carry = 0u128;
    for j in 0..n {
      let index = i.strict_add(j);
      let acc = u128::from(a[j])
        .strict_mul(u128::from(b[i]))
        .strict_add(u128::from(t[index]))
        .strict_add(carry);
      t[index] = acc as u64;
      carry = acc >> 64;
    }
    add_carry(t, i.strict_add(n), carry as u64);
  }

  for i in 0..n {
    let q = t[i].wrapping_mul(modulus.n0);
    let mut carry = 0u128;
    for j in 0..n {
      let index = i.strict_add(j);
      let acc = u128::from(q)
        .strict_mul(u128::from(modulus.limbs[j]))
        .strict_add(u128::from(t[index]))
        .strict_add(carry);
      t[index] = acc as u64;
      carry = acc >> 64;
    }
    add_carry(t, i.strict_add(n), carry as u64);
  }

  for (dst, src) in out.iter_mut().zip(t[n..n.strict_add(n)].iter().copied()) {
    *dst = src;
  }

  let extra = t[n.strict_add(n)..].iter().copied().fold(0u64, |acc, limb| acc | limb);
  subtract_modulus_if_needed(out, &modulus.limbs, extra);
}

#[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
fn mont_reduce(out: &mut [u64], value: &[u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  let n = modulus.limbs.len();
  debug_assert_eq!(out.len(), n);
  debug_assert_eq!(value.len(), n);
  debug_assert!(t.len() >= n.strict_mul(2).strict_add(2));

  t.fill(0);
  t[..n].copy_from_slice(value);

  for i in 0..n {
    let q = t[i].wrapping_mul(modulus.n0);
    let mut carry = 0u128;
    for j in 0..n {
      let index = i.strict_add(j);
      let acc = u128::from(q)
        .strict_mul(u128::from(modulus.limbs[j]))
        .strict_add(u128::from(t[index]))
        .strict_add(carry);
      t[index] = acc as u64;
      carry = acc >> 64;
    }
    add_carry(t, i.strict_add(n), carry as u64);
  }

  for (dst, src) in out.iter_mut().zip(t[n..n.strict_add(n)].iter().copied()) {
    *dst = src;
  }

  let extra = t[n.strict_add(n)..].iter().copied().fold(0u64, |acc, limb| acc | limb);
  subtract_modulus_if_needed(out, &modulus.limbs, extra);
}

#[allow(clippy::indexing_slicing)]
fn add_carry(t: &mut [u64], index: usize, mut carry: u64) {
  for limb in &mut t[index..] {
    let (sum, overflow) = limb.overflowing_add(carry);
    *limb = sum;
    carry = u64::from(overflow);
  }
}

struct DerReader<'a> {
  input: &'a [u8],
  offset: usize,
}

impl<'a> DerReader<'a> {
  #[inline]
  #[must_use]
  const fn new(input: &'a [u8]) -> Self {
    Self { input, offset: 0 }
  }

  fn peek_byte(&self) -> Option<u8> {
    self.input.get(self.offset).copied()
  }

  fn read_constructed(&mut self, tag: u8) -> Result<&'a [u8], RsaKeyError> {
    self.read_primitive(tag)
  }

  fn read_tlv(&mut self, tag: u8) -> Result<&'a [u8], RsaKeyError> {
    let start = self.offset;
    let _ = self.read_primitive(tag)?;
    self.input.get(start..self.offset).ok_or(RsaKeyError::MalformedDer)
  }

  fn read_primitive(&mut self, tag: u8) -> Result<&'a [u8], RsaKeyError> {
    let actual = self.read_byte()?;
    if actual != tag {
      return Err(RsaKeyError::MalformedDer);
    }

    let len = self.read_len()?;
    let end = self.offset.checked_add(len).ok_or(RsaKeyError::MalformedDer)?;
    if end > self.input.len() {
      return Err(RsaKeyError::MalformedDer);
    }

    let value = self.input.get(self.offset..end).ok_or(RsaKeyError::MalformedDer)?;
    self.offset = end;
    Ok(value)
  }

  fn finish(&self) -> Result<(), RsaKeyError> {
    if self.offset == self.input.len() {
      Ok(())
    } else {
      Err(RsaKeyError::MalformedDer)
    }
  }

  fn read_byte(&mut self) -> Result<u8, RsaKeyError> {
    let byte = *self.input.get(self.offset).ok_or(RsaKeyError::MalformedDer)?;
    self.offset = self.offset.strict_add(1);
    Ok(byte)
  }

  fn read_len(&mut self) -> Result<usize, RsaKeyError> {
    let first = self.read_byte()?;
    if first & 0x80 == 0 {
      return Ok(usize::from(first));
    }

    let len_len = usize::from(first & 0x7f);
    if len_len == 0 || len_len > core::mem::size_of::<usize>() {
      return Err(RsaKeyError::MalformedDer);
    }

    let first_len_byte = self.read_byte()?;
    if first_len_byte == 0 {
      return Err(RsaKeyError::MalformedDer);
    }

    let mut len = usize::from(first_len_byte);
    for _ in 1..len_len {
      len = len.checked_shl(8).ok_or(RsaKeyError::MalformedDer)?;
      len |= usize::from(self.read_byte()?);
    }

    if len < 128 {
      return Err(RsaKeyError::MalformedDer);
    }

    Ok(len)
  }
}

#[cfg(test)]
mod tests {
  use alloc::format;

  use proptest::prelude::*;
  #[cfg(feature = "getrandom")]
  use serde_json::Value;

  use super::*;

  #[cfg(feature = "getrandom")]
  const CAVP_KEYGEN_186_3_PROBABLE_PRIME: &str =
    include_str!("../../testdata/rsa/nist_cavp/rsa_keygen_186_3_probable_prime_subset.json");

  fn hex_to_vec(hex: &str) -> Vec<u8> {
    assert_eq!(hex.len() % 2, 0);
    let mut out = Vec::with_capacity(hex.len() / 2);
    for chunk in hex.as_bytes().chunks_exact(2) {
      out.push((hex_value(chunk[0]) << 4) | hex_value(chunk[1]));
    }
    out
  }

  const fn hex_value(byte: u8) -> u8 {
    match byte {
      b'0'..=b'9' => byte - b'0',
      b'a'..=b'f' => byte - b'a' + 10,
      b'A'..=b'F' => byte - b'A' + 10,
      _ => panic!("invalid hex digit"),
    }
  }

  #[cfg(feature = "getrandom")]
  fn json_field<'a>(value: &'a Value, name: &'static str) -> &'a str {
    value[name]
      .as_str()
      .unwrap_or_else(|| panic!("missing string field `{name}`"))
  }

  #[cfg(feature = "getrandom")]
  fn test_keygen_drbg(label: &'static [u8]) -> RsaKeygenDrbg {
    let mut seed = [0u8; RSA_KEYGEN_DRBG_ENTROPY_BYTES + RSA_KEYGEN_DRBG_NONCE_BYTES];
    for (index, byte) in seed.iter_mut().enumerate() {
      *byte = (index as u8).wrapping_mul(17).wrapping_add(0xa5);
    }
    RsaKeygenDrbg::new(&seed, label)
  }

  fn integer_unsigned(value: &[u8]) -> Vec<u8> {
    let first_nonzero = value.iter().position(|&byte| byte != 0);
    let value = first_nonzero.map_or(&[0u8][..], |index| &value[index..]);
    let mut encoded = Vec::with_capacity(value.len() + usize::from(value[0] & 0x80 != 0));
    if value[0] & 0x80 != 0 {
      encoded.push(0);
    }
    encoded.extend_from_slice(value);
    tlv(TAG_INTEGER, &encoded)
  }

  fn integer_der_value(value: &[u8]) -> Vec<u8> {
    tlv(TAG_INTEGER, value)
  }

  fn test_pkcs1_public_key(modulus: &[u8], exponent: &[u8]) -> Vec<u8> {
    let modulus = integer_unsigned(modulus);
    let exponent = integer_unsigned(exponent);
    let mut body = Vec::with_capacity(modulus.len().strict_add(exponent.len()));
    body.extend_from_slice(&modulus);
    body.extend_from_slice(&exponent);
    tlv(TAG_SEQUENCE, &body)
  }

  fn algorithm_identifier(algorithm_oid: &[u8], params: Option<&[u8]>) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&tlv(TAG_OBJECT_IDENTIFIER, algorithm_oid));
    if let Some(params) = params {
      body.extend_from_slice(params);
    }
    tlv(TAG_SEQUENCE, &body)
  }

  fn null() -> Vec<u8> {
    tlv(TAG_NULL, &[])
  }

  fn context_constructed(index: u8, value: &[u8]) -> Vec<u8> {
    tlv(0xa0 | index, value)
  }

  fn x509_hash_algorithm(profile: RsaPssProfile) -> Vec<u8> {
    let oid = match profile {
      RsaPssProfile::Sha256 => ID_SHA256_OID,
      RsaPssProfile::Sha384 => ID_SHA384_OID,
      RsaPssProfile::Sha512 => ID_SHA512_OID,
    };
    algorithm_identifier(oid, Some(&null()))
  }

  fn x509_mgf1_algorithm(profile: RsaPssProfile) -> Vec<u8> {
    let hash = x509_hash_algorithm(profile);
    algorithm_identifier(ID_MGF1_OID, Some(&hash))
  }

  fn x509_pss_algorithm(profile: RsaPssProfile, salt_len: usize) -> Vec<u8> {
    let mut params = Vec::new();
    params.extend_from_slice(&context_constructed(0, &x509_hash_algorithm(profile)));
    params.extend_from_slice(&context_constructed(1, &x509_mgf1_algorithm(profile)));
    params.extend_from_slice(&context_constructed(
      2,
      &integer_unsigned(&u64::try_from(salt_len).unwrap().to_be_bytes()),
    ));
    algorithm_identifier(ID_RSASSA_PSS_OID, Some(&tlv(TAG_SEQUENCE, &params)))
  }

  fn rsa_private_modulus() -> Vec<u8> {
    hex_to_vec(
      "\
d397b84d98a4c26138ed1b695a8106ead91d553bf06041b62d3fdc50a041e222b8f4529689c1b82c5e71554f5d\
d69fa2f4b6158cf0dbeb57811a0fc327e1f28e74fe74d3bc166c1eabdc1b8b57b934ca8be5b00b4f29975bcc\
99acaf415b59bb28a6782bb41a2c3c2976b3c18dbadef62f00c6bb226640095096c0cc60d22fe7ef987d75c\
6a81b10d96bf292028af110dc7cc1bbc43d22adab379a0cd5d8078cc780ff5cd6209dea34c922cf784f7717e\
428d75b5aec8ff30e5f0141510766e2e0ab8d473c84e8710b2b98227c3db095337ad3452f19e2b9bfbccdd8\
148abf6776fa552775e6e75956e45229ae5a9c46949bab1e622f0e48f56524a84ed3483b",
    )
  }

  fn rsa_private_exponent() -> Vec<u8> {
    hex_to_vec(
      "\
c4e70c689162c94c660828191b52b4d8392115df486a9adbe831e458d73958320dc1b755456e93701e9702d76\
fb0b92f90e01d1fe248153281fe79aa9763a92fae69d8d7ecd144de29fa135bd14f9573e349e45031e3b76982\
f583003826c552e89a397c1a06bd2163488630d92e8c2bb643d7abef700da95d685c941489a46f54b5316f62\
b5d2c3a7f1bbd134cb37353a44683fdc9d95d36458de22f6c44057fe74a0a436c4308f73f4da42f35c47ac1\
6a7138d483afc91e41dc3a1127382e0c0f5119b0221b4fc639d6b9c38177a6de9b526ebd88c38d7982c07f98\
a0efd877d508aae275b946915c02e2e1106d175d74ec6777f5e80d12c053d9c7be1e341",
    )
  }

  fn rsa_private_prime_p() -> Vec<u8> {
    hex_to_vec(
      "\
f827bbf3a41877c7cc59aebf42ed4b29c32defcb8ed96863d5b090a05a8930dd624a21c9dcf9838568fdfa0d\
f65b8462a5f2ac913d6c56f975532bd8e78fb07bd405ca99a484bcf59f019bbddcb3933f2bce706300b4f7b\
110120c5df9018159067c35da3061a56c8635a52b54273b31271b4311f0795df6021e6355e1a42e61",
    )
  }

  fn rsa_private_prime_q() -> Vec<u8> {
    hex_to_vec(
      "\
da4817ce0089dd36f2ade6a3ff410c73ec34bf1b4f6bda38431bfede11cef1f7f6efa70e5f8063a3b1f6e172\
96ffb15feefa0912a0325b8d1fd65a559e717b5b961ec345072e0ec5203d03441d29af4d64054a04507410cf\
1da78e7b6119d909ec66e6ad625bf995b279a4b3c5be7d895cd7c5b9c4c497fde730916fcdb4e41b",
    )
  }

  fn rsa_private_exponent_p() -> Vec<u8> {
    hex_to_vec(
      "\
1da6e9cf80212856e87522eb59bcef094b7836ba1514a7639e8a1d8dfba37f0245176498315e6337d2c6de554\
2c5c6b8dee973735b6a91adf735fbfc4c1720587b8a419e40495826e55c14d70803312a103af7b4ecc5b2ff26\
5371c4dcd730348a10d7827ddb7d1fcd9da561db09610a4b88f767b25b5e3de21ced73baa59aa1",
    )
  }

  fn rsa_private_exponent_q() -> Vec<u8> {
    hex_to_vec(
      "\
d737a7c8e43d0a10c85bf0011886a16996a6371b0d46b0c5325de3003f9cc47491539f6a0b7d82407f12851c\
bf86e1f34da3d7d8367d104967efa7e7ad2e04cbbb8b1f4aeb165d57bd3e8afed8a62602ef304bd74f1ff106\
d51d44dd9f52a5ed23da1d6d2c82b4e6052fecd5978e0726ad94cd8e295510eb35cc6c49491026ab",
    )
  }

  fn rsa_private_coefficient() -> Vec<u8> {
    hex_to_vec(
      "\
5268d7cf073479aebb2d2ed4dd66b8c89915b52d141e0c4932f56b0c0ed0936141894ec4d27d53bc86453cd8\
ca5b455045218c7e196209c1c651702ece090a15e3cbcc265971300023a86fe9d34ad527e9ef03b7adfe736e\
0680747abfd49839b82f2ffdec43bd0343ca30e13961b32af6cdeddd195672c76b53b76fc3ea76f8",
    )
  }

  fn test_pkcs1_private_key() -> Vec<u8> {
    test_pkcs1_private_key_with_public_exponent(&[0x01, 0x00, 0x01])
  }

  fn test_pkcs1_private_key_with_public_exponent(public_exponent: &[u8]) -> Vec<u8> {
    test_pkcs1_private_key_with_crt(
      public_exponent,
      &rsa_private_exponent_p(),
      &rsa_private_exponent_q(),
      &rsa_private_coefficient(),
    )
  }

  fn test_pkcs1_private_key_with_crt(
    public_exponent: &[u8],
    exponent_p: &[u8],
    exponent_q: &[u8],
    coefficient: &[u8],
  ) -> Vec<u8> {
    test_pkcs1_private_key_with_components(
      public_exponent,
      &rsa_private_exponent(),
      &rsa_private_prime_p(),
      &rsa_private_prime_q(),
      exponent_p,
      exponent_q,
      coefficient,
    )
  }

  fn test_pkcs1_private_key_with_components(
    public_exponent: &[u8],
    private_exponent: &[u8],
    prime_p: &[u8],
    prime_q: &[u8],
    exponent_p: &[u8],
    exponent_q: &[u8],
    coefficient: &[u8],
  ) -> Vec<u8> {
    let fields = [
      integer_unsigned(&[0]),
      integer_unsigned(&rsa_private_modulus()),
      integer_unsigned(public_exponent),
      integer_unsigned(private_exponent),
      integer_unsigned(prime_p),
      integer_unsigned(prime_q),
      integer_unsigned(exponent_p),
      integer_unsigned(exponent_q),
      integer_unsigned(coefficient),
    ];
    let len = fields.iter().map(Vec::len).sum();
    let mut body = Vec::with_capacity(len);
    for field in fields {
      body.extend_from_slice(&field);
    }
    tlv(TAG_SEQUENCE, &body)
  }

  fn test_pkcs8_private_key(pkcs1: &[u8], algorithm: &[u8]) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&integer_unsigned(&[0]));
    body.extend_from_slice(algorithm);
    body.extend_from_slice(&tlv(TAG_OCTET_STRING, pkcs1));
    tlv(TAG_SEQUENCE, &body)
  }

  fn masked_oaep_sha256_from_decoded_db(seed: &[u8; Sha256::OUTPUT_SIZE], decoded_db: &[u8]) -> Vec<u8> {
    let mut masked_db = decoded_db.to_vec();
    let mut db_mask = vec![0u8; masked_db.len()];
    mgf1::<Sha256>(seed, &mut db_mask);
    for (byte, mask) in masked_db.iter_mut().zip(db_mask.iter().copied()) {
      *byte ^= mask;
    }

    let mut masked_seed = seed.to_vec();
    let mut seed_mask = vec![0u8; Sha256::OUTPUT_SIZE];
    mgf1::<Sha256>(&masked_db, &mut seed_mask);
    for (byte, mask) in masked_seed.iter_mut().zip(seed_mask.iter().copied()) {
      *byte ^= mask;
    }

    let mut encoded = Vec::with_capacity(1usize.strict_add(masked_seed.len()).strict_add(masked_db.len()));
    encoded.push(0);
    encoded.extend_from_slice(&masked_seed);
    encoded.extend_from_slice(&masked_db);
    encoded
  }

  fn decoded_oaep_sha256_db(label: &[u8], message: &[u8], ps_len: usize) -> Vec<u8> {
    let label_hash = Sha256::digest(label);
    let mut db = Vec::with_capacity(
      Sha256::OUTPUT_SIZE
        .strict_add(ps_len)
        .strict_add(1)
        .strict_add(message.len()),
    );
    db.extend_from_slice(label_hash.as_ref());
    db.resize(Sha256::OUTPUT_SIZE.strict_add(ps_len), 0);
    db.push(1);
    db.extend_from_slice(message);
    db
  }

  fn x509_certificate_fixture_public_key() -> Vec<u8> {
    hex_to_vec(
      "\
30820122300d06092a864886f70d01010105000382010f003082010a0282010100a246ccf6bd59720287837151de9fa5\
5d4a811e456643f7fd0ced5a9ffa8fe52a89d52a8f6bd96246c9f0d23cd4f215609bfd0fd09dfcf13305440cae6e1b9a\
3c48e8e360438ca9993c1cd8ec03363cc3d79edbc4df7764c7f8ddb75f1148037847b356d2697f7d0158072a2e4f38f9\
40c8db08b70305dedb6fe97aeb530dccc009274f7864442f6f02cf6191b5a32268234bcbd7827bf3e570206c0cddf147\
df5169ceda6883b2169768878fd5b107a092ab7482d8ba7f46364b566aaa72153068b6a0174f2f5e0e5f9bcd0213dd4e\
8689d56ffa0be918a16fffcbe4830157eb8535c1a2a50636f8fc8a57f9ae0488b91159456ca94d7e64a1286babad3e92\
f70203010001",
    )
  }

  fn x509_pkcs1v15_certificate_fixture() -> Vec<u8> {
    hex_to_vec(
      "\
3082031b30820203a00302010202144abda3eea77ef52d888f7ab507cd9016cacc900f300d06092a864886f70d01010b\
0500301d311b301906035504030c12727363727970746f2d7273612d706b637331301e170d3236303532323232333332\
305a170d3236303632313232333332305a301d311b301906035504030c12727363727970746f2d7273612d706b637331\
30820122300d06092a864886f70d01010105000382010f003082010a0282010100a246ccf6bd59720287837151de9fa5\
5d4a811e456643f7fd0ced5a9ffa8fe52a89d52a8f6bd96246c9f0d23cd4f215609bfd0fd09dfcf13305440cae6e1b9a\
3c48e8e360438ca9993c1cd8ec03363cc3d79edbc4df7764c7f8ddb75f1148037847b356d2697f7d0158072a2e4f38f9\
40c8db08b70305dedb6fe97aeb530dccc009274f7864442f6f02cf6191b5a32268234bcbd7827bf3e570206c0cddf147\
df5169ceda6883b2169768878fd5b107a092ab7482d8ba7f46364b566aaa72153068b6a0174f2f5e0e5f9bcd0213dd4e\
8689d56ffa0be918a16fffcbe4830157eb8535c1a2a50636f8fc8a57f9ae0488b91159456ca94d7e64a1286babad3e92\
f70203010001a3533051301d0603551d0e04160414fd0e576ce3f05b08884ad67ef3e8b4d39039c65d301f0603551d23\
041830168014fd0e576ce3f05b08884ad67ef3e8b4d39039c65d300f0603551d130101ff040530030101ff300d06092a\
864886f70d01010b050003820101008ed399c2f78e0325f9ec4ae7cd0b978b5cd03d30af8fb61a91925213b6388adb00\
0a59f657dc6a3e983706ff8053b0a4049d5f532c00640e2b67aac3ce30518ad7c4b762078c3816c6f325e9a841920b80\
6d0c5ac16450e8b385c6e50434bf1dc575816a263696b9de661a8bf2ae143853951745d25fa1bbc49d66270197b572aa\
052b0d23d35243cd9087fdc2d79bd2b1d27ad2c67fc7c1960b370b77edc038aee1ee653bec34782bbca87c5aefb3dc92\
5eba1d3c83019a37696d52ea1f14366a13ec6c3f74bda1941c745771bd33ea117f81e6c0968b9692d4dd349743acc149\
73eb5c2a1fcd85691f6dcbd6937b03fe525cbe51610a1f5be86a189de2d2d3",
    )
  }

  fn tlv(tag: u8, value: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + der_len(value.len()).len() + value.len());
    out.push(tag);
    out.extend_from_slice(&der_len(value.len()));
    out.extend_from_slice(value);
    out
  }

  fn tlv_with_noncanonical_short_len(tag: u8, value: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(3 + value.len());
    out.push(tag);
    out.push(0x81);
    out.push(value.len() as u8);
    out.extend_from_slice(value);
    out
  }

  fn tlv_with_leading_zero_long_len(der: &[u8]) -> Vec<u8> {
    let tag = der[0];
    let len_len = usize::from(der[1] & 0x7f);
    assert_ne!(der[1] & 0x80, 0);
    assert_ne!(len_len, 0);

    let mut out = Vec::with_capacity(der.len().strict_add(1));
    out.push(tag);
    out.push(0x80 | (len_len.strict_add(1) as u8));
    out.push(0);
    out.extend_from_slice(&der[2..]);
    out
  }

  fn der_len(len: usize) -> Vec<u8> {
    if len < 128 {
      return vec![len as u8];
    }

    let bytes = len.to_be_bytes();
    let first_nonzero = bytes.iter().position(|&byte| byte != 0).unwrap();
    let len_bytes = &bytes[first_nonzero..];
    let mut out = Vec::with_capacity(1 + len_bytes.len());
    out.push(0x80 | len_bytes.len() as u8);
    out.extend_from_slice(len_bytes);
    out
  }

  fn pkcs1v15_encoded_sha256(message: &[u8], em_len: usize) -> Vec<u8> {
    let digest = Sha256::digest(message);
    let digest_info_len = SHA256_DIGEST_INFO_PREFIX.len().strict_add(Sha256::OUTPUT_SIZE);
    let ps_len = em_len.strict_sub(digest_info_len).strict_sub(3);
    let mut encoded = Vec::with_capacity(em_len);
    encoded.push(0);
    encoded.push(1);
    encoded.extend(core::iter::repeat_n(0xff, ps_len));
    encoded.push(0);
    encoded.extend_from_slice(SHA256_DIGEST_INFO_PREFIX);
    encoded.extend_from_slice(&digest);
    encoded
  }

  fn pss_encoded_sha256(message: &[u8], em_bits: usize, salt_len: usize) -> Vec<u8> {
    let em_len = em_bits.strict_add(7) / 8;
    let h_len = Sha256::OUTPUT_SIZE;
    let db_len = em_len.strict_sub(h_len).strict_sub(1);
    let ps_len = db_len.strict_sub(salt_len).strict_sub(1);

    let mut salt = Vec::with_capacity(salt_len);
    for index in 0..salt_len {
      salt.push((index as u8).wrapping_mul(17).wrapping_add(0xa5));
    }

    let m_hash = Sha256::digest(message);
    let mut h = Sha256::new();
    h.update(&[0u8; 8]);
    h.update(&m_hash);
    h.update(&salt);
    let h = h.finalize();

    let mut db = Vec::with_capacity(db_len);
    db.extend(core::iter::repeat_n(0, ps_len));
    db.push(1);
    db.extend_from_slice(&salt);

    let mut db_mask = vec![0u8; db_len];
    mgf1::<Sha256>(&h, &mut db_mask);
    for (byte, mask) in db.iter_mut().zip(db_mask.iter().copied()) {
      *byte ^= mask;
    }

    let unused_bits = em_len.strict_mul(8).strict_sub(em_bits);
    if unused_bits > 0
      && let Some(first) = db.first_mut()
    {
      *first &= 0xffu8 >> unused_bits;
    }

    let mut encoded = Vec::with_capacity(em_len);
    encoded.extend_from_slice(&db);
    encoded.extend_from_slice(&h);
    encoded.push(0xbc);
    encoded
  }

  fn factor_two_and_inverse(modulus: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let mut factor = vec![0u8; modulus.len()];
    factor[modulus.len().strict_sub(1)] = 2;

    let mut inverse = vec![0u8; modulus.len()];
    let mut carry = 0u8;
    for (dst, &byte) in inverse.iter_mut().zip(modulus) {
      *dst = (byte >> 1) | carry;
      carry = (byte & 1) << 7;
    }
    for byte in inverse.iter_mut().rev() {
      let (sum, overflow) = byte.overflowing_add(1);
      *byte = sum;
      if !overflow {
        break;
      }
    }

    (factor, inverse)
  }

  fn rsa_private_key_parts<'a>(
    modulus: &'a [u8],
    private_exponent: &'a [u8],
    prime_p: &'a [u8],
    prime_q: &'a [u8],
    exponent_p: &'a [u8],
    exponent_q: &'a [u8],
    coefficient: &'a [u8],
  ) -> RsaPrivateKeyParts<'a> {
    RsaPrivateKeyParts {
      modulus,
      public_exponent: 65_537,
      private_exponent,
      prime_p,
      prime_q,
      exponent_p,
      exponent_q,
      coefficient,
    }
  }

  fn wrong_width_private_scratch(key: &RsaPrivateKey) -> RsaPrivateScratch {
    let wrong_len = key.signature_len().strict_sub(1);
    RsaPrivateScratch {
      encoded: SecretBigEndianBuffer::zeroed(wrong_len),
      salt: SecretBigEndianBuffer::zeroed(wrong_len),
      blinding_factor: SecretBigEndianBuffer::zeroed(wrong_len),
      blinding_inverse: SecretBigEndianBuffer::zeroed(wrong_len),
      blinding_power: SecretBigEndianBuffer::zeroed(wrong_len),
      blinded: SecretBigEndianBuffer::zeroed(wrong_len),
      blinded_private_result: SecretBigEndianBuffer::zeroed(wrong_len),
      checked: SecretBigEndianBuffer::zeroed(wrong_len),
      one: SecretBigEndianBuffer::zeroed(wrong_len),
      public_scratch: key.public_key().public_scratch(),
      mul_scratch: RsaPrivateMulScratch::new(key.components.public.modulus.limbs.len()),
      exponent_p_scratch: RsaPrivateExponentScratch::new(key.components.prime_p_modulus.limbs.len()),
      exponent_q_scratch: RsaPrivateExponentScratch::new(key.components.prime_q_modulus.limbs.len()),
    }
  }

  #[test]
  fn pkcs1_private_key_parser_preserves_components_and_public_key() {
    let der = test_pkcs1_private_key();
    let key = parse_pkcs1_private_key_der_with_policy(&der, &RsaPublicKeyPolicy::legacy_verification()).unwrap();

    assert_eq!(key.public_key().modulus(), rsa_private_modulus());
    assert_eq!(key.public_key().public_exponent().as_u64(), 65_537);
    assert_eq!(key.private_exponent.as_bytes(), rsa_private_exponent());
    assert_eq!(key.prime_p.as_bytes(), rsa_private_prime_p());
    assert_eq!(key.prime_q.as_bytes(), rsa_private_prime_q());
    assert_eq!(key.exponent_p.as_bytes(), rsa_private_exponent_p());
    assert_eq!(key.exponent_q.as_bytes(), rsa_private_exponent_q());
    assert_eq!(key.coefficient.as_bytes(), rsa_private_coefficient());
  }

  #[test]
  fn private_key_components_debug_redacts_secret_material() {
    let der = test_pkcs1_private_key();
    let key = parse_pkcs1_private_key_der_with_policy(&der, &RsaPublicKeyPolicy::legacy_verification()).unwrap();
    let debug = format!("{key:?}");

    assert!(debug.contains("modulus_bits"));
    assert!(debug.contains("public_exponent"));
    assert!(debug.contains("private_exponent: \"****\""));
    assert!(debug.contains("prime_p: \"****\""));
    assert!(debug.contains("prime_q: \"****\""));
    assert!(debug.contains("coefficient: \"****\""));
    for secret in [
      rsa_private_exponent(),
      rsa_private_prime_p(),
      rsa_private_prime_q(),
      rsa_private_exponent_p(),
      rsa_private_exponent_q(),
      rsa_private_coefficient(),
    ] {
      assert!(!debug.contains(&format!("{secret:?}")));
    }
  }

  #[test]
  fn private_key_signs_pkcs1v15_and_pss_end_to_end() {
    let der = test_pkcs1_private_key();
    let key = RsaPrivateKey::from_pkcs1_der_with_policy(&der, &RsaPublicKeyPolicy::legacy_verification()).unwrap();
    let message = b"rscrypto RSA private signing roundtrip";
    let (blinding_factor, blinding_factor_inverse) = factor_two_and_inverse(key.public_key().modulus());

    let mut pkcs1v15_signature = vec![0u8; key.public_key().modulus().len()];
    key
      .sign_pkcs1v15_with_blinding_factor(
        RsaPkcs1v15Profile::Sha256,
        message,
        &blinding_factor,
        &blinding_factor_inverse,
        &mut pkcs1v15_signature,
      )
      .unwrap();
    key
      .public_key()
      .verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, message, &pkcs1v15_signature)
      .unwrap();
    assert_eq!(
      key
        .public_key()
        .verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, b"wrong message", &pkcs1v15_signature),
      Err(VerificationError::new())
    );
    let salt = [0xa5; Sha256::OUTPUT_SIZE];
    let mut pss_signature = vec![0u8; key.public_key().modulus().len()];
    key
      .sign_pss_with_salt_and_blinding_factor(
        RsaPssProfile::Sha256,
        message,
        &salt,
        &blinding_factor,
        &blinding_factor_inverse,
        &mut pss_signature,
      )
      .unwrap();
    key
      .public_key()
      .verify_pss_with_salt_len(RsaPssProfile::Sha256, salt.len(), message, &pss_signature)
      .unwrap();
    assert_eq!(
      key
        .public_key()
        .verify_pss_with_salt_len(RsaPssProfile::Sha256, salt.len(), b"wrong message", &pss_signature),
      Err(VerificationError::new())
    );
    let label = b"rscrypto-oaep-label";
    let plaintext = b"rsa oaep decrypt roundtrip";
    let mut ciphertext = vec![0u8; key.signature_len()];
    let seed = [0x53; Sha256::OUTPUT_SIZE];
    key
      .public_key()
      .encrypt_oaep_with_seed(RsaOaepProfile::Sha256, label, plaintext, &seed, &mut ciphertext)
      .unwrap();
    let short_seed = [0x53; Sha256::OUTPUT_SIZE - 1];
    ciphertext.fill(0xa5);
    assert_eq!(
      key
        .public_key()
        .encrypt_oaep_with_seed(RsaOaepProfile::Sha256, label, plaintext, &short_seed, &mut ciphertext),
      Err(RsaEncryptionError::InvalidLength)
    );
    assert!(is_zero_unsigned_be(&ciphertext));
    key
      .public_key()
      .encrypt_oaep_with_seed(RsaOaepProfile::Sha256, label, plaintext, &seed, &mut ciphertext)
      .unwrap();
    let mut decrypted = vec![0u8; key.signature_len()];
    let decrypted_len = key
      .decrypt_oaep_with_blinding_factor(
        RsaOaepProfile::Sha256,
        label,
        &ciphertext,
        &blinding_factor,
        &blinding_factor_inverse,
        &mut decrypted,
      )
      .unwrap();
    assert_eq!(&decrypted[..decrypted_len], plaintext);
    assert_eq!(
      key.decrypt_oaep_with_blinding_factor(
        RsaOaepProfile::Sha256,
        b"wrong label",
        &ciphertext,
        &blinding_factor,
        &blinding_factor_inverse,
        &mut decrypted,
      ),
      Err(RsaPrivateOpError::DecryptionFailed)
    );
    let pkcs1v15_plaintext = b"rsaes pkcs1v15 decrypt roundtrip";
    let pkcs1v15_padding = vec![0xa7; key.signature_len().strict_sub(pkcs1v15_plaintext.len()).strict_sub(3)];
    let mut pkcs1v15_ciphertext = vec![0u8; key.signature_len()];
    key
      .public_key()
      .encrypt_pkcs1v15_with_seed(pkcs1v15_plaintext, &pkcs1v15_padding, &mut pkcs1v15_ciphertext)
      .unwrap();
    let mut pkcs1v15_decrypted = vec![0u8; key.signature_len()];
    let pkcs1v15_decrypted_len = key
      .decrypt_pkcs1v15_with_blinding_factor(
        &pkcs1v15_ciphertext,
        &blinding_factor,
        &blinding_factor_inverse,
        &mut pkcs1v15_decrypted,
      )
      .unwrap();
    assert_eq!(&pkcs1v15_decrypted[..pkcs1v15_decrypted_len], pkcs1v15_plaintext);
    let mut scratch = key.private_scratch();
    let pkcs1v15_decrypted_len = key
      .decrypt_pkcs1v15_with_blinding_factor_and_scratch(
        &pkcs1v15_ciphertext,
        &blinding_factor,
        &blinding_factor_inverse,
        &mut pkcs1v15_decrypted,
        &mut scratch,
      )
      .unwrap();
    assert_eq!(&pkcs1v15_decrypted[..pkcs1v15_decrypted_len], pkcs1v15_plaintext);
  }

  #[test]
  fn private_key_invalid_blinding_clears_signing_and_decryption_outputs() {
    let der = test_pkcs1_private_key();
    let key = RsaPrivateKey::from_pkcs1_der_with_policy(&der, &RsaPublicKeyPolicy::legacy_verification()).unwrap();
    let message = b"rscrypto RSA invalid blinding output clearing";
    let (blinding_factor, blinding_factor_inverse) = factor_two_and_inverse(key.public_key().modulus());
    let mut bad_blinding_inverse = blinding_factor_inverse.clone();
    *bad_blinding_inverse.last_mut().unwrap() ^= 1;
    let mut scratch = key.private_scratch();

    let mut pkcs1v15_signature = vec![0xa5; key.signature_len()];
    assert_eq!(
      key.sign_pkcs1v15_with_blinding_factor(
        RsaPkcs1v15Profile::Sha256,
        message,
        &blinding_factor,
        &bad_blinding_inverse,
        &mut pkcs1v15_signature,
      ),
      Err(RsaPrivateOpError::InvalidBlindingFactor)
    );
    assert!(is_zero_unsigned_be(&pkcs1v15_signature));

    pkcs1v15_signature.fill(0xa5);
    assert_eq!(
      key.sign_pkcs1v15_with_blinding_factor_and_scratch(
        RsaPkcs1v15Profile::Sha256,
        message,
        &blinding_factor,
        &bad_blinding_inverse,
        &mut pkcs1v15_signature,
        &mut scratch,
      ),
      Err(RsaPrivateOpError::InvalidBlindingFactor)
    );
    assert!(is_zero_unsigned_be(&pkcs1v15_signature));

    let salt = [0x5c; Sha256::OUTPUT_SIZE];
    let mut pss_signature = vec![0xa5; key.signature_len()];
    assert_eq!(
      key.sign_pss_with_salt_and_blinding_factor(
        RsaPssProfile::Sha256,
        message,
        &salt,
        &blinding_factor,
        &bad_blinding_inverse,
        &mut pss_signature,
      ),
      Err(RsaPrivateOpError::InvalidBlindingFactor)
    );
    assert!(is_zero_unsigned_be(&pss_signature));

    pss_signature.fill(0xa5);
    assert_eq!(
      key.sign_pss_with_salt_and_blinding_factor_and_scratch(
        RsaPssProfile::Sha256,
        message,
        &salt,
        &blinding_factor,
        &bad_blinding_inverse,
        &mut pss_signature,
        &mut scratch,
      ),
      Err(RsaPrivateOpError::InvalidBlindingFactor)
    );
    assert!(is_zero_unsigned_be(&pss_signature));

    let label = b"invalid-blinding-clears-oaep";
    let plaintext = b"rsa oaep invalid blinding clear";
    let seed = [0x3d; Sha256::OUTPUT_SIZE];
    let mut ciphertext = vec![0u8; key.signature_len()];
    key
      .public_key()
      .encrypt_oaep_with_seed(RsaOaepProfile::Sha256, label, plaintext, &seed, &mut ciphertext)
      .unwrap();

    let mut decrypted = vec![0xa5; key.signature_len()];
    assert_eq!(
      key.decrypt_oaep_with_blinding_factor(
        RsaOaepProfile::Sha256,
        label,
        &ciphertext,
        &blinding_factor,
        &bad_blinding_inverse,
        &mut decrypted,
      ),
      Err(RsaPrivateOpError::InvalidBlindingFactor)
    );
    assert!(is_zero_unsigned_be(&decrypted));

    decrypted.fill(0xa5);
    assert_eq!(
      key.decrypt_oaep_with_blinding_factor_and_scratch(
        RsaOaepProfile::Sha256,
        label,
        &ciphertext,
        &blinding_factor,
        &bad_blinding_inverse,
        &mut decrypted,
        &mut scratch,
      ),
      Err(RsaPrivateOpError::InvalidBlindingFactor)
    );
    assert!(is_zero_unsigned_be(&decrypted));

    let pkcs1v15_plaintext = b"rsaes pkcs1v15 invalid blinding clear";
    let pkcs1v15_seed = vec![0x6d; key.signature_len().strict_sub(pkcs1v15_plaintext.len()).strict_sub(3)];
    let mut pkcs1v15_ciphertext = vec![0u8; key.signature_len()];
    key
      .public_key()
      .encrypt_pkcs1v15_with_seed(pkcs1v15_plaintext, &pkcs1v15_seed, &mut pkcs1v15_ciphertext)
      .unwrap();

    let mut pkcs1v15_decrypted = vec![0xa5; key.signature_len()];
    assert_eq!(
      key.decrypt_pkcs1v15_with_blinding_factor(
        &pkcs1v15_ciphertext,
        &blinding_factor,
        &bad_blinding_inverse,
        &mut pkcs1v15_decrypted,
      ),
      Err(RsaPrivateOpError::InvalidBlindingFactor)
    );
    assert!(is_zero_unsigned_be(&pkcs1v15_decrypted));

    pkcs1v15_decrypted.fill(0xa5);
    assert_eq!(
      key.decrypt_pkcs1v15_with_blinding_factor_and_scratch(
        &pkcs1v15_ciphertext,
        &blinding_factor,
        &bad_blinding_inverse,
        &mut pkcs1v15_decrypted,
        &mut scratch,
      ),
      Err(RsaPrivateOpError::InvalidBlindingFactor)
    );
    assert!(is_zero_unsigned_be(&pkcs1v15_decrypted));
  }

  #[test]
  fn oaep_decode_rejects_padding_oracle_classes_opaquely() {
    let label = b"rscrypto-oaep-label";
    let message = b"oaep message";
    let seed = [0x3cu8; Sha256::OUTPUT_SIZE];
    let decoded_db = decoded_oaep_sha256_db(label, message, 12);

    let mut valid = masked_oaep_sha256_from_decoded_db(&seed, &decoded_db);
    let mut out = vec![0u8; message.len()];
    let len = decode_oaep::<Sha256>(label, &mut valid, &mut out).unwrap();
    assert_eq!(len, message.len());
    assert_eq!(out, message);

    let mut bad_leading = masked_oaep_sha256_from_decoded_db(&seed, &decoded_db);
    bad_leading[0] = 1;
    assert_eq!(
      decode_oaep::<Sha256>(label, &mut bad_leading, &mut out),
      Err(RsaPrivateOpError::DecryptionFailed)
    );

    let mut bad_label_db = decoded_db.clone();
    bad_label_db[0] ^= 0x80;
    let mut bad_label = masked_oaep_sha256_from_decoded_db(&seed, &bad_label_db);
    assert_eq!(
      decode_oaep::<Sha256>(label, &mut bad_label, &mut out),
      Err(RsaPrivateOpError::DecryptionFailed)
    );

    let mut bad_padding_db = decoded_db.clone();
    bad_padding_db[Sha256::OUTPUT_SIZE.strict_add(3)] = 0x7f;
    let mut bad_padding = masked_oaep_sha256_from_decoded_db(&seed, &bad_padding_db);
    assert_eq!(
      decode_oaep::<Sha256>(label, &mut bad_padding, &mut out),
      Err(RsaPrivateOpError::DecryptionFailed)
    );

    let mut missing_separator_db = decoded_db.clone();
    missing_separator_db[Sha256::OUTPUT_SIZE..].fill(0);
    let mut missing_separator = masked_oaep_sha256_from_decoded_db(&seed, &missing_separator_db);
    assert_eq!(
      decode_oaep::<Sha256>(label, &mut missing_separator, &mut out),
      Err(RsaPrivateOpError::DecryptionFailed)
    );

    let mut valid_short_out = masked_oaep_sha256_from_decoded_db(&seed, &decoded_db);
    let mut short_out = vec![0u8; message.len().strict_sub(1)];
    assert_eq!(
      decode_oaep::<Sha256>(label, &mut valid_short_out, &mut short_out),
      Err(RsaPrivateOpError::InvalidLength)
    );
  }

  #[test]
  fn pkcs1v15_encryption_padding_accepts_valid_and_rejects_oracle_shapes() {
    let message = b"abc";
    let seed = [0x7bu8; 10];
    let mut encoded = [0u8; 16];
    encode_pkcs1v15_encryption_with_seed(message, &seed, &mut encoded).unwrap();

    let mut out = [0u8; 8];
    let len = decode_pkcs1v15_encryption(&encoded, &mut out).unwrap();
    assert_eq!(&out[..len], message);

    let mut zero_seed = seed;
    zero_seed[4] = 0;
    assert_eq!(
      encode_pkcs1v15_encryption_with_seed(message, &zero_seed, &mut encoded),
      Err(RsaEncryptionError::InvalidLength)
    );

    let mut bad_type = encoded;
    bad_type[1] = 1;
    assert_eq!(
      decode_pkcs1v15_encryption(&bad_type, &mut out),
      Err(RsaPrivateOpError::DecryptionFailed)
    );

    let mut short_padding = encoded;
    short_padding[5] = 0;
    assert_eq!(
      decode_pkcs1v15_encryption(&short_padding, &mut out),
      Err(RsaPrivateOpError::DecryptionFailed)
    );

    let mut missing_separator = encoded;
    missing_separator[12..].fill(0x55);
    assert_eq!(
      decode_pkcs1v15_encryption(&missing_separator, &mut out),
      Err(RsaPrivateOpError::DecryptionFailed)
    );
  }

  #[test]
  fn oaep_decrypt_api_rejects_same_width_oracle_classes_opaquely() {
    let der = test_pkcs1_private_key();
    let key = RsaPrivateKey::from_pkcs1_der_with_policy(&der, &RsaPublicKeyPolicy::legacy_verification()).unwrap();
    let (blinding_factor, blinding_factor_inverse) = factor_two_and_inverse(key.public_key().modulus());
    let label = b"rscrypto-oaep-label";
    let plaintext = b"oaep decrypt oracle regression";

    macro_rules! assert_profile_rejects_opaquely {
      ($profile:expr, $seed:expr) => {{
        let mut ciphertext = vec![0u8; key.signature_len()];
        key
          .public_key()
          .encrypt_oaep_with_seed($profile, label, plaintext, &$seed, &mut ciphertext)
          .unwrap();

        let mut out = vec![0u8; key.signature_len()];
        let len = key
          .decrypt_oaep_with_blinding_factor(
            $profile,
            label,
            &ciphertext,
            &blinding_factor,
            &blinding_factor_inverse,
            &mut out,
          )
          .unwrap();
        assert_eq!(&out[..len], plaintext);

        let mut scratch = key.private_scratch();
        let len = key
          .decrypt_oaep_with_blinding_factor_and_scratch(
            $profile,
            label,
            &ciphertext,
            &blinding_factor,
            &blinding_factor_inverse,
            &mut out,
            &mut scratch,
          )
          .unwrap();
        assert_eq!(&out[..len], plaintext);

        #[cfg(feature = "getrandom")]
        {
          let len = key.decrypt_oaep($profile, label, &ciphertext, &mut out).unwrap();
          assert_eq!(&out[..len], plaintext);

          let len = key
            .decrypt_oaep_with_scratch($profile, label, &ciphertext, &mut out, &mut scratch)
            .unwrap();
          assert_eq!(&out[..len], plaintext);
        }

        out.fill(0xa5);
        assert_eq!(
          key.decrypt_oaep_with_blinding_factor(
            $profile,
            b"wrong label",
            &ciphertext,
            &blinding_factor,
            &blinding_factor_inverse,
            &mut out,
          ),
          Err(RsaPrivateOpError::DecryptionFailed)
        );
        assert!(is_zero_unsigned_be(&out));
        out.fill(0xa5);
        assert_eq!(
          key.decrypt_oaep_with_blinding_factor_and_scratch(
            $profile,
            b"wrong label",
            &ciphertext,
            &blinding_factor,
            &blinding_factor_inverse,
            &mut out,
            &mut scratch,
          ),
          Err(RsaPrivateOpError::DecryptionFailed)
        );
        assert!(is_zero_unsigned_be(&out));

        #[cfg(feature = "getrandom")]
        {
          out.fill(0xa5);
          assert_eq!(
            key.decrypt_oaep($profile, b"wrong label", &ciphertext, &mut out),
            Err(RsaPrivateOpError::DecryptionFailed)
          );
          assert!(is_zero_unsigned_be(&out));
          out.fill(0xa5);
          assert_eq!(
            key.decrypt_oaep_with_scratch($profile, b"wrong label", &ciphertext, &mut out, &mut scratch),
            Err(RsaPrivateOpError::DecryptionFailed)
          );
          assert!(is_zero_unsigned_be(&out));
        }

        let mut tampered_tail = ciphertext.clone();
        *tampered_tail.last_mut().unwrap() ^= 0x01;
        out.fill(0xa5);
        assert_eq!(
          key.decrypt_oaep_with_blinding_factor(
            $profile,
            label,
            &tampered_tail,
            &blinding_factor,
            &blinding_factor_inverse,
            &mut out,
          ),
          Err(RsaPrivateOpError::DecryptionFailed)
        );
        assert!(is_zero_unsigned_be(&out));
        out.fill(0xa5);
        assert_eq!(
          key.decrypt_oaep_with_blinding_factor_and_scratch(
            $profile,
            label,
            &tampered_tail,
            &blinding_factor,
            &blinding_factor_inverse,
            &mut out,
            &mut scratch,
          ),
          Err(RsaPrivateOpError::DecryptionFailed)
        );
        assert!(is_zero_unsigned_be(&out));

        #[cfg(feature = "getrandom")]
        {
          out.fill(0xa5);
          assert_eq!(
            key.decrypt_oaep($profile, label, &tampered_tail, &mut out),
            Err(RsaPrivateOpError::DecryptionFailed)
          );
          assert!(is_zero_unsigned_be(&out));
          out.fill(0xa5);
          assert_eq!(
            key.decrypt_oaep_with_scratch($profile, label, &tampered_tail, &mut out, &mut scratch),
            Err(RsaPrivateOpError::DecryptionFailed)
          );
          assert!(is_zero_unsigned_be(&out));
        }

        let zero_representative = vec![0u8; key.signature_len()];
        out.fill(0xa5);
        assert_eq!(
          key.decrypt_oaep_with_blinding_factor(
            $profile,
            label,
            &zero_representative,
            &blinding_factor,
            &blinding_factor_inverse,
            &mut out,
          ),
          Err(RsaPrivateOpError::DecryptionFailed)
        );
        assert!(is_zero_unsigned_be(&out));
        out.fill(0xa5);
        assert_eq!(
          key.decrypt_oaep_with_blinding_factor_and_scratch(
            $profile,
            label,
            &zero_representative,
            &blinding_factor,
            &blinding_factor_inverse,
            &mut out,
            &mut scratch,
          ),
          Err(RsaPrivateOpError::DecryptionFailed)
        );
        assert!(is_zero_unsigned_be(&out));

        #[cfg(feature = "getrandom")]
        {
          out.fill(0xa5);
          assert_eq!(
            key.decrypt_oaep($profile, label, &zero_representative, &mut out),
            Err(RsaPrivateOpError::DecryptionFailed)
          );
          assert!(is_zero_unsigned_be(&out));
          out.fill(0xa5);
          assert_eq!(
            key.decrypt_oaep_with_scratch($profile, label, &zero_representative, &mut out, &mut scratch),
            Err(RsaPrivateOpError::DecryptionFailed)
          );
          assert!(is_zero_unsigned_be(&out));
        }
      }};
    }

    assert_profile_rejects_opaquely!(RsaOaepProfile::Sha256, [0x45; Sha256::OUTPUT_SIZE]);
    assert_profile_rejects_opaquely!(RsaOaepProfile::Sha384, [0x46; Sha384::OUTPUT_SIZE]);
    assert_profile_rejects_opaquely!(RsaOaepProfile::Sha512, [0x47; Sha512::OUTPUT_SIZE]);
  }

  #[test]
  fn pkcs1v15_encrypt_decrypt_rejects_oracle_classes_opaquely() {
    let der = test_pkcs1_private_key();
    let key = RsaPrivateKey::from_pkcs1_der_with_policy(&der, &RsaPublicKeyPolicy::legacy_verification()).unwrap();
    let (blinding_factor, blinding_factor_inverse) = factor_two_and_inverse(key.public_key().modulus());
    let plaintext = b"rsaes-pkcs1v15 legacy roundtrip";
    let seed = vec![0xb6; key.signature_len().strict_sub(plaintext.len()).strict_sub(3)];

    let mut ciphertext = vec![0u8; key.signature_len()];
    key
      .public_key()
      .encrypt_pkcs1v15_with_seed(plaintext, &seed, &mut ciphertext)
      .unwrap();
    let mut out = vec![0u8; key.signature_len()];
    let decrypted_len = key
      .decrypt_pkcs1v15_with_blinding_factor(&ciphertext, &blinding_factor, &blinding_factor_inverse, &mut out)
      .unwrap();
    assert_eq!(&out[..decrypted_len], plaintext);

    let mut scratch = key.private_scratch();
    let decrypted_len = key
      .decrypt_pkcs1v15_with_blinding_factor_and_scratch(
        &ciphertext,
        &blinding_factor,
        &blinding_factor_inverse,
        &mut out,
        &mut scratch,
      )
      .unwrap();
    assert_eq!(&out[..decrypted_len], plaintext);

    #[cfg(feature = "getrandom")]
    {
      let decrypted_len = key.decrypt_pkcs1v15(&ciphertext, &mut out).unwrap();
      assert_eq!(&out[..decrypted_len], plaintext);
      let decrypted_len = key
        .decrypt_pkcs1v15_with_scratch(&ciphertext, &mut out, &mut scratch)
        .unwrap();
      assert_eq!(&out[..decrypted_len], plaintext);
    }

    let mut zero_seed = seed.clone();
    zero_seed[3] = 0;
    ciphertext.fill(0xa5);
    assert_eq!(
      key
        .public_key()
        .encrypt_pkcs1v15_with_seed(plaintext, &zero_seed, &mut ciphertext),
      Err(RsaEncryptionError::InvalidLength)
    );
    assert!(is_zero_unsigned_be(&ciphertext));
    #[cfg(feature = "getrandom")]
    {
      let oversized_plaintext = vec![0x5a; key.signature_len().strict_sub(10)];
      ciphertext.fill(0xa5);
      assert_eq!(
        key.public_key().encrypt_pkcs1v15(&oversized_plaintext, &mut ciphertext),
        Err(RsaEncryptionError::MessageTooLong)
      );
      assert!(is_zero_unsigned_be(&ciphertext));
    }

    let mut bad_block = vec![0u8; key.signature_len()];
    bad_block[1] = 2;
    bad_block[2..10].fill(0xc5);
    bad_block[10] = 0;
    bad_block[11..11usize.strict_add(plaintext.len())].copy_from_slice(plaintext);
    key.public_key().public_operation(&bad_block, &mut ciphertext).unwrap();
    let mut assert_decrypt_error = |ciphertext: &[u8]| {
      out.fill(0xa5);
      assert_eq!(
        key.decrypt_pkcs1v15_with_blinding_factor(ciphertext, &blinding_factor, &blinding_factor_inverse, &mut out,),
        Err(RsaPrivateOpError::DecryptionFailed)
      );
      assert!(is_zero_unsigned_be(&out));
      out.fill(0xa5);
      assert_eq!(
        key.decrypt_pkcs1v15_with_blinding_factor_and_scratch(
          ciphertext,
          &blinding_factor,
          &blinding_factor_inverse,
          &mut out,
          &mut scratch,
        ),
        Err(RsaPrivateOpError::DecryptionFailed)
      );
      assert!(is_zero_unsigned_be(&out));
      #[cfg(feature = "getrandom")]
      {
        out.fill(0xa5);
        assert_eq!(
          key.decrypt_pkcs1v15(ciphertext, &mut out),
          Err(RsaPrivateOpError::DecryptionFailed)
        );
        assert!(is_zero_unsigned_be(&out));
        out.fill(0xa5);
        assert_eq!(
          key.decrypt_pkcs1v15_with_scratch(ciphertext, &mut out, &mut scratch),
          Err(RsaPrivateOpError::DecryptionFailed)
        );
        assert!(is_zero_unsigned_be(&out));
      }
    };

    bad_block[1] = 1;
    key.public_key().public_operation(&bad_block, &mut ciphertext).unwrap();
    assert_decrypt_error(&ciphertext);

    bad_block[1] = 2;
    bad_block[5] = 0;
    key.public_key().public_operation(&bad_block, &mut ciphertext).unwrap();
    assert_decrypt_error(&ciphertext);

    bad_block[5] = 0xc5;
    bad_block[10..].fill(0xc5);
    key.public_key().public_operation(&bad_block, &mut ciphertext).unwrap();
    assert_decrypt_error(&ciphertext);
  }

  #[test]
  fn key_der_exports_roundtrip_through_strict_importers() {
    let pkcs1 = test_pkcs1_private_key();
    let key = RsaPrivateKey::from_pkcs1_der_with_policy(&pkcs1, &RsaPublicKeyPolicy::legacy_verification()).unwrap();
    let rsa_algorithm = algorithm_identifier(RSA_ENCRYPTION_OID, Some(&null()));

    assert_eq!(
      key.public_key().to_pkcs1_der(),
      test_pkcs1_public_key(&rsa_private_modulus(), &[0x01, 0x00, 0x01])
    );
    assert_eq!(key.to_pkcs8_der(), test_pkcs8_private_key(&pkcs1, &rsa_algorithm),);

    let exported_pkcs1 = key.to_pkcs1_der();
    let imported_pkcs1 =
      RsaPrivateKey::from_pkcs1_der_with_policy(&exported_pkcs1, &RsaPublicKeyPolicy::legacy_verification()).unwrap();
    assert_eq!(imported_pkcs1.public_key(), key.public_key());

    let exported_pkcs8 = key.to_pkcs8_der();
    let imported_pkcs8 =
      RsaPrivateKey::from_pkcs8_der_with_policy(&exported_pkcs8, &RsaPublicKeyPolicy::legacy_verification()).unwrap();
    assert_eq!(imported_pkcs8.public_key(), key.public_key());

    let public_pkcs1 = key.public_key().to_pkcs1_der();
    let imported_public_pkcs1 =
      RsaPublicKey::from_pkcs1_der_with_policy(&public_pkcs1, &RsaPublicKeyPolicy::legacy_verification()).unwrap();
    assert_eq!(imported_public_pkcs1, *key.public_key());

    let spki = key.public_key().to_spki_der();
    let imported_spki =
      RsaPublicKey::from_spki_der_with_policy(&spki, &RsaPublicKeyPolicy::legacy_verification()).unwrap();
    assert_eq!(imported_spki, *key.public_key());
  }

  #[test]
  fn raw_public_and_private_components_import_end_to_end() {
    let modulus = rsa_private_modulus();
    let private_exponent = rsa_private_exponent();
    let prime_p = rsa_private_prime_p();
    let prime_q = rsa_private_prime_q();
    let exponent_p = rsa_private_exponent_p();
    let exponent_q = rsa_private_exponent_q();
    let coefficient = rsa_private_coefficient();
    let policy = RsaPublicKeyPolicy::legacy_verification();

    let public = RsaPublicKey::from_modulus_exponent_with_policy(&modulus, 65_537, &policy).unwrap();
    assert_eq!(public.modulus(), modulus);
    assert_eq!(public.public_exponent().as_u64(), 65_537);

    let key = RsaPrivateKey::from_components_with_policy(
      rsa_private_key_parts(
        &modulus,
        &private_exponent,
        &prime_p,
        &prime_q,
        &exponent_p,
        &exponent_q,
        &coefficient,
      ),
      &policy,
    )
    .unwrap();
    assert_eq!(key.public_key(), &public);

    let message = b"rscrypto raw RSA component import signing roundtrip";
    let (blinding_factor, blinding_factor_inverse) = factor_two_and_inverse(key.public_key().modulus());
    let mut signature = vec![0u8; key.signature_len()];
    key
      .sign_pkcs1v15_with_blinding_factor(
        RsaPkcs1v15Profile::Sha256,
        message,
        &blinding_factor,
        &blinding_factor_inverse,
        &mut signature,
      )
      .unwrap();
    public
      .verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, message, &signature)
      .unwrap();

    let mut bad_coefficient = coefficient;
    *bad_coefficient.last_mut().unwrap() ^= 1;
    assert_eq!(
      RsaPrivateKey::from_components_with_policy(
        rsa_private_key_parts(
          &modulus,
          &private_exponent,
          &prime_p,
          &prime_q,
          &exponent_p,
          &exponent_q,
          &bad_coefficient,
        ),
        &policy,
      )
      .err(),
      Some(RsaKeyError::InvalidModulus)
    );
    assert_eq!(
      RsaPublicKey::from_modulus_exponent_with_policy(&modulus, 17, &policy).err(),
      Some(RsaKeyError::InvalidPublicExponent)
    );
  }

  #[test]
  fn raw_private_components_reject_noncanonical_leading_zeroes() {
    let modulus = rsa_private_modulus();
    let private_exponent = rsa_private_exponent();
    let prime_p = rsa_private_prime_p();
    let prime_q = rsa_private_prime_q();
    let exponent_p = rsa_private_exponent_p();
    let exponent_q = rsa_private_exponent_q();
    let coefficient = rsa_private_coefficient();
    let policy = RsaPublicKeyPolicy::legacy_verification();

    let mut bad_private_exponent = private_exponent.clone();
    bad_private_exponent.insert(0, 0);
    assert_eq!(
      RsaPrivateKey::from_components_with_policy(
        rsa_private_key_parts(
          &modulus,
          &bad_private_exponent,
          &prime_p,
          &prime_q,
          &exponent_p,
          &exponent_q,
          &coefficient,
        ),
        &policy,
      )
      .err(),
      Some(RsaKeyError::InvalidModulus)
    );

    let mut bad_prime_p = prime_p.clone();
    bad_prime_p.insert(0, 0);
    assert_eq!(
      RsaPrivateKey::from_components_with_policy(
        rsa_private_key_parts(
          &modulus,
          &private_exponent,
          &bad_prime_p,
          &prime_q,
          &exponent_p,
          &exponent_q,
          &coefficient,
        ),
        &policy,
      )
      .err(),
      Some(RsaKeyError::InvalidModulus)
    );

    let mut bad_prime_q = prime_q.clone();
    bad_prime_q.insert(0, 0);
    assert_eq!(
      RsaPrivateKey::from_components_with_policy(
        rsa_private_key_parts(
          &modulus,
          &private_exponent,
          &prime_p,
          &bad_prime_q,
          &exponent_p,
          &exponent_q,
          &coefficient,
        ),
        &policy,
      )
      .err(),
      Some(RsaKeyError::InvalidModulus)
    );

    let mut bad_exponent_p = exponent_p.clone();
    bad_exponent_p.insert(0, 0);
    assert_eq!(
      RsaPrivateKey::from_components_with_policy(
        rsa_private_key_parts(
          &modulus,
          &private_exponent,
          &prime_p,
          &prime_q,
          &bad_exponent_p,
          &exponent_q,
          &coefficient,
        ),
        &policy,
      )
      .err(),
      Some(RsaKeyError::InvalidModulus)
    );

    let mut bad_exponent_q = exponent_q.clone();
    bad_exponent_q.insert(0, 0);
    assert_eq!(
      RsaPrivateKey::from_components_with_policy(
        rsa_private_key_parts(
          &modulus,
          &private_exponent,
          &prime_p,
          &prime_q,
          &exponent_p,
          &bad_exponent_q,
          &coefficient,
        ),
        &policy,
      )
      .err(),
      Some(RsaKeyError::InvalidModulus)
    );

    let mut bad_coefficient = coefficient.clone();
    bad_coefficient.insert(0, 0);
    assert_eq!(
      RsaPrivateKey::from_components_with_policy(
        rsa_private_key_parts(
          &modulus,
          &private_exponent,
          &prime_p,
          &prime_q,
          &exponent_p,
          &exponent_q,
          &bad_coefficient,
        ),
        &policy,
      )
      .err(),
      Some(RsaKeyError::InvalidModulus)
    );
  }

  #[cfg(feature = "getrandom")]
  #[test]
  fn private_key_getrandom_signs_and_decrypts_end_to_end() {
    let der = test_pkcs1_private_key();
    let key = RsaPrivateKey::from_pkcs1_der_with_policy(&der, &RsaPublicKeyPolicy::legacy_verification()).unwrap();
    let message = b"rscrypto RSA getrandom signing roundtrip";
    let mut scratch = key.private_scratch();

    let mut pkcs1v15_signature = vec![0u8; key.signature_len()];
    key
      .sign_pkcs1v15(RsaPkcs1v15Profile::Sha256, message, &mut pkcs1v15_signature)
      .unwrap();
    key
      .public_key()
      .verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, message, &pkcs1v15_signature)
      .unwrap();
    key
      .sign_pkcs1v15_with_scratch(
        RsaPkcs1v15Profile::Sha384,
        message,
        &mut pkcs1v15_signature,
        &mut scratch,
      )
      .unwrap();
    key
      .public_key()
      .verify_pkcs1v15(RsaPkcs1v15Profile::Sha384, message, &pkcs1v15_signature)
      .unwrap();

    let mut pss_signature = vec![0u8; key.signature_len()];
    key
      .sign_pss(RsaPssProfile::Sha256, message, &mut pss_signature)
      .unwrap();
    key
      .public_key()
      .verify_pss(RsaPssProfile::Sha256, message, &pss_signature)
      .unwrap();
    key
      .sign_pss_with_scratch(RsaPssProfile::Sha384, message, &mut pss_signature, &mut scratch)
      .unwrap();
    key
      .public_key()
      .verify_pss(RsaPssProfile::Sha384, message, &pss_signature)
      .unwrap();
    let pss_zero_salt = RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha256, 0);
    key
      .sign_signature_with_scratch(pss_zero_salt, message, &mut pss_signature, &mut scratch)
      .unwrap();
    key
      .public_key()
      .verify_signature(pss_zero_salt, message, &pss_signature)
      .unwrap();

    let label = b"rscrypto-getrandom-oaep";
    let plaintext = b"normal oaep api";
    let mut ciphertext = vec![0u8; key.signature_len()];
    key
      .public_key()
      .encrypt_oaep(RsaOaepProfile::Sha256, label, plaintext, &mut ciphertext)
      .unwrap();
    let mut decrypted = vec![0u8; key.signature_len()];
    let decrypted_len = key
      .decrypt_oaep(RsaOaepProfile::Sha256, label, &ciphertext, &mut decrypted)
      .unwrap();
    assert_eq!(&decrypted[..decrypted_len], plaintext);
    let decrypted_len = key
      .decrypt_oaep_with_scratch(RsaOaepProfile::Sha256, label, &ciphertext, &mut decrypted, &mut scratch)
      .unwrap();
    assert_eq!(&decrypted[..decrypted_len], plaintext);

    let pkcs1v15_plaintext = b"normal pkcs1v15 encryption api";
    key
      .public_key()
      .encrypt_pkcs1v15(pkcs1v15_plaintext, &mut ciphertext)
      .unwrap();
    let decrypted_len = key.decrypt_pkcs1v15(&ciphertext, &mut decrypted).unwrap();
    assert_eq!(&decrypted[..decrypted_len], pkcs1v15_plaintext);
    let decrypted_len = key
      .decrypt_pkcs1v15_with_scratch(&ciphertext, &mut decrypted, &mut scratch)
      .unwrap();
    assert_eq!(&decrypted[..decrypted_len], pkcs1v15_plaintext);
  }

  #[cfg(feature = "getrandom")]
  #[test]
  fn private_key_sign_signature_profile_and_explicit_pss_salt_len_roundtrip() {
    let der = test_pkcs1_private_key();
    let key = RsaPrivateKey::from_pkcs1_der_with_policy(&der, &RsaPublicKeyPolicy::legacy_verification()).unwrap();
    let message = b"rscrypto RSA typed signing profile roundtrip";

    let pkcs1_profile = RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha384);
    let mut pkcs1v15_signature = vec![0u8; key.signature_len()];
    key
      .sign_signature(pkcs1_profile, message, &mut pkcs1v15_signature)
      .unwrap();
    key
      .public_key()
      .verify_signature(pkcs1_profile, message, &pkcs1v15_signature)
      .unwrap();

    let pss_zero_salt = RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha256, 0);
    let mut pss_signature = vec![0u8; key.signature_len()];
    key.sign_signature(pss_zero_salt, message, &mut pss_signature).unwrap();
    key
      .public_key()
      .verify_signature(pss_zero_salt, message, &pss_signature)
      .unwrap();

    key
      .sign_pss_with_salt_len(RsaPssProfile::Sha512, 24, message, &mut pss_signature)
      .unwrap();
    key
      .public_key()
      .verify_pss_with_salt_len(RsaPssProfile::Sha512, 24, message, &pss_signature)
      .unwrap();

    let mut scratch = key.private_scratch();
    key
      .sign_pss_with_salt_len_and_scratch(RsaPssProfile::Sha384, 16, message, &mut pss_signature, &mut scratch)
      .unwrap();
    key
      .public_key()
      .verify_pss_with_salt_len(RsaPssProfile::Sha384, 16, message, &pss_signature)
      .unwrap();

    pss_signature.fill(0xa5);
    assert_eq!(
      key.sign_pss_with_salt_len(RsaPssProfile::Sha512, usize::MAX, message, &mut pss_signature),
      Err(RsaPrivateOpError::MessageTooLong)
    );
    assert!(is_zero_unsigned_be(&pss_signature));

    pss_signature.fill(0xa5);
    assert_eq!(
      key.sign_pss_with_salt_len_and_scratch(
        RsaPssProfile::Sha512,
        usize::MAX,
        message,
        &mut pss_signature,
        &mut scratch,
      ),
      Err(RsaPrivateOpError::MessageTooLong)
    );
    assert!(is_zero_unsigned_be(&pss_signature));
  }

  #[cfg(feature = "getrandom")]
  #[test]
  fn private_key_protocol_signing_helpers_roundtrip_and_reject_confusion() {
    let der = test_pkcs1_private_key();
    let key = RsaPrivateKey::from_pkcs1_der_with_policy(&der, &RsaPublicKeyPolicy::legacy_verification()).unwrap();
    let message = b"rscrypto RSA primitive protocol signing helper roundtrip";
    let mut signature = vec![0u8; key.signature_len()];
    let mut scratch = key.private_scratch();

    let x509_pkcs1v15 = algorithm_identifier(SHA384_WITH_RSA_ENCRYPTION_OID, Some(&null()));
    key
      .sign_x509_signature_algorithm_der(&x509_pkcs1v15, message, &mut signature)
      .unwrap();
    let x509_pkcs1v15_profile = RsaSignatureProfile::from_x509_signature_algorithm_der(&x509_pkcs1v15).unwrap();
    key
      .public_key()
      .verify_signature(x509_pkcs1v15_profile, message, &signature)
      .unwrap();
    key
      .sign_x509_signature_algorithm_der_with_scratch(&x509_pkcs1v15, message, &mut signature, &mut scratch)
      .unwrap();
    key
      .public_key()
      .verify_signature(x509_pkcs1v15_profile, message, &signature)
      .unwrap();

    let x509_pss = x509_pss_algorithm(RsaPssProfile::Sha256, 20);
    key
      .sign_x509_signature_algorithm_der(&x509_pss, message, &mut signature)
      .unwrap();
    let x509_pss_profile = RsaSignatureProfile::from_x509_signature_algorithm_der(&x509_pss).unwrap();
    key
      .public_key()
      .verify_signature(x509_pss_profile, message, &signature)
      .unwrap();
    key
      .sign_x509_signature_algorithm_der_with_scratch(&x509_pss, message, &mut signature, &mut scratch)
      .unwrap();
    key
      .public_key()
      .verify_signature(x509_pss_profile, message, &signature)
      .unwrap();

    key
      .sign_tls13_signature_scheme(0x0804, message, &mut signature)
      .unwrap();
    key
      .public_key()
      .verify_signature(
        RsaSignatureProfile::from_tls13_signature_scheme(0x0804).unwrap(),
        message,
        &signature,
      )
      .unwrap();
    key
      .sign_tls13_signature_scheme_with_scratch(0x0804, message, &mut signature, &mut scratch)
      .unwrap();
    key
      .public_key()
      .verify_signature(
        RsaSignatureProfile::from_tls13_signature_scheme(0x0804).unwrap(),
        message,
        &signature,
      )
      .unwrap();

    key
      .sign_tls_certificate_signature_scheme(0x0501, message, &mut signature)
      .unwrap();
    key
      .public_key()
      .verify_signature(
        RsaSignatureProfile::from_tls_certificate_signature_scheme(0x0501).unwrap(),
        message,
        &signature,
      )
      .unwrap();
    key
      .sign_tls_certificate_signature_scheme_with_scratch(0x0501, message, &mut signature, &mut scratch)
      .unwrap();
    key
      .public_key()
      .verify_signature(
        RsaSignatureProfile::from_tls_certificate_signature_scheme(0x0501).unwrap(),
        message,
        &signature,
      )
      .unwrap();

    key.sign_jwt_alg("PS512", message, &mut signature).unwrap();
    key.public_key().verify_jwt_alg("PS512", message, &signature).unwrap();
    key
      .sign_jwt_alg_with_scratch("PS512", message, &mut signature, &mut scratch)
      .unwrap();
    key.public_key().verify_jwt_alg("PS512", message, &signature).unwrap();

    key.sign_cose_algorithm_id(-257, message, &mut signature).unwrap();
    key
      .public_key()
      .verify_cose_algorithm_id(-257, message, &signature)
      .unwrap();
    key
      .sign_cose_algorithm_id_with_scratch(-257, message, &mut signature, &mut scratch)
      .unwrap();
    key
      .public_key()
      .verify_cose_algorithm_id(-257, message, &signature)
      .unwrap();

    signature.fill(0xa5);
    assert_eq!(
      key.sign_tls13_signature_scheme(0x0401, message, &mut signature),
      Err(RsaPrivateOpError::UnsupportedAlgorithm)
    );
    assert!(is_zero_unsigned_be(&signature));
    signature.fill(0xa5);
    assert_eq!(
      key.sign_tls13_signature_scheme_with_scratch(0x0401, message, &mut signature, &mut scratch),
      Err(RsaPrivateOpError::UnsupportedAlgorithm)
    );
    assert!(is_zero_unsigned_be(&signature));
    signature.fill(0xa5);
    assert_eq!(
      key.sign_tls_certificate_signature_scheme(0x0201, message, &mut signature),
      Err(RsaPrivateOpError::UnsupportedAlgorithm)
    );
    assert!(is_zero_unsigned_be(&signature));
    signature.fill(0xa5);
    assert_eq!(
      key.sign_tls_certificate_signature_scheme_with_scratch(0x0201, message, &mut signature, &mut scratch),
      Err(RsaPrivateOpError::UnsupportedAlgorithm)
    );
    assert!(is_zero_unsigned_be(&signature));
    signature.fill(0xa5);
    assert_eq!(
      key.sign_jwt_alg("HS256", message, &mut signature),
      Err(RsaPrivateOpError::UnsupportedAlgorithm)
    );
    assert!(is_zero_unsigned_be(&signature));
    signature.fill(0xa5);
    assert_eq!(
      key.sign_jwt_alg_with_scratch("HS256", message, &mut signature, &mut scratch),
      Err(RsaPrivateOpError::UnsupportedAlgorithm)
    );
    assert!(is_zero_unsigned_be(&signature));
    signature.fill(0xa5);
    assert_eq!(
      key.sign_cose_algorithm_id(-7, message, &mut signature),
      Err(RsaPrivateOpError::UnsupportedAlgorithm)
    );
    assert!(is_zero_unsigned_be(&signature));
    signature.fill(0xa5);
    assert_eq!(
      key.sign_cose_algorithm_id_with_scratch(-7, message, &mut signature, &mut scratch),
      Err(RsaPrivateOpError::UnsupportedAlgorithm)
    );
    assert!(is_zero_unsigned_be(&signature));

    let sha1 = algorithm_identifier(SHA1_WITH_RSA_ENCRYPTION_OID, Some(&null()));
    signature.fill(0xa5);
    assert_eq!(
      key.sign_x509_signature_algorithm_der(&sha1, message, &mut signature),
      Err(RsaPrivateOpError::UnsupportedAlgorithm)
    );
    assert!(is_zero_unsigned_be(&signature));
    signature.fill(0xa5);
    assert_eq!(
      key.sign_x509_signature_algorithm_der_with_scratch(&sha1, message, &mut signature, &mut scratch),
      Err(RsaPrivateOpError::UnsupportedAlgorithm)
    );
    assert!(is_zero_unsigned_be(&signature));
    signature.fill(0xa5);
    assert_eq!(
      key.sign_x509_signature_algorithm_der(&[0x30, 0x00], message, &mut signature),
      Err(RsaPrivateOpError::UnsupportedAlgorithm)
    );
    assert!(is_zero_unsigned_be(&signature));
    signature.fill(0xa5);
    assert_eq!(
      key.sign_x509_signature_algorithm_der_with_scratch(&[0x30, 0x00], message, &mut signature, &mut scratch),
      Err(RsaPrivateOpError::UnsupportedAlgorithm)
    );
    assert!(is_zero_unsigned_be(&signature));
  }

  #[cfg(feature = "getrandom")]
  #[test]
  fn private_key_random_blinding_factor_has_valid_crt_inverse() {
    let der = test_pkcs1_private_key();
    let key = RsaPrivateKey::from_pkcs1_der_with_policy(&der, &RsaPublicKeyPolicy::legacy_verification()).unwrap();
    let blinding = key.components.random_blinding_factor().unwrap();

    let mut check = vec![0u8; key.signature_len()];
    mod_mul_representatives(
      &key.public_key().modulus,
      blinding.factor(),
      blinding.inverse(),
      &mut check,
    )
    .unwrap();

    assert!(check[..check.len().strict_sub(1)].iter().all(|&byte| byte == 0));
    assert_eq!(check.last().copied(), Some(1));
  }

  #[cfg(feature = "getrandom")]
  #[test]
  fn private_key_blinding_inverse_rejects_non_invertible_factor() {
    let der = test_pkcs1_private_key();
    let key = RsaPrivateKey::from_pkcs1_der_with_policy(&der, &RsaPublicKeyPolicy::legacy_verification()).unwrap();
    let mut factor = vec![0u8; key.signature_len()];
    left_pad_be(&rsa_private_prime_p(), &mut factor).unwrap();
    let mut inverse = vec![0u8; key.signature_len()];

    assert_eq!(
      key.components.blinding_factor_inverse(&factor, &mut inverse),
      Err(RsaPrivateOpError::InvalidBlindingFactor)
    );
  }

  #[test]
  fn private_key_caller_scratch_covers_signing_and_decryption() {
    let der = test_pkcs1_private_key();
    let key = RsaPrivateKey::from_pkcs1_der_with_policy(&der, &RsaPublicKeyPolicy::legacy_verification()).unwrap();
    let (blinding_factor, blinding_factor_inverse) = factor_two_and_inverse(key.public_key().modulus());
    let mut scratch = key.private_scratch();
    let message = b"rscrypto RSA private scratch path";

    let mut pkcs1v15_signature = vec![0u8; key.signature_len()];
    key
      .sign_pkcs1v15_with_blinding_factor_and_scratch(
        RsaPkcs1v15Profile::Sha256,
        message,
        &blinding_factor,
        &blinding_factor_inverse,
        &mut pkcs1v15_signature,
        &mut scratch,
      )
      .unwrap();
    key
      .public_key()
      .verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, message, &pkcs1v15_signature)
      .unwrap();

    let salt = [0x42; Sha384::OUTPUT_SIZE];
    let mut pss_signature = vec![0u8; key.signature_len()];
    key
      .sign_pss_with_salt_and_blinding_factor_and_scratch(
        RsaPssProfile::Sha384,
        message,
        &salt,
        &blinding_factor,
        &blinding_factor_inverse,
        &mut pss_signature,
        &mut scratch,
      )
      .unwrap();
    key
      .public_key()
      .verify_pss(RsaPssProfile::Sha384, message, &pss_signature)
      .unwrap();

    let label = b"rscrypto-private-scratch-oaep";
    let plaintext = b"private scratch OAEP roundtrip";
    let seed = [0x24; Sha256::OUTPUT_SIZE];
    let mut ciphertext = vec![0u8; key.signature_len()];
    key
      .public_key()
      .encrypt_oaep_with_seed(RsaOaepProfile::Sha256, label, plaintext, &seed, &mut ciphertext)
      .unwrap();
    let mut decrypted = vec![0u8; key.signature_len()];
    let decrypted_len = key
      .decrypt_oaep_with_blinding_factor_and_scratch(
        RsaOaepProfile::Sha256,
        label,
        &ciphertext,
        &blinding_factor,
        &blinding_factor_inverse,
        &mut decrypted,
        &mut scratch,
      )
      .unwrap();
    assert_eq!(&decrypted[..decrypted_len], plaintext);

    let pkcs1v15_plaintext = b"private scratch RSAES-PKCS1-v1_5 roundtrip";
    let pkcs1v15_seed = vec![0x5a; key.signature_len().strict_sub(pkcs1v15_plaintext.len()).strict_sub(3)];
    key
      .public_key()
      .encrypt_pkcs1v15_with_seed(pkcs1v15_plaintext, &pkcs1v15_seed, &mut ciphertext)
      .unwrap();
    let decrypted_len = key
      .decrypt_pkcs1v15_with_blinding_factor_and_scratch(
        &ciphertext,
        &blinding_factor,
        &blinding_factor_inverse,
        &mut decrypted,
        &mut scratch,
      )
      .unwrap();
    assert_eq!(&decrypted[..decrypted_len], pkcs1v15_plaintext);

    let mut smaller_scratch = wrong_width_private_scratch(&key);
    assert_eq!(
      key.sign_pkcs1v15_with_blinding_factor_and_scratch(
        RsaPkcs1v15Profile::Sha256,
        message,
        &blinding_factor,
        &blinding_factor_inverse,
        &mut pkcs1v15_signature,
        &mut smaller_scratch,
      ),
      Err(RsaPrivateOpError::InvalidScratch)
    );
  }

  #[test]
  fn private_key_caller_scratch_rejects_wrong_width_before_operation() {
    let der = test_pkcs1_private_key();
    let key = RsaPrivateKey::from_pkcs1_der_with_policy(&der, &RsaPublicKeyPolicy::legacy_verification()).unwrap();
    let (blinding_factor, blinding_factor_inverse) = factor_two_and_inverse(key.public_key().modulus());
    let message = b"rscrypto RSA private scratch shape";
    let mut signature = vec![0xa5; key.signature_len()];
    let mut scratch = wrong_width_private_scratch(&key);

    assert_eq!(
      key.sign_pkcs1v15_with_blinding_factor_and_scratch(
        RsaPkcs1v15Profile::Sha256,
        message,
        &blinding_factor,
        &blinding_factor_inverse,
        &mut signature,
        &mut scratch,
      ),
      Err(RsaPrivateOpError::InvalidScratch)
    );
    assert!(is_zero_unsigned_be(&signature));
  }

  #[cfg(feature = "getrandom")]
  #[test]
  fn keygen_contract_is_explicitly_fips_186_5_a1_3_probable_prime() {
    assert_eq!(
      RsaPrivateKey::GENERATION_CONTRACT,
      RsaKeyGenerationContract::Fips1865A13ProbablePrime
    );
  }

  #[cfg(feature = "getrandom")]
  #[test]
  fn keygen_random_candidate_has_fips_a1_3_raw_shape() {
    let mut drbg = test_keygen_drbg(b"candidate-shape");
    for bits in [2usize, 3, 9, 128, 1024] {
      let candidate = keygen_random_odd_candidate(&mut drbg, bits);
      assert_eq!(candidate.len(), bits.strict_add(7) / 8);
      assert_eq!(candidate.last().copied().unwrap_or_default() & 1, 1);
      assert!(unsigned_be_bit_len(&candidate) <= bits);
    }
  }

  #[cfg(feature = "getrandom")]
  #[test]
  fn keygen_fips_a1_3_lower_bound_matches_square_test_at_boundary() {
    let below = [0xb5, 0x04, 0xf3, 0x33, 0xf9, 0xde, 0x64, 0x83];
    let at_or_above = [0xb5, 0x04, 0xf3, 0x33, 0xf9, 0xde, 0x64, 0x84];
    let above = [0xb5, 0x04, 0xf3, 0x33, 0xf9, 0xde, 0x64, 0x85];

    assert!(!keygen_probable_prime_meets_fips_lower_bound(&below, 64));
    assert_eq!(
      keygen_probable_prime_meets_fips_lower_bound(&at_or_above, 64),
      private_import_product_unsigned_be(&at_or_above, &at_or_above)
        .as_ref()
        .is_some_and(|square| unsigned_be_bit_len(square.as_slice()) >= 128)
    );
    assert!(keygen_probable_prime_meets_fips_lower_bound(&above, 64));
  }

  #[cfg(feature = "getrandom")]
  #[test]
  fn keygen_generated_prime_satisfies_fips_a1_3_prime_constraints() {
    let mut drbg = test_keygen_drbg(b"generated-prime-constraints");
    let prime = keygen_generate_prime(&mut drbg, 128, 256, None, 256usize.strict_mul(5)).unwrap();

    assert!(keygen_probable_prime_meets_fips_lower_bound(&prime, 128));
    assert_eq!(prime.last().copied().unwrap_or_default() & 1, 1);
    assert!(!keygen_has_small_prime_factor(&prime));
    assert!(!keygen_conflicts_with_public_exponent(&prime));
    assert!(keygen_is_probable_prime(&mut drbg, &prime).unwrap());
  }

  #[cfg(feature = "getrandom")]
  #[test]
  fn keygen_lcm_private_exponent_contract_is_enforced() {
    let p_minus_one = private_import_decrement_unsigned_be(&rsa_private_prime_p()).unwrap();
    let q_minus_one = private_import_decrement_unsigned_be(&rsa_private_prime_q()).unwrap();
    let lambda = keygen_lcm_unsigned_be(p_minus_one.as_slice(), q_minus_one.as_slice()).unwrap();
    let phi = private_import_product_unsigned_be(p_minus_one.as_slice(), q_minus_one.as_slice()).unwrap();

    assert_eq!(
      private_import_unsigned_be_mod(phi.as_slice(), lambda.as_slice()).as_slice(),
      [0]
    );
    assert!(unsigned_be_cmp(lambda.as_slice(), phi.as_slice()) != core::cmp::Ordering::Greater);
    assert!(keygen_private_exponent_is_large_enough(
      &rsa_private_exponent(),
      unsigned_be_bit_len(&rsa_private_modulus())
    ));
  }

  #[cfg(feature = "getrandom")]
  #[test]
  fn keygen_candidate_shape_helper_tracks_fips_search_acceptance_shape() {
    let mut drbg = test_keygen_drbg(b"accepted-candidate-shape");
    let mut accepted = None;
    while accepted.is_none() {
      let candidate = keygen_random_odd_candidate(&mut drbg, 128);
      if keygen_probable_prime_meets_fips_lower_bound(&candidate, 128) {
        accepted = Some(candidate);
      }
    }
    let candidate = accepted.unwrap();
    assert!(
      keygen_candidate_has_fixed_shape(&candidate, 128),
      "accepted FIPS A.1.3 candidate must be odd and full-width"
    );
  }

  #[cfg(feature = "getrandom")]
  #[test]
  fn keygen_prefilter_rejects_small_prime_factors_without_rejecting_larger_prime() {
    assert!(keygen_has_small_prime_factor(&999u16.to_be_bytes()));
    assert!(!keygen_has_small_prime_factor(&1009u16.to_be_bytes()));
  }

  #[cfg(feature = "getrandom")]
  #[test]
  fn keygen_public_exponent_conflict_detects_p_minus_one_not_coprime_to_e() {
    let conflicting = RSA_KEYGEN_PUBLIC_EXPONENT.strict_add(1).to_be_bytes();
    assert!(keygen_conflicts_with_public_exponent(&conflicting));
    assert!(!keygen_conflicts_with_public_exponent(
      &RSA_KEYGEN_PUBLIC_EXPONENT.to_be_bytes()
    ));
  }

  #[cfg(feature = "getrandom")]
  #[test]
  fn keygen_miller_rabin_uses_fixed_round_count_above_fips_table_minimums() {
    assert_eq!(keygen_miller_rabin_rounds(1024), 32);
    assert_eq!(keygen_miller_rabin_rounds(1536), 32);
    assert_eq!(keygen_miller_rabin_rounds(2048), 32);

    assert!(keygen_miller_rabin_rounds(1024) >= 5);
    assert!(keygen_miller_rabin_rounds(1536) >= 4);
    assert!(keygen_miller_rabin_rounds(2048) >= 4);
  }

  #[cfg(feature = "getrandom")]
  #[test]
  fn keygen_miller_rabin_accepts_prime_and_rejects_composite_for_fixed_bases() {
    let prime = 1009u16.to_be_bytes();
    let prime_modulus = private_component_modulus(&prime).unwrap();
    assert!(
      keygen_miller_rabin_accepts_base(&prime_modulus, &[63], 4, &1008u16.to_be_bytes(), &[0, 11]).unwrap(),
      "1009 must pass a direct Miller-Rabin round for base 11"
    );

    let composite = 341u16.to_be_bytes();
    let composite_modulus = private_component_modulus(&composite).unwrap();
    assert!(
      !keygen_miller_rabin_accepts_base(&composite_modulus, &[85], 2, &340u16.to_be_bytes(), &[0, 2]).unwrap(),
      "341 must fail a direct Miller-Rabin round for base 2"
    );
  }

  #[cfg(feature = "getrandom")]
  #[test]
  fn keygen_derives_private_components_from_fixture_primes_end_to_end() {
    let modulus = rsa_private_modulus();
    let components = keygen_build_private_key_from_primes(
      unsigned_be_bit_len(&modulus),
      &RsaPublicKeyPolicy::legacy_verification(),
      rsa_private_prime_p(),
      rsa_private_prime_q(),
    )
    .unwrap()
    .unwrap();
    let key = RsaPrivateKey { components };
    assert_eq!(key.public_key().modulus(), modulus);
    assert_eq!(key.public_key().public_exponent().as_u64(), RSA_KEYGEN_PUBLIC_EXPONENT);

    let exported = key.to_pkcs1_der();
    let imported =
      RsaPrivateKey::from_pkcs1_der_with_policy(&exported, &RsaPublicKeyPolicy::legacy_verification()).unwrap();
    assert_eq!(imported.public_key(), key.public_key());

    let message = b"rscrypto generated RSA component signing roundtrip";
    let (blinding_factor, blinding_factor_inverse) = factor_two_and_inverse(key.public_key().modulus());
    let mut signature = vec![0u8; key.signature_len()];
    key
      .sign_pkcs1v15_with_blinding_factor(
        RsaPkcs1v15Profile::Sha256,
        message,
        &blinding_factor,
        &blinding_factor_inverse,
        &mut signature,
      )
      .unwrap();
    key
      .public_key()
      .verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, message, &signature)
      .unwrap();

    let label = b"rscrypto-keygen-oaep";
    let plaintext = b"generated component oaep roundtrip";
    let seed = [0x7b; Sha256::OUTPUT_SIZE];
    let mut ciphertext = vec![0u8; key.signature_len()];
    key
      .public_key()
      .encrypt_oaep_with_seed(RsaOaepProfile::Sha256, label, plaintext, &seed, &mut ciphertext)
      .unwrap();
    let mut decrypted = vec![0u8; key.signature_len()];
    let decrypted_len = key
      .decrypt_oaep_with_blinding_factor(
        RsaOaepProfile::Sha256,
        label,
        &ciphertext,
        &blinding_factor,
        &blinding_factor_inverse,
        &mut decrypted,
      )
      .unwrap();
    assert_eq!(&decrypted[..decrypted_len], plaintext);

    let pkcs1v15_plaintext = b"generated component RSAES-PKCS1-v1_5 roundtrip";
    let pkcs1v15_seed = vec![0x3c; key.signature_len().strict_sub(pkcs1v15_plaintext.len()).strict_sub(3)];
    key
      .public_key()
      .encrypt_pkcs1v15_with_seed(pkcs1v15_plaintext, &pkcs1v15_seed, &mut ciphertext)
      .unwrap();
    let decrypted_len = key
      .decrypt_pkcs1v15_with_blinding_factor(&ciphertext, &blinding_factor, &blinding_factor_inverse, &mut decrypted)
      .unwrap();
    assert_eq!(&decrypted[..decrypted_len], pkcs1v15_plaintext);
  }

  #[cfg(feature = "getrandom")]
  #[test]
  fn keygen_derives_private_components_from_nist_cavp_probable_primes() {
    let suite: Value = serde_json::from_str(CAVP_KEYGEN_186_3_PROBABLE_PRIME).expect("CAVP keygen JSON must parse");
    assert_eq!(
      suite["source_files"][0].as_str(),
      Some("KeyGen_186-3_RandomProbablyPrime3_3_KAT.txt")
    );
    assert_eq!(suite["counts"]["total"].as_u64(), Some(2));

    let tests = suite["tests"].as_array().expect("CAVP keygen tests must be an array");
    let policy = RsaPublicKeyPolicy::legacy_verification();
    let mut covered = Vec::new();

    for test in tests {
      let modulus_bits = test["mod"].as_u64().expect("CAVP modulus size must be numeric") as usize;
      let components = keygen_build_private_key_from_primes(
        modulus_bits,
        &policy,
        hex_to_vec(json_field(test, "p")),
        hex_to_vec(json_field(test, "q")),
      )
      .expect("CAVP probable-prime candidates must derive cleanly")
      .expect("CAVP probable-prime candidates must produce the requested modulus width");
      let key = RsaPrivateKey { components };
      assert_eq!(key.public_key().modulus_bits(), modulus_bits);
      assert_eq!(key.public_key().public_exponent().as_u64(), RSA_KEYGEN_PUBLIC_EXPONENT);
      let mut primality_drbg = test_keygen_drbg(b"cavp-primality");
      assert!(keygen_is_probable_prime(&mut primality_drbg, key.components.prime_p.as_bytes()).unwrap());
      assert!(keygen_is_probable_prime(&mut primality_drbg, key.components.prime_q.as_bytes()).unwrap());

      let (blinding_factor, blinding_factor_inverse) = factor_two_and_inverse(key.public_key().modulus());
      let message = b"rscrypto NIST CAVP keygen candidate private operation";

      let mut pkcs1v15_signature = vec![0u8; key.signature_len()];
      key
        .sign_pkcs1v15_with_blinding_factor(
          RsaPkcs1v15Profile::Sha256,
          message,
          &blinding_factor,
          &blinding_factor_inverse,
          &mut pkcs1v15_signature,
        )
        .unwrap();
      key
        .public_key()
        .verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, message, &pkcs1v15_signature)
        .unwrap();

      let salt = [0xa5; Sha256::OUTPUT_SIZE];
      let mut pss_signature = vec![0u8; key.signature_len()];
      key
        .sign_pss_with_salt_and_blinding_factor(
          RsaPssProfile::Sha256,
          message,
          &salt,
          &blinding_factor,
          &blinding_factor_inverse,
          &mut pss_signature,
        )
        .unwrap();
      key
        .public_key()
        .verify_pss(RsaPssProfile::Sha256, message, &pss_signature)
        .unwrap();

      let label = b"rscrypto-cavp-keygen-oaep";
      let plaintext = b"NIST CAVP keygen candidate OAEP";
      let seed = [0x3d; Sha256::OUTPUT_SIZE];
      let mut ciphertext = vec![0u8; key.signature_len()];
      key
        .public_key()
        .encrypt_oaep_with_seed(RsaOaepProfile::Sha256, label, plaintext, &seed, &mut ciphertext)
        .unwrap();
      let mut decrypted = vec![0u8; key.signature_len()];
      let decrypted_len = key
        .decrypt_oaep_with_blinding_factor(
          RsaOaepProfile::Sha256,
          label,
          &ciphertext,
          &blinding_factor,
          &blinding_factor_inverse,
          &mut decrypted,
        )
        .unwrap();
      assert_eq!(&decrypted[..decrypted_len], plaintext);

      let pkcs1v15_plaintext = b"NIST CAVP keygen candidate RSAES-PKCS1-v1_5";
      let pkcs1v15_seed = vec![0x69; key.signature_len().strict_sub(pkcs1v15_plaintext.len()).strict_sub(3)];
      key
        .public_key()
        .encrypt_pkcs1v15_with_seed(pkcs1v15_plaintext, &pkcs1v15_seed, &mut ciphertext)
        .unwrap();
      let decrypted_len = key
        .decrypt_pkcs1v15_with_blinding_factor(&ciphertext, &blinding_factor, &blinding_factor_inverse, &mut decrypted)
        .unwrap();
      assert_eq!(&decrypted[..decrypted_len], pkcs1v15_plaintext);

      covered.push(modulus_bits);
    }

    assert_eq!(covered, [2048, 3072]);
  }

  #[cfg(feature = "getrandom")]
  #[test]
  fn keygen_generate_with_policy_produces_usable_private_key_end_to_end() {
    let key = RsaPrivateKey::generate_with_policy(2048, &RsaPublicKeyPolicy::legacy_verification()).unwrap();
    assert_eq!(key.public_key().modulus_bits(), 2048);
    assert_eq!(key.public_key().public_exponent().as_u64(), RSA_KEYGEN_PUBLIC_EXPONENT);

    let pkcs1 = key.to_pkcs1_der();
    let pkcs8 = key.to_pkcs8_der();
    let pkcs1_imported =
      RsaPrivateKey::from_pkcs1_der_with_policy(&pkcs1, &RsaPublicKeyPolicy::legacy_verification()).unwrap();
    let pkcs8_imported =
      RsaPrivateKey::from_pkcs8_der_with_policy(&pkcs8, &RsaPublicKeyPolicy::legacy_verification()).unwrap();
    assert_eq!(pkcs1_imported.public_key(), key.public_key());
    assert_eq!(pkcs8_imported.public_key(), key.public_key());

    macro_rules! assert_generated_key_profile {
      ($pkcs1_profile:expr, $pss_profile:expr, $oaep_profile:expr, $seed:expr, $plaintext:expr) => {{
        let message = b"rscrypto generated RSA key signs end to end";
        let mut pkcs1v15_signature = vec![0u8; key.signature_len()];
        key
          .sign_pkcs1v15($pkcs1_profile, message, &mut pkcs1v15_signature)
          .unwrap();
        key
          .public_key()
          .verify_pkcs1v15($pkcs1_profile, message, &pkcs1v15_signature)
          .unwrap();

        let mut pss_signature = vec![0u8; key.signature_len()];
        key.sign_pss($pss_profile, message, &mut pss_signature).unwrap();
        key
          .public_key()
          .verify_pss($pss_profile, message, &pss_signature)
          .unwrap();

        let label = b"rscrypto-generated-key-oaep";
        let mut ciphertext = vec![0u8; key.signature_len()];
        key
          .public_key()
          .encrypt_oaep_with_seed($oaep_profile, label, $plaintext, &$seed, &mut ciphertext)
          .unwrap();
        let mut decrypted = vec![0u8; key.signature_len()];
        let decrypted_len = key
          .decrypt_oaep($oaep_profile, label, &ciphertext, &mut decrypted)
          .unwrap();
        assert_eq!(&decrypted[..decrypted_len], $plaintext);

        let pkcs1v15_plaintext = b"generated key RSAES-PKCS1-v1_5";
        let pkcs1v15_seed = vec![0x63; key.signature_len().strict_sub(pkcs1v15_plaintext.len()).strict_sub(3)];
        key
          .public_key()
          .encrypt_pkcs1v15_with_seed(pkcs1v15_plaintext, &pkcs1v15_seed, &mut ciphertext)
          .unwrap();
        let decrypted_len = key.decrypt_pkcs1v15(&ciphertext, &mut decrypted).unwrap();
        assert_eq!(&decrypted[..decrypted_len], pkcs1v15_plaintext);
      }};
    }

    assert_generated_key_profile!(
      RsaPkcs1v15Profile::Sha256,
      RsaPssProfile::Sha256,
      RsaOaepProfile::Sha256,
      [0x26; Sha256::OUTPUT_SIZE],
      b"generated key sha256 oaep"
    );
    assert_generated_key_profile!(
      RsaPkcs1v15Profile::Sha384,
      RsaPssProfile::Sha384,
      RsaOaepProfile::Sha384,
      [0x38; Sha384::OUTPUT_SIZE],
      b"generated key sha384 oaep"
    );
    assert_generated_key_profile!(
      RsaPkcs1v15Profile::Sha512,
      RsaPssProfile::Sha512,
      RsaOaepProfile::Sha512,
      [0x52; Sha512::OUTPUT_SIZE],
      b"generated key sha512 oaep"
    );
  }

  #[cfg(feature = "getrandom")]
  #[test]
  fn keygen_generate_default_modern_key_produces_usable_private_key_end_to_end() {
    let key = RsaPrivateKey::generate(3072).unwrap();
    assert_eq!(key.public_key().modulus_bits(), 3072);
    assert_eq!(key.public_key().public_exponent().as_u64(), RSA_KEYGEN_PUBLIC_EXPONENT);

    let pkcs1 = key.to_pkcs1_der();
    let pkcs8 = key.to_pkcs8_der();
    assert_eq!(
      RsaPrivateKey::from_pkcs1_der(&pkcs1).unwrap().public_key(),
      key.public_key()
    );
    assert_eq!(
      RsaPrivateKey::from_pkcs8_der(&pkcs8).unwrap().public_key(),
      key.public_key()
    );

    let mut scratch = key.private_scratch();
    let message = b"rscrypto generated modern RSA key end to end";
    let mut signature = vec![0u8; key.signature_len()];

    let pkcs1v15_profile = RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256);
    key
      .sign_signature_with_scratch(pkcs1v15_profile, message, &mut signature, &mut scratch)
      .unwrap();
    key
      .public_key()
      .verify_signature(pkcs1v15_profile, message, &signature)
      .unwrap();

    let pss_profile = RsaSignatureProfile::pss(RsaPssProfile::Sha256);
    key
      .sign_signature_with_scratch(pss_profile, message, &mut signature, &mut scratch)
      .unwrap();
    key
      .public_key()
      .verify_signature(pss_profile, message, &signature)
      .unwrap();

    let label = b"rscrypto-generated-modern-key-oaep";
    let plaintext = b"modern generated key OAEP roundtrip";
    let seed = [0x30; Sha256::OUTPUT_SIZE];
    let mut ciphertext = vec![0u8; key.signature_len()];
    key
      .public_key()
      .encrypt_oaep_with_seed(RsaOaepProfile::Sha256, label, plaintext, &seed, &mut ciphertext)
      .unwrap();
    let mut decrypted = vec![0u8; key.signature_len()];
    let decrypted_len = key
      .decrypt_oaep_with_scratch(RsaOaepProfile::Sha256, label, &ciphertext, &mut decrypted, &mut scratch)
      .unwrap();
    assert_eq!(&decrypted[..decrypted_len], plaintext);

    let pkcs1v15_plaintext = b"modern generated key RSAES-PKCS1-v1_5";
    let pkcs1v15_seed = vec![0x64; key.signature_len().strict_sub(pkcs1v15_plaintext.len()).strict_sub(3)];
    key
      .public_key()
      .encrypt_pkcs1v15_with_seed(pkcs1v15_plaintext, &pkcs1v15_seed, &mut ciphertext)
      .unwrap();
    let decrypted_len = key
      .decrypt_pkcs1v15_with_scratch(&ciphertext, &mut decrypted, &mut scratch)
      .unwrap();
    assert_eq!(&decrypted[..decrypted_len], pkcs1v15_plaintext);
  }

  #[cfg(feature = "getrandom")]
  #[test]
  fn keygen_generate_rejects_policy_disallowed_sizes_before_entropy() {
    assert_eq!(
      RsaPrivateKey::generate(2048).err(),
      Some(RsaKeyGenerationError::InvalidModulusBits)
    );
    assert_eq!(
      RsaPrivateKey::generate(3071).err(),
      Some(RsaKeyGenerationError::InvalidModulusBits)
    );
    assert_eq!(
      RsaPrivateKey::generate(8193).err(),
      Some(RsaKeyGenerationError::InvalidModulusBits)
    );

    let impossible_policy = RsaPublicKeyPolicy::legacy_verification()
      .with_min_modulus_bits(4096)
      .with_max_modulus_bits(3072);
    assert_eq!(
      RsaPrivateKey::generate_with_policy(4096, &impossible_policy).err(),
      Some(RsaKeyGenerationError::InvalidModulusBits)
    );

    let undersized_policy = RsaPublicKeyPolicy::legacy_verification().with_min_modulus_bits(3072);
    assert_eq!(
      RsaPrivateKey::generate_with_policy(2048, &undersized_policy).err(),
      Some(RsaKeyGenerationError::InvalidModulusBits)
    );
  }

  #[cfg(feature = "getrandom")]
  #[test]
  fn keygen_build_private_key_from_primes_rejects_duplicate_and_wrong_width_candidates() {
    let modulus_bits = unsigned_be_bit_len(&rsa_private_modulus());
    let policy = RsaPublicKeyPolicy::legacy_verification();

    assert!(
      keygen_build_private_key_from_primes(modulus_bits, &policy, rsa_private_prime_p(), rsa_private_prime_p())
        .unwrap()
        .is_none()
    );
    assert!(
      keygen_build_private_key_from_primes(
        modulus_bits.strict_sub(1),
        &policy,
        rsa_private_prime_p(),
        rsa_private_prime_q()
      )
      .unwrap()
      .is_none()
    );
    assert!(
      keygen_build_private_key_from_primes(
        modulus_bits.strict_add(1),
        &policy,
        rsa_private_prime_p(),
        rsa_private_prime_q()
      )
      .unwrap()
      .is_none()
    );

    let mut close_prime_q = rsa_private_prime_p();
    let mut carry = 2u16;
    for byte in close_prime_q.iter_mut().rev() {
      let sum = u16::from(*byte).strict_add(carry);
      *byte = sum as u8;
      carry = sum >> 8;
      if carry == 0 {
        break;
      }
    }
    assert_eq!(carry, 0);
    assert!(
      !keygen_prime_distance_is_sufficient(&rsa_private_prime_p(), &close_prime_q, modulus_bits),
      "key generation must reject dangerously close prime candidates"
    );
    assert!(
      keygen_build_private_key_from_primes(modulus_bits, &policy, rsa_private_prime_p(), close_prime_q)
        .unwrap()
        .is_none()
    );

    let mut public_exponent_conflict = rsa_private_prime_p();
    let remainder = unsigned_be_mod_u64(&public_exponent_conflict, RSA_KEYGEN_PUBLIC_EXPONENT);
    let mut delta = (RSA_KEYGEN_PUBLIC_EXPONENT.strict_add(1).strict_sub(remainder)) % RSA_KEYGEN_PUBLIC_EXPONENT;
    for byte in public_exponent_conflict.iter_mut().rev() {
      if delta == 0 {
        break;
      }
      let sum = u64::from(*byte).strict_add(delta & 0xff);
      *byte = sum as u8;
      delta = (delta >> 8).strict_add(sum >> 8);
    }
    assert_eq!(delta, 0);
    assert!(
      keygen_conflicts_with_public_exponent(&public_exponent_conflict),
      "test candidate must conflict with RSA key-generation public exponent"
    );
    assert!(
      keygen_build_private_key_from_primes(modulus_bits, &policy, public_exponent_conflict, rsa_private_prime_q())
        .unwrap()
        .is_none()
    );
  }

  #[cfg(feature = "getrandom")]
  #[test]
  fn keygen_random_prime_search_returns_probable_prime() {
    let mut drbg = test_keygen_drbg(b"random-prime-search");
    let prime = keygen_generate_prime(&mut drbg, 128, 256, None, 256usize.strict_mul(5)).unwrap();

    assert_eq!(unsigned_be_bit_len(&prime), 128);
    assert_eq!(prime.last().copied().unwrap_or_default() & 1, 1);
    assert!(!keygen_has_small_prime_factor(&prime));
    assert!(!keygen_conflicts_with_public_exponent(&prime));
    assert!(keygen_probable_prime_meets_fips_lower_bound(&prime, 128));
    assert!(keygen_is_probable_prime(&mut drbg, &prime).unwrap());
  }

  #[test]
  fn pkcs1_private_key_parser_applies_modulus_policy_before_component_checks() {
    let mut bad_q = rsa_private_prime_q();
    *bad_q.last_mut().unwrap() ^= 0x02;
    let der = test_pkcs1_private_key_with_components(
      &[0x01, 0x00, 0x01],
      &rsa_private_exponent(),
      &rsa_private_prime_p(),
      &bad_q,
      &rsa_private_exponent_p(),
      &rsa_private_exponent_q(),
      &rsa_private_coefficient(),
    );
    let policy = RsaPublicKeyPolicy::legacy_verification().with_max_modulus_bits(1024);

    assert_eq!(
      parse_pkcs1_private_key_der_with_policy(&der, &policy).err(),
      Some(RsaKeyError::InvalidModulus)
    );
  }

  #[test]
  fn pkcs8_private_key_parser_requires_rsa_algorithm_and_strict_inner_key() {
    let pkcs1 = test_pkcs1_private_key();
    let rsa_algorithm = algorithm_identifier(RSA_ENCRYPTION_OID, Some(&null()));
    let der = test_pkcs8_private_key(&pkcs1, &rsa_algorithm);
    let key = parse_pkcs8_private_key_der_with_policy(&der, &RsaPublicKeyPolicy::legacy_verification()).unwrap();

    assert_eq!(key.public_key().modulus(), rsa_private_modulus());

    let pss_algorithm = algorithm_identifier(ID_RSASSA_PSS_OID, Some(&tlv(TAG_SEQUENCE, &[])));
    let wrong_algorithm = test_pkcs8_private_key(&pkcs1, &pss_algorithm);
    assert_eq!(
      parse_pkcs8_private_key_der_with_policy(&wrong_algorithm, &RsaPublicKeyPolicy::legacy_verification()).err(),
      Some(RsaKeyError::UnsupportedAlgorithm)
    );
  }

  #[test]
  fn private_key_parser_rejects_noncanonical_container_lengths_and_attributes() {
    let pkcs1 = test_pkcs1_private_key();
    assert_eq!(
      parse_pkcs1_private_key_der_with_policy(
        &tlv_with_leading_zero_long_len(&pkcs1),
        &RsaPublicKeyPolicy::legacy_verification()
      )
      .err(),
      Some(RsaKeyError::MalformedDer)
    );

    let mut pkcs1_body = Vec::new();
    for field in [
      tlv_with_noncanonical_short_len(TAG_INTEGER, &[0]),
      integer_unsigned(&rsa_private_modulus()),
      integer_unsigned(&[0x01, 0x00, 0x01]),
      integer_unsigned(&rsa_private_exponent()),
      integer_unsigned(&rsa_private_prime_p()),
      integer_unsigned(&rsa_private_prime_q()),
      integer_unsigned(&rsa_private_exponent_p()),
      integer_unsigned(&rsa_private_exponent_q()),
      integer_unsigned(&rsa_private_coefficient()),
    ] {
      pkcs1_body.extend_from_slice(&field);
    }
    assert_eq!(
      parse_pkcs1_private_key_der_with_policy(
        &tlv(TAG_SEQUENCE, &pkcs1_body),
        &RsaPublicKeyPolicy::legacy_verification()
      )
      .err(),
      Some(RsaKeyError::MalformedDer)
    );

    let rsa_algorithm = algorithm_identifier(RSA_ENCRYPTION_OID, Some(&null()));
    let pkcs8 = test_pkcs8_private_key(&pkcs1, &rsa_algorithm);
    assert_eq!(
      parse_pkcs8_private_key_der_with_policy(
        &tlv_with_leading_zero_long_len(&pkcs8),
        &RsaPublicKeyPolicy::legacy_verification()
      )
      .err(),
      Some(RsaKeyError::MalformedDer)
    );

    let mut pkcs8_with_attributes = Vec::new();
    pkcs8_with_attributes.extend_from_slice(&integer_unsigned(&[0]));
    pkcs8_with_attributes.extend_from_slice(&rsa_algorithm);
    pkcs8_with_attributes.extend_from_slice(&tlv(TAG_OCTET_STRING, &pkcs1));
    pkcs8_with_attributes.extend_from_slice(&context_constructed(0, &[]));
    assert_eq!(
      parse_pkcs8_private_key_der_with_policy(
        &tlv(TAG_SEQUENCE, &pkcs8_with_attributes),
        &RsaPublicKeyPolicy::legacy_verification()
      )
      .err(),
      Some(RsaKeyError::MalformedDer)
    );
  }

  #[test]
  fn pkcs1_private_key_parser_rejects_multiprime_and_inconsistent_components() {
    let mut multiprime_body = Vec::new();
    multiprime_body.extend_from_slice(&integer_unsigned(&[1]));
    multiprime_body.extend_from_slice(&integer_unsigned(&rsa_private_modulus()));
    multiprime_body.extend_from_slice(&integer_unsigned(&[0x01, 0x00, 0x01]));
    multiprime_body.extend_from_slice(&integer_unsigned(&rsa_private_exponent()));
    multiprime_body.extend_from_slice(&integer_unsigned(&rsa_private_prime_p()));
    multiprime_body.extend_from_slice(&integer_unsigned(&rsa_private_prime_q()));
    multiprime_body.extend_from_slice(&integer_unsigned(&rsa_private_exponent_p()));
    multiprime_body.extend_from_slice(&integer_unsigned(&rsa_private_exponent_q()));
    multiprime_body.extend_from_slice(&integer_unsigned(&rsa_private_coefficient()));
    let multiprime = tlv(TAG_SEQUENCE, &multiprime_body);
    assert_eq!(
      parse_pkcs1_private_key_der_with_policy(&multiprime, &RsaPublicKeyPolicy::legacy_verification()).err(),
      Some(RsaKeyError::UnsupportedAlgorithm)
    );

    let mut bad_q = rsa_private_prime_q();
    let last = bad_q.last_mut().unwrap();
    *last ^= 0x02;
    let mut inconsistent_body = Vec::new();
    for field in [
      integer_unsigned(&[0]),
      integer_unsigned(&rsa_private_modulus()),
      integer_unsigned(&[0x01, 0x00, 0x01]),
      integer_unsigned(&rsa_private_exponent()),
      integer_unsigned(&rsa_private_prime_p()),
      integer_unsigned(&bad_q),
      integer_unsigned(&rsa_private_exponent_p()),
      integer_unsigned(&rsa_private_exponent_q()),
      integer_unsigned(&rsa_private_coefficient()),
    ] {
      inconsistent_body.extend_from_slice(&field);
    }
    let inconsistent = tlv(TAG_SEQUENCE, &inconsistent_body);
    assert_eq!(
      parse_pkcs1_private_key_der_with_policy(&inconsistent, &RsaPublicKeyPolicy::legacy_verification()).err(),
      Some(RsaKeyError::InvalidModulus)
    );
  }

  #[test]
  fn pkcs1_private_key_parser_rejects_crt_congruence_mismatches() {
    let mut bad_exponent_p = rsa_private_exponent_p();
    *bad_exponent_p.last_mut().unwrap() ^= 0x02;
    let bad_dp = test_pkcs1_private_key_with_crt(
      &[0x01, 0x00, 0x01],
      &bad_exponent_p,
      &rsa_private_exponent_q(),
      &rsa_private_coefficient(),
    );
    assert_eq!(
      parse_pkcs1_private_key_der_with_policy(&bad_dp, &RsaPublicKeyPolicy::legacy_verification()).err(),
      Some(RsaKeyError::InvalidModulus)
    );

    let mut bad_exponent_q = rsa_private_exponent_q();
    *bad_exponent_q.last_mut().unwrap() ^= 0x02;
    let bad_dq = test_pkcs1_private_key_with_crt(
      &[0x01, 0x00, 0x01],
      &rsa_private_exponent_p(),
      &bad_exponent_q,
      &rsa_private_coefficient(),
    );
    assert_eq!(
      parse_pkcs1_private_key_der_with_policy(&bad_dq, &RsaPublicKeyPolicy::legacy_verification()).err(),
      Some(RsaKeyError::InvalidModulus)
    );

    let mut bad_coefficient = rsa_private_coefficient();
    *bad_coefficient.last_mut().unwrap() ^= 0x02;
    let bad_qinv = test_pkcs1_private_key_with_crt(
      &[0x01, 0x00, 0x01],
      &rsa_private_exponent_p(),
      &rsa_private_exponent_q(),
      &bad_coefficient,
    );
    assert_eq!(
      parse_pkcs1_private_key_der_with_policy(&bad_qinv, &RsaPublicKeyPolicy::legacy_verification()).err(),
      Some(RsaKeyError::InvalidModulus)
    );
  }

  #[test]
  fn pkcs1_private_key_parser_rejects_invalid_secret_ranges() {
    let policy = RsaPublicKeyPolicy::legacy_verification();

    let zero_private_exponent = test_pkcs1_private_key_with_components(
      &[0x01, 0x00, 0x01],
      &[0],
      &rsa_private_prime_p(),
      &rsa_private_prime_q(),
      &rsa_private_exponent_p(),
      &rsa_private_exponent_q(),
      &rsa_private_coefficient(),
    );
    assert_eq!(
      parse_pkcs1_private_key_der_with_policy(&zero_private_exponent, &policy).err(),
      Some(RsaKeyError::InvalidModulus)
    );

    let modulus_sized_private_exponent = test_pkcs1_private_key_with_components(
      &[0x01, 0x00, 0x01],
      &rsa_private_modulus(),
      &rsa_private_prime_p(),
      &rsa_private_prime_q(),
      &rsa_private_exponent_p(),
      &rsa_private_exponent_q(),
      &rsa_private_coefficient(),
    );
    assert_eq!(
      parse_pkcs1_private_key_der_with_policy(&modulus_sized_private_exponent, &policy).err(),
      Some(RsaKeyError::InvalidModulus)
    );

    let zero_prime = test_pkcs1_private_key_with_components(
      &[0x01, 0x00, 0x01],
      &rsa_private_exponent(),
      &[0],
      &rsa_private_prime_q(),
      &rsa_private_exponent_p(),
      &rsa_private_exponent_q(),
      &rsa_private_coefficient(),
    );
    assert_eq!(
      parse_pkcs1_private_key_der_with_policy(&zero_prime, &policy).err(),
      Some(RsaKeyError::InvalidModulus)
    );

    let even_prime = test_pkcs1_private_key_with_components(
      &[0x01, 0x00, 0x01],
      &rsa_private_exponent(),
      &[2],
      &rsa_private_prime_q(),
      &rsa_private_exponent_p(),
      &rsa_private_exponent_q(),
      &rsa_private_coefficient(),
    );
    assert_eq!(
      parse_pkcs1_private_key_der_with_policy(&even_prime, &policy).err(),
      Some(RsaKeyError::InvalidModulus)
    );

    let zero_crt_exponent = test_pkcs1_private_key_with_components(
      &[0x01, 0x00, 0x01],
      &rsa_private_exponent(),
      &rsa_private_prime_p(),
      &rsa_private_prime_q(),
      &[0],
      &rsa_private_exponent_q(),
      &rsa_private_coefficient(),
    );
    assert_eq!(
      parse_pkcs1_private_key_der_with_policy(&zero_crt_exponent, &policy).err(),
      Some(RsaKeyError::InvalidModulus)
    );

    let out_of_range_coefficient = test_pkcs1_private_key_with_components(
      &[0x01, 0x00, 0x01],
      &rsa_private_exponent(),
      &rsa_private_prime_p(),
      &rsa_private_prime_q(),
      &rsa_private_exponent_p(),
      &rsa_private_exponent_q(),
      &rsa_private_prime_p(),
    );
    assert_eq!(
      parse_pkcs1_private_key_der_with_policy(&out_of_range_coefficient, &policy).err(),
      Some(RsaKeyError::InvalidModulus)
    );
  }

  #[test]
  fn pkcs1_private_key_parser_rejects_public_private_exponent_mismatch() {
    let wrong_exponent = test_pkcs1_private_key_with_public_exponent(&[3]);
    assert_eq!(
      parse_pkcs1_private_key_der_with_policy(
        &wrong_exponent,
        &RsaPublicKeyPolicy::legacy_verification().allow_legacy_small_exponents()
      )
      .err(),
      Some(RsaKeyError::InvalidModulus)
    );
  }

  #[test]
  fn pkcs1_private_key_parser_rejects_noncanonical_integer_and_trailing_data() {
    let mut noncanonical_exponent_p = rsa_private_exponent_p();
    noncanonical_exponent_p.insert(0, 0);
    let mut noncanonical_body = Vec::new();
    for field in [
      integer_unsigned(&[0]),
      integer_unsigned(&rsa_private_modulus()),
      integer_unsigned(&[0x01, 0x00, 0x01]),
      integer_unsigned(&rsa_private_exponent()),
      integer_unsigned(&rsa_private_prime_p()),
      integer_unsigned(&rsa_private_prime_q()),
      integer_der_value(&noncanonical_exponent_p),
      integer_unsigned(&rsa_private_exponent_q()),
      integer_unsigned(&rsa_private_coefficient()),
    ] {
      noncanonical_body.extend_from_slice(&field);
    }
    let noncanonical = tlv(TAG_SEQUENCE, &noncanonical_body);
    assert_eq!(
      parse_pkcs1_private_key_der_with_policy(&noncanonical, &RsaPublicKeyPolicy::legacy_verification()).err(),
      Some(RsaKeyError::MalformedDer)
    );

    let mut trailing = test_pkcs1_private_key();
    trailing.extend_from_slice(&null());
    assert_eq!(
      parse_pkcs1_private_key_der_with_policy(&trailing, &RsaPublicKeyPolicy::legacy_verification()).err(),
      Some(RsaKeyError::MalformedDer)
    );
  }

  #[test]
  fn pkcs1v15_sha2_digest_info_prefixes_match_rfc8017_der() {
    assert_eq!(
      SHA256_DIGEST_INFO_PREFIX,
      &[
        0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, 0x05, 0x00, 0x04,
        0x20,
      ],
    );
    assert_eq!(
      SHA384_DIGEST_INFO_PREFIX,
      &[
        0x30, 0x41, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x02, 0x05, 0x00, 0x04,
        0x30,
      ],
    );
    assert_eq!(
      SHA512_DIGEST_INFO_PREFIX,
      &[
        0x30, 0x51, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x03, 0x05, 0x00, 0x04,
        0x40,
      ],
    );
  }

  #[test]
  fn pkcs1v15_encoded_sha256_accepts_exact_der_digest_info() {
    let message = b"rsa-pkcs1v15-encoded-message";
    let encoded = pkcs1v15_encoded_sha256(message, 256);

    assert_eq!(
      verify_pkcs1v15_encoded::<Sha256>(message, &encoded, SHA256_DIGEST_INFO_PREFIX),
      Ok(())
    );
  }

  #[test]
  fn pkcs1v15_encoded_sha256_rejects_short_padding_and_bad_separator() {
    let message = b"rsa-pkcs1v15-encoded-message";
    let digest = Sha256::digest(message);
    let mut short = Vec::new();
    short.push(0);
    short.push(1);
    short.extend(core::iter::repeat_n(0xff, 7));
    short.push(0);
    short.extend_from_slice(SHA256_DIGEST_INFO_PREFIX);
    short.extend_from_slice(&digest);
    assert!(verify_pkcs1v15_encoded::<Sha256>(message, &short, SHA256_DIGEST_INFO_PREFIX).is_err());

    let mut bad_separator = pkcs1v15_encoded_sha256(message, 256);
    let separator = bad_separator
      .len()
      .strict_sub(SHA256_DIGEST_INFO_PREFIX.len())
      .strict_sub(Sha256::OUTPUT_SIZE)
      .strict_sub(1);
    if let Some(byte) = bad_separator.get_mut(separator) {
      *byte = 0xff;
    }
    assert!(verify_pkcs1v15_encoded::<Sha256>(message, &bad_separator, SHA256_DIGEST_INFO_PREFIX).is_err());
  }

  #[test]
  fn pkcs1v15_encoded_sha256_rejects_digest_info_oid_and_digest_mismatches() {
    let message = b"rsa-pkcs1v15-encoded-message";
    let mut bad_oid = pkcs1v15_encoded_sha256(message, 256);
    let digest_info = bad_oid
      .len()
      .strict_sub(SHA256_DIGEST_INFO_PREFIX.len())
      .strict_sub(Sha256::OUTPUT_SIZE);
    if let Some(byte) = bad_oid.get_mut(digest_info.strict_add(14)) {
      *byte ^= 0x01;
    }
    assert!(verify_pkcs1v15_encoded::<Sha256>(message, &bad_oid, SHA256_DIGEST_INFO_PREFIX).is_err());

    let mut bad_digest = pkcs1v15_encoded_sha256(message, 256);
    if let Some(byte) = bad_digest.last_mut() {
      *byte ^= 0x01;
    }
    assert!(verify_pkcs1v15_encoded::<Sha256>(message, &bad_digest, SHA256_DIGEST_INFO_PREFIX).is_err());
  }

  #[test]
  fn pkcs1v15_encoded_sha256_rejects_historical_forgery_shapes() {
    let message = b"rsa-pkcs1v15-encoded-message";
    let digest = Sha256::digest(message);

    let mut early_separator = Vec::new();
    early_separator.push(0);
    early_separator.push(1);
    early_separator.extend(core::iter::repeat_n(0xff, 8));
    early_separator.push(0);
    early_separator.extend_from_slice(SHA256_DIGEST_INFO_PREFIX);
    early_separator.extend_from_slice(&digest);
    early_separator.extend(core::iter::repeat_n(0xaa, 256usize.strict_sub(early_separator.len())));
    assert!(verify_pkcs1v15_encoded::<Sha256>(message, &early_separator, SHA256_DIGEST_INFO_PREFIX).is_err());

    let mut missing_null_prefix = [
      0x30, 0x2f, 0x30, 0x0b, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, 0x04, 0x20,
    ]
    .to_vec();
    missing_null_prefix.extend_from_slice(&digest);
    let mut missing_null = Vec::new();
    missing_null.push(0);
    missing_null.push(1);
    missing_null.extend(core::iter::repeat_n(
      0xff,
      256usize.strict_sub(missing_null_prefix.len()).strict_sub(3),
    ));
    missing_null.push(0);
    missing_null.extend_from_slice(&missing_null_prefix);
    assert!(verify_pkcs1v15_encoded::<Sha256>(message, &missing_null, SHA256_DIGEST_INFO_PREFIX).is_err());

    let mut ber_indefinite_digest_info = Vec::new();
    ber_indefinite_digest_info.extend_from_slice(&[0x30, 0x80]);
    ber_indefinite_digest_info.extend_from_slice(&SHA256_DIGEST_INFO_PREFIX[2..SHA256_DIGEST_INFO_PREFIX.len()]);
    ber_indefinite_digest_info.extend_from_slice(&digest);
    ber_indefinite_digest_info.extend_from_slice(&[0x00, 0x00]);
    let mut ber = Vec::new();
    ber.push(0);
    ber.push(1);
    ber.extend(core::iter::repeat_n(
      0xff,
      256usize.strict_sub(ber_indefinite_digest_info.len()).strict_sub(3),
    ));
    ber.push(0);
    ber.extend_from_slice(&ber_indefinite_digest_info);
    assert!(verify_pkcs1v15_encoded::<Sha256>(message, &ber, SHA256_DIGEST_INFO_PREFIX).is_err());

    let mut padding_hole = pkcs1v15_encoded_sha256(message, 256);
    if let Some(byte) = padding_hole.get_mut(10) {
      *byte = 0;
    }
    assert!(verify_pkcs1v15_encoded::<Sha256>(message, &padding_hole, SHA256_DIGEST_INFO_PREFIX).is_err());
  }

  #[test]
  fn pss_encoded_sha256_accepts_valid_encoding() {
    let message = b"rsa-pss-encoded-message";
    let encoded = pss_encoded_sha256(message, 2047, Sha256::OUTPUT_SIZE);

    assert_eq!(verify_pss_encoded::<Sha256>(message, &encoded, 2047), Ok(()));
  }

  #[test]
  fn pss_encoded_sha256_accepts_explicit_zero_salt_and_rejects_leftmost_unused_bit() {
    let message = b"rsa-pss-encoded-message";
    let encoded = pss_encoded_sha256(message, 2047, 0);
    let mut db = vec![0u8; encoded.len()];
    let mut db_mask = vec![0u8; encoded.len()];

    assert_eq!(
      verify_pss_encoded_with_scratch::<Sha256>(message, &encoded, 2047, 0, &mut db, &mut db_mask),
      Ok(())
    );

    let mut bad_unused_bit = encoded;
    bad_unused_bit[0] |= 0x80;
    assert_eq!(
      verify_pss_encoded_with_scratch::<Sha256>(message, &bad_unused_bit, 2047, 0, &mut db, &mut db_mask),
      Err(VerificationError::new())
    );
  }

  #[test]
  fn pss_encoded_sha256_rejects_impossible_em_bits_without_panic() {
    let message = b"rsa-pss-encoded-message";
    let encoded = pss_encoded_sha256(message, 2047, Sha256::OUTPUT_SIZE);
    let mut db = vec![0u8; encoded.len()];
    let mut db_mask = vec![0u8; encoded.len()];

    assert_eq!(
      verify_pss_encoded_with_scratch::<Sha256>(
        message,
        &encoded,
        encoded.len().strict_mul(8).strict_add(1),
        Sha256::OUTPUT_SIZE,
        &mut db,
        &mut db_mask
      ),
      Err(VerificationError::new())
    );
    assert_eq!(
      verify_pss_encoded_with_scratch::<Sha256>(message, &encoded, 0, Sha256::OUTPUT_SIZE, &mut db, &mut db_mask),
      Err(VerificationError::new())
    );
  }

  #[test]
  fn montgomery_auto_threshold_uses_backend_cios_for_rsa8192() {
    let rsa4096 = RsaPublicModulus::new(&[0xff; 512], 4096);
    let rsa4160 = RsaPublicModulus::new(&[0xff; 520], 4160);
    let rsa8192 = RsaPublicModulus::new(&[0xff; 1024], 8192);

    assert!(use_cios_montgomery(&rsa4096));
    assert!(!use_cios_montgomery(&rsa4160));

    #[cfg(all(
      target_arch = "aarch64",
      target_os = "macos",
      not(feature = "portable-only"),
      not(miri)
    ))]
    assert!(use_cios_montgomery(&rsa8192));

    #[cfg(all(
      target_arch = "x86_64",
      target_os = "linux",
      not(feature = "portable-only"),
      not(miri)
    ))]
    assert_eq!(
      use_cios_montgomery(&rsa8192),
      rsa_x86_64_asm::supports_bignum_mont_words(128)
    );

    #[cfg(not(any(
      all(
        target_arch = "aarch64",
        target_os = "macos",
        not(feature = "portable-only"),
        not(miri)
      ),
      all(
        target_arch = "x86_64",
        target_os = "linux",
        not(feature = "portable-only"),
        not(miri)
      )
    )))]
    assert!(!use_cios_montgomery(&rsa8192));
  }

  fn reference_double_mod_in_place(value: &mut [u64], modulus: &[u64]) {
    let mut carry = 0u64;
    for limb in value.iter_mut() {
      let next = *limb >> 63;
      *limb = (*limb << 1) | carry;
      carry = next;
    }
    if carry != 0 {
      let _ = sub_modulus_in_place(value, modulus);
    } else if cmp_limbs(value, modulus) != core::cmp::Ordering::Less {
      let borrow = sub_modulus_in_place(value, modulus);
      debug_assert_eq!(borrow, 0);
    }
  }

  #[test]
  fn double_mod_in_place_matches_reference_edges() {
    let cases: [(&[u64], &[u64]); 5] = [
      (&[0x0000_0000_0000_0004], &[0x0000_0000_0000_000b]),
      (&[0x0000_0000_0000_0006], &[0x0000_0000_0000_000b]),
      (&[0x8000_0000_0000_0000], &[0xffff_ffff_ffff_fffb]),
      (&[0xffff_ffff_ffff_fffb], &[0xffff_ffff_ffff_fffd]),
      (
        &[0xffff_ffff_ffff_ffff, 0x7fff_ffff_ffff_ffff],
        &[0xffff_ffff_ffff_fffd, 0xffff_ffff_ffff_ffff],
      ),
    ];

    for (value, modulus) in cases {
      let mut optimized = value.to_vec();
      let mut reference = value.to_vec();
      double_mod_in_place(&mut optimized, modulus);
      reference_double_mod_in_place(&mut reference, modulus);
      assert_eq!(
        optimized, reference,
        "doubling mismatch for value={value:x?} modulus={modulus:x?}"
      );
    }
  }

  #[test]
  fn pow2_mod_direct_prefix_matches_naive_doubling() {
    let mut sparse_modulus = vec![0u8; 256];
    sparse_modulus[0] = 0x80;
    sparse_modulus[255] = 0x01;
    let dense_modulus = vec![0xff; 256];
    let mut unaligned_modulus = vec![0u8; 257];
    unaligned_modulus[0] = 0x01;
    unaligned_modulus[1] = 0x80;
    unaligned_modulus[256] = 0x01;

    for modulus_bytes in [&sparse_modulus, &dense_modulus, &unaligned_modulus] {
      let modulus = limbs_from_be(modulus_bytes);
      let modulus_bits = limb_bit_len(&modulus);
      let mut bit_counts = vec![
        0,
        1,
        63,
        64,
        65,
        modulus_bits.saturating_sub(2),
        modulus_bits.saturating_sub(1),
      ];
      bit_counts.extend([modulus_bits, modulus_bits.strict_add(1), modulus.len().strict_mul(128)]);

      for bits in bit_counts {
        let mut optimized = vec![0u64; modulus.len()];
        let mut naive = vec![0u64; modulus.len()];
        pow2_mod_into(&mut optimized, bits, &modulus);
        naive[0] = 1;
        for _ in 0..bits {
          reference_double_mod_in_place(&mut naive, &modulus);
        }
        assert_eq!(optimized, naive, "pow2_mod_into mismatch at {bits} bits");
      }
    }
  }

  #[cfg(feature = "diag")]
  #[test]
  fn diag_spki_public_key_validation_matches_import_metadata() {
    let spki = include_bytes!("../../benches/rsa_fixtures/rsa3072_spki.der");
    let key = RsaPublicKey::from_spki_der(spki).unwrap();
    let (modulus_len, modulus_bits, exponent) =
      diag_rsa_validate_spki_public_key_der(spki, &RsaPublicKeyPolicy::default()).unwrap();

    assert_eq!(modulus_len, key.modulus().len());
    assert_eq!(modulus_bits, key.modulus_bits());
    assert_eq!(exponent, key.public_exponent());
  }

  #[cfg(feature = "diag")]
  #[test]
  fn diag_montgomery_r2_precompute_matches_imported_key() {
    let spki = include_bytes!("../../benches/rsa_fixtures/rsa3072_spki.der");
    let key = RsaPublicKey::from_spki_der(spki).unwrap();
    let scratch = key.public_scratch();

    assert_eq!(
      diag_rsa_precompute_public_montgomery_r2(key.modulus()),
      Ok(limb_checksum(&scratch.r2))
    );
  }

  #[cfg(feature = "diag")]
  #[test]
  fn public_operation_generic_exponent_matches_specialized_fermat_paths() {
    let modulus = [0xff; 256];
    let mut input = [0xff; 256];
    input[255] = 0xfe;
    let policy = RsaPublicKeyPolicy::legacy_verification().allow_legacy_small_exponents();

    for exponent in [&[0x03][..], &[0x11][..], &[0x01, 0x00, 0x01][..]] {
      let key = RsaPublicKey::from_pkcs1_der_with_policy(&test_pkcs1_public_key(&modulus, exponent), &policy).unwrap();
      let mut specialized = vec![0u8; key.modulus().len()];
      let mut generic = vec![0u8; key.modulus().len()];
      let mut specialized_scratch = key.public_scratch();
      let mut generic_scratch = key.public_scratch();

      key
        .public_operation_with_scratch(&input, &mut specialized, &mut specialized_scratch)
        .unwrap();
      diag_rsa_public_operation_generic_exponent(&key, &input, &mut generic, &mut generic_scratch).unwrap();

      assert_eq!(
        generic, specialized,
        "generic exponent baseline diverged for exponent {exponent:02x?}"
      );
    }
  }

  #[cfg(feature = "diag")]
  #[test]
  fn diag_public_operation_backends_clear_output_on_error() {
    let modulus = [0xff; 256];
    let exponent = [0x01, 0x00, 0x01];
    let key = RsaPublicKey::from_pkcs1_der_with_policy(
      &test_pkcs1_public_key(&modulus, &exponent),
      &RsaPublicKeyPolicy::legacy_verification(),
    )
    .unwrap();
    let representative = key.modulus().to_vec();

    let mut out = vec![0xa5; key.modulus().len()];
    assert_eq!(
      diag_rsa_public_operation_bitserial(&key, &representative, &mut out),
      Err(RsaPublicOpError::RepresentativeOutOfRange)
    );
    assert!(is_zero_unsigned_be(&out));

    let mut product_scratch = key.public_scratch();
    out.fill(0xa5);
    assert_eq!(
      diag_rsa_public_operation_product(&key, &representative, &mut out, &mut product_scratch),
      Err(RsaPublicOpError::RepresentativeOutOfRange)
    );
    assert!(is_zero_unsigned_be(&out));

    let mut comba_scratch = key.public_scratch();
    out.fill(0xa5);
    assert_eq!(
      diag_rsa_public_operation_comba_product(&key, &representative, &mut out, &mut comba_scratch),
      Err(RsaPublicOpError::RepresentativeOutOfRange)
    );
    assert!(is_zero_unsigned_be(&out));

    let mut cios_scratch = key.public_scratch();
    out.fill(0xa5);
    assert_eq!(
      diag_rsa_public_operation_cios(&key, &representative, &mut out, &mut cios_scratch),
      Err(RsaPublicOpError::RepresentativeOutOfRange)
    );
    assert!(is_zero_unsigned_be(&out));

    let mut generic_scratch = key.public_scratch();
    out.fill(0xa5);
    assert_eq!(
      diag_rsa_public_operation_generic_exponent(&key, &representative, &mut out, &mut generic_scratch),
      Err(RsaPublicOpError::RepresentativeOutOfRange)
    );
    assert!(is_zero_unsigned_be(&out));

    let mut window2_scratch = key.public_scratch();
    out.fill(0xa5);
    assert_eq!(
      diag_rsa_public_operation_window2_exponent(&key, &representative, &mut out, &mut window2_scratch),
      Err(RsaPublicOpError::RepresentativeOutOfRange)
    );
    assert!(is_zero_unsigned_be(&out));
  }

  #[test]
  fn pss_encoded_sha256_rejects_invalid_salt_length_trailer_and_masked_db() {
    let message = b"rsa-pss-encoded-message";

    let short_salt = pss_encoded_sha256(message, 2047, Sha256::OUTPUT_SIZE.strict_sub(1));
    assert!(verify_pss_encoded::<Sha256>(message, &short_salt, 2047).is_err());

    let mut bad_trailer = pss_encoded_sha256(message, 2047, Sha256::OUTPUT_SIZE);
    if let Some(byte) = bad_trailer.last_mut() {
      *byte = 0xbd;
    }
    assert!(verify_pss_encoded::<Sha256>(message, &bad_trailer, 2047).is_err());

    let mut bad_db = pss_encoded_sha256(message, 2047, Sha256::OUTPUT_SIZE);
    if let Some(byte) = bad_db.get_mut(16) {
      *byte ^= 0x40;
    }
    assert!(verify_pss_encoded::<Sha256>(message, &bad_db, 2047).is_err());
  }

  #[test]
  fn pss_encoded_sha256_rejects_message_hash_mismatch() {
    let encoded = pss_encoded_sha256(b"rsa-pss-encoded-message", 2047, Sha256::OUTPUT_SIZE);

    assert!(verify_pss_encoded::<Sha256>(b"wrong message", &encoded, 2047).is_err());
  }

  #[test]
  fn public_operation_matches_independent_rsa2048_e65537_vector() {
    let modulus = hex_to_vec(
      "\
ef8bb02b8e4aec1abc6fac7a0d6fb1f2649bb86a1567423fee4a194a250461a9db702558e92e52cc\
907963d84731a7adaf4c609e1b7c7d7c187099a43857f7628f5d20416fcb48987c9d6f12cfc6bc\
260c9b5506be3fe3cd218ddb37ef5b30feb16172a9832312726ed135c0540ef9d3229b87b5566f\
3355c90f301b856aa822878269806079ab7267cdc6c7403d7be3fa652065b2d39f2dbf9fb61ed9\
71fee37432ebe31d9aa465dbae96b0edd5ffddf1b49e03346a02290fed1e4e31f6b3b6e1f839f\
d5add90a8a212c10dd997b0a4efcb3df990808509dcb28c504e0649827a83ffd864395d1f62f2\
9a004f44423a44b07de943a60fba844a9da3603ce5c5",
    );
    let input = hex_to_vec(
      "\
3450869c4ccbee98815e55cb42f2dd85a3427d3f65e33d29352293e18cde9582a9fbc54b440984\
1ba8d931a9a9411192516a9fbd3a7b886e7f8b8f3f7bb5403309eee9d7234df0b5934e18a1dc\
9e3b568a3fab6947cefe50500abcbda19fd9ab7b7e90a95801e36a020ba79bdc94346198d98131\
6864a06a43448b62acb7a8472661323175f04c5e447d0017e4073efc55f59f79f34aaa3be8ae7\
0d26db78b25e9dfb23856d1b1e024aedfcfd649d209412c0c80832ca3466965eeff539afe791f\
451b554e212cff4d92466438062c5202169b0adf0c95b7d3d31414602cf9d185252b550cc2e8f\
5be08b7fc71f51210ff88363badadfaf5c2915c3a10b2389e",
    );
    let expected = hex_to_vec(
      "\
020d016f2b1394b5ec00d4ddd1725435747a31fd4b2489fad76060b68b2259089d304b1d3c98\
e0a343c4b313d15b6022b0400dde22538f30e8474d483189c04ece5acc8aaf1481b362c2d2fe7\
fa853e856a0aba66cc47cf9e59052fdd4c5f4155bcc2a3f3330e2c48b7f45d1e66d8cd04829c\
0ba2e598569b4eeb8538c3cdf8e02c838d04bdc661b5d8c5291b0feebf284eb9deea03dd0226\
bdb322e180a6ab522ee40a02a0daf41094a2938d39698ab16381ed4d3ddd01bd05a8aa9113d8\
ec34e8c72cc58fd5324fbe1ddd9714909caedfaa38706cfa66d9bc1026ba3ec1188092392a54a\
3e94bf239ee74517b71ec2464551f8174dbd0f3952ffb41070c754",
    );
    let key = RsaPublicKey::from_pkcs1_der(&test_pkcs1_public_key(&modulus, &[0x01, 0x00, 0x01])).unwrap();
    let mut out = vec![0u8; key.modulus().len()];
    let mut scratch = key.public_scratch();

    key
      .public_operation_with_scratch(&input, &mut out, &mut scratch)
      .unwrap();

    assert_eq!(out, expected);
  }

  #[test]
  fn protocol_adapter_mappings_are_explicit_and_reject_confusion() {
    assert_eq!(
      RsaSignatureProfile::from_jwt_alg("PS256"),
      Ok(RsaSignatureProfile::pss(RsaPssProfile::Sha256))
    );
    assert_eq!(
      RsaSignatureProfile::from_jwt_alg("RS512"),
      Ok(RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha512))
    );
    for alg in ["none", "HS256", "ES256", "EdDSA", "RS1", "PS1", "rs256"] {
      assert_eq!(
        RsaSignatureProfile::from_jwt_alg(alg),
        Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
      );
    }

    assert_eq!(
      RsaSignatureProfile::from_cose_algorithm_id(-37),
      Ok(RsaSignatureProfile::pss(RsaPssProfile::Sha256))
    );
    assert_eq!(
      RsaSignatureProfile::from_cose_algorithm_id(-259),
      Ok(RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha512))
    );
    for algorithm in [0, 1, -7, -65535, i64::from(i32::MAX)] {
      assert_eq!(
        RsaSignatureProfile::from_cose_algorithm_id(algorithm),
        Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
      );
    }

    assert_eq!(
      RsaSignatureProfile::from_tls13_signature_scheme(0x0804),
      Ok(RsaSignatureProfile::pss(RsaPssProfile::Sha256))
    );
    assert_eq!(
      RsaSignatureProfile::from_tls13_signature_scheme(0x0401),
      Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
    );
    assert_eq!(
      RsaX509PublicKeyAlgorithm::RsaEncryption.signature_profile_from_tls13_signature_scheme(0x0809),
      Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
    );
    assert_eq!(
      RsaX509PublicKeyAlgorithm::RsaPss.signature_profile_from_tls_certificate_signature_scheme(0x0401),
      Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
    );

    let pss_default_sha1 = algorithm_identifier(ID_RSASSA_PSS_OID, Some(&tlv(TAG_SEQUENCE, &[])));
    assert_eq!(
      RsaSignatureProfile::from_x509_signature_algorithm_der(&pss_default_sha1),
      Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
    );

    let pkcs1v15 = algorithm_identifier(SHA256_WITH_RSA_ENCRYPTION_OID, Some(&null()));
    assert_eq!(
      RsaSignatureProfile::from_x509_signature_algorithm_der(&pkcs1v15),
      Ok(RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256))
    );
    let pss = x509_pss_algorithm(RsaPssProfile::Sha256, Sha256::OUTPUT_SIZE);
    assert_eq!(
      RsaSignatureProfile::from_x509_signature_algorithm_der(&pss),
      Ok(RsaSignatureProfile::pss(RsaPssProfile::Sha256))
    );
    let sha1 = algorithm_identifier(SHA1_WITH_RSA_ENCRYPTION_OID, Some(&null()));
    assert_eq!(
      RsaSignatureProfile::from_x509_signature_algorithm_der(&sha1),
      Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
    );
    let absent_params = algorithm_identifier(SHA256_WITH_RSA_ENCRYPTION_OID, None);
    assert_eq!(
      RsaSignatureProfile::from_x509_signature_algorithm_der(&absent_params),
      Ok(RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256))
    );
  }

  #[test]
  fn x509_certificate_signature_helper_accepts_real_certificate_and_rejects_tamper() {
    let issuer = RsaX509PublicKey::from_spki_der(&x509_certificate_fixture_public_key()).unwrap();
    let certificate = x509_pkcs1v15_certificate_fixture();

    assert!(issuer.verify_x509_certificate_signature_der(&certificate).is_ok());

    let mut tampered = certificate;
    if let Some(last) = tampered.last_mut() {
      *last ^= 0x01;
    }
    assert_eq!(
      issuer.verify_x509_certificate_signature_der(&tampered),
      Err(VerificationError::new())
    );
  }

  #[test]
  fn x509_certificate_signature_parser_rejects_trailing_and_noncanonical_der() {
    let certificate = x509_pkcs1v15_certificate_fixture();

    let mut trailing = certificate.clone();
    trailing.extend_from_slice(&null());
    assert_eq!(
      parse_x509_certificate_signature(&trailing).err(),
      Some(RsaProtocolAlgorithmError::MalformedAlgorithmIdentifier)
    );

    assert_eq!(
      parse_x509_certificate_signature(&tlv_with_leading_zero_long_len(&certificate)).err(),
      Some(RsaProtocolAlgorithmError::MalformedAlgorithmIdentifier)
    );
  }

  proptest! {
    #[test]
    fn representative_limb_conversion_round_trips(bytes in proptest::collection::vec(any::<u8>(), 0..600)) {
      let limbs = limbs_from_be(&bytes);
      let mut encoded = vec![0u8; bytes.len()];
      limbs_to_be(&limbs, &mut encoded);

      prop_assert_eq!(encoded, bytes);
    }

    #[test]
    fn representative_range_comparison_matches_big_endian_byte_order(
      mut left in proptest::collection::vec(any::<u8>(), 1..300),
      mut right in proptest::collection::vec(any::<u8>(), 1..300),
    ) {
      let len = core::cmp::max(left.len(), right.len());
      let mut left_padded = vec![0u8; len];
      let mut right_padded = vec![0u8; len];
      let left_offset = len.strict_sub(left.len());
      let right_offset = len.strict_sub(right.len());
      left_padded[left_offset..].copy_from_slice(&left);
      right_padded[right_offset..].copy_from_slice(&right);

      let left_limbs = limbs_from_be(&left_padded);
      let right_limbs = limbs_from_be(&right_padded);

      prop_assert_eq!(cmp_limbs(&left_limbs, &right_limbs), left_padded.cmp(&right_padded));

      left.clear();
      right.clear();
    }

    #[test]
    fn public_operation_rejects_equal_modulus_and_accepts_modulus_minus_one(mut modulus in proptest::collection::vec(any::<u8>(), 256)) {
      modulus[0] |= 0x80;
      modulus[255] |= 0x01;
      let exponent = RsaPublicExponent(65_537);
      let bits = validate_modulus(&modulus, &RsaPublicKeyPolicy::legacy_verification()).unwrap();
      let key = RsaPublicKey {
        modulus: RsaPublicModulus::new(&modulus, bits),
        exponent,
      };
      let mut out = vec![0u8; key.modulus().len()];
      let mut scratch = key.public_scratch();

      prop_assert_eq!(
        key.public_operation_with_scratch(&modulus, &mut out, &mut scratch),
        Err(RsaPublicOpError::RepresentativeOutOfRange)
      );

      let mut less = modulus.clone();
      for byte in less.iter_mut().rev() {
        if *byte != 0 {
          *byte = byte.strict_sub(1);
          break;
        }
        *byte = 0xff;
      }

      prop_assert!(key.public_operation_with_scratch(&less, &mut out, &mut scratch).is_ok());
    }
  }
}
