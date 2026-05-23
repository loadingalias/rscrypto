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
//! RSA support is verification-only today. Signing, decryption, OAEP, and key
//! generation are intentionally absent until the private-key side-channel,
//! blinding, zeroization, and fault-checking requirements are met.

use alloc::{boxed::Box, vec, vec::Vec};
use core::fmt;

use crate::{
  hashes::crypto::{Sha256, Sha384, Sha512},
  traits::{Digest, VerificationError, ct},
};

const TAG_SEQUENCE: u8 = 0x30;
const TAG_INTEGER: u8 = 0x02;
const TAG_BIT_STRING: u8 = 0x03;
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

/// Error returned when a protocol algorithm identifier has no supported RSA profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RsaProtocolAlgorithmError {
  /// The algorithm identifier is not valid DER for the expected protocol field.
  MalformedAlgorithmIdentifier,
  /// The identifier is unknown, non-RSA, ambiguous, or intentionally unsupported.
  UnsupportedAlgorithm,
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

/// Typed RSA signature verification profile.
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

  /// Map a TLS certificate signature scheme ID to an RSA verification profile.
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

  /// Map a JWT `alg` value to an RSA verification profile.
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

  /// Map a COSE algorithm ID to an RSA verification profile.
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
  /// SHA-2 `sha*WithRSAEncryption` identifiers must carry exact `NULL`
  /// parameters. RSASSA-PSS identifiers must carry explicit RFC 4055
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
pub fn diag_rsa_public_operation_bitserial(
  key: &RsaPublicKey,
  input: &[u8],
  out: &mut [u8],
) -> Result<(), RsaPublicOpError> {
  key.modulus.public_operation_bitserial(key.exponent, input, out)
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
pub fn diag_rsa_public_operation_product(
  key: &RsaPublicKey,
  input: &[u8],
  out: &mut [u8],
  scratch: &mut RsaPublicScratch,
) -> Result<(), RsaPublicOpError> {
  key.modulus.public_operation_product(key.exponent, input, out, scratch)
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
pub fn diag_rsa_public_operation_cios(
  key: &RsaPublicKey,
  input: &[u8],
  out: &mut [u8],
  scratch: &mut RsaPublicScratch,
) -> Result<(), RsaPublicOpError> {
  key.modulus.public_operation_cios(key.exponent, input, out, scratch)
}

/// Apply the RSA public operation with the generic square-and-multiply exponent loop.
///
/// Diagnostic-only benchmark baseline for exponentiation strategy. This forces
/// the generic public-exponent loop even when production uses a specialized
/// path for common Fermat exponents.
#[cfg(feature = "diag")]
#[inline]
pub fn diag_rsa_public_operation_generic_exponent(
  key: &RsaPublicKey,
  input: &[u8],
  out: &mut [u8],
  scratch: &mut RsaPublicScratch,
) -> Result<(), RsaPublicOpError> {
  key
    .modulus
    .public_operation_generic_exponent(key.exponent, input, out, scratch)
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

/// X.509 RSA public-key algorithm constraints.
///
/// `rsaEncryption` keys are unconstrained RSA public keys. `id-RSASSA-PSS`
/// keys are restricted to PSS signatures, and may additionally constrain the
/// hash/MGF1 profile and minimum salt length through their SPKI parameters.
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
  /// public-key algorithm. PKCS#1 v1.5 and SHA-1 schemes are rejected for
  /// TLS 1.3 handshake signatures.
  ///
  /// # Errors
  ///
  /// Returns [`RsaProtocolAlgorithmError::UnsupportedAlgorithm`] for unknown,
  /// non-RSA, SHA-1, PKCS#1 v1.5, or key-algorithm-confused scheme IDs.
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
  /// This includes legacy SHA-2 PKCS#1 v1.5 certificate signatures for
  /// `rsaEncryption` keys, plus TLS 1.3 RSA-PSS RSAE/PSS schemes with their
  /// required X.509 public-key algorithm.
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
    let mut scratch = self.key.public_scratch();
    self.verify_signature_from_x509_algorithm_der_with_scratch(
      signature_algorithm_der,
      message,
      signature,
      &mut scratch,
    )
  }

  /// Verify a signature using an X.509 signature `AlgorithmIdentifier` and caller-owned scratch.
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
    let mut scratch = self.key.public_scratch();
    self.verify_x509_certificate_signature_der_with_scratch(certificate_der, &mut scratch)
  }

  /// Verify the signature on an X.509 certificate DER object using caller-owned scratch.
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
    let mut scratch = self.key.public_scratch();
    self.verify_tls13_signature_scheme_with_scratch(scheme, message, signature, &mut scratch)
  }

  /// Verify a TLS 1.3 `CertificateVerify` RSA signature with caller-owned scratch.
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
    let mut scratch = self.key.public_scratch();
    self.verify_tls_certificate_signature_scheme_with_scratch(scheme, message, signature, &mut scratch)
  }

  /// Verify a TLS certificate RSA signature scheme with caller-owned scratch.
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
  pub fn public_operation_with_scratch(
    &self,
    input: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), RsaPublicOpError> {
    self.modulus.public_operation(self.exponent, input, out, scratch)
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
    match profile {
      RsaSignatureProfile::Pss { profile, salt_len } => {
        self.verify_pss_with_salt_len_and_scratch(profile, salt_len, message, signature, scratch)
      }
      RsaSignatureProfile::Pkcs1v15(profile) => self.verify_pkcs1v15_with_scratch(profile, message, signature, scratch),
    }
  }

  /// Verify an RSA JWT/JWS signature using an already-parsed JOSE `alg`.
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
    let mut scratch = self.public_scratch();
    self.verify_jwt_alg_with_scratch(alg, message, signature, &mut scratch)
  }

  /// Verify an RSA JWT/JWS signature using caller-owned RSA scratch space.
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
    let mut scratch = self.public_scratch();
    self.verify_cose_algorithm_id_with_scratch(algorithm, message, signature, &mut scratch)
  }

  /// Verify an RSA COSE signature using caller-owned RSA scratch space.
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
  limbs: Box<[u64]>,
  bytes: Box<[u8]>,
  limb_count: usize,
  byte_count: usize,
}

struct RsaPublicArithmeticScratch<'a> {
  t: &'a mut [u64],
  x: &'a mut [u64],
  tmp: &'a mut [u64],
  base: &'a mut [u64],
  acc: &'a mut [u64],
  r2: &'a [u64],
}

impl RsaPublicScratch {
  /// Allocate scratch space for `key`.
  #[must_use]
  pub fn new(key: &RsaPublicKey) -> Self {
    let limbs = key.modulus.limbs.len();
    let bytes = key.modulus.bytes.len();
    let mut limb_storage = vec![0u64; limbs.strict_mul(7).strict_add(2)].into_boxed_slice();
    let r2_start = limbs.strict_mul(6).strict_add(2);
    let (_, r2_and_tail) = limb_storage.split_at_mut(r2_start);
    let (r2, _) = r2_and_tail.split_at_mut(limbs);
    pow2_mod_into(r2, limbs.strict_mul(128), &key.modulus.limbs);
    Self {
      limbs: limb_storage,
      bytes: vec![0u8; bytes.strict_mul(3)].into_boxed_slice(),
      limb_count: limbs,
      byte_count: bytes,
    }
  }

  fn arithmetic_scratch(&mut self) -> RsaPublicArithmeticScratch<'_> {
    split_limb_scratch(&mut self.limbs, self.limb_count)
  }

  fn split_all(&mut self) -> (RsaPublicArithmeticScratch<'_>, &mut [u8], &mut [u8], &mut [u8]) {
    let arithmetic_scratch = split_limb_scratch(&mut self.limbs, self.limb_count);
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

fn split_limb_scratch(limbs: &mut [u64], limb_count: usize) -> RsaPublicArithmeticScratch<'_> {
  let t_len = limb_count.strict_mul(2).strict_add(2);
  let (t, rest) = limbs.split_at_mut(t_len);
  let (x, rest) = rest.split_at_mut(limb_count);
  let (tmp, rest) = rest.split_at_mut(limb_count);
  let (base, rest) = rest.split_at_mut(limb_count);
  let (acc, r2) = rest.split_at_mut(limb_count);
  RsaPublicArithmeticScratch {
    t,
    x,
    tmp,
    base,
    acc,
    r2,
  }
}

fn split_byte_scratch(bytes: &mut [u8], byte_count: usize) -> (&mut [u8], &mut [u8], &mut [u8]) {
  let (encoded, rest) = bytes.split_at_mut(byte_count);
  let (db, db_mask) = rest.split_at_mut(byte_count);
  (encoded, db, db_mask)
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct RsaPublicModulus {
  bytes: Box<[u8]>,
  limbs: Box<[u64]>,
  bits: usize,
  n0: u64,
}

impl RsaPublicModulus {
  fn new(bytes: &[u8], bits: usize) -> Self {
    let limbs = limbs_from_be(bytes);
    let n0 = montgomery_n0(limbs.first().copied().unwrap_or(1));
    Self {
      bytes: Box::from(bytes),
      limbs: limbs.into_boxed_slice(),
      bits,
      n0,
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
      t,
      x,
      tmp,
      base,
      acc,
      r2,
    } = scratch;
    let limbs = self.limbs.len();
    if input.len() != self.bytes.len() || out.len() != self.bytes.len() {
      return Err(RsaPublicOpError::InvalidLength);
    }
    if x.len() != limbs
      || base.len() != limbs
      || acc.len() != limbs
      || tmp.len() != limbs
      || r2.len() != limbs
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

    match exponent.as_u64() {
      3 => {
        mont_square_auto_in_place(acc, tmp, self, t);
        mont_mul_auto_in_place_left(acc, base, tmp, self, t);
      }
      17 => {
        for _ in 0..4 {
          mont_square_auto_in_place(acc, tmp, self, t);
        }
        mont_mul_auto_in_place_left(acc, base, tmp, self, t);
      }
      65_537 => {
        for _ in 0..16 {
          mont_square_auto_in_place(acc, tmp, self, t);
        }
        mont_mul_auto_in_place_left(acc, base, tmp, self, t);
      }
      exponent => {
        let top_bit = 63usize.strict_sub(exponent.leading_zeros() as usize);
        for bit in (0..top_bit).rev() {
          mont_square_auto_in_place(acc, tmp, self, t);
          if (exponent >> bit) & 1 == 1 {
            mont_mul_auto_in_place_left(acc, base, tmp, self, t);
          }
        }
      }
    }

    x.fill(0);
    if let Some(first) = x.first_mut() {
      *first = 1;
    }
    mont_mul_auto(base, acc, x, self, t);
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
      t,
      x,
      tmp,
      base,
      acc,
      r2,
    } = scratch.arithmetic_scratch();
    let limbs = self.limbs.len();
    if input.len() != self.bytes.len() || out.len() != self.bytes.len() {
      return Err(RsaPublicOpError::InvalidLength);
    }
    if x.len() != limbs
      || base.len() != limbs
      || acc.len() != limbs
      || tmp.len() != limbs
      || r2.len() != limbs
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

    x.fill(0);
    if let Some(first) = x.first_mut() {
      *first = 1;
    }
    mont_mul_auto(base, acc, x, self, t);
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
      t,
      x,
      tmp,
      base,
      acc,
      r2,
    } = scratch.arithmetic_scratch();
    let limbs = self.limbs.len();
    if input.len() != self.bytes.len() || out.len() != self.bytes.len() {
      return Err(RsaPublicOpError::InvalidLength);
    }
    if x.len() != limbs
      || base.len() != limbs
      || acc.len() != limbs
      || tmp.len() != limbs
      || r2.len() != limbs
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

    x.fill(0);
    if let Some(first) = x.first_mut() {
      *first = 1;
    }
    mont_mul(base, acc, x, self, t);
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
      t,
      x,
      tmp,
      base,
      acc,
      r2,
    } = scratch.arithmetic_scratch();
    let limbs = self.limbs.len();
    if input.len() != self.bytes.len() || out.len() != self.bytes.len() {
      return Err(RsaPublicOpError::InvalidLength);
    }
    if x.len() != limbs
      || base.len() != limbs
      || acc.len() != limbs
      || tmp.len() != limbs
      || r2.len() != limbs
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

    x.fill(0);
    if let Some(first) = x.first_mut() {
      *first = 1;
    }
    mont_mul_cios(base, acc, x, self, t);
    limbs_to_be(base, out);
    Ok(())
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
  let null = protocol_der(reader.read_primitive(TAG_NULL))?;
  if !null.is_empty() {
    return Err(RsaProtocolAlgorithmError::MalformedAlgorithmIdentifier);
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

  let unused_bits = em_len.strict_mul(8).strict_sub(em_bits);
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

  if encoded.first().copied() != Some(0x00) || encoded.get(1).copied() != Some(0x01) {
    return Err(VerificationError::new());
  }

  let separator_index = encoded.len().strict_sub(digest_info_len).strict_sub(1);
  let padding = encoded.get(2..separator_index).ok_or_else(VerificationError::new)?;
  if padding.len() < 8 || padding.iter().any(|&byte| byte != 0xff) {
    return Err(VerificationError::new());
  }
  if encoded.get(separator_index).copied() != Some(0x00) {
    return Err(VerificationError::new());
  }

  let digest_info = encoded
    .get(separator_index.strict_add(1)..)
    .ok_or_else(VerificationError::new)?;
  let (prefix, value) = digest_info.split_at(digest_info_prefix.len());
  if !ct::constant_time_eq(prefix, digest_info_prefix) {
    return Err(VerificationError::new());
  }
  if ct::constant_time_eq(value, digest.as_ref()) {
    Ok(())
  } else {
    Err(VerificationError::new())
  }
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
  out.fill(0);
  let mut limb_index = 0usize;
  let mut shift = 0u32;

  for &byte in bytes.iter().rev() {
    out[limb_index] |= u64::from(byte) << shift;
    shift += 8;
    if shift == 64 {
      shift = 0;
      limb_index = limb_index.strict_add(1);
    }
  }
}

#[allow(clippy::indexing_slicing)]
fn limbs_to_be(limbs: &[u64], out: &mut [u8]) {
  out.fill(0);
  let mut limb_index = 0usize;
  let mut shift = 0u32;

  for byte in out.iter_mut().rev() {
    *byte = (limbs[limb_index] >> shift) as u8;
    shift += 8;
    if shift == 64 {
      shift = 0;
      limb_index = limb_index.strict_add(1);
    }
  }
}

fn copy_limbs(dst: &mut [u64], src: &[u64]) {
  for (d, s) in dst.iter_mut().zip(src.iter().copied()) {
    *d = s;
  }
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
fn sub_modulus_in_place(value: &mut [u64], modulus: &[u64]) -> bool {
  debug_assert_eq!(value.len(), modulus.len());
  let mut borrow = false;
  for index in 0..value.len() {
    let (tmp, b1) = value[index].overflowing_sub(modulus[index]);
    let (tmp, b2) = tmp.overflowing_sub(u64::from(borrow));
    value[index] = tmp;
    borrow = b1 || b2;
  }
  borrow
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
    carry = u64::from(overflow || carry_overflow);
  }

  if carry != 0 || cmp_limbs(value, modulus) != core::cmp::Ordering::Less {
    let _ = sub_modulus_in_place(value, modulus);
  }
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
  if carry != 0 {
    let _ = sub_modulus_in_place(value, modulus);
  } else if cmp_limbs(value, modulus) != core::cmp::Ordering::Less {
    let borrow = sub_modulus_in_place(value, modulus);
    debug_assert!(!borrow);
  }
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
  out[0] = 1;
  for _ in 0..bits {
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
  copy_limbs(tmp, value);
  mont_mul(value, tmp, tmp, modulus, t);
}

#[allow(clippy::indexing_slicing)]
fn mont_mul_in_place_left(left: &mut [u64], right: &[u64], tmp: &mut [u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  copy_limbs(tmp, left);
  mont_mul(left, tmp, right, modulus, t);
}

#[allow(clippy::indexing_slicing)]
fn mont_square_cios_in_place(value: &mut [u64], tmp: &mut [u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
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
  copy_limbs(tmp, left);
  mont_mul_cios(left, tmp, right, modulus, t);
}

#[inline]
#[must_use]
fn use_cios_montgomery(modulus: &RsaPublicModulus) -> bool {
  // Current local threshold evidence favors CIOS through RSA-4096 and product
  // Montgomery for RSA-8192. Keep the cutoff explicit until broader target
  // measurements justify moving it.
  modulus.limbs.len() <= 64
}

#[allow(clippy::indexing_slicing)]
fn mont_square_auto_in_place(value: &mut [u64], tmp: &mut [u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  if use_cios_montgomery(modulus) {
    mont_square_cios_in_place(value, tmp, modulus, t);
  } else {
    mont_square_in_place(value, tmp, modulus, t);
  }
}

#[allow(clippy::indexing_slicing)]
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

fn mont_mul_auto(out: &mut [u64], a: &[u64], b: &[u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  if use_cios_montgomery(modulus) {
    mont_mul_cios(out, a, b, modulus, t);
  } else {
    mont_mul(out, a, b, modulus, t);
  }
}

#[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
fn mont_mul_cios(out: &mut [u64], a: &[u64], b: &[u64], modulus: &RsaPublicModulus, t: &mut [u64]) {
  let n = modulus.limbs.len();
  debug_assert_eq!(out.len(), n);
  debug_assert_eq!(a.len(), n);
  debug_assert_eq!(b.len(), n);
  debug_assert!(t.len() >= n.strict_add(2));

  t.fill(0);

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

  if t[n] != 0 {
    let _ = sub_modulus_in_place(out, &modulus.limbs);
  } else if cmp_limbs(out, &modulus.limbs) != core::cmp::Ordering::Less {
    let borrow = sub_modulus_in_place(out, &modulus.limbs);
    debug_assert!(!borrow);
  }
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

  if t[n.strict_add(n)..].iter().any(|&limb| limb != 0) {
    let _ = sub_modulus_in_place(out, &modulus.limbs);
  } else if cmp_limbs(out, &modulus.limbs) != core::cmp::Ordering::Less {
    let borrow = sub_modulus_in_place(out, &modulus.limbs);
    debug_assert!(!borrow);
  }
}

#[allow(clippy::indexing_slicing)]
fn add_carry(t: &mut [u64], mut index: usize, mut carry: u64) {
  while carry != 0 {
    let (sum, overflow) = t[index].overflowing_add(carry);
    t[index] = sum;
    carry = u64::from(overflow);
    index = index.strict_add(1);
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
  use proptest::prelude::*;

  use super::*;

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
  fn montgomery_auto_threshold_uses_cios_through_rsa4096_only() {
    let rsa4096 = RsaPublicModulus::new(&[0xff; 512], 4096);
    let rsa4160 = RsaPublicModulus::new(&[0xff; 520], 4160);
    let rsa8192 = RsaPublicModulus::new(&[0xff; 1024], 8192);

    assert!(use_cios_montgomery(&rsa4096));
    assert!(!use_cios_montgomery(&rsa4160));
    assert!(!use_cios_montgomery(&rsa8192));
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
    let missing_null = algorithm_identifier(SHA256_WITH_RSA_ENCRYPTION_OID, None);
    assert_eq!(
      RsaSignatureProfile::from_x509_signature_algorithm_der(&missing_null),
      Err(RsaProtocolAlgorithmError::MalformedAlgorithmIdentifier)
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
