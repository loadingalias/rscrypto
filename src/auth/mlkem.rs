//! ML-KEM typed key, ciphertext, and shared-secret foundations.
//!
//! This module defines the public type surface for FIPS 203 ML-KEM parameter
//! sets. Private operations select accelerated arithmetic where available and
//! retain the portable implementation as the semantic authority.
//!
//! Encapsulation keys have RFC 9935 SubjectPublicKeyInfo import and export.
//! Decapsulation keys have RFC 9935 PKCS #8 import and export: a seed owner
//! such as [`MlKem768Seed`] keeps the recommended `d || z` seed form, and a
//! decapsulation key exports the expanded form. PKCS #8 import of an expanded
//! key adds a pairwise consistency check to the FIPS 203 checks of raw import.

mod operations;

#[cfg(feature = "alloc")]
use alloc::boxed::Box;
#[cfg(feature = "alloc")]
use core::alloc::Allocator;
use core::{
  error::Error,
  fmt,
  hash::{Hash, Hasher},
};

use crate::{
  SecretBytes,
  backend::{
    der::MalformedDer,
    pkix::{self, KeyError},
  },
  secret::ZeroizingBytes,
  traits::{
    Kem,
    ct::{self, DataIndependentTiming},
  },
};

const ML_KEM_SEED_SIZE: usize = 32;
const ML_KEM_KEY_GENERATION_RANDOM_SIZE: usize = ML_KEM_SEED_SIZE * 2;
const ML_KEM_ENCAPSULATION_RANDOM_SIZE: usize = ML_KEM_SEED_SIZE;
const ML_KEM_KEY_HASH_SIZE: usize = 32;
const ML_KEM_SHARED_SECRET_SIZE: usize = 32;

/// ML-KEM operation error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MlKemError {
  /// The caller-provided random source failed.
  RandomGenerationFailed,
  /// The encapsulation key failed FIPS 203 input validation.
  InvalidEncapsulationKey,
  /// The decapsulation key failed FIPS 203 input validation.
  InvalidDecapsulationKey,
  /// The ciphertext failed FIPS 203 input validation.
  InvalidCiphertext,
}

impl fmt::Display for MlKemError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::RandomGenerationFailed => f.write_str("ML-KEM random generation failed"),
      Self::InvalidEncapsulationKey => f.write_str("ML-KEM encapsulation key failed validation"),
      Self::InvalidDecapsulationKey => f.write_str("ML-KEM decapsulation key failed validation"),
      Self::InvalidCiphertext => f.write_str("ML-KEM ciphertext failed validation"),
    }
  }
}

impl Error for MlKemError {}

/// ML-KEM SubjectPublicKeyInfo or PKCS #8 key-import failure.
///
/// Raw-byte import and the KEM operations use [`MlKemError`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum MlKemKeyError {
  /// DER input was malformed or non-canonical.
  MalformedDer,
  /// The algorithm identifier names another algorithm or ML-KEM parameter set.
  UnsupportedAlgorithm,
  /// The DER is a well-formed key of this algorithm in a form this import
  /// does not accept: PKCS #8 attributes, or an expanded-only private key
  /// where a seed is required.
  UnsupportedEncoding,
  /// The encapsulation key has the wrong length or fails the FIPS 203 modulus check.
  InvalidEncapsulationKey,
  /// The decapsulation key has the wrong length, fails validation, or
  /// disagrees with a redundant seed or public key.
  InvalidDecapsulationKey,
}

impl fmt::Display for MlKemKeyError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(match self {
      Self::MalformedDer => "malformed ML-KEM DER",
      Self::UnsupportedAlgorithm => "unsupported ML-KEM key algorithm",
      Self::UnsupportedEncoding => "unsupported ML-KEM key encoding",
      Self::InvalidEncapsulationKey => "invalid ML-KEM encapsulation key",
      Self::InvalidDecapsulationKey => "invalid ML-KEM decapsulation key",
    })
  }
}

impl Error for MlKemKeyError {}

impl MalformedDer for MlKemKeyError {
  const MALFORMED_DER: Self = Self::MalformedDer;
}

impl KeyError for MlKemKeyError {
  const UNSUPPORTED_ALGORITHM: Self = Self::UnsupportedAlgorithm;
  const UNSUPPORTED_ENCODING: Self = Self::UnsupportedEncoding;
  const INVALID_PUBLIC_KEY: Self = Self::InvalidEncapsulationKey;
  const INVALID_SECRET_KEY: Self = Self::InvalidDecapsulationKey;
}

macro_rules! define_mlkem_public_bytes {
  ($name:ident, $len:expr, $doc:expr) => {
    #[doc = $doc]
    #[derive(Clone)]
    pub struct $name([u8; Self::LENGTH]);

    impl $name {
      /// Length in bytes.
      pub const LENGTH: usize = $len;

      /// Construct the typed value from raw bytes.
      #[inline]
      #[must_use]
      pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
        Self(bytes)
      }

      /// Return the wrapped bytes.
      #[inline]
      #[must_use]
      pub const fn to_bytes(&self) -> [u8; Self::LENGTH] {
        self.0
      }

      /// Borrow the wrapped bytes.
      #[inline]
      #[must_use]
      pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
        &self.0
      }
    }

    impl AsRef<[u8]> for $name {
      #[inline]
      fn as_ref(&self) -> &[u8] {
        &self.0
      }
    }

    impl PartialEq for $name {
      #[inline]
      fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
      }
    }

    impl Eq for $name {}

    impl Hash for $name {
      #[inline]
      fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
      }
    }

    impl fmt::Debug for $name {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", stringify!($name))?;
        crate::hex::fmt_hex_lower(&self.0, f)?;
        write!(f, ")")
      }
    }

    impl_hex_fmt!($name);
    impl_serde_bytes!($name);
  };
}

macro_rules! define_mlkem_secret_bytes {
  ($name:ident, $len:expr, $doc:expr) => {
    #[doc = $doc]
    pub struct $name([u8; Self::LENGTH]);

    impl $name {
      /// Length in bytes.
      pub const LENGTH: usize = $len;

      /// Compare two secret values without exposing a branchable boolean.
      #[inline]
      pub fn ct_eq(&self, other: &Self) -> ct::CtDecision {
        ct::fixed_eq(&self.0, &other.0)
      }

      /// Construct the typed value from raw bytes.
      #[inline]
      #[must_use]
      pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
        Self(bytes)
      }

      /// Explicitly extract the secret bytes into a zeroizing wrapper.
      #[inline]
      #[must_use]
      pub fn expose_secret(&self) -> SecretBytes<{ Self::LENGTH }> {
        SecretBytes::new(self.0)
      }

      /// Explicitly duplicate this secret value.
      #[inline]
      #[must_use]
      pub const fn duplicate_secret(&self) -> Self {
        Self(self.0)
      }

      /// Borrow the secret bytes.
      #[inline]
      #[must_use]
      pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
        &self.0
      }
    }

    impl AsRef<[u8]> for $name {
      #[inline]
      fn as_ref(&self) -> &[u8] {
        &self.0
      }
    }

    impl fmt::Debug for $name {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(****)", stringify!($name))
      }
    }

    impl Drop for $name {
      fn drop(&mut self) {
        ct::zeroize(&mut self.0);
      }
    }

    impl_hex_fmt_secret!($name);
    impl_serde_secret_bytes!($name);
  };
}

macro_rules! define_mlkem_profile {
  (
    $profile:ident,
    $encapsulation_key:ident,
    $decapsulation_key:ident,
    $ciphertext:ident,
    $shared_secret:ident,
    $encapsulation_key_len:expr,
    $decapsulation_key_len:expr,
    $ciphertext_len:expr,
    $security_category:expr,
    $required_rbg_strength:expr,
    $doc_name:literal
  ) => {
    #[doc = concat!($doc_name, " parameter-set marker.")]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct $profile;

    impl $profile {
      /// Encapsulation key size in bytes.
      pub const ENCAPSULATION_KEY_SIZE: usize = $encapsulation_key_len;

      /// Decapsulation key size in bytes.
      pub const DECAPSULATION_KEY_SIZE: usize = $decapsulation_key_len;

      /// Ciphertext size in bytes.
      pub const CIPHERTEXT_SIZE: usize = $ciphertext_len;

      /// Shared-secret size in bytes.
      pub const SHARED_SECRET_SIZE: usize = ML_KEM_SHARED_SECRET_SIZE;

      /// Random bytes consumed by FIPS 203 ML-KEM.KeyGen.
      pub const KEY_GENERATION_RANDOM_SIZE: usize = ML_KEM_KEY_GENERATION_RANDOM_SIZE;

      /// Random bytes consumed by FIPS 203 ML-KEM.Encaps.
      pub const ENCAPSULATION_RANDOM_SIZE: usize = ML_KEM_ENCAPSULATION_RANDOM_SIZE;

      /// NIST post-quantum security category.
      pub const SECURITY_CATEGORY: u8 = $security_category;

      /// Required random-bit-generator strength in bits.
      pub const REQUIRED_RBG_STRENGTH_BITS: u16 = $required_rbg_strength;
    }

    define_mlkem_public_bytes!(
      $encapsulation_key,
      $encapsulation_key_len,
      concat!($doc_name, " encapsulation key bytes.")
    );
    define_mlkem_secret_bytes!(
      $decapsulation_key,
      $decapsulation_key_len,
      concat!($doc_name, " decapsulation key bytes.")
    );
    define_mlkem_public_bytes!($ciphertext, $ciphertext_len, concat!($doc_name, " ciphertext bytes."));
    define_mlkem_secret_bytes!(
      $shared_secret,
      ML_KEM_SHARED_SECRET_SIZE,
      concat!($doc_name, " shared-secret bytes.")
    );
  };
}

macro_rules! define_mlkem_prepared_keys {
  (
    $prepared_encapsulation_key:ident,
    $encapsulation_key:ident,
    $prepared_decapsulation_key:ident,
    $decapsulation_key:ident,
    $k:expr,
    $dk_pke_bytes:expr,
    $ek_bytes:expr,
    $dk_bytes:expr,
    $doc_name:literal
  ) => {
    #[doc = concat!("Validated, reusable ", $doc_name, " encapsulation key.")]
    #[derive(Clone)]
    pub struct $prepared_encapsulation_key {
      key: $encapsulation_key,
      key_hash: [u8; ML_KEM_KEY_HASH_SIZE],
      arithmetic: operations::PreparedEncapsulationArithmetic<$k>,
    }

    impl $prepared_encapsulation_key {
      /// Length in bytes of the wrapped encapsulation key.
      pub const LENGTH: usize = $encapsulation_key::LENGTH;

      /// Parse, validate, and prepare an encapsulation key from raw bytes.
      #[inline]
      pub fn try_from_slice(bytes: &[u8]) -> Result<Self, MlKemError> {
        if bytes.len() != Self::LENGTH {
          return Err(MlKemError::InvalidEncapsulationKey);
        }

        let mut key = [0u8; Self::LENGTH];
        key.copy_from_slice(bytes);
        Self::try_from($encapsulation_key::from_bytes(key))
      }

      /// Return the validated encapsulation key.
      #[inline]
      #[must_use]
      pub const fn encapsulation_key(&self) -> &$encapsulation_key {
        &self.key
      }

      /// Copy the wrapped encapsulation key bytes.
      #[inline]
      #[must_use]
      pub const fn to_bytes(&self) -> [u8; Self::LENGTH] {
        self.key.to_bytes()
      }

      /// Borrow the wrapped encapsulation key bytes.
      #[inline]
      #[must_use]
      pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
        self.key.as_bytes()
      }

      #[inline]
      const fn key_hash(&self) -> &[u8; ML_KEM_KEY_HASH_SIZE] {
        &self.key_hash
      }
    }

    impl core::convert::TryFrom<$encapsulation_key> for $prepared_encapsulation_key {
      type Error = MlKemError;

      #[inline]
      fn try_from(key: $encapsulation_key) -> Result<Self, Self::Error> {
        let arithmetic = operations::validate_and_prepare_encapsulation_key::<$k, $ek_bytes>(key.as_bytes())?;
        let key_hash = operations::encapsulation_key_hash(key.as_bytes());
        Ok(Self {
          key,
          key_hash,
          arithmetic,
        })
      }
    }

    impl core::convert::TryFrom<&$encapsulation_key> for $prepared_encapsulation_key {
      type Error = MlKemError;

      #[inline]
      fn try_from(key: &$encapsulation_key) -> Result<Self, Self::Error> {
        let arithmetic = operations::validate_and_prepare_encapsulation_key::<$k, $ek_bytes>(key.as_bytes())?;
        let key_hash = operations::encapsulation_key_hash(key.as_bytes());
        Ok(Self {
          key: key.clone(),
          key_hash,
          arithmetic,
        })
      }
    }

    impl AsRef<[u8]> for $prepared_encapsulation_key {
      #[inline]
      fn as_ref(&self) -> &[u8] {
        self.as_bytes()
      }
    }

    impl PartialEq for $prepared_encapsulation_key {
      #[inline]
      fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
      }
    }

    impl Eq for $prepared_encapsulation_key {}

    impl Hash for $prepared_encapsulation_key {
      #[inline]
      fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_bytes().hash(state);
      }
    }

    impl fmt::Debug for $prepared_encapsulation_key {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", stringify!($prepared_encapsulation_key))?;
        crate::hex::fmt_hex_lower(self.as_bytes(), f)?;
        write!(f, ")")
      }
    }

    #[doc = concat!("Validated, reusable ", $doc_name, " decapsulation key.")]
    pub struct $prepared_decapsulation_key {
      key: $decapsulation_key,
      arithmetic: operations::PreparedDecapsulationArithmetic<$k>,
    }

    impl $prepared_decapsulation_key {
      /// Length in bytes of the wrapped decapsulation key.
      pub const LENGTH: usize = $decapsulation_key::LENGTH;

      /// Parse, validate, and prepare a decapsulation key from raw bytes.
      #[inline]
      pub fn try_from_slice(bytes: &[u8]) -> Result<Self, MlKemError> {
        if bytes.len() != Self::LENGTH {
          return Err(MlKemError::InvalidDecapsulationKey);
        }

        let mut key = [0u8; Self::LENGTH];
        key.copy_from_slice(bytes);
        Self::try_from($decapsulation_key::from_bytes(key))
      }

      /// Return the validated decapsulation key.
      #[inline]
      #[must_use]
      pub const fn decapsulation_key(&self) -> &$decapsulation_key {
        &self.key
      }

      /// Explicitly extract the wrapped secret bytes into a zeroizing wrapper.
      #[inline]
      #[must_use]
      pub fn expose_secret(&self) -> SecretBytes<{ Self::LENGTH }> {
        self.key.expose_secret()
      }

      /// Explicitly duplicate this prepared secret key and its derived arithmetic.
      #[inline]
      #[must_use]
      pub fn duplicate_secret(&self) -> Self {
        Self {
          key: self.key.duplicate_secret(),
          arithmetic: self.arithmetic.clone(),
        }
      }

      /// Borrow the wrapped decapsulation key bytes.
      #[inline]
      #[must_use]
      pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
        self.key.as_bytes()
      }

      /// Compare two prepared secret keys without exposing a branchable boolean.
      #[inline]
      pub fn ct_eq(&self, other: &Self) -> ct::CtDecision {
        ct::fixed_eq(self.as_bytes(), other.as_bytes())
      }
    }

    impl core::convert::TryFrom<$decapsulation_key> for $prepared_decapsulation_key {
      type Error = MlKemError;

      #[inline]
      fn try_from(key: $decapsulation_key) -> Result<Self, Self::Error> {
        let arithmetic = operations::validate_and_prepare_decapsulation_key::<$k, $dk_pke_bytes, $ek_bytes, $dk_bytes>(
          key.as_bytes(),
        )?;
        Ok(Self { key, arithmetic })
      }
    }

    impl core::convert::TryFrom<&$decapsulation_key> for $prepared_decapsulation_key {
      type Error = MlKemError;

      #[inline]
      fn try_from(key: &$decapsulation_key) -> Result<Self, Self::Error> {
        let arithmetic = operations::validate_and_prepare_decapsulation_key::<$k, $dk_pke_bytes, $ek_bytes, $dk_bytes>(
          key.as_bytes(),
        )?;
        Ok(Self {
          key: key.duplicate_secret(),
          arithmetic,
        })
      }
    }

    impl AsRef<[u8]> for $prepared_decapsulation_key {
      #[inline]
      fn as_ref(&self) -> &[u8] {
        self.as_bytes()
      }
    }

    impl fmt::Debug for $prepared_decapsulation_key {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(****)", stringify!($prepared_decapsulation_key))
      }
    }
  };
}

define_mlkem_profile!(
  MlKem512,
  MlKem512EncapsulationKey,
  MlKem512DecapsulationKey,
  MlKem512Ciphertext,
  MlKem512SharedSecret,
  800,
  1632,
  768,
  1,
  128,
  "ML-KEM-512"
);

define_mlkem_profile!(
  MlKem768,
  MlKem768EncapsulationKey,
  MlKem768DecapsulationKey,
  MlKem768Ciphertext,
  MlKem768SharedSecret,
  1184,
  2400,
  1088,
  3,
  192,
  "ML-KEM-768"
);

define_mlkem_profile!(
  MlKem1024,
  MlKem1024EncapsulationKey,
  MlKem1024DecapsulationKey,
  MlKem1024Ciphertext,
  MlKem1024SharedSecret,
  1568,
  3168,
  1568,
  5,
  256,
  "ML-KEM-1024"
);

define_mlkem_prepared_keys!(
  MlKem512PreparedEncapsulationKey,
  MlKem512EncapsulationKey,
  MlKem512PreparedDecapsulationKey,
  MlKem512DecapsulationKey,
  2,
  768,
  800,
  1632,
  "ML-KEM-512"
);

define_mlkem_prepared_keys!(
  MlKem768PreparedEncapsulationKey,
  MlKem768EncapsulationKey,
  MlKem768PreparedDecapsulationKey,
  MlKem768DecapsulationKey,
  3,
  1152,
  1184,
  2400,
  "ML-KEM-768"
);

define_mlkem_prepared_keys!(
  MlKem1024PreparedEncapsulationKey,
  MlKem1024EncapsulationKey,
  MlKem1024PreparedDecapsulationKey,
  MlKem1024DecapsulationKey,
  4,
  1536,
  1568,
  3168,
  "ML-KEM-1024"
);

macro_rules! impl_mlkem_profile_ops {
  (
    $profile:ident,
    $encapsulation_key:ident,
    $decapsulation_key:ident,
    $seed:ident,
    $prepared_encapsulation_key:ident,
    $prepared_decapsulation_key:ident,
    $ciphertext:ident,
    $shared_secret:ident,
    $k:expr,
    $k_u8:expr,
    $eta1_random_bytes:expr,
    $dk_pke_bytes:expr,
    $ek_bytes:expr,
    $dk_bytes:expr,
    $ct_bytes:expr,
    $du:expr,
    $dv:expr,
    $poly_du_bytes:expr,
    $poly_dv_bytes:expr,
    $keygen:path,
    $encapsulate_prepared:path,
    $decapsulate_prepared:path,
    $arc:literal,
    $doc_name:literal
  ) => {
    impl $encapsulation_key {
      /// DER-encoded RFC 9935 SubjectPublicKeyInfo length in bytes.
      pub const SPKI_DER_LENGTH: usize = pkix::SPKI_HEADER_LENGTH.strict_add($ek_bytes);

      const SPKI_HEADER: [u8; pkix::SPKI_HEADER_LENGTH] = pkix::spki_header($profile::ALGORITHM, $ek_bytes);

      #[doc = concat!("Parse an RFC 9935 SubjectPublicKeyInfo for ", $doc_name, ".")]
      ///
      /// Accepts only the unique DER encoding: this parameter set's algorithm
      /// identifier with absent parameters, a BIT STRING with no unused bits,
      /// the exact key length, and no trailing input. The key must pass the
      /// FIPS 203 modulus check. No heap allocation.
      ///
      /// # Errors
      ///
      /// Returns [`MlKemKeyError::UnsupportedAlgorithm`] for a well-formed key
      /// of another algorithm or parameter set; [`MlKemKeyError::InvalidEncapsulationKey`]
      /// for a wrong length or a failed modulus check; and
      /// [`MlKemKeyError::MalformedDer`] for any other encoding.
      pub fn from_spki_der(der: &[u8]) -> Result<Self, MlKemKeyError> {
        let key = Self::from_bytes(*pkix::decode_spki::<MlKemKeyError, $ek_bytes>(der, &Self::SPKI_HEADER)?);
        key.validate().map_err(|_| MlKemKeyError::InvalidEncapsulationKey)?;
        Ok(key)
      }

      /// Encode the RFC 9935 SubjectPublicKeyInfo DER. No heap allocation.
      #[must_use]
      pub const fn to_spki_der(&self) -> [u8; Self::SPKI_DER_LENGTH] {
        pkix::concat(&Self::SPKI_HEADER, &self.0)
      }

      #[doc = concat!("Parse and validate an ", $doc_name, " encapsulation key from raw bytes.")]
      #[inline]
      pub fn try_from_slice(bytes: &[u8]) -> Result<Self, MlKemError> {
        if bytes.len() != Self::LENGTH {
          return Err(MlKemError::InvalidEncapsulationKey);
        }

        let mut key = [0u8; Self::LENGTH];
        key.copy_from_slice(bytes);
        let key = Self::from_bytes(key);
        key.validate()?;
        Ok(key)
      }

      /// Validate this encapsulation key using the FIPS 203 modulus check.
      #[inline]
      pub fn validate(&self) -> Result<(), MlKemError> {
        operations::validate_encapsulation_key::<$k, $ek_bytes>(self.as_bytes())
      }

      /// Validate once and prepare this key for repeated encapsulation.
      #[inline]
      pub fn prepare(&self) -> Result<$prepared_encapsulation_key, MlKemError> {
        $prepared_encapsulation_key::try_from(self)
      }
    }

    impl $decapsulation_key {
      #[doc = concat!("Parse and validate an ", $doc_name, " decapsulation key from raw bytes.")]
      ///
      /// Runs the FIPS 203 length and hash checks, which do not detect a secret
      /// vector that disagrees with the embedded encapsulation key. PKCS #8
      /// import of an expanded key adds a pairwise consistency check; see
      /// [`Self::from_pkcs8_der`].
      #[inline]
      pub fn try_from_slice(bytes: &[u8]) -> Result<Self, MlKemError> {
        if bytes.len() != Self::LENGTH {
          return Err(MlKemError::InvalidDecapsulationKey);
        }

        let mut key = [0u8; Self::LENGTH];
        key.copy_from_slice(bytes);
        let key = Self::from_bytes(key);
        key.validate()?;
        Ok(key)
      }

      /// Like [`Self::try_from_slice`], with the key imported into memory from `alloc`.
      ///
      /// The key is copied directly into its allocation, so moving the box moves
      /// only a pointer. The box clears the key on drop; a rejected key is
      /// cleared before returning. Allocation failure is handled as by
      /// [`Box::new_in`]. The caller retains responsibility for clearing `bytes`.
      ///
      /// # Errors
      ///
      /// Returns [`MlKemError::InvalidDecapsulationKey`] for a wrong length or a
      /// failed FIPS 203 embedded-key hash check.
      #[cfg(feature = "alloc")]
      pub fn try_from_slice_in<A: Allocator>(bytes: &[u8], alloc: A) -> Result<Box<Self, A>, MlKemError> {
        if bytes.len() != Self::LENGTH {
          return Err(MlKemError::InvalidDecapsulationKey);
        }
        let mut key = Box::new_in(Self::from_bytes([0; Self::LENGTH]), alloc);
        key.0.copy_from_slice(bytes);
        key.validate()?;
        Ok(key)
      }

      /// Pairwise consistency check for imported expanded keys.
      fn check_key_pair(&self) -> Result<(), MlKemKeyError> {
        operations::check_decapsulation_key_pair::<
          $k,
          $eta1_random_bytes,
          $dk_pke_bytes,
          $ek_bytes,
          $dk_bytes,
          $ct_bytes,
          $du,
          $dv,
          $poly_du_bytes,
          $poly_dv_bytes,
        >(self.as_bytes())
        .map_err(expanded_import_error)
      }

      /// DER length of this key as an RFC 9935 expanded-form PKCS #8 private key.
      pub const PKCS8_DER_LENGTH: usize = pkix::EXPANDED_HEADER_LENGTH.strict_add($dk_bytes);

      const PKCS8_HEADER: [u8; pkix::EXPANDED_HEADER_LENGTH] = pkix::expanded_header($profile::ALGORITHM, $dk_bytes);

      #[doc = concat!("Import an RFC 9935 private key for ", $doc_name, " from RFC 5958 OneAsymmetricKey (PKCS #8) DER.")]
      ///
      /// Accepts the seed, expanded, and both forms, in a version 1 container
      /// or a version 2 container with a public key. An expanded key gets the
      /// checks of [`Self::try_from_slice`] and a pairwise consistency check:
      /// a deterministic encapsulation to the embedded encapsulation key must
      /// decapsulate to the same shared secret, which rejects a secret vector
      /// that the hash check accepts (RFC 9935 Appendix C.4.1, second
      /// example). That costs one encapsulation and one decapsulation per
      /// import. A seed is expanded and not retained. In the both form, the expanded key must equal the seed's;
      /// a version 2 public key must equal the embedded encapsulation key. The
      /// caller retains responsibility for clearing `der`. No heap allocation.
      #[doc = concat!("Use [`", stringify!($seed), "::from_pkcs8_der`] to keep the seed.")]
      ///
      /// # Errors
      ///
      /// Returns [`MlKemKeyError::MalformedDer`] for malformed or non-canonical
      /// DER, including a version that disagrees with the public-key field;
      /// [`MlKemKeyError::UnsupportedAlgorithm`] for another algorithm or
      /// parameter set; [`MlKemKeyError::UnsupportedEncoding`] for attributes;
      /// [`MlKemKeyError::InvalidDecapsulationKey`] for a wrong-length or
      /// invalid key, or redundant fields that disagree; and
      /// [`MlKemKeyError::InvalidEncapsulationKey`] for a wrong-length public key.
      pub fn from_pkcs8_der(der: &[u8]) -> Result<Self, MlKemKeyError> {
        let _dit = DataIndependentTiming::enter();
        let decoded = pkix::decode_pkcs8::<MlKemKeyError, ML_KEM_KEY_GENERATION_RANDOM_SIZE, $dk_bytes, $ek_bytes>(
          der,
          $profile::ALGORITHM,
        )?;
        let (key, expanded) = match decoded.private_key {
          pkix::PrivateKey::Seed(seed) => ($profile::keypair_from_seed(seed).1, None),
          pkix::PrivateKey::Expanded(expanded) => {
            let key = Self::try_from_slice(expanded).map_err(expanded_import_error)?;
            key.check_key_pair()?;
            (key, None)
          }
          pkix::PrivateKey::Both { seed, expanded } => ($profile::keypair_from_seed(seed).1, Some(expanded)),
        };
        key.check_redundant(expanded, decoded.public_key)?;
        Ok(key)
      }

      /// Like [`Self::from_pkcs8_der`], with the key in memory from `alloc`.
      ///
      /// The key is written directly into its allocation, as by
      /// [`Self::try_from_slice_in`]. The box clears the key on drop; a
      /// rejected key is cleared before returning. Allocation failure is
      /// handled as by [`Box::new_in`].
      #[cfg(feature = "alloc")]
      pub fn from_pkcs8_der_in<A: Allocator>(der: &[u8], alloc: A) -> Result<Box<Self, A>, MlKemKeyError> {
        let _dit = DataIndependentTiming::enter();
        let decoded = pkix::decode_pkcs8::<MlKemKeyError, ML_KEM_KEY_GENERATION_RANDOM_SIZE, $dk_bytes, $ek_bytes>(
          der,
          $profile::ALGORITHM,
        )?;
        let (key, expanded) = match decoded.private_key {
          pkix::PrivateKey::Seed(seed) => ($profile::keypair_from_seed_in(seed, alloc).1, None),
          pkix::PrivateKey::Expanded(expanded) => {
            let key = Self::try_from_slice_in(expanded, alloc).map_err(expanded_import_error)?;
            key.check_key_pair()?;
            (key, None)
          }
          pkix::PrivateKey::Both { seed, expanded } => ($profile::keypair_from_seed_in(seed, alloc).1, Some(expanded)),
        };
        key.check_redundant(expanded, decoded.public_key)?;
        Ok(key)
      }

      /// Write this key as an RFC 9935 expanded-form private key in version 1
      /// OneAsymmetricKey (PKCS #8) DER. No heap allocation.
      ///
      /// `out` then holds the secret key, and the caller owns its cleanup.
      /// This key keeps no seed, so it cannot write the recommended seed form.
      #[doc = concat!("[`", stringify!($seed), "::to_pkcs8_der_into`] writes it.")]
      pub fn to_pkcs8_der_into(&self, out: &mut [u8; Self::PKCS8_DER_LENGTH]) {
        pkix::write(&Self::PKCS8_HEADER, &self.0, out);
      }

      /// Reject imported redundant fields that disagree with this key: an
      /// expanded key that is not its seed's (RFC 9935 section 8), or a
      /// public key that is not the embedded encapsulation key.
      fn check_redundant(
        &self,
        expanded: Option<&[u8; $dk_bytes]>,
        public: Option<&[u8; $ek_bytes]>,
      ) -> Result<(), MlKemKeyError> {
        if let Some(expanded) = expanded
          && !ct::fixed_eq(&self.0, expanded).declassify()
        {
          return Err(MlKemKeyError::InvalidDecapsulationKey);
        }
        let embedded = &self.0[$dk_pke_bytes..$dk_pke_bytes + $ek_bytes];
        if public.is_some_and(|public| public.as_slice() != embedded) {
          return Err(MlKemKeyError::InvalidDecapsulationKey);
        }
        Ok(())
      }

      /// Validate this decapsulation key using the FIPS 203 embedded-key hash check.
      #[inline]
      pub fn validate(&self) -> Result<(), MlKemError> {
        operations::validate_decapsulation_key::<$dk_pke_bytes, $ek_bytes, $dk_bytes>(self.as_bytes())
      }

      /// Validate once and prepare this key for repeated decapsulation.
      #[inline]
      pub fn prepare(&self) -> Result<$prepared_decapsulation_key, MlKemError> {
        $prepared_decapsulation_key::try_from(self)
      }

      /// Like [`Self::prepare`], with the prepared key in memory from `alloc`.
      ///
      /// The key is validated and prepared directly into its allocation, so
      /// moving the box moves only a pointer and no by-value copy of the
      /// prepared secrets is left behind. The box clears them on drop; a
      /// rejected key returns before they are written. Allocation failure is
      /// handled as by [`Box::new_in`].
      #[cfg(feature = "alloc")]
      pub fn prepare_in<A: Allocator>(&self, alloc: A) -> Result<Box<$prepared_decapsulation_key, A>, MlKemError> {
        let mut prepared = Box::new_in($prepared_decapsulation_key::zeroed(), alloc);
        operations::validate_and_prepare_decapsulation_key_into::<$k, $dk_pke_bytes, $ek_bytes, $dk_bytes>(
          self.as_bytes(),
          &mut prepared.arithmetic,
        )?;
        prepared.key.0.copy_from_slice(self.as_bytes());
        Ok(prepared)
      }
    }

    impl $prepared_decapsulation_key {
      /// Zero-filled owner that an allocation is filled from; never returned.
      #[cfg(feature = "alloc")]
      const fn zeroed() -> Self {
        Self {
          key: $decapsulation_key::from_bytes([0; $dk_bytes]),
          arithmetic: operations::PreparedDecapsulationArithmetic::zeroed(),
        }
      }
    }

    impl $ciphertext {
      #[doc = concat!("Parse and validate an ", $doc_name, " ciphertext from raw bytes.")]
      #[inline]
      pub fn try_from_slice(bytes: &[u8]) -> Result<Self, MlKemError> {
        if bytes.len() != Self::LENGTH {
          return Err(MlKemError::InvalidCiphertext);
        }

        let mut ciphertext = [0u8; Self::LENGTH];
        ciphertext.copy_from_slice(bytes);
        Ok(Self::from_bytes(ciphertext))
      }

      /// Validate this ciphertext's FIPS 203 type check.
      ///
      /// ML-KEM ciphertext validation is only a length/type check. The typed wrapper
      /// already enforces that invariant, so every constructed value is valid.
      #[inline]
      pub const fn validate(&self) -> Result<(), MlKemError> {
        let _ = self;
        Ok(())
      }
    }

    impl $profile {
      #[doc = concat!("RFC 9935 `id-alg-ml-kem-*` identifier: 2.16.840.1.101.3.4.4.", stringify!($arc), ".")]
      const ALGORITHM: pkix::Algorithm = pkix::Algorithm { family: 4, arc: $arc };

      /// FIPS 203 ML-KEM.KeyGen_internal from a `d || z` seed.
      fn keypair_from_seed(seed: &[u8; ML_KEM_KEY_GENERATION_RANDOM_SIZE]) -> ($encapsulation_key, $decapsulation_key) {
        let (ek, dk) = $keygen(seed);
        ($encapsulation_key::from_bytes(ek), $decapsulation_key::from_bytes(dk))
      }

      /// Like [`Self::keypair_from_seed`], with the decapsulation key generated
      /// directly into an allocation from `alloc`.
      #[cfg(feature = "alloc")]
      fn keypair_from_seed_in<A: Allocator>(
        seed: &[u8; ML_KEM_KEY_GENERATION_RANDOM_SIZE],
        alloc: A,
      ) -> ($encapsulation_key, Box<$decapsulation_key, A>) {
        let mut encapsulation_key = [0u8; $ek_bytes];
        let mut decapsulation_key = Box::new_in($decapsulation_key::from_bytes([0; $dk_bytes]), alloc);
        operations::keygen_into::<$k, $k_u8, $eta1_random_bytes, $dk_pke_bytes, $ek_bytes, $dk_bytes>(
          seed,
          &mut encapsulation_key,
          &mut decapsulation_key.0,
        );
        ($encapsulation_key::from_bytes(encapsulation_key), decapsulation_key)
      }

      /// Like [`Kem::generate_keypair`], with the decapsulation key in memory
      /// from `alloc`.
      ///
      /// The key is generated directly into its allocation, so moving the box
      /// moves only a pointer and no by-value copy of the key is left behind.
      /// The box clears the key on drop; entropy failure returns before any
      /// key is written. Allocation failure is handled as by [`Box::new_in`].
      ///
      /// # Errors
      ///
      /// Returns the error from `fill_random`.
      #[cfg(feature = "alloc")]
      pub fn generate_keypair_in<A: Allocator>(
        mut fill_random: impl FnMut(&mut [u8]) -> Result<(), MlKemError>,
        alloc: A,
      ) -> Result<($encapsulation_key, Box<$decapsulation_key, A>), MlKemError> {
        let mut random = ZeroizingBytes::<{ Self::KEY_GENERATION_RANDOM_SIZE }>::zeroed();
        fill_random(random.as_mut_array())?;
        let mut encapsulation_key = [0u8; $ek_bytes];
        let mut decapsulation_key = Box::new_in($decapsulation_key::from_bytes([0; $dk_bytes]), alloc);
        operations::keygen_into::<$k, $k_u8, $eta1_random_bytes, $dk_pke_bytes, $ek_bytes, $dk_bytes>(
          random.as_array(),
          &mut encapsulation_key,
          &mut decapsulation_key.0,
        );
        Ok(($encapsulation_key::from_bytes(encapsulation_key), decapsulation_key))
      }

      /// Like [`Self::try_generate_keypair`], with the decapsulation key in
      /// memory from `alloc`. See [`Self::generate_keypair_in`].
      ///
      /// # Errors
      ///
      /// Returns [`MlKemError::RandomGenerationFailed`] if the entropy source is unavailable.
      #[cfg(all(feature = "alloc", feature = "getrandom"))]
      #[cfg_attr(docsrs, doc(cfg(all(feature = "alloc", feature = "getrandom"))))]
      pub fn try_generate_keypair_in<A: Allocator>(
        alloc: A,
      ) -> Result<($encapsulation_key, Box<$decapsulation_key, A>), MlKemError> {
        Self::generate_keypair_in(
          |out| getrandom::fill(out).map_err(|_| MlKemError::RandomGenerationFailed),
          alloc,
        )
      }

      #[doc = concat!("Generate an ", $doc_name, " keypair from the platform entropy source.")]
      /// # Errors
      ///
      /// Returns [`MlKemError::RandomGenerationFailed`] if the entropy source is unavailable.
      #[cfg(feature = "getrandom")]
      #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
      #[inline]
      pub fn try_generate_keypair() -> Result<($encapsulation_key, $decapsulation_key), MlKemError> {
        Self::generate_keypair(|out| getrandom::fill(out).map_err(|_| MlKemError::RandomGenerationFailed))
      }

      #[doc = concat!("Encapsulate to an ", $doc_name, " key with platform entropy.")]
      /// # Errors
      ///
      /// Returns [`MlKemError`] if the encapsulation key is invalid or the entropy source is
      /// unavailable.
      #[cfg(feature = "getrandom")]
      #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
      #[inline]
      pub fn try_encapsulate(
        encapsulation_key: &$encapsulation_key,
      ) -> Result<($ciphertext, $shared_secret), MlKemError> {
        Self::encapsulate(encapsulation_key, |out| {
          getrandom::fill(out).map_err(|_| MlKemError::RandomGenerationFailed)
        })
      }

      #[doc = concat!("Validate and prepare an ", $doc_name, " encapsulation key for repeated operations.")]
      #[inline]
      pub fn prepare_encapsulation_key(
        encapsulation_key: &$encapsulation_key,
      ) -> Result<$prepared_encapsulation_key, MlKemError> {
        encapsulation_key.prepare()
      }

      #[doc = concat!("Validate and prepare an ", $doc_name, " decapsulation key for repeated operations.")]
      #[inline]
      pub fn prepare_decapsulation_key(
        decapsulation_key: &$decapsulation_key,
      ) -> Result<$prepared_decapsulation_key, MlKemError> {
        decapsulation_key.prepare()
      }

      #[doc = concat!("Encapsulate with a prepared ", $doc_name, " encapsulation key.")]
      #[inline]
      pub fn encapsulate_prepared(
        encapsulation_key: &$prepared_encapsulation_key,
        fill_random: impl FnMut(&mut [u8]) -> Result<(), MlKemError>,
      ) -> Result<($ciphertext, $shared_secret), MlKemError> {
        encapsulation_key.encapsulate(fill_random)
      }

      #[doc = concat!("Decapsulate with a prepared ", $doc_name, " decapsulation key.")]
      #[inline]
      pub fn decapsulate_prepared(
        decapsulation_key: &$prepared_decapsulation_key,
        ciphertext: &$ciphertext,
      ) -> Result<$shared_secret, MlKemError> {
        decapsulation_key.decapsulate(ciphertext)
      }
    }

    impl $prepared_encapsulation_key {
      #[doc = concat!("Encapsulate with this prepared ", $doc_name, " encapsulation key.")]
      #[inline]
      pub fn encapsulate(
        &self,
        mut fill_random: impl FnMut(&mut [u8]) -> Result<(), MlKemError>,
      ) -> Result<($ciphertext, $shared_secret), MlKemError> {
        let mut random = ZeroizingBytes::<{ $profile::ENCAPSULATION_RANDOM_SIZE }>::zeroed();
        fill_random(random.as_mut_array())?;
        let (ciphertext, shared_secret) = $encapsulate_prepared(&self.arithmetic, self.key_hash(), random.as_array());
        Ok((
          $ciphertext::from_bytes(ciphertext),
          $shared_secret::from_bytes(shared_secret),
        ))
      }
    }
    impl $prepared_decapsulation_key {
      #[doc = concat!("Decapsulate with this prepared ", $doc_name, " decapsulation key.")]
      #[inline]
      pub fn decapsulate(&self, ciphertext: &$ciphertext) -> Result<$shared_secret, MlKemError> {
        ciphertext.validate()?;
        Ok($shared_secret::from_bytes($decapsulate_prepared(
          self.as_bytes(),
          &self.arithmetic,
          ciphertext.as_bytes(),
        )))
      }
    }
    impl Kem for $profile {
      const ENCAPSULATION_KEY_SIZE: usize = Self::ENCAPSULATION_KEY_SIZE;
      const DECAPSULATION_KEY_SIZE: usize = Self::DECAPSULATION_KEY_SIZE;
      const CIPHERTEXT_SIZE: usize = Self::CIPHERTEXT_SIZE;
      const SHARED_SECRET_SIZE: usize = Self::SHARED_SECRET_SIZE;

      type EncapsulationKey = $encapsulation_key;
      type DecapsulationKey = $decapsulation_key;
      type Ciphertext = $ciphertext;
      type SharedSecret = $shared_secret;
      type KeyGenerationError = MlKemError;
      type EncapsulationError = MlKemError;
      type DecapsulationError = MlKemError;

      fn generate_keypair(
        mut fill_random: impl FnMut(&mut [u8]) -> Result<(), Self::KeyGenerationError>,
      ) -> Result<(Self::EncapsulationKey, Self::DecapsulationKey), Self::KeyGenerationError> {
        let mut random = ZeroizingBytes::<{ Self::KEY_GENERATION_RANDOM_SIZE }>::zeroed();
        fill_random(random.as_mut_array())?;
        let (ek, dk) = $keygen(random.as_array());
        Ok(($encapsulation_key::from_bytes(ek), $decapsulation_key::from_bytes(dk)))
      }

      fn encapsulate(
        encapsulation_key: &Self::EncapsulationKey,
        mut fill_random: impl FnMut(&mut [u8]) -> Result<(), Self::EncapsulationError>,
      ) -> Result<(Self::Ciphertext, Self::SharedSecret), Self::EncapsulationError> {
        encapsulation_key.validate()?;

        let mut random = ZeroizingBytes::<{ Self::ENCAPSULATION_RANDOM_SIZE }>::zeroed();
        fill_random(random.as_mut_array())?;
        let (ciphertext, shared_secret) = operations::encapsulate::<
          $k,
          $eta1_random_bytes,
          $dk_pke_bytes,
          $ek_bytes,
          $ct_bytes,
          $du,
          $dv,
          $poly_du_bytes,
          $poly_dv_bytes,
        >(encapsulation_key.as_bytes(), random.as_array());
        Ok((
          $ciphertext::from_bytes(ciphertext),
          $shared_secret::from_bytes(shared_secret),
        ))
      }

      fn decapsulate(
        decapsulation_key: &Self::DecapsulationKey,
        ciphertext: &Self::Ciphertext,
      ) -> Result<Self::SharedSecret, Self::DecapsulationError> {
        decapsulation_key.validate()?;
        ciphertext.validate()?;
        Ok($shared_secret::from_bytes(operations::decapsulate::<
          $k,
          $eta1_random_bytes,
          $dk_pke_bytes,
          $ek_bytes,
          $dk_bytes,
          $ct_bytes,
          $du,
          $dv,
          $poly_du_bytes,
          $poly_dv_bytes,
        >(
          decapsulation_key.as_bytes(),
          ciphertext.as_bytes(),
        )?))
      }
    }

    #[doc = concat!("FIPS 203 key-generation seed `d || z` for ", $doc_name, ": the RFC 9935 recommended private-key form.")]
    ///
    /// The seed determines the key pair, so protect it like the decapsulation
    /// key. Not `Clone` or `Copy`; zeroized on drop; `Debug` is redacted.
    pub struct $seed(ZeroizingBytes<ML_KEM_KEY_GENERATION_RANDOM_SIZE>);

    impl $seed {
      /// Seed length in bytes.
      pub const LENGTH: usize = ML_KEM_KEY_GENERATION_RANDOM_SIZE;

      /// DER length of the RFC 9935 seed-form PKCS #8 private key.
      pub const PKCS8_DER_LENGTH: usize = pkix::SEED_HEADER_LENGTH.strict_add(Self::LENGTH);

      const PKCS8_HEADER: [u8; pkix::SEED_HEADER_LENGTH] =
        pkix::seed_header::<ML_KEM_KEY_GENERATION_RANDOM_SIZE>($profile::ALGORITHM);

      /// Wrap a `d || z` seed. The caller retains responsibility for clearing `bytes`.
      #[must_use]
      pub const fn from_bytes(bytes: [u8; ML_KEM_KEY_GENERATION_RANDOM_SIZE]) -> Self {
        Self(ZeroizingBytes::new(bytes))
      }

      /// Generate a seed using 64 bytes from `fill_random`.
      /// The callback must fill the entire buffer with fresh cryptographic randomness.
      /// Failure clears even a partially filled buffer and returns no seed.
      pub fn generate(mut fill_random: impl FnMut(&mut [u8]) -> Result<(), MlKemError>) -> Result<Self, MlKemError> {
        let mut seed = Self(ZeroizingBytes::zeroed());
        fill_random(seed.0.as_mut_array())?;
        Ok(seed)
      }

      /// Generate a seed using OS entropy.
      #[cfg(feature = "getrandom")]
      #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
      pub fn try_generate() -> Result<Self, MlKemError> {
        Self::generate(|out| getrandom::fill(out).map_err(|_| MlKemError::RandomGenerationFailed))
      }

      /// Expand the seed into its key pair with FIPS 203 ML-KEM.KeyGen_internal.
      /// Gives the same keys as [`Kem::generate_keypair`] filled with this seed.
      #[must_use]
      pub fn keypair(&self) -> ($encapsulation_key, $decapsulation_key) {
        $profile::keypair_from_seed(self.0.as_array())
      }

      /// Like [`Self::keypair`], with the decapsulation key generated directly
      /// into an allocation from `alloc`. Allocation failure is handled as by
      /// [`Box::new_in`].
      #[cfg(feature = "alloc")]
      #[must_use]
      pub fn keypair_in<A: Allocator>(&self, alloc: A) -> ($encapsulation_key, Box<$decapsulation_key, A>) {
        $profile::keypair_from_seed_in(self.0.as_array(), alloc)
      }

      /// Explicitly export the seed into a zeroizing owner.
      #[must_use]
      pub fn expose_secret(&self) -> SecretBytes<ML_KEM_KEY_GENERATION_RANDOM_SIZE> {
        SecretBytes::new(*self.0.as_array())
      }

      #[doc = concat!("Import the seed from an RFC 9935 private key for ", $doc_name, " in RFC 5958 OneAsymmetricKey (PKCS #8) DER.")]
      ///
      /// Accepts the seed and both forms, in a version 1 container or a
      /// version 2 container with a public key. When the both form or a public
      /// key is present, the seed is expanded once to check them. The caller
      /// retains responsibility for clearing `der`. No heap allocation.
      ///
      /// # Errors
      ///
      #[doc = concat!("As [`", stringify!($decapsulation_key), "::from_pkcs8_der`], and [`MlKemKeyError::UnsupportedEncoding`]")]
      /// for an expanded-only key: expansion cannot recover a discarded seed.
      pub fn from_pkcs8_der(der: &[u8]) -> Result<Self, MlKemKeyError> {
        let _dit = DataIndependentTiming::enter();
        let decoded = pkix::decode_pkcs8::<MlKemKeyError, ML_KEM_KEY_GENERATION_RANDOM_SIZE, $dk_bytes, $ek_bytes>(
          der,
          $profile::ALGORITHM,
        )?;
        let (seed, expanded) = match decoded.private_key {
          pkix::PrivateKey::Seed(seed) => (seed, None),
          pkix::PrivateKey::Both { seed, expanded } => (seed, Some(expanded)),
          pkix::PrivateKey::Expanded(_) => return Err(MlKemKeyError::UnsupportedEncoding),
        };
        let owner = Self(ZeroizingBytes::new(*seed));
        if expanded.is_some() || decoded.public_key.is_some() {
          owner.keypair().1.check_redundant(expanded, decoded.public_key)?;
        }
        Ok(owner)
      }

      /// Write the seed as an RFC 9935 seed-form private key in version 1
      /// OneAsymmetricKey (PKCS #8) DER. No heap allocation.
      ///
      /// `out` then holds the seed, and the caller owns its cleanup.
      pub fn to_pkcs8_der_into(&self, out: &mut [u8; Self::PKCS8_DER_LENGTH]) {
        pkix::write(&Self::PKCS8_HEADER, self.0.as_array(), out);
      }
    }

    impl fmt::Debug for $seed {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(concat!(stringify!($seed), "(****)"))
      }
    }
  };
}

/// Expanded-key import and its pairwise check fail only as an invalid decapsulation key.
fn expanded_import_error(error: MlKemError) -> MlKemKeyError {
  debug_assert_eq!(error, MlKemError::InvalidDecapsulationKey);
  MlKemKeyError::InvalidDecapsulationKey
}

impl_mlkem_profile_ops!(
  MlKem512,
  MlKem512EncapsulationKey,
  MlKem512DecapsulationKey,
  MlKem512Seed,
  MlKem512PreparedEncapsulationKey,
  MlKem512PreparedDecapsulationKey,
  MlKem512Ciphertext,
  MlKem512SharedSecret,
  2,
  2,
  192,
  768,
  800,
  1632,
  768,
  10,
  4,
  320,
  128,
  operations::keygen::<2, 2, 192, 768, 800, 1632>,
  operations::encapsulate_prepared_512,
  operations::decapsulate_prepared_512,
  1,
  "ML-KEM-512"
);

impl_mlkem_profile_ops!(
  MlKem768,
  MlKem768EncapsulationKey,
  MlKem768DecapsulationKey,
  MlKem768Seed,
  MlKem768PreparedEncapsulationKey,
  MlKem768PreparedDecapsulationKey,
  MlKem768Ciphertext,
  MlKem768SharedSecret,
  3,
  3,
  128,
  1152,
  1184,
  2400,
  1088,
  10,
  4,
  320,
  128,
  operations::keygen::<3, 3, 128, 1152, 1184, 2400>,
  operations::encapsulate_prepared_768,
  operations::decapsulate_prepared_768,
  2,
  "ML-KEM-768"
);

impl_mlkem_profile_ops!(
  MlKem1024,
  MlKem1024EncapsulationKey,
  MlKem1024DecapsulationKey,
  MlKem1024Seed,
  MlKem1024PreparedEncapsulationKey,
  MlKem1024PreparedDecapsulationKey,
  MlKem1024Ciphertext,
  MlKem1024SharedSecret,
  4,
  4,
  128,
  1536,
  1568,
  3168,
  1568,
  11,
  5,
  352,
  160,
  operations::keygen_1024,
  operations::encapsulate_prepared_1024,
  operations::decapsulate_prepared_1024,
  3,
  "ML-KEM-1024"
);

macro_rules! mlkem_diag_keygen_secret_noise {
  ($name:ident, $k:expr, $eta1_random_bytes:expr, $dk_pke_bytes:expr, $ek_bytes:expr, $doc_name:literal) => {
    #[doc = concat!("Diagnostic digest for ", $doc_name, " PKE key generation with fixed public matrix seed.")]
    /// This is only available under `diag`; production key generation continues to derive
    /// both seeds through the FIPS 203 `G(d || k)` expansion.
    #[cfg(all(rscrypto_internal, feature = "diag"))]
    #[inline]
    #[must_use]
    pub fn $name(rho: [u8; ML_KEM_SEED_SIZE], sigma: [u8; ML_KEM_SEED_SIZE]) -> [u8; ML_KEM_SHARED_SECRET_SIZE] {
      operations::diag_keygen_secret_noise_digest::<$k, $eta1_random_bytes, $dk_pke_bytes, $ek_bytes>(&rho, &sigma)
    }
  };
}

mlkem_diag_keygen_secret_noise!(diag_mlkem512_keygen_secret_noise_digest, 2, 192, 768, 800, "ML-KEM-512");
mlkem_diag_keygen_secret_noise!(
  diag_mlkem768_keygen_secret_noise_digest,
  3,
  128,
  1152,
  1184,
  "ML-KEM-768"
);
mlkem_diag_keygen_secret_noise!(
  diag_mlkem1024_keygen_secret_noise_digest,
  4,
  128,
  1536,
  1568,
  "ML-KEM-1024"
);

#[cfg(all(rscrypto_internal, feature = "diag"))]
#[doc(hidden)]
#[inline]
#[must_use]
pub fn diag_mlkem_ntt_input_digest(poly: [u16; 256]) -> u16 {
  operations::diag_ntt_input_digest(poly)
}

/// Diagnostic digest for the s390x z/Vector NTT kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(
  rscrypto_internal,
  feature = "diag",
  target_arch = "s390x",
  not(miri),
  not(feature = "portable-only")
))]
#[doc(hidden)]
#[inline]
#[must_use]
pub unsafe fn diag_mlkem_s390x_ntt_input_digest(poly: [u16; 256]) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { operations::diag_s390x_ntt_input_digest(poly) }
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
#[doc(hidden)]
#[inline]
#[must_use]
pub fn diag_mlkem_inverse_ntt_montgomery_product_input_digest(poly: [u16; 256]) -> u16 {
  operations::diag_inverse_ntt_montgomery_product_input_digest(poly)
}

/// Diagnostic digest for the s390x z/Vector inverse-NTT kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(
  rscrypto_internal,
  feature = "diag",
  target_arch = "s390x",
  not(miri),
  not(feature = "portable-only")
))]
#[doc(hidden)]
#[inline]
#[must_use]
pub unsafe fn diag_mlkem_s390x_inverse_ntt_montgomery_product_input_digest(poly: [u16; 256]) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { operations::diag_s390x_inverse_ntt_montgomery_product_input_digest(poly) }
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
#[doc(hidden)]
#[inline]
#[must_use]
pub fn diag_mlkem_multiply_ntts_add_assign_input_digest(a: [u16; 256], b: [u16; 256], acc: [u16; 256]) -> u16 {
  operations::diag_multiply_ntts_add_assign_input_digest(a, b, acc)
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
#[doc(hidden)]
#[inline]
#[must_use]
pub fn diag_mlkem768_multiply_ntts_accumulate_input_digest(
  a: [[u16; 256]; 3],
  b: [[u16; 256]; 3],
  acc: [u16; 256],
) -> u16 {
  operations::diag_multiply_ntts_accumulate_k3_input_digest(a, b, acc)
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
#[doc(hidden)]
#[inline]
#[must_use]
pub fn diag_mlkem1024_multiply_ntts_accumulate_input_digest(
  a: [[u16; 256]; 4],
  b: [[u16; 256]; 4],
  acc: [u16; 256],
) -> u16 {
  operations::diag_multiply_ntts_accumulate_k4_input_digest(a, b, acc)
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
#[doc(hidden)]
#[inline]
#[must_use]
pub fn diag_mlkem_to_montgomery_product_domain_input_digest(poly: [u16; 256]) -> u16 {
  operations::diag_to_montgomery_product_domain_input_digest(poly)
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
#[doc(hidden)]
#[inline]
#[must_use]
pub fn diag_mlkem_from_montgomery_product_domain_input_digest(poly: [u16; 256]) -> u16 {
  operations::diag_from_montgomery_product_domain_input_digest(poly)
}

/// Diagnostic digest for the s390x z/Vector product-domain conversion kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(
  rscrypto_internal,
  feature = "diag",
  target_arch = "s390x",
  not(miri),
  not(feature = "portable-only")
))]
#[doc(hidden)]
#[inline]
#[must_use]
pub unsafe fn diag_mlkem_s390x_to_montgomery_product_domain_input_digest(poly: [u16; 256]) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { operations::diag_s390x_to_montgomery_product_domain_input_digest(poly) }
}

/// Diagnostic digest for the s390x z/Vector product-domain exit kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(
  rscrypto_internal,
  feature = "diag",
  target_arch = "s390x",
  not(miri),
  not(feature = "portable-only")
))]
#[doc(hidden)]
#[inline]
#[must_use]
pub unsafe fn diag_mlkem_s390x_from_montgomery_product_domain_input_digest(poly: [u16; 256]) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { operations::diag_s390x_from_montgomery_product_domain_input_digest(poly) }
}

/// Diagnostic digest for the s390x z/Vector base-multiply accumulator kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(
  rscrypto_internal,
  feature = "diag",
  target_arch = "s390x",
  not(miri),
  not(feature = "portable-only")
))]
#[doc(hidden)]
#[inline]
#[must_use]
pub unsafe fn diag_mlkem_s390x_multiply_ntts_add_assign_input_digest(
  a: [u16; 256],
  b: [u16; 256],
  acc: [u16; 256],
) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { operations::diag_s390x_multiply_ntts_add_assign_input_digest(a, b, acc) }
}

/// Diagnostic digest for the s390x z/Vector k=3 NTT dot-product kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(
  rscrypto_internal,
  feature = "diag",
  target_arch = "s390x",
  not(miri),
  not(feature = "portable-only")
))]
#[doc(hidden)]
#[inline]
#[must_use]
pub unsafe fn diag_mlkem_s390x_multiply_ntts_accumulate_k3_input_digest(
  a: [[u16; 256]; 3],
  b: [[u16; 256]; 3],
  acc: [u16; 256],
) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { operations::diag_s390x_multiply_ntts_accumulate_k3_input_digest(a, b, acc) }
}

/// Diagnostic digest for the s390x z/Vector k=4 NTT dot-product kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(
  rscrypto_internal,
  feature = "diag",
  target_arch = "s390x",
  not(miri),
  not(feature = "portable-only")
))]
#[doc(hidden)]
#[inline]
#[must_use]
pub unsafe fn diag_mlkem_s390x_multiply_ntts_accumulate_k4_input_digest(
  a: [[u16; 256]; 4],
  b: [[u16; 256]; 4],
  acc: [u16; 256],
) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { operations::diag_s390x_multiply_ntts_accumulate_k4_input_digest(a, b, acc) }
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
#[doc(hidden)]
#[inline]
#[must_use]
pub fn diag_mlkem_compress_decompress_values_digest(values: [u16; 4]) -> u16 {
  operations::diag_compress_decompress_values_digest(values)
}

/// Diagnostic digest for the s390x z/Vector compress/decompress kernels.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(
  rscrypto_internal,
  feature = "diag",
  target_arch = "s390x",
  not(miri),
  not(feature = "portable-only")
))]
#[doc(hidden)]
#[inline]
#[must_use]
pub unsafe fn diag_mlkem_s390x_compress_decompress_values_digest(values: [u16; 4]) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { operations::diag_s390x_compress_decompress_values_digest(values) }
}
