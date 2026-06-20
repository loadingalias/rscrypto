//! ML-KEM typed key, ciphertext, and shared-secret foundations.
//!
//! This module defines the public type surface for FIPS 203 ML-KEM parameter
//! sets. The portable arithmetic and operations are added separately so the
//! API contract can settle before backend work begins.

mod portable;

use core::{
  error::Error,
  fmt,
  hash::{Hash, Hasher},
};

use crate::{
  SecretBytes,
  traits::{Kem, ct},
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
        ct::constant_time_eq(&self.0, &other.0)
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

    impl_ct_eq!($name);
    impl_hex_fmt!($name);
    impl_serde_bytes!($name);
  };
}

macro_rules! define_mlkem_secret_bytes {
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

      /// Explicitly extract the secret bytes into a zeroizing wrapper.
      #[inline]
      #[must_use]
      pub fn expose_secret(&self) -> SecretBytes<{ Self::LENGTH }> {
        SecretBytes::new(self.0)
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

    impl PartialEq for $name {
      #[inline]
      fn eq(&self, other: &Self) -> bool {
        ct::constant_time_eq(&self.0, &other.0)
      }
    }

    impl Eq for $name {}

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

    impl_ct_eq!($name);
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
      arithmetic: portable::PreparedEncapsulationArithmetic<$k>,
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
        let arithmetic = portable::validate_and_prepare_encapsulation_key::<$k, $ek_bytes>(key.as_bytes())?;
        let key_hash = portable::encapsulation_key_hash(key.as_bytes());
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
        let arithmetic = portable::validate_and_prepare_encapsulation_key::<$k, $ek_bytes>(key.as_bytes())?;
        let key_hash = portable::encapsulation_key_hash(key.as_bytes());
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
        ct::constant_time_eq(self.as_bytes(), other.as_bytes())
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

    impl crate::traits::ConstantTimeEq for $prepared_encapsulation_key {
      #[inline]
      fn ct_eq(&self, other: &Self) -> bool {
        ct::constant_time_eq(self.as_bytes(), other.as_bytes())
      }
    }

    #[doc = concat!("Validated, reusable ", $doc_name, " decapsulation key.")]
    #[derive(Clone)]
    pub struct $prepared_decapsulation_key {
      key: $decapsulation_key,
      arithmetic: portable::PreparedDecapsulationArithmetic<$k>,
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

      /// Borrow the wrapped decapsulation key bytes.
      #[inline]
      #[must_use]
      pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
        self.key.as_bytes()
      }
    }

    impl core::convert::TryFrom<$decapsulation_key> for $prepared_decapsulation_key {
      type Error = MlKemError;

      #[inline]
      fn try_from(key: $decapsulation_key) -> Result<Self, Self::Error> {
        let arithmetic =
          portable::validate_and_prepare_decapsulation_key::<$k, $dk_pke_bytes, $ek_bytes, $dk_bytes>(key.as_bytes())?;
        Ok(Self { key, arithmetic })
      }
    }

    impl core::convert::TryFrom<&$decapsulation_key> for $prepared_decapsulation_key {
      type Error = MlKemError;

      #[inline]
      fn try_from(key: &$decapsulation_key) -> Result<Self, Self::Error> {
        let arithmetic =
          portable::validate_and_prepare_decapsulation_key::<$k, $dk_pke_bytes, $ek_bytes, $dk_bytes>(key.as_bytes())?;
        Ok(Self {
          key: key.clone(),
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

    impl PartialEq for $prepared_decapsulation_key {
      #[inline]
      fn eq(&self, other: &Self) -> bool {
        ct::constant_time_eq(self.as_bytes(), other.as_bytes())
      }
    }

    impl Eq for $prepared_decapsulation_key {}

    impl fmt::Debug for $prepared_decapsulation_key {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(****)", stringify!($prepared_decapsulation_key))
      }
    }

    impl crate::traits::ConstantTimeEq for $prepared_decapsulation_key {
      #[inline]
      fn ct_eq(&self, other: &Self) -> bool {
        ct::constant_time_eq(self.as_bytes(), other.as_bytes())
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
    $doc_name:literal
  ) => {
    impl $encapsulation_key {
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
        portable::validate_encapsulation_key::<$k, $ek_bytes>(self.as_bytes())
      }

      /// Validate once and prepare this key for repeated encapsulation.
      #[inline]
      pub fn prepare(&self) -> Result<$prepared_encapsulation_key, MlKemError> {
        $prepared_encapsulation_key::try_from(self)
      }
    }

    impl $decapsulation_key {
      #[doc = concat!("Parse and validate an ", $doc_name, " decapsulation key from raw bytes.")]
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

      /// Validate this decapsulation key using the FIPS 203 embedded-key hash check.
      #[inline]
      pub fn validate(&self) -> Result<(), MlKemError> {
        portable::validate_decapsulation_key::<$dk_pke_bytes, $ek_bytes, $dk_bytes>(self.as_bytes())
      }

      /// Validate once and prepare this key for repeated decapsulation.
      #[inline]
      pub fn prepare(&self) -> Result<$prepared_decapsulation_key, MlKemError> {
        $prepared_decapsulation_key::try_from(self)
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
        let mut random = [0u8; $profile::ENCAPSULATION_RANDOM_SIZE];
        fill_random(&mut random)?;
        let (ciphertext, shared_secret) = $encapsulate_prepared(&self.arithmetic, self.key_hash(), &random);
        ct::zeroize(&mut random);
        Ok((
          $ciphertext::from_bytes(ciphertext),
          $shared_secret::from_bytes(shared_secret),
        ))
      }

      #[cfg(feature = "diag")]
      #[doc(hidden)]
      #[inline]
      #[must_use]
      pub fn diag_pke_noise_ntt_digest(&self, seed: u8) -> u16 {
        let _ = self;
        portable::diag_pke_noise_ntt_digest::<$k, $eta1_random_bytes>(seed)
      }

      #[cfg(feature = "diag")]
      #[doc(hidden)]
      #[inline]
      #[must_use]
      pub fn diag_pke_matrix_u_digest(&self, seed: u16) -> u16 {
        portable::diag_pke_matrix_u_digest::<$k>(&self.arithmetic, seed)
      }

      #[cfg(feature = "diag")]
      #[doc(hidden)]
      #[inline]
      #[must_use]
      pub fn diag_pke_matrix_u_fused_digest(&self, seed: u16) -> u16 {
        portable::diag_pke_matrix_u_fused_digest::<$k>(&self.arithmetic, seed)
      }

      #[cfg(feature = "diag")]
      #[doc(hidden)]
      #[inline]
      #[must_use]
      pub fn diag_pke_inverse_u_add_digest(&self, seed: u16) -> u16 {
        let _ = self;
        portable::diag_pke_inverse_u_add_digest::<$k>(seed)
      }

      #[cfg(feature = "diag")]
      #[doc(hidden)]
      #[inline]
      #[must_use]
      pub fn diag_pke_v_digest(&self, seed: u16) -> u16 {
        portable::diag_pke_v_digest::<$k>(&self.arithmetic, seed)
      }

      #[cfg(feature = "diag")]
      #[doc(hidden)]
      #[inline]
      #[must_use]
      pub fn diag_pke_encode_digest(&self, seed: u16) -> u16 {
        let _ = self;
        portable::diag_pke_encode_digest::<$k, $ct_bytes, $du, $dv, $poly_du_bytes>(seed)
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

      #[cfg(feature = "diag")]
      #[doc(hidden)]
      #[inline]
      #[must_use]
      pub fn diag_decap_decrypt_digest(&self, ciphertext: &$ciphertext) -> u16 {
        portable::diag_decap_decrypt_digest::<$k, $ct_bytes, $du, $dv, $poly_du_bytes, $poly_dv_bytes>(
          &self.arithmetic,
          ciphertext.as_bytes(),
        )
      }

      #[cfg(feature = "diag")]
      #[doc(hidden)]
      #[inline]
      #[must_use]
      pub fn diag_decap_reencrypt_digest(&self, seed: u8) -> u16 {
        portable::diag_decap_reencrypt_digest::<
          $k,
          $eta1_random_bytes,
          $ct_bytes,
          $du,
          $dv,
          $poly_du_bytes,
          $poly_dv_bytes,
        >(&self.arithmetic, seed)
      }

      #[cfg(feature = "diag")]
      #[doc(hidden)]
      #[inline]
      #[must_use]
      pub fn diag_decap_hash_select_digest(&self, ciphertext: &$ciphertext, seed: u8) -> u16 {
        portable::diag_decap_hash_select_digest::<$dk_pke_bytes, $ek_bytes, $dk_bytes, $ct_bytes>(
          self.as_bytes(),
          ciphertext.as_bytes(),
          seed,
        )
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
        let mut random = [0u8; Self::KEY_GENERATION_RANDOM_SIZE];
        fill_random(&mut random)?;
        let (ek, dk) = $keygen(&random);
        ct::zeroize(&mut random);
        Ok(($encapsulation_key::from_bytes(ek), $decapsulation_key::from_bytes(dk)))
      }

      fn encapsulate(
        encapsulation_key: &Self::EncapsulationKey,
        mut fill_random: impl FnMut(&mut [u8]) -> Result<(), Self::EncapsulationError>,
      ) -> Result<(Self::Ciphertext, Self::SharedSecret), Self::EncapsulationError> {
        encapsulation_key.validate()?;

        let mut random = [0u8; Self::ENCAPSULATION_RANDOM_SIZE];
        fill_random(&mut random)?;
        let (ciphertext, shared_secret) = portable::encapsulate::<
          $k,
          $eta1_random_bytes,
          $dk_pke_bytes,
          $ek_bytes,
          $ct_bytes,
          $du,
          $dv,
          $poly_du_bytes,
          $poly_dv_bytes,
        >(encapsulation_key.as_bytes(), &random);
        ct::zeroize(&mut random);
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
        Ok($shared_secret::from_bytes(portable::decapsulate::<
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
        )))
      }
    }
  };
}

impl_mlkem_profile_ops!(
  MlKem512,
  MlKem512EncapsulationKey,
  MlKem512DecapsulationKey,
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
  portable::keygen::<2, 2, 192, 768, 800, 1632>,
  portable::encapsulate_prepared_512,
  portable::decapsulate_prepared_512,
  "ML-KEM-512"
);

impl_mlkem_profile_ops!(
  MlKem768,
  MlKem768EncapsulationKey,
  MlKem768DecapsulationKey,
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
  portable::keygen::<3, 3, 128, 1152, 1184, 2400>,
  portable::encapsulate_prepared_768,
  portable::decapsulate_prepared_768,
  "ML-KEM-768"
);

impl_mlkem_profile_ops!(
  MlKem1024,
  MlKem1024EncapsulationKey,
  MlKem1024DecapsulationKey,
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
  portable::keygen_1024,
  portable::encapsulate_prepared_1024,
  portable::decapsulate_prepared_1024,
  "ML-KEM-1024"
);

macro_rules! mlkem_diag_keygen_secret_noise {
  ($name:ident, $k:expr, $eta1_random_bytes:expr, $dk_pke_bytes:expr, $ek_bytes:expr, $doc_name:literal) => {
    #[doc = concat!("Diagnostic digest for ", $doc_name, " PKE key generation with fixed public matrix seed.")]
    /// This is only available under `diag`; production key generation continues to derive
    /// both seeds through the FIPS 203 `G(d || k)` expansion.
    #[cfg(feature = "diag")]
    #[inline]
    #[must_use]
    pub fn $name(rho: [u8; ML_KEM_SEED_SIZE], sigma: [u8; ML_KEM_SEED_SIZE]) -> [u8; ML_KEM_SHARED_SECRET_SIZE] {
      portable::diag_keygen_secret_noise_digest::<$k, $eta1_random_bytes, $dk_pke_bytes, $ek_bytes>(&rho, &sigma)
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

#[cfg(feature = "diag")]
macro_rules! mlkem_diag_matrix_sample {
  ($scalar:ident, $pair:ident, $quad:ident, $k:expr, $doc_name:literal) => {
    #[doc = concat!("Benchmark-only scalar matrix sampling digest for ", $doc_name, ".")]
    #[doc(hidden)]
    #[inline]
    #[must_use]
    pub fn $scalar(rho: &[u8; ML_KEM_SEED_SIZE]) -> u16 {
      portable::diag_matrix_sample_scalar_digest::<$k>(rho)
    }

    #[doc = concat!("Benchmark-only paired-XOF matrix sampling digest for ", $doc_name, ".")]
    #[doc(hidden)]
    #[inline]
    #[must_use]
    pub fn $pair(rho: &[u8; ML_KEM_SEED_SIZE]) -> u16 {
      portable::diag_matrix_sample_pair_digest::<$k>(rho)
    }

    #[doc = concat!("Benchmark-only quad-XOF matrix sampling digest for ", $doc_name, ".")]
    #[doc(hidden)]
    #[inline]
    #[must_use]
    pub fn $quad(rho: &[u8; ML_KEM_SEED_SIZE]) -> u16 {
      portable::diag_matrix_sample_quad_digest::<$k>(rho)
    }
  };
}

#[cfg(feature = "diag")]
mlkem_diag_matrix_sample!(
  diag_mlkem512_matrix_sample_scalar_digest,
  diag_mlkem512_matrix_sample_pair_digest,
  diag_mlkem512_matrix_sample_quad_digest,
  2,
  "ML-KEM-512"
);
#[cfg(feature = "diag")]
mlkem_diag_matrix_sample!(
  diag_mlkem768_matrix_sample_scalar_digest,
  diag_mlkem768_matrix_sample_pair_digest,
  diag_mlkem768_matrix_sample_quad_digest,
  3,
  "ML-KEM-768"
);
#[cfg(feature = "diag")]
mlkem_diag_matrix_sample!(
  diag_mlkem1024_matrix_sample_scalar_digest,
  diag_mlkem1024_matrix_sample_pair_digest,
  diag_mlkem1024_matrix_sample_quad_digest,
  4,
  "ML-KEM-1024"
);

#[cfg(feature = "diag")]
#[doc(hidden)]
#[inline]
#[must_use]
pub fn diag_mlkem_ntt_digest(seed: u16) -> u16 {
  portable::diag_ntt_digest(seed)
}

#[cfg(feature = "diag")]
#[doc(hidden)]
#[inline]
#[must_use]
pub fn diag_mlkem_ntt_input_digest(poly: [u16; 256]) -> u16 {
  portable::diag_ntt_input_digest(poly)
}

/// Diagnostic digest for the s390x z/Vector NTT kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(feature = "diag", target_arch = "s390x", not(miri), not(feature = "portable-only")))]
#[doc(hidden)]
#[inline]
#[must_use]
pub unsafe fn diag_mlkem_s390x_ntt_input_digest(poly: [u16; 256]) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { portable::diag_s390x_ntt_input_digest(poly) }
}

/// Diagnostic digest for the aarch64 NTT assembly kernel.
///
/// # Safety
///
/// The caller must only execute this on supported aarch64 Linux/macOS targets with baseline
/// Advanced SIMD available.
#[cfg(all(
  feature = "diag",
  target_arch = "aarch64",
  any(target_os = "macos", target_os = "linux"),
  not(miri),
  not(feature = "portable-only")
))]
#[doc(hidden)]
#[inline]
#[must_use]
pub unsafe fn diag_mlkem_aarch64_ntt_asm_digest(seed: u16) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { portable::diag_aarch64_ntt_asm_digest(seed) }
}

/// Diagnostic digest for the aarch64 NTT assembly kernel.
///
/// # Safety
///
/// The caller must only execute this on supported aarch64 Linux/macOS targets with baseline
/// Advanced SIMD available.
#[cfg(all(
  feature = "diag",
  target_arch = "aarch64",
  any(target_os = "macos", target_os = "linux"),
  not(miri),
  not(feature = "portable-only")
))]
#[doc(hidden)]
#[inline]
#[must_use]
pub unsafe fn diag_mlkem_aarch64_ntt_asm_input_digest(poly: [u16; 256]) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { portable::diag_aarch64_ntt_asm_input_digest(poly) }
}

#[cfg(feature = "diag")]
#[doc(hidden)]
#[inline]
#[must_use]
pub fn diag_mlkem_inverse_ntt_montgomery_product_digest(seed: u16) -> u16 {
  portable::diag_inverse_ntt_montgomery_product_digest(seed)
}

#[cfg(feature = "diag")]
#[doc(hidden)]
#[inline]
#[must_use]
pub fn diag_mlkem_inverse_ntt_montgomery_product_input_digest(poly: [u16; 256]) -> u16 {
  portable::diag_inverse_ntt_montgomery_product_input_digest(poly)
}

/// Diagnostic digest for the s390x z/Vector inverse-NTT kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(feature = "diag", target_arch = "s390x", not(miri), not(feature = "portable-only")))]
#[doc(hidden)]
#[inline]
#[must_use]
pub unsafe fn diag_mlkem_s390x_inverse_ntt_montgomery_product_input_digest(poly: [u16; 256]) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { portable::diag_s390x_inverse_ntt_montgomery_product_input_digest(poly) }
}

#[cfg(feature = "diag")]
#[doc(hidden)]
#[inline]
#[must_use]
pub fn diag_mlkem_multiply_ntts_add_assign_digest(seed: u16) -> u16 {
  portable::diag_multiply_ntts_add_assign_digest(seed)
}

#[cfg(feature = "diag")]
#[doc(hidden)]
#[inline]
#[must_use]
pub fn diag_mlkem_multiply_ntts_add_assign_input_digest(a: [u16; 256], b: [u16; 256], acc: [u16; 256]) -> u16 {
  portable::diag_multiply_ntts_add_assign_input_digest(a, b, acc)
}

/// Diagnostic digest for the rscrypto-owned aarch64 base-multiply accumulator.
///
/// # Safety
///
/// The caller must only execute this on supported aarch64 Linux/macOS targets with baseline
/// Advanced SIMD available.
#[cfg(all(
  feature = "diag",
  target_arch = "aarch64",
  any(target_os = "macos", target_os = "linux"),
  not(miri),
  not(feature = "portable-only")
))]
#[doc(hidden)]
#[inline]
#[must_use]
pub unsafe fn diag_mlkem_aarch64_multiply_ntts_add_assign_asm_digest(seed: u16) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { portable::diag_aarch64_multiply_ntts_add_assign_asm_digest(seed) }
}

/// Diagnostic digest for the rscrypto-owned aarch64 base-multiply accumulator.
///
/// # Safety
///
/// The caller must only execute this on supported aarch64 Linux/macOS targets with baseline
/// Advanced SIMD available.
#[cfg(all(
  feature = "diag",
  target_arch = "aarch64",
  any(target_os = "macos", target_os = "linux"),
  not(miri),
  not(feature = "portable-only")
))]
#[doc(hidden)]
#[inline]
#[must_use]
pub unsafe fn diag_mlkem_aarch64_multiply_ntts_add_assign_asm_input_digest(
  a: [u16; 256],
  b: [u16; 256],
  acc: [u16; 256],
) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { portable::diag_aarch64_multiply_ntts_add_assign_asm_input_digest(a, b, acc) }
}

#[cfg(feature = "diag")]
#[doc(hidden)]
#[inline]
#[must_use]
pub fn diag_mlkem768_multiply_ntts_accumulate_digest(seed: u16) -> u16 {
  portable::diag_multiply_ntts_accumulate_k3_digest(seed)
}

#[cfg(feature = "diag")]
#[doc(hidden)]
#[inline]
#[must_use]
pub fn diag_mlkem768_multiply_ntts_accumulate_input_digest(
  a: [[u16; 256]; 3],
  b: [[u16; 256]; 3],
  acc: [u16; 256],
) -> u16 {
  portable::diag_multiply_ntts_accumulate_k3_input_digest(a, b, acc)
}

#[cfg(feature = "diag")]
#[doc(hidden)]
#[inline]
#[must_use]
pub fn diag_mlkem1024_multiply_ntts_accumulate_digest(seed: u16) -> u16 {
  portable::diag_multiply_ntts_accumulate_k4_digest(seed)
}

#[cfg(feature = "diag")]
#[doc(hidden)]
#[inline]
#[must_use]
pub fn diag_mlkem1024_multiply_ntts_accumulate_input_digest(
  a: [[u16; 256]; 4],
  b: [[u16; 256]; 4],
  acc: [u16; 256],
) -> u16 {
  portable::diag_multiply_ntts_accumulate_k4_input_digest(a, b, acc)
}

/// Diagnostic digest for the rscrypto-owned aarch64 K=3 base-multiply accumulator.
///
/// # Safety
///
/// The caller must only execute this on supported aarch64 Linux/macOS targets with baseline
/// Advanced SIMD available.
#[cfg(all(
  feature = "diag",
  target_arch = "aarch64",
  any(target_os = "macos", target_os = "linux"),
  not(miri),
  not(feature = "portable-only")
))]
#[doc(hidden)]
#[inline]
#[must_use]
pub unsafe fn diag_mlkem768_aarch64_multiply_ntts_accumulate_asm_digest(seed: u16) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { portable::diag_aarch64_multiply_ntts_accumulate_k3_asm_digest(seed) }
}

/// Diagnostic digest for the rscrypto-owned aarch64 K=4 base-multiply accumulator.
///
/// # Safety
///
/// The caller must only execute this on supported aarch64 Linux/macOS targets with baseline
/// Advanced SIMD available.
#[cfg(all(
  feature = "diag",
  target_arch = "aarch64",
  any(target_os = "macos", target_os = "linux"),
  not(miri),
  not(feature = "portable-only")
))]
#[doc(hidden)]
#[inline]
#[must_use]
pub unsafe fn diag_mlkem1024_aarch64_multiply_ntts_accumulate_asm_digest(seed: u16) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { portable::diag_aarch64_multiply_ntts_accumulate_k4_asm_digest(seed) }
}

/// Diagnostic digest for the rscrypto-owned aarch64 K=3 base-multiply accumulator.
///
/// # Safety
///
/// The caller must only execute this on supported aarch64 Linux/macOS targets with baseline
/// Advanced SIMD available.
#[cfg(all(
  feature = "diag",
  target_arch = "aarch64",
  any(target_os = "macos", target_os = "linux"),
  not(miri),
  not(feature = "portable-only")
))]
#[doc(hidden)]
#[inline]
#[must_use]
pub unsafe fn diag_mlkem768_aarch64_multiply_ntts_accumulate_asm_input_digest(
  a: [[u16; 256]; 3],
  b: [[u16; 256]; 3],
  acc: [u16; 256],
) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { portable::diag_aarch64_multiply_ntts_accumulate_k3_asm_input_digest(a, b, acc) }
}

/// Diagnostic digest for the rscrypto-owned aarch64 K=4 base-multiply accumulator.
///
/// # Safety
///
/// The caller must only execute this on supported aarch64 Linux/macOS targets with baseline
/// Advanced SIMD available.
#[cfg(all(
  feature = "diag",
  target_arch = "aarch64",
  any(target_os = "macos", target_os = "linux"),
  not(miri),
  not(feature = "portable-only")
))]
#[doc(hidden)]
#[inline]
#[must_use]
pub unsafe fn diag_mlkem1024_aarch64_multiply_ntts_accumulate_asm_input_digest(
  a: [[u16; 256]; 4],
  b: [[u16; 256]; 4],
  acc: [u16; 256],
) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { portable::diag_aarch64_multiply_ntts_accumulate_k4_asm_input_digest(a, b, acc) }
}

#[cfg(feature = "diag")]
#[doc(hidden)]
#[inline]
#[must_use]
pub fn diag_mlkem_to_montgomery_product_domain_digest(seed: u16) -> u16 {
  portable::diag_to_montgomery_product_domain_digest(seed)
}

#[cfg(feature = "diag")]
#[doc(hidden)]
#[inline]
#[must_use]
pub fn diag_mlkem_to_montgomery_product_domain_input_digest(poly: [u16; 256]) -> u16 {
  portable::diag_to_montgomery_product_domain_input_digest(poly)
}

#[cfg(feature = "diag")]
#[doc(hidden)]
#[inline]
#[must_use]
pub fn diag_mlkem_from_montgomery_product_domain_digest(seed: u16) -> u16 {
  portable::diag_from_montgomery_product_domain_digest(seed)
}

#[cfg(feature = "diag")]
#[doc(hidden)]
#[inline]
#[must_use]
pub fn diag_mlkem_from_montgomery_product_domain_input_digest(poly: [u16; 256]) -> u16 {
  portable::diag_from_montgomery_product_domain_input_digest(poly)
}

/// Diagnostic digest for the s390x z/Vector product-domain conversion kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(feature = "diag", target_arch = "s390x", not(miri), not(feature = "portable-only")))]
#[doc(hidden)]
#[inline]
#[must_use]
pub unsafe fn diag_mlkem_s390x_to_montgomery_product_domain_input_digest(poly: [u16; 256]) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { portable::diag_s390x_to_montgomery_product_domain_input_digest(poly) }
}

/// Diagnostic digest for the s390x z/Vector product-domain exit kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(feature = "diag", target_arch = "s390x", not(miri), not(feature = "portable-only")))]
#[doc(hidden)]
#[inline]
#[must_use]
pub unsafe fn diag_mlkem_s390x_from_montgomery_product_domain_input_digest(poly: [u16; 256]) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { portable::diag_s390x_from_montgomery_product_domain_input_digest(poly) }
}

/// Diagnostic digest for the s390x z/Vector base-multiply accumulator kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(feature = "diag", target_arch = "s390x", not(miri), not(feature = "portable-only")))]
#[doc(hidden)]
#[inline]
#[must_use]
pub unsafe fn diag_mlkem_s390x_multiply_ntts_add_assign_input_digest(
  a: [u16; 256],
  b: [u16; 256],
  acc: [u16; 256],
) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { portable::diag_s390x_multiply_ntts_add_assign_input_digest(a, b, acc) }
}

/// Diagnostic digest for the s390x z/Vector k=3 NTT dot-product kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(feature = "diag", target_arch = "s390x", not(miri), not(feature = "portable-only")))]
#[doc(hidden)]
#[inline]
#[must_use]
pub unsafe fn diag_mlkem_s390x_multiply_ntts_accumulate_k3_input_digest(
  a: [[u16; 256]; 3],
  b: [[u16; 256]; 3],
  acc: [u16; 256],
) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { portable::diag_s390x_multiply_ntts_accumulate_k3_input_digest(a, b, acc) }
}

/// Diagnostic digest for the s390x z/Vector k=4 NTT dot-product kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(feature = "diag", target_arch = "s390x", not(miri), not(feature = "portable-only")))]
#[doc(hidden)]
#[inline]
#[must_use]
pub unsafe fn diag_mlkem_s390x_multiply_ntts_accumulate_k4_input_digest(
  a: [[u16; 256]; 4],
  b: [[u16; 256]; 4],
  acc: [u16; 256],
) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { portable::diag_s390x_multiply_ntts_accumulate_k4_input_digest(a, b, acc) }
}

#[cfg(feature = "diag")]
#[doc(hidden)]
#[inline]
#[must_use]
pub fn diag_mlkem_compress_decompress_digest(seed: u16) -> u16 {
  portable::diag_compress_decompress_digest(seed)
}

#[cfg(feature = "diag")]
#[doc(hidden)]
#[inline]
#[must_use]
pub fn diag_mlkem_compress_decompress_values_digest(values: [u16; 4]) -> u16 {
  portable::diag_compress_decompress_values_digest(values)
}

/// Diagnostic digest for the s390x z/Vector compress/decompress kernels.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(feature = "diag", target_arch = "s390x", not(miri), not(feature = "portable-only")))]
#[doc(hidden)]
#[inline]
#[must_use]
pub unsafe fn diag_mlkem_s390x_compress_decompress_values_digest(values: [u16; 4]) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { portable::diag_s390x_compress_decompress_values_digest(values) }
}
