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

const ML_KEM_KEY_GENERATION_RANDOM_SIZE: usize = 64;
const ML_KEM_ENCAPSULATION_RANDOM_SIZE: usize = 32;
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

impl MlKem768EncapsulationKey {
  /// Parse and validate an ML-KEM-768 encapsulation key from raw bytes.
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
    portable::validate_encapsulation_key(self.as_bytes())
  }
}

impl MlKem768DecapsulationKey {
  /// Parse and validate an ML-KEM-768 decapsulation key from raw bytes.
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
    portable::validate_decapsulation_key(self.as_bytes())
  }
}

impl MlKem768Ciphertext {
  /// Parse and validate an ML-KEM-768 ciphertext from raw bytes.
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

impl Kem for MlKem768 {
  const ENCAPSULATION_KEY_SIZE: usize = Self::ENCAPSULATION_KEY_SIZE;
  const DECAPSULATION_KEY_SIZE: usize = Self::DECAPSULATION_KEY_SIZE;
  const CIPHERTEXT_SIZE: usize = Self::CIPHERTEXT_SIZE;
  const SHARED_SECRET_SIZE: usize = Self::SHARED_SECRET_SIZE;

  type EncapsulationKey = MlKem768EncapsulationKey;
  type DecapsulationKey = MlKem768DecapsulationKey;
  type Ciphertext = MlKem768Ciphertext;
  type SharedSecret = MlKem768SharedSecret;
  type KeyGenerationError = MlKemError;
  type EncapsulationError = MlKemError;
  type DecapsulationError = MlKemError;

  fn generate_keypair(
    mut fill_random: impl FnMut(&mut [u8]) -> Result<(), Self::KeyGenerationError>,
  ) -> Result<(Self::EncapsulationKey, Self::DecapsulationKey), Self::KeyGenerationError> {
    let mut random = [0u8; Self::KEY_GENERATION_RANDOM_SIZE];
    fill_random(&mut random)?;
    let (ek, dk) = portable::keygen(&random);
    ct::zeroize(&mut random);
    Ok((
      MlKem768EncapsulationKey::from_bytes(ek),
      MlKem768DecapsulationKey::from_bytes(dk),
    ))
  }

  fn encapsulate(
    encapsulation_key: &Self::EncapsulationKey,
    mut fill_random: impl FnMut(&mut [u8]) -> Result<(), Self::EncapsulationError>,
  ) -> Result<(Self::Ciphertext, Self::SharedSecret), Self::EncapsulationError> {
    encapsulation_key.validate()?;

    let mut random = [0u8; Self::ENCAPSULATION_RANDOM_SIZE];
    fill_random(&mut random)?;
    let (ciphertext, shared_secret) = portable::encapsulate(encapsulation_key.as_bytes(), &random);
    ct::zeroize(&mut random);
    Ok((
      MlKem768Ciphertext::from_bytes(ciphertext),
      MlKem768SharedSecret::from_bytes(shared_secret),
    ))
  }

  fn decapsulate(
    decapsulation_key: &Self::DecapsulationKey,
    ciphertext: &Self::Ciphertext,
  ) -> Result<Self::SharedSecret, Self::DecapsulationError> {
    decapsulation_key.validate()?;
    ciphertext.validate()?;
    Ok(MlKem768SharedSecret::from_bytes(portable::decapsulate(
      decapsulation_key.as_bytes(),
      ciphertext.as_bytes(),
    )))
  }
}
