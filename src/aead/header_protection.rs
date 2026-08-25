//! Narrow fixed-size header-protection mask generation.
//!
//! These are the one-sample AES and ChaCha20 operations specified by
//! [RFC 9001 section 5.4](https://www.rfc-editor.org/rfc/rfc9001.html#section-5.4).
//! Packet parsing, header-bit selection, and packet protection remain protocol-layer concerns.

use core::fmt;

#[cfg(feature = "aes-gcm")]
use super::aes;
#[cfg(feature = "chacha20poly1305")]
use super::chacha20;
use crate::traits::ct;

/// Header-protection sample length in bytes.
const SAMPLE_SIZE: usize = 16;
/// Header-protection mask length in bytes.
const MASK_SIZE: usize = 5;

macro_rules! define_header_protection_key {
  ($name:ident, $len:expr, $doc:literal) => {
    #[doc = $doc]
    pub struct $name([u8; Self::LENGTH]);

    impl $name {
      /// Key length in bytes.
      pub const LENGTH: usize = $len;

      /// Construct a header-protection key from its protocol-derived bytes.
      #[inline]
      #[must_use]
      pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
        Self(bytes)
      }
    }

    impl fmt::Debug for $name {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(****)", stringify!($name))
      }
    }

    impl Drop for $name {
      #[inline]
      fn drop(&mut self) {
        ct::zeroize(&mut self.0);
      }
    }
  };
}

#[cfg(feature = "aes-gcm")]
define_header_protection_key!(
  Aes128HeaderProtectionKey,
  16,
  "A 128-bit AES header-protection key, distinct from packet-protection keys."
);

#[cfg(feature = "aes-gcm")]
define_header_protection_key!(
  Aes256HeaderProtectionKey,
  32,
  "A 256-bit AES header-protection key, distinct from packet-protection keys."
);

#[cfg(feature = "chacha20poly1305")]
define_header_protection_key!(
  ChaCha20HeaderProtectionKey,
  32,
  "A 256-bit ChaCha20 header-protection key, distinct from packet-protection keys."
);

/// AES-128 header-protection mask generator.
///
/// This is a deliberately narrow one-block capability. It accepts exactly one 16-byte sample and
/// returns only the first five bytes of AES-128 encryption. It does not expose ECB mode or the
/// remaining block bytes.
///
/// AES-NI and AES-CE backends extract the mask directly from vector state without materializing
/// the unused output bytes. Other backends clear their complete operation-local output block.
///
/// The context owns an expanded AES key schedule and is intentionally neither `Clone` nor `Copy`.
/// Mask generation does not allocate. On alloc-enabled RISC-V without a hardware AES backend,
/// construction retains the existing boxed fixslice schedule; no-alloc builds store it inline.
#[cfg(feature = "aes-gcm")]
pub struct Aes128HeaderProtection {
  expanded_key: aes::Aes128EncKey,
}

#[cfg(feature = "aes-gcm")]
impl fmt::Debug for Aes128HeaderProtection {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Aes128HeaderProtection").finish_non_exhaustive()
  }
}

#[cfg(feature = "aes-gcm")]
impl Aes128HeaderProtection {
  /// Construct a mask generator from a distinct header-protection key.
  #[inline]
  #[must_use]
  pub fn new(key: &Aes128HeaderProtectionKey) -> Self {
    Self {
      expanded_key: aes::aes128_expand_key(&key.0),
    }
  }

  /// Generate a five-byte mask from one 16-byte sample.
  #[inline]
  #[must_use]
  pub fn mask(&self, sample: &[u8; SAMPLE_SIZE]) -> [u8; MASK_SIZE] {
    aes::aes128_encrypt_block_prefix_5(&self.expanded_key, sample)
  }
}

/// AES-256 header-protection mask generator.
///
/// This is a deliberately narrow one-block capability. It accepts exactly one 16-byte sample and
/// returns only the first five bytes of AES-256 encryption. It does not expose ECB mode or the
/// remaining block bytes.
///
/// AES-NI and AES-CE backends extract the mask directly from vector state without materializing
/// the unused output bytes. Other backends clear their complete operation-local output block.
///
/// The context owns an expanded AES key schedule and is intentionally neither `Clone` nor `Copy`.
/// Mask generation does not allocate. On alloc-enabled RISC-V without a hardware AES backend,
/// construction retains the existing boxed fixslice schedule; no-alloc builds store it inline.
#[cfg(feature = "aes-gcm")]
pub struct Aes256HeaderProtection {
  expanded_key: aes::Aes256EncKey,
}

#[cfg(feature = "aes-gcm")]
impl fmt::Debug for Aes256HeaderProtection {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Aes256HeaderProtection").finish_non_exhaustive()
  }
}

#[cfg(feature = "aes-gcm")]
impl Aes256HeaderProtection {
  /// Construct a mask generator from a distinct header-protection key.
  #[inline]
  #[must_use]
  pub fn new(key: &Aes256HeaderProtectionKey) -> Self {
    Self {
      expanded_key: aes::aes256_expand_key(&key.0),
    }
  }

  /// Generate a five-byte mask from one 16-byte sample.
  #[inline]
  #[must_use]
  pub fn mask(&self, sample: &[u8; SAMPLE_SIZE]) -> [u8; MASK_SIZE] {
    aes::aes256_encrypt_block_prefix_5(&self.expanded_key, sample)
  }
}

/// ChaCha20 header-protection mask generator.
///
/// The sample's first four bytes are interpreted as a little-endian block counter and the
/// remaining twelve bytes as the nonce. The full temporary ChaCha20 block is cleared after its
/// first five bytes are copied into the returned mask.
///
/// The context owns key material and is intentionally neither `Clone` nor `Copy`.
#[cfg(feature = "chacha20poly1305")]
pub struct ChaCha20HeaderProtection {
  key: [u8; chacha20::KEY_SIZE],
}

#[cfg(feature = "chacha20poly1305")]
impl fmt::Debug for ChaCha20HeaderProtection {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("ChaCha20HeaderProtection").finish_non_exhaustive()
  }
}

#[cfg(feature = "chacha20poly1305")]
impl ChaCha20HeaderProtection {
  /// Construct a mask generator from a distinct header-protection key.
  #[inline]
  #[must_use]
  pub fn new(key: &ChaCha20HeaderProtectionKey) -> Self {
    Self { key: key.0 }
  }

  /// Generate a five-byte mask from one 16-byte sample.
  #[inline]
  #[must_use]
  pub fn mask(&self, sample: &[u8; SAMPLE_SIZE]) -> [u8; MASK_SIZE] {
    let counter = u32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]);
    let mut nonce = [0u8; chacha20::NONCE_SIZE];
    nonce.copy_from_slice(&sample[4..]);

    let mut block = chacha20::block(&self.key, counter, &nonce);
    let mut mask = [0u8; MASK_SIZE];
    mask.copy_from_slice(&block[..MASK_SIZE]);
    ct::zeroize(&mut block);
    mask
  }
}

#[cfg(feature = "chacha20poly1305")]
impl Drop for ChaCha20HeaderProtection {
  #[inline]
  fn drop(&mut self) {
    ct::zeroize(&mut self.key);
  }
}

#[cfg(all(feature = "diag", feature = "aes-gcm"))]
/// Exercise AES-128 header protection while retaining key, schedule, and temporary-block cleanup.
#[unsafe(no_mangle)]
#[inline(never)]
#[must_use]
pub fn diag_zeroize_aes128_header_protection(key: [u8; 16], sample: [u8; SAMPLE_SIZE]) -> [u8; MASK_SIZE] {
  let key = Aes128HeaderProtectionKey::from_bytes(key);
  Aes128HeaderProtection::new(&key).mask(&sample)
}

#[cfg(all(feature = "diag", feature = "aes-gcm"))]
/// Exercise AES-256 header protection while retaining key, schedule, and temporary-block cleanup.
#[unsafe(no_mangle)]
#[inline(never)]
#[must_use]
pub fn diag_zeroize_aes256_header_protection(key: [u8; 32], sample: [u8; SAMPLE_SIZE]) -> [u8; MASK_SIZE] {
  let key = Aes256HeaderProtectionKey::from_bytes(key);
  Aes256HeaderProtection::new(&key).mask(&sample)
}

#[cfg(all(feature = "diag", feature = "chacha20poly1305"))]
/// Exercise ChaCha20 header protection while retaining key, context, and temporary-block cleanup.
#[unsafe(no_mangle)]
#[inline(never)]
#[must_use]
pub fn diag_zeroize_chacha20_header_protection(key: [u8; 32], sample: [u8; SAMPLE_SIZE]) -> [u8; MASK_SIZE] {
  let key = ChaCha20HeaderProtectionKey::from_bytes(key);
  ChaCha20HeaderProtection::new(&key).mask(&sample)
}
