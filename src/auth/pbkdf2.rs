//! PBKDF2-HMAC-SHA2 key derivation (RFC 2898, NIST SP 800-132).
//!
//! Derives cryptographic keys from passwords using iterated HMAC-SHA256 or
//! HMAC-SHA512. Each iteration runs directly against cached SHA compress
//! function pointers with pre-computed HMAC prefix states — no per-iteration
//! struct creation, dispatch, or `Drop` overhead.
//!
//! # Examples
//!
//! ```rust
//! use rscrypto::Pbkdf2Sha256;
//!
//! let password = b"correct horse battery staple";
//! let salt = b"random-salt-value";
//!
//! let key = Pbkdf2Sha256::derive_key_array::<32>(password, salt, 600_000)?;
//! assert_ne!(key, [0u8; 32]);
//!
//! assert!(Pbkdf2Sha256::verify_password(password, salt, 600_000, &key).is_ok());
//! assert!(Pbkdf2Sha256::verify_password(b"wrong", salt, 600_000, &key).is_err());
//! # Ok::<(), rscrypto::auth::Pbkdf2Error>(())
//! ```
//!
//! # Security
//!
//! For NIST SP 800-132 / OWASP Password Storage Cheat Sheet compliance baselines, pair
//! type-specific minimums are:
//!
//! - `Pbkdf2Sha256`: at least `600_000` iterations
//! - `Pbkdf2Sha512`: at least `220_000` iterations
//! - salt length: at least `16` bytes (128 bits) for both types
//!
//! The one-shot password helpers enforce these minima by default. Use the
//! explicit primitive APIs only for RFC test vectors, protocol compatibility,
//! or migration code that has its own policy boundary.

use core::{fmt, num::NonZeroU32};

use super::hmac::{HmacSha256, hmac_prefix_state};
use crate::{
  hashes::crypto::{
    Sha256, Sha512,
    sha256::{H0 as SHA256_H0, dispatch as sha256_dispatch, kernels::CompressBlocksFn as Sha256CompressBlocksFn},
    sha512::{H0 as SHA512_H0, dispatch as sha512_dispatch, kernels::CompressBlocksFn as Sha512CompressBlocksFn},
  },
  traits::{VerificationError, ct},
};

const SHA256_OUTPUT_SIZE: usize = 32;
const SHA256_BLOCK_SIZE: usize = 64;
/// Maximum salt length whose `salt || INT_32_BE(block_index) || 0x80 || len`
/// payload fits in a single SHA-256 block: 64 - 4 (block index) - 1 (`0x80`)
/// - 8 (length) = 51.
const SHA256_INLINE_SALT_MAX: usize = SHA256_BLOCK_SIZE - 4 - 1 - 8;

const SHA512_OUTPUT_SIZE: usize = 64;
const SHA512_BLOCK_SIZE: usize = 128;
/// Maximum salt length whose `salt || INT_32_BE(block_index) || 0x80 || len`
/// payload fits in a single SHA-512 block: 128 - 4 - 1 - 16 = 107.
const SHA512_INLINE_SALT_MAX: usize = SHA512_BLOCK_SIZE - 4 - 1 - 16;

#[inline(always)]
#[allow(clippy::indexing_slicing)]
fn write_u32x8_be(dst: &mut [u8], words: &[u32; 8]) {
  dst[0..4].copy_from_slice(&words[0].to_be_bytes());
  dst[4..8].copy_from_slice(&words[1].to_be_bytes());
  dst[8..12].copy_from_slice(&words[2].to_be_bytes());
  dst[12..16].copy_from_slice(&words[3].to_be_bytes());
  dst[16..20].copy_from_slice(&words[4].to_be_bytes());
  dst[20..24].copy_from_slice(&words[5].to_be_bytes());
  dst[24..28].copy_from_slice(&words[6].to_be_bytes());
  dst[28..32].copy_from_slice(&words[7].to_be_bytes());
}

#[inline(always)]
#[allow(clippy::indexing_slicing)]
fn write_u64x8_be(dst: &mut [u8], words: &[u64; 8]) {
  dst[0..8].copy_from_slice(&words[0].to_be_bytes());
  dst[8..16].copy_from_slice(&words[1].to_be_bytes());
  dst[16..24].copy_from_slice(&words[2].to_be_bytes());
  dst[24..32].copy_from_slice(&words[3].to_be_bytes());
  dst[32..40].copy_from_slice(&words[4].to_be_bytes());
  dst[40..48].copy_from_slice(&words[5].to_be_bytes());
  dst[48..56].copy_from_slice(&words[6].to_be_bytes());
  dst[56..64].copy_from_slice(&words[7].to_be_bytes());
}

#[inline(always)]
fn zeroize_u32x8_no_fence(words: &mut [u32; 8]) {
  ct::zeroize_words_no_fence(words);
}

#[inline(always)]
fn zeroize_u64x8_no_fence(words: &mut [u64; 8]) {
  ct::zeroize_words_no_fence(words);
}

// ─── Error ──────────────────────────────────────────────────────────────────

/// Invalid PBKDF2 parameters.
///
/// Returned when the iteration count is zero or the derived key length exceeds
/// the RFC 2898 maximum of `(2^32 − 1) × hLen`.
///
/// # Examples
///
/// ```rust
/// use rscrypto::{Pbkdf2Sha256, auth::Pbkdf2Error};
///
/// let err = Pbkdf2Sha256::derive_key_primitive(b"pw", b"salt", 0, &mut [0u8; 32]);
/// assert_eq!(err, Err(Pbkdf2Error::InvalidIterations));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Pbkdf2Error {
  /// The iteration count must be at least 1. NIST/OWASP policy minima are documented on
  /// `Pbkdf2Sha256::MIN_RECOMMENDED_ITERATIONS` and
  /// `Pbkdf2Sha512::MIN_RECOMMENDED_ITERATIONS`.
  InvalidIterations,
  /// The iteration count is below the selected password-hashing policy.
  WeakIterations,
  /// The salt is shorter than the selected password-hashing policy allows.
  SaltTooShort,
  /// The requested output length exceeds `(2^32 − 1) × hLen`.
  OutputTooLong,
}

impl fmt::Display for Pbkdf2Error {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::InvalidIterations => f.write_str("PBKDF2 iteration count must be at least 1"),
      Self::WeakIterations => f.write_str("PBKDF2 iteration count is below the password policy minimum"),
      Self::SaltTooShort => f.write_str("PBKDF2 salt is below the password policy minimum"),
      Self::OutputTooLong => f.write_str("PBKDF2 output length exceeds algorithm maximum"),
    }
  }
}

impl core::error::Error for Pbkdf2Error {}

/// PBKDF2 password-hashing parameters.
///
/// `Pbkdf2Params` carries the salt and a nonzero iteration count. Use the
/// algorithm-specific `params` constructors for password storage so the
/// default policy is enforced. Use [`new_primitive`](Self::new_primitive) only
/// for RFC test vectors, protocol compatibility, or migration code that has
/// its own policy boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pbkdf2Params<'a> {
  salt: &'a [u8],
  iterations: NonZeroU32,
}

impl<'a> Pbkdf2Params<'a> {
  /// Build raw PBKDF2 parameters after only the RFC-level nonzero iteration
  /// check.
  ///
  /// This is the primitive/test-vector constructor. Application password
  /// storage should prefer `Pbkdf2Sha256::params` or `Pbkdf2Sha512::params`.
  ///
  /// # Errors
  ///
  /// Returns [`Pbkdf2Error::InvalidIterations`] when `iterations == 0`.
  pub fn new_primitive(salt: &'a [u8], iterations: u32) -> Result<Self, Pbkdf2Error> {
    let iterations = NonZeroU32::new(iterations).ok_or(Pbkdf2Error::InvalidIterations)?;
    Ok(Self { salt, iterations })
  }

  /// Borrow the salt.
  #[must_use]
  pub const fn salt(&self) -> &'a [u8] {
    self.salt
  }

  /// Return the iteration count.
  #[must_use]
  pub const fn iterations(&self) -> u32 {
    self.iterations.get()
  }
}

/// Policy minimums for PBKDF2 password verification and storage.
///
/// PBKDF2 has legitimate low-iteration test vectors, but password storage
/// should reject those values. The default one-shot password APIs use the
/// type-specific policy constants; use explicit policies only for migrations
/// or deployments with stricter local requirements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pbkdf2VerifyPolicy {
  /// Minimum accepted iteration count.
  pub min_iterations: u32,
  /// Minimum accepted salt length in bytes.
  pub min_salt_len: usize,
}

impl Pbkdf2VerifyPolicy {
  /// Build a policy from explicit lower bounds.
  #[must_use]
  pub const fn new(min_iterations: u32, min_salt_len: usize) -> Self {
    Self {
      min_iterations,
      min_salt_len,
    }
  }

  /// Return `true` when `params` satisfies this policy.
  #[must_use]
  pub const fn allows(&self, params: &Pbkdf2Params<'_>) -> bool {
    params.iterations() >= self.min_iterations && params.salt().len() >= self.min_salt_len
  }

  fn check(&self, params: &Pbkdf2Params<'_>) -> Result<(), Pbkdf2Error> {
    if params.iterations() < self.min_iterations {
      return Err(Pbkdf2Error::WeakIterations);
    }
    if params.salt().len() < self.min_salt_len {
      return Err(Pbkdf2Error::SaltTooShort);
    }
    Ok(())
  }
}

macro_rules! define_pbkdf2_sha2 {
  (
    $(#[$struct_meta:meta])*
    $name:ident {
      output_size_const: $output_size_const:ident,
      block_size_const: $block_size_const:ident,
      compress_ty: $compress_ty:ty,
      digest_ty: $digest_ty:ty,
      h0: $h0:path,
      dispatch: $dispatch:ident,
      new_fast_path: $new_fast_path:path,
      f_fn: $f_fn:path,
      iter1_fn: $iter1_fn:path,
      derive_key_fast_path: $derive_key_fast_path:path,
      test_oneshot: $test_oneshot:path,
      word_ty: $word_ty:ty,
      recommended_iterations: $recommended_iterations:expr,
    }
  ) => {
    $(#[$struct_meta])*
    #[derive(Clone)]
    pub struct $name {
      inner_init: [$word_ty; 8],
      outer_init: [$word_ty; 8],
      compress: $compress_ty,
    }

    impl fmt::Debug for $name {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(stringify!($name)).finish_non_exhaustive()
      }
    }

    impl $name {
      /// Digest output size in bytes.
      pub const OUTPUT_SIZE: usize = $output_size_const;
      /// Minimum iteration count recommended for compliance-sensitive deployments.
      pub const MIN_RECOMMENDED_ITERATIONS: u32 = $recommended_iterations;
      /// Minimum salt length (bytes) recommended for compliance-sensitive deployments.
      pub const MIN_SALT_LEN: usize = 16;
      /// Default password-hashing verification policy for this PBKDF2 variant.
      pub const DEFAULT_VERIFY_POLICY: Pbkdf2VerifyPolicy =
        Pbkdf2VerifyPolicy::new(Self::MIN_RECOMMENDED_ITERATIONS, Self::MIN_SALT_LEN);

      /// Build default-policy PBKDF2 password parameters.
      ///
      /// # Errors
      ///
      /// Returns [`Pbkdf2Error`] when the iteration count is zero, below the
      /// default policy minimum, or the salt is too short.
      pub fn params(salt: &[u8], iterations: u32) -> Result<Pbkdf2Params<'_>, Pbkdf2Error> {
        Self::params_with_policy(salt, iterations, &Self::DEFAULT_VERIFY_POLICY)
      }

      /// Build PBKDF2 password parameters under an explicit policy.
      ///
      /// # Errors
      ///
      /// Returns [`Pbkdf2Error`] when the iteration count is zero or the
      /// supplied policy rejects the parameters.
      pub fn params_with_policy<'a>(
        salt: &'a [u8],
        iterations: u32,
        policy: &Pbkdf2VerifyPolicy,
      ) -> Result<Pbkdf2Params<'a>, Pbkdf2Error> {
        let params = Pbkdf2Params::new_primitive(salt, iterations)?;
        policy.check(&params)?;
        Ok(params)
      }

      /// Pre-compute HMAC prefix states from `password`.
      #[must_use]
      #[allow(clippy::indexing_slicing)] // password.len() <= block size in the else branch.
      pub fn new(password: &[u8]) -> Self {
        if let Some(state) = $new_fast_path(password) {
          return state;
        }

        let compress = $dispatch::compress_dispatch().select(0);

        let mut key_block = [0u8; $block_size_const];
        if password.len() > $block_size_const {
          let digest = <$digest_ty>::digest(password);
          key_block[..$output_size_const].copy_from_slice(&digest);
        } else {
          key_block[..password.len()].copy_from_slice(password);
        }

        let (inner_init, outer_init) = hmac_prefix_state(&mut key_block, |ipad, opad| {
          let mut inner_init = $h0;
          compress(&mut inner_init, ipad);

          let mut outer_init = $h0;
          compress(&mut outer_init, opad);

          (inner_init, outer_init)
        });

        Self {
          inner_init,
          outer_init,
          compress,
        }
      }

      /// Derive a key into `okm`.
      #[inline]
      #[allow(clippy::indexing_slicing)]
      pub fn derive(&self, salt: &[u8], iterations: u32, okm: &mut [u8]) -> Result<(), Pbkdf2Error> {
        Self::derive_with_prefixes(self.compress, &self.inner_init, &self.outer_init, salt, iterations, okm)
      }

      /// Derive a key into `okm` using validated password parameters.
      #[inline]
      #[allow(clippy::indexing_slicing)]
      pub fn derive_with_params(&self, params: Pbkdf2Params<'_>, okm: &mut [u8]) -> Result<(), Pbkdf2Error> {
        self.derive(params.salt(), params.iterations(), okm)
      }

      #[inline]
      #[allow(clippy::indexing_slicing)]
      fn derive_with_prefixes(
        compress: $compress_ty,
        inner_init: &[$word_ty; 8],
        outer_init: &[$word_ty; 8],
        salt: &[u8],
        iterations: u32,
        okm: &mut [u8],
      ) -> Result<(), Pbkdf2Error> {
        if iterations == 0 {
          return Err(Pbkdf2Error::InvalidIterations);
        }
        if okm.is_empty() {
          return Ok(());
        }
        let num_blocks = okm.len().div_ceil($output_size_const);
        if num_blocks as u64 > u32::MAX as u64 {
          return Err(Pbkdf2Error::OutputTooLong);
        }

        if iterations == 1 {
          $iter1_fn(compress, inner_init, outer_init, salt, okm);
          return Ok(());
        }

        let mut block_index = 1u32;
        let mut chunks = okm.chunks_exact_mut($output_size_const);

        for chunk in chunks.by_ref() {
          // SAFETY: chunks_exact_mut yields slices whose length is exactly
          // $output_size_const, so this cast is to the same initialized bytes.
          let full_chunk = unsafe { &mut *(chunk.as_mut_ptr().cast::<[u8; $output_size_const]>()) };
          $f_fn(
            compress,
            inner_init,
            outer_init,
            salt,
            iterations,
            block_index,
            full_chunk,
          );
          block_index = block_index.strict_add(1);
        }

        let tail = chunks.into_remainder();
        if !tail.is_empty() {
          let mut block_out = [0u8; $output_size_const];
          $f_fn(
            compress,
            inner_init,
            outer_init,
            salt,
            iterations,
            block_index,
            &mut block_out,
          );
          tail.copy_from_slice(&block_out[..tail.len()]);
          ct::zeroize(&mut block_out);
        }
        Ok(())
      }

      /// Derive a key into a fixed-size array.
      pub fn derive_array<const N: usize>(&self, salt: &[u8], iterations: u32) -> Result<[u8; N], Pbkdf2Error> {
        let mut out = [0u8; N];
        self.derive(salt, iterations, &mut out)?;
        Ok(out)
      }

      /// Derive a key into a fixed-size array using validated password
      /// parameters.
      pub fn derive_array_with_params<const N: usize>(&self, params: Pbkdf2Params<'_>) -> Result<[u8; N], Pbkdf2Error> {
        let mut out = [0u8; N];
        self.derive_with_params(params, &mut out)?;
        Ok(out)
      }

      /// Verify `expected` against the derived key in constant time using the
      /// default password policy.
      #[allow(clippy::indexing_slicing)]
      #[must_use = "password verification must be checked; a dropped Result silently accepts the wrong password"]
      pub fn verify(&self, salt: &[u8], iterations: u32, expected: &[u8]) -> Result<(), VerificationError> {
        self.verify_with_policy(salt, iterations, expected, &Self::DEFAULT_VERIFY_POLICY)
      }

      /// Verify `expected` against the derived key in constant time using an
      /// explicit password policy.
      #[allow(clippy::indexing_slicing)]
      #[must_use = "password verification must be checked; a dropped Result silently accepts the wrong password"]
      pub fn verify_with_policy(
        &self,
        salt: &[u8],
        iterations: u32,
        expected: &[u8],
        policy: &Pbkdf2VerifyPolicy,
      ) -> Result<(), VerificationError> {
        let Ok(params) = Self::params_with_policy(salt, iterations, policy) else {
          return Err(VerificationError::new());
        };
        self.verify_primitive(params.salt(), params.iterations(), expected)
      }

      /// Verify `expected` against the derived key without password policy checks.
      ///
      /// This is the primitive/test-vector verification path. Stored password
      /// verification should use [`verify`](Self::verify),
      /// [`verify_with_policy`](Self::verify_with_policy), or
      /// [`verify_password`](Self::verify_password).
      #[allow(clippy::indexing_slicing)]
      #[must_use = "password verification must be checked; a dropped Result silently accepts the wrong password"]
      pub fn verify_primitive(&self, salt: &[u8], iterations: u32, expected: &[u8]) -> Result<(), VerificationError> {
        if iterations == 0 || expected.is_empty() {
          return Err(VerificationError::new());
        }
        let num_blocks = expected.len().div_ceil($output_size_const);
        if num_blocks as u64 > u32::MAX as u64 {
          return Err(VerificationError::new());
        }

        let compress = self.compress;

        let mut block_out = [0u8; $output_size_const];
        let mut acc = 0u8;

        for (i, chunk) in expected.chunks($output_size_const).enumerate() {
          let block_index = (i as u32).strict_add(1);
          $f_fn(
            compress,
            &self.inner_init,
            &self.outer_init,
            salt,
            iterations,
            block_index,
            &mut block_out,
          );
          for (&a, &b) in block_out[..chunk.len()].iter().zip(chunk.iter()) {
            acc |= a ^ b;
          }
        }

        ct::zeroize(&mut block_out);

        if core::hint::black_box(acc) == 0 {
          Ok(())
        } else {
          Err(VerificationError::new())
        }
      }

      /// Derive a password key in one shot using the default password policy.
      ///
      /// Use [`derive_key_primitive`](Self::derive_key_primitive) for RFC test
      /// vectors, low-iteration protocol compatibility, or non-password PBKDF2
      /// primitive use.
      #[inline]
      pub fn derive_key(password: &[u8], salt: &[u8], iterations: u32, okm: &mut [u8]) -> Result<(), Pbkdf2Error> {
        let params = Self::params(salt, iterations)?;
        Self::derive_key_with_params(password, params, okm)
      }

      /// Derive a password key in one shot using validated parameters.
      #[inline]
      pub fn derive_key_with_params(
        password: &[u8],
        params: Pbkdf2Params<'_>,
        okm: &mut [u8],
      ) -> Result<(), Pbkdf2Error> {
        Self::derive_key_primitive(password, params.salt(), params.iterations(), okm)
      }

      /// Derive a key in one shot without password policy checks.
      ///
      /// This is the PBKDF2 primitive/test-vector path. Application password
      /// storage should use [`derive_key`](Self::derive_key) or
      /// [`derive_key_with_params`](Self::derive_key_with_params).
      #[inline]
      pub fn derive_key_primitive(password: &[u8], salt: &[u8], iterations: u32, okm: &mut [u8]) -> Result<(), Pbkdf2Error> {
        if $derive_key_fast_path(password, salt, iterations, okm)? {
          return Ok(());
        }
        Self::new(password).derive(salt, iterations, okm)
      }

      /// Derive a password key into a fixed-size array in one shot using the
      /// default password policy.
      #[inline]
      pub fn derive_key_array<const N: usize>(
        password: &[u8],
        salt: &[u8],
        iterations: u32,
      ) -> Result<[u8; N], Pbkdf2Error> {
        let params = Self::params(salt, iterations)?;
        Self::derive_key_array_with_params(password, params)
      }

      /// Derive a password key into a fixed-size array using validated
      /// parameters.
      #[inline]
      pub fn derive_key_array_with_params<const N: usize>(
        password: &[u8],
        params: Pbkdf2Params<'_>,
      ) -> Result<[u8; N], Pbkdf2Error> {
        Self::derive_key_array_primitive(password, params.salt(), params.iterations())
      }

      /// Derive a key into a fixed-size array in one shot without password
      /// policy checks.
      #[inline]
      pub fn derive_key_array_primitive<const N: usize>(
        password: &[u8],
        salt: &[u8],
        iterations: u32,
      ) -> Result<[u8; N], Pbkdf2Error> {
        Self::new(password).derive_array(salt, iterations)
      }

      /// Verify a password in one shot using the default password policy.
      #[inline]
      #[must_use = "password verification must be checked; a dropped Result silently accepts the wrong password"]
      pub fn verify_password(
        password: &[u8],
        salt: &[u8],
        iterations: u32,
        expected: &[u8],
      ) -> Result<(), VerificationError> {
        Self::verify_password_with_policy(password, salt, iterations, expected, &Self::DEFAULT_VERIFY_POLICY)
      }

      /// Verify a password in one shot using an explicit password policy.
      #[inline]
      #[must_use = "password verification must be checked; a dropped Result silently accepts the wrong password"]
      pub fn verify_password_with_policy(
        password: &[u8],
        salt: &[u8],
        iterations: u32,
        expected: &[u8],
        policy: &Pbkdf2VerifyPolicy,
      ) -> Result<(), VerificationError> {
        Self::new(password).verify_with_policy(salt, iterations, expected, policy)
      }

      /// Verify a password in one shot without password policy checks.
      ///
      /// This is the primitive/test-vector verification path. Stored password
      /// verification should use [`verify_password`](Self::verify_password) or
      /// [`verify_password_with_policy`](Self::verify_password_with_policy).
      #[inline]
      #[must_use = "password verification must be checked; a dropped Result silently accepts the wrong password"]
      pub fn verify_password_primitive(
        password: &[u8],
        salt: &[u8],
        iterations: u32,
        expected: &[u8],
      ) -> Result<(), VerificationError> {
        Self::new(password).verify_primitive(salt, iterations, expected)
      }

      /// Test-only: build with a specific digest compress function.
      #[cfg(any(test, feature = "diag"))]
      #[allow(dead_code)]
      #[allow(clippy::indexing_slicing)]
      pub(crate) fn new_with_compress_for_test(password: &[u8], compress: $compress_ty) -> Self {
        let mut key_block = [0u8; $block_size_const];
        if password.len() > $block_size_const {
          key_block[..$output_size_const].copy_from_slice(&$test_oneshot(password, compress));
        } else {
          key_block[..password.len()].copy_from_slice(password);
        }

        let (inner_init, outer_init) = hmac_prefix_state(&mut key_block, |ipad, opad| {
          let mut inner_init = $h0;
          compress(&mut inner_init, ipad);

          let mut outer_init = $h0;
          compress(&mut outer_init, opad);

          (inner_init, outer_init)
        });

        Self {
          inner_init,
          outer_init,
          compress,
        }
      }
    }

    impl Drop for $name {
      fn drop(&mut self) {
        for word in self.inner_init.iter_mut().chain(self.outer_init.iter_mut()) {
          // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
          unsafe { core::ptr::write_volatile(word, 0) };
        }
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
      }
    }
  };
}

// ─── PBKDF2-HMAC-SHA256 ────────────────────────────────────────────────────

define_pbkdf2_sha2! {
  /// PBKDF2-HMAC-SHA256 key derivation (RFC 2898).
  ///
  /// Pre-computes the HMAC-SHA256 prefix states from the password so that
  /// subsequent `derive` and `verify` calls run the iteration loop directly
  /// against the SHA-256 compress function with no per-iteration overhead.
  ///
  /// # Examples
  ///
  /// ```rust
  /// use rscrypto::Pbkdf2Sha256;
  ///
  /// // Derive a 32-byte key
  /// let dk = Pbkdf2Sha256::derive_key_array::<32>(b"password", b"salt-16-bytes!!!", 600_000)?;
  ///
  /// // Re-use pre-computed state for multiple operations
  /// let state = Pbkdf2Sha256::new(b"password");
  /// let params = Pbkdf2Sha256::params(b"salt-16-bytes!!!", 600_000)?;
  /// let dk2 = state.derive_array_with_params::<32>(params)?;
  /// assert_eq!(dk, dk2);
  ///
  /// // Verify
  /// assert!(state.verify(params.salt(), params.iterations(), &dk).is_ok());
  /// # Ok::<(), rscrypto::auth::Pbkdf2Error>(())
  /// ```
  Pbkdf2Sha256 {
    output_size_const: SHA256_OUTPUT_SIZE,
    block_size_const: SHA256_BLOCK_SIZE,
    compress_ty: Sha256CompressBlocksFn,
    digest_ty: Sha256,
    h0: SHA256_H0,
    dispatch: sha256_dispatch,
    new_fast_path: pbkdf2_sha256_new_fast_path,
    f_fn: pbkdf2_sha256_f,
    iter1_fn: pbkdf2_sha256_iter1,
    derive_key_fast_path: pbkdf2_sha256_derive_key_fast_path,
    test_oneshot: sha256_oneshot_with_compress,
    word_ty: u32,
    recommended_iterations: 600_000,
  }
}

/// Test-only: one-shot SHA-256 digest using a specific compress function.
#[cfg(any(test, feature = "diag"))]
#[allow(clippy::indexing_slicing)]
fn sha256_oneshot_with_compress(data: &[u8], compress: Sha256CompressBlocksFn) -> [u8; SHA256_OUTPUT_SIZE] {
  let mut state = SHA256_H0;
  let mut pos = 0usize;
  while pos.strict_add(SHA256_BLOCK_SIZE) <= data.len() {
    compress(&mut state, &data[pos..pos.strict_add(SHA256_BLOCK_SIZE)]);
    pos = pos.strict_add(SHA256_BLOCK_SIZE);
  }
  let mut block = [0u8; SHA256_BLOCK_SIZE];
  let tail = data.len().strict_sub(pos);
  block[..tail].copy_from_slice(&data[pos..]);
  block[tail] = 0x80;
  if tail >= 56 {
    compress(&mut state, &block);
    block = [0u8; SHA256_BLOCK_SIZE];
  }
  block[56..64].copy_from_slice(&(data.len() as u64).strict_mul(8).to_be_bytes());
  compress(&mut state, &block);
  let mut out = [0u8; SHA256_OUTPUT_SIZE];
  for (chunk, &word) in out.chunks_exact_mut(4).zip(state.iter()) {
    chunk.copy_from_slice(&word.to_be_bytes());
  }
  out
}

#[cfg(feature = "diag")]
#[must_use]
pub fn diag_pbkdf2_sha256_verify_portable(
  password: &[u8; SHA256_OUTPUT_SIZE],
  expected: &[u8; SHA256_OUTPUT_SIZE],
) -> bool {
  let compress = crate::hashes::crypto::sha256::kernels::compress_blocks_fn(
    crate::hashes::crypto::sha256::kernels::Sha256KernelId::Portable,
  );
  Pbkdf2Sha256::new_with_compress_for_test(password, compress)
    .verify(b"salt", 1, expected)
    .is_ok()
}

#[cfg(feature = "diag")]
#[must_use]
pub fn diag_pbkdf2_sha512_verify_portable(
  password: &[u8; SHA512_OUTPUT_SIZE],
  expected: &[u8; SHA512_OUTPUT_SIZE],
) -> bool {
  let compress = crate::hashes::crypto::sha512::kernels::compress_blocks_fn(
    crate::hashes::crypto::sha512::kernels::Sha512KernelId::Portable,
  );
  Pbkdf2Sha512::new_with_compress_for_test(password, compress)
    .verify(b"salt", 1, expected)
    .is_ok()
}

#[inline]
#[allow(clippy::indexing_slicing)]
fn pbkdf2_sha256_new_fast_path(password: &[u8]) -> Option<Pbkdf2Sha256> {
  if !pbkdf2_sha256_spr_prefers_hmac_iter1() {
    return None;
  }

  let mut key_block = [0u8; SHA256_BLOCK_SIZE];
  if password.len() > SHA256_BLOCK_SIZE {
    let digest = Sha256::digest(password);
    key_block[..SHA256_OUTPUT_SIZE].copy_from_slice(&digest);
  } else {
    key_block[..password.len()].copy_from_slice(password);
  }

  let (inner_init, outer_init, compress) = hmac_prefix_state(&mut key_block, |ipad, opad| {
    let mut inner = Sha256::new();
    inner.update(ipad);
    let inner_prefix = inner.aligned_prefix();

    let mut outer = Sha256::new();
    outer.update(opad);
    let outer_prefix = outer.aligned_prefix();

    let (inner_init, compress) = inner_prefix.pbkdf2_parts();
    let (outer_init, _) = outer_prefix.pbkdf2_parts();
    (inner_init, outer_init, compress)
  });

  Some(Pbkdf2Sha256 {
    inner_init,
    outer_init,
    compress,
  })
}

#[inline]
#[must_use]
fn pbkdf2_sha256_spr_prefers_hmac_iter1() -> bool {
  #[cfg(target_arch = "x86_64")]
  {
    crate::platform::caps().has(crate::platform::caps::x86::INTEL_SAPPHIRE_RAPIDS)
  }

  #[cfg(not(target_arch = "x86_64"))]
  {
    false
  }
}

#[allow(clippy::indexing_slicing)]
#[inline]
fn pbkdf2_sha256_hmac_iter1_small(password: &[u8], salt: &[u8], okm: &mut [u8]) {
  debug_assert!(salt.len() <= SHA256_INLINE_SALT_MAX);
  debug_assert!(okm.len() <= SHA256_OUTPUT_SIZE * 2);

  let mut msg = [0u8; SHA256_INLINE_SALT_MAX + 4];
  msg[..salt.len()].copy_from_slice(salt);
  let msg_len = salt.len().strict_add(4);

  for (i, chunk) in okm.chunks_mut(SHA256_OUTPUT_SIZE).enumerate() {
    let block_index = (i as u32).strict_add(1);
    msg[salt.len()..msg_len].copy_from_slice(&block_index.to_be_bytes());
    let tag = HmacSha256::mac(password, &msg[..msg_len]);
    chunk.copy_from_slice(&tag.as_bytes()[..chunk.len()]);
  }
}

#[inline]
fn pbkdf2_sha256_derive_key_fast_path(
  password: &[u8],
  salt: &[u8],
  iterations: u32,
  okm: &mut [u8],
) -> Result<bool, Pbkdf2Error> {
  pbkdf2_sha256_derive_key_fast_path_with_preference(
    password,
    salt,
    iterations,
    okm,
    pbkdf2_sha256_spr_prefers_hmac_iter1(),
  )
}

#[inline]
fn pbkdf2_sha256_derive_key_fast_path_with_preference(
  password: &[u8],
  salt: &[u8],
  iterations: u32,
  okm: &mut [u8],
  prefer_hmac_iter1: bool,
) -> Result<bool, Pbkdf2Error> {
  if iterations != 1 || okm.is_empty() {
    return Ok(false);
  }
  if salt.len() > SHA256_INLINE_SALT_MAX || okm.len() > SHA256_OUTPUT_SIZE.strict_mul(2) {
    return Ok(false);
  }
  if !prefer_hmac_iter1 {
    return Ok(false);
  }

  pbkdf2_sha256_hmac_iter1_small(password, salt, okm);
  Ok(true)
}

/// Compute one PBKDF2-SHA256 block: `F(Password, Salt, c, i)`.
///
/// Each HMAC iteration in the hot loop runs exactly 2 SHA-256 compress calls
/// using pre-padded block templates — no hash struct creation, no dispatch
/// overhead, no padding recomputation.
#[allow(clippy::indexing_slicing)]
#[inline(always)]
fn pbkdf2_sha256_f(
  compress: Sha256CompressBlocksFn,
  inner_init: &[u32; 8],
  outer_init: &[u32; 8],
  salt: &[u8],
  iterations: u32,
  block_index: u32,
  output: &mut [u8; SHA256_OUTPUT_SIZE],
) {
  let mut state: [u32; 8];
  let mut u_words: [u32; 8];
  let mut result_words: [u32; 8];

  // ── U1 = HMAC(Password, Salt || INT_32_BE(block_index)) ──────────────
  state = *inner_init;
  let msg_len = salt.len().strict_add(4);
  let total_inner = (SHA256_BLOCK_SIZE as u64).strict_add(msg_len as u64);

  let mut block = [0u8; SHA256_BLOCK_SIZE];
  if salt.len() <= SHA256_INLINE_SALT_MAX {
    block[..salt.len()].copy_from_slice(salt);
    let pos = salt.len().strict_add(4);
    block[salt.len()..pos].copy_from_slice(&block_index.to_be_bytes());
    block[pos] = 0x80;
    block[56..SHA256_BLOCK_SIZE].copy_from_slice(&total_inner.strict_mul(8).to_be_bytes());
    compress(&mut state, &block);
  } else {
    let mut pos = 0usize;

    // Feed salt
    let mut salt_off = 0usize;
    while salt_off < salt.len() {
      let space = SHA256_BLOCK_SIZE.strict_sub(pos);
      let remaining = salt.len().strict_sub(salt_off);
      let take = if space < remaining { space } else { remaining };
      block[pos..pos.strict_add(take)].copy_from_slice(&salt[salt_off..salt_off.strict_add(take)]);
      pos = pos.strict_add(take);
      salt_off = salt_off.strict_add(take);
      if pos == SHA256_BLOCK_SIZE {
        compress(&mut state, &block);
        block = [0u8; SHA256_BLOCK_SIZE];
        pos = 0;
      }
    }

    // Feed block_index (4 bytes big-endian)
    for &b in &block_index.to_be_bytes() {
      block[pos] = b;
      pos = pos.strict_add(1);
      if pos == SHA256_BLOCK_SIZE {
        compress(&mut state, &block);
        block = [0u8; SHA256_BLOCK_SIZE];
        pos = 0;
      }
    }

    // SHA-256 padding
    block[pos] = 0x80;
    if pos.strict_add(1) > 56 {
      compress(&mut state, &block);
      block = [0u8; SHA256_BLOCK_SIZE];
    }
    block[56..SHA256_BLOCK_SIZE].copy_from_slice(&total_inner.strict_mul(8).to_be_bytes());
    compress(&mut state, &block);
  }

  // Outer hash of U1: single block
  let mut outer_block = [0u8; SHA256_BLOCK_SIZE];
  write_u32x8_be(&mut outer_block[..SHA256_OUTPUT_SIZE], &state);
  outer_block[SHA256_OUTPUT_SIZE] = 0x80;
  // total outer bytes = 64 (opad, pre-compressed) + 32 (inner hash) = 96 → 768 bits
  outer_block[56..SHA256_BLOCK_SIZE].copy_from_slice(&768u64.to_be_bytes());

  state = *outer_init;
  compress(&mut state, &outer_block);
  u_words = state;
  result_words = u_words;

  if iterations == 1 {
    write_u32x8_be(output, &result_words);
    ct::zeroize_no_fence(&mut outer_block);
    ct::zeroize_no_fence(&mut block);
    zeroize_u32x8_no_fence(&mut state);
    zeroize_u32x8_no_fence(&mut u_words);
    zeroize_u32x8_no_fence(&mut result_words);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    return;
  }

  // ── Iterations 2..=c (fixed-size HMAC, 2 compress calls each) ────────
  // Inner template: [32-byte U] [0x80] [zeros] [768-bit length]
  let mut inner_block = [0u8; SHA256_BLOCK_SIZE];
  inner_block[SHA256_OUTPUT_SIZE] = 0x80;
  inner_block[56..SHA256_BLOCK_SIZE].copy_from_slice(&768u64.to_be_bytes());

  for _ in 1..iterations {
    // Inner: compress(inner_init, U_{j-1} || padding)
    write_u32x8_be(&mut inner_block[..SHA256_OUTPUT_SIZE], &u_words);
    state = *inner_init;
    compress(&mut state, &inner_block);

    // Serialize inner hash directly into outer block
    write_u32x8_be(&mut outer_block[..SHA256_OUTPUT_SIZE], &state);

    // Outer: compress(outer_init, inner_hash || padding)
    state = *outer_init;
    compress(&mut state, &outer_block);

    u_words = state;

    // XOR into output
    for (dst, &word) in result_words.iter_mut().zip(u_words.iter()) {
      *dst ^= word;
    }
  }

  write_u32x8_be(output, &result_words);

  // Zeroize sensitive state
  ct::zeroize_no_fence(&mut inner_block);
  ct::zeroize_no_fence(&mut outer_block);
  ct::zeroize_no_fence(&mut block);
  zeroize_u32x8_no_fence(&mut state);
  zeroize_u32x8_no_fence(&mut u_words);
  zeroize_u32x8_no_fence(&mut result_words);
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}

#[allow(clippy::indexing_slicing)]
#[inline(always)]
fn pbkdf2_sha256_iter1(
  compress: Sha256CompressBlocksFn,
  inner_init: &[u32; 8],
  outer_init: &[u32; 8],
  salt: &[u8],
  okm: &mut [u8],
) {
  if salt.len() > SHA256_INLINE_SALT_MAX {
    let mut block_index = 1u32;
    let mut chunks = okm.chunks_exact_mut(SHA256_OUTPUT_SIZE);
    for chunk in chunks.by_ref() {
      // SAFETY: chunks_exact_mut yields slices whose length is exactly SHA256_OUTPUT_SIZE.
      let full_chunk = unsafe { &mut *(chunk.as_mut_ptr().cast::<[u8; SHA256_OUTPUT_SIZE]>()) };
      pbkdf2_sha256_f(compress, inner_init, outer_init, salt, 1, block_index, full_chunk);
      block_index = block_index.strict_add(1);
    }
    let tail = chunks.into_remainder();
    if !tail.is_empty() {
      let mut block_out = [0u8; SHA256_OUTPUT_SIZE];
      pbkdf2_sha256_f(compress, inner_init, outer_init, salt, 1, block_index, &mut block_out);
      tail.copy_from_slice(&block_out[..tail.len()]);
      ct::zeroize(&mut block_out);
    }
    return;
  }

  let msg_len = salt.len().strict_add(4);
  let total_inner = (SHA256_BLOCK_SIZE as u64).strict_add(msg_len as u64);
  let index_pos = salt.len();
  let pad_pos = index_pos.strict_add(4);

  let mut block = [0u8; SHA256_BLOCK_SIZE];
  block[..salt.len()].copy_from_slice(salt);
  block[pad_pos] = 0x80;
  block[56..SHA256_BLOCK_SIZE].copy_from_slice(&total_inner.strict_mul(8).to_be_bytes());

  let mut outer_block = [0u8; SHA256_BLOCK_SIZE];
  outer_block[SHA256_OUTPUT_SIZE] = 0x80;
  outer_block[56..SHA256_BLOCK_SIZE].copy_from_slice(&768u64.to_be_bytes());

  let mut state = [0u32; 8];
  let mut block_index = 1u32;

  let mut chunks = okm.chunks_exact_mut(SHA256_OUTPUT_SIZE);
  for chunk in chunks.by_ref() {
    block[index_pos..pad_pos].copy_from_slice(&block_index.to_be_bytes());

    state = *inner_init;
    compress(&mut state, &block);

    write_u32x8_be(&mut outer_block[..SHA256_OUTPUT_SIZE], &state);
    state = *outer_init;
    compress(&mut state, &outer_block);

    write_u32x8_be(chunk, &state);
    block_index = block_index.strict_add(1);
  }

  let tail = chunks.into_remainder();
  if !tail.is_empty() {
    block[index_pos..pad_pos].copy_from_slice(&block_index.to_be_bytes());

    state = *inner_init;
    compress(&mut state, &block);

    write_u32x8_be(&mut outer_block[..SHA256_OUTPUT_SIZE], &state);
    state = *outer_init;
    compress(&mut state, &outer_block);

    let mut block_out = [0u8; SHA256_OUTPUT_SIZE];
    write_u32x8_be(&mut block_out, &state);
    tail.copy_from_slice(&block_out[..tail.len()]);
    ct::zeroize_no_fence(&mut block_out);
  }

  ct::zeroize_no_fence(&mut block);
  ct::zeroize_no_fence(&mut outer_block);
  zeroize_u32x8_no_fence(&mut state);
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}

// ─── PBKDF2-HMAC-SHA512 ────────────────────────────────────────────────────

define_pbkdf2_sha2! {
  /// PBKDF2-HMAC-SHA512 key derivation (RFC 2898).
  ///
  /// Pre-computes the HMAC-SHA512 prefix states from the password so that
  /// subsequent `derive` and `verify` calls run the iteration loop directly
  /// against the SHA-512 compress function with no per-iteration overhead.
  ///
  /// # Examples
  ///
  /// ```rust
  /// use rscrypto::Pbkdf2Sha512;
  ///
  /// let dk = Pbkdf2Sha512::derive_key_array::<64>(b"password", b"salt-16-bytes!!!", 220_000)?;
  /// assert!(Pbkdf2Sha512::verify_password(b"password", b"salt-16-bytes!!!", 220_000, &dk).is_ok());
  /// # Ok::<(), rscrypto::auth::Pbkdf2Error>(())
  /// ```
  Pbkdf2Sha512 {
    output_size_const: SHA512_OUTPUT_SIZE,
    block_size_const: SHA512_BLOCK_SIZE,
    compress_ty: Sha512CompressBlocksFn,
    digest_ty: Sha512,
    h0: SHA512_H0,
    dispatch: sha512_dispatch,
    new_fast_path: pbkdf2_sha512_new_fast_path,
    f_fn: pbkdf2_sha512_f,
    iter1_fn: pbkdf2_sha512_iter1,
    derive_key_fast_path: pbkdf2_sha512_derive_key_fast_path,
    test_oneshot: sha512_oneshot_with_compress,
    word_ty: u64,
    recommended_iterations: 220_000,
  }
}

#[inline]
fn pbkdf2_sha512_new_fast_path(_password: &[u8]) -> Option<Pbkdf2Sha512> {
  None
}

#[inline]
fn pbkdf2_sha512_derive_key_fast_path(
  _password: &[u8],
  _salt: &[u8],
  _iterations: u32,
  _okm: &mut [u8],
) -> Result<bool, Pbkdf2Error> {
  Ok(false)
}

/// Test-only: one-shot SHA-512 digest using a specific compress function.
#[cfg(any(test, feature = "diag"))]
#[allow(dead_code)]
#[allow(clippy::indexing_slicing)]
fn sha512_oneshot_with_compress(data: &[u8], compress: Sha512CompressBlocksFn) -> [u8; SHA512_OUTPUT_SIZE] {
  let mut state = SHA512_H0;
  let mut pos = 0usize;
  while pos.strict_add(SHA512_BLOCK_SIZE) <= data.len() {
    compress(&mut state, &data[pos..pos.strict_add(SHA512_BLOCK_SIZE)]);
    pos = pos.strict_add(SHA512_BLOCK_SIZE);
  }
  let mut block = [0u8; SHA512_BLOCK_SIZE];
  let tail = data.len().strict_sub(pos);
  block[..tail].copy_from_slice(&data[pos..]);
  block[tail] = 0x80;
  if tail >= 112 {
    compress(&mut state, &block);
    block = [0u8; SHA512_BLOCK_SIZE];
  }
  block[112..128].copy_from_slice(&(data.len() as u128).strict_mul(8).to_be_bytes());
  compress(&mut state, &block);
  let mut out = [0u8; SHA512_OUTPUT_SIZE];
  for (chunk, &word) in out.chunks_exact_mut(8).zip(state.iter()) {
    chunk.copy_from_slice(&word.to_be_bytes());
  }
  out
}

/// Compute one PBKDF2-SHA512 block: `F(Password, Salt, c, i)`.
#[allow(clippy::indexing_slicing)]
#[inline(always)]
fn pbkdf2_sha512_f(
  compress: Sha512CompressBlocksFn,
  inner_init: &[u64; 8],
  outer_init: &[u64; 8],
  salt: &[u8],
  iterations: u32,
  block_index: u32,
  output: &mut [u8; SHA512_OUTPUT_SIZE],
) {
  let mut state: [u64; 8];
  let mut u_words: [u64; 8];
  let mut result_words: [u64; 8];

  // ── U1 = HMAC(Password, Salt || INT_32_BE(block_index)) ──────────────
  state = *inner_init;
  let msg_len = salt.len().strict_add(4);
  let total_inner = (SHA512_BLOCK_SIZE as u128).strict_add(msg_len as u128);

  let mut block = [0u8; SHA512_BLOCK_SIZE];
  if salt.len() <= SHA512_INLINE_SALT_MAX {
    block[..salt.len()].copy_from_slice(salt);
    let pos = salt.len().strict_add(4);
    block[salt.len()..pos].copy_from_slice(&block_index.to_be_bytes());
    block[pos] = 0x80;
    block[112..SHA512_BLOCK_SIZE].copy_from_slice(&total_inner.strict_mul(8).to_be_bytes());
    compress(&mut state, &block);
  } else {
    let mut pos = 0usize;

    // Feed salt
    let mut salt_off = 0usize;
    while salt_off < salt.len() {
      let space = SHA512_BLOCK_SIZE.strict_sub(pos);
      let remaining = salt.len().strict_sub(salt_off);
      let take = if space < remaining { space } else { remaining };
      block[pos..pos.strict_add(take)].copy_from_slice(&salt[salt_off..salt_off.strict_add(take)]);
      pos = pos.strict_add(take);
      salt_off = salt_off.strict_add(take);
      if pos == SHA512_BLOCK_SIZE {
        compress(&mut state, &block);
        block = [0u8; SHA512_BLOCK_SIZE];
        pos = 0;
      }
    }

    // Feed block_index (4 bytes big-endian)
    for &b in &block_index.to_be_bytes() {
      block[pos] = b;
      pos = pos.strict_add(1);
      if pos == SHA512_BLOCK_SIZE {
        compress(&mut state, &block);
        block = [0u8; SHA512_BLOCK_SIZE];
        pos = 0;
      }
    }

    // SHA-512 padding (16-byte length field)
    block[pos] = 0x80;
    if pos.strict_add(1) > 112 {
      compress(&mut state, &block);
      block = [0u8; SHA512_BLOCK_SIZE];
    }
    block[112..SHA512_BLOCK_SIZE].copy_from_slice(&total_inner.strict_mul(8).to_be_bytes());
    compress(&mut state, &block);
  }

  // Outer hash of U1: single block
  let mut outer_block = [0u8; SHA512_BLOCK_SIZE];
  write_u64x8_be(&mut outer_block[..SHA512_OUTPUT_SIZE], &state);
  outer_block[SHA512_OUTPUT_SIZE] = 0x80;
  // total outer bytes = 128 (opad, pre-compressed) + 64 (inner hash) = 192 → 1536 bits
  outer_block[112..SHA512_BLOCK_SIZE].copy_from_slice(&1536u128.to_be_bytes());

  state = *outer_init;
  compress(&mut state, &outer_block);
  u_words = state;
  result_words = u_words;

  // ── Iterations 2..=c (fixed-size HMAC, 2 compress calls each) ────────
  let mut inner_block = [0u8; SHA512_BLOCK_SIZE];
  inner_block[SHA512_OUTPUT_SIZE] = 0x80;
  inner_block[112..SHA512_BLOCK_SIZE].copy_from_slice(&1536u128.to_be_bytes());

  for _ in 1..iterations {
    write_u64x8_be(&mut inner_block[..SHA512_OUTPUT_SIZE], &u_words);
    state = *inner_init;
    compress(&mut state, &inner_block);

    write_u64x8_be(&mut outer_block[..SHA512_OUTPUT_SIZE], &state);

    state = *outer_init;
    compress(&mut state, &outer_block);

    u_words = state;

    for (dst, &word) in result_words.iter_mut().zip(u_words.iter()) {
      *dst ^= word;
    }
  }

  write_u64x8_be(output, &result_words);

  ct::zeroize_no_fence(&mut inner_block);
  ct::zeroize_no_fence(&mut outer_block);
  ct::zeroize_no_fence(&mut block);
  zeroize_u64x8_no_fence(&mut state);
  zeroize_u64x8_no_fence(&mut u_words);
  zeroize_u64x8_no_fence(&mut result_words);
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}

#[allow(clippy::indexing_slicing)]
#[inline(always)]
fn pbkdf2_sha512_iter1(
  compress: Sha512CompressBlocksFn,
  inner_init: &[u64; 8],
  outer_init: &[u64; 8],
  salt: &[u8],
  okm: &mut [u8],
) {
  if salt.len() > SHA512_INLINE_SALT_MAX {
    let mut block_index = 1u32;
    let mut chunks = okm.chunks_exact_mut(SHA512_OUTPUT_SIZE);
    for chunk in chunks.by_ref() {
      // SAFETY: chunks_exact_mut yields slices whose length is exactly SHA512_OUTPUT_SIZE.
      let full_chunk = unsafe { &mut *(chunk.as_mut_ptr().cast::<[u8; SHA512_OUTPUT_SIZE]>()) };
      pbkdf2_sha512_f(compress, inner_init, outer_init, salt, 1, block_index, full_chunk);
      block_index = block_index.strict_add(1);
    }
    let tail = chunks.into_remainder();
    if !tail.is_empty() {
      let mut block_out = [0u8; SHA512_OUTPUT_SIZE];
      pbkdf2_sha512_f(compress, inner_init, outer_init, salt, 1, block_index, &mut block_out);
      tail.copy_from_slice(&block_out[..tail.len()]);
      ct::zeroize(&mut block_out);
    }
    return;
  }

  let msg_len = salt.len().strict_add(4);
  let total_inner = (SHA512_BLOCK_SIZE as u128).strict_add(msg_len as u128);
  let index_pos = salt.len();
  let pad_pos = index_pos.strict_add(4);

  let mut block = [0u8; SHA512_BLOCK_SIZE];
  block[..salt.len()].copy_from_slice(salt);
  block[pad_pos] = 0x80;
  block[112..SHA512_BLOCK_SIZE].copy_from_slice(&total_inner.strict_mul(8).to_be_bytes());

  let mut outer_block = [0u8; SHA512_BLOCK_SIZE];
  outer_block[SHA512_OUTPUT_SIZE] = 0x80;
  outer_block[112..SHA512_BLOCK_SIZE].copy_from_slice(&1536u128.to_be_bytes());

  let mut state = [0u64; 8];
  let mut block_index = 1u32;

  let mut chunks = okm.chunks_exact_mut(SHA512_OUTPUT_SIZE);
  for chunk in chunks.by_ref() {
    block[index_pos..pad_pos].copy_from_slice(&block_index.to_be_bytes());

    state = *inner_init;
    compress(&mut state, &block);

    write_u64x8_be(&mut outer_block[..SHA512_OUTPUT_SIZE], &state);
    state = *outer_init;
    compress(&mut state, &outer_block);

    write_u64x8_be(chunk, &state);
    block_index = block_index.strict_add(1);
  }

  let tail = chunks.into_remainder();
  if !tail.is_empty() {
    block[index_pos..pad_pos].copy_from_slice(&block_index.to_be_bytes());

    state = *inner_init;
    compress(&mut state, &block);

    write_u64x8_be(&mut outer_block[..SHA512_OUTPUT_SIZE], &state);
    state = *outer_init;
    compress(&mut state, &outer_block);

    let mut block_out = [0u8; SHA512_OUTPUT_SIZE];
    write_u64x8_be(&mut block_out, &state);
    tail.copy_from_slice(&block_out[..tail.len()]);
    ct::zeroize_no_fence(&mut block_out);
  }

  ct::zeroize_no_fence(&mut block);
  ct::zeroize_no_fence(&mut outer_block);
  zeroize_u64x8_no_fence(&mut state);
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
  use alloc::vec;
  use core::sync::atomic::{AtomicUsize, Ordering};

  use super::*;

  // ── RFC 7914 §11 PBKDF2-HMAC-SHA256 test vectors ──────────────────────

  #[test]
  fn rfc7914_sha256_vector_1() {
    let mut dk = [0u8; 64];
    Pbkdf2Sha256::derive_key_primitive(b"passwd", b"salt", 1, &mut dk).unwrap();
    assert_eq!(
      dk,
      [
        0x55, 0xac, 0x04, 0x6e, 0x56, 0xe3, 0x08, 0x9f, 0xec, 0x16, 0x91, 0xc2, 0x25, 0x44, 0xb6, 0x05, 0xf9, 0x41,
        0x85, 0x21, 0x6d, 0xde, 0x04, 0x65, 0xe6, 0x8b, 0x9d, 0x57, 0xc2, 0x0d, 0xac, 0xbc, 0x49, 0xca, 0x9c, 0xcc,
        0xf1, 0x79, 0xb6, 0x45, 0x99, 0x16, 0x64, 0xb3, 0x9d, 0x77, 0xef, 0x31, 0x7c, 0x71, 0xb8, 0x45, 0xb1, 0xe3,
        0x0b, 0xd5, 0x09, 0x11, 0x20, 0x41, 0xd3, 0xa1, 0x97, 0x83,
      ]
    );
  }

  #[test]
  fn sha256_hmac_iter1_small_matches_state_path() {
    let password = [0x55u8; 32];
    let salt = [0x33u8; 16];
    let state = Pbkdf2Sha256::new(&password);

    for len in [1usize, 16, 31, 32, 33, 63, 64] {
      let mut expected = vec![0u8; len];
      let mut actual = vec![0u8; len];
      state.derive(&salt, 1, &mut expected).unwrap();
      pbkdf2_sha256_hmac_iter1_small(&password, &salt, &mut actual);
      assert_eq!(actual, expected, "len={len}");
    }
  }

  #[test]
  fn sha256_hmac_iter1_fast_path_selects_two_output_blocks_when_preferred() {
    let password = [0x55u8; 32];
    let salt = [0x33u8; 16];
    let mut expected = [0u8; 64];
    let mut actual = [0u8; 64];

    oracle_sha256(&password, &salt, 1, &mut expected);
    assert!(pbkdf2_sha256_derive_key_fast_path_with_preference(&password, &salt, 1, &mut actual, true).unwrap());
    assert_eq!(actual, expected);

    let mut rejected = [0u8; 65];
    assert!(!pbkdf2_sha256_derive_key_fast_path_with_preference(&password, &salt, 1, &mut rejected, true).unwrap());
    assert!(!pbkdf2_sha256_derive_key_fast_path_with_preference(&password, &salt, 2, &mut actual, true).unwrap());
    assert!(!pbkdf2_sha256_derive_key_fast_path_with_preference(&password, &salt, 1, &mut actual, false).unwrap());
  }

  #[cfg(not(miri))]
  #[test]
  fn rfc7914_sha256_vector_2() {
    let mut dk = [0u8; 64];
    Pbkdf2Sha256::derive_key_primitive(b"Password", b"NaCl", 80000, &mut dk).unwrap();
    assert_eq!(
      dk,
      [
        0x4d, 0xdc, 0xd8, 0xf6, 0x0b, 0x98, 0xbe, 0x21, 0x83, 0x0c, 0xee, 0x5e, 0xf2, 0x27, 0x01, 0xf9, 0x64, 0x1a,
        0x44, 0x18, 0xd0, 0x4c, 0x04, 0x14, 0xae, 0xff, 0x08, 0x87, 0x6b, 0x34, 0xab, 0x56, 0xa1, 0xd4, 0x25, 0xa1,
        0x22, 0x58, 0x33, 0x54, 0x9a, 0xdb, 0x84, 0x1b, 0x51, 0xc9, 0xb3, 0x17, 0x6a, 0x27, 0x2b, 0xde, 0xbb, 0xa1,
        0xd0, 0x78, 0x47, 0x8f, 0x62, 0xb3, 0x97, 0xf3, 0x3c, 0x8d,
      ]
    );
  }

  // ── Oracle: RustCrypto PBKDF2 primitive ────────────────────────────────

  fn oracle_sha256(password: &[u8], salt: &[u8], iterations: u32, out: &mut [u8]) {
    pbkdf2::pbkdf2_hmac::<sha2::Sha256>(password, salt, iterations, out);
  }

  fn oracle_sha512(password: &[u8], salt: &[u8], iterations: u32, out: &mut [u8]) {
    pbkdf2::pbkdf2_hmac::<sha2::Sha512>(password, salt, iterations, out);
  }

  #[test]
  fn sha256_matches_oracle() {
    #[cfg(not(miri))]
    let cases: &[(&[u8], &[u8], u32, usize)] = &[
      (b"password", b"salt", 1, 32),
      (b"password", b"salt", 2, 32),
      (b"password", b"salt", 4096, 32),
      (b"password", b"salt", 1, 64),
      (b"password", b"salt", 100, 64),
      (b"", b"salt", 1, 32),
      (b"password", b"", 1, 32),
      (b"", b"", 1, 32),
      (b"p", b"s", 1, 1),
      (b"password", b"salt", 1, 20),
      (b"password", b"salt", 1, 48),
      (b"password", b"salt", 1, 96),
      (b"password", b"salt", 1, 128),
      // Long salt (multi-block inner hash for U1)
      (&[0xAA; 100], b"salt", 1, 32),
      // Long salt spanning SHA-256 blocks
      (b"password", &[0xBB; 200], 1, 32),
      // Key longer than block size (hashed first)
      (&[0xCC; 128], b"salt", 1, 32),
    ];
    #[cfg(miri)]
    let cases: &[(&[u8], &[u8], u32, usize)] = &[
      (b"password", b"salt", 1, 32),
      (b"password", b"salt", 2, 32),
      (b"password", b"salt", 16, 64),
      (b"", b"", 1, 32),
      (b"p", b"s", 1, 1),
      (b"password", b"salt", 1, 96),
      (&[0xAA; 100], b"salt", 1, 32),
      (b"password", &[0xBB; 200], 1, 32),
      (&[0xCC; 128], b"salt", 1, 32),
    ];

    for &(password, salt, iterations, dk_len) in cases {
      let mut expected = vec![0u8; dk_len];
      oracle_sha256(password, salt, iterations, &mut expected);

      let mut actual = vec![0u8; dk_len];
      Pbkdf2Sha256::derive_key_primitive(password, salt, iterations, &mut actual).unwrap();

      assert_eq!(
        actual,
        expected,
        "SHA-256 mismatch: pw_len={} salt_len={} c={} dk_len={}",
        password.len(),
        salt.len(),
        iterations,
        dk_len,
      );
    }
  }

  #[test]
  fn sha512_matches_oracle() {
    #[cfg(not(miri))]
    let cases: &[(&[u8], &[u8], u32, usize)] = &[
      (b"password", b"salt", 1, 64),
      (b"password", b"salt", 2, 64),
      (b"password", b"salt", 4096, 64),
      (b"password", b"salt", 1, 128),
      (b"password", b"salt", 100, 128),
      (b"", b"salt", 1, 64),
      (b"password", b"", 1, 64),
      (b"", b"", 1, 64),
      (b"p", b"s", 1, 1),
      (b"password", b"salt", 1, 20),
      (b"password", b"salt", 1, 48),
      (b"password", b"salt", 1, 96),
      (b"password", b"salt", 1, 192),
      // Long salt
      (b"password", &[0xBB; 200], 1, 64),
      // Key longer than block size
      (&[0xCC; 200], b"salt", 1, 64),
    ];
    #[cfg(miri)]
    let cases: &[(&[u8], &[u8], u32, usize)] = &[
      (b"password", b"salt", 1, 64),
      (b"password", b"salt", 2, 64),
      (b"password", b"salt", 16, 128),
      (b"", b"", 1, 64),
      (b"p", b"s", 1, 1),
      (b"password", b"salt", 1, 192),
      (b"password", &[0xBB; 200], 1, 64),
      (&[0xCC; 200], b"salt", 1, 64),
    ];

    for &(password, salt, iterations, dk_len) in cases {
      let mut expected = vec![0u8; dk_len];
      oracle_sha512(password, salt, iterations, &mut expected);

      let mut actual = vec![0u8; dk_len];
      Pbkdf2Sha512::derive_key_primitive(password, salt, iterations, &mut actual).unwrap();

      assert_eq!(
        actual,
        expected,
        "SHA-512 mismatch: pw_len={} salt_len={} c={} dk_len={}",
        password.len(),
        salt.len(),
        iterations,
        dk_len,
      );
    }
  }

  // ── Verify ────────────────────────────────────────────────────────────

  #[test]
  fn sha256_verify_correct_password() {
    let dk = Pbkdf2Sha256::derive_key_array_primitive::<32>(b"password", b"salt", 100).unwrap();
    assert!(Pbkdf2Sha256::verify_password_primitive(b"password", b"salt", 100, &dk).is_ok());
  }

  #[test]
  fn sha256_verify_wrong_password() {
    let dk = Pbkdf2Sha256::derive_key_array_primitive::<32>(b"password", b"salt", 100).unwrap();
    assert!(Pbkdf2Sha256::verify_password_primitive(b"wrong", b"salt", 100, &dk).is_err());
  }

  #[test]
  fn sha256_verify_wrong_salt() {
    let dk = Pbkdf2Sha256::derive_key_array_primitive::<32>(b"password", b"salt", 100).unwrap();
    assert!(Pbkdf2Sha256::verify_password_primitive(b"password", b"wrong", 100, &dk).is_err());
  }

  #[test]
  fn sha256_verify_wrong_iterations() {
    let dk = Pbkdf2Sha256::derive_key_array_primitive::<32>(b"password", b"salt", 100).unwrap();
    assert!(Pbkdf2Sha256::verify_password_primitive(b"password", b"salt", 101, &dk).is_err());
  }

  #[test]
  fn sha512_verify_correct_password() {
    let dk = Pbkdf2Sha512::derive_key_array_primitive::<64>(b"password", b"salt", 100).unwrap();
    assert!(Pbkdf2Sha512::verify_password_primitive(b"password", b"salt", 100, &dk).is_ok());
  }

  #[test]
  fn sha512_verify_wrong_password() {
    let dk = Pbkdf2Sha512::derive_key_array_primitive::<64>(b"password", b"salt", 100).unwrap();
    assert!(Pbkdf2Sha512::verify_password_primitive(b"wrong", b"salt", 100, &dk).is_err());
  }

  // ── Error paths ───────────────────────────────────────────────────────

  #[test]
  fn sha256_zero_iterations_error() {
    let mut dk = [0u8; 32];
    assert_eq!(
      Pbkdf2Sha256::derive_key_primitive(b"pw", b"salt", 0, &mut dk),
      Err(Pbkdf2Error::InvalidIterations)
    );
  }

  #[test]
  fn sha256_default_password_policy_rejects_weak_params() {
    let strong_salt = [0xA5; Pbkdf2Sha256::MIN_SALT_LEN];
    let mut empty = [];

    assert_eq!(
      Pbkdf2Sha256::derive_key(
        b"pw",
        &strong_salt,
        Pbkdf2Sha256::MIN_RECOMMENDED_ITERATIONS.strict_sub(1),
        &mut empty,
      ),
      Err(Pbkdf2Error::WeakIterations)
    );
    assert_eq!(
      Pbkdf2Sha256::derive_key(
        b"pw",
        &strong_salt[..Pbkdf2Sha256::MIN_SALT_LEN.strict_sub(1)],
        Pbkdf2Sha256::MIN_RECOMMENDED_ITERATIONS,
        &mut empty,
      ),
      Err(Pbkdf2Error::SaltTooShort)
    );
    assert!(
      Pbkdf2Sha256::derive_key(
        b"pw",
        &strong_salt,
        Pbkdf2Sha256::MIN_RECOMMENDED_ITERATIONS,
        &mut empty
      )
      .is_ok()
    );
    assert!(
      Pbkdf2Sha256::verify_password(
        b"pw",
        &strong_salt,
        Pbkdf2Sha256::MIN_RECOMMENDED_ITERATIONS.strict_sub(1),
        &[0u8; 32],
      )
      .is_err()
    );
  }

  #[test]
  fn pbkdf2_params_can_use_explicit_policy_for_migrations() {
    let policy = Pbkdf2VerifyPolicy::new(1, 0);
    let params = Pbkdf2Sha256::params_with_policy(b"", 1, &policy).unwrap();
    assert_eq!(params.salt(), b"");
    assert_eq!(params.iterations(), 1);

    let mut empty = [];
    assert!(Pbkdf2Sha256::derive_key_with_params(b"pw", params, &mut empty).is_ok());
  }

  #[test]
  fn sha512_zero_iterations_error() {
    let mut dk = [0u8; 64];
    assert_eq!(
      Pbkdf2Sha512::derive_key_primitive(b"pw", b"salt", 0, &mut dk),
      Err(Pbkdf2Error::InvalidIterations)
    );
  }

  #[test]
  fn sha256_empty_output_ok() {
    assert!(Pbkdf2Sha256::derive_key_primitive(b"pw", b"salt", 1, &mut []).is_ok());
  }

  #[test]
  fn sha512_empty_output_ok() {
    assert!(Pbkdf2Sha512::derive_key_primitive(b"pw", b"salt", 1, &mut []).is_ok());
  }

  #[test]
  fn sha256_verify_zero_iterations() {
    assert!(Pbkdf2Sha256::verify_password_primitive(b"pw", b"salt", 0, &[0u8; 32]).is_err());
  }

  #[test]
  fn sha256_verify_empty_expected() {
    assert!(Pbkdf2Sha256::verify_password_primitive(b"pw", b"salt", 1, &[]).is_err());
  }

  #[test]
  fn sha256_verify_password_covers_output_lengths_and_mismatch_positions() {
    let password = [0xA5; 97];
    let salt = [0x5A; 200];
    #[cfg(not(miri))]
    let output_lengths = 1..=96;
    #[cfg(miri)]
    let output_lengths = [1usize, 2, 31, 32, 33, 63, 64, 65, 95, 96].into_iter();

    for out_len in output_lengths {
      let mut expected = vec![0u8; out_len];
      Pbkdf2Sha256::derive_key_primitive(&password, &salt, 2, &mut expected).unwrap();
      assert!(Pbkdf2Sha256::verify_password_primitive(&password, &salt, 2, &expected).is_ok());

      let mut wrong_first = expected.clone();
      wrong_first[0] ^= 1;
      assert!(Pbkdf2Sha256::verify_password_primitive(&password, &salt, 2, &wrong_first).is_err());

      let mut wrong_last = expected.clone();
      let last = wrong_last.len().strict_sub(1);
      wrong_last[last] ^= 1;
      assert!(Pbkdf2Sha256::verify_password_primitive(&password, &salt, 2, &wrong_last).is_err());
    }
  }

  #[test]
  fn sha512_verify_password_covers_output_lengths_and_mismatch_positions() {
    let password = [0x3C; 193];
    let salt = [0xC3; 260];
    #[cfg(not(miri))]
    let output_lengths = 1..=192;
    #[cfg(miri)]
    let output_lengths = [1usize, 2, 63, 64, 65, 127, 128, 129, 191, 192].into_iter();

    for out_len in output_lengths {
      let mut expected = vec![0u8; out_len];
      Pbkdf2Sha512::derive_key_primitive(&password, &salt, 2, &mut expected).unwrap();
      assert!(Pbkdf2Sha512::verify_password_primitive(&password, &salt, 2, &expected).is_ok());

      let mut wrong_first = expected.clone();
      wrong_first[0] ^= 1;
      assert!(Pbkdf2Sha512::verify_password_primitive(&password, &salt, 2, &wrong_first).is_err());

      let mut wrong_last = expected.clone();
      let last = wrong_last.len().strict_sub(1);
      wrong_last[last] ^= 1;
      assert!(Pbkdf2Sha512::verify_password_primitive(&password, &salt, 2, &wrong_last).is_err());
    }
  }

  // ── Streaming state reuse ─────────────────────────────────────────────

  #[test]
  fn sha256_state_reuse_matches_oneshot() {
    let state = Pbkdf2Sha256::new(b"password");
    let dk1 = state.derive_array::<32>(b"salt1", 100).unwrap();
    let dk2 = state.derive_array::<32>(b"salt2", 100).unwrap();

    let oneshot1 = Pbkdf2Sha256::derive_key_array_primitive::<32>(b"password", b"salt1", 100).unwrap();
    let oneshot2 = Pbkdf2Sha256::derive_key_array_primitive::<32>(b"password", b"salt2", 100).unwrap();

    assert_eq!(dk1, oneshot1);
    assert_eq!(dk2, oneshot2);
    assert_ne!(dk1, dk2);
  }

  // ── c=1 edge case ─────────────────────────────────────────────────────

  #[test]
  fn sha256_single_iteration() {
    let mut expected = [0u8; 32];
    oracle_sha256(b"pw", b"salt", 1, &mut expected);
    let actual = Pbkdf2Sha256::derive_key_array_primitive::<32>(b"pw", b"salt", 1).unwrap();
    assert_eq!(actual, expected);
  }

  #[test]
  fn sha512_single_iteration() {
    let mut expected = [0u8; 64];
    oracle_sha512(b"pw", b"salt", 1, &mut expected);
    let actual = Pbkdf2Sha512::derive_key_array_primitive::<64>(b"pw", b"salt", 1).unwrap();
    assert_eq!(actual, expected);
  }

  // ── Error traits ──────────────────────────────────────────────────────

  #[test]
  fn error_is_copy() {
    let e = Pbkdf2Error::InvalidIterations;
    let e2 = e;
    assert_eq!(e, e2);
  }

  #[test]
  fn error_display() {
    fn assert_display<T: core::fmt::Display>() {}
    assert_display::<Pbkdf2Error>();
  }

  #[test]
  fn error_debug() {
    fn assert_debug<T: core::fmt::Debug>() {}
    assert_debug::<Pbkdf2Error>();
  }

  #[test]
  fn error_implements_error_trait() {
    fn assert_error<T: core::error::Error>() {}
    assert_error::<Pbkdf2Error>();
  }

  // ── Forced-kernel oracle tests ──────────────────────────────────────

  use crate::hashes::crypto::{
    sha256::kernels::{
      Sha256KernelId, compress_blocks_fn as sha256_compress_blocks_fn, required_caps as sha256_required_caps,
    },
    sha512::kernels::{
      ALL as SHA512_KERNELS, Sha512KernelId, compress_blocks_fn as sha512_compress_blocks_fn,
      required_caps as sha512_required_caps,
    },
  };

  /// SHA-256 kernel list (sha256 module doesn't define ALL).
  const SHA256_KERNELS: &[Sha256KernelId] = &[
    Sha256KernelId::Portable,
    #[cfg(target_arch = "x86_64")]
    Sha256KernelId::X86Sha,
    #[cfg(target_arch = "aarch64")]
    Sha256KernelId::Aarch64Sha2,
    #[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
    Sha256KernelId::RiscvZknh,
    #[cfg(target_arch = "wasm32")]
    Sha256KernelId::WasmSimd128,
    #[cfg(target_arch = "s390x")]
    Sha256KernelId::S390xKimd,
  ];

  static SHA256_VERIFY_BLOCKS: AtomicUsize = AtomicUsize::new(0);
  static SHA512_VERIFY_BLOCKS: AtomicUsize = AtomicUsize::new(0);

  fn counting_sha256_compress(state: &mut [u32; 8], blocks: &[u8]) {
    SHA256_VERIFY_BLOCKS.fetch_add(blocks.len() / SHA256_BLOCK_SIZE, Ordering::Relaxed);
    sha256_compress_blocks_fn(Sha256KernelId::Portable)(state, blocks);
  }

  fn counting_sha512_compress(state: &mut [u64; 8], blocks: &[u8]) {
    SHA512_VERIFY_BLOCKS.fetch_add(blocks.len() / SHA512_BLOCK_SIZE, Ordering::Relaxed);
    sha512_compress_blocks_fn(Sha512KernelId::Portable)(state, blocks);
  }

  fn counted_sha256_verify(
    state: &Pbkdf2Sha256,
    salt: &[u8],
    iterations: u32,
    expected: &[u8],
  ) -> (Result<(), VerificationError>, usize) {
    SHA256_VERIFY_BLOCKS.store(0, Ordering::Relaxed);
    let result = state.verify_primitive(salt, iterations, expected);
    let blocks = SHA256_VERIFY_BLOCKS.swap(0, Ordering::Relaxed);
    (result, blocks)
  }

  fn counted_sha512_verify(
    state: &Pbkdf2Sha512,
    salt: &[u8],
    iterations: u32,
    expected: &[u8],
  ) -> (Result<(), VerificationError>, usize) {
    SHA512_VERIFY_BLOCKS.store(0, Ordering::Relaxed);
    let result = state.verify_primitive(salt, iterations, expected);
    let blocks = SHA512_VERIFY_BLOCKS.swap(0, Ordering::Relaxed);
    (result, blocks)
  }

  fn assert_pbkdf2_sha256_kernel(id: Sha256KernelId) {
    let compress = sha256_compress_blocks_fn(id);
    let cases: &[(&[u8], &[u8], u32, usize)] = &[
      (b"password", b"salt", 1, 32),
      (b"password", b"salt", 4, 32),
      (b"password", b"salt", 100, 32),
      (b"password", b"salt", 1, 64),      // multi-block
      (b"", b"salt", 1, 32),              // empty password
      (b"password", b"", 1, 32),          // empty salt
      (b"p", b"s", 1, 1),                 // minimal output
      (b"password", &[0xBB; 200], 1, 32), // long salt
      (&[0xCC; 128], b"salt", 1, 32),     // password > block_size (triggers key hashing)
    ];

    for &(password, salt, iterations, dk_len) in cases {
      let mut expected = vec![0u8; dk_len];
      oracle_sha256(password, salt, iterations, &mut expected);

      let state = Pbkdf2Sha256::new_with_compress_for_test(password, compress);
      let mut actual = vec![0u8; dk_len];
      state.derive(salt, iterations, &mut actual).unwrap();

      assert_eq!(
        actual,
        expected,
        "pbkdf2-sha256 forced mismatch kernel={} pw_len={} salt_len={} c={} dk_len={}",
        id.as_str(),
        password.len(),
        salt.len(),
        iterations,
        dk_len,
      );
    }
  }

  fn assert_pbkdf2_sha512_kernel(id: Sha512KernelId) {
    let compress = sha512_compress_blocks_fn(id);
    let cases: &[(&[u8], &[u8], u32, usize)] = &[
      (b"password", b"salt", 1, 64),
      (b"password", b"salt", 4, 64),
      (b"password", b"salt", 100, 64),
      (b"password", b"salt", 1, 128),     // multi-block
      (b"", b"salt", 1, 64),              // empty password
      (b"password", b"", 1, 64),          // empty salt
      (b"p", b"s", 1, 1),                 // minimal output
      (b"password", &[0xBB; 200], 1, 64), // long salt
      (&[0xCC; 200], b"salt", 1, 64),     // password > block_size
    ];

    for &(password, salt, iterations, dk_len) in cases {
      let mut expected = vec![0u8; dk_len];
      oracle_sha512(password, salt, iterations, &mut expected);

      let state = Pbkdf2Sha512::new_with_compress_for_test(password, compress);
      let mut actual = vec![0u8; dk_len];
      state.derive(salt, iterations, &mut actual).unwrap();

      assert_eq!(
        actual,
        expected,
        "pbkdf2-sha512 forced mismatch kernel={} pw_len={} salt_len={} c={} dk_len={}",
        id.as_str(),
        password.len(),
        salt.len(),
        iterations,
        dk_len,
      );
    }
  }

  #[test]
  fn pbkdf2_sha256_forced_kernels_match_oracle() {
    let caps = crate::platform::caps();
    for &id in SHA256_KERNELS {
      if caps.has(sha256_required_caps(id)) {
        assert_pbkdf2_sha256_kernel(id);
      }
    }
  }

  #[test]
  fn pbkdf2_sha512_forced_kernels_match_oracle() {
    let caps = crate::platform::caps();
    for &id in SHA512_KERNELS {
      if caps.has(sha512_required_caps(id)) {
        assert_pbkdf2_sha512_kernel(id);
      }
    }
  }

  #[test]
  fn sha256_verify_keeps_same_compress_work_for_match_and_mismatch_positions() {
    let password = [0x11; 89];
    let salt = [0x22; 200];
    let state = Pbkdf2Sha256::new_with_compress_for_test(&password, counting_sha256_compress);
    #[cfg(not(miri))]
    let output_lengths = 1..=96;
    #[cfg(miri)]
    let output_lengths = [1usize, 2, 31, 32, 33, 63, 64, 65, 95, 96].into_iter();

    for out_len in output_lengths {
      let mut expected = vec![0u8; out_len];
      state.derive(&salt, 3, &mut expected).unwrap();

      let (ok, ok_blocks) = counted_sha256_verify(&state, &salt, 3, &expected);
      assert!(ok.is_ok(), "sha256 verify must accept correct output_len={out_len}");

      let mut wrong_first = expected.clone();
      wrong_first[0] ^= 1;
      let (wrong_first_result, wrong_first_blocks) = counted_sha256_verify(&state, &salt, 3, &wrong_first);
      assert!(
        wrong_first_result.is_err(),
        "sha256 verify must reject first-byte mismatch output_len={out_len}"
      );

      let mut wrong_last = expected.clone();
      let last = wrong_last.len().strict_sub(1);
      wrong_last[last] ^= 1;
      let (wrong_last_result, wrong_last_blocks) = counted_sha256_verify(&state, &salt, 3, &wrong_last);
      assert!(
        wrong_last_result.is_err(),
        "sha256 verify must reject last-byte mismatch output_len={out_len}"
      );

      assert!(ok_blocks > 0, "sha256 verify must do real work output_len={out_len}");
      assert_eq!(
        ok_blocks, wrong_first_blocks,
        "sha256 verify must not short-circuit on first-byte mismatch output_len={out_len}"
      );
      assert_eq!(
        ok_blocks, wrong_last_blocks,
        "sha256 verify must not short-circuit on last-byte mismatch output_len={out_len}"
      );
    }
  }

  #[test]
  fn sha512_verify_keeps_same_compress_work_for_match_and_mismatch_positions() {
    let password = [0x44; 161];
    let salt = [0x55; 260];
    let state = Pbkdf2Sha512::new_with_compress_for_test(&password, counting_sha512_compress);
    #[cfg(not(miri))]
    let output_lengths = 1..=192;
    #[cfg(miri)]
    let output_lengths = [1usize, 2, 63, 64, 65, 127, 128, 129, 191, 192].into_iter();

    for out_len in output_lengths {
      let mut expected = vec![0u8; out_len];
      state.derive(&salt, 3, &mut expected).unwrap();

      let (ok, ok_blocks) = counted_sha512_verify(&state, &salt, 3, &expected);
      assert!(ok.is_ok(), "sha512 verify must accept correct output_len={out_len}");

      let mut wrong_first = expected.clone();
      wrong_first[0] ^= 1;
      let (wrong_first_result, wrong_first_blocks) = counted_sha512_verify(&state, &salt, 3, &wrong_first);
      assert!(
        wrong_first_result.is_err(),
        "sha512 verify must reject first-byte mismatch output_len={out_len}"
      );

      let mut wrong_last = expected.clone();
      let last = wrong_last.len().strict_sub(1);
      wrong_last[last] ^= 1;
      let (wrong_last_result, wrong_last_blocks) = counted_sha512_verify(&state, &salt, 3, &wrong_last);
      assert!(
        wrong_last_result.is_err(),
        "sha512 verify must reject last-byte mismatch output_len={out_len}"
      );

      assert!(ok_blocks > 0, "sha512 verify must do real work output_len={out_len}");
      assert_eq!(
        ok_blocks, wrong_first_blocks,
        "sha512 verify must not short-circuit on first-byte mismatch output_len={out_len}"
      );
      assert_eq!(
        ok_blocks, wrong_last_blocks,
        "sha512 verify must not short-circuit on last-byte mismatch output_len={out_len}"
      );
    }
  }
}
