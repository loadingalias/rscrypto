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

use core::fmt;

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

const SHA512_OUTPUT_SIZE: usize = 64;
const SHA512_BLOCK_SIZE: usize = 128;

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
/// let err = Pbkdf2Sha256::derive_key(b"pw", b"salt", 0, &mut [0u8; 32]);
/// assert_eq!(err, Err(Pbkdf2Error::InvalidIterations));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Pbkdf2Error {
  /// The iteration count must be at least 1.
  InvalidIterations,
  /// The requested output length exceeds `(2^32 − 1) × hLen`.
  OutputTooLong,
}

impl fmt::Display for Pbkdf2Error {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::InvalidIterations => f.write_str("PBKDF2 iteration count must be at least 1"),
      Self::OutputTooLong => f.write_str("PBKDF2 output length exceeds algorithm maximum"),
    }
  }
}

impl core::error::Error for Pbkdf2Error {}

// ─── PBKDF2-HMAC-SHA256 ────────────────────────────────────────────────────

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
/// let dk = Pbkdf2Sha256::derive_key_array::<32>(b"password", b"salt", 600_000)?;
///
/// // Re-use pre-computed state for multiple operations
/// let state = Pbkdf2Sha256::new(b"password");
/// let dk2 = state.derive_array::<32>(b"salt", 600_000)?;
/// assert_eq!(dk, dk2);
///
/// // Verify
/// assert!(state.verify(b"salt", 600_000, &dk).is_ok());
/// # Ok::<(), rscrypto::auth::Pbkdf2Error>(())
/// ```
#[derive(Clone)]
pub struct Pbkdf2Sha256 {
  inner_init: [u32; 8],
  outer_init: [u32; 8],
  compress: Sha256CompressBlocksFn,
}

impl fmt::Debug for Pbkdf2Sha256 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Pbkdf2Sha256").finish_non_exhaustive()
  }
}

impl Pbkdf2Sha256 {
  /// SHA-256 output size in bytes.
  pub const OUTPUT_SIZE: usize = SHA256_OUTPUT_SIZE;

  /// Pre-compute HMAC-SHA256 prefix states from `password`.
  #[must_use]
  #[allow(clippy::indexing_slicing)] // password.len() <= SHA256_BLOCK_SIZE in the else branch.
  pub fn new(password: &[u8]) -> Self {
    let compress = sha256_dispatch::compress_dispatch().select(0);

    let mut key_block = [0u8; SHA256_BLOCK_SIZE];
    if password.len() > SHA256_BLOCK_SIZE {
      let digest = Sha256::digest(password);
      key_block[..SHA256_OUTPUT_SIZE].copy_from_slice(&digest);
    } else {
      key_block[..password.len()].copy_from_slice(password);
    }

    let mut ipad = [0u8; SHA256_BLOCK_SIZE];
    let mut opad = [0u8; SHA256_BLOCK_SIZE];
    for ((ip, op), &kb) in ipad.iter_mut().zip(opad.iter_mut()).zip(key_block.iter()) {
      *ip = kb ^ 0x36;
      *op = kb ^ 0x5c;
    }

    let mut inner_init = SHA256_H0;
    compress(&mut inner_init, &ipad);

    let mut outer_init = SHA256_H0;
    compress(&mut outer_init, &opad);

    ct::zeroize(&mut key_block);
    ct::zeroize(&mut ipad);
    ct::zeroize(&mut opad);

    Self { inner_init, outer_init, compress }
  }

  /// Derive a key into `okm`.
  ///
  /// Returns [`Pbkdf2Error::InvalidIterations`] if `iterations` is zero, or
  /// [`Pbkdf2Error::OutputTooLong`] if `okm` exceeds `(2^32 − 1) × 32` bytes.
  #[allow(clippy::indexing_slicing)]
  pub fn derive(&self, salt: &[u8], iterations: u32, okm: &mut [u8]) -> Result<(), Pbkdf2Error> {
    if iterations == 0 {
      return Err(Pbkdf2Error::InvalidIterations);
    }
    if okm.is_empty() {
      return Ok(());
    }
    let num_blocks = okm.len().div_ceil(SHA256_OUTPUT_SIZE);
    if num_blocks as u64 > u32::MAX as u64 {
      return Err(Pbkdf2Error::OutputTooLong);
    }

    let mut block_out = [0u8; SHA256_OUTPUT_SIZE];
    for (i, chunk) in okm.chunks_mut(SHA256_OUTPUT_SIZE).enumerate() {
      let block_index = (i as u32).strict_add(1);
      pbkdf2_sha256_f(self.compress, &self.inner_init, &self.outer_init, salt, iterations, block_index, &mut block_out);
      chunk.copy_from_slice(&block_out[..chunk.len()]);
    }

    ct::zeroize(&mut block_out);
    Ok(())
  }

  /// Derive a key into a fixed-size array.
  pub fn derive_array<const N: usize>(&self, salt: &[u8], iterations: u32) -> Result<[u8; N], Pbkdf2Error> {
    let mut out = [0u8; N];
    self.derive(salt, iterations, &mut out)?;
    Ok(out)
  }

  /// Verify `expected` against the derived key in constant time.
  ///
  /// Returns [`VerificationError`] on mismatch or if parameters are invalid.
  #[allow(clippy::indexing_slicing)]
  pub fn verify(&self, salt: &[u8], iterations: u32, expected: &[u8]) -> Result<(), VerificationError> {
    if iterations == 0 || expected.is_empty() {
      return Err(VerificationError::new());
    }
    let num_blocks = expected.len().div_ceil(SHA256_OUTPUT_SIZE);
    if num_blocks as u64 > u32::MAX as u64 {
      return Err(VerificationError::new());
    }

    let mut block_out = [0u8; SHA256_OUTPUT_SIZE];
    let mut acc = 0u8;

    for (i, chunk) in expected.chunks(SHA256_OUTPUT_SIZE).enumerate() {
      let block_index = (i as u32).strict_add(1);
      pbkdf2_sha256_f(self.compress, &self.inner_init, &self.outer_init, salt, iterations, block_index, &mut block_out);
      for (&a, &b) in block_out[..chunk.len()].iter().zip(chunk.iter()) {
        acc |= a ^ b;
      }
    }

    ct::zeroize(&mut block_out);

    if core::hint::black_box(acc) == 0 { Ok(()) } else { Err(VerificationError::new()) }
  }

  /// Derive a key in one shot.
  #[inline]
  pub fn derive_key(password: &[u8], salt: &[u8], iterations: u32, okm: &mut [u8]) -> Result<(), Pbkdf2Error> {
    Self::new(password).derive(salt, iterations, okm)
  }

  /// Derive a key into a fixed-size array in one shot.
  #[inline]
  pub fn derive_key_array<const N: usize>(
    password: &[u8],
    salt: &[u8],
    iterations: u32,
  ) -> Result<[u8; N], Pbkdf2Error> {
    Self::new(password).derive_array(salt, iterations)
  }

  /// Verify a password in one shot.
  #[inline]
  pub fn verify_password(
    password: &[u8],
    salt: &[u8],
    iterations: u32,
    expected: &[u8],
  ) -> Result<(), VerificationError> {
    Self::new(password).verify(salt, iterations, expected)
  }

  /// Test-only: build with a specific SHA-256 compress function.
  #[cfg(test)]
  #[allow(clippy::indexing_slicing)]
  pub(crate) fn new_with_compress_for_test(password: &[u8], compress: Sha256CompressBlocksFn) -> Self {
    let mut key_block = [0u8; SHA256_BLOCK_SIZE];
    if password.len() > SHA256_BLOCK_SIZE {
      key_block[..SHA256_OUTPUT_SIZE].copy_from_slice(&sha256_oneshot_with_compress(password, compress));
    } else {
      key_block[..password.len()].copy_from_slice(password);
    }

    let mut ipad = [0u8; SHA256_BLOCK_SIZE];
    let mut opad = [0u8; SHA256_BLOCK_SIZE];
    for ((ip, op), &kb) in ipad.iter_mut().zip(opad.iter_mut()).zip(key_block.iter()) {
      *ip = kb ^ 0x36;
      *op = kb ^ 0x5c;
    }

    let mut inner_init = SHA256_H0;
    compress(&mut inner_init, &ipad);

    let mut outer_init = SHA256_H0;
    compress(&mut outer_init, &opad);

    ct::zeroize(&mut key_block);
    ct::zeroize(&mut ipad);
    ct::zeroize(&mut opad);

    Self { inner_init, outer_init, compress }
  }
}

/// Test-only: one-shot SHA-256 digest using a specific compress function.
#[cfg(test)]
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

impl Drop for Pbkdf2Sha256 {
  fn drop(&mut self) {
    for word in self.inner_init.iter_mut().chain(self.outer_init.iter_mut()) {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

/// Compute one PBKDF2-SHA256 block: `F(Password, Salt, c, i)`.
///
/// Each HMAC iteration in the hot loop runs exactly 2 SHA-256 compress calls
/// using pre-padded block templates — no hash struct creation, no dispatch
/// overhead, no padding recomputation.
#[allow(clippy::indexing_slicing)]
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
  let mut u = [0u8; SHA256_OUTPUT_SIZE];

  // ── U1 = HMAC(Password, Salt || INT_32_BE(block_index)) ──────────────
  state = *inner_init;
  let msg_len = salt.len().strict_add(4);
  let total_inner = (SHA256_BLOCK_SIZE as u64).strict_add(msg_len as u64);

  let mut block = [0u8; SHA256_BLOCK_SIZE];
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

  // Outer hash of U1: single block
  let mut outer_block = [0u8; SHA256_BLOCK_SIZE];
  for (chunk, &word) in outer_block[..SHA256_OUTPUT_SIZE].chunks_exact_mut(4).zip(state.iter()) {
    chunk.copy_from_slice(&word.to_be_bytes());
  }
  outer_block[SHA256_OUTPUT_SIZE] = 0x80;
  // total outer bytes = 64 (opad, pre-compressed) + 32 (inner hash) = 96 → 768 bits
  outer_block[56..SHA256_BLOCK_SIZE].copy_from_slice(&768u64.to_be_bytes());

  state = *outer_init;
  compress(&mut state, &outer_block);

  for (dst, &word) in u.chunks_exact_mut(4).zip(state.iter()) {
    dst.copy_from_slice(&word.to_be_bytes());
  }
  *output = u;

  // ── Iterations 2..=c (fixed-size HMAC, 2 compress calls each) ────────
  // Inner template: [32-byte U] [0x80] [zeros] [768-bit length]
  let mut inner_block = [0u8; SHA256_BLOCK_SIZE];
  inner_block[SHA256_OUTPUT_SIZE] = 0x80;
  inner_block[56..SHA256_BLOCK_SIZE].copy_from_slice(&768u64.to_be_bytes());

  for _ in 1..iterations {
    // Inner: compress(inner_init, U_{j-1} || padding)
    inner_block[..SHA256_OUTPUT_SIZE].copy_from_slice(&u);
    state = *inner_init;
    compress(&mut state, &inner_block);

    // Serialize inner hash directly into outer block
    for (chunk, &word) in outer_block[..SHA256_OUTPUT_SIZE].chunks_exact_mut(4).zip(state.iter()) {
      chunk.copy_from_slice(&word.to_be_bytes());
    }

    // Outer: compress(outer_init, inner_hash || padding)
    state = *outer_init;
    compress(&mut state, &outer_block);

    // Serialize Uj
    for (dst, &word) in u.chunks_exact_mut(4).zip(state.iter()) {
      dst.copy_from_slice(&word.to_be_bytes());
    }

    // XOR into output
    for (o, &x) in output.iter_mut().zip(u.iter()) {
      *o ^= x;
    }
  }

  // Zeroize sensitive state
  ct::zeroize_no_fence(&mut u);
  ct::zeroize_no_fence(&mut inner_block);
  ct::zeroize_no_fence(&mut outer_block);
  ct::zeroize_no_fence(&mut block);
  for word in state.iter_mut() {
    // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
    unsafe { core::ptr::write_volatile(word, 0) };
  }
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}

// ─── PBKDF2-HMAC-SHA512 ────────────────────────────────────────────────────

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
/// let dk = Pbkdf2Sha512::derive_key_array::<64>(b"password", b"salt", 210_000)?;
/// assert!(Pbkdf2Sha512::verify_password(b"password", b"salt", 210_000, &dk).is_ok());
/// # Ok::<(), rscrypto::auth::Pbkdf2Error>(())
/// ```
#[derive(Clone)]
pub struct Pbkdf2Sha512 {
  inner_init: [u64; 8],
  outer_init: [u64; 8],
  compress: Sha512CompressBlocksFn,
}

impl fmt::Debug for Pbkdf2Sha512 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Pbkdf2Sha512").finish_non_exhaustive()
  }
}

impl Pbkdf2Sha512 {
  /// SHA-512 output size in bytes.
  pub const OUTPUT_SIZE: usize = SHA512_OUTPUT_SIZE;

  /// Pre-compute HMAC-SHA512 prefix states from `password`.
  #[must_use]
  #[allow(clippy::indexing_slicing)] // password.len() <= SHA512_BLOCK_SIZE in the else branch.
  pub fn new(password: &[u8]) -> Self {
    let compress = sha512_dispatch::compress_dispatch().select(0);

    let mut key_block = [0u8; SHA512_BLOCK_SIZE];
    if password.len() > SHA512_BLOCK_SIZE {
      let digest = Sha512::digest(password);
      key_block[..SHA512_OUTPUT_SIZE].copy_from_slice(&digest);
    } else {
      key_block[..password.len()].copy_from_slice(password);
    }

    let mut ipad = [0u8; SHA512_BLOCK_SIZE];
    let mut opad = [0u8; SHA512_BLOCK_SIZE];
    for ((ip, op), &kb) in ipad.iter_mut().zip(opad.iter_mut()).zip(key_block.iter()) {
      *ip = kb ^ 0x36;
      *op = kb ^ 0x5c;
    }

    let mut inner_init = SHA512_H0;
    compress(&mut inner_init, &ipad);

    let mut outer_init = SHA512_H0;
    compress(&mut outer_init, &opad);

    ct::zeroize(&mut key_block);
    ct::zeroize(&mut ipad);
    ct::zeroize(&mut opad);

    Self { inner_init, outer_init, compress }
  }

  /// Derive a key into `okm`.
  #[allow(clippy::indexing_slicing)]
  pub fn derive(&self, salt: &[u8], iterations: u32, okm: &mut [u8]) -> Result<(), Pbkdf2Error> {
    if iterations == 0 {
      return Err(Pbkdf2Error::InvalidIterations);
    }
    if okm.is_empty() {
      return Ok(());
    }
    let num_blocks = okm.len().div_ceil(SHA512_OUTPUT_SIZE);
    if num_blocks as u64 > u32::MAX as u64 {
      return Err(Pbkdf2Error::OutputTooLong);
    }

    let mut block_out = [0u8; SHA512_OUTPUT_SIZE];
    for (i, chunk) in okm.chunks_mut(SHA512_OUTPUT_SIZE).enumerate() {
      let block_index = (i as u32).strict_add(1);
      pbkdf2_sha512_f(self.compress, &self.inner_init, &self.outer_init, salt, iterations, block_index, &mut block_out);
      chunk.copy_from_slice(&block_out[..chunk.len()]);
    }

    ct::zeroize(&mut block_out);
    Ok(())
  }

  /// Derive a key into a fixed-size array.
  pub fn derive_array<const N: usize>(&self, salt: &[u8], iterations: u32) -> Result<[u8; N], Pbkdf2Error> {
    let mut out = [0u8; N];
    self.derive(salt, iterations, &mut out)?;
    Ok(out)
  }

  /// Verify `expected` against the derived key in constant time.
  #[allow(clippy::indexing_slicing)]
  pub fn verify(&self, salt: &[u8], iterations: u32, expected: &[u8]) -> Result<(), VerificationError> {
    if iterations == 0 || expected.is_empty() {
      return Err(VerificationError::new());
    }
    let num_blocks = expected.len().div_ceil(SHA512_OUTPUT_SIZE);
    if num_blocks as u64 > u32::MAX as u64 {
      return Err(VerificationError::new());
    }

    let mut block_out = [0u8; SHA512_OUTPUT_SIZE];
    let mut acc = 0u8;

    for (i, chunk) in expected.chunks(SHA512_OUTPUT_SIZE).enumerate() {
      let block_index = (i as u32).strict_add(1);
      pbkdf2_sha512_f(self.compress, &self.inner_init, &self.outer_init, salt, iterations, block_index, &mut block_out);
      for (&a, &b) in block_out[..chunk.len()].iter().zip(chunk.iter()) {
        acc |= a ^ b;
      }
    }

    ct::zeroize(&mut block_out);

    if core::hint::black_box(acc) == 0 { Ok(()) } else { Err(VerificationError::new()) }
  }

  /// Derive a key in one shot.
  #[inline]
  pub fn derive_key(password: &[u8], salt: &[u8], iterations: u32, okm: &mut [u8]) -> Result<(), Pbkdf2Error> {
    Self::new(password).derive(salt, iterations, okm)
  }

  /// Derive a key into a fixed-size array in one shot.
  #[inline]
  pub fn derive_key_array<const N: usize>(
    password: &[u8],
    salt: &[u8],
    iterations: u32,
  ) -> Result<[u8; N], Pbkdf2Error> {
    Self::new(password).derive_array(salt, iterations)
  }

  /// Verify a password in one shot.
  #[inline]
  pub fn verify_password(
    password: &[u8],
    salt: &[u8],
    iterations: u32,
    expected: &[u8],
  ) -> Result<(), VerificationError> {
    Self::new(password).verify(salt, iterations, expected)
  }

  /// Test-only: build with a specific SHA-512 compress function.
  #[cfg(test)]
  #[allow(clippy::indexing_slicing)]
  pub(crate) fn new_with_compress_for_test(password: &[u8], compress: Sha512CompressBlocksFn) -> Self {
    let mut key_block = [0u8; SHA512_BLOCK_SIZE];
    if password.len() > SHA512_BLOCK_SIZE {
      key_block[..SHA512_OUTPUT_SIZE].copy_from_slice(&sha512_oneshot_with_compress(password, compress));
    } else {
      key_block[..password.len()].copy_from_slice(password);
    }

    let mut ipad = [0u8; SHA512_BLOCK_SIZE];
    let mut opad = [0u8; SHA512_BLOCK_SIZE];
    for ((ip, op), &kb) in ipad.iter_mut().zip(opad.iter_mut()).zip(key_block.iter()) {
      *ip = kb ^ 0x36;
      *op = kb ^ 0x5c;
    }

    let mut inner_init = SHA512_H0;
    compress(&mut inner_init, &ipad);

    let mut outer_init = SHA512_H0;
    compress(&mut outer_init, &opad);

    ct::zeroize(&mut key_block);
    ct::zeroize(&mut ipad);
    ct::zeroize(&mut opad);

    Self { inner_init, outer_init, compress }
  }
}

/// Test-only: one-shot SHA-512 digest using a specific compress function.
#[cfg(test)]
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

impl Drop for Pbkdf2Sha512 {
  fn drop(&mut self) {
    for word in self.inner_init.iter_mut().chain(self.outer_init.iter_mut()) {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

/// Compute one PBKDF2-SHA512 block: `F(Password, Salt, c, i)`.
#[allow(clippy::indexing_slicing)]
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
  let mut u = [0u8; SHA512_OUTPUT_SIZE];

  // ── U1 = HMAC(Password, Salt || INT_32_BE(block_index)) ──────────────
  state = *inner_init;
  let msg_len = salt.len().strict_add(4);
  let total_inner = (SHA512_BLOCK_SIZE as u128).strict_add(msg_len as u128);

  let mut block = [0u8; SHA512_BLOCK_SIZE];
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

  // Outer hash of U1: single block
  let mut outer_block = [0u8; SHA512_BLOCK_SIZE];
  for (chunk, &word) in outer_block[..SHA512_OUTPUT_SIZE].chunks_exact_mut(8).zip(state.iter()) {
    chunk.copy_from_slice(&word.to_be_bytes());
  }
  outer_block[SHA512_OUTPUT_SIZE] = 0x80;
  // total outer bytes = 128 (opad, pre-compressed) + 64 (inner hash) = 192 → 1536 bits
  outer_block[112..SHA512_BLOCK_SIZE].copy_from_slice(&1536u128.to_be_bytes());

  state = *outer_init;
  compress(&mut state, &outer_block);

  for (dst, &word) in u.chunks_exact_mut(8).zip(state.iter()) {
    dst.copy_from_slice(&word.to_be_bytes());
  }
  *output = u;

  // ── Iterations 2..=c (fixed-size HMAC, 2 compress calls each) ────────
  let mut inner_block = [0u8; SHA512_BLOCK_SIZE];
  inner_block[SHA512_OUTPUT_SIZE] = 0x80;
  inner_block[112..SHA512_BLOCK_SIZE].copy_from_slice(&1536u128.to_be_bytes());

  for _ in 1..iterations {
    inner_block[..SHA512_OUTPUT_SIZE].copy_from_slice(&u);
    state = *inner_init;
    compress(&mut state, &inner_block);

    for (chunk, &word) in outer_block[..SHA512_OUTPUT_SIZE].chunks_exact_mut(8).zip(state.iter()) {
      chunk.copy_from_slice(&word.to_be_bytes());
    }

    state = *outer_init;
    compress(&mut state, &outer_block);

    for (dst, &word) in u.chunks_exact_mut(8).zip(state.iter()) {
      dst.copy_from_slice(&word.to_be_bytes());
    }

    for (o, &x) in output.iter_mut().zip(u.iter()) {
      *o ^= x;
    }
  }

  ct::zeroize_no_fence(&mut u);
  ct::zeroize_no_fence(&mut inner_block);
  ct::zeroize_no_fence(&mut outer_block);
  ct::zeroize_no_fence(&mut block);
  for word in state.iter_mut() {
    // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
    unsafe { core::ptr::write_volatile(word, 0) };
  }
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
  use alloc::vec;

  use super::*;

  // ── RFC 7914 §11 PBKDF2-HMAC-SHA256 test vectors ──────────────────────

  #[test]
  fn rfc7914_sha256_vector_1() {
    let mut dk = [0u8; 64];
    Pbkdf2Sha256::derive_key(b"passwd", b"salt", 1, &mut dk).unwrap();
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
  fn rfc7914_sha256_vector_2() {
    let mut dk = [0u8; 64];
    Pbkdf2Sha256::derive_key(b"Password", b"NaCl", 80000, &mut dk).unwrap();
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

  // ── Oracle: inline PBKDF2 using the RustCrypto hmac + sha2 dev-deps ──

  use hmac::{Hmac, Mac as _, digest::KeyInit};

  type OracleHmacSha256 = Hmac<sha2::Sha256>;
  type OracleHmacSha512 = Hmac<sha2::Sha512>;

  fn oracle_sha256(password: &[u8], salt: &[u8], iterations: u32, out: &mut [u8]) {
    for (i, chunk) in out.chunks_mut(32).enumerate() {
      let block_index = (i as u32).strict_add(1);
      let mut mac = OracleHmacSha256::new_from_slice(password).unwrap();
      mac.update(salt);
      mac.update(&block_index.to_be_bytes());
      let mut u: [u8; 32] = mac.finalize().into_bytes().into();
      let mut result = u;
      for _ in 1..iterations {
        let mut mac = OracleHmacSha256::new_from_slice(password).unwrap();
        mac.update(&u);
        u = mac.finalize().into_bytes().into();
        for (r, &x) in result.iter_mut().zip(u.iter()) {
          *r ^= x;
        }
      }
      chunk.copy_from_slice(&result[..chunk.len()]);
    }
  }

  fn oracle_sha512(password: &[u8], salt: &[u8], iterations: u32, out: &mut [u8]) {
    for (i, chunk) in out.chunks_mut(64).enumerate() {
      let block_index = (i as u32).strict_add(1);
      let mut mac = OracleHmacSha512::new_from_slice(password).unwrap();
      mac.update(salt);
      mac.update(&block_index.to_be_bytes());
      let mut u: [u8; 64] = mac.finalize().into_bytes().into();
      let mut result = u;
      for _ in 1..iterations {
        let mut mac = OracleHmacSha512::new_from_slice(password).unwrap();
        mac.update(&u);
        u = mac.finalize().into_bytes().into();
        for (r, &x) in result.iter_mut().zip(u.iter()) {
          *r ^= x;
        }
      }
      chunk.copy_from_slice(&result[..chunk.len()]);
    }
  }

  #[test]
  fn sha256_matches_oracle() {
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

    for &(password, salt, iterations, dk_len) in cases {
      let mut expected = vec![0u8; dk_len];
      oracle_sha256(password, salt, iterations, &mut expected);

      let mut actual = vec![0u8; dk_len];
      Pbkdf2Sha256::derive_key(password, salt, iterations, &mut actual).unwrap();

      assert_eq!(
        actual, expected,
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

    for &(password, salt, iterations, dk_len) in cases {
      let mut expected = vec![0u8; dk_len];
      oracle_sha512(password, salt, iterations, &mut expected);

      let mut actual = vec![0u8; dk_len];
      Pbkdf2Sha512::derive_key(password, salt, iterations, &mut actual).unwrap();

      assert_eq!(
        actual, expected,
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
    let dk = Pbkdf2Sha256::derive_key_array::<32>(b"password", b"salt", 100).unwrap();
    assert!(Pbkdf2Sha256::verify_password(b"password", b"salt", 100, &dk).is_ok());
  }

  #[test]
  fn sha256_verify_wrong_password() {
    let dk = Pbkdf2Sha256::derive_key_array::<32>(b"password", b"salt", 100).unwrap();
    assert!(Pbkdf2Sha256::verify_password(b"wrong", b"salt", 100, &dk).is_err());
  }

  #[test]
  fn sha256_verify_wrong_salt() {
    let dk = Pbkdf2Sha256::derive_key_array::<32>(b"password", b"salt", 100).unwrap();
    assert!(Pbkdf2Sha256::verify_password(b"password", b"wrong", 100, &dk).is_err());
  }

  #[test]
  fn sha256_verify_wrong_iterations() {
    let dk = Pbkdf2Sha256::derive_key_array::<32>(b"password", b"salt", 100).unwrap();
    assert!(Pbkdf2Sha256::verify_password(b"password", b"salt", 101, &dk).is_err());
  }

  #[test]
  fn sha512_verify_correct_password() {
    let dk = Pbkdf2Sha512::derive_key_array::<64>(b"password", b"salt", 100).unwrap();
    assert!(Pbkdf2Sha512::verify_password(b"password", b"salt", 100, &dk).is_ok());
  }

  #[test]
  fn sha512_verify_wrong_password() {
    let dk = Pbkdf2Sha512::derive_key_array::<64>(b"password", b"salt", 100).unwrap();
    assert!(Pbkdf2Sha512::verify_password(b"wrong", b"salt", 100, &dk).is_err());
  }

  // ── Error paths ───────────────────────────────────────────────────────

  #[test]
  fn sha256_zero_iterations_error() {
    let mut dk = [0u8; 32];
    assert_eq!(Pbkdf2Sha256::derive_key(b"pw", b"salt", 0, &mut dk), Err(Pbkdf2Error::InvalidIterations));
  }

  #[test]
  fn sha512_zero_iterations_error() {
    let mut dk = [0u8; 64];
    assert_eq!(Pbkdf2Sha512::derive_key(b"pw", b"salt", 0, &mut dk), Err(Pbkdf2Error::InvalidIterations));
  }

  #[test]
  fn sha256_empty_output_ok() {
    assert!(Pbkdf2Sha256::derive_key(b"pw", b"salt", 1, &mut []).is_ok());
  }

  #[test]
  fn sha512_empty_output_ok() {
    assert!(Pbkdf2Sha512::derive_key(b"pw", b"salt", 1, &mut []).is_ok());
  }

  #[test]
  fn sha256_verify_zero_iterations() {
    assert!(Pbkdf2Sha256::verify_password(b"pw", b"salt", 0, &[0u8; 32]).is_err());
  }

  #[test]
  fn sha256_verify_empty_expected() {
    assert!(Pbkdf2Sha256::verify_password(b"pw", b"salt", 1, &[]).is_err());
  }

  // ── Streaming state reuse ─────────────────────────────────────────────

  #[test]
  fn sha256_state_reuse_matches_oneshot() {
    let state = Pbkdf2Sha256::new(b"password");
    let dk1 = state.derive_array::<32>(b"salt1", 100).unwrap();
    let dk2 = state.derive_array::<32>(b"salt2", 100).unwrap();

    let oneshot1 = Pbkdf2Sha256::derive_key_array::<32>(b"password", b"salt1", 100).unwrap();
    let oneshot2 = Pbkdf2Sha256::derive_key_array::<32>(b"password", b"salt2", 100).unwrap();

    assert_eq!(dk1, oneshot1);
    assert_eq!(dk2, oneshot2);
    assert_ne!(dk1, dk2);
  }

  // ── c=1 edge case ─────────────────────────────────────────────────────

  #[test]
  fn sha256_single_iteration() {
    let mut expected = [0u8; 32];
    oracle_sha256(b"pw", b"salt", 1, &mut expected);
    let actual = Pbkdf2Sha256::derive_key_array::<32>(b"pw", b"salt", 1).unwrap();
    assert_eq!(actual, expected);
  }

  #[test]
  fn sha512_single_iteration() {
    let mut expected = [0u8; 64];
    oracle_sha512(b"pw", b"salt", 1, &mut expected);
    let actual = Pbkdf2Sha512::derive_key_array::<64>(b"pw", b"salt", 1).unwrap();
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
    assert!(!Pbkdf2Error::InvalidIterations.to_string().is_empty());
    assert!(!Pbkdf2Error::OutputTooLong.to_string().is_empty());
  }

  #[test]
  fn error_debug() {
    assert!(!format!("{:?}", Pbkdf2Error::InvalidIterations).is_empty());
  }

  #[test]
  fn error_implements_error_trait() {
    fn assert_error<T: core::error::Error>() {}
    assert_error::<Pbkdf2Error>();
  }

  // ── Forced-kernel oracle tests ──────────────────────────────────────

  use crate::hashes::crypto::sha256::kernels::{
    Sha256KernelId, compress_blocks_fn as sha256_compress_blocks_fn,
    required_caps as sha256_required_caps,
  };
  use crate::hashes::crypto::sha512::kernels::{
    ALL as SHA512_KERNELS, Sha512KernelId, compress_blocks_fn as sha512_compress_blocks_fn,
    required_caps as sha512_required_caps,
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

  fn assert_pbkdf2_sha256_kernel(id: Sha256KernelId) {
    let compress = sha256_compress_blocks_fn(id);
    let cases: &[(&[u8], &[u8], u32, usize)] = &[
      (b"password", b"salt", 1, 32),
      (b"password", b"salt", 4, 32),
      (b"password", b"salt", 100, 32),
      (b"password", b"salt", 1, 64),    // multi-block
      (b"", b"salt", 1, 32),            // empty password
      (b"password", b"", 1, 32),        // empty salt
      (b"p", b"s", 1, 1),              // minimal output
      (b"password", &[0xBB; 200], 1, 32), // long salt
      (&[0xCC; 128], b"salt", 1, 32),   // password > block_size (triggers key hashing)
    ];

    for &(password, salt, iterations, dk_len) in cases {
      let mut expected = vec![0u8; dk_len];
      oracle_sha256(password, salt, iterations, &mut expected);

      let state = Pbkdf2Sha256::new_with_compress_for_test(password, compress);
      let mut actual = vec![0u8; dk_len];
      state.derive(salt, iterations, &mut actual).unwrap();

      assert_eq!(
        actual, expected,
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
      (b"password", b"salt", 1, 128),   // multi-block
      (b"", b"salt", 1, 64),            // empty password
      (b"password", b"", 1, 64),        // empty salt
      (b"p", b"s", 1, 1),              // minimal output
      (b"password", &[0xBB; 200], 1, 64), // long salt
      (&[0xCC; 200], b"salt", 1, 64),   // password > block_size
    ];

    for &(password, salt, iterations, dk_len) in cases {
      let mut expected = vec![0u8; dk_len];
      oracle_sha512(password, salt, iterations, &mut expected);

      let state = Pbkdf2Sha512::new_with_compress_for_test(password, compress);
      let mut actual = vec![0u8; dk_len];
      state.derive(salt, iterations, &mut actual).unwrap();

      assert_eq!(
        actual, expected,
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
}
