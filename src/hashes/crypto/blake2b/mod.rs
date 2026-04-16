//! Blake2b cryptographic hash (RFC 7693).
//!
//! Blake2b-256 and Blake2b-512, with optional keyed hashing (Blake2b-MAC).
//! Blake2b is a prerequisite for Argon2 and is independently useful in NOISE,
//! WireGuard, Zcash, and other protocols.
//!
//! # Examples
//!
//! ```rust
//! use rscrypto::{Blake2b256, Blake2b512, Digest};
//!
//! let hash = Blake2b256::digest(b"hello world");
//! assert_eq!(hash.len(), 32);
//!
//! let mut hasher = Blake2b512::new();
//! hasher.update(b"hello ");
//! hasher.update(b"world");
//! let hash = hasher.finalize();
//! assert_eq!(hash, Blake2b512::digest(b"hello world"));
//! ```
//!
//! ## Keyed Hashing
//!
//! ```rust
//! use rscrypto::{Blake2b256, Digest};
//!
//! let key = b"secret-key-up-to-64-bytes";
//! let tag = Blake2b256::keyed_hash(key, b"message");
//! assert_ne!(tag, Blake2b256::digest(b"message"));
//! ```

pub(crate) mod kernels;

#[cfg(target_arch = "aarch64")]
mod aarch64;
mod dispatch;
#[cfg(target_arch = "powerpc64")]
mod power;
#[cfg(target_arch = "riscv64")]
mod riscv64;
#[cfg(target_arch = "s390x")]
mod s390x;
#[cfg(target_arch = "wasm32")]
mod wasm;
#[cfg(target_arch = "x86_64")]
mod x86_64;

use core::fmt;

use kernels::IV;

use crate::traits::{Digest, ct};

const BLOCK_SIZE: usize = 128;
const MAX_KEY_LEN: usize = 64;
const MAX_OUTPUT_LEN: usize = 64;

// ─── Core state ─────────────────────────────────────────────────────────────

/// Internal Blake2b state shared by `Blake2b256` and `Blake2b512`.
///
/// Handles any output length (1-64 bytes) and optional keyed hashing.
#[derive(Clone)]
struct Core {
  h: [u64; 8],
  buf: [u8; BLOCK_SIZE],
  buf_len: u8,
  t: u128,
  nn: u8,
  kk: u8,
  key: [u8; MAX_KEY_LEN],
  compress: kernels::CompressFn,
}

impl Core {
  /// Create a new Blake2b state with output length `nn` and optional `key`.
  ///
  /// # Panics
  ///
  /// Panics if `nn` is 0 or > 64, or if `key` is longer than 64 bytes.
  #[allow(clippy::indexing_slicing)]
  fn new(nn: u8, key: &[u8]) -> Self {
    assert!(nn >= 1 && nn as usize <= MAX_OUTPUT_LEN, "Blake2b output length must be 1-64");
    assert!(key.len() <= MAX_KEY_LEN, "Blake2b key must be at most 64 bytes");

    let kk = key.len() as u8;

    // Parameter block (sequential mode): fanout=1, depth=1
    let p0 = nn as u64 | ((kk as u64) << 8) | (1u64 << 16) | (1u64 << 24);
    let mut h = IV;
    h[0] ^= p0;

    let mut stored_key = [0u8; MAX_KEY_LEN];
    stored_key[..key.len()].copy_from_slice(key);

    let mut buf = [0u8; BLOCK_SIZE];
    let buf_len;

    if kk > 0 {
      // Pad key to block size and buffer it
      buf[..key.len()].copy_from_slice(key);
      buf_len = BLOCK_SIZE as u8;
    } else {
      buf_len = 0;
    }

    Self { h, buf, buf_len, t: 0, nn, kk, key: stored_key, compress: dispatch::compress_dispatch() }
  }

  /// Feed data into the hash state.
  #[allow(clippy::indexing_slicing)]
  fn update(&mut self, data: &[u8]) {
    if data.is_empty() {
      return;
    }

    let mut offset = 0usize;
    let data_len = data.len();

    // If buffer has data and adding new data would exceed block size,
    // fill and compress the current buffer.
    if self.buf_len > 0 && (self.buf_len as usize).strict_add(data_len) > BLOCK_SIZE {
      let fill = BLOCK_SIZE.strict_sub(self.buf_len as usize);
      self.buf[self.buf_len as usize..BLOCK_SIZE].copy_from_slice(&data[..fill]);
      self.t = self.t.strict_add(BLOCK_SIZE as u128);
      (self.compress)(&mut self.h, &self.buf, self.t, false);
      self.buf_len = 0;
      offset = fill;
    }

    // Process full blocks directly from data, keeping the last chunk
    // in the buffer so finalize can set the final flag.
    while offset.strict_add(BLOCK_SIZE) < data_len {
      self.t = self.t.strict_add(BLOCK_SIZE as u128);
      // SAFETY: offset + BLOCK_SIZE < data_len, so 128 bytes are in bounds.
      // [u8; 128] has alignment 1, matching u8.
      let block = unsafe { &*data.as_ptr().add(offset).cast::<[u8; BLOCK_SIZE]>() };
      (self.compress)(&mut self.h, block, self.t, false);
      offset = offset.strict_add(BLOCK_SIZE);
    }

    // Buffer remaining data
    let remaining = data_len.strict_sub(offset);
    if remaining > 0 {
      let start = self.buf_len as usize;
      self.buf[start..start.strict_add(remaining)].copy_from_slice(&data[offset..]);
      self.buf_len = self.buf_len.strict_add(remaining as u8);
    }
  }

  /// Finalize and write the hash into `out` (must be exactly `nn` bytes).
  #[allow(clippy::indexing_slicing)]
  fn finalize_into(&self, out: &mut [u8]) {
    debug_assert!(out.len() == self.nn as usize);

    let mut h = self.h;
    let mut last_block = [0u8; BLOCK_SIZE];
    last_block[..self.buf_len as usize].copy_from_slice(&self.buf[..self.buf_len as usize]);

    let t = self.t.strict_add(self.buf_len as u128);
    (self.compress)(&mut h, &last_block, t, true);

    // Extract first nn bytes from h (little-endian word order)
    let nn = self.nn as usize;
    let full_words = nn / 8;
    for i in 0..full_words {
      let bytes = h[i].to_le_bytes();
      let off = i.strict_mul(8);
      out[off..off.strict_add(8)].copy_from_slice(&bytes);
    }
    let tail = nn % 8;
    if tail > 0 {
      let bytes = h[full_words].to_le_bytes();
      let off = full_words.strict_mul(8);
      out[off..off.strict_add(tail)].copy_from_slice(&bytes[..tail]);
    }

    // Zeroize local state
    for word in h.iter_mut() {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    ct::zeroize(&mut last_block);
  }

  #[cfg(test)]
  fn new_with_compress_for_test(nn: u8, key: &[u8], compress: kernels::CompressFn) -> Self {
    let mut core = Self::new(nn, key);
    core.compress = compress;
    core
  }

  /// Reset to the initial state (including re-buffering the key if keyed).
  #[allow(clippy::indexing_slicing)]
  fn reset(&mut self) {
    let p0 = self.nn as u64 | ((self.kk as u64) << 8) | (1u64 << 16) | (1u64 << 24);
    self.h = IV;
    self.h[0] ^= p0;

    self.buf = [0u8; BLOCK_SIZE];
    if self.kk > 0 {
      self.buf[..self.kk as usize].copy_from_slice(&self.key[..self.kk as usize]);
      self.buf_len = BLOCK_SIZE as u8;
    } else {
      self.buf_len = 0;
    }
    self.t = 0;
  }
}

impl Drop for Core {
  fn drop(&mut self) {
    for word in self.h.iter_mut() {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    ct::zeroize_no_fence(&mut self.buf);
    ct::zeroize_no_fence(&mut self.key);
    // SAFETY: buf_len, t, nn, kk are valid, aligned, dereferenceable pointers.
    unsafe {
      core::ptr::write_volatile(&mut self.buf_len, 0);
      core::ptr::write_volatile(&mut self.t, 0);
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

// ─── Blake2b256 ─────────────────────────────────────────────────────────────

/// Blake2b-256 cryptographic hash (32-byte output).
///
/// # Examples
///
/// ```rust
/// use rscrypto::{Blake2b256, Digest};
///
/// let hash = Blake2b256::digest(b"hello world");
/// assert_eq!(hash.len(), 32);
///
/// let mut h = Blake2b256::new();
/// h.update(b"hello ");
/// h.update(b"world");
/// assert_eq!(h.finalize(), hash);
/// ```
#[derive(Clone)]
pub struct Blake2b256(Core);

impl fmt::Debug for Blake2b256 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Blake2b256").finish_non_exhaustive()
  }
}

impl Blake2b256 {
  /// Output size in bytes.
  pub const OUTPUT_SIZE: usize = 32;

  /// Create a keyed Blake2b-256 instance (Blake2b-MAC).
  ///
  /// # Panics
  ///
  /// Panics if `key` is empty or longer than 64 bytes.
  #[must_use]
  pub fn keyed(key: &[u8]) -> Self {
    assert!(!key.is_empty(), "use Blake2b256::new() for unkeyed hashing");
    Self(Core::new(32, key))
  }

  /// Compute a keyed Blake2b-256 hash in one shot.
  ///
  /// # Panics
  ///
  /// Panics if `key` is empty or longer than 64 bytes.
  #[must_use]
  pub fn keyed_hash(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut h = Self::keyed(key);
    h.update(data);
    h.finalize()
  }
}

impl Blake2b256 {
  #[cfg(test)]
  pub(crate) fn new_with_compress_for_test(compress: kernels::CompressFn) -> Self {
    Self(Core::new_with_compress_for_test(32, &[], compress))
  }

  #[cfg(test)]
  pub(crate) fn keyed_with_compress_for_test(key: &[u8], compress: kernels::CompressFn) -> Self {
    assert!(!key.is_empty(), "use new_with_compress_for_test() for unkeyed hashing");
    Self(Core::new_with_compress_for_test(32, key, compress))
  }
}

impl Default for Blake2b256 {
  fn default() -> Self {
    Self(Core::new(32, &[]))
  }
}

impl Digest for Blake2b256 {
  const OUTPUT_SIZE: usize = 32;
  type Output = [u8; 32];

  #[inline]
  fn new() -> Self {
    Self(Core::new(32, &[]))
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.0.update(data);
  }

  fn finalize(&self) -> Self::Output {
    let mut out = [0u8; 32];
    self.0.finalize_into(&mut out);
    out
  }

  #[inline]
  fn reset(&mut self) {
    self.0.reset();
  }
}

impl Drop for Blake2b256 {
  fn drop(&mut self) {
    // Core::drop handles zeroization
  }
}

// ─── Blake2b512 ─────────────────────────────────────────────────────────────

/// Blake2b-512 cryptographic hash (64-byte output).
///
/// # Examples
///
/// ```rust
/// use rscrypto::{Blake2b512, Digest};
///
/// let hash = Blake2b512::digest(b"hello world");
/// assert_eq!(hash.len(), 64);
///
/// let mut h = Blake2b512::new();
/// h.update(b"hello ");
/// h.update(b"world");
/// assert_eq!(h.finalize(), hash);
/// ```
#[derive(Clone)]
pub struct Blake2b512(Core);

impl fmt::Debug for Blake2b512 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Blake2b512").finish_non_exhaustive()
  }
}

impl Blake2b512 {
  /// Output size in bytes.
  pub const OUTPUT_SIZE: usize = 64;

  /// Create a keyed Blake2b-512 instance (Blake2b-MAC).
  ///
  /// # Panics
  ///
  /// Panics if `key` is empty or longer than 64 bytes.
  #[must_use]
  pub fn keyed(key: &[u8]) -> Self {
    assert!(!key.is_empty(), "use Blake2b512::new() for unkeyed hashing");
    Self(Core::new(64, key))
  }

  /// Compute a keyed Blake2b-512 hash in one shot.
  ///
  /// # Panics
  ///
  /// Panics if `key` is empty or longer than 64 bytes.
  #[must_use]
  pub fn keyed_hash(key: &[u8], data: &[u8]) -> [u8; 64] {
    let mut h = Self::keyed(key);
    h.update(data);
    h.finalize()
  }
}

impl Blake2b512 {
  #[cfg(test)]
  pub(crate) fn new_with_compress_for_test(compress: kernels::CompressFn) -> Self {
    Self(Core::new_with_compress_for_test(64, &[], compress))
  }

  #[cfg(test)]
  pub(crate) fn keyed_with_compress_for_test(key: &[u8], compress: kernels::CompressFn) -> Self {
    assert!(!key.is_empty(), "use new_with_compress_for_test() for unkeyed hashing");
    Self(Core::new_with_compress_for_test(64, key, compress))
  }
}

impl Default for Blake2b512 {
  fn default() -> Self {
    Self(Core::new(64, &[]))
  }
}

impl Digest for Blake2b512 {
  const OUTPUT_SIZE: usize = 64;
  type Output = [u8; 64];

  #[inline]
  fn new() -> Self {
    Self(Core::new(64, &[]))
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.0.update(data);
  }

  fn finalize(&self) -> Self::Output {
    let mut out = [0u8; 64];
    self.0.finalize_into(&mut out);
    out
  }

  #[inline]
  fn reset(&mut self) {
    self.0.reset();
  }
}

impl Drop for Blake2b512 {
  fn drop(&mut self) {
    // Core::drop handles zeroization
  }
}

// ─── Variable-output helper for Argon2 ──────────────────────────────────────

/// One-shot Blake2b with variable output length (1-64 bytes).
///
/// Used internally by Argon2's variable-length hash function H'.
#[allow(clippy::indexing_slicing)]
pub(crate) fn blake2b_hash(data: &[u8], nn: u8, out: &mut [u8]) {
  debug_assert!(nn >= 1 && nn as usize <= MAX_OUTPUT_LEN);
  debug_assert!(out.len() == nn as usize);
  let mut core = Core::new(nn, &[]);
  core.update(data);
  core.finalize_into(out);
}

#[cfg(test)]
mod tests {
  use blake2::{Blake2b as OracleBlake2b, Blake2bMac, Digest as _};
  use digest::consts::{U32, U64};
  use digest::KeyInit;

  use super::*;

  type OracleBlake2b256 = OracleBlake2b<U32>;
  type OracleBlake2b512 = OracleBlake2b<U64>;
  type OracleBlake2bMac256 = Blake2bMac<U32>;

  fn oracle_hash_256(data: &[u8]) -> [u8; 32] {
    let mut h = OracleBlake2b256::new();
    h.update(data);
    let result = h.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
  }

  fn oracle_hash_512(data: &[u8]) -> [u8; 64] {
    let mut h = OracleBlake2b512::new();
    h.update(data);
    let result = h.finalize();
    let mut out = [0u8; 64];
    out.copy_from_slice(&result);
    out
  }

  // ── Unkeyed oracle tests ────────────────────────────────────────────

  #[test]
  fn blake2b256_matches_oracle() {
    let cases: &[&[u8]] = &[
      b"",
      b"a",
      b"abc",
      b"hello world",
      &[0u8; 64],
      &[0xFFu8; 128],
      &[0xAAu8; 129],
      &[0xBBu8; 256],
      &[0xCCu8; 1024],
    ];
    for &data in cases {
      let expected = oracle_hash_256(data);
      let actual = Blake2b256::digest(data);
      assert_eq!(actual, expected, "Blake2b256 mismatch for len={}", data.len());
    }
  }

  #[test]
  fn blake2b512_matches_oracle() {
    let cases: &[&[u8]] = &[
      b"",
      b"a",
      b"abc",
      b"hello world",
      &[0u8; 64],
      &[0xFFu8; 128],
      &[0xAAu8; 129],
      &[0xBBu8; 256],
      &[0xCCu8; 1024],
    ];
    for &data in cases {
      let expected = oracle_hash_512(data);
      let actual = Blake2b512::digest(data);
      assert_eq!(actual, expected, "Blake2b512 mismatch for len={}", data.len());
    }
  }

  // ── Streaming ─────────────────────────────────────────────────────────

  #[test]
  fn blake2b256_streaming_matches_oneshot() {
    let data = b"The quick brown fox jumps over the lazy dog";
    let oneshot = Blake2b256::digest(data);

    let mut h = Blake2b256::new();
    for byte in data.iter() {
      h.update(core::slice::from_ref(byte));
    }
    assert_eq!(h.finalize(), oneshot);
  }

  #[test]
  fn blake2b512_streaming_matches_oneshot() {
    let data = [0x42u8; 300];
    let oneshot = Blake2b512::digest(&data);

    // Feed in various chunk sizes
    let mut h = Blake2b512::new();
    let mut off = 0;
    for chunk_size in [1, 7, 63, 64, 127, 128, 129] {
      let end = (off + chunk_size).min(data.len());
      if off < end {
        h.update(&data[off..end]);
        off = end;
      }
    }
    if off < data.len() {
      h.update(&data[off..]);
    }
    assert_eq!(h.finalize(), oneshot);
  }

  // ── Reset ─────────────────────────────────────────────────────────────

  #[test]
  fn blake2b256_reset() {
    let hash1 = Blake2b256::digest(b"first");
    let hash2 = Blake2b256::digest(b"second");

    let mut h = Blake2b256::new();
    h.update(b"first");
    assert_eq!(h.finalize(), hash1);

    h.reset();
    h.update(b"second");
    assert_eq!(h.finalize(), hash2);
  }

  // ── Keyed hashing ────────────────────────────────────────────────────

  #[test]
  fn blake2b256_keyed_matches_oracle() {
    let key = b"secret-key";
    let data = b"hello world";

    let mut oracle = OracleBlake2bMac256::new_from_slice(key).unwrap();
    hmac::Mac::update(&mut oracle, data);
    let expected: [u8; 32] = hmac::Mac::finalize(oracle).into_bytes().into();

    let actual = Blake2b256::keyed_hash(key, data);
    assert_eq!(actual, expected);
  }

  #[test]
  fn blake2b256_keyed_empty_data() {
    let key = b"key";
    let mut oracle = OracleBlake2bMac256::new_from_slice(key).unwrap();
    hmac::Mac::update(&mut oracle, b"");
    let expected: [u8; 32] = hmac::Mac::finalize(oracle).into_bytes().into();

    let actual = Blake2b256::keyed_hash(key, b"");
    assert_eq!(actual, expected);
  }

  #[test]
  fn blake2b256_keyed_long_data() {
    let key = &[0xAA; 64]; // max key length
    let data = &[0xBB; 512];

    let mut oracle = OracleBlake2bMac256::new_from_slice(key).unwrap();
    hmac::Mac::update(&mut oracle, data);
    let expected: [u8; 32] = hmac::Mac::finalize(oracle).into_bytes().into();

    let actual = Blake2b256::keyed_hash(key, data);
    assert_eq!(actual, expected);
  }

  #[test]
  fn blake2b256_keyed_streaming_reset() {
    let key = b"my-key";
    let tag1 = Blake2b256::keyed_hash(key, b"msg1");
    let tag2 = Blake2b256::keyed_hash(key, b"msg2");

    let mut h = Blake2b256::keyed(key);
    h.update(b"msg1");
    assert_eq!(h.finalize(), tag1);

    h.reset();
    h.update(b"msg2");
    assert_eq!(h.finalize(), tag2);
  }

  #[test]
  fn keyed_differs_from_unkeyed() {
    let hash = Blake2b256::digest(b"hello");
    let tag = Blake2b256::keyed_hash(b"key", b"hello");
    assert_ne!(hash, tag);
  }

  // ── Variable output ───────────────────────────────────────────────────

  #[test]
  fn variable_output_1_byte() {
    let mut out = [0u8; 1];
    blake2b_hash(b"test", 1, &mut out);
    assert_ne!(out, [0u8; 1]);
  }

  #[test]
  fn variable_output_matches_fixed() {
    // 32-byte variable output should match Blake2b256
    let mut var_out = [0u8; 32];
    blake2b_hash(b"hello", 32, &mut var_out);
    let fixed_out = Blake2b256::digest(b"hello");
    assert_eq!(var_out, fixed_out);

    // 64-byte variable output should match Blake2b512
    let mut var_out = [0u8; 64];
    blake2b_hash(b"hello", 64, &mut var_out);
    let fixed_out = Blake2b512::digest(b"hello");
    assert_eq!(var_out, fixed_out);
  }

  // ── Edge cases ────────────────────────────────────────────────────────

  #[test]
  fn empty_input() {
    let expected = oracle_hash_256(b"");
    assert_eq!(Blake2b256::digest(b""), expected);
  }

  #[test]
  fn exactly_one_block() {
    let data = [0u8; 128];
    let expected = oracle_hash_256(&data);
    assert_eq!(Blake2b256::digest(&data), expected);
  }

  #[test]
  fn one_block_plus_one_byte() {
    let data = [0u8; 129];
    let expected = oracle_hash_256(&data);
    assert_eq!(Blake2b256::digest(&data), expected);
  }

  #[test]
  fn two_blocks_exact() {
    let data = [0u8; 256];
    let expected = oracle_hash_512(&data);
    assert_eq!(Blake2b512::digest(&data), expected);
  }

  #[test]
  #[should_panic]
  fn keyed_empty_key_panics() {
    let _ = Blake2b256::keyed(b"");
  }

  #[test]
  #[should_panic]
  fn keyed_overlength_key_panics() {
    let _ = Blake2b256::keyed(&[0u8; 65]);
  }

  // ── Finalize is non-destructive ───────────────────────────────────────

  #[test]
  fn finalize_is_idempotent() {
    let mut h = Blake2b256::new();
    h.update(b"test");
    let hash1 = h.finalize();
    let hash2 = h.finalize();
    assert_eq!(hash1, hash2);
  }

  // ── Forced-kernel oracle tests ────────────────────────────────────────

  use super::kernels::{
    ALL as BLAKE2B_KERNELS, Blake2bKernelId, compress_fn as blake2b_compress_fn,
    required_caps as blake2b_required_caps,
  };

  fn assert_blake2b_kernel(id: Blake2bKernelId) {
    let compress = blake2b_compress_fn(id);
    let cases: &[&[u8]] = &[
      b"",
      b"a",
      b"abc",
      b"hello world",
      &[0u8; 64],
      &[0xFFu8; 128],
      &[0xAAu8; 129],
      &[0xBBu8; 256],
      &[0xCCu8; 1024],
    ];

    // Unkeyed Blake2b-256
    for &data in cases {
      let expected = oracle_hash_256(data);
      let mut h = Blake2b256::new_with_compress_for_test(compress);
      h.update(data);
      assert_eq!(
        h.finalize(),
        expected,
        "blake2b-256 forced mismatch kernel={} len={}",
        id.as_str(),
        data.len(),
      );
    }

    // Unkeyed Blake2b-512
    for &data in cases {
      let expected = oracle_hash_512(data);
      let mut h = Blake2b512::new_with_compress_for_test(compress);
      h.update(data);
      assert_eq!(
        h.finalize(),
        expected,
        "blake2b-512 forced mismatch kernel={} len={}",
        id.as_str(),
        data.len(),
      );
    }

    // Keyed Blake2b-256
    for &key in &[&b"key"[..], &[0xAA; 64]] {
      let mut oracle = OracleBlake2bMac256::new_from_slice(key).unwrap();
      hmac::Mac::update(&mut oracle, b"message");
      let expected: [u8; 32] = hmac::Mac::finalize(oracle).into_bytes().into();

      let mut h = Blake2b256::keyed_with_compress_for_test(key, compress);
      h.update(b"message");
      assert_eq!(
        h.finalize(),
        expected,
        "blake2b-256 keyed forced mismatch kernel={} key_len={}",
        id.as_str(),
        key.len(),
      );
    }

    // Streaming with various chunk sizes
    let data = &[0x42u8; 300];
    let expected = oracle_hash_512(data);
    for chunk_size in [1, 7, 63, 64, 127, 128, 129] {
      let mut h = Blake2b512::new_with_compress_for_test(compress);
      for chunk in data.chunks(chunk_size) {
        h.update(chunk);
      }
      assert_eq!(
        h.finalize(),
        expected,
        "blake2b-512 streaming forced mismatch kernel={} chunk={}",
        id.as_str(),
        chunk_size,
      );
    }
  }

  #[test]
  fn blake2b_forced_kernels_match_oracle() {
    let caps = crate::platform::caps();
    for &id in BLAKE2B_KERNELS {
      if caps.has(blake2b_required_caps(id)) {
        assert_blake2b_kernel(id);
      }
    }
  }
}
