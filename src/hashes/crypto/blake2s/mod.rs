//! Blake2s cryptographic hash (RFC 7693).
//!
//! Blake2s-128 and Blake2s-256, with optional keyed hashing (Blake2s-MAC).
//! Blake2s uses 32-bit words (same rotation constants as ChaCha20) and is
//! optimized for 32-bit and embedded platforms.
//!
//! # Examples
//!
//! ```rust
//! use rscrypto::{Blake2s256, Digest};
//!
//! let hash = Blake2s256::digest(b"hello world");
//! assert_eq!(hash.len(), 32);
//!
//! let mut hasher = Blake2s256::new();
//! hasher.update(b"hello ");
//! hasher.update(b"world");
//! let hash = hasher.finalize();
//! assert_eq!(hash, Blake2s256::digest(b"hello world"));
//! ```
//!
//! ## Keyed Hashing
//!
//! ```rust
//! use rscrypto::{Blake2s256, Digest};
//!
//! let key = b"secret-key-up-to-32-bytes";
//! let tag = Blake2s256::keyed_hash(key, b"message");
//! assert_ne!(tag, Blake2s256::digest(b"message"));
//! ```

pub(crate) mod kernels;

#[cfg(target_arch = "aarch64")]
mod aarch64;
mod dispatch;

use core::fmt;

use kernels::IV;

use crate::traits::{Digest, ct};

const BLOCK_SIZE: usize = 64;
const MAX_KEY_LEN: usize = 32;
const MAX_OUTPUT_LEN: usize = 32;

// ─── Core state ─────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Core {
  h: [u32; 8],
  buf: [u8; BLOCK_SIZE],
  buf_len: u8,
  t: u64,
  nn: u8,
  kk: u8,
  key: [u8; MAX_KEY_LEN],
  compress: kernels::CompressFn,
}

impl Core {
  #[allow(clippy::indexing_slicing)]
  fn new(nn: u8, key: &[u8]) -> Self {
    assert!(
      nn >= 1 && nn as usize <= MAX_OUTPUT_LEN,
      "Blake2s output length must be 1-32"
    );
    assert!(key.len() <= MAX_KEY_LEN, "Blake2s key must be at most 32 bytes");

    let kk = key.len() as u8;

    // Parameter block: fanout=1, depth=1
    let p0 = nn as u32 | ((kk as u32) << 8) | (1u32 << 16) | (1u32 << 24);
    let mut h = IV;
    h[0] ^= p0;

    let mut stored_key = [0u8; MAX_KEY_LEN];
    stored_key[..key.len()].copy_from_slice(key);

    let mut buf = [0u8; BLOCK_SIZE];
    let buf_len;

    if kk > 0 {
      buf[..key.len()].copy_from_slice(key);
      buf_len = BLOCK_SIZE as u8;
    } else {
      buf_len = 0;
    }

    Self {
      h,
      buf,
      buf_len,
      t: 0,
      nn,
      kk,
      key: stored_key,
      compress: dispatch::compress_dispatch(),
    }
  }

  #[cfg(test)]
  fn new_with_compress_for_test(nn: u8, key: &[u8], compress: kernels::CompressFn) -> Self {
    let mut core = Self::new(nn, key);
    core.compress = compress;
    core
  }

  #[allow(clippy::indexing_slicing)]
  fn update(&mut self, data: &[u8]) {
    if data.is_empty() {
      return;
    }

    let mut offset = 0usize;
    let data_len = data.len();

    if self.buf_len > 0 && (self.buf_len as usize).strict_add(data_len) > BLOCK_SIZE {
      let fill = BLOCK_SIZE.strict_sub(self.buf_len as usize);
      self.buf[self.buf_len as usize..BLOCK_SIZE].copy_from_slice(&data[..fill]);
      self.t = self.t.strict_add(BLOCK_SIZE as u64);
      (self.compress)(&mut self.h, &self.buf, self.t, false);
      self.buf_len = 0;
      offset = fill;
    }

    while offset.strict_add(BLOCK_SIZE) < data_len {
      self.t = self.t.strict_add(BLOCK_SIZE as u64);
      // SAFETY: offset + BLOCK_SIZE < data_len, so 64 bytes are in bounds.
      let block = unsafe { &*data.as_ptr().add(offset).cast::<[u8; BLOCK_SIZE]>() };
      (self.compress)(&mut self.h, block, self.t, false);
      offset = offset.strict_add(BLOCK_SIZE);
    }

    let remaining = data_len.strict_sub(offset);
    if remaining > 0 {
      let start = self.buf_len as usize;
      self.buf[start..start.strict_add(remaining)].copy_from_slice(&data[offset..]);
      self.buf_len = self.buf_len.strict_add(remaining as u8);
    }
  }

  #[allow(clippy::indexing_slicing)]
  fn finalize_into(&self, out: &mut [u8]) {
    debug_assert!(out.len() == self.nn as usize);

    let mut h = self.h;
    let mut last_block = [0u8; BLOCK_SIZE];
    last_block[..self.buf_len as usize].copy_from_slice(&self.buf[..self.buf_len as usize]);

    let t = self.t.strict_add(self.buf_len as u64);
    (self.compress)(&mut h, &last_block, t, true);

    let nn = self.nn as usize;
    let full_words = nn / 4;
    for i in 0..full_words {
      let bytes = h[i].to_le_bytes();
      let off = i.strict_mul(4);
      out[off..off.strict_add(4)].copy_from_slice(&bytes);
    }
    let tail = nn % 4;
    if tail > 0 {
      let bytes = h[full_words].to_le_bytes();
      let off = full_words.strict_mul(4);
      out[off..off.strict_add(tail)].copy_from_slice(&bytes[..tail]);
    }

    for word in h.iter_mut() {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    ct::zeroize(&mut last_block);
  }

  #[allow(clippy::indexing_slicing)]
  fn reset(&mut self) {
    let p0 = self.nn as u32 | ((self.kk as u32) << 8) | (1u32 << 16) | (1u32 << 24);
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
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    ct::zeroize_no_fence(&mut self.buf);
    ct::zeroize_no_fence(&mut self.key);
    unsafe {
      core::ptr::write_volatile(&mut self.buf_len, 0);
      core::ptr::write_volatile(&mut self.t, 0);
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

// ─── Blake2s256 ─────────────────────────────────────────────────────────────

/// Blake2s-256 cryptographic hash (32-byte output).
///
/// # Examples
///
/// ```rust
/// use rscrypto::{Blake2s256, Digest};
///
/// let hash = Blake2s256::digest(b"hello world");
/// assert_eq!(hash.len(), 32);
///
/// let mut h = Blake2s256::new();
/// h.update(b"hello ");
/// h.update(b"world");
/// assert_eq!(h.finalize(), hash);
/// ```
#[derive(Clone)]
pub struct Blake2s256(Core);

impl fmt::Debug for Blake2s256 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Blake2s256").finish_non_exhaustive()
  }
}

impl Blake2s256 {
  /// Output size in bytes.
  pub const OUTPUT_SIZE: usize = 32;

  /// Create a keyed Blake2s-256 instance.
  ///
  /// # Panics
  ///
  /// Panics if `key` is empty or longer than 32 bytes.
  #[must_use]
  pub fn keyed(key: &[u8]) -> Self {
    assert!(!key.is_empty(), "use Blake2s256::new() for unkeyed hashing");
    Self(Core::new(32, key))
  }

  /// Compute a keyed Blake2s-256 hash in one shot.
  ///
  /// # Panics
  ///
  /// Panics if `key` is empty or longer than 32 bytes.
  #[must_use]
  pub fn keyed_hash(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut h = Self::keyed(key);
    h.update(data);
    h.finalize()
  }

  #[cfg(test)]
  pub(crate) fn new_with_compress_for_test(compress: kernels::CompressFn) -> Self {
    Self(Core::new_with_compress_for_test(32, &[], compress))
  }

  #[cfg(test)]
  pub(crate) fn keyed_with_compress_for_test(key: &[u8], compress: kernels::CompressFn) -> Self {
    assert!(!key.is_empty());
    Self(Core::new_with_compress_for_test(32, key, compress))
  }
}

impl Default for Blake2s256 {
  fn default() -> Self {
    Self(Core::new(32, &[]))
  }
}

impl Digest for Blake2s256 {
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

impl Drop for Blake2s256 {
  fn drop(&mut self) {
    // Core::drop handles zeroization
  }
}

// ─── Blake2s128 ─────────────────────────────────────────────────────────────

/// Blake2s-128 cryptographic hash (16-byte output).
///
/// # Examples
///
/// ```rust
/// use rscrypto::{Blake2s128, Digest};
///
/// let hash = Blake2s128::digest(b"hello world");
/// assert_eq!(hash.len(), 16);
/// ```
#[derive(Clone)]
pub struct Blake2s128(Core);

impl fmt::Debug for Blake2s128 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Blake2s128").finish_non_exhaustive()
  }
}

impl Blake2s128 {
  /// Output size in bytes.
  pub const OUTPUT_SIZE: usize = 16;

  /// Create a keyed Blake2s-128 instance.
  ///
  /// # Panics
  ///
  /// Panics if `key` is empty or longer than 32 bytes.
  #[must_use]
  pub fn keyed(key: &[u8]) -> Self {
    assert!(!key.is_empty(), "use Blake2s128::new() for unkeyed hashing");
    Self(Core::new(16, key))
  }

  /// Compute a keyed Blake2s-128 hash in one shot.
  #[must_use]
  pub fn keyed_hash(key: &[u8], data: &[u8]) -> [u8; 16] {
    let mut h = Self::keyed(key);
    h.update(data);
    h.finalize()
  }
}

impl Default for Blake2s128 {
  fn default() -> Self {
    Self(Core::new(16, &[]))
  }
}

impl Digest for Blake2s128 {
  const OUTPUT_SIZE: usize = 16;
  type Output = [u8; 16];

  #[inline]
  fn new() -> Self {
    Self(Core::new(16, &[]))
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.0.update(data);
  }

  fn finalize(&self) -> Self::Output {
    let mut out = [0u8; 16];
    self.0.finalize_into(&mut out);
    out
  }

  #[inline]
  fn reset(&mut self) {
    self.0.reset();
  }
}

impl Drop for Blake2s128 {
  fn drop(&mut self) {
    // Core::drop handles zeroization
  }
}

#[cfg(test)]
mod tests {
  use blake2::{Blake2s256 as OracleBlake2s256, Digest as _};

  use super::{
    kernels::{
      ALL as BLAKE2S_KERNELS, Blake2sKernelId, compress_fn as blake2s_compress_fn,
      required_caps as blake2s_required_caps,
    },
    *,
  };

  fn oracle_hash_256(data: &[u8]) -> [u8; 32] {
    let mut h = OracleBlake2s256::new();
    h.update(data);
    let result = h.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
  }

  // ── Unkeyed oracle tests ────────────────────────────────────────────

  #[test]
  fn blake2s256_matches_oracle() {
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
      let actual = Blake2s256::digest(data);
      assert_eq!(actual, expected, "Blake2s256 mismatch for len={}", data.len());
    }
  }

  // ── Streaming ─────────────────────────────────────────────────────────

  #[test]
  fn blake2s256_streaming_matches_oneshot() {
    let data = b"The quick brown fox jumps over the lazy dog";
    let oneshot = Blake2s256::digest(data);

    let mut h = Blake2s256::new();
    for byte in data.iter() {
      h.update(core::slice::from_ref(byte));
    }
    assert_eq!(h.finalize(), oneshot);
  }

  // ── Reset ─────────────────────────────────────────────────────────────

  #[test]
  fn blake2s256_reset() {
    let hash1 = Blake2s256::digest(b"first");
    let hash2 = Blake2s256::digest(b"second");

    let mut h = Blake2s256::new();
    h.update(b"first");
    assert_eq!(h.finalize(), hash1);

    h.reset();
    h.update(b"second");
    assert_eq!(h.finalize(), hash2);
  }

  // ── Keyed hashing ────────────────────────────────────────────────────

  #[test]
  fn keyed_differs_from_unkeyed() {
    let hash = Blake2s256::digest(b"hello");
    let tag = Blake2s256::keyed_hash(b"key", b"hello");
    assert_ne!(hash, tag);
  }

  // ── Edge cases ────────────────────────────────────────────────────────

  #[test]
  fn empty_input() {
    let expected = oracle_hash_256(b"");
    assert_eq!(Blake2s256::digest(b""), expected);
  }

  #[test]
  fn exactly_one_block() {
    let data = [0u8; 64];
    let expected = oracle_hash_256(&data);
    assert_eq!(Blake2s256::digest(&data), expected);
  }

  #[test]
  fn one_block_plus_one_byte() {
    let data = [0u8; 65];
    let expected = oracle_hash_256(&data);
    assert_eq!(Blake2s256::digest(&data), expected);
  }

  #[test]
  fn finalize_is_idempotent() {
    let mut h = Blake2s256::new();
    h.update(b"test");
    let hash1 = h.finalize();
    let hash2 = h.finalize();
    assert_eq!(hash1, hash2);
  }

  #[test]
  #[should_panic]
  fn keyed_empty_key_panics() {
    let _ = Blake2s256::keyed(b"");
  }

  #[test]
  #[should_panic]
  fn keyed_overlength_key_panics() {
    let _ = Blake2s256::keyed(&[0u8; 33]);
  }

  // ── Forced-kernel oracle tests ────────────────────────────────────────

  fn assert_blake2s_kernel(id: Blake2sKernelId) {
    let compress = blake2s_compress_fn(id);
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
      let mut h = Blake2s256::new_with_compress_for_test(compress);
      h.update(data);
      assert_eq!(
        h.finalize(),
        expected,
        "blake2s-256 forced mismatch kernel={} len={}",
        id.as_str(),
        data.len(),
      );
    }

    // Streaming
    let data = &[0x42u8; 300];
    let expected = oracle_hash_256(data);
    for chunk_size in [1, 7, 63, 64, 127, 128, 129] {
      let mut h = Blake2s256::new_with_compress_for_test(compress);
      for chunk in data.chunks(chunk_size) {
        h.update(chunk);
      }
      assert_eq!(
        h.finalize(),
        expected,
        "blake2s-256 streaming forced mismatch kernel={} chunk={}",
        id.as_str(),
        chunk_size,
      );
    }
  }

  #[test]
  fn blake2s_forced_kernels_match_oracle() {
    let caps = crate::platform::caps();
    for &id in BLAKE2S_KERNELS {
      if caps.has(blake2s_required_caps(id)) {
        assert_blake2s_kernel(id);
      }
    }
  }
}
