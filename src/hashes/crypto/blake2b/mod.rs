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
//! let tag = Blake2b256::keyed_digest(key, b"message");
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

use core::{fmt, mem::MaybeUninit};

use kernels::IV;

use crate::traits::{Digest, ct};

const BLOCK_SIZE: usize = 128;
const MAX_KEY_LEN: usize = 64;
const MAX_OUTPUT_LEN: usize = 64;

#[cfg(any(test, feature = "diag"))]
#[inline]
#[must_use]
pub(crate) fn kernel_name_for_len(len: usize) -> &'static str {
  dispatch::kernel_name_for_len(len)
}

#[cfg(feature = "diag")]
#[inline]
#[must_use]
pub fn diag_init_state_unkeyed(output_len: u8) -> [u64; 8] {
  init_state(output_len, 0)
}

#[cfg(feature = "diag")]
#[inline]
pub fn diag_compress_block(state: &mut [u64; 8], block: &[u8; BLOCK_SIZE], t: u128, last: bool) {
  compress_direct(state, block, t, last);
}

#[cfg(feature = "diag")]
#[inline]
pub fn diag_compress_block_portable(state: &mut [u64; 8], block: &[u8; BLOCK_SIZE], t: u128, last: bool) {
  kernels::compress(state, block, t, last);
}

#[cfg(all(feature = "diag", target_arch = "aarch64"))]
#[inline]
pub fn diag_compress_block_aarch64_neon(state: &mut [u64; 8], block: &[u8; BLOCK_SIZE], t: u128, last: bool) {
  // SAFETY: NEON is baseline on AArch64.
  unsafe { aarch64::compress_neon(state, block, t, last) }
}

// ─── Core state ─────────────────────────────────────────────────────────────

/// Internal Blake2b state shared by `Blake2b256` and `Blake2b512`.
///
/// Handles any output length (1-64 bytes), optional keyed hashing, and the
/// RFC 7693 §2.5 sequential-mode parameter block (`salt` + `personal`).
#[derive(Clone)]
struct Core {
  h: [u64; 8],
  buf: [u8; BLOCK_SIZE],
  buf_len: u8,
  t: u128,
  nn: u8,
  kk: u8,
  key: MaybeUninit<[u8; MAX_KEY_LEN]>,
  salt: [u8; SALT_LEN],
  personal: [u8; PERSONAL_LEN],
  compress: kernels::CompressFn,
}

impl Core {
  #[inline]
  fn new_unkeyed(nn: u8) -> Self {
    Self {
      h: init_state(nn, 0),
      buf: [0u8; BLOCK_SIZE],
      buf_len: 0,
      t: 0,
      nn,
      kk: 0,
      key: MaybeUninit::uninit(),
      salt: [0u8; SALT_LEN],
      personal: [0u8; PERSONAL_LEN],
      compress: dispatch::compress_dispatch(),
    }
  }

  /// Create a new Blake2b state with output length `nn` and optional `key`.
  ///
  /// # Panics
  ///
  /// Panics if `nn` is 0 or > 64, or if `key` is longer than 64 bytes.
  #[inline]
  fn new(nn: u8, key: &[u8]) -> Self {
    Self::new_with_params(nn, key, &[0u8; SALT_LEN], &[0u8; PERSONAL_LEN])
  }

  /// Create a new Blake2b state with output length `nn`, optional `key`, and
  /// spec-defined `salt` + `personal` parameter-block values (RFC 7693 §2.5).
  ///
  /// # Panics
  ///
  /// Panics if `nn` is 0 or > 64, or if `key` is longer than 64 bytes.
  #[allow(clippy::indexing_slicing)]
  fn new_with_params(nn: u8, key: &[u8], salt: &[u8; SALT_LEN], personal: &[u8; PERSONAL_LEN]) -> Self {
    assert!(
      nn >= 1 && nn as usize <= MAX_OUTPUT_LEN,
      "Blake2b output length must be 1-64"
    );
    assert!(key.len() <= MAX_KEY_LEN, "Blake2b key must be at most 64 bytes");

    let kk = key.len() as u8;
    let h = init_state_with_params(nn, kk, salt, personal);

    let stored_key = if kk > 0 {
      let mut bytes = [0u8; MAX_KEY_LEN];
      bytes[..key.len()].copy_from_slice(key);
      MaybeUninit::new(bytes)
    } else {
      MaybeUninit::uninit()
    };

    let mut buf = [0u8; BLOCK_SIZE];
    let buf_len = if kk > 0 {
      buf[..key.len()].copy_from_slice(key);
      BLOCK_SIZE as u8
    } else {
      0
    };

    Self {
      h,
      buf,
      buf_len,
      t: 0,
      nn,
      kk,
      key: stored_key,
      salt: *salt,
      personal: *personal,
      compress: dispatch::compress_dispatch(),
    }
  }

  #[inline(always)]
  fn zeroize_key_if_any(&mut self) {
    if self.kk > 0 {
      // SAFETY: when `kk > 0`, `self.key` was initialized in `new`.
      unsafe { ct::zeroize_no_fence(&mut *self.key.as_mut_ptr()) };
    }
  }

  #[inline(always)]
  fn wipe(&mut self) {
    for word in self.h.iter_mut() {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    ct::zeroize_no_fence(&mut self.buf);
    self.zeroize_key_if_any();
    // SAFETY: fields are valid, aligned, dereferenceable pointers.
    unsafe {
      core::ptr::write_volatile(&mut self.buf_len, 0);
      core::ptr::write_volatile(&mut self.t, 0);
      core::ptr::write_volatile(&mut self.nn, 0);
      core::ptr::write_volatile(&mut self.kk, 0);
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
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
    let t = self.t.strict_add(self.buf_len as u128);
    if self.buf_len as usize == BLOCK_SIZE {
      (self.compress)(&mut h, &self.buf, t, true);
    } else {
      let mut last_block = [0u8; BLOCK_SIZE];
      last_block[..self.buf_len as usize].copy_from_slice(&self.buf[..self.buf_len as usize]);
      (self.compress)(&mut h, &last_block, t, true);
      ct::zeroize(&mut last_block);
    }

    write_output(&h, self.nn, out);

    // Zeroize local state
    for word in h.iter_mut() {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
  }

  #[cfg(test)]
  fn new_with_compress_for_test(nn: u8, key: &[u8], compress: kernels::CompressFn) -> Self {
    let mut core = Self::new(nn, key);
    core.compress = compress;
    core
  }

  /// Reset to the initial state (including re-buffering the key if keyed and
  /// re-applying any salt/personalization parameter-block bytes).
  #[allow(clippy::indexing_slicing)]
  fn reset(&mut self) {
    self.h = init_state_with_params(self.nn, self.kk, &self.salt, &self.personal);

    if self.kk > 0 {
      let key_len = self.kk as usize;
      self.buf = [0u8; BLOCK_SIZE];
      // SAFETY: when `kk > 0`, `self.key` was initialized in `new`, and `self.buf`
      // has at least `key_len` bytes available.
      unsafe {
        core::ptr::copy_nonoverlapping(self.key.as_ptr().cast::<u8>(), self.buf.as_mut_ptr(), key_len);
      }
      self.buf_len = BLOCK_SIZE as u8;
    } else {
      self.buf = [0u8; BLOCK_SIZE];
      self.buf_len = 0;
    }
    self.t = 0;
  }
}

/// Spec-defined salt size for Blake2b (RFC 7693 §2.5).
const SALT_LEN: usize = 16;
/// Spec-defined personalization size for Blake2b (RFC 7693 §2.5).
const PERSONAL_LEN: usize = 16;

#[inline]
fn init_state(nn: u8, kk: u8) -> [u64; 8] {
  init_state_with_params(nn, kk, &[0u8; SALT_LEN], &[0u8; PERSONAL_LEN])
}

/// Initialize Blake2b chaining state from a sequential-mode parameter block.
///
/// Per RFC 7693 §2.5 the parameter block is XORed into the IV before the first
/// compression. In sequential mode (fanout=1, depth=1, leaf_length=0,
/// node_offset=0, node_depth=0, inner_length=0) only `h[0]` carries the
/// digest length / key length / fanout / depth bits, and `h[4..8]` carry the
/// salt and personalization words.
#[inline]
#[allow(clippy::indexing_slicing)]
fn init_state_with_params(nn: u8, kk: u8, salt: &[u8; SALT_LEN], personal: &[u8; PERSONAL_LEN]) -> [u64; 8] {
  let p0 = nn as u64 | ((kk as u64) << 8) | (1u64 << 16) | (1u64 << 24);
  let mut h = IV;
  h[0] ^= p0;
  // SAFETY: salt and personal are exactly 16 bytes; slice splits are in bounds.
  h[4] ^= u64::from_le_bytes(salt[0..8].try_into().unwrap());
  h[5] ^= u64::from_le_bytes(salt[8..16].try_into().unwrap());
  h[6] ^= u64::from_le_bytes(personal[0..8].try_into().unwrap());
  h[7] ^= u64::from_le_bytes(personal[8..16].try_into().unwrap());
  h
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon", not(target_os = "macos")))]
#[inline(always)]
fn compress_direct(h: &mut [u64; 8], block: &[u8; BLOCK_SIZE], t: u128, last: bool) {
  // SAFETY: NEON is enabled at compile time on this target.
  unsafe { aarch64::compress_neon(h, block, t, last) };
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon", not(target_os = "macos"))))]
#[inline(always)]
fn compress_direct(h: &mut [u64; 8], block: &[u8; BLOCK_SIZE], t: u128, last: bool) {
  let compress = dispatch::compress_dispatch();
  compress(h, block, t, last);
}

#[allow(clippy::indexing_slicing)]
fn write_output(h: &[u64; 8], nn: u8, out: &mut [u8]) {
  let nn = nn as usize;
  let full_words = nn / 8;
  let tail = nn % 8;

  #[cfg(target_endian = "little")]
  if tail == 0 {
    let bytes = full_words.strict_mul(8);
    // SAFETY: `out` is exactly `nn` bytes long and `h` contains at least
    // `full_words` initialized `u64` words. On little-endian targets, the
    // in-memory representation already matches the digest output layout.
    unsafe { core::ptr::copy_nonoverlapping(h.as_ptr().cast::<u8>(), out.as_mut_ptr(), bytes) };
    return;
  }

  for (i, word) in h.iter().enumerate().take(full_words) {
    let bytes = word.to_le_bytes();
    let off = i.strict_mul(8);
    out[off..off.strict_add(8)].copy_from_slice(&bytes);
  }
  if tail > 0 {
    let bytes = h[full_words].to_le_bytes();
    let off = full_words.strict_mul(8);
    out[off..off.strict_add(tail)].copy_from_slice(&bytes[..tail]);
  }
}

#[allow(clippy::indexing_slicing)]
fn oneshot_small_into_with_params(
  nn: u8,
  key: &[u8],
  salt: &[u8; SALT_LEN],
  personal: &[u8; PERSONAL_LEN],
  data: &[u8],
  out: &mut [u8],
) {
  let kk = key.len() as u8;
  let mut h = init_state_with_params(nn, kk, salt, personal);

  if kk == 0 {
    let mut block = [0u8; BLOCK_SIZE];
    block[..data.len()].copy_from_slice(data);
    compress_direct(&mut h, &block, data.len() as u128, true);
    write_output(&h, nn, out);
    return;
  }

  let mut key_block = [0u8; BLOCK_SIZE];
  key_block[..key.len()].copy_from_slice(key);

  if data.is_empty() {
    compress_direct(&mut h, &key_block, BLOCK_SIZE as u128, true);
  } else {
    compress_direct(&mut h, &key_block, BLOCK_SIZE as u128, false);

    let mut data_block = [0u8; BLOCK_SIZE];
    data_block[..data.len()].copy_from_slice(data);
    compress_direct(
      &mut h,
      &data_block,
      (BLOCK_SIZE as u128).strict_add(data.len() as u128),
      true,
    );
    ct::zeroize(&mut data_block);
  }

  write_output(&h, nn, out);
  for word in &mut h {
    // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
    unsafe { core::ptr::write_volatile(word, 0) };
  }
  ct::zeroize(&mut key_block);
}

#[allow(clippy::indexing_slicing)]
fn oneshot_hash_into_with_params(
  nn: u8,
  key: &[u8],
  salt: &[u8; SALT_LEN],
  personal: &[u8; PERSONAL_LEN],
  data: &[u8],
  out: &mut [u8],
) {
  debug_assert!(out.len() == nn as usize);
  assert!(
    nn >= 1 && nn as usize <= MAX_OUTPUT_LEN,
    "Blake2b output length must be 1-64"
  );
  assert!(key.len() <= MAX_KEY_LEN, "Blake2b key must be at most 64 bytes");

  if data.len() <= BLOCK_SIZE {
    oneshot_small_into_with_params(nn, key, salt, personal, data, out);
    return;
  }

  let kk = key.len() as u8;
  let mut h = init_state_with_params(nn, kk, salt, personal);
  let mut buf = [0u8; BLOCK_SIZE];
  let mut buf_len = if kk > 0 {
    buf[..key.len()].copy_from_slice(key);
    BLOCK_SIZE as u8
  } else {
    0
  };
  let mut t = 0u128;
  let mut offset = 0usize;
  let data_len = data.len();

  if buf_len > 0 && (buf_len as usize).strict_add(data_len) > BLOCK_SIZE {
    let fill = BLOCK_SIZE.strict_sub(buf_len as usize);
    if fill > 0 {
      buf[buf_len as usize..BLOCK_SIZE].copy_from_slice(&data[..fill]);
    }
    t = t.strict_add(BLOCK_SIZE as u128);
    compress_direct(&mut h, &buf, t, false);
    ct::zeroize(&mut buf);
    buf_len = 0;
    offset = fill;
  }

  while offset.strict_add(BLOCK_SIZE) < data_len {
    t = t.strict_add(BLOCK_SIZE as u128);
    // SAFETY: offset + BLOCK_SIZE < data_len, so 128 bytes are in bounds.
    let block = unsafe { &*data.as_ptr().add(offset).cast::<[u8; BLOCK_SIZE]>() };
    compress_direct(&mut h, block, t, false);
    offset = offset.strict_add(BLOCK_SIZE);
  }

  let remaining = data_len.strict_sub(offset);
  if remaining > 0 {
    buf[..remaining].copy_from_slice(&data[offset..]);
    buf_len = remaining as u8;
  }

  t = t.strict_add(buf_len as u128);
  compress_direct(&mut h, &buf, t, true);
  write_output(&h, nn, out);

  if kk > 0 {
    for word in &mut h {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    ct::zeroize(&mut buf);
  }
}

#[inline]
fn oneshot_hash_into(nn: u8, key: &[u8], data: &[u8], out: &mut [u8]) {
  oneshot_hash_into_with_params(nn, key, &[0u8; SALT_LEN], &[0u8; PERSONAL_LEN], data, out);
}

#[inline(always)]
fn oneshot_hash_array<const N: usize>(nn: u8, key: &[u8], data: &[u8]) -> [u8; N] {
  let mut out = MaybeUninit::<[u8; N]>::uninit();
  // SAFETY: `oneshot_hash_into` fully initializes exactly `N` output bytes
  // before returning, and the temporary out buffer is valid for mutable access.
  unsafe {
    oneshot_hash_into(nn, key, data, &mut *out.as_mut_ptr());
    out.assume_init()
  }
}

#[inline(always)]
fn oneshot_hash_array_with_params<const N: usize>(
  nn: u8,
  key: &[u8],
  salt: &[u8; SALT_LEN],
  personal: &[u8; PERSONAL_LEN],
  data: &[u8],
) -> [u8; N] {
  let mut out = MaybeUninit::<[u8; N]>::uninit();
  // SAFETY: `oneshot_hash_into_with_params` fully initializes exactly `N`
  // output bytes before returning.
  unsafe {
    oneshot_hash_into_with_params(nn, key, salt, personal, data, &mut *out.as_mut_ptr());
    out.assume_init()
  }
}

#[inline(always)]
fn finalize_array<const N: usize>(core: &Core) -> [u8; N] {
  let mut out = MaybeUninit::<[u8; N]>::uninit();
  // SAFETY: `finalize_into` writes the full fixed-size digest into the output
  // buffer before returning.
  unsafe {
    core.finalize_into(&mut *out.as_mut_ptr());
    out.assume_init()
  }
}

impl Drop for Core {
  fn drop(&mut self) {
    self.wipe();
  }
}

// ─── Blake2bParams ──────────────────────────────────────────────────────────

/// Builder for Blake2b hashers with optional key, salt, and personalization.
///
/// Implements the sequential-mode parameter block from RFC 7693 §2.5. Salt
/// (up to 16 bytes) and personalization (up to 16 bytes) are XORed into the
/// initial chaining value words `h[4..8]`, giving the same hasher with a
/// different domain.
///
/// Empty key produces an unkeyed hasher; non-empty key produces a Blake2b-MAC.
///
/// # Examples
///
/// ```rust
/// use rscrypto::Blake2bParams;
///
/// let tag = Blake2bParams::new()
///   .key(b"my-secret-key")
///   .salt(b"random-salt-1234")
///   .personal(b"app-v1-tagging00")
///   .hash_256(b"message");
/// assert_eq!(tag.len(), 32);
///
/// // Same input + different personalization → different output.
/// let other = Blake2bParams::new()
///   .key(b"my-secret-key")
///   .salt(b"random-salt-1234")
///   .personal(b"app-v2-tagging00")
///   .hash_256(b"message");
/// assert_ne!(tag, other);
/// ```
#[derive(Clone)]
pub struct Blake2bParams {
  key_buf: [u8; MAX_KEY_LEN],
  key_len: u8,
  salt: [u8; SALT_LEN],
  personal: [u8; PERSONAL_LEN],
}

impl Blake2bParams {
  /// Maximum key length (bytes).
  pub const MAX_KEY_LEN: usize = MAX_KEY_LEN;
  /// Maximum salt length (bytes).
  pub const SALT_LEN: usize = SALT_LEN;
  /// Maximum personalization length (bytes).
  pub const PERSONAL_LEN: usize = PERSONAL_LEN;

  /// Create a new params builder with no key, salt, or personalization.
  #[must_use]
  pub const fn new() -> Self {
    Self {
      key_buf: [0u8; MAX_KEY_LEN],
      key_len: 0,
      salt: [0u8; SALT_LEN],
      personal: [0u8; PERSONAL_LEN],
    }
  }

  /// Set the MAC key (0–64 bytes; empty disables keying).
  ///
  /// # Panics
  ///
  /// Panics if `key.len() > 64`.
  #[must_use]
  #[allow(clippy::indexing_slicing)]
  pub fn key(mut self, key: &[u8]) -> Self {
    assert!(key.len() <= MAX_KEY_LEN, "Blake2b key must be at most 64 bytes");
    self.key_buf = [0u8; MAX_KEY_LEN];
    self.key_buf[..key.len()].copy_from_slice(key);
    self.key_len = key.len() as u8;
    self
  }

  /// Set the salt (up to 16 bytes; shorter inputs are zero-padded per spec).
  ///
  /// # Panics
  ///
  /// Panics if `salt.len() > 16`.
  #[must_use]
  #[allow(clippy::indexing_slicing)]
  pub fn salt(mut self, salt: &[u8]) -> Self {
    assert!(salt.len() <= SALT_LEN, "Blake2b salt must be at most 16 bytes");
    self.salt = [0u8; SALT_LEN];
    self.salt[..salt.len()].copy_from_slice(salt);
    self
  }

  /// Set the personalization tag (up to 16 bytes; shorter inputs are zero-padded).
  ///
  /// # Panics
  ///
  /// Panics if `personal.len() > 16`.
  #[must_use]
  #[allow(clippy::indexing_slicing)]
  pub fn personal(mut self, personal: &[u8]) -> Self {
    assert!(
      personal.len() <= PERSONAL_LEN,
      "Blake2b personalization must be at most 16 bytes"
    );
    self.personal = [0u8; PERSONAL_LEN];
    self.personal[..personal.len()].copy_from_slice(personal);
    self
  }

  fn key_slice(&self) -> &[u8] {
    // SAFETY: key_len is set from a slice ≤ MAX_KEY_LEN in `key()`.
    &self.key_buf[..self.key_len as usize]
  }

  /// Build a streaming Blake2b-256 hasher initialized with these parameters.
  #[must_use]
  pub fn build_256(&self) -> Blake2b256 {
    Blake2b256(Core::new_with_params(32, self.key_slice(), &self.salt, &self.personal))
  }

  /// Build a streaming Blake2b-512 hasher initialized with these parameters.
  #[must_use]
  pub fn build_512(&self) -> Blake2b512 {
    Blake2b512(Core::new_with_params(64, self.key_slice(), &self.salt, &self.personal))
  }

  /// Compute a Blake2b-256 hash of `data` in one shot using these parameters.
  #[must_use]
  pub fn hash_256(&self, data: &[u8]) -> [u8; 32] {
    oneshot_hash_array_with_params::<32>(32, self.key_slice(), &self.salt, &self.personal, data)
  }

  /// Compute a Blake2b-512 hash of `data` in one shot using these parameters.
  #[must_use]
  pub fn hash_512(&self, data: &[u8]) -> [u8; 64] {
    oneshot_hash_array_with_params::<64>(64, self.key_slice(), &self.salt, &self.personal, data)
  }
}

impl Default for Blake2bParams {
  fn default() -> Self {
    Self::new()
  }
}

impl fmt::Debug for Blake2bParams {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Blake2bParams")
      .field("key_len", &self.key_len)
      .field("salt", &self.salt)
      .field("personal", &self.personal)
      .finish()
  }
}

impl Drop for Blake2bParams {
  fn drop(&mut self) {
    ct::zeroize(&mut self.key_buf);
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

  /// Compute an unkeyed Blake2b-256 hash in one shot.
  #[must_use]
  pub fn digest(data: &[u8]) -> [u8; 32] {
    oneshot_hash_array::<32>(32, &[], data)
  }

  /// Create a keyed Blake2b-256 streaming hasher (Blake2b-MAC).
  ///
  /// # Panics
  ///
  /// Panics if `key` is empty or longer than 64 bytes.
  #[must_use]
  pub fn new_keyed(key: &[u8]) -> Self {
    assert!(!key.is_empty(), "use Blake2b256::new() for unkeyed hashing");
    Self(Core::new(32, key))
  }

  /// Compute a keyed Blake2b-256 hash in one shot.
  ///
  /// # Panics
  ///
  /// Panics if `key` is empty or longer than 64 bytes.
  #[must_use]
  pub fn keyed_digest(key: &[u8], data: &[u8]) -> [u8; 32] {
    assert!(!key.is_empty(), "use Blake2b256::digest() for unkeyed hashing");
    oneshot_hash_array::<32>(32, key, data)
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
    Self(Core::new_unkeyed(32))
  }
}

impl Digest for Blake2b256 {
  const OUTPUT_SIZE: usize = 32;
  type Output = [u8; 32];

  #[inline]
  fn new() -> Self {
    Self(Core::new_unkeyed(32))
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.0.update(data);
  }

  fn finalize(&self) -> Self::Output {
    finalize_array::<32>(&self.0)
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

  /// Compute an unkeyed Blake2b-512 hash in one shot.
  #[must_use]
  pub fn digest(data: &[u8]) -> [u8; 64] {
    oneshot_hash_array::<64>(64, &[], data)
  }

  /// Create a keyed Blake2b-512 streaming hasher (Blake2b-MAC).
  ///
  /// # Panics
  ///
  /// Panics if `key` is empty or longer than 64 bytes.
  #[must_use]
  pub fn new_keyed(key: &[u8]) -> Self {
    assert!(!key.is_empty(), "use Blake2b512::new() for unkeyed hashing");
    Self(Core::new(64, key))
  }

  /// Compute a keyed Blake2b-512 hash in one shot.
  ///
  /// # Panics
  ///
  /// Panics if `key` is empty or longer than 64 bytes.
  #[must_use]
  pub fn keyed_digest(key: &[u8], data: &[u8]) -> [u8; 64] {
    assert!(!key.is_empty(), "use Blake2b512::digest() for unkeyed hashing");
    oneshot_hash_array::<64>(64, key, data)
  }
}

impl Blake2b512 {
  #[cfg(test)]
  pub(crate) fn new_with_compress_for_test(compress: kernels::CompressFn) -> Self {
    Self(Core::new_with_compress_for_test(64, &[], compress))
  }
}

impl Default for Blake2b512 {
  fn default() -> Self {
    Self(Core::new_unkeyed(64))
  }
}

impl Digest for Blake2b512 {
  const OUTPUT_SIZE: usize = 64;
  type Output = [u8; 64];

  #[inline]
  fn new() -> Self {
    Self(Core::new_unkeyed(64))
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.0.update(data);
  }

  fn finalize(&self) -> Self::Output {
    finalize_array::<64>(&self.0)
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
#[cfg(test)]
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
  use digest::{
    KeyInit,
    consts::{U32, U64},
  };

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

    let actual = Blake2b256::keyed_digest(key, data);
    assert_eq!(actual, expected);
  }

  #[test]
  fn blake2b256_keyed_empty_data() {
    let key = b"key";
    let mut oracle = OracleBlake2bMac256::new_from_slice(key).unwrap();
    hmac::Mac::update(&mut oracle, b"");
    let expected: [u8; 32] = hmac::Mac::finalize(oracle).into_bytes().into();

    let actual = Blake2b256::keyed_digest(key, b"");
    assert_eq!(actual, expected);
  }

  #[test]
  fn blake2b256_keyed_long_data() {
    let key = &[0xAA; 64]; // max key length
    let data = &[0xBB; 512];

    let mut oracle = OracleBlake2bMac256::new_from_slice(key).unwrap();
    hmac::Mac::update(&mut oracle, data);
    let expected: [u8; 32] = hmac::Mac::finalize(oracle).into_bytes().into();

    let actual = Blake2b256::keyed_digest(key, data);
    assert_eq!(actual, expected);
  }

  #[test]
  fn blake2b256_keyed_streaming_reset() {
    let key = b"my-key";
    let tag1 = Blake2b256::keyed_digest(key, b"msg1");
    let tag2 = Blake2b256::keyed_digest(key, b"msg2");

    let mut h = Blake2b256::new_keyed(key);
    h.update(b"msg1");
    assert_eq!(h.finalize(), tag1);

    h.reset();
    h.update(b"msg2");
    assert_eq!(h.finalize(), tag2);
  }

  #[test]
  fn keyed_differs_from_unkeyed() {
    let hash = Blake2b256::digest(b"hello");
    let tag = Blake2b256::keyed_digest(b"key", b"hello");
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
    let _ = Blake2b256::new_keyed(b"");
  }

  #[test]
  #[should_panic]
  fn keyed_overlength_key_panics() {
    let _ = Blake2b256::new_keyed(&[0u8; 65]);
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

  // ── Params (salt + personalization) ───────────────────────────────────

  #[test]
  fn params_default_matches_plain_digest() {
    // Empty salt + empty personal + empty key should equal Blake2b256::digest.
    let plain = Blake2b256::digest(b"hello");
    let via_params = Blake2bParams::new().hash_256(b"hello");
    assert_eq!(plain, via_params);
  }

  #[test]
  fn params_default_matches_plain_digest_512() {
    let plain = Blake2b512::digest(b"hello");
    let via_params = Blake2bParams::new().hash_512(b"hello");
    assert_eq!(plain, via_params);
  }

  #[test]
  fn params_key_matches_keyed_digest() {
    // Empty salt + empty personal + key should equal keyed_digest.
    let key = b"secret-key";
    let plain = Blake2b256::keyed_digest(key, b"hello");
    let via_params = Blake2bParams::new().key(key).hash_256(b"hello");
    assert_eq!(plain, via_params);
  }

  #[test]
  fn params_salt_changes_output() {
    let a = Blake2bParams::new().salt(b"salt-a").hash_256(b"msg");
    let b = Blake2bParams::new().salt(b"salt-b").hash_256(b"msg");
    let plain = Blake2b256::digest(b"msg");
    assert_ne!(a, b);
    assert_ne!(a, plain);
    assert_ne!(b, plain);
  }

  #[test]
  fn params_personal_changes_output() {
    let a = Blake2bParams::new().personal(b"ctx-a").hash_256(b"msg");
    let b = Blake2bParams::new().personal(b"ctx-b").hash_256(b"msg");
    let plain = Blake2b256::digest(b"msg");
    assert_ne!(a, b);
    assert_ne!(a, plain);
    assert_ne!(b, plain);
  }

  #[test]
  fn params_salt_and_personal_are_independent() {
    // Swapping which field carries the same bytes must change the output,
    // proving salt and personal XOR into different IV words.
    let both_a = Blake2bParams::new()
      .salt(b"AAAAAAAAAAAAAAAA")
      .personal(b"BBBBBBBBBBBBBBBB")
      .hash_256(b"msg");
    let swapped = Blake2bParams::new()
      .salt(b"BBBBBBBBBBBBBBBB")
      .personal(b"AAAAAAAAAAAAAAAA")
      .hash_256(b"msg");
    assert_ne!(both_a, swapped);
  }

  #[test]
  fn params_stable_under_repeat() {
    let a = Blake2bParams::new()
      .key(b"k")
      .salt(b"s")
      .personal(b"p")
      .hash_256(b"data");
    let b = Blake2bParams::new()
      .key(b"k")
      .salt(b"s")
      .personal(b"p")
      .hash_256(b"data");
    assert_eq!(a, b);
  }

  #[test]
  fn params_streaming_matches_oneshot() {
    let params = Blake2bParams::new().key(b"k").salt(b"s").personal(b"p");
    let oneshot = params.hash_256(b"hello world");

    let mut h = params.build_256();
    h.update(b"hello ");
    h.update(b"world");
    let stream = h.finalize();

    assert_eq!(oneshot, stream);
  }

  #[test]
  fn params_short_salt_is_zero_padded() {
    // Short salt zero-padded should match explicit zero-padded 16-byte salt.
    let short = Blake2bParams::new().salt(b"abc").hash_256(b"msg");
    let mut padded = [0u8; 16];
    padded[..3].copy_from_slice(b"abc");
    let full = Blake2bParams::new().salt(&padded).hash_256(b"msg");
    assert_eq!(short, full);
  }

  #[test]
  fn params_reset_preserves_salt_and_personal() {
    let params = Blake2bParams::new().salt(b"salt").personal(b"personal");
    let mut h = params.build_256();
    h.update(b"first");
    let _ = h.finalize();

    h.reset();
    h.update(b"hello world");
    let after_reset = h.finalize();

    let expected = params.hash_256(b"hello world");
    assert_eq!(after_reset, expected);
  }

  #[test]
  #[should_panic]
  fn params_overlong_salt_panics() {
    let _ = Blake2bParams::new().salt(&[0u8; 17]);
  }

  #[test]
  #[should_panic]
  fn params_overlong_personal_panics() {
    let _ = Blake2bParams::new().personal(&[0u8; 17]);
  }

  #[test]
  #[should_panic]
  fn params_overlong_key_panics() {
    let _ = Blake2bParams::new().key(&[0u8; 65]);
  }

  // ── Forced-kernel oracle tests ────────────────────────────────────────

  use super::kernels::{
    ALL as BLAKE2B_KERNELS, Blake2bKernelId, compress_fn as blake2b_compress_fn, required_caps as blake2b_required_caps,
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
