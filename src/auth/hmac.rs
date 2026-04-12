//! HMAC-SHA2 family (RFC 2104, FIPS 198-1).

use crate::{
  hashes::crypto::{
    Sha256, Sha384, Sha512,
    dispatch_util::len_hint_from_u64,
    sha256::{H0 as SHA256_H0, Sha256Prefix, dispatch as sha256_dispatch},
    sha384::{H0 as SHA384_H0, Sha384Prefix, dispatch as sha384_dispatch},
    sha512::{H0 as SHA512_H0, Sha512Prefix, dispatch as sha512_dispatch},
  },
  traits::{Digest, Mac, VerificationError, ct},
};

const SHA256_BLOCK_SIZE: usize = 64;
const SHA256_TAG_SIZE: usize = 32;
const SHA512_FAMILY_BLOCK_SIZE: usize = 128;
const SHA384_TAG_SIZE: usize = 48;
const SHA512_TAG_SIZE: usize = 64;

/// HMAC-SHA256 authentication state.
///
/// # Examples
///
/// ```rust
/// use rscrypto::{HmacSha256, Mac};
///
/// let key = b"shared-secret";
/// let data = b"auth message";
///
/// let tag = HmacSha256::mac(key, data);
///
/// let mut mac = HmacSha256::new(key);
/// mac.update(b"auth ");
/// mac.update(b"message");
/// assert_eq!(mac.finalize(), tag);
/// assert!(mac.verify(&tag).is_ok());
/// ```
#[derive(Clone)]
pub struct HmacSha256 {
  inner: Sha256,
  inner_init: Sha256Prefix,
  outer_init: Sha256Prefix,
}

impl core::fmt::Debug for HmacSha256 {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("HmacSha256").finish_non_exhaustive()
  }
}

impl HmacSha256 {
  /// HMAC-SHA256 block size in bytes.
  pub const BLOCK_SIZE: usize = SHA256_BLOCK_SIZE;

  /// HMAC-SHA256 tag size in bytes.
  pub const TAG_SIZE: usize = SHA256_TAG_SIZE;

  /// Compute the HMAC tag of `data` in one shot.
  #[inline]
  #[must_use]
  pub fn mac(key: &[u8], data: &[u8]) -> [u8; SHA256_TAG_SIZE] {
    <Self as Mac>::mac(key, data)
  }

  /// Verify `expected` against the HMAC tag of `data` in constant time.
  #[inline]
  pub fn verify_tag(key: &[u8], data: &[u8], expected: &[u8; SHA256_TAG_SIZE]) -> Result<(), VerificationError> {
    <Self as Mac>::verify_tag(key, data, expected)
  }
}

impl Mac for HmacSha256 {
  const TAG_SIZE: usize = SHA256_TAG_SIZE;
  type Tag = [u8; SHA256_TAG_SIZE];

  fn new(key: &[u8]) -> Self {
    let mut key_block = [0u8; SHA256_BLOCK_SIZE];
    if key.len() > SHA256_BLOCK_SIZE {
      let digest = Sha256::digest(key);
      for (dst, src) in key_block.iter_mut().zip(digest.iter()) {
        *dst = *src;
      }
    } else {
      for (dst, src) in key_block.iter_mut().zip(key.iter()) {
        *dst = *src;
      }
    }

    let mut ipad = [0x36u8; SHA256_BLOCK_SIZE];
    let mut opad = [0x5Cu8; SHA256_BLOCK_SIZE];
    for ((ipad_byte, opad_byte), key_byte) in ipad.iter_mut().zip(opad.iter_mut()).zip(key_block.iter().copied()) {
      *ipad_byte ^= key_byte;
      *opad_byte ^= key_byte;
    }

    let mut inner_init = Sha256::new();
    inner_init.update(&ipad);

    let mut outer_init = Sha256::new();
    outer_init.update(&opad);

    let inner_init_prefix = inner_init.aligned_prefix();
    let outer_init_prefix = outer_init.aligned_prefix();

    ct::zeroize(&mut key_block);
    ct::zeroize(&mut ipad);
    ct::zeroize(&mut opad);

    Self {
      inner: inner_init,
      inner_init: inner_init_prefix,
      outer_init: outer_init_prefix,
    }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.inner.update(data);
  }

  #[inline]
  fn finalize(&self) -> Self::Tag {
    let inner_hash = self.inner.finalize();
    let mut outer = Sha256::from_aligned_prefix(self.outer_init);
    outer.update(&inner_hash);
    outer.finalize()
  }

  #[inline]
  fn reset(&mut self) {
    self.inner.reset_to_aligned_prefix(self.inner_init);
  }

  /// Oneshot HMAC-SHA256: merges compress calls for small inputs and batches
  /// zeroization under a single compiler fence.
  ///
  /// For data <= 256 B the entire padded inner message is built on the stack
  /// and compressed in one call, eliminating per-call overhead (function-pointer
  /// dispatch, state save/restore) that dominates on fast SHA2-CE cores.
  /// The outer hash is always merged into a single 128-byte (2-block) call.
  #[inline]
  #[allow(clippy::indexing_slicing)] // All indices bounded by prior length checks + fixed-size arrays.
  fn mac(key: &[u8], data: &[u8]) -> Self::Tag {
    let mut ipad = [0x36u8; SHA256_BLOCK_SIZE];
    if key.len() > SHA256_BLOCK_SIZE {
      let digest = Sha256::digest(key);
      for (ip, &kb) in ipad[..SHA256_TAG_SIZE].iter_mut().zip(digest.iter()) {
        *ip = kb ^ 0x36;
      }
    } else {
      for (ip, &kb) in ipad[..key.len()].iter_mut().zip(key.iter()) {
        *ip = kb ^ 0x36;
      }
    }

    let total_inner = (SHA256_BLOCK_SIZE as u64).strict_add(data.len() as u64);
    let compress = sha256_dispatch::compress_dispatch().select(len_hint_from_u64(total_inner));
    let total_inner_bits = total_inner.strict_mul(8);

    let mut state = SHA256_H0;

    const INLINE_DATA_MAX: usize = 256;
    const INNER_BUF_LEN: usize = 384;
    let mut inner_buf = [0u8; INNER_BUF_LEN];
    let inner_used: usize;

    if data.len() <= INLINE_DATA_MAX {
      inner_buf[..SHA256_BLOCK_SIZE].copy_from_slice(&ipad);
      let data_end = SHA256_BLOCK_SIZE.strict_add(data.len());
      inner_buf[SHA256_BLOCK_SIZE..data_end].copy_from_slice(data);
      inner_buf[data_end] = 0x80;
      let padded = data_end.strict_add(9).strict_add(63).strict_div(64).strict_mul(64);
      inner_buf[padded.strict_sub(8)..padded].copy_from_slice(&total_inner_bits.to_be_bytes());
      compress(&mut state, &inner_buf[..padded]);
      inner_used = padded;
    } else {
      compress(&mut state, &ipad);

      let full_len = data.len().strict_sub(data.len() % SHA256_BLOCK_SIZE);
      if full_len != 0 {
        compress(&mut state, &data[..full_len]);
      }
      let rest = &data[full_len..];

      inner_buf[..rest.len()].copy_from_slice(rest);
      inner_buf[rest.len()] = 0x80;
      if rest.len() >= 56 {
        compress(&mut state, &inner_buf[..SHA256_BLOCK_SIZE]);
        inner_buf[..SHA256_BLOCK_SIZE].fill(0);
      }
      inner_buf[56..SHA256_BLOCK_SIZE].copy_from_slice(&total_inner_bits.to_be_bytes());
      compress(&mut state, &inner_buf[..SHA256_BLOCK_SIZE]);
      inner_used = SHA256_BLOCK_SIZE;
    }

    let mut outer = [0u8; 128];
    for (o, &ip) in outer[..SHA256_BLOCK_SIZE].iter_mut().zip(ipad.iter()) {
      *o = ip ^ 0x6a;
    }
    for (i, &word) in state.iter().enumerate() {
      let off = SHA256_BLOCK_SIZE.strict_add(i.strict_mul(4));
      outer[off..off.strict_add(4)].copy_from_slice(&word.to_be_bytes());
    }
    outer[SHA256_BLOCK_SIZE.strict_add(SHA256_TAG_SIZE)] = 0x80;
    outer[120..128].copy_from_slice(&768u64.to_be_bytes());

    state = SHA256_H0;
    compress(&mut state, &outer);

    let mut tag = [0u8; SHA256_TAG_SIZE];
    for (chunk, &word) in tag.chunks_exact_mut(4).zip(state.iter()) {
      chunk.copy_from_slice(&word.to_be_bytes());
    }

    ct::zeroize_no_fence(&mut ipad);
    ct::zeroize_no_fence(&mut inner_buf[..inner_used]);
    ct::zeroize_no_fence(&mut outer);
    for word in state.iter_mut() {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);

    tag
  }

  #[inline]
  fn verify(&self, expected: &Self::Tag) -> Result<(), VerificationError> {
    if ct::constant_time_eq(&self.finalize(), expected) {
      Ok(())
    } else {
      Err(VerificationError::new())
    }
  }
}

/// HMAC-SHA384 authentication state.
#[derive(Clone)]
pub struct HmacSha384 {
  inner: Sha384,
  inner_init: Sha384Prefix,
  outer_init: Sha384Prefix,
}

impl core::fmt::Debug for HmacSha384 {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("HmacSha384").finish_non_exhaustive()
  }
}

impl HmacSha384 {
  /// HMAC-SHA384 block size in bytes.
  pub const BLOCK_SIZE: usize = SHA512_FAMILY_BLOCK_SIZE;

  /// HMAC-SHA384 tag size in bytes.
  pub const TAG_SIZE: usize = SHA384_TAG_SIZE;

  /// Compute the HMAC tag of `data` in one shot.
  #[inline]
  #[must_use]
  pub fn mac(key: &[u8], data: &[u8]) -> [u8; SHA384_TAG_SIZE] {
    <Self as Mac>::mac(key, data)
  }

  /// Verify `expected` against the HMAC tag of `data` in constant time.
  #[inline]
  pub fn verify_tag(key: &[u8], data: &[u8], expected: &[u8; SHA384_TAG_SIZE]) -> Result<(), VerificationError> {
    <Self as Mac>::verify_tag(key, data, expected)
  }

  #[cfg(test)]
  pub(crate) fn new_with_compress_for_test(
    key: &[u8],
    compress: crate::hashes::crypto::sha384::kernels::CompressBlocksFn,
  ) -> Self {
    let mut key_block = [0u8; SHA512_FAMILY_BLOCK_SIZE];
    if key.len() > SHA512_FAMILY_BLOCK_SIZE {
      let digest = Sha384::digest(key);
      key_block[..SHA384_TAG_SIZE].copy_from_slice(&digest);
    } else {
      key_block[..key.len()].copy_from_slice(key);
    }

    let mut ipad = [0x36u8; SHA512_FAMILY_BLOCK_SIZE];
    let mut opad = [0x5Cu8; SHA512_FAMILY_BLOCK_SIZE];
    for ((ipad_byte, opad_byte), key_byte) in ipad.iter_mut().zip(opad.iter_mut()).zip(key_block.iter().copied()) {
      *ipad_byte ^= key_byte;
      *opad_byte ^= key_byte;
    }

    let mut inner = Sha384::new_with_compress_for_test(compress);
    inner.update(&ipad);
    let inner_init = inner.aligned_prefix();

    let mut outer = Sha384::new_with_compress_for_test(compress);
    outer.update(&opad);
    let outer_init = outer.aligned_prefix();

    ct::zeroize(&mut key_block);
    ct::zeroize(&mut ipad);
    ct::zeroize(&mut opad);

    Self {
      inner,
      inner_init,
      outer_init,
    }
  }

  #[cfg(test)]
  pub(crate) fn mac_with_compress_for_test(
    key: &[u8],
    data: &[u8],
    compress: crate::hashes::crypto::sha384::kernels::CompressBlocksFn,
  ) -> [u8; SHA384_TAG_SIZE] {
    let mut mac = Self::new_with_compress_for_test(key, compress);
    mac.update(data);
    mac.finalize()
  }
}

impl Mac for HmacSha384 {
  const TAG_SIZE: usize = SHA384_TAG_SIZE;
  type Tag = [u8; SHA384_TAG_SIZE];

  fn new(key: &[u8]) -> Self {
    let mut key_block = [0u8; SHA512_FAMILY_BLOCK_SIZE];
    if key.len() > SHA512_FAMILY_BLOCK_SIZE {
      let digest = Sha384::digest(key);
      for (dst, src) in key_block.iter_mut().zip(digest.iter()) {
        *dst = *src;
      }
    } else {
      for (dst, src) in key_block.iter_mut().zip(key.iter()) {
        *dst = *src;
      }
    }

    let mut ipad = [0x36u8; SHA512_FAMILY_BLOCK_SIZE];
    let mut opad = [0x5Cu8; SHA512_FAMILY_BLOCK_SIZE];
    for ((ipad_byte, opad_byte), key_byte) in ipad.iter_mut().zip(opad.iter_mut()).zip(key_block.iter().copied()) {
      *ipad_byte ^= key_byte;
      *opad_byte ^= key_byte;
    }

    let mut inner_init = Sha384::new();
    inner_init.update(&ipad);

    let mut outer_init = Sha384::new();
    outer_init.update(&opad);

    let inner_init_prefix = inner_init.aligned_prefix();
    let outer_init_prefix = outer_init.aligned_prefix();

    ct::zeroize(&mut key_block);
    ct::zeroize(&mut ipad);
    ct::zeroize(&mut opad);

    Self {
      inner: inner_init,
      inner_init: inner_init_prefix,
      outer_init: outer_init_prefix,
    }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.inner.update(data);
  }

  #[inline]
  fn finalize(&self) -> Self::Tag {
    let inner_hash = self.inner.finalize();
    let mut outer = Sha384::from_aligned_prefix(self.outer_init);
    outer.update(&inner_hash);
    outer.finalize()
  }

  #[inline]
  fn reset(&mut self) {
    self.inner.reset_to_aligned_prefix(self.inner_init);
  }

  #[inline]
  #[allow(clippy::indexing_slicing)] // All indices bounded by prior length checks + fixed-size arrays.
  fn mac(key: &[u8], data: &[u8]) -> Self::Tag {
    const INLINE_DATA_MAX: usize = 256;
    const INNER_BUF_LEN: usize = 512;

    let mut ipad = [0x36u8; SHA512_FAMILY_BLOCK_SIZE];
    if key.len() > SHA512_FAMILY_BLOCK_SIZE {
      let digest = Sha384::digest(key);
      for (ip, &kb) in ipad[..SHA384_TAG_SIZE].iter_mut().zip(digest.iter()) {
        *ip = kb ^ 0x36;
      }
    } else {
      for (ip, &kb) in ipad[..key.len()].iter_mut().zip(key.iter()) {
        *ip = kb ^ 0x36;
      }
    }

    let total_inner = (SHA512_FAMILY_BLOCK_SIZE as u64).strict_add(data.len() as u64);
    let compress = sha384_dispatch::compress_dispatch().select(len_hint_from_u64(total_inner));
    let total_inner_bits = total_inner.strict_mul(8);

    let mut state = SHA384_H0;
    let mut inner_buf = [0u8; INNER_BUF_LEN];
    let inner_sensitive_len: usize;

    if data.len() <= INLINE_DATA_MAX {
      inner_buf[..SHA512_FAMILY_BLOCK_SIZE].copy_from_slice(&ipad);
      let data_end = SHA512_FAMILY_BLOCK_SIZE.strict_add(data.len());
      inner_buf[SHA512_FAMILY_BLOCK_SIZE..data_end].copy_from_slice(data);
      inner_buf[data_end] = 0x80;
      let padded = data_end.strict_add(17).strict_add(127).strict_div(128).strict_mul(128);
      inner_buf[padded.strict_sub(8)..padded].copy_from_slice(&total_inner_bits.to_be_bytes());
      compress(&mut state, &inner_buf[..padded]);
      inner_sensitive_len = SHA512_FAMILY_BLOCK_SIZE;
    } else {
      compress(&mut state, &ipad);

      let full_len = data.len().strict_sub(data.len() % SHA512_FAMILY_BLOCK_SIZE);
      if full_len != 0 {
        compress(&mut state, &data[..full_len]);
      }
      let rest = &data[full_len..];

      inner_buf[..rest.len()].copy_from_slice(rest);
      inner_buf[rest.len()] = 0x80;
      if rest.len() >= 112 {
        compress(&mut state, &inner_buf[..SHA512_FAMILY_BLOCK_SIZE]);
        inner_buf[..SHA512_FAMILY_BLOCK_SIZE].fill(0);
      }
      inner_buf[120..SHA512_FAMILY_BLOCK_SIZE].copy_from_slice(&total_inner_bits.to_be_bytes());
      compress(&mut state, &inner_buf[..SHA512_FAMILY_BLOCK_SIZE]);
      inner_sensitive_len = 0;
    }

    let mut outer = [0u8; 256];
    for (o, &ip) in outer[..SHA512_FAMILY_BLOCK_SIZE].iter_mut().zip(ipad.iter()) {
      *o = ip ^ 0x6a;
    }
    for (i, &word) in state.iter().take(6).enumerate() {
      let off = SHA512_FAMILY_BLOCK_SIZE.strict_add(i.strict_mul(8));
      outer[off..off.strict_add(8)].copy_from_slice(&word.to_be_bytes());
    }
    outer[SHA512_FAMILY_BLOCK_SIZE.strict_add(SHA384_TAG_SIZE)] = 0x80;
    outer[248..256].copy_from_slice(&1408u64.to_be_bytes());

    state = SHA384_H0;
    compress(&mut state, &outer);

    let mut tag = [0u8; SHA384_TAG_SIZE];
    for (chunk, &word) in tag.chunks_exact_mut(8).zip(state.iter()) {
      chunk.copy_from_slice(&word.to_be_bytes());
    }

    ct::zeroize_no_fence(&mut ipad);
    if inner_sensitive_len != 0 {
      ct::zeroize_no_fence(&mut inner_buf[..inner_sensitive_len]);
    }
    ct::zeroize_no_fence(&mut outer[..SHA512_FAMILY_BLOCK_SIZE]);
    for word in state.iter_mut() {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);

    tag
  }

  #[inline]
  fn verify(&self, expected: &Self::Tag) -> Result<(), VerificationError> {
    if ct::constant_time_eq(self.finalize().as_ref(), expected.as_ref()) {
      Ok(())
    } else {
      Err(VerificationError::new())
    }
  }
}

/// HMAC-SHA512 authentication state.
#[derive(Clone)]
pub struct HmacSha512 {
  inner: Sha512,
  inner_init: Sha512Prefix,
  outer_init: Sha512Prefix,
}

impl core::fmt::Debug for HmacSha512 {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("HmacSha512").finish_non_exhaustive()
  }
}

impl HmacSha512 {
  /// HMAC-SHA512 block size in bytes.
  pub const BLOCK_SIZE: usize = SHA512_FAMILY_BLOCK_SIZE;

  /// HMAC-SHA512 tag size in bytes.
  pub const TAG_SIZE: usize = SHA512_TAG_SIZE;

  /// Compute the HMAC tag of `data` in one shot.
  #[inline]
  #[must_use]
  pub fn mac(key: &[u8], data: &[u8]) -> [u8; SHA512_TAG_SIZE] {
    <Self as Mac>::mac(key, data)
  }

  /// Verify `expected` against the HMAC tag of `data` in constant time.
  #[inline]
  pub fn verify_tag(key: &[u8], data: &[u8], expected: &[u8; SHA512_TAG_SIZE]) -> Result<(), VerificationError> {
    <Self as Mac>::verify_tag(key, data, expected)
  }

  #[cfg(test)]
  pub(crate) fn new_with_compress_for_test(
    key: &[u8],
    compress: crate::hashes::crypto::sha512::kernels::CompressBlocksFn,
  ) -> Self {
    let mut key_block = [0u8; SHA512_FAMILY_BLOCK_SIZE];
    if key.len() > SHA512_FAMILY_BLOCK_SIZE {
      let digest = Sha512::digest(key);
      key_block[..SHA512_TAG_SIZE].copy_from_slice(&digest);
    } else {
      key_block[..key.len()].copy_from_slice(key);
    }

    let mut ipad = [0x36u8; SHA512_FAMILY_BLOCK_SIZE];
    let mut opad = [0x5Cu8; SHA512_FAMILY_BLOCK_SIZE];
    for ((ipad_byte, opad_byte), key_byte) in ipad.iter_mut().zip(opad.iter_mut()).zip(key_block.iter().copied()) {
      *ipad_byte ^= key_byte;
      *opad_byte ^= key_byte;
    }

    let mut inner = Sha512::new_with_compress_for_test(compress);
    inner.update(&ipad);
    let inner_init = inner.aligned_prefix();

    let mut outer = Sha512::new_with_compress_for_test(compress);
    outer.update(&opad);
    let outer_init = outer.aligned_prefix();

    ct::zeroize(&mut key_block);
    ct::zeroize(&mut ipad);
    ct::zeroize(&mut opad);

    Self {
      inner,
      inner_init,
      outer_init,
    }
  }

  #[cfg(test)]
  pub(crate) fn mac_with_compress_for_test(
    key: &[u8],
    data: &[u8],
    compress: crate::hashes::crypto::sha512::kernels::CompressBlocksFn,
  ) -> [u8; SHA512_TAG_SIZE] {
    let mut mac = Self::new_with_compress_for_test(key, compress);
    mac.update(data);
    mac.finalize()
  }
}

impl Mac for HmacSha512 {
  const TAG_SIZE: usize = SHA512_TAG_SIZE;
  type Tag = [u8; SHA512_TAG_SIZE];

  fn new(key: &[u8]) -> Self {
    let mut key_block = [0u8; SHA512_FAMILY_BLOCK_SIZE];
    if key.len() > SHA512_FAMILY_BLOCK_SIZE {
      let digest = Sha512::digest(key);
      for (dst, src) in key_block.iter_mut().zip(digest.iter()) {
        *dst = *src;
      }
    } else {
      for (dst, src) in key_block.iter_mut().zip(key.iter()) {
        *dst = *src;
      }
    }

    let mut ipad = [0x36u8; SHA512_FAMILY_BLOCK_SIZE];
    let mut opad = [0x5Cu8; SHA512_FAMILY_BLOCK_SIZE];
    for ((ipad_byte, opad_byte), key_byte) in ipad.iter_mut().zip(opad.iter_mut()).zip(key_block.iter().copied()) {
      *ipad_byte ^= key_byte;
      *opad_byte ^= key_byte;
    }

    let mut inner_init = Sha512::new();
    inner_init.update(&ipad);

    let mut outer_init = Sha512::new();
    outer_init.update(&opad);

    let inner_init_prefix = inner_init.aligned_prefix();
    let outer_init_prefix = outer_init.aligned_prefix();

    ct::zeroize(&mut key_block);
    ct::zeroize(&mut ipad);
    ct::zeroize(&mut opad);

    Self {
      inner: inner_init,
      inner_init: inner_init_prefix,
      outer_init: outer_init_prefix,
    }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.inner.update(data);
  }

  #[inline]
  fn finalize(&self) -> Self::Tag {
    let inner_hash = self.inner.finalize();
    let mut outer = Sha512::from_aligned_prefix(self.outer_init);
    outer.update(&inner_hash);
    outer.finalize()
  }

  #[inline]
  fn reset(&mut self) {
    self.inner.reset_to_aligned_prefix(self.inner_init);
  }

  #[inline]
  #[allow(clippy::indexing_slicing)] // All indices bounded by prior length checks + fixed-size arrays.
  fn mac(key: &[u8], data: &[u8]) -> Self::Tag {
    const INLINE_DATA_MAX: usize = 256;
    const INNER_BUF_LEN: usize = 512;

    let mut ipad = [0x36u8; SHA512_FAMILY_BLOCK_SIZE];
    if key.len() > SHA512_FAMILY_BLOCK_SIZE {
      let digest = Sha512::digest(key);
      for (ip, &kb) in ipad[..SHA512_TAG_SIZE].iter_mut().zip(digest.iter()) {
        *ip = kb ^ 0x36;
      }
    } else {
      for (ip, &kb) in ipad[..key.len()].iter_mut().zip(key.iter()) {
        *ip = kb ^ 0x36;
      }
    }

    let total_inner = (SHA512_FAMILY_BLOCK_SIZE as u64).strict_add(data.len() as u64);
    let compress = sha512_dispatch::compress_dispatch().select(len_hint_from_u64(total_inner));
    let total_inner_bits = total_inner.strict_mul(8);

    let mut state = SHA512_H0;
    let mut inner_buf = [0u8; INNER_BUF_LEN];
    let inner_sensitive_len: usize;

    if data.len() <= INLINE_DATA_MAX {
      inner_buf[..SHA512_FAMILY_BLOCK_SIZE].copy_from_slice(&ipad);
      let data_end = SHA512_FAMILY_BLOCK_SIZE.strict_add(data.len());
      inner_buf[SHA512_FAMILY_BLOCK_SIZE..data_end].copy_from_slice(data);
      inner_buf[data_end] = 0x80;
      let padded = data_end.strict_add(17).strict_add(127).strict_div(128).strict_mul(128);
      inner_buf[padded.strict_sub(8)..padded].copy_from_slice(&total_inner_bits.to_be_bytes());
      compress(&mut state, &inner_buf[..padded]);
      inner_sensitive_len = SHA512_FAMILY_BLOCK_SIZE;
    } else {
      compress(&mut state, &ipad);

      let full_len = data.len().strict_sub(data.len() % SHA512_FAMILY_BLOCK_SIZE);
      if full_len != 0 {
        compress(&mut state, &data[..full_len]);
      }
      let rest = &data[full_len..];

      inner_buf[..rest.len()].copy_from_slice(rest);
      inner_buf[rest.len()] = 0x80;
      if rest.len() >= 112 {
        compress(&mut state, &inner_buf[..SHA512_FAMILY_BLOCK_SIZE]);
        inner_buf[..SHA512_FAMILY_BLOCK_SIZE].fill(0);
      }
      inner_buf[120..SHA512_FAMILY_BLOCK_SIZE].copy_from_slice(&total_inner_bits.to_be_bytes());
      compress(&mut state, &inner_buf[..SHA512_FAMILY_BLOCK_SIZE]);
      inner_sensitive_len = 0;
    }

    let mut outer = [0u8; 256];
    for (o, &ip) in outer[..SHA512_FAMILY_BLOCK_SIZE].iter_mut().zip(ipad.iter()) {
      *o = ip ^ 0x6a;
    }
    for (i, &word) in state.iter().enumerate() {
      let off = SHA512_FAMILY_BLOCK_SIZE.strict_add(i.strict_mul(8));
      outer[off..off.strict_add(8)].copy_from_slice(&word.to_be_bytes());
    }
    outer[SHA512_FAMILY_BLOCK_SIZE.strict_add(SHA512_TAG_SIZE)] = 0x80;
    outer[248..256].copy_from_slice(&1536u64.to_be_bytes());

    state = SHA512_H0;
    compress(&mut state, &outer);

    let mut tag = [0u8; SHA512_TAG_SIZE];
    for (chunk, &word) in tag.chunks_exact_mut(8).zip(state.iter()) {
      chunk.copy_from_slice(&word.to_be_bytes());
    }

    ct::zeroize_no_fence(&mut ipad);
    if inner_sensitive_len != 0 {
      ct::zeroize_no_fence(&mut inner_buf[..inner_sensitive_len]);
    }
    ct::zeroize_no_fence(&mut outer[..SHA512_FAMILY_BLOCK_SIZE]);
    for word in state.iter_mut() {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);

    tag
  }

  #[inline]
  fn verify(&self, expected: &Self::Tag) -> Result<(), VerificationError> {
    if ct::constant_time_eq(self.finalize().as_ref(), expected.as_ref()) {
      Ok(())
    } else {
      Err(VerificationError::new())
    }
  }
}

#[cfg(test)]
mod tests {
  use alloc::vec::Vec;

  use hmac::{Hmac, Mac as _, digest::KeyInit};

  use super::*;
  use crate::hashes::crypto::{
    sha384::kernels::{
      ALL as SHA384_KERNELS, Sha384KernelId, compress_blocks_fn as sha384_compress_blocks_fn,
      required_caps as sha384_required_caps,
    },
    sha512::kernels::{
      ALL as SHA512_KERNELS, Sha512KernelId, compress_blocks_fn as sha512_compress_blocks_fn,
      required_caps as sha512_required_caps,
    },
  };

  type RustCryptoHmacSha384 = Hmac<sha2::Sha384>;
  type RustCryptoHmacSha512 = Hmac<sha2::Sha512>;

  fn pattern(len: usize, mul: u8, add: u8) -> Vec<u8> {
    (0..len)
      .map(|i| {
        (i as u8)
          .wrapping_mul(mul)
          .wrapping_add(((i >> 3) as u8).wrapping_add(add))
      })
      .collect()
  }

  fn oracle_hmac_sha384(key: &[u8], data: &[u8]) -> [u8; SHA384_TAG_SIZE] {
    let mut mac = RustCryptoHmacSha384::new_from_slice(key).unwrap();
    mac.update(data);
    let bytes = mac.finalize().into_bytes();
    let mut tag = [0u8; SHA384_TAG_SIZE];
    tag.copy_from_slice(&bytes);
    tag
  }

  fn oracle_hmac_sha512(key: &[u8], data: &[u8]) -> [u8; SHA512_TAG_SIZE] {
    let mut mac = RustCryptoHmacSha512::new_from_slice(key).unwrap();
    mac.update(data);
    let bytes = mac.finalize().into_bytes();
    let mut tag = [0u8; SHA512_TAG_SIZE];
    tag.copy_from_slice(&bytes);
    tag
  }

  fn assert_hmac_sha384_kernel(id: Sha384KernelId) {
    let compress = sha384_compress_blocks_fn(id);
    let cases = [
      (0usize, 0usize, 1usize),
      (1, 1, 1),
      (16, 31, 7),
      (48, 127, 31),
      (80, 128, 64),
      (160, 129, 65),
      (256, 255, 128),
      (300, 1024, 257),
    ];

    for &(key_len, data_len, chunk_len) in &cases {
      let key = pattern(key_len, 17, 3);
      let data = pattern(data_len, 29, 11);
      let expected = oracle_hmac_sha384(&key, &data);

      assert_eq!(
        HmacSha384::mac(&key, &data),
        expected,
        "sha384 public oneshot mismatch kernel={} key_len={} data_len={}",
        id.as_str(),
        key_len,
        data_len
      );
      assert_eq!(
        HmacSha384::mac_with_compress_for_test(&key, &data, compress),
        expected,
        "sha384 forced oneshot mismatch kernel={} key_len={} data_len={}",
        id.as_str(),
        key_len,
        data_len
      );

      let mut streaming = HmacSha384::new_with_compress_for_test(&key, compress);
      for chunk in data.chunks(chunk_len) {
        streaming.update(chunk);
      }
      assert_eq!(
        streaming.finalize(),
        expected,
        "sha384 forced streaming mismatch kernel={} key_len={} data_len={} chunk_len={}",
        id.as_str(),
        key_len,
        data_len,
        chunk_len
      );

      streaming.reset();
      streaming.update(&data);
      assert_eq!(
        streaming.finalize(),
        expected,
        "sha384 forced reset mismatch kernel={} key_len={} data_len={}",
        id.as_str(),
        key_len,
        data_len
      );
    }
  }

  fn assert_hmac_sha512_kernel(id: Sha512KernelId) {
    let compress = sha512_compress_blocks_fn(id);
    let cases = [
      (0usize, 0usize, 1usize),
      (1, 1, 1),
      (32, 63, 7),
      (64, 127, 31),
      (96, 128, 64),
      (192, 129, 65),
      (256, 255, 128),
      (320, 1024, 257),
    ];

    for &(key_len, data_len, chunk_len) in &cases {
      let key = pattern(key_len, 23, 7);
      let data = pattern(data_len, 37, 13);
      let expected = oracle_hmac_sha512(&key, &data);

      assert_eq!(
        HmacSha512::mac(&key, &data),
        expected,
        "sha512 public oneshot mismatch kernel={} key_len={} data_len={}",
        id.as_str(),
        key_len,
        data_len
      );
      assert_eq!(
        HmacSha512::mac_with_compress_for_test(&key, &data, compress),
        expected,
        "sha512 forced oneshot mismatch kernel={} key_len={} data_len={}",
        id.as_str(),
        key_len,
        data_len
      );

      let mut streaming = HmacSha512::new_with_compress_for_test(&key, compress);
      for chunk in data.chunks(chunk_len) {
        streaming.update(chunk);
      }
      assert_eq!(
        streaming.finalize(),
        expected,
        "sha512 forced streaming mismatch kernel={} key_len={} data_len={} chunk_len={}",
        id.as_str(),
        key_len,
        data_len,
        chunk_len
      );

      streaming.reset();
      streaming.update(&data);
      assert_eq!(
        streaming.finalize(),
        expected,
        "sha512 forced reset mismatch kernel={} key_len={} data_len={}",
        id.as_str(),
        key_len,
        data_len
      );
    }
  }

  #[test]
  fn hmac_sha384_forced_kernels_match_oracle() {
    let caps = crate::platform::caps();
    for &id in SHA384_KERNELS {
      if caps.has(sha384_required_caps(id)) {
        assert_hmac_sha384_kernel(id);
      }
    }
  }

  #[test]
  fn hmac_sha512_forced_kernels_match_oracle() {
    let caps = crate::platform::caps();
    for &id in SHA512_KERNELS {
      if caps.has(sha512_required_caps(id)) {
        assert_hmac_sha512_kernel(id);
      }
    }
  }
}
