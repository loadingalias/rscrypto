//! HMAC-SHA256 (RFC 2104, FIPS 198-1).

use crate::{
  hashes::crypto::{
    Sha256,
    dispatch_util::len_hint_from_u64,
    sha256::{H0 as SHA256_H0, Sha256Prefix, dispatch as sha256_dispatch},
  },
  traits::{Digest, Mac, VerificationError, ct},
};

const BLOCK_SIZE: usize = 64;
const TAG_SIZE: usize = 32;

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
  pub const BLOCK_SIZE: usize = BLOCK_SIZE;

  /// HMAC-SHA256 tag size in bytes.
  pub const TAG_SIZE: usize = TAG_SIZE;

  /// Compute the HMAC tag of `data` in one shot.
  #[inline]
  #[must_use]
  pub fn mac(key: &[u8], data: &[u8]) -> [u8; TAG_SIZE] {
    <Self as Mac>::mac(key, data)
  }

  /// Verify `expected` against the HMAC tag of `data` in constant time.
  #[inline]
  pub fn verify_tag(key: &[u8], data: &[u8], expected: &[u8; TAG_SIZE]) -> Result<(), VerificationError> {
    <Self as Mac>::verify_tag(key, data, expected)
  }
}

impl Mac for HmacSha256 {
  const TAG_SIZE: usize = TAG_SIZE;
  type Tag = [u8; TAG_SIZE];

  fn new(key: &[u8]) -> Self {
    let mut key_block = [0u8; BLOCK_SIZE];
    if key.len() > BLOCK_SIZE {
      let digest = Sha256::digest(key);
      for (dst, src) in key_block.iter_mut().zip(digest.iter()) {
        *dst = *src;
      }
    } else {
      for (dst, src) in key_block.iter_mut().zip(key.iter()) {
        *dst = *src;
      }
    }

    let mut ipad = [0x36u8; BLOCK_SIZE];
    let mut opad = [0x5Cu8; BLOCK_SIZE];
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

  /// Oneshot HMAC-SHA256: bypasses streaming `Sha256` construction entirely.
  ///
  /// Works directly with `[u32; 8]` state arrays and the dispatch compress
  /// function. Avoids creating 3 `Sha256` structs, their `Drop` zeroization,
  /// dispatch lazy-init overhead, and the unused `inner_init` clone.
  #[inline]
  #[allow(clippy::indexing_slicing)] // All indices bounded by prior length checks + fixed-size arrays.
  fn mac(key: &[u8], data: &[u8]) -> Self::Tag {
    // --- Key normalization (RFC 2104 §2) ---
    let mut key_block = [0u8; BLOCK_SIZE];
    if key.len() > BLOCK_SIZE {
      let digest = Sha256::digest(key);
      for (dst, src) in key_block.iter_mut().zip(digest.iter()) {
        *dst = *src;
      }
    } else {
      for (dst, src) in key_block.iter_mut().zip(key.iter()) {
        *dst = *src;
      }
    }

    // --- Build ipad / opad ---
    let mut ipad = [0u8; BLOCK_SIZE];
    let mut opad = [0u8; BLOCK_SIZE];
    for ((ip, op), &kb) in ipad.iter_mut().zip(opad.iter_mut()).zip(key_block.iter()) {
      *ip = kb ^ 0x36;
      *op = kb ^ 0x5c;
    }

    // --- Resolve compress function once for the full operation ---
    let total_inner = (BLOCK_SIZE as u64).strict_add(data.len() as u64);
    let compress = sha256_dispatch::compress_dispatch().select(len_hint_from_u64(total_inner));

    // --- Inner hash: SHA-256(ipad ∥ data) ---
    let mut state = SHA256_H0;
    compress(&mut state, &ipad);

    let full_len = data.len().strict_sub(data.len() % BLOCK_SIZE);
    if full_len != 0 {
      compress(&mut state, &data[..full_len]);
    }
    let rest = &data[full_len..];

    let total_bits = total_inner.strict_mul(8);
    let mut block = [0u8; BLOCK_SIZE];
    block[..rest.len()].copy_from_slice(rest);
    block[rest.len()] = 0x80;

    if rest.len() >= 56 {
      compress(&mut state, &block);
      block = [0u8; BLOCK_SIZE];
    }
    block[56..BLOCK_SIZE].copy_from_slice(&total_bits.to_be_bytes());
    compress(&mut state, &block);

    // Serialize inner hash.
    let mut inner_hash = [0u8; TAG_SIZE];
    for (chunk, &word) in inner_hash.chunks_exact_mut(4).zip(state.iter()) {
      chunk.copy_from_slice(&word.to_be_bytes());
    }

    // --- Outer hash: SHA-256(opad ∥ inner_hash) ---
    // Fixed 96 bytes: opad (64) + inner_hash (32). Remainder 32 < 56, single final block.
    state = SHA256_H0;
    compress(&mut state, &opad);

    block = [0u8; BLOCK_SIZE];
    block[..TAG_SIZE].copy_from_slice(&inner_hash);
    block[TAG_SIZE] = 0x80;
    // 96 bytes × 8 = 768 bits
    block[56..BLOCK_SIZE].copy_from_slice(&768u64.to_be_bytes());
    compress(&mut state, &block);

    // Serialize tag.
    let mut tag = [0u8; TAG_SIZE];
    for (chunk, &word) in tag.chunks_exact_mut(4).zip(state.iter()) {
      chunk.copy_from_slice(&word.to_be_bytes());
    }

    // --- Zeroize sensitive material ---
    ct::zeroize(&mut key_block);
    ct::zeroize(&mut ipad);
    ct::zeroize(&mut opad);
    ct::zeroize(&mut inner_hash);
    ct::zeroize(&mut block);
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
