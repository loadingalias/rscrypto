//! SHA-512/256 (FIPS 180-4).
//!
//! SHA-512/256 is identical to SHA-512 except for initial hash values (H0) and
//! output truncation (32 bytes / 4 words). The compression function is shared.

#![allow(clippy::indexing_slicing)] // Fixed-size arrays in finalization

use self::kernels::CompressBlocksFn;
use super::sha512::Sha512;
use crate::{
  hashes::crypto::dispatch_util::{SizeClassDispatch, len_hint_from_u128},
  traits::Digest,
};

#[doc(hidden)]
pub(crate) mod dispatch;
#[doc(hidden)]
pub(crate) mod dispatch_tables;
#[cfg(test)]
mod kernel_test;
pub(crate) mod kernels;

const BLOCK_LEN: usize = 128;

// SHA-512/256 initial hash value (FIPS 180-4 SS5.3.6.2).
const H0: [u64; 8] = [
  0x2231_2194_fc2b_f72c,
  0x9f55_5fa3_c84c_64c2,
  0x2393_b86b_6f53_b151,
  0x9638_7719_5940_eabd,
  0x9628_3ee2_a88e_ffe3,
  0xbe5e_1e25_5386_3992,
  0x2b01_99fc_2c85_b8aa,
  0x0eb7_2ddc_81c5_2ca2,
];

/// SHA-512/256 digest state.
///
/// Standardized in FIPS 180-4.
///
/// # Examples
///
/// ```
/// use rscrypto::{Digest, Sha512_256};
///
/// let mut hasher = Sha512_256::new();
/// hasher.update(b"abc");
///
/// assert_eq!(hasher.finalize(), Sha512_256::digest(b"abc"));
/// ```
#[derive(Clone)]
pub struct Sha512_256 {
  state: [u64; 8],
  block: [u8; BLOCK_LEN],
  block_len: usize,
  bytes_hashed: u128,
  compress_blocks: CompressBlocksFn,
  dispatch: Option<SizeClassDispatch<CompressBlocksFn>>,
}

impl Sha512_256 {
  /// Compute the digest of `data` in one shot.
  ///
  /// This selects the best available kernel for the current platform and input
  /// length (cached after first use).
  #[inline]
  #[must_use]
  pub fn digest(data: &[u8]) -> [u8; 32] {
    dispatch::digest(data)
  }
}

impl core::fmt::Debug for Sha512_256 {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Sha512_256").finish_non_exhaustive()
  }
}

impl Default for Sha512_256 {
  #[inline]
  fn default() -> Self {
    Self {
      state: H0,
      block: [0u8; BLOCK_LEN],
      block_len: 0,
      bytes_hashed: 0,
      compress_blocks: Sha512::compress_blocks_portable,
      dispatch: None,
    }
  }
}

impl Sha512_256 {
  #[inline]
  fn select_compress(&mut self, incoming_len: usize) -> CompressBlocksFn {
    let dispatch = match self.dispatch {
      Some(d) => d,
      None => {
        let d = dispatch::compress_dispatch();
        self.dispatch = Some(d);
        d
      }
    };

    let total = self
      .bytes_hashed
      .strict_add(self.block_len as u128)
      .strict_add(incoming_len as u128);
    let compress = dispatch.select(len_hint_from_u128(total));
    self.compress_blocks = compress;
    compress
  }

  #[inline]
  fn update_with(&mut self, mut data: &[u8], compress_blocks: CompressBlocksFn) {
    if data.is_empty() {
      return;
    }

    if self.block_len != 0 {
      let take = core::cmp::min(BLOCK_LEN.strict_sub(self.block_len), data.len());
      self.block[self.block_len..self.block_len.strict_add(take)].copy_from_slice(&data[..take]);
      self.block_len = self.block_len.strict_add(take);
      data = &data[take..];

      if self.block_len == BLOCK_LEN {
        compress_blocks(&mut self.state, &self.block);
        self.bytes_hashed = self.bytes_hashed.strict_add(BLOCK_LEN as u128);
        self.block_len = 0;
      }
    }

    let full_len = data.len().strict_sub(data.len() % BLOCK_LEN);
    if full_len != 0 {
      let (blocks, rest) = data.split_at(full_len);
      compress_blocks(&mut self.state, blocks);
      self.bytes_hashed = self.bytes_hashed.strict_add(blocks.len() as u128);
      data = rest;
    }

    if !data.is_empty() {
      self.block[..data.len()].copy_from_slice(data);
      self.block_len = data.len();
    }
  }

  #[inline]
  fn finalize_inner_with(&self, compress_blocks: CompressBlocksFn) -> [u8; 32] {
    let mut state = self.state;
    let mut block = self.block;
    let mut block_len = self.block_len;
    let total_len = self.bytes_hashed.strict_add(block_len as u128);

    block[block_len] = 0x80;
    block_len = block_len.strict_add(1);

    if block_len > 112 {
      block[block_len..].fill(0);
      compress_blocks(&mut state, &block);
      block = [0u8; BLOCK_LEN];
      block_len = 0;
    }

    block[block_len..112].fill(0);

    let bit_len = total_len << 3;
    block[112..128].copy_from_slice(&bit_len.to_be_bytes());
    compress_blocks(&mut state, &block);

    let mut out = [0u8; 32];
    for (chunk, &word) in out.chunks_exact_mut(8).zip(state.iter()) {
      chunk.copy_from_slice(&word.to_be_bytes());
    }
    out
  }
}

impl Drop for Sha512_256 {
  fn drop(&mut self) {
    for word in self.state.iter_mut() {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    crate::traits::ct::zeroize(&mut self.block);
    // SAFETY: field is a valid, aligned, dereferenceable pointer to initialized memory.
    unsafe { core::ptr::write_volatile(&mut self.bytes_hashed, 0) };
    // SAFETY: field is a valid, aligned, dereferenceable pointer to initialized memory.
    unsafe { core::ptr::write_volatile(&mut self.block_len, 0) };
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

impl Digest for Sha512_256 {
  const OUTPUT_SIZE: usize = 32;
  type Output = [u8; 32];

  #[inline]
  fn new() -> Self {
    Self::default()
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    if data.is_empty() {
      return;
    }
    let compress = self.select_compress(data.len());
    self.update_with(data, compress);
  }

  #[inline]
  fn finalize(&self) -> Self::Output {
    self.finalize_inner_with(self.compress_blocks)
  }

  #[inline]
  fn reset(&mut self) {
    *self = Self::default();
  }

  #[inline]
  fn digest(data: &[u8]) -> Self::Output {
    dispatch::digest(data)
  }
}
