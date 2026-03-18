//! SHA-384 (FIPS 180-4).
//!
//! SHA-384 is identical to SHA-512 except for initial hash values (H0) and
//! output truncation (48 bytes / 6 words). The compression function is shared.

#![allow(clippy::indexing_slicing)] // Fixed-size arrays in finalization

use self::kernels::CompressBlocksFn;
use super::sha512::Sha512;
use crate::{
  hashes::crypto::dispatch_util::{SizeClassDispatch, len_hint_from_u128},
  traits::Digest,
};

#[doc(hidden)]
pub mod dispatch;
#[doc(hidden)]
pub mod dispatch_tables;
pub(crate) mod kernels;

const BLOCK_LEN: usize = 128;

// SHA-384 initial hash value (FIPS 180-4 SS5.3.4).
const H0: [u64; 8] = [
  0xcbbb_9d5d_c105_9ed8,
  0x629a_292a_367c_d507,
  0x9159_015a_3070_dd17,
  0x152f_ecd8_f70e_5939,
  0x6733_2667_ffc0_0b31,
  0x8eb4_4a87_6858_1511,
  0xdb0c_2e0d_64f9_8fa7,
  0x47b5_481d_befa_4fa4,
];

#[derive(Clone)]
pub struct Sha384 {
  state: [u64; 8],
  block: [u8; BLOCK_LEN],
  block_len: usize,
  bytes_hashed: u128,
  compress_blocks: CompressBlocksFn,
  dispatch: Option<SizeClassDispatch<CompressBlocksFn>>,
}

impl Sha384 {
  /// Compute the digest of `data` in one shot.
  ///
  /// This selects the best available kernel for the current platform and input
  /// length (cached after first use).
  #[inline]
  #[must_use]
  pub fn digest(data: &[u8]) -> [u8; 48] {
    dispatch::digest(data)
  }

  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn digest_portable(data: &[u8]) -> [u8; 48] {
    let mut h = Self::default();
    h.update_with(data, Sha512::compress_blocks_portable);
    h.finalize_inner_with(Sha512::compress_blocks_portable)
  }
}

impl core::fmt::Debug for Sha384 {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Sha384").finish_non_exhaustive()
  }
}

impl Default for Sha384 {
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

impl Sha384 {
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
      .saturating_add(self.block_len as u128)
      .saturating_add(incoming_len as u128);
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
  fn finalize_inner_with(&self, compress_blocks: CompressBlocksFn) -> [u8; 48] {
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

    let mut out = [0u8; 48];
    for (i, word) in state.iter().copied().enumerate().take(6) {
      let offset = i.strict_mul(8);
      out[offset..offset.strict_add(8)].copy_from_slice(&word.to_be_bytes());
    }
    out
  }
}

impl Drop for Sha384 {
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

impl Digest for Sha384 {
  const OUTPUT_SIZE: usize = 48;
  type Output = [u8; 48];

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
}

#[cfg(feature = "std")]
pub(crate) mod kernel_test;
