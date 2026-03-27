//! SHA-224 (FIPS 180-4).
//!
//! SHA-224 is identical to SHA-256 except for initial hash values (H0) and
//! output truncation (28 bytes / 7 words). The compression function is shared.

#![allow(clippy::indexing_slicing)] // Fixed-size arrays in finalization

use self::kernels::CompressBlocksFn;
use super::sha256::Sha256;
use crate::{
  hashes::crypto::dispatch_util::{SizeClassDispatch, len_hint_from_u64},
  traits::Digest,
};

#[doc(hidden)]
pub mod dispatch;
#[doc(hidden)]
pub mod dispatch_tables;
pub(crate) mod kernels;

const BLOCK_LEN: usize = 64;

// SHA-224 initial hash value (FIPS 180-4 SS5.3.2).
const H0: [u32; 8] = [
  0xc105_9ed8,
  0x367c_d507,
  0x3070_dd17,
  0xf70e_5939,
  0xffc0_0b31,
  0x6858_1511,
  0x64f9_8fa7,
  0xbefa_4fa4,
];

/// Maximum message length in bytes for SHA-224 (FIPS 180-4).
///
/// Same limit as SHA-256: the spec encodes message length as a 64-bit **bit**
/// count, so the maximum byte length is `(2^64 − 1) / 8 = 2^61 − 1`.
const MAX_MESSAGE_LEN: u64 = u64::MAX / 8;

#[derive(Clone)]
pub struct Sha224 {
  state: [u32; 8],
  block: [u8; BLOCK_LEN],
  block_len: usize,
  bytes_hashed: u64,
  compress_blocks: CompressBlocksFn,
  dispatch: Option<SizeClassDispatch<CompressBlocksFn>>,
}

impl core::fmt::Debug for Sha224 {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Sha224").finish_non_exhaustive()
  }
}

impl Default for Sha224 {
  #[inline]
  fn default() -> Self {
    Self {
      state: H0,
      block: [0u8; BLOCK_LEN],
      block_len: 0,
      bytes_hashed: 0,
      compress_blocks: Sha256::compress_blocks_portable,
      dispatch: None,
    }
  }
}

impl Sha224 {
  /// Compute the digest of `data` in one shot.
  ///
  /// Selects the best available kernel for the current platform and input
  /// length (cached after first use).
  #[inline]
  #[must_use]
  pub fn digest(data: &[u8]) -> [u8; 28] {
    dispatch::digest(data)
  }

  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn digest_portable(data: &[u8]) -> [u8; 28] {
    // Fast path for small inputs: process on stack without streaming state.
    // For `len < 120`, at most two blocks are needed (one full + one padded).
    if data.len() < 120 {
      let mut state = H0;

      let total_len = data.len() as u64;
      let bit_len = total_len.strict_mul(8);

      let (blocks, rest) = data.as_chunks::<BLOCK_LEN>();
      if !blocks.is_empty() {
        // For `len < 120`, there can be at most one full block here.
        Sha256::compress_blocks_portable(&mut state, &blocks[0]);
      }

      let mut block0 = [0u8; BLOCK_LEN];
      block0[..rest.len()].copy_from_slice(rest);
      block0[rest.len()] = 0x80;

      if data.len() < 56 {
        block0[56..64].copy_from_slice(&bit_len.to_be_bytes());
        Sha256::compress_blocks_portable(&mut state, &block0);
      } else if blocks.is_empty() {
        // `56 <= len < 64`: padding spills into a second block.
        Sha256::compress_blocks_portable(&mut state, &block0);
        let mut block1 = [0u8; BLOCK_LEN];
        block1[56..64].copy_from_slice(&bit_len.to_be_bytes());
        Sha256::compress_blocks_portable(&mut state, &block1);
      } else {
        // `64 <= len < 120`: remainder < 56, so length fits in the final block.
        block0[56..64].copy_from_slice(&bit_len.to_be_bytes());
        Sha256::compress_blocks_portable(&mut state, &block0);
      }

      let mut out = [0u8; 28];
      for (chunk, &word) in out.chunks_exact_mut(4).zip(state.iter()) {
        chunk.copy_from_slice(&word.to_be_bytes());
      }
      out
    } else {
      let mut h = Self::default();
      h.update_with(data, Sha256::compress_blocks_portable);
      h.finalize_inner_with(Sha256::compress_blocks_portable)
    }
  }

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
      .strict_add(self.block_len as u64)
      .strict_add(incoming_len as u64);
    let compress = dispatch.select(len_hint_from_u64(total));
    self.compress_blocks = compress;
    compress
  }

  #[inline]
  fn update_with(&mut self, mut data: &[u8], compress_blocks: CompressBlocksFn) {
    if data.is_empty() {
      return;
    }

    debug_assert!(
      self
        .bytes_hashed
        .strict_add(self.block_len as u64)
        .strict_add(data.len() as u64)
        <= MAX_MESSAGE_LEN,
      "SHA-224: total input exceeds FIPS 180-4 maximum of 2^61 − 1 bytes"
    );

    if self.block_len != 0 {
      let take = core::cmp::min(BLOCK_LEN.strict_sub(self.block_len), data.len());
      self.block[self.block_len..self.block_len.strict_add(take)].copy_from_slice(&data[..take]);
      self.block_len = self.block_len.strict_add(take);
      data = &data[take..];

      if self.block_len == BLOCK_LEN {
        compress_blocks(&mut self.state, &self.block);
        self.bytes_hashed = self.bytes_hashed.strict_add(BLOCK_LEN as u64);
        self.block_len = 0;
      }
    }

    let full_len = data.len().strict_sub(data.len() % BLOCK_LEN);
    if full_len != 0 {
      let (blocks, rest) = data.split_at(full_len);
      compress_blocks(&mut self.state, blocks);
      self.bytes_hashed = self.bytes_hashed.strict_add(blocks.len() as u64);
      data = rest;
    }

    if !data.is_empty() {
      self.block[..data.len()].copy_from_slice(data);
      self.block_len = data.len();
    }
  }

  #[inline]
  fn finalize_inner_with(&self, compress_blocks: CompressBlocksFn) -> [u8; 28] {
    let mut state = self.state;
    let mut block = self.block;
    let mut block_len = self.block_len;
    let total_len = self.bytes_hashed.strict_add(block_len as u64);

    block[block_len] = 0x80;
    block_len = block_len.strict_add(1);

    if block_len > 56 {
      block[block_len..].fill(0);
      compress_blocks(&mut state, &block);
      block = [0u8; BLOCK_LEN];
      block_len = 0;
    }

    block[block_len..56].fill(0);

    let bit_len = total_len.strict_mul(8);
    block[56..64].copy_from_slice(&bit_len.to_be_bytes());
    compress_blocks(&mut state, &block);

    let mut out = [0u8; 28];
    for (chunk, &word) in out.chunks_exact_mut(4).zip(state.iter()) {
      chunk.copy_from_slice(&word.to_be_bytes());
    }
    out
  }
}

impl Drop for Sha224 {
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

impl Digest for Sha224 {
  const OUTPUT_SIZE: usize = 28;
  type Output = [u8; 28];

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

#[cfg(feature = "std")]
pub(crate) mod kernel_test;

#[cfg(test)]
mod tests {
  use super::Sha224;

  fn hex28(bytes: &[u8; 28]) -> alloc::string::String {
    use alloc::string::String;
    use core::fmt::Write;
    let mut s = String::new();
    for &b in bytes {
      write!(&mut s, "{:02x}", b).unwrap();
    }
    s
  }

  #[test]
  fn known_vectors() {
    // NIST FIPS 180-4 test vectors.
    assert_eq!(
      hex28(&Sha224::digest(b"")),
      "d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f"
    );
    assert_eq!(
      hex28(&Sha224::digest(b"abc")),
      "23097d223405d8228642a477bda255b32aadbce4bda0b3f7e36c9da7"
    );
    assert_eq!(
      hex28(&Sha224::digest(
        b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
      )),
      "75388b16512776cc5dba5da1fd890150b0c6455cb4f58b1952522525"
    );

    // 1,000,000 repetitions of 'a'.
    let mut million_a = alloc::vec![b'a'; 1_000_000];
    assert_eq!(
      hex28(&Sha224::digest(&million_a)),
      "20794655980c91d8bbb4c1ea97618a4bf03f42581948b2ee4ee7ad67"
    );
    million_a.clear();
  }
}
