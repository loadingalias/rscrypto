//! BLAKE2s-256 (RFC 7693).
//!
//! Portable, `no_std`, pure Rust implementation (unkeyed).

#![allow(clippy::indexing_slicing)] // Compression schedule uses fixed indices

use traits::Digest;

use self::kernels::CompressFn;
use crate::crypto::dispatch_util::{SizeClassDispatch, len_hint_from_u64};

#[doc(hidden)]
pub mod dispatch;
#[doc(hidden)]
pub mod dispatch_tables;
pub(crate) mod kernels;

const BLOCK_LEN: usize = 64;

const IV: [u32; 8] = [
  0x6A09_E667,
  0xBB67_AE85,
  0x3C6E_F372,
  0xA54F_F53A,
  0x510E_527F,
  0x9B05_688C,
  0x1F83_D9AB,
  0x5BE0_CD19,
];

const SIGMA: [[usize; 16]; 10] = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
  [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
  [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
  [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
  [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
  [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
  [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
  [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
  [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

#[inline(always)]
fn rotr32(x: u32, n: u32) -> u32 {
  x.rotate_right(n)
}

#[inline(always)]
fn g(a: &mut u32, b: &mut u32, c: &mut u32, d: &mut u32, x: u32, y: u32) {
  *a = a.wrapping_add(*b).wrapping_add(x);
  *d = rotr32(*d ^ *a, 16);
  *c = c.wrapping_add(*d);
  *b = rotr32(*b ^ *c, 12);
  *a = a.wrapping_add(*b).wrapping_add(y);
  *d = rotr32(*d ^ *a, 8);
  *c = c.wrapping_add(*d);
  *b = rotr32(*b ^ *c, 7);
}

#[inline(always)]
fn compress(h: &mut [u32; 8], block: &[u8; BLOCK_LEN], t: u64, is_last: bool) {
  let (chunks, _) = block.as_chunks::<4>();
  let mut m = [0u32; 16];
  for (i, c) in chunks.iter().enumerate() {
    m[i] = u32::from_le_bytes(*c);
  }

  let mut v0 = h[0];
  let mut v1 = h[1];
  let mut v2 = h[2];
  let mut v3 = h[3];
  let mut v4 = h[4];
  let mut v5 = h[5];
  let mut v6 = h[6];
  let mut v7 = h[7];

  let mut v8 = IV[0];
  let mut v9 = IV[1];
  let mut v10 = IV[2];
  let mut v11 = IV[3];
  let mut v12 = IV[4] ^ (t as u32);
  let mut v13 = IV[5] ^ ((t >> 32) as u32);
  let mut v14 = IV[6] ^ if is_last { 0xFFFF_FFFF } else { 0 };
  let mut v15 = IV[7];

  macro_rules! round {
    ($r:expr) => {{
      let s = &SIGMA[$r];

      g(&mut v0, &mut v4, &mut v8, &mut v12, m[s[0]], m[s[1]]);
      g(&mut v1, &mut v5, &mut v9, &mut v13, m[s[2]], m[s[3]]);
      g(&mut v2, &mut v6, &mut v10, &mut v14, m[s[4]], m[s[5]]);
      g(&mut v3, &mut v7, &mut v11, &mut v15, m[s[6]], m[s[7]]);

      g(&mut v0, &mut v5, &mut v10, &mut v15, m[s[8]], m[s[9]]);
      g(&mut v1, &mut v6, &mut v11, &mut v12, m[s[10]], m[s[11]]);
      g(&mut v2, &mut v7, &mut v8, &mut v13, m[s[12]], m[s[13]]);
      g(&mut v3, &mut v4, &mut v9, &mut v14, m[s[14]], m[s[15]]);
    }};
  }

  round!(0);
  round!(1);
  round!(2);
  round!(3);
  round!(4);
  round!(5);
  round!(6);
  round!(7);
  round!(8);
  round!(9);

  h[0] ^= v0 ^ v8;
  h[1] ^= v1 ^ v9;
  h[2] ^= v2 ^ v10;
  h[3] ^= v3 ^ v11;
  h[4] ^= v4 ^ v12;
  h[5] ^= v5 ^ v13;
  h[6] ^= v6 ^ v14;
  h[7] ^= v7 ^ v15;
}

#[derive(Clone)]
pub struct Blake2s256 {
  h: [u32; 8],
  buf: [u8; BLOCK_LEN],
  buf_len: usize,
  bytes_hashed: u64,
  compress: CompressFn,
  dispatch: Option<SizeClassDispatch<CompressFn>>,
}

impl Blake2s256 {
  /// Compute the digest of `data` in one shot.
  ///
  /// This selects the best available kernel for the current platform and input
  /// length (cached after first use).
  #[inline]
  #[must_use]
  pub fn digest(data: &[u8]) -> [u8; 32] {
    dispatch::digest(data)
  }

  #[inline]
  #[must_use]
  pub(crate) fn digest_portable(data: &[u8]) -> [u8; 32] {
    let mut h = Self::default();
    h.update_with(data, Self::compress_portable);
    h.finalize_with(Self::compress_portable)
  }

  #[inline]
  pub(crate) fn compress_portable(
    h: &mut [u32; 8],
    blocks: &[u8],
    bytes_hashed: &mut u64,
    is_last: bool,
    last_block_len: u32,
  ) {
    debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);
    if blocks.is_empty() {
      return;
    }

    if is_last {
      debug_assert_eq!(blocks.len(), BLOCK_LEN);
      debug_assert!(last_block_len as usize <= BLOCK_LEN);
      *bytes_hashed = (*bytes_hashed).wrapping_add(last_block_len as u64);
      // SAFETY: `blocks` is exactly `BLOCK_LEN` bytes (asserted above).
      let block = unsafe { &*(blocks.as_ptr() as *const [u8; BLOCK_LEN]) };
      compress(h, block, *bytes_hashed, true);
      return;
    }

    let mut chunks = blocks.chunks_exact(BLOCK_LEN);
    for chunk in &mut chunks {
      *bytes_hashed = (*bytes_hashed).wrapping_add(BLOCK_LEN as u64);
      // SAFETY: `chunks_exact(BLOCK_LEN)` yields slices of exactly `BLOCK_LEN` bytes.
      let block = unsafe { &*(chunk.as_ptr() as *const [u8; BLOCK_LEN]) };
      compress(h, block, *bytes_hashed, false);
    }
    debug_assert!(chunks.remainder().is_empty());
  }

  #[inline]
  fn update_with(&mut self, mut data: &[u8], compress: CompressFn) {
    if data.is_empty() {
      return;
    }

    if self.buf_len != 0 {
      let take = core::cmp::min(BLOCK_LEN - self.buf_len, data.len());
      self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
      self.buf_len += take;
      data = &data[take..];

      // Keep a full block buffered until we know there is more input, so the
      // final block can be marked with the `is_last` flag.
      if self.buf_len == BLOCK_LEN && !data.is_empty() {
        compress(&mut self.h, &self.buf, &mut self.bytes_hashed, false, 0);
        self.buf_len = 0;
      }
    }

    let full_len = data.len() - (data.len() % BLOCK_LEN);
    if full_len != 0 {
      let (full, rest) = data.split_at(full_len);
      if rest.is_empty() {
        // Hold back the last full block for finalization.
        let split = full_len - BLOCK_LEN;
        let (to_compress, last_full) = full.split_at(split);
        compress(&mut self.h, to_compress, &mut self.bytes_hashed, false, 0);
        self.buf.copy_from_slice(last_full);
        self.buf_len = BLOCK_LEN;
      } else {
        compress(&mut self.h, full, &mut self.bytes_hashed, false, 0);
      }
      data = rest;
    }

    if !data.is_empty() {
      self.buf[..data.len()].copy_from_slice(data);
      self.buf_len = data.len();
    }
  }

  #[inline]
  fn finalize_with(&self, compress: CompressFn) -> [u8; 32] {
    let mut h = self.h;
    let mut buf = self.buf;
    let len = self.buf_len;

    buf[len..].fill(0);
    let mut t = self.bytes_hashed;
    compress(&mut h, &buf, &mut t, true, len as u32);

    let mut out = [0u8; 32];
    for (i, word) in h.iter().copied().enumerate() {
      out[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    out
  }
}

impl Default for Blake2s256 {
  #[inline]
  fn default() -> Self {
    let mut h = IV;
    // Parameter block word 0: outlen=32, keylen=0, fanout=1, depth=1.
    h[0] ^= 0x0101_0020;
    Self {
      h,
      buf: [0u8; BLOCK_LEN],
      buf_len: 0,
      bytes_hashed: 0,
      compress: Blake2s256::compress_portable,
      dispatch: None,
    }
  }
}

impl Digest for Blake2s256 {
  const OUTPUT_SIZE: usize = 32;
  type Output = [u8; 32];

  #[inline]
  fn new() -> Self {
    Self::default()
  }

  fn update(&mut self, data: &[u8]) {
    if data.is_empty() {
      return;
    }
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
      .saturating_add(self.buf_len as u64)
      .saturating_add(data.len() as u64);
    self.compress = dispatch.select(len_hint_from_u64(total));
    self.update_with(data, self.compress);
  }

  fn finalize(&self) -> Self::Output {
    self.finalize_with(self.compress)
  }

  #[inline]
  fn reset(&mut self) {
    *self = Self::default();
  }
}

#[cfg(feature = "std")]
pub(crate) mod kernel_test;
