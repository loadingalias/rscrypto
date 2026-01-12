//! BLAKE2b-512 (RFC 7693).
//!
//! Portable, `no_std`, pure Rust implementation (unkeyed).

#![allow(clippy::indexing_slicing)] // Compression schedule uses fixed indices

use traits::Digest;

const BLOCK_LEN: usize = 128;

const IV: [u64; 8] = [
  0x6a09_e667_f3bc_c908,
  0xbb67_ae85_84ca_a73b,
  0x3c6e_f372_fe94_f82b,
  0xa54f_f53a_5f1d_36f1,
  0x510e_527f_ade6_82d1,
  0x9b05_688c_2b3e_6c1f,
  0x1f83_d9ab_fb41_bd6b,
  0x5be0_cd19_137e_2179,
];

const SIGMA: [[usize; 16]; 12] = [
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
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

#[inline(always)]
fn rotr64(x: u64, n: u32) -> u64 {
  x.rotate_right(n)
}

#[inline(always)]
fn g(a: &mut u64, b: &mut u64, c: &mut u64, d: &mut u64, x: u64, y: u64) {
  *a = a.wrapping_add(*b).wrapping_add(x);
  *d = rotr64(*d ^ *a, 32);
  *c = c.wrapping_add(*d);
  *b = rotr64(*b ^ *c, 24);
  *a = a.wrapping_add(*b).wrapping_add(y);
  *d = rotr64(*d ^ *a, 16);
  *c = c.wrapping_add(*d);
  *b = rotr64(*b ^ *c, 63);
}

#[inline(always)]
fn compress(h: &mut [u64; 8], block: &[u8; BLOCK_LEN], t: u128, is_last: bool) {
  let (chunks, _) = block.as_chunks::<8>();
  let mut m = [0u64; 16];
  for (i, c) in chunks.iter().enumerate() {
    m[i] = u64::from_le_bytes(*c);
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
  let mut v12 = IV[4] ^ (t as u64);
  let mut v13 = IV[5] ^ ((t >> 64) as u64);
  let mut v14 = IV[6] ^ if is_last { u64::MAX } else { 0 };
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
  round!(10);
  round!(11);

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
pub struct Blake2b512 {
  h: [u64; 8],
  buf: [u8; BLOCK_LEN],
  buf_len: usize,
  bytes_hashed: u128,
}

impl Default for Blake2b512 {
  #[inline]
  fn default() -> Self {
    let mut h = IV;
    // Parameter block: outlen=64, keylen=0, fanout=1, depth=1.
    h[0] ^= 0x0101_0040;
    Self {
      h,
      buf: [0u8; BLOCK_LEN],
      buf_len: 0,
      bytes_hashed: 0,
    }
  }
}

impl Digest for Blake2b512 {
  const OUTPUT_SIZE: usize = 64;
  type Output = [u8; 64];

  #[inline]
  fn new() -> Self {
    Self::default()
  }

  fn update(&mut self, mut data: &[u8]) {
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
        self.bytes_hashed = self.bytes_hashed.wrapping_add(BLOCK_LEN as u128);
        compress(&mut self.h, &self.buf, self.bytes_hashed, false);
        self.buf_len = 0;
      }
    }

    let (blocks, rest) = data.as_chunks::<BLOCK_LEN>();
    if !blocks.is_empty() {
      // If `rest` is empty, hold back the last full block for finalization.
      let (to_compress, last_full) = if rest.is_empty() {
        (&blocks[..blocks.len() - 1], Some(blocks[blocks.len() - 1]))
      } else {
        (blocks, None)
      };

      for block in to_compress {
        self.bytes_hashed = self.bytes_hashed.wrapping_add(BLOCK_LEN as u128);
        compress(&mut self.h, block, self.bytes_hashed, false);
      }

      if let Some(last) = last_full {
        self.buf.copy_from_slice(&last);
        self.buf_len = BLOCK_LEN;
      }
    }
    data = rest;

    if !data.is_empty() {
      self.buf[..data.len()].copy_from_slice(data);
      self.buf_len = data.len();
    }
  }

  fn finalize(&self) -> Self::Output {
    let mut h = self.h;
    let mut buf = self.buf;
    let len = self.buf_len;

    buf[len..].fill(0);
    let t = self.bytes_hashed.wrapping_add(len as u128);
    compress(&mut h, &buf, t, true);

    let mut out = [0u8; 64];
    for (i, word) in h.iter().copied().enumerate() {
      out[i * 8..i * 8 + 8].copy_from_slice(&word.to_le_bytes());
    }
    out
  }

  #[inline]
  fn reset(&mut self) {
    *self = Self::default();
  }
}
