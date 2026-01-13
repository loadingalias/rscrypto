//! Ascon hash and XOF (NIST LWC).
//!
//! Portable, `no_std`, pure Rust implementation.

#![allow(clippy::indexing_slicing)] // Fixed-size state + sponge buffering

use traits::{Digest, Xof};

use crate::crypto::dispatch_util::SizeClassDispatch;

#[doc(hidden)]
pub mod dispatch;
#[doc(hidden)]
pub mod dispatch_tables;
pub(crate) mod kernels;

const RATE: usize = 8;

trait Permuter: Copy + Default {
  fn permute(self, state: &mut [u64; 5], len_hint: usize);
}

#[derive(Clone, Copy)]
struct DispatchPermuter {
  dispatch: SizeClassDispatch<fn(&mut [u64; 5])>,
}

impl Default for DispatchPermuter {
  #[inline]
  fn default() -> Self {
    Self {
      dispatch: dispatch::permute_dispatch(),
    }
  }
}

impl Permuter for DispatchPermuter {
  #[inline(always)]
  fn permute(self, state: &mut [u64; 5], len_hint: usize) {
    (self.dispatch.select(len_hint))(state);
  }
}

#[derive(Clone, Copy, Default)]
struct PortablePermuter;

impl Permuter for PortablePermuter {
  #[inline(always)]
  fn permute(self, state: &mut [u64; 5], _len_hint: usize) {
    permute_12_portable(state);
  }
}

// Ascon permutation round constants (12 rounds).
const RC: [u64; 12] = [0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87, 0x78, 0x69, 0x5A, 0x4B];

// Domain-specific IVs (from the Ascon hash/XOF specification).
const HASH256_IV: [u64; 5] = [
  0x9b1e_5494_e934_d681,
  0x4bc3_a01e_3337_51d2,
  0xae65_396c_6b34_b81a,
  0x3c7f_d4a4_d56a_4db3,
  0x1a5c_4649_06c5_976d,
];

const XOF128_IV: [u64; 5] = [
  0xda82_ce76_8d94_47eb,
  0xcc7c_e6c7_5f1e_f969,
  0xe750_8fd7_8008_5631,
  0x0ee0_ea53_416b_58cc,
  0xe054_7524_db6f_0bde,
];

#[inline(always)]
const fn pad(n: usize) -> u64 {
  // Produce the padding mask used by the reference construction:
  // XOR `pad(len)` into state[0], with state interpreted little-endian.
  0x01_u64 << (8 * n)
}

#[inline(always)]
pub(crate) fn permute_12_portable(s: &mut [u64; 5]) {
  for &c in &RC {
    round(s, c);
  }
}

#[inline(always)]
#[allow(dead_code)]
fn permute_12(s: &mut [u64; 5]) {
  dispatch::permute_12(s);
}

#[inline(always)]
fn round(s: &mut [u64; 5], c: u64) {
  let mut x0 = s[0];
  let mut x1 = s[1];
  let mut x2 = s[2];
  let mut x3 = s[3];
  let mut x4 = s[4];

  // Add round constant.
  x2 ^= c;

  // Substitution layer.
  x0 ^= x4;
  x4 ^= x3;
  x2 ^= x1;

  let t0 = (!x0) & x1;
  let t1 = (!x1) & x2;
  let t2 = (!x2) & x3;
  let t3 = (!x3) & x4;
  let t4 = (!x4) & x0;

  x0 ^= t1;
  x1 ^= t2;
  x2 ^= t3;
  x3 ^= t4;
  x4 ^= t0;

  x1 ^= x0;
  x0 ^= x4;
  x3 ^= x2;
  x2 = !x2;

  // Linear diffusion layer.
  x0 ^= x0.rotate_right(19) ^ x0.rotate_right(28);
  x1 ^= x1.rotate_right(61) ^ x1.rotate_right(39);
  x2 ^= x2.rotate_right(1) ^ x2.rotate_right(6);
  x3 ^= x3.rotate_right(10) ^ x3.rotate_right(17);
  x4 ^= x4.rotate_right(7) ^ x4.rotate_right(41);

  s[0] = x0;
  s[1] = x1;
  s[2] = x2;
  s[3] = x3;
  s[4] = x4;
}

#[derive(Clone)]
struct Sponge<P: Permuter, const IV0: u64, const IV1: u64, const IV2: u64, const IV3: u64, const IV4: u64> {
  state: [u64; 5],
  buf: [u8; RATE],
  buf_len: usize,
  bytes_hint: usize,
  permuter: P,
}

impl<P: Permuter, const IV0: u64, const IV1: u64, const IV2: u64, const IV3: u64, const IV4: u64> Default
  for Sponge<P, IV0, IV1, IV2, IV3, IV4>
{
  #[inline]
  fn default() -> Self {
    Self {
      state: [IV0, IV1, IV2, IV3, IV4],
      buf: [0u8; RATE],
      buf_len: 0,
      bytes_hint: 0,
      permuter: P::default(),
    }
  }
}

impl<P: Permuter, const IV0: u64, const IV1: u64, const IV2: u64, const IV3: u64, const IV4: u64>
  Sponge<P, IV0, IV1, IV2, IV3, IV4>
{
  #[inline(always)]
  fn absorb_block(&mut self, block: &[u8; RATE]) {
    self.state[0] ^= u64::from_le_bytes(*block);
    let permuter = self.permuter;
    let len_hint = self.bytes_hint;
    permuter.permute(&mut self.state, len_hint);
  }

  fn update(&mut self, mut data: &[u8]) {
    if data.is_empty() {
      return;
    }

    self.bytes_hint = self.bytes_hint.wrapping_add(data.len());
    if self.buf_len != 0 {
      let take = core::cmp::min(RATE - self.buf_len, data.len());
      self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
      self.buf_len += take;
      data = &data[take..];

      if self.buf_len == RATE {
        let block = self.buf;
        self.absorb_block(&block);
        self.buf_len = 0;
      }
    }

    let (blocks, rest) = data.as_chunks::<RATE>();
    for block in blocks {
      self.absorb_block(block);
    }
    data = rest;

    if !data.is_empty() {
      self.buf[..data.len()].copy_from_slice(data);
      self.buf_len = data.len();
    }
  }

  fn finalize_state(&self) -> [u64; 5] {
    let mut st = self.state;
    let permuter = self.permuter;
    let len_hint = self.bytes_hint;
    let last = &self.buf[..self.buf_len];

    let mut tmp = [0u8; RATE];
    tmp[..last.len()].copy_from_slice(last);
    st[0] ^= u64::from_le_bytes(tmp);
    st[0] ^= pad(last.len());
    permuter.permute(&mut st, len_hint);

    st
  }

  fn finalize_state_with_hint(&self) -> ([u64; 5], usize) {
    (self.finalize_state(), self.bytes_hint)
  }
}

/// Ascon-Hash256.
#[derive(Clone, Default)]
pub struct AsconHash256 {
  sponge: Sponge<
    DispatchPermuter,
    { HASH256_IV[0] },
    { HASH256_IV[1] },
    { HASH256_IV[2] },
    { HASH256_IV[3] },
    { HASH256_IV[4] },
  >,
}

impl AsconHash256 {
  #[inline]
  #[must_use]
  pub(crate) fn digest_portable(data: &[u8]) -> [u8; 32] {
    let mut sponge: Sponge<
      PortablePermuter,
      { HASH256_IV[0] },
      { HASH256_IV[1] },
      { HASH256_IV[2] },
      { HASH256_IV[3] },
      { HASH256_IV[4] },
    > = Sponge::default();
    sponge.update(data);

    let (mut st, _) = sponge.finalize_state_with_hint();
    let mut out = [0u8; 32];
    let mut off = 0usize;
    while off < 24 {
      out[off..off + 8].copy_from_slice(&st[0].to_le_bytes());
      permute_12_portable(&mut st);
      off += 8;
    }
    out[24..32].copy_from_slice(&st[0].to_le_bytes());
    out
  }
}

impl Digest for AsconHash256 {
  const OUTPUT_SIZE: usize = 32;
  type Output = [u8; 32];

  #[inline]
  fn new() -> Self {
    Self::default()
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.sponge.update(data);
  }

  fn finalize(&self) -> Self::Output {
    let (mut st, hint) = self.sponge.finalize_state_with_hint();

    let mut out = [0u8; 32];
    let mut off = 0usize;
    while off < 24 {
      out[off..off + 8].copy_from_slice(&st[0].to_le_bytes());
      dispatch::permute_12_for_len(&mut st, hint.wrapping_add(off));
      off += 8;
    }
    out[24..32].copy_from_slice(&st[0].to_le_bytes());
    out
  }

  #[inline]
  fn reset(&mut self) {
    *self = Self::default();
  }
}

/// Ascon-XOF128 hasher.
#[derive(Clone, Default)]
pub struct AsconXof128 {
  sponge:
    Sponge<DispatchPermuter, { XOF128_IV[0] }, { XOF128_IV[1] }, { XOF128_IV[2] }, { XOF128_IV[3] }, { XOF128_IV[4] }>,
}

impl AsconXof128 {
  #[inline]
  #[must_use]
  pub fn new() -> Self {
    Self::default()
  }

  #[inline]
  pub fn update(&mut self, data: &[u8]) {
    self.sponge.update(data);
  }

  #[inline]
  pub fn reset(&mut self) {
    *self = Self::default();
  }

  #[inline]
  #[must_use]
  pub fn finalize_xof(&self) -> AsconXof128Xof {
    let (state, hint) = self.sponge.finalize_state_with_hint();
    AsconXof128Xof {
      state,
      buf: [0u8; RATE],
      pos: RATE,
      hint,
      bytes_out: 0,
    }
  }

  #[inline]
  pub fn hash_into(data: &[u8], out: &mut [u8]) {
    let mut h = Self::new();
    h.update(data);
    let mut xof = h.finalize_xof();
    xof.squeeze(out);
  }

  #[inline]
  pub(crate) fn hash_into_portable(data: &[u8], mut out: &mut [u8]) {
    let mut sponge: Sponge<
      PortablePermuter,
      { XOF128_IV[0] },
      { XOF128_IV[1] },
      { XOF128_IV[2] },
      { XOF128_IV[3] },
      { XOF128_IV[4] },
    > = Sponge::default();
    sponge.update(data);
    let mut state = sponge.finalize_state();

    let mut buf = [0u8; RATE];
    let mut pos = RATE;
    while !out.is_empty() {
      if pos == RATE {
        buf = state[0].to_le_bytes();
        permute_12_portable(&mut state);
        pos = 0;
      }

      let take = core::cmp::min(RATE - pos, out.len());
      out[..take].copy_from_slice(&buf[pos..pos + take]);
      out = &mut out[take..];
      pos += take;
    }
  }
}

/// Ascon-XOF128 reader.
#[derive(Clone)]
pub struct AsconXof128Xof {
  state: [u64; 5],
  buf: [u8; RATE],
  pos: usize,
  hint: usize,
  bytes_out: usize,
}

impl AsconXof128Xof {
  #[inline(always)]
  fn refill(&mut self) {
    self.buf = self.state[0].to_le_bytes();
    dispatch::permute_12_for_len(&mut self.state, self.hint.wrapping_add(self.bytes_out));
    self.pos = 0;
  }
}

impl Xof for AsconXof128Xof {
  fn squeeze(&mut self, mut out: &mut [u8]) {
    while !out.is_empty() {
      if self.pos == RATE {
        self.refill();
      }

      let take = core::cmp::min(RATE - self.pos, out.len());
      out[..take].copy_from_slice(&self.buf[self.pos..self.pos + take]);
      self.bytes_out = self.bytes_out.wrapping_add(take);
      self.pos += take;
      out = &mut out[take..];
    }
  }
}

#[cfg(feature = "std")]
pub(crate) mod kernel_test;
