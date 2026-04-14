//! SHA-256 (FIPS 180-4).

#![allow(clippy::indexing_slicing)] // Fixed-size arrays + compression schedule

use self::kernels::CompressBlocksFn;
use crate::{
  hashes::{
    crypto::dispatch_util::{SizeClassDispatch, len_hint_from_u64},
    util::rotr32,
  },
  traits::Digest,
};

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;
#[doc(hidden)]
pub(crate) mod dispatch;
#[doc(hidden)]
pub(crate) mod dispatch_tables;
pub(crate) mod kernels;
#[cfg(target_arch = "powerpc64")]
pub(crate) mod ppc64;
#[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
pub(crate) mod riscv64;
#[cfg(target_arch = "s390x")]
pub(crate) mod s390x;
#[cfg(target_arch = "wasm32")]
pub(crate) mod wasm;
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64;

pub(crate) const BLOCK_LEN: usize = 64;

pub(crate) const H0: [u32; 8] = [
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

#[repr(C, align(64))]
struct AlignedK([u32; 64]);

impl core::ops::Deref for AlignedK {
  type Target = [u32; 64];
  #[inline(always)]
  fn deref(&self) -> &[u32; 64] {
    &self.0
  }
}

static K: AlignedK = AlignedK([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
  0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
  0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
  0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
  0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
  0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
  0xc67178f2,
]);

#[inline(always)]
fn ch(x: u32, y: u32, z: u32) -> u32 {
  (x & y) ^ (!x & z)
}

#[inline(always)]
fn maj(x: u32, y: u32, z: u32) -> u32 {
  (x & y) ^ (x & z) ^ (y & z)
}

#[inline(always)]
fn big_sigma0(x: u32) -> u32 {
  rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22)
}

#[inline(always)]
fn big_sigma1(x: u32) -> u32 {
  rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25)
}

#[inline(always)]
fn small_sigma0(x: u32) -> u32 {
  rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3)
}

#[inline(always)]
fn small_sigma1(x: u32) -> u32 {
  rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10)
}

/// Read SHA-256 round constant K[i].
///
/// On x86/x86_64, 32-bit constants can be encoded as immediate operands in
/// `add r32, imm32`, so inlining is optimal. On all other architectures
/// (POWER, aarch64 portable, s390x, RISC-V), materializing a 32-bit
/// immediate requires 2+ instructions (`lis`+`ori` on POWER, `movz`+`movk`
/// on ARM), so loading from the static K array via a single load instruction
/// is faster. `read_volatile` forces the compiler to emit a memory load
/// instead of inlining the constant.
///
/// This matches the `sha2` crate's `rk()` strategy and closes the 15-20%
/// gap on POWER10.
#[inline(always)]
fn rk(i: usize) -> u32 {
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  {
    // SAFETY: i is always in 0..64, and K has exactly 64 elements.
    unsafe { core::ptr::read(K.0.as_ptr().add(i)) }
  }
  #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
  {
    // SAFETY: i is always in 0..64, and K has exactly 64 elements.
    unsafe { core::ptr::read_volatile(K.0.as_ptr().add(i)) }
  }
}

/// Core SHA-256 block compression, parameterized over the four sigma/sum
/// operations. The portable kernel passes software rotate-xor-shift; RISC-V
/// Zknh passes hardware `sha256sum0/1` and `sha256sig0/1` intrinsics.
///
/// The compiler devirtualizes constant `fn` items at all optimization levels,
/// producing identical codegen to a hand-inlined version.
#[inline(always)]
pub(crate) fn compress_block_with(
  state: &mut [u32; 8],
  block: &[u8; BLOCK_LEN],
  big_s0: fn(u32) -> u32,
  big_s1: fn(u32) -> u32,
  small_s0: fn(u32) -> u32,
  small_s1: fn(u32) -> u32,
) {
  // 16-word ring buffer message schedule (lower memory traffic than a full
  // 64-word schedule, and typically faster in practice).
  //
  // Fully unrolled to avoid bounds checks and allow better instruction
  // scheduling in the scalar core.
  let mut w = [0u32; 16];
  let (chunks, _) = block.as_chunks::<4>();
  for (i, c) in chunks.iter().enumerate() {
    w[i] = u32::from_be_bytes(*c);
  }
  let [
    mut w0,
    mut w1,
    mut w2,
    mut w3,
    mut w4,
    mut w5,
    mut w6,
    mut w7,
    mut w8,
    mut w9,
    mut w10,
    mut w11,
    mut w12,
    mut w13,
    mut w14,
    mut w15,
  ] = w;

  let mut a = state[0];
  let mut b = state[1];
  let mut c = state[2];
  let mut d = state[3];
  let mut e = state[4];
  let mut f = state[5];
  let mut g = state[6];
  let mut h = state[7];

  macro_rules! round {
    ($k:expr, $wi:expr) => {{
      let t1 = h
        .wrapping_add(big_s1(e))
        .wrapping_add(ch(e, f, g))
        .wrapping_add($k)
        .wrapping_add($wi);
      let t2 = big_s0(a).wrapping_add(maj(a, b, c));

      h = g;
      g = f;
      f = e;
      e = d.wrapping_add(t1);
      d = c;
      c = b;
      b = a;
      a = t1.wrapping_add(t2);
    }};
  }

  macro_rules! sched {
    ($w_im2:expr, $w_im7:expr, $w_im15:expr, $w_im16:expr) => {{
      small_s1($w_im2)
        .wrapping_add($w_im7)
        .wrapping_add(small_s0($w_im15))
        .wrapping_add($w_im16)
    }};
  }

  round!(rk(0), w0);
  round!(rk(1), w1);
  round!(rk(2), w2);
  round!(rk(3), w3);
  round!(rk(4), w4);
  round!(rk(5), w5);
  round!(rk(6), w6);
  round!(rk(7), w7);
  round!(rk(8), w8);
  round!(rk(9), w9);
  round!(rk(10), w10);
  round!(rk(11), w11);
  round!(rk(12), w12);
  round!(rk(13), w13);
  round!(rk(14), w14);
  round!(rk(15), w15);

  w0 = sched!(w14, w9, w1, w0);
  round!(rk(16), w0);
  w1 = sched!(w15, w10, w2, w1);
  round!(rk(17), w1);
  w2 = sched!(w0, w11, w3, w2);
  round!(rk(18), w2);
  w3 = sched!(w1, w12, w4, w3);
  round!(rk(19), w3);
  w4 = sched!(w2, w13, w5, w4);
  round!(rk(20), w4);
  w5 = sched!(w3, w14, w6, w5);
  round!(rk(21), w5);
  w6 = sched!(w4, w15, w7, w6);
  round!(rk(22), w6);
  w7 = sched!(w5, w0, w8, w7);
  round!(rk(23), w7);
  w8 = sched!(w6, w1, w9, w8);
  round!(rk(24), w8);
  w9 = sched!(w7, w2, w10, w9);
  round!(rk(25), w9);
  w10 = sched!(w8, w3, w11, w10);
  round!(rk(26), w10);
  w11 = sched!(w9, w4, w12, w11);
  round!(rk(27), w11);
  w12 = sched!(w10, w5, w13, w12);
  round!(rk(28), w12);
  w13 = sched!(w11, w6, w14, w13);
  round!(rk(29), w13);
  w14 = sched!(w12, w7, w15, w14);
  round!(rk(30), w14);
  w15 = sched!(w13, w8, w0, w15);
  round!(rk(31), w15);
  w0 = sched!(w14, w9, w1, w0);
  round!(rk(32), w0);
  w1 = sched!(w15, w10, w2, w1);
  round!(rk(33), w1);
  w2 = sched!(w0, w11, w3, w2);
  round!(rk(34), w2);
  w3 = sched!(w1, w12, w4, w3);
  round!(rk(35), w3);
  w4 = sched!(w2, w13, w5, w4);
  round!(rk(36), w4);
  w5 = sched!(w3, w14, w6, w5);
  round!(rk(37), w5);
  w6 = sched!(w4, w15, w7, w6);
  round!(rk(38), w6);
  w7 = sched!(w5, w0, w8, w7);
  round!(rk(39), w7);
  w8 = sched!(w6, w1, w9, w8);
  round!(rk(40), w8);
  w9 = sched!(w7, w2, w10, w9);
  round!(rk(41), w9);
  w10 = sched!(w8, w3, w11, w10);
  round!(rk(42), w10);
  w11 = sched!(w9, w4, w12, w11);
  round!(rk(43), w11);
  w12 = sched!(w10, w5, w13, w12);
  round!(rk(44), w12);
  w13 = sched!(w11, w6, w14, w13);
  round!(rk(45), w13);
  w14 = sched!(w12, w7, w15, w14);
  round!(rk(46), w14);
  w15 = sched!(w13, w8, w0, w15);
  round!(rk(47), w15);
  w0 = sched!(w14, w9, w1, w0);
  round!(rk(48), w0);
  w1 = sched!(w15, w10, w2, w1);
  round!(rk(49), w1);
  w2 = sched!(w0, w11, w3, w2);
  round!(rk(50), w2);
  w3 = sched!(w1, w12, w4, w3);
  round!(rk(51), w3);
  w4 = sched!(w2, w13, w5, w4);
  round!(rk(52), w4);
  w5 = sched!(w3, w14, w6, w5);
  round!(rk(53), w5);
  w6 = sched!(w4, w15, w7, w6);
  round!(rk(54), w6);
  w7 = sched!(w5, w0, w8, w7);
  round!(rk(55), w7);
  w8 = sched!(w6, w1, w9, w8);
  round!(rk(56), w8);
  w9 = sched!(w7, w2, w10, w9);
  round!(rk(57), w9);
  w10 = sched!(w8, w3, w11, w10);
  round!(rk(58), w10);
  w11 = sched!(w9, w4, w12, w11);
  round!(rk(59), w11);
  w12 = sched!(w10, w5, w13, w12);
  round!(rk(60), w12);
  w13 = sched!(w11, w6, w14, w13);
  round!(rk(61), w13);
  w14 = sched!(w12, w7, w15, w14);
  round!(rk(62), w14);
  w15 = sched!(w13, w8, w0, w15);
  round!(rk(63), w15);

  state[0] = state[0].wrapping_add(a);
  state[1] = state[1].wrapping_add(b);
  state[2] = state[2].wrapping_add(c);
  state[3] = state[3].wrapping_add(d);
  state[4] = state[4].wrapping_add(e);
  state[5] = state[5].wrapping_add(f);
  state[6] = state[6].wrapping_add(g);
  state[7] = state[7].wrapping_add(h);
}

/// Maximum message length in bytes for SHA-256 (FIPS 180-4).
///
/// The spec encodes message length as a 64-bit **bit** count, so the maximum
/// byte length is `(2^64 − 1) / 8 = 2^61 − 1` (~2.3 exabytes). Beyond this,
/// the bit-length field wraps and the digest is silently incorrect.
const MAX_MESSAGE_LEN: u64 = u64::MAX / 8;

#[derive(Clone)]
pub struct Sha256 {
  state: [u32; 8],
  block: [u8; BLOCK_LEN],
  block_len: usize,
  bytes_hashed: u64,
  compress_blocks: CompressBlocksFn,
  dispatch: Option<SizeClassDispatch<CompressBlocksFn>>,
}

#[derive(Clone, Copy)]
#[cfg(feature = "hmac")]
pub(crate) struct Sha256Prefix {
  state: [u32; 8],
  bytes_hashed: u64,
  compress_blocks: CompressBlocksFn,
  dispatch: Option<SizeClassDispatch<CompressBlocksFn>>,
}

impl core::fmt::Debug for Sha256 {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Sha256").finish_non_exhaustive()
  }
}

impl Default for Sha256 {
  #[inline]
  fn default() -> Self {
    Self {
      state: H0,
      block: [0u8; BLOCK_LEN],
      block_len: 0,
      bytes_hashed: 0,
      compress_blocks: kernels::compile_time_best(),
      dispatch: None,
    }
  }
}

impl Sha256 {
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
  pub(crate) fn compress_blocks_portable(state: &mut [u32; 8], blocks: &[u8]) {
    debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);
    let mut chunks = blocks.chunks_exact(BLOCK_LEN);
    for chunk in &mut chunks {
      // SAFETY: `chunks_exact(BLOCK_LEN)` yields slices of exactly `BLOCK_LEN` bytes.
      let block = unsafe { &*(chunk.as_ptr() as *const [u8; BLOCK_LEN]) };
      Self::compress_block(state, block);
    }
    debug_assert!(chunks.remainder().is_empty());
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
      "SHA-256: total input exceeds FIPS 180-4 maximum of 2^61 − 1 bytes"
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

  #[inline(always)]
  fn compress_block(state: &mut [u32; 8], block: &[u8; BLOCK_LEN]) {
    compress_block_with(state, block, big_sigma0, big_sigma1, small_sigma0, small_sigma1);
  }

  #[inline]
  fn finalize_inner_with(&self, compress_blocks: CompressBlocksFn) -> [u8; 32] {
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

    let mut out = [0u8; 32];
    for (chunk, &word) in out.chunks_exact_mut(4).zip(state.iter()) {
      chunk.copy_from_slice(&word.to_be_bytes());
    }
    out
  }

  #[inline]
  #[must_use]
  #[cfg(feature = "hmac")]
  pub(crate) fn aligned_prefix(&self) -> Sha256Prefix {
    debug_assert_eq!(self.block_len, 0);
    Sha256Prefix {
      state: self.state,
      bytes_hashed: self.bytes_hashed,
      compress_blocks: self.compress_blocks,
      dispatch: self.dispatch,
    }
  }

  #[inline]
  #[must_use]
  #[cfg(feature = "hmac")]
  pub(crate) fn from_aligned_prefix(prefix: Sha256Prefix) -> Self {
    Self {
      state: prefix.state,
      block: [0u8; BLOCK_LEN],
      block_len: 0,
      bytes_hashed: prefix.bytes_hashed,
      compress_blocks: prefix.compress_blocks,
      dispatch: prefix.dispatch,
    }
  }

  #[inline]
  #[cfg(feature = "hmac")]
  pub(crate) fn reset_to_aligned_prefix(&mut self, prefix: Sha256Prefix) {
    self.state = prefix.state;
    self.block_len = 0;
    self.bytes_hashed = prefix.bytes_hashed;
    self.compress_blocks = prefix.compress_blocks;
    self.dispatch = prefix.dispatch;
  }
}

impl Drop for Sha256 {
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

impl Digest for Sha256 {
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
    if kernels::COMPILE_TIME_HW {
      self.update_with(data, kernels::compile_time_best());
      return;
    }
    let compress = self.select_compress(data.len());
    self.update_with(data, compress);
  }

  #[inline]
  fn finalize(&self) -> Self::Output {
    if kernels::COMPILE_TIME_HW {
      return self.finalize_inner_with(kernels::compile_time_best());
    }
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

#[cfg(test)]
mod tests {
  use super::Sha256;

  fn hex32(bytes: &[u8; 32]) -> alloc::string::String {
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
    // NIST FIPS 180-4 test vectors (short messages).
    assert_eq!(
      hex32(&Sha256::digest(b"")),
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
      hex32(&Sha256::digest(b"abc")),
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
      hex32(&Sha256::digest(
        b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
      )),
      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );

    // 1,000,000 repetitions of 'a'.
    let mut million_a = alloc::vec![b'a'; 1_000_000];
    assert_eq!(
      hex32(&Sha256::digest(&million_a)),
      "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
    million_a.clear();
  }
}
