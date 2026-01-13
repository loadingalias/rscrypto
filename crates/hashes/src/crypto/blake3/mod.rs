//! BLAKE3 (hash + XOF).
//!
//! This is a portable, dependency-free implementation suitable for `no_std`.

#![allow(clippy::indexing_slicing)] // Fixed-size arrays + internal block parsing

use core::{cmp::min, ptr};

use traits::{Digest, Xof};

#[doc(hidden)]
pub mod dispatch;
#[doc(hidden)]
pub mod dispatch_tables;
pub(crate) mod kernels;

use self::kernels::Kernel;
use crate::crypto::dispatch_util::{SizeClassDispatch, len_hint_from_u128};

const OUT_LEN: usize = 32;
const KEY_LEN: usize = 32;
const BLOCK_LEN: usize = 64;
const CHUNK_LEN: usize = 1024;

const CHUNK_START: u32 = 1 << 0;
const CHUNK_END: u32 = 1 << 1;
const PARENT: u32 = 1 << 2;
const ROOT: u32 = 1 << 3;
const KEYED_HASH: u32 = 1 << 4;
const DERIVE_KEY_CONTEXT: u32 = 1 << 5;
const DERIVE_KEY_MATERIAL: u32 = 1 << 6;

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

#[inline(always)]
fn words8_from_le_bytes_32(bytes: &[u8; 32]) -> [u32; 8] {
  let src = bytes.as_ptr() as *const u32;
  let mut out = [0u32; 8];
  let mut i = 0usize;
  while i < 8 {
    // SAFETY: `bytes` is exactly 32 bytes, so it contains 8 `u32` values;
    // `read_unaligned` supports the 1-byte alignment of `[u8; 32]`.
    out[i] = u32::from_le(unsafe { ptr::read_unaligned(src.add(i)) });
    i += 1;
  }
  out
}

#[inline(always)]
fn words16_from_le_bytes_64(bytes: &[u8; 64]) -> [u32; 16] {
  let src = bytes.as_ptr() as *const u32;
  let mut out = [0u32; 16];
  let mut i = 0usize;
  while i < 16 {
    // SAFETY: `bytes` is exactly 64 bytes, so it contains 16 `u32` values;
    // `read_unaligned` supports the 1-byte alignment of `[u8; 64]`.
    out[i] = u32::from_le(unsafe { ptr::read_unaligned(src.add(i)) });
    i += 1;
  }
  out
}

#[inline]
fn compress(chaining_value: &[u32; 8], block_words: &[u32; 16], counter: u64, block_len: u32, flags: u32) -> [u32; 16] {
  let m0 = block_words[0];
  let m1 = block_words[1];
  let m2 = block_words[2];
  let m3 = block_words[3];
  let m4 = block_words[4];
  let m5 = block_words[5];
  let m6 = block_words[6];
  let m7 = block_words[7];
  let m8 = block_words[8];
  let m9 = block_words[9];
  let m10 = block_words[10];
  let m11 = block_words[11];
  let m12 = block_words[12];
  let m13 = block_words[13];
  let m14 = block_words[14];
  let m15 = block_words[15];

  let counter_low = counter as u32;
  let counter_high = (counter >> 32) as u32;
  let mut v0 = chaining_value[0];
  let mut v1 = chaining_value[1];
  let mut v2 = chaining_value[2];
  let mut v3 = chaining_value[3];
  let mut v4 = chaining_value[4];
  let mut v5 = chaining_value[5];
  let mut v6 = chaining_value[6];
  let mut v7 = chaining_value[7];
  let mut v8 = IV[0];
  let mut v9 = IV[1];
  let mut v10 = IV[2];
  let mut v11 = IV[3];
  let mut v12 = counter_low;
  let mut v13 = counter_high;
  let mut v14 = block_len;
  let mut v15 = flags;

  macro_rules! g {
    ($a:ident, $b:ident, $c:ident, $d:ident, $mx:expr, $my:expr) => {{
      $a = $a.wrapping_add($b).wrapping_add($mx);
      $d = ($d ^ $a).rotate_right(16);
      $c = $c.wrapping_add($d);
      $b = ($b ^ $c).rotate_right(12);
      $a = $a.wrapping_add($b).wrapping_add($my);
      $d = ($d ^ $a).rotate_right(8);
      $c = $c.wrapping_add($d);
      $b = ($b ^ $c).rotate_right(7);
    }};
  }

  // One full BLAKE3 round, with an explicit message schedule. This lets the
  // compiler keep `v0..v15` and `m0..m15` in registers without indirect
  // indexing in the hottest loop.
  macro_rules! round {
    (
      $m0:expr, $m1:expr, $m2:expr, $m3:expr, $m4:expr, $m5:expr, $m6:expr, $m7:expr,
      $m8:expr, $m9:expr, $m10:expr, $m11:expr, $m12:expr, $m13:expr, $m14:expr, $m15:expr
    ) => {{
      g!(v0, v4, v8, v12, $m0, $m1);
      g!(v1, v5, v9, v13, $m2, $m3);
      g!(v2, v6, v10, v14, $m4, $m5);
      g!(v3, v7, v11, v15, $m6, $m7);

      g!(v0, v5, v10, v15, $m8, $m9);
      g!(v1, v6, v11, v12, $m10, $m11);
      g!(v2, v7, v8, v13, $m12, $m13);
      g!(v3, v4, v9, v14, $m14, $m15);
    }};
  }

  // Per-round schedules for the 7-round BLAKE3 compression function.
  round!(m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15);
  round!(m2, m6, m3, m10, m7, m0, m4, m13, m1, m11, m12, m5, m9, m14, m15, m8);
  round!(m3, m4, m10, m12, m13, m2, m7, m14, m6, m5, m9, m0, m11, m15, m8, m1);
  round!(m10, m7, m12, m9, m14, m3, m13, m15, m4, m0, m11, m2, m5, m8, m1, m6);
  round!(m12, m13, m9, m11, m15, m10, m14, m8, m7, m2, m5, m3, m0, m1, m6, m4);
  round!(m9, m14, m11, m5, m8, m12, m15, m1, m13, m3, m0, m10, m2, m6, m4, m7);
  round!(m11, m15, m5, m0, m1, m9, m8, m6, m14, m10, m2, m12, m3, m4, m7, m13);

  v0 ^= v8;
  v1 ^= v9;
  v2 ^= v10;
  v3 ^= v11;
  v4 ^= v12;
  v5 ^= v13;
  v6 ^= v14;
  v7 ^= v15;

  v8 ^= chaining_value[0];
  v9 ^= chaining_value[1];
  v10 ^= chaining_value[2];
  v11 ^= chaining_value[3];
  v12 ^= chaining_value[4];
  v13 ^= chaining_value[5];
  v14 ^= chaining_value[6];
  v15 ^= chaining_value[7];

  [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15]
}

#[inline(always)]
fn first_8_words(words: [u32; 16]) -> [u32; 8] {
  // SAFETY: fixed-size arrays
  [
    words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7],
  ]
}

#[derive(Clone, Copy)]
struct OutputState {
  kernel: Kernel,
  input_chaining_value: [u32; 8],
  block_words: [u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
}

impl OutputState {
  #[inline]
  fn chaining_value(&self) -> [u32; 8] {
    first_8_words((self.kernel.compress)(
      &self.input_chaining_value,
      &self.block_words,
      self.counter,
      self.block_len,
      self.flags,
    ))
  }

  #[inline]
  fn root_output_block_bytes(&self, output_block_counter: u64) -> [u8; 2 * OUT_LEN] {
    let words = (self.kernel.compress)(
      &self.input_chaining_value,
      &self.block_words,
      output_block_counter,
      self.block_len,
      self.flags | ROOT,
    );

    let mut out = [0u8; 2 * OUT_LEN];
    for (i, word) in words.iter().copied().enumerate() {
      let offset = i * 4;
      out[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
    }
    out
  }
}

#[derive(Clone, Copy)]
struct ChunkState {
  kernel: Kernel,
  chaining_value: [u32; 8],
  chunk_counter: u64,
  block: [u8; BLOCK_LEN],
  block_len: u8,
  blocks_compressed: u8,
  flags: u32,
}

impl ChunkState {
  #[inline]
  fn new(key_words: [u32; 8], chunk_counter: u64, flags: u32, kernel: Kernel) -> Self {
    Self {
      kernel,
      chaining_value: key_words,
      chunk_counter,
      block: [0u8; BLOCK_LEN],
      block_len: 0,
      blocks_compressed: 0,
      flags,
    }
  }

  #[inline]
  fn len(&self) -> usize {
    BLOCK_LEN * self.blocks_compressed as usize + self.block_len as usize
  }

  #[inline]
  fn start_flag(&self) -> u32 {
    if self.blocks_compressed == 0 { CHUNK_START } else { 0 }
  }

  fn update(&mut self, mut input: &[u8]) {
    while !input.is_empty() {
      if self.block_len as usize == BLOCK_LEN {
        debug_assert!(
          self.blocks_compressed < 15,
          "last chunk block stays buffered until output()"
        );
        (self.kernel.chunk_compress_blocks)(
          &mut self.chaining_value,
          self.chunk_counter,
          self.flags,
          &mut self.blocks_compressed,
          &self.block,
        );
        self.block_len = 0;
      }

      // Fast path: compress full blocks directly from the caller slice.
      //
      // We intentionally leave the *last* block of a full chunk buffered so
      // that the caller can decide whether this chunk is terminal (and thus
      // needs CHUNK_END at finalize) or will be followed by more input.
      if self.block_len == 0 && self.blocks_compressed < 15 && input.len() >= BLOCK_LEN {
        let max_blocks = 15usize - self.blocks_compressed as usize;
        let blocks_available = input.len() / BLOCK_LEN;
        let has_partial = !input.len().is_multiple_of(BLOCK_LEN);
        // Leave one block buffered when the input ends exactly on a block
        // boundary, so that finalize can apply CHUNK_END to the last block.
        let mut blocks_to_compress = if has_partial {
          blocks_available
        } else {
          blocks_available.saturating_sub(1)
        };
        blocks_to_compress = core::cmp::min(blocks_to_compress, max_blocks);
        if blocks_to_compress != 0 {
          let bytes = blocks_to_compress * BLOCK_LEN;
          (self.kernel.chunk_compress_blocks)(
            &mut self.chaining_value,
            self.chunk_counter,
            self.flags,
            &mut self.blocks_compressed,
            &input[..bytes],
          );
          input = &input[bytes..];
          continue;
        }
      }

      let want = BLOCK_LEN - self.block_len as usize;
      let take = min(want, input.len());
      self.block[self.block_len as usize..][..take].copy_from_slice(&input[..take]);
      self.block_len = self.block_len.wrapping_add(take as u8);
      input = &input[take..];
    }
  }

  #[inline]
  fn output(&self) -> OutputState {
    let mut block = self.block;
    if self.block_len as usize != BLOCK_LEN {
      block[self.block_len as usize..].fill(0);
    }
    let block_words = words16_from_le_bytes_64(&block);
    OutputState {
      kernel: self.kernel,
      input_chaining_value: self.chaining_value,
      block_words,
      counter: self.chunk_counter,
      block_len: self.block_len as u32,
      flags: self.flags | self.start_flag() | CHUNK_END,
    }
  }
}

#[inline]
fn parent_output(
  kernel: Kernel,
  left_child_cv: [u32; 8],
  right_child_cv: [u32; 8],
  key_words: [u32; 8],
  flags: u32,
) -> OutputState {
  let mut block_words = [0u32; 16];
  block_words[..8].copy_from_slice(&left_child_cv);
  block_words[8..].copy_from_slice(&right_child_cv);
  OutputState {
    kernel,
    input_chaining_value: key_words,
    block_words,
    counter: 0,
    block_len: BLOCK_LEN as u32,
    flags: PARENT | flags,
  }
}

#[inline]
fn single_chunk_output(
  kernel: Kernel,
  key_words: [u32; 8],
  chunk_counter: u64,
  flags: u32,
  input: &[u8],
) -> OutputState {
  debug_assert!(input.len() <= CHUNK_LEN);

  // We always emit an OutputState for the last chunk block, even when the
  // input is empty (which is treated as a single, 0-length block).
  let blocks = core::cmp::max(1usize, input.len().div_ceil(BLOCK_LEN));
  let (full_blocks, last_len) = if input.is_empty() {
    (0usize, 0usize)
  } else if input.len().is_multiple_of(BLOCK_LEN) {
    (blocks - 1, BLOCK_LEN)
  } else {
    (blocks - 1, input.len() % BLOCK_LEN)
  };

  let mut chaining_value = key_words;
  let mut blocks_compressed: u8 = 0;
  let full_bytes = full_blocks * BLOCK_LEN;
  (kernel.chunk_compress_blocks)(
    &mut chaining_value,
    chunk_counter,
    flags,
    &mut blocks_compressed,
    &input[..full_bytes],
  );

  let mut last_block = [0u8; BLOCK_LEN];
  if !input.is_empty() {
    let offset = full_blocks * BLOCK_LEN;
    last_block[..last_len].copy_from_slice(&input[offset..offset + last_len]);
  }
  let block_words = words16_from_le_bytes_64(&last_block);
  let start = if blocks_compressed == 0 { CHUNK_START } else { 0 };

  OutputState {
    kernel,
    input_chaining_value: chaining_value,
    block_words,
    counter: chunk_counter,
    block_len: last_len as u32,
    flags: flags | start | CHUNK_END,
  }
}

#[derive(Clone)]
pub struct Blake3 {
  kernel: Kernel,
  dispatch: Option<SizeClassDispatch<Kernel>>,
  chunk_state: ChunkState,
  key_words: [u32; 8],
  cv_stack: [[u32; 8]; 54],
  cv_stack_len: u8,
  flags: u32,
}

impl Default for Blake3 {
  #[inline]
  fn default() -> Self {
    Self::new()
  }
}

impl Blake3 {
  /// Compute the hash of `data` in one shot.
  ///
  /// This selects the best available kernel for the current platform and input
  /// length (cached after first use).
  #[inline]
  #[must_use]
  pub fn digest(data: &[u8]) -> [u8; OUT_LEN] {
    dispatch::digest(data)
  }

  #[inline]
  #[must_use]
  pub(crate) fn digest_portable(data: &[u8]) -> [u8; OUT_LEN] {
    let kernel = kernels::kernel(kernels::Blake3KernelId::Portable);
    if data.len() <= CHUNK_LEN {
      let output = single_chunk_output(kernel, IV, 0, 0, data);
      let block = output.root_output_block_bytes(0);
      let mut out = [0u8; OUT_LEN];
      out.copy_from_slice(&block[..OUT_LEN]);
      out
    } else {
      let mut h = Self::new_internal_with(IV, 0, kernel);
      h.update_with(data, kernel);
      h.finalize()
    }
  }

  #[inline]
  fn new_internal(key_words: [u32; 8], flags: u32) -> Self {
    let kernel = kernels::kernel(kernels::Blake3KernelId::Portable);
    Self::new_internal_with(key_words, flags, kernel)
  }

  #[inline]
  fn new_internal_with(key_words: [u32; 8], flags: u32, kernel: Kernel) -> Self {
    Self {
      kernel,
      dispatch: None,
      chunk_state: ChunkState::new(key_words, 0, flags, kernel),
      key_words,
      cv_stack: [[0u32; 8]; 54],
      cv_stack_len: 0,
      flags,
    }
  }

  fn update_with(&mut self, mut input: &[u8], kernel: Kernel) {
    self.kernel = kernel;
    self.chunk_state.kernel = kernel;

    while !input.is_empty() {
      if self.chunk_state.len() == CHUNK_LEN {
        let chunk_cv = self.chunk_state.output().chaining_value();
        let total_chunks = self.chunk_state.chunk_counter + 1;
        self.add_chunk_chaining_value(chunk_cv, total_chunks);
        self.chunk_state = ChunkState::new(self.key_words, total_chunks, self.flags, self.kernel);
      }

      let want = CHUNK_LEN - self.chunk_state.len();
      let take = min(want, input.len());
      self.chunk_state.update(&input[..take]);
      input = &input[take..];
    }
  }

  /// Construct a new hasher for the keyed hash function.
  #[must_use]
  pub fn new_keyed(key: &[u8; KEY_LEN]) -> Self {
    let key_words = words8_from_le_bytes_32(key);
    Self::new_internal(key_words, KEYED_HASH)
  }

  /// Construct a new hasher for the key derivation function.
  #[must_use]
  pub fn new_derive_key(context: &str) -> Self {
    let mut context_hasher = Self::new_internal(IV, DERIVE_KEY_CONTEXT);
    context_hasher.update(context.as_bytes());
    let context_key = context_hasher.finalize();
    let key_words = words8_from_le_bytes_32(&context_key);
    Self::new_internal(key_words, DERIVE_KEY_MATERIAL)
  }

  #[inline]
  fn push_stack(&mut self, cv: [u32; 8]) {
    self.cv_stack[self.cv_stack_len as usize] = cv;
    self.cv_stack_len = self.cv_stack_len.wrapping_add(1);
  }

  #[inline]
  fn pop_stack(&mut self) -> [u32; 8] {
    self.cv_stack_len = self.cv_stack_len.wrapping_sub(1);
    self.cv_stack[self.cv_stack_len as usize]
  }

  fn add_chunk_chaining_value(&mut self, mut new_cv: [u32; 8], mut total_chunks: u64) {
    while total_chunks & 1 == 0 {
      new_cv = (self.kernel.parent_cv)(self.pop_stack(), new_cv, self.key_words, self.flags);
      total_chunks >>= 1;
    }
    self.push_stack(new_cv);
  }

  fn root_output(&self) -> OutputState {
    let mut output = self.chunk_state.output();
    let mut parent_nodes_remaining = self.cv_stack_len as usize;
    while parent_nodes_remaining > 0 {
      parent_nodes_remaining -= 1;
      output = parent_output(
        self.kernel,
        self.cv_stack[parent_nodes_remaining],
        output.chaining_value(),
        self.key_words,
        self.flags,
      );
    }
    output
  }

  /// Finalize into an extendable output state (XOF).
  #[must_use]
  pub fn finalize_xof(&self) -> Blake3Xof {
    Blake3Xof::new(self.root_output())
  }
}

impl Digest for Blake3 {
  const OUTPUT_SIZE: usize = OUT_LEN;
  type Output = [u8; OUT_LEN];

  #[inline]
  fn new() -> Self {
    Self::new_internal(IV, 0)
  }

  fn update(&mut self, input: &[u8]) {
    if input.is_empty() {
      return;
    }

    let dispatch = match self.dispatch {
      Some(d) => d,
      None => {
        let d = dispatch::kernel_dispatch();
        self.dispatch = Some(d);
        d
      }
    };

    let bytes_so_far = (self.chunk_state.chunk_counter as u128)
      .saturating_mul(CHUNK_LEN as u128)
      .saturating_add(self.chunk_state.len() as u128);
    let len_hint = bytes_so_far.saturating_add(input.len() as u128);
    let kernel = dispatch.select(len_hint_from_u128(len_hint));
    self.update_with(input, kernel);
  }

  fn finalize(&self) -> Self::Output {
    let output = self.root_output();
    let block = output.root_output_block_bytes(0);
    let mut out = [0u8; OUT_LEN];
    out.copy_from_slice(&block[..OUT_LEN]);
    out
  }

  #[inline]
  fn reset(&mut self) {
    *self = Self::new_internal(self.key_words, self.flags);
  }
}

#[derive(Clone)]
pub struct Blake3Xof {
  output: OutputState,
  block_counter: u64,
  buf: [u8; 2 * OUT_LEN],
  buf_pos: usize,
}

impl Blake3Xof {
  #[inline]
  fn new(output: OutputState) -> Self {
    Self {
      output,
      block_counter: 0,
      buf: [0u8; 2 * OUT_LEN],
      buf_pos: 2 * OUT_LEN,
    }
  }

  #[inline]
  fn refill(&mut self) {
    self.buf = self.output.root_output_block_bytes(self.block_counter);
    self.block_counter = self.block_counter.wrapping_add(1);
    self.buf_pos = 0;
  }
}

impl Xof for Blake3Xof {
  fn squeeze(&mut self, mut out: &mut [u8]) {
    while !out.is_empty() {
      if self.buf_pos == self.buf.len() {
        self.refill();
      }
      let take = min(self.buf.len() - self.buf_pos, out.len());
      out[..take].copy_from_slice(&self.buf[self.buf_pos..self.buf_pos + take]);
      self.buf_pos += take;
      out = &mut out[take..];
    }
  }
}

#[cfg(feature = "std")]
pub(crate) mod kernel_test;

#[cfg(test)]
mod tests {
  use traits::{Digest, Xof};

  use super::{Blake3, OUT_LEN};

  const KEY: &[u8; 32] = b"whats the Elvish word for friend";
  const CONTEXT: &str = "BLAKE3 2019-12-27 16:29:52 test vectors context";

  fn hex_to_bytes(hex: &str, out: &mut [u8]) {
    assert_eq!(hex.len(), out.len() * 2);
    for (i, chunk) in hex.as_bytes().chunks_exact(2).enumerate() {
      let hi = (chunk[0] as char).to_digit(16).unwrap();
      let lo = (chunk[1] as char).to_digit(16).unwrap();
      out[i] = ((hi << 4) | lo) as u8;
    }
  }

  fn input_pattern(len: usize) -> alloc::vec::Vec<u8> {
    let mut v = alloc::vec::Vec::with_capacity(len);
    for i in 0..len {
      v.push((i % 251) as u8);
    }
    v
  }

  extern crate alloc;

  #[test]
  fn official_vectors_len0_hash_and_xof_prefix() {
    let mut hasher = Blake3::new();
    hasher.update(&input_pattern(0));

    let expected_hash_hex = "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262";
    let mut expected_hash = [0u8; OUT_LEN];
    hex_to_bytes(expected_hash_hex, &mut expected_hash);
    assert_eq!(hasher.finalize(), expected_hash);

    let expected_xof_prefix_hex = "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262e00f03e7b69af26b7faaf09fcd333050338ddfe085b8cc869ca98b206c08243a26f5487789e8f660afe6c99ef9e0c52b92e7393024a80459cf91f476f9ffdbda7001c22e159b402631f277ca96f2defdf1078282314e763699a31c5363165421cce14d";
    let mut expected_xof_prefix = [0u8; 131];
    hex_to_bytes(expected_xof_prefix_hex, &mut expected_xof_prefix);

    let mut xof = hasher.finalize_xof();
    let mut out = [0u8; 131];
    xof.squeeze(&mut out);
    assert_eq!(out, expected_xof_prefix);
  }

  #[test]
  fn official_vectors_len0_keyed_and_derive() {
    let mut keyed = Blake3::new_keyed(KEY);
    keyed.update(&input_pattern(0));
    let expected_keyed_hex = "92b2b75604ed3c761f9d6f62392c8a9227ad0ea3f09573e783f1498a4ed60d26";
    let mut expected_keyed = [0u8; OUT_LEN];
    hex_to_bytes(expected_keyed_hex, &mut expected_keyed);
    assert_eq!(keyed.finalize(), expected_keyed);

    let mut dk = Blake3::new_derive_key(CONTEXT);
    dk.update(&input_pattern(0));
    let expected_dk_hex = "2cc39783c223154fea8dfb7c1b1660f2ac2dcbd1c1de8277b0b0dd39b7e50d7d";
    let mut expected_dk = [0u8; OUT_LEN];
    hex_to_bytes(expected_dk_hex, &mut expected_dk);
    assert_eq!(dk.finalize(), expected_dk);
  }
}
