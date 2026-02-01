//! BLAKE3 (hash + XOF).
//!
//! This is a portable, dependency-free implementation suitable for `no_std`.

#![allow(clippy::indexing_slicing)] // Fixed-size arrays + internal block parsing

use core::{cmp::min, mem::MaybeUninit, ptr};

use traits::{Digest, Xof};

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;
#[doc(hidden)]
pub mod dispatch;
#[doc(hidden)]
pub mod dispatch_tables;
pub(crate) mod kernels;
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64;

use self::kernels::Kernel;

const OUT_LEN: usize = 32;
const KEY_LEN: usize = 32;
const BLOCK_LEN: usize = 64;
const CHUNK_LEN: usize = 1024;
const OUTPUT_BLOCK_LEN: usize = 2 * OUT_LEN;

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

/// BLAKE3 message schedule.
///
/// `MSG_SCHEDULE[round][i]` gives the index of the message word to use.
pub(crate) const MSG_SCHEDULE: [[usize; 16]; 7] = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
  [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
  [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
  [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
  [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
  [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

#[inline(always)]
fn uninit_cv_stack() -> [MaybeUninit<[u32; 8]>; 54] {
  // SAFETY: An uninitialized `[MaybeUninit<_>; N]` is valid, because
  // `MaybeUninit<T>` permits any bit pattern.
  unsafe { MaybeUninit::<[MaybeUninit<[u32; 8]>; 54]>::uninit().assume_init() }
}

#[inline(always)]
fn words8_from_le_bytes_32(bytes: &[u8; 32]) -> [u32; 8] {
  if cfg!(target_endian = "little") {
    let src = bytes.as_ptr() as *const u32;
    // SAFETY: `bytes` is exactly 32 bytes; `read_unaligned` supports the
    // 1-byte alignment of `[u8; 32]`.
    unsafe {
      [
        ptr::read_unaligned(src.add(0)),
        ptr::read_unaligned(src.add(1)),
        ptr::read_unaligned(src.add(2)),
        ptr::read_unaligned(src.add(3)),
        ptr::read_unaligned(src.add(4)),
        ptr::read_unaligned(src.add(5)),
        ptr::read_unaligned(src.add(6)),
        ptr::read_unaligned(src.add(7)),
      ]
    }
  } else {
    let src = bytes.as_ptr() as *const u32;
    // SAFETY: `bytes` is exactly 32 bytes; `read_unaligned` supports the
    // 1-byte alignment of `[u8; 32]`.
    unsafe {
      [
        u32::from_le(ptr::read_unaligned(src.add(0))),
        u32::from_le(ptr::read_unaligned(src.add(1))),
        u32::from_le(ptr::read_unaligned(src.add(2))),
        u32::from_le(ptr::read_unaligned(src.add(3))),
        u32::from_le(ptr::read_unaligned(src.add(4))),
        u32::from_le(ptr::read_unaligned(src.add(5))),
        u32::from_le(ptr::read_unaligned(src.add(6))),
        u32::from_le(ptr::read_unaligned(src.add(7))),
      ]
    }
  }
}

#[inline(always)]
fn words16_from_le_bytes_64(bytes: &[u8; 64]) -> [u32; 16] {
  if cfg!(target_endian = "little") {
    let src = bytes.as_ptr() as *const u32;
    // SAFETY: `bytes` is exactly 64 bytes; `read_unaligned` supports the
    // 1-byte alignment of `[u8; 64]`.
    unsafe {
      [
        ptr::read_unaligned(src.add(0)),
        ptr::read_unaligned(src.add(1)),
        ptr::read_unaligned(src.add(2)),
        ptr::read_unaligned(src.add(3)),
        ptr::read_unaligned(src.add(4)),
        ptr::read_unaligned(src.add(5)),
        ptr::read_unaligned(src.add(6)),
        ptr::read_unaligned(src.add(7)),
        ptr::read_unaligned(src.add(8)),
        ptr::read_unaligned(src.add(9)),
        ptr::read_unaligned(src.add(10)),
        ptr::read_unaligned(src.add(11)),
        ptr::read_unaligned(src.add(12)),
        ptr::read_unaligned(src.add(13)),
        ptr::read_unaligned(src.add(14)),
        ptr::read_unaligned(src.add(15)),
      ]
    }
  } else {
    let src = bytes.as_ptr() as *const u32;
    // SAFETY: `bytes` is exactly 64 bytes; `read_unaligned` supports the
    // 1-byte alignment of `[u8; 64]`.
    unsafe {
      [
        u32::from_le(ptr::read_unaligned(src.add(0))),
        u32::from_le(ptr::read_unaligned(src.add(1))),
        u32::from_le(ptr::read_unaligned(src.add(2))),
        u32::from_le(ptr::read_unaligned(src.add(3))),
        u32::from_le(ptr::read_unaligned(src.add(4))),
        u32::from_le(ptr::read_unaligned(src.add(5))),
        u32::from_le(ptr::read_unaligned(src.add(6))),
        u32::from_le(ptr::read_unaligned(src.add(7))),
        u32::from_le(ptr::read_unaligned(src.add(8))),
        u32::from_le(ptr::read_unaligned(src.add(9))),
        u32::from_le(ptr::read_unaligned(src.add(10))),
        u32::from_le(ptr::read_unaligned(src.add(11))),
        u32::from_le(ptr::read_unaligned(src.add(12))),
        u32::from_le(ptr::read_unaligned(src.add(13))),
        u32::from_le(ptr::read_unaligned(src.add(14))),
        u32::from_le(ptr::read_unaligned(src.add(15))),
      ]
    }
  }
}

#[inline(always)]
fn pow2_floor(n: usize) -> usize {
  debug_assert!(n != 0);
  1usize << (usize::BITS - 1 - n.leading_zeros())
}

#[inline]
fn reduce_power_of_two_chunk_cvs(kernel: Kernel, key_words: [u32; 8], flags: u32, cvs: &[[u32; 8]]) -> [u32; 8] {
  debug_assert!(cvs.len().is_power_of_two());
  debug_assert!(cvs.len() <= 16);

  if cvs.len() == 1 {
    return cvs[0];
  }

  let mut cur = [[0u32; 8]; 16];
  let mut next = [[0u32; 8]; 16];
  cur[..cvs.len()].copy_from_slice(cvs);
  let mut cur_len = cvs.len();

  while cur_len > 1 {
    let pairs = cur_len / 2;
    debug_assert!(pairs <= 8);
    kernels::parent_cvs_many_from_cvs_inline(kernel.id, &cur[..2 * pairs], key_words, flags, &mut next[..pairs]);
    cur[..pairs].copy_from_slice(&next[..pairs]);
    cur_len = pairs;
  }

  cur[0]
}

#[inline]
fn add_chunk_cvs_batched(
  kernel: Kernel,
  stack: &mut [MaybeUninit<[u32; 8]>; 54],
  stack_len: &mut usize,
  base_counter: u64,
  cvs: &[[u32; 8]],
  key_words: [u32; 8],
  flags: u32,
) {
  if cvs.is_empty() {
    return;
  }

  debug_assert!(cvs.len() <= 16);

  #[inline]
  fn push_stack(stack: &mut [MaybeUninit<[u32; 8]>; 54], len: &mut usize, cv: [u32; 8]) {
    stack[*len].write(cv);
    *len += 1;
  }

  #[inline]
  fn pop_stack(stack: &mut [MaybeUninit<[u32; 8]>; 54], len: &mut usize) -> [u32; 8] {
    *len -= 1;
    // SAFETY: `len` tracks the number of initialized entries.
    unsafe { stack[*len].assume_init_read() }
  }

  let mut offset = 0usize;
  let mut chunk_counter = base_counter;

  while offset < cvs.len() {
    let remaining = cvs.len() - offset;
    let mut size = pow2_floor(remaining);

    let aligned_max = if chunk_counter == 0 {
      usize::MAX
    } else {
      let tz = chunk_counter.trailing_zeros() as usize;
      if tz >= (usize::BITS as usize) {
        usize::MAX
      } else {
        1usize << tz
      }
    };

    size = size.min(aligned_max).min(remaining);
    debug_assert!(size.is_power_of_two());

    let subtree_cv = reduce_power_of_two_chunk_cvs(kernel, key_words, flags, &cvs[offset..offset + size]);
    chunk_counter = chunk_counter.wrapping_add(size as u64);

    // Merge this subtree into the global stack. Because `size` is a power of two
    // that divides `base_counter + offset`, there are no pending nodes below
    // this subtree's level, and popping proceeds in correct level order.
    let level = size.trailing_zeros();
    let mut total = chunk_counter >> level;
    let mut cv = subtree_cv;
    while total & 1 == 0 {
      cv = kernels::parent_cv_inline(kernel.id, pop_stack(stack, stack_len), cv, key_words, flags);
      total >>= 1;
    }
    push_stack(stack, stack_len, cv);

    offset += size;
  }
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

#[inline(always)]
fn words8_to_le_bytes(words: &[u32; 8]) -> [u8; OUT_LEN] {
  let mut out = [0u8; OUT_LEN];
  if cfg!(target_endian = "little") {
    // SAFETY: `words` is 8 u32s = 32 bytes, and `out` is 32 bytes.
    unsafe { ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), out.as_mut_ptr(), OUT_LEN) };
  } else {
    for (i, word) in words.iter().copied().enumerate() {
      let offset = i * 4;
      out[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
    }
  }
  out
}

#[inline(always)]
fn words16_to_le_bytes(words: &[u32; 16]) -> [u8; 2 * OUT_LEN] {
  let mut out = [0u8; 2 * OUT_LEN];
  if cfg!(target_endian = "little") {
    // SAFETY: `words` is 16 u32s = 64 bytes, and `out` is 64 bytes.
    unsafe { ptr::copy_nonoverlapping(words.as_ptr() as *const u8, out.as_mut_ptr(), 2 * OUT_LEN) };
  } else {
    for (i, word) in words.iter().copied().enumerate() {
      let offset = i * 4;
      out[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
    }
  }
  out
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
  fn root_hash_words(&self) -> [u32; 8] {
    first_8_words((self.kernel.compress)(
      &self.input_chaining_value,
      &self.block_words,
      0,
      self.block_len,
      self.flags | ROOT,
    ))
  }

  #[inline]
  fn root_hash_bytes(&self) -> [u8; OUT_LEN] {
    words8_to_le_bytes(&self.root_hash_words())
  }

  #[inline]
  fn root_output_blocks_into(&self, mut output_block_counter: u64, mut out: &mut [u8]) {
    debug_assert!(out.len().is_multiple_of(OUTPUT_BLOCK_LEN));
    let flags = self.flags | ROOT;

    while !out.is_empty() {
      let blocks_remaining = out.len() / OUTPUT_BLOCK_LEN;

      #[cfg(target_arch = "x86_64")]
      {
        match self.kernel.id {
          kernels::Blake3KernelId::X86Avx512 if blocks_remaining >= 16 => {
            // SAFETY: required CPU features are validated by dispatch before
            // selecting this kernel, and `out` has at least 16 blocks.
            unsafe {
              x86_64::avx512::root_output_blocks16(
                &self.input_chaining_value,
                &self.block_words,
                output_block_counter,
                self.block_len,
                flags,
                out.as_mut_ptr(),
              );
            }
            output_block_counter = output_block_counter.wrapping_add(16);
            out = &mut out[16 * OUTPUT_BLOCK_LEN..];
            continue;
          }
          kernels::Blake3KernelId::X86Avx512 if blocks_remaining >= 8 => {
            // SAFETY: AVX-512 implies AVX2 on the platforms we care about, and
            // dispatch only selects AVX-512 when the required caps are present.
            unsafe {
              x86_64::avx2::root_output_blocks8(
                &self.input_chaining_value,
                &self.block_words,
                output_block_counter,
                self.block_len,
                flags,
                out.as_mut_ptr(),
              );
            }
            output_block_counter = output_block_counter.wrapping_add(8);
            out = &mut out[8 * OUTPUT_BLOCK_LEN..];
            continue;
          }
          kernels::Blake3KernelId::X86Avx512 if blocks_remaining >= 4 => {
            // SAFETY: AVX-512 implies SSE4.1 on the platforms we care about, and
            // dispatch only selects AVX-512 when the required caps are present.
            unsafe {
              x86_64::sse41::root_output_blocks4(
                &self.input_chaining_value,
                &self.block_words,
                output_block_counter,
                self.block_len,
                flags,
                out.as_mut_ptr(),
              );
            }
            output_block_counter = output_block_counter.wrapping_add(4);
            out = &mut out[4 * OUTPUT_BLOCK_LEN..];
            continue;
          }
          kernels::Blake3KernelId::X86Avx2 if blocks_remaining >= 8 => {
            // SAFETY: required CPU features are validated by dispatch before
            // selecting this kernel, and `out` has at least 8 blocks.
            unsafe {
              x86_64::avx2::root_output_blocks8(
                &self.input_chaining_value,
                &self.block_words,
                output_block_counter,
                self.block_len,
                flags,
                out.as_mut_ptr(),
              );
            }
            output_block_counter = output_block_counter.wrapping_add(8);
            out = &mut out[8 * OUTPUT_BLOCK_LEN..];
            continue;
          }
          kernels::Blake3KernelId::X86Avx2 if blocks_remaining >= 4 => {
            // SAFETY: AVX2 implies SSE4.1 on the platforms we care about, and
            // dispatch only selects AVX2 when the required caps are present.
            unsafe {
              x86_64::sse41::root_output_blocks4(
                &self.input_chaining_value,
                &self.block_words,
                output_block_counter,
                self.block_len,
                flags,
                out.as_mut_ptr(),
              );
            }
            output_block_counter = output_block_counter.wrapping_add(4);
            out = &mut out[4 * OUTPUT_BLOCK_LEN..];
            continue;
          }
          kernels::Blake3KernelId::X86Sse41 if blocks_remaining >= 4 => {
            // SAFETY: required CPU features are validated by dispatch before
            // selecting this kernel, and `out` has at least 4 blocks.
            unsafe {
              x86_64::sse41::root_output_blocks4(
                &self.input_chaining_value,
                &self.block_words,
                output_block_counter,
                self.block_len,
                flags,
                out.as_mut_ptr(),
              );
            }
            output_block_counter = output_block_counter.wrapping_add(4);
            out = &mut out[4 * OUTPUT_BLOCK_LEN..];
            continue;
          }
          _ => {}
        }
      }

      #[cfg(target_arch = "aarch64")]
      {
        if self.kernel.id == kernels::Blake3KernelId::Aarch64Neon && blocks_remaining >= 4 {
          // SAFETY: required CPU features are validated by dispatch before
          // selecting this kernel, and `out` has at least 4 blocks.
          unsafe {
            aarch64::root_output_blocks4_neon(
              &self.input_chaining_value,
              &self.block_words,
              output_block_counter,
              self.block_len,
              flags,
              out.as_mut_ptr(),
            );
          }
          output_block_counter = output_block_counter.wrapping_add(4);
          out = &mut out[4 * OUTPUT_BLOCK_LEN..];
          continue;
        }
      }

      // Scalar fallback: generate one block at a time.
      let words = (self.kernel.compress)(
        &self.input_chaining_value,
        &self.block_words,
        output_block_counter,
        self.block_len,
        flags,
      );
      out[..OUTPUT_BLOCK_LEN].copy_from_slice(&words16_to_le_bytes(&words));
      output_block_counter = output_block_counter.wrapping_add(1);
      out = &mut out[OUTPUT_BLOCK_LEN..];
    }
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
    // Streaming fast path: when we receive exactly one whole chunk at a chunk
    // boundary, compute the internal state in one shot.
    #[cfg(target_arch = "aarch64")]
    {
      if self.kernel.id == kernels::Blake3KernelId::Aarch64Neon
        && self.blocks_compressed == 0
        && self.block_len == 0
        && input.len() == CHUNK_LEN
      {
        let mut cv_words = [0u32; 8];
        let mut last_block = [0u8; BLOCK_LEN];

        // SAFETY: input is exactly one full chunk.
        unsafe {
          aarch64::chunk_state_one_chunk_aarch64_out(
            input.as_ptr(),
            &self.chaining_value,
            self.chunk_counter,
            self.flags,
            cv_words.as_mut_ptr(),
            last_block.as_mut_ptr(),
          );
        }

        self.chaining_value = cv_words;
        self.block = last_block;
        self.block_len = BLOCK_LEN as u8;
        self.blocks_compressed = 15;
        return;
      }
    }

    // Phase 1: if we already have a buffered (partial or full) block, fill it
    // (or, if it's already full, compress it) before touching the caller
    // slice. This keeps the hot "many full blocks" path branch-light.
    if self.block_len != 0 {
      let want = BLOCK_LEN - self.block_len as usize;
      let take = min(want, input.len());
      self.block[self.block_len as usize..][..take].copy_from_slice(&input[..take]);
      self.block_len = self.block_len.wrapping_add(take as u8);
      input = &input[take..];

      // If the caller ended mid-block, we're done. Note that this also covers
      // the (rare) case where we just filled the final block of a chunk
      // (blocks_compressed == 15), in which case the full block must remain
      // buffered until output().
      if input.is_empty() {
        return;
      }

      if self.block_len as usize == BLOCK_LEN {
        debug_assert!(
          self.blocks_compressed < 15,
          "last chunk block stays buffered until output()"
        );
        kernels::chunk_compress_blocks_inline(
          self.kernel.id,
          &mut self.chaining_value,
          self.chunk_counter,
          self.flags,
          &mut self.blocks_compressed,
          &self.block,
        );
        self.block_len = 0;
      }
    }

    // Phase 2: we are block-aligned. Compress as many full blocks as possible
    // directly from the caller slice, leaving exactly one full block buffered
    // when the caller ends on a block boundary.
    debug_assert_eq!(self.block_len, 0);
    while !input.is_empty() {
      // Once we've compressed 15 blocks, the final block stays buffered.
      if self.blocks_compressed == 15 {
        debug_assert!(input.len() <= BLOCK_LEN);
        self.block[..input.len()].copy_from_slice(input);
        self.block_len = input.len() as u8;
        return;
      }

      let full_blocks = input.len() / BLOCK_LEN;
      if full_blocks != 0 {
        let max_blocks = 15usize - self.blocks_compressed as usize;
        let mut blocks_to_compress = full_blocks.min(max_blocks);

        // If we'd consume the entire input as full blocks, leave one block
        // buffered so finalize can apply CHUNK_END if the caller stops here.
        if input.len().is_multiple_of(BLOCK_LEN) && blocks_to_compress == full_blocks {
          blocks_to_compress = blocks_to_compress.saturating_sub(1);
        }

        if blocks_to_compress != 0 {
          let bytes = blocks_to_compress * BLOCK_LEN;
          kernels::chunk_compress_blocks_inline(
            self.kernel.id,
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

      // Remainder: buffer <= 64 bytes and return.
      let take = min(BLOCK_LEN, input.len());
      self.block[..take].copy_from_slice(&input[..take]);
      self.block_len = take as u8;
      return;
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

  // aarch64 fast path for a full chunk.
  //
  // This is used by oneshot XOF (and any other oneshot path that needs an OutputState)
  // to avoid per-block compression for 1024B inputs.
  #[cfg(target_arch = "aarch64")]
  {
    if kernel.id == kernels::Blake3KernelId::Aarch64Neon && input.len() == CHUNK_LEN {
      let mut cv_words = [0u32; 8];
      let mut last_block = [0u8; BLOCK_LEN];

      // SAFETY: `input` is exactly one full chunk, and this kernel is only selected
      // when its required CPU features are available.
      unsafe {
        aarch64::chunk_state_one_chunk_aarch64_out(
          input.as_ptr(),
          &key_words,
          chunk_counter,
          flags,
          cv_words.as_mut_ptr(),
          last_block.as_mut_ptr(),
        );
      }

      let block_words = words16_from_le_bytes_64(&last_block);
      return OutputState {
        kernel,
        input_chaining_value: cv_words,
        block_words,
        counter: chunk_counter,
        block_len: BLOCK_LEN as u32,
        flags: flags | CHUNK_END,
      };
    }
  }

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
  kernels::chunk_compress_blocks_inline(
    kernel.id,
    &mut chaining_value,
    chunk_counter,
    flags,
    &mut blocks_compressed,
    &input[..full_bytes],
  );

  let block_words = if cfg!(target_endian = "little") {
    let mut out = [0u32; 16];
    if !input.is_empty() {
      let offset = full_blocks * BLOCK_LEN;
      // SAFETY: `out` is 64 bytes, and `last_len <= 64`.
      unsafe {
        ptr::copy_nonoverlapping(input.as_ptr().add(offset), out.as_mut_ptr().cast::<u8>(), last_len);
      }
    }
    out
  } else {
    let mut last_block = [0u8; BLOCK_LEN];
    if !input.is_empty() {
      let offset = full_blocks * BLOCK_LEN;
      last_block[..last_len].copy_from_slice(&input[offset..offset + last_len]);
    }
    words16_from_le_bytes_64(&last_block)
  };
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

#[inline]
fn root_output_oneshot(kernel: Kernel, key_words: [u32; 8], flags: u32, input: &[u8]) -> OutputState {
  // Fast path for <= 1 chunk (root is the chunk itself).
  if input.len() <= CHUNK_LEN {
    return single_chunk_output(kernel, key_words, 0, flags, input);
  }

  let full_chunks = input.len() / CHUNK_LEN;
  let remainder = input.len() % CHUNK_LEN;

  // Local CV stack to avoid constructing a full streaming hasher.
  let mut cv_stack: [MaybeUninit<[u32; 8]>; 54] = uninit_cv_stack();
  let mut cv_stack_len = 0usize;

  const MAX_SIMD_DEGREE: usize = 16;
  let mut out_buf = [0u8; OUT_LEN * MAX_SIMD_DEGREE];

  // Hash all full chunks. If there is no remainder, we still hash the final
  // full chunk here, but we keep its CV as the "right child" instead of
  // pushing it to the stack (because the root output for multi-chunk inputs is
  // always a parent node).
  let mut last_full_chunk_cv = None;

  let mut chunk_counter = 0u64;
  let mut offset = 0usize;
  while chunk_counter < full_chunks as u64 {
    let remaining = (full_chunks as u64 - chunk_counter) as usize;
    let batch = core::cmp::min(remaining, core::cmp::min(kernel.simd_degree, MAX_SIMD_DEGREE));
    debug_assert!(batch != 0);

    // SAFETY: `offset` is within `input`, and `out_buf` is large enough for `batch`.
    unsafe {
      (kernel.hash_many_contiguous)(
        input.as_ptr().add(offset),
        batch,
        &key_words,
        chunk_counter,
        flags,
        out_buf.as_mut_ptr(),
      )
    };

    let mut cvs = [[0u32; 8]; MAX_SIMD_DEGREE];
    for (i, slot) in cvs.iter_mut().take(batch).enumerate() {
      let off = i * OUT_LEN;
      // SAFETY: `out_buf` is sized to `MAX_SIMD_DEGREE * OUT_LEN` and `off`
      // is `i * OUT_LEN` with `i < batch <= MAX_SIMD_DEGREE`.
      let cv = unsafe { words8_from_le_bytes_32(&*(out_buf.as_ptr().add(off) as *const [u8; OUT_LEN])) };
      *slot = cv;
    }

    let mut commit = batch;
    if remainder == 0 && chunk_counter + batch as u64 == full_chunks as u64 {
      last_full_chunk_cv = Some(cvs[batch - 1]);
      commit -= 1;
    }

    if commit != 0 {
      add_chunk_cvs_batched(
        kernel,
        &mut cv_stack,
        &mut cv_stack_len,
        chunk_counter,
        &cvs[..commit],
        key_words,
        flags,
      );
    }

    chunk_counter += batch as u64;
    offset += batch * CHUNK_LEN;
  }

  let right_cv = if remainder != 0 {
    let chunk_bytes = &input[full_chunks * CHUNK_LEN..];
    single_chunk_output(kernel, key_words, full_chunks as u64, flags, chunk_bytes).chaining_value()
  } else {
    // `input.len() > CHUNK_LEN` implies there are at least 2 chunks total, so
    // the root output is always derived from parent nodes rather than a chunk.
    match last_full_chunk_cv {
      Some(cv) => cv,
      None => unreachable!("missing last full chunk cv"),
    }
  };

  let mut parent_nodes_remaining = cv_stack_len;
  debug_assert!(parent_nodes_remaining > 0);
  parent_nodes_remaining -= 1;
  // SAFETY: `cv_stack_len` tracks the number of initialized entries.
  let left = unsafe { cv_stack[parent_nodes_remaining].assume_init_read() };
  let mut output = parent_output(kernel, left, right_cv, key_words, flags);
  while parent_nodes_remaining > 0 {
    parent_nodes_remaining -= 1;
    // SAFETY: `cv_stack_len` tracks the number of initialized entries.
    let left = unsafe { cv_stack[parent_nodes_remaining].assume_init_read() };
    output = parent_output(kernel, left, output.chaining_value(), key_words, flags);
  }

  output
}

#[inline]
fn digest_oneshot_words(kernel: Kernel, key_words: [u32; 8], flags: u32, input: &[u8]) -> [u32; 8] {
  #[cfg(target_arch = "x86_64")]
  {
    if input.len() <= CHUNK_LEN {
      match kernel.id {
        kernels::Blake3KernelId::X86Sse41 | kernels::Blake3KernelId::X86Avx2 | kernels::Blake3KernelId::X86Avx512 => {
          // SAFETY: x86 SIMD availability is validated by dispatch before selecting these kernels.
          return unsafe { digest_one_chunk_root_hash_words_x86(kernel, key_words, flags, input) };
        }
        _ => {}
      }
    }
  }

  // Fallback: construct the root output and extract the root hash words.
  let output = root_output_oneshot(kernel, key_words, flags, input);
  output.root_hash_words()
}

#[inline]
fn digest_oneshot(kernel: Kernel, key_words: [u32; 8], flags: u32, input: &[u8]) -> [u8; OUT_LEN] {
  #[cfg(target_arch = "aarch64")]
  {
    if kernel.id == kernels::Blake3KernelId::Aarch64Neon && input.len() == CHUNK_LEN {
      // SAFETY: aarch64 NEON is validated by dispatch before selecting this kernel.
      return unsafe { aarch64::root_hash_one_chunk_root_aarch64(input.as_ptr(), &key_words, flags) };
    }
  }

  words8_to_le_bytes(&digest_oneshot_words(kernel, key_words, flags, input))
}

#[derive(Clone)]
pub struct Blake3 {
  kernel: Kernel,
  bulk_kernel: Kernel,
  chunk_state: ChunkState,
  pending_chunk_cv: Option<[u32; 8]>,
  key_words: [u32; 8],
  cv_stack: [MaybeUninit<[u32; 8]>; 54],
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

  /// Compute the XOF output state of `data` in one shot.
  ///
  /// This avoids constructing a full streaming hasher. It's useful when you
  /// immediately want to squeeze output without incremental updates.
  #[inline]
  #[must_use]
  pub fn xof(data: &[u8]) -> Blake3Xof {
    dispatch::xof(data)
  }

  /// Compute the keyed hash of `data` in one shot.
  #[inline]
  #[must_use]
  pub fn keyed_digest(key: &[u8; KEY_LEN], data: &[u8]) -> [u8; OUT_LEN] {
    let key_words = words8_from_le_bytes_32(key);
    let kernel = dispatch::kernel_dispatch().select(data.len());
    digest_oneshot(kernel, key_words, KEYED_HASH, data)
  }

  /// Compute the keyed XOF output state of `data` in one shot.
  #[inline]
  #[must_use]
  pub fn keyed_xof(key: &[u8; KEY_LEN], data: &[u8]) -> Blake3Xof {
    let key_words = words8_from_le_bytes_32(key);
    let kernel = dispatch::kernel_dispatch().select(data.len());
    Blake3Xof::new(root_output_oneshot(kernel, key_words, KEYED_HASH, data))
  }

  /// Compute the derived key for `key_material` under `context`, in one shot.
  #[inline]
  #[must_use]
  pub fn derive_key(context: &str, key_material: &[u8]) -> [u8; OUT_LEN] {
    // Step 1: hash the context string under DERIVE_KEY_CONTEXT to get the
    // derived key context key.
    let context_bytes = context.as_bytes();
    let kernel_ctx = dispatch::kernel_dispatch().select(context_bytes.len());
    let context_key_words = digest_oneshot_words(kernel_ctx, IV, DERIVE_KEY_CONTEXT, context_bytes);

    // Step 2: hash the key material under DERIVE_KEY_MATERIAL with the derived
    // context key.
    let kernel_km = dispatch::kernel_dispatch().select(key_material.len());
    words8_to_le_bytes(&digest_oneshot_words(
      kernel_km,
      context_key_words,
      DERIVE_KEY_MATERIAL,
      key_material,
    ))
  }

  /// One-shot hash using an explicitly selected kernel.
  ///
  /// This is crate-internal glue for `hashes::bench` / `rscrypto-tune`.
  #[inline]
  #[must_use]
  pub(crate) fn digest_with_kernel_id(id: kernels::Blake3KernelId, data: &[u8]) -> [u8; OUT_LEN] {
    let kernel = kernels::kernel(id);
    digest_oneshot(kernel, IV, 0, data)
  }

  /// Streaming hash with fixed update chunking, using an explicitly selected kernel.
  ///
  /// This is crate-internal glue for `hashes::bench` / `rscrypto-tune`.
  #[inline]
  #[must_use]
  pub(crate) fn stream_chunks_with_kernel_id(
    id: kernels::Blake3KernelId,
    chunk_size: usize,
    data: &[u8],
  ) -> [u8; OUT_LEN] {
    let kernel = kernels::kernel(id);
    let mut h = Self::new_internal_with(IV, 0, kernel);
    for chunk in data.chunks(chunk_size) {
      h.update_with(chunk, kernel, kernel);
    }
    h.finalize()
  }

  #[inline]
  #[must_use]
  pub(crate) fn digest_portable(data: &[u8]) -> [u8; OUT_LEN] {
    let kernel = kernels::kernel(kernels::Blake3KernelId::Portable);
    if data.len() <= CHUNK_LEN {
      let output = single_chunk_output(kernel, IV, 0, 0, data);
      output.root_hash_bytes()
    } else {
      let mut h = Self::new_internal_with(IV, 0, kernel);
      h.update_with(data, kernel, kernel);
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
      bulk_kernel: kernel,
      chunk_state: ChunkState::new(key_words, 0, flags, kernel),
      pending_chunk_cv: None,
      key_words,
      cv_stack: uninit_cv_stack(),
      cv_stack_len: 0,
      flags,
    }
  }

  fn update_with(&mut self, mut input: &[u8], stream_kernel: Kernel, bulk_kernel: Kernel) {
    self.kernel = stream_kernel;
    self.bulk_kernel = bulk_kernel;
    self.chunk_state.kernel = stream_kernel;

    // If the previous update ended exactly on a chunk boundary, we may have
    // stored the last full chunk's CV instead of keeping a fully-buffered
    // `ChunkState`. As soon as more input arrives, that chunk is no longer
    // terminal and can be committed to the tree.
    if !input.is_empty()
      && let Some(cv) = self.pending_chunk_cv.take()
    {
      let total_chunks = self.chunk_state.chunk_counter;
      self.add_chunk_chaining_value(cv, total_chunks);
    }

    // When we're at a chunk boundary and we have more than one whole chunk
    // available, use the kernel's multi-chunk primitive to hash whole chunks
    // directly and feed their chaining values into the tree.
    //
    // Important: we always leave at least one full chunk (or partial) to be
    // processed by `ChunkState::update`, so that the streaming state retains
    // the "buffer the last block" invariant needed when the caller stops at
    // a block boundary and later calls `finalize`.
    const MAX_SIMD_DEGREE: usize = 16;
    let mut out_buf = [0u8; OUT_LEN * MAX_SIMD_DEGREE];

    while !input.is_empty() {
      if self.chunk_state.len() == CHUNK_LEN {
        let chunk_cv = self.chunk_state.output().chaining_value();
        let total_chunks = self.chunk_state.chunk_counter + 1;
        self.add_chunk_chaining_value(chunk_cv, total_chunks);
        self.chunk_state = ChunkState::new(self.key_words, total_chunks, self.flags, self.kernel);
      }

      if self.chunk_state.len() == 0 && self.bulk_kernel.simd_degree > 1 && input.len() > CHUNK_LEN {
        let full_chunks = input.len() / CHUNK_LEN;
        if full_chunks > 1 {
          let batch = core::cmp::min(
            full_chunks,
            core::cmp::min(self.bulk_kernel.simd_degree, MAX_SIMD_DEGREE),
          );

          if batch != 0 {
            let base_counter = self.chunk_state.chunk_counter;
            // SAFETY: `input` has at least `batch * CHUNK_LEN` bytes, `out_buf`
            // has at least `batch * OUT_LEN` bytes, and this kernel was selected
            // only when its required CPU features are available.
            unsafe {
              (self.bulk_kernel.hash_many_contiguous)(
                input.as_ptr(),
                batch,
                &self.key_words,
                base_counter,
                self.flags,
                out_buf.as_mut_ptr(),
              )
            };

            let keep_last_full_chunk = input.len().is_multiple_of(CHUNK_LEN) && batch == full_chunks;
            let commit = if keep_last_full_chunk { batch - 1 } else { batch };
            if commit != 0 {
              let mut cvs = [[0u32; 8]; MAX_SIMD_DEGREE];
              for (i, slot) in cvs.iter_mut().take(commit).enumerate() {
                let offset = i * OUT_LEN;
                // SAFETY: `out_buf` is `OUT_LEN * MAX_SIMD_DEGREE`, and `offset`
                // is `i * OUT_LEN` with `i < batch <= MAX_SIMD_DEGREE`.
                let cv = unsafe { words8_from_le_bytes_32(&*(out_buf.as_ptr().add(offset) as *const [u8; OUT_LEN])) };
                *slot = cv;
              }

              let mut stack_len = self.cv_stack_len as usize;
              add_chunk_cvs_batched(
                self.bulk_kernel,
                &mut self.cv_stack,
                &mut stack_len,
                base_counter,
                &cvs[..commit],
                self.key_words,
                self.flags,
              );
              self.cv_stack_len = stack_len as u8;
            }

            let new_counter = base_counter + batch as u64;
            self.chunk_state = ChunkState::new(self.key_words, new_counter, self.flags, self.kernel);
            if keep_last_full_chunk {
              let offset = (batch - 1) * OUT_LEN;
              // SAFETY: `out_buf` is `OUT_LEN * MAX_SIMD_DEGREE`, and `offset`
              // is `(batch - 1) * OUT_LEN` with `batch <= MAX_SIMD_DEGREE`.
              let cv = unsafe { words8_from_le_bytes_32(&*(out_buf.as_ptr().add(offset) as *const [u8; OUT_LEN])) };
              self.pending_chunk_cv = Some(cv);
            }
            input = &input[batch * CHUNK_LEN..];
            continue;
          }
        }
      }

      let want = CHUNK_LEN - self.chunk_state.len();
      let take = min(want, input.len());
      self.chunk_state.update(&input[..take]);
      input = &input[take..];
    }
  }

  /// Construct a new hasher for the keyed hash function.
  #[must_use]
  #[inline]
  pub fn new_keyed(key: &[u8; KEY_LEN]) -> Self {
    let key_words = words8_from_le_bytes_32(key);
    Self::new_internal(key_words, KEYED_HASH)
  }

  /// Construct a new hasher for the key derivation function.
  #[must_use]
  #[inline]
  pub fn new_derive_key(context: &str) -> Self {
    let context_bytes = context.as_bytes();
    let kernel_ctx = dispatch::kernel_dispatch().select(context_bytes.len());
    let key_words = digest_oneshot_words(kernel_ctx, IV, DERIVE_KEY_CONTEXT, context_bytes);
    Self::new_internal(key_words, DERIVE_KEY_MATERIAL)
  }

  #[inline]
  fn push_stack(&mut self, cv: [u32; 8]) {
    self.cv_stack[self.cv_stack_len as usize].write(cv);
    self.cv_stack_len = self.cv_stack_len.wrapping_add(1);
  }

  #[inline]
  fn pop_stack(&mut self) -> [u32; 8] {
    self.cv_stack_len = self.cv_stack_len.wrapping_sub(1);
    // SAFETY: `cv_stack_len` tracks the number of initialized entries.
    unsafe { self.cv_stack[self.cv_stack_len as usize].assume_init_read() }
  }

  fn add_chunk_chaining_value(&mut self, mut new_cv: [u32; 8], mut total_chunks: u64) {
    while total_chunks & 1 == 0 {
      new_cv = kernels::parent_cv_inline(self.kernel.id, self.pop_stack(), new_cv, self.key_words, self.flags);
      total_chunks >>= 1;
    }
    self.push_stack(new_cv);
  }

  fn root_output(&self) -> OutputState {
    let mut parent_nodes_remaining = self.cv_stack_len as usize;
    let mut output = if let Some(right_cv) = self.pending_chunk_cv {
      debug_assert!(
        parent_nodes_remaining > 0,
        "pending full chunk implies multi-chunk input"
      );
      parent_nodes_remaining -= 1;
      // SAFETY: `cv_stack_len` tracks the number of initialized entries.
      let left = unsafe { *self.cv_stack[parent_nodes_remaining].assume_init_ref() };
      parent_output(self.kernel, left, right_cv, self.key_words, self.flags)
    } else {
      self.chunk_state.output()
    };

    while parent_nodes_remaining > 0 {
      parent_nodes_remaining -= 1;
      // SAFETY: `cv_stack_len` tracks the number of initialized entries.
      let left = unsafe { *self.cv_stack[parent_nodes_remaining].assume_init_ref() };
      output = parent_output(self.kernel, left, output.chaining_value(), self.key_words, self.flags);
    }
    output
  }

  /// Finalize into an extendable output state (XOF).
  #[must_use]
  #[inline]
  pub fn finalize_xof(&self) -> Blake3Xof {
    // Mirror `finalize()` for the empty-input case: don't stay pinned to the
    // portable default when we could use the tuned streaming kernel.
    if self.chunk_state.chunk_counter == 0
      && self.chunk_state.len() == 0
      && self.cv_stack_len == 0
      && self.pending_chunk_cv.is_none()
    {
      let d = dispatch::streaming_dispatch();
      Blake3Xof::new(single_chunk_output(d.stream, self.key_words, 0, self.flags, &[]))
    } else {
      Blake3Xof::new(self.root_output())
    }
  }
}

impl Digest for Blake3 {
  const OUTPUT_SIZE: usize = OUT_LEN;
  type Output = [u8; OUT_LEN];

  #[inline]
  fn new() -> Self {
    Self::new_internal(IV, 0)
  }

  #[inline]
  fn update(&mut self, input: &[u8]) {
    if input.is_empty() {
      return;
    }

    // Streaming has two competing goals:
    // - Be very fast for large updates (file hashing, storage, etc.).
    // - Avoid SIMD setup overhead for tiny one-off updates (keyed/derive/tiny inputs).
    //
    // We start in portable mode and only "lock in" streaming dispatch once we
    // have enough data to amortize kernel setup costs.
    //
    // Important: keyed/derive modes are extremely sensitive to small-input
    // performance, and upstream implementations routinely use SIMD even for
    // tiny updates. Do not delay dispatch there.
    let is_plain_hash = self.flags == 0;
    if is_plain_hash
      && self.kernel.id == kernels::Blake3KernelId::Portable
      && self.chunk_state.chunk_counter == 0
      && self.cv_stack_len == 0
      && self.pending_chunk_cv.is_none()
    {
      let simd_enable_min = platform::tune().simd_threshold;
      if self.chunk_state.len().saturating_add(input.len()) < simd_enable_min {
        self.update_with(input, self.kernel, self.bulk_kernel);
        return;
      }
    }

    let (stream, table_bulk) = {
      let d = dispatch::streaming_dispatch();
      (d.stream, d.bulk)
    };

    let bulk = if input.len() >= 8 * 1024 {
      // For large updates, prefer the size-class tuned throughput kernel. This
      // keeps the streaming API competitive with one-shot hashing.
      dispatch::kernel_dispatch().select(input.len())
    } else {
      table_bulk
    };

    self.update_with(input, stream, bulk);
  }

  #[inline]
  fn finalize(&self) -> Self::Output {
    // If the caller never provided any input (or only provided empty updates),
    // we still want to use the tuned streaming kernel rather than staying
    // pinned to the portable default.
    let output = if self.chunk_state.chunk_counter == 0
      && self.chunk_state.len() == 0
      && self.cv_stack_len == 0
      && self.pending_chunk_cv.is_none()
    {
      let d = dispatch::streaming_dispatch();
      single_chunk_output(d.stream, self.key_words, 0, self.flags, &[])
    } else {
      self.root_output()
    };
    output.root_hash_bytes()
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
  buf: [u8; OUTPUT_BLOCK_LEN],
  buf_pos: usize,
}

impl Blake3Xof {
  #[inline]
  fn new(output: OutputState) -> Self {
    Self {
      output,
      block_counter: 0,
      buf: [0u8; OUTPUT_BLOCK_LEN],
      buf_pos: OUTPUT_BLOCK_LEN,
    }
  }

  #[inline]
  fn refill(&mut self) {
    self.output.root_output_blocks_into(self.block_counter, &mut self.buf);
    self.block_counter = self.block_counter.wrapping_add(1);
    self.buf_pos = 0;
  }
}

impl Xof for Blake3Xof {
  fn squeeze(&mut self, mut out: &mut [u8]) {
    if out.is_empty() {
      return;
    }

    // Drain any buffered bytes first.
    if self.buf_pos != self.buf.len() {
      let take = min(self.buf.len() - self.buf_pos, out.len());
      out[..take].copy_from_slice(&self.buf[self.buf_pos..self.buf_pos + take]);
      self.buf_pos += take;
      out = &mut out[take..];
      if out.is_empty() {
        return;
      }
    }

    // Generate any remaining full output blocks directly into the caller
    // buffer (lets the kernel choose its best batch size).
    let full = out.len() / OUTPUT_BLOCK_LEN * OUTPUT_BLOCK_LEN;
    if full != 0 {
      let blocks = (full / OUTPUT_BLOCK_LEN) as u64;
      self
        .output
        .root_output_blocks_into(self.block_counter, &mut out[..full]);
      self.block_counter = self.block_counter.wrapping_add(blocks);
      out = &mut out[full..];
    }

    // Tail: refill once and copy the remaining bytes.
    if !out.is_empty() {
      self.refill();
      let take = out.len();
      out.copy_from_slice(&self.buf[..take]);
      self.buf_pos = take;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// x86_64 tiny one-shot helpers (keyed/derive sensitive)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn digest_one_chunk_root_hash_words_x86(
  kernel: Kernel,
  key_words: [u32; 8],
  flags: u32,
  input: &[u8],
) -> [u32; 8] {
  debug_assert!(input.len() <= CHUNK_LEN);

  // We always have a "last block", even for the empty input (treated as a
  // single 0-length block).
  let blocks = core::cmp::max(1usize, input.len().div_ceil(BLOCK_LEN));
  let (full_blocks, last_len) = if input.is_empty() {
    (0usize, 0usize)
  } else if input.len().is_multiple_of(BLOCK_LEN) {
    (blocks - 1, BLOCK_LEN)
  } else {
    (blocks - 1, input.len() % BLOCK_LEN)
  };

  // Hash all full blocks except the final block, updating the CV. This keeps
  // ROOT out of the dependency chain until the last compress.
  let mut cv = key_words;
  let mut blocks_compressed: u8 = 0;
  let full_bytes = full_blocks * BLOCK_LEN;
  kernels::chunk_compress_blocks_inline(
    kernel.id,
    &mut cv,
    0,
    flags,
    &mut blocks_compressed,
    &input[..full_bytes],
  );

  let start = if blocks_compressed == 0 { CHUNK_START } else { 0 };
  let final_flags = flags | start | CHUNK_END | ROOT;

  // For partial blocks (including empty), pad to 64 bytes.
  let mut padded = [0u8; BLOCK_LEN];
  let block_ptr = if last_len == BLOCK_LEN && !input.is_empty() {
    // SAFETY: `full_blocks * BLOCK_LEN + BLOCK_LEN <= input.len()`.
    unsafe { input.as_ptr().add(full_blocks * BLOCK_LEN) }
  } else {
    if last_len != 0 {
      let offset = full_blocks * BLOCK_LEN;
      // SAFETY: `padded` is 64 bytes, and `last_len < 64` here.
      unsafe { ptr::copy_nonoverlapping(input.as_ptr().add(offset), padded.as_mut_ptr(), last_len) };
    }
    padded.as_ptr()
  };

  match kernel.id {
    kernels::Blake3KernelId::X86Sse41 => {
      // SAFETY: `block_ptr` points to 64 bytes (either into `input` or `padded`),
      // and dispatch only selects SSE4.1 when the required CPU features are present.
      unsafe { x86_64::compress_cv_sse41_bytes(&cv, block_ptr, 0, last_len as u32, final_flags) }
    }
    kernels::Blake3KernelId::X86Avx2 => {
      // SAFETY: `block_ptr` points to 64 bytes (either into `input` or `padded`),
      // and dispatch only selects AVX2 when the required CPU features are present.
      unsafe { x86_64::compress_cv_avx2_bytes(&cv, block_ptr, 0, last_len as u32, final_flags) }
    }
    kernels::Blake3KernelId::X86Avx512 => {
      // SAFETY: `block_ptr` points to 64 bytes (either into `input` or `padded`),
      // and dispatch only selects AVX-512 when the required CPU features are present.
      unsafe { x86_64::compress_cv_avx512_bytes(&cv, block_ptr, 0, last_len as u32, final_flags) }
    }
    _ => unreachable!("x86 tiny helper called for non-x86 SIMD kernel"),
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
