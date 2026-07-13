use core::{
  hash::{BuildHasher, Hasher},
  mem::MaybeUninit,
  ptr,
};

use super::{
  ACC_NB, DEFAULT_SECRET, DEFAULT_SECRET_SIZE, INITIAL_ACC, MID_SIZE_MAX, PRIME64_1, PRIME64_2, SECRET_CONSUME_RATE,
  SECRET_LASTACC_START, SECRET_MERGEACCS_START, STRIPE_LEN, accumulate_512, custom_default_secret, dispatch,
  kernels::StreamAccumulateFn, merge_accs,
};

const INTERNAL_BUFFER_SIZE: usize = 256;
const INTERNAL_BUFFER_STRIPES: usize = INTERNAL_BUFFER_SIZE / STRIPE_LEN;
const STRIPES_PER_BLOCK: usize = (DEFAULT_SECRET_SIZE - STRIPE_LEN) / SECRET_CONSUME_RATE;

#[inline(always)]
fn consume_stripes(
  stream_accumulate: StreamAccumulateFn,
  mut acc: [u64; ACC_NB],
  mut count: usize,
  mut accumulated: usize,
  input: &[u8],
  secret: &[u8; DEFAULT_SECRET_SIZE],
) -> ([u64; ACC_NB], usize) {
  debug_assert!(count.strict_mul(STRIPE_LEN) <= input.len());
  let mut input_offset = 0usize;
  while count != 0 {
    let to_block_end = STRIPES_PER_BLOCK.strict_sub(accumulated);
    let stripes = count.min(to_block_end);
    let scramble_after = stripes == to_block_end;
    let kernel = if stripes <= INTERNAL_BUFFER_STRIPES {
      super::stream_accumulate_portable
    } else {
      stream_accumulate
    };
    acc = kernel(
      acc,
      input,
      input_offset,
      secret,
      accumulated.strict_mul(SECRET_CONSUME_RATE),
      stripes,
      scramble_after,
    );
    input_offset = input_offset.strict_add(stripes.strict_mul(STRIPE_LEN));
    count = count.strict_sub(stripes);
    if scramble_after {
      accumulated = 0;
    } else {
      accumulated = accumulated.strict_add(stripes);
    }
  }
  (acc, accumulated)
}

#[inline(always)]
fn write_buffer(buffer: &mut [MaybeUninit<u8>; INTERNAL_BUFFER_SIZE], offset: usize, input: &[u8]) {
  debug_assert!(offset.strict_add(input.len()) <= buffer.len());
  // SAFETY: Copying initialized input into the inline buffer because:
  // 1. `offset + input.len() <= buffer.len()` is established by every caller and asserted above.
  // 2. `input` is an initialized shared byte slice and the destination is owned exclusively here.
  // 3. The source cannot overlap the hasher's private inline buffer through this safe API.
  unsafe {
    ptr::copy_nonoverlapping(
      input.as_ptr(),
      buffer.as_mut_ptr().cast::<u8>().add(offset),
      input.len(),
    );
  }
}

#[inline(always)]
fn initialized_buffer(buffer: &[MaybeUninit<u8>; INTERNAL_BUFFER_SIZE], len: usize) -> &[u8] {
  debug_assert!(len <= buffer.len());
  // SAFETY: Borrowing the initialized prefix because:
  // 1. `len <= INTERNAL_BUFFER_SIZE`, asserted above.
  // 2. State transitions increase `buffered` only after writing exactly that prefix.
  // 3. The returned lifetime is bounded by the shared borrow of `buffer`.
  unsafe { core::slice::from_raw_parts(buffer.as_ptr().cast::<u8>(), len) }
}

/// Allocation-free streaming XXH3-64 state.
///
/// Incremental writes produce the same result as hashing their concatenation
/// with [`super::Xxh3_64`]. Storage is fixed and independent of input size.
#[derive(Clone)]
pub struct Xxh3Hasher {
  acc: [u64; ACC_NB],
  custom_secret: MaybeUninit<[u8; DEFAULT_SECRET_SIZE]>,
  buffer: [MaybeUninit<u8>; INTERNAL_BUFFER_SIZE],
  buffered: usize,
  accumulated_stripes: usize,
  total_len: u64,
  seed: u64,
  stream_accumulate: Option<StreamAccumulateFn>,
}

impl Xxh3Hasher {
  /// Create an unseeded hasher.
  #[inline(always)]
  #[must_use]
  pub const fn new() -> Self {
    Self {
      acc: INITIAL_ACC,
      custom_secret: MaybeUninit::uninit(),
      buffer: [MaybeUninit::uninit(); INTERNAL_BUFFER_SIZE],
      buffered: 0,
      accumulated_stripes: 0,
      total_len: 0,
      seed: 0,
      stream_accumulate: None,
    }
  }

  /// Create a hasher with `seed`.
  #[inline]
  #[must_use]
  pub fn with_seed(seed: u64) -> Self {
    if seed == 0 {
      return Self::new();
    }
    Self {
      acc: INITIAL_ACC,
      custom_secret: MaybeUninit::new(custom_default_secret(seed)),
      buffer: [MaybeUninit::uninit(); INTERNAL_BUFFER_SIZE],
      buffered: 0,
      accumulated_stripes: 0,
      total_len: 0,
      seed,
      stream_accumulate: None,
    }
  }

  /// Reset the state without changing its seed.
  #[inline(always)]
  pub fn reset(&mut self) {
    self.acc = INITIAL_ACC;
    self.buffered = 0;
    self.accumulated_stripes = 0;
    self.total_len = 0;
  }

  #[inline(always)]
  fn secret(&self) -> &[u8; DEFAULT_SECRET_SIZE] {
    if self.seed == 0 {
      &DEFAULT_SECRET
    } else {
      // SAFETY: Borrowing the custom secret because:
      // 1. `with_seed` initializes `custom_secret` for every nonzero seed.
      // 2. `seed` is immutable after construction, so the initialized state cannot change.
      // 3. The returned lifetime is bounded by `self`.
      unsafe { self.custom_secret.assume_init_ref() }
    }
  }

  #[cold]
  #[inline(never)]
  fn write_inner(&mut self, bytes: &[u8]) {
    let Self {
      acc,
      custom_secret,
      buffer,
      buffered,
      accumulated_stripes,
      total_len: _,
      seed,
      stream_accumulate,
    } = self;
    let secret = if *seed == 0 {
      &DEFAULT_SECRET
    } else {
      // SAFETY: Borrowing the custom secret because:
      // 1. `with_seed` initializes it whenever `seed != 0`.
      // 2. This method never changes either `seed` or `custom_secret`.
      // 3. The shared secret borrow is disjoint from the mutable state fields used below.
      unsafe { custom_secret.assume_init_ref() }
    };

    let mut input = bytes;
    if *buffered > 0 {
      let fill = INTERNAL_BUFFER_SIZE.strict_sub(*buffered);
      write_buffer(buffer, *buffered, &input[..fill]);
      input = &input[fill..];
      let full = initialized_buffer(buffer, INTERNAL_BUFFER_SIZE);
      let kernel = *stream_accumulate.get_or_insert_with(dispatch::stream_accumulate_fn);
      (*acc, *accumulated_stripes) = consume_stripes(
        kernel,
        *acc,
        INTERNAL_BUFFER_STRIPES,
        *accumulated_stripes,
        full,
        secret,
      );
      *buffered = 0;
    }

    let direct_len = input.len().saturating_sub(1) / INTERNAL_BUFFER_SIZE * INTERNAL_BUFFER_SIZE;
    let processed_direct = direct_len != 0;
    if processed_direct {
      let kernel = *stream_accumulate.get_or_insert_with(dispatch::stream_accumulate_fn);
      (*acc, *accumulated_stripes) = consume_stripes(
        kernel,
        *acc,
        direct_len / STRIPE_LEN,
        *accumulated_stripes,
        &input[..direct_len],
        secret,
      );
      input = &input[direct_len..];
    }

    if processed_direct {
      let previous_tail_start = bytes.len().strict_sub(input.len()).strict_sub(STRIPE_LEN);
      write_buffer(
        buffer,
        INTERNAL_BUFFER_SIZE.strict_sub(STRIPE_LEN),
        &bytes[previous_tail_start..previous_tail_start.strict_add(STRIPE_LEN)],
      );
    }

    write_buffer(buffer, 0, input);
    *buffered = input.len();
  }

  #[inline(never)]
  fn final_acc(&self) -> [u64; ACC_NB] {
    let secret = self.secret();
    let stream_accumulate = self.stream_accumulate.unwrap_or_else(dispatch::stream_accumulate_fn);
    let mut acc = self.acc;
    let buffer = initialized_buffer(&self.buffer, self.buffered);

    if buffer.len() >= STRIPE_LEN {
      let stripes = buffer.len().strict_sub(1) / STRIPE_LEN;
      (acc, _) = consume_stripes(
        stream_accumulate,
        acc,
        stripes,
        self.accumulated_stripes,
        buffer,
        secret,
      );
      acc = accumulate_512(
        acc,
        &buffer[buffer.len().strict_sub(STRIPE_LEN)..],
        &secret[DEFAULT_SECRET_SIZE - STRIPE_LEN - SECRET_LASTACC_START..DEFAULT_SECRET_SIZE - SECRET_LASTACC_START],
      );
    } else {
      debug_assert!(!buffer.is_empty());
      let catchup = STRIPE_LEN.strict_sub(buffer.len());
      let old_start = INTERNAL_BUFFER_SIZE.strict_sub(STRIPE_LEN).strict_add(buffer.len());
      // SAFETY: Reconstructing the final stripe because:
      // 1. Long-state updates always retain the preceding 64-byte stripe at buffer offsets 192..256.
      // 2. `old_start..old_start + catchup` lies within that initialized retained stripe.
      // 3. `buffer` is the initialized current prefix and the two copies exactly fill `last_stripe`.
      let old_tail = unsafe { core::slice::from_raw_parts(self.buffer.as_ptr().cast::<u8>().add(old_start), catchup) };
      let mut last_stripe = [0u8; STRIPE_LEN];
      last_stripe[..catchup].copy_from_slice(old_tail);
      last_stripe[catchup..].copy_from_slice(buffer);
      acc = accumulate_512(
        acc,
        &last_stripe,
        &secret[DEFAULT_SECRET_SIZE - STRIPE_LEN - SECRET_LASTACC_START..DEFAULT_SECRET_SIZE - SECRET_LASTACC_START],
      );
    }

    acc
  }

  #[inline(never)]
  fn digest_long(&self) -> u64 {
    merge_accs(
      &self.final_acc(),
      self.secret(),
      SECRET_MERGEACCS_START,
      self.total_len.wrapping_mul(PRIME64_1),
    )
  }

  #[inline(never)]
  fn digest128_long(&self) -> u128 {
    let acc = self.final_acc();
    let secret = self.secret();
    let lo = merge_accs(
      &acc,
      secret,
      SECRET_MERGEACCS_START,
      self.total_len.wrapping_mul(PRIME64_1),
    );
    let hi = merge_accs(
      &acc,
      secret,
      DEFAULT_SECRET_SIZE
        .strict_sub(ACC_NB.strict_mul(core::mem::size_of::<u64>()))
        .strict_sub(SECRET_MERGEACCS_START),
      !self.total_len.wrapping_mul(PRIME64_2),
    );
    (lo as u128) | ((hi as u128) << 64)
  }

  #[inline(always)]
  fn digest(&self) -> u64 {
    if self.total_len <= MID_SIZE_MAX as u64 {
      dispatch::hash64_with_seed(self.seed, initialized_buffer(&self.buffer, self.buffered))
    } else {
      self.digest_long()
    }
  }

  #[inline(always)]
  fn digest128(&self) -> u128 {
    if self.total_len <= MID_SIZE_MAX as u64 {
      dispatch::hash128_with_seed(self.seed, initialized_buffer(&self.buffer, self.buffered))
    } else {
      self.digest128_long()
    }
  }
}

impl Default for Xxh3Hasher {
  #[inline(always)]
  fn default() -> Self {
    Self::new()
  }
}

impl core::fmt::Debug for Xxh3Hasher {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Xxh3Hasher")
      .field("seed", &self.seed)
      .field("buffered", &self.buffered)
      .field("total_len", &self.total_len)
      .finish_non_exhaustive()
  }
}

impl Hasher for Xxh3Hasher {
  #[inline(always)]
  fn write(&mut self, bytes: &[u8]) {
    self.total_len = self.total_len.strict_add(bytes.len() as u64);
    if bytes.len() <= INTERNAL_BUFFER_SIZE.strict_sub(self.buffered) {
      write_buffer(&mut self.buffer, self.buffered, bytes);
      self.buffered = self.buffered.strict_add(bytes.len());
    } else {
      self.write_inner(bytes);
    }
  }

  #[inline(always)]
  fn finish(&self) -> u64 {
    self.digest()
  }
}

/// Allocation-free streaming XXH3-128 state.
///
/// Incremental writes produce the same result as hashing their concatenation
/// with [`super::Xxh3_128`]. Storage is fixed and independent of input size.
#[derive(Clone, Debug, Default)]
pub struct Xxh3_128Hasher {
  inner: Xxh3Hasher,
}

impl Xxh3_128Hasher {
  /// Create an unseeded hasher.
  #[inline(always)]
  #[must_use]
  pub const fn new() -> Self {
    Self {
      inner: Xxh3Hasher::new(),
    }
  }

  /// Create a hasher with `seed`.
  #[inline]
  #[must_use]
  pub fn with_seed(seed: u64) -> Self {
    Self {
      inner: Xxh3Hasher::with_seed(seed),
    }
  }

  /// Append bytes to the hash stream.
  #[inline(always)]
  pub fn write(&mut self, bytes: &[u8]) {
    self.inner.write(bytes);
  }

  /// Reset the state without changing its seed.
  #[inline(always)]
  pub fn reset(&mut self) {
    self.inner.reset();
  }

  /// Return the current 128-bit hash without consuming the state.
  #[inline(always)]
  #[must_use]
  pub fn finish(&self) -> u128 {
    self.inner.digest128()
  }
}

/// Deterministic [`BuildHasher`] producing allocation-free [`Xxh3Hasher`]
/// instances for trusted collection keys.
///
/// The default seed is fixed at zero. Do not use this builder when an attacker
/// can choose keys; retain the standard library's randomized map hasher or use
/// another randomized collision-resistant builder for adversarial input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Xxh3BuildHasher {
  seed: u64,
}

impl Xxh3BuildHasher {
  /// Create a deterministic builder with seed zero.
  #[inline(always)]
  #[must_use]
  pub const fn new() -> Self {
    Self { seed: 0 }
  }

  /// Create a deterministic builder with `seed`.
  #[inline(always)]
  #[must_use]
  pub const fn with_seed(seed: u64) -> Self {
    Self { seed }
  }
}

impl Default for Xxh3BuildHasher {
  #[inline(always)]
  fn default() -> Self {
    Self::new()
  }
}

impl BuildHasher for Xxh3BuildHasher {
  type Hasher = Xxh3Hasher;

  #[inline(always)]
  fn build_hasher(&self) -> Self::Hasher {
    Xxh3Hasher::with_seed(self.seed)
  }
}

#[cfg(test)]
mod tests {
  use alloc::vec::Vec;

  use super::*;
  use crate::traits::FastHash;

  fn data(len: usize) -> Vec<u8> {
    (0..len).map(|i| i.wrapping_mul(131).wrapping_add(17) as u8).collect()
  }

  #[test]
  fn incremental_writes_match_one_shot_across_chunk_boundaries() {
    #[cfg(not(miri))]
    let mut lengths: Vec<usize> = (0..=320).collect();
    #[cfg(not(miri))]
    lengths.extend([383, 384, 385, 511, 512, 513, 1024, 4096]);
    #[cfg(miri)]
    let lengths = [0, 1, 16, 17, 239, 240, 241, 255, 256, 257, 319, 320, 321, 512, 513];
    #[cfg(not(miri))]
    let chunk_sizes = [
      1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 239, 240, 241, 255, 256, 257,
    ];
    #[cfg(miri)]
    let chunk_sizes = [1, 7, 16, 17, 63, 64, 65, 255, 256, 257];
    #[cfg(not(miri))]
    let seeds = [0, 1, u64::MAX, 0x243f_6a88_85a3_08d3];
    #[cfg(miri)]
    let seeds = [0, 42];

    for seed in seeds {
      for &len in &lengths {
        let input = data(len);
        let expected = super::super::Xxh3_64::hash_with_seed(seed, &input);
        for &chunk_size in &chunk_sizes {
          let mut hasher = Xxh3Hasher::with_seed(seed);
          for chunk in input.chunks(chunk_size) {
            hasher.write(chunk);
          }
          assert_eq!(
            hasher.finish(),
            expected,
            "seed={seed:#x}, len={len}, chunk={chunk_size}"
          );
          assert_eq!(hasher.finish(), expected, "finish must not mutate state");
        }
      }
    }
  }

  #[test]
  fn reset_preserves_seed_and_clears_stream_state() {
    let input = data(513);
    let mut hasher = Xxh3Hasher::with_seed(42);
    hasher.write(&input);
    hasher.reset();
    hasher.write(&input[..117]);
    assert_eq!(
      hasher.finish(),
      super::super::Xxh3_64::hash_with_seed(42, &input[..117])
    );
  }

  #[test]
  fn incremental_128bit_writes_match_one_shot_across_chunk_boundaries() {
    #[cfg(not(miri))]
    let lengths = [
      0, 1, 16, 17, 127, 128, 129, 239, 240, 241, 255, 256, 257, 511, 512, 513, 4096,
    ];
    #[cfg(miri)]
    let lengths = [0, 16, 17, 239, 240, 241, 255, 256, 257, 513];
    let chunk_sizes = [1, 7, 16, 63, 64, 127, 255, 256, 257];
    let seeds = [0, 1, 0x243f_6a88_85a3_08d3];

    for seed in seeds {
      for len in lengths {
        let input = data(len);
        let expected = super::super::Xxh3_128::hash_with_seed(seed, &input);
        for chunk_size in chunk_sizes {
          let mut hasher = Xxh3_128Hasher::with_seed(seed);
          for chunk in input.chunks(chunk_size) {
            hasher.write(chunk);
          }
          assert_eq!(
            hasher.finish(),
            expected,
            "seed={seed:#x}, len={len}, chunk={chunk_size}"
          );
          assert_eq!(hasher.finish(), expected, "finish must not mutate state");
        }
      }
    }
  }

  #[test]
  #[cfg(not(miri))]
  fn streaming_simd_kernels_match_portable_accumulator() {
    use super::super::kernels::{Xxh3KernelId, required_caps, stream_accumulate_fn};

    let input = data(INTERNAL_BUFFER_SIZE);
    let secret = custom_default_secret(42);
    let expected = stream_accumulate_fn(Xxh3KernelId::Portable)(INITIAL_ACC, &input, 0, &secret, 8, 4, true);
    #[cfg(target_arch = "x86_64")]
    let kernels = &[Xxh3KernelId::Avx2, Xxh3KernelId::Avx512][..];
    #[cfg(target_arch = "aarch64")]
    let kernels = &[Xxh3KernelId::Neon][..];
    #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
    let kernels = &[Xxh3KernelId::Vsx][..];
    #[cfg(target_arch = "s390x")]
    let kernels = &[Xxh3KernelId::Vector][..];
    #[cfg(not(any(
      target_arch = "x86_64",
      target_arch = "aarch64",
      all(target_arch = "powerpc64", target_endian = "little"),
      target_arch = "s390x"
    )))]
    let kernels = &[][..];

    let caps = crate::platform::caps();
    for &kernel in kernels {
      if caps.has(required_caps(kernel)) {
        assert_eq!(
          stream_accumulate_fn(kernel)(INITIAL_ACC, &input, 0, &secret, 8, 4, true),
          expected
        );
      }
    }
  }
}
