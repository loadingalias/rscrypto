use core::hash::{BuildHasher, Hasher};

use super::{
  DEFAULT_SECRETS, likely, rapid_mix, rapid_mum, rapidhash_fast_core_medium, rapidhash_seed_cpp, read_u32_le,
  read_u32_np, read_u64_le, read_u64_np,
};

const CHUNK_SIZE: usize = 112;
const PREVIOUS_TAIL_SIZE: usize = 16;
const BUFFER_SIZE: usize = PREVIOUS_TAIL_SIZE + CHUNK_SIZE;
// Indirection avoids materializing seven 64-bit constants in each cold streaming call.
static STREAM_SECRETS: [u64; 7] = DEFAULT_SECRETS;

#[inline(always)]
fn map_finish(mut a: u64, mut b: u64, seed: u64) -> u64 {
  a ^= DEFAULT_SECRETS[0];
  b ^= seed;
  (a, b) = rapid_mum(a, b);
  a ^ b
}

#[inline(always)]
fn map_hash_bytes(data: &[u8], mut seed: u64) -> u64 {
  if data.len() <= 16 {
    let (mut a, mut b) = (0, 0);
    if data.len() >= 8 {
      a = read_u64_np(data, 0);
      b = read_u64_np(data, data.len().strict_sub(8));
    } else if data.len() >= 4 {
      a = read_u32_np(data, 0) as u64;
      b = read_u32_np(data, data.len().strict_sub(4)) as u64;
    } else if !data.is_empty() {
      a = ((data[0] as u64) << 45) | data[data.len().strict_sub(1)] as u64;
      b = data[data.len() >> 1] as u64;
    }
    seed = seed.wrapping_add(data.len() as u64);
    map_finish(a, b, seed)
  } else {
    map_hash_bytes_long(data, seed)
  }
}

#[cold]
#[inline(never)]
fn map_hash_bytes_long(data: &[u8], mut seed: u64) -> u64 {
  debug_assert!(data.len() > 16);
  if data.len() <= 48 {
    seed = rapid_mix(read_u64_np(data, 0) ^ DEFAULT_SECRETS[0], read_u64_np(data, 8) ^ seed);
    if data.len() > 32 {
      seed = rapid_mix(read_u64_np(data, 16) ^ DEFAULT_SECRETS[0], read_u64_np(data, 24) ^ seed);
    }
    seed = seed.wrapping_add(data.len() as u64);
    map_finish(
      read_u64_np(data, data.len().strict_sub(16)),
      read_u64_np(data, data.len().strict_sub(8)),
      seed,
    )
  } else {
    // SAFETY: 1. This branch is reached only for inputs longer than 48 bytes, satisfying the
    // kernel's `data.len() > 16` precondition.
    unsafe { rapidhash_fast_core_medium(data, seed) }
  }
}

/// Allocation-free streaming rapidhash V3 state.
///
/// Incremental writes produce the same result as hashing their concatenation
/// with [`super::RapidHash64`]. Storage is fixed and independent of input size.
#[derive(Clone)]
pub struct RapidStreamHasher {
  initial_seed: u64,
  secrets: &'static [u64; 7],
  lanes: Option<[u64; 7]>,
  buffer: [u8; BUFFER_SIZE],
  buffered: usize,
}

impl RapidStreamHasher {
  /// Create an unseeded hasher.
  #[inline(always)]
  #[must_use]
  pub const fn new() -> Self {
    Self::with_seed(0)
  }

  /// Create a hasher with `seed`.
  #[inline(always)]
  #[must_use]
  pub const fn with_seed(seed: u64) -> Self {
    let initial_seed = rapidhash_seed_cpp(seed);
    Self {
      initial_seed,
      secrets: &STREAM_SECRETS,
      lanes: None,
      buffer: [0; BUFFER_SIZE],
      buffered: 0,
    }
  }

  /// Reset the state without changing its seed.
  #[inline(always)]
  pub fn reset(&mut self) {
    self.lanes = None;
    self.buffered = 0;
  }

  #[inline(always)]
  fn write_chunk(lanes: &mut [u64; 7], secrets: &[u64; 7], chunk: &[u8; CHUNK_SIZE]) {
    let mut lane = 0usize;
    while lane < lanes.len() {
      let offset = lane.strict_mul(16);
      lanes[lane] = rapid_mix(
        read_u64_le(chunk, offset) ^ secrets[lane],
        read_u64_le(chunk, offset.strict_add(8)) ^ lanes[lane],
      );
      lane = lane.strict_add(1);
    }
  }

  #[cold]
  #[inline(never)]
  fn write_inner(&mut self, data: &[u8]) {
    let lanes = self.lanes.get_or_insert([self.initial_seed; 7]);
    let remaining = if self.buffered == 0 {
      data
    } else {
      let copy_len = CHUNK_SIZE.strict_sub(self.buffered);
      let chunk_start = PREVIOUS_TAIL_SIZE.strict_add(self.buffered);
      self.buffer[chunk_start..BUFFER_SIZE].copy_from_slice(&data[..copy_len]);
      let chunk = &self.buffer[PREVIOUS_TAIL_SIZE..].as_chunks::<CHUNK_SIZE>().0[0];
      Self::write_chunk(lanes, self.secrets, chunk);
      &data[copy_len..]
    };

    let stop = remaining.len().saturating_sub(1) / CHUNK_SIZE * CHUNK_SIZE;
    let mut last_chunk = None;
    let mut offset = 0usize;
    while offset < stop {
      let end = offset.strict_add(CHUNK_SIZE);
      let chunk = &remaining[offset..end].as_chunks::<CHUNK_SIZE>().0[0];
      Self::write_chunk(lanes, self.secrets, chunk);
      last_chunk = Some(chunk);
      offset = end;
    }

    if let Some(chunk) = last_chunk {
      self.buffer[..PREVIOUS_TAIL_SIZE].copy_from_slice(&chunk[CHUNK_SIZE - PREVIOUS_TAIL_SIZE..]);
    } else {
      debug_assert!(self.buffered != 0);
      self
        .buffer
        .copy_within(BUFFER_SIZE - PREVIOUS_TAIL_SIZE..BUFFER_SIZE, 0);
    }

    let tail = &remaining[offset..];
    self.buffer[PREVIOUS_TAIL_SIZE..PREVIOUS_TAIL_SIZE.strict_add(tail.len())].copy_from_slice(tail);
    self.buffered = tail.len();
  }

  #[inline(always)]
  fn digest(&self) -> u64 {
    let mut seed = self.initial_seed;
    let (mut a, mut b, remainder);

    if self.lanes.is_none() && self.buffered <= 16 {
      let data = &self.buffer[PREVIOUS_TAIL_SIZE..PREVIOUS_TAIL_SIZE.strict_add(self.buffered)];
      if data.len() >= 4 {
        seed ^= data.len() as u64;
        if data.len() >= 8 {
          a = read_u64_le(data, 0);
          b = read_u64_le(data, data.len().strict_sub(8));
        } else {
          a = read_u32_le(data, 0) as u64;
          b = read_u32_le(data, data.len().strict_sub(4)) as u64;
        }
      } else if !data.is_empty() {
        a = ((data[0] as u64) << 45) | data[data.len().strict_sub(1)] as u64;
        b = data[data.len() >> 1] as u64;
      } else {
        a = 0;
        b = 0;
      }
      remainder = data.len() as u64;
    } else {
      if let Some(lanes) = self.lanes {
        seed = lanes.into_iter().fold(0, |merged, lane| merged ^ lane);
      }

      let tail = &self.buffer[PREVIOUS_TAIL_SIZE..PREVIOUS_TAIL_SIZE.strict_add(self.buffered)];
      if tail.len() > 16 {
        seed = rapid_mix(read_u64_le(tail, 0) ^ self.secrets[2], read_u64_le(tail, 8) ^ seed);
        for (offset, secret) in [(16usize, 2usize), (32, 1), (48, 1), (64, 2), (80, 1)] {
          if tail.len() > offset.strict_add(16) {
            seed = rapid_mix(
              read_u64_le(tail, offset) ^ self.secrets[secret],
              read_u64_le(tail, offset.strict_add(8)) ^ seed,
            );
          }
        }
      }

      let data = &self.buffer[..PREVIOUS_TAIL_SIZE.strict_add(self.buffered)];
      a = read_u64_le(data, data.len().strict_sub(16)) ^ tail.len() as u64;
      b = read_u64_le(data, data.len().strict_sub(8));
      remainder = self.buffered as u64;
    }

    a ^= self.secrets[1];
    b ^= seed;
    (a, b) = rapid_mum(a, b);
    rapid_mix(a ^ 0xaaaa_aaaa_aaaa_aaaa, b ^ self.secrets[1] ^ remainder)
  }
}

impl Default for RapidStreamHasher {
  #[inline(always)]
  fn default() -> Self {
    Self::new()
  }
}

impl core::fmt::Debug for RapidStreamHasher {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("RapidStreamHasher")
      .field("buffered", &self.buffered)
      .field("processed", &self.lanes.is_some())
      .finish_non_exhaustive()
  }
}

impl Hasher for RapidStreamHasher {
  #[inline(always)]
  fn write(&mut self, bytes: &[u8]) {
    if bytes.len() <= CHUNK_SIZE.strict_sub(self.buffered) {
      let start = PREVIOUS_TAIL_SIZE.strict_add(self.buffered);
      let end = start.strict_add(bytes.len());
      self.buffer[start..end].copy_from_slice(bytes);
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

/// Compact allocation-free hasher for `HashMap` and `HashSet` keys.
///
/// Unlike [`RapidStreamHasher`], this uses collection-oriented field mixing
/// and omits the final avalanche. Its output is not a stable C++-compatible
/// fingerprint. Use [`RapidStreamHasher`] when writes must equal hashing their
/// concatenation.
#[derive(Clone, Copy)]
pub struct RapidHasher {
  seed: u64,
  sponge: u128,
  sponge_bits: u8,
}

impl RapidHasher {
  /// Create an unseeded map hasher.
  #[inline(always)]
  #[must_use]
  pub const fn new() -> Self {
    Self::with_seed(0)
  }

  /// Create a map hasher with `seed`.
  #[inline(always)]
  #[must_use]
  pub const fn with_seed(seed: u64) -> Self {
    Self {
      seed: rapidhash_seed_cpp(seed),
      sponge: 0,
      sponge_bits: 0,
    }
  }

  #[inline(always)]
  fn flush_sponge(&mut self) {
    if self.sponge_bits != 0 {
      self.seed = rapid_mix(
        self.sponge as u64 ^ self.seed,
        (self.sponge >> 64) as u64 ^ DEFAULT_SECRETS[0],
      );
      self.sponge = 0;
      self.sponge_bits = 0;
    }
  }

  #[inline(always)]
  fn digest(&self) -> u64 {
    if self.sponge_bits == 0 {
      self.seed
    } else {
      rapid_mix(
        self.sponge as u64 ^ self.seed,
        (self.sponge >> 64) as u64 ^ DEFAULT_SECRETS[0],
      )
    }
  }
}

impl Default for RapidHasher {
  #[inline(always)]
  fn default() -> Self {
    Self::new()
  }
}

impl core::fmt::Debug for RapidHasher {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("RapidHasher").finish_non_exhaustive()
  }
}

macro_rules! write_integer {
  ($($method:ident, $ty:ty, $unsigned:ty),+ $(,)?) => {
    $(
      #[inline(always)]
      fn $method(&mut self, value: $ty) {
        const BITS: u8 = core::mem::size_of::<$ty>() as u8 * 8;
        let value = (value as $unsigned) as u128;
        let next_bits = self.sponge_bits.wrapping_add(BITS);
        if likely(next_bits <= 128) {
          self.sponge |= value << self.sponge_bits;
          self.sponge_bits = next_bits;
        } else {
          self.flush_sponge();
          self.sponge = value;
          self.sponge_bits = BITS;
        }
      }
    )+
  };
}

impl Hasher for RapidHasher {
  #[inline(always)]
  fn write(&mut self, bytes: &[u8]) {
    if bytes.is_empty() {
      return;
    }
    self.flush_sponge();
    self.seed = map_hash_bytes(bytes, self.seed);
  }

  #[inline(always)]
  fn finish(&self) -> u64 {
    self.digest()
  }

  write_integer!(
    write_u8,
    u8,
    u8,
    write_u16,
    u16,
    u16,
    write_u32,
    u32,
    u32,
    write_u64,
    u64,
    u64,
    write_usize,
    usize,
    usize,
    write_i8,
    i8,
    u8,
    write_i16,
    i16,
    u16,
    write_i32,
    i32,
    u32,
    write_i64,
    i64,
    u64,
    write_isize,
    isize,
    usize,
  );

  #[inline(always)]
  fn write_u128(&mut self, value: u128) {
    self.flush_sponge();
    self.sponge = value;
    self.sponge_bits = 128;
  }

  #[inline(always)]
  fn write_i128(&mut self, value: i128) {
    self.write_u128(value as u128);
  }
}

/// Deterministic [`BuildHasher`] producing allocation-free [`RapidHasher`]
/// instances for trusted collection keys.
///
/// The default seed is fixed at zero. Do not use this builder when an attacker
/// can choose keys; retain the standard library's randomized map hasher or use
/// another randomized collision-resistant builder for adversarial input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RapidBuildHasher {
  seed: u64,
}

impl RapidBuildHasher {
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

impl Default for RapidBuildHasher {
  #[inline(always)]
  fn default() -> Self {
    Self::new()
  }
}

impl BuildHasher for RapidBuildHasher {
  type Hasher = RapidHasher;

  #[inline(always)]
  fn build_hasher(&self) -> Self::Hasher {
    RapidHasher::with_seed(self.seed)
  }
}

#[cfg(test)]
mod tests {
  use alloc::vec::Vec;
  use core::hash::{BuildHasher, Hash};

  #[cfg(not(miri))]
  use proptest::prelude::*;

  use super::*;
  use crate::traits::FastHash;

  fn data(len: usize) -> Vec<u8> {
    (0..len).map(|i| i.wrapping_mul(131).wrapping_add(17) as u8).collect()
  }

  #[test]
  fn incremental_writes_match_one_shot_across_chunk_boundaries() {
    #[cfg(not(miri))]
    let mut lengths: Vec<usize> = (0..=260).collect();
    #[cfg(not(miri))]
    lengths.extend([335, 336, 337, 447, 448, 449, 1024, 4096]);
    #[cfg(miri)]
    let lengths = [0, 1, 16, 17, 111, 112, 113, 127, 128, 129, 224, 225, 256, 337, 513];
    #[cfg(not(miri))]
    let chunk_sizes = [
      1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 63, 64, 111, 112, 113, 127, 128, 129, 257,
    ];
    #[cfg(miri)]
    let chunk_sizes = [1, 7, 16, 17, 111, 112, 113, 257];
    #[cfg(not(miri))]
    let seeds = [0, 1, u64::MAX, 0x243f_6a88_85a3_08d3];
    #[cfg(miri)]
    let seeds = [0, 42];

    for seed in seeds {
      for &len in &lengths {
        let input = data(len);
        let expected = super::super::RapidHash64::hash_with_seed(seed, &input);
        for &chunk_size in &chunk_sizes {
          let mut hasher = RapidStreamHasher::with_seed(seed);
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
    let mut hasher = RapidStreamHasher::with_seed(42);
    hasher.write(&input);
    hasher.reset();
    hasher.write(&input[..117]);
    assert_eq!(
      hasher.finish(),
      super::super::RapidHash64::hash_with_seed(42, &input[..117])
    );
  }

  #[test]
  fn rapid_hasher_finish_is_repeatable_and_clone_preserves_state() {
    let mut hasher = RapidHasher::with_seed(42);
    hasher.write_u64(0x0123_4567_89ab_cdef);
    hasher.write(b"field");
    let cloned = hasher;

    assert_eq!(hasher.finish(), hasher.finish());
    assert_eq!(cloned.finish(), hasher.finish());
  }

  #[test]
  fn rapid_stream_hasher_clone_preserves_partial_stream() {
    let mut original = RapidStreamHasher::with_seed(42);
    original.write(&data(173));
    let mut cloned = original.clone();
    original.write(b"tail");
    cloned.write(b"tail");

    assert_eq!(cloned.finish(), original.finish());
  }

  #[test]
  fn rapid_hasher_preserves_mixed_field_order() {
    let mut integer_then_bytes = RapidHasher::new();
    integer_then_bytes.write_u64(7);
    integer_then_bytes.write(b"field");

    let mut bytes_then_integer = RapidHasher::new();
    bytes_then_integer.write(b"field");
    bytes_then_integer.write_u64(7);

    assert_ne!(integer_then_bytes.finish(), bytes_then_integer.finish());
  }

  #[test]
  fn rapid_hasher_nonempty_byte_writes_match_upstream_fast() {
    for seed in [0, 1, u64::MAX, 0x243f_6a88_85a3_08d3] {
      for len in 1..=512 {
        let input = data(len);
        let mut ours = RapidHasher::with_seed(seed);
        ours.write(&input);
        let mut upstream = rapidhash::fast::RapidHasher::new(seed);
        upstream.write(&input);
        assert_eq!(ours.finish(), upstream.finish(), "seed={seed:#x}, len={len}");
      }
    }
  }

  #[test]
  fn rapid_build_hasher_produces_rapid_hasher() {
    fn assert_builder<B: BuildHasher<Hasher = RapidHasher>>() {}
    assert_builder::<RapidBuildHasher>();
  }

  #[test]
  fn rapid_hasher_specialized_integer_methods_match_hash_trait() {
    macro_rules! assert_integer {
      ($method:ident, $value:expr) => {{
        let value = $value;
        let mut direct = RapidHasher::with_seed(42);
        direct.$method(value);
        let mut via_hash = RapidHasher::with_seed(42);
        value.hash(&mut via_hash);
        assert_eq!(direct.finish(), via_hash.finish(), stringify!($method));
      }};
    }

    assert_integer!(write_u8, 0xa5u8);
    assert_integer!(write_u16, 0xa5c3u16);
    assert_integer!(write_u32, 0xa5c3_17e9u32);
    assert_integer!(write_u64, 0xa5c3_17e9_6b4d_2f01u64);
    assert_integer!(write_u128, 0xa5c3_17e9_6b4d_2f01_0123_4567_89ab_cdefu128);
    assert_integer!(write_usize, usize::MAX.strict_sub(17));
    assert_integer!(write_i8, -37i8);
    assert_integer!(write_i16, -12_345i16);
    assert_integer!(write_i32, -1_234_567i32);
    assert_integer!(write_i64, -1_234_567_890_123i64);
    assert_integer!(write_i128, -1_234_567_890_123_456_789i128);
    assert_integer!(write_isize, -17isize);
  }

  #[cfg(not(miri))]
  proptest! {
    #[test]
    fn rapid_hasher_integer_methods_match_primitive_hash(value in any::<u128>(), seed in any::<u64>()) {
      let mut direct = RapidHasher::with_seed(seed);
      direct.write_u128(value);
      let mut via_hash = RapidHasher::with_seed(seed);
      value.hash(&mut via_hash);
      prop_assert_eq!(direct.finish(), via_hash.finish());
    }

    #[test]
    fn rapid_hasher_mixed_fields_are_deterministic(value in any::<u64>(), bytes in proptest::collection::vec(any::<u8>(), 0..256), seed in any::<u64>()) {
      let mut first = RapidHasher::with_seed(seed);
      first.write_u64(value);
      first.write(&bytes);
      let mut second = RapidHasher::with_seed(seed);
      value.hash(&mut second);
      second.write(&bytes);
      prop_assert_eq!(first.finish(), second.finish());
    }
  }
}
