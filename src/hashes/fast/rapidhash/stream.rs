use core::hash::{BuildHasher, Hasher};

use super::{
  DEFAULT_SECRETS, RapidSecrets, premix_seed, rapid_mix, rapid_mum, rapidhash_core, rapidhash_seed_cpp, read_u32_le,
  read_u64_le, u128_words,
};

const CHUNK_SIZE: usize = 112;
const PREVIOUS_TAIL_SIZE: usize = 16;
const BUFFER_SIZE: usize = PREVIOUS_TAIL_SIZE + CHUNK_SIZE;
/// Allocation-free streaming rapidhash V3 state.
///
/// Incremental writes produce the same result as hashing their concatenation
/// with [`super::RapidHash64`]. Storage is fixed and independent of input size.
#[derive(Clone)]
pub struct RapidStreamHasher {
  initial_seed: u64,
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
      Self::write_chunk(lanes, &DEFAULT_SECRETS, chunk);
      &data[copy_len..]
    };

    let stop = remaining
      .len()
      .saturating_sub(1)
      .strict_div(CHUNK_SIZE)
      .strict_mul(CHUNK_SIZE);
    let mut last_chunk = None;
    let mut offset = 0usize;
    while offset < stop {
      let end = offset.strict_add(CHUNK_SIZE);
      let chunk = &remaining[offset..end].as_chunks::<CHUNK_SIZE>().0[0];
      Self::write_chunk(lanes, &DEFAULT_SECRETS, chunk);
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
        seed = rapid_mix(read_u64_le(tail, 0) ^ DEFAULT_SECRETS[2], read_u64_le(tail, 8) ^ seed);
        for (offset, secret) in [(16usize, 2usize), (32, 1), (48, 1), (64, 2), (80, 1)] {
          if tail.len() > offset.strict_add(16) {
            seed = rapid_mix(
              read_u64_le(tail, offset) ^ DEFAULT_SECRETS[secret],
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

    a ^= DEFAULT_SECRETS[1];
    b ^= seed;
    (a, b) = rapid_mum(a, b);
    rapid_mix(a ^ 0xaaaa_aaaa_aaaa_aaaa, b ^ DEFAULT_SECRETS[1] ^ remainder)
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

/// Allocation-free hasher for `HashMap` and `HashSet` keys.
///
/// Unlike [`RapidStreamHasher`], separate writes need not equal hashing their
/// concatenation. Its collection-oriented output is not a stable
/// C++-compatible fingerprint.
///
/// Construct it through [`RapidSeededState`] for reproducible, trusted-key
/// collections. [`RapidRandomState`] adds per-state randomization when
/// RapidHash's limited HashDoS hardening is sufficient. It intentionally has no
/// public constructor or deterministic default.
#[derive(Clone, Copy)]
pub struct RapidHasher {
  seed: u64,
  random_word0: u64,
  sponge: u128,
  sponge_bits: u8,
}

impl RapidHasher {
  #[inline]
  const fn deterministic(seed: u64) -> Self {
    Self {
      seed,
      random_word0: 0,
      sponge: 0,
      sponge_bits: 0,
    }
  }

  #[inline]
  const fn randomized(seed: u64, random_word0: u64) -> Self {
    Self {
      seed,
      random_word0,
      sponge: 0,
      sponge_bits: 0,
    }
  }

  #[inline(always)]
  const fn word0(&self) -> u64 {
    if self.random_word0 == 0 {
      DEFAULT_SECRETS[0]
    } else {
      self.random_word0
    }
  }

  #[inline(always)]
  fn flush_sponge(&mut self) {
    if self.sponge_bits != 0 {
      let (low, high) = u128_words(self.sponge);
      self.seed = rapid_mix(low ^ self.seed, high ^ self.word0());
      self.sponge = 0;
      self.sponge_bits = 0;
    }
  }

  #[inline(always)]
  fn digest(&self) -> u64 {
    if self.sponge_bits == 0 {
      self.seed
    } else {
      let (low, high) = u128_words(self.sponge);
      rapid_mix(low ^ self.seed, high ^ self.word0())
    }
  }
}

impl core::fmt::Debug for RapidHasher {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("RapidHasher").finish_non_exhaustive()
  }
}

// Keep byte hashing out of fixed-size key callers; inlining either schedule bloats collection
// paths.
#[inline(never)]
fn hash_randomized(bytes: &[u8], seed: u64, random_word0: u64) -> u64 {
  let state = RapidSecrets::derived(seed, random_word0);
  rapidhash_core(bytes, state.seed, &state.words)
}

#[inline(never)]
fn hash_deterministic(bytes: &[u8], seed: u64) -> u64 {
  rapidhash_core(bytes, seed, &DEFAULT_SECRETS)
}

macro_rules! write_integer {
  (@convert unsigned, $value:expr) => {
    $value as u128
  };
  (@convert signed, $value:expr) => {
    $value.cast_unsigned() as u128
  };
  ($kind:ident; $($method:ident, $ty:ty),+ $(,)?) => {
    $(
      #[inline(always)]
      fn $method(&mut self, value: $ty) {
        let bits = u8::try_from(<$ty>::BITS).expect("Rust integer width should fit in u8");
        let value = write_integer!(@convert $kind, value);
        let next_bits = self.sponge_bits.strict_add(bits);
        if next_bits <= 128 {
          self.sponge |= value << self.sponge_bits;
          self.sponge_bits = next_bits;
        } else {
          self.flush_sponge();
          self.sponge = value;
          self.sponge_bits = bits;
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
    self.seed = if self.random_word0 == 0 {
      hash_deterministic(bytes, self.seed)
    } else {
      hash_randomized(bytes, self.seed, self.random_word0)
    };
  }

  #[inline(always)]
  fn finish(&self) -> u64 {
    self.digest()
  }

  write_integer!(
    unsigned;
    write_u8, u8,
    write_u16, u16,
    write_u32, u32,
    write_u64, u64,
    write_usize, usize,
  );
  write_integer!(
    signed;
    write_i8, i8,
    write_i16, i16,
    write_i32, i32,
    write_i64, i64,
    write_isize, isize,
  );

  #[inline(always)]
  fn write_u128(&mut self, value: u128) {
    self.flush_sponge();
    self.sponge = value;
    self.sponge_bits = 128;
  }

  #[inline(always)]
  fn write_i128(&mut self, value: i128) {
    self.write_u128(value.cast_unsigned());
  }
}

/// Deterministic [`BuildHasher`] for reproducible, trusted collection keys.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RapidSeededState {
  seed: u64,
}

impl RapidSeededState {
  /// Create a deterministic builder with `seed`.
  #[inline(always)]
  #[must_use]
  pub const fn new(seed: u64) -> Self {
    Self {
      seed: rapidhash_seed_cpp(seed),
    }
  }
}

impl BuildHasher for RapidSeededState {
  type Hasher = RapidHasher;

  #[inline]
  fn build_hasher(&self) -> Self::Hasher {
    RapidHasher::deterministic(self.seed)
  }
}

/// Fallible randomized [`BuildHasher`] with a per-state seed and secret schedule.
///
/// This changes collision behavior between states, but RapidHash remains
/// non-cryptographic.
#[derive(Clone)]
pub struct RapidRandomState {
  seed: u64,
  word0: u64,
}

impl RapidRandomState {
  /// Create a randomized state with caller-provided entropy.
  ///
  /// The callback must fill the entire supplied buffer before returning `Ok`.
  pub fn try_new_with<E>(fill: impl FnOnce(&mut [u8]) -> Result<(), E>) -> Result<Self, E> {
    let mut seed = [0u8; 8];
    fill(&mut seed)?;
    let seed = premix_seed(u64::from_le_bytes(seed), 0);
    Ok(Self {
      seed,
      word0: premix_seed(seed, 0),
    })
  }

  /// Create a randomized state from the platform entropy source.
  #[cfg(feature = "getrandom")]
  pub fn try_new() -> Result<Self, getrandom::Error> {
    Self::try_new_with(getrandom::fill)
  }
}

impl BuildHasher for RapidRandomState {
  type Hasher = RapidHasher;

  #[inline]
  fn build_hasher(&self) -> Self::Hasher {
    RapidHasher::randomized(self.seed, self.word0)
  }
}

impl core::fmt::Debug for RapidRandomState {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("RapidRandomState").finish_non_exhaustive()
  }
}

#[cfg(test)]
mod tests {
  use alloc::vec::Vec;
  use core::hash::{BuildHasher, Hash};

  #[cfg(not(miri))]
  use proptest::prelude::*;

  use super::*;

  fn data(len: usize) -> Vec<u8> {
    (0..len)
      .map(|i| i.wrapping_mul(131).wrapping_add(17).to_le_bytes()[0])
      .collect()
  }

  fn seeded_hasher(seed: u64) -> RapidHasher {
    RapidSeededState::new(seed).build_hasher()
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
    let mut hasher = seeded_hasher(42);
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
    let mut integer_then_bytes = seeded_hasher(0);
    integer_then_bytes.write_u64(7);
    integer_then_bytes.write(b"field");

    let mut bytes_then_integer = seeded_hasher(0);
    bytes_then_integer.write(b"field");
    bytes_then_integer.write_u64(7);

    assert_ne!(integer_then_bytes.finish(), bytes_then_integer.finish());
  }

  #[test]
  fn seeded_states_are_reproducible_and_seed_separated() {
    for seed in [0, 1, u64::MAX, 0x243f_6a88_85a3_08d3] {
      let state = RapidSeededState::new(seed);
      assert_eq!(state.hash_one(b"collection key"), state.hash_one(b"collection key"));
    }
    assert_ne!(
      RapidSeededState::new(1).hash_one(b"collection key"),
      RapidSeededState::new(2).hash_one(b"collection key")
    );
  }

  #[test]
  fn rapid_states_produce_rapid_hasher() {
    fn assert_builder<B: BuildHasher<Hasher = RapidHasher>>() {}
    assert_builder::<RapidSeededState>();
    assert_builder::<RapidRandomState>();
  }

  #[test]
  fn random_state_uses_fallible_caller_entropy() {
    let first = RapidRandomState::try_new_with(|seed| {
      seed.copy_from_slice(&1u64.to_le_bytes());
      Ok::<_, ()>(())
    })
    .expect("caller-provided entropy should initialize the first state");
    let same = RapidRandomState::try_new_with(|seed| {
      seed.copy_from_slice(&1u64.to_le_bytes());
      Ok::<_, ()>(())
    })
    .expect("caller-provided entropy should initialize the matching state");
    let second = RapidRandomState::try_new_with(|seed| {
      seed.copy_from_slice(&2u64.to_le_bytes());
      Ok::<_, ()>(())
    })
    .expect("caller-provided entropy should initialize the second state");

    assert_eq!(first.hash_one(b"collection key"), same.hash_one(b"collection key"));
    assert_ne!(first.hash_one(b"collection key"), second.hash_one(b"collection key"));
    RapidRandomState::try_new_with(|_| Err::<(), _>("entropy unavailable"))
      .expect_err("entropy-source failures should be returned");
  }

  #[test]
  fn rapid_hasher_specialized_integer_methods_match_hash_trait() {
    macro_rules! assert_integer {
      ($method:ident, $value:expr) => {{
        let value = $value;
        let mut direct = seeded_hasher(42);
        direct.$method(value);
        let mut via_hash = seeded_hasher(42);
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
      let mut direct = seeded_hasher(seed);
      direct.write_u128(value);
      let mut via_hash = seeded_hasher(seed);
      value.hash(&mut via_hash);
      prop_assert_eq!(direct.finish(), via_hash.finish());
    }

    #[test]
    fn rapid_hasher_mixed_fields_are_deterministic(value in any::<u64>(), bytes in proptest::collection::vec(any::<u8>(), 0..256), seed in any::<u64>()) {
      let mut first = seeded_hasher(seed);
      first.write_u64(value);
      first.write(&bytes);
      let mut second = seeded_hasher(seed);
      value.hash(&mut second);
      second.write(&bytes);
      prop_assert_eq!(first.finish(), second.finish());
    }
  }
}
