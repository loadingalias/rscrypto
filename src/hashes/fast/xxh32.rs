//! XXH32, the original 32-bit xxHash (**NOT CRYPTO**).
//!
//! This is XXH32 from the
//! [xxHash specification](https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md),
//! not XXH3. LZ4 frames use it with seed zero for their descriptor, block, and
//! content checksums. It detects accidental corruption only. A frame checksum
//! gives no authentication and no resistance to inputs chosen to collide. A
//! seed is not a key, and a secret seed does not resist hash flooding.
//!
//! The portable, safe Rust implementation is the sole semantic authority.

use crate::traits::FastHash;

const PRIME32_1: u32 = 0x9E37_79B1;
const PRIME32_2: u32 = 0x85EB_CA77;
const PRIME32_3: u32 = 0xC2B2_AE3D;
const PRIME32_4: u32 = 0x27D4_EB2F;
const PRIME32_5: u32 = 0x1656_67B1;

/// One stripe: a 32-bit word for each of the four lanes.
const STRIPE_LEN: usize = 16;

/// XXH32 with a 32-bit seed and a 32-bit output.
///
/// LZ4 stores this value little-endian. [`Xxh32Hasher`] computes it
/// incrementally.
#[derive(Clone, Debug, Default)]
pub struct Xxh32;

impl FastHash for Xxh32 {
  const OUTPUT_SIZE: usize = 4;
  type Output = u32;
  type Seed = u32;

  #[inline]
  fn hash_with_seed(seed: u32, data: &[u8]) -> u32 {
    let (stripes, tail) = data.as_chunks::<STRIPE_LEN>();
    let acc = if stripes.is_empty() {
      seed.wrapping_add(PRIME32_5)
    } else {
      converge(consume(initial_lanes(seed), stripes))
    };
    finalize(acc, data.len() as u64, tail)
  }
}

#[inline(always)]
const fn initial_lanes(seed: u32) -> [u32; 4] {
  [
    seed.wrapping_add(PRIME32_1).wrapping_add(PRIME32_2),
    seed.wrapping_add(PRIME32_2),
    seed,
    seed.wrapping_sub(PRIME32_1),
  ]
}

#[inline(always)]
const fn round(acc: u32, lane: u32) -> u32 {
  acc
    .wrapping_add(lane.wrapping_mul(PRIME32_2))
    .rotate_left(13)
    .wrapping_mul(PRIME32_1)
}

#[inline(always)]
fn consume(mut lanes: [u32; 4], stripes: &[[u8; STRIPE_LEN]]) -> [u32; 4] {
  for stripe in stripes {
    for (lane, word) in lanes.iter_mut().zip(stripe.as_chunks::<4>().0) {
      *lane = round(*lane, u32::from_le_bytes(*word));
    }
  }
  lanes
}

#[inline(always)]
const fn converge(lanes: [u32; 4]) -> u32 {
  lanes[0]
    .rotate_left(1)
    .wrapping_add(lanes[1].rotate_left(7))
    .wrapping_add(lanes[2].rotate_left(12))
    .wrapping_add(lanes[3].rotate_left(18))
}

/// Mix in the length and the final `tail` of fewer than 16 bytes, then avalanche.
#[inline(always)]
fn finalize(mut acc: u32, len: u64, tail: &[u8]) -> u32 {
  // The specification adds the input length modulo 2^32.
  let [len0, len1, len2, len3, ..] = len.to_le_bytes();
  acc = acc.wrapping_add(u32::from_le_bytes([len0, len1, len2, len3]));
  let (words, bytes) = tail.as_chunks::<4>();
  for word in words {
    acc = acc
      .wrapping_add(u32::from_le_bytes(*word).wrapping_mul(PRIME32_3))
      .rotate_left(17)
      .wrapping_mul(PRIME32_4);
  }
  for &byte in bytes {
    acc = acc
      .wrapping_add(u32::from(byte).wrapping_mul(PRIME32_5))
      .rotate_left(11)
      .wrapping_mul(PRIME32_1);
  }
  acc ^= acc >> 15;
  acc = acc.wrapping_mul(PRIME32_2);
  acc ^= acc >> 13;
  acc = acc.wrapping_mul(PRIME32_3);
  acc ^ (acc >> 16)
}

/// Allocation-free streaming XXH32 state.
///
/// Incremental writes produce the same result as hashing their concatenation
/// with [`Xxh32`]. The state keeps at most 15 bytes of input; storage does not
/// grow with the input. It counts the input length in 64 bits, so an input
/// longer than 4 GiB keeps the long-input path while the result mixes in the
/// length modulo 2^32, as the specification requires.
///
/// # Examples
///
/// An LZ4 decoder hashes each decoded block, then compares the content
/// checksum that the frame stores little-endian:
///
/// ```
/// use rscrypto::Xxh32Hasher;
///
/// let mut content = Xxh32Hasher::new();
/// for block in [&b"decoded "[..], b"blocks"] {
///   content.write(block);
/// }
/// assert_eq!(content.finish().to_le_bytes(), [0x19, 0xaf, 0x5a, 0xae]);
/// ```
#[derive(Clone)]
pub struct Xxh32Hasher {
  seed: u32,
  lanes: [u32; 4],
  buffer: [u8; STRIPE_LEN],
  buffered: usize,
  total_len: u64,
}

impl Xxh32Hasher {
  /// Create a hasher with seed zero, the LZ4 frame seed.
  #[inline]
  #[must_use]
  pub const fn new() -> Self {
    Self::with_seed(0)
  }

  /// Create a hasher with `seed`.
  #[inline]
  #[must_use]
  pub const fn with_seed(seed: u32) -> Self {
    Self {
      seed,
      lanes: initial_lanes(seed),
      buffer: [0; STRIPE_LEN],
      buffered: 0,
      total_len: 0,
    }
  }

  /// Append bytes to the hash stream.
  #[inline]
  pub fn write(&mut self, bytes: &[u8]) {
    self.total_len = self.total_len.strict_add(bytes.len() as u64);
    let mut input = bytes;
    if self.buffered != 0 {
      let (head, rest) = input.split_at(input.len().min(STRIPE_LEN.strict_sub(self.buffered)));
      let end = self.buffered.strict_add(head.len());
      self.buffer[self.buffered..end].copy_from_slice(head);
      self.buffered = end;
      if end < STRIPE_LEN {
        return;
      }
      self.lanes = consume(self.lanes, core::slice::from_ref(&self.buffer));
      input = rest;
    }
    let (stripes, tail) = input.as_chunks::<STRIPE_LEN>();
    self.lanes = consume(self.lanes, stripes);
    self.buffer[..tail.len()].copy_from_slice(tail);
    self.buffered = tail.len();
  }

  /// Reset the state without changing its seed.
  #[inline]
  pub fn reset(&mut self) {
    *self = Self::with_seed(self.seed);
  }

  /// Return the hash of the bytes written so far without consuming the state.
  #[inline]
  #[must_use]
  pub fn finish(&self) -> u32 {
    let acc = if self.total_len < STRIPE_LEN as u64 {
      self.seed.wrapping_add(PRIME32_5)
    } else {
      converge(self.lanes)
    };
    finalize(acc, self.total_len, &self.buffer[..self.buffered])
  }
}

impl Default for Xxh32Hasher {
  #[inline]
  fn default() -> Self {
    Self::new()
  }
}

impl core::fmt::Debug for Xxh32Hasher {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Xxh32Hasher")
      .field("seed", &self.seed)
      .field("buffered", &self.buffered)
      .field("total_len", &self.total_len)
      .finish_non_exhaustive()
  }
}
