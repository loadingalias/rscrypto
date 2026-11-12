//! XXH64, the original 64-bit xxHash (**NOT CRYPTO**).
//!
//! This is XXH64 from the
//! [xxHash specification](https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md),
//! not XXH3. Zstandard frames store the low 32 bits of the seed-zero XXH64 of
//! the content, little-endian. It detects accidental corruption only. A frame
//! checksum gives no authentication and no resistance to inputs chosen to
//! collide. A seed is not a key, and a secret seed does not resist hash
//! flooding.
//!
//! The portable, safe Rust implementation is the sole semantic authority.

use core::hash::Hasher;

use crate::traits::FastHash;

const PRIME64_1: u64 = 0x9E37_79B1_85EB_CA87;
const PRIME64_2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const PRIME64_3: u64 = 0x1656_67B1_9E37_79F9;
const PRIME64_4: u64 = 0x85EB_CA77_C2B2_AE63;
const PRIME64_5: u64 = 0x27D4_EB2F_1656_67C5;

/// One stripe: a 64-bit word for each of the four lanes.
const STRIPE_LEN: usize = 32;

/// XXH64 with a 64-bit seed and a 64-bit output.
///
/// [`Xxh64Hasher`] computes it incrementally.
#[derive(Clone, Debug, Default)]
pub struct Xxh64;

impl FastHash for Xxh64 {
  const OUTPUT_SIZE: usize = 8;
  type Output = u64;
  type Seed = u64;

  #[inline]
  fn hash_with_seed(seed: u64, data: &[u8]) -> u64 {
    let (stripes, tail) = data.as_chunks::<STRIPE_LEN>();
    let acc = if stripes.is_empty() {
      seed.wrapping_add(PRIME64_5)
    } else {
      converge(consume(initial_lanes(seed), stripes))
    };
    finalize(acc, data.len() as u64, tail)
  }
}

#[inline(always)]
const fn initial_lanes(seed: u64) -> [u64; 4] {
  [
    seed.wrapping_add(PRIME64_1).wrapping_add(PRIME64_2),
    seed.wrapping_add(PRIME64_2),
    seed,
    seed.wrapping_sub(PRIME64_1),
  ]
}

#[inline(always)]
const fn round(acc: u64, lane: u64) -> u64 {
  acc
    .wrapping_add(lane.wrapping_mul(PRIME64_2))
    .rotate_left(31)
    .wrapping_mul(PRIME64_1)
}

#[inline(always)]
fn consume(mut lanes: [u64; 4], stripes: &[[u8; STRIPE_LEN]]) -> [u64; 4] {
  for stripe in stripes {
    for (lane, word) in lanes.iter_mut().zip(stripe.as_chunks::<8>().0) {
      *lane = round(*lane, u64::from_le_bytes(*word));
    }
  }
  lanes
}

#[inline(always)]
fn converge(lanes: [u64; 4]) -> u64 {
  let acc = lanes[0]
    .rotate_left(1)
    .wrapping_add(lanes[1].rotate_left(7))
    .wrapping_add(lanes[2].rotate_left(12))
    .wrapping_add(lanes[3].rotate_left(18));
  lanes.into_iter().fold(acc, |acc, lane| {
    (acc ^ round(0, lane)).wrapping_mul(PRIME64_1).wrapping_add(PRIME64_4)
  })
}

/// Mix in the length and the final `tail` of fewer than 32 bytes, then avalanche.
#[inline(always)]
fn finalize(mut acc: u64, len: u64, tail: &[u8]) -> u64 {
  acc = acc.wrapping_add(len);
  let (words, rest) = tail.as_chunks::<8>();
  for word in words {
    acc = (acc ^ round(0, u64::from_le_bytes(*word)))
      .rotate_left(27)
      .wrapping_mul(PRIME64_1)
      .wrapping_add(PRIME64_4);
  }
  let (half_words, bytes) = rest.as_chunks::<4>();
  for half_word in half_words {
    acc = (acc ^ u64::from(u32::from_le_bytes(*half_word)).wrapping_mul(PRIME64_1))
      .rotate_left(23)
      .wrapping_mul(PRIME64_2)
      .wrapping_add(PRIME64_3);
  }
  for &byte in bytes {
    acc = (acc ^ u64::from(byte).wrapping_mul(PRIME64_5))
      .rotate_left(11)
      .wrapping_mul(PRIME64_1);
  }
  acc ^= acc >> 33;
  acc = acc.wrapping_mul(PRIME64_2);
  acc ^= acc >> 29;
  acc = acc.wrapping_mul(PRIME64_3);
  acc ^ (acc >> 32)
}

/// Allocation-free streaming XXH64 state.
///
/// Incremental writes produce the same result as hashing their concatenation
/// with [`Xxh64`]. The state keeps at most 31 bytes of input; storage does not
/// grow with the input.
///
/// # Examples
///
/// A Zstandard decoder hashes each decoded block, then compares the low 32
/// bits that the frame stores little-endian:
///
/// ```
/// use core::hash::Hasher;
///
/// use rscrypto::Xxh64Hasher;
///
/// let mut content = Xxh64Hasher::new();
/// for block in [&b"decoded "[..], b"blocks"] {
///   content.write(block);
/// }
/// assert_eq!(content.finish().to_le_bytes()[..4], [0x56, 0x9f, 0xf5, 0xec]);
/// ```
#[derive(Clone)]
pub struct Xxh64Hasher {
  seed: u64,
  lanes: [u64; 4],
  buffer: [u8; STRIPE_LEN],
  buffered: usize,
  total_len: u64,
}

impl Xxh64Hasher {
  /// Create a hasher with seed zero, the Zstandard frame seed.
  #[inline]
  #[must_use]
  pub const fn new() -> Self {
    Self::with_seed(0)
  }

  /// Create a hasher with `seed`.
  #[inline]
  #[must_use]
  pub const fn with_seed(seed: u64) -> Self {
    Self {
      seed,
      lanes: initial_lanes(seed),
      buffer: [0; STRIPE_LEN],
      buffered: 0,
      total_len: 0,
    }
  }

  /// Reset the state without changing its seed.
  #[inline]
  pub fn reset(&mut self) {
    *self = Self::with_seed(self.seed);
  }
}

impl Default for Xxh64Hasher {
  #[inline]
  fn default() -> Self {
    Self::new()
  }
}

impl core::fmt::Debug for Xxh64Hasher {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Xxh64Hasher")
      .field("seed", &self.seed)
      .field("buffered", &self.buffered)
      .field("total_len", &self.total_len)
      .finish_non_exhaustive()
  }
}

impl Hasher for Xxh64Hasher {
  #[inline]
  fn write(&mut self, bytes: &[u8]) {
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

  /// Return the hash of the bytes written so far without consuming the state.
  #[inline]
  fn finish(&self) -> u64 {
    let acc = if self.total_len < STRIPE_LEN as u64 {
      self.seed.wrapping_add(PRIME64_5)
    } else {
      converge(self.lanes)
    };
    finalize(acc, self.total_len, &self.buffer[..self.buffered])
  }
}
