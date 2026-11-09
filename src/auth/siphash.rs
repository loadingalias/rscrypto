//! SipHash-1-3 and SipHash-2-4 keyed hashing.
//!
//! SipHash (Aumasson and Bernstein, 2012) is a pseudorandom function with a
//! 128-bit key and a 64-bit output, built for short inputs. A hash table keyed
//! by values that an attacker chooses, such as QUIC connection IDs, needs a
//! keyed function like it: without the key, the attacker cannot predict which
//! inputs collide, so it cannot fill one bucket and turn lookups linear.
//!
//! - [`SipRandomState`] builds [`SipHash13`] hashers for `HashMap` and
//!   `HashSet`. SipHash-1-3 is the variant that the Rust standard library,
//!   Python, and Swift use for hash tables.
//! - [`SipHash24`] is the variant of the SipHash specification and its
//!   reference vectors, with more rounds, for other short-input uses.
//!
//! A 64-bit output is neither a collision-resistant hash nor a message
//! authentication tag: use HMAC or KMAC to authenticate messages.
//!
//! # Keys
//!
//! The key is secret: anyone who learns it can construct colliding inputs.
//! Draw it from a cryptographically secure source, one key per table or per
//! process. [`SipHashKey`] and [`SipRandomState`] clear their key material on
//! drop and print none of it in `Debug`. A hasher holds key-derived state for
//! one hash and does not clear it.
//!
//! The implementation has no branch or memory index that depends on the key or
//! the input bytes; its running time depends on the input length.
//!
//! # Example
//!
//! ```
//! use std::collections::HashMap;
//!
//! use rscrypto::SipRandomState;
//!
//! // Fill from a cryptographically secure source; this fixed key is for the example only.
//! let state = SipRandomState::try_new_with(|key| {
//!   key.fill(0x5a);
//!   Ok::<(), core::convert::Infallible>(())
//! })
//! .expect("the example source cannot fail");
//! let mut routes: HashMap<Vec<u8>, u32, _> = HashMap::with_hasher(state);
//! routes.insert(vec![0x8a, 0x2f, 0x11, 0x04], 7);
//! assert_eq!(routes.get([0x8a, 0x2f, 0x11, 0x04].as_slice()), Some(&7));
//! ```

use core::fmt;
use core::hash::{BuildHasher, Hasher};

/// A 128-bit SipHash key.
///
/// The bytes are `k0` then `k1`, each little-endian, as in the SipHash
/// specification. The key clears on drop.
#[derive(Clone)]
pub struct SipHashKey([u8; 16]);

impl SipHashKey {
  /// Key length in bytes.
  pub const LENGTH: usize = 16;

  /// Use 16 key bytes.
  #[must_use]
  pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
    Self(bytes)
  }

  /// Draw a key from caller-supplied entropy.
  ///
  /// The callback must fill the entire buffer before returning `Ok`.
  ///
  /// # Errors
  ///
  /// Returns the callback's error. No key is produced, and the partly filled
  /// buffer is cleared.
  pub fn try_generate_with<E>(fill: impl FnOnce(&mut [u8]) -> Result<(), E>) -> Result<Self, E> {
    let mut key = Self([0; Self::LENGTH]);
    fill(&mut key.0)?;
    Ok(key)
  }

  /// Draw a key from the operating system's entropy source.
  ///
  /// # Errors
  ///
  /// Returns the entropy source's error.
  #[cfg(feature = "getrandom")]
  pub fn try_generate() -> Result<Self, getrandom::Error> {
    Self::try_generate_with(getrandom::fill)
  }

  #[inline(always)]
  fn words(&self) -> (u64, u64) {
    let (k0, k1) = self.0.split_at(8);
    (
      u64::from_le_bytes(k0.try_into().expect("8-byte half")),
      u64::from_le_bytes(k1.try_into().expect("8-byte half")),
    )
  }
}

impl Drop for SipHashKey {
  fn drop(&mut self) {
    crate::traits::ct::zeroize(&mut self.0);
  }
}

impl fmt::Debug for SipHashKey {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("SipHashKey").finish_non_exhaustive()
  }
}

/// SipHash state with `C` compression rounds and `D` finalization rounds.
///
/// The pending bytes are the last `length % 8` input bytes, so the state keeps
/// no separate count.
#[derive(Clone)]
struct Sip<const C: usize, const D: usize> {
  v0: u64,
  v1: u64,
  v2: u64,
  v3: u64,
  /// The pending input bytes, little-endian, in the low bits; the higher bits
  /// are zero.
  tail: u64,
  /// Input length modulo 2^64; the final block encodes its low byte.
  length: u64,
}

/// `v0` through `v3` under `key`, before any input. They are equivalent to the key.
#[inline(always)]
fn keyed_words(key: &SipHashKey) -> [u64; 4] {
  let (k0, k1) = key.words();
  [
    k0 ^ 0x736f_6d65_7073_6575,
    k1 ^ 0x646f_7261_6e64_6f6d,
    k0 ^ 0x6c79_6765_6e65_7261,
    k1 ^ 0x7465_6462_7974_6573,
  ]
}

impl<const C: usize, const D: usize> Sip<C, D> {
  #[inline(always)]
  fn new(key: &SipHashKey) -> Self {
    Self::from_keyed(&keyed_words(key))
  }

  #[inline(always)]
  fn from_keyed(&[v0, v1, v2, v3]: &[u64; 4]) -> Self {
    Self {
      v0,
      v1,
      v2,
      v3,
      tail: 0,
      length: 0,
    }
  }

  #[inline(always)]
  fn round(&mut self) {
    self.v0 = self.v0.wrapping_add(self.v1);
    self.v1 = self.v1.rotate_left(13) ^ self.v0;
    self.v0 = self.v0.rotate_left(32);
    self.v2 = self.v2.wrapping_add(self.v3);
    self.v3 = self.v3.rotate_left(16) ^ self.v2;
    self.v0 = self.v0.wrapping_add(self.v3);
    self.v3 = self.v3.rotate_left(21) ^ self.v0;
    self.v2 = self.v2.wrapping_add(self.v1);
    self.v1 = self.v1.rotate_left(17) ^ self.v2;
    self.v2 = self.v2.rotate_left(32);
  }

  #[inline(always)]
  fn compress(&mut self, word: u64) {
    self.v3 ^= word;
    for _ in 0..C {
      self.round();
    }
    self.v0 ^= word;
  }

  /// The bits of `tail` in use: 8 for each pending byte.
  #[inline(always)]
  fn used_bits(&self) -> u32 {
    ((self.length & 7) as u32).strict_mul(8)
  }

  /// Merge `bits` bits of input, `bits` a multiple of 8 through 64, after the
  /// `used` pending bits. `x` holds the input little-endian in its low bits,
  /// with the higher bits zero. Compress the word if it fills.
  #[inline(always)]
  fn merge(&mut self, x: u64, used: u32, bits: u32) {
    self.tail |= x << used;
    if used.strict_add(bits) >= 64 {
      self.compress(self.tail);
      // The bits of `x` that did not fit; none when `used` is 0.
      self.tail = x.unbounded_shr(64u32.strict_sub(used));
    }
  }

  /// Append one integer of `BYTES` bytes in the byte order of `to_ne_bytes`,
  /// which `Hasher`'s default integer methods write.
  #[inline(always)]
  fn write_integer<const BYTES: u32>(&mut self, little_endian: u64) {
    let used = self.used_bits();
    // SipHash encodes the input length modulo 256; the counter wraps by specification.
    self.length = self.length.wrapping_add(u64::from(BYTES));
    self.merge(little_endian, used, BYTES.strict_mul(8));
  }

  #[inline]
  fn write(&mut self, mut bytes: &[u8]) {
    let used = self.used_bits();
    // SipHash encodes the input length modulo 256; the counter wraps by specification.
    self.length = self.length.wrapping_add(bytes.len() as u64);
    if used != 0 {
      // Complete the pending word first. After a whole integer or a length
      // prefix, `used` is 0 and the compiler drops this step.
      let open = 8usize.strict_sub((used / 8) as usize);
      let (first, rest) = bytes.split_at(open.min(bytes.len()));
      self.merge(load_partial(first), used, tail_bits(first.len()));
      if rest.is_empty() {
        return;
      }
      bytes = rest;
    }
    let (words, rest) = bytes.as_chunks::<8>();
    for word in words {
      self.compress(u64::from_le_bytes(*word));
    }
    self.tail = load_tail(bytes, tail_bits(rest.len()));
  }

  #[inline]
  fn finish(&self) -> u64 {
    self.clone().finalize((self.length << 56) | self.tail)
  }

  /// Hash one byte string without stream buffering.
  #[inline]
  fn hash(mut self, data: &[u8]) -> u64 {
    let (words, rest) = data.as_chunks::<8>();
    for word in words {
      self.compress(u64::from_le_bytes(*word));
    }
    // The final block encodes the length modulo 256.
    self.finalize(((data.len() as u64) << 56) | load_tail(data, tail_bits(rest.len())))
  }

  #[inline(always)]
  fn finalize(mut self, last: u64) -> u64 {
    self.compress(last);
    self.v2 ^= 0xff;
    for _ in 0..D {
      self.round();
    }
    self.v0 ^ self.v1 ^ self.v2 ^ self.v3
  }
}

/// The bit length of a tail of `rest` bytes, `rest` below 8.
#[inline(always)]
#[expect(clippy::cast_possible_truncation, reason = "a tail holds fewer than 8 bytes")]
const fn tail_bits(rest: usize) -> u32 {
  debug_assert!(rest < 8);
  ((rest & 7) as u32).strict_mul(8)
}

/// Load the last `rest_bits / 8` bytes of `bytes` as a little-endian word.
///
/// `rest_bits` is the bit length of the bytes after the last whole 8-byte
/// word. A string of 8 bytes or more takes one 8-byte load that ends with it
/// and drops the bytes before its tail.
#[inline(always)]
fn load_tail(bytes: &[u8], rest_bits: u32) -> u64 {
  match bytes.last_chunk::<8>() {
    // `rest_bits` is below 64, so the shift is 8 to 64 bits.
    Some(last) => u64::from_le_bytes(*last).unbounded_shr(64u32.strict_sub(rest_bits)),
    None => load_partial(bytes),
  }
}

/// Load up to seven bytes as a little-endian word.
///
/// Fixed-width reads chosen by the public length avoid a variable-length copy.
#[inline(always)]
fn load_partial(bytes: &[u8]) -> u64 {
  debug_assert!(bytes.len() < 8);
  let mut word = 0u64;
  let mut at = 0usize;
  if let Some(four) = bytes.first_chunk::<4>() {
    word = u64::from(u32::from_le_bytes(*four));
    at = 4;
  }
  if let Some(two) = bytes[at..].first_chunk::<2>() {
    word |= u64::from(u16::from_le_bytes(*two)) << at.strict_mul(8);
    at = at.strict_add(2);
  }
  if let Some(&one) = bytes.get(at) {
    word |= u64::from(one) << at.strict_mul(8);
  }
  word
}

macro_rules! sip_hasher {
  ($name:ident, $c:literal, $d:literal, $doc:literal) => {
    #[doc = $doc]
    ///
    /// As a [`Hasher`], it hashes the bytes that `Hash` implementations write.
    /// [`Self::hash`] hashes one byte string.
    #[derive(Clone)]
    pub struct $name(Sip<$c, $d>);

    impl $name {
      /// Start a hash under `key`.
      #[must_use]
      #[inline]
      pub fn new(key: &SipHashKey) -> Self {
        Self(Sip::new(key))
      }

      /// Hash `data` under `key`.
      #[must_use]
      #[inline]
      pub fn hash(key: &SipHashKey, data: &[u8]) -> u64 {
        Sip::<$c, $d>::new(key).hash(data)
      }
    }

    // The integer methods hash the bytes of `to_ne_bytes`, as the default
    // methods do, without the byte-string path. Signed integers reach them
    // through the default `write_i*` methods.
    impl Hasher for $name {
      #[inline]
      fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
      }

      #[inline]
      fn write_u8(&mut self, i: u8) {
        self.0.write_integer::<1>(u64::from(i));
      }

      #[inline]
      fn write_u16(&mut self, i: u16) {
        self.0.write_integer::<2>(u64::from(i.to_le()));
      }

      #[inline]
      fn write_u32(&mut self, i: u32) {
        self.0.write_integer::<4>(u64::from(i.to_le()));
      }

      #[inline]
      fn write_u64(&mut self, i: u64) {
        self.0.write_integer::<8>(i.to_le());
      }

      #[inline]
      #[expect(clippy::cast_possible_truncation, reason = "each cast takes one 64-bit half")]
      fn write_u128(&mut self, i: u128) {
        let i = i.to_le();
        self.0.write_integer::<8>(i as u64);
        self.0.write_integer::<8>((i >> 64) as u64);
      }

      #[inline]
      fn write_usize(&mut self, i: usize) {
        self.0.write_integer::<{ usize::BITS / 8 }>(i.to_le() as u64);
      }

      #[inline]
      fn finish(&self) -> u64 {
        self.0.finish()
      }
    }

    impl fmt::Debug for $name {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(stringify!($name)).finish_non_exhaustive()
      }
    }
  };
}

sip_hasher!(
  SipHash13,
  1,
  3,
  "SipHash-1-3: one compression round and three finalization rounds, for hash tables."
);
sip_hasher!(
  SipHash24,
  2,
  4,
  "SipHash-2-4: two compression rounds and four finalization rounds, as specified."
);

/// A [`BuildHasher`] that keys every [`SipHash13`] hasher with one secret key.
///
/// Use one state per table, or one per process, from a cryptographically secure
/// source. The state keeps SipHash's keyed starting words, which are equivalent
/// to the key, so a hasher starts without rederiving them. Cloning the state
/// copies the words; every copy clears them on drop.
#[derive(Clone)]
pub struct SipRandomState {
  keyed: [u64; 4],
}

impl SipRandomState {
  /// Build hashers under `key`.
  #[must_use]
  pub fn new(key: SipHashKey) -> Self {
    Self {
      keyed: keyed_words(&key),
    }
  }

  /// Draw the key from caller-supplied entropy.
  ///
  /// The callback must fill the entire buffer before returning `Ok`.
  ///
  /// # Errors
  ///
  /// Returns the callback's error.
  pub fn try_new_with<E>(fill: impl FnOnce(&mut [u8]) -> Result<(), E>) -> Result<Self, E> {
    SipHashKey::try_generate_with(fill).map(Self::new)
  }

  /// Draw the key from the operating system's entropy source.
  ///
  /// # Errors
  ///
  /// Returns the entropy source's error.
  #[cfg(feature = "getrandom")]
  pub fn try_new() -> Result<Self, getrandom::Error> {
    SipHashKey::try_generate().map(Self::new)
  }
}

impl BuildHasher for SipRandomState {
  type Hasher = SipHash13;

  #[inline]
  fn build_hasher(&self) -> SipHash13 {
    SipHash13(Sip::from_keyed(&self.keyed))
  }
}

impl Drop for SipRandomState {
  fn drop(&mut self) {
    crate::traits::ct::zeroize_words(&mut self.keyed);
  }
}

impl fmt::Debug for SipRandomState {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("SipRandomState").finish_non_exhaustive()
  }
}
