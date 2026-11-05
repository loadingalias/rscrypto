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
//! process. [`SipHashKey`] and [`SipRandomState`] clear their key bytes on drop
//! and print no key material in `Debug`. A hasher holds key-derived state for
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
#[derive(Clone)]
struct Sip<const C: usize, const D: usize> {
  v0: u64,
  v1: u64,
  v2: u64,
  v3: u64,
  /// Pending input bytes, little-endian, in the low `pending` bytes.
  tail: u64,
  pending: usize,
  /// Input length modulo 2^64; the final block encodes its low byte.
  length: u64,
}

impl<const C: usize, const D: usize> Sip<C, D> {
  #[inline(always)]
  fn new(key: &SipHashKey) -> Self {
    let (k0, k1) = key.words();
    Self {
      v0: k0 ^ 0x736f_6d65_7073_6575,
      v1: k1 ^ 0x646f_7261_6e64_6f6d,
      v2: k0 ^ 0x6c79_6765_6e65_7261,
      v3: k1 ^ 0x7465_6462_7974_6573,
      tail: 0,
      pending: 0,
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

  #[inline]
  fn write(&mut self, mut bytes: &[u8]) {
    // SipHash encodes the input length modulo 256; the counter wraps by specification.
    self.length = self.length.wrapping_add(bytes.len() as u64);
    if self.pending != 0 {
      // `pending` stays below 8 between calls.
      let take = 8usize.strict_sub(self.pending).min(bytes.len());
      self.tail |= load_partial(&bytes[..take]) << self.pending.strict_mul(8);
      self.pending = self.pending.strict_add(take);
      bytes = &bytes[take..];
      if self.pending < 8 {
        return;
      }
      self.compress(self.tail);
      self.tail = 0;
      self.pending = 0;
    }
    let (words, rest) = bytes.as_chunks::<8>();
    for word in words {
      self.compress(u64::from_le_bytes(*word));
    }
    self.tail = load_partial(rest);
    self.pending = rest.len();
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
    self.finalize(((data.len() as u64) << 56) | load_partial(rest))
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

    impl Hasher for $name {
      #[inline]
      fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
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
/// source. Cloning the state copies the key; every copy clears its key on drop.
#[derive(Clone)]
pub struct SipRandomState {
  key: SipHashKey,
}

impl SipRandomState {
  /// Build hashers under `key`.
  #[must_use]
  pub const fn new(key: SipHashKey) -> Self {
    Self { key }
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
    SipHash13::new(&self.key)
  }
}

impl fmt::Debug for SipRandomState {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("SipRandomState").finish_non_exhaustive()
  }
}
