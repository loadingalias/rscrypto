//! Deterministic random generation from a caller-supplied seed.
//!
//! [`ChaCha20Rng`] expands one 32-byte seed into an unbounded byte stream. It
//! never reads operating-system entropy by itself, so its output is exactly as
//! unpredictable as its seed. Use it where a protocol or simulation needs
//! reproducible randomness from a seed it controls.
//!
//! ```rust
//! use rscrypto::ChaCha20Rng;
//!
//! // Draw the seed from a cryptographically secure source in real use.
//! let mut rng = ChaCha20Rng::from_seed([0x5a; ChaCha20Rng::SEED_SIZE]);
//! let mut nonce = [0u8; 16];
//! rng.fill_bytes(&mut nonce);
//! let skip = rng.next_u32();
//!
//! // The stream is a function of the seed alone.
//! let mut replay = ChaCha20Rng::from_seed([0x5a; ChaCha20Rng::SEED_SIZE]);
//! let mut again = [0u8; 16];
//! replay.fill_bytes(&mut again);
//! assert_eq!(nonce, again);
//! assert_eq!(skip, replay.next_u32());
//! ```

use core::fmt;

use crate::{backend::chacha20, traits::ct};

const SEED_SIZE: usize = chacha20::KEY_SIZE;
/// Keystream bytes per refill: eight ChaCha20 blocks.
const CHUNK_SIZE: usize = 512;
/// Output bytes per refill. The final 32 keystream bytes become the next key.
const OUTPUT_SIZE: usize = CHUNK_SIZE - SEED_SIZE;

/// ChaCha20 deterministic random generator with fast key erasure.
///
/// # Output stream
///
/// The generator holds a ChaCha20 key; the seed is the first key. Each refill
/// computes 512 bytes of ChaCha20 keystream under the current key: the RFC 8439
/// block function with an all-zero nonce and block counters 0 to 7. The first
/// 480 bytes are the next output bytes. The last 32 bytes replace the key, and
/// the old key is erased. A refill happens only when a request needs more
/// output, so the stream has no length limit.
///
/// [`fill_bytes`](Self::fill_bytes), [`next_u32`](Self::next_u32), and
/// [`next_u64`](Self::next_u64) consume one byte stream in call order. The two
/// integer methods read 4 and 8 bytes as little-endian. Output does not depend
/// on how requests are split: two 8-byte requests return the same bytes as one
/// 16-byte request. The stream for each seed is a stable part of this API.
///
/// The first 480 output bytes equal the plain ChaCha20 keystream with the
/// seed as key and an all-zero nonce. Later bytes come from derived keys, so the
/// stream differs from generators that only advance a block counter, such as
/// `rand_chacha::ChaCha20Rng`.
///
/// # Security
///
/// - Seed it with 32 secret, uniformly random bytes. Generators with the same
///   seed return the same stream, so a seed must have one owner. To create an
///   independent generator, seed it from this generator's output.
/// - The generator clears each output byte from its state as it returns it, and
///   each refill erases the previous key. A copy of the state taken later does
///   not reveal output that was already returned.
/// - It never reseeds itself. Anyone who learns its state can compute all of its
///   later output. To recover from a compromise, replace it with a generator
///   built from a fresh seed. Dropping a generator clears its state.
/// - It does not implement `Clone`, because a clone would repeat the stream.
/// - Operation timing depends on the requested lengths and the call sequence,
///   not on the seed or the generator state.
/// - Rust moves copy the state. Dropping clears the final location only; earlier
///   copies made by moves are not cleared.
///
/// # `rand_core`
///
/// With the `rand_core` feature, the generator implements `rand_core` 0.10's
/// `TryRng` (with `Error = Infallible`, so `Rng` too), `TryCryptoRng` (so
/// `CryptoRng` too), and `SeedableRng` with a 32-byte seed. The trait methods
/// return the same stream as the inherent methods. `SeedableRng::from_rng` and
/// `try_from_rng` write the seed directly into the new generator, so `fork`
/// creates an independent generator from this one's output.
/// `SeedableRng::seed_from_u64` panics: a 64-bit value is not a 256-bit secret.
pub struct ChaCha20Rng {
  key: [u8; SEED_SIZE],
  /// `chunk[position..OUTPUT_SIZE]` is unread output. Every other byte is zero
  /// between calls, so a refill can XOR the keystream into the zeroed chunk.
  chunk: [u8; CHUNK_SIZE],
  position: usize,
}

impl ChaCha20Rng {
  /// Seed length in bytes.
  pub const SEED_SIZE: usize = SEED_SIZE;

  /// Create a generator from a 32-byte seed.
  ///
  /// The seed's copy in this call is cleared. The caller still owns, and should
  /// clear, its own copy.
  #[inline]
  #[must_use]
  pub fn from_seed(mut seed: [u8; SEED_SIZE]) -> Self {
    let rng = Self {
      key: seed,
      chunk: [0; CHUNK_SIZE],
      position: OUTPUT_SIZE,
    };
    ct::zeroize(&mut seed);
    rng
  }

  /// Try to create a generator from a seed written by `fill`.
  ///
  /// `fill` is called once with a 32-byte buffer and must fill all of it or
  /// return an error. The seed is written directly into the generator. On an
  /// error, any partial seed is cleared and the error is returned.
  ///
  /// # Errors
  ///
  /// Returns the error from `fill`.
  #[inline]
  pub fn try_generate_with<E>(mut fill: impl FnMut(&mut [u8]) -> Result<(), E>) -> Result<Self, E> {
    let mut rng = Self::from_seed([0; SEED_SIZE]);
    // On an error, dropping `rng` clears the partial seed.
    fill(&mut rng.key)?;
    Ok(rng)
  }

  /// Try to create a generator seeded from the platform entropy source.
  ///
  /// # Errors
  ///
  /// Returns a getrandom error if the platform entropy source is unavailable.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[inline]
  pub fn try_generate() -> Result<Self, getrandom::Error> {
    Self::try_generate_with(getrandom::fill)
  }

  /// Fill `output` with the next bytes of the stream.
  #[inline]
  pub fn fill_bytes(&mut self, output: &mut [u8]) {
    let mut output = output;
    loop {
      let unread = &mut self.chunk[self.position..OUTPUT_SIZE];
      let count = unread.len().min(output.len());
      let (now, rest) = output.split_at_mut(count);
      let (taken, _) = unread.split_at_mut(count);
      now.copy_from_slice(taken);
      ct::zeroize(taken);
      self.position = self.position.strict_add(count);
      output = rest;
      if output.is_empty() {
        return;
      }
      self.refill();
    }
  }

  /// Return the next 4 bytes of the stream as a little-endian `u32`.
  #[inline]
  #[must_use]
  pub fn next_u32(&mut self) -> u32 {
    let mut bytes = [0; 4];
    self.fill_bytes(&mut bytes);
    u32::from_le_bytes(bytes)
  }

  /// Return the next 8 bytes of the stream as a little-endian `u64`.
  #[inline]
  #[must_use]
  pub fn next_u64(&mut self) -> u64 {
    let mut bytes = [0; 8];
    self.fill_bytes(&mut bytes);
    u64::from_le_bytes(bytes)
  }

  #[inline(never)]
  fn refill(&mut self) {
    debug_assert_eq!(self.position, OUTPUT_SIZE);
    // Every output byte was cleared as it was read, so the chunk is zero and
    // XOR leaves the keystream itself.
    chacha20::xor_keystream_chunk(&self.key, &mut self.chunk);
    let (_, next_key) = self.chunk.split_at_mut(OUTPUT_SIZE);
    self.key.copy_from_slice(next_key);
    ct::zeroize(next_key);
    self.position = 0;
  }
}

#[cfg(feature = "rand_core")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand_core")))]
impl rand_core::TryRng for ChaCha20Rng {
  type Error = core::convert::Infallible;

  #[inline]
  fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
    Ok(Self::next_u32(self))
  }

  #[inline]
  fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
    Ok(Self::next_u64(self))
  }

  #[inline]
  fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
    Self::fill_bytes(self, dst);
    Ok(())
  }
}

#[cfg(feature = "rand_core")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand_core")))]
impl rand_core::TryCryptoRng for ChaCha20Rng {}

#[cfg(feature = "rand_core")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand_core")))]
impl rand_core::SeedableRng for ChaCha20Rng {
  type Seed = [u8; SEED_SIZE];

  /// Create a generator from a 32-byte seed; see [`ChaCha20Rng::from_seed`].
  #[inline]
  fn from_seed(seed: Self::Seed) -> Self {
    Self::from_seed(seed)
  }

  /// Refuse a 64-bit seed.
  ///
  /// # Panics
  ///
  /// Always. A `u64` holds at most 64 bits of entropy, so a generator seeded
  /// from it would be predictable. Use [`ChaCha20Rng::from_seed`] with 32
  /// secret, uniformly random bytes, or `from_rng` with a secure generator.
  #[track_caller]
  #[expect(
    clippy::panic,
    reason = "refusing a 64-bit seed is the documented contract; only an explicit seed_from_u64 call reaches it"
  )]
  fn seed_from_u64(_state: u64) -> Self {
    panic!("ChaCha20Rng needs a 32-byte secret seed; seed_from_u64 would give it at most 64 bits of entropy")
  }

  /// Create a generator seeded with the next 32 bytes of `rng`.
  ///
  /// The seed is written directly into the new generator.
  #[inline]
  fn from_rng<R: rand_core::Rng + ?Sized>(rng: &mut R) -> Self {
    let mut new = Self::from_seed([0; SEED_SIZE]);
    rng.fill_bytes(&mut new.key);
    new
  }

  /// Try to create a generator seeded with the next 32 bytes of `rng`.
  ///
  /// The seed is written directly into the new generator. On an error, any
  /// partial seed is cleared and the error is returned.
  #[inline]
  fn try_from_rng<R: rand_core::TryRng + ?Sized>(rng: &mut R) -> Result<Self, R::Error> {
    Self::try_generate_with(|seed| rng.try_fill_bytes(seed))
  }
}

impl Drop for ChaCha20Rng {
  #[inline]
  fn drop(&mut self) {
    ct::zeroize(&mut self.key);
    ct::zeroize(&mut self.chunk);
  }
}

impl fmt::Debug for ChaCha20Rng {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("ChaCha20Rng").finish_non_exhaustive()
  }
}

#[cfg(test)]
mod tests {
  use alloc::vec;

  use super::{CHUNK_SIZE, ChaCha20Rng, OUTPUT_SIZE, SEED_SIZE};

  fn assert_state_invariant(rng: &ChaCha20Rng) {
    assert!(rng.position <= OUTPUT_SIZE);
    assert!(
      rng.chunk[..rng.position].iter().all(|&byte| byte == 0),
      "read output is cleared"
    );
    assert!(
      rng.chunk[OUTPUT_SIZE..].iter().all(|&byte| byte == 0),
      "the next key leaves the chunk"
    );
  }

  #[test]
  fn reads_clear_output_and_refills_erase_the_key() {
    let seed = [0x42; SEED_SIZE];
    let mut rng = ChaCha20Rng::from_seed(seed);
    assert_eq!(rng.position, OUTPUT_SIZE);
    assert_state_invariant(&rng);
    let mut first = [0; 3];
    rng.fill_bytes(&mut first);
    assert_ne!(rng.key, seed, "the first refill replaces the seed");
    assert_state_invariant(&rng);
    for length in [0, 1, 7, OUTPUT_SIZE - 3, OUTPUT_SIZE, CHUNK_SIZE, 3 * CHUNK_SIZE + 5] {
      let key = rng.key;
      let refills = (rng.position + length).saturating_sub(1) / OUTPUT_SIZE;
      let mut output = vec![0; length];
      rng.fill_bytes(&mut output);
      assert_state_invariant(&rng);
      assert_eq!(rng.key == key, refills == 0 || length == 0);
    }
  }

  #[test]
  fn generator_failure_returns_the_error() {
    let mut calls = 0;
    let result = ChaCha20Rng::try_generate_with(|seed| {
      calls += 1;
      assert_eq!(seed.len(), SEED_SIZE);
      seed[..17].fill(0x42);
      Err("unavailable")
    });
    assert_eq!(result.expect_err("fill failure"), "unavailable");
    assert_eq!(calls, 1);
  }
}
