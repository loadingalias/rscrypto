//! Connection IDs and initial destination IDs from caller-seeded generators.
//!
//! Quinn's own generators read the thread-local, OS-seeded `rand::rng()` for
//! every ID. These replacements draw their bytes from [`ChaCha20Rng`]s seeded
//! by the caller's entropy source when the configuration is built, so the
//! endpoint reads no other entropy and runs deterministically from its sources.
//! Quinn's generator trait is infallible, so the seeded generators supply the
//! nonces of [`ConnectionIdKey`] IDs instead of a per-ID entropy draw.

use alloc::sync::Arc;
use core::time::Duration;
use std::sync::Mutex;

use quinn_proto::{ConnectionId, ConnectionIdGenerator, InvalidCid};
use rscrypto::ChaCha20Rng;
use rustls::crypto::{GetRandomFailed, SecureRandom};

use crate::quic::{CID_NONCE_LEN, ConnectionIdKey};

// Quinn's default initial destination ID length, the RFC 9000 maximum.
const INITIAL_ID_LEN: usize = 20;

fn seeded(random: &dyn SecureRandom) -> Result<ChaCha20Rng, GetRandomFailed> {
  ChaCha20Rng::try_generate_with(|seed| random.fill(seed))
}

/// Shared connection-ID material for the endpoints built from one key set.
pub(super) struct CidKeys {
  key: ConnectionIdKey,
  // Each endpoint's generator draws its seed here, so endpoints never share a stream.
  seeds: Mutex<ChaCha20Rng>,
}

impl CidKeys {
  /// Draw a 16-byte tag key, then a 32-byte seed for the endpoints' generators.
  pub(super) fn generate(random: &dyn SecureRandom) -> Result<Self, GetRandomFailed> {
    Ok(Self {
      key: ConnectionIdKey::generate(random)?,
      seeds: Mutex::new(seeded(random)?),
    })
  }

  /// A generator for one new endpoint, with its own nonce stream.
  pub(super) fn generator(self: &Arc<Self>) -> CidGenerator {
    // `fill_bytes` cannot panic, so a poisoned lock still holds a consistent generator.
    let mut seeds = self.seeds.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let rng = ChaCha20Rng::try_generate_with(|seed| {
      seeds.fill_bytes(seed);
      Ok::<(), core::convert::Infallible>(())
    })
    .unwrap_or_else(|never| match never {});
    CidGenerator {
      keys: Arc::clone(self),
      rng,
    }
  }
}

/// Eight-byte [`ConnectionIdKey`] IDs: a random nonce and its keyed tag.
///
/// The tag lets [`ConnectionIdGenerator::validate`] reject packets for IDs this
/// key set never issued before any connection lookup. Unlike Quinn's FxHash
/// filter, it is a truncated AES-128 PRF, so an observer cannot compute tags.
/// IDs carry no counter or endpoint index, so they cannot be linked.
pub(super) struct CidGenerator {
  keys: Arc<CidKeys>,
  rng: ChaCha20Rng,
}

impl ConnectionIdGenerator for CidGenerator {
  fn generate_cid(&mut self) -> ConnectionId {
    let mut nonce = [0u8; CID_NONCE_LEN];
    self.rng.fill_bytes(&mut nonce);
    ConnectionId::new(&self.keys.key.id(nonce))
  }

  fn validate(&self, cid: ConnectionId) -> Result<(), InvalidCid> {
    if self.keys.key.validate(&cid) {
      Ok(())
    } else {
      Err(InvalidCid)
    }
  }

  fn cid_len(&self) -> usize {
    ConnectionIdKey::ID_LEN
  }

  fn cid_lifetime(&self) -> Option<Duration> {
    None
  }
}

/// Initial destination connection IDs for one client configuration.
///
/// RFC 9000, section 7.2, requires an unpredictable value of at least eight
/// bytes. Each ID is the next 20 bytes of a generator seeded from the client's
/// entropy source, matching Quinn's default length.
pub(super) struct InitialCids(Mutex<ChaCha20Rng>);

impl InitialCids {
  pub(super) fn generate(random: &dyn SecureRandom) -> Result<Self, GetRandomFailed> {
    seeded(random).map(|rng| Self(Mutex::new(rng)))
  }

  pub(super) fn next(&self) -> ConnectionId {
    let mut id = [0u8; INITIAL_ID_LEN];
    self
      .0
      .lock()
      .unwrap_or_else(std::sync::PoisonError::into_inner)
      .fill_bytes(&mut id);
    ConnectionId::new(&id)
  }
}
