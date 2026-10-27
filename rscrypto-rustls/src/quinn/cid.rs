//! Connection IDs and initial destination IDs from caller-seeded generators.
//!
//! Quinn's own generators read the thread-local, OS-seeded `rand::rng()` for
//! every ID. These replacements draw their bytes from [`ChaCha20Rng`]s seeded
//! by the caller's entropy source when the configuration is built, so the
//! endpoint reads no other entropy and runs deterministically from its sources.

use alloc::sync::Arc;
use core::time::Duration;
use std::sync::Mutex;

use quinn_proto::{ConnectionId, ConnectionIdGenerator, InvalidCid};
use rscrypto::ChaCha20Rng;
use rscrypto::SecretBytes;
use rscrypto::aead::expert::header_protection::{Aes128HeaderProtection, Aes128HeaderProtectionKey};
use rustls::crypto::{GetRandomFailed, SecureRandom};

// Quinn's eight-byte layout: three random bytes, then a five-byte keyed tag.
const NONCE_LEN: usize = 3;
const TAG_LEN: usize = 5;
// Quinn's default initial destination ID length, the RFC 9000 maximum.
const INITIAL_ID_LEN: usize = 20;
const CID_LEN: usize = NONCE_LEN + TAG_LEN;
// Fills the AES block after the nonce; the key is used for nothing else.
const TAG_DOMAIN: [u8; 16 - NONCE_LEN] = *b"rscrypto-cid1";

fn seeded(random: &dyn SecureRandom) -> Result<ChaCha20Rng, GetRandomFailed> {
  ChaCha20Rng::try_generate_with(|seed| random.fill(seed))
}

/// Shared connection-ID material for the endpoints built from one key set.
pub(super) struct CidKeys {
  tag: Aes128HeaderProtection,
  // Each endpoint's generator draws its seed here, so endpoints never share a stream.
  seeds: Mutex<ChaCha20Rng>,
}

impl CidKeys {
  /// Draw a 16-byte tag key, then a 32-byte seed for the endpoints' generators.
  pub(super) fn generate(random: &dyn SecureRandom) -> Result<Self, GetRandomFailed> {
    let key = SecretBytes::<16>::try_fill_with(|out| random.fill(out))?;
    let tag = Aes128HeaderProtection::new(&Aes128HeaderProtectionKey::from_bytes(*key.as_bytes()));
    Ok(Self {
      tag,
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

  /// AES-128 of the domain-separated nonce, truncated to the five-byte tag.
  fn tag(&self, nonce: &[u8; NONCE_LEN]) -> [u8; TAG_LEN] {
    let mut sample = [0u8; 16];
    sample[..NONCE_LEN].copy_from_slice(nonce);
    sample[NONCE_LEN..].copy_from_slice(&TAG_DOMAIN);
    self.tag.mask(&sample)
  }
}

/// Eight-byte connection IDs: a random nonce and its keyed tag.
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
    let mut cid = [0u8; CID_LEN];
    let (nonce, tag) = cid.split_at_mut(NONCE_LEN);
    self.rng.fill_bytes(nonce);
    let nonce = <&[u8; NONCE_LEN]>::try_from(&*nonce).expect("fixed nonce length");
    tag.copy_from_slice(&self.keys.tag(nonce));
    ConnectionId::new(&cid)
  }

  fn validate(&self, cid: ConnectionId) -> Result<(), InvalidCid> {
    let Ok(cid) = <&[u8; CID_LEN]>::try_from(&cid[..]) else {
      return Err(InvalidCid);
    };
    let (nonce, tag) = cid.split_at(NONCE_LEN);
    let nonce = <&[u8; NONCE_LEN]>::try_from(nonce).map_err(|_| InvalidCid)?;
    let expected = self.keys.tag(nonce);
    // Compare without an early exit, so timing does not reveal a tag prefix.
    let difference = expected.iter().zip(tag).fold(0, |acc, (a, b)| acc | (a ^ b));
    if core::hint::black_box(difference) == 0 {
      Ok(())
    } else {
      Err(InvalidCid)
    }
  }

  fn cid_len(&self) -> usize {
    CID_LEN
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
