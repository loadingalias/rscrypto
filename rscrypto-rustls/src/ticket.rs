//! Stateless session tickets with AES-256-GCM and time-based key rotation.

use alloc::{sync::Arc, vec::Vec};
use core::fmt;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use core::time::Duration;
use std::sync::RwLock;
#[cfg(feature = "getrandom")]
use std::time::Instant;

use rscrypto::aead::{Nonce96, expert::AeadWithNonce};
use rscrypto::{Aes256Gcm, Aes256GcmKey, Aes256GcmTag, SecretBytes, SecretVec};
use rustls::crypto::{GetRandomFailed, SecureRandom, TicketProducer};
use rustls::time_provider::TimeProvider;

const ROTATION: Duration = Duration::from_secs(6 * 60 * 60);
const MAX_PLAINTEXT: usize = 16_384;
// Below RFC 8446 section 5.5's AES-GCM bound for 2^14-byte records.
const MAX_TICKETS: u32 = 1 << 24;
const NONCE_LEN: usize = 12;
const TAG_LEN: usize = 16;
const OVERHEAD: usize = NONCE_LEN + TAG_LEN;
const AAD: &[u8] = b"rscrypto-rustls ticket v1";

/// Create a stateless ticketer over OS entropy and the monotonic clock.
///
/// Behaves as [`ticketer_with`], with keys from OS entropy and rotation timed by
/// [`std::time::Instant`], which cannot move backward.
///
/// # Errors
///
/// Returns an error if OS entropy fails to produce the first key.
#[cfg(feature = "getrandom")]
pub fn ticketer() -> Result<Arc<dyn TicketProducer>, rustls::Error> {
  Rotator::new(crate::random::SECURE_RANDOM, Clock::Monotonic(Instant::now()))
    .map(|rotator| Arc::new(rotator) as Arc<dyn TicketProducer>)
}

/// Create a stateless ticketer with keys from `random` and rotation timed by `time`.
///
/// Assign the returned shared owner to `ServerConfig::ticketer` as `Some(owner)`. Keep it across
/// connections that should share resumption state. Pass the same clock as `ServerConfig`'s time
/// provider. The ticketer draws a 32-byte key from `random` now, and one more each time it rotates.
///
/// A key encrypts new tickets for six hours and opens them for twelve; the advertised ticket
/// lifetime is six hours. Rotation happens on use: the first encryption or decryption after a key
/// is six hours old replaces it, keeps it to open earlier tickets, and erases the key before it.
/// A key twelve or more hours old is erased on the first use, even after a long idle period. Idle
/// keys stay resident until then. Each key encrypts at most 2^24 tickets, using unique counter
/// nonces; exhaustion disables new tickets until rotation. Plaintext tickets are bounded to 16 KiB.
///
/// `time` alone decides key age. The ticketer never lets its view of time move backward: a clock
/// that falls behind delays rotation until it passes its latest reading. If `time` returns no
/// reading, the ticketer issues and opens no tickets for that call. If `random` fails during
/// rotation, no new tickets are issued until a later rotation succeeds; the previous key still
/// opens its tickets. These failures fall back to full handshakes and never fail the connection.
///
/// Keys are neither serialized nor shared across processes, and the ticket wire format is private
/// to this implementation.
///
/// # Errors
///
/// Returns an error if `time` has no reading or `random` fails to produce the first key.
pub fn ticketer_with(
  random: &'static dyn SecureRandom,
  time: Arc<dyn TimeProvider>,
) -> Result<Arc<dyn TicketProducer>, rustls::Error> {
  Rotator::new(random, Clock::Caller(time)).map(|rotator| Arc::new(rotator) as Arc<dyn TicketProducer>)
}

/// The rotation clock, as a duration since a fixed origin.
enum Clock {
  #[cfg(feature = "getrandom")]
  Monotonic(Instant),
  Caller(Arc<dyn TimeProvider>),
}

impl Clock {
  fn now(&self) -> Option<Duration> {
    match self {
      #[cfg(feature = "getrandom")]
      Self::Monotonic(origin) => Some(origin.elapsed()),
      Self::Caller(time) => time.current_time().map(|now| Duration::from_secs(now.as_secs())),
    }
  }
}

/// A ticket key and the clock reading at its creation.
struct Generation {
  key: TicketKey,
  created: Duration,
}

impl Generation {
  fn age(&self, now: Duration) -> Duration {
    // `now` never precedes a reading the rotator has already used.
    now.saturating_sub(self.created)
  }
}

#[derive(Default)]
struct Keys {
  current: Option<Generation>,
  previous: Option<Generation>,
}

impl Keys {
  fn is_current(&self, now: Duration) -> bool {
    self.current.as_ref().is_some_and(|key| key.age(now) < ROTATION)
      && self
        .previous
        .as_ref()
        .is_none_or(|key| key.age(now) < ROTATION.saturating_mul(2))
  }

  fn rotate(&mut self, now: Duration, random: &dyn SecureRandom) {
    if self
      .previous
      .as_ref()
      .is_some_and(|key| key.age(now) >= ROTATION.saturating_mul(2))
    {
      self.previous = None;
    }
    if self.current.as_ref().is_some_and(|key| key.age(now) < ROTATION) {
      return;
    }
    // A current key that can still open tickets replaces `previous`; dropping a key clears it.
    // After an earlier failed rotation there is no current key, and `previous` stays.
    if let Some(current) = self.current.take() {
      self.previous = (current.age(now) < ROTATION.saturating_mul(2)).then_some(current);
    }
    self.current = TicketKey::generate_with(|out| random.fill(out))
      .ok()
      .map(|key| Generation { key, created: now });
  }
}

struct Rotator {
  random: &'static dyn SecureRandom,
  clock: Clock,
  /// The latest clock reading in whole seconds, so a lagging clock cannot rewind key age.
  latest: AtomicU64,
  keys: RwLock<Keys>,
}

impl Rotator {
  fn new(random: &'static dyn SecureRandom, clock: Clock) -> Result<Self, rustls::Error> {
    let now = Duration::from_secs(clock.now().ok_or(rustls::Error::FailedToGetCurrentTime)?.as_secs());
    let key = TicketKey::generate_with(|out| random.fill(out))?;
    Ok(Self {
      random,
      clock,
      latest: AtomicU64::new(now.as_secs()),
      keys: RwLock::new(Keys {
        current: Some(Generation { key, created: now }),
        previous: None,
      }),
    })
  }

  fn now(&self) -> Option<Duration> {
    let reading = self.clock.now()?.as_secs();
    let latest = self.latest.fetch_max(reading, Ordering::Relaxed).max(reading);
    Some(Duration::from_secs(latest))
  }

  /// Run `use_keys` on keys that are current at the clock's present reading.
  fn with_keys<R>(&self, use_keys: impl FnOnce(&Keys) -> Option<R>) -> Option<R> {
    let now = self.now()?;
    {
      let keys = self.keys.read().ok()?;
      if keys.is_current(now) {
        return use_keys(&keys);
      }
    }
    let mut keys = self.keys.write().ok()?;
    keys.rotate(now, self.random);
    use_keys(&keys)
  }
}

impl fmt::Debug for Rotator {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Ticketer").finish_non_exhaustive()
  }
}

impl TicketProducer for Rotator {
  fn encrypt(&self, plain: &[u8]) -> Option<Vec<u8>> {
    self.with_keys(|keys| keys.current.as_ref()?.key.encrypt(plain))
  }

  fn decrypt(&self, cipher: &[u8]) -> Option<Vec<u8>> {
    self.with_keys(|keys| {
      [&keys.current, &keys.previous]
        .into_iter()
        .flatten()
        .find_map(|generation| generation.key.decrypt(cipher))
    })
  }

  fn lifetime(&self) -> Duration {
    ROTATION
  }
}

struct TicketKey {
  cipher: Aes256Gcm,
  next_nonce: AtomicU32,
}

impl TicketKey {
  fn generate_with(fill: impl FnOnce(&mut [u8; 32]) -> Result<(), GetRandomFailed>) -> Result<Self, GetRandomFailed> {
    let secret = SecretBytes::try_fill_with(fill)?;
    Ok(Self::from_key(&Aes256GcmKey::generate(|out| {
      out.copy_from_slice(secret.as_bytes());
    })))
  }

  fn from_key(key: &Aes256GcmKey) -> Self {
    Self {
      cipher: Aes256Gcm::new(key),
      next_nonce: AtomicU32::new(0),
    }
  }

  fn encrypt(&self, plain: &[u8]) -> Option<Vec<u8>> {
    if plain.len() > MAX_PLAINTEXT {
      return None;
    }
    // Relaxed ordering provides uniqueness; no other state is published here.
    // A nonce is consumed even if a later allocation or encryption fails.
    let counter = self
      .next_nonce
      .try_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
        (next < MAX_TICKETS).then(|| next.strict_add(1))
      })
      .ok()?;
    let mut nonce = [0; NONCE_LEN];
    nonce[8..].copy_from_slice(&counter.to_be_bytes());
    let ticket = SecretVec::try_fill_with(plain.len().strict_add(OVERHEAD), |out| {
      let (prefix, payload) = out.split_at_mut(NONCE_LEN);
      prefix.copy_from_slice(&nonce);
      let (body, tag) = payload.split_at_mut(plain.len());
      body.copy_from_slice(plain);
      let sealed = self.cipher.encrypt_in_place(&Nonce96::from_bytes(nonce), AAD, body)?;
      tag.copy_from_slice(sealed.as_bytes());
      Ok::<(), rscrypto::aead::SealError>(())
    })
    .ok()?;
    // Successful output contains only authenticated ciphertext and a public nonce.
    Some(ticket.into_unprotected_vec())
  }

  fn decrypt(&self, cipher: &[u8]) -> Option<Vec<u8>> {
    if !(OVERHEAD..=MAX_PLAINTEXT + OVERHEAD).contains(&cipher.len()) {
      return None;
    }
    let (nonce, payload) = cipher.split_at(NONCE_LEN);
    let nonce = Nonce96::from_bytes(nonce.try_into().ok()?);
    let (body, tag) = payload.split_at(payload.len().strict_sub(TAG_LEN));
    let tag = Aes256GcmTag::from_bytes(tag.try_into().ok()?);
    let plain = SecretVec::try_fill_with(body.len(), |out| {
      out.copy_from_slice(body);
      self.cipher.decrypt_in_place(&nonce, AAD, out, &tag)
    })
    .ok()?;
    // TicketProducer transfers plaintext ownership to rustls; it cannot borrow.
    Some(plain.into_unprotected_vec())
  }
}

#[cfg(test)]
#[path = "tests/ticket.rs"]
mod tests;
