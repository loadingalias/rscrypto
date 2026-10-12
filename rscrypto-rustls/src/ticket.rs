//! Stateless session tickets with AES-256-GCM and rustls key rotation.

use alloc::{boxed::Box, sync::Arc, vec::Vec};
use core::fmt;
use core::sync::atomic::{AtomicU32, Ordering};
use core::time::Duration;
use std::sync::RwLock;
use std::time::Instant;

use rscrypto::aead::{Nonce96, expert::AeadWithNonce};
use rscrypto::{Aes256Gcm, Aes256GcmKey, Aes256GcmTag, SecretBytes, SecretVec};
use rustls::TicketRotator;
use rustls::crypto::GetRandomFailed;
use rustls::crypto::TicketProducer;

const ROTATION_SECONDS: u32 = 6 * 60 * 60;
const MAX_PLAINTEXT: usize = 16_384;
// Below RFC 8446 section 5.5's AES-GCM bound for 2^14-byte records.
const MAX_TICKETS: u32 = 1 << 24;
const NONCE_LEN: usize = 12;
const TAG_LEN: usize = 16;
const OVERHEAD: usize = NONCE_LEN + TAG_LEN;
const AAD: &[u8] = b"rscrypto-rustls ticket v1";

/// Create a stateless ticketer with six-hour rotation and advertised ticket lifetime.
///
/// Assign the returned shared owner to `ServerConfig::ticketer` as `Some(owner)`. Keep it across
/// connections that should share resumption state. Each key encrypts at most
/// 2^24 tickets, using unique counter nonces; exhaustion disables new tickets
/// until rotation. Plaintext tickets are bounded to 16 KiB.
///
/// Rotation happens on use, as specified by rustls's [`TicketRotator`]. Old keys
/// may remain resident while idle. Each key also has a monotonic twelve-hour
/// deadline: its first use at or after expiry releases the cipher and rejects
/// the operation, even after a long idle period. Keys are neither serialized nor shared across
/// processes, and the ticket wire format is private to this implementation.
///
/// Returns an error if initial OS entropy acquisition fails. Later rotation or
/// encryption failure suppresses ticket issuance without failing the connection.
pub fn ticketer() -> Result<Arc<dyn TicketProducer>, rustls::Error> {
  Ok(Arc::new(TicketRotator::new(
    Duration::from_secs(u64::from(ROTATION_SECONDS)),
    generate,
  )?))
}

fn generate() -> Result<Box<dyn TicketProducer>, rustls::Error> {
  TicketKey::generate_with(|out| crate::random::SECURE_RANDOM.fill(out))
    .map(|key| Box::new(key) as Box<dyn TicketProducer>)
    .map_err(Into::into)
}

struct TicketKey {
  cipher: RwLock<Option<Aes256Gcm>>,
  created_at: Instant,
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
      cipher: RwLock::new(Some(Aes256Gcm::new(key))),
      created_at: Instant::now(),
      next_nonce: AtomicU32::new(0),
    }
  }

  fn with_cipher<R>(&self, now: Instant, use_key: impl FnOnce(&Aes256Gcm) -> Option<R>) -> Option<R> {
    // Keep this key's twelve-hour bound independent of the rotator's retained
    // generations. An expired key releases its cipher when called.
    if now.checked_duration_since(self.created_at)? >= Duration::from_secs(u64::from(ROTATION_SECONDS) * 2) {
      self.cipher.write().ok()?.take();
      return None;
    }
    let cipher = self.cipher.read().ok()?;
    use_key(cipher.as_ref()?)
  }
}

impl fmt::Debug for TicketKey {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("TicketKey").finish_non_exhaustive()
  }
}

impl TicketProducer for TicketKey {
  fn lifetime(&self) -> Duration {
    Duration::from_secs(u64::from(ROTATION_SECONDS) * 2)
  }

  fn encrypt(&self, plain: &[u8]) -> Option<Vec<u8>> {
    if plain.len() > MAX_PLAINTEXT {
      return None;
    }
    self.with_cipher(Instant::now(), |cipher| {
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
        let sealed = cipher.encrypt_in_place(&Nonce96::from_bytes(nonce), AAD, body)?;
        tag.copy_from_slice(sealed.as_bytes());
        Ok::<(), rscrypto::aead::SealError>(())
      })
      .ok()?;
      // Successful output contains only authenticated ciphertext and a public nonce.
      Some(ticket.into_unprotected_vec())
    })
  }

  fn decrypt(&self, cipher: &[u8]) -> Option<Vec<u8>> {
    self.decrypt_at(cipher, Instant::now())
  }
}

impl TicketKey {
  fn decrypt_at(&self, cipher: &[u8], now: Instant) -> Option<Vec<u8>> {
    if !(OVERHEAD..=MAX_PLAINTEXT + OVERHEAD).contains(&cipher.len()) {
      return None;
    }
    self.with_cipher(now, |key| {
      let (nonce, payload) = cipher.split_at(NONCE_LEN);
      let nonce = Nonce96::from_bytes(nonce.try_into().ok()?);
      let (body, tag) = payload.split_at(payload.len().strict_sub(TAG_LEN));
      let tag = Aes256GcmTag::from_bytes(tag.try_into().ok()?);
      let plain = SecretVec::try_fill_with(body.len(), |out| {
        out.copy_from_slice(body);
        key.decrypt_in_place(&nonce, AAD, out, &tag)
      })
      .ok()?;
      // TicketProducer transfers plaintext ownership to rustls; it cannot borrow.
      Some(plain.into_unprotected_vec())
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use rustls::crypto::cipher::{AeadKey, Iv};

  #[test]
  fn ticket_format_matches_independent_aes_and_rejects_every_corruption_class() {
    let key = TicketKey::from_key(&Aes256GcmKey::from_bytes([0x42; 32]));
    let suite = rustls_aws_lc_rs::cipher_suite::TLS13_AES_256_GCM_SHA384;
    for len in [0, 1, 16, MAX_PLAINTEXT] {
      let plain = vec![0x5a; len];
      let ticket = key.encrypt(&plain).expect("ticket");
      let peer = suite.quic.expect("QUIC AEAD").packet_key(
        AeadKey::from([0x42; 32]),
        Iv::from(<[u8; NONCE_LEN]>::try_from(&ticket[..NONCE_LEN]).expect("nonce")),
      );
      let mut expected = plain.clone();
      let tag = peer.encrypt_in_place(0, AAD, &mut expected, None).expect("peer seal");
      expected.extend_from_slice(tag.as_ref());
      assert_eq!(&ticket[NONCE_LEN..], expected);
      assert_eq!(key.decrypt(&ticket).expect("open"), plain);
      for position in [0, NONCE_LEN - 1, NONCE_LEN, ticket.len() - 1] {
        let mut corrupt = ticket.clone();
        corrupt[position] ^= 1;
        assert!(key.decrypt(&corrupt).is_none());
      }
      assert!(key.decrypt(&ticket[..ticket.len() - 1]).is_none());
    }
    assert!(key.encrypt(&vec![0; MAX_PLAINTEXT + 1]).is_none());
    for len in [0, OVERHEAD - 1, MAX_PLAINTEXT + OVERHEAD + 1] {
      assert!(key.decrypt(&vec![0; len]).is_none());
    }
    TicketKey::generate_with(|out| {
      out[..17].fill(0x42);
      Err(GetRandomFailed)
    })
    .expect_err("partial entropy failure");
  }

  #[test]
  fn concurrent_nonce_reservations_exhaust_without_wrap_or_reuse() {
    let key = TicketKey::from_key(&Aes256GcmKey::from_bytes([0x42; 32]));
    key.next_nonce.store(MAX_TICKETS - 128, Ordering::Relaxed);
    let tickets = std::thread::scope(|scope| {
      (0..4)
        .map(|_| scope.spawn(|| (0..64).filter_map(|_| key.encrypt(b"ticket")).collect::<Vec<_>>()))
        .collect::<Vec<_>>()
        .into_iter()
        .flat_map(|thread| thread.join().expect("ticket worker"))
        .collect::<Vec<_>>()
    });
    assert_eq!(tickets.len(), 128);
    let mut nonces = tickets.iter().map(|ticket| &ticket[..NONCE_LEN]).collect::<Vec<_>>();
    nonces.sort_unstable();
    nonces.dedup();
    assert_eq!(nonces.len(), tickets.len());
    for ticket in tickets {
      assert_eq!(key.decrypt(&ticket).expect("old ticket survives exhaustion"), b"ticket");
    }
    assert!(key.encrypt(b"exhausted").is_none());
    assert_eq!(key.next_nonce.load(Ordering::Relaxed), MAX_TICKETS);
  }

  #[test]
  fn expired_key_rejects_an_authentic_ticket_and_releases_its_cipher() {
    let key = TicketKey::from_key(&Aes256GcmKey::from_bytes([0x42; 32]));
    let ticket = key.encrypt(b"session secret").expect("ticket");
    let deadline = key
      .created_at
      .checked_add(Duration::from_secs(u64::from(ROTATION_SECONDS) * 2))
      .expect("test deadline");
    let before = deadline
      .checked_sub(Duration::from_nanos(1))
      .expect("just before expiry");
    assert_eq!(key.decrypt_at(&ticket, before).expect("live key"), b"session secret");
    assert!(key.decrypt_at(&ticket, deadline).is_none());
    assert!(key.cipher.read().expect("key lock").is_none());
    assert!(key.encrypt(b"cannot revive an expired key").is_none());
    assert!(key.decrypt_at(&ticket, before).is_none());
  }

  #[test]
  #[ignore = "bounded wall-clock rotation evidence; run explicitly"]
  fn rotator_accepts_previous_key_then_erases_it() {
    use core::time::Duration;
    let rotator = Arc::new(TicketRotator::new(Duration::from_secs(1), generate).expect("rotator"));
    let ticket = rotator.encrypt(b"session secret").expect("ticket");
    #[cfg(feature = "default")]
    let (client_config, server_config) = {
      use crate::tests::tls::{self, Credential};
      let (client, mut server) = tls::configs(
        rustls_ring::DEFAULT_TLS13_PROVIDER,
        crate::default_provider(),
        Credential::P256,
        false,
      );
      server.ticketer = Some(rotator.clone());
      server.session_storage = Arc::new(rustls::server::NoServerSessionStorage {});
      let configs = (Arc::new(client), Arc::new(server));
      let (mut client, mut server) = tls::connections_from_configs(configs.0.clone(), configs.1.clone());
      tls::drive(&mut client, &mut server).expect("handshake before rotation");
      assert_eq!(client.handshake_kind(), Some(rustls::HandshakeKind::Full));
      configs
    };
    std::thread::sleep(Duration::from_millis(1100));
    assert_eq!(rotator.decrypt(&ticket).expect("previous key"), b"session secret");
    #[cfg(feature = "default")]
    {
      use crate::tests::tls;
      let (mut client, mut server) = tls::connections_from_configs(client_config, server_config);
      tls::drive(&mut client, &mut server).expect("resumption across rotation");
      assert_eq!(client.handshake_kind(), Some(rustls::HandshakeKind::Resumed));
      assert_eq!(server.handshake_kind(), Some(rustls::HandshakeKind::Resumed));
      tls::application_exchange(&mut client, &mut server);
    }
    let next = rotator.encrypt(b"session secret").expect("rotated key");
    assert_ne!(next, ticket);
    std::thread::sleep(Duration::from_millis(1100));
    assert!(rotator.decrypt(&ticket).is_none());
    assert_eq!(rotator.decrypt(&next).expect("new previous key"), b"session secret");
  }

  #[cfg(all(
    feature = "hkdf",
    feature = "chacha20poly1305",
    feature = "x25519",
    feature = "p256-ecdh",
    feature = "p384-ecdh",
    feature = "ml-kem",
    feature = "ecdsa-p256"
  ))]
  #[test]
  fn tls_stateless_resumption_expiry_and_key_replacement() {
    use crate::tests::tls::{self, Credential};
    use core::sync::atomic::AtomicU64;
    use core::time::Duration;
    use rustls::HandshakeKind;
    use rustls::pki_types::UnixTime;

    #[derive(Debug)]
    struct Clock(AtomicU64);
    impl rustls::time_provider::TimeProvider for Clock {
      fn current_time(&self) -> Option<UnixTime> {
        Some(UnixTime::since_unix_epoch(Duration::from_secs(
          self.0.load(Ordering::Relaxed),
        )))
      }
    }
    let clock = Arc::new(Clock(AtomicU64::new(1_792_627_200)));
    let (client, mut server) = tls::configs_with_time(
      rustls_ring::DEFAULT_TLS13_PROVIDER,
      crate::default_provider(),
      Credential::P256,
      false,
      clock.clone(),
    );
    let tickets = ticketer().expect("production ticketer");
    assert_eq!(tickets.lifetime(), Duration::from_secs(u64::from(ROTATION_SECONDS)));
    server.ticketer = Some(tickets);
    server.session_storage = Arc::new(rustls::server::NoServerSessionStorage {});
    let client = Arc::new(client);
    let mut server = Arc::new(server);
    for expected in [HandshakeKind::Full, HandshakeKind::Resumed, HandshakeKind::Resumed] {
      let (mut client, mut server) = tls::connections_from_configs(client.clone(), server.clone());
      tls::drive(&mut client, &mut server).expect("stateless handshake");
      assert_eq!(client.handshake_kind(), Some(expected));
      assert_eq!(server.handshake_kind(), Some(expected));
      tls::application_exchange(&mut client, &mut server);
    }
    clock
      .0
      .fetch_add(u64::from(ROTATION_SECONDS * 2) + 1, Ordering::Relaxed);
    let (mut expired_client, mut expired_server) = tls::connections_from_configs(client.clone(), server.clone());
    tls::drive(&mut expired_client, &mut expired_server).expect("expired ticket falls back");
    assert_eq!(expired_client.handshake_kind(), Some(HandshakeKind::Full));
    drop((expired_client, expired_server));

    // A restarted server cannot open tickets made with the old process key.
    Arc::make_mut(&mut server).ticketer = Some(ticketer().expect("replacement ticketer"));
    let (mut client, mut server) = tls::connections_from_configs(client, server);
    tls::drive(&mut client, &mut server).expect("unknown key falls back");
    assert_eq!(client.handshake_kind(), Some(HandshakeKind::Full));
    tls::application_exchange(&mut client, &mut server);
  }
}
