use super::*;
use core::sync::atomic::AtomicBool;
use rustls::crypto::cipher::{AeadKey, Iv};
use rustls::pki_types::UnixTime;

const HOUR: u64 = 60 * 60;
const START: u64 = 1_792_627_200;

/// A caller clock in whole seconds that can report no reading.
#[derive(Debug)]
struct Clock {
  seconds: AtomicU64,
  available: AtomicBool,
}

impl Clock {
  fn new() -> Arc<Self> {
    Arc::new(Self {
      seconds: AtomicU64::new(START),
      available: AtomicBool::new(true),
    })
  }

  fn set(&self, seconds: u64) {
    self.seconds.store(seconds, Ordering::Relaxed);
  }
}

impl TimeProvider for Clock {
  fn current_time(&self) -> Option<UnixTime> {
    self
      .available
      .load(Ordering::Relaxed)
      .then(|| UnixTime::since_unix_epoch(Duration::from_secs(self.seconds.load(Ordering::Relaxed))))
  }
}

/// Distinct deterministic keys per draw; counts draws and can refuse them.
#[derive(Debug)]
struct Keys {
  draws: AtomicU32,
  refuse: AtomicBool,
}

impl Keys {
  fn leak() -> &'static Self {
    Box::leak(Box::new(Self {
      draws: AtomicU32::new(0),
      refuse: AtomicBool::new(false),
    }))
  }

  fn draws(&self) -> u32 {
    self.draws.load(Ordering::Relaxed)
  }
}

impl SecureRandom for Keys {
  fn fill(&self, output: &mut [u8]) -> Result<(), GetRandomFailed> {
    let draw = self.draws.fetch_add(1, Ordering::Relaxed);
    output.fill(u8::try_from(draw % 251).expect("small draw index").strict_add(1));
    if self.refuse.load(Ordering::Relaxed) {
      output[..7].fill(0x42);
      return Err(GetRandomFailed);
    }
    Ok(())
  }
}

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
  .err()
  .expect("partial entropy failure");
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
fn caller_clock_rotates_keeps_one_previous_key_and_erases_on_first_late_use() {
  let clock = Clock::new();
  let random = Keys::leak();
  let tickets = ticketer_with(random, clock.clone()).expect("ticketer");
  assert_eq!(tickets.lifetime(), Duration::from_secs(6 * HOUR));
  assert_eq!(random.draws(), 1, "the first key is drawn at construction");
  let first = tickets.encrypt(b"first").expect("first key");

  clock.set(START + 6 * HOUR - 1);
  let still_first = tickets.encrypt(b"still first").expect("first key");
  assert_eq!(random.draws(), 1, "no rotation before six hours");

  clock.set(START + 6 * HOUR);
  let second = tickets.encrypt(b"second").expect("second key");
  assert_eq!(random.draws(), 2, "the first use at six hours rotates");
  assert_ne!(second[NONCE_LEN..], first[NONCE_LEN..]);
  for (ticket, plain) in [
    (&first, &b"first"[..]),
    (&still_first, b"still first"),
    (&second, b"second"),
  ] {
    assert_eq!(tickets.decrypt(ticket).expect("current or previous key"), plain);
  }

  clock.set(START + 12 * HOUR - 1);
  assert_eq!(
    tickets.decrypt(&first).expect("previous key before twelve hours"),
    b"first"
  );

  clock.set(START + 12 * HOUR);
  assert!(tickets.decrypt(&first).is_none(), "a twelve-hour-old key is erased");
  assert_eq!(tickets.decrypt(&second).expect("now the previous key"), b"second");
  assert_eq!(random.draws(), 3);
  let third = tickets.encrypt(b"third").expect("third key");

  // After a long idle period, the first use erases both resident keys.
  clock.set(START + 100 * HOUR);
  assert!(tickets.decrypt(&second).is_none());
  assert!(tickets.decrypt(&third).is_none());
  assert_eq!(random.draws(), 4);
}

#[test]
fn lagging_or_missing_time_never_rewinds_or_revives_keys() {
  let clock = Clock::new();
  let random = Keys::leak();
  let tickets = ticketer_with(random, clock.clone()).expect("ticketer");
  let first = tickets.encrypt(b"first").expect("first key");
  clock.set(START + 13 * HOUR);
  assert!(tickets.decrypt(&first).is_none(), "first key erased");
  let draws = random.draws();

  // A clock that falls behind keeps the latest reading: no rotation, no revival.
  clock.set(START);
  assert!(tickets.decrypt(&first).is_none());
  let current = tickets.encrypt(b"current").expect("current key");
  assert_eq!(random.draws(), draws);
  clock.set(START + 13 * HOUR + 6 * HOUR - 1);
  assert_eq!(tickets.decrypt(&current).expect("same key"), b"current");
  assert_eq!(random.draws(), draws);

  clock.available.store(false, Ordering::Relaxed);
  assert!(tickets.encrypt(b"no time").is_none());
  assert!(tickets.decrypt(&current).is_none());
  assert_eq!(random.draws(), draws, "no reading, no rotation");
}

#[test]
fn rotation_entropy_failure_suppresses_new_tickets_but_keeps_the_previous_key() {
  let clock = Clock::new();
  let random = Keys::leak();
  let tickets = ticketer_with(random, clock.clone()).expect("ticketer");
  let first = tickets.encrypt(b"first").expect("first key");

  random.refuse.store(true, Ordering::Relaxed);
  clock.set(START + 6 * HOUR);
  assert!(tickets.encrypt(b"refused").is_none(), "no key, no ticket");
  assert_eq!(tickets.decrypt(&first).expect("previous key survives"), b"first");
  assert!(tickets.encrypt(b"retry").is_none());

  random.refuse.store(false, Ordering::Relaxed);
  let recovered = tickets.encrypt(b"recovered").expect("a later rotation succeeds");
  assert_eq!(tickets.decrypt(&first).expect("previous key still kept"), b"first");
  assert_eq!(tickets.decrypt(&recovered).expect("new key"), b"recovered");

  for (refuse, available, expected) in [
    (true, true, rustls::Error::FailedToGetRandomBytes),
    (false, false, rustls::Error::FailedToGetCurrentTime),
  ] {
    let random = Keys::leak();
    random.refuse.store(refuse, Ordering::Relaxed);
    let clock = Clock::new();
    clock.available.store(available, Ordering::Relaxed);
    assert_eq!(ticketer_with(random, clock).err(), Some(expected));
  }
}

#[cfg(feature = "getrandom")]
#[test]
fn os_ticketer_round_trips_with_a_six_hour_lifetime() {
  let tickets = ticketer().expect("OS ticketer");
  assert_eq!(tickets.lifetime(), ROTATION);
  let ticket = tickets.encrypt(b"session").expect("ticket");
  assert_eq!(tickets.decrypt(&ticket).expect("open"), b"session");
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
fn tls_stateless_resumption_follows_the_caller_clock_and_key_replacement() {
  use crate::tests::tls::{self, Credential};
  use rustls::HandshakeKind;

  let clock = Clock::new();
  let (client, mut server) = tls::configs_with_time(
    rustls_ring::DEFAULT_TLS13_PROVIDER,
    tls::provider(),
    Credential::P256,
    false,
    clock.clone(),
  );
  server.ticketer = Some(ticketer_with(&crate::tests::RANDOM, clock.clone()).expect("ticketer"));
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

  // A ticket issued just before rotation resumes through the previous key after it.
  for (time, stage) in [
    (START + 6 * HOUR - 10, "before rotation"),
    (START + 6 * HOUR + 10, "after rotation"),
  ] {
    clock.set(time);
    let (mut resumed_client, mut resumed_server) = tls::connections_from_configs(client.clone(), server.clone());
    tls::drive(&mut resumed_client, &mut resumed_server).expect(stage);
    assert_eq!(resumed_client.handshake_kind(), Some(HandshakeKind::Resumed), "{stage}");
  }

  clock.set(START + 13 * HOUR);
  let (mut expired_client, mut expired_server) = tls::connections_from_configs(client.clone(), server.clone());
  tls::drive(&mut expired_client, &mut expired_server).expect("expired ticket falls back");
  assert_eq!(expired_client.handshake_kind(), Some(HandshakeKind::Full));
  drop((expired_client, expired_server));

  // A restarted server cannot open tickets made with the old process key.
  Arc::make_mut(&mut server).ticketer = Some(ticketer_with(&crate::tests::RANDOM, clock).expect("replacement"));
  let (mut client, mut server) = tls::connections_from_configs(client, server);
  tls::drive(&mut client, &mut server).expect("unknown key falls back");
  assert_eq!(client.handshake_kind(), Some(HandshakeKind::Full));
  tls::application_exchange(&mut client, &mut server);
}
