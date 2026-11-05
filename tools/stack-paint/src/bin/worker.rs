//! Painted-stack high-water marks of a provider TLS worker.
//!
//! One scenario runs per process, so that every first-use path, such as cold
//! capability detection and a thread's first allocation, happens inside the
//! measurement. A scenario names a credential, key exchange group, cipher
//! suite, connection mode, injected fault, and certificate verifier. On an 8 MiB
//! painted stack, the first call builds the client and server configurations
//! and drives the connections in memory: handshake, application data in both
//! directions, close notification, and destruction. A second call repeats the
//! connections with the configurations already built. Each fault must end the
//! connection with a recoverable error; a scenario that does not fails.
//!
//! `--list` prints every scenario name; `scripts/stack/painted.py` runs them.

extern crate alloc;

use alloc::borrow::Cow;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::time::Duration;

use rscrypto_stack_paint::{GuardedStack, json_string, round16};
use rustls::client::{ClientSessionMemoryCache, Resumption, Tls12Resumption};
use rustls::crypto::kx::SupportedKxGroup;
use rustls::crypto::{
  CryptoProvider, GetRandomFailed, Identity, KeyProvider, SecureRandom, SignatureScheme, Signer, SigningKey,
};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName, SubjectPublicKeyInfoDer, UnixTime};
use rustls::{ClientConfig, ClientSide, Connection, Error, RootCertStore, ServerConfig, ServerSide, SideData, Tcp};

const STACK_BYTES: usize = 8 * 1024 * 1024;
const PAINT: u8 = 0xa5;
/// 2026-10-22 UTC, inside the fixtures' 2026–2036 validity period.
const TIME: u64 = 1_792_627_200;
/// 2037-01-01 UTC, after every fixture expires.
const EXPIRED: u64 = 2_114_380_800;

const CREDENTIALS: [&str; 9] = [
  "p256", "p384", "ed25519", "rsa2048", "rsa3072", "rsa4096", "mldsa44", "mldsa65", "mldsa87",
];
const CHAIN_CREDENTIALS: [&str; 7] = ["p256", "p384", "ed25519", "rsa3072", "mldsa44", "mldsa65", "mldsa87"];
const TLS12_CREDENTIALS: [&str; 5] = ["p256", "p384", "ed25519", "rsa2048", "rsa3072"];
const GROUPS: [&str; 8] = [
  "x25519mlkem768",
  "x25519",
  "secp256r1",
  "secp384r1",
  "secp256r1mlkem768",
  "secp384r1mlkem1024",
  "mlkem768",
  "mlkem1024",
];
const TLS12_GROUPS: [&str; 3] = ["x25519", "secp256r1", "secp384r1"];
const SUITES: [&str; 3] = ["aes128", "aes256", "chacha"];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
  ServerAuth,
  Mutual,
  Intermediate,
  Resumed,
  Stateful,
  Compact,
  Retry,
  Tls12,
  Tls12Resumed,
}

const MODES: [(&str, Mode); 9] = [
  ("server-auth", Mode::ServerAuth),
  ("mutual", Mode::Mutual),
  ("intermediate", Mode::Intermediate),
  ("resumed", Mode::Resumed),
  ("stateful", Mode::Stateful),
  ("compact", Mode::Compact),
  ("retry", Mode::Retry),
  ("tls12", Mode::Tls12),
  ("tls12-resumed", Mode::Tls12Resumed),
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fault {
  None,
  UntrustedServer,
  UntrustedClient,
  Expired,
  BadServerSignature,
  BadClientSignature,
  CorruptHandshake,
  CorruptClientFinished,
  CorruptServerRecord,
  CorruptClientRecord,
  InvalidClientShare,
  InvalidServerShare,
  /// Client entropy fails from the given request of the measured connection on.
  ClientEntropy(usize),
  /// Server entropy fails from the given request of the measured connection on.
  ServerEntropy(usize),
  /// Server entropy fails while the configuration loads its signing key.
  ServerKeyEntropy,
  TicketMismatch,
  NoCommonGroup,
  NoCommonSuite,
}

impl Fault {
  fn parse(name: &str) -> Option<Self> {
    let simple = [
      ("none", Self::None),
      ("untrusted-server", Self::UntrustedServer),
      ("untrusted-client", Self::UntrustedClient),
      ("expired", Self::Expired),
      ("bad-server-signature", Self::BadServerSignature),
      ("bad-client-signature", Self::BadClientSignature),
      ("corrupt-handshake", Self::CorruptHandshake),
      ("corrupt-client-finished", Self::CorruptClientFinished),
      ("corrupt-server-record", Self::CorruptServerRecord),
      ("corrupt-client-record", Self::CorruptClientRecord),
      ("invalid-client-share", Self::InvalidClientShare),
      ("invalid-server-share", Self::InvalidServerShare),
      ("ticket-mismatch", Self::TicketMismatch),
      ("no-common-group", Self::NoCommonGroup),
      ("no-common-suite", Self::NoCommonSuite),
      ("server-entropy-key", Self::ServerKeyEntropy),
    ];
    if let Some((_, fault)) = simple.iter().find(|(label, _)| *label == name) {
      return Some(*fault);
    }
    if let Some(count) = name.strip_prefix("client-entropy-") {
      return count.parse().ok().map(Self::ClientEntropy);
    }
    name
      .strip_prefix("server-entropy-")
      .and_then(|count| count.parse().ok())
      .map(Self::ServerEntropy)
  }

  /// Whether the connection must end with an error.
  const fn fails(self) -> bool {
    !matches!(self, Self::None | Self::TicketMismatch)
  }
}

#[derive(Clone, Copy, Debug)]
struct Scenario<'a> {
  credential: &'a str,
  group: &'a str,
  suite: &'a str,
  mode: Mode,
  fault: Fault,
  webpki: bool,
}

impl<'a> Scenario<'a> {
  fn parse(name: &'a str) -> Option<Self> {
    let fields: Vec<&str> = name.split('/').collect();
    let [credential, group, suite, mode, fault, verifier] = fields.as_slice() else {
      return None;
    };
    let mode = MODES.iter().find(|(label, _)| label == mode)?.1;
    let tls12 = matches!(mode, Mode::Tls12 | Mode::Tls12Resumed);
    let credentials: &[&str] = if mode == Mode::Intermediate {
      &CHAIN_CREDENTIALS
    } else if tls12 {
      &TLS12_CREDENTIALS
    } else {
      &CREDENTIALS
    };
    let groups: &[&str] = if tls12 { &TLS12_GROUPS } else { &GROUPS };
    let valid = credentials.contains(credential)
      && groups.contains(group)
      && SUITES.contains(suite)
      && matches!(*verifier, "prepared" | "webpki")
      && (mode != Mode::Compact || credential.starts_with("mldsa"));
    valid
      .then(|| {
        Fault::parse(fault).map(|fault| Self {
          credential,
          group,
          suite,
          mode,
          fault,
          webpki: *verifier == "webpki",
        })
      })
      .flatten()
  }

  const fn tls12(&self) -> bool {
    matches!(self.mode, Mode::Tls12 | Mode::Tls12Resumed)
  }

  const fn resumed(&self) -> bool {
    matches!(self.mode, Mode::Resumed | Mode::Stateful | Mode::Tls12Resumed)
      || matches!(self.fault, Fault::TicketMismatch)
  }

  const fn mutual(&self) -> bool {
    !matches!(self.mode, Mode::ServerAuth)
  }
}

fn catalog() -> Vec<String> {
  let mut names = Vec::new();
  let mut push = |credential: &str, group: &str, suite: &str, mode: &str, fault: &str, verifier: &str| {
    names.push(format!("{credential}/{group}/{suite}/{mode}/{fault}/{verifier}"));
  };
  let (base_credential, base_group, base_suite) = ("p256", "x25519mlkem768", "aes128");
  for credential in CREDENTIALS {
    for group in GROUPS {
      for suite in SUITES {
        push(credential, group, suite, "mutual", "none", "prepared");
      }
    }
    push(credential, base_group, base_suite, "mutual", "none", "webpki");
    push(credential, base_group, base_suite, "server-auth", "none", "prepared");
    for verifier in ["prepared", "webpki"] {
      push(
        credential,
        base_group,
        base_suite,
        "mutual",
        "bad-server-signature",
        verifier,
      );
      push(
        credential,
        base_group,
        base_suite,
        "mutual",
        "bad-client-signature",
        verifier,
      );
    }
  }
  for credential in CHAIN_CREDENTIALS {
    for verifier in ["prepared", "webpki"] {
      push(credential, base_group, base_suite, "intermediate", "none", verifier);
    }
  }
  for suite in SUITES {
    for credential in ["p256", "mldsa87"] {
      push(credential, base_group, suite, "resumed", "none", "prepared");
      push(credential, base_group, suite, "stateful", "none", "prepared");
    }
    push(
      base_credential,
      base_group,
      suite,
      "resumed",
      "ticket-mismatch",
      "prepared",
    );
    for fault in [
      "corrupt-handshake",
      "corrupt-client-finished",
      "corrupt-server-record",
      "corrupt-client-record",
    ] {
      push(base_credential, base_group, suite, "mutual", fault, "prepared");
      push(base_credential, "x25519", suite, "tls12", fault, "prepared");
    }
  }
  for credential in ["mldsa44", "mldsa65", "mldsa87"] {
    push(credential, base_group, base_suite, "compact", "none", "prepared");
  }
  for group in ["secp256r1", "x25519mlkem768", "mlkem1024"] {
    push(base_credential, group, base_suite, "retry", "none", "prepared");
  }
  for credential in TLS12_CREDENTIALS {
    for group in TLS12_GROUPS {
      for suite in SUITES {
        push(credential, group, suite, "tls12", "none", "prepared");
        push(credential, group, suite, "tls12-resumed", "none", "prepared");
      }
    }
  }
  for verifier in ["prepared", "webpki"] {
    for fault in ["untrusted-server", "untrusted-client", "expired"] {
      push(base_credential, base_group, base_suite, "mutual", fault, verifier);
    }
  }
  for group in GROUPS {
    push(
      base_credential,
      group,
      base_suite,
      "mutual",
      "invalid-client-share",
      "prepared",
    );
    push(
      base_credential,
      group,
      base_suite,
      "mutual",
      "invalid-server-share",
      "prepared",
    );
  }
  for group in TLS12_GROUPS {
    push(
      base_credential,
      group,
      base_suite,
      "tls12",
      "invalid-client-share",
      "prepared",
    );
    push(
      base_credential,
      group,
      base_suite,
      "tls12",
      "invalid-server-share",
      "prepared",
    );
  }
  // Each entropy request of a P-256 client in this connection: client random,
  // the hybrid share, and client signing.
  for count in 0..6 {
    push(
      base_credential,
      base_group,
      base_suite,
      "mutual",
      &format!("client-entropy-{count}"),
      "prepared",
    );
  }
  for credential in ["p256", "p384"] {
    push(
      credential,
      base_group,
      base_suite,
      "mutual",
      "server-entropy-key",
      "prepared",
    );
  }
  // Each entropy request of the server in this connection: server random, the
  // hybrid share, and signing (none for Ed25519, more for RSA blinding).
  for (credential, requests) in [("p256", 4), ("ed25519", 3), ("rsa2048", 6), ("mldsa87", 4)] {
    for count in 0..requests {
      push(
        credential,
        base_group,
        base_suite,
        "mutual",
        &format!("server-entropy-{count}"),
        "prepared",
      );
    }
  }
  push(
    base_credential,
    base_group,
    base_suite,
    "mutual",
    "no-common-group",
    "prepared",
  );
  push(
    base_credential,
    base_group,
    base_suite,
    "mutual",
    "no-common-suite",
    "prepared",
  );
  names
}

#[derive(Debug)]
struct FixtureTime(u64);

impl rustls::time_provider::TimeProvider for FixtureTime {
  fn current_time(&self) -> Option<UnixTime> {
    Some(UnixTime::since_unix_epoch(Duration::from_secs(self.0)))
  }
}

struct Credential {
  root: &'static [u8],
  leaf: &'static [u8],
  key: &'static [u8],
  intermediate: Option<&'static [u8]>,
}

fn credential(name: &str, chain: bool) -> Credential {
  macro_rules! file {
    ($name:literal, $suffix:literal) => {
      include_bytes!(concat!(
        "../../../../rscrypto-rustls/tests/vectors/credentials/",
        $name,
        $suffix
      ))
      .as_slice()
    };
  }
  macro_rules! files {
    ($name:literal) => {
      Credential {
        root: file!($name, "-root.der"),
        leaf: file!($name, "-leaf.der"),
        key: file!($name, "-leaf-key.der"),
        intermediate: None,
      }
    };
  }
  macro_rules! chain {
    ($name:literal) => {
      Credential {
        intermediate: Some(file!($name, "-intermediate.der")),
        ..files!($name)
      }
    };
  }
  match (name, chain) {
    ("p256", false) => files!("p256"),
    ("p384", false) => files!("p384"),
    ("ed25519", false) => files!("ed25519"),
    ("rsa2048", false) => files!("rsa2048"),
    ("rsa3072", false) => files!("rsa3072"),
    ("rsa4096", false) => files!("rsa4096"),
    ("mldsa44", false) => files!("mldsa44"),
    ("mldsa65", false) => files!("mldsa65"),
    ("mldsa87", false) => files!("mldsa87"),
    ("p256", true) => chain!("p256-chain"),
    ("p384", true) => chain!("p384-chain"),
    ("ed25519", true) => chain!("ed25519-chain"),
    ("rsa3072", true) => chain!("rsa3072-chain"),
    ("mldsa44", true) => chain!("mldsa44-chain"),
    ("mldsa65", true) => chain!("mldsa65-chain"),
    _ => {
      assert!(name == "mldsa87" && chain, "unknown credential {name}");
      chain!("mldsa87-chain")
    }
  }
}

/// Entropy that fails from its `fail_at`-th request on, with no fallback.
#[derive(Debug)]
struct Entropy {
  fail_at: AtomicUsize,
  calls: AtomicUsize,
}

impl SecureRandom for Entropy {
  fn fill(&self, output: &mut [u8]) -> Result<(), GetRandomFailed> {
    let call = self.calls.fetch_add(1, Ordering::Relaxed);
    if call >= self.fail_at.load(Ordering::Relaxed) {
      return Err(GetRandomFailed);
    }
    rscrypto_rustls::random::SECURE_RANDOM.fill(output)
  }
}

static CLIENT_ENTROPY: Entropy = Entropy {
  fail_at: AtomicUsize::new(usize::MAX),
  calls: AtomicUsize::new(0),
};
static SERVER_ENTROPY: Entropy = Entropy {
  fail_at: AtomicUsize::new(usize::MAX),
  calls: AtomicUsize::new(0),
};

/// Loads keys through `inner`, then corrupts one byte of every signature.
#[derive(Debug)]
struct Corrupting(&'static dyn KeyProvider);

#[derive(Debug)]
struct CorruptKey(Box<dyn SigningKey>);

#[derive(Debug)]
struct CorruptSigner(Box<dyn Signer>);

impl KeyProvider for Corrupting {
  fn load_private_key(&self, key_der: PrivateKeyDer<'static>) -> Result<Box<dyn SigningKey>, Error> {
    Ok(Box::new(CorruptKey(self.0.load_private_key(key_der)?)))
  }
}

impl SigningKey for CorruptKey {
  fn choose_scheme(&self, offered: &[SignatureScheme]) -> Option<Box<dyn Signer>> {
    self
      .0
      .choose_scheme(offered)
      .map(|signer| Box::new(CorruptSigner(signer)) as Box<dyn Signer>)
  }

  fn public_key(&self) -> Option<SubjectPublicKeyInfoDer<'_>> {
    self.0.public_key()
  }
}

impl Signer for CorruptSigner {
  fn sign(self: Box<Self>, message: &[u8]) -> Result<Vec<u8>, Error> {
    let mut signature = self.0.sign(message)?;
    let middle = signature.len() / 2;
    if let Some(byte) = signature.get_mut(middle) {
      *byte ^= 1;
    }
    Ok(signature)
  }

  fn scheme(&self) -> SignatureScheme {
    self.0.scheme()
  }
}

fn leak<T>(value: T) -> &'static T {
  Box::leak(Box::new(value))
}

fn group(name: &str, entropy: Option<&'static Entropy>) -> &'static dyn SupportedKxGroup {
  use rscrypto_rustls::kx_group as kx;
  match entropy {
    None => match name {
      "x25519mlkem768" => kx::X25519MLKEM768,
      "x25519" => kx::X25519,
      "secp256r1" => kx::SECP256R1,
      "secp384r1" => kx::SECP384R1,
      "secp256r1mlkem768" => kx::SECP256R1MLKEM768,
      "secp384r1mlkem1024" => kx::SECP384R1MLKEM1024,
      "mlkem768" => kx::MLKEM768,
      _ => {
        assert_eq!(name, "mlkem1024", "unknown group");
        kx::MLKEM1024
      }
    },
    Some(random) => match name {
      "x25519mlkem768" => leak(kx::X25519MlKem768Group::new(random)),
      "x25519" => leak(kx::X25519Group::new(random)),
      "secp256r1" => leak(kx::P256Group::new(random)),
      "secp384r1" => leak(kx::P384Group::new(random)),
      "secp256r1mlkem768" => leak(kx::P256MlKem768Group::new(random)),
      "secp384r1mlkem1024" => leak(kx::P384MlKem1024Group::new(random)),
      "mlkem768" => leak(kx::MlKem768Group::new(random)),
      _ => {
        assert_eq!(name, "mlkem1024", "unknown group");
        leak(kx::MlKem1024Group::new(random))
      }
    },
  }
}

fn tls13_suite(name: &str) -> &'static rustls::Tls13CipherSuite {
  use rscrypto_rustls::cipher_suite as suite;
  match name {
    "aes128" => suite::TLS13_AES_128_GCM_SHA256,
    "aes256" => suite::TLS13_AES_256_GCM_SHA384,
    _ => suite::TLS13_CHACHA20_POLY1305_SHA256,
  }
}

fn tls12_suite(name: &str, rsa: bool) -> &'static rustls::Tls12CipherSuite {
  use rscrypto_rustls::cipher_suite as suite;
  match (name, rsa) {
    ("aes128", false) => suite::TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
    ("aes256", false) => suite::TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
    (_, false) => suite::TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
    ("aes128", true) => suite::TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
    ("aes256", true) => suite::TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
    (_, true) => suite::TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
  }
}

/// One endpoint's provider for `scenario`, with optional failing entropy.
fn provider(
  scenario: &Scenario<'_>,
  entropy: Option<&'static Entropy>,
  corrupt: bool,
  group_name: &str,
) -> CryptoProvider {
  let mut provider = rscrypto_rustls::default_provider();
  provider.kx_groups = Cow::Owned(vec![group(group_name, entropy)]);
  if scenario.tls12() {
    provider.tls13_cipher_suites = Cow::Borrowed(&[]);
    provider.tls12_cipher_suites = Cow::Owned(vec![tls12_suite(
      scenario.suite,
      scenario.credential.starts_with("rsa"),
    )]);
  } else {
    provider.tls13_cipher_suites = Cow::Owned(vec![tls13_suite(scenario.suite)]);
  }
  let random: &'static dyn SecureRandom = match entropy {
    Some(entropy) => entropy,
    None => rscrypto_rustls::random::SECURE_RANDOM,
  };
  provider.secure_random = random;
  let keys = rscrypto_rustls::sign::Provider::new(random);
  let keys: &'static dyn KeyProvider = if scenario.mode == Mode::Compact {
    leak(keys.compact())
  } else if entropy.is_some() {
    leak(keys)
  } else {
    rscrypto_rustls::sign::KEY_PROVIDER
  };
  provider.key_provider = if corrupt { leak(Corrupting(keys)) } else { keys };
  provider
}

struct Configs {
  client: Arc<ClientConfig>,
  server: Arc<ServerConfig>,
  /// The server that resumes, when it differs from the issuing server.
  resuming: Option<Arc<ServerConfig>>,
}

fn configs(scenario: &Scenario<'_>) -> Result<Configs, Error> {
  let fault = scenario.fault;
  let client_entropy = matches!(fault, Fault::ClientEntropy(_)).then_some(&CLIENT_ENTROPY);
  let server_entropy = matches!(fault, Fault::ServerEntropy(_) | Fault::ServerKeyEntropy).then_some(&SERVER_ENTROPY);
  let server_group = match fault {
    Fault::NoCommonGroup => "x25519",
    _ => scenario.group,
  };
  let (client_group, retry) = match (scenario.mode, scenario.group) {
    (Mode::Retry, "x25519") => ("secp256r1", true),
    (Mode::Retry, _) => ("x25519", true),
    (_, _) if fault == Fault::NoCommonGroup => ("secp256r1", false),
    _ => (scenario.group, false),
  };
  let mut client_provider = provider(
    scenario,
    client_entropy,
    fault == Fault::BadClientSignature,
    client_group,
  );
  if retry {
    // The client's first share uses a group the server lacks; the server's
    // HelloRetryRequest then selects the scenario's group.
    client_provider.kx_groups.to_mut().push(group(scenario.group, None));
  }
  let mut server_provider = provider(
    scenario,
    server_entropy,
    fault == Fault::BadServerSignature,
    server_group,
  );
  if fault == Fault::NoCommonSuite && !scenario.tls12() {
    let other = if scenario.suite == "chacha" { "aes128" } else { "chacha" };
    server_provider.tls13_cipher_suites = Cow::Owned(vec![tls13_suite(other)]);
  }
  let chain = scenario.mode == Mode::Intermediate;
  let Credential {
    root,
    leaf,
    key,
    intermediate,
  } = credential(scenario.credential, chain);
  let mut certificates = vec![CertificateDer::from(leaf)];
  certificates.extend(intermediate.map(CertificateDer::from));
  let identity = Arc::new(Identity::from_cert_chain(certificates)?);
  // An untrusted peer chains to a different credential's root.
  let other_root = credential(if scenario.credential == "p384" { "p256" } else { "p384" }, false).root;
  let anchors = |trusted: bool| -> Result<Arc<RootCertStore>, Error> {
    let mut roots = RootCertStore::empty();
    roots.add(CertificateDer::from(if trusted { root } else { other_root }))?;
    Ok(Arc::new(roots))
  };
  let client_roots = anchors(fault != Fault::UntrustedServer)?;
  let server_roots = anchors(fault != Fault::UntrustedClient)?;
  let time = Arc::new(FixtureTime(if fault == Fault::Expired { EXPIRED } else { TIME }));
  let server_provider = Arc::new(server_provider);
  let supported = server_provider.signature_verification_algorithms;
  let verifier: Arc<dyn rustls::server::danger::ClientVerifier> = if scenario.webpki {
    Arc::new(
      rustls::server::WebPkiClientVerifier::builder(server_roots, &server_provider)
        .build()
        .map_err(|error| Error::General(format!("client verifier: {error}")))?,
    )
  } else {
    Arc::new(
      rscrypto_rustls::verify::PreparedClientVerifier::with_algorithms(server_roots, supported)
        .map_err(|error| Error::General(format!("client verifier: {error}")))?,
    )
  };
  let client_supported = client_provider.signature_verification_algorithms;
  let client = ClientConfig::builder_with_details(Arc::new(client_provider), time.clone());
  let client = if scenario.webpki {
    client.with_root_certificates(client_roots)
  } else {
    client.dangerous().with_custom_certificate_verifier(Arc::new(
      rscrypto_rustls::verify::PreparedServerVerifier::with_algorithms(client_roots, client_supported)
        .map_err(|error| Error::General(format!("server verifier: {error}")))?,
    ))
  };
  let mut client = if scenario.mutual() {
    client.with_client_auth_cert(
      identity.clone(),
      PrivateKeyDer::try_from(key.to_vec()).map_err(|error| Error::General(error.into()))?,
    )?
  } else {
    client.with_no_client_auth()?
  };
  let server = ServerConfig::builder_with_details(server_provider.clone(), time);
  let server = if scenario.mutual() {
    server.with_client_cert_verifier(verifier)
  } else {
    server.with_no_client_auth()
  };
  let mut server = server.with_single_cert(
    identity,
    PrivateKeyDer::try_from(key.to_vec()).map_err(|error| Error::General(error.into()))?,
  )?;
  client.alpn_protocols = vec![b"provider-test".to_vec().into()];
  server.alpn_protocols = client.alpn_protocols.clone();
  let cache = || Resumption::store(Arc::new(ClientSessionMemoryCache::new(256)));
  let mut resuming = None;
  if scenario.tls12() {
    client.require_ems = true;
    server.require_ems = true;
    server.ticketer = None;
    if scenario.mode == Mode::Tls12Resumed {
      client.resumption = cache().tls12_resumption(Tls12Resumption::SessionIdOnly);
      server.session_storage = rustls::server::ServerSessionMemoryCache::new(256);
    } else {
      client.resumption = cache().tls12_resumption(Tls12Resumption::Disabled);
      server.session_storage = Arc::new(rustls::server::NoServerSessionStorage {});
    }
  } else if scenario.resumed() {
    client.resumption = cache();
    if scenario.mode == Mode::Stateful {
      server.ticketer = None;
      server.session_storage = rustls::server::ServerSessionMemoryCache::new(256);
    } else {
      server.ticketer = Some(server_provider.ticketer_factory.ticketer()?);
    }
    server.send_tls13_tickets = rustls::server::Tls13Tickets { default: 1, max: 1 };
    if fault == Fault::TicketMismatch {
      // A server with its own ticket key cannot decrypt the first server's
      // tickets and falls back to a full handshake.
      let mut other = server.clone();
      other.ticketer = Some(server_provider.ticketer_factory.ticketer()?);
      resuming = Some(Arc::new(other));
    }
  } else {
    client.resumption = cache().tls12_resumption(Tls12Resumption::Disabled);
    server.send_tls13_tickets = rustls::server::Tls13Tickets { default: 0, max: 0 };
  }
  Ok(Configs {
    client: Arc::new(client),
    server: Arc::new(server),
    resuming,
  })
}

struct Peer<S: SideData> {
  connection: Connection<S, Tcp>,
  input: rustls::VecInput,
  output: Vec<u8>,
  plaintext: Vec<u8>,
  flights: usize,
}

impl<S: SideData> Peer<S> {
  fn new(connection: Connection<S, Tcp>, output: Vec<u8>) -> Self {
    Self {
      connection,
      input: rustls::VecInput::default(),
      output,
      plaintext: Vec::new(),
      flights: 0,
    }
  }
}

/// Finds handshake message `kind` in a flight of plaintext TLS records.
fn handshake_body(flight: &[u8], kind: u8) -> Option<(usize, usize)> {
  let mut record = 0usize;
  while record.checked_add(5)? <= flight.len() {
    let length = usize::from(u16::from_be_bytes([
      flight[record.strict_add(3)],
      flight[record.strict_add(4)],
    ]));
    let body = record.strict_add(5);
    let end = body.checked_add(length)?;
    if end > flight.len() {
      return None;
    }
    if flight[record] == 22 {
      let mut message = body;
      while message.checked_add(4)? <= end {
        let size = (usize::from(flight[message.strict_add(1)]) << 16)
          | (usize::from(flight[message.strict_add(2)]) << 8)
          | usize::from(flight[message.strict_add(3)]);
        if flight[message] == kind {
          return Some((message.strict_add(4), size));
        }
        message = message.strict_add(4).checked_add(size)?;
      }
    }
    record = end;
  }
  None
}

fn read16(bytes: &[u8], at: usize) -> Option<usize> {
  Some(usize::from(u16::from_be_bytes([
    *bytes.get(at)?,
    *bytes.get(at.checked_add(1)?)?,
  ])))
}

/// The offset and length of the first key share in a TLS 1.3 hello.
fn key_share(flight: &[u8], client: bool) -> Option<(usize, usize)> {
  let (body, size) = handshake_body(flight, if client { 1 } else { 2 })?;
  let end = body.checked_add(size)?;
  // legacy_version, random, then the legacy session ID.
  let mut at = body.checked_add(34)?;
  at = at.checked_add(1)?.checked_add(usize::from(*flight.get(at)?))?;
  if client {
    at = at.checked_add(2)?.checked_add(read16(flight, at)?)?;
    at = at.checked_add(1)?.checked_add(usize::from(*flight.get(at)?))?;
  } else {
    at = at.checked_add(3)?;
  }
  let extensions_end = at.checked_add(2)?.checked_add(read16(flight, at)?)?.min(end);
  at = at.checked_add(2)?;
  while at.checked_add(4)? <= extensions_end {
    let (kind, length) = (read16(flight, at)?, read16(flight, at.checked_add(2)?)?);
    let data = at.checked_add(4)?;
    if kind == 0x0033 {
      // A client lists its shares after a two-byte length; each share and the
      // server's single share carry a group and a two-byte key length.
      let share = if client { data.checked_add(2)? } else { data };
      let key = share.checked_add(4)?;
      return Some((key, read16(flight, share.checked_add(2)?)?));
    }
    at = data.checked_add(length)?;
  }
  None
}

/// The ECDHE public point in a TLS 1.2 ClientKeyExchange or ServerKeyExchange.
fn tls12_point(flight: &[u8], client: bool) -> Option<(usize, usize)> {
  if client {
    let (body, _) = handshake_body(flight, 16)?;
    Some((body.checked_add(1)?, usize::from(*flight.get(body)?)))
  } else {
    // curve_type (3, named curve), the named group, then the point.
    let (body, _) = handshake_body(flight, 12)?;
    let length = body.checked_add(3)?;
    Some((length.checked_add(1)?, usize::from(*flight.get(length)?)))
  }
}

/// Applies `scenario`'s fault to a flight in transit and reports whether it did.
fn inject(scenario: &Scenario<'_>, flight: &mut [u8], from_client: bool, index: usize, application: bool) -> bool {
  let flip_last = |flight: &mut [u8]| {
    if let Some(byte) = flight.last_mut() {
      *byte ^= 1;
    }
  };
  let share = |flight: &mut [u8], client: bool| -> bool {
    let located = if scenario.tls12() {
      tls12_point(flight, client)
    } else {
      key_share(flight, client)
    };
    let Some((at, length)) = located else {
      return false;
    };
    let Some(key) = at.checked_add(length).and_then(|end| flight.get_mut(at..end)) else {
      return false;
    };
    // An all-zero X25519 share is a low-order point; all-ones encodes neither a
    // curve point nor ML-KEM coefficients below the modulus.
    key.fill(if scenario.group == "x25519" { 0 } else { 0xff });
    true
  };
  match scenario.fault {
    // The server's last handshake record is encrypted: TLS 1.3's first flight
    // ends with Finished, and TLS 1.2's Finished follows its first flight.
    Fault::CorruptHandshake if !from_client && index == usize::from(scenario.tls12()) && !application => {
      flip_last(flight);
      true
    }
    Fault::CorruptClientFinished if from_client && index == 1 && !application => {
      flip_last(flight);
      true
    }
    Fault::CorruptServerRecord if !from_client && application => {
      flip_last(flight);
      true
    }
    Fault::CorruptClientRecord if from_client && application => {
      flip_last(flight);
      true
    }
    Fault::InvalidClientShare if from_client && index == usize::from(scenario.tls12()) && !application => {
      share(flight, true)
    }
    Fault::InvalidServerShare if !from_client && index == 0 && !application => share(flight, false),
    _ => false,
  }
}

struct Driver<'a> {
  scenario: &'a Scenario<'a>,
  injected: bool,
  application: bool,
}

impl Driver<'_> {
  fn transfer<A: SideData, B: SideData>(
    &mut self,
    source: &mut Peer<A>,
    destination: &mut Peer<B>,
    from_client: bool,
  ) -> Result<usize, Error> {
    if source.output.is_empty() {
      return Ok(0);
    }
    let mut flight = core::mem::take(&mut source.output);
    let index = source.flights;
    source.flights = source.flights.strict_add(1);
    if !self.injected {
      self.injected = inject(self.scenario, &mut flight, from_client, index, self.application);
    }
    let mut chunk = flight.as_slice();
    while !chunk.is_empty() {
      if destination
        .input
        .read(&mut chunk)
        .map_err(|error| Error::General(error.to_string()))?
        == 0
      {
        return Err(Error::General("receiver stalled".into()));
      }
      destination
        .connection
        .read_tls(&mut destination.input, &mut destination.output)
        .handle_all(&mut destination.plaintext)?;
    }
    Ok(flight.len())
  }

  fn drive(&mut self, client: &mut Peer<ClientSide>, server: &mut Peer<ServerSide>) -> Result<(), Error> {
    for _ in 0..64 {
      let progress = self
        .transfer(client, server, true)?
        .strict_add(self.transfer(server, client, false)?);
      if client.output.is_empty()
        && server.output.is_empty()
        && !client.connection.is_handshaking()
        && !server.connection.is_handshaking()
      {
        return Ok(());
      }
      if progress == 0 {
        return Err(Error::General("handshake stalled".into()));
      }
    }
    Err(Error::General("handshake exceeded its flight count".into()))
  }
}

/// One connection of `scenario`: handshake, data both ways, and close.
fn connection(
  scenario: &Scenario<'_>,
  client: &Arc<ClientConfig>,
  server: &Arc<ServerConfig>,
  measured: bool,
) -> Result<(), Error> {
  if measured {
    arm(scenario.fault, false);
  }
  let mut output = Vec::new();
  let client = client
    .connect(ServerName::try_from("localhost").map_err(|error| Error::General(error.to_string()))?)
    .build(&mut output)?;
  let mut client = Peer::new(client, output);
  let mut server = Peer::new(Connection::new(server.clone())?, Vec::new());
  let mut driver = Driver {
    scenario,
    injected: false,
    application: false,
  };
  driver.drive(&mut client, &mut server)?;
  driver.application = true;
  let request = b"client application data".repeat(700);
  let response = b"server application data".repeat(700);
  client.connection.write(request.as_slice().into(), &mut client.output)?;
  server
    .connection
    .write(response.as_slice().into(), &mut server.output)?;
  driver.drive(&mut client, &mut server)?;
  if server.plaintext != request || client.plaintext != response {
    return Err(Error::General("application data differs".into()));
  }
  client.connection.send_close_notify(&mut client.output)?;
  server.connection.send_close_notify(&mut server.output)?;
  driver.transfer(&mut client, &mut server, true)?;
  driver.transfer(&mut server, &mut client, false)?;
  Ok(())
}

/// Every connection of `scenario`: a resumed scenario first connects to obtain a ticket.
fn connections(scenario: &Scenario<'_>, configs: &Configs) -> Result<(), Error> {
  arm(Fault::None, false);
  if scenario.resumed() {
    connection(scenario, &configs.client, &configs.server, false)?;
  }
  connection(
    scenario,
    &configs.client,
    configs.resuming.as_ref().unwrap_or(&configs.server),
    true,
  )
}

/// Sets the failing entropy for configuration construction or for the measured connection.
fn arm(fault: Fault, configuration: bool) {
  let (entropy, fail_at) = match (fault, configuration) {
    (Fault::ClientEntropy(count), false) => (&CLIENT_ENTROPY, count),
    (Fault::ServerEntropy(count), false) => (&SERVER_ENTROPY, count),
    (Fault::ServerKeyEntropy, true) => (&SERVER_ENTROPY, 0),
    _ => {
      for entropy in [&CLIENT_ENTROPY, &SERVER_ENTROPY] {
        entropy.fail_at.store(usize::MAX, Ordering::Relaxed);
      }
      return;
    }
  };
  entropy.calls.store(0, Ordering::Relaxed);
  entropy.fail_at.store(fail_at, Ordering::Relaxed);
}

fn outcome(result: &Result<(), Error>) -> String {
  match result {
    Ok(()) => "completed".to_owned(),
    Err(error) => format!("error: {error}"),
  }
}

fn main() {
  let arguments: Vec<String> = std::env::args().skip(1).collect();
  if arguments.first().map(String::as_str) == Some("--list") {
    for name in catalog() {
      println!("{name}");
    }
    return;
  }
  let [name] = arguments.as_slice() else {
    eprintln!("usage: rscrypto-stack-worker --list | CREDENTIAL/GROUP/SUITE/MODE/FAULT/VERIFIER");
    std::process::exit(2);
  };
  let Some(scenario) = Scenario::parse(name) else {
    eprintln!("unknown scenario {name}; see --list");
    std::process::exit(2);
  };
  let mut stack = GuardedStack::new(STACK_BYTES);
  let mut built = None;
  let mut cold = Ok(());
  let cold_bytes = stack.run(PAINT, || {
    arm(scenario.fault, true);
    cold = configs(&scenario).and_then(|configs| {
      let result = connections(&scenario, &configs);
      built = Some(configs);
      result
    });
  });
  let mut warm = None;
  let warm_bytes = built.as_ref().map(|configs| {
    stack.run(PAINT, || {
      warm = Some(connections(&scenario, configs));
    })
  });
  let expected = |result: &Result<(), Error>| result.is_err() == scenario.fault.fails();
  // A fault may stop configuration construction (entropy for key loading);
  // there is then no second call.
  let passed = expected(&cold) && warm.as_ref().is_none_or(expected);
  println!(
    "{{\"scenario\": {}, \"stack_bytes\": {STACK_BYTES}, \"cold_bytes\": {}, \"warm_bytes\": {}, \"cold_outcome\": {}, \"warm_outcome\": {}, \"expected_failure\": {}, \"passed\": {passed}}}",
    json_string(name),
    round16(cold_bytes),
    warm_bytes.map_or_else(|| "null".to_owned(), |bytes| round16(bytes).to_string()),
    json_string(&outcome(&cold)),
    warm
      .as_ref()
      .map_or_else(|| "null".to_owned(), |result| json_string(&outcome(result))),
    scenario.fault.fails()
  );
  if !passed {
    std::process::exit(1);
  }
}
