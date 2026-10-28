//! Matched TLS handshakes and application traffic through production TLS stacks.

extern crate alloc;

#[path = "../../../benches/common/criterion.rs"]
#[expect(
  dead_code,
  reason = "rows build their fixtures lazily, so they register custom routines"
)]
mod bench_config;

use bench_config::checked_routine;
use core::hint::black_box;
use criterion::{Criterion, Throughput};
use rscrypto_provider_bench::{
  BACKENDS, CREDENTIALS, Fixture, GROUPS, Profile, RSA_CREDENTIALS, SUITES, Workload, supports,
};

fn connection(c: &mut Criterion, name: &str, profile: Profile<'_>, workload: Workload) {
  if !bench_config::selected(name) {
    return;
  }
  let mut measured = c.benchmark_group(name);
  measured.throughput(Throughput::Elements(1));
  for backend in BACKENDS {
    if !bench_config::selected(&format!("{name}/{backend}"))
      || !supports(backend, profile)
      || (workload == Workload::CompactKey && !backend.starts_with("rscrypto"))
    {
      continue;
    }
    // Discovery registers names without importing keys or connecting peers.
    measured.bench_function(
      backend,
      checked_routine(
        None,
        |fixture| {
          let checked = Fixture::with_workload(backend, profile, workload);
          drop(checked.checked_connect());
          *fixture = Some(checked);
        },
        |b, fixture| {
          let fixture = fixture.as_ref().expect("the check builds the fixture");
          b.iter(|| drop(black_box(fixture.connect())));
        },
      ),
    );
  }
  measured.finish();
}

fn connections(c: &mut Criterion) {
  for credential in CREDENTIALS {
    for group in GROUPS {
      for suite in SUITES {
        for resumed in [false, true] {
          let mode = if resumed { "resumed" } else { "full" };
          connection(
            c,
            &format!("tls13/handshake/{credential}/{group}/{suite}/{mode}"),
            Profile {
              credential,
              group,
              suite,
              resumed,
            },
            Workload::Mutual,
          );
        }
      }
    }
  }
}

fn expanded(c: &mut Criterion) {
  // Hold KX and record cipher fixed while changing authentication work.
  for credential in CREDENTIALS {
    for (family, workload) in [
      ("server-auth", Workload::ServerAuth),
      ("intermediate", Workload::Intermediate),
    ] {
      connection(
        c,
        &format!("tls13/{family}/{credential}/X25519/AES128-GCM-SHA256/full"),
        Profile {
          credential,
          group: "X25519",
          suite: "AES128-GCM-SHA256",
          resumed: false,
        },
        workload,
      );
    }
  }
  // Match the existing p256 stateless rows: resumption omits certificate signatures.
  for group in GROUPS {
    for suite in SUITES {
      connection(
        c,
        &format!("tls13/stateful/p256/{group}/{suite}/resumed"),
        Profile {
          credential: "p256",
          group,
          suite,
          resumed: true,
        },
        Workload::Stateful,
      );
    }
  }
  for credential in RSA_CREDENTIALS {
    for suite in SUITES {
      connection(
        c,
        &format!("tls13/rsa/{credential}/X25519/{suite}/full"),
        Profile {
          credential,
          group: "X25519",
          suite,
          resumed: false,
        },
        Workload::Mutual,
      );
    }
  }
  connection(
    c,
    "tls13/hrr/p256/secp256r1/AES128-GCM-SHA256/full",
    Profile {
      credential: "p256",
      group: "secp256r1",
      suite: "AES128-GCM-SHA256",
      resumed: false,
    },
    Workload::HelloRetryRequest,
  );
  connection(
    c,
    "tls13/compact-key/mldsa65/X25519/AES128-GCM-SHA256/full",
    Profile {
      credential: "mldsa65",
      group: "X25519",
      suite: "AES128-GCM-SHA256",
      resumed: false,
    },
    Workload::CompactKey,
  );
}

fn traffic(c: &mut Criterion) {
  for suite in SUITES {
    // Small writes expose per-record overhead; 64 KiB and 1 MiB amortize it over
    // multiple maximum-size records. Sizes are per direction for duplex rows.
    for size in [64usize, 256, 1024, 16_384, 65_536, 1_048_576] {
      for bidirectional in [false, true] {
        let family = if bidirectional {
          "traffic-bidirectional"
        } else {
          "traffic"
        };
        let name = format!("tls13/{family}/{suite}/{size}");
        if !bench_config::selected(&name) {
          continue;
        }
        let mut measured = c.benchmark_group(&name);
        let total = size.strict_mul(if bidirectional { 2 } else { 1 });
        measured.throughput(Throughput::Bytes(u64::try_from(total).expect("transfer size")));
        for backend in BACKENDS {
          if !bench_config::selected(&format!("{name}/{backend}")) {
            continue;
          }
          let transfer =
            |pair: &mut rscrypto_provider_bench::Pair, client_bytes: &[u8], server_bytes: &[u8], output: &mut [u8]| {
              if bidirectional {
                pair.transfer_both(black_box(client_bytes), black_box(server_bytes), black_box(output));
              } else {
                pair.transfer(black_box(client_bytes), black_box(output));
              }
            };
          measured.bench_function(
            backend,
            checked_routine(
              (vec![0x42; size], vec![0xa7; size], vec![0; total], None),
              |(client_bytes, server_bytes, output, connection)| {
                let fixture = Fixture::new(
                  backend,
                  Profile {
                    credential: "p256",
                    group: "X25519",
                    suite,
                    resumed: false,
                  },
                );
                let mut pair = fixture.checked_connect();
                // Change both directions between calls so replayed plaintext cannot pass.
                for (client_fill, server_fill) in [(0x42, 0xa7), (0x53, 0xb8)] {
                  client_bytes.fill(client_fill);
                  server_bytes.fill(server_fill);
                  output.fill(0);
                  transfer(&mut pair, client_bytes, server_bytes, output);
                  assert_eq!(
                    client_bytes.as_slice(),
                    &output[..size],
                    "authenticated client-to-server transfer"
                  );
                  if bidirectional {
                    assert_eq!(
                      server_bytes.as_slice(),
                      &output[size..],
                      "authenticated server-to-client transfer"
                    );
                  }
                }
                client_bytes.fill(0x42);
                server_bytes.fill(0xa7);
                *connection = Some((fixture, pair));
              },
              |b, (client_bytes, server_bytes, output, connection)| {
                let (_fixture, pair) = connection.as_mut().expect("the check opens the connection");
                b.iter(|| {
                  transfer(pair, client_bytes, server_bytes, output);
                  black_box(&*output);
                });
              },
            ),
          );
        }
        measured.finish();
      }
    }
  }
}

/// Server-certificate verification as a short-lived client pays it: build a
/// verifier over a Linux distribution bundle, then verify the chains that
/// top-secret's origins serve. TLS 1.3 CertificateVerify is excluded: both
/// rscrypto verifiers delegate it to the same rustls verifier.
fn server_verifiers(c: &mut Criterion) {
  use alloc::sync::Arc;
  use core::time::Duration;
  use rscrypto_rustls::verify::PreparedServerVerifier;
  use rustls::RootCertStore;
  use rustls::client::WebPkiServerVerifier;
  use rustls::client::danger::{ServerIdentity, ServerVerifier};
  use rustls::crypto::{CertificateIdentity, Identity};
  use rustls::pki_types::pem::PemObject;
  use rustls::pki_types::{CertificateDer, ServerName, UnixTime};

  macro_rules! fixture {
    ($file:literal) => {
      include_bytes!(concat!("../fixtures/server-verifier/", $file))
    };
  }
  const BUNDLES: [(&str, &[u8]); 2] = [("debian", fixture!("debian.crt")), ("fedora", fixture!("fedora.crt"))];
  const ORIGINS: [(&str, &[u8]); 4] = [
    ("index.crates.io", fixture!("index.crates.io.crt")),
    ("static.crates.io", fixture!("static.crates.io.crt")),
    ("static.rust-lang.org", fixture!("static.rust-lang.org.crt")),
    ("github.com", fixture!("github.com.crt")),
  ];
  // rustls's WebPKI verifier with each backend's algorithms, and the prepared verifier.
  const VERIFIERS: [&str; 3] = ["rscrypto", "rscrypto-prepared", "aws-lc-rs"];
  // The chains' capture time; see fixtures/server-verifier/README.md.
  const NOW: UnixTime = UnixTime::since_unix_epoch(Duration::from_secs(1_792_638_752));

  fn build(verifier: &str, roots: &Arc<RootCertStore>) -> Box<dyn ServerVerifier> {
    match verifier {
      "rscrypto-prepared" => Box::new(PreparedServerVerifier::new(Arc::clone(roots)).expect("bundle roots")),
      "rscrypto" => Box::new(
        WebPkiServerVerifier::builder(Arc::clone(roots), &rscrypto_rustls::DEFAULT_PROVIDER)
          .build()
          .expect("bundle roots"),
      ),
      _ => Box::new(
        WebPkiServerVerifier::builder(Arc::clone(roots), &rustls_aws_lc_rs::DEFAULT_TLS13_PROVIDER)
          .build()
          .expect("bundle roots"),
      ),
    }
  }

  fn verify(verifier: &dyn ServerVerifier, origin: &(Identity<'static>, ServerName<'static>)) {
    let (identity, name) = origin;
    black_box(verifier.verify_identity(&ServerIdentity::new(identity, name, NOW))).expect("origin chain");
  }

  let origins = ORIGINS.map(|(name, pem)| {
    let mut chain = CertificateDer::pem_slice_iter(pem).map(|certificate| certificate.expect("PEM fixture"));
    let leaf = chain.next().expect("leaf");
    let identity = Identity::X509(CertificateIdentity::new(leaf, chain.collect()));
    (name, (identity, ServerName::try_from(name).expect("origin name")))
  });
  for (bundle, pem) in BUNDLES {
    let name = format!("tls13/server-verifier/{bundle}");
    if !bench_config::selected(&name) {
      continue;
    }
    // Bundle parsing is the same for every verifier, so it stays outside timing.
    let mut roots = RootCertStore::empty();
    let (_, unparsed) =
      roots.add_parsable_certificates(CertificateDer::pem_slice_iter(pem).map(|root| root.expect("PEM fixture")));
    assert_eq!(unparsed, 0, "{bundle} roots parse");
    let roots = Arc::new(roots);
    let check = |verifier: &str| {
      let built = build(verifier, &roots);
      for (_, origin) in &origins {
        verify(built.as_ref(), origin);
      }
      built
    };

    let group = format!("{name}/build");
    if bench_config::selected(&group) {
      let mut measured = c.benchmark_group(&group);
      for verifier in VERIFIERS {
        if bench_config::selected(&format!("{group}/{verifier}")) {
          measured.bench_function(
            verifier,
            checked_routine(
              (),
              |()| drop(check(verifier)),
              |b, ()| b.iter(|| black_box(build(verifier, &roots))),
            ),
          );
        }
      }
      measured.finish();
    }
    for (origin_name, origin) in &origins {
      let group = format!("{name}/verify/{origin_name}");
      if !bench_config::selected(&group) {
        continue;
      }
      let mut measured = c.benchmark_group(&group);
      for verifier in VERIFIERS {
        if bench_config::selected(&format!("{group}/{verifier}")) {
          measured.bench_function(
            verifier,
            checked_routine(
              None,
              |built| *built = Some(check(verifier)),
              |b, built| {
                let built = built.as_deref().expect("the check builds the verifier");
                b.iter(|| verify(built, origin));
              },
            ),
          );
        }
      }
      measured.finish();
    }
    // A client lifetime: build, then verify one origin's chain per handshake.
    for (origin_name, origin) in origins
      .iter()
      .filter(|(name, _)| ["index.crates.io", "github.com"].contains(name))
    {
      for handshakes in [1, 3, 10] {
        let group = format!("{name}/lifetime/{origin_name}/{handshakes}");
        if !bench_config::selected(&group) {
          continue;
        }
        let mut measured = c.benchmark_group(&group);
        for verifier in &VERIFIERS[..2] {
          if bench_config::selected(&format!("{group}/{verifier}")) {
            measured.bench_function(
              *verifier,
              checked_routine(
                (),
                |()| drop(check(verifier)),
                |b, ()| {
                  b.iter(|| {
                    let built = build(verifier, &roots);
                    for _ in 0..handshakes {
                      verify(built.as_ref(), origin);
                    }
                    built
                  });
                },
              ),
            );
          }
        }
        measured.finish();
      }
    }
  }
}

fn record_gather(c: &mut Criterion) {
  use rustls::crypto::cipher::{AeadKey, EncodableVersion, Iv, OutboundPlain, Record, Tls13AeadAlgorithm};
  use rustls::enums::{ContentType, ProtocolVersion};

  let suites = [
    (
      "AES128-GCM-SHA256",
      rscrypto_rustls::cipher_suite::TLS13_AES_128_GCM_SHA256,
    ),
    (
      "AES256-GCM-SHA384",
      rscrypto_rustls::cipher_suite::TLS13_AES_256_GCM_SHA384,
    ),
    (
      "CHACHA20-POLY1305-SHA256",
      rscrypto_rustls::cipher_suite::TLS13_CHACHA20_POLY1305_SHA256,
    ),
  ];
  for (name, suite) in suites {
    let peer_algorithm = rustls_aws_lc_rs::DEFAULT_TLS13_PROVIDER
      .tls13_cipher_suites
      .iter()
      .find(|peer| peer.common.suite == suite.common.suite)
      .expect("matching AWS-LC suite")
      .aead_alg;
    let ring_algorithm = rustls_ring::DEFAULT_TLS13_PROVIDER
      .tls13_cipher_suites
      .iter()
      .find(|peer| peer.common.suite == suite.common.suite)
      .expect("matching ring suite")
      .aead_alg;
    let algorithms: [(&str, &dyn Tls13AeadAlgorithm); 3] = [
      ("rscrypto", suite.aead_alg),
      ("aws-lc-rs", peer_algorithm),
      ("ring", ring_algorithm),
    ];
    for count in [1usize, 4, 64] {
      let group = format!("tls13/record-gather/{name}/16384B/{count}-chunks");
      if !bench_config::selected(&group) {
        continue;
      }
      let plaintext: Vec<_> = (0usize..16_384).map(|i| i.to_le_bytes()[0]).collect();
      // Prepare genuinely separate input regions outside timing. Guards prevent
      // treating logical neighbors as one physical plaintext range.
      let storage: Vec<Vec<u8>> = plaintext
        .chunks(16_384usize.strict_div(count))
        .map(|chunk| {
          let mut fragment = Vec::with_capacity(chunk.len().strict_add(2));
          fragment.push(0xa5);
          fragment.extend_from_slice(chunk);
          fragment.push(0x5a);
          fragment
        })
        .collect();
      let chunks: Vec<_> = storage
        .iter()
        .map(|chunk| &chunk[1..chunk.len().strict_sub(1)])
        .collect();
      let message = || Record {
        typ: ContentType::ApplicationData,
        version: EncodableVersion::Legacy(ProtocolVersion::TLSv1_3),
        payload: OutboundPlain::new(&chunks),
      };
      let key = |algorithm: &dyn Tls13AeadAlgorithm| {
        if algorithm.key_len() == 16 {
          AeadKey::from([0x19; 16])
        } else {
          AeadKey::from([0x19; 32])
        }
      };
      let mut measured = c.benchmark_group(&group);
      measured.throughput(Throughput::Bytes(16_384));
      for (backend, algorithm) in algorithms {
        if !bench_config::selected(&format!("{group}/{backend}")) {
          continue;
        }
        measured.bench_function(
          backend,
          checked_routine(
            None,
            |state| {
              let mut encrypter = algorithm.encrypter(key(algorithm), Iv::from([0x71; 12]));
              let mut peer = peer_algorithm.encrypter(key(peer_algorithm), Iv::from([0x71; 12]));
              let mut actual = Vec::with_capacity(16_406);
              let mut expected = Vec::with_capacity(16_406);
              encrypter.encrypt(message(), 0, &mut actual).expect("record encryption");
              peer
                .encrypt(message(), 0, &mut expected)
                .expect("independent record encryption");
              assert_eq!(actual, expected, "{backend} record from {count} chunks");
              actual.clear();
              *state = Some((encrypter, actual, 1u64));
            },
            |b, state| {
              let (encrypter, out, sequence) = state.as_mut().expect("checked record fixture");
              b.iter(|| {
                out.clear();
                encrypter
                  .encrypt(message(), *sequence, out)
                  .expect("bounded record encryption");
                *sequence = sequence.strict_add(1);
                black_box(out.as_slice());
              });
            },
          ),
        );
      }
      measured.finish();
    }
  }
}

/// One TLS 1.3 application-data record sealed, or opened, through each
/// provider's record adapter, without connection or transport overhead.
/// Sealing appends the framed record to reserved output. Opening first copies
/// the sealed body into the record buffer, the same for every provider, because
/// it decrypts in place.
fn records(c: &mut Criterion) {
  use rustls::crypto::cipher::{
    AeadKey, EncodableVersion, InboundOpaque, Iv, OutboundPlain, Record, Tls13AeadAlgorithm,
  };
  use rustls::enums::{ContentType, ProtocolVersion};

  let suites = [
    (
      "AES128-GCM-SHA256",
      rscrypto_rustls::cipher_suite::TLS13_AES_128_GCM_SHA256,
    ),
    (
      "AES256-GCM-SHA384",
      rscrypto_rustls::cipher_suite::TLS13_AES_256_GCM_SHA384,
    ),
    (
      "CHACHA20-POLY1305-SHA256",
      rscrypto_rustls::cipher_suite::TLS13_CHACHA20_POLY1305_SHA256,
    ),
  ];
  for (name, suite) in suites {
    let find = |provider: &rustls::crypto::CryptoProvider| {
      provider
        .tls13_cipher_suites
        .iter()
        .find(|peer| peer.common.suite == suite.common.suite)
        .expect("matching suite")
        .aead_alg
    };
    let algorithms: [(&str, &dyn Tls13AeadAlgorithm); 3] = [
      ("rscrypto", suite.aead_alg),
      ("aws-lc-rs", find(&rustls_aws_lc_rs::DEFAULT_TLS13_PROVIDER)),
      ("ring", find(&rustls_ring::DEFAULT_TLS13_PROVIDER)),
    ];
    let peer = algorithms[1].1;
    let key = |algorithm: &dyn Tls13AeadAlgorithm| {
      if algorithm.key_len() == 16 {
        AeadKey::from([0x19; 16])
      } else {
        AeadKey::from([0x19; 32])
      }
    };
    for size in [64usize, 256, 1024, 16_384] {
      let plaintext: Vec<_> = (0..size).map(|i| i.to_le_bytes()[0]).collect();
      let chunks = [plaintext.as_slice()];
      let message = || Record {
        typ: ContentType::ApplicationData,
        version: EncodableVersion::Legacy(ProtocolVersion::TLSv1_3),
        payload: OutboundPlain::new(&chunks),
      };
      let mut sealed = Vec::new();
      peer
        .encrypter(key(peer), Iv::from([0x71; 12]))
        .encrypt(message(), 0, &mut sealed)
        .expect("independent record encryption");
      let body = sealed[5..].to_vec();

      let group = format!("tls13/record-seal/{name}/{size}");
      if bench_config::selected(&group) {
        let mut measured = c.benchmark_group(&group);
        measured.throughput(Throughput::Bytes(u64::try_from(size).expect("record size")));
        for (backend, algorithm) in algorithms {
          if !bench_config::selected(&format!("{group}/{backend}")) {
            continue;
          }
          measured.bench_function(
            backend,
            checked_routine(
              None,
              |state| {
                let mut encrypter = algorithm.encrypter(key(algorithm), Iv::from([0x71; 12]));
                let mut out = Vec::with_capacity(sealed.len());
                encrypter.encrypt(message(), 0, &mut out).expect("record encryption");
                assert_eq!(out, sealed, "{backend} sealed record");
                out.clear();
                *state = Some((encrypter, out, 1u64));
              },
              |b, state| {
                // AWS-LC derives its nonce mask from sequence 0 and rejects any
                // sequence that does not increase.
                let (encrypter, out, sequence) = state.as_mut().expect("checked record fixture");
                b.iter(|| {
                  out.clear();
                  encrypter.encrypt(message(), *sequence, out).expect("record encryption");
                  *sequence = sequence.strict_add(1);
                  black_box(out.as_slice());
                });
              },
            ),
          );
        }
        measured.finish();
      }

      let group = format!("tls13/record-open/{name}/{size}");
      if bench_config::selected(&group) {
        let mut measured = c.benchmark_group(&group);
        measured.throughput(Throughput::Bytes(u64::try_from(size).expect("record size")));
        for (backend, algorithm) in algorithms {
          if !bench_config::selected(&format!("{group}/{backend}")) {
            continue;
          }
          measured.bench_function(
            backend,
            checked_routine(
              None,
              |state| {
                let mut decrypter = algorithm.decrypter(key(algorithm), Iv::from([0x71; 12]));
                let mut buffer = body.clone();
                let opened = decrypter
                  .decrypt(
                    Record::new(
                      ContentType::ApplicationData,
                      EncodableVersion::Legacy(ProtocolVersion::TLSv1_2),
                      InboundOpaque(&mut buffer),
                    ),
                    0,
                  )
                  .expect("record decryption");
                assert_eq!(opened.payload, plaintext.as_slice(), "{backend} opened record");
                *state = Some((decrypter, buffer));
              },
              |b, state| {
                let (decrypter, buffer) = state.as_mut().expect("checked record fixture");
                b.iter(|| {
                  buffer.copy_from_slice(&body);
                  let opened = decrypter
                    .decrypt(
                      Record::new(
                        ContentType::ApplicationData,
                        EncodableVersion::Legacy(ProtocolVersion::TLSv1_2),
                        InboundOpaque(buffer),
                      ),
                      0,
                    )
                    .expect("record decryption");
                  black_box(opened.payload);
                });
              },
            ),
          );
        }
        measured.finish();
      }
    }
  }
}

fn main() {
  // Record the comparison libraries' identity in the run's output; discovery
  // and case matching keep their exact output.
  let first = std::env::args().nth(1);
  if !std::env::args().any(|arg| arg == "--list") && first.as_deref() != Some("--rscrypto-filter-cases") {
    eprintln!(
      "{}",
      serde_json::json!({"provider_bench_provenance": rscrypto_provider_bench::provenance()})
    );
  }
  bench_config::run(&[connections, expanded, traffic, server_verifiers, record_gather, records]);
}
