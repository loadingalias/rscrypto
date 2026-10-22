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
          let client_bytes = vec![0x42; size];
          let server_bytes = vec![0xa7; size];
          let transfer = |pair: &mut rscrypto_provider_bench::Pair, output: &mut [u8]| {
            if bidirectional {
              pair.transfer_both(black_box(&client_bytes), black_box(&server_bytes), black_box(output));
            } else {
              pair.transfer(black_box(&client_bytes), black_box(output));
            }
          };
          measured.bench_function(
            backend,
            checked_routine(
              (vec![0; total], None),
              |(output, connection)| {
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
                // Check consecutive operations too: no stale plaintext, sequence or direction mixup.
                for _ in 0..2 {
                  output.fill(0);
                  transfer(&mut pair, output);
                  assert_eq!(client_bytes, output[..size], "authenticated client-to-server transfer");
                  if bidirectional {
                    assert_eq!(server_bytes, output[size..], "authenticated server-to-client transfer");
                  }
                }
                *connection = Some((fixture, pair));
              },
              |b, (output, connection)| {
                let (_fixture, pair) = connection.as_mut().expect("the check opens the connection");
                b.iter(|| {
                  transfer(pair, output);
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
        WebPkiServerVerifier::builder(Arc::clone(roots), &rscrypto_rustls::default_provider())
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
  bench_config::run(&[connections, expanded, traffic, server_verifiers]);
}
