//! PBKDF2-HMAC-SHA2 benchmarks for rscrypto public APIs.
//!
//! Before timing, every derivation row must write RustCrypto's derived key, and the RustCrypto row must write ring's.
//! Every verification row must accept RustCrypto's derived key and reject it with one changed byte.

#[path = "common/criterion.rs"]
#[expect(
  dead_code,
  reason = "every row writes into a reused buffer or also checks a rejection"
)]
mod bench_config;

use core::{hint::black_box, num::NonZeroU32};

use bench_config::checked_with;
use criterion::{BenchmarkId, Criterion, Throughput};
use rscrypto::{Pbkdf2Sha256, Pbkdf2Sha512};

#[cfg(all(
  any(unix, windows),
  not(target_arch = "wasm32"),
  not(any(target_arch = "s390x", target_arch = "powerpc64"))
))]
macro_rules! aws_lc_bench {
  ($($tokens:tt)*) => {
    $($tokens)*
  };
}

#[cfg(not(all(
  any(unix, windows),
  not(target_arch = "wasm32"),
  not(any(target_arch = "s390x", target_arch = "powerpc64"))
)))]
macro_rules! aws_lc_bench {
  ($($tokens:tt)*) => {};
}

/// Iteration counts below the password policy, for the primitive API: one keying-dominated count and two
/// iteration-dominated counts.
const PRIMITIVE_ITERATIONS: [u32; 3] = [1, 100, 1000];

#[cfg(all(rscrypto_internal, feature = "diag"))]
fn print_pbkdf2_diag_once() {
  use std::sync::Once;

  static ONCE: Once = Once::new();
  ONCE.call_once(|| {
    use rscrypto::{Sha256, hashes::introspect::kernel_for};

    eprintln!("rscrypto-diag pbkdf2 runtime_caps={}", rscrypto::platform::caps());
    eprintln!("rscrypto-diag pbkdf2 static_caps={}", rscrypto::platform::caps_static());
    eprintln!(
      "rscrypto-diag pbkdf2 target_features sha={} sha512={} avx2={} avx512f={}",
      cfg!(target_feature = "sha"),
      cfg!(target_feature = "sha512"),
      cfg!(target_feature = "avx2"),
      cfg!(target_feature = "avx512f")
    );
    eprintln!(
      "rscrypto-diag pbkdf2 sha256_kernel 64={} 4096={} 1048576={}",
      kernel_for::<Sha256>(64),
      kernel_for::<Sha256>(4096),
      kernel_for::<Sha256>(1_048_576)
    );
  });
}

#[cfg(not(all(rscrypto_internal, feature = "diag")))]
#[inline]
fn print_pbkdf2_diag_once() {}

/// Returns a verification row. `verify` is the row's timed path: it must accept `expected`, and outside timing it
/// must reject `expected` with its last byte changed.
fn verify_row<'a>(
  verify: impl Fn(&[u8]) -> bool + 'a,
  expected: &'a [u8],
) -> impl FnMut(&mut criterion::Bencher<'_>) + 'a {
  let mut changed = expected.to_vec();
  *changed.last_mut().expect("a derived key is not empty") ^= 1;
  checked_with(
    verify,
    move |verify| {
      assert!(
        verify(black_box(expected)),
        "valid PBKDF2 benchmark verification must succeed"
      );
    },
    move |(), verify| assert!(!verify(&changed), "a changed derived key must fail"),
  )
}

macro_rules! pbkdf2_benches {
  (
    derive: $derive:ident,
    policy: $policy:ident,
    group: $group:literal,
    rscrypto: $rscrypto:ty,
    rustcrypto: $rustcrypto:ty,
    aws_lc_algorithm: $aws_algorithm:path,
    ring_algorithm: $ring_algorithm:path,
    password: $password:expr,
    salt: $salt:expr,
    output_lengths: $output_lengths:expr $(,)?
  ) => {
    // PBKDF2 one-shot derivation into a reused output buffer, at iteration counts below the password policy.
    // Timed: password keying, every iteration, and output. The rscrypto row calls the primitive API, because the
    // policy API rejects these counts. The `-state` group is rscrypto-only and reuses a keyed password state built
    // outside timing.
    fn $derive(c: &mut Criterion) {
      if !bench_config::selected($group) {
        return;
      }
      print_pbkdf2_diag_once();

      let password = $password;
      let salt = $salt;
      let state = <$rscrypto>::new(&password);
      let rustcrypto = |iterations: u32, len: usize| {
        let mut out = vec![0u8; len];
        pbkdf2::pbkdf2_hmac::<$rustcrypto>(&password, &salt, iterations, &mut out);
        out
      };
      let ring_expected = |iterations: NonZeroU32, len: usize| {
        let mut out = vec![0u8; len];
        ring::pbkdf2::derive($ring_algorithm, iterations, &salt, &password, &mut out);
        out
      };

      for iterations in PRIMITIVE_ITERATIONS {
        let nz_iters = NonZeroU32::new(iterations).expect("benchmark iteration counts are nonzero");
        let mut g = c.benchmark_group(format!("{}/iters={iterations}", $group));

        for out_len in $output_lengths {
          g.throughput(Throughput::Bytes(out_len as u64));
          let expected = rustcrypto(iterations, out_len);

          g.bench_function(
            BenchmarkId::new("rscrypto", out_len),
            checked_with(
              vec![0u8; out_len],
              |out| {
                <$rscrypto>::derive_key_primitive(
                  black_box(&password),
                  black_box(&salt),
                  iterations,
                  black_box(out.as_mut_slice()),
                )
                .expect("valid PBKDF2 benchmark derivation must succeed");
                black_box(out[0])
              },
              |_, out| assert_eq!(out, &expected),
            ),
          );

          g.bench_function(
            BenchmarkId::new("rustcrypto", out_len),
            checked_with(
              vec![0u8; out_len],
              |out| {
                pbkdf2::pbkdf2_hmac::<$rustcrypto>(
                  black_box(&password),
                  black_box(&salt),
                  iterations,
                  black_box(out.as_mut_slice()),
                );
                black_box(out[0])
              },
              |_, out| assert_eq!(out, &ring_expected(nz_iters, out_len)),
            ),
          );

          aws_lc_bench! {
            g.bench_function(
              BenchmarkId::new("aws-lc-rs", out_len),
              checked_with(
                vec![0u8; out_len],
                |out| {
                  aws_lc_rs::pbkdf2::derive(
                    $aws_algorithm,
                    nz_iters,
                    black_box(&salt),
                    black_box(&password),
                    black_box(out.as_mut_slice()),
                  );
                  black_box(out[0])
                },
                |_, out| assert_eq!(out, &expected),
              ),
            );
          }

          g.bench_function(
            BenchmarkId::new("ring", out_len),
            checked_with(
              vec![0u8; out_len],
              |out| {
                ring::pbkdf2::derive(
                  $ring_algorithm,
                  nz_iters,
                  black_box(&salt),
                  black_box(&password),
                  black_box(out.as_mut_slice()),
                );
                black_box(out[0])
              },
              |_, out| assert_eq!(out, &expected),
            ),
          );
        }

        g.finish();

        let mut g_state = c.benchmark_group(format!("{}-state/iters={iterations}", $group));
        for out_len in $output_lengths {
          g_state.throughput(Throughput::Bytes(out_len as u64));
          let expected = rustcrypto(iterations, out_len);
          g_state.bench_function(
            BenchmarkId::new("rscrypto", out_len),
            checked_with(
              vec![0u8; out_len],
              |out| {
                state
                  .derive(black_box(&salt), iterations, black_box(out.as_mut_slice()))
                  .expect("valid PBKDF2 benchmark derivation must succeed");
                black_box(out[0])
              },
              |_, out| assert_eq!(out, &expected),
            ),
          );
        }
        g_state.finish();
      }
    }

    // PBKDF2 at the minimum iteration count of rscrypto's default password policy, with one output block: the
    // stored-password shape. Each call takes tens of milliseconds to seconds, so the groups are gated separately.
    // - Derivation writes into a reused output buffer. Timed: password keying, every iteration, and output. The
    //   rscrypto row calls the policy API `derive_key`, which also checks the iteration count and salt length.
    // - Verification derives the key and compares it with the stored key. Timed: keying, every iteration, and the
    //   comparison. `verify_password` also checks the policy and bounds the work before it derives. ring and aws-lc-rs
    //   check neither. RustCrypto's `pbkdf2` has no raw-key verification API, so it is absent.
    fn $policy(c: &mut Criterion) {
      let iterations = <$rscrypto>::MIN_RECOMMENDED_ITERATIONS;
      let derive_group = format!("{}/iters={iterations}", $group);
      let verify_group = format!("{}/verify/iters={iterations}", $group);
      let derive_selected = bench_config::selected(&format!("{derive_group}/"));
      let verify_selected = bench_config::selected(&format!("{verify_group}/"));
      if !derive_selected && !verify_selected {
        return;
      }
      print_pbkdf2_diag_once();

      let password = $password;
      let salt = $salt;
      let nz_iters = NonZeroU32::new(iterations).expect("the policy iteration count is nonzero");
      let out_len = <$rscrypto>::OUTPUT_SIZE;
      let mut expected = vec![0u8; out_len];
      pbkdf2::pbkdf2_hmac::<$rustcrypto>(&password, &salt, iterations, &mut expected);

      if derive_selected {
        let mut ring_expected = vec![0u8; out_len];
        ring::pbkdf2::derive($ring_algorithm, nz_iters, &salt, &password, &mut ring_expected);
        let mut g = c.benchmark_group(derive_group);
        g.throughput(Throughput::Bytes(out_len as u64));

        g.bench_function(
          BenchmarkId::new("rscrypto", out_len),
          checked_with(
            vec![0u8; out_len],
            |out| {
              <$rscrypto>::derive_key(
                black_box(&password),
                black_box(&salt),
                iterations,
                black_box(out.as_mut_slice()),
              )
              .expect("valid PBKDF2 benchmark derivation must succeed");
              black_box(out[0])
            },
            |_, out| assert_eq!(out, &expected),
          ),
        );

        g.bench_function(
          BenchmarkId::new("rustcrypto", out_len),
          checked_with(
            vec![0u8; out_len],
            |out| {
              pbkdf2::pbkdf2_hmac::<$rustcrypto>(
                black_box(&password),
                black_box(&salt),
                iterations,
                black_box(out.as_mut_slice()),
              );
              black_box(out[0])
            },
            |_, out| assert_eq!(out, &ring_expected),
          ),
        );

        aws_lc_bench! {
          g.bench_function(
            BenchmarkId::new("aws-lc-rs", out_len),
            checked_with(
              vec![0u8; out_len],
              |out| {
                aws_lc_rs::pbkdf2::derive(
                  $aws_algorithm,
                  nz_iters,
                  black_box(&salt),
                  black_box(&password),
                  black_box(out.as_mut_slice()),
                );
                black_box(out[0])
              },
              |_, out| assert_eq!(out, &expected),
            ),
          );
        }

        g.bench_function(
          BenchmarkId::new("ring", out_len),
          checked_with(
            vec![0u8; out_len],
            |out| {
              ring::pbkdf2::derive(
                $ring_algorithm,
                nz_iters,
                black_box(&salt),
                black_box(&password),
                black_box(out.as_mut_slice()),
              );
              black_box(out[0])
            },
            |_, out| assert_eq!(out, &expected),
          ),
        );

        g.finish();
      }

      if verify_selected {
        let mut g = c.benchmark_group(verify_group);

        g.bench_function(
          BenchmarkId::new("rscrypto", out_len),
          verify_row(
            |stored| <$rscrypto>::verify_password(black_box(&password), black_box(&salt), iterations, stored).is_ok(),
            &expected,
          ),
        );

        aws_lc_bench! {
          g.bench_function(
            BenchmarkId::new("aws-lc-rs", out_len),
            verify_row(
              |stored| {
                aws_lc_rs::pbkdf2::verify($aws_algorithm, nz_iters, black_box(&salt), black_box(&password), stored)
                  .is_ok()
              },
              &expected,
            ),
          );
        }

        g.bench_function(
          BenchmarkId::new("ring", out_len),
          verify_row(
            |stored| {
              ring::pbkdf2::verify(
                $ring_algorithm,
                nz_iters,
                black_box(&salt),
                black_box(&password),
                stored,
              )
              .is_ok()
            },
            &expected,
          ),
        );

        g.finish();
      }
    }
  };
}

pbkdf2_benches!(
  derive: pbkdf2_sha256_derive,
  policy: pbkdf2_sha256_policy,
  group: "pbkdf2-sha256",
  rscrypto: Pbkdf2Sha256,
  rustcrypto: sha2::Sha256,
  aws_lc_algorithm: aws_lc_rs::pbkdf2::PBKDF2_HMAC_SHA256,
  ring_algorithm: ring::pbkdf2::PBKDF2_HMAC_SHA256,
  password: [0x55u8; 32],
  salt: [0x33u8; 16],
  output_lengths: [32usize, 64],
);

pbkdf2_benches!(
  derive: pbkdf2_sha512_derive,
  policy: pbkdf2_sha512_policy,
  group: "pbkdf2-sha512",
  rscrypto: Pbkdf2Sha512,
  rustcrypto: sha2::Sha512,
  aws_lc_algorithm: aws_lc_rs::pbkdf2::PBKDF2_HMAC_SHA512,
  ring_algorithm: ring::pbkdf2::PBKDF2_HMAC_SHA512,
  password: [0x66u8; 48],
  salt: [0x44u8; 16],
  output_lengths: [64usize, 128],
);

fn pbkdf2_sha256_internal(c: &mut Criterion) {
  if !bench_config::selected("pbkdf2-sha256/internal/") {
    return;
  }
  print_pbkdf2_diag_once();

  let password = [0x55u8; 32];
  let salt = [0x33u8; 16];
  let state = Pbkdf2Sha256::new(&password);
  // Every row must write RustCrypto's derived key into its reused output buffer; the RustCrypto row must write
  // ring's.
  let rustcrypto = |iterations: u32, len: usize| {
    let mut out = vec![0u8; len];
    pbkdf2::pbkdf2_hmac::<sha2::Sha256>(&password, &salt, iterations, &mut out);
    out
  };
  let ring_expected = |iterations: NonZeroU32, len: usize| {
    let mut out = vec![0u8; len];
    ring::pbkdf2::derive(ring::pbkdf2::PBKDF2_HMAC_SHA256, iterations, &salt, &password, &mut out);
    out
  };

  for iterations in PRIMITIVE_ITERATIONS {
    let nz_iters = NonZeroU32::new(iterations).expect("benchmark iteration counts are nonzero");
    let mut g = c.benchmark_group(format!("pbkdf2-sha256/internal/iters={iterations}"));

    for out_len in [32usize, 64] {
      g.throughput(Throughput::Bytes(out_len as u64));
      let expected = rustcrypto(iterations, out_len);

      g.bench_function(
        BenchmarkId::new("rscrypto-oneshot", out_len),
        checked_with(
          vec![0u8; out_len],
          |out| {
            Pbkdf2Sha256::derive_key_primitive(
              black_box(&password),
              black_box(&salt),
              iterations,
              black_box(out.as_mut_slice()),
            )
            .expect("valid PBKDF2 benchmark derivation must succeed");
            black_box(out[0])
          },
          |_, out| assert_eq!(out, &expected),
        ),
      );

      g.bench_function(
        BenchmarkId::new("rscrypto-state", out_len),
        checked_with(
          vec![0u8; out_len],
          |out| {
            state
              .derive(black_box(&salt), iterations, black_box(out.as_mut_slice()))
              .expect("valid PBKDF2 benchmark derivation must succeed");
            black_box(out[0])
          },
          |_, out| assert_eq!(out, &expected),
        ),
      );

      g.bench_function(
        BenchmarkId::new("rustcrypto", out_len),
        checked_with(
          vec![0u8; out_len],
          |out| {
            pbkdf2::pbkdf2_hmac::<sha2::Sha256>(
              black_box(&password),
              black_box(&salt),
              iterations,
              black_box(out.as_mut_slice()),
            );
            black_box(out[0])
          },
          |_, out| assert_eq!(out, &ring_expected(nz_iters, out_len)),
        ),
      );

      aws_lc_bench! {
        g.bench_function(
          BenchmarkId::new("aws-lc-rs", out_len),
          checked_with(
            vec![0u8; out_len],
            |out| {
              aws_lc_rs::pbkdf2::derive(
                aws_lc_rs::pbkdf2::PBKDF2_HMAC_SHA256,
                nz_iters,
                black_box(&salt),
                black_box(&password),
                black_box(out.as_mut_slice()),
              );
              black_box(out[0])
            },
            |_, out| assert_eq!(out, &expected),
          ),
        );
      }

      g.bench_function(
        BenchmarkId::new("ring", out_len),
        checked_with(
          vec![0u8; out_len],
          |out| {
            ring::pbkdf2::derive(
              ring::pbkdf2::PBKDF2_HMAC_SHA256,
              nz_iters,
              black_box(&salt),
              black_box(&password),
              black_box(out.as_mut_slice()),
            );
            black_box(out[0])
          },
          |_, out| assert_eq!(out, &expected),
        ),
      );
    }

    g.finish();
  }
}

fn main() {
  bench_config::run(&[
    pbkdf2_sha256_derive,
    pbkdf2_sha256_policy,
    pbkdf2_sha256_internal,
    pbkdf2_sha512_derive,
    pbkdf2_sha512_policy,
  ]);
}
