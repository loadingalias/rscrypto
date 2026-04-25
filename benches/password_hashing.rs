//! Argon2 / scrypt password-hashing benchmarks.
//!
//! Differential against the `argon2` and `scrypt` (RustCrypto) crates.
//! Organised by cost-parameter classes so CI can run the fast groups on
//! every push and reserve the OWASP-scale group for dedicated perf runs.

#![allow(clippy::unwrap_used)]

use core::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rscrypto::{Argon2Error, Argon2Params, Argon2Version, Argon2d, Argon2i, Argon2id, Scrypt, ScryptParams};

/// Function pointer shared by all three `Argon2{d,i,id}::hash` variants.
type ArgonHashFn = fn(&Argon2Params, &[u8], &[u8], &mut [u8]) -> Result<(), Argon2Error>;

const PASSWORD: &[u8] = b"correct horse battery staple";
const SALT: &[u8] = b"rscrypto-bench-salt-16bytes!";

/// Build rscrypto params.
fn rs_params(m_kib: u32, t: u32, p: u32, out_len: u32) -> Argon2Params {
  Argon2Params::new()
    .memory_cost_kib(m_kib)
    .time_cost(t)
    .parallelism(p)
    .output_len(out_len)
    .version(Argon2Version::V0x13)
    .build()
    .unwrap()
}

/// Build RustCrypto oracle context.
fn oracle_ctx(algo: argon2::Algorithm, m_kib: u32, t: u32, p: u32, out_len: usize) -> argon2::Argon2<'static> {
  let params = argon2::Params::new(m_kib, t, p, Some(out_len)).unwrap();
  argon2::Argon2::new(algo, argon2::Version::V0x13, params)
}

// ─── Small-parameter matrix (CI-friendly) ───────────────────────────────────

/// Tiny / fast parameter matrix used for all three variants. Every tuple
/// completes in a few milliseconds so the bench run stays CI-friendly.
const SMALL_MATRIX: &[(u32, u32, u32)] = &[
  // (m KiB, t, p)
  (8, 1, 1),
  (32, 2, 1),
  (64, 3, 2),
];

fn bench_small_variant(c: &mut Criterion, group_name: &str, rs_hash: ArgonHashFn, oracle_algo: argon2::Algorithm) {
  let mut g = c.benchmark_group(group_name);
  g.sample_size(30);

  for &(m, t, p) in SMALL_MATRIX {
    let out_len = 32usize;
    let param_id = format!("m={m}_t={t}_p={p}");
    let rs_params = rs_params(m, t, p, out_len as u32);
    let oracle = oracle_ctx(oracle_algo, m, t, p, out_len);

    g.bench_with_input(BenchmarkId::new("rscrypto", &param_id), &rs_params, |b, params| {
      let mut out = [0u8; 32];
      b.iter(|| {
        rs_hash(
          black_box(params),
          black_box(PASSWORD),
          black_box(SALT),
          black_box(&mut out),
        )
        .unwrap()
      });
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto", &param_id), &oracle, |b, ctx| {
      let mut out = [0u8; 32];
      b.iter(|| {
        ctx
          .hash_password_into(black_box(PASSWORD), black_box(SALT), black_box(&mut out))
          .unwrap();
      });
    });
  }
  g.finish();
}

fn argon2d_small(c: &mut Criterion) {
  bench_small_variant(c, "argon2d-small", Argon2d::hash, argon2::Algorithm::Argon2d);
}

fn argon2i_small(c: &mut Criterion) {
  bench_small_variant(c, "argon2i-small", Argon2i::hash, argon2::Algorithm::Argon2i);
}

fn argon2id_small(c: &mut Criterion) {
  bench_small_variant(c, "argon2id-small", Argon2id::hash, argon2::Algorithm::Argon2id);
}

// ─── OWASP-scale parameter matrix ───────────────────────────────────────────

/// OWASP 2024 recommended password-hashing parameters (m=19 MiB, t=2, p=1).
/// These are slow — Criterion runs a reduced sample size so the group
/// completes in reasonable time for real perf measurement.
fn argon2id_owasp(c: &mut Criterion) {
  let mut g = c.benchmark_group("argon2id-owasp");
  g.sample_size(10);
  g.measurement_time(Duration::from_secs(30));

  let out_len = 32usize;

  // OWASP 2024: m=19MiB, t=2, p=1
  let rs_params = rs_params(19 * 1024, 2, 1, out_len as u32);
  let oracle = oracle_ctx(argon2::Algorithm::Argon2id, 19 * 1024, 2, 1, out_len);

  g.bench_function(BenchmarkId::new("rscrypto", "m=19MiB_t=2_p=1"), |b| {
    let mut out = [0u8; 32];
    b.iter(|| {
      Argon2id::hash(
        black_box(&rs_params),
        black_box(PASSWORD),
        black_box(SALT),
        black_box(&mut out),
      )
      .unwrap()
    });
  });

  g.bench_function(BenchmarkId::new("rustcrypto", "m=19MiB_t=2_p=1"), |b| {
    let mut out = [0u8; 32];
    b.iter(|| {
      oracle
        .hash_password_into(black_box(PASSWORD), black_box(SALT), black_box(&mut out))
        .unwrap();
    });
  });

  g.finish();
}

// ─── scrypt parameter matrix ───────────────────────────────────────────────

/// Build rscrypto scrypt params.
fn rs_scrypt_params(log_n: u8, r: u32, p: u32, out_len: u32) -> ScryptParams {
  ScryptParams::new()
    .log_n(log_n)
    .r(r)
    .p(p)
    .output_len(out_len)
    .build()
    .unwrap()
}

/// Build RustCrypto scrypt oracle params.
fn oracle_scrypt_params(log_n: u8, r: u32, p: u32, out_len: usize) -> scrypt::Params {
  scrypt::Params::new(log_n, r, p, out_len).unwrap()
}

/// Small / CI-friendly scrypt matrix: (log_n, r, p).
const SCRYPT_SMALL_MATRIX: &[(u8, u32, u32)] = &[(10, 8, 1), (12, 8, 1), (14, 8, 1), (10, 8, 4)];

fn scrypt_small(c: &mut Criterion) {
  let mut g = c.benchmark_group("scrypt-small");
  g.sample_size(15);

  for &(log_n, r, p) in SCRYPT_SMALL_MATRIX {
    let out_len = 32usize;
    let id = format!("log_n={log_n}_r={r}_p={p}");
    let rs = rs_scrypt_params(log_n, r, p, out_len as u32);
    let oracle = oracle_scrypt_params(log_n, r, p, out_len);

    g.bench_with_input(BenchmarkId::new("rscrypto", &id), &rs, |b, params| {
      let mut out = [0u8; 32];
      b.iter(|| {
        Scrypt::hash(
          black_box(params),
          black_box(PASSWORD),
          black_box(SALT),
          black_box(&mut out),
        )
        .unwrap();
      });
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto", &id), &oracle, |b, params| {
      let mut out = [0u8; 32];
      b.iter(|| {
        scrypt::scrypt(black_box(PASSWORD), black_box(SALT), params, black_box(&mut out)).unwrap();
      });
    });
  }
  g.finish();
}

/// OWASP 2024 scrypt shape (log_n = 17 → N = 131072). Long-running — uses
/// the same reduced sample size as `argon2id-owasp`.
fn scrypt_owasp(c: &mut Criterion) {
  let mut g = c.benchmark_group("scrypt-owasp");
  g.sample_size(10);
  g.measurement_time(Duration::from_secs(30));

  let out_len = 32usize;
  let rs = rs_scrypt_params(17, 8, 1, out_len as u32);
  let oracle = oracle_scrypt_params(17, 8, 1, out_len);

  g.bench_function(BenchmarkId::new("rscrypto", "log_n=17_r=8_p=1"), |b| {
    let mut out = [0u8; 32];
    b.iter(|| {
      Scrypt::hash(
        black_box(&rs),
        black_box(PASSWORD),
        black_box(SALT),
        black_box(&mut out),
      )
      .unwrap();
    });
  });

  g.bench_function(BenchmarkId::new("rustcrypto", "log_n=17_r=8_p=1"), |b| {
    let mut out = [0u8; 32];
    b.iter(|| {
      scrypt::scrypt(black_box(PASSWORD), black_box(SALT), &oracle, black_box(&mut out)).unwrap();
    });
  });

  g.finish();
}

#[cfg(feature = "phc-strings")]
fn scrypt_phc_roundtrip(c: &mut Criterion) {
  let mut g = c.benchmark_group("scrypt-phc-roundtrip");
  g.sample_size(20);

  let params = rs_scrypt_params(10, 8, 1, 32);
  g.bench_function("hash_string_with_salt", |b| {
    b.iter(|| Scrypt::hash_string_with_salt(black_box(&params), black_box(PASSWORD), black_box(SALT)).unwrap());
  });

  let encoded = Scrypt::hash_string_with_salt(&params, PASSWORD, SALT).unwrap();
  g.bench_function("verify_string", |b| {
    b.iter(|| Scrypt::verify_string(black_box(PASSWORD), black_box(&encoded)).unwrap());
  });

  g.finish();
}

#[cfg(not(feature = "phc-strings"))]
fn scrypt_phc_roundtrip(_c: &mut Criterion) {
  // Stub when the PHC feature is disabled.
}

// ─── PHC string round-trip ─────────────────────────────────────────────────

#[cfg(feature = "phc-strings")]
fn argon2id_phc_roundtrip(c: &mut Criterion) {
  let mut g = c.benchmark_group("argon2id-phc-roundtrip");
  g.sample_size(30);

  let params = rs_params(32, 2, 1, 32);
  g.bench_function("hash_string_with_salt", |b| {
    b.iter(|| Argon2id::hash_string_with_salt(black_box(&params), black_box(PASSWORD), black_box(SALT)).unwrap());
  });

  let encoded = Argon2id::hash_string_with_salt(&params, PASSWORD, SALT).unwrap();
  g.bench_function("verify_string", |b| {
    b.iter(|| Argon2id::verify_string(black_box(PASSWORD), black_box(&encoded)).unwrap());
  });

  g.finish();
}

#[cfg(not(feature = "phc-strings"))]
fn argon2id_phc_roundtrip(_c: &mut Criterion) {
  // Stub when the PHC feature is disabled.
}

// ─── Argon2 lane parallelism scaling (Phase 3) ─────────────────────────────

/// CI-friendly lane parallelism scaling curve. With `parallel` enabled,
/// `p > 1` triggers the `rayon::scope`-driven slice driver; `p == 1`
/// takes the fast-path that skips rayon entirely. Same memory cost
/// across every `p`, so the per-iteration WORK is identical (Argon2's
/// matrix size depends on `m`, not `p`); only the scheduling differs.
///
/// At 4 MiB the rayon task-dispatch overhead per slice is a measurable
/// fraction of the per-lane compute, which caps the observed efficiency
/// around 50 % at `p=4`. The overhead-to-work ratio improves with
/// larger matrices — see [`argon2id_parallel_owasp`] for the realistic
/// deployment-scale scaling curve.
#[cfg(feature = "parallel")]
fn argon2id_parallel_scaling(c: &mut Criterion) {
  let mut g = c.benchmark_group("argon2id-parallel");
  g.sample_size(15);
  g.measurement_time(Duration::from_secs(15));

  let m_kib = 4 * 1024; // 4 MiB — CI-friendly; per-iteration time stays in the low-ms range at every `p`.
  let t = 2u32;
  let out_len = 32usize;

  for &p in &[1u32, 4, 8, 16] {
    let id = format!("p={p}");
    let params = rs_params(m_kib, t, p, out_len as u32);

    g.bench_with_input(BenchmarkId::new("rscrypto", &id), &params, |b, params| {
      let mut out = [0u8; 32];
      b.iter(|| {
        Argon2id::hash(
          black_box(params),
          black_box(PASSWORD),
          black_box(SALT),
          black_box(&mut out),
        )
        .unwrap();
      });
    });
  }

  g.finish();
}

/// OWASP-scale lane parallelism scaling curve. Memory budget large
/// enough that the rayon task-dispatch overhead fades into noise and
/// the measurement reflects true compute parallelism. Sample size and
/// measurement window match [`argon2id_owasp`] so the group stays
/// comparable.
#[cfg(feature = "parallel")]
fn argon2id_parallel_owasp(c: &mut Criterion) {
  let mut g = c.benchmark_group("argon2id-parallel-owasp");
  g.sample_size(10);
  g.measurement_time(Duration::from_secs(30));

  let m_kib = 19 * 1024; // OWASP 2024 recommended memory cost.
  let t = 2u32;
  let out_len = 32usize;

  for &p in &[1u32, 4, 8, 16] {
    let id = format!("p={p}");
    let params = rs_params(m_kib, t, p, out_len as u32);

    g.bench_with_input(BenchmarkId::new("rscrypto", &id), &params, |b, params| {
      let mut out = [0u8; 32];
      b.iter(|| {
        Argon2id::hash(
          black_box(params),
          black_box(PASSWORD),
          black_box(SALT),
          black_box(&mut out),
        )
        .unwrap();
      });
    });
  }

  g.finish();
}

#[cfg(not(feature = "parallel"))]
fn argon2id_parallel_scaling(_c: &mut Criterion) {
  // Stub when the parallel feature is disabled.
}

#[cfg(not(feature = "parallel"))]
fn argon2id_parallel_owasp(_c: &mut Criterion) {
  // Stub when the parallel feature is disabled.
}

// ─── Phase 4: forced-kernel microbench ─────────────────────────────────────

/// Single-block BlaMka compression throughput, per kernel.
///
/// The `diag` feature exposes direct entry points to each shipped kernel
/// (portable + per-arch SIMD). Measuring one 1 KiB compression in
/// isolation cleanly separates kernel throughput from H'/H0/matrix-alloc
/// overhead, which is the right signal for the Phase 4 gate ("kernel
/// beats portable by a measurable amount").
#[cfg(feature = "diag")]
fn argon2_kernel_raw(c: &mut Criterion) {
  use rscrypto::auth::argon2::{self, DIAG_BLOCK_WORDS};

  let mut g = c.benchmark_group("argon2-kernel-raw");
  g.sample_size(40);
  g.measurement_time(Duration::from_secs(6));

  // Stable pseudo-random inputs; representative of mid-hash block contents.
  let x: [u64; DIAG_BLOCK_WORDS] =
    core::array::from_fn(|i| (i as u64).wrapping_mul(0x5a5a_a5a5_c3c3_3c3c) ^ (i as u64).rotate_left(13));
  let y: [u64; DIAG_BLOCK_WORDS] =
    core::array::from_fn(|i| (i as u64).wrapping_mul(0xdead_beef_feed_face) ^ (i as u64).rotate_right(7));

  g.bench_function("portable", |b| {
    let mut dst = [0u64; DIAG_BLOCK_WORDS];
    b.iter(|| {
      argon2::diag_compress_portable(black_box(&mut dst), black_box(&x), black_box(&y), black_box(false));
      black_box(&dst);
    });
  });

  #[cfg(target_arch = "aarch64")]
  g.bench_function("aarch64-neon", |b| {
    let mut dst = [0u64; DIAG_BLOCK_WORDS];
    b.iter(|| {
      argon2::diag_compress_aarch64_neon(black_box(&mut dst), black_box(&x), black_box(&y), black_box(false));
      black_box(&dst);
    });
  });

  #[cfg(target_arch = "x86_64")]
  {
    let host = rscrypto::platform::caps();

    if host.has(rscrypto::auth::argon2::required_caps(argon2::KernelId::X86Avx2)) {
      g.bench_function("x86-avx2", |b| {
        let mut dst = [0u64; DIAG_BLOCK_WORDS];
        b.iter(|| {
          argon2::diag_compress_x86_avx2(black_box(&mut dst), black_box(&x), black_box(&y), black_box(false));
          black_box(&dst);
        });
      });
    }

    if host.has(rscrypto::auth::argon2::required_caps(argon2::KernelId::X86Avx512)) {
      g.bench_function("x86-avx512", |b| {
        let mut dst = [0u64; DIAG_BLOCK_WORDS];
        b.iter(|| {
          argon2::diag_compress_x86_avx512(black_box(&mut dst), black_box(&x), black_box(&y), black_box(false));
          black_box(&dst);
        });
      });
    }
  }

  #[cfg(target_arch = "powerpc64")]
  if rscrypto::platform::caps().has(rscrypto::auth::argon2::required_caps(argon2::KernelId::PowerVsx)) {
    g.bench_function("power-vsx", |b| {
      let mut dst = [0u64; DIAG_BLOCK_WORDS];
      b.iter(|| {
        argon2::diag_compress_power_vsx(black_box(&mut dst), black_box(&x), black_box(&y), black_box(false));
        black_box(&dst);
      });
    });
  }

  #[cfg(target_arch = "s390x")]
  if rscrypto::platform::caps().has(rscrypto::auth::argon2::required_caps(argon2::KernelId::S390xVector)) {
    g.bench_function("s390x-vector", |b| {
      let mut dst = [0u64; DIAG_BLOCK_WORDS];
      b.iter(|| {
        argon2::diag_compress_s390x_vector(black_box(&mut dst), black_box(&x), black_box(&y), black_box(false));
        black_box(&dst);
      });
    });
  }

  #[cfg(target_arch = "riscv64")]
  if rscrypto::platform::caps().has(rscrypto::auth::argon2::required_caps(argon2::KernelId::Riscv64V)) {
    g.bench_function("riscv64-v", |b| {
      let mut dst = [0u64; DIAG_BLOCK_WORDS];
      b.iter(|| {
        argon2::diag_compress_riscv64_v(black_box(&mut dst), black_box(&x), black_box(&y), black_box(false));
        black_box(&dst);
      });
    });
  }

  #[cfg(target_arch = "wasm32")]
  if rscrypto::platform::caps().has(rscrypto::auth::argon2::required_caps(argon2::KernelId::WasmSimd128)) {
    g.bench_function("wasm-simd128", |b| {
      let mut dst = [0u64; DIAG_BLOCK_WORDS];
      b.iter(|| {
        argon2::diag_compress_wasm_simd128(black_box(&mut dst), black_box(&x), black_box(&y), black_box(false));
        black_box(&dst);
      });
    });
  }

  g.finish();
}

#[cfg(not(feature = "diag"))]
fn argon2_kernel_raw(_c: &mut Criterion) {
  // Stub when the diag feature is disabled.
}

/// End-to-end hash latency per kernel at a mid-size matrix.
///
/// Complements `argon2-kernel-raw` (one compress call) with a real hash
/// pipeline (matrix alloc, H0, H', all passes, finalisation). Pinning the
/// kernel explicitly via `diag_hash_*` isolates per-kernel end-to-end
/// behaviour so the Phase 4 gate ("kernel beats portable by a measurable
/// amount") can be evaluated for every kernel, not just the active one.
#[cfg(feature = "diag")]
fn argon2_kernel_end_to_end(c: &mut Criterion) {
  use rscrypto::auth::argon2::{self, Argon2Variant};

  let mut g = c.benchmark_group("argon2id-kernel-end-to-end");
  g.sample_size(20);
  g.measurement_time(Duration::from_secs(8));

  // 256 KiB matrix, t=2 — large enough to amortise alloc/init but short
  // enough for a CI bench window.
  let params = rs_params(256, 2, 1, 32);

  g.bench_function("portable", |b| {
    let mut out = [0u8; 32];
    b.iter(|| {
      argon2::diag_hash_portable(
        black_box(&params),
        black_box(PASSWORD),
        black_box(SALT),
        Argon2Variant::Argon2id,
        black_box(&mut out),
      )
      .unwrap();
    });
  });

  #[cfg(target_arch = "aarch64")]
  g.bench_function("aarch64-neon", |b| {
    let mut out = [0u8; 32];
    b.iter(|| {
      argon2::diag_hash_aarch64_neon(
        black_box(&params),
        black_box(PASSWORD),
        black_box(SALT),
        Argon2Variant::Argon2id,
        black_box(&mut out),
      )
      .unwrap();
    });
  });

  #[cfg(target_arch = "x86_64")]
  {
    let host = rscrypto::platform::caps();

    if host.has(rscrypto::auth::argon2::required_caps(argon2::KernelId::X86Avx2)) {
      g.bench_function("x86-avx2", |b| {
        let mut out = [0u8; 32];
        b.iter(|| {
          argon2::diag_hash_x86_avx2(
            black_box(&params),
            black_box(PASSWORD),
            black_box(SALT),
            Argon2Variant::Argon2id,
            black_box(&mut out),
          )
          .unwrap();
        });
      });
    }

    if host.has(rscrypto::auth::argon2::required_caps(argon2::KernelId::X86Avx512)) {
      g.bench_function("x86-avx512", |b| {
        let mut out = [0u8; 32];
        b.iter(|| {
          argon2::diag_hash_x86_avx512(
            black_box(&params),
            black_box(PASSWORD),
            black_box(SALT),
            Argon2Variant::Argon2id,
            black_box(&mut out),
          )
          .unwrap();
        });
      });
    }
  }

  #[cfg(target_arch = "powerpc64")]
  if rscrypto::platform::caps().has(rscrypto::auth::argon2::required_caps(argon2::KernelId::PowerVsx)) {
    g.bench_function("power-vsx", |b| {
      let mut out = [0u8; 32];
      b.iter(|| {
        argon2::diag_hash_power_vsx(
          black_box(&params),
          black_box(PASSWORD),
          black_box(SALT),
          Argon2Variant::Argon2id,
          black_box(&mut out),
        )
        .unwrap();
      });
    });
  }

  #[cfg(target_arch = "s390x")]
  if rscrypto::platform::caps().has(rscrypto::auth::argon2::required_caps(argon2::KernelId::S390xVector)) {
    g.bench_function("s390x-vector", |b| {
      let mut out = [0u8; 32];
      b.iter(|| {
        argon2::diag_hash_s390x_vector(
          black_box(&params),
          black_box(PASSWORD),
          black_box(SALT),
          Argon2Variant::Argon2id,
          black_box(&mut out),
        )
        .unwrap();
      });
    });
  }

  #[cfg(target_arch = "riscv64")]
  if rscrypto::platform::caps().has(rscrypto::auth::argon2::required_caps(argon2::KernelId::Riscv64V)) {
    g.bench_function("riscv64-v", |b| {
      let mut out = [0u8; 32];
      b.iter(|| {
        argon2::diag_hash_riscv64_v(
          black_box(&params),
          black_box(PASSWORD),
          black_box(SALT),
          Argon2Variant::Argon2id,
          black_box(&mut out),
        )
        .unwrap();
      });
    });
  }

  #[cfg(target_arch = "wasm32")]
  if rscrypto::platform::caps().has(rscrypto::auth::argon2::required_caps(argon2::KernelId::WasmSimd128)) {
    g.bench_function("wasm-simd128", |b| {
      let mut out = [0u8; 32];
      b.iter(|| {
        argon2::diag_hash_wasm_simd128(
          black_box(&params),
          black_box(PASSWORD),
          black_box(SALT),
          Argon2Variant::Argon2id,
          black_box(&mut out),
        )
        .unwrap();
      });
    });
  }

  g.finish();
}

#[cfg(not(feature = "diag"))]
fn argon2_kernel_end_to_end(_c: &mut Criterion) {
  // Stub when the diag feature is disabled.
}

criterion_group!(
  benches,
  argon2_kernel_raw,
  argon2_kernel_end_to_end,
  argon2d_small,
  argon2i_small,
  argon2id_small,
  argon2id_phc_roundtrip,
  argon2id_parallel_scaling,
  scrypt_small,
  scrypt_phc_roundtrip,
  argon2id_owasp,
  argon2id_parallel_owasp,
  scrypt_owasp,
);
criterion_main!(benches);
