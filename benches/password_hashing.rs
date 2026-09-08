//! Argon2 / scrypt password-hashing benchmarks.
//!
//! Differential against the `argon2` and `scrypt` (RustCrypto) crates.
//! All parameter classes use the shared Criterion configuration and remain
//! included in their algorithm selectors.

#[path = "common/criterion.rs"]
mod bench_config;

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use dryoc::{
  classic::crypto_pwhash::{PasswordHashAlgorithm, crypto_pwhash},
  constants::{CRYPTO_PWHASH_ARGON2I_OPSLIMIT_MIN, CRYPTO_PWHASH_ARGON2ID_OPSLIMIT_MIN},
};
use rscrypto::{
  Argon2Error, Argon2Params, Argon2d, Argon2i, Argon2id, Argon2idPassword, Scrypt, ScryptParams, ScryptPassword,
};

/// Function pointer shared by all three raw Argon2 variants.
type ArgonHashFn = fn(&Argon2Params, &[u8], &[u8], &mut [u8]) -> Result<(), Argon2Error>;

const PASSWORD: &[u8] = b"correct horse battery staple";
const SALT: &[u8] = b"rscrypto-bench-salt-16bytes!";
const ARGON2_SALT: &[u8; 16] = b"rscrypto-bench-s";

/// Build rscrypto params.
fn rs_params(m_kib: u32, t: u32, p: u32, _out_len: u32) -> Argon2Params {
  Argon2Params::new(m_kib, t, p).expect("supported password-hashing benchmark parameters must succeed")
}

/// Build RustCrypto oracle context.
fn oracle_ctx(algo: argon2::Algorithm, m_kib: u32, t: u32, p: u32, out_len: usize) -> argon2::Argon2<'static> {
  let params = argon2::Params::new(m_kib, t, p, Some(out_len))
    .expect("supported password-hashing benchmark parameters must succeed");
  argon2::Argon2::new(algo, argon2::Version::V0x13, params)
}

/// Tiny / fast parameter matrix used for all three variants. Every tuple
/// completes in a few milliseconds so the bench run stays bounded.
const SMALL_MATRIX: &[(u32, u32, u32)] = &[
  // (m KiB, t, p)
  (8, 1, 1),
  (32, 2, 1),
  (64, 3, 2),
];

fn dryoc_supports_small_row(algorithm: PasswordHashAlgorithm, time_cost: u32, parallelism: u32) -> bool {
  let minimum_time_cost = match algorithm {
    PasswordHashAlgorithm::Argon2i13 => CRYPTO_PWHASH_ARGON2I_OPSLIMIT_MIN,
    PasswordHashAlgorithm::Argon2id13 => CRYPTO_PWHASH_ARGON2ID_OPSLIMIT_MIN,
  };

  parallelism == 1 && u64::from(time_cost) >= minimum_time_cost
}

fn bench_small_variant(
  c: &mut Criterion,
  group_name: &str,
  rs_hash: ArgonHashFn,
  oracle_algo: argon2::Algorithm,
  dryoc_algo: Option<PasswordHashAlgorithm>,
) {
  if !bench_config::selected(group_name) {
    return;
  }
  let mut g = c.benchmark_group(format!("{group_name}/salt16-raw32"));

  for &(m, t, p) in SMALL_MATRIX {
    let out_len = 32usize;
    let param_id = format!("m={m}_t={t}_p={p}");
    let rs_params = rs_params(
      m,
      t,
      p,
      u32::try_from(out_len).expect("benchmark output length must fit u32"),
    );
    let oracle = oracle_ctx(oracle_algo, m, t, p, out_len);
    let mut expected = [0u8; 32];
    rs_hash(&rs_params, PASSWORD, ARGON2_SALT, &mut expected).expect("Argon2 fixture");
    let mut actual = [0u8; 32];
    oracle
      .hash_password_into(PASSWORD, ARGON2_SALT, &mut actual)
      .expect("RustCrypto Argon2 fixture");
    assert_eq!(actual, expected, "RustCrypto Argon2 comparison: {param_id}");

    g.bench_with_input(BenchmarkId::new("rscrypto", &param_id), &rs_params, |b, params| {
      let mut out = [0u8; 32];
      b.iter(|| {
        rs_hash(
          black_box(params),
          black_box(PASSWORD),
          black_box(ARGON2_SALT),
          black_box(&mut out),
        )
        .expect("supported password-hashing benchmark parameters must succeed")
      });
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto", &param_id), &oracle, |b, ctx| {
      let mut out = [0u8; 32];
      b.iter(|| {
        ctx
          .hash_password_into(black_box(PASSWORD), black_box(ARGON2_SALT), black_box(&mut out))
          .expect("supported password-hashing benchmark parameters must succeed");
      });
    });

    if let Some(algorithm) = dryoc_algo.filter(|&algorithm| dryoc_supports_small_row(algorithm, t, p)) {
      let memlimit_bytes = usize::try_from(m)
        .expect("benchmark memory cost must fit usize")
        .strict_mul(1024);
      crypto_pwhash(
        &mut actual,
        PASSWORD,
        ARGON2_SALT,
        u64::from(t),
        memlimit_bytes,
        algorithm,
      )
      .expect("dryoc Argon2 fixture");
      assert_eq!(actual, expected, "dryoc Argon2 comparison: {param_id}");
      g.bench_with_input(BenchmarkId::new("dryoc", &param_id), &algorithm, |b, algorithm| {
        let mut out = [0u8; 32];
        b.iter(|| {
          crypto_pwhash(
            black_box(&mut out),
            black_box(PASSWORD),
            black_box(ARGON2_SALT),
            u64::from(t),
            memlimit_bytes,
            *algorithm,
          )
          .expect("supported password-hashing benchmark parameters must succeed");
        });
      });
    }
  }
  g.finish();
}

fn argon2d_small(c: &mut Criterion) {
  // dryoc has no Argon2d (libsodium only ships Argon2i and Argon2id).
  bench_small_variant(c, "argon2d-small", Argon2d::derive, argon2::Algorithm::Argon2d, None);
}

fn argon2i_small(c: &mut Criterion) {
  bench_small_variant(
    c,
    "argon2i-small",
    Argon2i::derive,
    argon2::Algorithm::Argon2i,
    Some(PasswordHashAlgorithm::Argon2i13),
  );
}

fn argon2id_small(c: &mut Criterion) {
  bench_small_variant(
    c,
    "argon2id-small",
    Argon2id::derive,
    argon2::Algorithm::Argon2id,
    Some(PasswordHashAlgorithm::Argon2id13),
  );
}

/// OWASP 2024 recommended password-hashing parameters (m=19 MiB, t=2, p=1).
/// Uses the same Criterion settings as every other workload.
fn argon2id_owasp(c: &mut Criterion) {
  if !bench_config::selected("argon2id-owasp/salt16-raw32") {
    return;
  }
  let mut g = c.benchmark_group("argon2id-owasp/salt16-raw32");

  let out_len = 32usize;

  // OWASP 2024: m=19MiB, t=2, p=1
  let rs_params = rs_params(
    19 * 1024,
    2,
    1,
    u32::try_from(out_len).expect("benchmark output length must fit u32"),
  );
  let oracle = oracle_ctx(argon2::Algorithm::Argon2id, 19 * 1024, 2, 1, out_len);
  let mut expected = [0u8; 32];
  Argon2id::derive(&rs_params, PASSWORD, ARGON2_SALT, &mut expected).expect("Argon2 OWASP fixture");
  let mut actual = [0u8; 32];
  oracle
    .hash_password_into(PASSWORD, ARGON2_SALT, &mut actual)
    .expect("RustCrypto Argon2 OWASP fixture");
  assert_eq!(actual, expected, "RustCrypto Argon2 OWASP comparison");

  g.bench_function(BenchmarkId::new("rscrypto", "m=19MiB_t=2_p=1"), |b| {
    let mut out = [0u8; 32];
    b.iter(|| {
      Argon2id::derive(
        black_box(&rs_params),
        black_box(PASSWORD),
        black_box(ARGON2_SALT),
        black_box(&mut out),
      )
      .expect("supported password-hashing benchmark parameters must succeed")
    });
  });

  g.bench_function(BenchmarkId::new("rustcrypto", "m=19MiB_t=2_p=1"), |b| {
    let mut out = [0u8; 32];
    b.iter(|| {
      oracle
        .hash_password_into(black_box(PASSWORD), black_box(ARGON2_SALT), black_box(&mut out))
        .expect("supported password-hashing benchmark parameters must succeed");
    });
  });

  // dryoc / libsodium-classic Argon2id at OWASP parameters (memlimit in bytes).
  let dryoc_memlimit = 19usize.strict_mul(1024).strict_mul(1024);
  crypto_pwhash(
    &mut actual,
    PASSWORD,
    ARGON2_SALT,
    2,
    dryoc_memlimit,
    PasswordHashAlgorithm::Argon2id13,
  )
  .expect("dryoc Argon2 OWASP fixture");
  assert_eq!(actual, expected, "dryoc Argon2 OWASP comparison");
  g.bench_function(BenchmarkId::new("dryoc", "m=19MiB_t=2_p=1"), |b| {
    let mut out = [0u8; 32];
    b.iter(|| {
      crypto_pwhash(
        black_box(&mut out),
        black_box(PASSWORD),
        black_box(ARGON2_SALT),
        2u64,
        dryoc_memlimit,
        PasswordHashAlgorithm::Argon2id13,
      )
      .expect("supported password-hashing benchmark parameters must succeed");
    });
  });

  g.finish();
}

/// Build rscrypto scrypt params.
fn rs_scrypt_params(log_n: u8, r: u32, p: u32, _out_len: u32) -> ScryptParams {
  ScryptParams::new(log_n, r, p).expect("supported password-hashing benchmark parameters must succeed")
}

/// Build RustCrypto scrypt oracle params.
fn oracle_scrypt_params(log_n: u8, r: u32, p: u32, _out_len: usize) -> scrypt::Params {
  scrypt::Params::new(log_n, r, p).expect("supported password-hashing benchmark parameters must succeed")
}

/// Small scrypt matrix: (log_n, r, p).
const SCRYPT_SMALL_MATRIX: &[(u8, u32, u32)] = &[(10, 8, 1), (12, 8, 1), (14, 8, 1), (10, 8, 4)];

fn scrypt_small(c: &mut Criterion) {
  if !bench_config::selected("scrypt-small") {
    return;
  }
  let mut g = c.benchmark_group("scrypt-small");

  for &(log_n, r, p) in SCRYPT_SMALL_MATRIX {
    let out_len = 32usize;
    let id = format!("log_n={log_n}_r={r}_p={p}");
    let rs = rs_scrypt_params(
      log_n,
      r,
      p,
      u32::try_from(out_len).expect("benchmark output length must fit u32"),
    );
    let oracle = oracle_scrypt_params(log_n, r, p, out_len);

    g.bench_with_input(BenchmarkId::new("rscrypto", &id), &rs, |b, params| {
      let mut out = [0u8; 32];
      b.iter(|| {
        Scrypt::derive(
          black_box(params),
          black_box(PASSWORD),
          black_box(SALT),
          black_box(&mut out),
        )
        .expect("supported password-hashing benchmark parameters must succeed");
      });
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto", &id), &oracle, |b, params| {
      let mut out = [0u8; 32];
      b.iter(|| {
        scrypt::scrypt(black_box(PASSWORD), black_box(SALT), params, black_box(&mut out))
          .expect("supported password-hashing benchmark parameters must succeed");
      });
    });
  }
  g.finish();
}

/// OWASP 2024 scrypt shape (log_n = 17 → N = 131072), using the shared settings.
fn scrypt_owasp(c: &mut Criterion) {
  if !bench_config::selected("scrypt-owasp") {
    return;
  }
  let mut g = c.benchmark_group("scrypt-owasp");

  let out_len = 32usize;
  let rs = rs_scrypt_params(
    17,
    8,
    1,
    u32::try_from(out_len).expect("benchmark output length must fit u32"),
  );
  let oracle = oracle_scrypt_params(17, 8, 1, out_len);

  g.bench_function(BenchmarkId::new("rscrypto", "log_n=17_r=8_p=1"), |b| {
    let mut out = [0u8; 32];
    b.iter(|| {
      Scrypt::derive(
        black_box(&rs),
        black_box(PASSWORD),
        black_box(SALT),
        black_box(&mut out),
      )
      .expect("supported password-hashing benchmark parameters must succeed");
    });
  });

  g.bench_function(BenchmarkId::new("rustcrypto", "log_n=17_r=8_p=1"), |b| {
    let mut out = [0u8; 32];
    b.iter(|| {
      scrypt::scrypt(black_box(PASSWORD), black_box(SALT), &oracle, black_box(&mut out))
        .expect("supported password-hashing benchmark parameters must succeed");
    });
  });

  g.finish();
}

fn scrypt_phc_roundtrip(c: &mut Criterion) {
  if !bench_config::selected("scrypt-phc-roundtrip") {
    return;
  }
  let mut g = c.benchmark_group("scrypt-phc-roundtrip");

  let params = rs_scrypt_params(10, 8, 1, 32);
  let password = ScryptPassword::new(params).expect("supported password-hashing benchmark parameters must succeed");
  g.bench_function("hash_password", |b| {
    b.iter(|| {
      password
        .hash_password(black_box(PASSWORD))
        .expect("supported password-hashing benchmark parameters must succeed")
    });
  });

  let encoded = password
    .hash_password(PASSWORD)
    .expect("supported password-hashing benchmark parameters must succeed");
  g.bench_function("verify_password", |b| {
    b.iter(|| {
      password
        .verify_password(black_box(PASSWORD), black_box(&encoded))
        .expect("supported password-hashing benchmark parameters must succeed")
    });
  });

  g.finish();
}

fn argon2id_phc_roundtrip(c: &mut Criterion) {
  if !bench_config::selected("argon2id-phc-roundtrip") {
    return;
  }
  let mut g = c.benchmark_group("argon2id-phc-roundtrip");

  let params = rs_params(32, 2, 1, 32);
  let password = Argon2idPassword::new(params).expect("supported password-hashing benchmark parameters must succeed");
  g.bench_function("hash_password", |b| {
    b.iter(|| {
      password
        .hash_password(black_box(PASSWORD))
        .expect("supported password-hashing benchmark parameters must succeed")
    });
  });

  let encoded = password
    .hash_password(PASSWORD)
    .expect("supported password-hashing benchmark parameters must succeed");
  g.bench_function("verify_password", |b| {
    b.iter(|| {
      password
        .verify_password(black_box(PASSWORD), black_box(&encoded))
        .expect("supported password-hashing benchmark parameters must succeed")
    });
  });

  g.finish();
}

/// Bounded lane-parallel scaling curve. With `parallel` enabled,
/// `p > 1` uses the `rayon::scope` slice driver; `p == 1` skips Rayon.
/// Every row holds total memory and time cost constant while varying the
/// lane count.
#[cfg(feature = "parallel")]
fn argon2id_parallel_scaling(c: &mut Criterion) {
  if !bench_config::selected("argon2id-parallel/salt16-raw32") {
    return;
  }
  let mut g = c.benchmark_group("argon2id-parallel/salt16-raw32");

  let m_kib = 4 * 1024; // 4 MiB; per-iteration time stays in the low-ms range at every `p`.
  let t = 2u32;
  let out_len = 32usize;

  for &p in &[1u32, 4, 8, 16] {
    let id = format!("p={p}");
    let params = rs_params(
      m_kib,
      t,
      p,
      u32::try_from(out_len).expect("benchmark output length must fit u32"),
    );

    g.bench_with_input(BenchmarkId::new("rscrypto", &id), &params, |b, params| {
      let mut out = [0u8; 32];
      b.iter(|| {
        Argon2id::derive(
          black_box(params),
          black_box(PASSWORD),
          black_box(ARGON2_SALT),
          black_box(&mut out),
        )
        .expect("supported password-hashing benchmark parameters must succeed");
      });
    });
  }

  g.finish();
}

/// OWASP-scale lane-parallel scaling curve. Sample size and measurement
/// window match [`argon2id_owasp`] so the raw rows are comparable.
#[cfg(feature = "parallel")]
fn argon2id_parallel_owasp(c: &mut Criterion) {
  if !bench_config::selected("argon2id-parallel-owasp/salt16-raw32") {
    return;
  }
  let mut g = c.benchmark_group("argon2id-parallel-owasp/salt16-raw32");

  let m_kib = 19 * 1024; // OWASP 2024 recommended memory cost.
  let t = 2u32;
  let out_len = 32usize;

  for &p in &[1u32, 4, 8, 16] {
    let id = format!("p={p}");
    let params = rs_params(
      m_kib,
      t,
      p,
      u32::try_from(out_len).expect("benchmark output length must fit u32"),
    );

    g.bench_with_input(BenchmarkId::new("rscrypto", &id), &params, |b, params| {
      let mut out = [0u8; 32];
      b.iter(|| {
        Argon2id::derive(
          black_box(params),
          black_box(PASSWORD),
          black_box(ARGON2_SALT),
          black_box(&mut out),
        )
        .expect("supported password-hashing benchmark parameters must succeed");
      });
    });
  }

  g.finish();
}

fn main() {
  bench_config::run(&[
    argon2d_small,
    argon2i_small,
    argon2id_small,
    argon2id_phc_roundtrip,
    #[cfg(feature = "parallel")]
    argon2id_parallel_scaling,
    scrypt_small,
    scrypt_phc_roundtrip,
    argon2id_owasp,
    #[cfg(feature = "parallel")]
    argon2id_parallel_owasp,
    scrypt_owasp,
  ]);
}
