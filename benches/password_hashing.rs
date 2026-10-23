//! Argon2 / scrypt password-hashing benchmarks.
//!
//! Differential against the `argon2` and `scrypt` (RustCrypto) crates.
//! All parameter classes use the shared Criterion configuration and remain
//! included in their algorithm selectors.
//!
//! Before timing, every raw row must return the RustCrypto output for its
//! parameters. Each oracle runs at most once per parameter set, and only when
//! one of its rows runs. Argon2id PHC records must pass dryoc's independent
//! verifier. The scrypt PHC round-trip rows check the encoded profile and
//! rscrypto's accept/reject decisions; the PHC verification groups also check
//! every record with RustCrypto's verifier.

#[path = "common/criterion.rs"]
mod bench_config;

use core::{cell::LazyCell, hint::black_box};

use bench_config::{checked, checked_with};
use criterion::{BenchmarkId, Criterion};
use dryoc::{
  classic::crypto_pwhash::{PasswordHashAlgorithm, crypto_pwhash, crypto_pwhash_str_verify},
  constants::{CRYPTO_PWHASH_ARGON2I_OPSLIMIT_MIN, CRYPTO_PWHASH_ARGON2ID_OPSLIMIT_MIN},
};
use rscrypto::{
  Argon2Block, Argon2Error, Argon2Params, Argon2d, Argon2i, Argon2id, Argon2idPassword, PasswordStatus, Scrypt,
  ScryptBlock, ScryptParams, ScryptPassword,
};

/// Function pointer shared by all three raw Argon2 variants.
type ArgonHashFn = fn(&Argon2Params, &[u8], &[u8], &mut [u8]) -> Result<(), Argon2Error>;

const PASSWORD: &[u8] = b"correct horse battery staple";
const WRONG_PASSWORD: &[u8] = b"correct horse battery stapler";
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

/// RustCrypto Argon2 output for the benchmark password and salt.
fn argon2_expected(oracle: &argon2::Argon2<'_>) -> [u8; 32] {
  let mut out = [0u8; 32];
  oracle
    .hash_password_into(PASSWORD, ARGON2_SALT, &mut out)
    .expect("RustCrypto Argon2 oracle");
  out
}

/// RustCrypto scrypt output for the benchmark password and salt.
fn scrypt_expected(params: &scrypt::Params) -> [u8; 32] {
  let mut out = [0u8; 32];
  scrypt::scrypt(PASSWORD, SALT, params, &mut out).expect("RustCrypto scrypt oracle");
  out
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
    let expected = LazyCell::new(|| argon2_expected(&oracle));

    g.bench_function(
      BenchmarkId::new("rscrypto", &param_id),
      checked_with(
        [0u8; 32],
        |out| {
          rs_hash(
            black_box(&rs_params),
            black_box(PASSWORD),
            black_box(ARGON2_SALT),
            black_box(out),
          )
          .expect("supported password-hashing benchmark parameters must succeed")
        },
        |_, out| assert_eq!(*out, *expected),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rustcrypto", &param_id),
      checked_with(
        [0u8; 32],
        |out| {
          oracle
            .hash_password_into(black_box(PASSWORD), black_box(ARGON2_SALT), black_box(out))
            .expect("supported password-hashing benchmark parameters must succeed");
        },
        |_, out| assert_eq!(*out, *expected),
      ),
    );

    if let Some(algorithm) = dryoc_algo.filter(|&algorithm| dryoc_supports_small_row(algorithm, t, p)) {
      let memlimit_bytes = usize::try_from(m)
        .expect("benchmark memory cost must fit usize")
        .strict_mul(1024);
      g.bench_function(
        BenchmarkId::new("dryoc", &param_id),
        checked_with(
          [0u8; 32],
          |out| {
            crypto_pwhash(
              black_box(out),
              black_box(PASSWORD),
              black_box(ARGON2_SALT),
              u64::from(t),
              memlimit_bytes,
              black_box(algorithm),
            )
            .expect("supported password-hashing benchmark parameters must succeed");
          },
          |_, out| assert_eq!(*out, *expected),
        ),
      );
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
  let expected = LazyCell::new(|| argon2_expected(&oracle));

  g.bench_function(
    BenchmarkId::new("rscrypto", "m=19MiB_t=2_p=1"),
    checked_with(
      [0u8; 32],
      |out| {
        Argon2id::derive(
          black_box(&rs_params),
          black_box(PASSWORD),
          black_box(ARGON2_SALT),
          black_box(out),
        )
        .expect("supported password-hashing benchmark parameters must succeed")
      },
      |_, out| assert_eq!(*out, *expected),
    ),
  );

  g.bench_function(
    BenchmarkId::new("rustcrypto", "m=19MiB_t=2_p=1"),
    checked_with(
      [0u8; 32],
      |out| {
        oracle
          .hash_password_into(black_box(PASSWORD), black_box(ARGON2_SALT), black_box(out))
          .expect("supported password-hashing benchmark parameters must succeed");
      },
      |_, out| assert_eq!(*out, *expected),
    ),
  );

  // dryoc / libsodium-classic Argon2id at OWASP parameters (memlimit in bytes).
  let dryoc_memlimit = 19usize.strict_mul(1024).strict_mul(1024);
  g.bench_function(
    BenchmarkId::new("dryoc", "m=19MiB_t=2_p=1"),
    checked_with(
      [0u8; 32],
      |out| {
        crypto_pwhash(
          black_box(out),
          black_box(PASSWORD),
          black_box(ARGON2_SALT),
          2u64,
          dryoc_memlimit,
          PasswordHashAlgorithm::Argon2id13,
        )
        .expect("supported password-hashing benchmark parameters must succeed");
      },
      |_, out| assert_eq!(*out, *expected),
    ),
  );

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
    let expected = LazyCell::new(|| scrypt_expected(&oracle));

    g.bench_function(
      BenchmarkId::new("rscrypto", &id),
      checked_with(
        [0u8; 32],
        |out| {
          Scrypt::derive(black_box(&rs), black_box(PASSWORD), black_box(SALT), black_box(out))
            .expect("supported password-hashing benchmark parameters must succeed");
        },
        |_, out| assert_eq!(*out, *expected),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rustcrypto", &id),
      checked_with(
        [0u8; 32],
        |out| {
          scrypt::scrypt(black_box(PASSWORD), black_box(SALT), &oracle, black_box(out))
            .expect("supported password-hashing benchmark parameters must succeed");
        },
        |_, out| assert_eq!(*out, *expected),
      ),
    );
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
  let expected = LazyCell::new(|| scrypt_expected(&oracle));

  g.bench_function(
    BenchmarkId::new("rscrypto", "log_n=17_r=8_p=1"),
    checked_with(
      [0u8; 32],
      |out| {
        Scrypt::derive(black_box(&rs), black_box(PASSWORD), black_box(SALT), black_box(out))
          .expect("supported password-hashing benchmark parameters must succeed");
      },
      |_, out| assert_eq!(*out, *expected),
    ),
  );

  g.bench_function(
    BenchmarkId::new("rustcrypto", "log_n=17_r=8_p=1"),
    checked_with(
      [0u8; 32],
      |out| {
        scrypt::scrypt(black_box(PASSWORD), black_box(SALT), &oracle, black_box(out))
          .expect("supported password-hashing benchmark parameters must succeed");
      },
      |_, out| assert_eq!(*out, *expected),
    ),
  );

  g.finish();
}

/// Fresh versus reused work memory at the OWASP shapes; see the comparison
/// contract in `docs/benchmarking.md`.
///
/// `rscrypto/fresh-allocation` calls `derive`, which allocates, zero-fills, and
/// clears its work memory on every call. `reused-memory` rows lend one buffer,
/// allocated and freed outside timing, to every call; rscrypto still clears the
/// blocks it used before returning. The RustCrypto row reuses its own `Block`
/// buffer and, built without its `zeroize` feature, does not clear it. Every
/// row's checked output comes from a call on memory that an earlier call used.
fn reused_memory(c: &mut Criterion) {
  let argon2_group = "argon2id-owasp/salt16-raw32-reused-memory";
  if bench_config::selected(argon2_group) {
    let mut g = c.benchmark_group(argon2_group);
    let params = rs_params(19 * 1024, 2, 1, 32);
    let oracle = oracle_ctx(argon2::Algorithm::Argon2id, 19 * 1024, 2, 1, 32);
    let expected = LazyCell::new(|| argon2_expected(&oracle));
    let blocks = usize::try_from(params.memory_blocks()).expect("benchmark block count must fit usize");

    g.bench_function(
      BenchmarkId::new("rscrypto/fresh-allocation", "m=19MiB_t=2_p=1"),
      checked_with(
        [0u8; 32],
        |out| {
          Argon2id::derive(
            black_box(&params),
            black_box(PASSWORD),
            black_box(ARGON2_SALT),
            black_box(out),
          )
          .expect("supported password-hashing benchmark parameters must succeed")
        },
        |_, out| assert_eq!(*out, *expected),
      ),
    );
    g.bench_function(
      BenchmarkId::new("rscrypto/reused-memory", "m=19MiB_t=2_p=1"),
      checked_with(
        ([0u8; 32], vec![Argon2Block::ZERO; blocks]),
        |(out, memory)| {
          Argon2id::derive_with_memory(
            black_box(&params),
            black_box(PASSWORD),
            black_box(ARGON2_SALT),
            black_box(out),
            black_box(memory),
          )
          .expect("supported password-hashing benchmark parameters must succeed")
        },
        |_, (out, _)| assert_eq!(*out, *expected),
      ),
    );
    g.bench_function(
      BenchmarkId::new("rustcrypto/reused-memory", "m=19MiB_t=2_p=1"),
      checked_with(
        ([0u8; 32], vec![argon2::Block::new(); blocks]),
        |(out, memory)| {
          oracle
            .hash_password_into_with_memory(
              black_box(PASSWORD),
              black_box(ARGON2_SALT),
              black_box(out),
              black_box(memory),
            )
            .expect("supported password-hashing benchmark parameters must succeed")
        },
        |_, (out, _)| assert_eq!(*out, *expected),
      ),
    );
    g.finish();
  }

  let scrypt_group = "scrypt-owasp-reused-memory";
  if bench_config::selected(scrypt_group) {
    let mut g = c.benchmark_group(scrypt_group);
    let params = rs_scrypt_params(17, 8, 1, 32);
    let oracle = oracle_scrypt_params(17, 8, 1, 32);
    let expected = LazyCell::new(|| scrypt_expected(&oracle));
    let blocks = params.memory_blocks().expect("OWASP scrypt shape must fit");

    g.bench_function(
      BenchmarkId::new("rscrypto/fresh-allocation", "log_n=17_r=8_p=1"),
      checked_with(
        [0u8; 32],
        |out| {
          Scrypt::derive(black_box(&params), black_box(PASSWORD), black_box(SALT), black_box(out))
            .expect("supported password-hashing benchmark parameters must succeed")
        },
        |_, out| assert_eq!(*out, *expected),
      ),
    );
    g.bench_function(
      BenchmarkId::new("rscrypto/reused-memory", "log_n=17_r=8_p=1"),
      checked_with(
        ([0u8; 32], vec![ScryptBlock::ZERO; blocks]),
        |(out, memory)| {
          Scrypt::derive_with_memory(
            black_box(&params),
            black_box(PASSWORD),
            black_box(SALT),
            black_box(out),
            black_box(memory),
          )
          .expect("supported password-hashing benchmark parameters must succeed")
        },
        |_, (out, _)| assert_eq!(*out, *expected),
      ),
    );
    g.finish();
  }
}

fn scrypt_phc_roundtrip(c: &mut Criterion) {
  if !bench_config::selected("scrypt-phc-roundtrip") {
    return;
  }
  let mut g = c.benchmark_group("scrypt-phc-roundtrip");

  let params = rs_scrypt_params(10, 8, 1, 32);
  let password = ScryptPassword::new(params).expect("supported password-hashing benchmark parameters must succeed");
  g.bench_function(
    "hash_password",
    checked(
      || {
        password
          .hash_password(black_box(PASSWORD))
          .expect("supported password-hashing benchmark parameters must succeed")
      },
      |encoded| {
        assert!(
          encoded.starts_with("$scrypt$ln=10,r=8,p=1$"),
          "scrypt PHC profile: {encoded}"
        );
        let status = password
          .verify_password(PASSWORD, &encoded)
          .expect("scrypt PHC record must verify");
        assert_eq!(status, PasswordStatus::Current);
        password
          .verify_password(WRONG_PASSWORD, &encoded)
          .expect_err("scrypt PHC record must reject a wrong password");
      },
    ),
  );

  let encoded = password
    .hash_password(PASSWORD)
    .expect("supported password-hashing benchmark parameters must succeed");
  g.bench_function(
    "verify_password",
    checked(
      || {
        password
          .verify_password(black_box(PASSWORD), black_box(&encoded))
          .expect("supported password-hashing benchmark parameters must succeed")
      },
      |status| {
        assert_eq!(status, PasswordStatus::Current);
        password
          .verify_password(WRONG_PASSWORD, &encoded)
          .expect_err("scrypt PHC verification must reject a wrong password");
      },
    ),
  );

  g.finish();
}

fn argon2id_phc_roundtrip(c: &mut Criterion) {
  if !bench_config::selected("argon2id-phc-roundtrip") {
    return;
  }
  let mut g = c.benchmark_group("argon2id-phc-roundtrip");

  let params = rs_params(32, 2, 1, 32);
  let password = Argon2idPassword::new(params).expect("supported password-hashing benchmark parameters must succeed");
  g.bench_function(
    "hash_password",
    checked(
      || {
        password
          .hash_password(black_box(PASSWORD))
          .expect("supported password-hashing benchmark parameters must succeed")
      },
      |encoded| {
        assert!(
          encoded.starts_with("$argon2id$v=19$m=32,t=2,p=1$"),
          "Argon2id PHC profile: {encoded}"
        );
        crypto_pwhash_str_verify(&encoded, PASSWORD).expect("dryoc must verify the Argon2id PHC record");
        crypto_pwhash_str_verify(&encoded, WRONG_PASSWORD)
          .expect_err("dryoc must reject a wrong password for the Argon2id PHC record");
      },
    ),
  );

  let encoded = password
    .hash_password(PASSWORD)
    .expect("supported password-hashing benchmark parameters must succeed");
  g.bench_function(
    "verify_password",
    checked(
      || {
        password
          .verify_password(black_box(PASSWORD), black_box(&encoded))
          .expect("supported password-hashing benchmark parameters must succeed")
      },
      |status| {
        crypto_pwhash_str_verify(&encoded, PASSWORD).expect("dryoc must verify the Argon2id PHC record");
        assert_eq!(status, PasswordStatus::Current);
        password
          .verify_password(WRONG_PASSWORD, &encoded)
          .expect_err("Argon2id PHC verification must reject a wrong password");
      },
    ),
  );

  g.finish();
}

/// PHC password verification with fresh versus caller-owned work memory.
///
/// Task: verify one password against one stored PHC record. Timed: record parsing and limit approval,
/// derivation, the constant-time comparison, and work-memory handling. The record is created once
/// outside timing.
///
/// - `rscrypto/fresh-allocation` calls `verify_password`, which allocates, zero-fills, and clears its
///   work memory in each call.
/// - `rscrypto/reused-memory` calls `verify_password_with_memory` with one buffer allocated and freed
///   outside timing; rscrypto still clears the blocks it used before returning.
/// - `rustcrypto/fresh-allocation` calls `PasswordVerifier::verify_password` on the same PHC string,
///   parsed in each call. RustCrypto allocates its work memory in each call and, built without its
///   `zeroize` feature, does not clear it.
/// - `dryoc/fresh-allocation` (Argon2id only) calls `crypto_pwhash_str_verify`, which parses the record
///   and allocates its work memory in each call. dryoc has no scrypt PHC verifier.
/// - No external verifier here takes caller-owned work memory, so the reused-memory rows have no
///   external peer.
///
/// Argon2id uses the OWASP shape of `argon2id_owasp`. scrypt uses `log_n=14` (16 MiB) from the small
/// matrix rather than the 128 MiB OWASP shape, to keep the slowest runner within one run budget.
/// Before timing, every row must accept the password from a call that reuses its state, and reject a
/// wrong one. Each record must also pass RustCrypto's independent verifier, and Argon2id records dryoc's.
fn phc_verify(c: &mut Criterion) {
  let argon2_group = "argon2id-phc-verify";
  if bench_config::selected(argon2_group) {
    let mut g = c.benchmark_group(argon2_group);
    let params = rs_params(19 * 1024, 2, 1, 32);
    let password = Argon2idPassword::new(params).expect("supported password-hashing benchmark parameters must succeed");
    let encoded = password
      .hash_password(PASSWORD)
      .expect("supported password-hashing benchmark parameters must succeed");
    assert!(
      encoded.starts_with("$argon2id$v=19$m=19456,t=2,p=1$"),
      "Argon2id PHC profile: {encoded}"
    );
    crypto_pwhash_str_verify(&encoded, PASSWORD).expect("dryoc must verify the Argon2id PHC record");
    crypto_pwhash_str_verify(&encoded, WRONG_PASSWORD)
      .expect_err("dryoc must reject a wrong password for the Argon2id PHC record");
    let oracle = argon2::Argon2::default();
    let rustcrypto_verify = |password: &[u8], encoded: &str| {
      <argon2::Argon2<'_> as argon2::PasswordVerifier<str>>::verify_password(&oracle, password, encoded)
    };
    rustcrypto_verify(PASSWORD, &encoded).expect("RustCrypto must verify the Argon2id PHC record");
    let blocks = usize::try_from(params.memory_blocks()).expect("benchmark block count must fit usize");

    g.bench_function(
      BenchmarkId::new("rscrypto/fresh-allocation", "m=19MiB_t=2_p=1"),
      checked(
        || {
          password
            .verify_password(black_box(PASSWORD), black_box(&encoded))
            .expect("Argon2id PHC record must verify")
        },
        |status| {
          assert_eq!(status, PasswordStatus::Current);
          password
            .verify_password(WRONG_PASSWORD, &encoded)
            .expect_err("Argon2id PHC verification must reject a wrong password");
        },
      ),
    );
    g.bench_function(
      BenchmarkId::new("rscrypto/reused-memory", "m=19MiB_t=2_p=1"),
      checked_with(
        vec![Argon2Block::ZERO; blocks],
        |memory| {
          password
            .verify_password_with_memory(black_box(PASSWORD), black_box(&encoded), black_box(memory))
            .expect("Argon2id PHC record must verify")
        },
        |status, _| {
          assert_eq!(status, PasswordStatus::Current);
          let mut memory = vec![Argon2Block::ZERO; blocks];
          password
            .verify_password_with_memory(WRONG_PASSWORD, &encoded, &mut memory)
            .expect_err("Argon2id PHC verification must reject a wrong password");
        },
      ),
    );
    g.bench_function(
      BenchmarkId::new("rustcrypto/fresh-allocation", "m=19MiB_t=2_p=1"),
      checked(
        || rustcrypto_verify(black_box(PASSWORD), black_box(&encoded)),
        |result| {
          result.expect("RustCrypto must verify the Argon2id PHC record");
          rustcrypto_verify(WRONG_PASSWORD, &encoded).expect_err("RustCrypto must reject a wrong password");
        },
      ),
    );
    g.bench_function(
      BenchmarkId::new("dryoc/fresh-allocation", "m=19MiB_t=2_p=1"),
      checked(
        || crypto_pwhash_str_verify(black_box(&encoded), black_box(PASSWORD)),
        |result| {
          result.expect("dryoc must verify the Argon2id PHC record");
          crypto_pwhash_str_verify(&encoded, WRONG_PASSWORD).expect_err("dryoc must reject a wrong password");
        },
      ),
    );
    g.finish();
  }

  let scrypt_group = "scrypt-phc-verify";
  if bench_config::selected(scrypt_group) {
    let mut g = c.benchmark_group(scrypt_group);
    let params = rs_scrypt_params(14, 8, 1, 32);
    let password = ScryptPassword::new(params).expect("supported password-hashing benchmark parameters must succeed");
    let encoded = password
      .hash_password(PASSWORD)
      .expect("supported password-hashing benchmark parameters must succeed");
    assert!(
      encoded.starts_with("$scrypt$ln=14,r=8,p=1$"),
      "scrypt PHC profile: {encoded}"
    );
    let oracle = scrypt::Scrypt::new();
    // Parsing and verification report different error types; either failure rejects the record.
    let rustcrypto_verify = |password: &[u8], encoded: &str| {
      scrypt::phc::PasswordHash::new(encoded).is_ok_and(|hash| {
        <scrypt::Scrypt as scrypt::password_hash::PasswordVerifier<_>>::verify_password(&oracle, password, &hash)
          .is_ok()
      })
    };
    assert!(
      rustcrypto_verify(PASSWORD, &encoded),
      "RustCrypto must verify the scrypt PHC record"
    );
    assert!(
      !rustcrypto_verify(WRONG_PASSWORD, &encoded),
      "RustCrypto must reject a wrong scrypt password"
    );
    let blocks = params.memory_blocks().expect("benchmark scrypt shape must fit");

    g.bench_function(
      BenchmarkId::new("rscrypto/fresh-allocation", "log_n=14_r=8_p=1"),
      checked(
        || {
          password
            .verify_password(black_box(PASSWORD), black_box(&encoded))
            .expect("scrypt PHC record must verify")
        },
        |status| {
          assert_eq!(status, PasswordStatus::Current);
          password
            .verify_password(WRONG_PASSWORD, &encoded)
            .expect_err("scrypt PHC verification must reject a wrong password");
        },
      ),
    );
    g.bench_function(
      BenchmarkId::new("rscrypto/reused-memory", "log_n=14_r=8_p=1"),
      checked_with(
        vec![ScryptBlock::ZERO; blocks],
        |memory| {
          password
            .verify_password_with_memory(black_box(PASSWORD), black_box(&encoded), black_box(memory))
            .expect("scrypt PHC record must verify")
        },
        |status, _| {
          assert_eq!(status, PasswordStatus::Current);
          let mut memory = vec![ScryptBlock::ZERO; blocks];
          password
            .verify_password_with_memory(WRONG_PASSWORD, &encoded, &mut memory)
            .expect_err("scrypt PHC verification must reject a wrong password");
        },
      ),
    );
    g.bench_function(
      BenchmarkId::new("rustcrypto/fresh-allocation", "log_n=14_r=8_p=1"),
      checked(
        || rustcrypto_verify(black_box(PASSWORD), black_box(&encoded)),
        |verified| {
          assert!(verified, "RustCrypto must verify the scrypt PHC record");
          assert!(
            !rustcrypto_verify(WRONG_PASSWORD, &encoded),
            "RustCrypto must reject a wrong password"
          );
        },
      ),
    );
    g.finish();
  }
}

/// Bounded lane-parallel scaling curve. With `parallel` enabled,
/// `p > 1` uses the `rayon::scope` slice driver; `p == 1` skips Rayon.
/// Every row holds total memory and time cost constant while varying the
/// lane count. RustCrypto computes each lane count's expected output serially.
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
    let oracle = oracle_ctx(argon2::Algorithm::Argon2id, m_kib, t, p, out_len);

    g.bench_function(
      BenchmarkId::new("rscrypto", &id),
      checked_with(
        [0u8; 32],
        |out| {
          Argon2id::derive(
            black_box(&params),
            black_box(PASSWORD),
            black_box(ARGON2_SALT),
            black_box(out),
          )
          .expect("supported password-hashing benchmark parameters must succeed");
        },
        |_, out| assert_eq!(*out, argon2_expected(&oracle)),
      ),
    );
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
    let oracle = oracle_ctx(argon2::Algorithm::Argon2id, m_kib, t, p, out_len);

    g.bench_function(
      BenchmarkId::new("rscrypto", &id),
      checked_with(
        [0u8; 32],
        |out| {
          Argon2id::derive(
            black_box(&params),
            black_box(PASSWORD),
            black_box(ARGON2_SALT),
            black_box(out),
          )
          .expect("supported password-hashing benchmark parameters must succeed");
        },
        |_, out| assert_eq!(*out, argon2_expected(&oracle)),
      ),
    );
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
    reused_memory,
    phc_verify,
  ]);
}
