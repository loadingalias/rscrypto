//! Blake2 comparison benchmarks: rscrypto vs RustCrypto blake2 crate.
//!
//! Before timing, every row must return RustCrypto blake2's output for its input.

#[path = "common/criterion.rs"]
mod bench_config;

mod common;

use core::hint::black_box;

use bench_config::{checked, checked_with};
use blake2::{
  Blake2b as RustCryptoBlake2b, Blake2b512 as RustCryptoBlake2b512, Blake2bMac, Blake2s as RustCryptoBlake2s,
  Blake2s256 as RustCryptoBlake2s256, Blake2sMac,
  digest::{
    Digest as _, KeyInit as _, Mac as _,
    consts::{U16, U32, U64},
  },
};
use criterion::{BenchmarkId, Criterion};
use rscrypto::{
  Blake2b256, Blake2b512, Blake2bKey, Blake2bParams, Blake2s128, Blake2s256, Blake2sKey, Blake2sParams, Digest,
};

type RustCryptoBlake2bMac256 = Blake2bMac<U32>;
type RustCryptoBlake2bMac512 = Blake2bMac<U64>;
type RustCryptoBlake2sMac128 = Blake2sMac<U16>;
type RustCryptoBlake2sMac256 = Blake2sMac<U32>;
type RustCryptoBlake2b256 = RustCryptoBlake2b<U32>;
type RustCryptoBlake2s128 = RustCryptoBlake2s<U16>;

/// One byte past one BLAKE2s block (64 bytes) and one BLAKE2b block (128 bytes): the shortest messages that need a
/// second compression. The exact 128-byte block is in `short-oneshot` and `short-keyed`.
const BLOCK_BOUNDARIES: [usize; 2] = [65, 129];

fn oneshot(c: &mut Criterion) {
  if !bench_config::selected("blake2") {
    return;
  }
  use dryoc::classic::crypto_generichash::crypto_generichash;

  let inputs = common::comp_sizes(&BLOCK_BOUNDARIES);
  // Timed: complete one-shot hashing and output destruction.
  // Untimed: input/key fixtures and any typed-key validation.
  let mut g = c.benchmark_group("blake2");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let b256 = RustCryptoBlake2b256::digest(data);
    let b512 = RustCryptoBlake2b512::digest(data);
    let s128 = RustCryptoBlake2s128::digest(data);
    let s256 = RustCryptoBlake2s256::digest(data);

    g.bench_function(
      BenchmarkId::new("rscrypto/blake2b256", len),
      checked(
        || Blake2b256::digest(black_box(data)),
        |out| assert_eq!(out.as_slice(), b256.as_slice()),
      ),
    );
    g.bench_function(
      BenchmarkId::new("rustcrypto/blake2b256", len),
      checked(
        || RustCryptoBlake2b256::digest(black_box(data)),
        |out| assert_eq!(out, b256),
      ),
    );
    g.bench_function(
      BenchmarkId::new("dryoc/blake2b256", len),
      checked_with(
        [0u8; 32],
        |out| {
          crypto_generichash(black_box(&mut *out), black_box(data), None)
            .expect("valid BLAKE2 benchmark operation must succeed");
          black_box(*out)
        },
        |out, _| assert_eq!(out.as_slice(), b256.as_slice()),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rscrypto/blake2b512", len),
      checked(
        || Blake2b512::digest(black_box(data)),
        |out| assert_eq!(out.as_slice(), b512.as_slice()),
      ),
    );
    g.bench_function(
      BenchmarkId::new("rustcrypto/blake2b512", len),
      checked(
        || RustCryptoBlake2b512::digest(black_box(data)),
        |out| assert_eq!(out, b512),
      ),
    );
    g.bench_function(
      BenchmarkId::new("dryoc/blake2b512", len),
      checked_with(
        [0u8; 64],
        |out| {
          crypto_generichash(black_box(&mut *out), black_box(data), None)
            .expect("valid BLAKE2 benchmark operation must succeed");
          black_box(*out)
        },
        |out, _| assert_eq!(out.as_slice(), b512.as_slice()),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rscrypto/blake2s128", len),
      checked(
        || Blake2s128::digest(black_box(data)),
        |out| assert_eq!(out.as_slice(), s128.as_slice()),
      ),
    );
    g.bench_function(
      BenchmarkId::new("rustcrypto/blake2s128", len),
      checked(
        || RustCryptoBlake2s128::digest(black_box(data)),
        |out| assert_eq!(out, s128),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rscrypto/blake2s256", len),
      checked(
        || Blake2s256::digest(black_box(data)),
        |out| assert_eq!(out.as_slice(), s256.as_slice()),
      ),
    );
    g.bench_function(
      BenchmarkId::new("rustcrypto/blake2s256", len),
      checked(
        || RustCryptoBlake2s256::digest(black_box(data)),
        |out| assert_eq!(out, s256),
      ),
    );
  }

  g.finish();
}

fn short_messages(c: &mut Criterion) {
  if !bench_config::selected("blake2/short-") && !bench_config::selected("blake2/single-update/") {
    return;
  }
  // The main groups already cover 0, 1, 32, and 64 bytes. Keep only the missing sizes.
  let inputs = [16, 128].map(|len| (len, common::random_bytes(len)));
  let key_b = [0x42u8; 64];
  let key_s = [0x24u8; 32];
  let key_b_typed = Blake2bKey::new(black_box(&key_b[..32])).expect("valid BLAKE2 benchmark operation must succeed");
  let key_s_typed = Blake2sKey::new(black_box(&key_s)).expect("valid BLAKE2 benchmark operation must succeed");
  let rustcrypto_b256_mac = |data: &[u8]| {
    let mut mac = RustCryptoBlake2bMac256::new_from_slice(black_box(&key_b[..32]))
      .expect("valid BLAKE2 benchmark operation must succeed");
    mac.update(black_box(data));
    mac.finalize().into_bytes()
  };
  let rustcrypto_s256_mac = |data: &[u8]| {
    let mut mac = RustCryptoBlake2sMac256::new_from_slice(black_box(&key_s))
      .expect("valid BLAKE2 benchmark operation must succeed");
    mac.update(black_box(data));
    mac.finalize().into_bytes()
  };

  // Timed: the complete one-shot hash and output destruction; input generation is untimed.
  let mut oneshot = c.benchmark_group("blake2/short-oneshot");
  for (len, data) in &inputs {
    common::set_throughput(&mut oneshot, *len);
    let b256 = RustCryptoBlake2b256::digest(data);
    let s256 = RustCryptoBlake2s256::digest(data);

    oneshot.bench_function(
      BenchmarkId::new("rscrypto/blake2b256", len),
      checked(
        || Blake2b256::digest(black_box(data)),
        |out| assert_eq!(out.as_slice(), b256.as_slice()),
      ),
    );
    oneshot.bench_function(
      BenchmarkId::new("rustcrypto/blake2b256", len),
      checked(
        || RustCryptoBlake2b256::digest(black_box(data)),
        |out| assert_eq!(out, b256),
      ),
    );

    oneshot.bench_function(
      BenchmarkId::new("rscrypto/blake2s256", len),
      checked(
        || Blake2s256::digest(black_box(data)),
        |out| assert_eq!(out.as_slice(), s256.as_slice()),
      ),
    );
    oneshot.bench_function(
      BenchmarkId::new("rustcrypto/blake2s256", len),
      checked(
        || RustCryptoBlake2s256::digest(black_box(data)),
        |out| assert_eq!(out, s256),
      ),
    );
  }
  oneshot.finish();

  // Timed: keyed initialization, hashing, finalization, and per-call destruction.
  // Untimed: input/key fixtures and rscrypto typed-key validation.
  let mut keyed = c.benchmark_group("blake2/short-keyed");
  for (len, data) in &inputs {
    common::set_throughput(&mut keyed, *len);
    let b256 = rustcrypto_b256_mac(data);
    let s256 = rustcrypto_s256_mac(data);

    keyed.bench_function(
      BenchmarkId::new("rscrypto/blake2b256", len),
      checked(
        || Blake2b256::keyed_digest(key_b_typed, black_box(data)),
        |out| assert_eq!(out.as_slice(), b256.as_slice()),
      ),
    );
    keyed.bench_function(
      BenchmarkId::new("rustcrypto/blake2b256", len),
      checked(|| rustcrypto_b256_mac(data), |out| assert_eq!(out, b256)),
    );

    keyed.bench_function(
      BenchmarkId::new("rscrypto/blake2s256", len),
      checked(
        || Blake2s256::keyed_digest(key_s_typed, black_box(data)),
        |out| assert_eq!(out.as_slice(), s256.as_slice()),
      ),
    );
    keyed.bench_function(
      BenchmarkId::new("rustcrypto/blake2s256", len),
      checked(|| rustcrypto_s256_mac(data), |out| assert_eq!(out, s256)),
    );
  }
  keyed.finish();

  // Timed: construct, update once (including an empty update), finalize, and drop outputs.
  // Untimed: immutable input fixtures. Multi-chunk streaming is a separate workload.
  let inputs = [0, 1, 16, 32, 64, 128].map(|len| (len, common::random_bytes(len)));
  let mut stream = c.benchmark_group("blake2/single-update");
  for (len, data) in &inputs {
    common::set_throughput(&mut stream, *len);
    let b256 = RustCryptoBlake2b256::digest(data);
    let s256 = RustCryptoBlake2s256::digest(data);

    stream.bench_function(
      BenchmarkId::new("rscrypto/blake2b256", len),
      checked(
        || {
          let mut h = Blake2b256::new();
          h.update(black_box(data));
          h.finalize()
        },
        |out| assert_eq!(out.as_slice(), b256.as_slice()),
      ),
    );
    stream.bench_function(
      BenchmarkId::new("rustcrypto/blake2b256", len),
      checked(
        || {
          let mut h = RustCryptoBlake2b256::new();
          h.update(black_box(data));
          h.finalize()
        },
        |out| assert_eq!(out, b256),
      ),
    );

    stream.bench_function(
      BenchmarkId::new("rscrypto/blake2s256", len),
      checked(
        || {
          let mut h = Blake2s256::new();
          h.update(black_box(data));
          h.finalize()
        },
        |out| assert_eq!(out.as_slice(), s256.as_slice()),
      ),
    );
    stream.bench_function(
      BenchmarkId::new("rustcrypto/blake2s256", len),
      checked(
        || {
          let mut h = RustCryptoBlake2s256::new();
          h.update(black_box(data));
          h.finalize()
        },
        |out| assert_eq!(out, s256),
      ),
    );
  }
  stream.finish();
}

fn keyed(c: &mut Criterion) {
  if !bench_config::selected("blake2/keyed") {
    return;
  }
  use dryoc::classic::crypto_generichash::crypto_generichash;

  let inputs = common::comp_sizes(&BLOCK_BOUNDARIES);
  let key_b = [0x42u8; 64];
  let key_s = [0x24u8; 32];
  let key_b_256 = Blake2bKey::new(black_box(&key_b[..32])).expect("valid BLAKE2 benchmark operation must succeed");
  let key_b_512 = Blake2bKey::new(black_box(&key_b)).expect("valid BLAKE2 benchmark operation must succeed");
  let key_s_128 = Blake2sKey::new(black_box(&key_s[..16])).expect("valid BLAKE2 benchmark operation must succeed");
  let key_s_256 = Blake2sKey::new(black_box(&key_s)).expect("valid BLAKE2 benchmark operation must succeed");
  let rustcrypto_b256_mac = |data: &[u8]| {
    let mut mac = RustCryptoBlake2bMac256::new_from_slice(black_box(&key_b[..32]))
      .expect("valid BLAKE2 benchmark operation must succeed");
    mac.update(black_box(data));
    mac.finalize().into_bytes()
  };
  let rustcrypto_b512_mac = |data: &[u8]| {
    let mut mac = RustCryptoBlake2bMac512::new_from_slice(black_box(&key_b))
      .expect("valid BLAKE2 benchmark operation must succeed");
    mac.update(black_box(data));
    mac.finalize().into_bytes()
  };
  let rustcrypto_s128_mac = |data: &[u8]| {
    let mut mac = RustCryptoBlake2sMac128::new_from_slice(black_box(&key_s[..16]))
      .expect("valid BLAKE2 benchmark operation must succeed");
    mac.update(black_box(data));
    mac.finalize().into_bytes()
  };
  let rustcrypto_s256_mac = |data: &[u8]| {
    let mut mac = RustCryptoBlake2sMac256::new_from_slice(black_box(&key_s))
      .expect("valid BLAKE2 benchmark operation must succeed");
    mac.update(black_box(data));
    mac.finalize().into_bytes()
  };
  // Timed: keyed initialization, hashing, finalization, and per-call destruction.
  // Untimed: input/key fixtures and any typed-key validation.
  let mut g = c.benchmark_group("blake2/keyed");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let b256 = rustcrypto_b256_mac(data);
    let b512 = rustcrypto_b512_mac(data);
    let s128 = rustcrypto_s128_mac(data);
    let s256 = rustcrypto_s256_mac(data);

    g.bench_function(
      BenchmarkId::new("rscrypto/blake2b256", len),
      checked(
        || Blake2b256::keyed_digest(key_b_256, black_box(data)),
        |out| assert_eq!(out.as_slice(), b256.as_slice()),
      ),
    );
    g.bench_function(
      BenchmarkId::new("rustcrypto/blake2b256", len),
      checked(|| rustcrypto_b256_mac(data), |out| assert_eq!(out, b256)),
    );
    g.bench_function(
      BenchmarkId::new("dryoc/blake2b256", len),
      checked_with(
        [0u8; 32],
        |out| {
          crypto_generichash(black_box(&mut *out), black_box(data), Some(black_box(&key_b[..32])))
            .expect("valid BLAKE2 benchmark operation must succeed");
          black_box(*out)
        },
        |out, _| assert_eq!(out.as_slice(), b256.as_slice()),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rscrypto/blake2b512", len),
      checked(
        || Blake2b512::keyed_digest(key_b_512, black_box(data)),
        |out| assert_eq!(out.as_slice(), b512.as_slice()),
      ),
    );
    g.bench_function(
      BenchmarkId::new("rustcrypto/blake2b512", len),
      checked(|| rustcrypto_b512_mac(data), |out| assert_eq!(out, b512)),
    );
    g.bench_function(
      BenchmarkId::new("dryoc/blake2b512", len),
      checked_with(
        [0u8; 64],
        |out| {
          crypto_generichash(black_box(&mut *out), black_box(data), Some(black_box(&key_b[..])))
            .expect("valid BLAKE2 benchmark operation must succeed");
          black_box(*out)
        },
        |out, _| assert_eq!(out.as_slice(), b512.as_slice()),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rscrypto/blake2s128", len),
      checked(
        || Blake2s128::keyed_digest(key_s_128, black_box(data)),
        |out| assert_eq!(out.as_slice(), s128.as_slice()),
      ),
    );
    g.bench_function(
      BenchmarkId::new("rustcrypto/blake2s128", len),
      checked(|| rustcrypto_s128_mac(data), |out| assert_eq!(out, s128)),
    );

    g.bench_function(
      BenchmarkId::new("rscrypto/blake2s256", len),
      checked(
        || Blake2s256::keyed_digest(key_s_256, black_box(data)),
        |out| assert_eq!(out.as_slice(), s256.as_slice()),
      ),
    );
    g.bench_function(
      BenchmarkId::new("rustcrypto/blake2s256", len),
      checked(|| rustcrypto_s256_mac(data), |out| assert_eq!(out, s256)),
    );
  }

  g.finish();
}

fn streaming(c: &mut Criterion) {
  if !bench_config::selected("blake2/streaming") {
    return;
  }
  use dryoc::classic::crypto_generichash::{
    crypto_generichash_final, crypto_generichash_init, crypto_generichash_update,
  };

  let data = common::random_bytes(1048576);
  let b256 = RustCryptoBlake2b256::digest(&data);
  let s256 = RustCryptoBlake2s256::digest(&data);
  // Timed: hasher construction, all chunk updates, finalization, and per-call destruction.
  // Untimed: input/key fixtures and any typed-key validation.
  let mut g = c.benchmark_group("blake2/streaming");
  g.throughput(criterion::Throughput::Bytes(data.len() as u64));

  for chunk_size in [64, 4096, 65536] {
    g.bench_function(
      format!("rscrypto/blake2b256/{chunk_size}B"),
      checked(
        || {
          let mut h = Blake2b256::new();
          for chunk in data.chunks(chunk_size) {
            h.update(black_box(chunk));
          }
          h.finalize()
        },
        |out| assert_eq!(out.as_slice(), b256.as_slice()),
      ),
    );
    g.bench_function(
      format!("rustcrypto/blake2b256/{chunk_size}B"),
      checked(
        || {
          let mut h = RustCryptoBlake2b256::new();
          for chunk in data.chunks(chunk_size) {
            h.update(black_box(chunk));
          }
          h.finalize()
        },
        |out| assert_eq!(out, b256),
      ),
    );
    g.bench_function(
      format!("dryoc/blake2b256/{chunk_size}B"),
      checked(
        || {
          let mut state = crypto_generichash_init(None, 32).expect("valid BLAKE2 benchmark operation must succeed");
          for chunk in data.chunks(chunk_size) {
            crypto_generichash_update(&mut state, black_box(chunk));
          }
          let mut out = [0u8; 32];
          crypto_generichash_final(state, &mut out).expect("valid BLAKE2 benchmark operation must succeed");
          out
        },
        |out| assert_eq!(out.as_slice(), b256.as_slice()),
      ),
    );

    g.bench_function(
      format!("rscrypto/blake2s256/{chunk_size}B"),
      checked(
        || {
          let mut h = Blake2s256::new();
          for chunk in data.chunks(chunk_size) {
            h.update(black_box(chunk));
          }
          h.finalize()
        },
        |out| assert_eq!(out.as_slice(), s256.as_slice()),
      ),
    );
    g.bench_function(
      format!("rustcrypto/blake2s256/{chunk_size}B"),
      checked(
        || {
          let mut h = RustCryptoBlake2s256::new();
          for chunk in data.chunks(chunk_size) {
            h.update(black_box(chunk));
          }
          h.finalize()
        },
        |out| assert_eq!(out, s256),
      ),
    );
  }

  g.finish();
}

/// Complete salt/personalization hashing; the main one-shot groups own the plain baselines.
fn params(c: &mut Criterion) {
  if !bench_config::selected("blake2/params") {
    return;
  }
  let sizes = [64usize, 4096, 65_536];
  // Timed: parameter construction, salt/personalization hashing, and per-call destruction.
  // Untimed: input/key fixtures and any typed-key validation.
  let mut g = c.benchmark_group("blake2/params");

  let salt_b = [0x11u8; 16];
  let personal_b = [0x22u8; 16];
  let salt_s = [0x33u8; 8];
  let personal_s = [0x44u8; 8];

  for len in sizes {
    let data = common::random_bytes(len);
    g.throughput(criterion::Throughput::Bytes(len as u64));
    // Oracle: RustCrypto's unkeyed `new_with_salt_and_personal` BLAKE2 parameter block.
    let mut b256 = RustCryptoBlake2bMac256::new_with_salt_and_personal(None, &salt_b, &personal_b)
      .expect("valid BLAKE2 benchmark operation must succeed");
    b256.update(&data);
    let b256 = b256.finalize().into_bytes();
    let mut s256 = RustCryptoBlake2sMac256::new_with_salt_and_personal(None, &salt_s, &personal_s)
      .expect("valid BLAKE2 benchmark operation must succeed");
    s256.update(&data);
    let s256 = s256.finalize().into_bytes();

    g.bench_function(
      BenchmarkId::new("rscrypto/blake2b256/salt+personal", len),
      checked(
        || {
          Blake2bParams::new()
            .salt(black_box(salt_b))
            .personal(black_box(personal_b))
            .hash_256(black_box(&data))
        },
        |out| assert_eq!(out.as_slice(), b256.as_slice()),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rscrypto/blake2s256/salt+personal", len),
      checked(
        || {
          Blake2sParams::new()
            .salt(black_box(salt_s))
            .personal(black_box(personal_s))
            .hash_256(black_box(&data))
        },
        |out| assert_eq!(out.as_slice(), s256.as_slice()),
      ),
    );
  }

  g.finish();
}

fn main() {
  bench_config::run(&[oneshot, short_messages, keyed, streaming, params]);
}
