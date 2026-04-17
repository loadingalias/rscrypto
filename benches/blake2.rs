//! Blake2 comparison benchmarks: rscrypto vs RustCrypto blake2 crate.

mod common;

use core::hint::black_box;

use blake2::{
  Blake2b256 as RustCryptoBlake2b256, Blake2b512 as RustCryptoBlake2b512, Blake2bMac,
  Blake2s128 as RustCryptoBlake2s128, Blake2s256 as RustCryptoBlake2s256, Blake2sMac, Digest as _,
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use digest::typenum::{U16, U32, U64};
use hmac::{Mac as _, digest::KeyInit};
use rscrypto::{Blake2b256, Blake2b512, Blake2s128, Blake2s256, Digest};

type RustCryptoBlake2bMac256 = Blake2bMac<U32>;
type RustCryptoBlake2bMac512 = Blake2bMac<U64>;
type RustCryptoBlake2sMac128 = Blake2sMac<U16>;
type RustCryptoBlake2sMac256 = Blake2sMac<U32>;

fn oneshot(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("blake2");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto/blake2b256", len), data, |b, d| {
      b.iter(|| black_box(Blake2b256::digest(black_box(d))))
    });
    g.bench_with_input(BenchmarkId::new("rustcrypto/blake2b256", len), data, |b, d| {
      b.iter(|| black_box(RustCryptoBlake2b256::digest(black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("rscrypto/blake2b512", len), data, |b, d| {
      b.iter(|| black_box(Blake2b512::digest(black_box(d))))
    });
    g.bench_with_input(BenchmarkId::new("rustcrypto/blake2b512", len), data, |b, d| {
      b.iter(|| black_box(RustCryptoBlake2b512::digest(black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("rscrypto/blake2s128", len), data, |b, d| {
      b.iter(|| black_box(Blake2s128::digest(black_box(d))))
    });
    g.bench_with_input(BenchmarkId::new("rustcrypto/blake2s128", len), data, |b, d| {
      b.iter(|| black_box(RustCryptoBlake2s128::digest(black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("rscrypto/blake2s256", len), data, |b, d| {
      b.iter(|| black_box(Blake2s256::digest(black_box(d))))
    });
    g.bench_with_input(BenchmarkId::new("rustcrypto/blake2s256", len), data, |b, d| {
      b.iter(|| black_box(RustCryptoBlake2s256::digest(black_box(d))))
    });
  }

  g.finish();
}

fn keyed(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let key_b = [0x42u8; 64];
  let key_s = [0x24u8; 32];
  let mut g = c.benchmark_group("blake2/keyed");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto/blake2b256", len), data, |b, d| {
      b.iter(|| black_box(Blake2b256::keyed_hash(black_box(&key_b[..32]), black_box(d))))
    });
    g.bench_with_input(BenchmarkId::new("rustcrypto/blake2b256", len), data, |b, d| {
      b.iter(|| {
        let mut mac = RustCryptoBlake2bMac256::new_from_slice(black_box(&key_b[..32])).unwrap();
        mac.update(black_box(d));
        black_box(mac.finalize().into_bytes())
      })
    });

    g.bench_with_input(BenchmarkId::new("rscrypto/blake2b512", len), data, |b, d| {
      b.iter(|| black_box(Blake2b512::keyed_hash(black_box(&key_b), black_box(d))))
    });
    g.bench_with_input(BenchmarkId::new("rustcrypto/blake2b512", len), data, |b, d| {
      b.iter(|| {
        let mut mac = RustCryptoBlake2bMac512::new_from_slice(black_box(&key_b)).unwrap();
        mac.update(black_box(d));
        black_box(mac.finalize().into_bytes())
      })
    });

    g.bench_with_input(BenchmarkId::new("rscrypto/blake2s128", len), data, |b, d| {
      b.iter(|| black_box(Blake2s128::keyed_hash(black_box(&key_s[..16]), black_box(d))))
    });
    g.bench_with_input(BenchmarkId::new("rustcrypto/blake2s128", len), data, |b, d| {
      b.iter(|| {
        let mut mac = RustCryptoBlake2sMac128::new_from_slice(black_box(&key_s[..16])).unwrap();
        mac.update(black_box(d));
        black_box(mac.finalize().into_bytes())
      })
    });

    g.bench_with_input(BenchmarkId::new("rscrypto/blake2s256", len), data, |b, d| {
      b.iter(|| black_box(Blake2s256::keyed_hash(black_box(&key_s), black_box(d))))
    });
    g.bench_with_input(BenchmarkId::new("rustcrypto/blake2s256", len), data, |b, d| {
      b.iter(|| {
        let mut mac = RustCryptoBlake2sMac256::new_from_slice(black_box(&key_s)).unwrap();
        mac.update(black_box(d));
        black_box(mac.finalize().into_bytes())
      })
    });
  }

  g.finish();
}

fn streaming(c: &mut Criterion) {
  let data = common::random_bytes(1048576);
  let mut g = c.benchmark_group("blake2/streaming");
  g.throughput(criterion::Throughput::Bytes(data.len() as u64));

  for chunk_size in [64, 4096, 65536] {
    g.bench_function(format!("rscrypto/blake2b256/{chunk_size}B"), |b| {
      b.iter(|| {
        let mut h = Blake2b256::new();
        for chunk in data.chunks(chunk_size) {
          h.update(black_box(chunk));
        }
        black_box(h.finalize())
      })
    });
    g.bench_function(format!("rustcrypto/blake2b256/{chunk_size}B"), |b| {
      b.iter(|| {
        let mut h = RustCryptoBlake2b256::new();
        for chunk in data.chunks(chunk_size) {
          h.update(black_box(chunk));
        }
        black_box(h.finalize())
      })
    });

    g.bench_function(format!("rscrypto/blake2s256/{chunk_size}B"), |b| {
      b.iter(|| {
        let mut h = Blake2s256::new();
        for chunk in data.chunks(chunk_size) {
          h.update(black_box(chunk));
        }
        black_box(h.finalize())
      })
    });
    g.bench_function(format!("rustcrypto/blake2s256/{chunk_size}B"), |b| {
      b.iter(|| {
        let mut h = RustCryptoBlake2s256::new();
        for chunk in data.chunks(chunk_size) {
          h.update(black_box(chunk));
        }
        black_box(h.finalize())
      })
    });
  }

  g.finish();
}

criterion_group!(benches, oneshot, keyed, streaming);
criterion_main!(benches);
