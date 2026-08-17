//! Blake2 comparison benchmarks: rscrypto vs RustCrypto blake2 crate.

mod common;

use core::hint::black_box;

use blake2::{
  Blake2b as RustCryptoBlake2b, Blake2b512 as RustCryptoBlake2b512, Blake2bMac, Blake2s as RustCryptoBlake2s,
  Blake2s256 as RustCryptoBlake2s256, Blake2sMac,
  digest::{
    Digest as _, Mac as _,
    consts::{U16, U32, U64},
  },
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rscrypto::{
  Blake2b256, Blake2b512, Blake2bKey, Blake2bParams, Blake2s128, Blake2s256, Blake2sKey, Blake2sParams, Digest,
};

type RustCryptoBlake2bMac256 = Blake2bMac<U32>;
type RustCryptoBlake2bMac512 = Blake2bMac<U64>;
type RustCryptoBlake2sMac128 = Blake2sMac<U16>;
type RustCryptoBlake2sMac256 = Blake2sMac<U32>;
type RustCryptoBlake2b256 = RustCryptoBlake2b<U32>;
type RustCryptoBlake2s128 = RustCryptoBlake2s<U16>;

fn oneshot(c: &mut Criterion) {
  use dryoc::classic::crypto_generichash::crypto_generichash;

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
    g.bench_with_input(BenchmarkId::new("dryoc/blake2b256", len), data, |b, d| {
      let mut out = [0u8; 32];
      b.iter(|| {
        crypto_generichash(black_box(&mut out), black_box(d), None)
          .expect("valid BLAKE2 benchmark operation must succeed");
        black_box(out)
      })
    });

    g.bench_with_input(BenchmarkId::new("rscrypto/blake2b512", len), data, |b, d| {
      b.iter(|| black_box(Blake2b512::digest(black_box(d))))
    });
    g.bench_with_input(BenchmarkId::new("rustcrypto/blake2b512", len), data, |b, d| {
      b.iter(|| black_box(RustCryptoBlake2b512::digest(black_box(d))))
    });
    g.bench_with_input(BenchmarkId::new("dryoc/blake2b512", len), data, |b, d| {
      let mut out = [0u8; 64];
      b.iter(|| {
        crypto_generichash(black_box(&mut out), black_box(d), None)
          .expect("valid BLAKE2 benchmark operation must succeed");
        black_box(out)
      })
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

fn tiny_inputs() -> Vec<(usize, Vec<u8>)> {
  [0, 1, 16, 32, 64, 128]
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect()
}

fn host_overhead(c: &mut Criterion) {
  let inputs = tiny_inputs();
  let key_b = [0x42u8; 64];
  let key_s = [0x24u8; 32];
  let key_b_typed = Blake2bKey::new(black_box(&key_b[..32])).expect("valid BLAKE2 benchmark operation must succeed");
  let key_s_typed = Blake2sKey::new(black_box(&key_s)).expect("valid BLAKE2 benchmark operation must succeed");

  let mut oneshot = c.benchmark_group("blake2/host-overhead");
  for (len, data) in &inputs {
    common::set_throughput(&mut oneshot, *len);

    oneshot.bench_with_input(BenchmarkId::new("rscrypto/blake2b256", len), data, |b, d| {
      b.iter(|| black_box(Blake2b256::digest(black_box(d))))
    });
    oneshot.bench_with_input(BenchmarkId::new("rustcrypto/blake2b256", len), data, |b, d| {
      b.iter(|| black_box(RustCryptoBlake2b256::digest(black_box(d))))
    });

    oneshot.bench_with_input(BenchmarkId::new("rscrypto/blake2s256", len), data, |b, d| {
      b.iter(|| black_box(Blake2s256::digest(black_box(d))))
    });
    oneshot.bench_with_input(BenchmarkId::new("rustcrypto/blake2s256", len), data, |b, d| {
      b.iter(|| black_box(RustCryptoBlake2s256::digest(black_box(d))))
    });
  }
  oneshot.finish();

  let mut keyed = c.benchmark_group("blake2/host-keyed-overhead");
  for (len, data) in &inputs {
    common::set_throughput(&mut keyed, *len);

    keyed.bench_with_input(BenchmarkId::new("rscrypto/blake2b256", len), data, |b, d| {
      b.iter(|| black_box(Blake2b256::keyed_digest(key_b_typed, black_box(d))))
    });
    keyed.bench_with_input(BenchmarkId::new("rustcrypto/blake2b256", len), data, |b, d| {
      b.iter(|| {
        let mut mac = RustCryptoBlake2bMac256::new_from_slice(black_box(&key_b[..32]))
          .expect("valid BLAKE2 benchmark operation must succeed");
        mac.update(black_box(d));
        black_box(mac.finalize().into_bytes())
      })
    });

    keyed.bench_with_input(BenchmarkId::new("rscrypto/blake2s256", len), data, |b, d| {
      b.iter(|| black_box(Blake2s256::keyed_digest(key_s_typed, black_box(d))))
    });
    keyed.bench_with_input(BenchmarkId::new("rustcrypto/blake2s256", len), data, |b, d| {
      b.iter(|| {
        let mut mac = RustCryptoBlake2sMac256::new_from_slice(black_box(&key_s))
          .expect("valid BLAKE2 benchmark operation must succeed");
        mac.update(black_box(d));
        black_box(mac.finalize().into_bytes())
      })
    });
  }
  keyed.finish();

  let mut stream = c.benchmark_group("blake2/host-stream-overhead");
  for (len, data) in &inputs {
    common::set_throughput(&mut stream, *len);

    stream.bench_with_input(BenchmarkId::new("rscrypto/blake2b256", len), data, |b, d| {
      b.iter(|| {
        let mut h = Blake2b256::new();
        h.update(black_box(d));
        black_box(h.finalize())
      })
    });
    stream.bench_with_input(BenchmarkId::new("rustcrypto/blake2b256", len), data, |b, d| {
      b.iter(|| {
        let mut h = RustCryptoBlake2b256::new();
        h.update(black_box(d));
        black_box(h.finalize())
      })
    });

    stream.bench_with_input(BenchmarkId::new("rscrypto/blake2s256", len), data, |b, d| {
      b.iter(|| {
        let mut h = Blake2s256::new();
        h.update(black_box(d));
        black_box(h.finalize())
      })
    });
    stream.bench_with_input(BenchmarkId::new("rustcrypto/blake2s256", len), data, |b, d| {
      b.iter(|| {
        let mut h = RustCryptoBlake2s256::new();
        h.update(black_box(d));
        black_box(h.finalize())
      })
    });
  }
  stream.finish();
}

fn keyed(c: &mut Criterion) {
  use dryoc::classic::crypto_generichash::crypto_generichash;

  let inputs = common::comp_sizes();
  let key_b = [0x42u8; 64];
  let key_s = [0x24u8; 32];
  let key_b_256 = Blake2bKey::new(black_box(&key_b[..32])).expect("valid BLAKE2 benchmark operation must succeed");
  let key_b_512 = Blake2bKey::new(black_box(&key_b)).expect("valid BLAKE2 benchmark operation must succeed");
  let key_s_128 = Blake2sKey::new(black_box(&key_s[..16])).expect("valid BLAKE2 benchmark operation must succeed");
  let key_s_256 = Blake2sKey::new(black_box(&key_s)).expect("valid BLAKE2 benchmark operation must succeed");
  let mut g = c.benchmark_group("blake2/keyed");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto/blake2b256", len), data, |b, d| {
      b.iter(|| black_box(Blake2b256::keyed_digest(key_b_256, black_box(d))))
    });
    g.bench_with_input(BenchmarkId::new("rustcrypto/blake2b256", len), data, |b, d| {
      b.iter(|| {
        let mut mac = RustCryptoBlake2bMac256::new_from_slice(black_box(&key_b[..32]))
          .expect("valid BLAKE2 benchmark operation must succeed");
        mac.update(black_box(d));
        black_box(mac.finalize().into_bytes())
      })
    });
    g.bench_with_input(BenchmarkId::new("dryoc/blake2b256", len), data, |b, d| {
      let mut out = [0u8; 32];
      b.iter(|| {
        crypto_generichash(black_box(&mut out), black_box(d), Some(black_box(&key_b[..32])))
          .expect("valid BLAKE2 benchmark operation must succeed");
        black_box(out)
      })
    });

    g.bench_with_input(BenchmarkId::new("rscrypto/blake2b512", len), data, |b, d| {
      b.iter(|| black_box(Blake2b512::keyed_digest(key_b_512, black_box(d))))
    });
    g.bench_with_input(BenchmarkId::new("rustcrypto/blake2b512", len), data, |b, d| {
      b.iter(|| {
        let mut mac = RustCryptoBlake2bMac512::new_from_slice(black_box(&key_b))
          .expect("valid BLAKE2 benchmark operation must succeed");
        mac.update(black_box(d));
        black_box(mac.finalize().into_bytes())
      })
    });
    g.bench_with_input(BenchmarkId::new("dryoc/blake2b512", len), data, |b, d| {
      let mut out = [0u8; 64];
      b.iter(|| {
        crypto_generichash(black_box(&mut out), black_box(d), Some(black_box(&key_b[..])))
          .expect("valid BLAKE2 benchmark operation must succeed");
        black_box(out)
      })
    });

    g.bench_with_input(BenchmarkId::new("rscrypto/blake2s128", len), data, |b, d| {
      b.iter(|| black_box(Blake2s128::keyed_digest(key_s_128, black_box(d))))
    });
    g.bench_with_input(BenchmarkId::new("rustcrypto/blake2s128", len), data, |b, d| {
      b.iter(|| {
        let mut mac = RustCryptoBlake2sMac128::new_from_slice(black_box(&key_s[..16]))
          .expect("valid BLAKE2 benchmark operation must succeed");
        mac.update(black_box(d));
        black_box(mac.finalize().into_bytes())
      })
    });

    g.bench_with_input(BenchmarkId::new("rscrypto/blake2s256", len), data, |b, d| {
      b.iter(|| black_box(Blake2s256::keyed_digest(key_s_256, black_box(d))))
    });
    g.bench_with_input(BenchmarkId::new("rustcrypto/blake2s256", len), data, |b, d| {
      b.iter(|| {
        let mut mac = RustCryptoBlake2sMac256::new_from_slice(black_box(&key_s))
          .expect("valid BLAKE2 benchmark operation must succeed");
        mac.update(black_box(d));
        black_box(mac.finalize().into_bytes())
      })
    });
  }

  g.finish();
}

fn streaming(c: &mut Criterion) {
  use dryoc::classic::crypto_generichash::{
    crypto_generichash_final, crypto_generichash_init, crypto_generichash_update,
  };

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
    g.bench_function(format!("dryoc/blake2b256/{chunk_size}B"), |b| {
      b.iter(|| {
        let mut state = crypto_generichash_init(None, 32).expect("valid BLAKE2 benchmark operation must succeed");
        for chunk in data.chunks(chunk_size) {
          crypto_generichash_update(&mut state, black_box(chunk));
        }
        let mut out = [0u8; 32];
        crypto_generichash_final(state, &mut out).expect("valid BLAKE2 benchmark operation must succeed");
        black_box(out)
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

/// Parameter-block path (salt + personalization): verifies that the init-only
/// cost of XORing salt/personal into IV[4..8] does not perturb the hot path
/// relative to the unsalted `digest()` one-shot.
fn params(c: &mut Criterion) {
  let sizes = [64usize, 4096, 65_536];
  let mut g = c.benchmark_group("blake2/params");

  let salt_b = [0x11u8; 16];
  let personal_b = [0x22u8; 16];
  let salt_s = [0x33u8; 8];
  let personal_s = [0x44u8; 8];

  for len in sizes {
    let data = common::random_bytes(len);
    g.throughput(criterion::Throughput::Bytes(len as u64));

    g.bench_with_input(BenchmarkId::new("rscrypto/blake2b256/plain", len), &data, |b, d| {
      b.iter(|| black_box(Blake2b256::digest(black_box(d))))
    });
    g.bench_with_input(
      BenchmarkId::new("rscrypto/blake2b256/salt+personal", len),
      &data,
      |b, d| {
        b.iter(|| {
          black_box(
            Blake2bParams::new()
              .salt(black_box(salt_b))
              .personal(black_box(personal_b))
              .hash_256(black_box(d)),
          )
        })
      },
    );

    g.bench_with_input(BenchmarkId::new("rscrypto/blake2s256/plain", len), &data, |b, d| {
      b.iter(|| black_box(Blake2s256::digest(black_box(d))))
    });
    g.bench_with_input(
      BenchmarkId::new("rscrypto/blake2s256/salt+personal", len),
      &data,
      |b, d| {
        b.iter(|| {
          black_box(
            Blake2sParams::new()
              .salt(black_box(salt_s))
              .personal(black_box(personal_s))
              .hash_256(black_box(d)),
          )
        })
      },
    );
  }

  g.finish();
}

criterion_group!(benches, oneshot, host_overhead, keyed, streaming, params);
criterion_main!(benches);
