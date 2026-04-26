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
#[cfg(feature = "diag")]
use rscrypto::hashes::crypto::{blake2b, blake2s};
use rscrypto::{Blake2b256, Blake2b512, Blake2bParams, Blake2s128, Blake2s256, Blake2sParams, Digest};

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
      b.iter(|| black_box(Blake2b256::keyed_digest(black_box(&key_b[..32]), black_box(d))))
    });
    keyed.bench_with_input(BenchmarkId::new("rustcrypto/blake2b256", len), data, |b, d| {
      b.iter(|| {
        let mut mac = RustCryptoBlake2bMac256::new_from_slice(black_box(&key_b[..32])).unwrap();
        mac.update(black_box(d));
        black_box(mac.finalize().into_bytes())
      })
    });

    keyed.bench_with_input(BenchmarkId::new("rscrypto/blake2s256", len), data, |b, d| {
      b.iter(|| black_box(Blake2s256::keyed_digest(black_box(&key_s), black_box(d))))
    });
    keyed.bench_with_input(BenchmarkId::new("rustcrypto/blake2s256", len), data, |b, d| {
      b.iter(|| {
        let mut mac = RustCryptoBlake2sMac256::new_from_slice(black_box(&key_s)).unwrap();
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
  let inputs = common::comp_sizes();
  let key_b = [0x42u8; 64];
  let key_s = [0x24u8; 32];
  let mut g = c.benchmark_group("blake2/keyed");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto/blake2b256", len), data, |b, d| {
      b.iter(|| black_box(Blake2b256::keyed_digest(black_box(&key_b[..32]), black_box(d))))
    });
    g.bench_with_input(BenchmarkId::new("rustcrypto/blake2b256", len), data, |b, d| {
      b.iter(|| {
        let mut mac = RustCryptoBlake2bMac256::new_from_slice(black_box(&key_b[..32])).unwrap();
        mac.update(black_box(d));
        black_box(mac.finalize().into_bytes())
      })
    });

    g.bench_with_input(BenchmarkId::new("rscrypto/blake2b512", len), data, |b, d| {
      b.iter(|| black_box(Blake2b512::keyed_digest(black_box(&key_b), black_box(d))))
    });
    g.bench_with_input(BenchmarkId::new("rustcrypto/blake2b512", len), data, |b, d| {
      b.iter(|| {
        let mut mac = RustCryptoBlake2bMac512::new_from_slice(black_box(&key_b)).unwrap();
        mac.update(black_box(d));
        black_box(mac.finalize().into_bytes())
      })
    });

    g.bench_with_input(BenchmarkId::new("rscrypto/blake2s128", len), data, |b, d| {
      b.iter(|| black_box(Blake2s128::keyed_digest(black_box(&key_s[..16]), black_box(d))))
    });
    g.bench_with_input(BenchmarkId::new("rustcrypto/blake2s128", len), data, |b, d| {
      b.iter(|| {
        let mut mac = RustCryptoBlake2sMac128::new_from_slice(black_box(&key_s[..16])).unwrap();
        mac.update(black_box(d));
        black_box(mac.finalize().into_bytes())
      })
    });

    g.bench_with_input(BenchmarkId::new("rscrypto/blake2s256", len), data, |b, d| {
      b.iter(|| black_box(Blake2s256::keyed_digest(black_box(&key_s), black_box(d))))
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
              .salt(black_box(&salt_b))
              .personal(black_box(&personal_b))
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
              .salt(black_box(&salt_s))
              .personal(black_box(&personal_s))
              .hash_256(black_box(d)),
          )
        })
      },
    );
  }

  g.finish();
}

#[cfg(feature = "diag")]
fn compress_kernel(c: &mut Criterion) {
  let block_b = [0xA5u8; 128];
  let block_s = [0x5Au8; 64];

  let mut g = c.benchmark_group("blake2/compress-kernel");

  g.bench_function("rscrypto/blake2b256/last", |b| {
    b.iter(|| {
      let mut state = blake2b::diag_init_state_unkeyed(32);
      blake2b::diag_compress_block(
        black_box(&mut state),
        black_box(&block_b),
        black_box(128),
        black_box(true),
      );
      black_box(state)
    })
  });

  g.bench_function("rscrypto/blake2b256/mid", |b| {
    b.iter(|| {
      let mut state = blake2b::diag_init_state_unkeyed(32);
      blake2b::diag_compress_block(
        black_box(&mut state),
        black_box(&block_b),
        black_box(128),
        black_box(false),
      );
      black_box(state)
    })
  });

  g.bench_function("rscrypto/blake2s256/last", |b| {
    b.iter(|| {
      let mut state = blake2s::diag_init_state_unkeyed(32);
      blake2s::diag_compress_block(
        black_box(&mut state),
        black_box(&block_s),
        black_box(64),
        black_box(true),
      );
      black_box(state)
    })
  });

  g.bench_function("rscrypto/blake2s256/mid", |b| {
    b.iter(|| {
      let mut state = blake2s::diag_init_state_unkeyed(32);
      blake2s::diag_compress_block(
        black_box(&mut state),
        black_box(&block_s),
        black_box(64),
        black_box(false),
      );
      black_box(state)
    })
  });

  g.finish();
}

#[cfg(all(feature = "diag", any(target_arch = "aarch64", target_arch = "riscv64")))]
fn forced_kernel_compare(c: &mut Criterion) {
  let mut group = c.benchmark_group("blake2/forced-kernel");

  let mut block_b = [0u8; 128];
  block_b[..64].fill(0x42);
  let mut block_s = [0u8; 64];
  block_s[..32].fill(0x42);

  group.bench_function("rscrypto/blake2b256/portable/64b", |b| {
    b.iter(|| {
      let mut state = blake2b::diag_init_state_unkeyed(32);
      blake2b::diag_compress_block_portable(&mut state, black_box(&block_b), 64, true);
      black_box(state)
    });
  });

  #[cfg(target_arch = "aarch64")]
  {
    group.bench_function("rscrypto/blake2b256/aarch64-neon/64b", |b| {
      b.iter(|| {
        let mut state = blake2b::diag_init_state_unkeyed(32);
        blake2b::diag_compress_block_aarch64_neon(&mut state, black_box(&block_b), 64, true);
        black_box(state)
      });
    });
  }

  #[cfg(target_arch = "riscv64")]
  if rscrypto::platform::caps().has(rscrypto::platform::caps::riscv::V) {
    group.bench_function("rscrypto/blake2b256/riscv64-v/64b", |b| {
      b.iter(|| {
        let mut state = blake2b::diag_init_state_unkeyed(32);
        blake2b::diag_compress_block_riscv64_v(&mut state, black_box(&block_b), 64, true);
        black_box(state)
      });
    });
  }

  group.bench_function("rscrypto/blake2s256/portable/32b", |b| {
    b.iter(|| {
      let mut state = blake2s::diag_init_state_unkeyed(32);
      blake2s::diag_compress_block_portable(&mut state, black_box(&block_s), 32, true);
      black_box(state)
    });
  });

  #[cfg(target_arch = "aarch64")]
  {
    group.bench_function("rscrypto/blake2s256/aarch64-neon/32b", |b| {
      b.iter(|| {
        let mut state = blake2s::diag_init_state_unkeyed(32);
        blake2s::diag_compress_block_aarch64_neon(&mut state, black_box(&block_s), 32, true);
        black_box(state)
      });
    });
  }

  #[cfg(target_arch = "riscv64")]
  if rscrypto::platform::caps().has(rscrypto::platform::caps::riscv::V) {
    group.bench_function("rscrypto/blake2s256/riscv64-v/32b", |b| {
      b.iter(|| {
        let mut state = blake2s::diag_init_state_unkeyed(32);
        blake2s::diag_compress_block_riscv64_v(&mut state, black_box(&block_s), 32, true);
        black_box(state)
      });
    });
  }

  group.finish();
}

#[cfg(not(feature = "diag"))]
criterion_group!(benches, oneshot, host_overhead, keyed, streaming, params);
#[cfg(feature = "diag")]
#[cfg(not(any(target_arch = "aarch64", target_arch = "riscv64")))]
criterion_group!(
  benches,
  oneshot,
  host_overhead,
  keyed,
  streaming,
  params,
  compress_kernel
);
#[cfg(feature = "diag")]
#[cfg(any(target_arch = "aarch64", target_arch = "riscv64"))]
criterion_group!(
  benches,
  oneshot,
  host_overhead,
  keyed,
  streaming,
  params,
  compress_kernel,
  forced_kernel_compare
);
criterion_main!(benches);
