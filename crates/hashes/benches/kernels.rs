use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hashes::{
  crypto::{Blake3, Sha3_256, Sha3_512, Sha256, Sha512, Shake256},
  fast::{RapidHash64, RapidHash128, SipHash13, SipHash24, Xxh3_64, Xxh3_128},
};
use platform as _;
use traits::{Digest as _, FastHash as _, Xof as _};

mod common;

fn print_blake3_dispatch_info(oneshot_lens: impl Iterator<Item = usize>) {
  let tune = platform::tune();
  let caps = platform::caps();
  let (stream, bulk) = hashes::crypto::blake3::dispatch::streaming_kernel_names();
  eprintln!(
    "BLAKE3 dispatch: tune.kind={:?} ({}) caps={:?}",
    tune.kind,
    tune.kind.name(),
    caps
  );
  eprintln!("BLAKE3 streaming dispatch: stream={stream} bulk={bulk}");

  for len in oneshot_lens {
    let kernel = hashes::crypto::blake3::dispatch::kernel_name_for_len(len);
    eprintln!("BLAKE3 oneshot dispatch: len={len} kernel={kernel}");
  }
}

fn oneshot(c: &mut Criterion) {
  let inputs = common::sized_inputs();
  let mut group = c.benchmark_group("hashes/oneshot");

  print_blake3_dispatch_info(inputs.iter().map(|(len, _)| *len));

  for (len, data) in &inputs {
    common::set_throughput(&mut group, *len);

    group.bench_with_input(BenchmarkId::new("sha256", len), data, |b, d| {
      b.iter(|| black_box(Sha256::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("sha3_256", len), data, |b, d| {
      b.iter(|| black_box(Sha3_256::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("sha3_512", len), data, |b, d| {
      b.iter(|| black_box(Sha3_512::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("sha512", len), data, |b, d| {
      b.iter(|| black_box(Sha512::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("shake256/32B", len), data, |b, d| {
      b.iter(|| {
        let mut out = [0u8; 32];
        Shake256::hash_into(black_box(d), &mut out);
        black_box(out)
      })
    });
    group.bench_with_input(BenchmarkId::new("blake3", len), data, |b, d| {
      b.iter(|| black_box(Blake3::digest(black_box(d))))
    });
  }

  group.finish();
}

fn fast_oneshot(c: &mut Criterion) {
  let inputs = common::sized_inputs();
  let mut group = c.benchmark_group("fasthash/oneshot");

  for (len, data) in &inputs {
    common::set_throughput(&mut group, *len);

    group.bench_with_input(BenchmarkId::new("xxh3_64", len), data, |b, d| {
      b.iter(|| black_box(Xxh3_64::hash(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("xxh3_128", len), data, |b, d| {
      b.iter(|| black_box(Xxh3_128::hash(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("rapidhash64", len), data, |b, d| {
      b.iter(|| black_box(RapidHash64::hash(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("rapidhash128", len), data, |b, d| {
      b.iter(|| black_box(RapidHash128::hash(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("siphash13", len), data, |b, d| {
      b.iter(|| black_box(SipHash13::hash_with_seed([0u64; 2], black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("siphash24", len), data, |b, d| {
      b.iter(|| black_box(SipHash24::hash_with_seed([0u64; 2], black_box(d))))
    });
  }

  group.finish();
}

fn streaming(c: &mut Criterion) {
  let mut group = c.benchmark_group("hashes/streaming");
  let data = common::pseudo_random_bytes(1024 * 1024, 0xA11C_E5ED_5EED_0001);
  let data = black_box(data);
  group.throughput(Throughput::Bytes(data.len() as u64));

  {
    let (stream, bulk) = hashes::crypto::blake3::dispatch::streaming_kernel_names();
    eprintln!("BLAKE3 streaming dispatch (for benches): stream={stream} bulk={bulk}");
  }

  group.bench_function("sha256/64B-chunks", |b| {
    b.iter(|| {
      let mut h = Sha256::new();
      for chunk in data.chunks(64) {
        h.update(chunk);
      }
      black_box(h.finalize())
    })
  });

  group.bench_function("sha256/4KiB-chunks", |b| {
    b.iter(|| {
      let mut h = Sha256::new();
      for chunk in data.chunks(4 * 1024) {
        h.update(chunk);
      }
      black_box(h.finalize())
    })
  });

  group.bench_function("sha3_256/64B-chunks", |b| {
    b.iter(|| {
      let mut h = Sha3_256::new();
      for chunk in data.chunks(64) {
        h.update(chunk);
      }
      black_box(h.finalize())
    })
  });

  group.bench_function("sha3_256/4KiB-chunks", |b| {
    b.iter(|| {
      let mut h = Sha3_256::new();
      for chunk in data.chunks(4 * 1024) {
        h.update(chunk);
      }
      black_box(h.finalize())
    })
  });

  group.bench_function("sha3_512/64B-chunks", |b| {
    b.iter(|| {
      let mut h = Sha3_512::new();
      for chunk in data.chunks(64) {
        h.update(chunk);
      }
      black_box(h.finalize())
    })
  });

  group.bench_function("sha3_512/4KiB-chunks", |b| {
    b.iter(|| {
      let mut h = Sha3_512::new();
      for chunk in data.chunks(4 * 1024) {
        h.update(chunk);
      }
      black_box(h.finalize())
    })
  });

  group.bench_function("sha512/64B-chunks", |b| {
    b.iter(|| {
      let mut h = Sha512::new();
      for chunk in data.chunks(64) {
        h.update(chunk);
      }
      black_box(h.finalize())
    })
  });

  group.bench_function("sha512/4KiB-chunks", |b| {
    b.iter(|| {
      let mut h = Sha512::new();
      for chunk in data.chunks(4 * 1024) {
        h.update(chunk);
      }
      black_box(h.finalize())
    })
  });

  group.bench_function("shake256/64B-chunks/32B-out", |b| {
    b.iter(|| {
      let mut h = Shake256::new();
      for chunk in data.chunks(64) {
        h.update(chunk);
      }
      let mut xof = h.finalize_xof();
      let mut out = [0u8; 32];
      xof.squeeze(&mut out);
      black_box(out)
    })
  });

  group.bench_function("shake256/4KiB-chunks/32B-out", |b| {
    b.iter(|| {
      let mut h = Shake256::new();
      for chunk in data.chunks(4 * 1024) {
        h.update(chunk);
      }
      let mut xof = h.finalize_xof();
      let mut out = [0u8; 32];
      xof.squeeze(&mut out);
      black_box(out)
    })
  });

  group.bench_function("blake3/64B-chunks", |b| {
    b.iter(|| {
      let mut h = Blake3::new();
      for chunk in data.chunks(64) {
        h.update(chunk);
      }
      black_box(h.finalize())
    })
  });

  group.bench_function("blake3/4KiB-chunks", |b| {
    b.iter(|| {
      let mut h = Blake3::new();
      for chunk in data.chunks(4 * 1024) {
        h.update(chunk);
      }
      black_box(h.finalize())
    })
  });

  group.finish();
}

criterion_group!(benches, oneshot, fast_oneshot, streaming);
criterion_main!(benches);
