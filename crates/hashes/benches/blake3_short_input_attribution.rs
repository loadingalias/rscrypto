//! BLAKE3 Phase 2 instrumentation benchmarks.
//!
//! Focused on short-input (64..1024B) attribution and dispatch-overhead isolation.

use core::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main};
use hashes::{
  bench as microbench,
  crypto::{Blake3, blake3::dispatch},
};
use traits::{Digest as _, Xof as _};

mod common;
const _: fn() -> Vec<(usize, Vec<u8>)> = common::sized_inputs;

const SHORT_SIZES: [usize; 6] = [64, 65, 128, 256, 512, 1024];
const KEY: [u8; 32] = *b"rscrypto-blake3-benchmark-key!!_";
const DERIVE_CONTEXT: &str = "rscrypto benchmark 2024-01-01 derive key context";

#[inline]
fn configure_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
  group.sample_size(30);
  group.warm_up_time(Duration::from_secs(1));
  group.measurement_time(Duration::from_secs(3));
  group.sampling_mode(SamplingMode::Flat);
}

#[inline]
fn black_box_ref<T: ?Sized>(value: &T) {
  let _ = black_box(value);
}

#[inline]
fn black_box_mut<T: ?Sized>(value: &mut T) {
  let _ = black_box(value);
}

fn blake3_short_path_split(c: &mut Criterion) {
  let mut group = c.benchmark_group("blake3/short-input/short-path-split");
  configure_group(&mut group);

  for len in SHORT_SIZES {
    let data = common::pseudo_random_bytes(len, 0xB1AE_E3B1_A1E3_2001 ^ len as u64);
    common::set_throughput(&mut group, len);

    group.bench_function(BenchmarkId::new("rscrypto/plain/init", len), |b| {
      b.iter(|| {
        let h = Blake3::new();
        black_box_ref(&h);
      })
    });

    group.bench_with_input(BenchmarkId::new("rscrypto/plain/init+update", len), &data, |b, d| {
      b.iter(|| {
        let mut h = Blake3::new();
        h.update(black_box(d));
        black_box_mut(&mut h)
      })
    });

    group.bench_function(BenchmarkId::new("rscrypto/plain/init+finalize-empty", len), |b| {
      b.iter(|| {
        let h = Blake3::new();
        black_box(h.finalize())
      })
    });

    group.bench_with_input(BenchmarkId::new("rscrypto/plain/full", len), &data, |b, d| {
      b.iter(|| {
        let mut h = Blake3::new();
        h.update(black_box(d));
        black_box(h.finalize())
      })
    });

    let mut rs_pre = Blake3::new();
    rs_pre.update(&data);
    group.bench_function(BenchmarkId::new("rscrypto/plain/finalize-only(clone)", len), |b| {
      b.iter(|| {
        let h = black_box(rs_pre.clone());
        black_box(h.finalize())
      })
    });

    group.bench_with_input(BenchmarkId::new("rscrypto/keyed/full", len), &data, |b, d| {
      b.iter(|| {
        let mut h = Blake3::new_keyed(&KEY);
        h.update(black_box(d));
        black_box(h.finalize())
      })
    });

    group.bench_with_input(BenchmarkId::new("rscrypto/derive/full", len), &data, |b, d| {
      b.iter(|| {
        let mut h = Blake3::new_derive_key(DERIVE_CONTEXT);
        h.update(black_box(d));
        black_box(h.finalize())
      })
    });

    group.bench_with_input(BenchmarkId::new("official/plain/full", len), &data, |b, d| {
      b.iter(|| black_box(*blake3::hash(black_box(d)).as_bytes()))
    });

    group.bench_with_input(BenchmarkId::new("official/keyed/full", len), &data, |b, d| {
      b.iter(|| black_box(*blake3::keyed_hash(&KEY, black_box(d)).as_bytes()))
    });

    group.bench_with_input(BenchmarkId::new("official/derive/full", len), &data, |b, d| {
      b.iter(|| black_box(blake3::derive_key(DERIVE_CONTEXT, black_box(d))))
    });

    let mut off_pre = blake3::Hasher::new();
    off_pre.update(&data);
    group.bench_function(BenchmarkId::new("official/plain/finalize-only(clone)", len), |b| {
      b.iter(|| {
        let h = black_box(off_pre.clone());
        black_box(*h.finalize().as_bytes())
      })
    });
  }

  group.finish();
}

fn blake3_xof_target_attribution(c: &mut Criterion) {
  let mut group = c.benchmark_group("blake3/short-input/xof-target-attribution");
  configure_group(&mut group);

  for len in [1, 64, 1024] {
    let data = common::pseudo_random_bytes(len, 0xB1AE_E3B1_A1E3_2005 ^ len as u64);
    let name = format!("{len}B-in");

    group.throughput(Throughput::Elements(1));
    group.bench_function(BenchmarkId::new("rscrypto/init-only(ref)", &name), |b| {
      b.iter(|| {
        let h = Blake3::new();
        black_box_ref(&h);
      })
    });
    group.bench_function(BenchmarkId::new("official/init-only(ref)", &name), |b| {
      b.iter(|| {
        let h = blake3::Hasher::new();
        black_box_ref(&h);
      })
    });

    group.throughput(Throughput::Bytes(len as u64));
    group.bench_with_input(BenchmarkId::new("rscrypto/init+update(ref)", &name), &data, |b, d| {
      b.iter(|| {
        let mut h = Blake3::new();
        h.update(black_box(d));
        black_box_mut(&mut h);
      })
    });
    group.bench_with_input(BenchmarkId::new("official/init+update(ref)", &name), &data, |b, d| {
      b.iter(|| {
        let mut h = blake3::Hasher::new();
        h.update(black_box(d));
        black_box_mut(&mut h);
      })
    });

    group.throughput(Throughput::Bytes(len as u64));
    group.bench_with_input(
      BenchmarkId::new("rscrypto/init+update+finalize-xof(ref)", &name),
      &data,
      |b, d| {
        b.iter(|| {
          let mut h = Blake3::new();
          h.update(black_box(d));
          let xof = h.finalize_xof();
          black_box_ref(&xof);
        })
      },
    );
    group.bench_with_input(
      BenchmarkId::new("official/init+update+finalize-xof(ref)", &name),
      &data,
      |b, d| {
        b.iter(|| {
          let mut h = blake3::Hasher::new();
          h.update(black_box(d));
          let reader = h.finalize_xof();
          black_box_ref(&reader);
        })
      },
    );

    group.throughput(Throughput::Bytes(32));
    group.bench_with_input(
      BenchmarkId::new("rscrypto/finalize-xof+read32(clone)", &name),
      &data,
      |b, d| {
        let mut h = Blake3::new();
        h.update(d);
        b.iter(|| {
          let mut xof = h.clone().finalize_xof();
          let mut out = [0u8; 32];
          xof.squeeze(&mut out);
          black_box_ref(&out);
        })
      },
    );
    group.bench_with_input(
      BenchmarkId::new("official/finalize-xof+read32(clone)", &name),
      &data,
      |b, d| {
        let mut h = blake3::Hasher::new();
        h.update(d);
        b.iter(|| {
          let mut reader = h.clone().finalize_xof();
          let mut out = [0u8; 32];
          reader.fill(&mut out);
          black_box_ref(&out);
        })
      },
    );

    group.bench_with_input(
      BenchmarkId::new("rscrypto/init+read32(target)", &name),
      &data,
      |b, d| {
        let mut out = [0u8; 32];
        b.iter(|| {
          let mut h = Blake3::new();
          h.update(black_box(d));
          let mut xof = h.finalize_xof();
          xof.squeeze(&mut out);
          black_box_ref(&out);
        })
      },
    );
    group.bench_with_input(
      BenchmarkId::new("official/init+read32(target)", &name),
      &data,
      |b, d| {
        let mut out = [0u8; 32];
        b.iter(|| {
          let mut h = blake3::Hasher::new();
          h.update(black_box(d));
          let mut reader = h.finalize_xof();
          reader.fill(&mut out);
          black_box_ref(&out);
        })
      },
    );

    group.bench_with_input(BenchmarkId::new("rscrypto/read32-only", &name), &data, |b, d| {
      let mut h = Blake3::new();
      h.update(d);
      let base = h.finalize_xof();
      let mut out = [0u8; 32];
      b.iter(|| {
        let mut xof = base.clone();
        xof.squeeze(&mut out);
        black_box_ref(&out);
      })
    });
    group.bench_with_input(BenchmarkId::new("official/read32-only", &name), &data, |b, d| {
      let mut h = blake3::Hasher::new();
      h.update(d);
      let base = h.finalize_xof();
      let mut out = [0u8; 32];
      b.iter(|| {
        let mut reader = base.clone();
        reader.fill(&mut out);
        black_box_ref(&out);
      })
    });
  }

  group.finish();
}

fn blake3_stream_target_attribution(c: &mut Criterion) {
  let data_1mb = common::pseudo_random_bytes(1024 * 1024, 0xB1AE_E3B1_A1E3_2006);
  let mut group = c.benchmark_group("blake3/short-input/stream-target-attribution");
  configure_group(&mut group);

  for chunk_size in SHORT_SIZES {
    let name = format!("{chunk_size}B-chunks");
    group.throughput(Throughput::Bytes(data_1mb.len() as u64));

    group.bench_function(BenchmarkId::new("rscrypto/update-loop-only(ref)", &name), |b| {
      b.iter(|| {
        let mut h = Blake3::new();
        for chunk in data_1mb.chunks(chunk_size) {
          h.update(black_box(chunk));
        }
        black_box_mut(&mut h);
      })
    });
    group.bench_function(BenchmarkId::new("official/update-loop-only(ref)", &name), |b| {
      b.iter(|| {
        let mut h = blake3::Hasher::new();
        for chunk in data_1mb.chunks(chunk_size) {
          h.update(black_box(chunk));
        }
        black_box_mut(&mut h);
      })
    });

    group.bench_function(BenchmarkId::new("rscrypto/finalize-only(clone)", &name), |b| {
      let mut h = Blake3::new();
      for chunk in data_1mb.chunks(chunk_size) {
        h.update(chunk);
      }
      b.iter(|| {
        let cloned = h.clone();
        black_box(cloned.finalize());
      })
    });
    group.bench_function(BenchmarkId::new("official/finalize-only(clone)", &name), |b| {
      let mut h = blake3::Hasher::new();
      for chunk in data_1mb.chunks(chunk_size) {
        h.update(chunk);
      }
      b.iter(|| {
        let cloned = h.clone();
        black_box(*cloned.finalize().as_bytes());
      })
    });

    group.bench_function(BenchmarkId::new("rscrypto/full(target)", &name), |b| {
      b.iter(|| {
        let mut h = Blake3::new();
        for chunk in data_1mb.chunks(chunk_size) {
          h.update(black_box(chunk));
        }
        black_box(h.finalize());
      })
    });
    group.bench_function(BenchmarkId::new("official/full(target)", &name), |b| {
      b.iter(|| {
        let mut h = blake3::Hasher::new();
        for chunk in data_1mb.chunks(chunk_size) {
          h.update(black_box(chunk));
        }
        black_box(*h.finalize().as_bytes());
      })
    });
  }

  group.finish();
}

fn blake3_oneshot_dispatch_overhead(c: &mut Criterion) {
  let mut group = c.benchmark_group("blake3/short-input/oneshot-dispatch-overhead");
  configure_group(&mut group);

  for len in SHORT_SIZES {
    let data = common::pseudo_random_bytes(len, 0xB1AE_E3B1_A1E3_2002 ^ len as u64);
    common::set_throughput(&mut group, len);

    let plain_auto_kernel = microbench::kernel_name_for_len("blake3", len).unwrap_or("unknown");
    eprintln!("[short-input][oneshot] len={len} algo=blake3 auto_kernel={plain_auto_kernel}");

    group.bench_with_input(BenchmarkId::new("rscrypto/plain/auto", len), &data, |b, d| {
      b.iter(|| black_box(microbench::run_auto("blake3", black_box(d)).unwrap_or(0)))
    });

    if let Some(k) = microbench::get_kernel("blake3", plain_auto_kernel) {
      group.bench_with_input(
        BenchmarkId::new(format!("rscrypto/plain/direct/{}", k.name), len),
        &data,
        |b, d| b.iter(|| black_box((k.func)(black_box(d)))),
      );
    }

    if plain_auto_kernel != "portable"
      && let Some(k) = microbench::get_kernel("blake3", "portable")
    {
      group.bench_with_input(
        BenchmarkId::new("rscrypto/plain/direct/portable", len),
        &data,
        |b, d| b.iter(|| black_box((k.func)(black_box(d)))),
      );
    }

    for (algo, mode) in [("blake3-keyed", "keyed"), ("blake3-derive", "derive")] {
      let auto_kernel = microbench::kernel_name_for_len(algo, len).unwrap_or("unknown");
      eprintln!("[short-input][oneshot] len={len} algo={algo} auto_kernel={auto_kernel}");

      group.bench_with_input(BenchmarkId::new(format!("rscrypto/{mode}/auto"), len), &data, |b, d| {
        b.iter(|| black_box(microbench::run_auto(algo, black_box(d)).unwrap_or(0)))
      });

      if let Some(k) = microbench::get_kernel(algo, auto_kernel) {
        group.bench_with_input(
          BenchmarkId::new(format!("rscrypto/{mode}/direct/{}", k.name), len),
          &data,
          |b, d| b.iter(|| black_box((k.func)(black_box(d)))),
        );
      }
    }
  }

  group.finish();
}

fn blake3_stream_dispatch_overhead(c: &mut Criterion) {
  let mut group = c.benchmark_group("blake3/short-input/stream-dispatch-overhead");
  configure_group(&mut group);

  let modes: &[(&str, &str, u32)] = &[
    ("plain", "blake3-stream64", 0),
    ("keyed", "blake3-stream64-keyed", dispatch::FLAGS_KEYED_HASH),
    ("derive", "blake3-stream64-derive", dispatch::FLAGS_DERIVE_KEY_MATERIAL),
  ];

  for len in SHORT_SIZES {
    let data = common::pseudo_random_bytes(len, 0xB1AE_E3B1_A1E3_2003 ^ len as u64);
    common::set_throughput(&mut group, len);

    for (mode, algo, flags) in modes {
      let info = dispatch::streaming_dispatch_info(*flags, len);
      let kernel_pair = format!("{}+{}", info.stream_kernel, info.bulk_kernel);
      eprintln!(
        "[short-input][stream64] mode={mode} len={len} pair={kernel_pair} parallel={} min_bytes={} max_threads={} \
         actual_threads={}",
        if info.would_parallelize { "YES" } else { "no" },
        info.parallel_min_bytes,
        info.parallel_max_threads,
        info.parallel_threads,
      );

      group.bench_with_input(BenchmarkId::new(format!("rscrypto/{mode}/auto"), len), &data, |b, d| {
        b.iter(|| black_box(microbench::run_auto(algo, black_box(d)).unwrap_or(0)))
      });

      if let Some(k) = microbench::get_kernel(algo, info.stream_kernel) {
        group.bench_with_input(
          BenchmarkId::new(format!("rscrypto/{mode}/direct/{}", k.name), len),
          &data,
          |b, d| b.iter(|| black_box((k.func)(black_box(d)))),
        );
      }

      if info.stream_kernel != "portable"
        && let Some(k) = microbench::get_kernel(algo, "portable")
      {
        group.bench_with_input(
          BenchmarkId::new(format!("rscrypto/{mode}/direct/portable"), len),
          &data,
          |b, d| b.iter(|| black_box((k.func)(black_box(d)))),
        );
      }
    }
  }

  group.finish();
}

fn blake3_oneshot_apples(c: &mut Criterion) {
  let mut group = c.benchmark_group("blake3/short-input/oneshot-apples");
  configure_group(&mut group);

  for len in SHORT_SIZES {
    let data = common::pseudo_random_bytes(len, 0xB1AE_E3B1_A1E3_2004 ^ len as u64);
    common::set_throughput(&mut group, len);

    let plain_auto_kernel = microbench::kernel_name_for_len("blake3", len).unwrap_or("unknown");
    eprintln!("[short-input][apples] len={len} mode=plain auto_kernel={plain_auto_kernel}");

    group.bench_with_input(BenchmarkId::new("rscrypto/plain/api", len), &data, |b, d| {
      b.iter(|| {
        let mut h = Blake3::new();
        h.update(black_box(d));
        black_box(h.finalize())
      })
    });

    group.bench_with_input(BenchmarkId::new("rscrypto/plain/auto-kernel", len), &data, |b, d| {
      b.iter(|| black_box(microbench::run_auto("blake3", black_box(d)).unwrap_or(0)))
    });

    group.bench_with_input(BenchmarkId::new("official/plain/api", len), &data, |b, d| {
      b.iter(|| black_box(*blake3::hash(black_box(d)).as_bytes()))
    });

    group.bench_with_input(BenchmarkId::new("official/plain/reuse-hasher", len), &data, |b, d| {
      b.iter(|| {
        let mut h = blake3::Hasher::new();
        h.update(black_box(d));
        black_box(*h.finalize().as_bytes())
      })
    });

    let keyed_auto_kernel = microbench::kernel_name_for_len("blake3-keyed", len).unwrap_or("unknown");
    eprintln!("[short-input][apples] len={len} mode=keyed auto_kernel={keyed_auto_kernel}");

    group.bench_with_input(BenchmarkId::new("rscrypto/keyed/api", len), &data, |b, d| {
      b.iter(|| {
        let mut h = Blake3::new_keyed(&KEY);
        h.update(black_box(d));
        black_box(h.finalize())
      })
    });

    group.bench_with_input(BenchmarkId::new("rscrypto/keyed/auto-kernel", len), &data, |b, d| {
      b.iter(|| black_box(microbench::run_auto("blake3-keyed", black_box(d)).unwrap_or(0)))
    });

    group.bench_with_input(BenchmarkId::new("official/keyed/api", len), &data, |b, d| {
      b.iter(|| black_box(*blake3::keyed_hash(&KEY, black_box(d)).as_bytes()))
    });
  }

  group.finish();
}

criterion_group!(
  benches,
  blake3_short_path_split,
  blake3_oneshot_dispatch_overhead,
  blake3_stream_dispatch_overhead,
  blake3_oneshot_apples,
  blake3_xof_target_attribution,
  blake3_stream_target_attribution,
);
criterion_main!(benches);
