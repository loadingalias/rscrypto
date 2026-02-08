use core::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main};
use hashes::{
  crypto::{
    AsconHash256, AsconXof128, Blake2b512, Blake2s256, Blake3, Sha3_224, Sha3_256, Sha3_384, Sha3_512, Sha224, Sha256,
    Sha384, Sha512, Sha512_224, Sha512_256, Shake128, Shake256,
  },
  fast::{RapidHash64, SipHash13, SipHash24, Xxh3_64, Xxh3_128},
};
use traits::{Digest as _, FastHash as _, Xof as _};

mod common;

fn blake3_comp(c: &mut Criterion) {
  // Keep the comparison matrix crisp and stable across CI runners.
  // This is intentionally aligned with the BLAKE3-specific matrix in
  // `crates/hashes/src/crypto/blake3/TASK.md`.
  let oneshot_sizes = [
    0usize,
    1,
    31,
    32,
    63,
    64,
    65,
    128,
    1024,
    4 * 1024,
    16 * 1024,
    64 * 1024,
    1024 * 1024,
  ];

  // One-shot hash (rscrypto vs official).
  {
    let mut group = c.benchmark_group("blake3/hash");
    group.sample_size(40);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(4));
    group.sampling_mode(SamplingMode::Flat);

    for &len in &oneshot_sizes {
      let data = common::pseudo_random_bytes(len, 0xB1AE_E3B1_A1E3_0002);
      common::set_throughput(&mut group, len);

      group.bench_with_input(BenchmarkId::new("rscrypto/blake3", len), &data, |b, d| {
        b.iter(|| black_box(Blake3::digest(black_box(d))))
      });
      group.bench_with_input(BenchmarkId::new("official/blake3", len), &data, |b, d| {
        b.iter(|| black_box(*blake3::hash(black_box(d)).as_bytes()))
      });
      group.bench_with_input(BenchmarkId::new("official/blake3-rayon", len), &data, |b, d| {
        b.iter(|| {
          let mut h = blake3::Hasher::new();
          h.update_rayon(black_box(d));
          black_box(*h.finalize().as_bytes())
        })
      });
    }

    group.finish();
  }

  // Streaming update+finalize overhead (1 MiB total, varying chunk sizes).
  {
    let data_1mb = common::pseudo_random_bytes(1024 * 1024, 0xB1AE_E3B1_A1E3_0003);
    let data_1mb = black_box(data_1mb);
    let mut group = c.benchmark_group("blake3/streaming");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(4));
    group.sampling_mode(SamplingMode::Flat);
    group.throughput(Throughput::Bytes(data_1mb.len() as u64));

    for chunk_size in [64usize, 128, 256, 512, 1024, 4096, 16384, 65536] {
      let case = format!("{chunk_size}B-chunks");

      group.bench_with_input(BenchmarkId::new("rscrypto/blake3", &case), &chunk_size, |b, &cs| {
        b.iter(|| {
          let mut h = Blake3::new();
          for chunk in data_1mb.chunks(cs) {
            h.update(chunk);
          }
          black_box(h.finalize())
        })
      });

      group.bench_with_input(BenchmarkId::new("official/blake3", &case), &chunk_size, |b, &cs| {
        b.iter(|| {
          let mut h = blake3::Hasher::new();
          for chunk in data_1mb.chunks(cs) {
            h.update(chunk);
          }
          black_box(*h.finalize().as_bytes())
        })
      });

      group.bench_with_input(
        BenchmarkId::new("official-rayon/blake3", &case),
        &chunk_size,
        |b, &_cs| {
          b.iter(|| {
            let mut h = blake3::Hasher::new();
            h.update_rayon(&data_1mb[..]);
            black_box(*h.finalize().as_bytes())
          })
        },
      );
    }

    group.finish();
  }

  // XOF (hash + finalize_xof + read), limited matrix to keep runtime bounded.
  {
    let input_sizes = [1usize, 64, 1024, 64 * 1024];
    let output_sizes = [32usize, 1024];

    let mut group = c.benchmark_group("blake3/xof");
    group.sample_size(25);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sampling_mode(SamplingMode::Flat);

    for &output_size in &output_sizes {
      for &len in &input_sizes {
        let data =
          common::pseudo_random_bytes(len, 0xB1AE_E3B1_A1E3_0004 ^ (len as u64) ^ ((output_size as u64) << 32));
        let case = format!("{len}B-in/{output_size}B-out");
        group.throughput(Throughput::Bytes((len + output_size) as u64));

        group.bench_with_input(BenchmarkId::new("rscrypto/blake3", &case), &data, |b, d| {
          let mut out = vec![0u8; output_size];
          b.iter(|| {
            let mut h = Blake3::new();
            h.update(black_box(d));
            let mut xof = h.finalize_xof();
            xof.squeeze(&mut out);
            black_box(&out);
          })
        });

        group.bench_with_input(BenchmarkId::new("official/blake3", &case), &data, |b, d| {
          let mut out = vec![0u8; output_size];
          b.iter(|| {
            let mut h = blake3::Hasher::new();
            h.update(black_box(d));
            let mut reader = h.finalize_xof();
            reader.fill(&mut out);
            black_box(&out);
          })
        });

        // Add official-rayon comparison for larger inputs where parallelization matters
        if len >= 1024 {
          group.bench_with_input(BenchmarkId::new("official-rayon/blake3", &case), &data, |b, d| {
            let mut out = vec![0u8; output_size];
            b.iter(|| {
              let mut h = blake3::Hasher::new();
              h.update_rayon(black_box(d));
              let mut reader = h.finalize_xof();
              reader.fill(&mut out);
              black_box(&out);
            })
          });
        }
      }
    }

    group.finish();
  }

  // Keyed hash (one-shot).
  {
    let key: [u8; 32] = *b"rscrypto-blake3-benchmark-key!!_";

    let mut group = c.benchmark_group("blake3/keyed-hash");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sampling_mode(SamplingMode::Flat);

    for &len in &oneshot_sizes {
      let data = common::pseudo_random_bytes(len, 0xB1AE_E3B1_A1E3_0005);
      common::set_throughput(&mut group, len);

      group.bench_with_input(BenchmarkId::new("rscrypto/blake3", len), &data, |b, d| {
        b.iter(|| black_box(Blake3::keyed_digest(&key, black_box(d))))
      });
      group.bench_with_input(BenchmarkId::new("official/blake3", len), &data, |b, d| {
        b.iter(|| black_box(*blake3::keyed_hash(&key, black_box(d)).as_bytes()))
      });

      // Add official-rayon comparison for larger inputs where parallelization matters
      if len >= 1024 {
        group.bench_with_input(BenchmarkId::new("official-rayon/blake3", len), &data, |b, d| {
          b.iter(|| {
            let mut h = blake3::Hasher::new_keyed(&key);
            h.update_rayon(black_box(d));
            black_box(*h.finalize().as_bytes())
          })
        });
      }
    }

    group.finish();
  }

  // Derive key (one-shot).
  {
    let context = "rscrypto benchmark 2024-01-01 derive key context";

    let mut group = c.benchmark_group("blake3/derive-key");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sampling_mode(SamplingMode::Flat);

    for &len in &oneshot_sizes {
      let data = common::pseudo_random_bytes(len, 0xB1AE_E3B1_A1E3_0006);
      common::set_throughput(&mut group, len);

      group.bench_with_input(BenchmarkId::new("rscrypto/blake3", len), &data, |b, d| {
        b.iter(|| black_box(Blake3::derive_key(context, black_box(d))))
      });
      group.bench_with_input(BenchmarkId::new("official/blake3", len), &data, |b, d| {
        b.iter(|| black_box(blake3::derive_key(context, black_box(d))))
      });

      // Add official-rayon comparison for larger inputs where parallelization matters
      if len >= 1024 {
        group.bench_with_input(BenchmarkId::new("official-rayon/blake3", len), &data, |b, d| {
          b.iter(|| {
            let mut h = blake3::Hasher::new_derive_key(context);
            h.update_rayon(black_box(d));
            black_box(h.finalize())
          })
        });
      }
    }

    group.finish();
  }
}

fn comp(c: &mut Criterion) {
  let inputs = common::sized_inputs();
  let mut group = c.benchmark_group("hashes/comp");

  for (len, data) in &inputs {
    common::set_throughput(&mut group, *len);

    group.bench_with_input(BenchmarkId::new("sha224/rscrypto", len), data, |b, d| {
      b.iter(|| black_box(Sha224::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("sha224/sha2", len), data, |b, d| {
      b.iter(|| {
        use sha2::Digest as _;
        let out = sha2::Sha224::digest(black_box(d));
        black_box(out)
      })
    });

    group.bench_with_input(BenchmarkId::new("sha256/rscrypto", len), data, |b, d| {
      b.iter(|| black_box(Sha256::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("sha256/sha2", len), data, |b, d| {
      b.iter(|| {
        use sha2::Digest as _;
        let out = sha2::Sha256::digest(black_box(d));
        black_box(out)
      })
    });

    group.bench_with_input(BenchmarkId::new("sha384/rscrypto", len), data, |b, d| {
      b.iter(|| black_box(Sha384::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("sha384/sha2", len), data, |b, d| {
      b.iter(|| {
        use sha2::Digest as _;
        let out = sha2::Sha384::digest(black_box(d));
        black_box(out)
      })
    });

    group.bench_with_input(BenchmarkId::new("sha3_256/rscrypto", len), data, |b, d| {
      b.iter(|| black_box(Sha3_256::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("sha3_256/sha3", len), data, |b, d| {
      b.iter(|| {
        use sha3::Digest as _;
        let out = sha3::Sha3_256::digest(black_box(d));
        black_box(out)
      })
    });

    group.bench_with_input(BenchmarkId::new("sha3_224/rscrypto", len), data, |b, d| {
      b.iter(|| black_box(Sha3_224::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("sha3_224/sha3", len), data, |b, d| {
      b.iter(|| {
        use sha3::Digest as _;
        let out = sha3::Sha3_224::digest(black_box(d));
        black_box(out)
      })
    });

    group.bench_with_input(BenchmarkId::new("sha3_384/rscrypto", len), data, |b, d| {
      b.iter(|| black_box(Sha3_384::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("sha3_384/sha3", len), data, |b, d| {
      b.iter(|| {
        use sha3::Digest as _;
        let out = sha3::Sha3_384::digest(black_box(d));
        black_box(out)
      })
    });

    group.bench_with_input(BenchmarkId::new("sha3_512/rscrypto", len), data, |b, d| {
      b.iter(|| black_box(Sha3_512::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("sha3_512/sha3", len), data, |b, d| {
      b.iter(|| {
        use sha3::Digest as _;
        let out = sha3::Sha3_512::digest(black_box(d));
        black_box(out)
      })
    });

    group.bench_with_input(BenchmarkId::new("sha512_224/rscrypto", len), data, |b, d| {
      b.iter(|| black_box(Sha512_224::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("sha512_224/sha2", len), data, |b, d| {
      b.iter(|| {
        use sha2::Digest as _;
        let out = sha2::Sha512_224::digest(black_box(d));
        black_box(out)
      })
    });

    group.bench_with_input(BenchmarkId::new("sha512_256/rscrypto", len), data, |b, d| {
      b.iter(|| black_box(Sha512_256::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("sha512_256/sha2", len), data, |b, d| {
      b.iter(|| {
        use sha2::Digest as _;
        let out = sha2::Sha512_256::digest(black_box(d));
        black_box(out)
      })
    });

    group.bench_with_input(BenchmarkId::new("sha512/rscrypto", len), data, |b, d| {
      b.iter(|| black_box(Sha512::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("sha512/sha2", len), data, |b, d| {
      b.iter(|| {
        use sha2::Digest as _;
        let out = sha2::Sha512::digest(black_box(d));
        black_box(out)
      })
    });

    group.bench_with_input(BenchmarkId::new("shake128/rscrypto/32B-out", len), data, |b, d| {
      b.iter(|| {
        let mut out = [0u8; 32];
        Shake128::hash_into(black_box(d), &mut out);
        black_box(out)
      })
    });
    group.bench_with_input(BenchmarkId::new("shake128/sha3/32B-out", len), data, |b, d| {
      b.iter(|| {
        use sha3::digest::{ExtendableOutput, Update, XofReader};
        let mut h = sha3::Shake128::default();
        h.update(black_box(d));
        let mut reader = h.finalize_xof();
        let mut out = [0u8; 32];
        reader.read(&mut out);
        black_box(out)
      })
    });

    group.bench_with_input(BenchmarkId::new("shake256/rscrypto", len), data, |b, d| {
      b.iter(|| {
        let mut out = [0u8; 32];
        Shake256::hash_into(black_box(d), &mut out);
        black_box(out)
      })
    });
    group.bench_with_input(BenchmarkId::new("shake256/sha3", len), data, |b, d| {
      b.iter(|| {
        use sha3::digest::{ExtendableOutput, Update, XofReader};
        let mut h = sha3::Shake256::default();
        h.update(black_box(d));
        let mut reader = h.finalize_xof();
        let mut out = [0u8; 32];
        reader.read(&mut out);
        black_box(out)
      })
    });

    group.bench_with_input(BenchmarkId::new("ascon_hash256/rscrypto", len), data, |b, d| {
      b.iter(|| black_box(AsconHash256::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("ascon_hash256/ascon-hash256", len), data, |b, d| {
      b.iter(|| {
        use ascon_hash256::Digest as _;
        let out = ascon_hash256::AsconHash256::digest(black_box(d));
        black_box(out)
      })
    });

    group.bench_with_input(BenchmarkId::new("ascon_xof128/rscrypto/32B-out", len), data, |b, d| {
      b.iter(|| {
        let mut out = [0u8; 32];
        AsconXof128::hash_into(black_box(d), &mut out);
        black_box(out)
      })
    });
    group.bench_with_input(
      BenchmarkId::new("ascon_xof128/ascon-hash256/32B-out", len),
      data,
      |b, d| {
        b.iter(|| {
          use ascon_hash256::digest::{ExtendableOutput, Update, XofReader};
          let mut h = ascon_hash256::AsconXof128::default();
          h.update(black_box(d));
          let mut reader = h.finalize_xof();
          let mut out = [0u8; 32];
          reader.read(&mut out);
          black_box(out)
        })
      },
    );

    group.bench_with_input(BenchmarkId::new("blake2s256/rscrypto", len), data, |b, d| {
      b.iter(|| black_box(Blake2s256::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("blake2s256/blake2", len), data, |b, d| {
      b.iter(|| {
        use blake2::Digest as _;
        let out = blake2::Blake2s256::digest(black_box(d));
        black_box(out)
      })
    });

    group.bench_with_input(BenchmarkId::new("blake2b512/rscrypto", len), data, |b, d| {
      b.iter(|| black_box(Blake2b512::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("blake2b512/blake2", len), data, |b, d| {
      b.iter(|| {
        use blake2::Digest as _;
        let out = blake2::Blake2b512::digest(black_box(d));
        black_box(out)
      })
    });
  }

  group.finish();
}

fn fast_comp(c: &mut Criterion) {
  let inputs = common::sized_inputs();
  let mut group = c.benchmark_group("fasthash/comp");
  let rapid_seed = 0u64;
  let rapid_secrets = rapidhash::v3::RapidSecrets::seed_cpp(rapid_seed);
  let sip_key = [0u64; 2];

  for (len, data) in &inputs {
    common::set_throughput(&mut group, *len);

    group.bench_with_input(BenchmarkId::new("xxh3_64/rscrypto", len), data, |b, d| {
      b.iter(|| black_box(Xxh3_64::hash(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("xxh3_64/xxhash-rust", len), data, |b, d| {
      b.iter(|| black_box(xxhash_rust::xxh3::xxh3_64(black_box(d))))
    });

    group.bench_with_input(BenchmarkId::new("xxh3_128/rscrypto", len), data, |b, d| {
      b.iter(|| black_box(Xxh3_128::hash(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("xxh3_128/xxhash-rust", len), data, |b, d| {
      b.iter(|| black_box(xxhash_rust::xxh3::xxh3_128(black_box(d))))
    });

    group.bench_with_input(BenchmarkId::new("rapidhash_v3/rscrypto", len), data, |b, d| {
      b.iter(|| black_box(RapidHash64::hash_with_seed(rapid_seed, black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("rapidhash_v3/rapidhash", len), data, |b, d| {
      b.iter(|| black_box(rapidhash::v3::rapidhash_v3_seeded(black_box(d), &rapid_secrets)))
    });

    group.bench_with_input(BenchmarkId::new("siphash13/rscrypto", len), data, |b, d| {
      b.iter(|| black_box(SipHash13::hash_with_seed(sip_key, black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("siphash13/siphasher", len), data, |b, d| {
      b.iter(|| {
        use core::hash::Hasher as _;
        let mut h = siphasher::sip::SipHasher13::new_with_keys(sip_key[0], sip_key[1]);
        h.write(black_box(d));
        black_box(h.finish())
      })
    });

    group.bench_with_input(BenchmarkId::new("siphash24/rscrypto", len), data, |b, d| {
      b.iter(|| black_box(SipHash24::hash_with_seed(sip_key, black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("siphash24/siphasher", len), data, |b, d| {
      b.iter(|| {
        use core::hash::Hasher as _;
        let mut h = siphasher::sip::SipHasher24::new_with_keys(sip_key[0], sip_key[1]);
        h.write(black_box(d));
        black_box(h.finish())
      })
    });
  }

  group.finish();
}

criterion_group!(benches, blake3_comp, comp, fast_comp);
criterion_main!(benches);
