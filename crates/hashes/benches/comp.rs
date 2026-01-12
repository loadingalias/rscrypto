use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use hashes::{
  crypto::{
    AsconHash256, AsconXof128, Blake2b512, Blake2s256, Blake3, CShake128, CShake256, Kmac128, Kmac256, Sha3_224,
    Sha3_256, Sha3_384, Sha3_512, Sha224, Sha256, Sha384, Sha512, Sha512_224, Sha512_256, Shake128, Shake256,
  },
  fast::{RapidHash64, SipHash13, SipHash24, Xxh3_64, Xxh3_128},
};
use traits::{Digest as _, FastHash as _};

mod common;

fn comp(c: &mut Criterion) {
  let inputs = common::sized_inputs();
  let mut group = c.benchmark_group("hashes/comp");
  let cshake_fn = b"rscrypto";
  let cshake_custom = b"bench";
  let kmac_key = b"rscrypto kmac bench key (32 bytes)";
  let kmac_custom = b"bench";

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

    group.bench_with_input(BenchmarkId::new("cshake128/rscrypto/32B-out", len), data, |b, d| {
      b.iter(|| {
        let mut out = [0u8; 32];
        CShake128::hash_into(cshake_fn, cshake_custom, black_box(d), &mut out);
        black_box(out)
      })
    });
    group.bench_with_input(BenchmarkId::new("cshake128/sha3/32B-out", len), data, |b, d| {
      b.iter(|| {
        use sha3::digest::{ExtendableOutput, Update, XofReader};
        let core = sha3::CShake128Core::new_with_function_name(cshake_fn, cshake_custom);
        let mut h = sha3::CShake128::from_core(core);
        h.update(black_box(d));
        let mut reader = h.finalize_xof();
        let mut out = [0u8; 32];
        reader.read(&mut out);
        black_box(out)
      })
    });

    group.bench_with_input(BenchmarkId::new("cshake256/rscrypto/32B-out", len), data, |b, d| {
      b.iter(|| {
        let mut out = [0u8; 32];
        CShake256::hash_into(cshake_fn, cshake_custom, black_box(d), &mut out);
        black_box(out)
      })
    });
    group.bench_with_input(BenchmarkId::new("cshake256/sha3/32B-out", len), data, |b, d| {
      b.iter(|| {
        use sha3::digest::{ExtendableOutput, Update, XofReader};
        let core = sha3::CShake256Core::new_with_function_name(cshake_fn, cshake_custom);
        let mut h = sha3::CShake256::from_core(core);
        h.update(black_box(d));
        let mut reader = h.finalize_xof();
        let mut out = [0u8; 32];
        reader.read(&mut out);
        black_box(out)
      })
    });

    group.bench_with_input(BenchmarkId::new("kmac128/rscrypto/32B-out", len), data, |b, d| {
      b.iter(|| {
        let mut out = [0u8; 32];
        let mut h = Kmac128::new(kmac_key, kmac_custom);
        h.update(black_box(d));
        h.finalize_into(&mut out);
        black_box(out)
      })
    });
    group.bench_with_input(BenchmarkId::new("kmac128/tiny-keccak/32B-out", len), data, |b, d| {
      b.iter(|| {
        use tiny_keccak::{Hasher, Kmac};
        let mut out = [0u8; 32];
        let mut h = Kmac::v128(kmac_key, kmac_custom);
        h.update(black_box(d));
        h.finalize(&mut out);
        black_box(out)
      })
    });

    group.bench_with_input(BenchmarkId::new("kmac256/rscrypto/32B-out", len), data, |b, d| {
      b.iter(|| {
        let mut out = [0u8; 32];
        let mut h = Kmac256::new(kmac_key, kmac_custom);
        h.update(black_box(d));
        h.finalize_into(&mut out);
        black_box(out)
      })
    });
    group.bench_with_input(BenchmarkId::new("kmac256/tiny-keccak/32B-out", len), data, |b, d| {
      b.iter(|| {
        use tiny_keccak::{Hasher, Kmac};
        let mut out = [0u8; 32];
        let mut h = Kmac::v256(kmac_key, kmac_custom);
        h.update(black_box(d));
        h.finalize(&mut out);
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

    group.bench_with_input(BenchmarkId::new("blake3/rscrypto", len), data, |b, d| {
      b.iter(|| black_box(Blake3::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("blake3/official", len), data, |b, d| {
      b.iter(|| black_box(blake3::hash(black_box(d))))
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

criterion_group!(benches, comp, fast_comp);
criterion_main!(benches);
