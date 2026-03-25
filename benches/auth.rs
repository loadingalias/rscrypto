//! Auth benchmarks for rscrypto public APIs.

mod common;

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ed25519_dalek::{Signer as _, SigningKey};
use hkdf::Hkdf as RustCryptoHkdf;
use hmac::Hmac;
use rscrypto::{Ed25519Keypair, Ed25519PublicKey, Ed25519SecretKey, HkdfSha256, HmacSha256, Mac as _};

type RustCryptoHmacSha256 = Hmac<sha2::Sha256>;
type RustCryptoHkdfSha256 = RustCryptoHkdf<sha2::Sha256>;

fn hmac_sha256(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let key = [0x42u8; 32];
  let mut g = c.benchmark_group("hmac-sha256");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(HmacSha256::mac(black_box(&key), black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto", len), data, |b, d| {
      b.iter(|| {
        use hmac::Mac as _;

        let mut mac = RustCryptoHmacSha256::new_from_slice(black_box(&key)).unwrap();
        mac.update(black_box(d));
        black_box(mac.finalize().into_bytes())
      })
    });
  }

  g.finish();
}

fn hmac_sha256_streaming(c: &mut Criterion) {
  let data = common::random_bytes(1048576);
  let key = [0x24u8; 32];
  let mut g = c.benchmark_group("hmac-sha256/streaming");
  g.throughput(criterion::Throughput::Bytes(data.len() as u64));

  for chunk_size in [64, 4096] {
    g.bench_function(format!("rscrypto/{chunk_size}B"), |b| {
      b.iter(|| {
        let mut mac = HmacSha256::new(&key);
        for chunk in data.chunks(chunk_size) {
          mac.update(black_box(chunk));
        }
        black_box(mac.finalize())
      })
    });

    g.bench_function(format!("rustcrypto/{chunk_size}B"), |b| {
      b.iter(|| {
        use hmac::Mac as _;

        let mut mac = RustCryptoHmacSha256::new_from_slice(&key).unwrap();
        for chunk in data.chunks(chunk_size) {
          mac.update(black_box(chunk));
        }
        black_box(mac.finalize().into_bytes())
      })
    });
  }

  g.finish();
}

fn hkdf_sha256_expand(c: &mut Criterion) {
  let salt = [0x11u8; 32];
  let ikm = [0x22u8; 32];
  let info = [0x33u8; 48];
  let hkdf = HkdfSha256::new(&salt, &ikm);
  let rustcrypto = RustCryptoHkdfSha256::new(Some(&salt), &ikm);
  let mut g = c.benchmark_group("hkdf-sha256/expand");

  for out_len in [32usize, 64, 256, 1024] {
    g.throughput(criterion::Throughput::Bytes(out_len as u64));
    g.bench_with_input(BenchmarkId::new("rscrypto", out_len), &out_len, |b, &len| {
      let mut out = vec![0u8; len];
      b.iter(|| {
        hkdf.expand(black_box(&info), black_box(&mut out)).unwrap();
        black_box(out[0])
      })
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto", out_len), &out_len, |b, &len| {
      let mut out = vec![0u8; len];
      b.iter(|| {
        rustcrypto.expand(black_box(&info), black_box(&mut out)).unwrap();
        black_box(out[0])
      })
    });
  }

  g.finish();
}

fn ed25519_public_key(c: &mut Criterion) {
  let secret_bytes = [7u8; 32];
  let mut g = c.benchmark_group("ed25519/public-key-from-secret");

  g.bench_function("rscrypto", |b| {
    b.iter(|| {
      let secret = Ed25519SecretKey::from_bytes(*black_box(&secret_bytes));
      black_box(secret.public_key())
    })
  });

  g.bench_function("dalek", |b| {
    b.iter(|| {
      let signing_key = SigningKey::from_bytes(black_box(&secret_bytes));
      black_box(signing_key.verifying_key())
    })
  });

  g.finish();
}

fn ed25519_keypair_from_secret(c: &mut Criterion) {
  let secret_bytes = [8u8; 32];
  let mut g = c.benchmark_group("ed25519/keypair-from-secret");

  g.bench_function("rscrypto", |b| {
    b.iter(|| {
      let secret = Ed25519SecretKey::from_bytes(*black_box(&secret_bytes));
      black_box(Ed25519Keypair::from_secret_key(secret))
    })
  });

  g.bench_function("dalek", |b| {
    b.iter(|| black_box(SigningKey::from_bytes(black_box(&secret_bytes))))
  });

  g.finish();
}

fn ed25519_sign(c: &mut Criterion) {
  let secret_bytes = [9u8; 32];
  let secret = Ed25519SecretKey::from_bytes(secret_bytes);
  let keypair = Ed25519Keypair::from_secret_key(secret);
  let signing_key = SigningKey::from_bytes(&secret_bytes);
  let inputs = [0usize, 32, 1024, 16384]
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect::<Vec<_>>();
  let mut g = c.benchmark_group("ed25519/sign");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(black_box(&keypair).sign(black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("dalek", len), data, |b, d| {
      b.iter(|| black_box(black_box(&signing_key).sign(black_box(d))))
    });
  }

  g.finish();
}

fn ed25519_verify(c: &mut Criterion) {
  let secret_bytes = [13u8; 32];
  let secret = Ed25519SecretKey::from_bytes(secret_bytes);
  let keypair = Ed25519Keypair::from_secret_key(secret);
  let public: Ed25519PublicKey = keypair.public_key();
  let signing_key = SigningKey::from_bytes(&secret_bytes);
  let verifying_key = signing_key.verifying_key();
  let inputs = [0usize, 32, 1024, 16384]
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect::<Vec<_>>();
  let mut g = c.benchmark_group("ed25519/verify");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let ours = keypair.sign(data);
    let dalek = signing_key.sign(data);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| {
        black_box(&public).verify(black_box(d), black_box(&ours)).unwrap();
        black_box(())
      })
    });

    g.bench_with_input(BenchmarkId::new("dalek", len), data, |b, d| {
      b.iter(|| {
        black_box(&verifying_key)
          .verify_strict(black_box(d), black_box(&dalek))
          .unwrap();
        black_box(())
      })
    });
  }

  g.finish();
}

criterion_group!(
  benches,
  hmac_sha256,
  hmac_sha256_streaming,
  hkdf_sha256_expand,
  ed25519_public_key,
  ed25519_keypair_from_secret,
  ed25519_sign,
  ed25519_verify
);
criterion_main!(benches);
