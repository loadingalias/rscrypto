//! Auth benchmarks for rscrypto public APIs.

mod common;

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ed25519_dalek::{Signer as _, SigningKey};
use fips203::{
  ml_kem_768 as FipsMlKem768,
  traits::{Decaps as _, Encaps as _, KeyGen as _},
};
use hkdf::Hkdf as RustCryptoHkdf;
use hmac::{Hmac, KeyInit};
use libcrux_ml_kem::mlkem768 as LibcruxMlKem768;
use p256::ecdsa::{Signature as P256OracleSignature, SigningKey as P256OracleSigningKey, signature::Verifier as _};
use p384::ecdsa::{Signature as P384OracleSignature, SigningKey as P384OracleSigningKey};
use rscrypto::{
  EcdsaP256Keypair, EcdsaP256PublicKey, EcdsaP256SecretKey, EcdsaP256Signature, EcdsaP384Keypair, EcdsaP384PublicKey,
  EcdsaP384SecretKey, EcdsaP384Signature, Ed25519Keypair, Ed25519PublicKey, Ed25519SecretKey, HkdfSha256, HkdfSha384,
  HmacSha256, HmacSha384, HmacSha512, Kem as _, Mac as _, MlKem768, MlKemError, Pbkdf2Sha256, Pbkdf2Sha512,
  X25519SecretKey,
};
use rustcrypto_ml_kem::{
  B32 as RustCryptoMlKemB32, DecapsulationKey as RustCryptoMlKemDecapsulationKey, KeyExport as _,
  MlKem768 as RustCryptoMlKem768, Seed as RustCryptoMlKemSeed, kem::Decapsulate as _,
};
use x25519_dalek::{PublicKey as DalekX25519PublicKey, StaticSecret as DalekX25519Secret};

type RustCryptoHmacSha256 = Hmac<sha2::Sha256>;
type RustCryptoHmacSha384 = Hmac<sha2::Sha384>;
type RustCryptoHmacSha512 = Hmac<sha2::Sha512>;
type RustCryptoHkdfSha256 = RustCryptoHkdf<sha2::Sha256>;
type RustCryptoHkdfSha384 = RustCryptoHkdf<sha2::Sha384>;

fn array_from_slice<const N: usize>(slice: &[u8]) -> [u8; N] {
  let mut out = [0u8; N];
  out.copy_from_slice(slice);
  out
}

fn deterministic_bytes<const N: usize>(offset: u8) -> [u8; N] {
  let mut out = [0u8; N];
  for (i, byte) in out.iter_mut().enumerate() {
    *byte = offset.wrapping_add(i as u8);
  }
  out
}

#[cfg(all(
  any(unix, windows),
  not(target_arch = "wasm32"),
  not(any(target_arch = "s390x", target_arch = "powerpc64"))
))]
macro_rules! aws_lc_bench {
  ($($tokens:tt)*) => {
    $($tokens)*
  };
}

#[cfg(not(all(
  any(unix, windows),
  not(target_arch = "wasm32"),
  not(any(target_arch = "s390x", target_arch = "powerpc64"))
)))]
macro_rules! aws_lc_bench {
  ($($tokens:tt)*) => {};
}

// `KeyType` newtype so `aws_lc_rs::hkdf::Prk::expand(...)` can produce a
// variable-length OKM matching the bench's `out_len`.
aws_lc_bench! {
  use aws_lc_rs::kem::{
    Ciphertext as AwsMlKemCiphertext, DecapsulationKey as AwsMlKemDecapsulationKey, ML_KEM_768 as AWS_ML_KEM_768,
  };

  struct AwsHkdfLen(usize);
  impl aws_lc_rs::hkdf::KeyType for AwsHkdfLen {
    fn len(&self) -> usize {
      self.0
    }
  }
}

/// `KeyType` newtype for ring's variable-length HKDF expand.
struct RingHkdfLen(usize);
impl ring::hkdf::KeyType for RingHkdfLen {
  fn len(&self) -> usize {
    self.0
  }
}

#[cfg(feature = "diag")]
fn print_auth_diag_once() {
  use std::sync::Once;

  static ONCE: Once = Once::new();
  ONCE.call_once(|| {
    use rscrypto::{Sha256, hashes::introspect::kernel_for};

    eprintln!("rscrypto-diag auth runtime_caps={}", rscrypto::platform::caps());
    eprintln!("rscrypto-diag auth static_caps={}", rscrypto::platform::caps_static());
    eprintln!(
      "rscrypto-diag auth target_features sha={} sha512={} avx2={} avx512f={}",
      cfg!(target_feature = "sha"),
      cfg!(target_feature = "sha512"),
      cfg!(target_feature = "avx2"),
      cfg!(target_feature = "avx512f")
    );
    eprintln!(
      "rscrypto-diag auth sha256_kernel 64={} 4096={} 1048576={}",
      kernel_for::<Sha256>(64),
      kernel_for::<Sha256>(4096),
      kernel_for::<Sha256>(1_048_576)
    );
  });
}

#[cfg(not(feature = "diag"))]
#[inline]
fn print_auth_diag_once() {}

fn hmac_sha256(c: &mut Criterion) {
  print_auth_diag_once();

  let inputs = common::comp_sizes();
  let key = [0x42u8; 32];
  aws_lc_bench! {
    let aws_key = aws_lc_rs::hmac::Key::new(aws_lc_rs::hmac::HMAC_SHA256, &key);
  }
  let ring_key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, &key);
  let mut g = c.benchmark_group("hmac-sha256");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      let mut mac = HmacSha256::new(&key);
      b.iter(|| {
        mac.update(black_box(d));
        let tag = mac.finalize();
        mac.reset();
        black_box(tag)
      })
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto", len), data, |b, d| {
      b.iter(|| {
        use hmac::Mac as _;

        let mut mac = RustCryptoHmacSha256::new_from_slice(black_box(&key)).unwrap();
        mac.update(black_box(d));
        black_box(mac.finalize().into_bytes())
      })
    });

    aws_lc_bench! {
      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), data, |b, d| {
        b.iter(|| black_box(aws_lc_rs::hmac::sign(&aws_key, black_box(d))))
      });
    }

    g.bench_with_input(BenchmarkId::new("ring", len), data, |b, d| {
      b.iter(|| black_box(ring::hmac::sign(&ring_key, black_box(d))))
    });
  }

  g.finish();
}

fn hmac_sha384(c: &mut Criterion) {
  print_auth_diag_once();

  let inputs = common::comp_sizes();
  let key = [0x42u8; 48];
  aws_lc_bench! {
    let aws_key = aws_lc_rs::hmac::Key::new(aws_lc_rs::hmac::HMAC_SHA384, &key);
  }
  let ring_key = ring::hmac::Key::new(ring::hmac::HMAC_SHA384, &key);
  let mut g = c.benchmark_group("hmac-sha384");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      let mut mac = HmacSha384::new(&key);
      b.iter(|| {
        mac.update(black_box(d));
        let tag = mac.finalize();
        mac.reset();
        black_box(tag)
      })
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto", len), data, |b, d| {
      let base_mac = RustCryptoHmacSha384::new_from_slice(&key).unwrap();
      b.iter(|| {
        use hmac::Mac as _;

        let mut mac = base_mac.clone();
        mac.update(black_box(d));
        black_box(mac.finalize().into_bytes())
      })
    });

    aws_lc_bench! {
      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), data, |b, d| {
        b.iter(|| black_box(aws_lc_rs::hmac::sign(&aws_key, black_box(d))))
      });
    }

    g.bench_with_input(BenchmarkId::new("ring", len), data, |b, d| {
      b.iter(|| black_box(ring::hmac::sign(&ring_key, black_box(d))))
    });
  }

  g.finish();
}

fn hmac_sha512(c: &mut Criterion) {
  print_auth_diag_once();

  let inputs = common::comp_sizes();
  let key = [0x42u8; 64];
  aws_lc_bench! {
    let aws_key = aws_lc_rs::hmac::Key::new(aws_lc_rs::hmac::HMAC_SHA512, &key);
  }
  let ring_key = ring::hmac::Key::new(ring::hmac::HMAC_SHA512, &key);
  let mut g = c.benchmark_group("hmac-sha512");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      let mut mac = HmacSha512::new(&key);
      b.iter(|| {
        mac.update(black_box(d));
        let tag = mac.finalize();
        mac.reset();
        black_box(tag)
      })
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto", len), data, |b, d| {
      let base_mac = RustCryptoHmacSha512::new_from_slice(&key).unwrap();
      b.iter(|| {
        use hmac::Mac as _;

        let mut mac = base_mac.clone();
        mac.update(black_box(d));
        black_box(mac.finalize().into_bytes())
      })
    });

    aws_lc_bench! {
      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), data, |b, d| {
        b.iter(|| black_box(aws_lc_rs::hmac::sign(&aws_key, black_box(d))))
      });
    }

    g.bench_with_input(BenchmarkId::new("ring", len), data, |b, d| {
      b.iter(|| black_box(ring::hmac::sign(&ring_key, black_box(d))))
    });
  }

  g.finish();
}

fn hmac_sha256_streaming(c: &mut Criterion) {
  print_auth_diag_once();

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

fn hmac_sha256_internal(c: &mut Criterion) {
  print_auth_diag_once();

  let data = common::random_bytes(4096);
  let key = [0x24u8; 32];
  aws_lc_bench! {
    let aws_key = aws_lc_rs::hmac::Key::new(aws_lc_rs::hmac::HMAC_SHA256, &key);
  }
  let ring_key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, &key);
  let mut g = c.benchmark_group("hmac-sha256/internal/fixed-message");

  for len in [32usize, 64, 256, 4096] {
    let msg = &data[..len];
    common::set_throughput(&mut g, len);

    g.bench_with_input(BenchmarkId::new("rscrypto-oneshot", len), msg, |b, d| {
      b.iter(|| black_box(HmacSha256::mac(black_box(&key), black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("rscrypto-stream-new", len), msg, |b, d| {
      b.iter(|| {
        let mut mac = HmacSha256::new(black_box(&key));
        mac.update(black_box(d));
        black_box(mac.finalize())
      })
    });

    g.bench_with_input(BenchmarkId::new("rscrypto-stream-reuse", len), msg, |b, d| {
      let mut mac = HmacSha256::new(&key);
      b.iter(|| {
        mac.update(black_box(d));
        let tag = mac.finalize();
        mac.reset();
        black_box(tag)
      })
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto-oneshot", len), msg, |b, d| {
      b.iter(|| {
        use hmac::Mac as _;

        let mut mac = RustCryptoHmacSha256::new_from_slice(black_box(&key)).unwrap();
        mac.update(black_box(d));
        black_box(mac.finalize().into_bytes())
      })
    });

    aws_lc_bench! {
      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), msg, |b, d| {
        b.iter(|| black_box(aws_lc_rs::hmac::sign(&aws_key, black_box(d))))
      });
    }

    g.bench_with_input(BenchmarkId::new("ring", len), msg, |b, d| {
      b.iter(|| black_box(ring::hmac::sign(&ring_key, black_box(d))))
    });
  }

  g.finish();
}

fn hkdf_sha256_expand(c: &mut Criterion) {
  print_auth_diag_once();

  let salt = [0x11u8; 32];
  let ikm = [0x22u8; 32];
  let info = [0x33u8; 48];
  let hkdf = HkdfSha256::new(&salt, &ikm);
  let rustcrypto = RustCryptoHkdfSha256::new(Some(&salt), &ikm);
  aws_lc_bench! {
    let aws_prk = aws_lc_rs::hkdf::Salt::new(aws_lc_rs::hkdf::HKDF_SHA256, &salt).extract(&ikm);
  }
  let ring_prk = ring::hkdf::Salt::new(ring::hkdf::HKDF_SHA256, &salt).extract(&ikm);
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

    aws_lc_bench! {
      g.bench_with_input(BenchmarkId::new("aws-lc-rs", out_len), &out_len, |b, &len| {
        let mut out = vec![0u8; len];
        b.iter(|| {
          aws_prk
            .expand(&[black_box(&info)], AwsHkdfLen(len))
            .unwrap()
            .fill(black_box(&mut out))
            .unwrap();
          black_box(out[0])
        })
      });
    }

    g.bench_with_input(BenchmarkId::new("ring", out_len), &out_len, |b, &len| {
      let mut out = vec![0u8; len];
      b.iter(|| {
        ring_prk
          .expand(&[black_box(&info)], RingHkdfLen(len))
          .unwrap()
          .fill(black_box(&mut out))
          .unwrap();
        black_box(out[0])
      })
    });
  }

  g.finish();
}

fn hkdf_sha384_expand(c: &mut Criterion) {
  print_auth_diag_once();

  let salt = [0x11u8; 48];
  let ikm = [0x22u8; 48];
  let info = [0x33u8; 80];
  let hkdf = HkdfSha384::new(&salt, &ikm);
  let rustcrypto = RustCryptoHkdfSha384::new(Some(&salt), &ikm);
  aws_lc_bench! {
    let aws_prk = aws_lc_rs::hkdf::Salt::new(aws_lc_rs::hkdf::HKDF_SHA384, &salt).extract(&ikm);
  }
  let ring_prk = ring::hkdf::Salt::new(ring::hkdf::HKDF_SHA384, &salt).extract(&ikm);
  let mut g = c.benchmark_group("hkdf-sha384/expand");

  for out_len in [48usize, 96, 256, 1024] {
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

    aws_lc_bench! {
      g.bench_with_input(BenchmarkId::new("aws-lc-rs", out_len), &out_len, |b, &len| {
        let mut out = vec![0u8; len];
        b.iter(|| {
          aws_prk
            .expand(&[black_box(&info)], AwsHkdfLen(len))
            .unwrap()
            .fill(black_box(&mut out))
            .unwrap();
          black_box(out[0])
        })
      });
    }

    g.bench_with_input(BenchmarkId::new("ring", out_len), &out_len, |b, &len| {
      let mut out = vec![0u8; len];
      b.iter(|| {
        ring_prk
          .expand(&[black_box(&info)], RingHkdfLen(len))
          .unwrap()
          .fill(black_box(&mut out))
          .unwrap();
        black_box(out[0])
      })
    });
  }

  g.finish();
}

fn pbkdf2_sha256_derive(c: &mut Criterion) {
  print_auth_diag_once();

  let password = [0x55u8; 32];
  let salt = [0x33u8; 16];
  let state = Pbkdf2Sha256::new(&password);

  for &iterations in &[1u32, 100, 1000] {
    let nz_iters = core::num::NonZeroU32::new(iterations).unwrap();
    let mut g = c.benchmark_group(format!("pbkdf2-sha256/iters={iterations}"));

    for &out_len in &[32usize, 64] {
      g.throughput(criterion::Throughput::Bytes(out_len as u64));

      g.bench_with_input(BenchmarkId::new("rscrypto", out_len), &out_len, |b, &len| {
        let mut out = vec![0u8; len];
        b.iter(|| {
          Pbkdf2Sha256::derive_key_primitive(black_box(&password), black_box(&salt), iterations, black_box(&mut out))
            .unwrap();
          black_box(out[0])
        })
      });

      g.bench_with_input(BenchmarkId::new("rustcrypto", out_len), &out_len, |b, &len| {
        let mut out = vec![0u8; len];
        b.iter(|| {
          pbkdf2::pbkdf2_hmac::<sha2::Sha256>(black_box(&password), black_box(&salt), iterations, black_box(&mut out));
          black_box(out[0])
        })
      });

      aws_lc_bench! {
        g.bench_with_input(BenchmarkId::new("aws-lc-rs", out_len), &out_len, |b, &len| {
          let mut out = vec![0u8; len];
          b.iter(|| {
            aws_lc_rs::pbkdf2::derive(
              aws_lc_rs::pbkdf2::PBKDF2_HMAC_SHA256,
              nz_iters,
              black_box(&salt),
              black_box(&password),
              black_box(&mut out),
            );
            black_box(out[0])
          })
        });
      }

      g.bench_with_input(BenchmarkId::new("ring", out_len), &out_len, |b, &len| {
        let mut out = vec![0u8; len];
        b.iter(|| {
          ring::pbkdf2::derive(
            ring::pbkdf2::PBKDF2_HMAC_SHA256,
            nz_iters,
            black_box(&salt),
            black_box(&password),
            black_box(&mut out),
          );
          black_box(out[0])
        })
      });
    }

    g.finish();

    let mut g_state = c.benchmark_group(format!("pbkdf2-sha256-state/iters={iterations}"));
    for &out_len in &[32usize, 64] {
      g_state.throughput(criterion::Throughput::Bytes(out_len as u64));
      g_state.bench_with_input(BenchmarkId::new("rscrypto", out_len), &out_len, |b, &len| {
        let mut out = vec![0u8; len];
        b.iter(|| {
          state.derive(black_box(&salt), iterations, black_box(&mut out)).unwrap();
          black_box(out[0])
        })
      });
    }
    g_state.finish();
  }
}

fn pbkdf2_sha256_internal(c: &mut Criterion) {
  print_auth_diag_once();

  let password = [0x55u8; 32];
  let salt = [0x33u8; 16];
  let state = Pbkdf2Sha256::new(&password);

  for &iterations in &[1u32, 100, 1000] {
    let nz_iters = core::num::NonZeroU32::new(iterations).unwrap();
    let mut g = c.benchmark_group(format!("pbkdf2-sha256/internal/iters={iterations}"));

    for &out_len in &[32usize, 64] {
      g.throughput(criterion::Throughput::Bytes(out_len as u64));

      g.bench_with_input(BenchmarkId::new("rscrypto-oneshot", out_len), &out_len, |b, &len| {
        let mut out = vec![0u8; len];
        b.iter(|| {
          Pbkdf2Sha256::derive_key_primitive(black_box(&password), black_box(&salt), iterations, black_box(&mut out))
            .unwrap();
          black_box(out[0])
        })
      });

      g.bench_with_input(BenchmarkId::new("rscrypto-state", out_len), &out_len, |b, &len| {
        let mut out = vec![0u8; len];
        b.iter(|| {
          state.derive(black_box(&salt), iterations, black_box(&mut out)).unwrap();
          black_box(out[0])
        })
      });

      g.bench_with_input(BenchmarkId::new("rustcrypto", out_len), &out_len, |b, &len| {
        let mut out = vec![0u8; len];
        b.iter(|| {
          pbkdf2::pbkdf2_hmac::<sha2::Sha256>(black_box(&password), black_box(&salt), iterations, black_box(&mut out));
          black_box(out[0])
        })
      });

      aws_lc_bench! {
        g.bench_with_input(BenchmarkId::new("aws-lc-rs", out_len), &out_len, |b, &len| {
          let mut out = vec![0u8; len];
          b.iter(|| {
            aws_lc_rs::pbkdf2::derive(
              aws_lc_rs::pbkdf2::PBKDF2_HMAC_SHA256,
              nz_iters,
              black_box(&salt),
              black_box(&password),
              black_box(&mut out),
            );
            black_box(out[0])
          })
        });
      }

      g.bench_with_input(BenchmarkId::new("ring", out_len), &out_len, |b, &len| {
        let mut out = vec![0u8; len];
        b.iter(|| {
          ring::pbkdf2::derive(
            ring::pbkdf2::PBKDF2_HMAC_SHA256,
            nz_iters,
            black_box(&salt),
            black_box(&password),
            black_box(&mut out),
          );
          black_box(out[0])
        })
      });
    }

    g.finish();
  }
}

fn pbkdf2_sha512_derive(c: &mut Criterion) {
  print_auth_diag_once();

  let password = [0x66u8; 48];
  let salt = [0x44u8; 16];
  let state = Pbkdf2Sha512::new(&password);

  for &iterations in &[1u32, 100, 1000] {
    let nz_iters = core::num::NonZeroU32::new(iterations).unwrap();
    let mut g = c.benchmark_group(format!("pbkdf2-sha512/iters={iterations}"));

    for &out_len in &[64usize, 128] {
      g.throughput(criterion::Throughput::Bytes(out_len as u64));

      g.bench_with_input(BenchmarkId::new("rscrypto", out_len), &out_len, |b, &len| {
        let mut out = vec![0u8; len];
        b.iter(|| {
          Pbkdf2Sha512::derive_key_primitive(black_box(&password), black_box(&salt), iterations, black_box(&mut out))
            .unwrap();
          black_box(out[0])
        })
      });

      g.bench_with_input(BenchmarkId::new("rustcrypto", out_len), &out_len, |b, &len| {
        let mut out = vec![0u8; len];
        b.iter(|| {
          pbkdf2::pbkdf2_hmac::<sha2::Sha512>(black_box(&password), black_box(&salt), iterations, black_box(&mut out));
          black_box(out[0])
        })
      });

      aws_lc_bench! {
        g.bench_with_input(BenchmarkId::new("aws-lc-rs", out_len), &out_len, |b, &len| {
          let mut out = vec![0u8; len];
          b.iter(|| {
            aws_lc_rs::pbkdf2::derive(
              aws_lc_rs::pbkdf2::PBKDF2_HMAC_SHA512,
              nz_iters,
              black_box(&salt),
              black_box(&password),
              black_box(&mut out),
            );
            black_box(out[0])
          })
        });
      }

      g.bench_with_input(BenchmarkId::new("ring", out_len), &out_len, |b, &len| {
        let mut out = vec![0u8; len];
        b.iter(|| {
          ring::pbkdf2::derive(
            ring::pbkdf2::PBKDF2_HMAC_SHA512,
            nz_iters,
            black_box(&salt),
            black_box(&password),
            black_box(&mut out),
          );
          black_box(out[0])
        })
      });
    }

    g.finish();

    let mut g_state = c.benchmark_group(format!("pbkdf2-sha512-state/iters={iterations}"));
    for &out_len in &[64usize, 128] {
      g_state.throughput(criterion::Throughput::Bytes(out_len as u64));
      g_state.bench_with_input(BenchmarkId::new("rscrypto", out_len), &out_len, |b, &len| {
        let mut out = vec![0u8; len];
        b.iter(|| {
          state.derive(black_box(&salt), iterations, black_box(&mut out)).unwrap();
          black_box(out[0])
        })
      });
    }
    g_state.finish();
  }
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

fn ecdsa_p256_verify(c: &mut Criterion) {
  let secret_bytes = [0x11u8; 32];
  let signing_key = P256OracleSigningKey::from_slice(&secret_bytes).unwrap();
  let verifying_key = signing_key.verifying_key();
  let sec1 = verifying_key.to_encoded_point(false);
  let public = EcdsaP256PublicKey::from_sec1_bytes(sec1.as_bytes()).unwrap();
  let ring_upk = ring::signature::UnparsedPublicKey::new(&ring::signature::ECDSA_P256_SHA256_FIXED, sec1.as_bytes());
  aws_lc_bench! {
    let aws_upk =
      aws_lc_rs::signature::UnparsedPublicKey::new(&aws_lc_rs::signature::ECDSA_P256_SHA256_FIXED, sec1.as_bytes());
  }

  let inputs = [0usize, 32, 1024, 16384]
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect::<Vec<_>>();
  let mut g = c.benchmark_group("ecdsa-p256/verify");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let oracle_signature: P256OracleSignature = signing_key.sign(data);
    let signature = EcdsaP256Signature::from_bytes(array_from_slice(oracle_signature.to_bytes().as_ref())).unwrap();

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| {
        black_box(&public).verify(black_box(d), black_box(&signature)).unwrap();
        black_box(())
      })
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto-p256", len), data, |b, d| {
      b.iter(|| {
        black_box(verifying_key)
          .verify(black_box(d), black_box(&oracle_signature))
          .unwrap();
        black_box(())
      })
    });

    g.bench_with_input(BenchmarkId::new("ring", len), data, |b, d| {
      b.iter(|| {
        ring_upk.verify(black_box(d), black_box(signature.as_bytes())).unwrap();
        black_box(())
      })
    });

    aws_lc_bench! {
      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), data, |b, d| {
        b.iter(|| {
          aws_upk.verify(black_box(d), black_box(signature.as_bytes())).unwrap();
          black_box(())
        })
      });
    }
  }

  g.finish();
}

fn ecdsa_p256_sign(c: &mut Criterion) {
  let secret_bytes = [0x11u8; 32];
  let secret = EcdsaP256SecretKey::from_bytes(secret_bytes).unwrap();
  let keypair = EcdsaP256Keypair::from_secret_key(secret);
  let blind = [0x5cu8; 64];
  let signing_key = P256OracleSigningKey::from_slice(&secret_bytes).unwrap();
  let sec1 = keypair.public_key().to_sec1_bytes();
  let ring_rng = ring::rand::SystemRandom::new();
  let ring_key = ring::signature::EcdsaKeyPair::from_private_key_and_public_key(
    &ring::signature::ECDSA_P256_SHA256_FIXED_SIGNING,
    &secret_bytes,
    &sec1,
    &ring_rng,
  )
  .unwrap();
  aws_lc_bench! {
    let aws_rng = aws_lc_rs::rand::SystemRandom::new();
    let aws_key = aws_lc_rs::signature::EcdsaKeyPair::from_private_key_and_public_key(
      &aws_lc_rs::signature::ECDSA_P256_SHA256_FIXED_SIGNING,
      &secret_bytes,
      &sec1,
    )
    .unwrap();
  }

  let inputs = [0usize, 32, 1024, 16384]
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect::<Vec<_>>();
  let mut g = c.benchmark_group("ecdsa-p256/sign");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto-deterministic", len), data, |b, d| {
      b.iter(|| black_box(black_box(&keypair).try_sign(black_box(d)).unwrap()))
    });

    g.bench_with_input(BenchmarkId::new("rscrypto-blinded", len), data, |b, d| {
      b.iter(|| {
        black_box(
          black_box(&keypair)
            .try_sign_blinded(black_box(d), |out| out.copy_from_slice(black_box(&blind)))
            .unwrap(),
        )
      })
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto-p256", len), data, |b, d| {
      b.iter(|| {
        let signature: P256OracleSignature = black_box(&signing_key).sign(black_box(d));
        black_box(signature)
      })
    });

    g.bench_with_input(BenchmarkId::new("ring", len), data, |b, d| {
      b.iter(|| black_box(ring_key.sign(&ring_rng, black_box(d)).unwrap()))
    });

    aws_lc_bench! {
      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), data, |b, d| {
        b.iter(|| black_box(aws_key.sign(&aws_rng, black_box(d)).unwrap()))
      });
    }
  }

  g.finish();
}

fn ecdsa_p384_verify(c: &mut Criterion) {
  let secret_bytes = [0x31u8; 48];
  let signing_key = P384OracleSigningKey::from_slice(&secret_bytes).unwrap();
  let verifying_key = signing_key.verifying_key();
  let sec1 = verifying_key.to_encoded_point(false);
  let public = EcdsaP384PublicKey::from_sec1_bytes(sec1.as_bytes()).unwrap();
  let ring_upk = ring::signature::UnparsedPublicKey::new(&ring::signature::ECDSA_P384_SHA384_FIXED, sec1.as_bytes());
  aws_lc_bench! {
    let aws_upk =
      aws_lc_rs::signature::UnparsedPublicKey::new(&aws_lc_rs::signature::ECDSA_P384_SHA384_FIXED, sec1.as_bytes());
  }

  let inputs = [0usize, 32, 1024, 16384]
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect::<Vec<_>>();
  let mut g = c.benchmark_group("ecdsa-p384/verify");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let oracle_signature: P384OracleSignature = signing_key.sign(data);
    let signature = EcdsaP384Signature::from_bytes(array_from_slice(oracle_signature.to_bytes().as_ref())).unwrap();

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| {
        black_box(&public).verify(black_box(d), black_box(&signature)).unwrap();
        black_box(())
      })
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto-p384", len), data, |b, d| {
      b.iter(|| {
        black_box(verifying_key)
          .verify(black_box(d), black_box(&oracle_signature))
          .unwrap();
        black_box(())
      })
    });

    g.bench_with_input(BenchmarkId::new("ring", len), data, |b, d| {
      b.iter(|| {
        ring_upk.verify(black_box(d), black_box(signature.as_bytes())).unwrap();
        black_box(())
      })
    });

    aws_lc_bench! {
      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), data, |b, d| {
        b.iter(|| {
          aws_upk.verify(black_box(d), black_box(signature.as_bytes())).unwrap();
          black_box(())
        })
      });
    }
  }

  g.finish();
}

fn ecdsa_p384_sign(c: &mut Criterion) {
  let secret_bytes = [0x31u8; 48];
  let secret = EcdsaP384SecretKey::from_bytes(secret_bytes).unwrap();
  let keypair = EcdsaP384Keypair::from_secret_key(secret);
  let blind = [0xa3u8; 96];
  let signing_key = P384OracleSigningKey::from_slice(&secret_bytes).unwrap();
  let sec1 = keypair.public_key().to_sec1_bytes();
  let ring_rng = ring::rand::SystemRandom::new();
  let ring_key = ring::signature::EcdsaKeyPair::from_private_key_and_public_key(
    &ring::signature::ECDSA_P384_SHA384_FIXED_SIGNING,
    &secret_bytes,
    &sec1,
    &ring_rng,
  )
  .unwrap();
  aws_lc_bench! {
    let aws_rng = aws_lc_rs::rand::SystemRandom::new();
    let aws_key = aws_lc_rs::signature::EcdsaKeyPair::from_private_key_and_public_key(
      &aws_lc_rs::signature::ECDSA_P384_SHA384_FIXED_SIGNING,
      &secret_bytes,
      &sec1,
    )
    .unwrap();
  }

  let inputs = [0usize, 32, 1024, 16384]
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect::<Vec<_>>();
  let mut g = c.benchmark_group("ecdsa-p384/sign");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto-deterministic", len), data, |b, d| {
      b.iter(|| black_box(black_box(&keypair).try_sign(black_box(d)).unwrap()))
    });

    g.bench_with_input(BenchmarkId::new("rscrypto-blinded", len), data, |b, d| {
      b.iter(|| {
        black_box(
          black_box(&keypair)
            .try_sign_blinded(black_box(d), |out| out.copy_from_slice(black_box(&blind)))
            .unwrap(),
        )
      })
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto-p384", len), data, |b, d| {
      b.iter(|| {
        let signature: P384OracleSignature = black_box(&signing_key).sign(black_box(d));
        black_box(signature)
      })
    });

    g.bench_with_input(BenchmarkId::new("ring", len), data, |b, d| {
      b.iter(|| black_box(ring_key.sign(&ring_rng, black_box(d)).unwrap()))
    });

    aws_lc_bench! {
      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), data, |b, d| {
        b.iter(|| black_box(aws_key.sign(&aws_rng, black_box(d)).unwrap()))
      });
    }
  }

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
  use dryoc::classic::crypto_sign::{crypto_sign_detached, crypto_sign_seed_keypair};

  let secret_bytes = [9u8; 32];
  let secret = Ed25519SecretKey::from_bytes(secret_bytes);
  let keypair = Ed25519Keypair::from_secret_key(secret);
  let signing_key = SigningKey::from_bytes(&secret_bytes);
  aws_lc_bench! {
    let aws_kp = aws_lc_rs::signature::Ed25519KeyPair::from_seed_unchecked(&secret_bytes).unwrap();
  }
  let ring_kp = ring::signature::Ed25519KeyPair::from_seed_unchecked(&secret_bytes).unwrap();
  let (_dryoc_pk, dryoc_sk) = crypto_sign_seed_keypair(&secret_bytes);
  let mut dryoc_sig: [u8; 64] = [0u8; 64];
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

    aws_lc_bench! {
      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), data, |b, d| {
        b.iter(|| black_box(aws_kp.sign(black_box(d))))
      });
    }

    g.bench_with_input(BenchmarkId::new("ring", len), data, |b, d| {
      b.iter(|| black_box(ring_kp.sign(black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("dryoc", len), data, |b, d| {
      b.iter(|| {
        crypto_sign_detached(&mut dryoc_sig, black_box(d), &dryoc_sk).unwrap();
        black_box(&dryoc_sig);
      })
    });
  }

  g.finish();
}

fn ed25519_verify(c: &mut Criterion) {
  aws_lc_bench! {
    use aws_lc_rs::signature::KeyPair as _;
  }
  use dryoc::classic::crypto_sign::{crypto_sign_detached, crypto_sign_seed_keypair, crypto_sign_verify_detached};
  use ring::signature::KeyPair as _;

  let secret_bytes = [13u8; 32];
  let secret = Ed25519SecretKey::from_bytes(secret_bytes);
  let keypair = Ed25519Keypair::from_secret_key(secret);
  let public: Ed25519PublicKey = keypair.public_key();
  let signing_key = SigningKey::from_bytes(&secret_bytes);
  let verifying_key = signing_key.verifying_key();
  aws_lc_bench! {
    let aws_kp = aws_lc_rs::signature::Ed25519KeyPair::from_seed_unchecked(&secret_bytes).unwrap();
    let aws_pubkey: Vec<u8> = aws_kp.public_key().as_ref().to_vec();
    let aws_upk = aws_lc_rs::signature::UnparsedPublicKey::new(&aws_lc_rs::signature::ED25519, aws_pubkey);
  }
  let ring_kp = ring::signature::Ed25519KeyPair::from_seed_unchecked(&secret_bytes).unwrap();
  let ring_pubkey: Vec<u8> = ring_kp.public_key().as_ref().to_vec();
  let ring_upk = ring::signature::UnparsedPublicKey::new(&ring::signature::ED25519, ring_pubkey);
  let (dryoc_pk, dryoc_sk) = crypto_sign_seed_keypair(&secret_bytes);
  let inputs = [0usize, 32, 1024, 16384]
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect::<Vec<_>>();
  let mut g = c.benchmark_group("ed25519/verify");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let ours = keypair.sign(data);
    let dalek = signing_key.sign(data);
    aws_lc_bench! {
      let aws_sig = aws_kp.sign(data);
    }
    let ring_sig = ring_kp.sign(data);
    let mut dryoc_sig: [u8; 64] = [0u8; 64];
    crypto_sign_detached(&mut dryoc_sig, data, &dryoc_sk).unwrap();

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

    aws_lc_bench! {
      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), data, |b, d| {
        b.iter(|| {
          aws_upk.verify(black_box(d), aws_sig.as_ref()).unwrap();
          black_box(())
        })
      });
    }

    g.bench_with_input(BenchmarkId::new("ring", len), data, |b, d| {
      b.iter(|| {
        ring_upk.verify(black_box(d), ring_sig.as_ref()).unwrap();
        black_box(())
      })
    });

    g.bench_with_input(BenchmarkId::new("dryoc", len), data, |b, d| {
      b.iter(|| {
        crypto_sign_verify_detached(&dryoc_sig, black_box(d), &dryoc_pk).unwrap();
        black_box(())
      })
    });
  }

  g.finish();
}

// `ring` is omitted from x25519 benches: ring 0.17 only exposes
// `EphemeralPrivateKey` (consumed by `agree_ephemeral`) and provides no
// reusable static-key API. Including it would force a full keygen-and-discard
// per iteration, which is not apples-to-apples against the static-key DH
// path that rscrypto / dalek / aws-lc-rs / dryoc all share.
fn x25519_public_key(c: &mut Criterion) {
  use dryoc::classic::crypto_core::crypto_scalarmult_base;

  let secret_bytes = [0x2au8; 32];
  let mut g = c.benchmark_group("x25519/public-key-from-secret");

  g.bench_function("rscrypto", |b| {
    b.iter(|| {
      let secret = X25519SecretKey::from_bytes(*black_box(&secret_bytes));
      black_box(secret.public_key())
    })
  });

  g.bench_function("dalek", |b| {
    b.iter(|| {
      let secret = DalekX25519Secret::from(*black_box(&secret_bytes));
      black_box(DalekX25519PublicKey::from(&secret))
    })
  });

  aws_lc_bench! {
    g.bench_function("aws-lc-rs", |b| {
      b.iter(|| {
        let priv_key =
          aws_lc_rs::agreement::PrivateKey::from_private_key(&aws_lc_rs::agreement::X25519, black_box(&secret_bytes))
            .unwrap();
        black_box(priv_key.compute_public_key().unwrap())
      })
    });
  }

  g.bench_function("dryoc", |b| {
    let mut public = [0u8; 32];
    b.iter(|| {
      crypto_scalarmult_base(&mut public, black_box(&secret_bytes));
      black_box(public)
    })
  });

  g.finish();
}

fn x25519_diffie_hellman(c: &mut Criterion) {
  use dryoc::classic::crypto_core::{crypto_scalarmult, crypto_scalarmult_base};

  let alice_bytes = [0x18u8; 32];
  let bob_bytes = [0x34u8; 32];

  let alice = X25519SecretKey::from_bytes(alice_bytes);
  let bob_public = X25519SecretKey::from_bytes(bob_bytes).public_key();
  let dalek_alice = DalekX25519Secret::from(alice_bytes);
  let dalek_bob_public = DalekX25519PublicKey::from(&DalekX25519Secret::from(bob_bytes));
  aws_lc_bench! {
    let aws_alice =
      aws_lc_rs::agreement::PrivateKey::from_private_key(&aws_lc_rs::agreement::X25519, &alice_bytes).unwrap();
    let aws_bob_pub_bytes: [u8; 32] = {
      let bob_priv =
        aws_lc_rs::agreement::PrivateKey::from_private_key(&aws_lc_rs::agreement::X25519, &bob_bytes).unwrap();
      let pk = bob_priv.compute_public_key().unwrap();
      let mut out = [0u8; 32];
      out.copy_from_slice(pk.as_ref());
      out
    };
    let aws_bob_unparsed =
      aws_lc_rs::agreement::UnparsedPublicKey::new(&aws_lc_rs::agreement::X25519, aws_bob_pub_bytes);
  }
  let mut dryoc_bob_pub = [0u8; 32];
  crypto_scalarmult_base(&mut dryoc_bob_pub, &bob_bytes);
  let mut g = c.benchmark_group("x25519/diffie-hellman");

  g.bench_function("rscrypto", |b| {
    b.iter(|| black_box(black_box(&alice).diffie_hellman(black_box(&bob_public)).unwrap()))
  });

  g.bench_function("dalek", |b| {
    b.iter(|| black_box(black_box(&dalek_alice).diffie_hellman(black_box(&dalek_bob_public))))
  });

  aws_lc_bench! {
    g.bench_function("aws-lc-rs", |b| {
      b.iter(|| {
        let shared = aws_lc_rs::agreement::agree(black_box(&aws_alice), black_box(&aws_bob_unparsed), (), |bytes| {
          let mut out = [0u8; 32];
          out.copy_from_slice(bytes);
          Ok::<[u8; 32], ()>(out)
        })
        .unwrap();
        black_box(shared)
      })
    });
  }

  g.bench_function("dryoc", |b| {
    let mut shared = [0u8; 32];
    b.iter(|| {
      crypto_scalarmult(&mut shared, black_box(&alice_bytes), black_box(&dryoc_bob_pub));
      black_box(shared)
    })
  });

  g.finish();
}

fn mlkem768_keygen(c: &mut Criterion) {
  let key_random = deterministic_bytes::<{ MlKem768::KEY_GENERATION_RANDOM_SIZE }>(0x10);
  let d = array_from_slice::<32>(&key_random[..32]);
  let z = array_from_slice::<32>(&key_random[32..]);
  let mut g = c.benchmark_group("mlkem768/keygen");

  g.bench_function("rscrypto", |b| {
    b.iter(|| {
      MlKem768::generate_keypair(|out| {
        out.copy_from_slice(black_box(&key_random));
        Ok::<(), MlKemError>(())
      })
      .unwrap()
    })
  });

  g.bench_function("libcrux", |b| {
    b.iter(|| LibcruxMlKem768::generate_key_pair(black_box(key_random)))
  });

  aws_lc_bench! {
    g.bench_function("aws-lc-rs", |b| {
      b.iter(|| black_box(AwsMlKemDecapsulationKey::generate(&AWS_ML_KEM_768).unwrap()))
    });
  }

  g.bench_function("fips203", |b| {
    b.iter(|| FipsMlKem768::KG::keygen_from_seed(black_box(d), black_box(z)))
  });

  g.bench_function("rustcrypto", |b| {
    b.iter(|| {
      let dk = RustCryptoMlKemDecapsulationKey::<RustCryptoMlKem768>::from_seed(RustCryptoMlKemSeed::from(key_random));
      black_box(dk.encapsulation_key().to_bytes());
      black_box(dk)
    })
  });

  g.finish();
}

fn mlkem768_encapsulate(c: &mut Criterion) {
  let key_random = deterministic_bytes::<{ MlKem768::KEY_GENERATION_RANDOM_SIZE }>(0x20);
  let encaps_random = deterministic_bytes::<{ MlKem768::ENCAPSULATION_RANDOM_SIZE }>(0x80);
  let (ek, _) = MlKem768::generate_keypair(|out| {
    out.copy_from_slice(&key_random);
    Ok::<(), MlKemError>(())
  })
  .unwrap();
  let (fips_ek, _) = FipsMlKem768::KG::keygen_from_seed(
    array_from_slice::<32>(&key_random[..32]),
    array_from_slice::<32>(&key_random[32..]),
  );
  let rustcrypto_dk =
    RustCryptoMlKemDecapsulationKey::<RustCryptoMlKem768>::from_seed(RustCryptoMlKemSeed::from(key_random));
  let rustcrypto_ek = rustcrypto_dk.encapsulation_key().clone();
  let libcrux_keypair = LibcruxMlKem768::generate_key_pair(key_random);
  let libcrux_ek = libcrux_keypair.public_key().clone();
  aws_lc_bench! {
    let aws_dk = AwsMlKemDecapsulationKey::generate(&AWS_ML_KEM_768).unwrap();
    let aws_ek = aws_dk.encapsulation_key().unwrap();
  }
  let mut g = c.benchmark_group("mlkem768/encapsulate");

  g.bench_function("rscrypto", |b| {
    b.iter(|| {
      MlKem768::encapsulate(black_box(&ek), |out| {
        out.copy_from_slice(black_box(&encaps_random));
        Ok::<(), MlKemError>(())
      })
      .unwrap()
    })
  });

  g.bench_function("libcrux", |b| {
    b.iter(|| {
      black_box(LibcruxMlKem768::encapsulate(
        black_box(&libcrux_ek),
        black_box(encaps_random),
      ))
    })
  });

  aws_lc_bench! {
    g.bench_function("aws-lc-rs", |b| {
      b.iter(|| black_box(aws_ek.encapsulate().unwrap()))
    });
  }

  g.bench_function("fips203", |b| {
    b.iter(|| black_box(fips_ek.encaps_from_seed(black_box(&encaps_random))))
  });

  g.bench_function("rustcrypto", |b| {
    b.iter(|| black_box(rustcrypto_ek.encapsulate_deterministic(black_box(&RustCryptoMlKemB32::from(encaps_random)))))
  });

  g.finish();
}

fn mlkem768_decapsulate(c: &mut Criterion) {
  let key_random = deterministic_bytes::<{ MlKem768::KEY_GENERATION_RANDOM_SIZE }>(0x30);
  let encaps_random = deterministic_bytes::<{ MlKem768::ENCAPSULATION_RANDOM_SIZE }>(0x90);
  let (ek, dk) = MlKem768::generate_keypair(|out| {
    out.copy_from_slice(&key_random);
    Ok::<(), MlKemError>(())
  })
  .unwrap();
  let (ciphertext, _) = MlKem768::encapsulate(&ek, |out| {
    out.copy_from_slice(&encaps_random);
    Ok::<(), MlKemError>(())
  })
  .unwrap();
  let (fips_ek, fips_dk) = FipsMlKem768::KG::keygen_from_seed(
    array_from_slice::<32>(&key_random[..32]),
    array_from_slice::<32>(&key_random[32..]),
  );
  let (_, fips_ciphertext) = fips_ek.encaps_from_seed(&encaps_random);
  let rustcrypto_dk =
    RustCryptoMlKemDecapsulationKey::<RustCryptoMlKem768>::from_seed(RustCryptoMlKemSeed::from(key_random));
  let (rustcrypto_ciphertext, _) = rustcrypto_dk
    .encapsulation_key()
    .encapsulate_deterministic(&RustCryptoMlKemB32::from(encaps_random));
  let libcrux_keypair = LibcruxMlKem768::generate_key_pair(key_random);
  let libcrux_ek = libcrux_keypair.public_key().clone();
  let libcrux_dk = libcrux_keypair.private_key().clone();
  let (libcrux_ciphertext, _) = LibcruxMlKem768::encapsulate(&libcrux_ek, encaps_random);
  aws_lc_bench! {
    let aws_dk = AwsMlKemDecapsulationKey::generate(&AWS_ML_KEM_768).unwrap();
    let aws_ek = aws_dk.encapsulation_key().unwrap();
    let (aws_ciphertext, _) = aws_ek.encapsulate().unwrap();
  }
  let mut g = c.benchmark_group("mlkem768/decapsulate");

  g.bench_function("rscrypto", |b| {
    b.iter(|| MlKem768::decapsulate(black_box(&dk), black_box(&ciphertext)).unwrap())
  });

  g.bench_function("libcrux", |b| {
    b.iter(|| {
      black_box(LibcruxMlKem768::decapsulate(
        black_box(&libcrux_dk),
        black_box(&libcrux_ciphertext),
      ))
    })
  });

  aws_lc_bench! {
    g.bench_function("aws-lc-rs", |b| {
      b.iter(|| {
        black_box(
          aws_dk
            .decapsulate(AwsMlKemCiphertext::from(black_box(aws_ciphertext.as_ref())))
            .unwrap(),
        )
      })
    });
  }

  g.bench_function("fips203", |b| {
    b.iter(|| black_box(fips_dk.try_decaps(black_box(&fips_ciphertext)).unwrap()))
  });

  g.bench_function("rustcrypto", |b| {
    b.iter(|| black_box(rustcrypto_dk.decapsulate(black_box(&rustcrypto_ciphertext))))
  });

  g.finish();
}

criterion_group!(
  benches,
  hmac_sha256,
  hmac_sha384,
  hmac_sha512,
  hmac_sha256_streaming,
  hmac_sha256_internal,
  hkdf_sha256_expand,
  hkdf_sha384_expand,
  pbkdf2_sha256_derive,
  pbkdf2_sha256_internal,
  pbkdf2_sha512_derive,
  ecdsa_p256_sign,
  ecdsa_p256_verify,
  ecdsa_p384_sign,
  ecdsa_p384_verify,
  ed25519_public_key,
  ed25519_keypair_from_secret,
  ed25519_sign,
  ed25519_verify,
  x25519_public_key,
  x25519_diffie_hellman,
  mlkem768_keygen,
  mlkem768_encapsulate,
  mlkem768_decapsulate
);
criterion_main!(benches);
