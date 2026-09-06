//! Auth benchmarks for rscrypto public APIs.

mod common;

use core::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use ed25519_dalek::{Signer as _, SigningKey};
use fips203::{
  ml_kem_512 as FipsMlKem512, ml_kem_768 as FipsMlKem768, ml_kem_1024 as FipsMlKem1024,
  traits::{Decaps as _, Encaps as _, KeyGen as _},
};
use hkdf::Hkdf as RustCryptoHkdf;
use hmac::{Hmac, KeyInit};
use libcrux_ml_kem::{mlkem512 as LibcruxMlKem512, mlkem768 as LibcruxMlKem768, mlkem1024 as LibcruxMlKem1024};
use p256::SecretKey as P256OracleSecretKey;
use p256::ecdsa::{Signature as P256OracleSignature, SigningKey as P256OracleSigningKey};
use p256::elliptic_curve::sec1::ToSec1Point as _;
use p384::SecretKey as P384OracleSecretKey;
use p384::ecdsa::{Signature as P384OracleSignature, SigningKey as P384OracleSigningKey};
use rscrypto::{
  EcdsaP256Keypair, EcdsaP256PublicKey, EcdsaP256SecretKey, EcdsaP256Signature, EcdsaP384Keypair, EcdsaP384PublicKey,
  EcdsaP384SecretKey, EcdsaP384Signature, Ed25519Keypair, Ed25519PublicKey, Ed25519SecretKey, HkdfSha256, HkdfSha384,
  HmacSha256, HmacSha384, HmacSha512, Kem as _, Mac as _, MlKem512, MlKem768, MlKem1024, MlKemError,
  P256EphemeralSecret, P256PublicKey, Pbkdf2Sha256, Pbkdf2Sha512, X25519SecretKey,
};
use rustcrypto_ml_kem::{
  B32 as RustCryptoMlKemB32, DecapsulationKey as RustCryptoMlKemDecapsulationKey, KeyExport as _,
  MlKem512 as RustCryptoMlKem512, MlKem768 as RustCryptoMlKem768, MlKem1024 as RustCryptoMlKem1024,
  Seed as RustCryptoMlKemSeed, kem::Decapsulate as _,
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
    let low_index = u8::try_from(i & usize::from(u8::MAX)).expect("masked deterministic-byte index must fit u8");
    *byte = offset.wrapping_add(low_index);
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

#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
macro_rules! ring_p256_bench {
  ($($tokens:tt)*) => {
    $($tokens)*
  };
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
macro_rules! ring_p256_bench {
  ($($tokens:tt)*) => {};
}

// `KeyType` newtype so `aws_lc_rs::hkdf::Prk::expand(...)` can produce a
// variable-length OKM matching the bench's `out_len`.
aws_lc_bench! {
  use aws_lc_rs::kem::{
    Ciphertext as AwsMlKemCiphertext, DecapsulationKey as AwsMlKemDecapsulationKey, ML_KEM_512 as AWS_ML_KEM_512,
    ML_KEM_768 as AWS_ML_KEM_768, ML_KEM_1024 as AWS_ML_KEM_1024,
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
      let base_mac =
        RustCryptoHmacSha256::new_from_slice(&key).expect("valid authentication benchmark operation must succeed");
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
      let base_mac =
        RustCryptoHmacSha384::new_from_slice(&key).expect("valid authentication benchmark operation must succeed");
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
      let base_mac =
        RustCryptoHmacSha512::new_from_slice(&key).expect("valid authentication benchmark operation must succeed");
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

        let mut mac =
          RustCryptoHmacSha256::new_from_slice(&key).expect("valid authentication benchmark operation must succeed");
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

        let mut mac = RustCryptoHmacSha256::new_from_slice(black_box(&key))
          .expect("valid authentication benchmark operation must succeed");
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
        hkdf
          .expand(black_box(&info), black_box(&mut out))
          .expect("valid authentication benchmark operation must succeed");
        black_box(out[0])
      })
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto", out_len), &out_len, |b, &len| {
      let mut out = vec![0u8; len];
      b.iter(|| {
        rustcrypto
          .expand(black_box(&info), black_box(&mut out))
          .expect("valid authentication benchmark operation must succeed");
        black_box(out[0])
      })
    });

    aws_lc_bench! {
      g.bench_with_input(BenchmarkId::new("aws-lc-rs", out_len), &out_len, |b, &len| {
        let mut out = vec![0u8; len];
        b.iter(|| {
          aws_prk
            .expand(&[black_box(&info)], AwsHkdfLen(len))
            .expect("valid authentication benchmark operation must succeed")
            .fill(black_box(&mut out))
            .expect("valid authentication benchmark operation must succeed");
          black_box(out[0])
        })
      });
    }

    g.bench_with_input(BenchmarkId::new("ring", out_len), &out_len, |b, &len| {
      let mut out = vec![0u8; len];
      b.iter(|| {
        ring_prk
          .expand(&[black_box(&info)], RingHkdfLen(len))
          .expect("valid authentication benchmark operation must succeed")
          .fill(black_box(&mut out))
          .expect("valid authentication benchmark operation must succeed");
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
        hkdf
          .expand(black_box(&info), black_box(&mut out))
          .expect("valid authentication benchmark operation must succeed");
        black_box(out[0])
      })
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto", out_len), &out_len, |b, &len| {
      let mut out = vec![0u8; len];
      b.iter(|| {
        rustcrypto
          .expand(black_box(&info), black_box(&mut out))
          .expect("valid authentication benchmark operation must succeed");
        black_box(out[0])
      })
    });

    aws_lc_bench! {
      g.bench_with_input(BenchmarkId::new("aws-lc-rs", out_len), &out_len, |b, &len| {
        let mut out = vec![0u8; len];
        b.iter(|| {
          aws_prk
            .expand(&[black_box(&info)], AwsHkdfLen(len))
            .expect("valid authentication benchmark operation must succeed")
            .fill(black_box(&mut out))
            .expect("valid authentication benchmark operation must succeed");
          black_box(out[0])
        })
      });
    }

    g.bench_with_input(BenchmarkId::new("ring", out_len), &out_len, |b, &len| {
      let mut out = vec![0u8; len];
      b.iter(|| {
        ring_prk
          .expand(&[black_box(&info)], RingHkdfLen(len))
          .expect("valid authentication benchmark operation must succeed")
          .fill(black_box(&mut out))
          .expect("valid authentication benchmark operation must succeed");
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
    let nz_iters =
      core::num::NonZeroU32::new(iterations).expect("valid authentication benchmark operation must succeed");
    let mut g = c.benchmark_group(format!("pbkdf2-sha256/iters={iterations}"));

    for &out_len in &[32usize, 64] {
      g.throughput(criterion::Throughput::Bytes(out_len as u64));

      g.bench_with_input(BenchmarkId::new("rscrypto", out_len), &out_len, |b, &len| {
        let mut out = vec![0u8; len];
        b.iter(|| {
          Pbkdf2Sha256::derive_key_primitive(black_box(&password), black_box(&salt), iterations, black_box(&mut out))
            .expect("valid authentication benchmark operation must succeed");
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
          state
            .derive(black_box(&salt), iterations, black_box(&mut out))
            .expect("valid authentication benchmark operation must succeed");
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
    let nz_iters =
      core::num::NonZeroU32::new(iterations).expect("valid authentication benchmark operation must succeed");
    let mut g = c.benchmark_group(format!("pbkdf2-sha256/internal/iters={iterations}"));

    for &out_len in &[32usize, 64] {
      g.throughput(criterion::Throughput::Bytes(out_len as u64));

      g.bench_with_input(BenchmarkId::new("rscrypto-oneshot", out_len), &out_len, |b, &len| {
        let mut out = vec![0u8; len];
        b.iter(|| {
          Pbkdf2Sha256::derive_key_primitive(black_box(&password), black_box(&salt), iterations, black_box(&mut out))
            .expect("valid authentication benchmark operation must succeed");
          black_box(out[0])
        })
      });

      g.bench_with_input(BenchmarkId::new("rscrypto-state", out_len), &out_len, |b, &len| {
        let mut out = vec![0u8; len];
        b.iter(|| {
          state
            .derive(black_box(&salt), iterations, black_box(&mut out))
            .expect("valid authentication benchmark operation must succeed");
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
    let nz_iters =
      core::num::NonZeroU32::new(iterations).expect("valid authentication benchmark operation must succeed");
    let mut g = c.benchmark_group(format!("pbkdf2-sha512/iters={iterations}"));

    for &out_len in &[64usize, 128] {
      g.throughput(criterion::Throughput::Bytes(out_len as u64));

      g.bench_with_input(BenchmarkId::new("rscrypto", out_len), &out_len, |b, &len| {
        let mut out = vec![0u8; len];
        b.iter(|| {
          Pbkdf2Sha512::derive_key_primitive(black_box(&password), black_box(&salt), iterations, black_box(&mut out))
            .expect("valid authentication benchmark operation must succeed");
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
          state
            .derive(black_box(&salt), iterations, black_box(&mut out))
            .expect("valid authentication benchmark operation must succeed");
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
  let signing_key =
    P256OracleSigningKey::from_slice(&secret_bytes).expect("valid authentication benchmark operation must succeed");
  let verifying_key = signing_key.verifying_key();
  let sec1 = EcdsaP256SecretKey::from_bytes(secret_bytes)
    .expect("valid authentication benchmark operation must succeed")
    .public_key()
    .to_sec1_bytes();
  let public = EcdsaP256PublicKey::from_sec1_bytes(sec1.as_slice())
    .expect("valid authentication benchmark operation must succeed");
  let ring_upk = ring::signature::UnparsedPublicKey::new(&ring::signature::ECDSA_P256_SHA256_FIXED, sec1.as_slice());
  aws_lc_bench! {
    let aws_upk =
      aws_lc_rs::signature::UnparsedPublicKey::new(&aws_lc_rs::signature::ECDSA_P256_SHA256_FIXED, sec1.as_slice());
  }

  let inputs = [0usize, 32, 1024, 16384]
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect::<Vec<_>>();
  let mut g = c.benchmark_group("ecdsa-p256/verify");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let oracle_signature: P256OracleSignature = p256::ecdsa::signature::Signer::sign(&signing_key, data);
    let signature = EcdsaP256Signature::from_bytes(array_from_slice(oracle_signature.to_bytes().as_ref()))
      .expect("valid authentication benchmark operation must succeed");

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| {
        black_box(&public)
          .verify(black_box(d), black_box(&signature))
          .expect("valid authentication benchmark operation must succeed");
        black_box(())
      })
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto-p256", len), data, |b, d| {
      b.iter(|| {
        p256::ecdsa::signature::Verifier::verify(black_box(verifying_key), black_box(d), black_box(&oracle_signature))
          .expect("valid authentication benchmark operation must succeed");
        black_box(())
      })
    });

    g.bench_with_input(BenchmarkId::new("ring", len), data, |b, d| {
      b.iter(|| {
        ring_upk
          .verify(black_box(d), black_box(signature.as_bytes()))
          .expect("valid authentication benchmark operation must succeed");
        black_box(())
      })
    });

    aws_lc_bench! {
      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), data, |b, d| {
        b.iter(|| {
          aws_upk.verify(black_box(d), black_box(signature.as_bytes())).expect("valid authentication benchmark operation must succeed");
          black_box(())
        })
      });
    }
  }

  g.finish();
}

fn ecdsa_p256_sign(c: &mut Criterion) {
  let secret_bytes = [0x11u8; 32];
  let secret =
    EcdsaP256SecretKey::from_bytes(secret_bytes).expect("valid authentication benchmark operation must succeed");
  let keypair = EcdsaP256Keypair::from_secret_key(secret);
  let blind = [0x5cu8; 64];
  let signing_key =
    P256OracleSigningKey::from_slice(&secret_bytes).expect("valid authentication benchmark operation must succeed");
  let sec1 = keypair.public_key().to_sec1_bytes();
  let ring_rng = ring::rand::SystemRandom::new();
  let ring_key = ring::signature::EcdsaKeyPair::from_private_key_and_public_key(
    &ring::signature::ECDSA_P256_SHA256_FIXED_SIGNING,
    &secret_bytes,
    &sec1,
    &ring_rng,
  )
  .expect("valid authentication benchmark operation must succeed");
  aws_lc_bench! {
    let aws_rng = aws_lc_rs::rand::SystemRandom::new();
    let aws_key = aws_lc_rs::signature::EcdsaKeyPair::from_private_key_and_public_key(
      &aws_lc_rs::signature::ECDSA_P256_SHA256_FIXED_SIGNING,
      &secret_bytes,
      &sec1,
    )
    .expect("valid authentication benchmark operation must succeed");
  }

  let inputs = [0usize, 32, 1024, 16384]
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect::<Vec<_>>();
  let mut g = c.benchmark_group("ecdsa-p256/sign");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto-deterministic", len), data, |b, d| {
      b.iter(|| {
        black_box(
          black_box(&keypair)
            .try_sign(black_box(d))
            .expect("valid authentication benchmark operation must succeed"),
        )
      })
    });

    g.bench_with_input(BenchmarkId::new("rscrypto-blinded", len), data, |b, d| {
      b.iter(|| {
        black_box(
          black_box(&keypair)
            .try_sign_blinded_with(black_box(d), |out| {
              out.copy_from_slice(black_box(&blind));
              Ok::<(), core::convert::Infallible>(())
            })
            .expect("valid authentication benchmark operation must succeed"),
        )
      })
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto-p256", len), data, |b, d| {
      b.iter(|| {
        let signature: P256OracleSignature =
          p256::ecdsa::signature::Signer::sign(black_box(&signing_key), black_box(d));
        black_box(signature)
      })
    });

    g.bench_with_input(BenchmarkId::new("ring", len), data, |b, d| {
      b.iter(|| {
        black_box(
          ring_key
            .sign(&ring_rng, black_box(d))
            .expect("valid authentication benchmark operation must succeed"),
        )
      })
    });

    aws_lc_bench! {
      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), data, |b, d| {
        b.iter(|| black_box(aws_key.sign(&aws_rng, black_box(d)).expect("valid authentication benchmark operation must succeed")))
      });
    }
  }

  g.finish();
}

fn ecdsa_p256_public_key(c: &mut Criterion) {
  let secret_bytes = [0x11u8; 32];
  let secret =
    EcdsaP256SecretKey::from_bytes(secret_bytes).expect("valid authentication benchmark operation must succeed");
  let blind = [0x5cu8; 64];
  let oracle =
    P256OracleSecretKey::from_slice(&secret_bytes).expect("valid authentication benchmark operation must succeed");
  let expected = secret.public_key().to_sec1_bytes();
  assert_eq!(oracle.public_key().to_sec1_bytes().as_ref(), expected);

  let mut g = c.benchmark_group("ecdsa-p256/public-key");
  g.bench_function("rscrypto-blinded", |b| {
    b.iter(|| {
      black_box(
        black_box(&secret)
          .try_public_key_blinded_with(|out| {
            out.copy_from_slice(black_box(&blind));
            Ok::<(), core::convert::Infallible>(())
          })
          .expect("valid authentication benchmark operation must succeed"),
      )
    })
  });
  g.bench_function("rustcrypto-p256", |b| {
    b.iter(|| black_box(black_box(&oracle).public_key()))
  });
  g.finish();
}

fn ecdsa_p384_verify(c: &mut Criterion) {
  let secret_bytes = [0x31u8; 48];
  let signing_key =
    P384OracleSigningKey::from_slice(&secret_bytes).expect("valid authentication benchmark operation must succeed");
  let verifying_key = signing_key.verifying_key();
  let sec1 = EcdsaP384SecretKey::from_bytes(secret_bytes)
    .expect("valid authentication benchmark operation must succeed")
    .public_key()
    .to_sec1_bytes();
  let public = EcdsaP384PublicKey::from_sec1_bytes(sec1.as_slice())
    .expect("valid authentication benchmark operation must succeed");
  let ring_upk = ring::signature::UnparsedPublicKey::new(&ring::signature::ECDSA_P384_SHA384_FIXED, sec1.as_slice());
  aws_lc_bench! {
    let aws_upk =
      aws_lc_rs::signature::UnparsedPublicKey::new(&aws_lc_rs::signature::ECDSA_P384_SHA384_FIXED, sec1.as_slice());
  }

  let inputs = [0usize, 32, 1024, 16384]
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect::<Vec<_>>();
  let mut g = c.benchmark_group("ecdsa-p384/verify");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let oracle_signature: P384OracleSignature = p384::ecdsa::signature::Signer::sign(&signing_key, data);
    let signature = EcdsaP384Signature::from_bytes(array_from_slice(oracle_signature.to_bytes().as_ref()))
      .expect("valid authentication benchmark operation must succeed");

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| {
        black_box(&public)
          .verify(black_box(d), black_box(&signature))
          .expect("valid authentication benchmark operation must succeed");
        black_box(())
      })
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto-p384", len), data, |b, d| {
      b.iter(|| {
        p384::ecdsa::signature::Verifier::verify(black_box(verifying_key), black_box(d), black_box(&oracle_signature))
          .expect("valid authentication benchmark operation must succeed");
        black_box(())
      })
    });

    g.bench_with_input(BenchmarkId::new("ring", len), data, |b, d| {
      b.iter(|| {
        ring_upk
          .verify(black_box(d), black_box(signature.as_bytes()))
          .expect("valid authentication benchmark operation must succeed");
        black_box(())
      })
    });

    aws_lc_bench! {
      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), data, |b, d| {
        b.iter(|| {
          aws_upk.verify(black_box(d), black_box(signature.as_bytes())).expect("valid authentication benchmark operation must succeed");
          black_box(())
        })
      });
    }
  }

  g.finish();
}

fn ecdsa_p384_sign(c: &mut Criterion) {
  let secret_bytes = [0x31u8; 48];
  let secret =
    EcdsaP384SecretKey::from_bytes(secret_bytes).expect("valid authentication benchmark operation must succeed");
  let keypair = EcdsaP384Keypair::from_secret_key(secret);
  let blind = [0xa3u8; 96];
  let signing_key =
    P384OracleSigningKey::from_slice(&secret_bytes).expect("valid authentication benchmark operation must succeed");
  let sec1 = keypair.public_key().to_sec1_bytes();
  let ring_rng = ring::rand::SystemRandom::new();
  let ring_key = ring::signature::EcdsaKeyPair::from_private_key_and_public_key(
    &ring::signature::ECDSA_P384_SHA384_FIXED_SIGNING,
    &secret_bytes,
    &sec1,
    &ring_rng,
  )
  .expect("valid authentication benchmark operation must succeed");
  aws_lc_bench! {
    let aws_rng = aws_lc_rs::rand::SystemRandom::new();
    let aws_key = aws_lc_rs::signature::EcdsaKeyPair::from_private_key_and_public_key(
      &aws_lc_rs::signature::ECDSA_P384_SHA384_FIXED_SIGNING,
      &secret_bytes,
      &sec1,
    )
    .expect("valid authentication benchmark operation must succeed");
  }

  let inputs = [0usize, 32, 1024, 16384]
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect::<Vec<_>>();
  let mut g = c.benchmark_group("ecdsa-p384/sign");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto-deterministic", len), data, |b, d| {
      b.iter(|| {
        black_box(
          black_box(&keypair)
            .try_sign(black_box(d))
            .expect("valid authentication benchmark operation must succeed"),
        )
      })
    });

    g.bench_with_input(BenchmarkId::new("rscrypto-blinded", len), data, |b, d| {
      b.iter(|| {
        black_box(
          black_box(&keypair)
            .try_sign_blinded_with(black_box(d), |out| {
              out.copy_from_slice(black_box(&blind));
              Ok::<(), core::convert::Infallible>(())
            })
            .expect("valid authentication benchmark operation must succeed"),
        )
      })
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto-p384", len), data, |b, d| {
      b.iter(|| {
        let signature: P384OracleSignature =
          p384::ecdsa::signature::Signer::sign(black_box(&signing_key), black_box(d));
        black_box(signature)
      })
    });

    g.bench_with_input(BenchmarkId::new("ring", len), data, |b, d| {
      b.iter(|| {
        black_box(
          ring_key
            .sign(&ring_rng, black_box(d))
            .expect("valid authentication benchmark operation must succeed"),
        )
      })
    });

    aws_lc_bench! {
      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), data, |b, d| {
        b.iter(|| black_box(aws_key.sign(&aws_rng, black_box(d)).expect("valid authentication benchmark operation must succeed")))
      });
    }
  }

  g.finish();
}

fn ecdsa_p384_public_key(c: &mut Criterion) {
  let secret_bytes = [0x31u8; 48];
  let secret =
    EcdsaP384SecretKey::from_bytes(secret_bytes).expect("valid authentication benchmark operation must succeed");
  let blind = [0xa3u8; 96];
  let oracle =
    P384OracleSecretKey::from_slice(&secret_bytes).expect("valid authentication benchmark operation must succeed");
  let expected = secret.public_key().to_sec1_bytes();
  assert_eq!(oracle.public_key().to_sec1_bytes().as_ref(), expected);

  let mut g = c.benchmark_group("ecdsa-p384/public-key");
  g.bench_function("rscrypto-blinded", |b| {
    b.iter(|| {
      black_box(
        black_box(&secret)
          .try_public_key_blinded_with(|out| {
            out.copy_from_slice(black_box(&blind));
            Ok::<(), core::convert::Infallible>(())
          })
          .expect("valid authentication benchmark operation must succeed"),
      )
    })
  });
  g.bench_function("rustcrypto-p384", |b| {
    b.iter(|| black_box(black_box(&oracle).public_key()))
  });
  g.finish();
}

#[cfg(all(feature = "diag", feature = "ecdsa-p256"))]
fn ecdsa_p256_internal(c: &mut Criterion) {
  use rscrypto::auth::{
    diag_ecdsa_p256_basepoint_blinded_limb_digest, diag_ecdsa_p256_final_multiply_limb_digest,
    diag_ecdsa_p256_nonce_inverse_blinded_limb_digest, diag_ecdsa_p256_nonce_inverse_limb_digest,
    diag_ecdsa_p256_nonce_reduce_limb_digest, diag_ecdsa_p256_order_mul_blinded_fixed_r_limb_digest,
    diag_ecdsa_p256_order_mul_fixed_r_limb_digest, diag_ecdsa_p256_reduce_wide_order_limb_digest,
    diag_ecdsa_p256_scalar_finish_limb_digest, diag_ecdsa_p256_select_signing_generator_affine_limb_digest,
  };

  let secret = [0x11u8; 32];
  let blind = [0x5cu8; 64];
  let nonce_wide = [0x5bu8; 64];
  let inputs = [0usize, 32, 1024, 16384]
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect::<Vec<_>>();
  let mut g = c.benchmark_group("ecdsa-p256/internal");

  g.bench_function("select-generator-affine", |b| {
    b.iter(|| {
      black_box(diag_ecdsa_p256_select_signing_generator_affine_limb_digest(black_box(
        173,
      )))
    })
  });
  g.bench_function("reduce-wide-order", |b| {
    b.iter(|| black_box(diag_ecdsa_p256_reduce_wide_order_limb_digest(black_box(nonce_wide))))
  });
  g.bench_function("order-mul-fixed-r", |b| {
    b.iter(|| black_box(diag_ecdsa_p256_order_mul_fixed_r_limb_digest(black_box(secret))))
  });
  g.bench_function("order-mul-blinded-fixed-r", |b| {
    b.iter(|| {
      black_box(diag_ecdsa_p256_order_mul_blinded_fixed_r_limb_digest(
        black_box(secret),
        black_box(blind),
      ))
    })
  });

  for (len, data) in &inputs {
    g.bench_with_input(BenchmarkId::new("nonce-reduce", len), data, |b, d| {
      b.iter(|| {
        black_box(diag_ecdsa_p256_nonce_reduce_limb_digest(
          black_box(secret),
          black_box(d),
        ))
      })
    });
    g.bench_with_input(BenchmarkId::new("basepoint-blinded", len), data, |b, d| {
      b.iter(|| {
        black_box(diag_ecdsa_p256_basepoint_blinded_limb_digest(
          black_box(secret),
          black_box(blind),
          black_box(d),
        ))
      })
    });
    g.bench_with_input(BenchmarkId::new("scalar-finish", len), data, |b, d| {
      b.iter(|| {
        black_box(diag_ecdsa_p256_scalar_finish_limb_digest(
          black_box(secret),
          black_box(nonce_wide),
          black_box(d),
        ))
      })
    });
    g.bench_with_input(BenchmarkId::new("nonce-inverse", len), data, |b, d| {
      b.iter(|| {
        black_box(diag_ecdsa_p256_nonce_inverse_limb_digest(
          black_box(secret),
          black_box(d),
        ))
      })
    });
    g.bench_with_input(BenchmarkId::new("nonce-inverse-blinded", len), data, |b, d| {
      b.iter(|| {
        black_box(diag_ecdsa_p256_nonce_inverse_blinded_limb_digest(
          black_box(secret),
          black_box(blind),
          black_box(d),
        ))
      })
    });
    g.bench_with_input(BenchmarkId::new("final-multiply", len), data, |b, d| {
      b.iter(|| {
        black_box(diag_ecdsa_p256_final_multiply_limb_digest(
          black_box(secret),
          black_box(nonce_wide),
          black_box(d),
        ))
      })
    });
  }

  g.finish();
}

#[cfg(not(all(feature = "diag", feature = "ecdsa-p256")))]
fn ecdsa_p256_internal(_: &mut Criterion) {}

#[cfg(all(feature = "diag", feature = "ecdsa-p384"))]
fn ecdsa_p384_internal(c: &mut Criterion) {
  use rscrypto::auth::{
    diag_ecdsa_p384_basepoint_blinded_limb_digest, diag_ecdsa_p384_basepoint_r_limb_digest,
    diag_ecdsa_p384_final_multiply_limb_digest, diag_ecdsa_p384_nonce_inverse_blinded_limb_digest,
    diag_ecdsa_p384_nonce_inverse_limb_digest, diag_ecdsa_p384_nonce_reduce_limb_digest,
    diag_ecdsa_p384_order_mul_fixed_r_limb_digest, diag_ecdsa_p384_reduce_wide_order_limb_digest,
    diag_ecdsa_p384_scalar_finish_limb_digest, diag_ecdsa_p384_select_signing_generator_affine_limb_digest,
  };

  let secret = [0x31u8; 48];
  let blind = [0xa3u8; 96];
  let nonce_wide = [0x5bu8; 96];
  let inputs = [0usize, 32, 1024, 16384]
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect::<Vec<_>>();
  let mut g = c.benchmark_group("ecdsa-p384/internal");

  g.bench_function("select-generator-affine", |b| {
    b.iter(|| {
      black_box(diag_ecdsa_p384_select_signing_generator_affine_limb_digest(black_box(
        173,
      )))
    })
  });
  g.bench_function("reduce-wide-order", |b| {
    b.iter(|| black_box(diag_ecdsa_p384_reduce_wide_order_limb_digest(black_box(nonce_wide))))
  });
  g.bench_function("order-mul-fixed-r", |b| {
    b.iter(|| black_box(diag_ecdsa_p384_order_mul_fixed_r_limb_digest(black_box(secret))))
  });

  for (len, data) in &inputs {
    g.bench_with_input(BenchmarkId::new("nonce-reduce", len), data, |b, d| {
      b.iter(|| {
        black_box(diag_ecdsa_p384_nonce_reduce_limb_digest(
          black_box(secret),
          black_box(d),
        ))
      })
    });
    g.bench_with_input(BenchmarkId::new("basepoint-blinded", len), data, |b, d| {
      b.iter(|| {
        black_box(diag_ecdsa_p384_basepoint_blinded_limb_digest(
          black_box(secret),
          black_box(blind),
          black_box(d),
        ))
      })
    });
    g.bench_with_input(BenchmarkId::new("basepoint-r", len), data, |b, d| {
      b.iter(|| black_box(diag_ecdsa_p384_basepoint_r_limb_digest(black_box(secret), black_box(d))))
    });
    g.bench_with_input(BenchmarkId::new("scalar-finish", len), data, |b, d| {
      b.iter(|| {
        black_box(diag_ecdsa_p384_scalar_finish_limb_digest(
          black_box(secret),
          black_box(nonce_wide),
          black_box(d),
        ))
      })
    });
    g.bench_with_input(BenchmarkId::new("nonce-inverse", len), data, |b, d| {
      b.iter(|| {
        black_box(diag_ecdsa_p384_nonce_inverse_limb_digest(
          black_box(secret),
          black_box(d),
        ))
      })
    });
    g.bench_with_input(BenchmarkId::new("nonce-inverse-blinded", len), data, |b, d| {
      b.iter(|| {
        black_box(diag_ecdsa_p384_nonce_inverse_blinded_limb_digest(
          black_box(secret),
          black_box(blind),
          black_box(d),
        ))
      })
    });
    g.bench_with_input(BenchmarkId::new("final-multiply", len), data, |b, d| {
      b.iter(|| {
        black_box(diag_ecdsa_p384_final_multiply_limb_digest(
          black_box(secret),
          black_box(nonce_wide),
          black_box(d),
        ))
      })
    });
  }

  g.finish();
}

#[cfg(not(all(feature = "diag", feature = "ecdsa-p384")))]
fn ecdsa_p384_internal(_: &mut Criterion) {}

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
  let keypair = Ed25519Keypair::from_secret_key(secret.duplicate_secret());
  let signing_key = SigningKey::from_bytes(&secret_bytes);
  aws_lc_bench! {
    let aws_kp = aws_lc_rs::signature::Ed25519KeyPair::from_seed_unchecked(&secret_bytes).expect("valid authentication benchmark operation must succeed");
  }
  let ring_kp = ring::signature::Ed25519KeyPair::from_seed_unchecked(&secret_bytes)
    .expect("valid authentication benchmark operation must succeed");
  let (_dryoc_pk, dryoc_sk) = crypto_sign_seed_keypair(&secret_bytes);
  let mut dryoc_sig: [u8; 64] = [0u8; 64];
  let inputs = [0usize, 32, 1024, 16384]
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect::<Vec<_>>();
  let mut g = c.benchmark_group("ed25519/sign");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto-direct-secret", len), data, |b, d| {
      b.iter(|| black_box(black_box(&secret).sign(black_box(d))))
    });

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
        crypto_sign_detached(&mut dryoc_sig, black_box(d), &dryoc_sk)
          .expect("valid authentication benchmark operation must succeed");
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
    let aws_kp = aws_lc_rs::signature::Ed25519KeyPair::from_seed_unchecked(&secret_bytes).expect("valid authentication benchmark operation must succeed");
    let aws_pubkey: Vec<u8> = aws_kp.public_key().as_ref().to_vec();
    let aws_upk = aws_lc_rs::signature::UnparsedPublicKey::new(&aws_lc_rs::signature::ED25519, aws_pubkey);
  }
  let ring_kp = ring::signature::Ed25519KeyPair::from_seed_unchecked(&secret_bytes)
    .expect("valid authentication benchmark operation must succeed");
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
    crypto_sign_detached(&mut dryoc_sig, data, &dryoc_sk)
      .expect("valid authentication benchmark operation must succeed");

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| {
        black_box(&public)
          .verify(black_box(d), black_box(&ours))
          .expect("valid authentication benchmark operation must succeed");
        black_box(())
      })
    });

    g.bench_with_input(BenchmarkId::new("dalek", len), data, |b, d| {
      b.iter(|| {
        black_box(&verifying_key)
          .verify_strict(black_box(d), black_box(&dalek))
          .expect("valid authentication benchmark operation must succeed");
        black_box(())
      })
    });

    aws_lc_bench! {
      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), data, |b, d| {
        b.iter(|| {
          aws_upk.verify(black_box(d), aws_sig.as_ref()).expect("valid authentication benchmark operation must succeed");
          black_box(())
        })
      });
    }

    g.bench_with_input(BenchmarkId::new("ring", len), data, |b, d| {
      b.iter(|| {
        ring_upk
          .verify(black_box(d), ring_sig.as_ref())
          .expect("valid authentication benchmark operation must succeed");
        black_box(())
      })
    });

    g.bench_with_input(BenchmarkId::new("dryoc", len), data, |b, d| {
      b.iter(|| {
        crypto_sign_verify_detached(&dryoc_sig, black_box(d), &dryoc_pk)
          .expect("valid authentication benchmark operation must succeed");
        black_box(())
      })
    });
  }

  g.finish();
}

#[cfg(feature = "diag")]
fn ed25519_verify_phase(c: &mut Criterion) {
  use rscrypto::auth::{
    diag_ed25519_verify_challenge_reduce_digest, diag_ed25519_verify_portable_double_scalar_digest,
    diag_ed25519_verify_public_decode_digest, diag_ed25519_verify_r_decode_digest, diag_ed25519_verify_scalars,
  };

  print_auth_diag_once();

  let secret_bytes = [13u8; 32];
  let secret = Ed25519SecretKey::from_bytes(secret_bytes);
  let keypair = Ed25519Keypair::from_secret_key(secret);
  let public = keypair.public_key();
  let inputs = [0usize, 32, 1024, 16384]
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect::<Vec<_>>();
  let mut g = c.benchmark_group("ed25519/verify-phase");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let signature = keypair.sign(data);
    let scalars = diag_ed25519_verify_scalars(&public, &signature, data)
      .expect("valid authentication benchmark operation must succeed");

    g.bench_with_input(BenchmarkId::new("challenge-reduce", len), data, |b, d| {
      b.iter(|| {
        black_box(diag_ed25519_verify_challenge_reduce_digest(
          black_box(&public),
          black_box(&signature),
          black_box(d),
        ))
      })
    });

    g.bench_function(BenchmarkId::new("public-decode", len), |b| {
      b.iter(|| black_box(diag_ed25519_verify_public_decode_digest(black_box(&scalars.public_key))))
    });

    g.bench_function(BenchmarkId::new("r-decode", len), |b| {
      b.iter(|| black_box(diag_ed25519_verify_r_decode_digest(black_box(&scalars.r_bytes))))
    });

    g.bench_function(BenchmarkId::new("portable-double-scalar", len), |b| {
      b.iter(|| {
        black_box(diag_ed25519_verify_portable_double_scalar_digest(
          black_box(&scalars.s_canonical),
          black_box(&scalars.neg_challenge),
          black_box(&scalars.public_key),
        ))
      })
    });

    #[cfg(all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      not(feature = "portable-only"),
      not(miri)
    ))]
    g.bench_function(BenchmarkId::new("aarch64-asm-double-scalar", len), |b| {
      b.iter(|| {
        black_box(rscrypto::auth::diag_ed25519_verify_aarch64_asm_double_scalar_digest(
          black_box(&scalars.s_canonical),
          black_box(&scalars.neg_challenge),
          black_box(&scalars.public_key),
        ))
      })
    });
  }

  g.finish();
}

#[cfg(not(feature = "diag"))]
fn ed25519_verify_phase(_: &mut Criterion) {}

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
            .expect("valid authentication benchmark operation must succeed");
        black_box(priv_key.compute_public_key().expect("valid authentication benchmark operation must succeed"))
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
      aws_lc_rs::agreement::PrivateKey::from_private_key(&aws_lc_rs::agreement::X25519, &alice_bytes).expect("valid authentication benchmark operation must succeed");
    let aws_bob_pub_bytes: [u8; 32] = {
      let bob_priv =
        aws_lc_rs::agreement::PrivateKey::from_private_key(&aws_lc_rs::agreement::X25519, &bob_bytes).expect("valid authentication benchmark operation must succeed");
      let pk = bob_priv.compute_public_key().expect("valid authentication benchmark operation must succeed");
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
    b.iter(|| {
      black_box(
        black_box(&alice)
          .diffie_hellman(black_box(&bob_public))
          .expect("valid authentication benchmark operation must succeed"),
      )
    })
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
        .expect("valid authentication benchmark operation must succeed");
        black_box(shared)
      })
    });
  }

  g.bench_function("dryoc", |b| {
    let mut shared = [0u8; 32];
    b.iter(|| {
      crypto_scalarmult(&mut shared, black_box(&alice_bytes), black_box(&dryoc_bob_pub))
        .expect("valid authentication benchmark operation must succeed");
      black_box(shared)
    })
  });

  g.finish();
}

fn p256_ephemeral(bytes: [u8; 32]) -> P256EphemeralSecret {
  P256EphemeralSecret::try_generate_with(|candidate| {
    candidate.copy_from_slice(&bytes);
    Ok::<(), core::convert::Infallible>(())
  })
  .expect("valid authentication benchmark scalar must be accepted")
}

const P256_NIST_PRIVATE: [u8; 32] = [
  0x7d, 0x7d, 0xc5, 0xf7, 0x1e, 0xb2, 0x9d, 0xda, 0xf8, 0x0d, 0x62, 0x14, 0x63, 0x2e, 0xea, 0xe0, 0x3d, 0x90, 0x58,
  0xaf, 0x1f, 0xb6, 0xd2, 0x2e, 0xd8, 0x0b, 0xad, 0xb6, 0x2b, 0xc1, 0xa5, 0x34,
];
const P256_NIST_PUBLIC: [u8; 65] = [
  0x04, 0xea, 0xd2, 0x18, 0x59, 0x01, 0x19, 0xe8, 0x87, 0x6b, 0x29, 0x14, 0x6f, 0xf8, 0x9c, 0xa6, 0x17, 0x70, 0xc4,
  0xed, 0xbb, 0xf9, 0x7d, 0x38, 0xce, 0x38, 0x5e, 0xd2, 0x81, 0xd8, 0xa6, 0xb2, 0x30, 0x28, 0xaf, 0x61, 0x28, 0x1f,
  0xd3, 0x5e, 0x2f, 0xa7, 0x00, 0x25, 0x23, 0xac, 0xc8, 0x5a, 0x42, 0x9c, 0xb0, 0x6e, 0xe6, 0x64, 0x83, 0x25, 0x38,
  0x9f, 0x59, 0xed, 0xfc, 0xe1, 0x40, 0x51, 0x41,
];
const P256_NIST_PEER: [u8; 65] = [
  0x04, 0x70, 0x0c, 0x48, 0xf7, 0x7f, 0x56, 0x58, 0x4c, 0x5c, 0xc6, 0x32, 0xca, 0x65, 0x64, 0x0d, 0xb9, 0x1b, 0x6b,
  0xac, 0xce, 0x3a, 0x4d, 0xf6, 0xb4, 0x2c, 0xe7, 0xcc, 0x83, 0x88, 0x33, 0xd2, 0x87, 0xdb, 0x71, 0xe5, 0x09, 0xe3,
  0xfd, 0x9b, 0x06, 0x0d, 0xdb, 0x20, 0xba, 0x5c, 0x51, 0xdc, 0xc5, 0x94, 0x8d, 0x46, 0xfb, 0xf6, 0x40, 0xdf, 0xe0,
  0x44, 0x17, 0x82, 0xca, 0xb8, 0x5f, 0xa4, 0xac,
];
const P256_NIST_SHARED: [u8; 32] = [
  0x46, 0xfc, 0x62, 0x10, 0x64, 0x20, 0xff, 0x01, 0x2e, 0x54, 0xa4, 0x34, 0xfb, 0xdd, 0x2d, 0x25, 0xcc, 0xc5, 0x85,
  0x20, 0x60, 0x56, 0x1e, 0x68, 0x04, 0x0d, 0xd7, 0x77, 0x89, 0x97, 0xbd, 0x7b,
];

fn crrl_p256_scalar(bytes: [u8; 32]) -> crrl::p256::Scalar {
  let mut little_endian = bytes;
  little_endian.reverse();
  crrl::p256::Scalar::decode(&little_endian).expect("valid CRRL benchmark scalar")
}

fn p256_benchmark_preflight() {
  use std::sync::Once;

  static PREFLIGHT: Once = Once::new();
  PREFLIGHT.call_once(|| {
    let ours = p256_ephemeral(P256_NIST_PRIVATE);
    assert_eq!(ours.public_key().to_sec1_bytes(), P256_NIST_PUBLIC);
    let peer = P256PublicKey::from_sec1_bytes(&P256_NIST_PEER).expect("valid NIST benchmark peer");
    assert_eq!(ours.diffie_hellman(&peer).as_bytes(), &P256_NIST_SHARED);

    let rustcrypto = P256OracleSecretKey::from_slice(&P256_NIST_PRIVATE).expect("valid RustCrypto benchmark scalar");
    assert_eq!(
      rustcrypto.public_key().to_sec1_point(false).as_bytes(),
      P256_NIST_PUBLIC
    );
    let rustcrypto_peer = p256::PublicKey::from_sec1_bytes(&P256_NIST_PEER).expect("valid RustCrypto benchmark peer");
    assert_eq!(
      p256::ecdh::diffie_hellman(rustcrypto.to_nonzero_scalar(), rustcrypto_peer.as_affine())
        .raw_secret_bytes()
        .as_slice(),
      P256_NIST_SHARED
    );

    let crrl_scalar = crrl_p256_scalar(P256_NIST_PRIVATE);
    assert_eq!(
      crrl::p256::Point::mulgen(&crrl_scalar).encode_uncompressed(),
      P256_NIST_PUBLIC
    );
    let crrl_peer = crrl::p256::Point::decode(&P256_NIST_PEER).expect("valid CRRL benchmark peer");
    assert_eq!(
      &core::ops::Mul::mul(crrl_peer, crrl_scalar).encode_uncompressed()[1..33],
      &P256_NIST_SHARED
    );

    let mut libcrux_public = [0u8; 64];
    assert!(libcrux_p256::dh_initiator(&mut libcrux_public, &P256_NIST_PRIVATE));
    assert_eq!(libcrux_public, P256_NIST_PUBLIC[1..]);
    let mut libcrux_shared = [0u8; 64];
    assert!(libcrux_p256::dh_responder(
      &mut libcrux_shared,
      &P256_NIST_PEER[1..],
      &P256_NIST_PRIVATE,
    ));
    assert_eq!(libcrux_shared[..32], P256_NIST_SHARED);

    aws_lc_bench! {
      let aws_secret = aws_lc_rs::agreement::PrivateKey::from_private_key(
        &aws_lc_rs::agreement::ECDH_P256,
        &P256_NIST_PRIVATE,
      )
      .expect("valid AWS-LC benchmark scalar");
      assert_eq!(
        aws_secret.compute_public_key().expect("AWS-LC public derivation").as_ref(),
        P256_NIST_PUBLIC,
      );
      let aws_peer = aws_lc_rs::agreement::UnparsedPublicKey::new(
        &aws_lc_rs::agreement::ECDH_P256,
        P256_NIST_PEER,
      );
      let aws_shared = aws_lc_rs::agreement::agree(&aws_secret, aws_peer, (), |bytes| {
        Ok::<[u8; 32], ()>(array_from_slice(bytes))
      })
      .expect("AWS-LC benchmark agreement");
      assert_eq!(aws_shared, P256_NIST_SHARED);
    }

    ring_p256_bench! {
      let rng = ring::rand::SystemRandom::new();
      let ring_secret = ring::agreement::EphemeralPrivateKey::generate(&ring::agreement::ECDH_P256, &rng)
        .expect("ring benchmark scalar generation");
      let ring_public = ring_secret.compute_public_key().expect("ring benchmark public derivation");
      let ours_peer = P256PublicKey::from_sec1_bytes(ring_public.as_ref()).expect("valid ring benchmark public key");
      let ours_shared = p256_ephemeral(P256_NIST_PRIVATE).diffie_hellman(&ours_peer);
      let ring_peer = ring::agreement::UnparsedPublicKey::new(&ring::agreement::ECDH_P256, P256_NIST_PUBLIC);
      let ring_shared = ring::agreement::agree_ephemeral(ring_secret, &ring_peer, array_from_slice::<32>)
        .expect("ring benchmark agreement");
      assert_eq!(ours_shared.as_bytes(), &ring_shared);
    }
  });
}

fn p256_ecdh_key_generation(c: &mut Criterion) {
  p256_benchmark_preflight();
  let scalar = P256_NIST_PRIVATE;
  let mut g = c.benchmark_group("p256-ecdh/key-generation");
  g.bench_function("rscrypto-caller-fill", |b| {
    b.iter(|| black_box(p256_ephemeral(*black_box(&scalar))))
  });
  g.bench_function("rustcrypto-p256-import", |b| {
    b.iter(|| black_box(P256OracleSecretKey::from_slice(black_box(&scalar)).expect("valid oracle scalar")))
  });
  g.bench_function("crrl-import", |b| {
    b.iter(|| black_box(crrl::p256::PrivateKey::decode(black_box(&scalar)).expect("valid CRRL scalar")))
  });
  aws_lc_bench! {
    g.bench_function("aws-lc-rs-native-import-and-precompute", |b| {
      b.iter(|| {
        black_box(
          aws_lc_rs::agreement::PrivateKey::from_private_key(
            &aws_lc_rs::agreement::ECDH_P256,
            black_box(&scalar),
          )
          .expect("valid AWS-LC scalar"),
        )
      })
    });
  }
  g.finish();
}

fn p256_ecdh_public_key(c: &mut Criterion) {
  p256_benchmark_preflight();
  let scalar = P256_NIST_PRIVATE;
  let ours = p256_ephemeral(scalar);
  let rustcrypto = P256OracleSecretKey::from_slice(&scalar).expect("valid RustCrypto scalar");
  let crrl_scalar = crrl_p256_scalar(scalar);
  aws_lc_bench! {
    let aws_secret = aws_lc_rs::agreement::PrivateKey::from_private_key(
      &aws_lc_rs::agreement::ECDH_P256,
      &scalar,
    )
    .expect("valid AWS-LC scalar");
  }
  let mut g = c.benchmark_group("p256-ecdh/public-key");
  g.bench_function("rscrypto-selected", |b| b.iter(|| black_box(ours.public_key())));
  g.bench_function("rscrypto-ecdsa-public-api", |b| {
    b.iter(|| {
      let secret = EcdsaP256SecretKey::from_bytes(*black_box(&scalar)).expect("valid ECDSA benchmark scalar");
      black_box(secret.public_key())
    })
  });
  g.bench_function("rustcrypto-p256-pure-rust", |b| {
    b.iter(|| black_box(rustcrypto.public_key().to_sec1_point(false)))
  });
  g.bench_function("crrl-pure-rust", |b| {
    b.iter(|| black_box(crrl::p256::Point::mulgen(black_box(&crrl_scalar)).encode_uncompressed()))
  });
  g.bench_function("libcrux-hacl-pure-rust", |b| {
    b.iter(|| {
      let mut public = [0u8; 64];
      assert!(libcrux_p256::dh_initiator(black_box(&mut public), black_box(&scalar)));
      black_box(public)
    })
  });
  aws_lc_bench! {
    g.bench_function("aws-lc-rs-native-cached", |b| {
      b.iter(|| black_box(aws_secret.compute_public_key().expect("AWS-LC public derivation")))
    });
    g.bench_function("aws-lc-rs-native-import-and-public", |b| {
      b.iter(|| {
        let secret = aws_lc_rs::agreement::PrivateKey::from_private_key(
          &aws_lc_rs::agreement::ECDH_P256,
          black_box(&scalar),
        )
        .expect("valid AWS-LC scalar");
        black_box(secret.compute_public_key().expect("AWS-LC public derivation"))
      })
    });
  }
  ring_p256_bench! {
    let rng = ring::rand::SystemRandom::new();
    g.bench_function("ring-native", |b| {
      b.iter_batched(
        || {
          ring::agreement::EphemeralPrivateKey::generate(&ring::agreement::ECDH_P256, &rng)
            .expect("ring scalar generation")
        },
        |secret| black_box(secret.compute_public_key().expect("ring public derivation")),
        BatchSize::SmallInput,
      )
    });
  }
  g.finish();
}

fn p256_ecdh_parse(c: &mut Criterion) {
  p256_benchmark_preflight();
  let encoded = p256_ephemeral([0x24; 32]).public_key().to_sec1_bytes();
  let mut g = c.benchmark_group("p256-ecdh/parse");
  g.bench_function("rscrypto", |b| {
    b.iter(|| black_box(P256PublicKey::from_sec1_bytes(black_box(&encoded)).expect("valid benchmark point")))
  });
  g.bench_function("rustcrypto-p256-pure-rust", |b| {
    b.iter(|| black_box(p256::PublicKey::from_sec1_bytes(black_box(&encoded)).expect("valid oracle point")))
  });
  g.bench_function("crrl-pure-rust", |b| {
    b.iter(|| black_box(crrl::p256::Point::decode(black_box(&encoded)).expect("valid CRRL point")))
  });
  aws_lc_bench! {
    g.bench_function("aws-lc-rs-native", |b| {
      b.iter(|| {
        let unparsed = aws_lc_rs::agreement::UnparsedPublicKey::new(
          &aws_lc_rs::agreement::ECDH_P256,
          black_box(encoded),
        );
        black_box(
          aws_lc_rs::agreement::ParsedPublicKey::try_from(unparsed).expect("valid AWS-LC point"),
        )
      })
    });
  }
  g.finish();
}

fn p256_ecdh_agreement(c: &mut Criterion) {
  p256_benchmark_preflight();
  let scalar = P256_NIST_PRIVATE;
  let peer = P256PublicKey::from_sec1_bytes(&P256_NIST_PEER).expect("valid benchmark peer point");
  let rustcrypto_secret = P256OracleSecretKey::from_slice(&scalar).expect("valid RustCrypto scalar");
  let rustcrypto_peer = p256::PublicKey::from_sec1_bytes(&P256_NIST_PEER).expect("valid RustCrypto peer point");
  let crrl_scalar = crrl_p256_scalar(scalar);
  let crrl_peer = crrl::p256::Point::decode(&P256_NIST_PEER).expect("valid CRRL peer point");
  let libcrux_peer = array_from_slice::<64>(&P256_NIST_PEER[1..]);
  aws_lc_bench! {
    let aws_secret = aws_lc_rs::agreement::PrivateKey::from_private_key(
      &aws_lc_rs::agreement::ECDH_P256,
      &scalar,
    )
    .expect("valid AWS-LC scalar");
    let aws_peer = aws_lc_rs::agreement::ParsedPublicKey::try_from(
      aws_lc_rs::agreement::UnparsedPublicKey::new(&aws_lc_rs::agreement::ECDH_P256, P256_NIST_PEER),
    )
    .expect("valid AWS-LC peer point");
  }
  let mut g = c.benchmark_group("p256-ecdh/agreement");
  g.bench_function("rscrypto-selected", |b| {
    b.iter_batched(
      || p256_ephemeral(scalar),
      |secret| black_box(secret.diffie_hellman(black_box(&peer))),
      BatchSize::SmallInput,
    )
  });
  g.bench_function("rustcrypto-p256-pure-rust", |b| {
    b.iter(|| {
      black_box(p256::ecdh::diffie_hellman(
        rustcrypto_secret.to_nonzero_scalar(),
        black_box(rustcrypto_peer.as_affine()),
      ))
    })
  });
  g.bench_function("crrl-pure-rust", |b| {
    b.iter(|| black_box(core::ops::Mul::mul(black_box(crrl_peer), black_box(crrl_scalar)).encode_uncompressed()))
  });
  g.bench_function("libcrux-hacl-pure-rust", |b| {
    b.iter(|| {
      let mut shared = [0u8; 64];
      assert!(libcrux_p256::dh_responder(
        black_box(&mut shared),
        black_box(&libcrux_peer),
        black_box(&scalar),
      ));
      black_box(shared)
    })
  });
  aws_lc_bench! {
    g.bench_function("aws-lc-rs-native", |b| {
      b.iter(|| {
        black_box(
          aws_lc_rs::agreement::agree(&aws_secret, aws_peer.clone(), (), |bytes| {
            Ok::<[u8; 32], ()>(array_from_slice(bytes))
          })
          .expect("AWS-LC benchmark agreement"),
        )
      })
    });
  }
  ring_p256_bench! {
    let rng = ring::rand::SystemRandom::new();
    let ring_peer = ring::agreement::UnparsedPublicKey::new(&ring::agreement::ECDH_P256, P256_NIST_PEER);
    g.bench_function("ring-native", |b| {
      b.iter_batched(
        || {
          ring::agreement::EphemeralPrivateKey::generate(&ring::agreement::ECDH_P256, &rng)
            .expect("ring scalar generation")
        },
        |secret| {
          black_box(
            ring::agreement::agree_ephemeral(secret, black_box(&ring_peer), array_from_slice::<32>)
              .expect("ring benchmark agreement"),
          )
        },
        BatchSize::SmallInput,
      )
    });
  }
  g.finish();
}

fn p256_ecdh_tls_roundtrip(c: &mut Criterion) {
  let alice = [0x42; 32];
  let bob = [0x24; 32];
  let mut g = c.benchmark_group("p256-ecdh/tls-shaped-roundtrip");
  g.bench_function("rscrypto", |b| {
    b.iter(|| {
      let alice_secret = p256_ephemeral(*black_box(&alice));
      let bob_secret = p256_ephemeral(*black_box(&bob));
      let alice_public = alice_secret.public_key().to_sec1_bytes();
      let bob_public = bob_secret.public_key().to_sec1_bytes();
      let alice_peer = P256PublicKey::from_sec1_bytes(black_box(&bob_public)).expect("valid benchmark peer");
      let bob_peer = P256PublicKey::from_sec1_bytes(black_box(&alice_public)).expect("valid benchmark peer");
      let alice_shared = alice_secret.diffie_hellman(&alice_peer);
      let bob_shared = bob_secret.diffie_hellman(&bob_peer);
      black_box((alice_shared, bob_shared))
    })
  });
  g.finish();
}

macro_rules! mlkem_profile_benches {
  (
    $keygen_fn:ident,
    $encapsulate_fn:ident,
    $decapsulate_fn:ident,
    $group:literal,
    $profile:ty,
    $fips:ident,
    $libcrux:ident,
    $rustcrypto:ty,
    $aws_algorithm:ident
  ) => {
    fn $keygen_fn(c: &mut Criterion) {
      let key_random = deterministic_bytes::<{ <$profile>::KEY_GENERATION_RANDOM_SIZE }>(0x10);
      let d = array_from_slice::<32>(&key_random[..32]);
      let z = array_from_slice::<32>(&key_random[32..]);
      let mut g = c.benchmark_group(concat!($group, "/keygen"));

      g.bench_function("rscrypto", |b| {
        b.iter(|| {
          <$profile>::generate_keypair(|out| {
            out.copy_from_slice(black_box(&key_random));
            Ok::<(), MlKemError>(())
          })
          .expect("valid authentication benchmark operation must succeed")
        })
      });

      g.bench_function("libcrux", |b| {
        b.iter(|| $libcrux::generate_key_pair(black_box(key_random)))
      });

      aws_lc_bench! {
        g.bench_function("aws-lc-rs", |b| {
          b.iter(|| black_box(AwsMlKemDecapsulationKey::generate(&$aws_algorithm).expect("valid authentication benchmark operation must succeed")))
        });
      }

      g.bench_function("fips203", |b| {
        b.iter(|| $fips::KG::keygen_from_seed(black_box(d), black_box(z)))
      });

      g.bench_function("rustcrypto", |b| {
        b.iter(|| {
          let dk = RustCryptoMlKemDecapsulationKey::<$rustcrypto>::from_seed(RustCryptoMlKemSeed::from(key_random));
          black_box(dk.encapsulation_key().to_bytes());
          black_box(dk)
        })
      });

      g.finish();
    }

    fn $encapsulate_fn(c: &mut Criterion) {
      let key_random = deterministic_bytes::<{ <$profile>::KEY_GENERATION_RANDOM_SIZE }>(0x20);
      let encaps_random = deterministic_bytes::<{ <$profile>::ENCAPSULATION_RANDOM_SIZE }>(0x80);
      let (ek, _) = <$profile>::generate_keypair(|out| {
        out.copy_from_slice(&key_random);
        Ok::<(), MlKemError>(())
      })
      .expect("valid authentication benchmark operation must succeed");
      let prepared_ek = ek.prepare().expect("valid authentication benchmark operation must succeed");
      let (fips_ek, _) = $fips::KG::keygen_from_seed(
        array_from_slice::<32>(&key_random[..32]),
        array_from_slice::<32>(&key_random[32..]),
      );
      let rustcrypto_dk =
        RustCryptoMlKemDecapsulationKey::<$rustcrypto>::from_seed(RustCryptoMlKemSeed::from(key_random));
      let rustcrypto_ek = rustcrypto_dk.encapsulation_key().clone();
      let libcrux_keypair = $libcrux::generate_key_pair(key_random);
      let libcrux_ek = libcrux_keypair.public_key().clone();
      aws_lc_bench! {
        let aws_dk = AwsMlKemDecapsulationKey::generate(&$aws_algorithm).expect("valid authentication benchmark operation must succeed");
        let aws_ek = aws_dk.encapsulation_key().expect("valid authentication benchmark operation must succeed");
      }
      let mut g = c.benchmark_group(concat!($group, "/encapsulate"));

      g.bench_function("rscrypto", |b| {
        b.iter(|| {
          black_box(&prepared_ek)
            .encapsulate(|out| {
              out.copy_from_slice(black_box(&encaps_random));
              Ok::<(), MlKemError>(())
            })
            .expect("valid authentication benchmark operation must succeed")
        })
      });

      g.bench_function("libcrux", |b| {
        b.iter(|| black_box($libcrux::encapsulate(black_box(&libcrux_ek), black_box(encaps_random))))
      });

      aws_lc_bench! {
        g.bench_function("aws-lc-rs", |b| {
          b.iter(|| black_box(aws_ek.encapsulate().expect("valid authentication benchmark operation must succeed")))
        });
      }

      g.bench_function("fips203", |b| {
        b.iter(|| black_box(fips_ek.encaps_from_seed(black_box(&encaps_random))))
      });

      g.bench_function("rustcrypto", |b| {
        b.iter(|| {
          black_box(rustcrypto_ek.encapsulate_deterministic(black_box(&RustCryptoMlKemB32::from(encaps_random))))
        })
      });

      g.finish();
    }

    fn $decapsulate_fn(c: &mut Criterion) {
      let key_random = deterministic_bytes::<{ <$profile>::KEY_GENERATION_RANDOM_SIZE }>(0x30);
      let encaps_random = deterministic_bytes::<{ <$profile>::ENCAPSULATION_RANDOM_SIZE }>(0x90);
      let (ek, dk) = <$profile>::generate_keypair(|out| {
        out.copy_from_slice(&key_random);
        Ok::<(), MlKemError>(())
      })
      .expect("valid authentication benchmark operation must succeed");
      let prepared_ek = ek.prepare().expect("valid authentication benchmark operation must succeed");
      let prepared_dk = dk.prepare().expect("valid authentication benchmark operation must succeed");
      let (ciphertext, _) = prepared_ek
        .encapsulate(|out| {
          out.copy_from_slice(&encaps_random);
          Ok::<(), MlKemError>(())
        })
        .expect("valid authentication benchmark operation must succeed");
      let (fips_ek, fips_dk) = $fips::KG::keygen_from_seed(
        array_from_slice::<32>(&key_random[..32]),
        array_from_slice::<32>(&key_random[32..]),
      );
      let (_, fips_ciphertext) = fips_ek.encaps_from_seed(&encaps_random);
      let rustcrypto_dk =
        RustCryptoMlKemDecapsulationKey::<$rustcrypto>::from_seed(RustCryptoMlKemSeed::from(key_random));
      let (rustcrypto_ciphertext, _) = rustcrypto_dk
        .encapsulation_key()
        .encapsulate_deterministic(&RustCryptoMlKemB32::from(encaps_random));
      let libcrux_keypair = $libcrux::generate_key_pair(key_random);
      let libcrux_ek = libcrux_keypair.public_key().clone();
      let libcrux_dk = libcrux_keypair.private_key().clone();
      let (libcrux_ciphertext, _) = $libcrux::encapsulate(&libcrux_ek, encaps_random);
      aws_lc_bench! {
        let aws_dk = AwsMlKemDecapsulationKey::generate(&$aws_algorithm).expect("valid authentication benchmark operation must succeed");
        let aws_ek = aws_dk.encapsulation_key().expect("valid authentication benchmark operation must succeed");
        let (aws_ciphertext, _) = aws_ek.encapsulate().expect("valid authentication benchmark operation must succeed");
      }
      let mut g = c.benchmark_group(concat!($group, "/decapsulate"));

      g.bench_function("rscrypto", |b| {
        b.iter(|| black_box(&prepared_dk).decapsulate(black_box(&ciphertext)).expect("valid authentication benchmark operation must succeed"))
      });

      g.bench_function("libcrux", |b| {
        b.iter(|| {
          black_box($libcrux::decapsulate(
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
                .expect("valid authentication benchmark operation must succeed"),
            )
          })
        });
      }

      g.bench_function("fips203", |b| {
        b.iter(|| black_box(fips_dk.try_decaps(black_box(&fips_ciphertext)).expect("valid authentication benchmark operation must succeed")))
      });

      g.bench_function("rustcrypto", |b| {
        b.iter(|| black_box(rustcrypto_dk.decapsulate(black_box(&rustcrypto_ciphertext))))
      });

      g.finish();
    }
  };
}

mlkem_profile_benches!(
  mlkem512_keygen,
  mlkem512_encapsulate,
  mlkem512_decapsulate,
  "mlkem512",
  MlKem512,
  FipsMlKem512,
  LibcruxMlKem512,
  RustCryptoMlKem512,
  AWS_ML_KEM_512
);
mlkem_profile_benches!(
  mlkem768_keygen,
  mlkem768_encapsulate,
  mlkem768_decapsulate,
  "mlkem768",
  MlKem768,
  FipsMlKem768,
  LibcruxMlKem768,
  RustCryptoMlKem768,
  AWS_ML_KEM_768
);
mlkem_profile_benches!(
  mlkem1024_keygen,
  mlkem1024_encapsulate,
  mlkem1024_decapsulate,
  "mlkem1024",
  MlKem1024,
  FipsMlKem1024,
  LibcruxMlKem1024,
  RustCryptoMlKem1024,
  AWS_ML_KEM_1024
);

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
  ecdsa_p256_public_key,
  ecdsa_p256_sign,
  ecdsa_p256_verify,
  ecdsa_p256_internal,
  ecdsa_p384_public_key,
  ecdsa_p384_sign,
  ecdsa_p384_verify,
  ecdsa_p384_internal,
  ed25519_public_key,
  ed25519_keypair_from_secret,
  ed25519_sign,
  ed25519_verify,
  ed25519_verify_phase,
  x25519_public_key,
  x25519_diffie_hellman,
  p256_ecdh_key_generation,
  p256_ecdh_public_key,
  p256_ecdh_parse,
  p256_ecdh_agreement,
  p256_ecdh_tls_roundtrip,
  mlkem512_keygen,
  mlkem512_encapsulate,
  mlkem512_decapsulate,
  mlkem768_keygen,
  mlkem768_encapsulate,
  mlkem768_decapsulate,
  mlkem1024_keygen,
  mlkem1024_encapsulate,
  mlkem1024_decapsulate
);
criterion_main!(benches);
