//! Auth benchmarks for rscrypto public APIs.
//!
//! Before timing, each row's output must match an independent implementation or a published vector. Rows with
//! randomized output must instead verify under an independent implementation.

#[path = "common/criterion.rs"]
mod bench_config;

mod common;

use core::hint::black_box;

use bench_config::{checked, checked_routine, checked_with};
use criterion::{BatchSize, BenchmarkId, Criterion};
use ed25519_dalek::{Signature as DalekSignature, Signer as _, SigningKey};
use fips203::{
  ml_kem_512 as FipsMlKem512, ml_kem_768 as FipsMlKem768, ml_kem_1024 as FipsMlKem1024,
  traits::{Decaps as _, Encaps as _, KeyGen as _, SerDes as _},
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
  EcdsaP384SecretKey, EcdsaP384Signature, Ed25519Keypair, Ed25519PublicKey, Ed25519SecretKey, Ed25519Signature,
  HkdfSha256, HkdfSha384, HkdfSha512, HmacSha256, HmacSha384, HmacSha512, Kem as _, Mac as _, MlKem512, MlKem768,
  MlKem1024, MlKemError, P256EphemeralSecret, P256PublicKey, P384EphemeralSecret, P384PublicKey, Pbkdf2Sha256,
  Pbkdf2Sha512, X25519SecretKey,
};
use rustcrypto_ml_kem::{
  DecapsulationKey as RustCryptoMlKemDecapsulationKey, MlKem512 as RustCryptoMlKem512, MlKem768 as RustCryptoMlKem768,
  MlKem1024 as RustCryptoMlKem1024, Seed as RustCryptoMlKemSeed, kem::Decapsulate as _,
};
use x25519_dalek::{PublicKey as DalekX25519PublicKey, StaticSecret as DalekX25519Secret};

type RustCryptoHmacSha256 = Hmac<sha2::Sha256>;
type RustCryptoHmacSha384 = Hmac<sha2::Sha384>;
type RustCryptoHmacSha512 = Hmac<sha2::Sha512>;
type RustCryptoHkdfSha256 = RustCryptoHkdf<sha2::Sha256>;
type RustCryptoHkdfSha384 = RustCryptoHkdf<sha2::Sha384>;
type RustCryptoHkdfSha512 = RustCryptoHkdf<sha2::Sha512>;

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

/// Decodes a fixed-length hexadecimal test vector.
fn hex<const N: usize>(text: &str) -> [u8; N] {
  assert_eq!(text.len(), N.strict_mul(2), "hex vector length");
  let mut out = [0u8; N];
  for (byte, pair) in out.iter_mut().zip(text.as_bytes().as_chunks::<2>().0) {
    let pair = core::str::from_utf8(pair).expect("hex vector is ASCII");
    *byte = u8::from_str_radix(pair, 16).expect("hex vector digit pair");
  }
  out
}

/// The RustCrypto HMAC tag of `message`.
fn rustcrypto_hmac<M: hmac::Mac + KeyInit>(key: &[u8], message: &[u8]) -> Vec<u8> {
  let mut mac = <M as KeyInit>::new_from_slice(key).expect("valid authentication benchmark operation must succeed");
  mac.update(message);
  mac.finalize().into_bytes().to_vec()
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

#[cfg(all(rscrypto_internal, feature = "diag"))]
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

#[cfg(not(all(rscrypto_internal, feature = "diag")))]
#[inline]
fn print_auth_diag_once() {}

// Each HMAC row times message absorption, finalization, and state reuse or destruction; keying is untimed.
// Every row must return RustCrypto's tag.
macro_rules! hmac_bench {
  (
    $name:ident,
    $group:literal,
    $rscrypto:ty,
    $rustcrypto:ty,
    $key_len:literal,
    $aws_algorithm:path,
    $ring_algorithm:path
  ) => {
    fn $name(c: &mut Criterion) {
      if !bench_config::selected($group) {
        return;
      }
      print_auth_diag_once();

      let inputs = common::comp_sizes();
      let key = [0x42u8; $key_len];
      let rustcrypto =
        <$rustcrypto>::new_from_slice(&key).expect("valid authentication benchmark operation must succeed");
      aws_lc_bench! {
        let aws_key = aws_lc_rs::hmac::Key::new($aws_algorithm, &key);
      }
      let ring_key = ring::hmac::Key::new($ring_algorithm, &key);
      let mut g = c.benchmark_group($group);

      for (len, data) in &inputs {
        common::set_throughput(&mut g, *len);
        let expected = rustcrypto_hmac::<$rustcrypto>(&key, data);

        let mut mac = <$rscrypto>::new(&key);
        g.bench_function(
          BenchmarkId::new("rscrypto", len),
          checked(
            || {
              mac.update(black_box(data));
              let tag = mac.finalize();
              mac.reset();
              black_box(tag)
            },
            |tag| assert_eq!(tag.as_bytes().as_slice(), expected.as_slice()),
          ),
        );

        g.bench_function(
          BenchmarkId::new("rustcrypto", len),
          checked(
            || {
              use hmac::Mac as _;

              let mut mac = rustcrypto.clone();
              mac.update(black_box(data));
              black_box(mac.finalize().into_bytes())
            },
            |tag| assert_eq!(tag.as_slice(), expected.as_slice()),
          ),
        );

        aws_lc_bench! {
          g.bench_function(
            BenchmarkId::new("aws-lc-rs", len),
            checked(
              || black_box(aws_lc_rs::hmac::sign(&aws_key, black_box(data))),
              |tag| assert_eq!(tag.as_ref(), expected.as_slice()),
            ),
          );
        }

        g.bench_function(
          BenchmarkId::new("ring", len),
          checked(
            || black_box(ring::hmac::sign(&ring_key, black_box(data))),
            |tag| assert_eq!(tag.as_ref(), expected.as_slice()),
          ),
        );
      }

      g.finish();
    }
  };
}

hmac_bench!(
  hmac_sha256,
  "hmac-sha256",
  HmacSha256,
  RustCryptoHmacSha256,
  32,
  aws_lc_rs::hmac::HMAC_SHA256,
  ring::hmac::HMAC_SHA256
);
hmac_bench!(
  hmac_sha384,
  "hmac-sha384",
  HmacSha384,
  RustCryptoHmacSha384,
  48,
  aws_lc_rs::hmac::HMAC_SHA384,
  ring::hmac::HMAC_SHA384
);
hmac_bench!(
  hmac_sha512,
  "hmac-sha512",
  HmacSha512,
  RustCryptoHmacSha512,
  64,
  aws_lc_rs::hmac::HMAC_SHA512,
  ring::hmac::HMAC_SHA512
);

fn hmac_sha256_streaming(c: &mut Criterion) {
  if !bench_config::selected("hmac-sha256/streaming") {
    return;
  }
  print_auth_diag_once();

  let data = common::random_bytes(1048576);
  let key = [0x24u8; 32];
  let expected = rustcrypto_hmac::<RustCryptoHmacSha256>(&key, &data);
  let mut g = c.benchmark_group("hmac-sha256/streaming");
  g.throughput(criterion::Throughput::Bytes(data.len() as u64));

  for chunk_size in [64, 4096] {
    g.bench_function(
      format!("rscrypto/{chunk_size}B"),
      checked(
        || {
          let mut mac = HmacSha256::new(&key);
          for chunk in data.chunks(chunk_size) {
            mac.update(black_box(chunk));
          }
          black_box(mac.finalize())
        },
        |tag| assert_eq!(tag.as_bytes().as_slice(), expected.as_slice()),
      ),
    );

    g.bench_function(
      format!("rustcrypto/{chunk_size}B"),
      checked(
        || {
          use hmac::Mac as _;

          let mut mac =
            RustCryptoHmacSha256::new_from_slice(&key).expect("valid authentication benchmark operation must succeed");
          for chunk in data.chunks(chunk_size) {
            mac.update(black_box(chunk));
          }
          black_box(mac.finalize().into_bytes())
        },
        |tag| assert_eq!(tag.as_slice(), expected.as_slice()),
      ),
    );
  }

  g.finish();
}

fn hmac_sha256_internal(c: &mut Criterion) {
  if !bench_config::selected("hmac-sha256/internal/fixed-message") {
    return;
  }
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
    let expected = rustcrypto_hmac::<RustCryptoHmacSha256>(&key, msg);

    g.bench_function(
      BenchmarkId::new("rscrypto-oneshot", len),
      checked(
        || black_box(HmacSha256::mac(black_box(&key), black_box(msg))),
        |tag| assert_eq!(tag.as_bytes().as_slice(), expected.as_slice()),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rscrypto-stream-new", len),
      checked(
        || {
          let mut mac = HmacSha256::new(black_box(&key));
          mac.update(black_box(msg));
          black_box(mac.finalize())
        },
        |tag| assert_eq!(tag.as_bytes().as_slice(), expected.as_slice()),
      ),
    );

    let mut mac = HmacSha256::new(&key);
    g.bench_function(
      BenchmarkId::new("rscrypto-stream-reuse", len),
      checked(
        || {
          mac.update(black_box(msg));
          let tag = mac.finalize();
          mac.reset();
          black_box(tag)
        },
        |tag| assert_eq!(tag.as_bytes().as_slice(), expected.as_slice()),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rustcrypto-oneshot", len),
      checked(
        || {
          use hmac::Mac as _;

          let mut mac = RustCryptoHmacSha256::new_from_slice(black_box(&key))
            .expect("valid authentication benchmark operation must succeed");
          mac.update(black_box(msg));
          black_box(mac.finalize().into_bytes())
        },
        |tag| assert_eq!(tag.as_slice(), expected.as_slice()),
      ),
    );

    aws_lc_bench! {
      g.bench_function(
        BenchmarkId::new("aws-lc-rs", len),
        checked(
          || black_box(aws_lc_rs::hmac::sign(&aws_key, black_box(msg))),
          |tag| assert_eq!(tag.as_ref(), expected.as_slice()),
        ),
      );
    }

    g.bench_function(
      BenchmarkId::new("ring", len),
      checked(
        || black_box(ring::hmac::sign(&ring_key, black_box(msg))),
        |tag| assert_eq!(tag.as_ref(), expected.as_slice()),
      ),
    );
  }

  g.finish();
}

// HKDF-Expand rows write each library's output into a reused caller buffer, and every row must return
// RustCrypto's output. Salt and IKM are HashLen bytes, and `info` keeps T(1) in one compression block while each
// later T(i) needs two.
macro_rules! hkdf_expand_bench {
  (
    $name:ident,
    $group:literal,
    $rscrypto:ty,
    $rustcrypto:ty,
    $aws_algorithm:path,
    $ring_algorithm:path,
    $hash_len:literal,
    $info_len:literal,
    [$($out_len:literal),+]
    $(,)?
  ) => {
    fn $name(c: &mut Criterion) {
      if !bench_config::selected($group) {
        return;
      }
      print_auth_diag_once();

      let salt = [0x11u8; $hash_len];
      let ikm = [0x22u8; $hash_len];
      let info = [0x33u8; $info_len];
      let hkdf = <$rscrypto>::new(&salt, &ikm);
      let rustcrypto = <$rustcrypto>::new(Some(&salt), &ikm);
      aws_lc_bench! {
        let aws_prk = aws_lc_rs::hkdf::Salt::new($aws_algorithm, &salt).extract(&ikm);
      }
      let ring_prk = ring::hkdf::Salt::new($ring_algorithm, &salt).extract(&ikm);
      let mut g = c.benchmark_group($group);

      for out_len in [$($out_len),+] {
        let mut expected = vec![0u8; out_len];
        rustcrypto
          .expand(&info, &mut expected)
          .expect("valid authentication benchmark operation must succeed");

        g.throughput(criterion::Throughput::Bytes(out_len as u64));
        g.bench_function(
          BenchmarkId::new("rscrypto", out_len),
          checked_with(
            vec![0u8; out_len],
            |out| {
              hkdf
                .expand(black_box(&info), black_box(out.as_mut_slice()))
                .expect("valid authentication benchmark operation must succeed");
              black_box(out[0])
            },
            |_, out| {
              assert_eq!(out, &expected);
            },
          ),
        );

        g.bench_function(
          BenchmarkId::new("rustcrypto", out_len),
          checked_with(
            vec![0u8; out_len],
            |out| {
              rustcrypto
                .expand(black_box(&info), black_box(out.as_mut_slice()))
                .expect("valid authentication benchmark operation must succeed");
              black_box(out[0])
            },
            |_, out| assert_eq!(out, &expected),
          ),
        );

        aws_lc_bench! {
          g.bench_function(
            BenchmarkId::new("aws-lc-rs", out_len),
            checked_with(
              vec![0u8; out_len],
              |out| {
                aws_prk
                  .expand(&[black_box(&info)], AwsHkdfLen(out_len))
                  .expect("valid authentication benchmark operation must succeed")
                  .fill(black_box(out.as_mut_slice()))
                  .expect("valid authentication benchmark operation must succeed");
                black_box(out[0])
              },
              |_, out| assert_eq!(out, &expected),
            ),
          );
        }

        g.bench_function(
          BenchmarkId::new("ring", out_len),
          checked_with(
            vec![0u8; out_len],
            |out| {
              ring_prk
                .expand(&[black_box(&info)], RingHkdfLen(out_len))
                .expect("valid authentication benchmark operation must succeed")
                .fill(black_box(out.as_mut_slice()))
                .expect("valid authentication benchmark operation must succeed");
              black_box(out[0])
            },
            |_, out| assert_eq!(out, &expected),
          ),
        );
      }

      g.finish();
    }
  };
}

hkdf_expand_bench!(
  hkdf_sha256_expand,
  "hkdf-sha256/expand",
  HkdfSha256,
  RustCryptoHkdfSha256,
  aws_lc_rs::hkdf::HKDF_SHA256,
  ring::hkdf::HKDF_SHA256,
  32,
  48,
  [32, 64, 256, 1024]
);
hkdf_expand_bench!(
  hkdf_sha384_expand,
  "hkdf-sha384/expand",
  HkdfSha384,
  RustCryptoHkdfSha384,
  aws_lc_rs::hkdf::HKDF_SHA384,
  ring::hkdf::HKDF_SHA384,
  48,
  80,
  [48, 96, 256, 1024]
);
hkdf_expand_bench!(
  hkdf_sha512_expand,
  "hkdf-sha512/expand",
  HkdfSha512,
  RustCryptoHkdfSha512,
  aws_lc_rs::hkdf::HKDF_SHA512,
  ring::hkdf::HKDF_SHA512,
  64,
  80,
  [64, 128, 256, 1024]
);

// HKDF-Extract rows time salt keying, HMAC(salt, IKM), the PRK HMAC key
// schedule that each library builds for later expansion, and state drop.
// Salt is HashLen bytes. IKM is HashLen bytes, the TLS 1.3 early- and
// master-secret shape; SHA-256 adds the 64-byte X25519MLKEM768 shared secret,
// which crosses the SHA-256 block boundary. aws-lc-rs has no row: its
// `Salt::extract` only copies the IKM and defers HKDF-Extract to `Okm::fill`.
macro_rules! hkdf_extract_bench {
  (
    $name:ident,
    $group:literal,
    $rscrypto:ty,
    $rustcrypto:ty,
    $ring_algorithm:path,
    $hash_len:literal,
    [$($ikm_len:literal),+]
    $(,)?
  ) => {
    fn $name(c: &mut Criterion) {
      if !bench_config::selected($group) {
        return;
      }
      print_auth_diag_once();

      let salt = [0x11u8; $hash_len];
      let mut g = c.benchmark_group($group);
      for ikm_len in [$($ikm_len),+] {
        let ikm = vec![0x22u8; ikm_len];
        // RustCrypto exposes the PRK. ring does not, so its row must reproduce T(1), the first expanded block.
        let (expected_prk, expected_hkdf) = <$rustcrypto>::extract(Some(&salt), &ikm);
        let mut expected_t1 = [0u8; $hash_len];
        expected_hkdf
          .expand(&[], &mut expected_t1)
          .expect("valid authentication benchmark operation must succeed");

        g.bench_function(
          BenchmarkId::new("rscrypto", ikm_len),
          checked(
            || black_box(<$rscrypto>::extract(black_box(&salt), black_box(&ikm))),
            |hkdf| {
              assert_eq!(hkdf.prk().as_slice(), expected_prk.as_slice());
            },
          ),
        );
        g.bench_function(
          BenchmarkId::new("rustcrypto", ikm_len),
          checked(
            || black_box(<$rustcrypto>::extract(Some(black_box(&salt)), black_box(&ikm))),
            |(prk, _)| assert_eq!(prk.as_slice(), expected_prk.as_slice()),
          ),
        );
        g.bench_function(
          BenchmarkId::new("ring", ikm_len),
          checked(
            || black_box(ring::hkdf::Salt::new($ring_algorithm, black_box(&salt)).extract(black_box(&ikm))),
            |prk| {
              let mut t1 = [0u8; $hash_len];
              prk
                .expand(&[], RingHkdfLen($hash_len))
                .expect("valid authentication benchmark operation must succeed")
                .fill(&mut t1)
                .expect("valid authentication benchmark operation must succeed");
              assert_eq!(t1, expected_t1);
            },
          ),
        );
      }
      g.finish();
    }
  };
}

hkdf_extract_bench!(
  hkdf_sha256_extract,
  "hkdf-sha256/extract",
  HkdfSha256,
  RustCryptoHkdfSha256,
  ring::hkdf::HKDF_SHA256,
  32,
  [32, 64]
);
hkdf_extract_bench!(
  hkdf_sha384_extract,
  "hkdf-sha384/extract",
  HkdfSha384,
  RustCryptoHkdfSha384,
  ring::hkdf::HKDF_SHA384,
  48,
  [48]
);
hkdf_extract_bench!(
  hkdf_sha512_extract,
  "hkdf-sha512/extract",
  HkdfSha512,
  RustCryptoHkdfSha512,
  ring::hkdf::HKDF_SHA512,
  64,
  [64]
);

fn pbkdf2_sha256_derive(c: &mut Criterion) {
  if !bench_config::selected("pbkdf2-sha256") {
    return;
  }
  print_auth_diag_once();

  let password = [0x55u8; 32];
  let salt = [0x33u8; 16];
  let state = Pbkdf2Sha256::new(&password);
  // Every row must write RustCrypto's derived key into its reused output buffer.
  let rustcrypto = |iterations: u32, len: usize| {
    let mut out = vec![0u8; len];
    pbkdf2::pbkdf2_hmac::<sha2::Sha256>(&password, &salt, iterations, &mut out);
    out
  };

  for &iterations in &[1u32, 100, 1000] {
    let nz_iters =
      core::num::NonZeroU32::new(iterations).expect("valid authentication benchmark operation must succeed");
    let mut g = c.benchmark_group(format!("pbkdf2-sha256/iters={iterations}"));

    for &out_len in &[32usize, 64] {
      g.throughput(criterion::Throughput::Bytes(out_len as u64));
      let expected = rustcrypto(iterations, out_len);

      g.bench_function(
        BenchmarkId::new("rscrypto", out_len),
        checked_with(
          vec![0u8; out_len],
          |out| {
            Pbkdf2Sha256::derive_key_primitive(
              black_box(&password),
              black_box(&salt),
              iterations,
              black_box(out.as_mut_slice()),
            )
            .expect("valid authentication benchmark operation must succeed");
            black_box(out[0])
          },
          |_, out| assert_eq!(out, &expected),
        ),
      );

      g.bench_function(
        BenchmarkId::new("rustcrypto", out_len),
        checked_with(
          vec![0u8; out_len],
          |out| {
            pbkdf2::pbkdf2_hmac::<sha2::Sha256>(
              black_box(&password),
              black_box(&salt),
              iterations,
              black_box(out.as_mut_slice()),
            );
            black_box(out[0])
          },
          |_, out| assert_eq!(out, &expected),
        ),
      );

      aws_lc_bench! {
        g.bench_function(
          BenchmarkId::new("aws-lc-rs", out_len),
          checked_with(
            vec![0u8; out_len],
            |out| {
              aws_lc_rs::pbkdf2::derive(
                aws_lc_rs::pbkdf2::PBKDF2_HMAC_SHA256,
                nz_iters,
                black_box(&salt),
                black_box(&password),
                black_box(out.as_mut_slice()),
              );
              black_box(out[0])
            },
            |_, out| assert_eq!(out, &expected),
          ),
        );
      }

      g.bench_function(
        BenchmarkId::new("ring", out_len),
        checked_with(
          vec![0u8; out_len],
          |out| {
            ring::pbkdf2::derive(
              ring::pbkdf2::PBKDF2_HMAC_SHA256,
              nz_iters,
              black_box(&salt),
              black_box(&password),
              black_box(out.as_mut_slice()),
            );
            black_box(out[0])
          },
          |_, out| assert_eq!(out, &expected),
        ),
      );
    }

    g.finish();

    let mut g_state = c.benchmark_group(format!("pbkdf2-sha256-state/iters={iterations}"));
    for &out_len in &[32usize, 64] {
      g_state.throughput(criterion::Throughput::Bytes(out_len as u64));
      let expected = rustcrypto(iterations, out_len);
      g_state.bench_function(
        BenchmarkId::new("rscrypto", out_len),
        checked_with(
          vec![0u8; out_len],
          |out| {
            state
              .derive(black_box(&salt), iterations, black_box(out.as_mut_slice()))
              .expect("valid authentication benchmark operation must succeed");
            black_box(out[0])
          },
          |_, out| assert_eq!(out, &expected),
        ),
      );
    }
    g_state.finish();
  }
}

fn pbkdf2_sha256_internal(c: &mut Criterion) {
  if !bench_config::selected("pbkdf2-sha256/internal/") {
    return;
  }
  print_auth_diag_once();

  let password = [0x55u8; 32];
  let salt = [0x33u8; 16];
  let state = Pbkdf2Sha256::new(&password);
  // Every row must write RustCrypto's derived key into its reused output buffer.
  let rustcrypto = |iterations: u32, len: usize| {
    let mut out = vec![0u8; len];
    pbkdf2::pbkdf2_hmac::<sha2::Sha256>(&password, &salt, iterations, &mut out);
    out
  };

  for &iterations in &[1u32, 100, 1000] {
    let nz_iters =
      core::num::NonZeroU32::new(iterations).expect("valid authentication benchmark operation must succeed");
    let mut g = c.benchmark_group(format!("pbkdf2-sha256/internal/iters={iterations}"));

    for &out_len in &[32usize, 64] {
      g.throughput(criterion::Throughput::Bytes(out_len as u64));
      let expected = rustcrypto(iterations, out_len);

      g.bench_function(
        BenchmarkId::new("rscrypto-oneshot", out_len),
        checked_with(
          vec![0u8; out_len],
          |out| {
            Pbkdf2Sha256::derive_key_primitive(
              black_box(&password),
              black_box(&salt),
              iterations,
              black_box(out.as_mut_slice()),
            )
            .expect("valid authentication benchmark operation must succeed");
            black_box(out[0])
          },
          |_, out| assert_eq!(out, &expected),
        ),
      );

      g.bench_function(
        BenchmarkId::new("rscrypto-state", out_len),
        checked_with(
          vec![0u8; out_len],
          |out| {
            state
              .derive(black_box(&salt), iterations, black_box(out.as_mut_slice()))
              .expect("valid authentication benchmark operation must succeed");
            black_box(out[0])
          },
          |_, out| assert_eq!(out, &expected),
        ),
      );

      g.bench_function(
        BenchmarkId::new("rustcrypto", out_len),
        checked_with(
          vec![0u8; out_len],
          |out| {
            pbkdf2::pbkdf2_hmac::<sha2::Sha256>(
              black_box(&password),
              black_box(&salt),
              iterations,
              black_box(out.as_mut_slice()),
            );
            black_box(out[0])
          },
          |_, out| assert_eq!(out, &expected),
        ),
      );

      aws_lc_bench! {
        g.bench_function(
          BenchmarkId::new("aws-lc-rs", out_len),
          checked_with(
            vec![0u8; out_len],
            |out| {
              aws_lc_rs::pbkdf2::derive(
                aws_lc_rs::pbkdf2::PBKDF2_HMAC_SHA256,
                nz_iters,
                black_box(&salt),
                black_box(&password),
                black_box(out.as_mut_slice()),
              );
              black_box(out[0])
            },
            |_, out| assert_eq!(out, &expected),
          ),
        );
      }

      g.bench_function(
        BenchmarkId::new("ring", out_len),
        checked_with(
          vec![0u8; out_len],
          |out| {
            ring::pbkdf2::derive(
              ring::pbkdf2::PBKDF2_HMAC_SHA256,
              nz_iters,
              black_box(&salt),
              black_box(&password),
              black_box(out.as_mut_slice()),
            );
            black_box(out[0])
          },
          |_, out| assert_eq!(out, &expected),
        ),
      );
    }

    g.finish();
  }
}

fn pbkdf2_sha512_derive(c: &mut Criterion) {
  if !bench_config::selected("pbkdf2-sha512") {
    return;
  }
  print_auth_diag_once();

  let password = [0x66u8; 48];
  let salt = [0x44u8; 16];
  let state = Pbkdf2Sha512::new(&password);
  // Every row must write RustCrypto's derived key into its reused output buffer.
  let rustcrypto = |iterations: u32, len: usize| {
    let mut out = vec![0u8; len];
    pbkdf2::pbkdf2_hmac::<sha2::Sha512>(&password, &salt, iterations, &mut out);
    out
  };

  for &iterations in &[1u32, 100, 1000] {
    let nz_iters =
      core::num::NonZeroU32::new(iterations).expect("valid authentication benchmark operation must succeed");
    let mut g = c.benchmark_group(format!("pbkdf2-sha512/iters={iterations}"));

    for &out_len in &[64usize, 128] {
      g.throughput(criterion::Throughput::Bytes(out_len as u64));
      let expected = rustcrypto(iterations, out_len);

      g.bench_function(
        BenchmarkId::new("rscrypto", out_len),
        checked_with(
          vec![0u8; out_len],
          |out| {
            Pbkdf2Sha512::derive_key_primitive(
              black_box(&password),
              black_box(&salt),
              iterations,
              black_box(out.as_mut_slice()),
            )
            .expect("valid authentication benchmark operation must succeed");
            black_box(out[0])
          },
          |_, out| assert_eq!(out, &expected),
        ),
      );

      g.bench_function(
        BenchmarkId::new("rustcrypto", out_len),
        checked_with(
          vec![0u8; out_len],
          |out| {
            pbkdf2::pbkdf2_hmac::<sha2::Sha512>(
              black_box(&password),
              black_box(&salt),
              iterations,
              black_box(out.as_mut_slice()),
            );
            black_box(out[0])
          },
          |_, out| assert_eq!(out, &expected),
        ),
      );

      aws_lc_bench! {
        g.bench_function(
          BenchmarkId::new("aws-lc-rs", out_len),
          checked_with(
            vec![0u8; out_len],
            |out| {
              aws_lc_rs::pbkdf2::derive(
                aws_lc_rs::pbkdf2::PBKDF2_HMAC_SHA512,
                nz_iters,
                black_box(&salt),
                black_box(&password),
                black_box(out.as_mut_slice()),
              );
              black_box(out[0])
            },
            |_, out| assert_eq!(out, &expected),
          ),
        );
      }

      g.bench_function(
        BenchmarkId::new("ring", out_len),
        checked_with(
          vec![0u8; out_len],
          |out| {
            ring::pbkdf2::derive(
              ring::pbkdf2::PBKDF2_HMAC_SHA512,
              nz_iters,
              black_box(&salt),
              black_box(&password),
              black_box(out.as_mut_slice()),
            );
            black_box(out[0])
          },
          |_, out| assert_eq!(out, &expected),
        ),
      );
    }

    g.finish();

    let mut g_state = c.benchmark_group(format!("pbkdf2-sha512-state/iters={iterations}"));
    for &out_len in &[64usize, 128] {
      g_state.throughput(criterion::Throughput::Bytes(out_len as u64));
      let expected = rustcrypto(iterations, out_len);
      g_state.bench_function(
        BenchmarkId::new("rscrypto", out_len),
        checked_with(
          vec![0u8; out_len],
          |out| {
            state
              .derive(black_box(&salt), iterations, black_box(out.as_mut_slice()))
              .expect("valid authentication benchmark operation must succeed");
            black_box(out[0])
          },
          |_, out| assert_eq!(out, &expected),
        ),
      );
    }
    g_state.finish();
  }
}

fn ed25519_public_key(c: &mut Criterion) {
  if !bench_config::selected("ed25519/public-key-from-secret") {
    return;
  }
  let secret_bytes = [7u8; 32];
  let expected = SigningKey::from_bytes(&secret_bytes).verifying_key().to_bytes();
  let mut g = c.benchmark_group("ed25519/public-key-from-secret");

  g.bench_function(
    "rscrypto",
    checked(
      || {
        let secret = Ed25519SecretKey::from_bytes(*black_box(&secret_bytes));
        black_box(secret.public_key())
      },
      |public| assert_eq!(public.to_bytes(), expected),
    ),
  );

  g.bench_function(
    "dalek",
    checked(
      || {
        let signing_key = SigningKey::from_bytes(black_box(&secret_bytes));
        black_box(signing_key.verifying_key())
      },
      |public| assert_eq!(public.to_bytes(), expected),
    ),
  );

  g.finish();
}

fn ecdsa_p256_verify(c: &mut Criterion) {
  if !bench_config::selected("ecdsa-p256/verify") {
    return;
  }
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
    // Every row verifies RustCrypto's signature and must reject it once the last byte of r changes.
    let oracle_signature: P256OracleSignature = p256::ecdsa::signature::Signer::sign(&signing_key, data);
    let signature_bytes: [u8; 64] = array_from_slice(oracle_signature.to_bytes().as_ref());
    let signature =
      EcdsaP256Signature::from_bytes(signature_bytes).expect("valid authentication benchmark operation must succeed");
    let mut tampered_bytes = signature_bytes;
    tampered_bytes[31] ^= 1;
    let tampered = EcdsaP256Signature::from_bytes(tampered_bytes).expect("a changed r remains a valid scalar");
    let oracle_tampered = P256OracleSignature::from_slice(&tampered_bytes).expect("a changed r remains a valid scalar");

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || {
          black_box(&public)
            .verify(black_box(data), black_box(&signature))
            .expect("valid authentication benchmark operation must succeed");
          black_box(())
        },
        |()| {
          public
            .verify(data, &tampered)
            .expect_err("a changed ECDSA P-256 signature must fail");
        },
      ),
    );

    g.bench_function(
      BenchmarkId::new("rscrypto-import", len),
      checked(
        || {
          EcdsaP256PublicKey::from_sec1_bytes(black_box(sec1.as_slice()))
            .expect("valid authentication benchmark operation must succeed")
            .verify(black_box(data), black_box(&signature))
            .expect("valid authentication benchmark operation must succeed");
          black_box(())
        },
        |()| {
          EcdsaP256PublicKey::from_sec1_bytes(sec1.as_slice())
            .expect("valid authentication benchmark operation must succeed")
            .verify(data, &tampered)
            .expect_err("a changed ECDSA P-256 signature must fail");
        },
      ),
    );

    g.bench_function(
      BenchmarkId::new("rustcrypto-p256", len),
      checked(
        || {
          p256::ecdsa::signature::Verifier::verify(
            black_box(verifying_key),
            black_box(data),
            black_box(&oracle_signature),
          )
          .expect("valid authentication benchmark operation must succeed");
          black_box(())
        },
        |()| {
          p256::ecdsa::signature::Verifier::verify(verifying_key, data, &oracle_tampered)
            .expect_err("a changed ECDSA P-256 signature must fail");
        },
      ),
    );

    g.bench_function(
      BenchmarkId::new("ring", len),
      checked(
        || {
          ring_upk
            .verify(black_box(data), black_box(signature.as_bytes()))
            .expect("valid authentication benchmark operation must succeed");
          black_box(())
        },
        |()| {
          ring_upk
            .verify(data, &tampered_bytes)
            .expect_err("a changed ECDSA P-256 signature must fail");
        },
      ),
    );

    aws_lc_bench! {
      g.bench_function(
        BenchmarkId::new("aws-lc-rs", len),
        checked(
          || {
            aws_upk.verify(black_box(data), black_box(signature.as_bytes())).expect("valid authentication benchmark operation must succeed");
            black_box(())
          },
          |()| {
            aws_upk.verify(data, &tampered_bytes).expect_err("a changed ECDSA P-256 signature must fail");
          },
        ),
      );
    }
  }

  g.finish();
}

fn ecdsa_p256_sign(c: &mut Criterion) {
  if !bench_config::selected("ecdsa-p256/sign") {
    return;
  }
  let secret_bytes = [0x11u8; 32];
  let secret =
    EcdsaP256SecretKey::from_bytes(secret_bytes).expect("valid authentication benchmark operation must succeed");
  let keypair = EcdsaP256Keypair::from_secret_key(secret);
  let blind = [0x5cu8; 64];
  let signing_key =
    P256OracleSigningKey::from_slice(&secret_bytes).expect("valid authentication benchmark operation must succeed");
  let verifying_key = signing_key.verifying_key();
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
  // rscrypto's deterministic nonce is not RFC 6979, and the competitors draw random nonces, so no row has a
  // fixed signature. RustCrypto must verify every row's signature, and blinding must not change rscrypto's.
  let verifies = |data: &[u8], signature: &[u8]| {
    let signature = P256OracleSignature::from_slice(signature).expect("ECDSA P-256 signature must parse");
    p256::ecdsa::signature::Verifier::verify(verifying_key, data, &signature)
      .expect("RustCrypto must verify the ECDSA P-256 signature");
  };

  let inputs = [0usize, 32, 1024, 16384]
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect::<Vec<_>>();
  let mut g = c.benchmark_group("ecdsa-p256/sign");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_function(
      BenchmarkId::new("rscrypto-deterministic", len),
      checked(
        || {
          black_box(
            black_box(&keypair)
              .try_sign(black_box(data))
              .expect("valid authentication benchmark operation must succeed"),
          )
        },
        |signature| verifies(data, signature.as_bytes()),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rscrypto-blinded", len),
      checked(
        || {
          black_box(
            black_box(&keypair)
              .try_sign_blinded_with(black_box(data), |out| {
                out.copy_from_slice(black_box(&blind));
                Ok::<(), core::convert::Infallible>(())
              })
              .expect("valid authentication benchmark operation must succeed"),
          )
        },
        |signature| {
          let deterministic = keypair
            .try_sign(data)
            .expect("valid authentication benchmark operation must succeed");
          assert_eq!(signature, deterministic);
          verifies(data, signature.as_bytes());
        },
      ),
    );

    g.bench_function(
      BenchmarkId::new("rustcrypto-p256", len),
      checked(
        || {
          let signature: P256OracleSignature =
            p256::ecdsa::signature::Signer::sign(black_box(&signing_key), black_box(data));
          black_box(signature)
        },
        |signature| verifies(data, signature.to_bytes().as_slice()),
      ),
    );

    g.bench_function(
      BenchmarkId::new("ring", len),
      checked(
        || {
          black_box(
            ring_key
              .sign(&ring_rng, black_box(data))
              .expect("valid authentication benchmark operation must succeed"),
          )
        },
        |signature| verifies(data, signature.as_ref()),
      ),
    );

    aws_lc_bench! {
      g.bench_function(
        BenchmarkId::new("aws-lc-rs", len),
        checked(
          || black_box(aws_key.sign(&aws_rng, black_box(data)).expect("valid authentication benchmark operation must succeed")),
          |signature| verifies(data, signature.as_ref()),
        ),
      );
    }
  }

  g.finish();
}

fn ecdsa_p256_public_key(c: &mut Criterion) {
  if !bench_config::selected("ecdsa-p256/public-key") {
    return;
  }
  let secret_bytes = [0x11u8; 32];
  let secret =
    EcdsaP256SecretKey::from_bytes(secret_bytes).expect("valid authentication benchmark operation must succeed");
  let blind = [0x5cu8; 64];
  let oracle =
    P256OracleSecretKey::from_slice(&secret_bytes).expect("valid authentication benchmark operation must succeed");
  let expected = oracle.public_key().to_sec1_bytes();

  let mut g = c.benchmark_group("ecdsa-p256/public-key");
  g.bench_function(
    "rscrypto-blinded",
    checked(
      || {
        black_box(
          black_box(&secret)
            .try_public_key_blinded_with(|out| {
              out.copy_from_slice(black_box(&blind));
              Ok::<(), core::convert::Infallible>(())
            })
            .expect("valid authentication benchmark operation must succeed"),
        )
      },
      |public| assert_eq!(public.to_sec1_bytes().as_slice(), &*expected),
    ),
  );
  g.bench_function(
    "rustcrypto-p256",
    checked(
      || black_box(black_box(&oracle).public_key()),
      |public| assert_eq!(public.to_sec1_bytes(), expected),
    ),
  );
  g.finish();
}

fn ecdsa_p384_verify(c: &mut Criterion) {
  if !bench_config::selected("ecdsa-p384/verify") {
    return;
  }
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
    // Every row verifies RustCrypto's signature and must reject it once the last byte of r changes.
    let oracle_signature: P384OracleSignature = p384::ecdsa::signature::Signer::sign(&signing_key, data);
    let signature_bytes: [u8; 96] = array_from_slice(oracle_signature.to_bytes().as_ref());
    let signature =
      EcdsaP384Signature::from_bytes(signature_bytes).expect("valid authentication benchmark operation must succeed");
    let mut tampered_bytes = signature_bytes;
    tampered_bytes[47] ^= 1;
    let tampered = EcdsaP384Signature::from_bytes(tampered_bytes).expect("a changed r remains a valid scalar");
    let oracle_tampered = P384OracleSignature::from_slice(&tampered_bytes).expect("a changed r remains a valid scalar");

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || {
          black_box(&public)
            .verify(black_box(data), black_box(&signature))
            .expect("valid authentication benchmark operation must succeed");
          black_box(())
        },
        |()| {
          public
            .verify(data, &tampered)
            .expect_err("a changed ECDSA P-384 signature must fail");
        },
      ),
    );

    g.bench_function(
      BenchmarkId::new("rscrypto-import", len),
      checked(
        || {
          EcdsaP384PublicKey::from_sec1_bytes(black_box(sec1.as_slice()))
            .expect("valid authentication benchmark operation must succeed")
            .verify(black_box(data), black_box(&signature))
            .expect("valid authentication benchmark operation must succeed");
          black_box(())
        },
        |()| {
          EcdsaP384PublicKey::from_sec1_bytes(sec1.as_slice())
            .expect("valid authentication benchmark operation must succeed")
            .verify(data, &tampered)
            .expect_err("a changed ECDSA P-384 signature must fail");
        },
      ),
    );

    g.bench_function(
      BenchmarkId::new("rustcrypto-p384", len),
      checked(
        || {
          p384::ecdsa::signature::Verifier::verify(
            black_box(verifying_key),
            black_box(data),
            black_box(&oracle_signature),
          )
          .expect("valid authentication benchmark operation must succeed");
          black_box(())
        },
        |()| {
          p384::ecdsa::signature::Verifier::verify(verifying_key, data, &oracle_tampered)
            .expect_err("a changed ECDSA P-384 signature must fail");
        },
      ),
    );

    g.bench_function(
      BenchmarkId::new("ring", len),
      checked(
        || {
          ring_upk
            .verify(black_box(data), black_box(signature.as_bytes()))
            .expect("valid authentication benchmark operation must succeed");
          black_box(())
        },
        |()| {
          ring_upk
            .verify(data, &tampered_bytes)
            .expect_err("a changed ECDSA P-384 signature must fail");
        },
      ),
    );

    aws_lc_bench! {
      g.bench_function(
        BenchmarkId::new("aws-lc-rs", len),
        checked(
          || {
            aws_upk.verify(black_box(data), black_box(signature.as_bytes())).expect("valid authentication benchmark operation must succeed");
            black_box(())
          },
          |()| {
            aws_upk.verify(data, &tampered_bytes).expect_err("a changed ECDSA P-384 signature must fail");
          },
        ),
      );
    }
  }

  g.finish();
}

fn ecdsa_p384_sign(c: &mut Criterion) {
  if !bench_config::selected("ecdsa-p384/sign") {
    return;
  }
  let secret_bytes = [0x31u8; 48];
  let secret =
    EcdsaP384SecretKey::from_bytes(secret_bytes).expect("valid authentication benchmark operation must succeed");
  let keypair = EcdsaP384Keypair::from_secret_key(secret);
  let blind = [0xa3u8; 96];
  let signing_key =
    P384OracleSigningKey::from_slice(&secret_bytes).expect("valid authentication benchmark operation must succeed");
  let verifying_key = signing_key.verifying_key();
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
  // rscrypto's deterministic nonce is not RFC 6979, and the competitors draw random nonces, so no row has a
  // fixed signature. RustCrypto must verify every row's signature, and blinding must not change rscrypto's.
  let verifies = |data: &[u8], signature: &[u8]| {
    let signature = P384OracleSignature::from_slice(signature).expect("ECDSA P-384 signature must parse");
    p384::ecdsa::signature::Verifier::verify(verifying_key, data, &signature)
      .expect("RustCrypto must verify the ECDSA P-384 signature");
  };

  let inputs = [0usize, 32, 1024, 16384]
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect::<Vec<_>>();
  let mut g = c.benchmark_group("ecdsa-p384/sign");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_function(
      BenchmarkId::new("rscrypto-deterministic", len),
      checked(
        || {
          black_box(
            black_box(&keypair)
              .try_sign(black_box(data))
              .expect("valid authentication benchmark operation must succeed"),
          )
        },
        |signature| verifies(data, signature.as_bytes()),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rscrypto-blinded", len),
      checked(
        || {
          black_box(
            black_box(&keypair)
              .try_sign_blinded_with(black_box(data), |out| {
                out.copy_from_slice(black_box(&blind));
                Ok::<(), core::convert::Infallible>(())
              })
              .expect("valid authentication benchmark operation must succeed"),
          )
        },
        |signature| {
          let deterministic = keypair
            .try_sign(data)
            .expect("valid authentication benchmark operation must succeed");
          assert_eq!(signature, deterministic);
          verifies(data, signature.as_bytes());
        },
      ),
    );

    g.bench_function(
      BenchmarkId::new("rustcrypto-p384", len),
      checked(
        || {
          let signature: P384OracleSignature =
            p384::ecdsa::signature::Signer::sign(black_box(&signing_key), black_box(data));
          black_box(signature)
        },
        |signature| verifies(data, signature.to_bytes().as_slice()),
      ),
    );

    g.bench_function(
      BenchmarkId::new("ring", len),
      checked(
        || {
          black_box(
            ring_key
              .sign(&ring_rng, black_box(data))
              .expect("valid authentication benchmark operation must succeed"),
          )
        },
        |signature| verifies(data, signature.as_ref()),
      ),
    );

    aws_lc_bench! {
      g.bench_function(
        BenchmarkId::new("aws-lc-rs", len),
        checked(
          || black_box(aws_key.sign(&aws_rng, black_box(data)).expect("valid authentication benchmark operation must succeed")),
          |signature| verifies(data, signature.as_ref()),
        ),
      );
    }
  }

  g.finish();
}

fn ecdsa_p384_public_key(c: &mut Criterion) {
  if !bench_config::selected("ecdsa-p384/public-key") {
    return;
  }
  let secret_bytes = [0x31u8; 48];
  let secret =
    EcdsaP384SecretKey::from_bytes(secret_bytes).expect("valid authentication benchmark operation must succeed");
  let blind = [0xa3u8; 96];
  let oracle =
    P384OracleSecretKey::from_slice(&secret_bytes).expect("valid authentication benchmark operation must succeed");
  let expected = oracle.public_key().to_sec1_bytes();

  let mut g = c.benchmark_group("ecdsa-p384/public-key");
  g.bench_function(
    "rscrypto-blinded",
    checked(
      || {
        black_box(
          black_box(&secret)
            .try_public_key_blinded_with(|out| {
              out.copy_from_slice(black_box(&blind));
              Ok::<(), core::convert::Infallible>(())
            })
            .expect("valid authentication benchmark operation must succeed"),
        )
      },
      |public| assert_eq!(public.to_sec1_bytes().as_slice(), &*expected),
    ),
  );
  g.bench_function(
    "rustcrypto-p384",
    checked(
      || black_box(black_box(&oracle).public_key()),
      |public| assert_eq!(public.to_sec1_bytes(), expected),
    ),
  );
  g.finish();
}

// Independent expectations for the diagnostic ECDSA component rows, from RustCrypto point arithmetic and
// crypto-bigint modular arithmetic. Components return little-endian 64-bit limbs. A Montgomery value is
// multiplied by 2^(64 * limbs) modulo its field prime or group order. A nonce row passes only when the
// x-coordinate of its k * G is the r of rscrypto's public signature.
#[cfg(all(rscrypto_internal, feature = "diag"))]
macro_rules! ecdsa_component_oracle {
  (
    $module:ident,
    $curve:ident,
    $uint:ident,
    $secret_key:ident,
    $hash:ty,
    $len:literal,
    $words:literal,
    $order:literal,
    $field:literal,
    $comb_width:literal,
    $comb_rows:literal,
    $comb_entries:literal
  ) => {
    mod $module {
      use sha2::Digest as _;
      use $curve::{
        AffinePoint, ProjectivePoint, Scalar,
        elliptic_curve::{bigint::NonZero, ops::Reduce, point::AffineCoordinates as _},
        $uint,
      };

      const ORDER: NonZero<$uint> = NonZero::<$uint>::new_unwrap(<$uint>::from_be_hex($order));
      const FIELD: NonZero<$uint> = NonZero::<$uint>::new_unwrap(<$uint>::from_be_hex($field));

      /// The integer whose little-endian 64-bit limbs a component returned.
      pub(super) fn from_limbs(limbs: &[u64]) -> $uint {
        let bytes: Vec<u8> = limbs.iter().rev().flat_map(|limb| limb.to_be_bytes()).collect();
        <$uint>::from_be_slice(&bytes)
      }

      /// The coordinate pair, or `r || s`, of a two-integer component.
      pub(super) fn pair(limbs: &[u64]) -> ($uint, $uint) {
        let (first, second) = limbs.split_at($words);
        (from_limbs(first), from_limbs(second))
      }

      fn nonzero(value: $uint) -> $uint {
        if value == <$uint>::ZERO { <$uint>::ONE } else { value }
      }

      fn reduce(bytes: &[u8]) -> $uint {
        <$uint>::rem_wide((<$uint>::from_be_slice(bytes), <$uint>::ZERO), &ORDER)
      }

      fn mul(lhs: &$uint, rhs: &$uint) -> $uint {
        lhs.mul_mod(rhs, &ORDER)
      }

      fn inverse(value: &$uint) -> $uint {
        value.invert_mod(&ORDER).expect("a nonzero scalar is invertible")
      }

      /// Multiplies by 2^bits, which is 2^bits - modulus because each modulus exceeds 2^(bits - 1).
      fn montgomery(value: &$uint, modulus: &NonZero<$uint>) -> $uint {
        value.mul_mod(&<$uint>::ZERO.wrapping_sub(modulus.as_ref()), modulus)
      }

      /// Affine coordinates of `scalar * G`.
      fn base_mul(scalar: &$uint) -> ($uint, $uint) {
        let point =
          core::ops::Mul::mul(ProjectivePoint::GENERATOR, <Scalar as Reduce<$uint>>::reduce(scalar)).to_affine();
        (<$uint>::from_be_slice(&point.x()), <$uint>::from_be_slice(&point.y()))
      }

      /// rscrypto's nonzero reduction of a double-width big-endian value.
      pub(super) fn reduce_wide(bytes: &[u8]) -> $uint {
        let (high, low) = bytes.split_at($len);
        nonzero(<$uint>::rem_wide(
          (<$uint>::from_be_slice(low), <$uint>::from_be_slice(high)),
          &ORDER,
        ))
      }

      /// The nonce blind that rscrypto derives from caller bytes. s390x draws it from the second half, with
      /// the top bit cleared; other targets reduce all of the bytes.
      fn nonce_blind(blind: &[u8]) -> $uint {
        if cfg!(target_arch = "s390x") {
          let mut half = [0u8; $len];
          half.copy_from_slice(&blind[$len..]);
          half[0] &= 0x7f;
          nonzero(<$uint>::from_be_slice(&half))
        } else {
          reduce_wide(blind)
        }
      }

      /// The `r` of rscrypto's public deterministic signature.
      pub(super) fn signature_r(secret: [u8; $len], message: &[u8]) -> $uint {
        let signature = rscrypto::$secret_key::from_bytes(secret)
          .expect("valid authentication benchmark operation must succeed")
          .try_sign(message)
          .expect("valid authentication benchmark operation must succeed")
          .to_bytes();
        <$uint>::from_be_slice(&signature[..$len])
      }

      /// The nonce in `limbs`, once its point reproduces the public signature for `message`.
      pub(super) fn nonce(limbs: &[u64], secret: [u8; $len], message: &[u8]) -> $uint {
        let nonce = from_limbs(limbs);
        let (x, _) = base_mul(&nonce);
        assert_eq!(
          <$uint>::rem_wide((x, <$uint>::ZERO), &ORDER),
          signature_r(secret, message),
          "the component nonce must produce the public signature"
        );
        nonce
      }

      /// The fixed public `r` of the scalar rows: the generator's x-coordinate modulo n.
      pub(super) fn fixed_r() -> $uint {
        reduce(&AffinePoint::GENERATOR.x())
      }

      /// r * d mod n for the fixed `r`.
      pub(super) fn r_times_secret(secret: [u8; $len]) -> $uint {
        mul(&fixed_r(), &reduce(&secret))
      }

      /// k^-1 (z + r d) mod n for the fixed `r`, before low-s normalization.
      pub(super) fn raw_s(secret: [u8; $len], nonce: &$uint, message: &[u8]) -> $uint {
        let z = reduce(&<$hash>::digest(message));
        mul(&inverse(nonce), &z.add_mod(&r_times_secret(secret), &ORDER))
      }

      pub(super) fn low_s(s: $uint) -> $uint {
        if s > ORDER.as_ref().shr(1) {
          s.neg_mod(&ORDER)
        } else {
          s
        }
      }

      pub(super) fn montgomery_scalar(value: &$uint) -> $uint {
        montgomery(value, &ORDER)
      }

      pub(super) fn montgomery_inverse(value: &$uint) -> $uint {
        montgomery(&inverse(value), &ORDER)
      }

      /// The nonce times the blind that rscrypto derives from `blind`.
      pub(super) fn blinded(nonce: &$uint, blind: &[u8]) -> $uint {
        mul(nonce, &nonce_blind(blind))
      }

      /// Montgomery affine coordinates of `scalar * G`.
      pub(super) fn montgomery_point(scalar: &$uint) -> ($uint, $uint) {
        let (x, y) = base_mul(scalar);
        (montgomery(&x, &FIELD), montgomery(&y, &FIELD))
      }

      /// The signing-comb entry for `digit`: digit bit `c` adds 2^(c * rows) G. Digit 0 and digits past the
      /// table select entry 0, which holds G.
      pub(super) fn comb_entry(digit: u8) -> ($uint, $uint) {
        let mut scalar = <$uint>::ZERO;
        if u32::from(digit) < $comb_entries {
          for column in 0..$comb_width {
            if u32::from(digit).checked_shr(column).expect("comb column is below 32") & 1 == 1 {
              scalar = scalar.wrapping_add(&<$uint>::ONE.shl(column.strict_mul($comb_rows)));
            }
          }
        }
        montgomery_point(&nonzero(scalar))
      }
    }
  };
}

#[cfg(all(rscrypto_internal, feature = "diag", feature = "ecdsa-p256"))]
ecdsa_component_oracle!(
  p256_components,
  p256,
  U256,
  EcdsaP256SecretKey,
  sha2::Sha256,
  32,
  4,
  "FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551",
  "FFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF",
  7,
  37,
  128
);

#[cfg(all(rscrypto_internal, feature = "diag", feature = "ecdsa-p384"))]
ecdsa_component_oracle!(
  p384_components,
  p384,
  U384,
  EcdsaP384SecretKey,
  sha2::Sha384,
  48,
  6,
  "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFC7634D81F4372DDF581A0DB248B0A77AECEC196ACCC52973",
  "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFF0000000000000000FFFFFFFF",
  8,
  48,
  256
);

#[cfg(all(rscrypto_internal, feature = "diag", feature = "ecdsa-p256"))]
fn ecdsa_p256_internal(c: &mut Criterion) {
  if !bench_config::selected("ecdsa-p256/internal") {
    return;
  }
  use crate::p256_components as oracle;
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

  g.bench_function(
    "select-generator-affine",
    checked(
      || {
        black_box(diag_ecdsa_p256_select_signing_generator_affine_limb_digest(black_box(
          173,
        )))
      },
      |limbs| assert_eq!(oracle::pair(&limbs), oracle::comb_entry(173)),
    ),
  );
  g.bench_function(
    "reduce-wide-order",
    checked(
      || black_box(diag_ecdsa_p256_reduce_wide_order_limb_digest(black_box(nonce_wide))),
      |limbs| assert_eq!(oracle::from_limbs(&limbs), oracle::reduce_wide(&nonce_wide)),
    ),
  );
  g.bench_function(
    "order-mul-fixed-r",
    checked(
      || black_box(diag_ecdsa_p256_order_mul_fixed_r_limb_digest(black_box(secret))),
      |limbs| assert_eq!(oracle::from_limbs(&limbs), oracle::r_times_secret(secret)),
    ),
  );
  g.bench_function(
    "order-mul-blinded-fixed-r",
    checked(
      || {
        black_box(diag_ecdsa_p256_order_mul_blinded_fixed_r_limb_digest(
          black_box(secret),
          black_box(blind),
        ))
      },
      |limbs| assert_eq!(oracle::from_limbs(&limbs), oracle::r_times_secret(secret)),
    ),
  );

  for (len, data) in &inputs {
    let nonce = || oracle::nonce(&diag_ecdsa_p256_nonce_reduce_limb_digest(secret, data), secret, data);
    let wide_s = || oracle::raw_s(secret, &oracle::reduce_wide(&nonce_wide), data);

    g.bench_function(
      BenchmarkId::new("nonce-reduce", len),
      checked(
        || {
          black_box(diag_ecdsa_p256_nonce_reduce_limb_digest(
            black_box(secret),
            black_box(data),
          ))
        },
        |limbs| {
          oracle::nonce(&limbs, secret, data);
        },
      ),
    );
    g.bench_function(
      BenchmarkId::new("basepoint-blinded", len),
      checked(
        || {
          black_box(diag_ecdsa_p256_basepoint_blinded_limb_digest(
            black_box(secret),
            black_box(blind),
            black_box(data),
          ))
        },
        |limbs| assert_eq!(oracle::pair(&limbs), oracle::montgomery_point(&nonce())),
      ),
    );
    g.bench_function(
      BenchmarkId::new("scalar-finish", len),
      checked(
        || {
          black_box(diag_ecdsa_p256_scalar_finish_limb_digest(
            black_box(secret),
            black_box(nonce_wide),
            black_box(data),
          ))
        },
        |limbs| assert_eq!(oracle::pair(&limbs), (oracle::fixed_r(), oracle::low_s(wide_s()))),
      ),
    );
    g.bench_function(
      BenchmarkId::new("nonce-inverse", len),
      checked(
        || {
          black_box(diag_ecdsa_p256_nonce_inverse_limb_digest(
            black_box(secret),
            black_box(data),
          ))
        },
        |limbs| assert_eq!(oracle::from_limbs(&limbs), oracle::montgomery_inverse(&nonce())),
      ),
    );
    g.bench_function(
      BenchmarkId::new("nonce-inverse-blinded", len),
      checked(
        || {
          black_box(diag_ecdsa_p256_nonce_inverse_blinded_limb_digest(
            black_box(secret),
            black_box(blind),
            black_box(data),
          ))
        },
        |limbs| {
          assert_eq!(
            oracle::from_limbs(&limbs),
            oracle::montgomery_inverse(&oracle::blinded(&nonce(), &blind))
          )
        },
      ),
    );
    g.bench_function(
      BenchmarkId::new("final-multiply", len),
      checked(
        || {
          black_box(diag_ecdsa_p256_final_multiply_limb_digest(
            black_box(secret),
            black_box(nonce_wide),
            black_box(data),
          ))
        },
        |limbs| assert_eq!(oracle::from_limbs(&limbs), oracle::montgomery_scalar(&wide_s())),
      ),
    );
  }

  g.finish();
}

#[cfg(all(rscrypto_internal, feature = "diag", feature = "ecdsa-p384"))]
fn ecdsa_p384_internal(c: &mut Criterion) {
  if !bench_config::selected("ecdsa-p384/internal") {
    return;
  }
  use crate::p384_components as oracle;
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

  g.bench_function(
    "select-generator-affine",
    checked(
      || {
        black_box(diag_ecdsa_p384_select_signing_generator_affine_limb_digest(black_box(
          173,
        )))
      },
      |limbs| assert_eq!(oracle::pair(&limbs), oracle::comb_entry(173)),
    ),
  );
  g.bench_function(
    "reduce-wide-order",
    checked(
      || black_box(diag_ecdsa_p384_reduce_wide_order_limb_digest(black_box(nonce_wide))),
      |limbs| assert_eq!(oracle::from_limbs(&limbs), oracle::reduce_wide(&nonce_wide)),
    ),
  );
  g.bench_function(
    "order-mul-fixed-r",
    checked(
      || black_box(diag_ecdsa_p384_order_mul_fixed_r_limb_digest(black_box(secret))),
      |limbs| assert_eq!(oracle::from_limbs(&limbs), oracle::r_times_secret(secret)),
    ),
  );

  for (len, data) in &inputs {
    let nonce = || oracle::nonce(&diag_ecdsa_p384_nonce_reduce_limb_digest(secret, data), secret, data);
    let wide_s = || oracle::raw_s(secret, &oracle::reduce_wide(&nonce_wide), data);

    g.bench_function(
      BenchmarkId::new("nonce-reduce", len),
      checked(
        || {
          black_box(diag_ecdsa_p384_nonce_reduce_limb_digest(
            black_box(secret),
            black_box(data),
          ))
        },
        |limbs| {
          oracle::nonce(&limbs, secret, data);
        },
      ),
    );
    g.bench_function(
      BenchmarkId::new("basepoint-blinded", len),
      checked(
        || {
          black_box(diag_ecdsa_p384_basepoint_blinded_limb_digest(
            black_box(secret),
            black_box(blind),
            black_box(data),
          ))
        },
        |limbs| assert_eq!(oracle::pair(&limbs), oracle::montgomery_point(&nonce())),
      ),
    );
    g.bench_function(
      BenchmarkId::new("basepoint-r", len),
      checked(
        || {
          black_box(diag_ecdsa_p384_basepoint_r_limb_digest(
            black_box(secret),
            black_box(data),
          ))
        },
        |limbs| assert_eq!(oracle::from_limbs(&limbs), oracle::signature_r(secret, data)),
      ),
    );
    g.bench_function(
      BenchmarkId::new("scalar-finish", len),
      checked(
        || {
          black_box(diag_ecdsa_p384_scalar_finish_limb_digest(
            black_box(secret),
            black_box(nonce_wide),
            black_box(data),
          ))
        },
        |limbs| assert_eq!(oracle::pair(&limbs), (oracle::fixed_r(), oracle::low_s(wide_s()))),
      ),
    );
    g.bench_function(
      BenchmarkId::new("nonce-inverse", len),
      checked(
        || {
          black_box(diag_ecdsa_p384_nonce_inverse_limb_digest(
            black_box(secret),
            black_box(data),
          ))
        },
        |limbs| assert_eq!(oracle::from_limbs(&limbs), oracle::montgomery_inverse(&nonce())),
      ),
    );
    g.bench_function(
      BenchmarkId::new("nonce-inverse-blinded", len),
      checked(
        || {
          black_box(diag_ecdsa_p384_nonce_inverse_blinded_limb_digest(
            black_box(secret),
            black_box(blind),
            black_box(data),
          ))
        },
        |limbs| {
          assert_eq!(
            oracle::from_limbs(&limbs),
            oracle::montgomery_inverse(&oracle::blinded(&nonce(), &blind))
          )
        },
      ),
    );
    g.bench_function(
      BenchmarkId::new("final-multiply", len),
      checked(
        || {
          black_box(diag_ecdsa_p384_final_multiply_limb_digest(
            black_box(secret),
            black_box(nonce_wide),
            black_box(data),
          ))
        },
        |limbs| assert_eq!(oracle::from_limbs(&limbs), oracle::montgomery_scalar(&wide_s())),
      ),
    );
  }

  g.finish();
}

fn ed25519_keypair_from_secret(c: &mut Criterion) {
  if !bench_config::selected("ed25519/keypair-from-secret") {
    return;
  }
  let secret_bytes = [8u8; 32];
  let expected = SigningKey::from_bytes(&secret_bytes).verifying_key().to_bytes();
  let mut g = c.benchmark_group("ed25519/keypair-from-secret");

  g.bench_function(
    "rscrypto",
    checked(
      || {
        let secret = Ed25519SecretKey::from_bytes(*black_box(&secret_bytes));
        black_box(Ed25519Keypair::from_secret_key(secret))
      },
      |keypair| assert_eq!(keypair.public_key().to_bytes(), expected),
    ),
  );

  g.bench_function(
    "dalek",
    checked(
      || black_box(SigningKey::from_bytes(black_box(&secret_bytes))),
      |signing_key| assert_eq!(signing_key.verifying_key().to_bytes(), expected),
    ),
  );

  g.finish();
}

fn ed25519_sign(c: &mut Criterion) {
  if !bench_config::selected("ed25519/sign") {
    return;
  }
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
  let inputs = [0usize, 32, 1024, 16384]
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect::<Vec<_>>();
  let mut g = c.benchmark_group("ed25519/sign");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    // RFC 8032 signing is deterministic, so every row must return Dalek's signature.
    let expected = || signing_key.sign(data).to_bytes();

    g.bench_function(
      BenchmarkId::new("rscrypto-direct-secret", len),
      checked(
        || black_box(black_box(&secret).sign(black_box(data))),
        |signature| assert_eq!(signature.to_bytes(), expected()),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || black_box(black_box(&keypair).sign(black_box(data))),
        |signature| assert_eq!(signature.to_bytes(), expected()),
      ),
    );

    g.bench_function(
      BenchmarkId::new("dalek", len),
      checked(
        || black_box(black_box(&signing_key).sign(black_box(data))),
        |signature| assert_eq!(signature.to_bytes(), expected()),
      ),
    );

    aws_lc_bench! {
      g.bench_function(
        BenchmarkId::new("aws-lc-rs", len),
        checked(
          || black_box(aws_kp.sign(black_box(data))),
          |signature| assert_eq!(signature.as_ref(), expected().as_slice()),
        ),
      );
    }

    g.bench_function(
      BenchmarkId::new("ring", len),
      checked(
        || black_box(ring_kp.sign(black_box(data))),
        |signature| assert_eq!(signature.as_ref(), expected().as_slice()),
      ),
    );

    g.bench_function(
      BenchmarkId::new("dryoc", len),
      checked_with(
        [0u8; 64],
        |signature| {
          crypto_sign_detached(signature, black_box(data), &dryoc_sk)
            .expect("valid authentication benchmark operation must succeed");
          black_box(&*signature);
        },
        |(), signature| assert_eq!(*signature, expected()),
      ),
    );
  }

  g.finish();
}

fn ed25519_verify(c: &mut Criterion) {
  if !bench_config::selected("ed25519/verify") {
    return;
  }
  use dryoc::classic::crypto_sign::crypto_sign_verify_detached;

  let secret_bytes = [13u8; 32];
  let signing_key = SigningKey::from_bytes(&secret_bytes);
  let verifying_key = signing_key.verifying_key();
  let public_bytes = verifying_key.to_bytes();
  let public = Ed25519PublicKey::from_bytes(public_bytes);
  aws_lc_bench! {
    let aws_upk = aws_lc_rs::signature::UnparsedPublicKey::new(&aws_lc_rs::signature::ED25519, public_bytes);
  }
  let ring_upk = ring::signature::UnparsedPublicKey::new(&ring::signature::ED25519, public_bytes);
  let inputs = [0usize, 32, 1024, 16384]
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect::<Vec<_>>();
  let mut g = c.benchmark_group("ed25519/verify");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    // Every row verifies Dalek's signature, which RFC 8032 makes every library's signature for this key and
    // message, and must reject it once one byte of R changes.
    let signature_bytes = signing_key.sign(data).to_bytes();
    let mut tampered_bytes = signature_bytes;
    tampered_bytes[0] ^= 1;
    let ours = Ed25519Signature::from_bytes(signature_bytes);
    let tampered = Ed25519Signature::from_bytes(tampered_bytes);
    let dalek = DalekSignature::from_bytes(&signature_bytes);
    let dalek_tampered = DalekSignature::from_bytes(&tampered_bytes);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || {
          black_box(&public)
            .verify(black_box(data), black_box(&ours))
            .expect("valid authentication benchmark operation must succeed");
          black_box(())
        },
        |()| {
          public
            .verify(data, &tampered)
            .expect_err("a changed Ed25519 signature must fail");
        },
      ),
    );

    g.bench_function(
      BenchmarkId::new("rscrypto-import", len),
      checked(
        || {
          Ed25519PublicKey::from_bytes(*black_box(&public_bytes))
            .verify(black_box(data), black_box(&ours))
            .expect("valid authentication benchmark operation must succeed");
          black_box(())
        },
        |()| {
          Ed25519PublicKey::from_bytes(public_bytes)
            .verify(data, &tampered)
            .expect_err("a changed Ed25519 signature must fail");
        },
      ),
    );

    g.bench_function(
      BenchmarkId::new("dalek", len),
      checked(
        || {
          black_box(&verifying_key)
            .verify_strict(black_box(data), black_box(&dalek))
            .expect("valid authentication benchmark operation must succeed");
          black_box(())
        },
        |()| {
          verifying_key
            .verify_strict(data, &dalek_tampered)
            .expect_err("a changed Ed25519 signature must fail");
        },
      ),
    );

    aws_lc_bench! {
      g.bench_function(
        BenchmarkId::new("aws-lc-rs", len),
        checked(
          || {
            aws_upk.verify(black_box(data), &signature_bytes).expect("valid authentication benchmark operation must succeed");
            black_box(())
          },
          |()| {
            aws_upk.verify(data, &tampered_bytes).expect_err("a changed Ed25519 signature must fail");
          },
        ),
      );
    }

    g.bench_function(
      BenchmarkId::new("ring", len),
      checked(
        || {
          ring_upk
            .verify(black_box(data), &signature_bytes)
            .expect("valid authentication benchmark operation must succeed");
          black_box(())
        },
        |()| {
          ring_upk
            .verify(data, &tampered_bytes)
            .expect_err("a changed Ed25519 signature must fail");
        },
      ),
    );

    g.bench_function(
      BenchmarkId::new("dryoc", len),
      checked(
        || {
          crypto_sign_verify_detached(&signature_bytes, black_box(data), &public_bytes)
            .expect("valid authentication benchmark operation must succeed");
          black_box(())
        },
        |()| {
          crypto_sign_verify_detached(&tampered_bytes, data, &public_bytes)
            .expect_err("a changed Ed25519 signature must fail");
        },
      ),
    );
  }

  g.finish();
}

/// Adds two little-endian 256-bit integers whose sum fits in 256 bits.
#[cfg(all(rscrypto_internal, feature = "diag"))]
fn add_le_256(lhs: &[u8; 32], rhs: &[u8; 32]) -> [u8; 32] {
  let mut carry = 0u8;
  let sum = core::array::from_fn(|index| {
    let total = u16::from(lhs[index])
      .strict_add(u16::from(rhs[index]))
      .strict_add(u16::from(carry));
    let [low, high] = total.to_le_bytes();
    carry = high;
    low
  });
  assert_eq!(carry, 0, "the sum must fit in 256 bits");
  sum
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
fn ed25519_verify_phase(c: &mut Criterion) {
  if !bench_config::selected("ed25519/verify-phase") {
    return;
  }
  use rscrypto::auth::{
    diag_ed25519_verify_challenge_reduce_digest, diag_ed25519_verify_portable_double_scalar_digest,
    diag_ed25519_verify_public_decode_digest, diag_ed25519_verify_r_decode_digest, diag_ed25519_verify_scalars,
  };

  print_auth_diag_once();

  // Dalek's signature satisfies [s]B + [-k]A = R. Decoded points must re-encode to their inputs, and the
  // double-scalar rows must return R; with that equation, k + (-k) must equal the group order L.
  let order = hex::<32>("edd3f55c1a631258d69cf7a2def9de1400000000000000000000000000000010");
  let signing_key = SigningKey::from_bytes(&[13u8; 32]);
  let public = Ed25519PublicKey::from_bytes(signing_key.verifying_key().to_bytes());
  let inputs = [0usize, 32, 1024, 16384]
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect::<Vec<_>>();
  let mut g = c.benchmark_group("ed25519/verify-phase");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let signature = Ed25519Signature::from_bytes(signing_key.sign(data).to_bytes());
    let scalars = diag_ed25519_verify_scalars(&public, &signature, data)
      .expect("valid authentication benchmark operation must succeed");
    let r_bytes = scalars.r_bytes;

    g.bench_function(
      BenchmarkId::new("challenge-reduce", len),
      checked(
        || {
          black_box(diag_ed25519_verify_challenge_reduce_digest(
            black_box(&public),
            black_box(&signature),
            black_box(data),
          ))
        },
        |challenge| {
          let point = diag_ed25519_verify_portable_double_scalar_digest(
            &scalars.s_canonical,
            &scalars.neg_challenge,
            &scalars.public_key,
          );
          assert_eq!(point, r_bytes);
          assert_eq!(add_le_256(&challenge, &scalars.neg_challenge), order);
        },
      ),
    );

    g.bench_function(
      BenchmarkId::new("public-decode", len),
      checked(
        || black_box(diag_ed25519_verify_public_decode_digest(black_box(&scalars.public_key))),
        |encoded| assert_eq!(encoded, *public.as_bytes()),
      ),
    );

    g.bench_function(
      BenchmarkId::new("r-decode", len),
      checked(
        || black_box(diag_ed25519_verify_r_decode_digest(black_box(&scalars.r_bytes))),
        |encoded| assert_eq!(encoded, r_bytes),
      ),
    );

    g.bench_function(
      BenchmarkId::new("portable-double-scalar", len),
      checked(
        || {
          black_box(diag_ed25519_verify_portable_double_scalar_digest(
            black_box(&scalars.s_canonical),
            black_box(&scalars.neg_challenge),
            black_box(&scalars.public_key),
          ))
        },
        |point| assert_eq!(point, r_bytes),
      ),
    );

    #[cfg(all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      not(feature = "portable-only"),
      not(miri)
    ))]
    g.bench_function(
      BenchmarkId::new("aarch64-asm-double-scalar", len),
      checked(
        || {
          black_box(rscrypto::auth::diag_ed25519_verify_aarch64_asm_double_scalar_digest(
            black_box(&scalars.s_canonical),
            black_box(&scalars.neg_challenge),
            black_box(&scalars.public_key),
          ))
        },
        |point| assert_eq!(point, r_bytes),
      ),
    );
  }

  g.finish();
}

/// RFC 7748 Section 6.1 X25519 vector through the public rscrypto API.
fn x25519_rfc7748_check() {
  let alice = X25519SecretKey::from_bytes(hex("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a"));
  let bob = X25519SecretKey::from_bytes(hex("5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb"));
  assert_eq!(
    alice.public_key().to_bytes(),
    hex("8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a")
  );
  assert_eq!(
    bob.public_key().to_bytes(),
    hex("de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f")
  );
  let shared = alice
    .diffie_hellman(&bob.public_key())
    .expect("the RFC 7748 public key is valid");
  assert_eq!(
    *shared.as_bytes(),
    hex::<32>("4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742")
  );
}

// `ring` is omitted from x25519 benches: ring 0.17 only exposes
// `EphemeralPrivateKey` (consumed by `agree_ephemeral`) and provides no
// reusable static-key API. Including it would force a full keygen-and-discard
// per iteration, which is not apples-to-apples against the static-key DH
// path that rscrypto / dalek / aws-lc-rs / dryoc all share.
fn x25519_public_key(c: &mut Criterion) {
  if !bench_config::selected("x25519/public-key-from-secret") {
    return;
  }
  use dryoc::classic::crypto_core::crypto_scalarmult_base;

  let secret_bytes = [0x2au8; 32];
  // Every row must return Dalek's public key.
  let expected = DalekX25519PublicKey::from(&DalekX25519Secret::from(secret_bytes)).to_bytes();
  let mut g = c.benchmark_group("x25519/public-key-from-secret");

  g.bench_function(
    "rscrypto",
    checked(
      || {
        let secret = X25519SecretKey::from_bytes(*black_box(&secret_bytes));
        black_box(secret.public_key())
      },
      |public| assert_eq!(public.to_bytes(), expected),
    ),
  );

  g.bench_function(
    "dalek",
    checked(
      || {
        let secret = DalekX25519Secret::from(*black_box(&secret_bytes));
        black_box(DalekX25519PublicKey::from(&secret))
      },
      |public| assert_eq!(public.to_bytes(), expected),
    ),
  );

  aws_lc_bench! {
    g.bench_function(
      "aws-lc-rs",
      checked(
        || {
          let priv_key =
            aws_lc_rs::agreement::PrivateKey::from_private_key(&aws_lc_rs::agreement::X25519, black_box(&secret_bytes))
              .expect("valid authentication benchmark operation must succeed");
          black_box(priv_key.compute_public_key().expect("valid authentication benchmark operation must succeed"))
        },
        |public| assert_eq!(public.as_ref(), expected.as_slice()),
      ),
    );
  }

  let mut public = [0u8; 32];
  g.bench_function(
    "dryoc",
    checked(
      || {
        crypto_scalarmult_base(&mut public, black_box(&secret_bytes));
        black_box(public)
      },
      |public| assert_eq!(public, expected),
    ),
  );

  g.finish();
}

fn x25519_diffie_hellman(c: &mut Criterion) {
  if !bench_config::selected("x25519/diffie-hellman") {
    return;
  }
  use dryoc::classic::crypto_core::{crypto_scalarmult, crypto_scalarmult_base};

  let alice_bytes = [0x18u8; 32];
  let bob_bytes = [0x34u8; 32];

  let alice = X25519SecretKey::from_bytes(alice_bytes);
  let bob_public = X25519SecretKey::from_bytes(bob_bytes).public_key();
  let dalek_alice = DalekX25519Secret::from(alice_bytes);
  let dalek_bob_public = DalekX25519PublicKey::from(&DalekX25519Secret::from(bob_bytes));
  // Every row must return Dalek's shared secret; the rscrypto row also reproduces RFC 7748.
  let expected = *dalek_alice.diffie_hellman(&dalek_bob_public).as_bytes();
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

  g.bench_function(
    "rscrypto",
    checked(
      || {
        black_box(
          black_box(&alice)
            .diffie_hellman(black_box(&bob_public))
            .expect("valid authentication benchmark operation must succeed"),
        )
      },
      |shared| {
        assert_eq!(*shared.as_bytes(), expected);
        x25519_rfc7748_check();
      },
    ),
  );

  g.bench_function(
    "dalek",
    checked(
      || black_box(black_box(&dalek_alice).diffie_hellman(black_box(&dalek_bob_public))),
      |shared| assert_eq!(*shared.as_bytes(), expected),
    ),
  );

  aws_lc_bench! {
    g.bench_function(
      "aws-lc-rs",
      checked(
        || {
          let shared = aws_lc_rs::agreement::agree(black_box(&aws_alice), black_box(&aws_bob_unparsed), (), |bytes| {
            let mut out = [0u8; 32];
            out.copy_from_slice(bytes);
            Ok::<[u8; 32], ()>(out)
          })
          .expect("valid authentication benchmark operation must succeed");
          black_box(shared)
        },
        |shared| assert_eq!(shared, expected),
      ),
    );
  }

  let mut shared = [0u8; 32];
  g.bench_function(
    "dryoc",
    checked(
      || {
        crypto_scalarmult(&mut shared, black_box(&alice_bytes), black_box(&dryoc_bob_pub))
          .expect("valid authentication benchmark operation must succeed");
        black_box(shared)
      },
      |shared| assert_eq!(shared, expected),
    ),
  );

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

// Fixed-key rows must reproduce the NIST CAVP vector above. The parse rows must agree with RustCrypto when they
// pair the parsed key with the NIST private key.
fn p256_ecdh_key_generation(c: &mut Criterion) {
  if !bench_config::selected("p256-ecdh/key-generation") {
    return;
  }
  let scalar = P256_NIST_PRIVATE;
  let mut g = c.benchmark_group("p256-ecdh/key-generation");
  g.bench_function(
    "rscrypto-caller-fill",
    checked(
      || black_box(p256_ephemeral(*black_box(&scalar))),
      |secret| assert_eq!(secret.public_key().to_sec1_bytes(), P256_NIST_PUBLIC),
    ),
  );
  g.bench_function(
    "rustcrypto-p256-import",
    checked(
      || black_box(P256OracleSecretKey::from_slice(black_box(&scalar)).expect("valid oracle scalar")),
      |secret| assert_eq!(secret.public_key().to_sec1_point(false).as_bytes(), P256_NIST_PUBLIC),
    ),
  );
  g.bench_function(
    "crrl-import",
    checked(
      || black_box(crrl::p256::PrivateKey::decode(black_box(&scalar)).expect("valid CRRL scalar")),
      |secret| assert_eq!(secret.to_public_key().encode_uncompressed(), P256_NIST_PUBLIC),
    ),
  );
  aws_lc_bench! {
    g.bench_function(
      "aws-lc-rs-native-import-and-precompute",
      checked(
        || {
          black_box(
            aws_lc_rs::agreement::PrivateKey::from_private_key(
              &aws_lc_rs::agreement::ECDH_P256,
              black_box(&scalar),
            )
            .expect("valid AWS-LC scalar"),
          )
        },
        |secret| assert_eq!(secret.compute_public_key().expect("AWS-LC public derivation").as_ref(), P256_NIST_PUBLIC),
      ),
    );
  }
  g.finish();
}

fn p256_ecdh_public_key(c: &mut Criterion) {
  if !bench_config::selected("p256-ecdh/public-key") {
    return;
  }
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
  g.bench_function(
    "rscrypto-selected",
    checked(
      || black_box(ours.public_key()),
      |public| assert_eq!(public.to_sec1_bytes(), P256_NIST_PUBLIC),
    ),
  );
  g.bench_function(
    "rscrypto-ecdsa-public-api",
    checked(
      || {
        let secret = EcdsaP256SecretKey::from_bytes(*black_box(&scalar)).expect("valid ECDSA benchmark scalar");
        black_box(secret.public_key())
      },
      |public| assert_eq!(public.to_sec1_bytes(), P256_NIST_PUBLIC),
    ),
  );
  g.bench_function(
    "rustcrypto-p256-pure-rust",
    checked(
      || black_box(rustcrypto.public_key().to_sec1_point(false)),
      |public| assert_eq!(public.as_bytes(), P256_NIST_PUBLIC),
    ),
  );
  g.bench_function(
    "crrl-pure-rust",
    checked(
      || black_box(crrl::p256::Point::mulgen(black_box(&crrl_scalar)).encode_uncompressed()),
      |public| assert_eq!(public, P256_NIST_PUBLIC),
    ),
  );
  g.bench_function(
    "libcrux-hacl-pure-rust",
    checked(
      || {
        let mut public = [0u8; 64];
        assert!(libcrux_p256::dh_initiator(black_box(&mut public), black_box(&scalar)));
        black_box(public)
      },
      |public| assert_eq!(public, P256_NIST_PUBLIC[1..]),
    ),
  );
  aws_lc_bench! {
    g.bench_function(
      "aws-lc-rs-native-cached",
      checked(
        || black_box(aws_secret.compute_public_key().expect("AWS-LC public derivation")),
        |public| assert_eq!(public.as_ref(), P256_NIST_PUBLIC),
      ),
    );
    g.bench_function(
      "aws-lc-rs-native-import-and-public",
      checked(
        || {
          let secret = aws_lc_rs::agreement::PrivateKey::from_private_key(
            &aws_lc_rs::agreement::ECDH_P256,
            black_box(&scalar),
          )
          .expect("valid AWS-LC scalar");
          black_box(secret.compute_public_key().expect("AWS-LC public derivation"))
        },
        |public| assert_eq!(public.as_ref(), P256_NIST_PUBLIC),
      ),
    );
  }
  ring_p256_bench! {
    let rng = ring::rand::SystemRandom::new();
    let generate = || {
      ring::agreement::EphemeralPrivateKey::generate(&ring::agreement::ECDH_P256, &rng)
        .expect("ring scalar generation")
    };
    g.bench_function(
      "ring-native",
      checked_routine(
        (),
        |_| {
          // A random key has no fixed public key: its agreement with the NIST public key must match
          // RustCrypto's agreement between the NIST private key and the derived public key.
          let secret = generate();
          let public = secret.compute_public_key().expect("ring public derivation");
          let nist_public = ring::agreement::UnparsedPublicKey::new(&ring::agreement::ECDH_P256, P256_NIST_PUBLIC);
          let shared = ring::agreement::agree_ephemeral(secret, &nist_public, array_from_slice::<32>)
            .expect("ring benchmark agreement");
          let ring_public = p256::PublicKey::from_sec1_bytes(public.as_ref()).expect("valid ring public key");
          let expected = p256::ecdh::diffie_hellman(rustcrypto.to_nonzero_scalar(), ring_public.as_affine());
          assert_eq!(shared.as_slice(), expected.raw_secret_bytes().as_slice());
        },
        |b, _| {
          b.iter_batched(
            &generate,
            |secret| black_box(secret.compute_public_key().expect("ring public derivation")),
            BatchSize::SmallInput,
          )
        },
      ),
    );
  }
  g.finish();
}

fn p256_ecdh_parse(c: &mut Criterion) {
  if !bench_config::selected("p256-ecdh/parse") {
    return;
  }
  let encoded: [u8; 65] = array_from_slice(
    P256OracleSecretKey::from_slice(&[0x24; 32])
      .expect("valid RustCrypto scalar")
      .public_key()
      .to_sec1_point(false)
      .as_bytes(),
  );
  let rustcrypto_secret = P256OracleSecretKey::from_slice(&P256_NIST_PRIVATE).expect("valid RustCrypto scalar");
  let rustcrypto_shared = |peer: &p256::PublicKey| -> [u8; 32] {
    array_from_slice(
      p256::ecdh::diffie_hellman(rustcrypto_secret.to_nonzero_scalar(), peer.as_affine())
        .raw_secret_bytes()
        .as_slice(),
    )
  };
  let expected = rustcrypto_shared(&p256::PublicKey::from_sec1_bytes(&encoded).expect("valid oracle point"));
  aws_lc_bench! {
    let aws_secret = aws_lc_rs::agreement::PrivateKey::from_private_key(
      &aws_lc_rs::agreement::ECDH_P256,
      &P256_NIST_PRIVATE,
    )
    .expect("valid AWS-LC scalar");
  }
  let mut g = c.benchmark_group("p256-ecdh/parse");
  g.bench_function(
    "rscrypto",
    checked(
      || black_box(P256PublicKey::from_sec1_bytes(black_box(&encoded)).expect("valid benchmark point")),
      |public| {
        assert_eq!(
          *p256_ephemeral(P256_NIST_PRIVATE).diffie_hellman(&public).as_bytes(),
          expected
        )
      },
    ),
  );
  g.bench_function(
    "rustcrypto-p256-pure-rust",
    checked(
      || black_box(p256::PublicKey::from_sec1_bytes(black_box(&encoded)).expect("valid oracle point")),
      |public| assert_eq!(rustcrypto_shared(&public), expected),
    ),
  );
  g.bench_function(
    "crrl-pure-rust",
    checked(
      || black_box(crrl::p256::Point::decode(black_box(&encoded)).expect("valid CRRL point")),
      |point| {
        let shared = core::ops::Mul::mul(point, crrl_p256_scalar(P256_NIST_PRIVATE)).encode_uncompressed();
        assert_eq!(&shared[1..33], &expected);
      },
    ),
  );
  aws_lc_bench! {
    g.bench_function(
      "aws-lc-rs-native",
      checked(
        || {
          let unparsed = aws_lc_rs::agreement::UnparsedPublicKey::new(
            &aws_lc_rs::agreement::ECDH_P256,
            black_box(encoded),
          );
          black_box(
            aws_lc_rs::agreement::ParsedPublicKey::try_from(unparsed).expect("valid AWS-LC point"),
          )
        },
        |public| {
          let shared = aws_lc_rs::agreement::agree(&aws_secret, public, (), |bytes| {
            Ok::<[u8; 32], ()>(array_from_slice(bytes))
          })
          .expect("AWS-LC benchmark agreement");
          assert_eq!(shared, expected);
        },
      ),
    );
  }
  g.finish();
}

fn p256_ecdh_agreement(c: &mut Criterion) {
  if !bench_config::selected("p256-ecdh/agreement") {
    return;
  }
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
  g.bench_function(
    "rscrypto-selected",
    checked_routine(
      (),
      |_| {
        assert_eq!(
          p256_ephemeral(scalar).diffie_hellman(&peer).as_bytes(),
          &P256_NIST_SHARED
        )
      },
      |b, _| {
        b.iter_batched(
          || p256_ephemeral(scalar),
          |secret| black_box(secret.diffie_hellman(black_box(&peer))),
          BatchSize::SmallInput,
        )
      },
    ),
  );
  g.bench_function(
    "rustcrypto-p256-pure-rust",
    checked(
      || {
        black_box(p256::ecdh::diffie_hellman(
          rustcrypto_secret.to_nonzero_scalar(),
          black_box(rustcrypto_peer.as_affine()),
        ))
      },
      |shared| assert_eq!(shared.raw_secret_bytes().as_slice(), P256_NIST_SHARED),
    ),
  );
  g.bench_function(
    "crrl-pure-rust",
    checked(
      || black_box(core::ops::Mul::mul(black_box(crrl_peer), black_box(crrl_scalar)).encode_uncompressed()),
      |shared| assert_eq!(&shared[1..33], &P256_NIST_SHARED),
    ),
  );
  g.bench_function(
    "libcrux-hacl-pure-rust",
    checked(
      || {
        let mut shared = [0u8; 64];
        assert!(libcrux_p256::dh_responder(
          black_box(&mut shared),
          black_box(&libcrux_peer),
          black_box(&scalar),
        ));
        black_box(shared)
      },
      |shared| assert_eq!(shared[..32], P256_NIST_SHARED),
    ),
  );
  aws_lc_bench! {
    g.bench_function(
      "aws-lc-rs-native",
      checked(
        || {
          black_box(
            aws_lc_rs::agreement::agree(&aws_secret, aws_peer.clone(), (), |bytes| {
              Ok::<[u8; 32], ()>(array_from_slice(bytes))
            })
            .expect("AWS-LC benchmark agreement"),
          )
        },
        |shared| assert_eq!(shared, P256_NIST_SHARED),
      ),
    );
  }
  ring_p256_bench! {
    let rng = ring::rand::SystemRandom::new();
    let ring_peer = ring::agreement::UnparsedPublicKey::new(&ring::agreement::ECDH_P256, P256_NIST_PEER);
    let generate = || {
      ring::agreement::EphemeralPrivateKey::generate(&ring::agreement::ECDH_P256, &rng)
        .expect("ring scalar generation")
    };
    g.bench_function(
      "ring-native",
      checked_routine(
        (),
        |_| {
          // The NIST peer has no published private key, so the check pairs a random key with the NIST public
          // key and compares RustCrypto's agreement between the NIST private key and the random public key.
          let secret = generate();
          let public = secret.compute_public_key().expect("ring public derivation");
          let nist_public = ring::agreement::UnparsedPublicKey::new(&ring::agreement::ECDH_P256, P256_NIST_PUBLIC);
          let shared = ring::agreement::agree_ephemeral(secret, &nist_public, array_from_slice::<32>)
            .expect("ring benchmark agreement");
          let ring_public = p256::PublicKey::from_sec1_bytes(public.as_ref()).expect("valid ring public key");
          let expected = p256::ecdh::diffie_hellman(rustcrypto_secret.to_nonzero_scalar(), ring_public.as_affine());
          assert_eq!(shared.as_slice(), expected.raw_secret_bytes().as_slice());
        },
        |b, _| {
          b.iter_batched(
            &generate,
            |secret| {
              black_box(
                ring::agreement::agree_ephemeral(secret, black_box(&ring_peer), array_from_slice::<32>)
                  .expect("ring benchmark agreement"),
              )
            },
            BatchSize::SmallInput,
          )
        },
      ),
    );
  }
  g.finish();
}

fn p256_ecdh_tls_roundtrip(c: &mut Criterion) {
  if !bench_config::selected("p256-ecdh/tls-shaped-roundtrip") {
    return;
  }
  let alice = [0x42; 32];
  let bob = [0x24; 32];
  // Both parties must derive RustCrypto's shared secret.
  let expected = {
    let alice = P256OracleSecretKey::from_slice(&alice).expect("valid RustCrypto scalar");
    let bob = P256OracleSecretKey::from_slice(&bob).expect("valid RustCrypto scalar");
    p256::ecdh::diffie_hellman(alice.to_nonzero_scalar(), bob.public_key().as_affine())
  };
  let mut g = c.benchmark_group("p256-ecdh/tls-shaped-roundtrip");
  g.bench_function(
    "rscrypto",
    checked(
      || {
        let alice_secret = p256_ephemeral(*black_box(&alice));
        let bob_secret = p256_ephemeral(*black_box(&bob));
        let alice_public = alice_secret.public_key().to_sec1_bytes();
        let bob_public = bob_secret.public_key().to_sec1_bytes();
        let alice_peer = P256PublicKey::from_sec1_bytes(black_box(&bob_public)).expect("valid benchmark peer");
        let bob_peer = P256PublicKey::from_sec1_bytes(black_box(&alice_public)).expect("valid benchmark peer");
        let alice_shared = alice_secret.diffie_hellman(&alice_peer);
        let bob_shared = bob_secret.diffie_hellman(&bob_peer);
        black_box((alice_shared, bob_shared))
      },
      |(alice_shared, bob_shared)| {
        assert_eq!(
          alice_shared.as_bytes().as_slice(),
          expected.raw_secret_bytes().as_slice()
        );
        assert_eq!(bob_shared.as_bytes().as_slice(), expected.raw_secret_bytes().as_slice());
      },
    ),
  );
  g.finish();
}

const P384_NIST_PRIVATE: [u8; 48] = [
  0x3c, 0xc3, 0x12, 0x2a, 0x68, 0xf0, 0xd9, 0x50, 0x27, 0xad, 0x38, 0xc0, 0x67, 0x91, 0x6b, 0xa0, 0xeb, 0x8c, 0x38,
  0x89, 0x4d, 0x22, 0xe1, 0xb1, 0x56, 0x18, 0xb6, 0x81, 0x8a, 0x66, 0x17, 0x74, 0xad, 0x46, 0x3b, 0x20, 0x5d, 0xa8,
  0x8c, 0xf6, 0x99, 0xab, 0x4d, 0x43, 0xc9, 0xcf, 0x98, 0xa1,
];
const P384_NIST_PUBLIC: [u8; 97] = [
  0x04, 0x98, 0x03, 0x80, 0x7f, 0x2f, 0x6d, 0x2f, 0xd9, 0x66, 0xcd, 0xd0, 0x29, 0x0b, 0xd4, 0x10, 0xc0, 0x19, 0x03,
  0x52, 0xfb, 0xec, 0x7f, 0xf6, 0x24, 0x7d, 0xe1, 0x30, 0x2d, 0xf8, 0x6f, 0x25, 0xd3, 0x4f, 0xe4, 0xa9, 0x7b, 0xef,
  0x60, 0xcf, 0xf5, 0x48, 0x35, 0x5c, 0x01, 0x5d, 0xbb, 0x3e, 0x5f, 0xba, 0x26, 0xca, 0x69, 0xec, 0x2f, 0x5b, 0x5d,
  0x9d, 0xad, 0x20, 0xcc, 0x9d, 0xa7, 0x11, 0x38, 0x3a, 0x9d, 0xbe, 0x34, 0xea, 0x3f, 0xa5, 0xa2, 0xaf, 0x75, 0xb4,
  0x65, 0x02, 0x62, 0x9a, 0xd5, 0x4d, 0xd8, 0xb7, 0xd7, 0x3a, 0x8a, 0xbb, 0x06, 0xa3, 0xa3, 0xbe, 0x47, 0xd6, 0x50,
  0xcc, 0x99,
];
const P384_NIST_PEER: [u8; 97] = [
  0x04, 0xa7, 0xc7, 0x6b, 0x97, 0x0c, 0x3b, 0x5f, 0xe8, 0xb0, 0x5d, 0x28, 0x38, 0xae, 0x04, 0xab, 0x47, 0x69, 0x7b,
  0x9e, 0xaf, 0x52, 0xe7, 0x64, 0x59, 0x2e, 0xfd, 0xa2, 0x7f, 0xe7, 0x51, 0x32, 0x72, 0x73, 0x44, 0x66, 0xb4, 0x00,
  0x09, 0x1a, 0xdb, 0xf2, 0xd6, 0x8c, 0x58, 0xe0, 0xc5, 0x00, 0x66, 0xac, 0x68, 0xf1, 0x9f, 0x2e, 0x1c, 0xb8, 0x79,
  0xae, 0xd4, 0x3a, 0x99, 0x69, 0xb9, 0x1a, 0x08, 0x39, 0xc4, 0xc3, 0x8a, 0x49, 0x74, 0x9b, 0x66, 0x1e, 0xfe, 0xdf,
  0x24, 0x34, 0x51, 0x91, 0x5e, 0xd0, 0x90, 0x5a, 0x32, 0xb0, 0x60, 0x99, 0x2b, 0x46, 0x8c, 0x64, 0x76, 0x6f, 0xc8,
  0x43, 0x7a,
];
const P384_NIST_SHARED: [u8; 48] = [
  0x5f, 0x9d, 0x29, 0xdc, 0x5e, 0x31, 0xa1, 0x63, 0x06, 0x03, 0x56, 0x21, 0x36, 0x69, 0xc8, 0xce, 0x13, 0x2e, 0x22,
  0xf5, 0x7c, 0x9a, 0x04, 0xf4, 0x0b, 0xa7, 0xfc, 0xea, 0xd4, 0x93, 0xb4, 0x57, 0xe5, 0x62, 0x1e, 0x76, 0x6c, 0x40,
  0xa2, 0xe3, 0xd4, 0xd6, 0xa0, 0x4b, 0x25, 0xe5, 0x33, 0xf1,
];

fn p384_ephemeral(bytes: [u8; 48]) -> P384EphemeralSecret {
  P384EphemeralSecret::try_generate_with(|candidate| {
    candidate.copy_from_slice(&bytes);
    Ok::<(), core::convert::Infallible>(())
  })
  .expect("valid authentication benchmark scalar must be accepted")
}

// Fixed-key rows must reproduce the NIST CAVP vector above.
fn p384_ecdh_key_generation(c: &mut Criterion) {
  if !bench_config::selected("p384-ecdh/key-generation") {
    return;
  }
  let scalar = P384_NIST_PRIVATE;
  let mut g = c.benchmark_group("p384-ecdh/key-generation");
  g.bench_function(
    "rscrypto-caller-fill",
    checked(
      || black_box(p384_ephemeral(*black_box(&scalar))),
      |secret| assert_eq!(secret.public_key().to_sec1_bytes(), P384_NIST_PUBLIC),
    ),
  );
  g.bench_function(
    "rustcrypto-p384-import",
    checked(
      || black_box(P384OracleSecretKey::from_slice(black_box(&scalar)).expect("valid oracle scalar")),
      |secret| assert_eq!(secret.public_key().to_sec1_point(false).as_bytes(), P384_NIST_PUBLIC),
    ),
  );
  aws_lc_bench! {
    g.bench_function(
      "aws-lc-rs-native-import-and-precompute",
      checked(
        || {
          black_box(
            aws_lc_rs::agreement::PrivateKey::from_private_key(
              &aws_lc_rs::agreement::ECDH_P384,
              black_box(&scalar),
            )
            .expect("valid AWS-LC scalar"),
          )
        },
        |secret| assert_eq!(secret.compute_public_key().expect("AWS-LC public derivation").as_ref(), P384_NIST_PUBLIC),
      ),
    );
  }
  g.finish();
}

fn p384_ecdh_public_key(c: &mut Criterion) {
  if !bench_config::selected("p384-ecdh/public-key") {
    return;
  }
  let scalar = P384_NIST_PRIVATE;
  let ours = p384_ephemeral(scalar);
  let rustcrypto = P384OracleSecretKey::from_slice(&scalar).expect("valid RustCrypto scalar");
  aws_lc_bench! {
    let aws_secret = aws_lc_rs::agreement::PrivateKey::from_private_key(
      &aws_lc_rs::agreement::ECDH_P384,
      &scalar,
    )
    .expect("valid AWS-LC scalar");
  }
  let mut g = c.benchmark_group("p384-ecdh/public-key");
  g.bench_function(
    "rscrypto-selected",
    checked(
      || black_box(ours.public_key()),
      |public| assert_eq!(public.to_sec1_bytes(), P384_NIST_PUBLIC),
    ),
  );
  g.bench_function(
    "rustcrypto-p384-pure-rust",
    checked(
      || black_box(rustcrypto.public_key().to_sec1_point(false)),
      |public| assert_eq!(public.as_bytes(), P384_NIST_PUBLIC),
    ),
  );
  aws_lc_bench! {
    g.bench_function(
      "aws-lc-rs-native-cached",
      checked(
        || black_box(aws_secret.compute_public_key().expect("AWS-LC public derivation")),
        |public| assert_eq!(public.as_ref(), P384_NIST_PUBLIC),
      ),
    );
  }
  ring_p256_bench! {
    let rng = ring::rand::SystemRandom::new();
    let generate = || {
      ring::agreement::EphemeralPrivateKey::generate(&ring::agreement::ECDH_P384, &rng)
        .expect("ring scalar generation")
    };
    g.bench_function(
      "ring-native",
      checked_routine(
        (),
        |_| {
          // A random key has no fixed public key: its agreement with the NIST public key must match
          // RustCrypto's agreement between the NIST private key and the derived public key.
          let secret = generate();
          let public = secret.compute_public_key().expect("ring public derivation");
          let nist_public = ring::agreement::UnparsedPublicKey::new(&ring::agreement::ECDH_P384, P384_NIST_PUBLIC);
          let shared = ring::agreement::agree_ephemeral(secret, &nist_public, array_from_slice::<48>)
            .expect("ring benchmark agreement");
          let ring_public = p384::PublicKey::from_sec1_bytes(public.as_ref()).expect("valid ring public key");
          let expected = p384::ecdh::diffie_hellman(rustcrypto.to_nonzero_scalar(), ring_public.as_affine());
          assert_eq!(shared.as_slice(), expected.raw_secret_bytes().as_slice());
        },
        |b, _| {
          b.iter_batched(
            &generate,
            |secret| black_box(secret.compute_public_key().expect("ring public derivation")),
            BatchSize::SmallInput,
          )
        },
      ),
    );
  }
  g.finish();
}

fn p384_ecdh_parse(c: &mut Criterion) {
  if !bench_config::selected("p384-ecdh/parse") {
    return;
  }
  let rustcrypto_secret = P384OracleSecretKey::from_slice(&P384_NIST_PRIVATE).expect("valid RustCrypto scalar");
  let mut g = c.benchmark_group("p384-ecdh/parse");
  g.bench_function(
    "rscrypto",
    checked(
      || black_box(P384PublicKey::from_sec1_bytes(black_box(&P384_NIST_PEER)).expect("valid benchmark point")),
      |public| {
        assert_eq!(
          p384_ephemeral(P384_NIST_PRIVATE).diffie_hellman(&public).as_bytes(),
          &P384_NIST_SHARED
        )
      },
    ),
  );
  g.bench_function(
    "rustcrypto-p384-pure-rust",
    checked(
      || black_box(p384::PublicKey::from_sec1_bytes(black_box(&P384_NIST_PEER)).expect("valid oracle point")),
      |public| {
        let shared = p384::ecdh::diffie_hellman(rustcrypto_secret.to_nonzero_scalar(), public.as_affine());
        assert_eq!(shared.raw_secret_bytes().as_slice(), P384_NIST_SHARED);
      },
    ),
  );
  g.finish();
}

fn p384_ecdh_agreement(c: &mut Criterion) {
  if !bench_config::selected("p384-ecdh/agreement") {
    return;
  }
  let scalar = P384_NIST_PRIVATE;
  let peer = P384PublicKey::from_sec1_bytes(&P384_NIST_PEER).expect("valid benchmark peer point");
  let rustcrypto_secret = P384OracleSecretKey::from_slice(&scalar).expect("valid RustCrypto scalar");
  let rustcrypto_peer = p384::PublicKey::from_sec1_bytes(&P384_NIST_PEER).expect("valid RustCrypto peer point");
  aws_lc_bench! {
    let aws_secret = aws_lc_rs::agreement::PrivateKey::from_private_key(
      &aws_lc_rs::agreement::ECDH_P384,
      &scalar,
    )
    .expect("valid AWS-LC scalar");
    let aws_peer = aws_lc_rs::agreement::ParsedPublicKey::try_from(
      aws_lc_rs::agreement::UnparsedPublicKey::new(&aws_lc_rs::agreement::ECDH_P384, P384_NIST_PEER),
    )
    .expect("valid AWS-LC peer point");
  }
  let mut g = c.benchmark_group("p384-ecdh/agreement");
  g.bench_function(
    "rscrypto-selected",
    checked_routine(
      (),
      |_| {
        assert_eq!(
          p384_ephemeral(scalar).diffie_hellman(&peer).as_bytes(),
          &P384_NIST_SHARED
        )
      },
      |b, _| {
        b.iter_batched(
          || p384_ephemeral(scalar),
          |secret| black_box(secret.diffie_hellman(black_box(&peer))),
          BatchSize::SmallInput,
        )
      },
    ),
  );
  g.bench_function(
    "rustcrypto-p384-pure-rust",
    checked(
      || {
        black_box(p384::ecdh::diffie_hellman(
          rustcrypto_secret.to_nonzero_scalar(),
          black_box(rustcrypto_peer.as_affine()),
        ))
      },
      |shared| assert_eq!(shared.raw_secret_bytes().as_slice(), P384_NIST_SHARED),
    ),
  );
  aws_lc_bench! {
    g.bench_function(
      "aws-lc-rs-native",
      checked(
        || {
          black_box(
            aws_lc_rs::agreement::agree(&aws_secret, aws_peer.clone(), (), |bytes| {
              Ok::<[u8; 48], ()>(array_from_slice(bytes))
            })
            .expect("AWS-LC benchmark agreement"),
          )
        },
        |shared| assert_eq!(shared, P384_NIST_SHARED),
      ),
    );
  }
  ring_p256_bench! {
    let rng = ring::rand::SystemRandom::new();
    let ring_peer = ring::agreement::UnparsedPublicKey::new(&ring::agreement::ECDH_P384, P384_NIST_PEER);
    let generate = || {
      ring::agreement::EphemeralPrivateKey::generate(&ring::agreement::ECDH_P384, &rng)
        .expect("ring scalar generation")
    };
    g.bench_function(
      "ring-native",
      checked_routine(
        (),
        |_| {
          // The NIST peer has no published private key, so the check pairs a random key with the NIST public
          // key and compares RustCrypto's agreement between the NIST private key and the random public key.
          let secret = generate();
          let public = secret.compute_public_key().expect("ring public derivation");
          let nist_public = ring::agreement::UnparsedPublicKey::new(&ring::agreement::ECDH_P384, P384_NIST_PUBLIC);
          let shared = ring::agreement::agree_ephemeral(secret, &nist_public, array_from_slice::<48>)
            .expect("ring benchmark agreement");
          let ring_public = p384::PublicKey::from_sec1_bytes(public.as_ref()).expect("valid ring public key");
          let expected = p384::ecdh::diffie_hellman(rustcrypto_secret.to_nonzero_scalar(), ring_public.as_affine());
          assert_eq!(shared.as_slice(), expected.raw_secret_bytes().as_slice());
        },
        |b, _| {
          b.iter_batched(
            &generate,
            |secret| {
              black_box(
                ring::agreement::agree_ephemeral(secret, black_box(&ring_peer), array_from_slice::<48>)
                  .expect("ring benchmark agreement"),
              )
            },
            BatchSize::SmallInput,
          )
        },
      ),
    );
  }
  g.finish();
}

// Every deterministic row must reproduce libcrux, an independent FIPS 203 implementation, from the same seed, key,
// ciphertext, and randomness. AWS-LC rows draw internal entropy, so rscrypto and AWS-LC must instead open each
// other's outputs. Fixtures are public deterministic benchmark data; output conversion and drop remain timed. See
// docs/benchmarking.md for the comparison boundaries.
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
      if !bench_config::selected(concat!($group, "/keygen/")) { return; }
      let key_random = deterministic_bytes::<64>(0x10);
      let expected = || {
        let keys = $libcrux::generate_key_pair(key_random);
        (*keys.public_key().as_slice(), *keys.private_key().as_slice())
      };
      let mut g = c.benchmark_group(concat!($group, "/keygen/derand-encoded"));
      g.bench_function("rscrypto", checked(|| {
        let (ek, dk) = <$profile>::generate_keypair(|out| {
          out.copy_from_slice(black_box(&key_random));
          Ok::<(), MlKemError>(())
        }).expect("ML-KEM key generation");
        (ek.to_bytes(), *dk.as_bytes())
      }, |keys| assert_eq!(keys, expected())));
      g.bench_function("libcrux", checked(|| {
        let keys = $libcrux::generate_key_pair(black_box(key_random));
        (*keys.public_key().as_slice(), *keys.private_key().as_slice())
      }, |keys| assert_eq!(keys, expected())));
      g.bench_function("fips203", checked(|| {
        let seed = black_box(&key_random);
        let (ek, dk) = $fips::KG::keygen_from_seed(array_from_slice(&seed[..32]), array_from_slice(&seed[32..]));
        (ek.into_bytes(), dk.into_bytes())
      }, |keys| assert_eq!(keys, expected())));
      g.finish();

      aws_lc_bench! {
        let mut g = c.benchmark_group(concat!($group, "/keygen/internal-entropy-encoded"));
        g.bench_function("aws-lc-rs", checked(|| {
          let dk = AwsMlKemDecapsulationKey::generate(&$aws_algorithm).expect("AWS-LC key generation");
          let ek = dk.encapsulation_key().expect("AWS-LC public key");
          (array_from_slice::<{ <$profile>::ENCAPSULATION_KEY_SIZE }>(ek.key_bytes().expect("AWS-LC key export").as_ref()),
           array_from_slice::<{ <$profile>::DECAPSULATION_KEY_SIZE }>(dk.key_bytes().expect("AWS-LC key export").as_ref()))
        }, |(ek_bytes, dk_bytes)| {
          let ek = <<$profile as rscrypto::Kem>::EncapsulationKey>::try_from_slice(&ek_bytes).expect("AWS-LC public key import");
          let dk = <<$profile as rscrypto::Kem>::DecapsulationKey>::try_from_slice(&dk_bytes).expect("AWS-LC secret key import");
          let (ct, ss) = <$profile>::encapsulate(&ek, |out| { out.fill(0x42); Ok::<(), MlKemError>(()) }).expect("AWS-LC key interoperability");
          assert_eq!(<$profile>::decapsulate(&dk, &ct).expect("AWS-LC key interoperability").as_bytes(), ss.as_bytes());
          let aws_dk = AwsMlKemDecapsulationKey::new(&$aws_algorithm, &dk_bytes).expect("AWS-LC key import");
          assert_eq!(aws_dk.decapsulate(AwsMlKemCiphertext::from(ct.as_ref())).expect("AWS-LC key interoperability").as_ref(), ss.as_ref());
        }));
        g.finish();
      }
    }

    fn $encapsulate_fn(c: &mut Criterion) {
      if !bench_config::selected(concat!($group, "/encapsulate/")) { return; }
      let key_random = deterministic_bytes::<64>(0x20);
      let encaps_random = deterministic_bytes::<32>(0x80);
      let (ek, _dk) = <$profile>::generate_keypair(|out| {
        out.copy_from_slice(&key_random);
        Ok::<(), MlKemError>(())
      }).expect("ML-KEM fixture key generation");
      let ek_bytes = ek.to_bytes();
      let prepared_ek = ek.prepare().expect("ML-KEM key preparation");
      let rustcrypto_ek = rustcrypto_ml_kem::EncapsulationKey::<$rustcrypto>::new(&ek_bytes.into()).expect("RustCrypto key import");
      let fips_ek = $fips::EncapsKey::try_from_bytes(ek_bytes).expect("fips203 key import");
      let libcrux_ek = libcrux_ml_kem::MlKemPublicKey::from(ek_bytes);
      let expected = || {
        let (ct, ss) = $libcrux::encapsulate(&libcrux_ek, encaps_random);
        (*ct.as_slice(), ss)
      };
      let mut g = c.benchmark_group(concat!($group, "/encapsulate/derand-reuse-matrix-prepared"));
      g.bench_function("rscrypto", checked(|| {
        let (ct, ss) = black_box(&prepared_ek).encapsulate(|out| {
          out.copy_from_slice(black_box(&encaps_random)); Ok::<(), MlKemError>(())
        }).expect("ML-KEM encapsulation");
        (ct.to_bytes(), *ss.as_bytes())
      }, |output| assert_eq!(output, expected())));
      g.finish();
      let mut g = c.benchmark_group(concat!($group, "/encapsulate/derand-reuse-decoded"));
      g.bench_function("rustcrypto", checked(|| {
        let (ct, ss) = black_box(&rustcrypto_ek).encapsulate_deterministic(&(*black_box(&encaps_random)).into());
        (array_from_slice::<{ <$profile>::CIPHERTEXT_SIZE }>(ct.as_slice()), array_from_slice::<32>(ss.as_slice()))
      }, |output| assert_eq!(output, expected())));
      g.finish();
      let mut g = c.benchmark_group(concat!($group, "/encapsulate/derand-reuse-encoded"));
      g.bench_function("rscrypto", checked(|| {
        let (ct, ss) = <$profile>::encapsulate(black_box(&ek), |out| {
          out.copy_from_slice(black_box(&encaps_random)); Ok::<(), MlKemError>(())
        }).expect("ML-KEM encapsulation");
        (ct.to_bytes(), *ss.as_bytes())
      }, |output| assert_eq!(output, expected())));
      g.bench_function("libcrux", checked(|| {
        let (ct, ss) = $libcrux::encapsulate(black_box(&libcrux_ek), *black_box(&encaps_random));
        (*ct.as_slice(), ss)
      }, |output| assert_eq!(output, expected())));
      g.bench_function("fips203", checked(|| {
        let (ss, ct) = black_box(&fips_ek).encaps_from_seed(black_box(&encaps_random));
        (ct.into_bytes(), ss.into_bytes())
      }, |output| assert_eq!(output, expected())));
      g.finish();

      let mut g = c.benchmark_group(concat!($group, "/encapsulate/derand-import-encoded"));
      g.bench_function("rscrypto", checked(|| {
        let ek = <<$profile as rscrypto::Kem>::EncapsulationKey>::try_from_slice(black_box(&ek_bytes)).expect("ML-KEM key import");
        let (ct, ss) = <$profile>::encapsulate(&ek, |out| {
          out.copy_from_slice(black_box(&encaps_random)); Ok::<(), MlKemError>(())
        }).expect("ML-KEM encapsulation");
        (ct.to_bytes(), *ss.as_bytes())
      }, |output| assert_eq!(output, expected())));
      g.bench_function("rustcrypto", checked(|| {
        let ek = rustcrypto_ml_kem::EncapsulationKey::<$rustcrypto>::new(&(*black_box(&ek_bytes)).into()).expect("RustCrypto key import");
        let (ct, ss) = ek.encapsulate_deterministic(&(*black_box(&encaps_random)).into());
        (array_from_slice::<{ <$profile>::CIPHERTEXT_SIZE }>(ct.as_slice()), array_from_slice::<32>(ss.as_slice()))
      }, |output| assert_eq!(output, expected())));
      g.bench_function("libcrux", checked(|| {
        let ek = libcrux_ml_kem::MlKemPublicKey::from(*black_box(&ek_bytes));
        let (ct, ss) = $libcrux::encapsulate(&ek, *black_box(&encaps_random));
        (*ct.as_slice(), ss)
      }, |output| assert_eq!(output, expected())));
      g.bench_function("fips203", checked(|| {
        let ek = $fips::EncapsKey::try_from_bytes(*black_box(&ek_bytes)).expect("fips203 key import");
        let (ss, ct) = ek.encaps_from_seed(black_box(&encaps_random));
        (ct.into_bytes(), ss.into_bytes())
      }, |output| assert_eq!(output, expected())));
      g.finish();

      aws_lc_bench! {
        let aws_ek = aws_lc_rs::kem::EncapsulationKey::new(&$aws_algorithm, &ek_bytes).expect("AWS-LC key import");
        // Randomized rows cannot match a fixed ciphertext: rscrypto must open each AWS-LC ciphertext to its secret.
        for import in [false, true] {
          let group = if import { concat!($group, "/encapsulate/internal-entropy-import-encoded") }
            else { concat!($group, "/encapsulate/internal-entropy-reuse-native") };
          let mut g = c.benchmark_group(group);
          g.bench_function("aws-lc-rs", checked(|| {
            let imported;
            let key = if import {
              imported = aws_lc_rs::kem::EncapsulationKey::new(&$aws_algorithm, black_box(&ek_bytes)).expect("AWS-LC key import");
              &imported
            } else { black_box(&aws_ek) };
            let (ct, ss) = key.encapsulate().expect("AWS-LC encapsulation");
            (array_from_slice::<{ <$profile>::CIPHERTEXT_SIZE }>(ct.as_ref()), array_from_slice::<32>(ss.as_ref()))
          }, |(ct_bytes, ss)| {
            let ct = <<$profile as rscrypto::Kem>::Ciphertext>::try_from_slice(&ct_bytes).expect("AWS-LC ciphertext import");
            assert_eq!(<$profile>::decapsulate(&_dk, &ct).expect("AWS-LC encapsulation interoperability").as_bytes(), &ss);
          }));
          g.finish();
        }
      }
    }

    fn $decapsulate_fn(c: &mut Criterion) {
      if !bench_config::selected(concat!($group, "/decapsulate/")) { return; }
      let key_random = deterministic_bytes::<64>(0x30);
      let encaps_random = deterministic_bytes::<32>(0x90);
      let (ek, dk) = <$profile>::generate_keypair(|out| {
        out.copy_from_slice(&key_random); Ok::<(), MlKemError>(())
      }).expect("ML-KEM fixture key generation");
      let (ct, _shared) = <$profile>::encapsulate(&ek, |out| {
        out.copy_from_slice(&encaps_random); Ok::<(), MlKemError>(())
      }).expect("ML-KEM fixture encapsulation");
      let dk_bytes = *dk.as_bytes();
      let ct_bytes = ct.to_bytes();
      let prepared_dk = dk.prepare().expect("ML-KEM key preparation");
      let rustcrypto_dk = RustCryptoMlKemDecapsulationKey::<$rustcrypto>::from_seed(RustCryptoMlKemSeed::from(key_random));
      let rustcrypto_ct = ct_bytes.into();
      let fips_dk = $fips::DecapsKey::try_from_bytes(dk_bytes).expect("fips203 key import");
      let fips_ct = $fips::CipherText::try_from_bytes(ct_bytes).expect("fips203 ciphertext import");
      let libcrux_dk = libcrux_ml_kem::MlKemPrivateKey::from(dk_bytes);
      let libcrux_ct = libcrux_ml_kem::MlKemCiphertext::from(ct_bytes);
      let expected = || $libcrux::decapsulate(&libcrux_dk, &libcrux_ct);
      // Preparation decodes the key, expands the public matrix, and derives the
      // decryption mask once; the reuse row below amortizes all of it.
      let mut g = c.benchmark_group(concat!($group, "/decapsulate/prepare-key"));
      g.bench_function("rscrypto", checked(
        || black_box(black_box(&dk).prepare().expect("ML-KEM key preparation")),
        |prepared| assert_eq!(*prepared.decapsulate(&ct).expect("ML-KEM decapsulation").as_bytes(), expected()),
      ));
      g.finish();
      let mut g = c.benchmark_group(concat!($group, "/decapsulate/reuse-matrix-prepared"));
      g.bench_function("rscrypto", checked(|| {
        *black_box(&prepared_dk).decapsulate(black_box(&ct)).expect("ML-KEM decapsulation").as_bytes()
      }, |secret| assert_eq!(secret, expected())));
      g.finish();
      let mut g = c.benchmark_group(concat!($group, "/decapsulate/reuse-decoded"));
      g.bench_function("rustcrypto", checked(|| {
        array_from_slice::<32>(black_box(&rustcrypto_dk).decapsulate(black_box(&rustcrypto_ct)).as_slice())
      }, |secret| assert_eq!(secret, expected())));
      g.finish();
      let mut g = c.benchmark_group(concat!($group, "/decapsulate/reuse-encoded"));
      g.bench_function("rscrypto", checked(|| {
        *<$profile>::decapsulate(black_box(&dk), black_box(&ct)).expect("ML-KEM decapsulation").as_bytes()
      }, |secret| assert_eq!(secret, expected())));
      g.bench_function("libcrux", checked(|| {
        $libcrux::decapsulate(black_box(&libcrux_dk), black_box(&libcrux_ct))
      }, |secret| assert_eq!(secret, expected())));
      g.bench_function("fips203", checked(|| {
        black_box(&fips_dk).try_decaps(black_box(&fips_ct)).expect("fips203 decapsulation").into_bytes()
      }, |secret| assert_eq!(secret, expected())));
      g.finish();
      aws_lc_bench! {
        let aws_dk = AwsMlKemDecapsulationKey::new(&$aws_algorithm, &dk_bytes).expect("AWS-LC key import");
        let mut g = c.benchmark_group(concat!($group, "/decapsulate/reuse-native"));
        g.bench_function("aws-lc-rs", checked(|| {
          array_from_slice::<32>(black_box(&aws_dk).decapsulate(AwsMlKemCiphertext::from(black_box(ct_bytes.as_slice()))).expect("AWS-LC decapsulation").as_ref())
        }, |secret| assert_eq!(secret, expected())));
        g.finish();
      }
      let mut g = c.benchmark_group(concat!($group, "/decapsulate/import-encoded"));
      g.bench_function("rscrypto", checked(|| {
        let dk = <<$profile as rscrypto::Kem>::DecapsulationKey>::try_from_slice(black_box(&dk_bytes)).expect("ML-KEM key import");
        let ct = <<$profile as rscrypto::Kem>::Ciphertext>::try_from_slice(black_box(&ct_bytes)).expect("ML-KEM ciphertext import");
        *<$profile>::decapsulate(&dk, &ct).expect("ML-KEM decapsulation").as_bytes()
      }, |secret| assert_eq!(secret, expected())));
      g.bench_function("libcrux", checked(|| {
        let dk = libcrux_ml_kem::MlKemPrivateKey::from(*black_box(&dk_bytes));
        let ct = libcrux_ml_kem::MlKemCiphertext::from(*black_box(&ct_bytes));
        $libcrux::decapsulate(&dk, &ct)
      }, |secret| assert_eq!(secret, expected())));
      g.bench_function("fips203", checked(|| {
        let dk = $fips::DecapsKey::try_from_bytes(*black_box(&dk_bytes)).expect("fips203 key import");
        let ct = $fips::CipherText::try_from_bytes(*black_box(&ct_bytes)).expect("fips203 ciphertext import");
        dk.try_decaps(&ct).expect("fips203 decapsulation").into_bytes()
      }, |secret| assert_eq!(secret, expected())));
      aws_lc_bench! {
        g.bench_function("aws-lc-rs", checked(|| {
          let dk = AwsMlKemDecapsulationKey::new(&$aws_algorithm, black_box(&dk_bytes)).expect("AWS-LC key import");
          array_from_slice::<32>(dk.decapsulate(AwsMlKemCiphertext::from(black_box(ct_bytes.as_slice()))).expect("AWS-LC decapsulation").as_ref())
        }, |secret| assert_eq!(secret, expected())));
      }
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

fn main() {
  // Exercise the AVX2 Curve25519 fallback on IFMA hosts through the production
  // dispatcher. Apply the restriction before any fixture initializes detection.
  if let Some(value) = std::env::var_os("RSCRYPTO_BENCH_DISABLE_IFMA") {
    assert_eq!(value, "1", "RSCRYPTO_BENCH_DISABLE_IFMA must be 1");
    #[cfg(not(target_arch = "x86_64"))]
    {
      eprintln!("RSCRYPTO_BENCH_DISABLE_IFMA requires x86_64");
      std::process::exit(2);
    }
    #[cfg(target_arch = "x86_64")]
    {
      use rscrypto::platform::{caps::x86, expert};

      let mut detected = expert::detect_uncached();
      assert!(detected.caps.has(x86::AVX2), "AVX2 benchmark requires an AVX2 host");
      detected.caps = detected.caps.difference(x86::AVX512IFMA);
      expert::try_set_override(Some(detected)).expect("restrict capabilities before initialization");
      eprintln!("rscrypto-bench IFMA disabled; runtime_caps={}", detected.caps);
    }
  }
  bench_config::run(&[
    hmac_sha256,
    hmac_sha384,
    hmac_sha512,
    hmac_sha256_streaming,
    hmac_sha256_internal,
    hkdf_sha256_expand,
    hkdf_sha384_expand,
    hkdf_sha512_expand,
    hkdf_sha256_extract,
    hkdf_sha384_extract,
    hkdf_sha512_extract,
    pbkdf2_sha256_derive,
    pbkdf2_sha256_internal,
    pbkdf2_sha512_derive,
    ecdsa_p256_public_key,
    ecdsa_p256_sign,
    ecdsa_p256_verify,
    #[cfg(all(rscrypto_internal, feature = "diag", feature = "ecdsa-p256"))]
    ecdsa_p256_internal,
    ecdsa_p384_public_key,
    ecdsa_p384_sign,
    ecdsa_p384_verify,
    #[cfg(all(rscrypto_internal, feature = "diag", feature = "ecdsa-p384"))]
    ecdsa_p384_internal,
    ed25519_public_key,
    ed25519_keypair_from_secret,
    ed25519_sign,
    ed25519_verify,
    #[cfg(all(rscrypto_internal, feature = "diag"))]
    ed25519_verify_phase,
    x25519_public_key,
    x25519_diffie_hellman,
    p256_ecdh_key_generation,
    p256_ecdh_public_key,
    p256_ecdh_parse,
    p256_ecdh_agreement,
    p256_ecdh_tls_roundtrip,
    p384_ecdh_key_generation,
    p384_ecdh_public_key,
    p384_ecdh_parse,
    p384_ecdh_agreement,
    mlkem512_keygen,
    mlkem512_encapsulate,
    mlkem512_decapsulate,
    mlkem768_keygen,
    mlkem768_encapsulate,
    mlkem768_decapsulate,
    mlkem1024_keygen,
    mlkem1024_encapsulate,
    mlkem1024_decapsulate,
  ]);
}
