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
use ed25519_dalek::{Signature as DalekSignature, Signer as _, SigningKey, VerifyingKey as DalekVerifyingKey};
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
  MlKem1024, MlKemError, P256EphemeralSecret, P256PublicKey, P384EphemeralSecret, P384PublicKey, X25519PublicKey,
  X25519SecretKey,
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

/// `point` with the lowest bit of its last byte flipped. The caller must confirm that the result is off the curve.
fn flip_last_bit<const N: usize>(mut point: [u8; N]) -> [u8; N] {
  *point.last_mut().expect("a point encoding is not empty") ^= 1;
  point
}

/// Returns a row whose timed path validates its encoded input. `operation` is that path: it must accept `input`, and
/// `check` tests its output. Outside timing, the same path must reject `malformed` by returning `None`.
fn validating_row<'a, I, T: 'a>(
  operation: impl Fn(&I) -> Option<T> + 'a,
  input: &'a I,
  malformed: &'a I,
  check: impl FnOnce(T) + 'a,
) -> impl FnMut(&mut criterion::Bencher<'_>) + 'a {
  checked_with(
    operation,
    move |operation| {
      black_box(operation(black_box(input)).expect("valid authentication benchmark operation must succeed"))
    },
    move |output, operation| {
      check(output);
      assert!(operation(malformed).is_none(), "a malformed input must fail");
    },
  )
}

/// Returns an import-and-verify row. `verify` is the row's timed path and must accept the valid key, message, and
/// signature. Outside timing, the same path must reject the changed signature and the malformed key.
fn import_and_verify<'a, K, S>(
  verify: impl Fn(&K, &[u8], &S) -> bool + 'a,
  key: &'a K,
  malformed_key: &'a K,
  message: &'a [u8],
  signature: &'a S,
  tampered: &'a S,
) -> impl FnMut(&mut criterion::Bencher<'_>) + 'a {
  checked_with(
    verify,
    move |verify| {
      assert!(
        verify(black_box(key), black_box(message), black_box(signature)),
        "valid authentication benchmark operation must succeed"
      );
    },
    move |(), verify| {
      assert!(!verify(key, message, tampered), "a changed signature must fail");
      assert!(
        !verify(malformed_key, message, signature),
        "a malformed public key must fail"
      );
    },
  )
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

// HMAC with a key that each library prepares outside timing. Timed: message absorption, finalization, and state
// reuse or destruction.
// - rscrypto resets its keyed state after each tag. RustCrypto clones its keyed state. aws-lc-rs and ring copy the
//   key's precomputed inner and outer states in `sign`.
// Every row must return RustCrypto's tag; the RustCrypto row must return ring's.
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

      let inputs = common::comp_sizes(&[]);
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
        let ring_expected = ring::hmac::sign(&ring_key, data);

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
            |tag| assert_eq!(tag.as_slice(), ring_expected.as_ref()),
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

// HMAC-SHA256 over 1 MiB in fixed-size updates. Timed: keying, every update, and finalization. aws-lc-rs and ring
// have streaming APIs but no rows here. Every row must return RustCrypto's tag; the RustCrypto row must return
// ring's.
fn hmac_sha256_streaming(c: &mut Criterion) {
  if !bench_config::selected("hmac-sha256/streaming") {
    return;
  }
  print_auth_diag_once();

  let data = common::random_bytes(1048576);
  let key = [0x24u8; 32];
  let expected = rustcrypto_hmac::<RustCryptoHmacSha256>(&key, &data);
  let ring_expected = ring::hmac::sign(&ring::hmac::Key::new(ring::hmac::HMAC_SHA256, &key), &data);
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
        |tag| assert_eq!(tag.as_slice(), ring_expected.as_ref()),
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
    let ring_expected = ring::hmac::sign(&ring_key, msg);

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
        |tag| assert_eq!(tag.as_slice(), ring_expected.as_ref()),
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

/// Returns a tag-verification row. `verify` is the row's timed path: it must accept `tag`, and outside timing it must
/// reject `changed`.
fn tag_verification_row<'a, T: ?Sized>(
  verify: impl FnMut(&T) -> bool + 'a,
  tag: &'a T,
  changed: &'a T,
) -> impl FnMut(&mut criterion::Bencher<'_>) + 'a {
  checked_routine(
    verify,
    move |verify| {
      // The second acceptance follows one call that reused state has already served.
      for _ in 0..2 {
        assert!(verify(tag), "valid authentication benchmark verification must succeed");
      }
      assert!(!verify(changed), "a changed tag must fail");
    },
    move |b, verify| b.iter(|| verify(black_box(tag))),
  )
}

/// Message lengths for HMAC verification: a short token and a typical request body. Verification adds a fixed-cost
/// comparison to tagging, which the tag groups measure across the full size matrix.
const HMAC_VERIFY_SIZES: [usize; 2] = [32, 1024];

// HMAC tag verification over one message. Every row must accept RustCrypto's tag and reject it with its last bit
// flipped. Each library compares the full tag in constant time.
// - `verify/reuse-key`: each library prepares its key outside timing. Timed: message absorption, finalization, the
//   comparison, and state reuse or destruction. rscrypto absorbs into its keyed state, calls `verify`, and resets
//   the state. RustCrypto clones its keyed state and calls `verify_slice`. aws-lc-rs and ring copy the key's
//   precomputed inner and outer states in `hmac::verify`.
// - `verify/import-key`: every library receives the key bytes in each call. Timed: keying, absorption,
//   finalization, the comparison, and destruction. rscrypto calls `verify_tag`. RustCrypto calls `new_from_slice`.
//   aws-lc-rs and ring build an `hmac::Key`.
macro_rules! hmac_verify_bench {
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
      if !bench_config::selected(concat!($group, "/verify/")) {
        return;
      }
      print_auth_diag_once();

      let key = [0x42u8; $key_len];
      let rustcrypto =
        <$rustcrypto>::new_from_slice(&key).expect("valid authentication benchmark operation must succeed");
      aws_lc_bench! {
        let aws_key = aws_lc_rs::hmac::Key::new($aws_algorithm, &key);
      }
      let ring_key = ring::hmac::Key::new($ring_algorithm, &key);
      let inputs = HMAC_VERIFY_SIZES
        .into_iter()
        .map(|len| (len, common::random_bytes(len)))
        .collect::<Vec<_>>();
      let fixtures = inputs
        .iter()
        .map(|(len, data)| {
          let tag = rustcrypto_hmac::<$rustcrypto>(&key, data);
          let mut changed = tag.clone();
          *changed.last_mut().expect("an HMAC tag is not empty") ^= 1;
          let tag_rs = <<$rscrypto as rscrypto::Mac>::Tag>::try_from(tag.as_slice())
            .expect("RustCrypto's tag has the HMAC output length");
          let changed_rs = <<$rscrypto as rscrypto::Mac>::Tag>::try_from(changed.as_slice())
            .expect("RustCrypto's tag has the HMAC output length");
          (*len, data, tag, changed, tag_rs, changed_rs)
        })
        .collect::<Vec<_>>();

      let mut g = c.benchmark_group(concat!($group, "/verify/reuse-key"));
      for (len, data, tag, changed, tag_rs, changed_rs) in &fixtures {
        common::set_throughput(&mut g, *len);
        let mut mac = <$rscrypto>::new(&key);
        g.bench_function(
          BenchmarkId::new("rscrypto", len),
          tag_verification_row(
            move |tag| {
              mac.update(black_box(data));
              let verified = mac.verify(tag).is_ok();
              mac.reset();
              verified
            },
            tag_rs,
            changed_rs,
          ),
        );
        g.bench_function(
          BenchmarkId::new("rustcrypto", len),
          tag_verification_row(
            |tag: &[u8]| {
              use hmac::Mac as _;

              let mut mac = rustcrypto.clone();
              mac.update(black_box(data));
              mac.verify_slice(tag).is_ok()
            },
            tag.as_slice(),
            changed.as_slice(),
          ),
        );
        aws_lc_bench! {
          g.bench_function(
            BenchmarkId::new("aws-lc-rs", len),
            tag_verification_row(
              |tag: &[u8]| aws_lc_rs::hmac::verify(&aws_key, black_box(data), tag).is_ok(),
              tag.as_slice(),
              changed.as_slice(),
            ),
          );
        }
        g.bench_function(
          BenchmarkId::new("ring", len),
          tag_verification_row(
            |tag: &[u8]| ring::hmac::verify(&ring_key, black_box(data), tag).is_ok(),
            tag.as_slice(),
            changed.as_slice(),
          ),
        );
      }
      g.finish();

      let mut g = c.benchmark_group(concat!($group, "/verify/import-key"));
      for (len, data, tag, changed, tag_rs, changed_rs) in &fixtures {
        common::set_throughput(&mut g, *len);
        g.bench_function(
          BenchmarkId::new("rscrypto", len),
          tag_verification_row(
            |tag| <$rscrypto>::verify_tag(black_box(&key), black_box(data), tag).is_ok(),
            tag_rs,
            changed_rs,
          ),
        );
        g.bench_function(
          BenchmarkId::new("rustcrypto", len),
          tag_verification_row(
            |tag: &[u8]| {
              use hmac::Mac as _;

              let mut mac = <$rustcrypto>::new_from_slice(black_box(&key))
                .expect("valid authentication benchmark operation must succeed");
              mac.update(black_box(data));
              mac.verify_slice(tag).is_ok()
            },
            tag.as_slice(),
            changed.as_slice(),
          ),
        );
        aws_lc_bench! {
          g.bench_function(
            BenchmarkId::new("aws-lc-rs", len),
            tag_verification_row(
              |tag: &[u8]| {
                aws_lc_rs::hmac::verify(&aws_lc_rs::hmac::Key::new($aws_algorithm, black_box(&key)), black_box(data), tag)
                  .is_ok()
              },
              tag.as_slice(),
              changed.as_slice(),
            ),
          );
        }
        g.bench_function(
          BenchmarkId::new("ring", len),
          tag_verification_row(
            |tag: &[u8]| {
              ring::hmac::verify(&ring::hmac::Key::new($ring_algorithm, black_box(&key)), black_box(data), tag).is_ok()
            },
            tag.as_slice(),
            changed.as_slice(),
          ),
        );
      }
      g.finish();
    }
  };
}

hmac_verify_bench!(
  hmac_sha256_verify,
  "hmac-sha256",
  HmacSha256,
  RustCryptoHmacSha256,
  32,
  aws_lc_rs::hmac::HMAC_SHA256,
  ring::hmac::HMAC_SHA256
);
hmac_verify_bench!(
  hmac_sha384_verify,
  "hmac-sha384",
  HmacSha384,
  RustCryptoHmacSha384,
  48,
  aws_lc_rs::hmac::HMAC_SHA384,
  ring::hmac::HMAC_SHA384
);
hmac_verify_bench!(
  hmac_sha512_verify,
  "hmac-sha512",
  HmacSha512,
  RustCryptoHmacSha512,
  64,
  aws_lc_rs::hmac::HMAC_SHA512,
  ring::hmac::HMAC_SHA512
);

// HKDF-Expand from a PRK that each library holds outside timing. Each row writes into a reused caller buffer. Salt
// and IKM are HashLen bytes, and `info` keeps T(1) in one compression block while each later T(i) needs two.
// Every row must return RustCrypto's output; the RustCrypto row must return ring's.
// - rscrypto, RustCrypto, and ring hold the PRK as an HMAC key with its inner and outer states precomputed.
// - aws-lc-rs holds the PRK bytes from `Prk::new_less_safe`, built from RustCrypto's PRK. A PRK from its
//   `Salt::extract` only stores the salt and IKM, and `fill` would then run HKDF-Extract too. `fill` still keys
//   HMAC from the PRK in each call: two more compression-function calls than the other rows.
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
        let prk = <$rustcrypto>::extract(Some(&salt), &ikm).0;
        let aws_prk = aws_lc_rs::hkdf::Prk::new_less_safe($aws_algorithm, prk.as_slice());
      }
      let ring_prk = ring::hkdf::Salt::new($ring_algorithm, &salt).extract(&ikm);
      let mut g = c.benchmark_group($group);

      for out_len in [$($out_len),+] {
        let mut expected = vec![0u8; out_len];
        rustcrypto
          .expand(&info, &mut expected)
          .expect("valid authentication benchmark operation must succeed");
        let mut ring_expected = vec![0u8; out_len];
        ring_prk
          .expand(&[&info], RingHkdfLen(out_len))
          .expect("valid authentication benchmark operation must succeed")
          .fill(&mut ring_expected)
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
            |_, out| assert_eq!(out, &ring_expected),
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
// rscrypto must return RustCrypto's PRK. RustCrypto and ring must expand their
// PRK state to each other's T(1), the first expanded block.
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
        let (expected_prk, expected_hkdf) = <$rustcrypto>::extract(Some(&salt), &ikm);
        let mut expected_t1 = [0u8; $hash_len];
        expected_hkdf
          .expand(&[], &mut expected_t1)
          .expect("valid authentication benchmark operation must succeed");
        let mut ring_t1 = [0u8; $hash_len];
        ring::hkdf::Salt::new($ring_algorithm, &salt)
          .extract(&ikm)
          .expand(&[], RingHkdfLen($hash_len))
          .expect("valid authentication benchmark operation must succeed")
          .fill(&mut ring_t1)
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
            |(_, hkdf)| {
              let mut t1 = [0u8; $hash_len];
              hkdf
                .expand(&[], &mut t1)
                .expect("valid authentication benchmark operation must succeed");
              assert_eq!(t1, ring_t1);
            },
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

/// RFC 8032 Section 7.1 TEST 1 secret key and public key.
const ED25519_RFC8032_SECRET: &str = "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60";
const ED25519_RFC8032_PUBLIC: &str = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a";

// Ed25519 public key from 32 secret-key bytes. Timed: secret import, SHA-512 key expansion, and the fixed-base
// scalar multiplication. Dalek derives the public key while it constructs `SigningKey`. Every row must return
// the RFC 8032 public key.
fn ed25519_public_key(c: &mut Criterion) {
  if !bench_config::selected("ed25519/public-key-from-secret") {
    return;
  }
  let secret_bytes = hex::<32>(ED25519_RFC8032_SECRET);
  let expected = hex::<32>(ED25519_RFC8032_PUBLIC);
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

// ECDSA P-256 verification. Every row verifies RustCrypto's fixed-width signature over the same message, and its
// check requires a rejection once the last byte of r changes.
//
// `ecdsa-p256/verify/reuse-key`: each library's key object is built outside timing. Timed: verification.
// - rscrypto reuses a parsed key with its precomputed verification table, and a parsed signature.
// - RustCrypto reuses a `VerifyingKey` and a parsed signature.
// - aws-lc-rs reuses a `ParsedPublicKey`. It takes the signature bytes and re-encodes them as DER in each call.
// - ring is absent: it has no API that reuses a parsed key.
//
// `ecdsa-p256/verify/import-key`: every row gets the same 65-byte uncompressed SEC1 key and 64-byte signature.
// Timed: key parsing and validation, signature parsing, verification, and drop of per-call objects.
// - rscrypto also builds the verification table that key reuse amortizes.
// - aws-lc-rs allocates an EVP_PKEY and re-encodes the signature as DER.
// Each row's check also requires the same path to reject a key that is not on the curve.
fn ecdsa_p256_verify(c: &mut Criterion) {
  if !bench_config::selected("ecdsa-p256/verify/") {
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
  let off_curve = flip_last_bit(sec1);
  p256::PublicKey::from_sec1_bytes(&off_curve).expect_err("RustCrypto must reject the changed point");
  let public =
    EcdsaP256PublicKey::from_sec1_bytes(&sec1).expect("valid authentication benchmark operation must succeed");
  aws_lc_bench! {
    let aws_key = aws_lc_rs::signature::ParsedPublicKey::new(&aws_lc_rs::signature::ECDSA_P256_SHA256_FIXED, sec1)
      .expect("valid authentication benchmark operation must succeed");
  }
  let inputs = [0usize, 32, 1024, 16384]
    .into_iter()
    .map(|len| {
      let data = common::random_bytes(len);
      let signature: P256OracleSignature = p256::ecdsa::signature::Signer::sign(&signing_key, &data);
      let signature: [u8; 64] = array_from_slice(signature.to_bytes().as_ref());
      let mut tampered = signature;
      tampered[31] ^= 1;
      (len, data, signature, tampered)
    })
    .collect::<Vec<_>>();

  let mut g = c.benchmark_group("ecdsa-p256/verify/reuse-key");
  for (len, data, signature_bytes, tampered_bytes) in &inputs {
    common::set_throughput(&mut g, *len);
    let signature =
      EcdsaP256Signature::from_bytes(*signature_bytes).expect("valid authentication benchmark operation must succeed");
    let tampered = EcdsaP256Signature::from_bytes(*tampered_bytes).expect("a changed r remains a valid scalar");
    let oracle_signature =
      P256OracleSignature::from_slice(signature_bytes).expect("valid authentication benchmark operation must succeed");
    let oracle_tampered = P256OracleSignature::from_slice(tampered_bytes).expect("a changed r remains a valid scalar");

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

    aws_lc_bench! {
      g.bench_function(
        BenchmarkId::new("aws-lc-rs", len),
        checked(
          || {
            black_box(&aws_key)
              .verify_sig(black_box(data), black_box(signature_bytes))
              .expect("valid authentication benchmark operation must succeed");
            black_box(())
          },
          |()| {
            aws_key.verify_sig(data, tampered_bytes).expect_err("a changed ECDSA P-256 signature must fail");
          },
        ),
      );
    }
  }
  g.finish();

  // Each closure is one row's timed import-and-verify path.
  let rscrypto_import = |key: &[u8; 65], message: &[u8], signature: &[u8; 64]| {
    let (Ok(key), Ok(signature)) = (
      EcdsaP256PublicKey::from_sec1_bytes(key),
      EcdsaP256Signature::from_bytes(*signature),
    ) else {
      return false;
    };
    key.verify(message, &signature).is_ok()
  };
  let rustcrypto_import = |key: &[u8; 65], message: &[u8], signature: &[u8; 64]| {
    let (Ok(key), Ok(signature)) = (
      p256::ecdsa::VerifyingKey::from_sec1_bytes(key),
      P256OracleSignature::from_slice(signature),
    ) else {
      return false;
    };
    p256::ecdsa::signature::Verifier::verify(&key, message, &signature).is_ok()
  };
  let ring_import = |key: &[u8; 65], message: &[u8], signature: &[u8; 64]| {
    ring::signature::UnparsedPublicKey::new(&ring::signature::ECDSA_P256_SHA256_FIXED, key)
      .verify(message, signature)
      .is_ok()
  };
  let mut g = c.benchmark_group("ecdsa-p256/verify/import-key");
  for (len, data, signature, tampered) in &inputs {
    common::set_throughput(&mut g, *len);
    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      import_and_verify(rscrypto_import, &sec1, &off_curve, data, signature, tampered),
    );
    g.bench_function(
      BenchmarkId::new("rustcrypto-p256", len),
      import_and_verify(rustcrypto_import, &sec1, &off_curve, data, signature, tampered),
    );
    g.bench_function(
      BenchmarkId::new("ring", len),
      import_and_verify(ring_import, &sec1, &off_curve, data, signature, tampered),
    );
    aws_lc_bench! {
      let aws_import = |key: &[u8; 65], message: &[u8], signature: &[u8; 64]| {
        aws_lc_rs::signature::UnparsedPublicKey::new(&aws_lc_rs::signature::ECDSA_P256_SHA256_FIXED, key)
          .verify(message, signature)
          .is_ok()
      };
      g.bench_function(
        BenchmarkId::new("aws-lc-rs", len),
        import_and_verify(aws_import, &sec1, &off_curve, data, signature, tampered),
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
  // ECDSA P-256 signing with a key held outside timing. Timed: hashing, nonce generation, signing, and the
  // signature object. The nonce source differs by library and stays timed:
  // - `rscrypto` is `try_sign`, the default API: a deterministic nonce that is not RFC 6979, with no entropy
  //   source. `rscrypto-blinded` adds caller-supplied nonce blinding and must return the same signature.
  // - RustCrypto derives the RFC 6979 nonce.
  // - ring draws nonce randomness from `SystemRandom` in each call; aws-lc-rs uses AWS-LC's internal generator.
  // No row has a fixed signature. RustCrypto must verify every other row's signature, and ring must verify
  // RustCrypto's.
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
      BenchmarkId::new("rscrypto", len),
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
        |signature| {
          ring::signature::UnparsedPublicKey::new(&ring::signature::ECDSA_P256_SHA256_FIXED, &sec1)
            .verify(data, signature.to_bytes().as_slice())
            .expect("ring must verify RustCrypto's ECDSA P-256 signature");
        },
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

// ECDSA P-256 public key from a secret key held outside timing. Timed: the fixed-base scalar multiplication and the
// library's public-key object. rscrypto's object also precomputes its verification table; RustCrypto's holds
// only the affine point. `rscrypto-blinded` adds caller-supplied scalar blinding. Every row must return the NIST
// CAVP public key for the NIST private key.
fn ecdsa_p256_public_key(c: &mut Criterion) {
  if !bench_config::selected("ecdsa-p256/public-key") {
    return;
  }
  let secret =
    EcdsaP256SecretKey::from_bytes(P256_NIST_PRIVATE).expect("valid authentication benchmark operation must succeed");
  let blind = [0x5cu8; 64];
  let oracle =
    P256OracleSecretKey::from_slice(&P256_NIST_PRIVATE).expect("valid authentication benchmark operation must succeed");

  let mut g = c.benchmark_group("ecdsa-p256/public-key");
  g.bench_function(
    "rscrypto",
    checked(
      || black_box(black_box(&secret).public_key()),
      |public| assert_eq!(public.to_sec1_bytes(), P256_NIST_PUBLIC),
    ),
  );
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
      |public| assert_eq!(public.to_sec1_bytes(), P256_NIST_PUBLIC),
    ),
  );
  g.bench_function(
    "rustcrypto-p256",
    checked(
      || black_box(black_box(&oracle).public_key()),
      |public| assert_eq!(public.to_sec1_point(false).as_bytes(), P256_NIST_PUBLIC),
    ),
  );
  g.finish();
}

// ECDSA P-384 verification. Every row verifies RustCrypto's fixed-width signature over the same message, and its
// check requires a rejection once the last byte of r changes.
//
// `ecdsa-p384/verify/reuse-key`: each library's key object is built outside timing. Timed: verification.
// - rscrypto reuses a parsed key with its precomputed verification table, and a parsed signature.
// - RustCrypto reuses a `VerifyingKey` and a parsed signature.
// - aws-lc-rs reuses a `ParsedPublicKey`. It takes the signature bytes and re-encodes them as DER in each call.
// - ring is absent: it has no API that reuses a parsed key.
//
// `ecdsa-p384/verify/import-key`: every row gets the same 97-byte uncompressed SEC1 key and 96-byte signature.
// Timed: key parsing and validation, signature parsing, verification, and drop of per-call objects.
// - rscrypto also builds the verification table that key reuse amortizes.
// - aws-lc-rs allocates an EVP_PKEY and re-encodes the signature as DER.
// Each row's check also requires the same path to reject a key that is not on the curve.
fn ecdsa_p384_verify(c: &mut Criterion) {
  if !bench_config::selected("ecdsa-p384/verify/") {
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
  let off_curve = flip_last_bit(sec1);
  p384::PublicKey::from_sec1_bytes(&off_curve).expect_err("RustCrypto must reject the changed point");
  let public =
    EcdsaP384PublicKey::from_sec1_bytes(&sec1).expect("valid authentication benchmark operation must succeed");
  aws_lc_bench! {
    let aws_key = aws_lc_rs::signature::ParsedPublicKey::new(&aws_lc_rs::signature::ECDSA_P384_SHA384_FIXED, sec1)
      .expect("valid authentication benchmark operation must succeed");
  }
  let inputs = [0usize, 32, 1024, 16384]
    .into_iter()
    .map(|len| {
      let data = common::random_bytes(len);
      let signature: P384OracleSignature = p384::ecdsa::signature::Signer::sign(&signing_key, &data);
      let signature: [u8; 96] = array_from_slice(signature.to_bytes().as_ref());
      let mut tampered = signature;
      tampered[47] ^= 1;
      (len, data, signature, tampered)
    })
    .collect::<Vec<_>>();

  let mut g = c.benchmark_group("ecdsa-p384/verify/reuse-key");
  for (len, data, signature_bytes, tampered_bytes) in &inputs {
    common::set_throughput(&mut g, *len);
    let signature =
      EcdsaP384Signature::from_bytes(*signature_bytes).expect("valid authentication benchmark operation must succeed");
    let tampered = EcdsaP384Signature::from_bytes(*tampered_bytes).expect("a changed r remains a valid scalar");
    let oracle_signature =
      P384OracleSignature::from_slice(signature_bytes).expect("valid authentication benchmark operation must succeed");
    let oracle_tampered = P384OracleSignature::from_slice(tampered_bytes).expect("a changed r remains a valid scalar");

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

    aws_lc_bench! {
      g.bench_function(
        BenchmarkId::new("aws-lc-rs", len),
        checked(
          || {
            black_box(&aws_key)
              .verify_sig(black_box(data), black_box(signature_bytes))
              .expect("valid authentication benchmark operation must succeed");
            black_box(())
          },
          |()| {
            aws_key.verify_sig(data, tampered_bytes).expect_err("a changed ECDSA P-384 signature must fail");
          },
        ),
      );
    }
  }
  g.finish();

  // Each closure is one row's timed import-and-verify path.
  let rscrypto_import = |key: &[u8; 97], message: &[u8], signature: &[u8; 96]| {
    let (Ok(key), Ok(signature)) = (
      EcdsaP384PublicKey::from_sec1_bytes(key),
      EcdsaP384Signature::from_bytes(*signature),
    ) else {
      return false;
    };
    key.verify(message, &signature).is_ok()
  };
  let rustcrypto_import = |key: &[u8; 97], message: &[u8], signature: &[u8; 96]| {
    let (Ok(key), Ok(signature)) = (
      p384::ecdsa::VerifyingKey::from_sec1_bytes(key),
      P384OracleSignature::from_slice(signature),
    ) else {
      return false;
    };
    p384::ecdsa::signature::Verifier::verify(&key, message, &signature).is_ok()
  };
  let ring_import = |key: &[u8; 97], message: &[u8], signature: &[u8; 96]| {
    ring::signature::UnparsedPublicKey::new(&ring::signature::ECDSA_P384_SHA384_FIXED, key)
      .verify(message, signature)
      .is_ok()
  };
  let mut g = c.benchmark_group("ecdsa-p384/verify/import-key");
  for (len, data, signature, tampered) in &inputs {
    common::set_throughput(&mut g, *len);
    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      import_and_verify(rscrypto_import, &sec1, &off_curve, data, signature, tampered),
    );
    g.bench_function(
      BenchmarkId::new("rustcrypto-p384", len),
      import_and_verify(rustcrypto_import, &sec1, &off_curve, data, signature, tampered),
    );
    g.bench_function(
      BenchmarkId::new("ring", len),
      import_and_verify(ring_import, &sec1, &off_curve, data, signature, tampered),
    );
    aws_lc_bench! {
      let aws_import = |key: &[u8; 97], message: &[u8], signature: &[u8; 96]| {
        aws_lc_rs::signature::UnparsedPublicKey::new(&aws_lc_rs::signature::ECDSA_P384_SHA384_FIXED, key)
          .verify(message, signature)
          .is_ok()
      };
      g.bench_function(
        BenchmarkId::new("aws-lc-rs", len),
        import_and_verify(aws_import, &sec1, &off_curve, data, signature, tampered),
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
  // ECDSA P-384 signing with a key held outside timing. Timed: hashing, nonce generation, signing, and the
  // signature object. The nonce source differs by library and stays timed:
  // - `rscrypto` is `try_sign`, the default API: a deterministic nonce that is not RFC 6979, with no entropy
  //   source. `rscrypto-blinded` adds caller-supplied nonce blinding and must return the same signature.
  // - RustCrypto derives the RFC 6979 nonce.
  // - ring draws nonce randomness from `SystemRandom` in each call; aws-lc-rs uses AWS-LC's internal generator.
  // No row has a fixed signature. RustCrypto must verify every other row's signature, and ring must verify
  // RustCrypto's.
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
      BenchmarkId::new("rscrypto", len),
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
        |signature| {
          ring::signature::UnparsedPublicKey::new(&ring::signature::ECDSA_P384_SHA384_FIXED, &sec1)
            .verify(data, signature.to_bytes().as_slice())
            .expect("ring must verify RustCrypto's ECDSA P-384 signature");
        },
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

// ECDSA P-384 public key from a secret key held outside timing. Timed: the fixed-base scalar multiplication and the
// library's public-key object. rscrypto's object also precomputes its verification table; RustCrypto's holds
// only the affine point. `rscrypto-blinded` adds caller-supplied scalar blinding. Every row must return the NIST
// CAVP public key for the NIST private key.
fn ecdsa_p384_public_key(c: &mut Criterion) {
  if !bench_config::selected("ecdsa-p384/public-key") {
    return;
  }
  let secret =
    EcdsaP384SecretKey::from_bytes(P384_NIST_PRIVATE).expect("valid authentication benchmark operation must succeed");
  let blind = [0xa3u8; 96];
  let oracle =
    P384OracleSecretKey::from_slice(&P384_NIST_PRIVATE).expect("valid authentication benchmark operation must succeed");

  let mut g = c.benchmark_group("ecdsa-p384/public-key");
  g.bench_function(
    "rscrypto",
    checked(
      || black_box(black_box(&secret).public_key()),
      |public| assert_eq!(public.to_sec1_bytes(), P384_NIST_PUBLIC),
    ),
  );
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
      |public| assert_eq!(public.to_sec1_bytes(), P384_NIST_PUBLIC),
    ),
  );
  g.bench_function(
    "rustcrypto-p384",
    checked(
      || black_box(black_box(&oracle).public_key()),
      |public| assert_eq!(public.to_sec1_point(false).as_bytes(), P384_NIST_PUBLIC),
    ),
  );
  g.finish();
}

// ECDSA signature DER codec (RFC 3279 `Ecdsa-Sig-Value`), the form that X.509 and TLS carry. The fixture is
// RustCrypto's RFC 6979 signature of a 32-byte message, which ring must verify in both its fixed-width and DER
// forms before any row runs.
// - `encode`: timed encoding of a parsed signature. rscrypto writes into a reused buffer; RustCrypto returns its
//   fixed-capacity `der::Signature`. Every encoding must equal RustCrypto's and verify under ring's DER verifier.
// - `decode`: timed parsing and range validation of the DER bytes. Every row must return the fixed-width signature
//   and reject the encoding with its last byte removed.
// ring and aws-lc-rs have no standalone signature codec; they parse DER only inside verification.
macro_rules! ecdsa_der_bench {
  (
    $name:ident,
    $group:literal,
    $signature:ty,
    $oracle_signing_key:ty,
    $oracle_signature:ty,
    $secret_len:literal,
    $ring_fixed:path,
    $ring_der:path
  ) => {
    fn $name(c: &mut Criterion) {
      if !bench_config::selected(concat!($group, "/signature-der/")) {
        return;
      }
      let message = common::random_bytes(32);
      let signing_key = <$oracle_signing_key>::from_slice(&[0x11u8; $secret_len])
        .expect("valid authentication benchmark operation must succeed");
      let sec1 = signing_key.verifying_key().to_sec1_point(false);
      let signature: $oracle_signature = p256::ecdsa::signature::Signer::sign(&signing_key, &message);
      let fixed = signature.to_bytes().to_vec();
      let der = signature.to_der().as_bytes().to_vec();
      let truncated = der[..der.len().strict_sub(1)].to_vec();
      let ring_verifies_der = |encoding: &[u8]| {
        ring::signature::UnparsedPublicKey::new(&$ring_der, sec1.as_bytes())
          .verify(&message, encoding)
          .expect("ring must verify the DER signature");
      };
      ring::signature::UnparsedPublicKey::new(&$ring_fixed, sec1.as_bytes())
        .verify(&message, &fixed)
        .expect("ring must verify the fixed-width signature");
      ring_verifies_der(&der);
      let parsed = <$signature>::from_bytes(fixed.as_slice().try_into().expect("fixed-width signature length"))
        .expect("valid authentication benchmark operation must succeed");

      let mut g = c.benchmark_group(concat!($group, "/signature-der/encode"));
      g.bench_function(
        "rscrypto",
        checked_with(
          [0u8; <$signature>::DER_MAX_LENGTH],
          |out| black_box(&parsed).to_der_into(out).len(),
          |len, out| {
            assert_eq!(&out[..len], der.as_slice());
            ring_verifies_der(&out[..len]);
          },
        ),
      );
      g.bench_function(
        "rustcrypto",
        checked(
          || black_box(&signature).to_der(),
          |encoding| ring_verifies_der(encoding.as_bytes()),
        ),
      );
      g.finish();

      let mut g = c.benchmark_group(concat!($group, "/signature-der/decode"));
      g.bench_function(
        "rscrypto",
        validating_row(
          |encoding: &Vec<u8>| <$signature>::from_der(encoding).ok(),
          &der,
          &truncated,
          |decoded| assert_eq!(decoded.as_bytes().as_slice(), fixed.as_slice()),
        ),
      );
      g.bench_function(
        "rustcrypto",
        validating_row(
          |encoding: &Vec<u8>| <$oracle_signature>::from_der(encoding).ok(),
          &der,
          &truncated,
          |decoded| assert_eq!(decoded.to_bytes().as_slice(), fixed.as_slice()),
        ),
      );
      g.finish();
    }
  };
}

ecdsa_der_bench!(
  ecdsa_p256_signature_der,
  "ecdsa-p256",
  EcdsaP256Signature,
  P256OracleSigningKey,
  P256OracleSignature,
  32,
  ring::signature::ECDSA_P256_SHA256_FIXED,
  ring::signature::ECDSA_P256_SHA256_ASN1
);
ecdsa_der_bench!(
  ecdsa_p384_signature_der,
  "ecdsa-p384",
  EcdsaP384Signature,
  P384OracleSigningKey,
  P384OracleSignature,
  48,
  ring::signature::ECDSA_P384_SHA384_FIXED,
  ring::signature::ECDSA_P384_SHA384_ASN1
);

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

// Ed25519 signing keypair from 32 secret-key bytes. Timed: secret import, key expansion, public-key derivation,
// and the keypair object. Every row must hold the RFC 8032 public key.
fn ed25519_keypair_from_secret(c: &mut Criterion) {
  if !bench_config::selected("ed25519/keypair-from-secret") {
    return;
  }
  let secret_bytes = hex::<32>(ED25519_RFC8032_SECRET);
  let expected = hex::<32>(ED25519_RFC8032_PUBLIC);
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

// Ed25519 signing with a key object built outside timing. Timed: signing and the signature. Each library keeps
// what its key object caches: `rscrypto` holds the expanded secret and public key; `rscrypto-direct-secret`
// signs from the bare secret key and derives the public key in each call.
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
    // RFC 8032 signing is deterministic, so every row must return Dalek's signature; the Dalek row must return
    // ring's.
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
        |signature| assert_eq!(signature.to_bytes().as_slice(), ring_kp.sign(data).as_ref()),
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

/// An encoded Ed25519 public key that is not a curve point: no x satisfies the curve equation for y = 2.
const ED25519_OFF_CURVE_KEY: [u8; 32] = {
  let mut key = [0u8; 32];
  key[0] = 2;
  key
};

// Ed25519 verification. Every row verifies Dalek's signature, which RFC 8032 makes every library's signature for
// this key and message. Each row's check requires a rejection once one byte of R changes. Verification rules
// differ: rscrypto and Dalek `verify_strict` reject small-order keys and R values; the other libraries apply
// their own rules.
//
// `ed25519/verify/reuse-key`: each library's key object is built outside timing. Timed: verification.
// - rscrypto and Dalek reuse the decoded public point.
// - aws-lc-rs reuses a `ParsedPublicKey`, which saves parsing and allocation; AWS-LC still decodes the point in
//   each call.
// - ring and dryoc are absent: they have no API that reuses a parsed key.
//
// `ed25519/verify/import-key`: every row gets the same 32-byte key and 64-byte signature. Timed: key import and
// point decoding, signature parsing, verification, and drop of per-call objects. aws-lc-rs allocates an EVP_PKEY.
// Each row's check also requires the same path to reject an encoding that is not a curve point.
fn ed25519_verify(c: &mut Criterion) {
  if !bench_config::selected("ed25519/verify/") {
    return;
  }
  use dryoc::classic::crypto_sign::crypto_sign_verify_detached;

  let secret_bytes = [13u8; 32];
  let signing_key = SigningKey::from_bytes(&secret_bytes);
  let verifying_key = signing_key.verifying_key();
  let public_bytes = verifying_key.to_bytes();
  let public = Ed25519PublicKey::from_bytes(public_bytes);
  aws_lc_bench! {
    let aws_key = aws_lc_rs::signature::ParsedPublicKey::new(&aws_lc_rs::signature::ED25519, public_bytes)
      .expect("valid authentication benchmark operation must succeed");
  }
  let inputs = [0usize, 32, 1024, 16384]
    .into_iter()
    .map(|len| {
      let data = common::random_bytes(len);
      let signature = signing_key.sign(&data).to_bytes();
      let mut tampered = signature;
      tampered[0] ^= 1;
      (len, data, signature, tampered)
    })
    .collect::<Vec<_>>();

  let mut g = c.benchmark_group("ed25519/verify/reuse-key");
  for (len, data, signature_bytes, tampered_bytes) in &inputs {
    common::set_throughput(&mut g, *len);
    let ours = Ed25519Signature::from_bytes(*signature_bytes);
    let tampered = Ed25519Signature::from_bytes(*tampered_bytes);
    let dalek = DalekSignature::from_bytes(signature_bytes);
    let dalek_tampered = DalekSignature::from_bytes(tampered_bytes);

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
            black_box(&aws_key)
              .verify_sig(black_box(data), black_box(signature_bytes))
              .expect("valid authentication benchmark operation must succeed");
            black_box(())
          },
          |()| {
            aws_key.verify_sig(data, tampered_bytes).expect_err("a changed Ed25519 signature must fail");
          },
        ),
      );
    }
  }
  g.finish();

  // Each closure is one row's timed import-and-verify path.
  let rscrypto_import = |key: &[u8; 32], message: &[u8], signature: &[u8; 64]| {
    Ed25519PublicKey::from_bytes(*key)
      .verify(message, &Ed25519Signature::from_bytes(*signature))
      .is_ok()
  };
  let dalek_import = |key: &[u8; 32], message: &[u8], signature: &[u8; 64]| {
    DalekVerifyingKey::from_bytes(key).is_ok_and(|key| {
      key
        .verify_strict(message, &DalekSignature::from_bytes(signature))
        .is_ok()
    })
  };
  let ring_import = |key: &[u8; 32], message: &[u8], signature: &[u8; 64]| {
    ring::signature::UnparsedPublicKey::new(&ring::signature::ED25519, key)
      .verify(message, signature)
      .is_ok()
  };
  let dryoc_import =
    |key: &[u8; 32], message: &[u8], signature: &[u8; 64]| crypto_sign_verify_detached(signature, message, key).is_ok();
  let mut g = c.benchmark_group("ed25519/verify/import-key");
  for (len, data, signature, tampered) in &inputs {
    common::set_throughput(&mut g, *len);
    let key = &public_bytes;
    let malformed = &ED25519_OFF_CURVE_KEY;
    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      import_and_verify(rscrypto_import, key, malformed, data, signature, tampered),
    );
    g.bench_function(
      BenchmarkId::new("dalek", len),
      import_and_verify(dalek_import, key, malformed, data, signature, tampered),
    );
    aws_lc_bench! {
      let aws_import = |key: &[u8; 32], message: &[u8], signature: &[u8; 64]| {
        aws_lc_rs::signature::UnparsedPublicKey::new(&aws_lc_rs::signature::ED25519, key)
          .verify(message, signature)
          .is_ok()
      };
      g.bench_function(
        BenchmarkId::new("aws-lc-rs", len),
        import_and_verify(aws_import, key, malformed, data, signature, tampered),
      );
    }
    g.bench_function(
      BenchmarkId::new("ring", len),
      import_and_verify(ring_import, key, malformed, data, signature, tampered),
    );
    g.bench_function(
      BenchmarkId::new("dryoc", len),
      import_and_verify(dryoc_import, key, malformed, data, signature, tampered),
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

/// RFC 7748 Section 6.1 X25519 vectors: Alice's private and public keys, Bob's public key, and their shared secret.
const X25519_RFC7748_ALICE_PRIVATE: &str = "77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a";
const X25519_RFC7748_ALICE_PUBLIC: &str = "8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a";
const X25519_RFC7748_BOB_PUBLIC: &str = "de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f";
const X25519_RFC7748_SHARED: &str = "4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742";

// X25519 public key from 32 secret-key bytes. Timed: secret import and the fixed-base scalar multiplication.
// aws-lc-rs derives the public key while it imports the private key into an EVP_PKEY. Every row must return
// RFC 7748 Alice's public key. ring is absent: it cannot import a caller secret.
fn x25519_public_key(c: &mut Criterion) {
  if !bench_config::selected("x25519/public-key-from-secret") {
    return;
  }
  use dryoc::classic::crypto_core::crypto_scalarmult_base;

  let secret_bytes = hex::<32>(X25519_RFC7748_ALICE_PRIVATE);
  let expected = hex::<32>(X25519_RFC7748_ALICE_PUBLIC);
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

// X25519 with a private key held outside timing and the peer's encoded 32-byte public key. Every row gets the same
// peer bytes and must return the RFC 7748 shared secret. Timed: peer decoding, the Montgomery ladder, the check
// that rejects an all-zero output, and the output. Each row's check also requires the same path to reject the
// low-order peer u = 0.
// - Dalek checks its output with `was_contributory`.
// - aws-lc-rs parses the peer into an EVP_PKEY in each call. Its `ParsedPublicKey` would move that parse out of
//   timing, but `agree` consumes the key, so reuse needs a clone in each call.
// - ring is absent: its private keys are one-use, so it has no reusable private key.
fn x25519_diffie_hellman(c: &mut Criterion) {
  if !bench_config::selected("x25519/diffie-hellman/") {
    return;
  }
  use dryoc::classic::crypto_core::crypto_scalarmult;

  let alice_bytes = hex::<32>(X25519_RFC7748_ALICE_PRIVATE);
  let peer = hex::<32>(X25519_RFC7748_BOB_PUBLIC);
  let low_order = [0u8; 32];
  let expected = hex::<32>(X25519_RFC7748_SHARED);
  let alice = X25519SecretKey::from_bytes(alice_bytes);
  let dalek_alice = DalekX25519Secret::from(alice_bytes);
  aws_lc_bench! {
    let aws_alice = aws_lc_rs::agreement::PrivateKey::from_private_key(&aws_lc_rs::agreement::X25519, &alice_bytes)
      .expect("valid authentication benchmark operation must succeed");
  }
  // Each closure is one row's timed path. It returns the shared secret, or `None` when it rejects the peer.
  let rscrypto_agree = |peer: &[u8; 32]| {
    alice
      .diffie_hellman(&X25519PublicKey::from_bytes(*peer))
      .ok()
      .map(|shared| *shared.as_bytes())
  };
  let dalek_agree = |peer: &[u8; 32]| {
    let shared = dalek_alice.diffie_hellman(&DalekX25519PublicKey::from(*peer));
    shared.was_contributory().then(|| *shared.as_bytes())
  };
  let dryoc_agree = |peer: &[u8; 32]| {
    let mut shared = [0u8; 32];
    crypto_scalarmult(&mut shared, &alice_bytes, peer).ok().map(|()| shared)
  };
  let mut g = c.benchmark_group("x25519/diffie-hellman/encoded-peer");
  let matches = |shared: [u8; 32]| assert_eq!(shared, expected);
  g.bench_function("rscrypto", validating_row(rscrypto_agree, &peer, &low_order, matches));
  g.bench_function("dalek", validating_row(dalek_agree, &peer, &low_order, matches));
  aws_lc_bench! {
    let aws_agree = |peer: &[u8; 32]| {
      aws_lc_rs::agreement::agree(
        &aws_alice,
        aws_lc_rs::agreement::UnparsedPublicKey::new(&aws_lc_rs::agreement::X25519, peer),
        (),
        |bytes| Ok::<[u8; 32], ()>(array_from_slice(bytes)),
      )
      .ok()
    };
    g.bench_function("aws-lc-rs", validating_row(aws_agree, &peer, &low_order, matches));
  }
  g.bench_function("dryoc", validating_row(dryoc_agree, &peer, &low_order, matches));
  g.finish();
}

/// Imports caller scalar bytes as a one-use P-256 secret, or returns `None` when the scalar is zero or at least
/// the group order.
fn try_p256_ephemeral(bytes: &[u8; 32]) -> Option<P256EphemeralSecret> {
  P256EphemeralSecret::try_generate_with(|candidate| {
    candidate.copy_from_slice(bytes);
    Ok::<(), core::convert::Infallible>(())
  })
  .ok()
}

fn p256_ephemeral(bytes: [u8; 32]) -> P256EphemeralSecret {
  try_p256_ephemeral(&bytes).expect("valid authentication benchmark scalar must be accepted")
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

/// The NIST P-256 peer key with its y-coordinate changed so that the point is off the curve.
fn p256_off_curve_peer() -> [u8; 65] {
  let off_curve = flip_last_bit(P256_NIST_PEER);
  p256::PublicKey::from_sec1_bytes(&off_curve).expect_err("RustCrypto must reject the changed point");
  off_curve
}

// P-256 ECDH private key from caller-filled scalar bytes. Timed: the scalar range check and the private-key object,
// with no public-key derivation. Each row's check derives the public key outside timing and must reproduce the
// NIST CAVP vector, and the same import path must reject the zero scalar.
// - aws-lc-rs is absent: its import also derives the public point. `p256-ecdh/public-key-from-secret` times it.
fn p256_ecdh_key_generation(c: &mut Criterion) {
  if !bench_config::selected("p256-ecdh/key-generation") {
    return;
  }
  let scalar = P256_NIST_PRIVATE;
  let zero = [0u8; 32];
  let mut g = c.benchmark_group("p256-ecdh/key-generation");
  g.bench_function(
    "rscrypto",
    validating_row(try_p256_ephemeral, &scalar, &zero, |secret| {
      assert_eq!(secret.public_key().to_sec1_bytes(), P256_NIST_PUBLIC)
    }),
  );
  g.bench_function(
    "rustcrypto-p256",
    validating_row(
      |scalar: &[u8; 32]| P256OracleSecretKey::from_slice(scalar).ok(),
      &scalar,
      &zero,
      |secret| assert_eq!(secret.public_key().to_sec1_point(false).as_bytes(), P256_NIST_PUBLIC),
    ),
  );
  g.bench_function(
    "crrl",
    validating_row(
      |scalar: &[u8; 32]| crrl::p256::PrivateKey::decode(scalar),
      &scalar,
      &zero,
      |secret| assert_eq!(secret.to_public_key().encode_uncompressed(), P256_NIST_PUBLIC),
    ),
  );
  g.finish();
}

// P-256 ECDH public key from a private key held outside timing. Timed: the fixed-base scalar multiplication,
// affine conversion, and the uncompressed SEC1 encoding. Fixed-key rows must reproduce the NIST CAVP vector.
// - ring derives from a random one-use key generated outside timing. Its check pairs that key with the NIST public
//   key and compares RustCrypto's agreement.
// - aws-lc-rs is absent: `PrivateKey` derives the public point at import, and `compute_public_key` only copies
//   it out. libcrux is absent: it has no private-key object. `p256-ecdh/public-key-from-secret` times both.
fn p256_ecdh_public_key(c: &mut Criterion) {
  if !bench_config::selected("p256-ecdh/public-key/") {
    return;
  }
  let scalar = P256_NIST_PRIVATE;
  let ours = p256_ephemeral(scalar);
  let rustcrypto = P256OracleSecretKey::from_slice(&scalar).expect("valid RustCrypto scalar");
  let crrl_scalar = crrl_p256_scalar(scalar);
  let mut g = c.benchmark_group("p256-ecdh/public-key");
  g.bench_function(
    "rscrypto",
    checked(
      || black_box(black_box(&ours).public_key()),
      |public| assert_eq!(public.to_sec1_bytes(), P256_NIST_PUBLIC),
    ),
  );
  g.bench_function(
    "rustcrypto-p256",
    checked(
      || black_box(black_box(&rustcrypto).public_key().to_sec1_point(false)),
      |public| assert_eq!(public.as_bytes(), P256_NIST_PUBLIC),
    ),
  );
  g.bench_function(
    "crrl",
    checked(
      || black_box(crrl::p256::Point::mulgen(black_box(&crrl_scalar)).encode_uncompressed()),
      |public| assert_eq!(public, P256_NIST_PUBLIC),
    ),
  );
  ring_p256_bench! {
    let rng = ring::rand::SystemRandom::new();
    let generate = || {
      ring::agreement::EphemeralPrivateKey::generate(&ring::agreement::ECDH_P256, &rng)
        .expect("ring scalar generation")
    };
    g.bench_function(
      "ring",
      checked_routine(
        (),
        |_| {
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

// P-256 ECDH public key from caller scalar bytes. Every row gets the same 32-byte scalar. Timed: scalar validation,
// any private-key object, the fixed-base scalar multiplication, and the public-key encoding. Each row must return
// the NIST CAVP public key, and the same path must reject the zero scalar.
// - aws-lc-rs allocates an EVP_PKEY and derives the public point in `PrivateKey::from_private_key`;
//   `compute_public_key` copies it out.
// - libcrux `dh_initiator` writes the 64 coordinate bytes without the 0x04 prefix.
// - ring is absent: it cannot import a caller scalar for agreement.
fn p256_ecdh_public_key_from_secret(c: &mut Criterion) {
  if !bench_config::selected("p256-ecdh/public-key-from-secret") {
    return;
  }
  let scalar = P256_NIST_PRIVATE;
  let zero = [0u8; 32];
  let mut g = c.benchmark_group("p256-ecdh/public-key-from-secret");
  g.bench_function(
    "rscrypto",
    validating_row(
      |scalar: &[u8; 32]| try_p256_ephemeral(scalar).map(|secret| secret.public_key()),
      &scalar,
      &zero,
      |public| assert_eq!(public.to_sec1_bytes(), P256_NIST_PUBLIC),
    ),
  );
  g.bench_function(
    "rustcrypto-p256",
    validating_row(
      |scalar: &[u8; 32]| {
        P256OracleSecretKey::from_slice(scalar)
          .ok()
          .map(|secret| secret.public_key().to_sec1_point(false))
      },
      &scalar,
      &zero,
      |public| assert_eq!(public.as_bytes(), P256_NIST_PUBLIC),
    ),
  );
  g.bench_function(
    "crrl",
    validating_row(
      |scalar: &[u8; 32]| {
        crrl::p256::PrivateKey::decode(scalar).map(|secret| secret.to_public_key().encode_uncompressed())
      },
      &scalar,
      &zero,
      |public| assert_eq!(public, P256_NIST_PUBLIC),
    ),
  );
  g.bench_function(
    "libcrux",
    validating_row(
      |scalar: &[u8; 32]| {
        let mut public = [0u8; 64];
        libcrux_p256::dh_initiator(&mut public, scalar).then_some(public)
      },
      &scalar,
      &zero,
      |public| assert_eq!(public, P256_NIST_PUBLIC[1..]),
    ),
  );
  aws_lc_bench! {
    g.bench_function(
      "aws-lc-rs",
      validating_row(
        |scalar: &[u8; 32]| {
          aws_lc_rs::agreement::PrivateKey::from_private_key(&aws_lc_rs::agreement::ECDH_P256, scalar)
            .ok()
            .and_then(|secret| secret.compute_public_key().ok())
        },
        &scalar,
        &zero,
        |public| assert_eq!(public.as_ref(), P256_NIST_PUBLIC),
      ),
    );
  }
  g.finish();
}

// P-256 public-key parsing. Every row gets the NIST CAVP peer key as 65 uncompressed SEC1 bytes. Timed: format,
// range, and on-curve validation and the parsed-key object; aws-lc-rs also allocates an EVP_PKEY. Each row's
// check pairs the parsed key with the NIST private key and must return the NIST shared secret, and the same path
// must reject a point that is not on the curve.
fn p256_ecdh_parse(c: &mut Criterion) {
  if !bench_config::selected("p256-ecdh/parse") {
    return;
  }
  let off_curve = p256_off_curve_peer();
  let rustcrypto_secret = P256OracleSecretKey::from_slice(&P256_NIST_PRIVATE).expect("valid RustCrypto scalar");
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
    validating_row(
      |encoded: &[u8; 65]| P256PublicKey::from_sec1_bytes(encoded).ok(),
      &P256_NIST_PEER,
      &off_curve,
      |public| {
        assert_eq!(
          *p256_ephemeral(P256_NIST_PRIVATE).diffie_hellman(&public).as_bytes(),
          P256_NIST_SHARED
        )
      },
    ),
  );
  g.bench_function(
    "rustcrypto-p256",
    validating_row(
      |encoded: &[u8; 65]| p256::PublicKey::from_sec1_bytes(encoded).ok(),
      &P256_NIST_PEER,
      &off_curve,
      |public| {
        let shared = p256::ecdh::diffie_hellman(rustcrypto_secret.to_nonzero_scalar(), public.as_affine());
        assert_eq!(shared.raw_secret_bytes().as_slice(), P256_NIST_SHARED);
      },
    ),
  );
  g.bench_function(
    "crrl",
    validating_row(
      |encoded: &[u8; 65]| crrl::p256::Point::decode(encoded),
      &P256_NIST_PEER,
      &off_curve,
      |point| {
        let shared = core::ops::Mul::mul(point, crrl_p256_scalar(P256_NIST_PRIVATE)).encode_uncompressed();
        assert_eq!(shared[1..33], P256_NIST_SHARED);
      },
    ),
  );
  aws_lc_bench! {
    g.bench_function(
      "aws-lc-rs",
      validating_row(
        |encoded: &[u8; 65]| {
          aws_lc_rs::agreement::ParsedPublicKey::try_from(aws_lc_rs::agreement::UnparsedPublicKey::new(
            &aws_lc_rs::agreement::ECDH_P256,
            encoded,
          ))
          .ok()
        },
        &P256_NIST_PEER,
        &off_curve,
        |public| {
          let shared = aws_lc_rs::agreement::agree(&aws_secret, public, (), |bytes| {
            Ok::<[u8; 32], ()>(array_from_slice(bytes))
          })
          .expect("AWS-LC benchmark agreement");
          assert_eq!(shared, P256_NIST_SHARED);
        },
      ),
    );
  }
  g.finish();
}

// P-256 ECDH agreement with the NIST CAVP key pair. Fixed-key rows must return the NIST shared secret. ring draws a
// random one-use key outside timing, so its check pairs that key with the NIST public key and compares
// RustCrypto's agreement.
//
// `p256-ecdh/agreement/parsed-peer`: the peer key is parsed and validated outside timing. Timed: the scalar
// multiplication, the shared-secret output, and drop of per-call objects.
// - rscrypto consumes a one-use secret built outside timing. RustCrypto and crrl reuse a scalar; crrl encodes both
//   coordinates of the result.
// - aws-lc-rs reuses a `PrivateKey`, but `agree` consumes the peer, so each call clones the `ParsedPublicKey`: an
//   atomic reference-count increment and a 65-byte heap copy.
// - ring and libcrux are absent: neither has a parsed-peer API.
//
// `p256-ecdh/agreement/encoded-peer`: every row gets the NIST peer key as 65 uncompressed SEC1 bytes. Timed: peer
// parsing and validation, then the agreement as above. aws-lc-rs allocates an EVP_PKEY for the peer. libcrux
// checks the 0x04 prefix with `uncompressed_to_raw`, validates the point and the scalar in `dh_responder`, and
// writes both coordinates. Each row's check also requires the same path to reject a point that is not on the
// curve.
fn p256_ecdh_agreement(c: &mut Criterion) {
  if !bench_config::selected("p256-ecdh/agreement/") {
    return;
  }
  let scalar = P256_NIST_PRIVATE;
  let off_curve = p256_off_curve_peer();
  let peer = P256PublicKey::from_sec1_bytes(&P256_NIST_PEER).expect("valid benchmark peer point");
  let rustcrypto_secret = P256OracleSecretKey::from_slice(&scalar).expect("valid RustCrypto scalar");
  let rustcrypto_peer = p256::PublicKey::from_sec1_bytes(&P256_NIST_PEER).expect("valid RustCrypto peer point");
  let crrl_scalar = crrl_p256_scalar(scalar);
  let crrl_peer = crrl::p256::Point::decode(&P256_NIST_PEER).expect("valid CRRL peer point");
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

  let mut g = c.benchmark_group("p256-ecdh/agreement/parsed-peer");
  g.bench_function(
    "rscrypto",
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
    "rustcrypto-p256",
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
    "crrl",
    checked(
      || black_box(core::ops::Mul::mul(black_box(crrl_peer), black_box(crrl_scalar)).encode_uncompressed()),
      |shared| assert_eq!(shared[1..33], P256_NIST_SHARED),
    ),
  );
  aws_lc_bench! {
    g.bench_function(
      "aws-lc-rs",
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
  g.finish();

  // Each closure is one row's timed path. It returns the shared secret, or `None` when it rejects the peer.
  let rscrypto_agree = |secret: P256EphemeralSecret, peer: &[u8; 65]| {
    P256PublicKey::from_sec1_bytes(peer)
      .ok()
      .map(|peer| secret.diffie_hellman(&peer))
  };
  let mut g = c.benchmark_group("p256-ecdh/agreement/encoded-peer");
  g.bench_function(
    "rscrypto",
    checked_routine(
      (),
      |_| {
        let shared = rscrypto_agree(p256_ephemeral(scalar), &P256_NIST_PEER).expect("valid benchmark peer point");
        assert_eq!(shared.as_bytes(), &P256_NIST_SHARED);
        assert!(
          rscrypto_agree(p256_ephemeral(scalar), &off_curve).is_none(),
          "an off-curve peer must fail"
        );
      },
      |b, _| {
        b.iter_batched(
          || p256_ephemeral(scalar),
          |secret| black_box(rscrypto_agree(secret, black_box(&P256_NIST_PEER)).expect("valid benchmark peer point")),
          BatchSize::SmallInput,
        )
      },
    ),
  );
  g.bench_function(
    "rustcrypto-p256",
    validating_row(
      |peer: &[u8; 65]| {
        p256::PublicKey::from_sec1_bytes(peer)
          .ok()
          .map(|peer| p256::ecdh::diffie_hellman(rustcrypto_secret.to_nonzero_scalar(), peer.as_affine()))
      },
      &P256_NIST_PEER,
      &off_curve,
      |shared| assert_eq!(shared.raw_secret_bytes().as_slice(), P256_NIST_SHARED),
    ),
  );
  g.bench_function(
    "crrl",
    validating_row(
      |peer: &[u8; 65]| {
        crrl::p256::Point::decode(peer).map(|peer| core::ops::Mul::mul(peer, crrl_scalar).encode_uncompressed())
      },
      &P256_NIST_PEER,
      &off_curve,
      |shared| assert_eq!(shared[1..33], P256_NIST_SHARED),
    ),
  );
  g.bench_function(
    "libcrux",
    validating_row(
      |peer: &[u8; 65]| {
        let mut raw = [0u8; 64];
        let mut shared = [0u8; 64];
        (libcrux_p256::uncompressed_to_raw(peer, &mut raw) && libcrux_p256::dh_responder(&mut shared, &raw, &scalar))
          .then_some(shared)
      },
      &P256_NIST_PEER,
      &off_curve,
      |shared| assert_eq!(shared[..32], P256_NIST_SHARED),
    ),
  );
  aws_lc_bench! {
    g.bench_function(
      "aws-lc-rs",
      validating_row(
        |peer: &[u8; 65]| {
          aws_lc_rs::agreement::agree(
            &aws_secret,
            aws_lc_rs::agreement::UnparsedPublicKey::new(&aws_lc_rs::agreement::ECDH_P256, peer),
            (),
            |bytes| Ok::<[u8; 32], ()>(array_from_slice(bytes)),
          )
          .ok()
        },
        &P256_NIST_PEER,
        &off_curve,
        |shared| assert_eq!(shared, P256_NIST_SHARED),
      ),
    );
  }
  ring_p256_bench! {
    let rng = ring::rand::SystemRandom::new();
    let generate = || {
      ring::agreement::EphemeralPrivateKey::generate(&ring::agreement::ECDH_P256, &rng)
        .expect("ring scalar generation")
    };
    let ring_agree = |secret: ring::agreement::EphemeralPrivateKey, peer: &[u8; 65]| {
      ring::agreement::agree_ephemeral(
        secret,
        &ring::agreement::UnparsedPublicKey::new(&ring::agreement::ECDH_P256, peer),
        array_from_slice::<32>,
      )
      .ok()
    };
    g.bench_function(
      "ring",
      checked_routine(
        (),
        |_| {
          // The NIST peer has no published private key, so the check pairs a random key with the NIST public
          // key and compares RustCrypto's agreement between the NIST private key and the random public key.
          let secret = generate();
          let public = secret.compute_public_key().expect("ring public derivation");
          let shared = ring_agree(secret, &P256_NIST_PUBLIC).expect("ring benchmark agreement");
          let ring_public = p256::PublicKey::from_sec1_bytes(public.as_ref()).expect("valid ring public key");
          let expected = p256::ecdh::diffie_hellman(rustcrypto_secret.to_nonzero_scalar(), ring_public.as_affine());
          assert_eq!(shared.as_slice(), expected.raw_secret_bytes().as_slice());
          assert!(ring_agree(generate(), &off_curve).is_none(), "an off-curve peer must fail");
        },
        |b, _| {
          b.iter_batched(
            &generate,
            |secret| black_box(ring_agree(secret, black_box(&P256_NIST_PEER)).expect("ring benchmark agreement")),
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

/// Imports caller scalar bytes as a one-use P-384 secret, or returns `None` when the scalar is zero or at least
/// the group order.
fn try_p384_ephemeral(bytes: &[u8; 48]) -> Option<P384EphemeralSecret> {
  P384EphemeralSecret::try_generate_with(|candidate| {
    candidate.copy_from_slice(bytes);
    Ok::<(), core::convert::Infallible>(())
  })
  .ok()
}

fn p384_ephemeral(bytes: [u8; 48]) -> P384EphemeralSecret {
  try_p384_ephemeral(&bytes).expect("valid authentication benchmark scalar must be accepted")
}

/// The NIST P-384 peer key with its y-coordinate changed so that the point is off the curve.
fn p384_off_curve_peer() -> [u8; 97] {
  let off_curve = flip_last_bit(P384_NIST_PEER);
  p384::PublicKey::from_sec1_bytes(&off_curve).expect_err("RustCrypto must reject the changed point");
  off_curve
}

// P-384 ECDH private key from caller-filled scalar bytes. Timed: the scalar range check and the private-key object,
// with no public-key derivation. Each row's check derives the public key outside timing and must reproduce the
// NIST CAVP vector, and the same import path must reject the zero scalar.
// - aws-lc-rs is absent: its import also derives the public point. `p384-ecdh/public-key-from-secret` times it.
fn p384_ecdh_key_generation(c: &mut Criterion) {
  if !bench_config::selected("p384-ecdh/key-generation") {
    return;
  }
  let scalar = P384_NIST_PRIVATE;
  let zero = [0u8; 48];
  let mut g = c.benchmark_group("p384-ecdh/key-generation");
  g.bench_function(
    "rscrypto",
    validating_row(try_p384_ephemeral, &scalar, &zero, |secret| {
      assert_eq!(secret.public_key().to_sec1_bytes(), P384_NIST_PUBLIC)
    }),
  );
  g.bench_function(
    "rustcrypto-p384",
    validating_row(
      |scalar: &[u8; 48]| P384OracleSecretKey::from_slice(scalar).ok(),
      &scalar,
      &zero,
      |secret| assert_eq!(secret.public_key().to_sec1_point(false).as_bytes(), P384_NIST_PUBLIC),
    ),
  );
  g.finish();
}

// P-384 ECDH public key from a private key held outside timing. Timed: the fixed-base scalar multiplication,
// affine conversion, and the uncompressed SEC1 encoding. Fixed-key rows must reproduce the NIST CAVP vector.
// - ring derives from a random one-use key generated outside timing. Its check pairs that key with the NIST public
//   key and compares RustCrypto's agreement.
// - aws-lc-rs is absent: `PrivateKey` derives the public point at import, and `compute_public_key` only copies
//   it out. `p384-ecdh/public-key-from-secret` times that import.
fn p384_ecdh_public_key(c: &mut Criterion) {
  if !bench_config::selected("p384-ecdh/public-key/") {
    return;
  }
  let scalar = P384_NIST_PRIVATE;
  let ours = p384_ephemeral(scalar);
  let rustcrypto = P384OracleSecretKey::from_slice(&scalar).expect("valid RustCrypto scalar");
  let mut g = c.benchmark_group("p384-ecdh/public-key");
  g.bench_function(
    "rscrypto",
    checked(
      || black_box(black_box(&ours).public_key()),
      |public| assert_eq!(public.to_sec1_bytes(), P384_NIST_PUBLIC),
    ),
  );
  g.bench_function(
    "rustcrypto-p384",
    checked(
      || black_box(black_box(&rustcrypto).public_key().to_sec1_point(false)),
      |public| assert_eq!(public.as_bytes(), P384_NIST_PUBLIC),
    ),
  );
  ring_p256_bench! {
    let rng = ring::rand::SystemRandom::new();
    let generate = || {
      ring::agreement::EphemeralPrivateKey::generate(&ring::agreement::ECDH_P384, &rng)
        .expect("ring scalar generation")
    };
    g.bench_function(
      "ring",
      checked_routine(
        (),
        |_| {
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

// P-384 ECDH public key from caller scalar bytes. Every row gets the same 48-byte scalar. Timed: scalar validation,
// any private-key object, the fixed-base scalar multiplication, and the public-key encoding. Each row must return
// the NIST CAVP public key, and the same path must reject the zero scalar.
// - aws-lc-rs allocates an EVP_PKEY and derives the public point in `PrivateKey::from_private_key`;
//   `compute_public_key` copies it out.
// - ring is absent: it cannot import a caller scalar for agreement.
fn p384_ecdh_public_key_from_secret(c: &mut Criterion) {
  if !bench_config::selected("p384-ecdh/public-key-from-secret") {
    return;
  }
  let scalar = P384_NIST_PRIVATE;
  let zero = [0u8; 48];
  let mut g = c.benchmark_group("p384-ecdh/public-key-from-secret");
  g.bench_function(
    "rscrypto",
    validating_row(
      |scalar: &[u8; 48]| try_p384_ephemeral(scalar).map(|secret| secret.public_key()),
      &scalar,
      &zero,
      |public| assert_eq!(public.to_sec1_bytes(), P384_NIST_PUBLIC),
    ),
  );
  g.bench_function(
    "rustcrypto-p384",
    validating_row(
      |scalar: &[u8; 48]| {
        P384OracleSecretKey::from_slice(scalar)
          .ok()
          .map(|secret| secret.public_key().to_sec1_point(false))
      },
      &scalar,
      &zero,
      |public| assert_eq!(public.as_bytes(), P384_NIST_PUBLIC),
    ),
  );
  aws_lc_bench! {
    g.bench_function(
      "aws-lc-rs",
      validating_row(
        |scalar: &[u8; 48]| {
          aws_lc_rs::agreement::PrivateKey::from_private_key(&aws_lc_rs::agreement::ECDH_P384, scalar)
            .ok()
            .and_then(|secret| secret.compute_public_key().ok())
        },
        &scalar,
        &zero,
        |public| assert_eq!(public.as_ref(), P384_NIST_PUBLIC),
      ),
    );
  }
  g.finish();
}

// P-384 public-key parsing. Every row gets the NIST CAVP peer key as 97 uncompressed SEC1 bytes. Timed: format,
// range, and on-curve validation and the parsed-key object. Each row's check pairs the parsed key with the NIST
// private key and must return the NIST shared secret, and the same path must reject a point that is not on the
// curve.
fn p384_ecdh_parse(c: &mut Criterion) {
  if !bench_config::selected("p384-ecdh/parse") {
    return;
  }
  let off_curve = p384_off_curve_peer();
  let rustcrypto_secret = P384OracleSecretKey::from_slice(&P384_NIST_PRIVATE).expect("valid RustCrypto scalar");
  let mut g = c.benchmark_group("p384-ecdh/parse");
  g.bench_function(
    "rscrypto",
    validating_row(
      |encoded: &[u8; 97]| P384PublicKey::from_sec1_bytes(encoded).ok(),
      &P384_NIST_PEER,
      &off_curve,
      |public| {
        assert_eq!(
          *p384_ephemeral(P384_NIST_PRIVATE).diffie_hellman(&public).as_bytes(),
          P384_NIST_SHARED
        )
      },
    ),
  );
  g.bench_function(
    "rustcrypto-p384",
    validating_row(
      |encoded: &[u8; 97]| p384::PublicKey::from_sec1_bytes(encoded).ok(),
      &P384_NIST_PEER,
      &off_curve,
      |public| {
        let shared = p384::ecdh::diffie_hellman(rustcrypto_secret.to_nonzero_scalar(), public.as_affine());
        assert_eq!(shared.raw_secret_bytes().as_slice(), P384_NIST_SHARED);
      },
    ),
  );
  g.finish();
}

// P-384 ECDH agreement with the NIST CAVP key pair. Fixed-key rows must return the NIST shared secret. ring draws a
// random one-use key outside timing, so its check pairs that key with the NIST public key and compares
// RustCrypto's agreement.
//
// `p384-ecdh/agreement/parsed-peer`: the peer key is parsed and validated outside timing. Timed: the scalar
// multiplication, the shared-secret output, and drop of per-call objects.
// - rscrypto consumes a one-use secret built outside timing. RustCrypto reuses a scalar.
// - aws-lc-rs reuses a `PrivateKey`, but `agree` consumes the peer, so each call clones the `ParsedPublicKey`: an
//   atomic reference-count increment and a 97-byte heap copy.
// - ring is absent: it has no parsed-peer API.
//
// `p384-ecdh/agreement/encoded-peer`: every row gets the NIST peer key as 97 uncompressed SEC1 bytes. Timed: peer
// parsing and validation, then the agreement as above. aws-lc-rs allocates an EVP_PKEY for the peer. Each row's
// check also requires the same path to reject a point that is not on the curve.
fn p384_ecdh_agreement(c: &mut Criterion) {
  if !bench_config::selected("p384-ecdh/agreement/") {
    return;
  }
  let scalar = P384_NIST_PRIVATE;
  let off_curve = p384_off_curve_peer();
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

  let mut g = c.benchmark_group("p384-ecdh/agreement/parsed-peer");
  g.bench_function(
    "rscrypto",
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
    "rustcrypto-p384",
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
      "aws-lc-rs",
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
  g.finish();

  // Each closure is one row's timed path. It returns the shared secret, or `None` when it rejects the peer.
  let rscrypto_agree = |secret: P384EphemeralSecret, peer: &[u8; 97]| {
    P384PublicKey::from_sec1_bytes(peer)
      .ok()
      .map(|peer| secret.diffie_hellman(&peer))
  };
  let mut g = c.benchmark_group("p384-ecdh/agreement/encoded-peer");
  g.bench_function(
    "rscrypto",
    checked_routine(
      (),
      |_| {
        let shared = rscrypto_agree(p384_ephemeral(scalar), &P384_NIST_PEER).expect("valid benchmark peer point");
        assert_eq!(shared.as_bytes(), &P384_NIST_SHARED);
        assert!(
          rscrypto_agree(p384_ephemeral(scalar), &off_curve).is_none(),
          "an off-curve peer must fail"
        );
      },
      |b, _| {
        b.iter_batched(
          || p384_ephemeral(scalar),
          |secret| black_box(rscrypto_agree(secret, black_box(&P384_NIST_PEER)).expect("valid benchmark peer point")),
          BatchSize::SmallInput,
        )
      },
    ),
  );
  g.bench_function(
    "rustcrypto-p384",
    validating_row(
      |peer: &[u8; 97]| {
        p384::PublicKey::from_sec1_bytes(peer)
          .ok()
          .map(|peer| p384::ecdh::diffie_hellman(rustcrypto_secret.to_nonzero_scalar(), peer.as_affine()))
      },
      &P384_NIST_PEER,
      &off_curve,
      |shared| assert_eq!(shared.raw_secret_bytes().as_slice(), P384_NIST_SHARED),
    ),
  );
  aws_lc_bench! {
    g.bench_function(
      "aws-lc-rs",
      validating_row(
        |peer: &[u8; 97]| {
          aws_lc_rs::agreement::agree(
            &aws_secret,
            aws_lc_rs::agreement::UnparsedPublicKey::new(&aws_lc_rs::agreement::ECDH_P384, peer),
            (),
            |bytes| Ok::<[u8; 48], ()>(array_from_slice(bytes)),
          )
          .ok()
        },
        &P384_NIST_PEER,
        &off_curve,
        |shared| assert_eq!(shared, P384_NIST_SHARED),
      ),
    );
  }
  ring_p256_bench! {
    let rng = ring::rand::SystemRandom::new();
    let generate = || {
      ring::agreement::EphemeralPrivateKey::generate(&ring::agreement::ECDH_P384, &rng)
        .expect("ring scalar generation")
    };
    let ring_agree = |secret: ring::agreement::EphemeralPrivateKey, peer: &[u8; 97]| {
      ring::agreement::agree_ephemeral(
        secret,
        &ring::agreement::UnparsedPublicKey::new(&ring::agreement::ECDH_P384, peer),
        array_from_slice::<48>,
      )
      .ok()
    };
    g.bench_function(
      "ring",
      checked_routine(
        (),
        |_| {
          // The NIST peer has no published private key, so the check pairs a random key with the NIST public
          // key and compares RustCrypto's agreement between the NIST private key and the random public key.
          let secret = generate();
          let public = secret.compute_public_key().expect("ring public derivation");
          let shared = ring_agree(secret, &P384_NIST_PUBLIC).expect("ring benchmark agreement");
          let ring_public = p384::PublicKey::from_sec1_bytes(public.as_ref()).expect("valid ring public key");
          let expected = p384::ecdh::diffie_hellman(rustcrypto_secret.to_nonzero_scalar(), ring_public.as_affine());
          assert_eq!(shared.as_slice(), expected.raw_secret_bytes().as_slice());
          assert!(ring_agree(generate(), &off_curve).is_none(), "an off-curve peer must fail");
        },
        |b, _| {
          b.iter_batched(
            &generate,
            |secret| black_box(ring_agree(secret, black_box(&P384_NIST_PEER)).expect("ring benchmark agreement")),
            BatchSize::SmallInput,
          )
        },
      ),
    );
  }
  g.finish();
}

// Every deterministic row must reproduce libcrux, an independent FIPS 203 implementation, from the same seed, key,
// ciphertext, and randomness; the libcrux rows must reproduce fips203. AWS-LC rows draw internal entropy, so
// rscrypto and AWS-LC must instead open each other's outputs. Fixtures are public deterministic benchmark data;
// output conversion and drop remain timed. See docs/benchmarking.md for the comparison boundaries.
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
      // The libcrux row must instead reproduce fips203.
      let independent = || {
        let (ek, dk) =
          $fips::KG::keygen_from_seed(array_from_slice(&key_random[..32]), array_from_slice(&key_random[32..]));
        (ek.into_bytes(), dk.into_bytes())
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
      }, |keys| assert_eq!(keys, independent())));
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
      // Only the AWS-LC rows use the decapsulation key, so other targets leave it unused.
      let (ek, _dk) = <$profile>::generate_keypair(|out| {
        out.copy_from_slice(&key_random);
        Ok::<(), MlKemError>(())
      }).expect("ML-KEM fixture key generation");
      let ek_bytes = ek.to_bytes();
      // Coefficient 0 becomes 4095, at least q = 3329, so the FIPS 203 Section 7.2 modulus check must fail.
      let mut malformed_ek = ek_bytes;
      malformed_ek[0] = 0xff;
      malformed_ek[1] |= 0x0f;
      let prepared_ek = ek.prepare().expect("ML-KEM key preparation");
      let rustcrypto_ek = rustcrypto_ml_kem::EncapsulationKey::<$rustcrypto>::new(&ek_bytes.into()).expect("RustCrypto key import");
      let fips_ek = $fips::EncapsKey::try_from_bytes(ek_bytes).expect("fips203 key import");
      let libcrux_ek = libcrux_ml_kem::MlKemPublicKey::from(ek_bytes);
      assert!($libcrux::validate_public_key(&libcrux_ek), "libcrux must accept the fixture key");
      let expected = || {
        let (ct, ss) = $libcrux::encapsulate(&libcrux_ek, encaps_random);
        (*ct.as_slice(), ss)
      };
      // The libcrux rows must instead reproduce fips203.
      let independent = || {
        let (ss, ct) = fips_ek.encaps_from_seed(&encaps_random);
        (ct.into_bytes(), ss.into_bytes())
      };
      // Groups by key reuse. FIPS 203 Section 7.2 requires the modulus check on an encapsulation key.
      // - `derand-reuse-key`: each library's best reusable key, built and checked once outside timing, with caller
      //   randomness. The keys cache different work. rscrypto's prepared key holds the decoded key, the expanded
      //   public matrix, and H(ek). RustCrypto's key holds the decoded key and H(ek), and samples the matrix in
      //   each call. libcrux and fips203 keys hold the encoded bytes, so decoding, matrix sampling, and H(ek) stay
      //   timed. AWS-LC is absent: it cannot take caller randomness.
      // - `derand-reuse-encoded`: every row reuses an encoded-key wrapper, so decoding, matrix sampling, and H(ek)
      //   stay timed. rscrypto also runs the modulus check in every `Kem::encapsulate`; libcrux and fips203
      //   checked the key once outside timing.
      // - `derand-import-encoded`: every row gets the same key bytes and runs the check inside timing. rscrypto runs
      //   it twice, in `try_from_slice` and in `Kem::encapsulate`. libcrux import does not check, so its row calls
      //   `validate_public_key`. Each row's check requires the same path to reject a key with a coefficient of at
      //   least q.
      // - `internal-entropy-import-encoded`: AWS-LC only. It imports the key bytes, checks the key in
      //   encapsulation, and draws internal entropy.
      let mut g = c.benchmark_group(concat!($group, "/encapsulate/derand-reuse-key"));
      g.bench_function("rscrypto", checked(|| {
        let (ct, ss) = black_box(&prepared_ek).encapsulate(|out| {
          out.copy_from_slice(black_box(&encaps_random)); Ok::<(), MlKemError>(())
        }).expect("ML-KEM encapsulation");
        (ct.to_bytes(), *ss.as_bytes())
      }, |output| assert_eq!(output, expected())));
      g.bench_function("rustcrypto", checked(|| {
        let (ct, ss) = black_box(&rustcrypto_ek).encapsulate_deterministic(&(*black_box(&encaps_random)).into());
        (array_from_slice::<{ <$profile>::CIPHERTEXT_SIZE }>(ct.as_slice()), array_from_slice::<32>(ss.as_slice()))
      }, |output| assert_eq!(output, expected())));
      g.bench_function("libcrux", checked(|| {
        let (ct, ss) = $libcrux::encapsulate(black_box(&libcrux_ek), *black_box(&encaps_random));
        (*ct.as_slice(), ss)
      }, |output| assert_eq!(output, independent())));
      g.bench_function("fips203", checked(|| {
        let (ss, ct) = black_box(&fips_ek).encaps_from_seed(black_box(&encaps_random));
        (ct.into_bytes(), ss.into_bytes())
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
      }, |output| assert_eq!(output, independent())));
      g.bench_function("fips203", checked(|| {
        let (ss, ct) = black_box(&fips_ek).encaps_from_seed(black_box(&encaps_random));
        (ct.into_bytes(), ss.into_bytes())
      }, |output| assert_eq!(output, expected())));
      g.finish();

      // Each closure is one row's timed import path. It returns `None` when it rejects the key.
      let rscrypto_import = |ek_bytes: &[u8; <$profile>::ENCAPSULATION_KEY_SIZE]| -> Option<_> {
        let ek = <<$profile as rscrypto::Kem>::EncapsulationKey>::try_from_slice(ek_bytes).ok()?;
        let (ct, ss) = <$profile>::encapsulate(&ek, |out| {
          out.copy_from_slice(black_box(&encaps_random)); Ok::<(), MlKemError>(())
        }).ok()?;
        Some((ct.to_bytes(), *ss.as_bytes()))
      };
      let rustcrypto_import = |ek_bytes: &[u8; <$profile>::ENCAPSULATION_KEY_SIZE]| {
        rustcrypto_ml_kem::EncapsulationKey::<$rustcrypto>::new(&(*ek_bytes).into()).ok().map(|ek| {
          let (ct, ss) = ek.encapsulate_deterministic(&(*black_box(&encaps_random)).into());
          (array_from_slice::<{ <$profile>::CIPHERTEXT_SIZE }>(ct.as_slice()), array_from_slice::<32>(ss.as_slice()))
        })
      };
      let libcrux_import = |ek_bytes: &[u8; <$profile>::ENCAPSULATION_KEY_SIZE]| {
        let ek = libcrux_ml_kem::MlKemPublicKey::from(*ek_bytes);
        $libcrux::validate_public_key(&ek).then(|| {
          let (ct, ss) = $libcrux::encapsulate(&ek, *black_box(&encaps_random));
          (*ct.as_slice(), ss)
        })
      };
      let fips_import = |ek_bytes: &[u8; <$profile>::ENCAPSULATION_KEY_SIZE]| {
        $fips::EncapsKey::try_from_bytes(*ek_bytes).ok().map(|ek| {
          let (ss, ct) = ek.encaps_from_seed(black_box(&encaps_random));
          (ct.into_bytes(), ss.into_bytes())
        })
      };
      let matches = |output| assert_eq!(output, expected());
      let mut g = c.benchmark_group(concat!($group, "/encapsulate/derand-import-encoded"));
      g.bench_function("rscrypto", validating_row(rscrypto_import, &ek_bytes, &malformed_ek, matches));
      g.bench_function("rustcrypto", validating_row(rustcrypto_import, &ek_bytes, &malformed_ek, matches));
      let reproduces_fips203 = |output| assert_eq!(output, independent());
      g.bench_function("libcrux", validating_row(libcrux_import, &ek_bytes, &malformed_ek, reproduces_fips203));
      g.bench_function("fips203", validating_row(fips_import, &ek_bytes, &malformed_ek, matches));
      g.finish();

      aws_lc_bench! {
        let aws_encapsulate = |key: &aws_lc_rs::kem::EncapsulationKey| {
          key.encapsulate().ok().map(|(ct, ss)| {
            (array_from_slice::<{ <$profile>::CIPHERTEXT_SIZE }>(ct.as_ref()), array_from_slice::<32>(ss.as_ref()))
          })
        };
        // Randomized rows cannot match a fixed ciphertext: rscrypto must open each AWS-LC ciphertext to its secret.
        let opens = |(ct_bytes, ss): ([u8; <$profile>::CIPHERTEXT_SIZE], [u8; 32])| {
          let ct =
            <<$profile as rscrypto::Kem>::Ciphertext>::try_from_slice(&ct_bytes).expect("AWS-LC ciphertext import");
          let secret = <$profile>::decapsulate(&_dk, &ct).expect("AWS-LC encapsulation interoperability");
          assert_eq!(secret.as_bytes(), &ss);
        };
        let mut g = c.benchmark_group(concat!($group, "/encapsulate/internal-entropy-import-encoded"));
        g.bench_function("aws-lc-rs", validating_row(
          |ek_bytes: &[u8; <$profile>::ENCAPSULATION_KEY_SIZE]| {
            aws_lc_rs::kem::EncapsulationKey::new(&$aws_algorithm, ek_bytes).ok().and_then(|key| aws_encapsulate(&key))
          },
          &ek_bytes,
          &malformed_ek,
          opens,
        ));
        g.finish();
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
      // The last 64 bytes of an expanded decapsulation key are H(ek) and z. A changed H(ek) must fail the FIPS 203
      // Section 7.3 hash check.
      let mut malformed_dk = dk_bytes;
      malformed_dk[dk_bytes.len().strict_sub(64)] ^= 1;
      let ct_bytes = ct.to_bytes();
      let prepared_dk = dk.prepare().expect("ML-KEM key preparation");
      let rustcrypto_dk = RustCryptoMlKemDecapsulationKey::<$rustcrypto>::from_seed(RustCryptoMlKemSeed::from(key_random));
      let rustcrypto_ct = ct_bytes.into();
      let fips_dk = $fips::DecapsKey::try_from_bytes(dk_bytes).expect("fips203 key import");
      let fips_ct = $fips::CipherText::try_from_bytes(ct_bytes).expect("fips203 ciphertext import");
      let libcrux_dk = libcrux_ml_kem::MlKemPrivateKey::from(dk_bytes);
      let libcrux_ct = libcrux_ml_kem::MlKemCiphertext::from(ct_bytes);
      assert!($libcrux::validate_private_key(&libcrux_dk, &libcrux_ct), "libcrux must accept the fixture key");
      let expected = || $libcrux::decapsulate(&libcrux_dk, &libcrux_ct);
      // The libcrux rows must instead reproduce fips203.
      let independent = || fips_dk.try_decaps(&fips_ct).expect("fips203 decapsulation").into_bytes();
      // Groups by key reuse. FIPS 203 Section 7.3 requires the hash check on a decapsulation key.
      // - `prepare-key`: rscrypto only. It times the preparation that its `reuse-key` row amortizes: decoding the
      //   key, expanding the public matrix, deriving the decryption mask, and the hash check.
      // - `reuse-key`: each library's best reusable key and ciphertext, built outside timing. The keys cache
      //   different work. rscrypto's prepared key holds the result of `prepare-key`. RustCrypto's key, built from
      //   the seed, holds the decoded keys and samples the matrix for re-encryption in each call. libcrux and
      //   fips203 keys hold the encoded bytes and were checked once, so decoding and matrix sampling stay timed.
      //   AWS-LC holds the encoded key, takes borrowed ciphertext bytes, and runs the hash check in each call.
      // - `reuse-encoded`: every row reuses an encoded-key wrapper, so decoding and matrix sampling stay timed.
      //   rscrypto also runs the hash check in every `Kem::decapsulate`; libcrux and fips203 checked the key once
      //   outside timing.
      // - `import-encoded`: every row gets the same key and ciphertext bytes and runs the check inside timing.
      //   rscrypto runs it twice, in `try_from_slice` and in `Kem::decapsulate`; fips203 import also runs the
      //   modulus check on the embedded encapsulation key; AWS-LC runs it in decapsulation. libcrux import does
      //   not check, so its row calls `validate_private_key`. Each row's check requires the same path to reject a
      //   key with a changed H(ek).
      let mut g = c.benchmark_group(concat!($group, "/decapsulate/prepare-key"));
      g.bench_function("rscrypto", checked(
        || black_box(black_box(&dk).prepare().expect("ML-KEM key preparation")),
        |prepared| assert_eq!(*prepared.decapsulate(&ct).expect("ML-KEM decapsulation").as_bytes(), expected()),
      ));
      g.finish();
      let mut g = c.benchmark_group(concat!($group, "/decapsulate/reuse-key"));
      g.bench_function("rscrypto", checked(|| {
        *black_box(&prepared_dk).decapsulate(black_box(&ct)).expect("ML-KEM decapsulation").as_bytes()
      }, |secret| assert_eq!(secret, expected())));
      g.bench_function("rustcrypto", checked(|| {
        array_from_slice::<32>(black_box(&rustcrypto_dk).decapsulate(black_box(&rustcrypto_ct)).as_slice())
      }, |secret| assert_eq!(secret, expected())));
      g.bench_function("libcrux", checked(|| {
        $libcrux::decapsulate(black_box(&libcrux_dk), black_box(&libcrux_ct))
      }, |secret| assert_eq!(secret, independent())));
      g.bench_function("fips203", checked(|| {
        black_box(&fips_dk).try_decaps(black_box(&fips_ct)).expect("fips203 decapsulation").into_bytes()
      }, |secret| assert_eq!(secret, expected())));
      aws_lc_bench! {
        let aws_dk = AwsMlKemDecapsulationKey::new(&$aws_algorithm, &dk_bytes).expect("AWS-LC key import");
        g.bench_function("aws-lc-rs", checked(|| {
          let secret = black_box(&aws_dk)
            .decapsulate(AwsMlKemCiphertext::from(black_box(ct_bytes.as_slice())))
            .expect("AWS-LC decapsulation");
          array_from_slice::<32>(secret.as_ref())
        }, |secret| assert_eq!(secret, expected())));
      }
      g.finish();
      let mut g = c.benchmark_group(concat!($group, "/decapsulate/reuse-encoded"));
      g.bench_function("rscrypto", checked(|| {
        *<$profile>::decapsulate(black_box(&dk), black_box(&ct)).expect("ML-KEM decapsulation").as_bytes()
      }, |secret| assert_eq!(secret, expected())));
      g.bench_function("libcrux", checked(|| {
        $libcrux::decapsulate(black_box(&libcrux_dk), black_box(&libcrux_ct))
      }, |secret| assert_eq!(secret, independent())));
      g.bench_function("fips203", checked(|| {
        black_box(&fips_dk).try_decaps(black_box(&fips_ct)).expect("fips203 decapsulation").into_bytes()
      }, |secret| assert_eq!(secret, expected())));
      g.finish();

      // Each closure is one row's timed import path. It returns `None` when it rejects the key.
      let rscrypto_import = |dk_bytes: &[u8; <$profile>::DECAPSULATION_KEY_SIZE]| -> Option<[u8; 32]> {
        let dk = <<$profile as rscrypto::Kem>::DecapsulationKey>::try_from_slice(dk_bytes).ok()?;
        let ct = <<$profile as rscrypto::Kem>::Ciphertext>::try_from_slice(black_box(&ct_bytes)).ok()?;
        <$profile>::decapsulate(&dk, &ct).ok().map(|secret| *secret.as_bytes())
      };
      let libcrux_import = |dk_bytes: &[u8; <$profile>::DECAPSULATION_KEY_SIZE]| {
        let dk = libcrux_ml_kem::MlKemPrivateKey::from(*dk_bytes);
        let ct = libcrux_ml_kem::MlKemCiphertext::from(*black_box(&ct_bytes));
        $libcrux::validate_private_key(&dk, &ct).then(|| $libcrux::decapsulate(&dk, &ct))
      };
      let fips_import = |dk_bytes: &[u8; <$profile>::DECAPSULATION_KEY_SIZE]| -> Option<[u8; 32]> {
        let dk = $fips::DecapsKey::try_from_bytes(*dk_bytes).ok()?;
        let ct = $fips::CipherText::try_from_bytes(*black_box(&ct_bytes)).ok()?;
        dk.try_decaps(&ct).ok().map(|secret| secret.into_bytes())
      };
      let matches = |secret| assert_eq!(secret, expected());
      let mut g = c.benchmark_group(concat!($group, "/decapsulate/import-encoded"));
      g.bench_function("rscrypto", validating_row(rscrypto_import, &dk_bytes, &malformed_dk, matches));
      let reproduces_fips203 = |secret| assert_eq!(secret, independent());
      g.bench_function("libcrux", validating_row(libcrux_import, &dk_bytes, &malformed_dk, reproduces_fips203));
      g.bench_function("fips203", validating_row(fips_import, &dk_bytes, &malformed_dk, matches));
      aws_lc_bench! {
        g.bench_function("aws-lc-rs", validating_row(
          |dk_bytes: &[u8; <$profile>::DECAPSULATION_KEY_SIZE]| -> Option<[u8; 32]> {
            let dk = AwsMlKemDecapsulationKey::new(&$aws_algorithm, dk_bytes).ok()?;
            let secret = dk.decapsulate(AwsMlKemCiphertext::from(black_box(ct_bytes.as_slice()))).ok()?;
            Some(array_from_slice::<32>(secret.as_ref()))
          },
          &dk_bytes,
          &malformed_dk,
          matches,
        ));
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
    hmac_sha256_verify,
    hmac_sha384_verify,
    hmac_sha512_verify,
    hmac_sha256_streaming,
    hmac_sha256_internal,
    hkdf_sha256_expand,
    hkdf_sha384_expand,
    hkdf_sha512_expand,
    hkdf_sha256_extract,
    hkdf_sha384_extract,
    hkdf_sha512_extract,
    ecdsa_p256_public_key,
    ecdsa_p256_sign,
    ecdsa_p256_verify,
    #[cfg(all(rscrypto_internal, feature = "diag", feature = "ecdsa-p256"))]
    ecdsa_p256_internal,
    ecdsa_p384_public_key,
    ecdsa_p384_sign,
    ecdsa_p384_verify,
    ecdsa_p256_signature_der,
    ecdsa_p384_signature_der,
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
    p256_ecdh_public_key_from_secret,
    p256_ecdh_parse,
    p256_ecdh_agreement,
    p256_ecdh_tls_roundtrip,
    p384_ecdh_key_generation,
    p384_ecdh_public_key,
    p384_ecdh_public_key_from_secret,
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
