//! OpenSSL EVP comparisons, including SHA3-224 streaming and native IBM Z CPACF candidates.
//! All rows call production APIs. Context construction and destruction are timed;
//! input generation and output-buffer allocation are not.

#[path = "../../../benches/common/criterion.rs"]
mod bench_config;
#[path = "../../../benches/common/mod.rs"]
mod common;

use core::hint::black_box;

use bench_config::{checked, checked_with};
use criterion::{BenchmarkId, Criterion, Throughput};
use openssl::{
  hash::{Hasher, MessageDigest},
  symm::{Cipher, Crypter, Mode},
};
use rscrypto::{Digest, aead::expert::AeadWithNonce};

macro_rules! digest_rows {
  ($function:ident, $name:literal, $ours:ty, $algorithm:expr, $boundaries:expr) => {
    fn $function(c: &mut Criterion) {
      if !bench_config::selected(concat!("openssl-primitives/", $name, "/")) {
        return;
      }
      let algorithm = $algorithm;
      let inputs = common::comp_sizes(&$boundaries);
      let mut group = c.benchmark_group(concat!("openssl-primitives/", $name, "/oneshot"));
      for (len, data) in &inputs {
        common::set_throughput(&mut group, *len);
        let expected = openssl::hash::hash(algorithm, data).expect("EVP digest");
        let ours = <$ours>::digest(data);
        group.bench_function(
          BenchmarkId::new("rscrypto", len),
          checked(
            || <$ours>::digest(black_box(data)),
            |out| assert_eq!(out.as_ref(), expected.as_ref()),
          ),
        );
        group.bench_function(
          BenchmarkId::new("openssl", len),
          checked(
            || openssl::hash::hash(algorithm, black_box(data)).expect("EVP digest"),
            |out| assert_eq!(out.as_ref(), ours.as_ref()),
          ),
        );
      }
      group.finish();

      let data = common::random_bytes(1_048_576);
      let expected = openssl::hash::hash(algorithm, &data).expect("EVP digest");
      let ours = <$ours>::digest(&data);
      let mut group = c.benchmark_group(concat!("openssl-primitives/", $name, "/streaming"));
      group.throughput(Throughput::Bytes(data.len() as u64));
      for chunk_size in [64, 4096] {
        group.bench_function(
          format!("rscrypto/{chunk_size}B"),
          checked(
            || {
              let mut context = <$ours>::new();
              for chunk in data.chunks(chunk_size) {
                context.update(black_box(chunk));
              }
              context.finalize()
            },
            |out| assert_eq!(out.as_ref(), expected.as_ref()),
          ),
        );
        group.bench_function(
          format!("openssl/{chunk_size}B"),
          checked(
            || {
              let mut context = Hasher::new(algorithm).expect("EVP digest context");
              for chunk in data.chunks(chunk_size) {
                context.update(black_box(chunk)).expect("EVP digest update");
              }
              context.finish().expect("EVP digest finish")
            },
            |out| assert_eq!(out.as_ref(), ours.as_ref()),
          ),
        );
      }
      group.finish();
    }
  };
}

digest_rows!(sha224, "sha224", rscrypto::Sha224, MessageDigest::sha224(), [55, 56]);
digest_rows!(sha256, "sha256", rscrypto::Sha256, MessageDigest::sha256(), [55, 56]);
digest_rows!(sha384, "sha384", rscrypto::Sha384, MessageDigest::sha384(), [111, 112]);
digest_rows!(sha512, "sha512", rscrypto::Sha512, MessageDigest::sha512(), [111, 112]);
digest_rows!(
  sha512_256,
  "sha512-256",
  rscrypto::Sha512_256,
  MessageDigest::from_name("SHA512-256").expect("SHA512-256 descriptor"),
  [111, 112]
);
digest_rows!(
  sha3_224,
  "sha3-224",
  rscrypto::Sha3_224,
  MessageDigest::sha3_224(),
  [143, 144]
);
digest_rows!(
  sha3_256,
  "sha3-256",
  rscrypto::Sha3_256,
  MessageDigest::sha3_256(),
  [135, 136]
);
digest_rows!(
  sha3_384,
  "sha3-384",
  rscrypto::Sha3_384,
  MessageDigest::sha3_384(),
  [103, 104]
);
digest_rows!(
  sha3_512,
  "sha3-512",
  rscrypto::Sha3_512,
  MessageDigest::sha3_512(),
  [71, 72]
);

// Public benchmark-only material. Repeated nonces here do not protect application data.
const NONCE: [u8; 12] = [0x07; 12];
const AAD: &[u8] = b"rscrypto-bench";

fn openssl_seal(cipher: Cipher, key: &[u8], data: &[u8], output: &mut [u8]) {
  let mut context = Crypter::new(cipher, Mode::Encrypt, key, Some(&NONCE)).expect("EVP cipher context");
  context.aad_update(AAD).expect("EVP AAD");
  let size = context.update(data, output).expect("EVP encryption");
  let tail = context.finalize(&mut output[size..]).expect("EVP encryption finish");
  assert_eq!(size.strict_add(tail), data.len());
  context.get_tag(&mut output[data.len()..]).expect("EVP tag");
}

fn openssl_open(cipher: Cipher, key: &[u8], sealed: &[u8], output: &mut [u8]) -> bool {
  let (ciphertext, tag) = sealed.split_at(sealed.len().strict_sub(16));
  let mut context = Crypter::new(cipher, Mode::Decrypt, key, Some(&NONCE)).expect("EVP cipher context");
  context.aad_update(AAD).expect("EVP AAD");
  context.set_tag(tag).expect("EVP tag");
  let size = context.update(ciphertext, output).expect("EVP decryption");
  match context.finalize(&mut output[size..]) {
    Ok(tail) => {
      assert_eq!(size.strict_add(tail), ciphertext.len());
      true
    }
    Err(_) => false,
  }
}

macro_rules! aead_rows {
  ($function:ident, $name:literal, $ours:ty, $key:ty, $bytes:expr, $cipher:expr) => {
    fn $function(c: &mut Criterion) {
      if !bench_config::selected(concat!("openssl-primitives/", $name, "/")) {
        return;
      }
      let key_bytes = $bytes;
      let key = <$key>::from_bytes(key_bytes);
      let nonce = rscrypto::aead::Nonce96::from_bytes(NONCE);
      let cipher = $cipher;
      // Empty, partial/exact blocks, record-sized and bulk messages expose setup and steady-state costs.
      let inputs = common::comp_sizes(&[15, 16, 17]);
      for operation in ["seal", "open"] {
        let mut group = c.benchmark_group(format!("openssl-primitives/{}/new-key/{operation}", $name));
        for (len, data) in &inputs {
          common::set_throughput(&mut group, *len);
          let mut expected = vec![0u8; len.strict_add(16)];
          openssl_seal(cipher, &key_bytes, data, &mut expected);
          let mut ours = vec![0u8; expected.len()];
          <$ours>::new(&key)
            .encrypt(&nonce, AAD, data, &mut ours)
            .expect("seal fixture");
          assert_eq!(ours, expected, "independent ciphertext and tag");
          if operation == "seal" {
            group.bench_function(
              BenchmarkId::new("rscrypto", len),
              checked_with(
                vec![0u8; expected.len()],
                |output| {
                  <$ours>::new(black_box(&key))
                    .encrypt(&nonce, AAD, black_box(data), black_box(output))
                    .expect("seal");
                },
                |(), output| assert_eq!(output, &expected),
              ),
            );
            group.bench_function(
              BenchmarkId::new("openssl", len),
              checked_with(
                vec![0u8; expected.len()],
                |output| {
                  openssl_seal(cipher, black_box(&key_bytes), black_box(data), black_box(output));
                },
                |(), output| assert_eq!(output, &ours),
              ),
            );
          } else {
            let mut tampered = expected.clone();
            *tampered.last_mut().expect("tag") ^= 1;
            group.bench_function(
              BenchmarkId::new("rscrypto", len),
              checked_with(
                vec![0u8; *len],
                |output| {
                  <$ours>::new(black_box(&key))
                    .decrypt(&nonce, AAD, black_box(&expected), black_box(output))
                    .expect("open");
                },
                |(), output| {
                  assert_eq!(output, data);
                  let mut rejected = output.clone();
                  assert!(
                    <$ours>::new(&key)
                      .decrypt(&nonce, AAD, &tampered, &mut rejected)
                      .is_err(),
                    "changed tag"
                  );
                },
              ),
            );
            // EVP requires scratch space for a final block, even though GCM preserves length.
            group.bench_function(
              BenchmarkId::new("openssl", len),
              checked_with(
                vec![0u8; expected.len()],
                |output| {
                  assert!(openssl_open(
                    cipher,
                    black_box(&key_bytes),
                    black_box(&ours),
                    black_box(output)
                  ));
                },
                |(), output| {
                  assert_eq!(&output[..*len], data);
                  let mut rejected = output.clone();
                  assert!(
                    !openssl_open(cipher, &key_bytes, &tampered, &mut rejected),
                    "changed tag"
                  );
                },
              ),
            );
          }
        }
        group.finish();
      }
    }
  };
}

aead_rows!(
  aes128,
  "aes128-gcm",
  rscrypto::Aes128Gcm,
  rscrypto::Aes128GcmKey,
  [0x42; 16],
  Cipher::aes_128_gcm()
);
aead_rows!(
  aes256,
  "aes256-gcm",
  rscrypto::Aes256Gcm,
  rscrypto::Aes256GcmKey,
  [0x42; 32],
  Cipher::aes_256_gcm()
);

fn main() {
  let first = std::env::args().nth(1);
  if !std::env::args().any(|arg| arg == "--list") && first.as_deref() != Some("--rscrypto-filter-cases") {
    assert!(
      openssl::version::version().starts_with("OpenSSL 4.0.3 "),
      "OpenSSL 4.0.3 required"
    );
    eprintln!(
      "{}",
      serde_json::json!({"provider_bench_provenance": rscrypto_provider_bench::provenance()})
    );
  }
  bench_config::run(&[
    sha224, sha256, sha384, sha512, sha512_256, sha3_224, sha3_256, sha3_384, sha3_512, aes128, aes256,
  ]);
}
