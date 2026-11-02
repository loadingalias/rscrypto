//! Production provider key-schedule costs, separate from connections and records.

use core::hint::black_box;

use criterion::Criterion;
use openssl::{hash::MessageDigest, pkey::PKey, sign::Signer};
use rustls::crypto::{CipherSuite, hmac, tls12, tls13};

use super::bench_config::{checked_routine, selected};

fn hmac_oracle(digest: MessageDigest, key: &[u8], parts: &[&[u8]]) -> Vec<u8> {
  assert!(
    openssl::version::version().starts_with("OpenSSL 4.0.3 "),
    "benchmark oracle requires OpenSSL 4.0.3"
  );
  let key = PKey::hmac(key).expect("fixture HMAC key");
  let mut signer = Signer::new(digest, &key).expect("independent HMAC");
  for part in parts {
    signer.update(part).expect("independent HMAC input");
  }
  signer.sign_to_vec().expect("independent HMAC output")
}

#[expect(clippy::unreachable, reason = "the component rows select only the three declared PRF backends")]
fn prf(backend: &str, suite: CipherSuite, key_block: usize) -> &'static dyn tls12::Prf {
  let suites = match backend {
    "rscrypto" => rscrypto_rustls::cipher_suite::ALL_TLS12_CIPHER_SUITES,
    "aws-lc-rs" => rustls_aws_lc_rs::ALL_TLS12_CIPHER_SUITES,
    "ring" => rustls_ring::ALL_TLS12_CIPHER_SUITES,
    _ => unreachable!("declared key-schedule backends"),
  };
  let selected = suites
    .iter()
    .find(|entry| entry.common.suite == suite)
    .expect("selected suite");
  let shape = selected.aead_alg.key_block_shape();
  // Match rustls ConnectionSecrets::make_key_block, including its extra
  // explicit-nonce seed. This is 48/80 bytes for AES-GCM, rather than 40/72.
  assert_eq!(
    key_block,
    shape
      .enc_key_len
      .strict_add(shape.fixed_iv_len)
      .strict_mul(2)
      .strict_add(shape.explicit_nonce_len),
    "production key-block length"
  );
  selected.prf_provider
}

pub(super) fn key_schedule(c: &mut Criterion) {
  // Each PRF uses a master secret prepared once, as the production connection
  // does. Label/seed construction and output capacity are outside timing.
  for (cipher, suite, key_block, hash_len) in [
    (
      "AES128-GCM-SHA256",
      CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
      48,
      32,
    ),
    (
      "AES256-GCM-SHA384",
      CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
      80,
      48,
    ),
    (
      "CHACHA20-POLY1305-SHA256",
      CipherSuite::TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
      88,
      32,
    ),
  ] {
    for (operation, label, seed_len, output_len) in [
      ("key-expansion", b"key expansion".as_slice(), 64, key_block),
      ("finished", b"client finished".as_slice(), hash_len, 12),
    ] {
      let name = format!("tls12/prf/{cipher}/{operation}");
      if !selected(&name) {
        continue;
      }
      let mut group = c.benchmark_group(&name);
      for backend in ["rscrypto", "aws-lc-rs", "ring"] {
        if !selected(&format!("{name}/{backend}")) {
          continue;
        }
        group.bench_function(
          backend,
          checked_routine(
            (None, vec![0; output_len]),
            |(secret, output)| {
              let actual = prf(backend, suite, key_block).new_secret(&[0x42; 48]);
              let oracle = prf(
                if backend == "aws-lc-rs" { "ring" } else { "aws-lc-rs" },
                suite,
                key_block,
              )
              .new_secret(&[0x42; 48]);
              let mut expected = vec![0; output_len];
              for fill in [0x17, 0x92] {
                actual.prf(output, label, &[fill; 64][..seed_len]);
                oracle.prf(&mut expected, label, &[fill; 64][..seed_len]);
                assert_eq!(*output, expected, "independent PRF and reusable secret");
              }
              *secret = Some(actual);
            },
            |b, (secret, output)| {
              let secret = secret.as_ref().expect("checked PRF secret");
              b.iter(|| {
                secret.prf(black_box(output), black_box(label), black_box(&[0x17; 64][..seed_len]));
                black_box(&*output);
              });
            },
          ),
        );
      }
      group.finish();
    }
  }

  for (hash, algorithm, hkdf, digest) in [
    (
      "sha256",
      rscrypto_rustls::hash::HMAC_SHA256,
      rscrypto_rustls::hash::HKDF_SHA256,
      MessageDigest::sha256(),
    ),
    (
      "sha384",
      rscrypto_rustls::hash::HMAC_SHA384,
      rscrypto_rustls::hash::HKDF_SHA384,
      MessageDigest::sha384(),
    ),
  ] {
    let hash_len = algorithm.hash_output_len();
    for import in [false, true] {
      let mode = if import { "import-key" } else { "reuse-key" };
      let name = format!("tls/hmac/{hash}/{mode}");
      if !selected(&name) || !selected(&format!("{name}/rscrypto")) {
        continue;
      }
      let mut group = c.benchmark_group(&name);
      // The input shape is one TLS 1.2 P_hash output term: A(i), label and seed.
      // Both rows include the same sign_concat and clearing Tag destruction;
      // only import-key also constructs and destroys the boxed provider key.
      group.bench_function(
        "rscrypto",
        checked_routine(
          None::<Box<dyn hmac::Key>>,
          |key| {
            let prepared = algorithm.with_key(&[0x42; 48]);
            for fill in [0x17, 0x92] {
              let first = &[fill; 48][..hash_len];
              let tag = prepared.sign_concat(first, &[b"key expansion", &[0x92; 64]], &[]);
              assert_eq!(
                tag.as_ref(),
                hmac_oracle(digest, &[0x42; 48], &[first, b"key expansion", &[0x92; 64]])
              );
            }
            *key = Some(prepared);
          },
          |b, key| {
            b.iter(|| {
              let imported;
              let key = if import {
                imported = algorithm.with_key(black_box(&[0x42; 48]));
                &imported
              } else {
                key.as_ref().expect("checked HMAC key")
              };
              black_box(key.sign_concat(
                black_box(&[0x17; 48][..hash_len]),
                black_box(&[b"key expansion", &[0x92; 64]]),
                &[],
              ))
            });
          },
        ),
      );
      group.finish();
    }

    let name = format!("tls13/finished-hmac/{hash}");
    if !selected(&name) || !selected(&format!("{name}/rscrypto")) {
      continue;
    }
    let mut group = c.benchmark_group(&name);
    group.bench_function(
      "rscrypto",
      checked_routine(
        None,
        |key| {
          let prepared = tls13::OkmBlock::new(&[0x42; 48][..hash_len]);
          for fill in [0x17, 0x92] {
            let message = &[fill; 48][..hash_len];
            assert_eq!(
              hkdf.hmac_sign(&prepared, message).as_ref(),
              hmac_oracle(digest, prepared.as_ref(), &[message])
            );
          }
          *key = Some(prepared);
        },
        |b, key| {
          b.iter(|| {
            black_box(hkdf.hmac_sign(
              key.as_ref().expect("checked Finished key"),
              black_box(&[0x17; 48][..hash_len]),
            ))
          })
        },
      ),
    );
    group.finish();
  }
}
