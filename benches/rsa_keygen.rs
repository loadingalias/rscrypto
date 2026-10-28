//! RSA key-generation benchmarks for rscrypto's public API, RustCrypto, and AWS-LC.
//!
//! Each group generates a key pair of one modulus size with public exponent 65537.
//! Before timing, every row checks its key's size and exponent and signs a message
//! that a different library verifies.

#[path = "common/criterion.rs"]
mod bench_config;

#[path = "common/aws_lc.rs"]
mod aws_lc_support;
use aws_lc_support::aws_lc_bench;

use core::hint::black_box;

aws_lc_bench! {
  use aws_lc_rs::{rsa as aws_rsa, signature as aws_signature};
}
use bench_config::checked;
use criterion::Criterion;
use rsa::{
  RsaPrivateKey as RustCryptoRsaPrivateKey, RsaPublicKey as RustCryptoRsaPublicKey,
  pkcs1::EncodeRsaPublicKey as _,
  pkcs1v15::{
    Signature as RustCryptoPkcs1v15Signature, SigningKey as RustCryptoPkcs1v15SigningKey,
    VerifyingKey as RustCryptoPkcs1v15VerifyingKey,
  },
  pkcs8::DecodePublicKey as _,
  rand_core::{CryptoRng as RustCryptoCryptoRng, Error as RustCryptoRngError, RngCore as RustCryptoRngCore},
  signature::{SignatureEncoding as _, Signer as _, Verifier as _},
};
use rscrypto::{RsaPkcs1v15Profile, RsaPrivateKey, RsaPublicKey, RsaPublicKeyPolicy};

const PRIVATE_SIGNING_MESSAGE: &[u8] = b"rscrypto TLS-shaped RSA private signing benchmark";

/// Deterministic SHA-256 counter-mode byte stream.
///
/// Fixture input, not an entropy source. It gives RustCrypto key generation a
/// reproducible stream.
struct SeededRng {
  seed: [u8; 32],
  counter: u64,
  block: [u8; 32],
  used: usize,
}

impl SeededRng {
  fn new(label: &[u8]) -> Self {
    Self {
      seed: <sha2_010::Sha256 as sha2_010::Digest>::digest(label).into(),
      counter: 0,
      block: [0; 32],
      used: 32,
    }
  }
}

impl RustCryptoRngCore for SeededRng {
  fn next_u32(&mut self) -> u32 {
    rsa::rand_core::impls::next_u32_via_fill(self)
  }

  fn next_u64(&mut self) -> u64 {
    rsa::rand_core::impls::next_u64_via_fill(self)
  }

  fn fill_bytes(&mut self, dest: &mut [u8]) {
    for byte in dest {
      if self.used == self.block.len() {
        let mut hasher = <sha2_010::Sha256 as sha2_010::Digest>::new();
        sha2_010::Digest::update(&mut hasher, self.seed);
        sha2_010::Digest::update(&mut hasher, self.counter.to_be_bytes());
        self.block = sha2_010::Digest::finalize(hasher).into();
        self.counter = self.counter.strict_add(1);
        self.used = 0;
      }
      *byte = self.block[self.used];
      self.used = self.used.strict_add(1);
    }
  }

  fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), RustCryptoRngError> {
    self.fill_bytes(dest);
    Ok(())
  }
}

impl RustCryptoCryptoRng for SeededRng {}

/// RSA-2048 keys need the legacy policy; larger keys use the modern default.
fn key_policy(bits: usize) -> RsaPublicKeyPolicy {
  if bits < 3072 {
    RsaPublicKeyPolicy::modulus_2048_to_8192()
  } else {
    RsaPublicKeyPolicy::modulus_3072_to_8192()
  }
}

fn rsa_keygen(c: &mut Criterion) {
  for bits in [2048, 3072, 4096] {
    rsa_keygen_for_size(c, bits);
  }
}

/// Check a generated public key's size and exponent with rscrypto, and verify
/// its RSA-PKCS1-v1_5-SHA256 signature over [`PRIVATE_SIGNING_MESSAGE`].
fn assert_generated_public_key(bits: usize, public_key_pkcs1: &[u8], signature: &[u8]) {
  let key = RsaPublicKey::from_pkcs1_der_with_policy(public_key_pkcs1, &key_policy(bits))
    .expect("rscrypto must import the generated RSA public key");
  assert_eq!(key.modulus_bits(), bits, "generated RSA modulus size");
  assert_eq!(key.public_exponent().as_u64(), 65_537, "generated RSA public exponent");
  key
    .verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, PRIVATE_SIGNING_MESSAGE, signature)
    .expect("rscrypto must verify a signature from the generated RSA key");
}

/// Key generation with public exponent 65537: one group per size, with one row
/// per library.
///
/// Prime search makes the cost of each key vary widely. Flat sampling runs the
/// same number of generations in every sample, so each estimate is a mean over
/// at least `sample_size` keys; read its spread before comparing rows. The
/// rscrypto and aws-lc-rs APIs take no RNG: rscrypto draws from `getrandom`
/// and aws-lc-rs from AWS-LC's DRBG, so their keys differ between runs.
/// RustCrypto draws from one [`SeededRng`] stream, so runs with the same
/// iteration counts generate the same keys. Timing includes each library's key
/// checks, CRT precomputation, and destruction. RSA-2048 is a legacy size for
/// rscrypto, so every size calls `generate_with_policy`;
/// `RsaPrivateKey::generate` is that call with the modern policy. ring has no
/// RSA key generation API.
///
/// Untimed checks: each row's key has the requested size and exponent, and
/// signs a message that a different library verifies.
fn rsa_keygen_for_size(c: &mut Criterion, bits: usize) {
  let name = format!("rsa-{bits}-keygen");
  if !bench_config::selected(&format!("{name}/")) {
    return;
  }
  let policy = key_policy(bits);
  let mut rustcrypto_rng = SeededRng::new(name.as_bytes());
  aws_lc_bench! {
    use aws_signature::KeyPair as _;

    let aws_size = [aws_rsa::KeySize::Rsa2048, aws_rsa::KeySize::Rsa3072, aws_rsa::KeySize::Rsa4096]
      .into_iter()
      .find(|size| size.len() == bits / 8)
      .expect("aws-lc-rs must support the RSA key size");
  }

  let mut group = c.benchmark_group(name);
  group.sampling_mode(criterion::SamplingMode::Flat);
  group.bench_function(
    "rscrypto",
    checked(
      || {
        black_box(
          RsaPrivateKey::generate_with_policy(black_box(bits), &policy)
            .expect("rscrypto RSA key generation must succeed"),
        )
      },
      |key| {
        assert_eq!(key.public_key().modulus_bits(), bits, "rscrypto RSA modulus size");
        assert_eq!(
          key.public_key().public_exponent().as_u64(),
          65_537,
          "rscrypto RSA public exponent"
        );
        let mut signature = vec![0u8; key.signature_len()];
        key
          .sign_pkcs1v15(RsaPkcs1v15Profile::Sha256, PRIVATE_SIGNING_MESSAGE, &mut signature)
          .expect("a generated rscrypto RSA key must sign");
        RustCryptoPkcs1v15VerifyingKey::<sha2_010::Sha256>::new(
          RustCryptoRsaPublicKey::from_public_key_der(&key.public_key().to_spki_der())
            .expect("RustCrypto must import the generated rscrypto RSA public key"),
        )
        .verify(
          PRIVATE_SIGNING_MESSAGE,
          &RustCryptoPkcs1v15Signature::try_from(signature.as_slice())
            .expect("RSA signature must have the modulus length"),
        )
        .expect("RustCrypto must verify a signature from a generated rscrypto RSA key");
      },
    ),
  );
  group.bench_function(
    "rustcrypto-rsa",
    checked(
      || {
        black_box(
          RustCryptoRsaPrivateKey::new(&mut rustcrypto_rng, black_box(bits))
            .expect("RustCrypto RSA key generation must succeed"),
        )
      },
      |key| {
        let public_key = key
          .to_public_key()
          .to_pkcs1_der()
          .expect("RustCrypto must encode the generated RSA public key");
        let signature = RustCryptoPkcs1v15SigningKey::<sha2_010::Sha256>::new(key)
          .try_sign(PRIVATE_SIGNING_MESSAGE)
          .expect("a generated RustCrypto RSA key must sign");
        assert_generated_public_key(bits, public_key.as_bytes(), &signature.to_vec());
      },
    ),
  );
  aws_lc_bench! {
    group.bench_function(
      "aws-lc-rs",
      checked(
        || {
          black_box(aws_rsa::KeyPair::generate(black_box(aws_size)).expect("aws-lc-rs RSA key generation must succeed"))
        },
        |aws_key| {
          let mut signature = vec![0u8; aws_key.public_modulus_len()];
          aws_key
            .sign(
              &aws_signature::RSA_PKCS1_SHA256,
              &aws_lc_rs::rand::SystemRandom::new(),
              PRIVATE_SIGNING_MESSAGE,
              &mut signature,
            )
            .expect("a generated aws-lc-rs RSA key must sign");
          assert_generated_public_key(bits, aws_key.public_key().as_ref(), &signature);
        },
      ),
    );
  }
  group.finish();
}

fn main() {
  bench_config::run(&[rsa_keygen]);
}
