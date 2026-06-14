#![cfg(all(feature = "rsa", feature = "diag", feature = "getrandom"))]

use core::hint::black_box;
use std::time::Instant;

use rscrypto::{
  RsaOaepProfile, RsaPkcs1v15Profile, RsaPrivateKey, RsaPssProfile, RsaPublicKeyPolicy,
  auth::rsa::diag_rsa_blinding_factor_inverse,
};
use serde_json::Value;

const OAEP_SHA256: &str = include_str!("../testdata/rsa/wycheproof/rsa_oaep_2048_sha256_mgf1sha256_test.json");

#[derive(Clone, Copy)]
struct XorShift64 {
  state: u64,
}

impl XorShift64 {
  const fn new(seed: u64) -> Self {
    Self { state: seed }
  }

  fn next_u64(&mut self) -> u64 {
    let mut x = self.state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    self.state = x;
    x
  }

  fn fill(&mut self, out: &mut [u8]) {
    for chunk in out.chunks_mut(8) {
      let word = self.next_u64().to_le_bytes();
      let len = chunk.len();
      chunk.copy_from_slice(&word[..len]);
    }
  }
}

#[derive(Default)]
struct OnlineStats {
  n: usize,
  mean: f64,
  m2: f64,
}

impl OnlineStats {
  fn push(&mut self, value: f64) {
    self.n += 1;
    let delta = value - self.mean;
    self.mean += delta / self.n as f64;
    let delta2 = value - self.mean;
    self.m2 += delta * delta2;
  }

  fn variance(&self) -> f64 {
    if self.n > 1 { self.m2 / (self.n - 1) as f64 } else { 0.0 }
  }
}

fn welch_t(left: &OnlineStats, right: &OnlineStats) -> f64 {
  let standard_error = (left.variance() / left.n as f64 + right.variance() / right.n as f64).sqrt();
  if standard_error == 0.0 {
    if left.mean == right.mean {
      0.0
    } else {
      f64::INFINITY.copysign(left.mean - right.mean)
    }
  } else {
    (left.mean - right.mean) / standard_error
  }
}

fn env_usize(name: &str, default: usize) -> usize {
  std::env::var(name)
    .ok()
    .and_then(|value| value.parse().ok())
    .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
  std::env::var(name)
    .ok()
    .and_then(|value| value.parse().ok())
    .unwrap_or(default)
}

fn measure_case<F>(name: &str, samples_per_class: usize, warmup: usize, threshold: f64, mut operation: F)
where
  F: FnMut(bool, usize),
{
  for index in 0..warmup {
    operation(false, index);
    operation(true, index);
  }

  let mut rng = XorShift64::new(0x7261_6e64_6f6d_697a);
  let mut fixed_remaining = samples_per_class;
  let mut random_remaining = samples_per_class;
  let mut fixed_index = 0usize;
  let mut random_index = 0usize;
  let mut fixed = OnlineStats::default();
  let mut random = OnlineStats::default();

  while fixed_remaining != 0 || random_remaining != 0 {
    let use_random = if fixed_remaining == 0 {
      true
    } else if random_remaining == 0 {
      false
    } else {
      rng.next_u64() & 1 == 1
    };
    let index = if use_random {
      let index = random_index;
      random_index += 1;
      random_remaining -= 1;
      index
    } else {
      let index = fixed_index;
      fixed_index += 1;
      fixed_remaining -= 1;
      index
    };

    let start = Instant::now();
    operation(use_random, index);
    let elapsed = start.elapsed().as_nanos() as f64;
    if use_random {
      random.push(elapsed);
    } else {
      fixed.push(elapsed);
    }
  }

  let t = welch_t(&fixed, &random);
  println!(
    "{{\"case\":\"{name}\",\"fixed_n\":{},\"random_n\":{},\"fixed_mean_ns\":{:.1},\"random_mean_ns\":{:.1},\"welch_t\"\
     :{:.4},\"threshold\":{:.4}}}",
    fixed.n, random.n, fixed.mean, random.mean, t, threshold
  );
  assert!(
    t.abs() < threshold,
    "RSA leakage case `{name}` exceeded Welch t threshold: |{t:.4}| >= {threshold:.4}"
  );
}

fn hex_to_vec(hex: &str) -> Vec<u8> {
  assert_eq!(hex.len() % 2, 0);
  let mut out = Vec::with_capacity(hex.len() / 2);
  for chunk in hex.as_bytes().chunks_exact(2) {
    out.push((hex_value(chunk[0]) << 4) | hex_value(chunk[1]));
  }
  out
}

fn hex_value(byte: u8) -> u8 {
  match byte {
    b'0'..=b'9' => byte - b'0',
    b'a'..=b'f' => byte - b'a' + 10,
    b'A'..=b'F' => byte - b'A' + 10,
    _ => panic!("invalid hex digit"),
  }
}

fn legacy_rsa2048_fixture_key() -> RsaPrivateKey {
  let suite: Value = serde_json::from_str(OAEP_SHA256).expect("Wycheproof OAEP JSON must parse");
  let group = suite["testGroups"]
    .as_array()
    .and_then(|groups| groups.first())
    .expect("Wycheproof OAEP group must exist");
  let der = hex_to_vec(group["privateKeyPkcs8"].as_str().expect("privateKeyPkcs8 must exist"));
  let key = RsaPrivateKey::from_pkcs8_der_with_policy(&der, &RsaPublicKeyPolicy::legacy_verification())
    .expect("Wycheproof RSA-2048 private key must parse with explicit legacy policy");
  assert_eq!(key.public_key().modulus_bits(), 2048);
  key
}

#[test]
fn rsa_leakage_fixture_uses_explicit_legacy_rsa2048_policy() {
  let key = legacy_rsa2048_fixture_key();
  assert_eq!(key.public_key().public_exponent().as_u64(), 65_537);
}

fn factor_two_and_inverse(modulus: &[u8]) -> (Vec<u8>, Vec<u8>) {
  let mut factor = vec![0u8; modulus.len()];
  if let Some(last) = factor.last_mut() {
    *last = 2;
  }

  let mut plus_one = modulus.to_vec();
  let mut carry = 1u16;
  for byte in plus_one.iter_mut().rev() {
    let sum = u16::from(*byte) + carry;
    *byte = sum as u8;
    carry = sum >> 8;
    if carry == 0 {
      break;
    }
  }
  if carry != 0 {
    plus_one.insert(0, carry as u8);
  }

  let mut quotient = Vec::with_capacity(plus_one.len());
  let mut remainder = 0u16;
  for byte in plus_one {
    let value = (remainder << 8) | u16::from(byte);
    quotient.push((value / 2) as u8);
    remainder = value % 2;
  }
  let first_nonzero = quotient
    .iter()
    .position(|&byte| byte != 0)
    .unwrap_or(quotient.len() - 1);
  let inverse = &quotient[first_nonzero..];

  let mut inverse_fixed = vec![0u8; modulus.len()];
  inverse_fixed[modulus.len() - inverse.len()..].copy_from_slice(inverse);
  (factor, inverse_fixed)
}

fn fixed_and_random_messages(len: usize, count: usize) -> (Vec<u8>, Vec<Vec<u8>>) {
  let fixed = vec![0x42; len];
  let mut rng = XorShift64::new(0x6d65_7373_6167_6573);
  let mut random = Vec::with_capacity(count);
  for _ in 0..count {
    let mut message = vec![0u8; len];
    rng.fill(&mut message);
    random.push(message);
  }
  (fixed, random)
}

fn fixed_and_random_oaep_ciphertexts(key: &RsaPrivateKey, count: usize) -> (Vec<u8>, Vec<Vec<u8>>) {
  let len = key.public_key().modulus().len();
  let mut fixed = vec![0u8; len];
  let fixed_seed = vec![0x11; RsaOaepProfile::Sha256.digest_len()];
  key
    .public_key()
    .diag_encrypt_oaep_with_seed(
      RsaOaepProfile::Sha256,
      b"leakage-label",
      b"fixed oaep leakage text",
      &fixed_seed,
      &mut fixed,
    )
    .expect("fixed OAEP ciphertext must encrypt");

  let mut rng = XorShift64::new(0x6f61_6570_6374_7874);
  let mut random = Vec::with_capacity(count);
  for _ in 0..count {
    let mut seed = vec![0u8; RsaOaepProfile::Sha256.digest_len()];
    let mut plaintext = vec![0u8; b"fixed oaep leakage text".len()];
    rng.fill(&mut seed);
    rng.fill(&mut plaintext);
    let mut ciphertext = vec![0u8; len];
    key
      .public_key()
      .diag_encrypt_oaep_with_seed(
        RsaOaepProfile::Sha256,
        b"leakage-label",
        &plaintext,
        &seed,
        &mut ciphertext,
      )
      .expect("random OAEP ciphertext must encrypt");
    random.push(ciphertext);
  }
  (fixed, random)
}

fn fixed_and_random_pkcs1v15_ciphertexts(key: &RsaPrivateKey, count: usize) -> (Vec<u8>, Vec<Vec<u8>>) {
  let len = key.public_key().modulus().len();
  let plaintext_len = b"fixed pkcs1v15 leakage text".len();
  let padding_len = len - plaintext_len - 3;
  let fixed_seed = vec![0x5a; padding_len];
  let mut fixed = vec![0u8; len];
  key
    .public_key()
    .diag_encrypt_pkcs1v15_with_seed(b"fixed pkcs1v15 leakage text", &fixed_seed, &mut fixed)
    .expect("fixed PKCS1v1.5 ciphertext must encrypt");

  let mut rng = XorShift64::new(0x706b_6373_6374_7874);
  let mut random = Vec::with_capacity(count);
  for _ in 0..count {
    let mut seed = vec![0u8; padding_len];
    for byte in &mut seed {
      while *byte == 0 {
        *byte = rng.next_u64() as u8;
      }
    }
    let mut plaintext = vec![0u8; plaintext_len];
    rng.fill(&mut plaintext);
    let mut ciphertext = vec![0u8; len];
    key
      .public_key()
      .diag_encrypt_pkcs1v15_with_seed(&plaintext, &seed, &mut ciphertext)
      .expect("random PKCS1v1.5 ciphertext must encrypt");
    random.push(ciphertext);
  }
  (fixed, random)
}

fn random_blinding_factors(key: &RsaPrivateKey, count: usize) -> Vec<Vec<u8>> {
  let len = key.public_key().modulus().len();
  let mut rng = XorShift64::new(0x626c_696e_6469_6e67);
  let mut factors = Vec::with_capacity(count);
  while factors.len() < count {
    let mut factor = vec![0u8; len];
    rng.fill(&mut factor);
    factor[0] &= 0x7f;
    if let Some(last) = factor.last_mut() {
      *last |= 1;
    }
    if factor.iter().all(|&byte| byte == 0) {
      continue;
    }
    let mut inverse = vec![0u8; len];
    if diag_rsa_blinding_factor_inverse(key, &factor, &mut inverse).is_ok() {
      factors.push(factor);
    }
  }
  factors
}

#[test]
#[ignore = "release-only statistical timing gate; run scripts/test/test-rsa-leakage.sh"]
fn rsa_private_operations_do_not_show_first_order_timing_leakage() {
  let samples = env_usize("RSCRYPTO_RSA_LEAKAGE_SAMPLES", 2000);
  let warmup = env_usize("RSCRYPTO_RSA_LEAKAGE_WARMUP", 64);
  let threshold = env_f64("RSCRYPTO_RSA_LEAKAGE_T_THRESHOLD", 8.0);
  assert!(samples >= 4, "leakage gate needs at least four samples per class");
  let input_pool = samples.max(warmup).min(256);

  let key = legacy_rsa2048_fixture_key();
  let len = key.public_key().modulus().len();
  let (blinding_factor, blinding_inverse) = factor_two_and_inverse(key.public_key().modulus());
  let (fixed_message, random_messages) = fixed_and_random_messages(64, input_pool);
  let pss_salt = vec![0x33; RsaPssProfile::Sha256.digest_len()];
  let (fixed_oaep, random_oaep) = fixed_and_random_oaep_ciphertexts(&key, input_pool);
  let (fixed_pkcs1v15, random_pkcs1v15) = fixed_and_random_pkcs1v15_ciphertexts(&key, input_pool);
  let random_factors = random_blinding_factors(&key, input_pool);
  let fixed_factor = random_factors[0].clone();

  measure_case(
    "rsa2048_pkcs1v15_sign_fixed_blinding",
    samples,
    warmup,
    threshold,
    |use_random, index| {
      let message = if use_random {
        &random_messages[index % random_messages.len()]
      } else {
        &fixed_message
      };
      let mut out = vec![0u8; len];
      let mut scratch = key.private_scratch();
      key
        .sign_pkcs1v15_with_blinding_factor_and_scratch(
          RsaPkcs1v15Profile::Sha256,
          black_box(message),
          &blinding_factor,
          &blinding_inverse,
          black_box(&mut out),
          &mut scratch,
        )
        .expect("PKCS1v1.5 signing must succeed");
      black_box(out[0]);
    },
  );

  measure_case(
    "rsa2048_pss_sign_fixed_salt_and_blinding",
    samples,
    warmup,
    threshold,
    |use_random, index| {
      let message = if use_random {
        &random_messages[index % random_messages.len()]
      } else {
        &fixed_message
      };
      let mut out = vec![0u8; len];
      let mut scratch = key.private_scratch();
      key
        .sign_pss_with_salt_and_blinding_factor_and_scratch(
          RsaPssProfile::Sha256,
          black_box(message),
          &pss_salt,
          &blinding_factor,
          &blinding_inverse,
          black_box(&mut out),
          &mut scratch,
        )
        .expect("PSS signing must succeed");
      black_box(out[0]);
    },
  );

  measure_case(
    "rsa2048_oaep_decrypt_fixed_blinding",
    samples,
    warmup,
    threshold,
    |use_random, index| {
      let ciphertext = if use_random {
        &random_oaep[index % random_oaep.len()]
      } else {
        &fixed_oaep
      };
      let mut out = vec![0u8; len];
      let mut scratch = key.private_scratch();
      let plaintext_len = key
        .decrypt_oaep_with_blinding_factor_and_scratch(
          RsaOaepProfile::Sha256,
          b"leakage-label",
          black_box(ciphertext),
          &blinding_factor,
          &blinding_inverse,
          black_box(&mut out),
          &mut scratch,
        )
        .expect("OAEP decryption must succeed");
      black_box(plaintext_len);
    },
  );

  measure_case(
    "rsa2048_pkcs1v15_decrypt_fixed_blinding",
    samples,
    warmup,
    threshold,
    |use_random, index| {
      let ciphertext = if use_random {
        &random_pkcs1v15[index % random_pkcs1v15.len()]
      } else {
        &fixed_pkcs1v15
      };
      let mut out = vec![0u8; len];
      let mut scratch = key.private_scratch();
      let plaintext_len = key
        .decrypt_pkcs1v15_with_blinding_factor_and_scratch(
          black_box(ciphertext),
          &blinding_factor,
          &blinding_inverse,
          black_box(&mut out),
          &mut scratch,
        )
        .expect("PKCS1v1.5 decryption must succeed");
      black_box(plaintext_len);
    },
  );

  measure_case(
    "rsa2048_blinding_inverse",
    samples,
    warmup,
    threshold,
    |use_random, index| {
      let factor = if use_random {
        &random_factors[index % random_factors.len()]
      } else {
        &fixed_factor
      };
      let mut inverse = vec![0u8; len];
      diag_rsa_blinding_factor_inverse(&key, black_box(factor), black_box(&mut inverse))
        .expect("blinding inverse must derive");
      black_box(inverse[0]);
    },
  );

  measure_case(
    "rsa2048_pkcs1v15_sign_os_blinding",
    samples,
    warmup,
    threshold,
    |use_random, index| {
      let message = if use_random {
        &random_messages[index % random_messages.len()]
      } else {
        &fixed_message
      };
      let mut out = vec![0u8; len];
      let mut scratch = key.private_scratch();
      key
        .sign_pkcs1v15_with_scratch(
          RsaPkcs1v15Profile::Sha256,
          black_box(message),
          black_box(&mut out),
          &mut scratch,
        )
        .expect("OS-blinded PKCS1v1.5 signing must succeed");
      black_box(out[0]);
    },
  );
}
