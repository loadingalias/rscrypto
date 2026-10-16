//! SP 800-185 and HMAC-SHA3 comparison benchmarks.
//!
//! Every group runs the shared comparison size matrix and times one complete
//! output per iteration:
//!
//! - `kmac128`, `kmac256`: one-shot KMAC with a fixed key and customization
//!   string and a 32-byte output. Key absorption is timed. rscrypto vs
//!   tiny-keccak.
//! - `cshake128`, `cshake256`: one-shot cSHAKE with an empty function name, a
//!   fixed customization string, and a 32-byte output. rscrypto vs
//!   tiny-keccak.
//! - `hmac-sha3-224` through `hmac-sha3-512`: HMAC-SHA3 with a key as long as
//!   the tag. Keying happens once outside timing. The timed work is message
//!   absorption, finalization, and state reuse: rscrypto resets its keyed
//!   state, and RustCrypto `hmac::SimpleHmac` over `sha3` clones one.
//!   `hmac::Hmac` needs a block-level hash core that `sha3` 0.12 does not
//!   expose. aws-lc-rs and ring do not offer HMAC-SHA3.
//!
//! Before registering timed rows, each group checks outside timing that
//! rscrypto reproduces a published vector and matches the competitor at every
//! size.

#[path = "common/criterion.rs"]
mod bench_config;

mod common;

use core::{fmt::Write as _, hint::black_box};

use criterion::{BenchmarkId, Criterion};
use hmac::{KeyInit, SimpleHmac};
use rscrypto::{HmacSha3_224, HmacSha3_256, HmacSha3_384, HmacSha3_512, Mac};

const KMAC_KEY: &[u8] = b"rscrypto-kmac-bench-key-32bytes";
const CUSTOMIZATION: &[u8] = b"rscrypto-bench";
const FUNCTION_NAME: &[u8] = b"";
const OUTPUT_LEN: usize = 32;

/// Message of the NIST SP 800-185 KMAC and cSHAKE samples checked here.
const SP800_185_SAMPLE_DATA: &[u8] = &[0x00, 0x01, 0x02, 0x03];
/// Customization string of the NIST SP 800-185 cSHAKE samples checked here.
const CSHAKE_SAMPLE_CUSTOMIZATION: &[u8] = b"Email Signature";
/// Message of NIST HMAC-SHA3 example 1, whose key is `0x00, 0x01, ...` and as
/// long as the tag.
const HMAC_SHA3_SAMPLE_MESSAGE: &[u8] = b"Sample message for keylen<blocklen";

/// Key `0x40..=0x5f` of the NIST SP 800-185 KMAC samples.
fn kmac_sample_key() -> Vec<u8> {
  (0x40u8..=0x5f).collect()
}

fn hex(bytes: &[u8]) -> String {
  let mut out = String::with_capacity(bytes.len().strict_mul(2));
  for byte in bytes {
    write!(out, "{byte:02x}").expect("writing to a String must succeed");
  }
  out
}

/// Registers the rscrypto and tiny-keccak rows of one SP 800-185 group after
/// checking outside timing that both rows produce the same output at every
/// size. `key_or_name` is the KMAC key or the cSHAKE function name.
fn sp800_185_group(
  c: &mut Criterion,
  group: &str,
  key_or_name: &[u8],
  rscrypto: impl Fn(&[u8], &[u8], &[u8]) -> [u8; OUTPUT_LEN],
  tiny_keccak: impl Fn(&[u8], &[u8], &[u8]) -> [u8; OUTPUT_LEN],
) {
  let inputs = common::comp_sizes();
  for (len, data) in &inputs {
    assert_eq!(
      rscrypto(key_or_name, CUSTOMIZATION, data),
      tiny_keccak(key_or_name, CUSTOMIZATION, data),
      "{group}/{len}: rscrypto output must match tiny-keccak"
    );
  }

  let mut g = c.benchmark_group(group);

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(rscrypto(black_box(key_or_name), black_box(CUSTOMIZATION), black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("tiny-keccak", len), data, |b, d| {
      b.iter(|| {
        black_box(tiny_keccak(
          black_box(key_or_name),
          black_box(CUSTOMIZATION),
          black_box(d),
        ))
      })
    });
  }

  g.finish();
}

fn kmac128(c: &mut Criterion) {
  if !bench_config::selected("kmac128") {
    return;
  }
  let sample = rscrypto::Kmac128::mac_array::<32>(&kmac_sample_key(), b"", SP800_185_SAMPLE_DATA);
  assert_eq!(
    hex(&sample),
    "e5780b0d3ea6f7d3a429c5706aa43a00fadbd7d49628839e3187243f456ee14e",
    "kmac128 must reproduce NIST SP 800-185 KMAC sample 1"
  );

  sp800_185_group(
    c,
    "kmac128",
    KMAC_KEY,
    rscrypto::Kmac128::mac_array,
    |key, customization, data| {
      use tiny_keccak::Hasher as _;

      let mut out = [0u8; OUTPUT_LEN];
      let mut kmac = tiny_keccak::Kmac::v128(key, customization);
      kmac.update(data);
      kmac.finalize(&mut out);
      out
    },
  );
}

fn kmac256(c: &mut Criterion) {
  if !bench_config::selected("kmac256") {
    return;
  }
  let sample = rscrypto::Kmac256::mac_array::<64>(&kmac_sample_key(), b"My Tagged Application", SP800_185_SAMPLE_DATA);
  assert_eq!(
    hex(&sample),
    concat!(
      "20c570c31346f703c9ac36c61c03cb64c3970d0cfc787e9b79599d273a68d2f7",
      "f69d4cc3de9d104a351689f27cf6f5951f0103f33f4f24871024d9c27773a8dd"
    ),
    "kmac256 must reproduce NIST SP 800-185 KMAC sample 4"
  );

  sp800_185_group(
    c,
    "kmac256",
    KMAC_KEY,
    rscrypto::Kmac256::mac_array,
    |key, customization, data| {
      use tiny_keccak::Hasher as _;

      let mut out = [0u8; OUTPUT_LEN];
      let mut kmac = tiny_keccak::Kmac::v256(key, customization);
      kmac.update(data);
      kmac.finalize(&mut out);
      out
    },
  );
}

fn cshake128(c: &mut Criterion) {
  if !bench_config::selected("cshake128") {
    return;
  }
  let mut sample = [0u8; 32];
  rscrypto::Cshake128::hash_into(b"", CSHAKE_SAMPLE_CUSTOMIZATION, SP800_185_SAMPLE_DATA, &mut sample);
  assert_eq!(
    hex(&sample),
    "c1c36925b6409a04f1b504fcbca9d82b4017277cb5ed2b2065fc1d3814d5aaf5",
    "cshake128 must reproduce NIST SP 800-185 cSHAKE sample 1"
  );

  sp800_185_group(
    c,
    "cshake128",
    FUNCTION_NAME,
    |function_name, customization, data| {
      let mut out = [0u8; OUTPUT_LEN];
      rscrypto::Cshake128::hash_into(function_name, customization, data, &mut out);
      out
    },
    |function_name, customization, data| {
      use tiny_keccak::Hasher as _;

      let mut out = [0u8; OUTPUT_LEN];
      let mut cshake = tiny_keccak::CShake::v128(function_name, customization);
      cshake.update(data);
      cshake.finalize(&mut out);
      out
    },
  );
}

fn cshake256(c: &mut Criterion) {
  if !bench_config::selected("cshake256") {
    return;
  }
  let mut sample = [0u8; 64];
  rscrypto::Cshake256::hash_into(b"", CSHAKE_SAMPLE_CUSTOMIZATION, SP800_185_SAMPLE_DATA, &mut sample);
  assert_eq!(
    hex(&sample),
    concat!(
      "d008828e2b80ac9d2218ffee1d070c48b8e4c87bff32c9699d5b6896eee0edd1",
      "64020e2be0560858d9c00c037e34a96937c561a74c412bb4c746469527281c8c"
    ),
    "cshake256 must reproduce NIST SP 800-185 cSHAKE sample 3"
  );

  sp800_185_group(
    c,
    "cshake256",
    FUNCTION_NAME,
    |function_name, customization, data| {
      let mut out = [0u8; OUTPUT_LEN];
      rscrypto::Cshake256::hash_into(function_name, customization, data, &mut out);
      out
    },
    |function_name, customization, data| {
      use tiny_keccak::Hasher as _;

      let mut out = [0u8; OUTPUT_LEN];
      let mut cshake = tiny_keccak::CShake::v256(function_name, customization);
      cshake.update(data);
      cshake.finalize(&mut out);
      out
    },
  );
}

/// Registers the rscrypto and RustCrypto rows of one HMAC-SHA3 group after
/// checking outside timing that rscrypto reproduces NIST HMAC-SHA3 example 1
/// and that both rows produce the same tag at every size.
fn hmac_sha3_group<Ours: Mac, Theirs: hmac::Mac + KeyInit + Clone>(
  c: &mut Criterion,
  group: &str,
  nist_example_1: &str,
) {
  if !bench_config::selected(group) {
    return;
  }
  let sample_key: Vec<u8> = (0..=u8::MAX).take(Ours::TAG_SIZE).collect();
  assert_eq!(
    hex(Ours::mac(&sample_key, HMAC_SHA3_SAMPLE_MESSAGE).as_ref()),
    nist_example_1,
    "{group} must reproduce NIST HMAC-SHA3 example 1"
  );

  let inputs = common::comp_sizes();
  let key = vec![0x42u8; Ours::TAG_SIZE];
  let base_mac = Theirs::new_from_slice(&key).expect("valid authentication benchmark operation must succeed");
  let rscrypto_tag = |mac: &mut Ours, data: &[u8]| {
    mac.update(data);
    let tag = mac.finalize();
    mac.reset();
    tag
  };
  let rustcrypto_tag = |data: &[u8]| {
    let mut mac = base_mac.clone();
    mac.update(data);
    mac.finalize().into_bytes()
  };

  let mut mac = Ours::new(&key);
  for (len, data) in &inputs {
    assert_eq!(
      rscrypto_tag(&mut mac, data).as_ref(),
      rustcrypto_tag(data).as_slice(),
      "{group}/{len}: rscrypto tag must match RustCrypto"
    );
  }

  let mut g = c.benchmark_group(group);

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      let mut mac = Ours::new(&key);
      b.iter(|| black_box(rscrypto_tag(&mut mac, black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto", len), data, |b, d| {
      b.iter(|| black_box(rustcrypto_tag(black_box(d))))
    });
  }

  g.finish();
}

fn hmac_sha3_224(c: &mut Criterion) {
  hmac_sha3_group::<HmacSha3_224, SimpleHmac<sha3::Sha3_224>>(
    c,
    "hmac-sha3-224",
    "332cfd59347fdb8e576e77260be4aba2d6dc53117b3bfb52c6d18c04",
  );
}

fn hmac_sha3_256(c: &mut Criterion) {
  hmac_sha3_group::<HmacSha3_256, SimpleHmac<sha3::Sha3_256>>(
    c,
    "hmac-sha3-256",
    "4fe8e202c4f058e8dddc23d8c34e467343e23555e24fc2f025d598f558f67205",
  );
}

fn hmac_sha3_384(c: &mut Criterion) {
  hmac_sha3_group::<HmacSha3_384, SimpleHmac<sha3::Sha3_384>>(
    c,
    "hmac-sha3-384",
    concat!(
      "d588a3c51f3f2d906e8298c1199aa8ff6296218127f6b38a",
      "90b6afe2c5617725bc99987f79b22a557b6520db710b7f42"
    ),
  );
}

fn hmac_sha3_512(c: &mut Criterion) {
  hmac_sha3_group::<HmacSha3_512, SimpleHmac<sha3::Sha3_512>>(
    c,
    "hmac-sha3-512",
    concat!(
      "4efd629d6c71bf86162658f29943b1c308ce27cdfa6db0d9c3ce81763f9cbce5",
      "f7ebe9868031db1a8f8eb7b6b95e5c5e3f657a8996c86a2f6527e307f0213196"
    ),
  );
}

fn main() {
  bench_config::run(&[
    kmac128,
    kmac256,
    cshake128,
    cshake256,
    hmac_sha3_224,
    hmac_sha3_256,
    hmac_sha3_384,
    hmac_sha3_512,
  ]);
}
