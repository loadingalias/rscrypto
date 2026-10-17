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
//! Before timing, every row must return the competitor's output for its input.

#[path = "common/criterion.rs"]
mod bench_config;

mod common;

use core::hint::black_box;

use bench_config::{checked, checked_with};
use criterion::{BenchmarkId, Criterion};
use hmac::{KeyInit, SimpleHmac};
use rscrypto::{HmacSha3_224, HmacSha3_256, HmacSha3_384, HmacSha3_512, Mac};

const KMAC_KEY: &[u8] = b"rscrypto-kmac-bench-key-32bytes";
const CUSTOMIZATION: &[u8] = b"rscrypto-bench";
const FUNCTION_NAME: &[u8] = b"";
const OUTPUT_LEN: usize = 32;

/// Registers the rscrypto and tiny-keccak rows of one SP 800-185 group. Both
/// must return tiny-keccak's output. `key_or_name` is the KMAC key or the
/// cSHAKE function name.
fn sp800_185_group(
  c: &mut Criterion,
  group: &str,
  key_or_name: &[u8],
  rscrypto: impl Fn(&[u8], &[u8], &[u8]) -> [u8; OUTPUT_LEN],
  tiny_keccak: impl Fn(&[u8], &[u8], &[u8]) -> [u8; OUTPUT_LEN],
) {
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group(group);

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = tiny_keccak(key_or_name, CUSTOMIZATION, data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto(black_box(key_or_name), black_box(CUSTOMIZATION), black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );

    g.bench_function(
      BenchmarkId::new("tiny-keccak", len),
      checked(
        || tiny_keccak(black_box(key_or_name), black_box(CUSTOMIZATION), black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
  }

  g.finish();
}

fn kmac128(c: &mut Criterion) {
  if !bench_config::selected("kmac128") {
    return;
  }
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

/// Registers the rscrypto and RustCrypto rows of one HMAC-SHA3 group. Both must
/// return RustCrypto's tag.
fn hmac_sha3_group<Ours: Mac, Theirs: hmac::Mac + KeyInit + Clone>(c: &mut Criterion, group: &str) {
  if !bench_config::selected(group) {
    return;
  }
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

  let mut g = c.benchmark_group(group);

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = rustcrypto_tag(data);

    // The check sees the second tag, after one reset of the reused state.
    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked_with(
        Ours::new(&key),
        |mac| rscrypto_tag(mac, black_box(data)),
        |tag, _| assert_eq!(tag.as_ref(), expected.as_slice()),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rustcrypto", len),
      checked(|| rustcrypto_tag(black_box(data)), |tag| assert_eq!(tag, expected)),
    );
  }

  g.finish();
}

fn hmac_sha3_224(c: &mut Criterion) {
  hmac_sha3_group::<HmacSha3_224, SimpleHmac<sha3::Sha3_224>>(c, "hmac-sha3-224");
}

fn hmac_sha3_256(c: &mut Criterion) {
  hmac_sha3_group::<HmacSha3_256, SimpleHmac<sha3::Sha3_256>>(c, "hmac-sha3-256");
}

fn hmac_sha3_384(c: &mut Criterion) {
  hmac_sha3_group::<HmacSha3_384, SimpleHmac<sha3::Sha3_384>>(c, "hmac-sha3-384");
}

fn hmac_sha3_512(c: &mut Criterion) {
  hmac_sha3_group::<HmacSha3_512, SimpleHmac<sha3::Sha3_512>>(c, "hmac-sha3-512");
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
