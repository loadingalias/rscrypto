#![cfg(feature = "chacha20-rng")]

use chacha20::cipher::{KeyIvInit as _, StreamCipher as _};
use proptest::prelude::*;
use rscrypto::ChaCha20Rng;

mod common;
use common::decode_hex_vec as decode_hex;

/// Keystream bytes per refill and the output bytes it yields, from `ChaCha20Rng`'s documented stream.
const CHUNK: usize = 512;
const OUTPUT: usize = CHUNK - 32;

/// The documented stream, built from RustCrypto's independent ChaCha20.
///
/// Each 512-byte keystream chunk under the current key (nonce zero, counter zero) yields its first 480 bytes as
/// output; its last 32 bytes are the next key.
fn independent_stream(seed: [u8; 32], len: usize) -> Vec<u8> {
  let mut key = seed;
  let mut stream = Vec::with_capacity(len.strict_add(OUTPUT));
  while stream.len() < len {
    let mut chunk = [0u8; CHUNK];
    chacha20::ChaCha20::new(&key.into(), &[0u8; 12].into()).apply_keystream(&mut chunk);
    stream.extend_from_slice(&chunk[..OUTPUT]);
    key.copy_from_slice(&chunk[OUTPUT..]);
  }
  stream.truncate(len);
  stream
}

fn take(rng: &mut ChaCha20Rng, len: usize) -> Vec<u8> {
  let mut output = vec![0; len];
  rng.fill_bytes(&mut output);
  output
}

#[test]
fn first_output_is_the_rfc_8439_appendix_a1_keystream() {
  // RFC 8439 Appendix A.1, test vectors 1 and 2: all-zero key and nonce, block counters 0 and 1.
  let mut zero = ChaCha20Rng::from_seed([0; 32]);
  assert_eq!(
    take(&mut zero, 128),
    decode_hex(concat!(
      "76b8e0ada0f13d90405d6ae55386bd28bdd219b8a08ded1aa836efcc8b770dc7",
      "da41597c5157488d7724e03fb8d84a376a43b8f41518a11cc387b669b2ee6586",
      "9f07e7be5551387a98ba977c732d080dcb0f29a048e3656912c6533e32ee7aed",
      "29b721769ce64e43d57133b074d839d531ed1f28510afb45ace10a1f4b794d6f",
    ))
  );

  // Test vector 3: the key's last byte is 1, block counter 1, so output bytes 64 to 127.
  let mut seed = [0; 32];
  seed[31] = 1;
  let mut rng = ChaCha20Rng::from_seed(seed);
  rng.fill_bytes(&mut [0; 64]);
  assert_eq!(
    take(&mut rng, 64),
    decode_hex(concat!(
      "3aeb5224ecf849929b9d828db1ced4dd832025e8018b8160b82284f3c949aa5a",
      "8eca00bbb4a73bdad192b5c42f73f2fd4e273644c8b36125a64addeb006c13a0",
    ))
  );

  // Test vector 4: key byte 1 is 0xff, block counter 2, so output bytes 128 to 191.
  let mut seed = [0; 32];
  seed[1] = 0xff;
  let mut rng = ChaCha20Rng::from_seed(seed);
  rng.fill_bytes(&mut [0; 128]);
  assert_eq!(
    take(&mut rng, 64),
    decode_hex(concat!(
      "72d54dfbf12ec44b362692df94137f328fea8da73990265ec1bbbea1ae9af0ca",
      "13b25aa26cb4a648cb9b9d1be65b2c0924a66c54d545ec1b7374f4872e99f096",
    ))
  );
}

#[test]
fn stream_matches_independent_chacha20_across_key_refills() {
  let mut patterned = [0u8; 32];
  for (index, byte) in patterned.iter_mut().enumerate() {
    *byte = u8::try_from(index)
      .expect("seed index")
      .wrapping_mul(29)
      .wrapping_add(7);
  }
  for seed in [[0; 32], [0xff; 32], patterned] {
    let len = OUTPUT.strict_mul(20).strict_add(17);
    assert_eq!(
      take(&mut ChaCha20Rng::from_seed(seed), len),
      independent_stream(seed, len)
    );
  }
}

#[test]
fn stream_boundaries_cross_each_refill_exactly() {
  let seed = [0x6b; 32];
  let expected = independent_stream(seed, OUTPUT.strict_mul(3));
  for split in [OUTPUT - 1, OUTPUT, OUTPUT + 1, 2 * OUTPUT] {
    let mut rng = ChaCha20Rng::from_seed(seed);
    let mut output = take(&mut rng, split);
    output.extend(take(&mut rng, OUTPUT.strict_mul(3).strict_sub(split)));
    assert_eq!(output, expected, "split at {split}");
  }
}

#[test]
fn caller_entropy_constructor_matches_from_seed() {
  let seed = [0x37; 32];
  let mut calls = 0;
  let mut filled = ChaCha20Rng::try_generate_with(|output| {
    calls += 1;
    output.copy_from_slice(&seed);
    Ok::<(), ()>(())
  })
  .expect("complete fill");
  assert_eq!(calls, 1);
  assert_eq!(take(&mut filled, 1000), take(&mut ChaCha20Rng::from_seed(seed), 1000));

  let error = ChaCha20Rng::try_generate_with(|output| {
    output[..5].fill(0x42);
    Err("source unavailable")
  })
  .expect_err("failed fill");
  assert_eq!(error, "source unavailable");
}

#[test]
fn distinct_seeds_produce_distinct_streams() {
  let mut seed = [0x11; 32];
  let base = take(&mut ChaCha20Rng::from_seed(seed), 64);
  for index in 0..seed.len() {
    seed[index] ^= 1;
    assert_ne!(take(&mut ChaCha20Rng::from_seed(seed), 64), base, "seed byte {index}");
    seed[index] ^= 1;
  }
}

/// `(label, index)` steps from a parent to a descendant generator.
type ChildPath = &'static [([u8; 8], u64)];

/// A path of `(label, index)` steps from the root seed `00 01 … 1f`, and the first 64 output bytes of its generator.
///
/// RustCrypto `chacha20` gives the expected bytes. The branch key is the first 32 bytes of the block at counter 0
/// with nonce `rscrypto-rng`. Let `x` be the label and the little-endian index. A child seed is the first 32 bytes of
/// the block under the branch key at counter `le32(x[0..4])` with nonce `x[4..16]`. OpenSSL 4.1's `chacha20` gives
/// the same bytes.
const CHILD_VECTORS: [(ChildPath, &str); 5] = [
  (
    &[(*b"rsasync\0", 0)],
    "7435f7b65c7fe3797d8a47eedf8edca942c2a9a78735b64516bf7a13652036765f36d64721eaf502fa0c3dfa5240382a6462bce7fb3fe23ca3ce03ff371d6255",
  ),
  (
    &[(*b"rsasync\0", 1)],
    "8f707dc3edfe049e33c436e871ecae768158c7934c214ee2092e8738c163d9dc63f30bed9bdbb37e5f881080185598f770ec23c1d391004fea0fb84de4fde4e1",
  ),
  (
    &[(*b"rsasync\0", u64::MAX)],
    "687d7baf7e25df512d3aef30ee34a1320c939612ac41b5045f2cedcec0f5cc5bc735c71a04f761c33783d2001374136af01d13175ed280d0d9cc0c1b52004b55",
  ),
  (
    &[(*b"rsnet\0\0\0", 0)],
    "77eebe7c31b22c9c50847e98a8c70be65a4fbbfa03a5476924a90b0365585f2e1e653f32ad97304f561994d95fce387d2f74ed3e3be10e8d5fc9da6127bccbec",
  ),
  (
    &[(*b"rsasync\0", 0), (*b"worker\0\0", 3)],
    "9046af4ab6b5aea49e72610ec95baf88dd72df532076fcde227f71f459920b549fd513ff76c723b649047ec72669e6ee25042a0685a0822d0111820295952fba",
  ),
];

fn child_root() -> [u8; 32] {
  core::array::from_fn(|index| u8::try_from(index).expect("seed index"))
}

/// Walk `path` from `parent` with `ChaCha20Rng::child`, and return the first 64 output bytes of its generator.
fn child_output(parent: &ChaCha20Rng, path: ChildPath) -> Vec<u8> {
  let ((label, index), rest) = path.split_first().expect("non-empty path");
  let mut rng = parent.child(label, *index);
  for (label, index) in rest {
    rng = rng.child(label, *index);
  }
  take(&mut rng, 64)
}

fn assert_child_vectors(parent: &ChaCha20Rng, context: &str) {
  for (path, expected) in CHILD_VECTORS {
    assert_eq!(
      child_output(parent, path),
      decode_hex(expected),
      "{context}: path {path:?}"
    );
  }
}

#[test]
fn child_derivation_matches_independent_chacha20() {
  assert_child_vectors(&ChaCha20Rng::from_seed(child_root()), "fresh parent");
}

#[test]
fn children_do_not_depend_on_parent_draws() {
  // A fresh parent (zero draws) is `child_derivation_matches_independent_chacha20`.
  for draws in [1, OUTPUT - 1, OUTPUT, OUTPUT + 1, 4096] {
    let mut parent = ChaCha20Rng::from_seed(child_root());
    take(&mut parent, draws);
    assert_child_vectors(&parent, &format!("after {draws} parent bytes"));
  }
}

#[test]
fn children_do_not_depend_on_the_parent_constructor() {
  let parent = ChaCha20Rng::try_generate_with(|seed| {
    seed.copy_from_slice(&child_root());
    Ok::<(), ()>(())
  })
  .expect("complete fill");
  assert_child_vectors(&parent, "try_generate_with parent");
}

#[derive(Clone, Debug)]
enum Request {
  Bytes(usize),
  U32,
  U64,
}

fn request() -> impl Strategy<Value = Request> {
  prop_oneof![
    (0usize..1100).prop_map(Request::Bytes),
    Just(Request::U32),
    Just(Request::U64),
  ]
}

proptest! {
  #[test]
  fn output_is_one_byte_stream_for_any_request_sequence(
    seed in any::<[u8; 32]>(),
    requests in proptest::collection::vec(request(), 0..24),
  ) {
    let mut rng = ChaCha20Rng::from_seed(seed);
    let mut produced = Vec::new();
    for request in requests {
      match request {
        Request::Bytes(len) => produced.extend(take(&mut rng, len)),
        Request::U32 => produced.extend(rng.next_u32().to_le_bytes()),
        Request::U64 => produced.extend(rng.next_u64().to_le_bytes()),
      }
    }
    prop_assert_eq!(&produced, &independent_stream(seed, produced.len()));
  }
}

#[cfg(feature = "rand_core")]
mod rand_core_traits {
  use core::{convert::Infallible, fmt};

  use proptest::prelude::*;
  use rand_core::{CryptoRng, Rng, SeedableRng, TryRng};
  use rscrypto::ChaCha20Rng;

  use super::{OUTPUT, Request, independent_stream, request, take};

  /// Draw through the traits only, as a generic `rand_core` consumer does.
  fn trait_take(rng: &mut (impl Rng + ?Sized), len: usize) -> Vec<u8> {
    let mut output = vec![0; len];
    rng.fill_bytes(&mut output);
    output
  }

  #[test]
  fn seedable_and_boxed_crypto_generators_return_the_documented_stream() {
    let seed = [0x5d; 32];
    let expected = independent_stream(seed, 2 * OUTPUT + 12);

    let mut rng = <ChaCha20Rng as SeedableRng>::from_seed(seed);
    let mut produced = trait_take(&mut rng, OUTPUT - 1);
    produced.extend(Rng::next_u32(&mut rng).to_le_bytes());
    produced.extend(
      rng
        .try_next_u64()
        .map(u64::to_le_bytes)
        .unwrap_or_else(|never| match never {}),
    );
    let mut rest = vec![0; OUTPUT + 1];
    rng
      .try_fill_bytes(&mut rest)
      .unwrap_or_else(|never: Infallible| match never {});
    produced.extend(rest);
    assert_eq!(produced, expected);

    // Callers can store generators as boxed `CryptoRng` trait objects.
    let mut boxed: Box<dyn CryptoRng + Send + Sync> = Box::new(ChaCha20Rng::from_seed(seed));
    assert_eq!(trait_take(&mut boxed, expected.len()), expected);
  }

  #[test]
  fn from_rng_and_fork_seed_the_child_from_the_parent_stream() {
    let seed = [0x29; 32];
    let parent_stream = independent_stream(seed, 32 + 100);
    let child_seed: [u8; 32] = parent_stream[..32].try_into().expect("32 bytes");
    let child_stream = independent_stream(child_seed, 600);

    let mut parent = ChaCha20Rng::from_seed(seed);
    let mut child = ChaCha20Rng::from_rng(&mut parent);
    assert_eq!(take(&mut child, 600), child_stream);
    assert_eq!(take(&mut parent, 100), parent_stream[32..]);

    let mut parent = ChaCha20Rng::from_seed(seed);
    assert_eq!(take(&mut parent.fork(), 600), child_stream);

    let mut parent = ChaCha20Rng::from_seed(seed);
    let mut child = ChaCha20Rng::try_from_rng(&mut parent).unwrap_or_else(|never| match never {});
    assert_eq!(take(&mut child, 600), child_stream);

    let mut parent = ChaCha20Rng::from_seed(seed);
    let mut child = parent.try_fork().unwrap_or_else(|never| match never {});
    assert_eq!(take(&mut child, 600), child_stream);
  }

  #[derive(Debug, PartialEq)]
  struct Unavailable;

  impl fmt::Display for Unavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      f.write_str("entropy source unavailable")
    }
  }

  impl core::error::Error for Unavailable {}

  /// Writes part of a request, then fails.
  struct FailingSource;

  impl TryRng for FailingSource {
    type Error = Unavailable;

    fn try_next_u32(&mut self) -> Result<u32, Unavailable> {
      Err(Unavailable)
    }

    fn try_next_u64(&mut self) -> Result<u64, Unavailable> {
      Err(Unavailable)
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Unavailable> {
      dst[..7].fill(0x42);
      Err(Unavailable)
    }
  }

  #[test]
  fn try_from_rng_returns_the_source_error() {
    assert_eq!(
      ChaCha20Rng::try_from_rng(&mut FailingSource).expect_err("failing source"),
      Unavailable
    );
  }

  #[test]
  #[should_panic(expected = "ChaCha20Rng needs a 32-byte secret seed")]
  fn seed_from_u64_refuses_a_64_bit_seed() {
    let _rng = ChaCha20Rng::seed_from_u64(7);
  }

  proptest! {
    #[test]
    fn trait_and_inherent_requests_share_one_stream(
      seed in any::<[u8; 32]>(),
      requests in proptest::collection::vec((request(), any::<bool>()), 0..24),
    ) {
      let mut rng = ChaCha20Rng::from_seed(seed);
      let mut produced = Vec::new();
      for (request, through_trait) in requests {
        match (request, through_trait) {
          (Request::Bytes(len), false) => produced.extend(take(&mut rng, len)),
          (Request::Bytes(len), true) => produced.extend(trait_take(&mut rng, len)),
          (Request::U32, false) => produced.extend(rng.next_u32().to_le_bytes()),
          (Request::U32, true) => produced.extend(Rng::next_u32(&mut rng).to_le_bytes()),
          (Request::U64, false) => produced.extend(rng.next_u64().to_le_bytes()),
          (Request::U64, true) => produced.extend(Rng::next_u64(&mut rng).to_le_bytes()),
        }
      }
      prop_assert_eq!(&produced, &independent_stream(seed, produced.len()));
    }
  }
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
#[test]
fn portable_diagnostic_fill_matches_the_documented_stream() {
  // Binary constant-time analysis runs this entry point, so it must take the production state machine.
  let seed: [u8; 32] = core::array::from_fn(|index| u8::try_from(index).unwrap_or(0).wrapping_mul(29).wrapping_add(3));
  for len in [
    0,
    1,
    32,
    OUTPUT - 1,
    OUTPUT,
    OUTPUT + 1,
    2 * OUTPUT,
    2 * OUTPUT + 7,
    3 * CHUNK,
  ] {
    let mut output = vec![0; len];
    rscrypto::rng::diag_chacha20_rng_fill_portable(seed, &mut output);
    assert_eq!(output, independent_stream(seed, len), "len={len}");
  }
}
