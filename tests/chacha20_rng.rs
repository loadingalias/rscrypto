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
