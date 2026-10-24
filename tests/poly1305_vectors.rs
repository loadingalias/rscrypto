#![cfg(feature = "poly1305")]

use rscrypto::{Poly1305, Poly1305OneTimeKey, Poly1305Tag};

mod common;
#[path = "common/array.rs"]
mod hex_array;
use hex_array::decode_hex_array;

#[test]
fn poly1305_matches_rfc_8439_section_2_5_2() {
  let key = Poly1305OneTimeKey::from_bytes(decode_hex_array(
    "85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b",
  ));
  let message = b"Cryptographic Forum Research Group";
  let expected = Poly1305Tag::from_bytes(decode_hex_array("a8061dc1305136c6c22b8baf0c0127a9"));

  let tag = Poly1305::authenticate_once(key, message);
  assert!(tag.ct_eq(&expected).declassify());
}

#[test]
fn poly1305_streaming_matches_oneshot() {
  let key = Poly1305OneTimeKey::from_bytes([0x42; Poly1305OneTimeKey::LENGTH]);
  let expected = Poly1305::authenticate_once(Poly1305OneTimeKey::from_bytes([0x42; 32]), b"abcdef");

  let mut poly1305 = Poly1305::new(key);
  poly1305.update(b"a");
  poly1305.update(b"bcde");
  poly1305.update(b"f");

  assert!(poly1305.finalize().ct_eq(&expected).declassify());
}

#[test]
fn poly1305_block_boundaries_and_carries_match_dryoc() {
  use dryoc::classic::crypto_onetimeauth::crypto_onetimeauth;

  // The all-one key/message exercises clamp masks and long carry chains. The
  // zero case covers the additive pad; mixed inputs vary all five limbs.
  for seed in [0u8, 0xff, 0x42] {
    let key = core::array::from_fn(|i| {
      if seed == 0x42 {
        seed.wrapping_add(u8::try_from(i).expect("key index fits u8").wrapping_mul(73))
      } else {
        seed
      }
    });
    let data: Vec<u8> = (0..4098)
      .map(|i| {
        if seed == 0x42 {
          u8::try_from(i % 251).expect("remainder fits u8")
        } else {
          seed
        }
      })
      .collect();
    for len in (0..=64).chain([127, 128, 129, 255, 256, 257, 1024, 4097]) {
      // Offset the input to cover unaligned loads as well as every short tail.
      let message = &data[1..=len];
      let mut expected = [0; 16];
      crypto_onetimeauth(&mut expected, message, &key);
      let actual = Poly1305::authenticate_once(Poly1305OneTimeKey::from_bytes(key), message);
      assert_eq!(actual.to_bytes(), expected, "seed={seed} len={len}");
      for chunk in [1, 15, 16, 17, 31, 64] {
        let mut state = Poly1305::new(Poly1305OneTimeKey::from_bytes(key));
        for part in message.chunks(chunk) {
          state.update(part);
          state.update(&[]);
        }
        assert_eq!(
          state.finalize().to_bytes(),
          expected,
          "seed={seed} len={len} chunk={chunk}"
        );
      }
    }
  }
}

#[test]
fn poly1305_verify_rejects_corrupted_tag() {
  let key = Poly1305OneTimeKey::from_bytes([0x24; Poly1305OneTimeKey::LENGTH]);
  let message = b"message";
  let tag = Poly1305::authenticate_once(Poly1305OneTimeKey::from_bytes([0x24; 32]), message);
  let mut corrupted = tag.to_bytes();
  corrupted[7] ^= 0x80;
  let corrupted = Poly1305Tag::from_bytes(corrupted);

  Poly1305::verify_once(key, message, &corrupted).expect_err("corrupted Poly1305 tag must be rejected");
}

#[test]
fn poly1305_try_generate_with_uses_fallible_fill() {
  let key = Poly1305OneTimeKey::try_generate_with(|out| {
    out.fill(0x11);
    Ok::<(), ()>(())
  })
  .expect("deterministic entropy callback must generate a Poly1305 key");

  assert_eq!(key.as_bytes(), &[0x11; Poly1305OneTimeKey::LENGTH]);
  Poly1305OneTimeKey::try_generate_with(|_| Err("rng")).expect_err("entropy-source failure must be returned");
}

/// RFC 8439 Appendix A.3 Poly1305 vectors. Vectors 5 to 11 exercise the final reduction and the
/// carries at the 2^130 boundary.
#[test]
fn poly1305_matches_rfc_8439_appendix_a3() {
  use dryoc::classic::crypto_onetimeauth::crypto_onetimeauth;

  const IETF: &str = "Any submission to the IETF intended by the Contributor for publication as all or part of an \
                      IETF Internet-Draft or RFC and any statement made within the context of an IETF activity is \
                      considered an \"IETF Contribution\". Such statements include oral statements in IETF sessions, \
                      as well as written and electronic communications made at any time or place, which are \
                      addressed to";
  const JABBERWOCKY: &str = "'Twas brillig, and the slithy toves\nDid gyre and gimble in the wabe:\nAll mimsy were the \
                             borogoves,\nAnd the mome raths outgrabe.";
  let vector10 = "e33594d7505e43b900000000000000003394d7505e4379cd010000000000000000000000000000000000000000000000\
                  01000000000000000000000000000000";
  let cases: [(&str, Vec<u8>, &str); 11] = [
    (
      "0000000000000000000000000000000000000000000000000000000000000000",
      vec![0; 64],
      "00000000000000000000000000000000",
    ),
    (
      "0000000000000000000000000000000036e5f6b5c5e06070f0efca96227a863e",
      IETF.as_bytes().to_vec(),
      "36e5f6b5c5e06070f0efca96227a863e",
    ),
    (
      "36e5f6b5c5e06070f0efca96227a863e00000000000000000000000000000000",
      IETF.as_bytes().to_vec(),
      "f3477e7cd95417af89a6b8794c310cf0",
    ),
    (
      "1c9240a5eb55d38af333888604f6b5f0473917c1402b80099dca5cbc207075c0",
      JABBERWOCKY.as_bytes().to_vec(),
      "4541669a7eaaee61e708dc7cbcc5eb62",
    ),
    (
      "0200000000000000000000000000000000000000000000000000000000000000",
      common::decode_hex_vec("ffffffffffffffffffffffffffffffff"),
      "03000000000000000000000000000000",
    ),
    (
      "02000000000000000000000000000000ffffffffffffffffffffffffffffffff",
      common::decode_hex_vec("02000000000000000000000000000000"),
      "03000000000000000000000000000000",
    ),
    (
      "0100000000000000000000000000000000000000000000000000000000000000",
      common::decode_hex_vec(
        "fffffffffffffffffffffffffffffffff0ffffffffffffffffffffffffffffff11000000000000000000000000000000",
      ),
      "05000000000000000000000000000000",
    ),
    (
      "0100000000000000000000000000000000000000000000000000000000000000",
      common::decode_hex_vec(
        "fffffffffffffffffffffffffffffffffbfefefefefefefefefefefefefefefe01010101010101010101010101010101",
      ),
      "00000000000000000000000000000000",
    ),
    (
      "0200000000000000000000000000000000000000000000000000000000000000",
      common::decode_hex_vec("fdffffffffffffffffffffffffffffff"),
      "faffffffffffffffffffffffffffffff",
    ),
    (
      "0100000000000000040000000000000000000000000000000000000000000000",
      common::decode_hex_vec(vector10),
      "14000000000000005500000000000000",
    ),
    (
      "0100000000000000040000000000000000000000000000000000000000000000",
      common::decode_hex_vec(vector10)[..48].to_vec(),
      "13000000000000000000000000000000",
    ),
  ];
  for (index, (key, message, tag)) in cases.iter().enumerate() {
    let number = index.strict_add(1);
    let key: [u8; 32] = decode_hex_array(key);
    let expected: [u8; 16] = decode_hex_array(tag);
    // The independent implementation confirms each transcribed vector.
    let mut independent = [0u8; 16];
    crypto_onetimeauth(&mut independent, message, &key);
    assert_eq!(independent, expected, "vector {number}");
    let actual = Poly1305::authenticate_once(Poly1305OneTimeKey::from_bytes(key), message);
    assert_eq!(actual.to_bytes(), expected, "vector {number}");
    let mut streaming = Poly1305::new(Poly1305OneTimeKey::from_bytes(key));
    for byte in message.chunks(1) {
      streaming.update(byte);
    }
    assert_eq!(streaming.finalize().to_bytes(), expected, "streamed vector {number}");
  }
}

/// A deterministic xorshift64 sequence for inputs and split points.
struct Xorshift(u64);

impl Xorshift {
  fn next(&mut self) -> u64 {
    self.0 ^= self.0 << 13;
    self.0 ^= self.0 >> 7;
    self.0 ^= self.0 << 17;
    self.0
  }

  fn below(&mut self, bound: usize) -> usize {
    let bound = u64::try_from(bound).expect("bound fits u64");
    let value = self.next().checked_rem(bound).expect("bound is nonzero");
    usize::try_from(value).expect("value below a usize bound fits usize")
  }
}

fn streamed(key: [u8; 32], message: &[u8], splits: &mut dyn FnMut(usize) -> usize) -> [u8; 16] {
  let mut state = Poly1305::new(Poly1305OneTimeKey::from_bytes(key));
  let mut rest = message;
  while !rest.is_empty() {
    let take = splits(rest.len()).min(rest.len());
    state.update(&rest[..take]);
    rest = &rest[take..];
  }
  state.finalize().to_bytes()
}

/// Every length to 1100 bytes and larger sizes, at shifted alignments, through one-shot, byte-by-byte,
/// random, and block-boundary splits. dryoc and RustCrypto supply the expected tags. Native builds run
/// the dispatched kernels; `portable-only` builds run the portable authority.
#[test]
fn poly1305_split_patterns_and_alignments_match_independent_implementations() {
  use dryoc::classic::crypto_onetimeauth::crypto_onetimeauth;
  use poly1305::universal_hash::KeyInit as _;

  let mut rng = Xorshift(0x9e37_79b9_7f4a_7c15);
  let data: Vec<u8> = (0..70_000)
    .map(|_| {
      let [byte, ..] = rng.next().to_le_bytes();
      byte
    })
    .collect();
  let ones = vec![0xffu8; 70_000];
  let mut patterned_key = [0u8; 32];
  for (index, byte) in patterned_key.iter_mut().enumerate() {
    *byte = u8::try_from(index)
      .expect("key index fits u8")
      .wrapping_mul(97)
      .wrapping_add(5);
  }
  let check = |key: [u8; 32], message: &[u8], rng: &mut Xorshift, byte_by_byte: bool| {
    let len = message.len();
    let mut expected = [0u8; 16];
    crypto_onetimeauth(&mut expected, message, &key);
    let rustcrypto: [u8; 16] = poly1305::Poly1305::new(&key.into()).compute_unpadded(message).into();
    assert_eq!(rustcrypto, expected, "oracles disagree at len {len}");
    let once = Poly1305::authenticate_once(Poly1305OneTimeKey::from_bytes(key), message);
    assert_eq!(once.to_bytes(), expected, "one-shot len {len}");
    if byte_by_byte {
      assert_eq!(streamed(key, message, &mut |_| 1), expected, "byte-by-byte len {len}");
    }
    assert_eq!(
      streamed(key, message, &mut |_| rng.below(48)),
      expected,
      "short random splits len {len}"
    );
    assert_eq!(
      streamed(key, message, &mut |remaining| rng.below(remaining.strict_add(1))),
      expected,
      "random splits len {len}"
    );
    let block_split = rng.below(10).strict_mul(16).max(16);
    assert_eq!(
      streamed(key, message, &mut |_| block_split),
      expected,
      "block-boundary splits len {len}"
    );
    assert_eq!(
      streamed(key, message, &mut |_| block_split.strict_add(1)),
      expected,
      "off-boundary splits len {len}"
    );
    assert!(
      Poly1305::verify_once(Poly1305OneTimeKey::from_bytes(key), message, &once).is_ok(),
      "verify_once len {len}"
    );
    let mut verifier = Poly1305::new(Poly1305OneTimeKey::from_bytes(key));
    verifier.update(message);
    assert!(verifier.verify(&once).is_ok(), "verify len {len}");
  };

  for key in [[0xffu8; 32], patterned_key] {
    for len in 0..=1100usize {
      check(key, &data[1..len.strict_add(1)], &mut rng, len <= 320);
      check(key, &ones[..len], &mut rng, false);
    }
    for len in [0usize, 1, 15, 16, 17, 64, 127, 128, 129, 255, 256, 257, 1024, 1039] {
      for offset in 0..64usize {
        check(key, &data[offset..offset.strict_add(len)], &mut rng, false);
      }
    }
    for len in [2047usize, 4096, 16_384, 16_401, 65_541] {
      check(key, &data[5..len.strict_add(5)], &mut rng, false);
      check(key, &ones[..len], &mut rng, false);
    }
  }
}

/// Changing any one tag bit makes both verification paths reject, on lengths that reach the scalar
/// and vector kernels.
#[test]
fn poly1305_verification_rejects_every_tag_bit_flip() {
  for len in [0usize, 15, 64, 129, 1024] {
    let message: Vec<u8> = (0..len)
      .map(|index| u8::try_from(index % 251).expect("remainder fits u8"))
      .collect();
    let key = [0x37u8; 32];
    let tag = Poly1305::authenticate_once(Poly1305OneTimeKey::from_bytes(key), &message).to_bytes();
    for bit in 0..128usize {
      let mut corrupted = tag;
      corrupted[bit / 8] ^= 1 << (bit % 8);
      let corrupted = Poly1305Tag::from_bytes(corrupted);
      assert!(
        Poly1305::verify_once(Poly1305OneTimeKey::from_bytes(key), &message, &corrupted).is_err(),
        "verify_once accepted bit {bit} at len {len}"
      );
      let mut verifier = Poly1305::new(Poly1305OneTimeKey::from_bytes(key));
      verifier.update(&message);
      assert!(
        verifier.verify(&corrupted).is_err(),
        "verify accepted bit {bit} at len {len}"
      );
    }
  }
}
