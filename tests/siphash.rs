#![cfg(feature = "siphash")]

use core::hash::{BuildHasher, Hasher};

use rscrypto::{SipHash13, SipHash24, SipHashKey, SipRandomState};

/// SipHash-2-4 64-bit outputs for key `00..0f` and messages `00..i-1`, from the
/// reference implementation's `vectors.h` (`vectors_sip64`, veorq/SipHash at
/// 32d067603b93b47828700880649198e0bfbbcffa), read as little-endian words.
const REFERENCE_24: [u64; 64] = [
  0x726fdb47dd0e0e31,
  0x74f839c593dc67fd,
  0x0d6c8009d9a94f5a,
  0x85676696d7fb7e2d,
  0xcf2794e0277187b7,
  0x18765564cd99a68d,
  0xcbc9466e58fee3ce,
  0xab0200f58b01d137,
  0x93f5f5799a932462,
  0x9e0082df0ba9e4b0,
  0x7a5dbbc594ddb9f3,
  0xf4b32f46226bada7,
  0x751e8fbc860ee5fb,
  0x14ea5627c0843d90,
  0xf723ca908e7af2ee,
  0xa129ca6149be45e5,
  0x3f2acc7f57c29bdb,
  0x699ae9f52cbe4794,
  0x4bc1b3f0968dd39c,
  0xbb6dc91da77961bd,
  0xbed65cf21aa2ee98,
  0xd0f2cbb02e3b67c7,
  0x93536795e3a33e88,
  0xa80c038ccd5ccec8,
  0xb8ad50c6f649af94,
  0xbce192de8a85b8ea,
  0x17d835b85bbb15f3,
  0x2f2e6163076bcfad,
  0xde4daaaca71dc9a5,
  0xa6a2506687956571,
  0xad87a3535c49ef28,
  0x32d892fad841c342,
  0x7127512f72f27cce,
  0xa7f32346f95978e3,
  0x12e0b01abb051238,
  0x15e034d40fa197ae,
  0x314dffbe0815a3b4,
  0x027990f029623981,
  0xcadcd4e59ef40c4d,
  0x9abfd8766a33735c,
  0x0e3ea96b5304a7d0,
  0xad0c42d6fc585992,
  0x187306c89bc215a9,
  0xd4a60abcf3792b95,
  0xf935451de4f21df2,
  0xa9538f0419755787,
  0xdb9acddff56ca510,
  0xd06c98cd5c0975eb,
  0xe612a3cb9ecba951,
  0xc766e62cfcadaf96,
  0xee64435a9752fe72,
  0xa192d576b245165a,
  0x0a8787bf8ecb74b2,
  0x81b3e73d20b49b6f,
  0x7fa8220ba3b2ecea,
  0x245731c13ca42499,
  0xb78dbfaf3a8d83bd,
  0xea1ad565322a1a0b,
  0x60e61c23a3795013,
  0x6606d7e446282b93,
  0x6ca4ecb15c5f91e1,
  0x9f626da15c9625f3,
  0xe51b38608ef25f57,
  0x958a324ceb064572,
];

fn reference_key() -> [u8; 16] {
  core::array::from_fn(|i| u8::try_from(i).expect("small"))
}

fn message(length: usize) -> Vec<u8> {
  (0..length).map(|i| u8::try_from(i % 251).expect("small")).collect()
}

#[test]
fn siphash24_matches_the_reference_vectors() {
  let key = SipHashKey::from_bytes(reference_key());
  for (length, expected) in REFERENCE_24.into_iter().enumerate() {
    assert_eq!(SipHash24::hash(&key, &message(length)), expected, "length {length}");
  }
}

/// Independent oracles: Rust std's SipHash-2-4 and the `siphasher` crate, for
/// several keys, every length through 64 bytes, and every split of a streamed write.
#[test]
fn both_variants_match_independent_implementations_in_one_shot_and_streamed() {
  #[expect(deprecated, reason = "std's SipHasher is an independent SipHash-2-4 oracle")]
  fn std24(key: &[u8; 16], data: &[u8]) -> u64 {
    let (k0, k1) = key.split_at(8);
    let mut hasher = core::hash::SipHasher::new_with_keys(
      u64::from_le_bytes(k0.try_into().expect("half")),
      u64::from_le_bytes(k1.try_into().expect("half")),
    );
    hasher.write(data);
    hasher.finish()
  }
  for seed in [0u8, 0x5a, 0xff] {
    let bytes: [u8; 16] = core::array::from_fn(|i| seed.wrapping_mul(31).wrapping_add(u8::try_from(i).expect("small")));
    let key = SipHashKey::from_bytes(bytes);
    for length in 0..=64 {
      let data = message(length);
      let expected13 = {
        let mut oracle = siphasher::sip::SipHasher13::new_with_key(&bytes);
        oracle.write(&data);
        oracle.finish()
      };
      let expected24 = {
        let mut oracle = siphasher::sip::SipHasher24::new_with_key(&bytes);
        oracle.write(&data);
        oracle.finish()
      };
      assert_eq!(std24(&bytes, &data), expected24, "oracles agree at length {length}");
      assert_eq!(SipHash13::hash(&key, &data), expected13, "1-3 length {length}");
      assert_eq!(SipHash24::hash(&key, &data), expected24, "2-4 length {length}");
      for split in 0..=length {
        for second in [split, split.min(length).max(split.saturating_add(3).min(length))] {
          let mut hasher13 = SipHash13::new(&key);
          let mut hasher24 = SipHash24::new(&key);
          for part in [&data[..split], &data[split..second], &data[second..]] {
            hasher13.write(part);
            hasher24.write(part);
          }
          assert_eq!(hasher13.finish(), expected13, "1-3 split {split}/{second} of {length}");
          assert_eq!(hasher24.finish(), expected24, "2-4 split {split}/{second} of {length}");
        }
      }
    }
  }
}

/// One `Hasher` write: a byte string or an integer method.
#[derive(Clone, Copy, Debug)]
enum Write<'a> {
  Bytes(&'a [u8]),
  U8(u8),
  U16(u16),
  U32(u32),
  U64(u64),
  U128(u128),
  Usize(usize),
  I32(i32),
  Isize(isize),
}

impl Write<'_> {
  fn apply(self, hasher: &mut impl Hasher) {
    match self {
      Self::Bytes(bytes) => hasher.write(bytes),
      Self::U8(i) => hasher.write_u8(i),
      Self::U16(i) => hasher.write_u16(i),
      Self::U32(i) => hasher.write_u32(i),
      Self::U64(i) => hasher.write_u64(i),
      Self::U128(i) => hasher.write_u128(i),
      Self::Usize(i) => hasher.write_usize(i),
      Self::I32(i) => hasher.write_i32(i),
      Self::Isize(i) => hasher.write_isize(i),
    }
  }

  /// The bytes the write hashes, as `Hasher`'s default methods define them.
  fn bytes(self) -> Vec<u8> {
    match self {
      Self::Bytes(bytes) => bytes.to_vec(),
      Self::U8(i) => i.to_ne_bytes().to_vec(),
      Self::U16(i) => i.to_ne_bytes().to_vec(),
      Self::U32(i) => i.to_ne_bytes().to_vec(),
      Self::U64(i) => i.to_ne_bytes().to_vec(),
      Self::U128(i) => i.to_ne_bytes().to_vec(),
      Self::Usize(i) => i.to_ne_bytes().to_vec(),
      Self::I32(i) => i.to_ne_bytes().to_vec(),
      Self::Isize(i) => i.to_ne_bytes().to_vec(),
    }
  }
}

/// Integer methods hash the bytes of `to_ne_bytes`, at every pending offset
/// and for every mix with byte strings. The oracle is `siphasher` fed the same
/// bytes in one write, so it shares no buffering logic with rscrypto.
#[test]
fn integer_and_byte_writes_hash_their_byte_stream_at_every_offset() {
  let key_bytes: [u8; 16] = core::array::from_fn(|i| u8::try_from(i).expect("small").wrapping_mul(17));
  let key = SipHashKey::from_bytes(key_bytes);
  let data = message(40);
  let integers = [
    Write::U8(0xa5),
    Write::U16(0xbeef),
    Write::U32(0x0123_4567),
    Write::U64(0x0123_4567_89ab_cdef),
    Write::U128(0x0011_2233_4455_6677_8899_aabb_ccdd_eeff),
    Write::Usize(usize::MAX - 2),
    Write::I32(-7),
    Write::Isize(isize::MIN + 3),
  ];
  let oracle = |stream: &[u8]| {
    let mut hasher13 = siphasher::sip::SipHasher13::new_with_key(&key_bytes);
    let mut hasher24 = siphasher::sip::SipHasher24::new_with_key(&key_bytes);
    hasher13.write(stream);
    hasher24.write(stream);
    (hasher13.finish(), hasher24.finish())
  };
  // A prefix of each length 0..=8 puts every write at every pending offset;
  // the byte strings after each write cross one or more word boundaries.
  for prefix in 0..=8 {
    for first in integers {
      for second in integers {
        for suffix in [0, 1, 7, 8, 9, 23, 40] {
          let writes = [
            Write::Bytes(&data[..prefix]),
            first,
            Write::Bytes(&data[prefix..prefix + suffix.min(40 - prefix)]),
            second,
            Write::Bytes(&data[..suffix]),
          ];
          let stream: Vec<u8> = writes.iter().flat_map(|write| write.bytes()).collect();
          let mut hasher13 = SipHash13::new(&key);
          let mut hasher24 = SipHash24::new(&key);
          for write in writes {
            write.apply(&mut hasher13);
            write.apply(&mut hasher24);
          }
          assert_eq!((hasher13.finish(), hasher24.finish()), oracle(&stream), "{writes:?}");
          assert_eq!(SipHash13::hash(&key, &stream), oracle(&stream).0, "one-shot {writes:?}");
        }
      }
    }
  }
}

/// `finish` reads the state without consuming it: later writes continue the
/// same input.
#[test]
fn finish_does_not_end_the_input() {
  let key = SipHashKey::from_bytes(reference_key());
  let data = message(37);
  let mut hasher = SipHash24::new(&key);
  for (index, byte) in data.iter().enumerate() {
    assert_eq!(hasher.finish(), REFERENCE_24[index], "after {index} bytes");
    hasher.write_u8(*byte);
  }
  assert_eq!(hasher.finish(), REFERENCE_24[37]);
}

#[test]
fn random_state_keys_sip13_and_fails_closed() {
  let state = SipRandomState::try_new_with(|key| {
    key.copy_from_slice(&reference_key());
    Ok::<(), ()>(())
  })
  .expect("fixture key");
  let key = SipHashKey::from_bytes(reference_key());
  for length in [0, 8, 20, 33] {
    let data = message(length);
    let mut hasher = state.build_hasher();
    hasher.write(&data);
    assert_eq!(hasher.finish(), SipHash13::hash(&key, &data));
  }
  // A failing source yields no state, and no key material reaches Debug output.
  SipRandomState::try_new_with(|key| {
    key[..3].fill(1);
    Err("source failed")
  })
  .expect_err("a failing source yields no state");
  SipHashKey::try_generate_with(|_| Err(())).expect_err("a failing source yields no key");
  for debug in [
    format!("{state:?}"),
    format!("{key:?}"),
    format!("{:?}", SipHash13::new(&key)),
  ] {
    assert!(!debug.contains("0, 1, 2") && !debug.contains("00010203"), "{debug}");
  }
}
