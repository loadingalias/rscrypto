//! XXH32 and XXH64 against xxhsum 0.8.4, the xxHash reference implementation,
//! and against the independent `xxhash-rust` crate.

#![cfg(any(feature = "xxh32", feature = "xxh64"))]

/// The xxHash sanity buffer: a 64-bit state starts at 2654435761, each byte is
/// its top byte, and the state then multiplies by 11400714785074694797.
fn sanity_bytes(len: usize) -> Vec<u8> {
  let mut state: u64 = 2_654_435_761;
  (0..len)
    .map(|_| {
      let byte = state.to_be_bytes()[0];
      state = state.wrapping_mul(11_400_714_785_074_694_797);
      byte
    })
    .collect()
}

/// xxhsum 0.8.4 results for `sanity_bytes(length)`:
/// `xxhsum -H0`, `xxhsum -H0 -s2654435761`, `xxhsum -H1`, and `xxhsum -H1 -s11400714785074694791`.
/// The lengths cover the empty input, each tail shape, the stripe boundaries
/// (16 bytes for XXH32, 32 for XXH64), and two long inputs.
const KNOWN_ANSWERS: [(usize, u32, u32, u64, u64); 19] = [
  (0, 0x02cc5d05, 0x36b78ae7, 0xef46db3751d8e999, 0x6ec6d05f61c7e7a7),
  (1, 0xcf65b03e, 0xb4545aa4, 0xe934a84adb052768, 0x60508b0ced72c717),
  (3, 0xc23884f5, 0x1a269947, 0xff7e1959cb50794a, 0xb7c97337300aa844),
  (4, 0xa9de7ce9, 0x2baafe83, 0x9136a0dca57457ee, 0x05c571d4638902d1),
  (5, 0xeb1734bb, 0x5874dab0, 0x9b046fb1397f09a5, 0x6b970531073f0ac7),
  (8, 0xa3f6f44b, 0xc2a8e239, 0xcdbcf538e71d1348, 0xc9bae69468995dd2),
  (12, 0xe89b5f9b, 0x05a6c4b5, 0x0723bf50086ead9a, 0x86966626b6d03591),
  (13, 0xc537de02, 0x1298add0, 0xc2e5013e3c40bcf7, 0x752e5af344e83bae),
  (15, 0x6b859e14, 0xad53090d, 0x180719316d622d84, 0xd10799c88a4c631f),
  (16, 0x93ba3759, 0xa94fc1e1, 0x98c90b57fdfcb55c, 0xd547a25480a2febb),
  (17, 0x89fdc23e, 0xc9910739, 0x0d39a2d051a30c2c, 0x62dbbb84377a27ca),
  (31, 0x5f40e562, 0x5c0c3350, 0x299b39a290e6d783, 0xd5b2dbbfa83b0b60),
  (32, 0xd89829ec, 0xa5c44467, 0x18b216492bb44b70, 0xaafef1645d1b13d9),
  (33, 0x31a427e5, 0x0de5b1f9, 0x55c8dc3e578f5b59, 0x07493c92a23a6825),
  (63, 0xf1d48fdb, 0x956b3d77, 0xa9efbe0fa0f3f4e7, 0x626845aeea2bfc12),
  (64, 0x02e95dbb, 0xcf82f830, 0xef558f8acac2b5cd, 0x97973ade8b590fe4),
  (65, 0x16992b3d, 0x428eec5f, 0xde0f20dc2631af7a, 0x8f1925dc517d9b78),
  (4587, 0x48e1661e, 0x1e36e686, 0x055d09759da0c73b, 0x1ac3ccd932a7e485),
  (
    1_048_589,
    0x4499a43b,
    0x4b9689c5,
    0xf02046e12badfec3,
    0xf89cb5f117187a65,
  ),
];

/// Copies of `data` that start at each byte offset from 0 to 7 of a buffer.
fn at_each_offset(data: &[u8]) -> impl Iterator<Item = (usize, Vec<u8>)> {
  (0..8usize).map(move |offset| {
    let mut buffer = vec![0xa5; offset.strict_add(data.len())];
    buffer[offset..].copy_from_slice(data);
    (offset, buffer)
  })
}

/// Every two-part split, then every uniform chunk size from 1 to 33 bytes.
fn splits(len: usize) -> impl Iterator<Item = Vec<core::ops::Range<usize>>> {
  let two_part = (0..=len).map(move |at| vec![0..at, at..len]);
  let uniform = (1..=33usize).map(move |size| {
    (0..len.div_ceil(size))
      .map(|index| index.strict_mul(size)..index.strict_add(1).strict_mul(size).min(len))
      .collect()
  });
  two_part.chain(uniform)
}

#[cfg(feature = "xxh32")]
mod xxh32 {
  use rscrypto::{FastHash, Xxh32, Xxh32Hasher};

  use super::{KNOWN_ANSWERS, at_each_offset, sanity_bytes, splits};

  const SEEDS: [u32; 4] = [0, 1, 0x9e37_79b1, u32::MAX];

  #[test]
  fn one_shot_and_stream_match_xxhsum_at_every_offset() {
    for (len, seed0, seed_prime, _, _) in KNOWN_ANSWERS {
      let data = sanity_bytes(len);
      for (seed, expected) in [(0, seed0), (0x9e37_79b1, seed_prime)] {
        for (offset, buffer) in at_each_offset(&data) {
          assert_eq!(
            Xxh32::hash_with_seed(seed, &buffer[offset..]),
            expected,
            "len={len}, seed={seed:#x}, offset={offset}"
          );
        }
        let mut hasher = Xxh32Hasher::with_seed(seed);
        for chunk in data.chunks(61) {
          hasher.write(chunk);
        }
        assert_eq!(hasher.finish(), expected, "stream len={len}, seed={seed:#x}");
      }
    }
  }

  #[test]
  fn every_split_matches_xxhash_rust() {
    for seed in SEEDS {
      for len in 0..=130 {
        let data = sanity_bytes(len);
        let expected = xxhash_rust::xxh32::xxh32(&data, seed);
        assert_eq!(
          Xxh32::hash_with_seed(seed, &data),
          expected,
          "len={len}, seed={seed:#x}"
        );
        for split in splits(len) {
          let mut hasher = Xxh32Hasher::with_seed(seed);
          for range in &split {
            hasher.write(&data[range.clone()]);
          }
          assert_eq!(hasher.finish(), expected, "len={len}, seed={seed:#x}, split={split:?}");
        }
      }
    }
  }

  #[test]
  fn reset_keeps_the_seed_and_drops_written_input() {
    let data = sanity_bytes(100);
    let mut hasher = Xxh32Hasher::with_seed(0x9e37_79b1);
    hasher.write(&data);
    hasher.reset();
    assert_eq!(hasher.finish(), xxhash_rust::xxh32::xxh32(b"", 0x9e37_79b1));
    hasher.write(&data[..37]);
    hasher.reset();
    hasher.write(&data[..21]);
    assert_eq!(hasher.finish(), xxhash_rust::xxh32::xxh32(&data[..21], 0x9e37_79b1));
  }

  /// XXH32 mixes in the length modulo 2^32, but an input of 2^32 + 7 bytes is
  /// still a long input: a 32-bit counter would wrap to 7 and select the
  /// short-input path. `head -c 4294967303 /dev/zero | xxhsum -H0` gives
  /// 844cb0a7 with xxhsum 0.8.4. The test streams the input from one 64 KiB
  /// buffer.
  #[test]
  fn input_beyond_4_gib_keeps_the_long_input_path() {
    let zeros = [0u8; 1 << 16];
    let mut hasher = Xxh32Hasher::new();
    for _ in 0..1u32 << 16 {
      hasher.write(&zeros);
    }
    hasher.write(&zeros[..7]);
    assert_eq!(hasher.finish(), 0x844c_b0a7);
  }
}

#[cfg(feature = "xxh64")]
mod xxh64 {
  use core::hash::Hasher;

  use rscrypto::{FastHash, Xxh64, Xxh64Hasher};

  use super::{KNOWN_ANSWERS, at_each_offset, sanity_bytes, splits};

  const SEEDS: [u64; 4] = [0, 1, 0x9e37_79b1_85eb_ca87, u64::MAX];

  #[test]
  fn one_shot_and_stream_match_xxhsum_at_every_offset() {
    for (len, _, _, seed0, seed_prime) in KNOWN_ANSWERS {
      let data = sanity_bytes(len);
      for (seed, expected) in [(0, seed0), (0x9e37_79b1_85eb_ca87, seed_prime)] {
        for (offset, buffer) in at_each_offset(&data) {
          assert_eq!(
            Xxh64::hash_with_seed(seed, &buffer[offset..]),
            expected,
            "len={len}, seed={seed:#x}, offset={offset}"
          );
        }
        let mut hasher = Xxh64Hasher::with_seed(seed);
        for chunk in data.chunks(61) {
          hasher.write(chunk);
        }
        assert_eq!(hasher.finish(), expected, "stream len={len}, seed={seed:#x}");
      }
    }
  }

  #[test]
  fn every_split_matches_xxhash_rust() {
    for seed in SEEDS {
      for len in 0..=130 {
        let data = sanity_bytes(len);
        let expected = xxhash_rust::xxh64::xxh64(&data, seed);
        assert_eq!(
          Xxh64::hash_with_seed(seed, &data),
          expected,
          "len={len}, seed={seed:#x}"
        );
        for split in splits(len) {
          let mut hasher = Xxh64Hasher::with_seed(seed);
          for range in &split {
            hasher.write(&data[range.clone()]);
          }
          assert_eq!(hasher.finish(), expected, "len={len}, seed={seed:#x}, split={split:?}");
        }
      }
    }
  }

  #[test]
  fn reset_keeps_the_seed_and_drops_written_input() {
    let data = sanity_bytes(100);
    let mut hasher = Xxh64Hasher::with_seed(0x9e37_79b1_85eb_ca87);
    hasher.write(&data);
    hasher.reset();
    assert_eq!(hasher.finish(), xxhash_rust::xxh64::xxh64(b"", 0x9e37_79b1_85eb_ca87));
    hasher.write(&data[..37]);
    hasher.reset();
    hasher.write(&data[..41]);
    assert_eq!(
      hasher.finish(),
      xxhash_rust::xxh64::xxh64(&data[..41], 0x9e37_79b1_85eb_ca87)
    );
  }
}
