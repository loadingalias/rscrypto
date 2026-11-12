//! Adler-32 against CPython's `zlib` module (zlib 1.2.12) and the independent
//! `simd-adler32` crate, and the zlib fixtures in `testdata/adler32`.

#![cfg(feature = "adler32")]

use rscrypto::{Adler32, Checksum, ChecksumCombine};

/// `zlib.adler32` (CPython 3.14.8, zlib 1.2.12) results.
const SHORT_ANSWERS: [(&[u8], u32); 5] = [
  (b"", 0x0000_0001),
  (b"a", 0x0062_0062),
  (b"abc", 0x024d_0127),
  (b"123456789", 0x091e_01de),
  (b"Wikipedia", 0x11e6_0398),
];

/// `zlib.adler32(bytes(n))`, `zlib.adler32(b"\xff" * n)`, and
/// `zlib.adler32(b"\xff" * n, MAX_START)`. The lengths sit at the 16-byte step
/// and at one, two, and three times the 5,552-byte reduction distance (zlib's
/// NMAX), and past many reductions. All-0xff input from `MAX_START` drives both
/// sums to the largest values that the portable kernel's `NMAX_FITS` bounds, so
/// a reduction distance past that bound makes the last column wrong.
const LONG_ANSWERS: [(usize, u32, u32, u32); 16] = [
  (1, 0x0001_0001, 0x0100_0100, 0x00fd_00fe),
  (15, 0x000f_0001, 0x7797_0ef2, 0x7778_0ef0),
  (16, 0x0010_0001, 0x8788_0ff1, 0x8767_0fef),
  (17, 0x0011_0001, 0x9878_10f0, 0x9855_10ee),
  (5551, 0x15af_0001, 0x5603_9a8d, 0x2aa4_9a8b),
  (5552, 0x15b0_0001, 0xf18f_9b8c, 0xc62e_9b8a),
  (5553, 0x15b1_0001, 0x8e29_9c8b, 0x62c6_9c89),
  (11_103, 0x2b5f_0001, 0xc849_3627, 0x718a_3625),
  (11_104, 0x2b60_0001, 0xff6f_3726, 0xa8ae_3724),
  (11_105, 0x2b61_0001, 0x37a3_3825, 0xe0d1_3823),
  (16_655, 0x410f_0001, 0x56e0_d1b2, 0xd4b2_d1b0),
  (16_656, 0x4110_0001, 0x29a0_d2b1, 0xa770_d2af),
  (16_657, 0x4111_0001, 0xfd50_d3b0, 0x7b2d_d3ae),
  (139_269, 0x2023_0001, 0x49e1_04be, 0x099a_04bc),
  (524_319, 0x0097_0001, 0xaa57_966a, 0xa928_9668),
  (1_048_589, 0x00fd_0001, 0x0dcd_fc04, 0x0bd2_fc02),
];

/// Both sums at 65520, the largest reduced value.
const MAX_START: u32 = 0xfff0_fff0;

fn oracle(data: &[u8]) -> u32 {
  let mut hasher = simd_adler32::Adler32::new();
  hasher.write(data);
  hasher.finish()
}

fn patterned(len: usize) -> Vec<u8> {
  (0..len)
    .map(|i| i.wrapping_mul(131).wrapping_add(17).to_le_bytes()[0])
    .collect()
}

#[test]
fn checksums_match_zlib_for_short_and_long_input() {
  for (data, expected) in SHORT_ANSWERS {
    assert_eq!(Adler32::checksum(data), expected, "input={data:?}");
  }
  for (len, zeros, ones, ones_from_max) in LONG_ANSWERS {
    let data = vec![0xff; len];
    assert_eq!(Adler32::checksum(&vec![0; len]), zeros, "{len} zero bytes");
    assert_eq!(Adler32::checksum(&data), ones, "{len} 0xff bytes");
    let mut hasher = Adler32::resume(MAX_START);
    hasher.update(&data);
    assert_eq!(hasher.finalize(), ones_from_max, "{len} 0xff bytes from MAX_START");
  }
}

/// Streaming across the reduction boundaries at other offsets than one-shot
/// hashing uses must give zlib's value.
#[test]
fn long_streams_match_zlib_at_every_chunk_size() {
  let (len, _, expected, _) = LONG_ANSWERS
    .into_iter()
    .find(|&(len, ..)| len == 139_269)
    .expect("the table holds 139,269 bytes");
  let data = vec![0xff; len];
  for chunk in [1, 15, 17, 5551, 5553, 65_543] {
    let mut hasher = Adler32::new();
    for part in data.chunks(chunk) {
      hasher.update(part);
    }
    assert_eq!(hasher.finalize(), expected, "chunk={chunk}");
  }
}

#[test]
fn every_split_resume_and_combine_matches_simd_adler32() {
  for len in 0..=300 {
    let data = patterned(len);
    let expected = oracle(&data);
    assert_eq!(Adler32::checksum(&data), expected, "len={len}");
    for at in 0..=len {
      let (head, tail) = data.split_at(at);
      let mut hasher = Adler32::new();
      hasher.update(head);
      hasher.update(tail);
      assert_eq!(hasher.finalize(), expected, "len={len}, split={at}");
      let mut resumed = Adler32::with_initial(oracle(head));
      resumed.update(tail);
      assert_eq!(resumed.finalize(), expected, "len={len}, resumed at {at}");
      let mut resumed = Adler32::resume(oracle(head));
      resumed.update(tail);
      assert_eq!(resumed.finalize(), expected, "len={len}, const resume at {at}");
      assert_eq!(
        Adler32::combine(oracle(head), oracle(tail), tail.len()),
        expected,
        "len={len}, combined at {at}"
      );
    }
    for size in 1..=33 {
      let mut hasher = Adler32::new();
      for part in data.chunks(size) {
        hasher.update(part);
      }
      assert_eq!(hasher.finalize(), expected, "len={len}, chunk={size}");
    }
  }
}

/// zlib's `adler32_combine` needs only the second length modulo 65521. The
/// expected values are `zlib.adler32` of the concatenation (CPython 3.14.8).
/// The first part's `a` sum is 1 for "Wikipedia" and 0 for 256 bytes of 0xff
/// and one 0xf0, where `a - 1` is negative.
#[test]
fn combine_matches_zlib_for_lengths_that_are_multiples_of_65521() {
  let wikipedia = b"Wikipedia".as_slice();
  let a_zero = [[0xff; 256].as_slice(), &[0xf0]].concat();
  assert_eq!(Adler32::checksum(&a_zero), 0x0800_0000);
  for (len, expected) in [
    (65_521, [0xc0a5_8272, 0xb6bf_7eda]),
    (131_042, [0x094f_017e, 0xff5a_fdd7]),
  ] {
    let suffix = patterned(len);
    for (prefix, expected) in [wikipedia, &a_zero].into_iter().zip(expected) {
      let combined = Adler32::combine(Adler32::checksum(prefix), Adler32::checksum(&suffix), len);
      assert_eq!(combined, expected, "len={len}, prefix={} bytes", prefix.len());
    }
  }
}

/// `zlib.adler32` over "Wikipedia" then 2^32 + 7 bytes of 0xff, streamed in
/// CPython, and over the 0xff bytes alone. A length truncated to 32 bits would
/// give the result for 7 bytes instead.
#[cfg(target_pointer_width = "64")]
#[test]
fn combine_takes_a_length_past_2_pow_32_exactly() {
  let len = (1usize << 32).strict_add(7);
  let combined = Adler32::combine(Adler32::checksum(b"Wikipedia"), 0x317b_e719, len);
  assert_eq!(combined, 0x8466_eab0);
}

/// Coalescing short writes must not change the checksum: writes around the
/// 16- and 64-byte flush thresholds and the 512-byte buffer, against the
/// unbuffered checksum of the whole input.
#[cfg(feature = "alloc")]
#[test]
fn buffered_writes_match_the_unbuffered_checksum() {
  let data = patterned(3000);
  let expected = oracle(&data);
  for size in [1, 3, 15, 16, 17, 63, 64, 65, 511, 512, 513, 3000] {
    let mut buffered = Adler32::buffered();
    for part in data.chunks(size) {
      buffered.update(part);
    }
    assert_eq!(buffered.finalize(), expected, "chunk={size}");
    assert_eq!(
      buffered.finalize(),
      expected,
      "finalize keeps the buffered state, chunk={size}"
    );
    buffered.reset();
    buffered.update(&data[..100]);
    assert_eq!(buffered.finalize(), oracle(&data[..100]), "reset, chunk={size}");
  }
}

#[test]
fn reset_restores_the_standard_initial_state() {
  let data = patterned(1000);
  let mut hasher = Adler32::with_initial(0x1234_5678);
  hasher.update(&data);
  hasher.reset();
  assert_eq!(hasher.finalize(), 1);
  hasher.update(&data[..77]);
  assert_eq!(hasher.finalize(), oracle(&data[..77]));
}

/// RFC 1950, section 2.2: CMF and FLG open the stream, and `(CMF·256 + FLG)` is
/// a multiple of 31. With FLG.FDICT (bit 5), the next four bytes are DICTID,
/// the Adler-32 of the preset dictionary. The stream ends with the Adler-32 of
/// the uncompressed data. Both checksums are big-endian.
#[test]
fn zlib_trailer_and_dictionary_id_are_adler32() {
  let payload = include_bytes!("../testdata/adler32/payload.bin");
  let dictionary = include_bytes!("../testdata/adler32/dictionary.bin");
  let plain = include_bytes!("../testdata/adler32/payload.zlib");
  let with_dictionary = include_bytes!("../testdata/adler32/payload-dictionary.zlib");
  let be_u32 = |bytes: &[u8]| u32::from_be_bytes(bytes.try_into().expect("four bytes"));

  let trailer = be_u32(&plain[plain.len().strict_sub(4)..]);
  for stream in [&plain[..], &with_dictionary[..]] {
    assert_eq!(stream[0] & 0x0f, 8, "CM is deflate");
    assert_eq!(u16::from_be_bytes([stream[0], stream[1]]) % 31, 0, "FCHECK");
    assert_eq!(be_u32(&stream[stream.len().strict_sub(4)..]), trailer, "same payload");
  }
  assert_eq!(trailer, Adler32::checksum(payload), "trailer");
  assert_eq!(plain[1] & 0x20, 0, "the plain stream has no dictionary");
  assert_ne!(with_dictionary[1] & 0x20, 0, "FDICT");
  let dictionary_id = be_u32(&with_dictionary[2..6]);
  assert_eq!(dictionary_id, Adler32::checksum(dictionary), "DICTID");
  assert_ne!(dictionary_id, trailer, "DICTID is not the payload checksum");

  // CRC-32 of the same bytes differs, so these fixtures fail an
  // implementation that computes CRC-32 in place of Adler-32.
  #[cfg(feature = "crc32")]
  {
    assert_ne!(trailer, rscrypto::Crc32::checksum(payload));
    assert_ne!(dictionary_id, rscrypto::Crc32::checksum(dictionary));
  }
}
