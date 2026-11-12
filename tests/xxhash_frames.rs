//! Frame checksums in fixtures that the LZ4 1.11.0 and Zstandard 1.5.7 CLIs
//! made; `testdata/xxhash/README.md` records their provenance.
//!
//! Each test walks the frame layout only as far as the checksum fields, at the
//! offsets that the LZ4 and Zstandard frame format documents give, and does
//! not decompress.

#![cfg(any(feature = "xxh32", feature = "xxh64"))]

const CONTENT: &[u8] = include_bytes!("../testdata/xxhash/content.bin");

fn u32_le(bytes: &[u8], at: usize) -> u32 {
  u32::from_le_bytes(bytes[at..at.strict_add(4)].try_into().expect("four bytes"))
}

/// LZ4 frame: magic, FLG, BD, an optional 8-byte content size, then HC, the
/// second byte of the seed-zero XXH32 of the descriptor. Each block is a
/// 4-byte size word (high bit: stored uncompressed), its data, and with
/// FLG.B.Checksum the XXH32 of that data. A zero size word ends the blocks;
/// with FLG.C.Checksum the XXH32 of the content follows. Every checksum is
/// little-endian.
#[cfg(feature = "xxh32")]
#[test]
fn lz4_descriptor_block_and_content_checksums_are_xxh32() {
  use rscrypto::{FastHash, Xxh32};

  let frame = include_bytes!("../testdata/xxhash/content.lz4");
  assert_eq!(u32_le(frame, 0), 0x184d_2204, "LZ4 frame magic");
  let flags = frame[4];
  assert_eq!(
    flags, 0b0111_1100,
    "version 01, independent blocks, block checksums, content size and checksum"
  );
  let descriptor = &frame[4..14];
  assert_eq!(frame[14], Xxh32::hash(descriptor).to_le_bytes()[1], "header checksum");

  let mut at = 15;
  let mut blocks = 0u32;
  let mut stored_uncompressed = 0u32;
  loop {
    let word = u32_le(frame, at);
    at = at.strict_add(4);
    if word == 0 {
      break;
    }
    let len = usize::try_from(word & 0x7fff_ffff).expect("block size fits usize");
    let data = &frame[at..at.strict_add(len)];
    at = at.strict_add(len);
    assert_eq!(u32_le(frame, at), Xxh32::hash(data), "block checksum at {at}");
    at = at.strict_add(4);
    blocks = blocks.strict_add(1);
    stored_uncompressed = stored_uncompressed.strict_add(word >> 31);
  }
  assert_eq!(
    (blocks, stored_uncompressed),
    (4, 1),
    "four blocks, one stored uncompressed"
  );
  assert_eq!(&frame[at..], Xxh32::hash(CONTENT).to_le_bytes(), "content checksum");
}

/// Zstandard frame: magic, then the frame header descriptor (FHD), an optional
/// window descriptor, dictionary ID, and content size, then blocks with 3-byte
/// headers. With FHD.Content_Checksum_flag, the low 32 bits of the seed-zero
/// XXH64 of the content follow the last block, little-endian.
#[cfg(feature = "xxh64")]
#[test]
fn zstd_content_checksum_is_low_half_of_xxh64() {
  use rscrypto::{FastHash, Xxh64};

  let frame = include_bytes!("../testdata/xxhash/content.zst");
  assert_eq!(u32_le(frame, 0), 0xfd2f_b528, "Zstandard frame magic");
  let descriptor = frame[4];
  assert_ne!(descriptor & 0b100, 0, "content checksum flag");
  let single_segment = descriptor & 0b10_0000 != 0;
  let window = usize::from(!single_segment);
  let dictionary_id = [0, 1, 2, 4][usize::from(descriptor & 0b11)];
  let content_size = match descriptor >> 6 {
    0 => usize::from(single_segment),
    1 => 2,
    2 => 4,
    _ => 8,
  };
  let mut at = 5usize
    .strict_add(window)
    .strict_add(dictionary_id)
    .strict_add(content_size);
  loop {
    let header = u32::from_le_bytes([frame[at], frame[at.strict_add(1)], frame[at.strict_add(2)], 0]);
    at = at.strict_add(3);
    let len = usize::try_from(header >> 3).expect("block size fits usize");
    // A run-length block stores one byte; raw and compressed blocks store `len`.
    at = at.strict_add(if (header >> 1) & 0b11 == 1 { 1 } else { len });
    if header & 1 == 1 {
      break;
    }
  }
  let stored = &frame[at..];
  assert_eq!(stored, &Xxh64::hash(CONTENT).to_le_bytes()[..4], "content checksum");

  // XXH3-64 of the same content has different low 32 bits, so this fixture
  // fails an implementation that computes XXH3 in place of XXH64.
  #[cfg(feature = "xxh3")]
  assert_ne!(stored, &rscrypto::Xxh3::hash(CONTENT).to_le_bytes()[..4]);
}
