//! Deterministic invariant coverage that stays enabled under Miri.
//!
//! The randomized property tests in `tests/` and selected module-local proptests
//! are intentionally excluded from Miri because proptest's failure persistence
//! touches filesystem APIs that Miri isolation rejects. These tests mirror the
//! important invariants with boundary-heavy fixed inputs so the Miri lane still
//! checks real behavior.
//!
//! For checksums, the Miri lane covers one representative algorithm per width.
//! The full algorithm/polynomial matrix stays in the normal and property-test
//! lanes, where breadth is cheap and randomized input is viable.

#[cfg(feature = "checksums")]
use alloc::vec::Vec;

use crate::platform::Caps;
#[cfg(feature = "checksums")]
use crate::traits::{Checksum, ChecksumCombine};

#[cfg(feature = "checksums")]
const CRC_MIRI_LENGTHS: &[usize] = &[0, 1, 16, 17, 32, 33, 64, 65];

#[cfg(feature = "checksums")]
const CRC_MIRI_CHUNKS: &[usize] = &[1, 16];

#[cfg(feature = "checksums")]
fn deterministic_bytes(len: usize) -> Vec<u8> {
  let mut out = alloc::vec![0u8; len];
  let mut x = 0x243f_6a88_85a3_08d3u64;
  for (i, byte) in out.iter_mut().enumerate() {
    x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *byte = (x >> 56) as u8 ^ ((i as u8).rotate_left((i & 7) as u32));
  }
  out
}

#[cfg(feature = "checksums")]
fn interesting_splits(len: usize) -> Vec<usize> {
  let mut splits = alloc::vec![0, len / 2, len];
  if len >= 1 {
    splits.push(1);
    splits.push(len - 1);
  }
  if len >= 2 {
    splits.push(2);
  }
  if len >= 3 {
    splits.push(len / 3);
    splits.push((len * 2) / 3);
  }
  splits.sort_unstable();
  splits.dedup();
  splits
}

#[cfg(feature = "checksums")]
fn run_crc_shadow_suite<C>()
where
  C: Checksum + ChecksumCombine,
  C::Output: Eq + core::fmt::Debug,
{
  assert_crc_streaming_equals_oneshot::<C>(&[]);

  for &len in CRC_MIRI_LENGTHS {
    let data = deterministic_bytes(len);

    assert_crc_streaming_equals_oneshot::<C>(&data);
    assert_crc_finalize_idempotent::<C>(&data);
    assert_crc_reset::<C>(&data);
    assert_crc_combine_empty_suffix::<C>(&data);
    assert_crc_combine_empty_prefix::<C>(&data);

    if len <= 32 {
      assert_crc_streaming_byte_at_a_time::<C>(&data);
    }

    for split in interesting_splits(len) {
      assert_crc_combine_property::<C>(&data, split);
    }

    if matches!(len, 0 | 1 | 16 | 17) {
      assert_crc_combine_all_splits::<C>(&data);
    }

    for &chunk_size in CRC_MIRI_CHUNKS {
      assert_crc_streaming_chunked::<C>(&data, chunk_size);
    }

    if !data.is_empty() {
      assert_crc_streaming_and_combine::<C>(&data);
    }
  }
}

#[cfg(feature = "checksums")]
fn assert_crc_streaming_equals_oneshot<C>(data: &[u8])
where
  C: Checksum,
  C::Output: Eq + core::fmt::Debug,
{
  let oneshot = C::checksum(data);
  let mut hasher = C::new();
  hasher.update(data);
  assert_eq!(
    hasher.finalize(),
    oneshot,
    "streaming != oneshot for len={}",
    data.len()
  );
}

#[cfg(feature = "checksums")]
fn assert_crc_streaming_byte_at_a_time<C>(data: &[u8])
where
  C: Checksum,
  C::Output: Eq + core::fmt::Debug,
{
  let oneshot = C::checksum(data);
  let mut hasher = C::new();
  for &byte in data {
    hasher.update(&[byte]);
  }
  assert_eq!(
    hasher.finalize(),
    oneshot,
    "byte-at-a-time != oneshot for len={}",
    data.len()
  );
}

#[cfg(feature = "checksums")]
fn assert_crc_streaming_chunked<C>(data: &[u8], chunk_size: usize)
where
  C: Checksum,
  C::Output: Eq + core::fmt::Debug,
{
  let oneshot = C::checksum(data);
  let mut hasher = C::new();
  for chunk in data.chunks(chunk_size) {
    hasher.update(chunk);
  }
  assert_eq!(
    hasher.finalize(),
    oneshot,
    "chunked streaming != oneshot for len={} chunk={chunk_size}",
    data.len()
  );
}

#[cfg(feature = "checksums")]
fn assert_crc_finalize_idempotent<C>(data: &[u8])
where
  C: Checksum,
  C::Output: Eq + core::fmt::Debug,
{
  let mut hasher = C::new();
  hasher.update(data);
  let first = hasher.finalize();
  let second = hasher.finalize();
  let third = hasher.finalize();
  assert_eq!(first, second, "finalize not idempotent for len={}", data.len());
  assert_eq!(second, third, "finalize not idempotent for len={}", data.len());
}

#[cfg(feature = "checksums")]
fn assert_crc_reset<C>(data: &[u8])
where
  C: Checksum,
  C::Output: Eq + core::fmt::Debug,
{
  let oneshot = C::checksum(data);
  let mut hasher = C::new();
  hasher.update(b"discard me");
  hasher.reset();
  hasher.update(data);
  assert_eq!(hasher.finalize(), oneshot, "reset failed for len={}", data.len());
}

#[cfg(feature = "checksums")]
fn assert_crc_combine_property<C>(data: &[u8], split: usize)
where
  C: Checksum + ChecksumCombine,
  C::Output: Eq + core::fmt::Debug,
{
  let split = split.min(data.len());
  let (a, b) = data.split_at(split);
  let combined = C::combine(C::checksum(a), C::checksum(b), b.len());
  assert_eq!(
    combined,
    C::checksum(data),
    "combine failed for len={} split={split}",
    data.len()
  );
}

#[cfg(feature = "checksums")]
fn assert_crc_combine_all_splits<C>(data: &[u8])
where
  C: Checksum + ChecksumCombine,
  C::Output: Eq + core::fmt::Debug,
{
  for split in 0..=data.len() {
    assert_crc_combine_property::<C>(data, split);
  }
}

#[cfg(feature = "checksums")]
fn assert_crc_combine_empty_suffix<C>(data: &[u8])
where
  C: Checksum + ChecksumCombine,
  C::Output: Eq + core::fmt::Debug,
{
  let combined = C::combine(C::checksum(data), C::checksum(&[]), 0);
  assert_eq!(
    combined,
    C::checksum(data),
    "empty suffix combine failed for len={}",
    data.len()
  );
}

#[cfg(feature = "checksums")]
fn assert_crc_combine_empty_prefix<C>(data: &[u8])
where
  C: Checksum + ChecksumCombine,
  C::Output: Eq + core::fmt::Debug,
{
  let combined = C::combine(C::checksum(&[]), C::checksum(data), data.len());
  assert_eq!(
    combined,
    C::checksum(data),
    "empty prefix combine failed for len={}",
    data.len()
  );
}

#[cfg(feature = "checksums")]
fn assert_crc_streaming_and_combine<C>(data: &[u8])
where
  C: Checksum + ChecksumCombine,
  C::Output: Eq + core::fmt::Debug,
{
  let mid = data.len() / 2;
  let (left, right) = data.split_at(mid);
  let mut hasher = C::new();
  hasher.update(left);
  let combined = C::combine(hasher.finalize(), C::checksum(right), right.len());
  assert_eq!(
    combined,
    C::checksum(data),
    "streaming+combine failed for len={}",
    data.len()
  );
}

macro_rules! shadow_crc_test {
  ($feature:literal, $name:ident, $ty:ty) => {
    #[test]
    #[cfg(feature = $feature)]
    fn $name() {
      run_crc_shadow_suite::<$ty>();
    }
  };
}

shadow_crc_test!("crc16", miri_shadow_crc16_ccitt_invariants, crate::Crc16Ccitt);
shadow_crc_test!("crc32", miri_shadow_crc32c_invariants, crate::Crc32C);
shadow_crc_test!("crc64", miri_shadow_crc64_nvme_invariants, crate::Crc64Nvme);

#[test]
fn miri_shadow_caps_boolean_algebra() {
  let samples = [
    Caps::NONE,
    Caps::bit(0),
    Caps::bit(63),
    Caps::bit(64),
    Caps::bit(127),
    Caps::bit(128),
    Caps::bit(191),
    Caps::bit(192),
    Caps::bit(255),
    Caps::from_words([0x0123_4567_89ab_cdef, 0, 0, 0]),
    Caps::from_words([0, 0xf0f0_f0f0_0f0f_0f0f, 0, 0]),
    Caps::from_words([
      0xffff_0000_ffff_0000,
      0x0000_ffff_0000_ffff,
      0xaaaa_5555_aaaa_5555,
      0x1357_9bdf_2468_ace0,
    ]),
  ];

  for &caps in &samples {
    assert_eq!(caps | Caps::NONE, caps, "union identity failed for {caps:?}");
    assert_eq!(
      caps & Caps::NONE,
      Caps::NONE,
      "intersection absorbing failed for {caps:?}"
    );
    assert!(caps.has(caps), "self containment failed for {caps:?}");
  }

  for &a in &samples {
    for &b in &samples {
      assert_eq!(a | b, b | a, "union commutativity failed for {a:?} and {b:?}");
      assert_eq!(a & b, b & a, "intersection commutativity failed for {a:?} and {b:?}");

      let union = a | b;
      assert!(union.has(a), "union must contain lhs: {a:?} | {b:?}");
      assert!(union.has(b), "union must contain rhs: {a:?} | {b:?}");
    }
  }

  for &a in &samples {
    for &b in &samples {
      for &c in &samples {
        assert_eq!((a | b) | c, a | (b | c), "union associativity failed");
        assert_eq!((a & b) & c, a & (b & c), "intersection associativity failed");
      }
    }
  }
}
