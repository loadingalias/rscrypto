//! Fuzz target for vectored CRC APIs.
//!
//! Verifies that:
//! - `update_vectored()` matches oneshot for arbitrary segmentations
//! - `dispatch::*_vectored()` matches the type-level oneshot
//! - `IoSlice` helpers (std) match oneshot

#![no_main]

use arbitrary::Arbitrary;
use checksum::{
  Checksum, Crc16Ccitt, Crc16Ibm, Crc24OpenPgp, Crc32, Crc32C, Crc64, Crc64Nvme, dispatch,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
  data: Vec<u8>,
  /// Chunk sizes used to create the vectored slices.
  chunk_sizes: Vec<usize>,
  /// Optional grouping for multiple `update_vectored` calls.
  group_sizes: Vec<usize>,
}

fuzz_target!(|input: Input| {
  let data = input.data;
  let slices = split_into_slices(&data, &input.chunk_sizes);

  // Type-level: oneshot vs vectored.
  assert_eq!(
    checksum_via_update_vectored::<Crc64>(&slices),
    Crc64::checksum(&data),
    "crc64 update_vectored mismatch"
  );
  assert_eq!(
    checksum_via_update_vectored::<Crc64Nvme>(&slices),
    Crc64Nvme::checksum(&data),
    "crc64/nvme update_vectored mismatch"
  );
  assert_eq!(
    checksum_via_update_vectored::<Crc32>(&slices),
    Crc32::checksum(&data),
    "crc32 update_vectored mismatch"
  );
  assert_eq!(
    checksum_via_update_vectored::<Crc32C>(&slices),
    Crc32C::checksum(&data),
    "crc32c update_vectored mismatch"
  );
  assert_eq!(
    checksum_via_update_vectored::<Crc16Ccitt>(&slices),
    Crc16Ccitt::checksum(&data),
    "crc16/ccitt update_vectored mismatch"
  );
  assert_eq!(
    checksum_via_update_vectored::<Crc16Ibm>(&slices),
    Crc16Ibm::checksum(&data),
    "crc16/ibm update_vectored mismatch"
  );
  assert_eq!(
    checksum_via_update_vectored::<Crc24OpenPgp>(&slices),
    Crc24OpenPgp::checksum(&data),
    "crc24/openpgp update_vectored mismatch"
  );

  // Type-level: multiple vectored updates.
  assert_eq!(
    checksum_via_grouped_update_vectored::<Crc64>(&slices, &input.group_sizes),
    Crc64::checksum(&data),
    "crc64 grouped update_vectored mismatch"
  );
  assert_eq!(
    checksum_via_grouped_update_vectored::<Crc32>(&slices, &input.group_sizes),
    Crc32::checksum(&data),
    "crc32 grouped update_vectored mismatch"
  );

  // Dispatch: oneshot vectored APIs.
  assert_eq!(
    dispatch::crc64_xz_vectored(&slices),
    Crc64::checksum(&data),
    "dispatch crc64/xz vectored mismatch"
  );
  assert_eq!(
    dispatch::crc64_nvme_vectored(&slices),
    Crc64Nvme::checksum(&data),
    "dispatch crc64/nvme vectored mismatch"
  );
  assert_eq!(
    dispatch::crc32_ieee_vectored(&slices),
    Crc32::checksum(&data),
    "dispatch crc32/ieee vectored mismatch"
  );
  assert_eq!(
    dispatch::crc32c_vectored(&slices),
    Crc32C::checksum(&data),
    "dispatch crc32c vectored mismatch"
  );
  assert_eq!(
    dispatch::crc16_ccitt_vectored(&slices),
    Crc16Ccitt::checksum(&data),
    "dispatch crc16/ccitt vectored mismatch"
  );
  assert_eq!(
    dispatch::crc16_ibm_vectored(&slices),
    Crc16Ibm::checksum(&data),
    "dispatch crc16/ibm vectored mismatch"
  );
  assert_eq!(
    dispatch::crc24_openpgp_vectored(&slices),
    Crc24OpenPgp::checksum(&data),
    "dispatch crc24/openpgp vectored mismatch"
  );

  // IoSlice helpers (fuzz targets run with std).
  use std::io::IoSlice;
  let io_slices: Vec<IoSlice<'_>> = slices.iter().map(|s| IoSlice::new(*s)).collect();

  let mut h64 = Crc64::new();
  h64.update_io_slices(&io_slices);
  assert_eq!(h64.finalize(), Crc64::checksum(&data), "crc64 update_io_slices mismatch");

  assert_eq!(
    dispatch::std_io::crc64_xz_io_slices(&io_slices),
    Crc64::checksum(&data),
    "dispatch crc64/xz io_slices mismatch"
  );
  assert_eq!(
    dispatch::std_io::crc32_ieee_io_slices(&io_slices),
    Crc32::checksum(&data),
    "dispatch crc32/ieee io_slices mismatch"
  );
});

fn split_into_slices<'a>(data: &'a [u8], chunk_sizes: &[usize]) -> Vec<&'a [u8]> {
  if data.is_empty() {
    return Vec::new();
  }

  let mut slices = Vec::new();
  let mut offset = 0usize;
  let mut chunk_idx = 0usize;

  while offset < data.len() {
    let chunk_size = if chunk_sizes.is_empty() {
      1usize
    } else {
      let idx = chunk_idx.strict_rem(chunk_sizes.len());
      chunk_sizes[idx].strict_rem(512).max(1)
    };

    let end = offset.strict_add(chunk_size).min(data.len());
    slices.push(&data[offset..end]);
    offset = end;
    chunk_idx = chunk_idx.strict_add(1);
  }

  slices
}

fn checksum_via_update_vectored<C: Checksum>(slices: &[&[u8]]) -> C::Output {
  let mut h = C::new();
  h.update_vectored(slices);
  h.finalize()
}

fn checksum_via_grouped_update_vectored<C: Checksum>(slices: &[&[u8]], group_sizes: &[usize]) -> C::Output {
  let mut h = C::new();

  if slices.is_empty() {
    h.update(b"");
    return h.finalize();
  }

  let mut offset = 0usize;
  let mut group_idx = 0usize;

  while offset < slices.len() {
    let group_len = if group_sizes.is_empty() {
      1usize
    } else {
      let idx = group_idx.strict_rem(group_sizes.len());
      group_sizes[idx].strict_rem(32).max(1)
    };

    let end = offset.strict_add(group_len).min(slices.len());
    h.update_vectored(&slices[offset..end]);
    offset = end;
    group_idx = group_idx.strict_add(1);
  }

  h.finalize()
}
