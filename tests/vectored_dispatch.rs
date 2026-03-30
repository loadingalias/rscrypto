#![cfg(feature = "checksums")]

use rscrypto::{Checksum, Crc16Ccitt, Crc16Ibm, Crc24OpenPgp, Crc32, Crc32C, Crc64, Crc64Nvme};

#[test]
fn checksum_vectored_matches_oneshot() {
  let data = b"The quick brown fox jumps over the lazy dog";
  let (a, b) = data.split_at(7);
  let (b, c) = b.split_at(13);
  let bufs: [&[u8]; 3] = [a, b, c];

  assert_eq!(Crc16Ccitt::checksum_vectored(&bufs), Crc16Ccitt::checksum(data));
  assert_eq!(Crc16Ibm::checksum_vectored(&bufs), Crc16Ibm::checksum(data));
  assert_eq!(Crc24OpenPgp::checksum_vectored(&bufs), Crc24OpenPgp::checksum(data));
  assert_eq!(Crc32::checksum_vectored(&bufs), Crc32::checksum(data));
  assert_eq!(Crc32C::checksum_vectored(&bufs), Crc32C::checksum(data));
  assert_eq!(Crc64::checksum_vectored(&bufs), Crc64::checksum(data));
  assert_eq!(Crc64Nvme::checksum_vectored(&bufs), Crc64Nvme::checksum(data));
}

#[cfg(feature = "std")]
#[test]
fn checksum_io_slices_matches_oneshot() {
  use std::io::IoSlice;

  let data = b"The quick brown fox jumps over the lazy dog";
  let (a, b) = data.split_at(7);
  let (b, c) = b.split_at(13);
  let bufs = [IoSlice::new(a), IoSlice::new(b), IoSlice::new(c)];

  assert_eq!(Crc16Ccitt::checksum_io_slices(&bufs), Crc16Ccitt::checksum(data));
  assert_eq!(Crc16Ibm::checksum_io_slices(&bufs), Crc16Ibm::checksum(data));
  assert_eq!(Crc24OpenPgp::checksum_io_slices(&bufs), Crc24OpenPgp::checksum(data));
  assert_eq!(Crc32::checksum_io_slices(&bufs), Crc32::checksum(data));
  assert_eq!(Crc32C::checksum_io_slices(&bufs), Crc32C::checksum(data));
  assert_eq!(Crc64::checksum_io_slices(&bufs), Crc64::checksum(data));
  assert_eq!(Crc64Nvme::checksum_io_slices(&bufs), Crc64Nvme::checksum(data));
}
