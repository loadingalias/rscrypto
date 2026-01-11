use checksum::{Checksum, Crc16Ccitt, Crc16Ibm, Crc24OpenPgp, Crc32, Crc32C, Crc64, Crc64Nvme, dispatch};

#[test]
fn dispatch_vectored_matches_oneshot() {
  let data = b"The quick brown fox jumps over the lazy dog";
  let (a, b) = data.split_at(7);
  let (b, c) = b.split_at(13);
  let bufs: [&[u8]; 3] = [a, b, c];

  assert_eq!(dispatch::crc16_ccitt_vectored(&bufs), Crc16Ccitt::checksum(data));
  assert_eq!(dispatch::crc16_ibm_vectored(&bufs), Crc16Ibm::checksum(data));
  assert_eq!(dispatch::crc24_openpgp_vectored(&bufs), Crc24OpenPgp::checksum(data));
  assert_eq!(dispatch::crc32_ieee_vectored(&bufs), Crc32::checksum(data));
  assert_eq!(dispatch::crc32c_vectored(&bufs), Crc32C::checksum(data));
  assert_eq!(dispatch::crc64_xz_vectored(&bufs), Crc64::checksum(data));
  assert_eq!(dispatch::crc64_nvme_vectored(&bufs), Crc64Nvme::checksum(data));
}

#[cfg(feature = "std")]
#[test]
fn dispatch_io_slices_matches_oneshot() {
  use std::io::IoSlice;

  let data = b"The quick brown fox jumps over the lazy dog";
  let (a, b) = data.split_at(7);
  let (b, c) = b.split_at(13);
  let bufs = [IoSlice::new(a), IoSlice::new(b), IoSlice::new(c)];

  assert_eq!(
    dispatch::std_io::crc16_ccitt_io_slices(&bufs),
    Crc16Ccitt::checksum(data)
  );
  assert_eq!(dispatch::std_io::crc16_ibm_io_slices(&bufs), Crc16Ibm::checksum(data));
  assert_eq!(
    dispatch::std_io::crc24_openpgp_io_slices(&bufs),
    Crc24OpenPgp::checksum(data)
  );
  assert_eq!(dispatch::std_io::crc32_ieee_io_slices(&bufs), Crc32::checksum(data));
  assert_eq!(dispatch::std_io::crc32c_io_slices(&bufs), Crc32C::checksum(data));
  assert_eq!(dispatch::std_io::crc64_xz_io_slices(&bufs), Crc64::checksum(data));
  assert_eq!(dispatch::std_io::crc64_nvme_io_slices(&bufs), Crc64Nvme::checksum(data));
}
