#![allow(unused_imports)]

#[cfg(all(feature = "checksums", feature = "std"))]
use std::io::{Cursor, Read, Write};

#[cfg(feature = "hashes")]
use rscrypto::{
  AsconXof, Blake3, Digest, Sha3_224, Sha3_256, Sha3_384, Sha3_512, Sha224, Sha256, Sha384, Sha512, Sha512_256,
  Shake128, Shake256, Xof,
};

#[cfg(feature = "hashes")]
fn assert_digest_api<D>()
where
  D: Digest,
  D::Output: PartialEq + core::fmt::Debug,
{
  let mut h = D::new();
  h.update(b"abc");
  let expected = h.finalize();
  h.reset();
  h.update(b"abc");
  assert_eq!(h.finalize(), expected);
}

#[cfg(feature = "hashes")]
fn squeeze_32(mut reader: impl Xof) -> [u8; 32] {
  let mut out = [0u8; 32];
  reader.squeeze(&mut out);
  out
}

#[test]
#[cfg(feature = "hashes")]
fn all_digests_follow_new_update_finalize_reset() {
  assert_digest_api::<Sha224>();
  assert_digest_api::<Sha256>();
  assert_digest_api::<Sha384>();
  assert_digest_api::<Sha512>();
  assert_digest_api::<Sha512_256>();
  assert_digest_api::<Sha3_224>();
  assert_digest_api::<Sha3_256>();
  assert_digest_api::<Sha3_384>();
  assert_digest_api::<Sha3_512>();
  assert_digest_api::<Blake3>();
}

#[test]
#[cfg(feature = "hashes")]
fn all_xofs_follow_new_update_finalize_xof_and_xof() {
  macro_rules! assert_xof_api {
    ($ty:ty) => {{
      let data = b"abc";
      let mut h = <$ty>::new();
      h.update(data);
      let streaming = squeeze_32(h.finalize_xof());
      h.reset();
      let oneshot = squeeze_32(<$ty>::xof(data));
      assert_eq!(streaming, oneshot);
    }};
  }

  assert_xof_api!(Shake128);
  assert_xof_api!(Shake256);
  assert_xof_api!(Blake3);
  assert_xof_api!(AsconXof);
}

#[test]
#[cfg(all(feature = "checksums", feature = "std"))]
fn checksum_adapters_use_checksum() -> std::io::Result<()> {
  use rscrypto::{Checksum as _, Crc32C};

  let mut reader = Crc32C::reader(Cursor::new(b"abc".to_vec()));
  std::io::copy(&mut reader, &mut std::io::sink())?;
  assert_eq!(reader.checksum(), Crc32C::checksum(b"abc"));

  let mut writer = Crc32C::writer(Vec::new());
  writer.write_all(b"abc")?;
  assert_eq!(writer.checksum(), Crc32C::checksum(b"abc"));

  Ok(())
}
