#![allow(unused_imports)]

#[cfg(all(feature = "checksums", feature = "std"))]
use std::io::{Cursor, Read, Write};

#[cfg(feature = "aead")]
use rscrypto::{
  Aead, Aegis256, Aegis256Key, ChaCha20Poly1305, ChaCha20Poly1305Key, VerificationError, XChaCha20Poly1305,
  XChaCha20Poly1305Key,
  aead::{AeadBufferError, Nonce96, Nonce192, Nonce256},
};
#[cfg(feature = "hashes")]
use rscrypto::{
  AsconCxof128, AsconXof, Blake3, Cshake256, Digest, Sha3_224, Sha3_256, Sha3_384, Sha3_512, Sha224, Sha256, Sha384,
  Sha512, Sha512_256, Shake128, Shake256, Xof,
};
#[cfg(feature = "auth")]
use rscrypto::{HmacSha256, Kmac256, Mac};

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

#[cfg(feature = "auth")]
fn assert_mac_api<M>()
where
  M: Mac,
  M::Tag: PartialEq + core::fmt::Debug,
{
  let key = b"api-consistency-key";
  let mut mac = M::new(key);
  mac.update(b"abc");
  let expected = mac.finalize();
  mac.reset();
  mac.update(b"abc");
  assert_eq!(mac.finalize(), expected);
  assert!(mac.verify(&expected).is_ok());
}

#[cfg(feature = "aead")]
#[derive(Clone, PartialEq, Eq)]
struct ApiKey([u8; 32]);

#[cfg(feature = "aead")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ApiTag([u8; 16]);

#[cfg(feature = "aead")]
impl AsRef<[u8]> for ApiTag {
  fn as_ref(&self) -> &[u8] {
    &self.0
  }
}

#[cfg(feature = "aead")]
#[derive(Clone)]
struct ApiAead {
  key: ApiKey,
}

#[cfg(feature = "aead")]
impl ApiAead {
  fn mask(&self, nonce: &Nonce96, aad: &[u8]) -> u8 {
    self.key.0[0] ^ nonce.as_bytes()[0] ^ (aad.len() as u8)
  }

  fn compute_tag(&self, nonce: &Nonce96, aad: &[u8], ciphertext: &[u8]) -> ApiTag {
    let mut tag = [0u8; 16];
    tag[0] = self.mask(nonce, aad);
    tag[1] = ciphertext.len() as u8;
    for (index, &byte) in ciphertext.iter().enumerate() {
      tag[index % 14 + 2] ^= byte;
    }
    ApiTag(tag)
  }
}

#[cfg(feature = "aead")]
impl Aead for ApiAead {
  const KEY_SIZE: usize = 32;
  const NONCE_SIZE: usize = Nonce96::LENGTH;
  const TAG_SIZE: usize = 16;

  type Key = ApiKey;
  type Nonce = Nonce96;
  type Tag = ApiTag;

  fn new(key: &Self::Key) -> Self {
    Self { key: key.clone() }
  }

  fn tag_from_slice(bytes: &[u8]) -> Result<Self::Tag, AeadBufferError> {
    if bytes.len() != Self::TAG_SIZE {
      return Err(AeadBufferError::new());
    }

    let mut tag = [0u8; Self::TAG_SIZE];
    tag.copy_from_slice(bytes);
    Ok(ApiTag(tag))
  }

  fn encrypt_in_place(&self, nonce: &Self::Nonce, aad: &[u8], buffer: &mut [u8]) -> Self::Tag {
    let mask = self.mask(nonce, aad);
    for byte in buffer.iter_mut() {
      *byte ^= mask;
    }
    self.compute_tag(nonce, aad, buffer)
  }

  fn decrypt_in_place(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &Self::Tag,
  ) -> Result<(), VerificationError> {
    if self.compute_tag(nonce, aad, buffer) != *tag {
      return Err(VerificationError::new());
    }

    let mask = self.mask(nonce, aad);
    for byte in buffer.iter_mut() {
      *byte ^= mask;
    }
    Ok(())
  }
}

#[cfg(feature = "aead")]
fn assert_aead_api<A>(key: A::Key, nonce: A::Nonce)
where
  A: Aead,
{
  let aead = A::new(&key);
  let plaintext = b"abc";

  let mut sealed = [0u8; 19];
  aead.encrypt(&nonce, b"aad", plaintext, &mut sealed).unwrap();

  let mut opened = [0u8; 3];
  aead.decrypt(&nonce, b"aad", &sealed, &mut opened).unwrap();
  assert_eq!(&opened, plaintext);

  let mut detached = *b"abc";
  let tag = aead.encrypt_in_place_detached(&nonce, b"aad", &mut detached);
  aead.decrypt_in_place(&nonce, b"aad", &mut detached, &tag).unwrap();
  assert_eq!(&detached, plaintext);
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

  let data = b"abc";
  let mut cshake = Cshake256::new(b"", b"ctx=v1");
  cshake.update(data);
  let streaming = squeeze_32(cshake.finalize_xof());
  cshake.reset();
  cshake.update(data);
  assert_eq!(streaming, squeeze_32(cshake.finalize_xof()));

  let mut cxof = AsconCxof128::new(b"ctx=v1").unwrap();
  cxof.update(data);
  let streaming = squeeze_32(cxof.finalize_xof());
  cxof.reset();
  cxof.update(data);
  assert_eq!(streaming, squeeze_32(cxof.finalize_xof()));
}

#[test]
#[cfg(feature = "auth")]
fn all_macs_follow_new_update_finalize_reset() {
  assert_mac_api::<HmacSha256>();

  let mut kmac = Kmac256::new(b"api-consistency-key", b"ctx=v1");
  kmac.update(b"abc");
  let expected = Kmac256::mac_array::<32>(b"api-consistency-key", b"ctx=v1", b"abc");
  let mut actual = [0u8; 32];
  kmac.finalize_into(&mut actual);
  assert_eq!(actual, expected);
  kmac.reset();
  kmac.update(b"abc");
  kmac.finalize_into(&mut actual);
  assert_eq!(actual, expected);
  assert!(kmac.verify_tag(&expected).is_ok());
}

#[test]
#[cfg(feature = "aead")]
fn all_aeads_follow_new_encrypt_decrypt_and_detached_aliases() {
  assert_aead_api::<ApiAead>(ApiKey([7u8; 32]), Nonce96::from_bytes([9u8; Nonce96::LENGTH]));
  assert_aead_api::<ChaCha20Poly1305>(
    ChaCha20Poly1305Key::from_bytes([2u8; ChaCha20Poly1305::KEY_SIZE]),
    Nonce96::from_bytes([4u8; Nonce96::LENGTH]),
  );
  assert_aead_api::<XChaCha20Poly1305>(
    XChaCha20Poly1305Key::from_bytes([3u8; XChaCha20Poly1305::KEY_SIZE]),
    Nonce192::from_bytes([5u8; Nonce192::LENGTH]),
  );
  assert_aead_api::<Aegis256>(
    Aegis256Key::from_bytes([6u8; Aegis256::KEY_SIZE]),
    Nonce256::from_bytes([7u8; Nonce256::LENGTH]),
  );
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
