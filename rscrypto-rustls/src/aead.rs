//! TLS record and QUIC packet protection with protocol-owned nonces.

use alloc::{boxed::Box, vec::Vec};

use rscrypto::aead::{Nonce96, expert::AeadWithNonce};
#[cfg(feature = "aes-gcm")]
use rscrypto::{Aes128Gcm, Aes128GcmKey, Aes128GcmTag, Aes256Gcm, Aes256GcmKey, Aes256GcmTag};
#[cfg(feature = "chacha20poly1305")]
use rscrypto::{ChaCha20Poly1305, ChaCha20Poly1305Key, ChaCha20Poly1305Tag};
use rustls::crypto::cipher::{
  AeadKey, InboundOpaque, Iv, Nonce, OutboundPlain, Record, RecordDecrypter, RecordEncrypter, Tls13AeadAlgorithm,
  UnsupportedOperationError, encode_record_header, make_tls13_aad,
};
use rustls::enums::ContentType;
use rustls::quic::{PacketKey, Tag};
use rustls::{ConnectionTrafficSecrets, Error};

const TAG_LEN: usize = 16;

#[derive(Clone, Copy)]
pub(crate) enum Algorithm {
  #[cfg(feature = "aes-gcm")]
  Aes128,
  #[cfg(feature = "aes-gcm")]
  Aes256,
  #[cfg(feature = "chacha20poly1305")]
  ChaCha20,
}

impl Algorithm {
  pub(crate) const fn key_len(self) -> usize {
    match self {
      #[cfg(feature = "aes-gcm")]
      Self::Aes128 => 16,
      #[cfg(feature = "aes-gcm")]
      Self::Aes256 => 32,
      #[cfg(feature = "chacha20poly1305")]
      Self::ChaCha20 => 32,
    }
  }

  fn cipher(self, key: &[u8]) -> Cipher {
    match self {
      #[cfg(feature = "aes-gcm")]
      Self::Aes128 => <&[u8; 16]>::try_from(key)
        .map(|bytes| {
          Cipher::Aes128(Aes128Gcm::new(&Aes128GcmKey::generate(|out| {
            out.copy_from_slice(bytes)
          })))
        })
        .unwrap_or(Cipher::InvalidKeyLength),
      #[cfg(feature = "aes-gcm")]
      Self::Aes256 => <&[u8; 32]>::try_from(key)
        .map(|bytes| {
          Cipher::Aes256(Aes256Gcm::new(&Aes256GcmKey::generate(|out| {
            out.copy_from_slice(bytes)
          })))
        })
        .unwrap_or(Cipher::InvalidKeyLength),
      #[cfg(feature = "chacha20poly1305")]
      Self::ChaCha20 => <&[u8; 32]>::try_from(key)
        .map(|bytes| {
          Cipher::ChaCha20(ChaCha20Poly1305::new(&ChaCha20Poly1305Key::generate(|out| {
            out.copy_from_slice(bytes)
          })))
        })
        .unwrap_or(Cipher::InvalidKeyLength),
    }
  }
}

impl Tls13AeadAlgorithm for Algorithm {
  fn encrypter(&self, key: AeadKey, iv: Iv) -> Box<dyn RecordEncrypter> {
    Box::new(RecordKey {
      cipher: self.cipher(key.as_ref()),
      iv,
    })
  }

  fn decrypter(&self, key: AeadKey, iv: Iv) -> Box<dyn RecordDecrypter> {
    Box::new(RecordKey {
      cipher: self.cipher(key.as_ref()),
      iv,
    })
  }

  fn key_len(&self) -> usize {
    (*self).key_len()
  }

  fn extract_keys(&self, key: AeadKey, iv: Iv) -> Result<ConnectionTrafficSecrets, UnsupportedOperationError> {
    if key.as_ref().len() != (*self).key_len() {
      return Err(UnsupportedOperationError);
    }
    Ok(match self {
      #[cfg(feature = "aes-gcm")]
      Self::Aes128 => ConnectionTrafficSecrets::Aes128Gcm { key, iv },
      #[cfg(feature = "aes-gcm")]
      Self::Aes256 => ConnectionTrafficSecrets::Aes256Gcm { key, iv },
      #[cfg(feature = "chacha20poly1305")]
      Self::ChaCha20 => ConnectionTrafficSecrets::Chacha20Poly1305 { key, iv },
    })
  }
}

#[cfg_attr(
  all(target_arch = "s390x", feature = "aes-gcm"),
  expect(
    clippy::large_enum_variant,
    reason = "the enclosing rustls key is already boxed; boxing variants adds another allocation"
  )
)]
enum Cipher {
  #[cfg(feature = "aes-gcm")]
  Aes128(Aes128Gcm),
  #[cfg(feature = "aes-gcm")]
  Aes256(Aes256Gcm),
  #[cfg(feature = "chacha20poly1305")]
  ChaCha20(ChaCha20Poly1305),
  InvalidKeyLength,
}

impl Cipher {
  fn seal(&self, nonce: Nonce, aad: &[u8], payload: &mut [u8]) -> Result<[u8; TAG_LEN], Error> {
    let nonce = Nonce96::from_bytes(nonce.to_array().map_err(|_| Error::EncryptError)?);
    match self {
      #[cfg(feature = "aes-gcm")]
      Self::Aes128(key) => key.encrypt_in_place(&nonce, aad, payload).map(|tag| *tag.as_bytes()),
      #[cfg(feature = "aes-gcm")]
      Self::Aes256(key) => key.encrypt_in_place(&nonce, aad, payload).map(|tag| *tag.as_bytes()),
      #[cfg(feature = "chacha20poly1305")]
      Self::ChaCha20(key) => key.encrypt_in_place(&nonce, aad, payload).map(|tag| *tag.as_bytes()),
      Self::InvalidKeyLength => return Err(Error::EncryptError),
    }
    .map_err(|_| Error::EncryptError)
  }

  fn open<'a>(&self, nonce: Nonce, aad: &[u8], payload: &'a mut [u8]) -> Result<&'a [u8], Error> {
    let plain_len = payload.len().checked_sub(TAG_LEN).ok_or(Error::DecryptError)?;
    let (body, tag) = payload.split_at_mut(plain_len);
    let tag: [u8; TAG_LEN] = (&*tag).try_into().map_err(|_| Error::DecryptError)?;
    let nonce = Nonce96::from_bytes(nonce.to_array().map_err(|_| Error::DecryptError)?);
    let result = match self {
      #[cfg(feature = "aes-gcm")]
      Self::Aes128(key) => key.decrypt_in_place(&nonce, aad, body, &Aes128GcmTag::from_bytes(tag)),
      #[cfg(feature = "aes-gcm")]
      Self::Aes256(key) => key.decrypt_in_place(&nonce, aad, body, &Aes256GcmTag::from_bytes(tag)),
      #[cfg(feature = "chacha20poly1305")]
      Self::ChaCha20(key) => key.decrypt_in_place(&nonce, aad, body, &ChaCha20Poly1305Tag::from_bytes(tag)),
      Self::InvalidKeyLength => return Err(Error::DecryptError),
    };
    if result.is_err() {
      rscrypto::traits::ct::zeroize(body);
      return Err(Error::DecryptError);
    }
    Ok(body)
  }
}

struct RecordKey {
  cipher: Cipher,
  iv: Iv,
}

impl RecordEncrypter for RecordKey {
  fn encrypt(&mut self, msg: Record<OutboundPlain<'_>>, seq: u64, out: &mut Vec<u8>) -> Result<(), Error> {
    // rustls fragments records at 2^14 bytes. Reject misuse before writing.
    if msg.payload.len() > 16_384 || matches!(self.cipher, Cipher::InvalidKeyLength) {
      return Err(Error::EncryptError);
    }
    let total_len = self.encrypted_payload_len(msg.payload.len());
    let header = encode_record_header(
      ContentType::ApplicationData,
      msg.version,
      u16::try_from(total_len).map_err(|_| Error::EncryptError)?,
    );
    let start = out.len();
    // One reservation: plaintext is never left behind in a reallocated buffer.
    out.reserve(header.len().strict_add(total_len));
    out.extend_from_slice(&header);
    let body = out.len();
    msg.payload.copy_to_vec(out);
    out.push(u8::from(msg.typ));
    match self.cipher.seal(
      Nonce::new(&self.iv, seq),
      &make_tls13_aad(ContentType::ApplicationData, msg.version.encode(), total_len),
      &mut out[body..],
    ) {
      Ok(tag) => {
        out.extend_from_slice(&tag);
        Ok(())
      }
      Err(error) => {
        rscrypto::traits::ct::zeroize(&mut out[start..]);
        out.truncate(start);
        Err(error)
      }
    }
  }

  fn encrypted_payload_len(&self, payload_len: usize) -> usize {
    payload_len.strict_add(1).strict_add(TAG_LEN)
  }
}

impl RecordDecrypter for RecordKey {
  fn decrypt<'a>(&mut self, mut msg: Record<InboundOpaque<'a>>, seq: u64) -> Result<Record<&'a [u8]>, Error> {
    let aad = make_tls13_aad(msg.typ, msg.version.encode(), msg.payload.len());
    let plain_len = self
      .cipher
      .open(Nonce::new(&self.iv, seq), &aad, &mut msg.payload)?
      .len();
    msg.payload.truncate(plain_len);
    msg.into_tls13_unpadded_record()
  }
}

pub(crate) struct PacketKeyImpl {
  cipher: Cipher,
  iv: Iv,
  algorithm: Algorithm,
}

impl PacketKeyImpl {
  pub(crate) fn new(algorithm: Algorithm, key: AeadKey, iv: Iv) -> Self {
    Self {
      cipher: algorithm.cipher(key.as_ref()),
      iv,
      algorithm,
    }
  }
}

impl PacketKey for PacketKeyImpl {
  fn encrypt_in_place(
    &self,
    packet_number: u64,
    header: &[u8],
    payload: &mut [u8],
    path_id: Option<u32>,
  ) -> Result<Tag, Error> {
    if packet_number >= 1 << 62 {
      return Err(Error::EncryptError);
    }
    self
      .cipher
      .seal(Nonce::quic(path_id, &self.iv, packet_number), header, payload)
      .map(|tag| Tag::from(tag.as_slice()))
  }

  fn decrypt_in_place<'a>(
    &self,
    packet_number: u64,
    header: &[u8],
    payload: &'a mut [u8],
    path_id: Option<u32>,
  ) -> Result<&'a [u8], Error> {
    if packet_number >= 1 << 62 {
      return Err(Error::DecryptError);
    }
    self
      .cipher
      .open(Nonce::quic(path_id, &self.iv, packet_number), header, payload)
  }

  fn tag_len(&self) -> usize {
    TAG_LEN
  }

  fn confidentiality_limit(&self) -> u64 {
    match self.algorithm {
      #[cfg(feature = "aes-gcm")]
      Algorithm::Aes128 | Algorithm::Aes256 => 1 << 23,
      #[cfg(feature = "chacha20poly1305")]
      Algorithm::ChaCha20 => u64::MAX,
    }
  }

  fn integrity_limit(&self) -> u64 {
    match self.algorithm {
      #[cfg(feature = "aes-gcm")]
      Algorithm::Aes128 | Algorithm::Aes256 => 1 << 52,
      #[cfg(feature = "chacha20poly1305")]
      Algorithm::ChaCha20 => 1 << 36,
    }
  }
}

/// AES-128-GCM record protection with rustls-owned nonces.
#[cfg(feature = "aes-gcm")]
pub static AES_128_GCM: &dyn Tls13AeadAlgorithm = &Algorithm::Aes128;
/// AES-256-GCM record protection with rustls-owned nonces.
#[cfg(feature = "aes-gcm")]
pub static AES_256_GCM: &dyn Tls13AeadAlgorithm = &Algorithm::Aes256;
/// ChaCha20-Poly1305 record protection with rustls-owned nonces.
#[cfg(feature = "chacha20poly1305")]
pub static CHACHA20_POLY1305: &dyn Tls13AeadAlgorithm = &Algorithm::ChaCha20;

#[cfg(all(test, feature = "hkdf", feature = "aes-gcm", feature = "chacha20poly1305"))]
#[path = "tests/records.rs"]
mod tests;
