//! QUIC packet and header protection (RFC 9001).

use alloc::{boxed::Box, string::ToString};

use crate::aead::{Algorithm, ConstructionStackScrub, PacketKeyImpl};
#[cfg(feature = "aes-gcm")]
use rscrypto::aead::expert::header_protection::{
  Aes128HeaderProtection, Aes128HeaderProtectionKey, Aes256HeaderProtection, Aes256HeaderProtectionKey,
};
#[cfg(feature = "chacha20poly1305")]
use rscrypto::aead::expert::header_protection::{ChaCha20HeaderProtection, ChaCha20HeaderProtectionKey};
use rustls::Error;
use rustls::crypto::cipher::{AeadKey, Iv};
use rustls::quic::{HeaderProtectionKey, PacketKey};

const SAMPLE_LEN: usize = 16;
const MAX_PACKET_NUMBER_LEN: usize = 4;

impl rustls::quic::Algorithm for Algorithm {
  fn packet_key(&self, key: AeadKey, iv: Iv) -> Box<dyn PacketKey> {
    let _scrub = ConstructionStackScrub;
    boxed_packet_key(*self, key.as_ref(), iv)
  }

  fn header_protection_key(&self, key: AeadKey) -> Box<dyn HeaderProtectionKey> {
    let _scrub = ConstructionStackScrub;
    boxed_header_key(*self, key.as_ref())
  }

  fn aead_key_len(&self) -> usize {
    (*self).key_len()
  }

  fn fips(&self) -> rustls::pki_types::FipsStatus {
    // rscrypto is not currently a FIPS-validated module.
    rustls::pki_types::FipsStatus::Unvalidated
  }
}

#[inline(never)]
fn boxed_packet_key(algorithm: Algorithm, key: &[u8], iv: Iv) -> Box<PacketKeyImpl> {
  Box::new(PacketKeyImpl::new(algorithm, key, iv))
}

#[inline(never)]
fn boxed_header_key(algorithm: Algorithm, key: &[u8]) -> Box<HeaderKey> {
  Box::new(HeaderKey::from_bytes(algorithm, key))
}

#[cfg_attr(
  all(target_arch = "s390x", feature = "aes-gcm"),
  expect(
    clippy::large_enum_variant,
    reason = "rustls boxes this key; boxing variants adds another allocation"
  )
)]
enum HeaderKey {
  #[cfg(feature = "aes-gcm")]
  Aes128(Aes128HeaderProtection),
  #[cfg(feature = "aes-gcm")]
  Aes256(Aes256HeaderProtection),
  #[cfg(feature = "chacha20poly1305")]
  ChaCha20(ChaCha20HeaderProtection),
  InvalidKeyLength,
}

impl HeaderKey {
  fn from_bytes(algorithm: Algorithm, bytes: &[u8]) -> Self {
    match algorithm {
      #[cfg(feature = "aes-gcm")]
      Algorithm::Aes128 => match <[u8; 16]>::try_from(bytes) {
        Ok(bytes) => {
          let key = Aes128HeaderProtectionKey::from_bytes(bytes);
          Self::Aes128(Aes128HeaderProtection::new(&key))
        }
        Err(_) => Self::InvalidKeyLength,
      },
      #[cfg(feature = "aes-gcm")]
      Algorithm::Aes256 => match <[u8; 32]>::try_from(bytes) {
        Ok(bytes) => {
          let key = Aes256HeaderProtectionKey::from_bytes(bytes);
          Self::Aes256(Aes256HeaderProtection::new(&key))
        }
        Err(_) => Self::InvalidKeyLength,
      },
      #[cfg(feature = "chacha20poly1305")]
      Algorithm::ChaCha20 => match <[u8; 32]>::try_from(bytes) {
        Ok(bytes) => {
          let key = ChaCha20HeaderProtectionKey::from_bytes(bytes);
          Self::ChaCha20(ChaCha20HeaderProtection::new(&key))
        }
        Err(_) => Self::InvalidKeyLength,
      },
    }
  }

  fn mask(&self, sample: &[u8; SAMPLE_LEN]) -> Result<[u8; 5], Error> {
    match self {
      #[cfg(feature = "aes-gcm")]
      Self::Aes128(key) => Ok(key.mask(sample)),
      #[cfg(feature = "aes-gcm")]
      Self::Aes256(key) => Ok(key.mask(sample)),
      #[cfg(feature = "chacha20poly1305")]
      Self::ChaCha20(key) => Ok(key.mask(sample)),
      Self::InvalidKeyLength => Err(Error::General(
        "QUIC header-protection key has invalid length".to_string(),
      )),
    }
  }

  fn xor_in_place(
    &self,
    sample: &[u8],
    first: &mut u8,
    packet_number: &mut [u8],
    decrypting: bool,
  ) -> Result<(), Error> {
    let sample: &[u8; SAMPLE_LEN] = sample
      .try_into()
      .map_err(|_| Error::General("sample of invalid length".to_string()))?;
    if packet_number.len() > MAX_PACKET_NUMBER_LEN {
      return Err(Error::General("packet number too long".to_string()));
    }

    let mask = self.mask(sample)?;
    const LONG_HEADER_FORM: u8 = 0x80;
    let first_bits = if *first & LONG_HEADER_FORM == LONG_HEADER_FORM {
      0x0f
    } else {
      0x1f
    };
    let first_plain = if decrypting {
      *first ^ (mask[0] & first_bits)
    } else {
      *first
    };
    let packet_number_len = usize::from(first_plain & 0x03).strict_add(1);

    if packet_number.len() < packet_number_len {
      return Err(Error::General("truncated QUIC packet number".to_string()));
    }
    *first ^= mask[0] & first_bits;
    for (byte, mask_byte) in packet_number.iter_mut().zip(&mask[1..]).take(packet_number_len) {
      *byte ^= mask_byte;
    }
    Ok(())
  }
}

impl HeaderProtectionKey for HeaderKey {
  fn encrypt_in_place(&self, sample: &[u8], first: &mut u8, packet_number: &mut [u8]) -> Result<(), Error> {
    self.xor_in_place(sample, first, packet_number, false)
  }

  fn decrypt_in_place(&self, sample: &[u8], first: &mut u8, packet_number: &mut [u8]) -> Result<(), Error> {
    self.xor_in_place(sample, first, packet_number, true)
  }

  fn sample_len(&self) -> usize {
    SAMPLE_LEN
  }
}

/// AES-128-GCM packets and AES-128 header protection.
#[cfg(feature = "aes-gcm")]
pub static AES_128_GCM: &dyn rustls::quic::Algorithm = &Algorithm::Aes128;
/// AES-256-GCM packets and AES-256 header protection.
#[cfg(feature = "aes-gcm")]
pub static AES_256_GCM: &dyn rustls::quic::Algorithm = &Algorithm::Aes256;
/// ChaCha20-Poly1305 packets and ChaCha20 header protection.
#[cfg(feature = "chacha20poly1305")]
pub static CHACHA20_POLY1305: &dyn rustls::quic::Algorithm = &Algorithm::ChaCha20;
