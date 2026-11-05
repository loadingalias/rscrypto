//! QUIC v1 cryptography for any transport (RFC 9000 and RFC 9001).
//!
//! The [`AES_128_GCM`], [`AES_256_GCM`], and [`CHACHA20_POLY1305`] algorithms
//! supply packet and header protection to rustls's QUIC API. With `aes-gcm` and
//! `hkdf`, this module also supplies the endpoint cryptography that rustls does
//! not own, independent of any transport crate:
//!
//! - [`initial_keys`] derives Initial packet keys (RFC 9001, section 5.2).
//! - [`retry_integrity_tag`] and [`is_valid_retry`] compute and check Retry
//!   integrity tags (RFC 9001, section 5.8).
//! - [`ConnectionIdKey`] issues connection IDs and recognizes its own IDs.
//! - [`StatelessResetKey`] derives stateless-reset tokens (RFC 9000, section 10.3).
//! - [`AddressTokenKey`] seals and opens Retry and NEW_TOKEN address tokens
//!   (RFC 9000, section 8.1).
//!
//! The `quinn` feature adapts these same types to Quinn.
//!
//! # Entropy
//!
//! Every random value comes from the caller's [`rustls::crypto::SecureRandom`]
//! source, passed to each call that draws it: key generation, connection-ID
//! nonces, and address-token nonces. A failing source fails the call, without a
//! fallback to the operating system, and leaves caller data unchanged. With
//! fixed sources, every output is deterministic. The source must provide
//! cryptographically secure random bytes; a source that repeats output repeats
//! connection IDs and address-token nonces.
//!
//! # Example
//!
//! A server answers a client's first Initial packet with a Retry: it issues a
//! connection ID, seals the client's original ID into an address token in its
//! own format, and appends the integrity tag. The client checks the tag, and the
//! server opens the token that the client returns.
//!
//! ```
//! # #[cfg(all(feature = "aes-gcm", feature = "hkdf", feature = "getrandom"))]
//! # {
//! use rscrypto_rustls::quic::{self, AddressTokenKey, ConnectionIdKey};
//! use rscrypto_rustls::random::SECURE_RANDOM;
//!
//! let ids = ConnectionIdKey::generate(SECURE_RANDOM)?;
//! let tokens = AddressTokenKey::generate(SECURE_RANDOM)?;
//! let client_id = [0x11; 8];
//! let original_id = [0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08];
//!
//! let mut token = original_id.to_vec();
//! tokens.seal(SECURE_RANDOM, &mut token, b"retry 192.0.2.7:50000")?;
//! let server_id = ids.issue(SECURE_RANDOM)?;
//! // Long header, Retry type, version 1, then both connection IDs and the token.
//! let mut retry = vec![0xf0, 0, 0, 0, 1, 8];
//! retry.extend_from_slice(&client_id);
//! retry.push(8);
//! retry.extend_from_slice(&server_id);
//! retry.extend_from_slice(&token);
//! let tag = quic::retry_integrity_tag(&original_id, &retry).expect("an 8-byte ID");
//! retry.extend_from_slice(&tag);
//! assert!(quic::is_valid_retry(&original_id, &retry));
//!
//! // The client's next Initial echoes the token and addresses the issued ID.
//! assert!(ids.validate(&server_id));
//! assert_eq!(tokens.open(&mut token, b"retry 192.0.2.7:50000")?, original_id);
//! # }
//! # Ok::<(), rustls::Error>(())
//! ```

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

#[cfg(all(feature = "aes-gcm", feature = "hkdf"))]
mod cid;
#[cfg(all(feature = "aes-gcm", feature = "hkdf"))]
mod initial;
#[cfg(all(feature = "aes-gcm", feature = "hkdf", test))]
mod tests;
#[cfg(all(feature = "aes-gcm", feature = "hkdf"))]
mod token;

#[cfg(all(feature = "aes-gcm", feature = "hkdf"))]
pub use cid::ConnectionIdKey;
#[cfg(all(feature = "aes-gcm", feature = "hkdf", feature = "quinn"))]
pub(crate) use cid::NONCE_LEN as CID_NONCE_LEN;
#[cfg(all(feature = "aes-gcm", feature = "hkdf"))]
pub use initial::{MAX_CONNECTION_ID_LEN, RETRY_INTEGRITY_TAG_LEN, initial_keys, is_valid_retry, retry_integrity_tag};
#[cfg(all(feature = "aes-gcm", feature = "hkdf", feature = "quinn"))]
pub(crate) use initial::{retry_tag, valid_retry};
#[cfg(all(feature = "aes-gcm", feature = "hkdf", feature = "quinn"))]
pub(crate) use token::TokenCipher;
#[cfg(all(feature = "aes-gcm", feature = "hkdf"))]
pub use token::{AddressTokenKey, StatelessResetKey};
