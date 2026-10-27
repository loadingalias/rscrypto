//! TLS record and QUIC packet protection with protocol-owned nonces.

use alloc::{boxed::Box, vec::Vec};

use rscrypto::aead::{Nonce96, expert::AeadWithNonce};
#[cfg(feature = "aes-gcm")]
use rscrypto::{Aes128Gcm, Aes128GcmKey, Aes128GcmTag, Aes256Gcm, Aes256GcmKey, Aes256GcmTag};
#[cfg(feature = "chacha20poly1305")]
use rscrypto::{ChaCha20Poly1305, ChaCha20Poly1305Key, ChaCha20Poly1305Tag};
use rustls::crypto::cipher::{
  AeadKey, InboundOpaque, Iv, KeyBlockShape, NONCE_LEN, Nonce, OutboundPlain, Record, RecordDecrypter, RecordEncrypter,
  Tls12AeadAlgorithm, Tls13AeadAlgorithm, UnsupportedOperationError, encode_record_header, make_tls12_aad,
  make_tls13_aad,
};
use rustls::enums::ContentType;
use rustls::quic::{PacketKey, Tag};
use rustls::{ConnectionTrafficSecrets, Error};

const TAG_LEN: usize = 16;
// rustls fragments plaintext records at 2^14 bytes (RFC 5246 §6.2.1, RFC 9846 §5.1).
const MAX_FRAGMENT_LEN: usize = 16_384;

// Keep key/context temporaries in workers that return only a heap owner. The
// post-call scrub must cover their dead frames in the exact linked artifact.
const CONSTRUCTION_SCRUB_BYTES: usize = 32 * 1024;

pub(crate) struct ConstructionStackScrub;

impl Drop for ConstructionStackScrub {
  fn drop(&mut self) {
    scrub_construction_stack();
  }
}

#[inline(never)]
fn scrub_construction_stack() {
  let mut scratch = [0u8; CONSTRUCTION_SCRUB_BYTES];
  rscrypto::traits::ct::zeroize(&mut scratch);
  core::hint::black_box(&mut scratch);
}

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
        .unwrap_or(Cipher::InvalidKeyMaterial),
      #[cfg(feature = "aes-gcm")]
      Self::Aes256 => <&[u8; 32]>::try_from(key)
        .map(|bytes| {
          Cipher::Aes256(Aes256Gcm::new(&Aes256GcmKey::generate(|out| {
            out.copy_from_slice(bytes)
          })))
        })
        .unwrap_or(Cipher::InvalidKeyMaterial),
      #[cfg(feature = "chacha20poly1305")]
      Self::ChaCha20 => <&[u8; 32]>::try_from(key)
        .map(|bytes| {
          Cipher::ChaCha20(ChaCha20Poly1305::new(&ChaCha20Poly1305Key::generate(|out| {
            out.copy_from_slice(bytes)
          })))
        })
        .unwrap_or(Cipher::InvalidKeyMaterial),
    }
  }
}

impl Tls13AeadAlgorithm for Algorithm {
  fn encrypter(&self, key: AeadKey, iv: Iv) -> Box<dyn RecordEncrypter> {
    let _scrub = ConstructionStackScrub;
    RecordKey::boxed(*self, key.as_ref(), iv)
  }

  fn decrypter(&self, key: AeadKey, iv: Iv) -> Box<dyn RecordDecrypter> {
    let _scrub = ConstructionStackScrub;
    RecordKey::boxed(*self, key.as_ref(), iv)
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

/// TLS 1.2 record protection: AES-GCM as in RFC 5288, ChaCha20-Poly1305 as in RFC 7905.
#[derive(Clone, Copy)]
pub(crate) struct Tls12(Algorithm);

impl Tls12 {
  /// Nonce bytes each record carries: RFC 5288's 8-byte `nonce_explicit` for
  /// AES-GCM, none for ChaCha20-Poly1305.
  const fn explicit_nonce_len(self) -> usize {
    match self.0 {
      #[cfg(feature = "aes-gcm")]
      Algorithm::Aes128 | Algorithm::Aes256 => 8,
      #[cfg(feature = "chacha20poly1305")]
      Algorithm::ChaCha20 => 0,
    }
  }

  /// The key block's `fixed_iv`: the 4-byte AES-GCM salt or the 12-byte ChaCha20-Poly1305 IV.
  const fn fixed_iv_len(self) -> usize {
    NONCE_LEN.strict_sub(self.explicit_nonce_len())
  }

  /// `fixed ‖ explicit`, if both have this algorithm's key-block lengths.
  fn nonce_base(self, fixed: &[u8], explicit: &[u8]) -> Option<[u8; NONCE_LEN]> {
    let fixed_len = self.fixed_iv_len();
    if fixed.len() != fixed_len || explicit.len() != self.explicit_nonce_len() {
      return None;
    }
    let mut nonce = [0; NONCE_LEN];
    nonce[..fixed_len].copy_from_slice(fixed);
    nonce[fixed_len..].copy_from_slice(explicit);
    Some(nonce)
  }
}

impl Tls12AeadAlgorithm for Tls12 {
  fn encrypter(&self, key: AeadKey, iv: &[u8], extra: &[u8]) -> Box<dyn RecordEncrypter> {
    let _scrub = ConstructionStackScrub;
    // RFC 5288 §3 leaves the explicit part to the sender but requires it to be
    // unique per key. As in rustls's other providers, it is the key block's
    // `extra` bytes XOR the sequence number, the TLS 1.3 nonce construction.
    Tls12RecordKey::boxed(*self, key.as_ref(), self.nonce_base(iv, extra))
  }

  fn decrypter(&self, key: AeadKey, iv: &[u8]) -> Box<dyn RecordDecrypter> {
    let _scrub = ConstructionStackScrub;
    // An AES-GCM decrypter takes each record's explicit part from the wire.
    let explicit = [0; NONCE_LEN];
    Tls12RecordKey::boxed(
      *self,
      key.as_ref(),
      self.nonce_base(iv, &explicit[..self.explicit_nonce_len()]),
    )
  }

  fn key_block_shape(&self) -> KeyBlockShape {
    KeyBlockShape {
      enc_key_len: self.0.key_len(),
      fixed_iv_len: self.fixed_iv_len(),
      explicit_nonce_len: self.explicit_nonce_len(),
    }
  }

  fn extract_keys(
    &self,
    key: AeadKey,
    iv: &[u8],
    explicit: &[u8],
  ) -> Result<ConnectionTrafficSecrets, UnsupportedOperationError> {
    let nonce = self.nonce_base(iv, explicit).ok_or(UnsupportedOperationError)?;
    if key.as_ref().len() != self.0.key_len() {
      return Err(UnsupportedOperationError);
    }
    let iv = Iv::from(nonce);
    Ok(match self.0 {
      #[cfg(feature = "aes-gcm")]
      Algorithm::Aes128 => ConnectionTrafficSecrets::Aes128Gcm { key, iv },
      #[cfg(feature = "aes-gcm")]
      Algorithm::Aes256 => ConnectionTrafficSecrets::Aes256Gcm { key, iv },
      #[cfg(feature = "chacha20poly1305")]
      Algorithm::ChaCha20 => ConnectionTrafficSecrets::Chacha20Poly1305 { key, iv },
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
  /// Key or IV lengths that do not fit the algorithm; every record is rejected.
  InvalidKeyMaterial,
}

impl Cipher {
  fn seal_gather<'input>(
    &self,
    nonce: [u8; NONCE_LEN],
    aad: &[u8],
    plaintext: impl IntoIterator<Item = &'input [u8]>,
    plaintext_len: usize,
    out: &mut Vec<u8>,
  ) -> Result<[u8; TAG_LEN], Error> {
    let nonce = Nonce96::from_bytes(nonce);
    match self {
      #[cfg(feature = "aes-gcm")]
      Self::Aes128(key) => key
        .encrypt_gather_append(&nonce, aad, plaintext, plaintext_len, out)
        .map(|tag| *tag.as_bytes()),
      #[cfg(feature = "aes-gcm")]
      Self::Aes256(key) => key
        .encrypt_gather_append(&nonce, aad, plaintext, plaintext_len, out)
        .map(|tag| *tag.as_bytes()),
      #[cfg(feature = "chacha20poly1305")]
      Self::ChaCha20(key) => key
        .encrypt_gather_append(&nonce, aad, plaintext, plaintext_len, out)
        .map(|tag| *tag.as_bytes()),
      Self::InvalidKeyMaterial => return Err(Error::EncryptError),
    }
    .map_err(|_| Error::EncryptError)
  }

  fn seal(&self, nonce: [u8; NONCE_LEN], aad: &[u8], payload: &mut [u8]) -> Result<[u8; TAG_LEN], Error> {
    let nonce = Nonce96::from_bytes(nonce);
    match self {
      #[cfg(feature = "aes-gcm")]
      Self::Aes128(key) => key.encrypt_in_place(&nonce, aad, payload).map(|tag| *tag.as_bytes()),
      #[cfg(feature = "aes-gcm")]
      Self::Aes256(key) => key.encrypt_in_place(&nonce, aad, payload).map(|tag| *tag.as_bytes()),
      #[cfg(feature = "chacha20poly1305")]
      Self::ChaCha20(key) => key.encrypt_in_place(&nonce, aad, payload).map(|tag| *tag.as_bytes()),
      Self::InvalidKeyMaterial => return Err(Error::EncryptError),
    }
    .map_err(|_| Error::EncryptError)
  }

  fn open<'a>(&self, nonce: [u8; NONCE_LEN], aad: &[u8], payload: &'a mut [u8]) -> Result<&'a [u8], Error> {
    let plain_len = payload.len().checked_sub(TAG_LEN).ok_or(Error::DecryptError)?;
    let (body, tag) = payload.split_at_mut(plain_len);
    let tag: [u8; TAG_LEN] = (&*tag).try_into().map_err(|_| Error::DecryptError)?;
    let nonce = Nonce96::from_bytes(nonce);
    let result = match self {
      #[cfg(feature = "aes-gcm")]
      Self::Aes128(key) => key.decrypt_in_place(&nonce, aad, body, &Aes128GcmTag::from_bytes(tag)),
      #[cfg(feature = "aes-gcm")]
      Self::Aes256(key) => key.decrypt_in_place(&nonce, aad, body, &Aes256GcmTag::from_bytes(tag)),
      #[cfg(feature = "chacha20poly1305")]
      Self::ChaCha20(key) => key.decrypt_in_place(&nonce, aad, body, &ChaCha20Poly1305Tag::from_bytes(tag)),
      Self::InvalidKeyMaterial => return Err(Error::DecryptError),
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

impl RecordKey {
  #[inline(never)]
  fn boxed(algorithm: Algorithm, key: &[u8], iv: Iv) -> Box<Self> {
    Box::new(Self {
      cipher: algorithm.cipher(key),
      iv,
    })
  }
}

impl RecordEncrypter for RecordKey {
  fn encrypt(&mut self, msg: Record<OutboundPlain<'_>>, seq: u64, out: &mut Vec<u8>) -> Result<(), Error> {
    // rustls fragments records at 2^14 bytes. Reject misuse before writing.
    if msg.payload.len() > MAX_FRAGMENT_LEN || matches!(self.cipher, Cipher::InvalidKeyMaterial) {
      return Err(Error::EncryptError);
    }
    let nonce = Nonce::new(&self.iv, seq).to_array().map_err(|_| Error::EncryptError)?;
    let total_len = self.encrypted_payload_len(msg.payload.len());
    let header = encode_record_header(
      ContentType::ApplicationData,
      msg.version,
      u16::try_from(total_len).map_err(|_| Error::EncryptError)?,
    );
    let start = out.len();
    // One reservation covers the header, ciphertext, and tag. Plaintext is read
    // directly from the payload chunks and never enters the output allocation.
    out.reserve(header.len().strict_add(total_len));
    out.extend_from_slice(&header);
    let plaintext_len = msg.payload.len().strict_add(1);
    let content_type = [u8::from(msg.typ)];
    match self.cipher.seal_gather(
      nonce,
      &make_tls13_aad(ContentType::ApplicationData, msg.version.encode(), total_len),
      msg.payload.chunks().chain(core::iter::once(content_type.as_slice())),
      plaintext_len,
      out,
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
    let nonce = Nonce::new(&self.iv, seq).to_array().map_err(|_| Error::DecryptError)?;
    let plain_len = self.cipher.open(nonce, &aad, &mut msg.payload)?.len();
    msg.payload.truncate(plain_len);
    msg.into_tls13_unpadded_record()
  }
}

struct Tls12RecordKey {
  cipher: Cipher,
  // `fixed_iv ‖ explicit` from the key block. An AES-GCM decrypter keeps only
  // the salt; each record supplies its explicit part.
  nonce: [u8; NONCE_LEN],
  explicit_nonce_len: usize,
}

impl Tls12RecordKey {
  #[inline(never)]
  fn boxed(algorithm: Tls12, key: &[u8], nonce: Option<[u8; NONCE_LEN]>) -> Box<Self> {
    Box::new(Self {
      cipher: nonce.map_or(Cipher::InvalidKeyMaterial, |_| algorithm.0.cipher(key)),
      nonce: nonce.unwrap_or_default(),
      explicit_nonce_len: algorithm.explicit_nonce_len(),
    })
  }

  /// The nonce base with the big-endian sequence number XORed into its last
  /// eight bytes (RFC 7905 §2, and the GCM sender construction above).
  fn sequence_nonce(&self, seq: u64) -> [u8; NONCE_LEN] {
    let mut nonce = self.nonce;
    for (byte, seq) in nonce[NONCE_LEN.strict_sub(8)..].iter_mut().zip(seq.to_be_bytes()) {
      *byte ^= seq;
    }
    nonce
  }
}

impl RecordEncrypter for Tls12RecordKey {
  fn encrypt(&mut self, msg: Record<OutboundPlain<'_>>, seq: u64, out: &mut Vec<u8>) -> Result<(), Error> {
    let plaintext_len = msg.payload.len();
    // rustls fragments records at 2^14 bytes. Reject misuse before writing.
    if plaintext_len > MAX_FRAGMENT_LEN || matches!(self.cipher, Cipher::InvalidKeyMaterial) {
      return Err(Error::EncryptError);
    }
    let nonce = self.sequence_nonce(seq);
    let total_len = self.encrypted_payload_len(plaintext_len);
    let header = encode_record_header(
      msg.typ,
      msg.version,
      u16::try_from(total_len).map_err(|_| Error::EncryptError)?,
    );
    let start = out.len();
    // One reservation covers the header, explicit nonce, ciphertext, and tag.
    // Plaintext is read directly from the payload chunks and never enters the
    // output allocation.
    out.reserve(header.len().strict_add(total_len));
    out.extend_from_slice(&header);
    out.extend_from_slice(&nonce[NONCE_LEN.strict_sub(self.explicit_nonce_len)..]);
    match self.cipher.seal_gather(
      nonce,
      &make_tls12_aad(seq, msg.typ, msg.version.encode(), plaintext_len),
      msg.payload.chunks(),
      plaintext_len,
      out,
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
    payload_len.strict_add(self.explicit_nonce_len).strict_add(TAG_LEN)
  }
}

impl RecordDecrypter for Tls12RecordKey {
  fn decrypt<'a>(&mut self, mut msg: Record<InboundOpaque<'a>>, seq: u64) -> Result<Record<&'a [u8]>, Error> {
    let explicit_len = self.explicit_nonce_len;
    let plain_len = msg
      .payload
      .len()
      .checked_sub(explicit_len.strict_add(TAG_LEN))
      .ok_or(Error::DecryptError)?;
    // RFC 5246 §6.2.3 bounds the plaintext; reject before decrypting.
    if plain_len > MAX_FRAGMENT_LEN {
      return Err(Error::PeerSentOversizedRecord);
    }
    // The AAD carries the record's wire version, as the sender authenticated it.
    let aad = make_tls12_aad(seq, msg.typ, msg.version.version(), plain_len);
    let (explicit, body) = msg.payload.split_at_mut(explicit_len);
    let nonce = if explicit_len == 0 {
      self.sequence_nonce(seq)
    } else {
      let mut nonce = self.nonce;
      nonce[NONCE_LEN.strict_sub(explicit_len)..].copy_from_slice(explicit);
      nonce
    };
    self.cipher.open(nonce, &aad, body)?;
    Ok(msg.into_plain_record_range(explicit_len..explicit_len.strict_add(plain_len)))
  }
}

pub(crate) struct PacketKeyImpl {
  cipher: Cipher,
  iv: Iv,
  algorithm: Algorithm,
}

impl PacketKeyImpl {
  pub(crate) fn new(algorithm: Algorithm, key: &[u8], iv: Iv) -> Self {
    Self {
      cipher: algorithm.cipher(key),
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
    let nonce = Nonce::quic(path_id, &self.iv, packet_number)
      .to_array()
      .map_err(|_| Error::EncryptError)?;
    self
      .cipher
      .seal(nonce, header, payload)
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
    let nonce = Nonce::quic(path_id, &self.iv, packet_number)
      .to_array()
      .map_err(|_| Error::DecryptError)?;
    self.cipher.open(nonce, header, payload)
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

/// TLS 1.2 AES-128-GCM record protection (RFC 5288) with rustls-owned nonces.
#[cfg(feature = "aes-gcm")]
pub static TLS12_AES_128_GCM: &dyn Tls12AeadAlgorithm = &Tls12(Algorithm::Aes128);
/// TLS 1.2 AES-256-GCM record protection (RFC 5288) with rustls-owned nonces.
#[cfg(feature = "aes-gcm")]
pub static TLS12_AES_256_GCM: &dyn Tls12AeadAlgorithm = &Tls12(Algorithm::Aes256);
/// TLS 1.2 ChaCha20-Poly1305 record protection (RFC 7905) with rustls-owned nonces.
#[cfg(feature = "chacha20poly1305")]
pub static TLS12_CHACHA20_POLY1305: &dyn Tls12AeadAlgorithm = &Tls12(Algorithm::ChaCha20);

#[cfg(all(test, feature = "hkdf", feature = "aes-gcm", feature = "chacha20poly1305"))]
#[path = "tests/records.rs"]
mod tests;

#[cfg(all(test, feature = "aes-gcm", feature = "chacha20poly1305"))]
#[path = "tests/tls12_records.rs"]
mod tls12_tests;
