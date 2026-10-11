use alloc::{boxed::Box, vec::Vec};
use core::sync::atomic::{AtomicBool, Ordering};
use quinn_proto::crypto::{AeadKey, CryptoError, HandshakeTokenKey, HmacKey};
use rscrypto::aead::{Nonce96, expert::AeadWithNonce};
use rscrypto::{Aes256Gcm, Aes256GcmKey, Aes256GcmTag, HkdfSha256, HmacSha256, HmacSha256Tag, Mac, SecretBytes};
use rustls::crypto::GetRandomFailed;

const MAX_TOKEN_PLAINTEXT: usize = 4096;

pub(super) struct ResetKey(SecretBytes<32>);

impl ResetKey {
  pub(super) fn generate_with(
    fill: impl FnOnce(&mut [u8; 32]) -> Result<(), GetRandomFailed>,
  ) -> Result<Self, GetRandomFailed> {
    SecretBytes::try_fill_with(fill).map(Self)
  }
}

impl HmacKey for ResetKey {
  fn sign(&self, data: &[u8], output: &mut [u8]) {
    output.copy_from_slice(HmacSha256::mac(self.0.as_bytes(), data).as_ref());
  }

  fn signature_len(&self) -> usize {
    32
  }

  fn verify(&self, data: &[u8], signature: &[u8]) -> Result<(), CryptoError> {
    let signature = HmacSha256Tag::from_bytes(signature.try_into().map_err(|_| CryptoError)?);
    let mut mac = HmacSha256::new(self.0.as_bytes());
    mac.update(data);
    mac.verify(&signature).map_err(|_| CryptoError)
  }
}

pub(super) struct TokenKey(HkdfSha256);

impl TokenKey {
  pub(super) fn generate_with(
    fill: impl FnOnce(&mut [u8; 32]) -> Result<(), GetRandomFailed>,
  ) -> Result<Self, GetRandomFailed> {
    let key = SecretBytes::try_fill_with(fill)?;
    Ok(Self(HkdfSha256::from_prk(key.as_bytes())))
  }
}

impl HandshakeTokenKey for TokenKey {
  fn aead_from_hkdf(&self, random_bytes: &[u8]) -> Box<dyn AeadKey> {
    let key = SecretBytes::<32>::try_fill_with(|out| {
      self
        .0
        .expand_vectored(&[b"rscrypto-rustls QUIC token v1", random_bytes], out)
    })
    .expect("one SHA-256 block is a valid HKDF output");
    let key = Aes256GcmKey::generate(|out| out.copy_from_slice(key.as_bytes()));
    Box::new(TokenCipher {
      cipher: Aes256Gcm::new(&key),
      sealed: AtomicBool::new(false),
    })
  }
}

struct TokenCipher {
  cipher: Aes256Gcm,
  sealed: AtomicBool,
}

impl AeadKey for TokenCipher {
  fn seal(&self, data: &mut Vec<u8>, additional_data: &[u8]) -> Result<(), CryptoError> {
    if data.len() > MAX_TOKEN_PLAINTEXT || self.sealed.swap(true, Ordering::Relaxed) {
      return Err(CryptoError);
    }
    // Quinn derives a fresh key from its random 128-bit token nonce. Each
    // derived key seals once; the fixed AEAD nonce is therefore never reused.
    let tag = self
      .cipher
      .encrypt_in_place(&Nonce96::from_bytes([0; 12]), additional_data, data)
      .map_err(|_| CryptoError)?;
    data.extend_from_slice(tag.as_bytes());
    Ok(())
  }

  fn open<'a>(&self, data: &'a mut [u8], additional_data: &[u8]) -> Result<&'a mut [u8], CryptoError> {
    if !(16..=MAX_TOKEN_PLAINTEXT + 16).contains(&data.len()) {
      return Err(CryptoError);
    }
    let (body, tag) = data.split_at_mut(data.len().strict_sub(16));
    let tag = Aes256GcmTag::from_bytes((&*tag).try_into().map_err(|_| CryptoError)?);
    self
      .cipher
      .decrypt_in_place(&Nonce96::from_bytes([0; 12]), additional_data, body, &tag)
      .map_err(|_| CryptoError)?;
    Ok(body)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use rustls::crypto::cipher::{AeadKey as RustlsAeadKey, Iv};
  use rustls::crypto::tls13::OkmBlock;

  #[test]
  fn reset_mac_matches_independent_peer_and_rejects_forgery() {
    let key = ResetKey(SecretBytes::new([0x0b; 32]));
    let mut tag = [0; 32];
    key.sign(b"Hi There", &mut tag);
    // This adapter fixes the random key length to 32 bytes; an independent
    // AWS-LC HMAC covers that exact key while the primitive owns RFC vectors.
    let peer = rustls::crypto::aws_lc_rs::cipher_suite::TLS13_AES_128_GCM_SHA256
      .tls13()
      .expect("suite");
    let expected = peer.hkdf_provider.hmac_sign(&OkmBlock::new(&[0x0b; 32]), b"Hi There");
    assert_eq!(&tag, expected.as_ref());
    key.verify(b"Hi There", &tag).expect("valid reset MAC");
    for i in 0..tag.len() {
      let mut bad = tag;
      bad[i] ^= 1;
      assert!(key.verify(b"Hi There", &bad).is_err());
    }
    assert!(key.verify(b"wrong connection ID", &tag).is_err());
    assert!(key.verify(b"Hi There", &tag[..16]).is_err());
  }

  #[test]
  fn token_kdf_and_aead_match_independent_peer_and_reject_reuse() {
    let root = [0x42; 32];
    let key = TokenKey(HkdfSha256::from_prk(&root));
    let random = [0x13; 16];
    let peer = rustls::crypto::aws_lc_rs::cipher_suite::TLS13_AES_256_GCM_SHA384
      .tls13()
      .expect("AES-256");
    let hkdf = rustls::crypto::aws_lc_rs::cipher_suite::TLS13_AES_128_GCM_SHA256
      .tls13()
      .expect("SHA-256")
      .hkdf_provider;
    let mut derived = [0; 32];
    hkdf
      .expander_for_okm(&OkmBlock::new(&root))
      .expand_slice(&[b"rscrypto-rustls QUIC token v1", &random], &mut derived)
      .expect("independent HKDF");
    let oracle = peer
      .quic
      .expect("QUIC AEAD")
      .packet_key(RustlsAeadKey::from(derived), Iv::from([0; 12]));
    for length in [0, 1, 32, MAX_TOKEN_PLAINTEXT] {
      let cipher = key.aead_from_hkdf(&random);
      let plaintext = vec![0x57; length];
      let mut sealed = plaintext.clone();
      cipher.seal(&mut sealed, b"address context").expect("seal");
      let mut expected = plaintext.clone();
      let tag = oracle
        .encrypt_in_place(0, b"address context", &mut expected)
        .expect("independent AEAD");
      expected.extend_from_slice(tag.as_ref());
      assert_eq!(sealed, expected);
      let mut second = plaintext.clone();
      assert!(cipher.seal(&mut second, b"address context").is_err());
      assert_eq!(second, plaintext, "a second seal cannot reuse this key/nonce");
      let mut opened = sealed.clone();
      assert_eq!(cipher.open(&mut opened, b"address context").expect("open"), plaintext);
      for position in [0, sealed.len().strict_sub(1)] {
        let mut corrupt = sealed.clone();
        corrupt[position] ^= 1;
        cipher
          .open(&mut corrupt, b"address context")
          .expect_err("tampered token");
      }
      cipher
        .open(&mut sealed.clone(), b"wrong address context")
        .expect_err("wrong context");
      key
        .aead_from_hkdf(&[0x14; 16])
        .open(&mut sealed, b"address context")
        .expect_err("wrong token key");
    }
    assert!(
      key
        .aead_from_hkdf(&random)
        .seal(&mut vec![0; MAX_TOKEN_PLAINTEXT + 1], b"")
        .is_err()
    );
    key
      .aead_from_hkdf(&random)
      .open(&mut [0; 15], b"")
      .expect_err("short tag");
  }

  #[test]
  fn partial_entropy_cannot_publish_a_reset_or_token_key() {
    fn fail(out: &mut [u8; 32]) -> Result<(), GetRandomFailed> {
      out[..17].fill(0x42);
      Err(GetRandomFailed)
    }
    assert!(ResetKey::generate_with(fail).is_err());
    assert!(TokenKey::generate_with(fail).is_err());
  }
}
