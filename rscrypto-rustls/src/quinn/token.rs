//! Quinn's reset and token-key traits over the transport-neutral QUIC keys.

use alloc::{boxed::Box, vec::Vec};
use quinn_proto::crypto::{AeadKey, CryptoError, HandshakeTokenKey, HmacKey};

use crate::quic::{AddressTokenKey, StatelessResetKey, TokenCipher};

impl HmacKey for StatelessResetKey {
  fn sign(&self, data: &[u8], output: &mut [u8]) {
    output.copy_from_slice(self.mac(data).as_bytes());
  }

  fn signature_len(&self) -> usize {
    32
  }

  fn verify(&self, data: &[u8], signature: &[u8]) -> Result<(), CryptoError> {
    if self.verify_mac(data, signature) {
      Ok(())
    } else {
      Err(CryptoError)
    }
  }
}

impl HandshakeTokenKey for AddressTokenKey {
  fn aead_from_hkdf(&self, random_bytes: &[u8]) -> Box<dyn AeadKey> {
    // Quinn draws a fresh 128-bit token nonce from its seeded generator.
    self.cipher(random_bytes)
  }
}

impl AeadKey for TokenCipher {
  fn seal(&self, data: &mut Vec<u8>, additional_data: &[u8]) -> Result<(), CryptoError> {
    self.seal_in_place(data, additional_data).map_err(|_| CryptoError)
  }

  fn open<'a>(&self, data: &'a mut [u8], additional_data: &[u8]) -> Result<&'a mut [u8], CryptoError> {
    self.open_in_place(data, additional_data).map_err(|_| CryptoError)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::tests::Fixed;
  use rustls::crypto::cipher::{AeadKey as RustlsAeadKey, Iv};
  use rustls::crypto::tls13::OkmBlock;

  const MAX_TOKEN_PLAINTEXT: usize = AddressTokenKey::MAX_PLAINTEXT_LEN;

  #[test]
  fn reset_mac_matches_independent_peer_and_rejects_forgery() {
    let key = StatelessResetKey::generate(&Fixed(0x0b)).expect("fixed reset key");
    let mut tag = [0; 32];
    key.sign(b"Hi There", &mut tag);
    // This adapter fixes the random key length to 32 bytes; an independent
    // AWS-LC HMAC covers that exact key while the primitive owns RFC vectors.
    let peer = rustls_aws_lc_rs::cipher_suite::TLS13_AES_128_GCM_SHA256;
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
    let key = AddressTokenKey::generate(&Fixed(0x42)).expect("fixed token key");
    let random = [0x13; 16];
    let peer = rustls_aws_lc_rs::cipher_suite::TLS13_AES_256_GCM_SHA384;
    let hkdf = rustls_aws_lc_rs::cipher_suite::TLS13_AES_128_GCM_SHA256.hkdf_provider;
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
        .encrypt_in_place(0, b"address context", &mut expected, None)
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
    use rustls::crypto::{GetRandomFailed, SecureRandom};
    #[derive(Debug)]
    struct Partial;
    impl SecureRandom for Partial {
      fn fill(&self, out: &mut [u8]) -> Result<(), GetRandomFailed> {
        out[..17].fill(0x42);
        Err(GetRandomFailed)
      }
    }
    StatelessResetKey::generate(&Partial).expect_err("partial entropy");
    AddressTokenKey::generate(&Partial).expect_err("partial entropy");
  }
}
