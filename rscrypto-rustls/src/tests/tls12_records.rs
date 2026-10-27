use alloc::{vec, vec::Vec};
use rustls::Tls12CipherSuite;
use rustls::crypto::cipher::EncodableVersion;
use rustls::enums::ProtocolVersion;

use super::*;

/// Each enabled TLS 1.2 AEAD with the same suite from both independent peers.
fn suites() -> Vec<(&'static dyn Tls12AeadAlgorithm, [&'static Tls12CipherSuite; 2])> {
  let peer = |provider: rustls::crypto::CryptoProvider, suite| {
    *provider
      .tls12_cipher_suites
      .iter()
      .find(|candidate| candidate.common.suite == suite)
      .expect("independent peer suite")
  };
  [
    (
      TLS12_AES_128_GCM,
      rustls::crypto::CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
    ),
    (
      TLS12_AES_256_GCM,
      rustls::crypto::CipherSuite::TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
    ),
    (
      TLS12_CHACHA20_POLY1305,
      rustls::crypto::CipherSuite::TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
    ),
  ]
  .into_iter()
  .map(|(ours, suite)| {
    (
      ours,
      [
        peer(rustls_aws_lc_rs::DEFAULT_TLS12_PROVIDER, suite),
        peer(rustls_ring::DEFAULT_TLS12_PROVIDER, suite),
      ],
    )
  })
  .collect()
}

fn key(shape: &KeyBlockShape) -> AeadKey {
  if shape.enc_key_len == 16 {
    AeadKey::from([0x19; 16])
  } else {
    AeadKey::from([0x19; 32])
  }
}

/// A key of the other supported length.
fn wrong_key(shape: &KeyBlockShape) -> AeadKey {
  if shape.enc_key_len == 16 {
    AeadKey::from([0x19; 32])
  } else {
    AeadKey::from([0x19; 16])
  }
}

fn inbound(typ: ContentType, payload: &mut [u8]) -> Record<InboundOpaque<'_>> {
  Record::new(
    typ,
    EncodableVersion::Legacy(ProtocolVersion::TLSv1_2),
    InboundOpaque(payload),
  )
}

#[test]
fn records_match_independent_peers_and_reject_every_corruption() {
  let (iv, extra) = ([0x71; NONCE_LEN], [0x3c; 8]);
  for (ours, peers) in suites() {
    let shape = ours.key_block_shape();
    let (fixed, explicit) = (&iv[..shape.fixed_iv_len], &extra[..shape.explicit_nonce_len]);
    for peer in peers {
      let theirs = peer.aead_alg.key_block_shape();
      assert_eq!(
        (shape.enc_key_len, shape.fixed_iv_len, shape.explicit_nonce_len),
        (theirs.enc_key_len, theirs.fixed_iv_len, theirs.explicit_nonce_len)
      );
      for seq in [0, 1, 1 << 32, u64::MAX.strict_sub(1)] {
        for len in [0, 1, 16, MAX_FRAGMENT_LEN] {
          for typ in [ContentType::ApplicationData, ContentType::Handshake, ContentType::Alert] {
            let plain = vec![0xa3; len];
            let (head, tail) = plain.split_at(len / 3);
            let chunks = [head, tail];
            let message = || Record {
              typ,
              version: EncodableVersion::Legacy(ProtocolVersion::TLSv1_2),
              payload: OutboundPlain::new(&chunks),
            };
            let mut encrypter = ours.encrypter(key(&shape), fixed, explicit);
            let mut actual = vec![0xa5];
            encrypter.encrypt(message(), seq, &mut actual).expect("encrypt");
            let mut expected = Vec::new();
            peer
              .aead_alg
              .encrypter(key(&shape), fixed, explicit)
              .encrypt(message(), seq, &mut expected)
              .expect("independent encrypt");
            assert_eq!(actual[0], 0xa5, "encryption must preserve existing output");
            assert_eq!(actual[1..], expected, "{:?} seq={seq} len={len}", peer.common.suite);
            assert_eq!(expected.len(), encrypter.encrypted_payload_len(len).strict_add(5));

            let mut decrypter = ours.decrypter(key(&shape), fixed);
            let mut payload = expected[5..].to_vec();
            let opened = decrypter
              .decrypt(inbound(typ, &mut payload), seq)
              .expect("decrypt independent record");
            assert_eq!(
              (opened.typ, opened.version, opened.payload),
              (
                typ,
                EncodableVersion::Legacy(ProtocolVersion::TLSv1_2),
                plain.as_slice()
              )
            );

            // Any changed explicit nonce, ciphertext, or tag byte fails, and
            // the unauthenticated plaintext is cleared.
            let body = shape.explicit_nonce_len..shape.explicit_nonce_len.strict_add(len);
            let last = expected.len().strict_sub(6);
            for position in [0, last / 2, last] {
              let mut corrupt = expected[5..].to_vec();
              corrupt[position] ^= 1;
              assert!(matches!(
                decrypter.decrypt(inbound(typ, &mut corrupt), seq),
                Err(Error::DecryptError)
              ));
              assert!(corrupt[body.clone()].iter().all(|&byte| byte == 0));
            }
            // The sequence number and content type are authenticated.
            let mut wrong_seq = expected[5..].to_vec();
            assert!(matches!(
              decrypter.decrypt(inbound(typ, &mut wrong_seq), seq ^ 1),
              Err(Error::DecryptError)
            ));
            let mut wrong_type = expected[5..].to_vec();
            assert!(matches!(
              decrypter.decrypt(inbound(ContentType::ChangeCipherSpec, &mut wrong_type), seq),
              Err(Error::DecryptError)
            ));
          }
        }
      }
    }
  }
}

#[test]
fn key_extraction_matches_independent_peers() {
  let (iv, extra) = ([0x71; NONCE_LEN], [0x3c; 8]);
  for (ours, peers) in suites() {
    let shape = ours.key_block_shape();
    let (fixed, explicit) = (&iv[..shape.fixed_iv_len], &extra[..shape.explicit_nonce_len]);
    let secrets = |algorithm: &dyn Tls12AeadAlgorithm| {
      let (name, key, iv) = match algorithm.extract_keys(key(&shape), fixed, explicit).expect("extract") {
        ConnectionTrafficSecrets::Aes128Gcm { key, iv } => Some(("AES-128-GCM", key, iv)),
        ConnectionTrafficSecrets::Aes256Gcm { key, iv } => Some(("AES-256-GCM", key, iv)),
        ConnectionTrafficSecrets::Chacha20Poly1305 { key, iv } => Some(("ChaCha20-Poly1305", key, iv)),
        _ => None,
      }
      .expect("an AEAD traffic secret");
      (name, key.as_ref().to_vec(), iv.as_ref().to_vec())
    };
    for peer in peers {
      assert_eq!(secrets(ours), secrets(peer.aead_alg));
    }
    assert!(ours.extract_keys(wrong_key(&shape), fixed, explicit).is_err());
    assert!(ours.extract_keys(key(&shape), &iv[..5], explicit).is_err());
  }
}

#[test]
fn malformed_records_and_key_material_fail_before_releasing_data() {
  fn message(payload: &[u8]) -> Record<OutboundPlain<'_>> {
    Record {
      typ: ContentType::ApplicationData,
      version: EncodableVersion::Legacy(ProtocolVersion::TLSv1_2),
      payload: payload.into(),
    }
  }
  let (iv, extra) = ([0x71; NONCE_LEN], [0x3c; 8]);
  for (ours, _) in suites() {
    let shape = ours.key_block_shape();
    let (fixed, explicit) = (&iv[..shape.fixed_iv_len], &extra[..shape.explicit_nonce_len]);
    let overhead = shape.explicit_nonce_len.strict_add(TAG_LEN);
    let mut decrypter = ours.decrypter(key(&shape), fixed);
    // Too short for the explicit nonce and tag.
    let mut short = vec![0; overhead.strict_sub(1)];
    assert!(matches!(
      decrypter.decrypt(inbound(ContentType::ApplicationData, &mut short), 0),
      Err(Error::DecryptError)
    ));
    // RFC 5246 §6.2.3: plaintext above 2^14 bytes is an overflow, rejected unopened.
    let mut oversized = vec![0x5a; MAX_FRAGMENT_LEN.strict_add(1).strict_add(overhead)];
    assert!(matches!(
      decrypter.decrypt(inbound(ContentType::ApplicationData, &mut oversized), 0),
      Err(Error::PeerSentOversizedRecord)
    ));
    assert!(oversized.iter().all(|&byte| byte == 0x5a));

    let mut encrypter = ours.encrypter(key(&shape), fixed, explicit);
    let large = vec![0; MAX_FRAGMENT_LEN.strict_add(1)];
    let mut out = vec![0xa5; 3];
    assert!(matches!(
      encrypter.encrypt(message(&large), 0, &mut out),
      Err(Error::EncryptError)
    ));
    assert_eq!(out, [0xa5; 3]);

    // Key or IV lengths outside the key-block shape reject every record.
    let mut invalid = vec![
      ours.encrypter(wrong_key(&shape), fixed, explicit),
      ours.encrypter(key(&shape), &iv[..fixed.len().strict_sub(1)], explicit),
    ];
    if let Some(shorter) = explicit.len().checked_sub(1) {
      invalid.push(ours.encrypter(key(&shape), fixed, &extra[..shorter]));
    }
    for mut invalid in invalid {
      let mut out = vec![0xa5; 3];
      assert!(matches!(
        invalid.encrypt(message(b"plaintext"), 0, &mut out),
        Err(Error::EncryptError)
      ));
      assert_eq!(out, [0xa5; 3]);
    }
    let mut invalid = ours.decrypter(wrong_key(&shape), fixed);
    let mut record = vec![0; overhead.strict_add(4)];
    assert!(matches!(
      invalid.decrypt(inbound(ContentType::ApplicationData, &mut record), 0),
      Err(Error::DecryptError)
    ));
  }
}
