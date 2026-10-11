use super::{fields, hex};
use alloc::vec;

use rustls::crypto::{cipher::AeadKey, cipher::Iv, tls13::OkmBlock};
use rustls::{CipherSuite, Error, Side, SupportedCipherSuite, quic::Version};

use crate::{cipher_suite, hash};

fn peers() -> impl Iterator<Item = (SupportedCipherSuite, SupportedCipherSuite)> {
  let aws = rustls::crypto::aws_lc_rs::default_provider();
  cipher_suite::ALL_CIPHER_SUITES.iter().map(move |ours| {
    let peer = aws
      .cipher_suites
      .iter()
      .find(|peer| peer.suite() == ours.suite())
      .expect("AWS-LC suite");
    (*ours, *peer)
  })
}

#[test]
fn implementation_inventory_covers_every_enabled_suite() {
  // These are the wire algorithms and limits promised by the profile, independent
  // of the tables used to build the provider. The differential tests exercise
  // the selected operations; ci-provider checks their normal/build provenance.
  let expected = [
    (CipherSuite::TLS13_AES_256_GCM_SHA384, 48, 32, 1 << 24, 1 << 23, 1 << 52),
    (CipherSuite::TLS13_AES_128_GCM_SHA256, 32, 16, 1 << 24, 1 << 23, 1 << 52),
    (
      CipherSuite::TLS13_CHACHA20_POLY1305_SHA256,
      32,
      32,
      u64::MAX,
      u64::MAX,
      1 << 36,
    ),
  ];
  assert_eq!(cipher_suite::ALL_CIPHER_SUITES.len(), expected.len());
  for (suite, (id, hash_len, key_len, record_limit, packet_limit, forgery_limit)) in
    cipher_suite::ALL_CIPHER_SUITES.iter().zip(expected)
  {
    let suite = suite.tls13().expect("TLS 1.3 only");
    assert_eq!(suite.common.suite, id);
    assert_eq!(suite.common.hash_provider.output_len(), hash_len);
    assert_eq!(suite.hkdf_provider.extract_from_zero_ikm(None).hash_len(), hash_len);
    assert_eq!(suite.aead_alg.key_len(), key_len);
    assert_eq!(suite.common.confidentiality_limit, record_limit);
    assert!(!suite.fips());
    let quic = suite.quic_suite().expect("each suite has QUIC protection");
    let keys = quic.keys(b"inventory", Side::Client, Version::V1);
    assert_eq!(keys.local.packet.tag_len(), 16);
    assert_eq!(keys.local.header.sample_len(), 16);
    assert_eq!(keys.local.packet.confidentiality_limit(), packet_limit);
    assert_eq!(keys.local.packet.integrity_limit(), forgery_limit);
  }
}

#[test]
fn transcript_forks_and_hmac_scatter_reuse_match_aws_lc() {
  for (ours, peer) in peers().take(2) {
    let ours = ours.tls13().expect("TLS 1.3");
    let peer = peer.tls13().expect("TLS 1.3");
    for length in [0, 1, 55, 56, 63, 64, 65, 111, 112, 127, 128, 129, 1025] {
      let prefix = vec![0xa5; length];
      let expected = peer.common.hash_provider.hash(&prefix);
      assert_eq!(ours.common.hash_provider.hash(&prefix).as_ref(), expected.as_ref());
      let mut state = ours.common.hash_provider.start();
      for part in prefix.chunks(17) {
        state.update(part);
      }
      assert_eq!(state.fork_finish().as_ref(), expected.as_ref());
      let mut fork = state.fork();
      state.update(b"client");
      fork.update(b"server");
      for (state, suffix) in [(state, b"client"), (fork, b"server")] {
        let mut expected = peer.common.hash_provider.start();
        expected.update(&prefix);
        expected.update(suffix);
        assert_eq!(state.finish().as_ref(), expected.finish().as_ref());
      }
    }
    let hmac = if ours.common.hash_provider.output_len() == 32 {
      hash::HMAC_SHA256
    } else {
      hash::HMAC_SHA384
    };
    for key_len in [0, 16, 32, 48, 64] {
      let key = vec![0x0b; key_len];
      let ours = hmac.with_key(&key);
      for middle in [&b""[..], &b"middle"[..]] {
        let expected = peer
          .hkdf_provider
          .hmac_sign(&OkmBlock::new(&key), &[&b"first"[..], middle, b"last"].concat());
        let actual = ours.sign_concat(b"first", &[middle, b""], b"last");
        assert_eq!(actual.as_ref(), expected.as_ref());
        assert_eq!(ours.tag_len(), expected.as_ref().len());
      }
    }
  }
}

#[test]
fn hkdf_extract_import_scatter_and_bounds_match_aws_lc() {
  for (ours, peer) in peers().take(2) {
    let ours = ours.tls13().expect("TLS 1.3").hkdf_provider;
    let peer = peer.tls13().expect("TLS 1.3").hkdf_provider;
    let size = ours.extract_from_zero_ikm(None).hash_len();
    let prk = OkmBlock::new(&[0x5a; 48][..size]);
    for (ours, peer) in [
      (ours.expander_for_okm(&prk), peer.expander_for_okm(&prk)),
      (
        ours.extract_from_secret(Some(b"salt"), b"secret"),
        peer.extract_from_secret(Some(b"salt"), b"secret"),
      ),
      (
        ours.extract_from_secret(Some(b""), b""),
        peer.extract_from_secret(Some(b""), b""),
      ),
      (ours.extract_from_zero_ikm(None), peer.extract_from_zero_ikm(None)),
      (
        ours.extract_from_zero_ikm(Some(b"salt")),
        peer.extract_from_zero_ikm(Some(b"salt")),
      ),
    ] {
      for len in [0, 1, size, size.strict_add(1), size.strict_mul(255)] {
        let mut actual = vec![0; len];
        let mut expected = vec![0; len];
        ours
          .expand_slice(&[b"tls13 ", b"", b"info"], &mut actual)
          .expect("HKDF bound");
        peer.expand_slice(&[b"tls13 info"], &mut expected).expect("HKDF bound");
        assert_eq!(actual, expected);
      }
      assert_eq!(
        ours.expand_block(&[b"block"]).as_ref(),
        peer.expand_block(&[b"block"]).as_ref()
      );
      let mut excessive = vec![0xa5; size.strict_mul(255).strict_add(1)];
      assert!(matches!(
        ours.expand_slice(&[b"info"], &mut excessive),
        Err(rustls::crypto::tls13::OutputLengthError)
      ));
      assert!(excessive.iter().all(|&byte| byte == 0xa5));
    }
  }
}

#[test]
fn quic_packets_match_aws_lc_and_reject_corruption() {
  for (ours, peer) in peers() {
    let ours =
      ours
        .tls13()
        .expect("TLS 1.3")
        .quic_suite()
        .expect("QUIC")
        .keys(b"packet-test", Side::Client, Version::V1);
    let peer =
      peer
        .tls13()
        .expect("TLS 1.3")
        .quic_suite()
        .expect("QUIC")
        .keys(b"packet-test", Side::Client, Version::V1);
    for len in [0, 1, 31, 1200, 16_384] {
      for number in [0, 7, (1_u64 << 62).strict_sub(1)] {
        for path in [0, 0x0302_0100] {
          let plain = vec![0x5a; len];
          let mut actual = plain.clone();
          let mut expected = plain.clone();
          let tag = ours
            .local
            .packet
            .encrypt_in_place_for_path(path, number, b"header", &mut actual)
            .expect("encrypt");
          let peer_tag = peer
            .local
            .packet
            .encrypt_in_place_for_path(path, number, b"header", &mut expected)
            .expect("peer encrypt");
          assert_eq!(actual, expected);
          assert_eq!(tag.as_ref(), peer_tag.as_ref());
          actual.extend_from_slice(tag.as_ref());
          let ciphertext = actual.clone();
          assert_eq!(
            peer
              .local
              .packet
              .decrypt_in_place_for_path(path, number, b"header", &mut actual)
              .expect("peer decrypt"),
            plain
          );
          let mut valid = ciphertext.clone();
          assert_eq!(
            ours
              .local
              .packet
              .decrypt_in_place_for_path(path, number, b"header", &mut valid)
              .expect("decrypt"),
            plain
          );
          for corruption in 0..4 {
            let mut invalid = ciphertext.clone();
            let (pn, header, path) = match corruption {
              0 => {
                *invalid.last_mut().expect("tag") ^= 1;
                (number, &b"header"[..], path)
              }
              1 => (number ^ 1, &b"header"[..], path),
              2 => (number, &b"tamper"[..], path),
              _ => (number, &b"header"[..], path ^ 1),
            };
            assert!(matches!(
              ours
                .local
                .packet
                .decrypt_in_place_for_path(path, pn, header, &mut invalid),
              Err(Error::DecryptError)
            ));
            assert!(invalid[..len].iter().all(|&byte| byte == 0));
          }
        }
      }
    }
    for len in 0..16 {
      assert!(matches!(
        ours.local.packet.decrypt_in_place(0, b"", &mut vec![0; len]),
        Err(Error::DecryptError)
      ));
    }
    let mut payload = [0x55; 8];
    assert!(matches!(
      ours.local.packet.encrypt_in_place(1 << 62, b"", &mut payload),
      Err(Error::EncryptError)
    ));
    assert_eq!(payload, [0x55; 8]);
  }
}

#[test]
fn header_protection_matches_aws_lc_and_rejects_invalid_slices_atomically() {
  for (ours, peer) in peers() {
    let ours =
      ours
        .tls13()
        .expect("TLS 1.3")
        .quic_suite()
        .expect("QUIC")
        .keys(b"header-test", Side::Client, Version::V1);
    let peer =
      peer
        .tls13()
        .expect("TLS 1.3")
        .quic_suite()
        .expect("QUIC")
        .keys(b"header-test", Side::Client, Version::V1);
    for first in u8::MIN..=u8::MAX {
      let sample = [first; 16];
      let (mut a, mut b) = (first, first);
      let (mut pn, mut peer_pn) = ([0x10, 0x20, 0x30, 0x40], [0x10, 0x20, 0x30, 0x40]);
      ours
        .local
        .header
        .encrypt_in_place(&sample, &mut a, &mut pn)
        .expect("header");
      peer
        .local
        .header
        .encrypt_in_place(&sample, &mut b, &mut peer_pn)
        .expect("peer header");
      assert_eq!((a, pn), (b, peer_pn));
      ours
        .local
        .header
        .decrypt_in_place(&sample, &mut a, &mut pn)
        .expect("header open");
      assert_eq!((a, pn), (first, [0x10, 0x20, 0x30, 0x40]));
    }
    for (sample_len, pn_len) in [(15, 4), (17, 4), (16, 5), (16, 0), (16, 3)] {
      let mut first = 0xc3; // Encodes a four-byte packet number.
      let mut pn = vec![0x42; pn_len];
      let before = pn.clone();
      assert!(matches!(
        ours
          .local
          .header
          .encrypt_in_place(&vec![0; sample_len], &mut first, &mut pn),
        Err(Error::General(_))
      ));
      assert_eq!((first, pn), (0xc3, before));
    }
  }
}

#[test]
fn rfc9001_initial_packets_cover_derivation_packet_and_header_protection() {
  let suite = cipher_suite::TLS13_AES_128_GCM_SHA256
    .tls13()
    .expect("TLS 1.3")
    .quic_suite()
    .expect("QUIC");
  let cid = hex("8394c8f03e515708");
  for (fixture, side, number, padded_len) in [
    (
      include_str!("../../tests/vectors/rfc9001-client.txt"),
      Side::Client,
      2,
      1162,
    ),
    (
      include_str!("../../tests/vectors/rfc9001-server.txt"),
      Side::Server,
      1,
      99,
    ),
  ] {
    let v = fields(fixture);
    let mut payload = v["frame"].clone();
    payload.resize(padded_len, 0);
    let plaintext = payload.clone();
    let mut header = v["header"].clone();
    let pn_len = usize::from(header[0] & 3).strict_add(1);
    let pn_offset = header.len().strict_sub(pn_len);
    let keys = suite.keys(&cid, side, Version::V1);
    let tag = keys
      .local
      .packet
      .encrypt_in_place(number, &header, &mut payload)
      .expect("packet seal");
    payload.extend_from_slice(tag.as_ref());
    let sample_offset = 4_usize.strict_sub(pn_len);
    let sample = &payload[sample_offset..sample_offset.strict_add(16)];
    let (first, rest) = header.split_first_mut().expect("first byte");
    keys
      .local
      .header
      .encrypt_in_place(sample, first, &mut rest[pn_offset.strict_sub(1)..])
      .expect("header seal");
    assert_eq!([header.as_slice(), payload.as_slice()].concat(), v["packet"]);
    let (first, rest) = header.split_first_mut().expect("first byte");
    keys
      .local
      .header
      .decrypt_in_place(sample, first, &mut rest[pn_offset.strict_sub(1)..])
      .expect("header open");
    assert_eq!(header, v["header"]);
    assert_eq!(
      keys
        .local
        .packet
        .decrypt_in_place(number, &header, &mut payload)
        .expect("packet open"),
      plaintext
    );
  }
}

#[test]
fn rfc9001_chacha_short_packet_and_key_update() {
  let hkdf = hash::HKDF_SHA256.expander_for_okm(&OkmBlock::new(&hex(
    "9ac312a7f877468ebe69422748ad00a15443f18203a07d6060f688f30f21632b",
  )));
  let mut key = [0; 32];
  let mut iv = [0; 12];
  let mut hp = [0; 32];
  let mut update = [0; 32];
  for (info, out, expected) in [
    (
      "00200e746c7331332071756963206b657900",
      &mut key[..],
      "c6d98ff3441c3fe1b2182094f69caa2ed4b716b65488960a7a984979fb23e1c8",
    ),
    (
      "000c0d746c733133207175696320697600",
      &mut iv[..],
      "e0459b3474bdd0e44a41c144",
    ),
    (
      "00200d746c733133207175696320687000",
      &mut hp[..],
      "25a282b9e82f06f21f488917a4fc8f1b73573685608597d0efcb076b0ab7a7a4",
    ),
    (
      "00200d746c7331332071756963206b7500",
      &mut update[..],
      "1223504755036d556342ee9361d253421a826c9ecdf3c7148684b36b714881f9",
    ),
  ] {
    hkdf.expand_slice(&[&hex(info)], out).expect("HKDF label");
    assert_eq!(out, hex(expected));
  }
  let alg = crate::quic::CHACHA20_POLY1305;
  let packet = alg.packet_key(AeadKey::from(key), Iv::from(iv));
  let header = alg.header_protection_key(AeadKey::from(hp));
  let mut plaintext = [1];
  let aad = hex("4200bff4");
  let tag = packet
    .encrypt_in_place(654_360_564, &aad, &mut plaintext)
    .expect("seal");
  let ciphertext = [plaintext.as_slice(), tag.as_ref()].concat();
  let mut first = aad[0];
  let mut pn = aad[1..].to_vec();
  header
    .encrypt_in_place(&ciphertext[1..], &mut first, &mut pn)
    .expect("header seal");
  assert_eq!(
    [&[first], pn.as_slice(), ciphertext.as_slice()].concat(),
    hex("4cfe4189655e5cd55c41f69080575d7999c25a5bfb")
  );
}
