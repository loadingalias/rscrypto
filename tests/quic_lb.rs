//! QUIC-LB connection-ID cipher (draft-ietf-quic-load-balancers-22, section 5.4).

#![cfg(feature = "aes-gcm")]

mod common;

use aes::cipher::{Array, BlockCipherDecrypt as _, BlockCipherEncrypt as _, KeyInit as _};
use common::decode_hex_vec;
use rscrypto::aead::expert::quic_lb::{QuicLbCipher, QuicLbKey};

const LEN: usize = QuicLbCipher::MAX_PLAINTEXT_LEN;

fn key(hex: &str) -> QuicLbKey {
  QuicLbKey::from_bytes(decode_hex_vec(hex).try_into().expect("16-octet key"))
}

fn block(bytes: &[u8]) -> [u8; LEN] {
  let mut block = [0u8; LEN];
  block[..bytes.len()].copy_from_slice(bytes);
  block
}

/// A deterministic stream of distinct test inputs.
fn generated<const N: usize>(state: &mut u64) -> [u8; N] {
  let mut out = [0u8; N];
  for byte in &mut out {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *byte = state.to_le_bytes()[0];
  }
  out
}

/// Every (server-ID length, nonce length) pair that section 5.3 allows: 120 pairs.
fn allowed_lengths() -> impl Iterator<Item = (usize, usize)> {
  (1..=LEN)
    .flat_map(|server_id_len| (4..=LEN.strict_sub(server_id_len)).map(move |nonce_len| (server_id_len, nonce_len)))
}

/// The four encrypted CIDs of Appendix B.2 and the section 5.4.2.4 example. Each body is the CID
/// without its first octet, which carries the rotation and length and is not encrypted.
///
/// Erratum: in the section 5.4.2.4 example, step 10 prints `0xb3e4357c` as the truncated output
/// of `0xb334357c...`. The final CID `0767947d29be054a` matches the correct value.
const VECTORS: &[(&str, &str, &str, &str)] = &[
  // (key, server ID, nonce, encrypted body)
  (
    "fdf726a9893ec05c0632d3956680baf0",
    "31441a",
    "9c69c275",
    "67947d29be054a",
  ),
  (
    "8f95f09245765f80256934e50c66207f",
    "ed793a",
    "ee080dbf",
    "20b1d07b359d3c",
  ),
  // The server ID is longer than the nonce, so decryption needs the fourth pass.
  (
    "8f95f09245765f80256934e50c66207f",
    "ed793a51d49b8f5fab65",
    "ee080dbf48",
    "cc381bc74cb4fbad2823a3d1f8fed2",
  ),
  // Single pass: the server ID and nonce total 16 octets.
  (
    "8f95f09245765f80256934e50c66207f",
    "ed793a51d49b8f5f",
    "ee080dbf48c0d1e5",
    "4dd2d05a7b0de9b2b9907afb5ecf8cc3",
  ),
  // Erratum: this B.2 row lists cr_bits 3, but its CID begins 0x12, which encodes rotation 0.
  // The body does not depend on the first octet.
  (
    "8f95f09245765f80256934e50c66207f",
    "ed793a51d49b8f5fab",
    "ee080dbf48c0d1e55d",
    "5779c9cc86beb3a3a4a3ca96fce4bfe0cdbc",
  ),
];

/// Encryption reproduces the draft's ciphertexts, and decryption recovers each server ID.
///
/// Decryption inverts section 5.4.2.3. Erratum: the section 5.5.2 pseudocode clears the first bits
/// of `left_1` after pass 3, where `right_1` is meant.
#[test]
fn draft_vectors_encrypt_and_decrypt() {
  for &(key_hex, server_id, nonce, body) in VECTORS {
    let (server_id, nonce, body) = (decode_hex_vec(server_id), decode_hex_vec(nonce), decode_hex_vec(body));
    let cipher = QuicLbCipher::new(&key(key_hex), server_id.len(), nonce.len()).expect("draft lengths");

    let mut encrypted = block(&[server_id.as_slice(), nonce.as_slice()].concat());
    cipher.encrypt(&mut encrypted);
    assert_eq!(encrypted, block(&body), "encrypt {key_hex} {}", server_id.len());

    cipher.decrypt_server_id(&mut encrypted);
    assert_eq!(encrypted, block(&server_id), "decrypt {key_hex} {}", server_id.len());
  }
}

/// With a 16-octet total, the body is one AES-128 block (section 5.4.1), checked against an
/// independent AES implementation in both directions.
#[test]
fn single_pass_matches_independent_aes() {
  let mut state = 0x5155_4943_2d4c_4231u64;
  for server_id_len in 1..=12 {
    let raw_key: [u8; 16] = generated(&mut state);
    let plaintext: [u8; 16] = generated(&mut state);
    let aes = aes::Aes128::new(&Array::from(raw_key));
    let mut expected = Array::from(plaintext);
    aes.encrypt_block(&mut expected);

    let cipher = QuicLbCipher::new(
      &QuicLbKey::from_bytes(raw_key),
      server_id_len,
      16usize.strict_sub(server_id_len),
    )
    .expect("single-pass lengths");
    let mut encrypted = block(&plaintext);
    cipher.encrypt(&mut encrypted);
    assert_eq!(encrypted[..16], expected[..], "encrypt, server ID {server_id_len}");

    let ciphertext: [u8; 16] = generated(&mut state);
    let mut expected = Array::from(ciphertext);
    aes.decrypt_block(&mut expected);
    let mut decrypted = block(&ciphertext);
    cipher.decrypt_server_id(&mut decrypted);
    assert_eq!(
      decrypted,
      block(&expected[..server_id_len]),
      "decrypt, server ID {server_id_len}"
    );
  }
}

/// Decryption recovers the server ID for every allowed length pair. The inputs set every bit,
/// including both halves of the octet that odd lengths split, and the server IDs that end in that
/// octet need the fourth decryption pass.
#[test]
fn decrypt_recovers_the_server_id_for_every_length_pair() {
  let mut state = 0x726f_7574_652d_6964u64;
  for (server_id_len, nonce_len) in allowed_lengths() {
    let len = server_id_len.strict_add(nonce_len);
    let cipher = QuicLbCipher::new(&QuicLbKey::from_bytes(generated(&mut state)), server_id_len, nonce_len)
      .expect("allowed lengths");
    for plaintext in [[0xffu8; LEN], generated(&mut state)] {
      let plaintext = block(&plaintext[..len]);
      let mut buffer = plaintext;
      cipher.encrypt(&mut buffer);
      assert_ne!(
        buffer[..len],
        plaintext[..len],
        "encrypt changes {server_id_len}+{nonce_len}"
      );
      cipher.decrypt_server_id(&mut buffer);
      assert_eq!(
        buffer,
        block(&plaintext[..server_id_len]),
        "round trip {server_id_len}+{nonce_len}"
      );
    }
  }
}

/// The octets after the plaintext keep their values through encryption, and neither output
/// depends on them. Decryption leaves zeros after the server ID. Each trailing octet holds a
/// distinct value, so a moved, swapped, or merged octet shows.
#[test]
fn buffer_octets_after_the_plaintext_follow_the_contract() {
  let mut state = 0x6275_6666_6572_2d31u64;
  for (server_id_len, nonce_len) in allowed_lengths() {
    let len = server_id_len.strict_add(nonce_len);
    let cipher = QuicLbCipher::new(&QuicLbKey::from_bytes(generated(&mut state)), server_id_len, nonce_len)
      .expect("allowed lengths");
    let plaintext: [u8; LEN] = generated(&mut state);
    let tails: [[u8; LEN]; 2] = [
      core::array::from_fn(|i| 0xa0 ^ u8::try_from(i).expect("small index")),
      core::array::from_fn(|i| {
        0x31_u8
          .wrapping_mul(u8::try_from(i).expect("small index"))
          .wrapping_add(7)
      }),
    ];
    let buffers = tails.map(|tail| {
      let mut buffer = plaintext;
      buffer[len..].copy_from_slice(&tail[len..]);
      buffer
    });

    let encrypted = buffers.map(|mut buffer| {
      cipher.encrypt(&mut buffer);
      buffer
    });
    for (encrypted, tail) in encrypted.iter().zip(&tails) {
      assert_eq!(
        encrypted[len..],
        tail[len..],
        "encrypt changes the tail, {server_id_len}+{nonce_len}"
      );
    }
    assert_eq!(
      encrypted[0][..len],
      encrypted[1][..len],
      "encrypt reads the tail, {server_id_len}+{nonce_len}"
    );

    let decrypted = encrypted.map(|mut buffer| {
      cipher.decrypt_server_id(&mut buffer);
      buffer
    });
    assert_eq!(
      decrypted[0], decrypted[1],
      "decrypt reads the tail, {server_id_len}+{nonce_len}"
    );
    assert!(
      decrypted[0][server_id_len..].iter().all(|&byte| byte == 0),
      "decrypt leaves octets after the server ID, {server_id_len}+{nonce_len}"
    );
  }
}

/// Construction succeeds exactly for the lengths of section 5.3: a server ID of at least one
/// octet, a nonce of at least four, and a total of at most 19.
#[test]
fn construction_accepts_exactly_the_draft_lengths() {
  let key = QuicLbKey::from_bytes([0x42; 16]);
  for server_id_len in 0..=21 {
    for nonce_len in 0..=21 {
      let allowed = server_id_len >= 1 && nonce_len >= 4 && server_id_len + nonce_len <= 19;
      let cipher = QuicLbCipher::new(&key, server_id_len, nonce_len);
      assert_eq!(cipher.is_some(), allowed, "{server_id_len}+{nonce_len}");
    }
  }
  assert!(QuicLbCipher::new(&key, usize::MAX, 4).is_none(), "overflowing total");
}

/// Every allowed length pair matches an independent implementation: google/quiche for 117 pairs,
/// and hopf-quic for the 3 pairs with a nonce of more than 16 octets, which quiche refuses. Each
/// line is "rotation key server_id nonce connection_id", and the first octet of the connection ID
/// is not encrypted. The provenance is in `testdata/quic_lb/README.md`.
#[test]
fn every_length_pair_matches_independent_implementations() {
  let files = [
    include_str!("../testdata/quic_lb/quiche-vectors.txt"),
    include_str!("../testdata/quic_lb/hopf-vectors.txt"),
  ];
  let mut pairs = Vec::new();
  for line in files.iter().flat_map(|file| file.lines()) {
    let [_, key_hex, server_id, nonce, id]: [&str; 5] =
      line.split(' ').collect::<Vec<_>>().try_into().expect("five fields");
    let (server_id, nonce, id) = (decode_hex_vec(server_id), decode_hex_vec(nonce), decode_hex_vec(id));
    let cipher = QuicLbCipher::new(&key(key_hex), server_id.len(), nonce.len()).expect("allowed lengths");

    let mut buffer = block(&[server_id.as_slice(), nonce.as_slice()].concat());
    cipher.encrypt(&mut buffer);
    assert_eq!(buffer, block(&id[1..]), "encrypt {line}");

    cipher.decrypt_server_id(&mut buffer);
    assert_eq!(buffer, block(&server_id), "decrypt {line}");
    pairs.push((server_id.len(), nonce.len()));
  }
  pairs.sort_unstable();
  pairs.dedup();
  assert!(pairs.into_iter().eq(allowed_lengths()), "every allowed length pair");
}
