#![cfg(feature = "aead")]

use chacha20poly1305::{
  ChaCha20Poly1305 as Oracle, KeyInit,
  aead::{Aead as _, AeadInOut, Payload, array::Array},
};
use rscrypto::{ChaCha20Poly1305, ChaCha20Poly1305Key, ChaCha20Poly1305Tag, aead::Nonce96};

mod common;
use common::decode_hex_vec as decode_hex;

fn pattern_bytes(len: usize, seed: u8) -> Vec<u8> {
  let mut out = vec![0u8; len];
  for (index, byte) in out.iter_mut().enumerate() {
    *byte = seed
      .wrapping_add((index as u8).wrapping_mul(17))
      .wrapping_add((index as u8).rotate_left(1));
  }
  out
}

#[test]
fn chacha20poly1305_matches_rfc_8439_vector() {
  let plaintext = decode_hex(
    "4c616469657320616e642047656e746c656d656e206f662074686520636c617373206f66202739393a204966204920636f756c64206f6666657220796f75206f6e6c79206f6e652074697020666f7220746865206675747572652c2073756e73637265656e20776f756c642062652069742e",
  );
  let aad = decode_hex("50515253c0c1c2c3c4c5c6c7");
  let key = decode_hex("808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f");
  let nonce = decode_hex("070000004041424344454647");
  let expected_ciphertext = decode_hex(
    "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d63dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b3692ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc3ff4def08e4b7a9de576d26586cec64b6116",
  );
  let expected_tag = decode_hex("1ae10b594f09e26a7e902ecbd0600691");

  let key = ChaCha20Poly1305Key::from_bytes(key.try_into().unwrap());
  let nonce = Nonce96::from_bytes(nonce.try_into().unwrap());
  let cipher = ChaCha20Poly1305::new(&key);

  let mut sealed = vec![0u8; plaintext.len() + ChaCha20Poly1305::TAG_SIZE];
  cipher.encrypt(&nonce, &aad, &plaintext, &mut sealed).unwrap();

  assert_eq!(&sealed[..plaintext.len()], expected_ciphertext.as_slice());
  assert_eq!(&sealed[plaintext.len()..], expected_tag.as_slice());

  let mut opened = vec![0u8; plaintext.len()];
  cipher.decrypt(&nonce, &aad, &sealed, &mut opened).unwrap();
  assert_eq!(opened, plaintext);
}

#[test]
fn chacha20poly1305_matches_rustcrypto_oracle() {
  let key_bytes = [0x42u8; ChaCha20Poly1305::KEY_SIZE];
  let nonce_bytes = [0x24u8; Nonce96::LENGTH];
  let aad = b"rscrypto-chacha-aead";
  let plaintext = b"portable baseline first, SIMD later";

  let key = ChaCha20Poly1305Key::from_bytes(key_bytes);
  let nonce = Nonce96::from_bytes(nonce_bytes);
  let cipher = ChaCha20Poly1305::new(&key);

  let oracle = Oracle::new(&Array(key_bytes));
  let oracle_nonce = Array(nonce_bytes);

  let mut ours = plaintext.to_vec();
  let tag = cipher.encrypt_in_place(&nonce, aad, &mut ours).unwrap();

  let mut oracle_buffer = plaintext.to_vec();
  let oracle_tag = oracle
    .encrypt_inout_detached(&oracle_nonce, aad, oracle_buffer.as_mut_slice().into())
    .unwrap();

  assert_eq!(ours, oracle_buffer);
  assert_eq!(tag.as_bytes(), oracle_tag.as_slice());

  let typed_tag = ChaCha20Poly1305Tag::from_bytes(tag.to_bytes());
  cipher.decrypt_in_place(&nonce, aad, &mut ours, &typed_tag).unwrap();
  assert_eq!(ours, plaintext);
}

#[test]
fn chacha20poly1305_rejects_modified_tag() {
  let key = ChaCha20Poly1305Key::from_bytes([0x11; ChaCha20Poly1305::KEY_SIZE]);
  let nonce = Nonce96::from_bytes([0x22; Nonce96::LENGTH]);
  let cipher = ChaCha20Poly1305::new(&key);

  let mut buffer = *b"forgery-check";
  let mut tag = cipher.encrypt_in_place(&nonce, b"aad", &mut buffer).unwrap().to_bytes();
  tag[0] ^= 1;

  assert!(
    cipher
      .decrypt_in_place(&nonce, b"aad", &mut buffer, &ChaCha20Poly1305Tag::from_bytes(tag))
      .is_err()
  );
}

#[cfg(feature = "diag")]
#[test]
fn chacha20poly1305_diag_owned_decrypt_large_inputs_match_normal_path() {
  const PLAINTEXT_LENS: &[usize] = &[1024, 1025, 4095, 4096, 4097, 16_384];

  let key = ChaCha20Poly1305Key::from_bytes([0x42; ChaCha20Poly1305::KEY_SIZE]);
  let nonce = Nonce96::from_bytes([0x24; Nonce96::LENGTH]);
  let cipher = ChaCha20Poly1305::new(&key);

  for &plaintext_len in PLAINTEXT_LENS {
    let plaintext = pattern_bytes(plaintext_len, 0x51);
    let aad = pattern_bytes(257, 0xa7);
    let mut ciphertext = plaintext.clone();
    let tag = cipher.encrypt_in_place(&nonce, &aad, &mut ciphertext).unwrap();

    rscrypto::aead::diag_chacha20poly1305_decrypt_in_place_owned(&cipher, &nonce, &aad, &mut ciphertext, &tag).unwrap();

    assert_eq!(
      ciphertext, plaintext,
      "owned diagnostic decrypt mismatch at plaintext_len={plaintext_len}"
    );
  }
}

#[cfg(feature = "diag")]
#[test]
fn chacha20poly1305_diag_owned_decrypt_zeroes_large_buffer_on_bad_tag() {
  let key = ChaCha20Poly1305Key::from_bytes([0x11; ChaCha20Poly1305::KEY_SIZE]);
  let nonce = Nonce96::from_bytes([0x22; Nonce96::LENGTH]);
  let cipher = ChaCha20Poly1305::new(&key);

  let plaintext = pattern_bytes(4097, 0x6d);
  let aad = pattern_bytes(33, 0x95);
  let mut ciphertext = plaintext.clone();
  let mut tag = cipher
    .encrypt_in_place(&nonce, &aad, &mut ciphertext)
    .unwrap()
    .to_bytes();
  tag[7] ^= 0x80;

  let result = rscrypto::aead::diag_chacha20poly1305_decrypt_in_place_owned(
    &cipher,
    &nonce,
    &aad,
    &mut ciphertext,
    &ChaCha20Poly1305Tag::from_bytes(tag),
  );

  assert!(result.is_err());
  assert!(
    ciphertext.iter().all(|&byte| byte == 0),
    "owned diagnostic decrypt must zero caller buffer on verification failure"
  );
}

#[cfg(all(feature = "diag", target_arch = "x86_64"))]
#[test]
fn chacha20poly1305_diag_x86_short_fused_encrypt_matches_normal_path() {
  const AAD_LENS: &[usize] = &[0, 1, 15, 16, 17, 31, 32, 33, 64];

  let key = ChaCha20Poly1305Key::from_bytes([0x42; ChaCha20Poly1305::KEY_SIZE]);
  let nonce = Nonce96::from_bytes([0x24; Nonce96::LENGTH]);
  let cipher = ChaCha20Poly1305::new(&key);

  for plaintext_len in 0..=512 {
    let plaintext = pattern_bytes(plaintext_len, 0x51);

    for &aad_len in AAD_LENS {
      let aad = pattern_bytes(aad_len, 0xa7);
      let mut actual = plaintext.clone();
      let actual_tag =
        rscrypto::aead::diag_chacha20poly1305_encrypt_in_place_x86_64_short_fused(&cipher, &nonce, &aad, &mut actual);

      if !(1..=256).contains(&plaintext_len) {
        assert!(
          actual_tag.is_none(),
          "x86 short fused path must not apply at plaintext_len={plaintext_len}"
        );
        continue;
      }

      let mut expected = plaintext.clone();
      let expected_tag = cipher.encrypt_in_place(&nonce, &aad, &mut expected).unwrap();
      let actual_tag = actual_tag
        .expect("x86 short fused path must apply for plaintext lengths 1..=256")
        .unwrap();

      assert_eq!(
        actual, expected,
        "x86 short fused ciphertext mismatch plaintext_len={plaintext_len} aad_len={aad_len}"
      );
      assert_eq!(
        actual_tag, expected_tag,
        "x86 short fused tag mismatch plaintext_len={plaintext_len} aad_len={aad_len}"
      );
    }
  }
}

#[cfg(all(
  feature = "diag",
  target_arch = "aarch64",
  any(target_os = "linux", target_os = "macos")
))]
#[test]
fn chacha20poly1305_diag_owned_par4_encrypt_matches_normal_path() {
  const PLAINTEXT_LENS: &[usize] = &[0, 1, 15, 16, 17, 511, 512, 513, 1024, 1025, 4095, 4096, 4097, 16_384];
  const AAD_LENS: &[usize] = &[0, 1, 14, 15, 16, 17, 63, 64, 65, 257];

  let key = ChaCha20Poly1305Key::from_bytes([0x42; ChaCha20Poly1305::KEY_SIZE]);
  let nonce = Nonce96::from_bytes([0x24; Nonce96::LENGTH]);
  let cipher = ChaCha20Poly1305::new(&key);

  for &plaintext_len in PLAINTEXT_LENS {
    let plaintext = pattern_bytes(plaintext_len, 0x51);

    for &aad_len in AAD_LENS {
      let aad = pattern_bytes(aad_len, 0xa7);
      let mut expected = plaintext.clone();
      let expected_tag = cipher.encrypt_in_place(&nonce, &aad, &mut expected).unwrap();

      let mut actual = plaintext.clone();
      let actual_tag =
        rscrypto::aead::diag_chacha20poly1305_encrypt_in_place_owned_par4_aarch64(&cipher, &nonce, &aad, &mut actual)
          .unwrap();

      assert_eq!(
        actual, expected,
        "owned par4 ciphertext mismatch plaintext_len={plaintext_len} aad_len={aad_len}"
      );
      assert_eq!(
        actual_tag, expected_tag,
        "owned par4 tag mismatch plaintext_len={plaintext_len} aad_len={aad_len}"
      );
    }
  }
}

#[cfg(all(
  feature = "diag",
  target_arch = "aarch64",
  any(target_os = "linux", target_os = "macos")
))]
#[test]
fn chacha20poly1305_diag_owned_par4_auth_matches_dispatched_path() {
  const CIPHERTEXT_LENS: &[usize] = &[0, 1, 15, 16, 17, 63, 64, 65, 255, 256, 257, 511, 512, 513, 4096, 4097];
  const AAD_LENS: &[usize] = &[0, 1, 15, 16, 17, 63, 64, 65, 257];

  let key = [0x42; ChaCha20Poly1305::KEY_SIZE];

  for &ciphertext_len in CIPHERTEXT_LENS {
    let ciphertext = pattern_bytes(ciphertext_len, 0x51);

    for &aad_len in AAD_LENS {
      let aad = pattern_bytes(aad_len, 0xa7);
      let expected = rscrypto::aead::diag_chacha20poly1305_authenticate_aead(&aad, &ciphertext, &key).unwrap();
      let actual =
        rscrypto::aead::diag_chacha20poly1305_authenticate_aead_aarch64_owned_par4(&aad, &ciphertext, &key).unwrap();

      assert_eq!(
        actual, expected,
        "owned par4 auth mismatch ciphertext_len={ciphertext_len} aad_len={aad_len}"
      );
    }
  }
}

#[test]
fn chacha20poly1305_rejects_wrong_tag_length() {
  assert!(ChaCha20Poly1305::tag_from_slice(&[0u8; 0]).is_err());
  assert!(ChaCha20Poly1305::tag_from_slice(&[0u8; 15]).is_err());
  assert!(ChaCha20Poly1305::tag_from_slice(&[0u8; 17]).is_err());
  assert!(ChaCha20Poly1305::tag_from_slice(&[0u8; 16]).is_ok());
}

#[test]
fn chacha20poly1305_boundary_and_large_inputs_match_oracle() {
  const PLAINTEXT_LENS: &[usize] = &[
    0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1023, 1024, 4095, 4096, 16_383, 16_384,
  ];
  const AAD_LENS: &[usize] = &[0, 1, 15, 16, 17, 31, 32, 33, 255, 256];

  let key_bytes = [0x42u8; ChaCha20Poly1305::KEY_SIZE];
  let nonce_bytes = [0x24u8; Nonce96::LENGTH];
  let key = ChaCha20Poly1305Key::from_bytes(key_bytes);
  let nonce = Nonce96::from_bytes(nonce_bytes);
  let cipher = ChaCha20Poly1305::new(&key);

  let oracle = Oracle::new(&Array(key_bytes));
  let oracle_nonce = Array(nonce_bytes);

  for &plaintext_len in PLAINTEXT_LENS {
    let plaintext = pattern_bytes(plaintext_len, 0x31);

    for &aad_len in AAD_LENS {
      let aad = pattern_bytes(aad_len, 0x9b);

      let mut combined = vec![0u8; plaintext_len + ChaCha20Poly1305::TAG_SIZE];
      cipher.encrypt(&nonce, &aad, &plaintext, &mut combined).unwrap();

      let oracle_combined = oracle
        .encrypt(
          &oracle_nonce,
          Payload {
            msg: &plaintext,
            aad: &aad,
          },
        )
        .unwrap();
      assert_eq!(
        combined, oracle_combined,
        "combined ciphertext mismatch pt_len={plaintext_len} aad_len={aad_len}"
      );

      let mut opened = vec![0u8; plaintext_len];
      cipher.decrypt(&nonce, &aad, &oracle_combined, &mut opened).unwrap();
      assert_eq!(
        opened, plaintext,
        "combined decrypt mismatch pt_len={plaintext_len} aad_len={aad_len}"
      );

      let oracle_opened = oracle
        .decrypt(
          &oracle_nonce,
          Payload {
            msg: &combined,
            aad: &aad,
          },
        )
        .unwrap();
      assert_eq!(
        oracle_opened, plaintext,
        "oracle decrypt mismatch pt_len={plaintext_len} aad_len={aad_len}"
      );

      let mut detached = plaintext.clone();
      let tag = cipher.encrypt_in_place(&nonce, &aad, &mut detached).unwrap();
      assert_eq!(
        detached,
        oracle_combined[..plaintext_len],
        "detached ciphertext mismatch pt_len={plaintext_len} aad_len={aad_len}"
      );
      assert_eq!(
        tag.as_bytes(),
        &oracle_combined[plaintext_len..],
        "detached tag mismatch pt_len={plaintext_len} aad_len={aad_len}"
      );

      let typed_tag = ChaCha20Poly1305Tag::from_bytes(tag.to_bytes());
      cipher
        .decrypt_in_place(&nonce, &aad, &mut detached, &typed_tag)
        .unwrap();
      assert_eq!(
        detached, plaintext,
        "detached decrypt mismatch pt_len={plaintext_len} aad_len={aad_len}"
      );
    }
  }
}
