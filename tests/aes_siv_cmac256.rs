#![cfg(feature = "aes-siv")]

use aes_siv::{KeyInit as _, siv::Aes128Siv};
use rscrypto::{
  AesSivCmac256, AesSivCmac256Key, AesSivCmac256Nonce, AesSivCmac256NonceError, AesSivCmac256Tag,
  aead::{OpenError, SealError},
};

fn generated_bytes(state: &mut u64, out: &mut [u8]) {
  for byte in out {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *byte = state.to_le_bytes()[0];
  }
}

fn oracle(key: &[u8; 32], nonce: &[u8], aad: &[u8], plaintext: &[u8]) -> Vec<u8> {
  let mut cipher = Aes128Siv::new(key.into());
  cipher
    .encrypt([aad, nonce], plaintext)
    .expect("generated oracle input stays within the header-count bound")
}

#[test]
fn nonce_is_borrowed_non_empty_and_publicly_formatted() {
  assert_eq!(
    AesSivCmac256Nonce::try_from(&[][..]),
    Err(AesSivCmac256NonceError::new())
  );
  let bytes = [0x01, 0x23, 0x45];
  let nonce = AesSivCmac256Nonce::try_from(bytes.as_slice()).expect("non-empty nonce");
  assert_eq!(nonce.as_bytes(), bytes);
  assert_eq!(format!("{nonce:?}"), "AesSivCmac256Nonce(012345)");
}

#[test]
fn public_profile_matches_rustcrypto_across_boundaries_and_generated_inputs() {
  let lengths = [0usize, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1232];
  let nonce_lengths = [1usize, 2, 15, 16, 17, 31, 32, 127];
  let mut state = 0x6165_732d_7369_7621u64;

  for case in 0..256usize {
    let mut key = [0u8; 32];
    generated_bytes(&mut state, &mut key);
    let mut nonce_bytes = vec![0u8; nonce_lengths[case % nonce_lengths.len()]];
    let mut aad = vec![0u8; lengths[(case.strict_mul(5).strict_add(3)) % lengths.len()]];
    let mut plaintext = vec![0u8; lengths[case % lengths.len()]];
    generated_bytes(&mut state, &mut nonce_bytes);
    generated_bytes(&mut state, &mut aad);
    generated_bytes(&mut state, &mut plaintext);

    let cipher = AesSivCmac256::new(&AesSivCmac256Key::from_bytes(key));
    let nonce = AesSivCmac256Nonce::try_from(nonce_bytes.as_slice()).expect("generated nonce is non-empty");
    let expected = oracle(&key, &nonce_bytes, &aad, &plaintext);

    let mut combined = vec![0u8; plaintext.len().strict_add(AesSivCmac256::TAG_SIZE)];
    cipher
      .seal(nonce, &aad, &plaintext, &mut combined)
      .expect("output shape is exact");
    assert_eq!(combined, expected);

    let mut opened = vec![0u8; plaintext.len()];
    cipher
      .open(nonce, &aad, &combined, &mut opened)
      .expect("oracle-matching ciphertext authenticates");
    assert_eq!(opened, plaintext);

    let mut in_place = plaintext.clone();
    let tag = cipher.seal_in_place(nonce, &aad, &mut in_place);
    assert_eq!(tag.as_bytes(), &expected[..AesSivCmac256::TAG_SIZE]);
    assert_eq!(in_place, expected[AesSivCmac256::TAG_SIZE..]);
    cipher
      .open_in_place(nonce, &aad, &mut in_place, &tag)
      .expect("detached round trip authenticates");
    assert_eq!(in_place, plaintext);
  }
}

#[test]
fn every_tag_corruption_and_representative_input_corruption_clears_plaintext() {
  let key = AesSivCmac256Key::from_bytes([0x11; 32]);
  let cipher = AesSivCmac256::new(&key);
  let nonce_bytes = [0x22; 16];
  let nonce = AesSivCmac256Nonce::try_from(nonce_bytes.as_slice()).expect("non-empty nonce");
  let aad = [0x33; 33];
  let plaintext = [0x44; 65];
  let mut sealed = [0u8; 81];
  cipher.seal(nonce, &aad, &plaintext, &mut sealed).expect("exact output");

  for index in 0..AesSivCmac256::TAG_SIZE {
    let mut corrupted = sealed;
    corrupted[index] ^= 1;
    let mut out = [0xAA; 65];
    assert_eq!(
      cipher.open(nonce, &aad, &corrupted, &mut out),
      Err(OpenError::verification())
    );
    assert_eq!(out, [0u8; 65]);
  }

  for index in [AesSivCmac256::TAG_SIZE, 48, sealed.len().strict_sub(1)] {
    let mut corrupted = sealed;
    corrupted[index] ^= 1;
    let mut out = [0xAA; 65];
    assert_eq!(
      cipher.open(nonce, &aad, &corrupted, &mut out),
      Err(OpenError::verification())
    );
    assert_eq!(out, [0u8; 65]);
  }

  let mut wrong_aad = aad;
  wrong_aad[16] ^= 1;
  let mut out = [0xAA; 65];
  assert_eq!(
    cipher.open(nonce, &wrong_aad, &sealed, &mut out),
    Err(OpenError::verification())
  );
  assert_eq!(out, [0u8; 65]);

  let wrong_nonce_bytes = [0x23; 16];
  let wrong_nonce = AesSivCmac256Nonce::try_from(wrong_nonce_bytes.as_slice()).expect("non-empty nonce");
  out.fill(0xAA);
  assert_eq!(
    cipher.open(wrong_nonce, &aad, &sealed, &mut out),
    Err(OpenError::verification())
  );
  assert_eq!(out, [0u8; 65]);
}

#[test]
fn structural_errors_do_not_mutate_output() {
  let cipher = AesSivCmac256::new(&AesSivCmac256Key::from_bytes([0x11; 32]));
  let nonce_bytes = [0x22];
  let nonce = AesSivCmac256Nonce::try_from(nonce_bytes.as_slice()).expect("non-empty nonce");
  let mut out = [0xAA; 8];

  assert_eq!(cipher.seal(nonce, b"", b"data", &mut out), Err(SealError::buffer()));
  assert_eq!(out, [0xAA; 8]);
  assert_eq!(cipher.open(nonce, b"", &[0u8; 15], &mut out), Err(OpenError::buffer()));
  assert_eq!(out, [0xAA; 8]);
  assert_eq!(cipher.open(nonce, b"", &[0u8; 20], &mut out), Err(OpenError::buffer()));
  assert_eq!(out, [0xAA; 8]);
  assert!(matches!(
    AesSivCmac256::tag_from_slice(&[0u8; 15]),
    Err(err) if err == rscrypto::aead::AeadBufferError::new()
  ));
  assert_eq!(
    AesSivCmac256::tag_from_slice(&[0u8; 16]).map(AesSivCmac256Tag::to_bytes),
    Ok([0u8; 16])
  );
}

#[test]
fn key_context_and_tag_debug_follow_secret_boundaries() {
  let key = AesSivCmac256Key::from_bytes([0x53; 32]);
  assert_eq!(format!("{key:?}"), "AesSivCmac256Key(****)");
  assert_eq!(format!("{:?}", AesSivCmac256::new(&key)), "AesSivCmac256 { .. }");
  let tag = AesSivCmac256Tag::from_bytes([0x53; 16]);
  assert_eq!(format!("{tag:?}"), "AesSivCmac256Tag(53535353535353535353535353535353)");
}
