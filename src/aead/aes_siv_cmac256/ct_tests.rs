use super::*;

impl AesSivCmac256 {
  fn new_forced_portable(key: &AesSivCmac256Key) -> Self {
    let (cmac_key_bytes, ctr_key_bytes) = key
      .as_bytes()
      .split_first_chunk::<16>()
      .expect("AES-SIV-CMAC-256 key has two fixed 16-byte halves");
    let ctr_key_bytes = ctr_key_bytes
      .first_chunk::<16>()
      .expect("AES-SIV-CMAC-256 key has two fixed 16-byte halves");

    let cmac_key = aes::ct_tests::aes128_expand_key_forced_portable(cmac_key_bytes);
    let ctr_key = aes::ct_tests::aes128_expand_key_forced_portable(ctr_key_bytes);

    let mut l = [0u8; BLOCK_SIZE];
    aes::aes128_encrypt_block(&cmac_key, &mut l);
    let mut cmac_subkey1 = l;
    double_block(&mut cmac_subkey1);
    let mut cmac_subkey2 = cmac_subkey1;
    double_block(&mut cmac_subkey2);
    ct::zeroize(&mut l);

    Self {
      cmac_key,
      ctr_key,
      cmac_subkey1,
      cmac_subkey2,
    }
  }
}

/// Evaluate the private CMAC/S2V state through the forced-portable AES authority.
///
/// This diagnostic exists only for constant-time and backend-equivalence evidence. It is not a
/// supported CMAC, S2V, or deterministic-SIV product API.
#[inline(never)]
#[must_use]
pub(crate) fn diag_aes_siv_cmac256_s2v_portable(
  key: &[u8; KEY_SIZE],
  nonce: &[u8; BLOCK_SIZE],
  aad: &[u8; 32],
  plaintext: &[u8; 48],
) -> [u8; TAG_SIZE] {
  let key = AesSivCmac256Key::from_bytes(*key);
  let cipher = AesSivCmac256::new_forced_portable(&key);
  cipher.s2v(&[aad, nonce], plaintext)
}

/// Run a complete fixed-shape open through the forced-portable AES authority.
///
/// This diagnostic exists only for generated-code and timing evidence. The returned byte is an
/// opaque success indicator; authentication failure still clears the complete plaintext buffer.
#[inline(never)]
#[must_use]
pub(crate) fn diag_aes_siv_cmac256_open_portable(
  key: &[u8; KEY_SIZE],
  nonce: &[u8; BLOCK_SIZE],
  aad: &[u8; 32],
  ciphertext: &[u8; 48],
  tag: &[u8; TAG_SIZE],
) -> u8 {
  let key = AesSivCmac256Key::from_bytes(*key);
  let cipher = AesSivCmac256::new_forced_portable(&key);
  let nonce = AesSivCmac256Nonce::try_from(nonce.as_slice()).expect("fixed diagnostic nonce is non-empty");
  let tag = AesSivCmac256Tag::from_bytes(*tag);
  let mut buffer = *ciphertext;
  let accepted = cipher.open_in_place(nonce, aad, &mut buffer, &tag).is_ok();
  let digest = buffer.iter().copied().fold(0u8, |acc, byte| acc ^ byte);
  digest ^ u8::from(accepted)
}

#[test]
fn forced_portable_cmac_s2v_and_ctr_match_selected_backend() {
  const LENGTHS: &[usize] = &[0, 1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 127, 128, 129];

  for case in 0u8..32 {
    let mut key_bytes = [0u8; KEY_SIZE];
    for (index, byte) in key_bytes.iter_mut().enumerate() {
      let index = u8::try_from(index).expect("AES-SIV key index fits in u8");
      *byte = case.wrapping_mul(29).wrapping_add(index).rotate_left(1);
    }
    let key = AesSivCmac256Key::from_bytes(key_bytes);
    let selected = AesSivCmac256::new(&key);
    let portable = AesSivCmac256::new_forced_portable(&key);

    let mut storage = [0u8; 131];
    for (index, byte) in storage.iter_mut().enumerate() {
      let index = u8::try_from(index).expect("AES-SIV test input index fits in u8");
      *byte = case.wrapping_mul(17).wrapping_add(index).rotate_left(3);
    }
    let offset = usize::from(case & 1);
    let nonce = &storage[offset..offset.strict_add(16)];
    let aad = &storage[offset..offset.strict_add(32)];

    for &len in LENGTHS {
      let input = &storage[offset..offset.strict_add(len)];
      assert_eq!(selected.cmac(input), portable.cmac(input), "CMAC case={case} len={len}");
      assert_eq!(
        selected.s2v(&[aad, nonce], input),
        portable.s2v(&[aad, nonce], input),
        "S2V case={case} len={len}"
      );

      let tag = selected.s2v(&[aad, nonce], input);
      let mut selected_ctr = input.to_vec();
      let mut portable_ctr = selected_ctr.clone();
      selected.ctr_xor(&tag, &mut selected_ctr);
      portable.ctr_xor(&tag, &mut portable_ctr);
      assert_eq!(selected_ctr, portable_ctr, "CTR case={case} len={len}");
    }
  }
}
