use super::*;

impl<K: AesKey> Gcm<K> {
  /// Exercise counter-mode encryption and fold the fixed diagnostic output to one block.
  #[inline(always)]
  fn diag_ctr32_be(&self, nonce: &Nonce96, plaintext: &[u8; 44]) -> [u8; 16] {
    let (_, ctr_block) = make_j0_and_ctr(nonce);
    let mut buffer = *plaintext;
    self.ek.ctr32_encrypt_be(&ctr_block, &mut buffer);
    crate::aead::ct_tests::fold16(&buffer)
  }
  /// Return the GHASH accumulator for diagnostic backend comparison.
  #[inline(always)]
  fn diag_ghash(&self, aad: &[u8], ciphertext: &[u8]) -> [u8; 16] {
    let h_polyval = self.h_powers_4()[3];
    let mut acc = GhashAccumulator(0);
    if self.should_use_wide_ghash(aad.len(), ciphertext.len()) {
      acc.0 = ghash_update_padded_wide(acc.0, h_polyval, self.h_powers_4(), aad);
      acc.0 = ghash_update_padded_wide(acc.0, h_polyval, self.h_powers_4(), ciphertext);
    } else {
      acc.0 = ghash_update_padded(acc.0, h_polyval, aad);
      acc.0 = ghash_update_padded(acc.0, h_polyval, ciphertext);
    }
    let length_block = match crate::aead::AeadByteLengths::try_new_bit_lengths(aad.len(), ciphertext.len()) {
      Ok(lengths) => lengths.to_be_bits_block(),
      Err(_) => return [0u8; 16],
    };
    acc.0 ^= u128::from_be_bytes(length_block);
    acc.0 = polyval::clmul128_reduce(acc.0, h_polyval);
    acc.0.to_be_bytes()
  }
  /// Encrypt a diagnostic GHASH accumulator into the final tag.
  #[inline(always)]
  fn diag_tag_aes(&self, nonce: &Nonce96, acc: &[u8; 16]) -> [u8; 16] {
    let (j0, _) = make_j0_and_ctr(nonce);
    self.encrypt_j0_tag(&j0, u128::from_be_bytes(*acc))
  }
}

/// Exercise AES-128 counter-mode encryption and fold the fixed diagnostic output to one block.
#[must_use]
pub(crate) fn diag_aes128gcm_ctr32_be(cipher: &Aes128Gcm, nonce: &Nonce96, plaintext: &[u8; 44]) -> [u8; 16] {
  cipher.inner.diag_ctr32_be(nonce, plaintext)
}

/// Return the AES-128-GCM GHASH accumulator for diagnostic backend comparison.
#[unsafe(no_mangle)]
#[inline(never)]
#[must_use]
pub(crate) fn diag_aes128gcm_ghash(cipher: &Aes128Gcm, aad: &[u8], ciphertext: &[u8]) -> [u8; 16] {
  cipher.inner.diag_ghash(aad, ciphertext)
}

/// Encrypt a diagnostic GHASH accumulator into the final AES-128-GCM tag.
#[must_use]
pub(crate) fn diag_aes128gcm_tag_aes(cipher: &Aes128Gcm, nonce: &Nonce96, acc: &[u8; 16]) -> [u8; 16] {
  cipher.inner.diag_tag_aes(nonce, acc)
}

/// Exercise AES-256 counter-mode encryption and fold the fixed diagnostic output to one block.
#[must_use]
pub(crate) fn diag_aes256gcm_ctr32_be(cipher: &Aes256Gcm, nonce: &Nonce96, plaintext: &[u8; 44]) -> [u8; 16] {
  cipher.inner.diag_ctr32_be(nonce, plaintext)
}

/// Return the AES-256-GCM GHASH accumulator for diagnostic backend comparison.
#[unsafe(no_mangle)]
#[inline(never)]
#[must_use]
pub(crate) fn diag_aes256gcm_ghash(cipher: &Aes256Gcm, aad: &[u8], ciphertext: &[u8]) -> [u8; 16] {
  cipher.inner.diag_ghash(aad, ciphertext)
}

/// Encrypt a diagnostic GHASH accumulator into the final AES-256-GCM tag.
#[must_use]
pub(crate) fn diag_aes256gcm_tag_aes(cipher: &Aes256Gcm, nonce: &Nonce96, acc: &[u8; 16]) -> [u8; 16] {
  cipher.inner.diag_tag_aes(nonce, acc)
}
