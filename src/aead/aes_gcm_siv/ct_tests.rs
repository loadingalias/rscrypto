use super::*;

impl<K: AesKey> GcmSiv<K> {
  /// Derive the per-nonce keys through the production dispatch, for diagnostic comparison.
  #[inline(always)]
  fn diag_derive_keys(&self, nonce: &Nonce96) -> ([u8; 16], K::Bytes) {
    #[cfg(target_arch = "aarch64")]
    if matches!(
      self.backend,
      AeadBackend::Aarch64AesPmull | AeadBackend::Aarch64Sve2AesPmull
    ) {
      // SAFETY: mirror the production AArch64 GCM-SIV key-derivation dispatch because:
      // 1. Backend resolution selected an AArch64 AES+PMULL backend.
      // 2. The method falls back to the portable KDF when `master_ek` is not an AES-CE schedule.
      // 3. `nonce.as_bytes()` is exactly the 96-bit GCM-SIV nonce.
      return unsafe { self.master_ek.aarch64_gcmsiv_derive_keys(nonce.as_bytes()) };
    }
    self.master_ek.gcmsiv_derive_keys(nonce.as_bytes())
  }
}

/// Encrypt one diagnostic tag block with a raw key.
#[inline(always)]
fn diag_raw_tag_aes<K: AesKey>(enc_key: &K::Bytes, block: &[u8; 16]) -> [u8; 16] {
  let mut out = *block;
  #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
  {
    // SAFETY: diagnostic s390x CT runs execute on the native MSA runner.
    unsafe { K::s390x_encrypt_block_raw(enc_key, &mut out) };
  }
  #[cfg(not(all(target_arch = "s390x", not(feature = "portable-only"))))]
  K::expand(enc_key).encrypt_block(&mut out);
  out
}

/// Exercise counter-mode encryption with a raw key and fold the fixed diagnostic output to one
/// block.
#[inline(always)]
fn diag_ctr32<K: AesKey>(enc_key: &K::Bytes, tag: &[u8; 16], plaintext: &[u8; 44]) -> [u8; 16] {
  let counter_block = counter_block(tag);
  let mut buffer = *plaintext;
  #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
  {
    let ctr = u32::from_le_bytes([counter_block[0], counter_block[1], counter_block[2], counter_block[3]]);
    ctr32_le_xor_blocks(&counter_block, ctr, &mut buffer, |block| {
      // SAFETY: diagnostic s390x CT runs execute on the native MSA runner.
      unsafe { K::s390x_encrypt_block_raw(enc_key, block) }
    });
  }
  #[cfg(not(all(target_arch = "s390x", not(feature = "portable-only"))))]
  K::expand(enc_key).ctr32_encrypt(&counter_block, &mut buffer);
  crate::aead::ct_tests::fold16(&buffer)
}

/// Derive the AES-128-GCM-SIV per-nonce keys for diagnostic comparison.
#[must_use]
pub(crate) fn diag_aes128gcmsiv_derive_keys(cipher: &Aes128GcmSiv, nonce: &Nonce96) -> ([u8; 16], [u8; 16]) {
  cipher.inner.diag_derive_keys(nonce)
}

/// Derive the AES-256-GCM-SIV per-nonce keys for diagnostic comparison.
#[must_use]
pub(crate) fn diag_aes256gcmsiv_derive_keys(cipher: &Aes256GcmSiv, nonce: &Nonce96) -> ([u8; 16], [u8; 32]) {
  cipher.inner.diag_derive_keys(nonce)
}

/// Return the AES-GCM-SIV POLYVAL digest before nonce and AES tag finalization.
#[must_use]
pub(crate) fn diag_aes128gcmsiv_polyval_digest(auth_key: &[u8; 16], aad: &[u8], plaintext: &[u8]) -> [u8; 16] {
  let mut pv = polyval::Polyval::new(auth_key);
  pv.update_padded(aad);
  pv.update_padded(plaintext);
  let length_block = crate::aead::AeadByteLengths::from_usize(aad.len(), plaintext.len()).to_le_bits_block();
  pv.update_block(&length_block);
  pv.finalize()
}

/// Encrypt one diagnostic tag block with a raw AES-128 key.
#[must_use]
pub(crate) fn diag_aes128gcmsiv_raw_tag_aes(enc_key: &[u8; 16], block: &[u8; 16]) -> [u8; 16] {
  diag_raw_tag_aes::<aes::Aes128EncKey>(enc_key, block)
}

/// Encrypt one diagnostic tag block with a raw AES-256 key.
#[must_use]
pub(crate) fn diag_aes256gcmsiv_raw_tag_aes(enc_key: &[u8; 32], block: &[u8; 16]) -> [u8; 16] {
  diag_raw_tag_aes::<aes::Aes256EncKey>(enc_key, block)
}

/// Exercise AES-128 counter-mode encryption and fold the fixed diagnostic output to one block.
#[must_use]
pub(crate) fn diag_aes128gcmsiv_ctr32(enc_key: &[u8; 16], tag: &[u8; 16], plaintext: &[u8; 44]) -> [u8; 16] {
  diag_ctr32::<aes::Aes128EncKey>(enc_key, tag, plaintext)
}

/// Exercise AES-256 counter-mode encryption and fold the fixed diagnostic output to one block.
#[must_use]
pub(crate) fn diag_aes256gcmsiv_ctr32(enc_key: &[u8; 32], tag: &[u8; 16], plaintext: &[u8; 44]) -> [u8; 16] {
  diag_ctr32::<aes::Aes256EncKey>(enc_key, tag, plaintext)
}
