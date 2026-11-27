//! RSAES-PKCS1-v1_5 implicit rejection follows section 7 of draft-irtf-cfrg-rsa-guidance-10.
//! On a padding failure, the private exponent and ciphertext determine the synthetic message.

//! RSAES-PKCS1-v1_5 decryption with implicit rejection.

use super::*;

impl RsaPrivateKeyComponents {
  pub(super) fn decrypt_pkcs1v15_with_blinding_factor(
    &self,
    ciphertext: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
  ) -> Result<usize, RsaPrivateOpError> {
    let mut encoded = vec![0u8; self.public.modulus().len()];
    let mut alternative = vec![0u8; self.public.modulus().len()];
    let result = self
      .private_operation_with_blinding_factor(ciphertext, blinding, &mut encoded)
      .and_then(|()| {
        decode_pkcs1v15_with_implicit_rejection(
          self.private_exponent.as_bytes(),
          ciphertext,
          &encoded,
          &mut alternative,
          out,
        )
      });
    ct::zeroize(&mut encoded);
    ct::zeroize(&mut alternative);
    clear_decryption_output_on_error(result, out)
  }

  #[cfg(feature = "getrandom")]
  pub(super) fn decrypt_pkcs1v15_with_stored_blinding_and_scratch(
    &self,
    ciphertext: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<usize, RsaPrivateOpError> {
    let result = scratch.ensure_len(self.public.modulus().len()).and_then(|()| {
      if ciphertext.len() != scratch.encoded.as_slice().len() {
        Err(RsaPrivateOpError::InvalidLength)
      } else {
        scratch.encoded.as_mut_slice().copy_from_slice(ciphertext);
        self
          .private_operation_from_scratch_encoded_with_stored_blinding(scratch)
          .and_then(|()| {
            decode_pkcs1v15_with_implicit_rejection(
              self.private_exponent.as_bytes(),
              ciphertext,
              scratch.blinded_private_result.as_slice(),
              scratch.blinding_power.as_mut_slice(),
              out,
            )
          })
      }
    });
    clear_decryption_output_on_error(result, out)
  }

  pub(super) fn decrypt_pkcs1v15_with_blinding_factor_and_scratch(
    &self,
    ciphertext: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<usize, RsaPrivateOpError> {
    let result = scratch.ensure_len(self.public.modulus().len()).and_then(|()| {
      if ciphertext.len() != scratch.encoded.as_slice().len() {
        Err(RsaPrivateOpError::InvalidLength)
      } else {
        scratch.encoded.as_mut_slice().copy_from_slice(ciphertext);
        self
          .private_operation_from_scratch_encoded(blinding, scratch)
          .and_then(|()| {
            decode_pkcs1v15_with_implicit_rejection(
              self.private_exponent.as_bytes(),
              ciphertext,
              scratch.blinded_private_result.as_slice(),
              scratch.blinding_power.as_mut_slice(),
              out,
            )
          })
      }
    });
    scratch.clear();
    clear_decryption_output_on_error(result, out)
  }
}

/// Finish RSAES-PKCS1-v1_5 decryption of `encoded` (EM) for `ciphertext`
/// with implicit rejection. `ciphertext` has already passed the length and
/// range checks; `alternative` is `k` octets of scratch, cleared on return.
fn decode_pkcs1v15_with_implicit_rejection(
  private_exponent: &[u8],
  ciphertext: &[u8],
  encoded: &[u8],
  alternative: &mut [u8],
  out: &mut [u8],
) -> Result<usize, RsaPrivateOpError> {
  let mut kdk = implicit_rejection_kdk(private_exponent, ciphertext);
  let mut candidates = [0u8; IMPLICIT_REJECTION_CANDIDATE_BYTES];
  let result = implicit_rejection_prf(&kdk, b"length", &mut candidates)
    .and_then(|()| implicit_rejection_prf(&kdk, b"message", alternative))
    .and_then(|()| {
      let max_len = encoded
        .len()
        .checked_sub(RSAES_PKCS1V15_OVERHEAD)
        .ok_or(RsaPrivateOpError::InvalidLength)?;
      let alternative_len = implicit_rejection_length(&candidates, max_len);
      decode_pkcs1v15_implicit_rejection(encoded, alternative, alternative_len, out)
    });
  ct::zeroize(&mut kdk);
  ct::zeroize(&mut candidates);
  ct::zeroize(alternative);
  result
}

/// Derive the implicit-rejection key derivation key (draft section 7.2, step 3.1):
/// `KDK = HMAC-SHA256(SHA256(I2OSP(d, k)), C)`.
///
/// `d` is retained at the public modulus width on import or generation, never
/// recomputed from CRT values. This keeps the derivation's lengths and memory
/// accesses independent of the exponent's significant-byte count.
pub(super) fn implicit_rejection_kdk(private_exponent: &[u8], ciphertext: &[u8]) -> [u8; HMAC_SHA256_OUT_BYTES] {
  debug_assert_eq!(private_exponent.len(), ciphertext.len());
  let mut exponent_hash = Sha256::digest_secret(private_exponent);
  let kdk = hmac_sha256_parts(&exponent_hash, [ciphertext]);
  ct::zeroize(&mut exponent_hash);
  kdk
}

/// Fill `out` with the implicit-rejection PRF (draft section 7.1):
/// HMAC-SHA256 blocks over `I || label || bitLength`, with two-octet
/// big-endian `I` and `bitLength`.
pub(super) fn implicit_rejection_prf(
  kdk: &[u8; HMAC_SHA256_OUT_BYTES],
  label: &[u8],
  out: &mut [u8],
) -> Result<(), RsaPrivateOpError> {
  // The draft bounds the output at 8192 octets; every supported modulus is far below it.
  let bit_length = out
    .len()
    .checked_mul(8)
    .and_then(|bits| u16::try_from(bits).ok())
    .ok_or(RsaPrivateOpError::InvalidLength)?
    .to_be_bytes();
  for (index, chunk) in out.chunks_mut(HMAC_SHA256_OUT_BYTES).enumerate() {
    let index = u16::try_from(index)
      .map_err(|_| RsaPrivateOpError::InvalidLength)?
      .to_be_bytes();
    let mut block = hmac_sha256_parts(kdk, [index.as_slice(), label, &bit_length]);
    chunk.copy_from_slice(&block[..chunk.len()]);
    ct::zeroize(&mut block);
  }
  Ok(())
}

/// Select the synthetic message length (draft section 7.2, step 3.3): mask each
/// candidate to the bit length of `max_len`, then keep the last one that is at
/// most `max_len`, or zero if none is.
pub(super) fn implicit_rejection_length(
  candidates: &[u8; IMPLICIT_REJECTION_CANDIDATE_BYTES],
  max_len: usize,
) -> usize {
  let bits = usize::BITS.strict_sub(max_len.leading_zeros());
  let mask = (1usize << bits).strict_sub(1);
  let mut selected = 0usize;
  for pair in candidates.as_chunks::<2>().0 {
    let candidate = usize::from(u16::from_be_bytes([pair[0], pair[1]])) & mask;
    // Both values are below 2^16, so the borrow bit decides `candidate <= max_len`.
    let fits = (max_len.wrapping_sub(candidate) >> (usize::BITS - 1)).to_le_bytes()[0] ^ 1;
    selected = ct_select_usize(selected, candidate, fits);
  }
  selected
}

/// Decode `encoded` (EM) with implicit rejection (draft section 7.2, steps 4-5).
///
/// `alternative` holds the synthetic message AM on entry; `alternative_len` is
/// its selected length. Both are read in full whatever the padding check
/// decides. Writes the selected message to the front of `out[..k - 11]`, zero
/// fills the rest of that window, clears `alternative`, and returns the
/// message length. The write pattern does not depend on the decision or the
/// length.
pub(super) fn decode_pkcs1v15_implicit_rejection(
  encoded: &[u8],
  alternative: &mut [u8],
  alternative_len: usize,
  out: &mut [u8],
) -> Result<usize, RsaPrivateOpError> {
  let k = encoded.len();
  let max_len = k
    .checked_sub(RSAES_PKCS1V15_OVERHEAD)
    .ok_or(RsaPrivateOpError::InvalidLength)?;
  if alternative.len() != k {
    return Err(RsaPrivateOpError::InvalidLength);
  }
  debug_assert!(alternative_len <= max_len);
  let out = out.get_mut(..max_len).ok_or(RsaPrivateOpError::InvalidLength)?;

  let mut bad = u8::from(encoded.first().copied() != Some(0));
  bad |= u8::from(encoded.get(1).copied() != Some(2));
  let mut seen_separator = 0u8;
  let mut separator = 0usize;
  for (index, &byte) in encoded.iter().enumerate().skip(2) {
    let before_separator = seen_separator ^ 1;
    // SECURITY: RISC-V LLVM lowered `before_separator & (byte == 0)` to a
    // branch on the plaintext octet. Keep the comparison result opaque.
    let is_zero = core::hint::black_box(u8::from(byte == 0));
    let separator_at_index = before_separator & is_zero;
    separator = ct_select_usize(separator, index, separator_at_index);
    seen_separator |= separator_at_index;
  }
  bad |= seen_separator ^ 1;
  // PS = EM[2..separator] must hold at least eight octets.
  bad |= u8::from(separator < 10);

  // L is the length after the separator, or zero without one (step 4). The
  // separator index is below `k`; wrapping arithmetic keeps the secret index
  // out of an overflow-panic branch.
  debug_assert!(separator < k);
  let message_len = ct_select_usize(0, k.wrapping_sub(1).wrapping_sub(separator), seen_separator);
  let selected_len = ct_select_usize(message_len, alternative_len, bad);

  // SECURITY: Keep the decision opaque so LLVM cannot unswitch these loops or
  // lower the selection to branches on the padding result.
  let use_alternative = core::hint::black_box(0u8.wrapping_sub(bad & 1));
  for (chosen, &byte) in alternative.iter_mut().zip(encoded) {
    *chosen = byte ^ (use_alternative & (byte ^ *chosen));
  }

  // The selected message is the last `selected_len` octets. Within the
  // `max_len`-octet window that ends at octet `k`, it starts at
  // `max_len - selected_len`; shift it to the front in log2 fixed-pattern
  // steps, filling with zeros.
  let window = &mut alternative[RSAES_PKCS1V15_OVERHEAD..];
  // Valid messages and every candidate length are at most `max_len`.
  debug_assert!(selected_len <= max_len);
  let shift = max_len.wrapping_sub(selected_len);
  let mut bit = 0u32;
  while (1usize << bit) <= max_len {
    let step = 1usize << bit;
    let take = core::hint::black_box(0u8.wrapping_sub(((shift >> bit) & 1).to_le_bytes()[0]));
    for index in 0..max_len {
      let source = match index.checked_add(step) {
        Some(source) if source < max_len => window[source],
        _ => 0,
      };
      window[index] ^= take & (window[index] ^ source);
    }
    bit = bit.strict_add(1);
  }
  out.copy_from_slice(window);
  ct::zeroize(alternative);
  Ok(selected_len)
}

#[cfg(test)]
mod tests;

#[cfg(all(test, not(miri), feature = "portable-only", feature = "rsa"))]
mod binsec {
  use super::*;
  use crate::binsec::*;

  #[unsafe(no_mangle)]
  #[inline(never)]
  pub(super) extern "C" fn ct_binsec_rsa_implicit_rejection_kdk_256() -> ! {
    // SAFETY: Each pointer references an initialized fixed-size harness global.
    let exponent = unsafe { array_from_global(&raw const CT_BINSEC_RSA_EXPONENT_256) };
    // SAFETY: This pointer references an initialized fixed-size harness global.
    let ciphertext = unsafe { array_from_global(&raw const CT_BINSEC_RSA_CIPHERTEXT_256) };
    let kdk = implicit_rejection_kdk(&exponent, &ciphertext);
    ct_binsec_done(kdk.iter().fold(0, |acc, byte| acc ^ byte))
  }

  #[unsafe(no_mangle)]
  #[inline(never)]
  pub(super) extern "C" fn ct_binsec_rsa_implicit_rejection_expand_256() -> ! {
    // SAFETY: This pointer references an initialized fixed-size harness global.
    let kdk = unsafe { array_from_global(&raw const CT_BINSEC_KEY_32) };
    let mut candidates = [0u8; 256];
    let mut alternative = [0u8; 256];
    let result = implicit_rejection_prf(&kdk, b"length", &mut candidates)
      .and_then(|()| implicit_rejection_prf(&kdk, b"message", &mut alternative));
    let digest = candidates.iter().chain(&alternative).fold(0, |acc, byte| acc ^ byte);
    ct_binsec_done(digest ^ u8::from(result.is_ok()))
  }

  #[unsafe(no_mangle)]
  #[inline(never)]
  pub(super) extern "C" fn ct_binsec_rsa_implicit_rejection_decode_256() -> ! {
    // SAFETY: Each pointer references an initialized fixed-size harness global.
    let encoded = unsafe { array_from_global(&raw const CT_BINSEC_RSA_ENCODED_256) };
    // SAFETY: This pointer references an initialized fixed-size harness global.
    let candidates = unsafe { array_from_global(&raw const CT_BINSEC_RSA_CANDIDATES_256) };
    // SAFETY: This pointer references an initialized fixed-size harness global.
    let mut alternative = unsafe { array_from_global(&raw const CT_BINSEC_RSA_ALTERNATIVE_256) };
    let mut out = [0u8; 245];
    let len = decode_pkcs1v15_implicit_rejection(
      &encoded,
      &mut alternative,
      implicit_rejection_length(&candidates, 245),
      &mut out,
    )
    .unwrap_or(0);
    let digest = out.iter().fold(0, |acc, byte| acc ^ byte);
    ct_binsec_done(digest ^ len.to_le_bytes()[0])
  }
}
