use super::{RsaBlindingPair, RsaPrivateOpError, RsaPrivateScratch};

/// RSAES-PKCS1-v1_5 decryption of legacy data, with implicit rejection.
///
/// Import this trait explicitly only to decrypt existing RSAES-PKCS1-v1_5
/// data, such as OpenPGP, CMS key-transport, or JWE `RSA1_5` messages. New
/// protocols must use RSAES-OAEP through
/// [`RsaPrivateKey::decrypt_oaep`](super::RsaPrivateKey::decrypt_oaep). Do not
/// expose this operation as an online decryption endpoint.
///
/// Decryption follows the implicit rejection of
/// [draft-irtf-cfrg-rsa-guidance-10](https://datatracker.ietf.org/doc/draft-irtf-cfrg-rsa-guidance/10/)
/// section 7.2, which OpenSSL 3.2 and later also apply by default. A
/// ciphertext with invalid padding does not produce an error: it decrypts to a
/// synthetic message of at most `k - 11` bytes, derived from the private
/// exponent and the ciphertext, and the same ciphertext always yields the same
/// message. A returned message therefore does not show that the padding was
/// valid. Authenticate the recovered data, for example with the protocol's MAC
/// or AEAD, before acting on it. Errors report only public structure (the
/// output length, the ciphertext length, and a ciphertext representative that
/// is not below the modulus) and blinding, scratch, entropy, or fault-check
/// failures.
///
/// TLS 1.2 RSA key exchange needs the fixed-length premaster-secret handling
/// of RFC 5246 section 7.4.7.1, which this trait does not provide.
pub trait RsaPkcs1v15Decrypt {
  /// Decrypt an RSAES-PKCS1-v1_5 ciphertext using OS-backed blinding.
  ///
  /// `out` must hold at least `k - 11` bytes for a `k`-byte modulus. A shorter
  /// `out` fails with [`RsaPrivateOpError::InvalidLength`] before decryption,
  /// whatever the ciphertext. On success the message is `out[..len]`, where
  /// `len` is the returned length, and the rest of `out[..k - 11]` is zero.
  /// On failure `out` is zeroed.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, a length is
  /// invalid, or the ciphertext representative is out of range. Invalid
  /// padding is not an error.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA decryption failure must be checked; a dropped Result silently discards plaintext"]
  fn decrypt_pkcs1v15(&self, ciphertext: &[u8], out: &mut [u8]) -> Result<usize, RsaPrivateOpError>;

  /// Decrypt an RSAES-PKCS1-v1_5 ciphertext using OS-backed blinding and
  /// caller-owned scratch.
  ///
  /// Reusing scratch avoids top-level private-operation temporary allocation
  /// after setup. The `out` contract is the same as
  /// [`Self::decrypt_pkcs1v15`].
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, `scratch` was
  /// allocated for a different modulus width, a length is invalid, or the
  /// ciphertext representative is out of range. Invalid padding is not an
  /// error.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA decryption failure must be checked; a dropped Result silently discards plaintext"]
  fn decrypt_pkcs1v15_with_scratch(
    &self,
    ciphertext: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<usize, RsaPrivateOpError>;

  /// Decrypt an RSAES-PKCS1-v1_5 ciphertext with caller-supplied blinding.
  ///
  /// The blinding pair must satisfy the requirements of
  /// [`RsaBlindingPair::new`]. The `out` contract is the same as for OS-backed
  /// blinding.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if a length is invalid, the blinding pair is
  /// invalid, or the ciphertext representative is out of range. Invalid
  /// padding is not an error.
  #[must_use = "RSA decryption failure must be checked; a dropped Result silently discards plaintext"]
  fn decrypt_pkcs1v15_with_blinding_factor(
    &self,
    ciphertext: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
  ) -> Result<usize, RsaPrivateOpError>;

  /// Decrypt an RSAES-PKCS1-v1_5 ciphertext with caller-supplied blinding and
  /// scratch.
  ///
  /// Reusing scratch avoids top-level private-operation temporary allocation
  /// after setup. The blinding-pair and `out` contracts are the same as
  /// [`Self::decrypt_pkcs1v15_with_blinding_factor`].
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if a length is invalid, `scratch` was
  /// allocated for a different modulus width, the blinding pair is invalid,
  /// or the ciphertext representative is out of range. Invalid padding is not
  /// an error.
  #[must_use = "RSA decryption failure must be checked; a dropped Result silently discards plaintext"]
  fn decrypt_pkcs1v15_with_blinding_factor_and_scratch(
    &self,
    ciphertext: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<usize, RsaPrivateOpError>;
}
