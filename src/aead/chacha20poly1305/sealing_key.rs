#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::fmt;

use super::{ChaCha20Poly1305, ChaCha20Poly1305Key, ChaCha20Poly1305Tag};
use crate::aead::{Nonce96, NonceCounterSealError, expert::AeadWithNonce, nonce_sequence::NonceSequence};

/// A ChaCha20-Poly1305 key that owns its nonce sequence.
///
/// Give each sender a fresh key. Use that key for sealing only through this instance.
/// After a restart, fork, or snapshot restore, start with a new key.
/// A detected fork disables the inherited sealing key.
/// See [`ProcessOrigin`](crate::platform::ProcessOrigin) for detection limits, including builds without `std`.
/// Protocols that persist a key can reserve unique nonces durably and use [`AeadWithNonce`].
///
/// Each nonce contains four zero bytes followed by a big-endian 64-bit counter that starts at zero.
/// Each seal reserves a nonce before encryption, even when encryption fails.
/// The sequence stops permanently after `u64::MAX` attempts.
/// Store or send the returned nonce with the ciphertext.
///
/// This example uses `getrandom` to generate the key.
///
/// ```
/// # #[cfg(feature = "getrandom")]
/// # {
/// use rscrypto::{ChaCha20Poly1305, ChaCha20Poly1305Key, ChaCha20Poly1305SealingKey};
///
/// let key = ChaCha20Poly1305Key::try_random()?;
/// let opener = ChaCha20Poly1305::new(&key);
/// let mut sealer = ChaCha20Poly1305SealingKey::new(key);
/// let mut message = *b"hello";
/// let (nonce, tag) = sealer.seal_in_place(b"header", &mut message)?;
/// opener.decrypt_in_place(&nonce, b"header", &mut message, &tag)?;
/// assert_eq!(&message, b"hello");
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ChaCha20Poly1305SealingKey {
  cipher: ChaCha20Poly1305,
  nonces: NonceSequence<{ u64::MAX }>,
}

impl fmt::Debug for ChaCha20Poly1305SealingKey {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("ChaCha20Poly1305SealingKey").finish_non_exhaustive()
  }
}

impl ChaCha20Poly1305SealingKey {
  /// Bind a fresh key to a single nonce sequence.
  #[must_use]
  pub fn new(key: ChaCha20Poly1305Key) -> Self {
    Self {
      cipher: ChaCha20Poly1305 { key },
      nonces: NonceSequence::new(0),
    }
  }

  /// Seal into `out` as `ciphertext || tag` and return the nonce.
  ///
  /// # Errors
  ///
  /// Returns [`NonceCounterSealError`] if the sequence cannot issue a nonce or encryption fails.
  pub fn seal(&mut self, aad: &[u8], plaintext: &[u8], out: &mut [u8]) -> Result<Nonce96, NonceCounterSealError> {
    let nonce = self.nonces.advance([0; 4])?;
    self.cipher.encrypt(&nonce, aad, plaintext, out)?;
    Ok(nonce)
  }

  /// Seal in place and return the nonce and detached tag.
  ///
  /// # Errors
  ///
  /// Returns [`NonceCounterSealError`] if the sequence cannot issue a nonce or encryption fails.
  pub fn seal_in_place(
    &mut self,
    aad: &[u8],
    buffer: &mut [u8],
  ) -> Result<(Nonce96, ChaCha20Poly1305Tag), NonceCounterSealError> {
    let nonce = self.nonces.advance([0; 4])?;
    let tag = self.cipher.encrypt_in_place(&nonce, aad, buffer)?;
    Ok((nonce, tag))
  }

  /// Seal ordered plaintext slices into `out` and return the nonce and detached tag.
  ///
  /// # Errors
  ///
  /// Returns [`NonceCounterSealError`] if the sequence cannot issue a nonce or gather encryption fails.
  pub fn seal_gather<'input>(
    &mut self,
    aad: &[u8],
    plaintext: impl IntoIterator<Item = &'input [u8]>,
    out: &mut [u8],
  ) -> Result<(Nonce96, ChaCha20Poly1305Tag), NonceCounterSealError> {
    let nonce = self.nonces.advance([0; 4])?;
    let tag = self.cipher.encrypt_gather(&nonce, aad, plaintext, out)?;
    Ok((nonce, tag))
  }

  /// Allocate `ciphertext || tag` and return it with the nonce.
  ///
  /// # Errors
  ///
  /// Returns [`NonceCounterSealError`] if the sequence cannot issue a nonce or encryption fails.
  #[cfg(feature = "alloc")]
  #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
  pub fn seal_to_vec(&mut self, aad: &[u8], plaintext: &[u8]) -> Result<(Nonce96, Vec<u8>), NonceCounterSealError> {
    let nonce = self.nonces.advance([0; 4])?;
    let sealed = self.cipher.encrypt_to_vec(&nonce, aad, plaintext)?;
    Ok((nonce, sealed))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn chacha_sealing_key_stops_permanently_before_wrap() {
    let mut sealer = ChaCha20Poly1305SealingKey::new(ChaCha20Poly1305Key::from_bytes([0x42; 32]));
    sealer.nonces = NonceSequence::new(u64::MAX - 1);
    let mut out = [0; 16];
    let nonce = sealer.seal(b"", b"", &mut out).expect("last nonce");
    assert_eq!(nonce.to_bytes(), [0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 254]);
    let last = out;
    for _ in 0..2 {
      assert_eq!(sealer.seal(b"", b"", &mut out), Err(NonceCounterSealError::exhausted()));
      assert_eq!(out, last);
    }
  }
}
