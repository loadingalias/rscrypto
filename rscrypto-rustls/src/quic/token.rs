//! Stateless-reset tokens and address tokens.

use alloc::{boxed::Box, vec::Vec};
use core::sync::atomic::{AtomicBool, Ordering};

use rscrypto::aead::{Nonce96, expert::AeadWithNonce};
use rscrypto::{Aes256Gcm, Aes256GcmKey, Aes256GcmTag, HkdfSha256, HmacSha256, HmacSha256Tag, SecretBytes};
use rustls::Error;
use rustls::crypto::{GetRandomFailed, SecureRandom};

const RESET_TOKEN_LEN: usize = 16;
const TOKEN_NONCE_LEN: usize = 16;
const TOKEN_TAG_LEN: usize = 16;
const MAX_TOKEN_PLAINTEXT: usize = 4096;
const TOKEN_LABEL: &[u8] = b"rscrypto-rustls QUIC token v1";

/// Derives QUIC stateless-reset tokens from a static key (RFC 9000, section 10.3.2).
///
/// Each token is the first 16 bytes of HMAC-SHA256, under this key, of the
/// connection ID that it accompanies. Send it with the ID in a
/// NEW_CONNECTION_ID frame or the `stateless_reset_token` transport parameter.
/// To reset a connection whose state is lost, end a Stateless Reset with the
/// token of the packet's Destination Connection ID.
///
/// Keep one key for endpoints that replace each other, for example after a
/// restart on the same address, so that they can reset each other's
/// connections. The key clears on drop. Share one key across threads with `Arc`.
pub struct StatelessResetKey(SecretBytes<32>);

impl StatelessResetKey {
  /// Length of every token.
  pub const TOKEN_LEN: usize = RESET_TOKEN_LEN;

  /// Generate a key from 32 bytes of `random`.
  ///
  /// The source must provide cryptographically secure random bytes. A failure
  /// returns the error without fallback.
  pub fn generate(random: &dyn SecureRandom) -> Result<Self, GetRandomFailed> {
    SecretBytes::try_fill_with(|out| random.fill(out)).map(Self)
  }

  /// The stateless-reset token for `connection_id`.
  #[must_use]
  pub fn token(&self, connection_id: &[u8]) -> [u8; RESET_TOKEN_LEN] {
    let mut token = [0u8; RESET_TOKEN_LEN];
    token.copy_from_slice(&self.mac(connection_id).as_bytes()[..RESET_TOKEN_LEN]);
    token
  }

  /// The full HMAC-SHA256 of `data`.
  pub(crate) fn mac(&self, data: &[u8]) -> HmacSha256Tag {
    HmacSha256::mac(self.0.as_bytes(), data)
  }

  /// Verify a full 32-byte HMAC-SHA256 of `data` without an early exit.
  #[cfg(feature = "quinn")]
  pub(crate) fn verify_mac(&self, data: &[u8], signature: &[u8]) -> bool {
    use rscrypto::Mac as _;
    let Ok(signature) = <[u8; 32]>::try_from(signature) else {
      return false;
    };
    let mut mac = HmacSha256::new(self.0.as_bytes());
    mac.update(data);
    mac.verify(&HmacSha256Tag::from_bytes(signature)).is_ok()
  }
}

impl core::fmt::Debug for StatelessResetKey {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("StatelessResetKey").finish_non_exhaustive()
  }
}

/// Seals and opens QUIC address tokens for Retry and NEW_TOKEN frames (RFC 9000, section 8.1).
///
/// A token is `ciphertext || tag || nonce`. [`Self::seal`] draws the 16-byte
/// nonce from the caller's source. HKDF-SHA256 expands this key with a fixed
/// label and the nonce into a one-use AES-256-GCM key, which encrypts with a
/// zero AEAD nonce. An AEAD key therefore never encrypts twice unless the
/// source repeats a nonce. Quinn writes the same layout, so the `quinn`
/// adapter's address tokens open here under the same key.
///
/// The caller owns the plaintext format, such as the client address, the
/// original Destination Connection ID, and the issue time. Authenticate context
/// that the token does not carry through `aad`. A token stays valid for its key's
/// lifetime and can be replayed: check expiry and any single-use policy after
/// [`Self::open`]. Distinguish Retry tokens from NEW_TOKEN tokens in the
/// plaintext or `aad`.
///
/// The key clears on drop. Share one key across threads with `Arc`.
pub struct AddressTokenKey(HkdfSha256);

impl AddressTokenKey {
  /// Length of the nonce at the end of every token.
  pub const NONCE_LEN: usize = TOKEN_NONCE_LEN;
  /// Bytes that sealing adds to the plaintext: the tag and the nonce.
  pub const OVERHEAD: usize = TOKEN_TAG_LEN + TOKEN_NONCE_LEN;
  /// Longest plaintext that a token can carry.
  pub const MAX_PLAINTEXT_LEN: usize = MAX_TOKEN_PLAINTEXT;

  /// Generate a key from 32 bytes of `random`.
  ///
  /// The source must provide cryptographically secure random bytes. A failure
  /// returns the error without fallback.
  pub fn generate(random: &dyn SecureRandom) -> Result<Self, GetRandomFailed> {
    let key = SecretBytes::<32>::try_fill_with(|out| random.fill(out))?;
    Ok(Self(HkdfSha256::from_prk(key.as_bytes())))
  }

  /// Seal the plaintext in `data` into a token, in place.
  ///
  /// Draws a 16-byte nonce from `random`, encrypts `data`, then appends the
  /// tag and the nonce.
  ///
  /// # Errors
  ///
  /// Returns [`Error::EncryptError`] if `data` exceeds
  /// [`Self::MAX_PLAINTEXT_LEN`] bytes, before drawing entropy, and
  /// [`Error::FailedToGetRandomBytes`] if the source fails. `data` is
  /// unchanged after an error.
  pub fn seal(&self, random: &dyn SecureRandom, data: &mut Vec<u8>, aad: &[u8]) -> Result<(), Error> {
    if data.len() > MAX_TOKEN_PLAINTEXT {
      return Err(Error::EncryptError);
    }
    let mut nonce = [0u8; TOKEN_NONCE_LEN];
    random.fill(&mut nonce)?;
    self.cipher(&nonce).seal_in_place(data, aad)?;
    data.extend_from_slice(&nonce);
    Ok(())
  }

  /// Open `token` in place and return its plaintext, a prefix of `token`.
  ///
  /// # Errors
  ///
  /// Returns [`Error::DecryptError`] for every rejection: a length outside
  /// [`Self::OVERHEAD`] through `MAX_PLAINTEXT_LEN + OVERHEAD`, a wrong key,
  /// wrong `aad`, or any altered byte. A length rejection leaves `token`
  /// unchanged; an authentication failure zeroes the bytes before the tag.
  pub fn open<'a>(&self, token: &'a mut [u8], aad: &[u8]) -> Result<&'a mut [u8], Error> {
    if !(Self::OVERHEAD..=MAX_TOKEN_PLAINTEXT.strict_add(Self::OVERHEAD)).contains(&token.len()) {
      return Err(Error::DecryptError);
    }
    let (sealed, nonce) = token.split_at_mut(token.len().strict_sub(TOKEN_NONCE_LEN));
    self.cipher(nonce).open_in_place(sealed, aad)
  }

  /// The one-use AEAD context for one token nonce.
  pub(crate) fn cipher(&self, nonce: &[u8]) -> Box<TokenCipher> {
    let key = SecretBytes::<32>::try_fill_with(|out| self.0.expand_vectored(&[TOKEN_LABEL, nonce], out))
      .expect("one SHA-256 block is a valid HKDF output");
    let key = Aes256GcmKey::generate(|out| out.copy_from_slice(key.as_bytes()));
    #[cfg(all(
      target_arch = "aarch64",
      target_endian = "little",
      target_os = "linux",
      target_env = "gnu",
      not(debug_assertions),
      not(feature = "portable-only")
    ))]
    {
      // Clear retired construction frames before the borrowed input key drops.
      let _scrub = crate::aead::ConstructionStackScrub;
      boxed_token_cipher(&key)
    }
    #[cfg(not(all(
      target_arch = "aarch64",
      target_endian = "little",
      target_os = "linux",
      target_env = "gnu",
      not(debug_assertions),
      not(feature = "portable-only")
    )))]
    Box::new(TokenCipher {
      cipher: Aes256Gcm::new(&key),
      sealed: AtomicBool::new(false),
    })
  }
}

impl core::fmt::Debug for AddressTokenKey {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("AddressTokenKey").finish_non_exhaustive()
  }
}

// Return only the final heap owner; the caller clears this retired worker.
#[cfg(all(
  target_arch = "aarch64",
  target_endian = "little",
  target_os = "linux",
  target_env = "gnu",
  not(debug_assertions),
  not(feature = "portable-only")
))]
#[inline(never)]
fn boxed_token_cipher(key: &Aes256GcmKey) -> Box<TokenCipher> {
  Box::new(TokenCipher {
    cipher: Aes256Gcm::new(key),
    sealed: AtomicBool::new(false),
  })
}

/// One derived token key, which seals at most once.
pub(crate) struct TokenCipher {
  cipher: Aes256Gcm,
  sealed: AtomicBool,
}

impl TokenCipher {
  /// Encrypt `data` in place and append the tag; fails for oversized input or a second seal.
  pub(crate) fn seal_in_place(&self, data: &mut Vec<u8>, aad: &[u8]) -> Result<(), Error> {
    if data.len() > MAX_TOKEN_PLAINTEXT || self.sealed.swap(true, Ordering::Relaxed) {
      return Err(Error::EncryptError);
    }
    // Each derived key seals once, so the fixed AEAD nonce is never reused
    // under one key while token nonces stay distinct.
    let tag = self
      .cipher
      .encrypt_in_place(&Nonce96::from_bytes([0; 12]), aad, data)
      .map_err(|_| Error::EncryptError)?;
    data.extend_from_slice(tag.as_bytes());
    Ok(())
  }

  /// Decrypt `data`, which ends in the tag, and return the plaintext prefix.
  pub(crate) fn open_in_place<'a>(&self, data: &'a mut [u8], aad: &[u8]) -> Result<&'a mut [u8], Error> {
    if !(TOKEN_TAG_LEN..=MAX_TOKEN_PLAINTEXT.strict_add(TOKEN_TAG_LEN)).contains(&data.len()) {
      return Err(Error::DecryptError);
    }
    let (body, tag) = data.split_at_mut(data.len().strict_sub(TOKEN_TAG_LEN));
    let tag = Aes256GcmTag::from_bytes((&*tag).try_into().map_err(|_| Error::DecryptError)?);
    self
      .cipher
      .decrypt_in_place(&Nonce96::from_bytes([0; 12]), aad, body, &tag)
      .map_err(|_| Error::DecryptError)?;
    Ok(body)
  }
}
