//! Connection IDs that their issuing key recognizes.

use rscrypto::SecretBytes;
use rscrypto::aead::expert::header_protection::{Aes128HeaderProtection, Aes128HeaderProtectionKey};
use rustls::crypto::{GetRandomFailed, SecureRandom};

// Eight-byte layout: three random bytes, then a five-byte keyed tag.
const NONCE_LEN: usize = 3;
const TAG_LEN: usize = 5;
const ID_LEN: usize = NONCE_LEN + TAG_LEN;
// Fills the AES block after the nonce; the key is used for nothing else.
const TAG_DOMAIN: [u8; 16 - NONCE_LEN] = *b"rscrypto-cid1";

/// Issues eight-byte QUIC connection IDs and recognizes the IDs it issued.
///
/// Each ID is a three-byte random nonce followed by a five-byte tag: AES-128,
/// under this key, of the nonce and a fixed domain label, truncated to five
/// bytes. [`Self::validate`] lets an endpoint reject an ID that this key never
/// issued before any table lookup. Observers cannot compute tags, and IDs carry
/// no counter or endpoint index, so an observer cannot link two IDs with
/// different nonces.
///
/// IDs are not unique. With a 24-bit nonce, a repeated ID becomes likely after
/// a few thousand IDs, and a repeated nonce repeats the whole ID. Reject an
/// issued ID that is already in use and issue another. An ID carries no route;
/// endpoints that share a key accept each other's IDs.
///
/// The key clears on drop. Share one key across threads with `Arc`.
pub struct ConnectionIdKey {
  tag: Aes128HeaderProtection,
}

impl ConnectionIdKey {
  /// Length of every issued ID.
  pub const ID_LEN: usize = ID_LEN;

  /// Generate a key from 16 bytes of `random`.
  ///
  /// The source must provide cryptographically secure random bytes. A failure
  /// returns the error without fallback.
  pub fn generate(random: &dyn SecureRandom) -> Result<Self, GetRandomFailed> {
    let key = SecretBytes::<16>::try_fill_with(|out| random.fill(out))?;
    Ok(Self {
      tag: Aes128HeaderProtection::new(&Aes128HeaderProtectionKey::from_bytes(*key.as_bytes())),
    })
  }

  /// Issue an ID with a nonce of three bytes from `random`.
  ///
  /// A failure returns the error without fallback.
  pub fn issue(&self, random: &dyn SecureRandom) -> Result<[u8; ID_LEN], GetRandomFailed> {
    let mut nonce = [0u8; NONCE_LEN];
    random.fill(&mut nonce)?;
    Ok(self.id(nonce))
  }

  /// Whether this key issued `id`.
  ///
  /// Returns `false` for any length other than [`Self::ID_LEN`]. The tag
  /// comparison has no early exit, so its timing does not reveal how many tag
  /// bytes match.
  #[must_use]
  pub fn validate(&self, id: &[u8]) -> bool {
    let Ok(id) = <&[u8; ID_LEN]>::try_from(id) else {
      return false;
    };
    let (nonce, tag) = id.split_at(NONCE_LEN);
    let nonce = <[u8; NONCE_LEN]>::try_from(nonce).expect("fixed nonce length");
    let expected = self.tag(&nonce);
    let difference = expected.iter().zip(tag).fold(0, |acc, (a, b)| acc | (a ^ b));
    core::hint::black_box(difference) == 0
  }

  /// The ID for `nonce`.
  fn id(&self, nonce: [u8; NONCE_LEN]) -> [u8; ID_LEN] {
    let mut id = [0u8; ID_LEN];
    let (prefix, tag) = id.split_at_mut(NONCE_LEN);
    prefix.copy_from_slice(&nonce);
    tag.copy_from_slice(&self.tag(&nonce));
    id
  }

  /// AES-128 of the domain-separated nonce, truncated to the five-byte tag.
  fn tag(&self, nonce: &[u8; NONCE_LEN]) -> [u8; TAG_LEN] {
    let mut sample = [0u8; 16];
    sample[..NONCE_LEN].copy_from_slice(nonce);
    sample[NONCE_LEN..].copy_from_slice(&TAG_DOMAIN);
    self.tag.mask(&sample)
  }
}

impl core::fmt::Debug for ConnectionIdKey {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("ConnectionIdKey").finish_non_exhaustive()
  }
}
