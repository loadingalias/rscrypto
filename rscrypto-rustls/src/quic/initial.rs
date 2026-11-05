//! Initial keys and Retry integrity for QUIC v1 (RFC 9001, sections 5.2 and 5.8).

use alloc::vec::Vec;

use rscrypto::aead::{Nonce96, expert::AeadWithNonce};
use rscrypto::{Aes128Gcm, Aes128GcmKey, Aes128GcmTag};
use rustls::quic::{Keys, Side, Suite, Version};

/// Maximum length of a QUIC v1 connection ID (RFC 9000, section 17.2).
pub const MAX_CONNECTION_ID_LEN: usize = 20;

/// Length of a Retry integrity tag (RFC 9001, section 5.8).
pub const RETRY_INTEGRITY_TAG_LEN: usize = 16;

// RFC 9001 section 5.8: public fixed Retry key and nonce for QUIC v1.
const RETRY_KEY: [u8; 16] = [
  0xbe, 0x0c, 0x69, 0x0b, 0x9f, 0x66, 0x57, 0x5a, 0x1d, 0x76, 0x6b, 0x54, 0xe3, 0x68, 0xc8, 0x4e,
];
const RETRY_NONCE: [u8; 12] = [0x46, 0x15, 0x99, 0xd3, 0x5d, 0x63, 0x2b, 0xf2, 0x23, 0x98, 0x25, 0xbb];

/// Derive the QUIC v1 Initial packet and header keys for one endpoint.
///
/// `client_dst_connection_id` is the Destination Connection ID field of the
/// client's first Initial packet, or of its Initial packets after a Retry.
/// `side` names the local endpoint: `local` keys protect its packets and
/// `remote` keys remove the peer's protection. Initial packets always use
/// AES-128-GCM and AES-128 header protection, independent of the negotiated
/// suite (RFC 9001, section 5.2). Rustls derives later packet-space keys.
pub fn initial_keys(client_dst_connection_id: &[u8], side: Side) -> Keys {
  let suite = Suite::try_from(crate::cipher_suite::TLS13_AES_128_GCM_SHA256).expect("static QUIC suite");
  Keys::initial(Version::V1, suite, client_dst_connection_id, side)
}

/// Compute the integrity tag of a QUIC v1 Retry packet (RFC 9001, section 5.8).
///
/// `retry` holds the Retry packet from its first byte through the Retry Token
/// field. `original_dst_connection_id` is the Destination Connection ID of
/// the client Initial packet that this Retry answers. Append the returned tag
/// to complete the packet.
///
/// Returns `None` if `original_dst_connection_id` exceeds
/// [`MAX_CONNECTION_ID_LEN`] bytes.
#[must_use]
pub fn retry_integrity_tag(original_dst_connection_id: &[u8], retry: &[u8]) -> Option<[u8; RETRY_INTEGRITY_TAG_LEN]> {
  let aad = retry_aad(original_dst_connection_id, retry)?;
  let tag = retry_cipher()
    .encrypt_in_place(&Nonce96::from_bytes(RETRY_NONCE), &aad, &mut [])
    .expect("Retry has an empty plaintext");
  Some(*tag.as_bytes())
}

/// Check the integrity tag of a received QUIC v1 Retry packet (RFC 9001, section 5.8).
///
/// `packet` holds the complete Retry packet, ending in its 16-byte integrity
/// tag. `original_dst_connection_id` is the Destination Connection ID of the
/// client's first Initial packet. A client must discard a Retry that fails this
/// check.
///
/// Returns `false` for a wrong tag, a packet shorter than the tag, or an
/// original connection ID longer than [`MAX_CONNECTION_ID_LEN`] bytes.
#[must_use]
pub fn is_valid_retry(original_dst_connection_id: &[u8], packet: &[u8]) -> bool {
  let Some(length) = packet.len().checked_sub(RETRY_INTEGRITY_TAG_LEN) else {
    return false;
  };
  let (body, tag) = packet.split_at(length);
  let Ok(tag) = <[u8; RETRY_INTEGRITY_TAG_LEN]>::try_from(tag) else {
    return false;
  };
  let Some(aad) = retry_aad(original_dst_connection_id, body) else {
    return false;
  };
  retry_cipher()
    .decrypt_in_place(
      &Nonce96::from_bytes(RETRY_NONCE),
      &aad,
      &mut [],
      &Aes128GcmTag::from_bytes(tag),
    )
    .is_ok()
}

/// The Retry pseudo-packet: the original connection ID with its length, then the packet without its tag.
fn retry_aad(cid: &[u8], packet: &[u8]) -> Option<Vec<u8>> {
  if cid.len() > MAX_CONNECTION_ID_LEN {
    return None;
  }
  let length = u8::try_from(cid.len()).expect("the length is at most 20");
  let mut aad = Vec::with_capacity(1usize.strict_add(cid.len()).strict_add(packet.len()));
  aad.push(length);
  aad.extend_from_slice(cid);
  aad.extend_from_slice(packet);
  Some(aad)
}

fn retry_cipher() -> Aes128Gcm {
  Aes128Gcm::new(&Aes128GcmKey::from_bytes(RETRY_KEY))
}
