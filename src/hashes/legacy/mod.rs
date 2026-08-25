//! Capability-confined operations required by legacy protocol standards.
//!
//! These APIs expose complete protocol operations, not general-purpose access
//! to superseded primitives. They are excluded from every umbrella feature so
//! applications must opt in to the exact compatibility capability they need.
//!
//! The capability type is deliberately not available at the crate root:
//!
//! ```compile_fail
//! use rscrypto::WebSocketAcceptDigest;
//! ```

mod sha1;

const WEBSOCKET_GUID: &[u8; 36] = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

/// The 20-byte digest used to construct an RFC 6455 WebSocket accept value.
///
/// [`Self::compute`] hashes the `Sec-WebSocket-Key` field value exactly as
/// received, followed by RFC 6455's fixed GUID. The caller remains responsible
/// for HTTP field parsing and Base64 encoding the returned public bytes.
///
/// This type is a protocol compatibility value. SHA-1 collision resistance is
/// broken, and this operation provides no authentication or integrity claim.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WebSocketAcceptDigest([u8; 20]);

impl WebSocketAcceptDigest {
  /// Computes the RFC 6455 WebSocket accept digest without allocating.
  ///
  /// `sec_websocket_key` is hashed byte-for-byte. This method does not trim,
  /// decode, validate, or otherwise interpret the HTTP field value.
  #[must_use]
  #[inline]
  pub fn compute(sec_websocket_key: &[u8]) -> Self {
    Self(sha1::digest_websocket_key(sec_websocket_key))
  }
}

impl AsRef<[u8]> for WebSocketAcceptDigest {
  #[inline]
  fn as_ref(&self) -> &[u8] {
    &self.0
  }
}

#[cfg(test)]
mod tests {
  use super::WebSocketAcceptDigest;

  #[test]
  fn digest_type_has_public_value_semantics() {
    fn assert_traits<T: Copy + core::fmt::Debug + Eq + AsRef<[u8]>>() {}

    assert_traits::<WebSocketAcceptDigest>();
  }
}
