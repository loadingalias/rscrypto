#![cfg(feature = "websocket-sha1")]

use rscrypto::hashes::legacy::WebSocketAcceptDigest;

#[path = "common/allocations.rs"]
mod allocations;
use allocations::measure_allocations;

#[test]
fn websocket_accept_digest_does_not_allocate() {
  let key = b"dGhlIHNhbXBsZSBub25jZQ==";
  let allocations = measure_allocations(|| {
    let digest = WebSocketAcceptDigest::compute(key);
    assert_eq!(digest.as_ref().len(), 20);
  });

  assert_eq!(allocations, 0, "WebSocket accept digest must not allocate");
}
