use rscrypto::hashes::legacy::WebSocketAcceptDigest;
use sha1::{Digest as _, Sha1};

const WEBSOCKET_GUID: &[u8] = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

pub(super) fn run(data: &[u8]) {
  let ours = WebSocketAcceptDigest::compute(data);

  let mut oracle = Sha1::new();
  oracle.update(data);
  oracle.update(WEBSOCKET_GUID);
  let expected = oracle.finalize();

  assert_eq!(ours.as_ref(), expected.as_slice(), "WebSocket accept digest mismatch");
}
