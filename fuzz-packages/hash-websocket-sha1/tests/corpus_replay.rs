use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/hash_websocket_sha1.rs"]
mod hash_websocket_sha1;

#[test]
fn replay_hash_websocket_sha1_corpus() {
  let replayed = replay_corpus_dir(
    "hash_websocket_sha1",
    corpus_dir("hash_websocket_sha1"),
    hash_websocket_sha1::run,
  );
  assert_ne!(replayed, 0, "hash_websocket_sha1 corpus should not be empty");
}
