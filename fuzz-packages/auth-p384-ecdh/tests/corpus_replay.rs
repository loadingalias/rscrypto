use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/auth_p384_ecdh.rs"]
mod auth_p384_ecdh;

#[test]
fn replay_auth_p384_ecdh_corpus() {
  let replayed = replay_corpus_dir("auth_p384_ecdh", corpus_dir("auth_p384_ecdh"), auth_p384_ecdh::run);
  assert_ne!(replayed, 0, "auth_p384_ecdh corpus should not be empty");
}
