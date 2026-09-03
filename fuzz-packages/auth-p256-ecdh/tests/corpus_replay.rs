use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/auth_p256_ecdh.rs"]
mod auth_p256_ecdh;

#[test]
fn replay_auth_p256_ecdh_corpus() {
  let replayed = replay_corpus_dir("auth_p256_ecdh", corpus_dir("auth_p256_ecdh"), auth_p256_ecdh::run);
  assert_ne!(replayed, 0, "auth_p256_ecdh corpus should not be empty");
}
