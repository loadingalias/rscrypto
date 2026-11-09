use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/auth_siphash.rs"]
mod auth_siphash;

#[test]
fn replay_auth_siphash_corpus() {
  let replayed = replay_corpus_dir("auth_siphash", corpus_dir("auth_siphash"), auth_siphash::run);
  assert_ne!(replayed, 0, "auth_siphash corpus should not be empty");
}
