use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/fast_xxh64.rs"]
mod fast_xxh64;

#[test]
fn replay_fast_xxh64_corpus() {
  let replayed = replay_corpus_dir("fast_xxh64", corpus_dir("fast_xxh64"), fast_xxh64::run);
  assert_ne!(replayed, 0, "fast_xxh64 corpus should not be empty");
}
