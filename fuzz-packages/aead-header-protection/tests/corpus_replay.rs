use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/aead_header_protection.rs"]
mod aead_header_protection;

#[test]
fn replay_aead_header_protection_corpus() {
  let replayed = replay_corpus_dir(
    "aead_header_protection",
    corpus_dir("aead_header_protection"),
    aead_header_protection::run,
  );
  assert_ne!(replayed, 0, "aead_header_protection corpus should not be empty");
}
