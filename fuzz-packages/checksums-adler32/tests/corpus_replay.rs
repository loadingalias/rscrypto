use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/checksum_adler32.rs"]
mod checksum_adler32;

#[test]
fn replay_checksum_adler32_corpus() {
  let replayed = replay_corpus_dir(
    "checksum_adler32",
    corpus_dir("checksum_adler32"),
    checksum_adler32::run,
  );
  assert_ne!(replayed, 0, "checksum_adler32 corpus should not be empty");
}
