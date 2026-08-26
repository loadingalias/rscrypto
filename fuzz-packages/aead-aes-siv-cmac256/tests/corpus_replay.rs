use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/aead_aes_siv_cmac256.rs"]
mod aead_aes_siv_cmac256;

#[test]
fn replay_aead_aes_siv_cmac256_corpus() {
  let replayed = replay_corpus_dir(
    "aead_aes_siv_cmac256",
    corpus_dir("aead_aes_siv_cmac256"),
    aead_aes_siv_cmac256::run,
  );
  assert_ne!(replayed, 0, "aead_aes_siv_cmac256 corpus should not be empty");
}
