use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/aead_aes128gcm.rs"]
mod aead_aes128gcm;

#[test]
fn replay_aead_aes128gcm_corpus() {
    let replayed = replay_corpus_dir("aead_aes128gcm", corpus_dir("aead_aes128gcm"), aead_aes128gcm::run);
    assert_ne!(replayed, 0, "aead_aes128gcm corpus should not be empty");
}
