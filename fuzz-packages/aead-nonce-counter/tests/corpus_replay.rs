use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/aead_nonce_counter.rs"]
mod aead_nonce_counter;

#[test]
fn replay_aead_nonce_counter_corpus() {
    let replayed = replay_corpus_dir("aead_nonce_counter", corpus_dir("aead_nonce_counter"), aead_nonce_counter::run);
    assert_ne!(replayed, 0, "aead_nonce_counter corpus should not be empty");
}

