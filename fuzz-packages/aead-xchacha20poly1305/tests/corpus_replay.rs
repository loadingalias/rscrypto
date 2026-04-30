use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/aead_xchacha20poly1305.rs"]
mod aead_xchacha20poly1305;

#[test]
fn replay_aead_xchacha20poly1305_corpus() {
    let replayed = replay_corpus_dir("aead_xchacha20poly1305", corpus_dir("aead_xchacha20poly1305"), aead_xchacha20poly1305::run);
    assert_ne!(replayed, 0, "aead_xchacha20poly1305 corpus should not be empty");
}

