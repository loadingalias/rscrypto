use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/aead_aes256gcm.rs"]
mod aead_aes256gcm;

#[test]
fn replay_aead_aes256gcm_corpus() {
    let replayed = replay_corpus_dir("aead_aes256gcm", corpus_dir("aead_aes256gcm"), aead_aes256gcm::run);
    assert_ne!(replayed, 0, "aead_aes256gcm corpus should not be empty");
}

