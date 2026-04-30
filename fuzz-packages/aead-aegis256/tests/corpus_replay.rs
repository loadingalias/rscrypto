use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/aead_aegis256.rs"]
mod aead_aegis256;

#[test]
fn replay_aead_aegis256_corpus() {
    let replayed = replay_corpus_dir("aead_aegis256", corpus_dir("aead_aegis256"), aead_aegis256::run);
    assert_ne!(replayed, 0, "aead_aegis256 corpus should not be empty");
}

