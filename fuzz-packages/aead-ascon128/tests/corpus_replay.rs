use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/aead_ascon128.rs"]
mod aead_ascon128;

#[test]
fn replay_aead_ascon128_corpus() {
    let replayed = replay_corpus_dir("aead_ascon128", corpus_dir("aead_ascon128"), aead_ascon128::run);
    assert_ne!(replayed, 0, "aead_ascon128 corpus should not be empty");
}

