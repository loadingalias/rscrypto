use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/auth_phc.rs"]
mod auth_phc;

#[test]
fn replay_auth_phc_corpus() {
    let replayed = replay_corpus_dir("auth_phc", corpus_dir("auth_phc"), auth_phc::run);
    assert_ne!(replayed, 0, "auth_phc corpus should not be empty");
}

