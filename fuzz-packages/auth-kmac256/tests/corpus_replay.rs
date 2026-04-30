use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/auth_kmac256.rs"]
mod auth_kmac256;

#[test]
fn replay_auth_kmac256_corpus() {
    let replayed = replay_corpus_dir("auth_kmac256", corpus_dir("auth_kmac256"), auth_kmac256::run);
    assert_ne!(replayed, 0, "auth_kmac256 corpus should not be empty");
}

