use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/auth_pbkdf2.rs"]
mod auth_pbkdf2;

#[test]
fn replay_auth_pbkdf2_corpus() {
    let replayed = replay_corpus_dir("auth_pbkdf2", corpus_dir("auth_pbkdf2"), auth_pbkdf2::run);
    assert_ne!(replayed, 0, "auth_pbkdf2 corpus should not be empty");
}

