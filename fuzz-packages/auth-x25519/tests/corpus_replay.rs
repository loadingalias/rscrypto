use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/auth_x25519.rs"]
mod auth_x25519;

#[test]
fn replay_auth_x25519_corpus() {
    let replayed = replay_corpus_dir("auth_x25519", corpus_dir("auth_x25519"), auth_x25519::run);
    assert_ne!(replayed, 0, "auth_x25519 corpus should not be empty");
}

