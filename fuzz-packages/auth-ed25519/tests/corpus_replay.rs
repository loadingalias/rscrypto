use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/auth_ed25519.rs"]
mod auth_ed25519;

#[path = "../../../fuzz/target_impls/auth_ed25519_verify.rs"]
mod auth_ed25519_verify;

#[test]
fn replay_auth_ed25519_corpus() {
    let replayed = replay_corpus_dir("auth_ed25519", corpus_dir("auth_ed25519"), auth_ed25519::run);
    assert_ne!(replayed, 0, "auth_ed25519 corpus should not be empty");
}

#[test]
fn replay_auth_ed25519_verify_corpus() {
    let replayed = replay_corpus_dir("auth_ed25519_verify", corpus_dir("auth_ed25519_verify"), auth_ed25519_verify::run);
    assert_ne!(replayed, 0, "auth_ed25519_verify corpus should not be empty");
}

