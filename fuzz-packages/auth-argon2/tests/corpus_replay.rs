use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/auth_argon2d.rs"]
mod auth_argon2d;

#[path = "../../../fuzz/target_impls/auth_argon2i.rs"]
mod auth_argon2i;

#[path = "../../../fuzz/target_impls/auth_argon2id.rs"]
mod auth_argon2id;

#[test]
fn replay_auth_argon2d_corpus() {
    let replayed = replay_corpus_dir("auth_argon2d", corpus_dir("auth_argon2d"), auth_argon2d::run);
    assert_ne!(replayed, 0, "auth_argon2d corpus should not be empty");
}

#[test]
fn replay_auth_argon2i_corpus() {
    let replayed = replay_corpus_dir("auth_argon2i", corpus_dir("auth_argon2i"), auth_argon2i::run);
    assert_ne!(replayed, 0, "auth_argon2i corpus should not be empty");
}

#[test]
fn replay_auth_argon2id_corpus() {
    let replayed = replay_corpus_dir("auth_argon2id", corpus_dir("auth_argon2id"), auth_argon2id::run);
    assert_ne!(replayed, 0, "auth_argon2id corpus should not be empty");
}

