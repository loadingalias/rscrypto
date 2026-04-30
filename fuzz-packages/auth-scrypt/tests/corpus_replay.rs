use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/auth_scrypt.rs"]
mod auth_scrypt;

#[test]
fn replay_auth_scrypt_corpus() {
    let replayed = replay_corpus_dir("auth_scrypt", corpus_dir("auth_scrypt"), auth_scrypt::run);
    assert_ne!(replayed, 0, "auth_scrypt corpus should not be empty");
}

