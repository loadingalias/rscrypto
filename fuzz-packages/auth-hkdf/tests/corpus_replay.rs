use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/auth_hkdf_sha256.rs"]
mod auth_hkdf_sha256;

#[path = "../../../fuzz/target_impls/auth_hkdf_sha384.rs"]
mod auth_hkdf_sha384;

#[test]
fn replay_auth_hkdf_sha256_corpus() {
    let replayed = replay_corpus_dir("auth_hkdf_sha256", corpus_dir("auth_hkdf_sha256"), auth_hkdf_sha256::run);
    assert_ne!(replayed, 0, "auth_hkdf_sha256 corpus should not be empty");
}

#[test]
fn replay_auth_hkdf_sha384_corpus() {
    let replayed = replay_corpus_dir("auth_hkdf_sha384", corpus_dir("auth_hkdf_sha384"), auth_hkdf_sha384::run);
    assert_ne!(replayed, 0, "auth_hkdf_sha384 corpus should not be empty");
}

