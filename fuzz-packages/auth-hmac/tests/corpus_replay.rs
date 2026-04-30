use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/auth_hmac_sha256.rs"]
mod auth_hmac_sha256;

#[path = "../../../fuzz/target_impls/auth_hmac_sha384.rs"]
mod auth_hmac_sha384;

#[path = "../../../fuzz/target_impls/auth_hmac_sha512.rs"]
mod auth_hmac_sha512;

#[test]
fn replay_auth_hmac_sha256_corpus() {
    let replayed = replay_corpus_dir("auth_hmac_sha256", corpus_dir("auth_hmac_sha256"), auth_hmac_sha256::run);
    assert_ne!(replayed, 0, "auth_hmac_sha256 corpus should not be empty");
}

#[test]
fn replay_auth_hmac_sha384_corpus() {
    let replayed = replay_corpus_dir("auth_hmac_sha384", corpus_dir("auth_hmac_sha384"), auth_hmac_sha384::run);
    assert_ne!(replayed, 0, "auth_hmac_sha384 corpus should not be empty");
}

#[test]
fn replay_auth_hmac_sha512_corpus() {
    let replayed = replay_corpus_dir("auth_hmac_sha512", corpus_dir("auth_hmac_sha512"), auth_hmac_sha512::run);
    assert_ne!(replayed, 0, "auth_hmac_sha512 corpus should not be empty");
}

