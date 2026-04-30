use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/hash_cshake256.rs"]
mod hash_cshake256;

#[path = "../../../fuzz/target_impls/hash_sha3.rs"]
mod hash_sha3;

#[test]
fn replay_hash_cshake256_corpus() {
    let replayed = replay_corpus_dir("hash_cshake256", corpus_dir("hash_cshake256"), hash_cshake256::run);
    assert_ne!(replayed, 0, "hash_cshake256 corpus should not be empty");
}

#[test]
fn replay_hash_sha3_corpus() {
    let replayed = replay_corpus_dir("hash_sha3", corpus_dir("hash_sha3"), hash_sha3::run);
    assert_ne!(replayed, 0, "hash_sha3 corpus should not be empty");
}

