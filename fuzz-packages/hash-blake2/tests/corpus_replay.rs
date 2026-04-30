use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/hash_blake2b.rs"]
mod hash_blake2b;

#[path = "../../../fuzz/target_impls/hash_blake2s.rs"]
mod hash_blake2s;

#[test]
fn replay_hash_blake2b_corpus() {
    let replayed = replay_corpus_dir("hash_blake2b", corpus_dir("hash_blake2b"), hash_blake2b::run);
    assert_ne!(replayed, 0, "hash_blake2b corpus should not be empty");
}

#[test]
fn replay_hash_blake2s_corpus() {
    let replayed = replay_corpus_dir("hash_blake2s", corpus_dir("hash_blake2s"), hash_blake2s::run);
    assert_ne!(replayed, 0, "hash_blake2s corpus should not be empty");
}

