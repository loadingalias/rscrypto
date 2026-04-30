use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/hash_blake3.rs"]
mod hash_blake3;

#[path = "../../../fuzz/target_impls/hash_blake3_derive.rs"]
mod hash_blake3_derive;

#[path = "../../../fuzz/target_impls/hash_blake3_keyed.rs"]
mod hash_blake3_keyed;

#[test]
fn replay_hash_blake3_corpus() {
    let replayed = replay_corpus_dir("hash_blake3", corpus_dir("hash_blake3"), hash_blake3::run);
    assert_ne!(replayed, 0, "hash_blake3 corpus should not be empty");
}

#[test]
fn replay_hash_blake3_derive_corpus() {
    let replayed = replay_corpus_dir("hash_blake3_derive", corpus_dir("hash_blake3_derive"), hash_blake3_derive::run);
    assert_ne!(replayed, 0, "hash_blake3_derive corpus should not be empty");
}

#[test]
fn replay_hash_blake3_keyed_corpus() {
    let replayed = replay_corpus_dir("hash_blake3_keyed", corpus_dir("hash_blake3_keyed"), hash_blake3_keyed::run);
    assert_ne!(replayed, 0, "hash_blake3_keyed corpus should not be empty");
}

