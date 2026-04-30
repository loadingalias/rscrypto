use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/hash_sha2.rs"]
mod hash_sha2;

#[test]
fn replay_hash_sha2_corpus() {
    let replayed = replay_corpus_dir("hash_sha2", corpus_dir("hash_sha2"), hash_sha2::run);
    assert_ne!(replayed, 0, "hash_sha2 corpus should not be empty");
}

