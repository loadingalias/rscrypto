use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/fast_rapidhash.rs"]
mod fast_rapidhash;

#[test]
fn replay_fast_rapidhash_corpus() {
    let replayed = replay_corpus_dir("fast_rapidhash", corpus_dir("fast_rapidhash"), fast_rapidhash::run);
    assert_ne!(replayed, 0, "fast_rapidhash corpus should not be empty");
}

