use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/fast_xxh3.rs"]
mod fast_xxh3;

#[test]
fn replay_fast_xxh3_corpus() {
    let replayed = replay_corpus_dir("fast_xxh3", corpus_dir("fast_xxh3"), fast_xxh3::run);
    assert_ne!(replayed, 0, "fast_xxh3 corpus should not be empty");
}

