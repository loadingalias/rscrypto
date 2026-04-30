use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/traits_io.rs"]
mod traits_io;

#[test]
fn replay_traits_io_corpus() {
    let replayed = replay_corpus_dir("traits_io", corpus_dir("traits_io"), traits_io::run);
    assert_ne!(replayed, 0, "traits_io corpus should not be empty");
}

