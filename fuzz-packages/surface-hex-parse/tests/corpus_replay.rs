use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/hex_parse.rs"]
mod hex_parse;

#[test]
fn replay_hex_parse_corpus() {
    let replayed = replay_corpus_dir("hex_parse", corpus_dir("hex_parse"), hex_parse::run);
    assert_ne!(replayed, 0, "hex_parse corpus should not be empty");
}

