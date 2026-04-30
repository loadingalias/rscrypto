use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/checksum_crc24.rs"]
mod checksum_crc24;

#[test]
fn replay_checksum_crc24_corpus() {
    let replayed = replay_corpus_dir("checksum_crc24", corpus_dir("checksum_crc24"), checksum_crc24::run);
    assert_ne!(replayed, 0, "checksum_crc24 corpus should not be empty");
}

