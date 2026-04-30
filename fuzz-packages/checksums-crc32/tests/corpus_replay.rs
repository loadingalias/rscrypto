use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/checksum_crc32.rs"]
mod checksum_crc32;

#[test]
fn replay_checksum_crc32_corpus() {
    let replayed = replay_corpus_dir("checksum_crc32", corpus_dir("checksum_crc32"), checksum_crc32::run);
    assert_ne!(replayed, 0, "checksum_crc32 corpus should not be empty");
}

