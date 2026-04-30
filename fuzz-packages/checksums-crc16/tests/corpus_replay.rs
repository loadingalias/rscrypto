use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/checksum_crc16.rs"]
mod checksum_crc16;

#[test]
fn replay_checksum_crc16_corpus() {
    let replayed = replay_corpus_dir("checksum_crc16", corpus_dir("checksum_crc16"), checksum_crc16::run);
    assert_ne!(replayed, 0, "checksum_crc16 corpus should not be empty");
}

