use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/hash_ascon.rs"]
mod hash_ascon;

#[path = "../../../fuzz/target_impls/hash_ascon_cxof.rs"]
mod hash_ascon_cxof;

#[test]
fn replay_hash_ascon_corpus() {
    let replayed = replay_corpus_dir("hash_ascon", corpus_dir("hash_ascon"), hash_ascon::run);
    assert_ne!(replayed, 0, "hash_ascon corpus should not be empty");
}

#[test]
fn replay_hash_ascon_cxof_corpus() {
    let replayed = replay_corpus_dir("hash_ascon_cxof", corpus_dir("hash_ascon_cxof"), hash_ascon_cxof::run);
    assert_ne!(replayed, 0, "hash_ascon_cxof corpus should not be empty");
}

