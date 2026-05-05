use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../../../fuzz/target_impls/aead_aes128gcmsiv.rs"]
mod aead_aes128gcmsiv;

#[test]
fn replay_aead_aes128gcmsiv_corpus() {
    let replayed = replay_corpus_dir("aead_aes128gcmsiv", corpus_dir("aead_aes128gcmsiv"), aead_aes128gcmsiv::run);
    assert_ne!(replayed, 0, "aead_aes128gcmsiv corpus should not be empty");
}
