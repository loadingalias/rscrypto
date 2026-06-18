use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

#[path = "../target_impls/aead_aegis256.rs"]
mod aead_aegis256;

#[path = "../target_impls/aead_aes256gcm.rs"]
mod aead_aes256gcm;

#[path = "../target_impls/aead_aes256gcmsiv.rs"]
mod aead_aes256gcmsiv;

#[path = "../target_impls/aead_ascon128.rs"]
mod aead_ascon128;

#[path = "../target_impls/aead_chacha20poly1305.rs"]
mod aead_chacha20poly1305;

#[path = "../target_impls/aead_nonce_counter.rs"]
mod aead_nonce_counter;

#[path = "../target_impls/aead_xchacha20poly1305.rs"]
mod aead_xchacha20poly1305;

#[path = "../target_impls/auth_argon2d.rs"]
mod auth_argon2d;

#[path = "../target_impls/auth_argon2i.rs"]
mod auth_argon2i;

#[path = "../target_impls/auth_argon2id.rs"]
mod auth_argon2id;

#[path = "../target_impls/auth_ed25519.rs"]
mod auth_ed25519;

#[path = "../target_impls/auth_ed25519_verify.rs"]
mod auth_ed25519_verify;

#[path = "../target_impls/auth_ecdsa_verify.rs"]
mod auth_ecdsa_verify;

#[path = "../target_impls/auth_ecdsa_sign.rs"]
mod auth_ecdsa_sign;

#[path = "../target_impls/auth_hkdf_sha256.rs"]
mod auth_hkdf_sha256;

#[path = "../target_impls/auth_hkdf_sha384.rs"]
mod auth_hkdf_sha384;

#[path = "../target_impls/auth_hmac_sha256.rs"]
mod auth_hmac_sha256;

#[path = "../target_impls/auth_hmac_sha384.rs"]
mod auth_hmac_sha384;

#[path = "../target_impls/auth_hmac_sha512.rs"]
mod auth_hmac_sha512;

#[path = "../target_impls/auth_kmac256.rs"]
mod auth_kmac256;

#[path = "../target_impls/auth_mlkem512.rs"]
mod auth_mlkem512;

#[path = "../target_impls/auth_mlkem768.rs"]
mod auth_mlkem768;

#[path = "../target_impls/auth_mlkem1024.rs"]
mod auth_mlkem1024;

#[path = "../target_impls/auth_pbkdf2.rs"]
mod auth_pbkdf2;

#[path = "../target_impls/auth_phc.rs"]
mod auth_phc;

#[path = "../target_impls/auth_rsa_public_key.rs"]
mod auth_rsa_public_key;

#[path = "../target_impls/auth_rsa_import.rs"]
mod auth_rsa_import;

#[path = "../target_impls/auth_rsa_protocol.rs"]
mod auth_rsa_protocol;

#[path = "../target_impls/auth_rsa_private_ops.rs"]
mod auth_rsa_private_ops;

#[path = "../target_impls/auth_rsa_verify.rs"]
mod auth_rsa_verify;

#[path = "../target_impls/auth_scrypt.rs"]
mod auth_scrypt;

#[path = "../target_impls/auth_x25519.rs"]
mod auth_x25519;

#[path = "../target_impls/checksum_crc.rs"]
mod checksum_crc;

#[path = "../target_impls/fast_rapidhash.rs"]
mod fast_rapidhash;

#[path = "../target_impls/fast_xxh3.rs"]
mod fast_xxh3;

#[path = "../target_impls/hash_ascon.rs"]
mod hash_ascon;

#[path = "../target_impls/hash_ascon_cxof.rs"]
mod hash_ascon_cxof;

#[path = "../target_impls/hash_blake2b.rs"]
mod hash_blake2b;

#[path = "../target_impls/hash_blake2s.rs"]
mod hash_blake2s;

#[path = "../target_impls/hash_blake3.rs"]
mod hash_blake3;

#[path = "../target_impls/hash_blake3_derive.rs"]
mod hash_blake3_derive;

#[path = "../target_impls/hash_blake3_keyed.rs"]
mod hash_blake3_keyed;

#[path = "../target_impls/hash_cshake256.rs"]
mod hash_cshake256;

#[path = "../target_impls/hash_sha2.rs"]
mod hash_sha2;

#[path = "../target_impls/hash_sha3.rs"]
mod hash_sha3;

#[path = "../target_impls/hex_parse.rs"]
mod hex_parse;

#[path = "../target_impls/traits_io.rs"]
mod traits_io;

#[test]
fn replay_aead_aegis256_corpus() {
    let replayed = replay_corpus_dir("aead_aegis256", corpus_dir("aead_aegis256"), aead_aegis256::run);
    assert_ne!(replayed, 0, "aead_aegis256 corpus should not be empty");
}

#[test]
fn replay_aead_aes256gcm_corpus() {
    let replayed = replay_corpus_dir("aead_aes256gcm", corpus_dir("aead_aes256gcm"), aead_aes256gcm::run);
    assert_ne!(replayed, 0, "aead_aes256gcm corpus should not be empty");
}

#[test]
fn replay_aead_aes256gcmsiv_corpus() {
    let replayed = replay_corpus_dir("aead_aes256gcmsiv", corpus_dir("aead_aes256gcmsiv"), aead_aes256gcmsiv::run);
    assert_ne!(replayed, 0, "aead_aes256gcmsiv corpus should not be empty");
}

#[test]
fn replay_aead_ascon128_corpus() {
    let replayed = replay_corpus_dir("aead_ascon128", corpus_dir("aead_ascon128"), aead_ascon128::run);
    assert_ne!(replayed, 0, "aead_ascon128 corpus should not be empty");
}

#[test]
fn replay_aead_chacha20poly1305_corpus() {
    let replayed = replay_corpus_dir("aead_chacha20poly1305", corpus_dir("aead_chacha20poly1305"), aead_chacha20poly1305::run);
    assert_ne!(replayed, 0, "aead_chacha20poly1305 corpus should not be empty");
}

#[test]
fn replay_aead_nonce_counter_corpus() {
    let replayed = replay_corpus_dir("aead_nonce_counter", corpus_dir("aead_nonce_counter"), aead_nonce_counter::run);
    assert_ne!(replayed, 0, "aead_nonce_counter corpus should not be empty");
}

#[test]
fn replay_aead_xchacha20poly1305_corpus() {
    let replayed = replay_corpus_dir("aead_xchacha20poly1305", corpus_dir("aead_xchacha20poly1305"), aead_xchacha20poly1305::run);
    assert_ne!(replayed, 0, "aead_xchacha20poly1305 corpus should not be empty");
}

#[test]
fn replay_auth_argon2d_corpus() {
    let replayed = replay_corpus_dir("auth_argon2d", corpus_dir("auth_argon2d"), auth_argon2d::run);
    assert_ne!(replayed, 0, "auth_argon2d corpus should not be empty");
}

#[test]
fn replay_auth_argon2i_corpus() {
    let replayed = replay_corpus_dir("auth_argon2i", corpus_dir("auth_argon2i"), auth_argon2i::run);
    assert_ne!(replayed, 0, "auth_argon2i corpus should not be empty");
}

#[test]
fn replay_auth_argon2id_corpus() {
    let replayed = replay_corpus_dir("auth_argon2id", corpus_dir("auth_argon2id"), auth_argon2id::run);
    assert_ne!(replayed, 0, "auth_argon2id corpus should not be empty");
}

#[test]
fn replay_auth_ed25519_corpus() {
    let replayed = replay_corpus_dir("auth_ed25519", corpus_dir("auth_ed25519"), auth_ed25519::run);
    assert_ne!(replayed, 0, "auth_ed25519 corpus should not be empty");
}

#[test]
fn replay_auth_ed25519_verify_corpus() {
    let replayed = replay_corpus_dir("auth_ed25519_verify", corpus_dir("auth_ed25519_verify"), auth_ed25519_verify::run);
    assert_ne!(replayed, 0, "auth_ed25519_verify corpus should not be empty");
}

#[test]
fn replay_auth_ecdsa_verify_corpus() {
    let replayed = replay_corpus_dir("auth_ecdsa_verify", corpus_dir("auth_ecdsa_verify"), auth_ecdsa_verify::run);
    assert_ne!(replayed, 0, "auth_ecdsa_verify corpus should not be empty");
}

#[test]
fn replay_auth_ecdsa_sign_corpus() {
    let replayed = replay_corpus_dir("auth_ecdsa_sign", corpus_dir("auth_ecdsa_sign"), auth_ecdsa_sign::run);
    assert_ne!(replayed, 0, "auth_ecdsa_sign corpus should not be empty");
}

#[test]
fn replay_auth_hkdf_sha256_corpus() {
    let replayed = replay_corpus_dir("auth_hkdf_sha256", corpus_dir("auth_hkdf_sha256"), auth_hkdf_sha256::run);
    assert_ne!(replayed, 0, "auth_hkdf_sha256 corpus should not be empty");
}

#[test]
fn replay_auth_hkdf_sha384_corpus() {
    let replayed = replay_corpus_dir("auth_hkdf_sha384", corpus_dir("auth_hkdf_sha384"), auth_hkdf_sha384::run);
    assert_ne!(replayed, 0, "auth_hkdf_sha384 corpus should not be empty");
}

#[test]
fn replay_auth_hmac_sha256_corpus() {
    let replayed = replay_corpus_dir("auth_hmac_sha256", corpus_dir("auth_hmac_sha256"), auth_hmac_sha256::run);
    assert_ne!(replayed, 0, "auth_hmac_sha256 corpus should not be empty");
}

#[test]
fn replay_auth_hmac_sha384_corpus() {
    let replayed = replay_corpus_dir("auth_hmac_sha384", corpus_dir("auth_hmac_sha384"), auth_hmac_sha384::run);
    assert_ne!(replayed, 0, "auth_hmac_sha384 corpus should not be empty");
}

#[test]
fn replay_auth_hmac_sha512_corpus() {
    let replayed = replay_corpus_dir("auth_hmac_sha512", corpus_dir("auth_hmac_sha512"), auth_hmac_sha512::run);
    assert_ne!(replayed, 0, "auth_hmac_sha512 corpus should not be empty");
}

#[test]
fn replay_auth_kmac256_corpus() {
    let replayed = replay_corpus_dir("auth_kmac256", corpus_dir("auth_kmac256"), auth_kmac256::run);
    assert_ne!(replayed, 0, "auth_kmac256 corpus should not be empty");
}

#[test]
fn replay_auth_mlkem512_corpus() {
    let replayed = replay_corpus_dir("auth_mlkem512", corpus_dir("auth_mlkem512"), auth_mlkem512::run);
    assert_ne!(replayed, 0, "auth_mlkem512 corpus should not be empty");
}

#[test]
fn replay_auth_mlkem768_corpus() {
    let replayed = replay_corpus_dir("auth_mlkem768", corpus_dir("auth_mlkem768"), auth_mlkem768::run);
    assert_ne!(replayed, 0, "auth_mlkem768 corpus should not be empty");
}

#[test]
fn replay_auth_mlkem1024_corpus() {
    let replayed = replay_corpus_dir("auth_mlkem1024", corpus_dir("auth_mlkem1024"), auth_mlkem1024::run);
    assert_ne!(replayed, 0, "auth_mlkem1024 corpus should not be empty");
}

#[test]
fn replay_auth_pbkdf2_corpus() {
    let replayed = replay_corpus_dir("auth_pbkdf2", corpus_dir("auth_pbkdf2"), auth_pbkdf2::run);
    assert_ne!(replayed, 0, "auth_pbkdf2 corpus should not be empty");
}

#[test]
fn replay_auth_phc_corpus() {
    let replayed = replay_corpus_dir("auth_phc", corpus_dir("auth_phc"), auth_phc::run);
    assert_ne!(replayed, 0, "auth_phc corpus should not be empty");
}

#[test]
fn replay_auth_rsa_public_key_corpus() {
    let replayed = replay_corpus_dir("auth_rsa_public_key", corpus_dir("auth_rsa_public_key"), auth_rsa_public_key::run);
    assert_ne!(replayed, 0, "auth_rsa_public_key corpus should not be empty");
}

#[test]
fn replay_auth_rsa_import_corpus() {
    let replayed = replay_corpus_dir("auth_rsa_import", corpus_dir("auth_rsa_import"), auth_rsa_import::run);
    assert_ne!(replayed, 0, "auth_rsa_import corpus should not be empty");
}

#[test]
fn replay_auth_rsa_protocol_corpus() {
    let replayed = replay_corpus_dir("auth_rsa_protocol", corpus_dir("auth_rsa_protocol"), auth_rsa_protocol::run);
    assert_ne!(replayed, 0, "auth_rsa_protocol corpus should not be empty");
}

#[test]
fn replay_auth_rsa_private_ops_corpus() {
    let replayed = replay_corpus_dir("auth_rsa_private_ops", corpus_dir("auth_rsa_private_ops"), auth_rsa_private_ops::run);
    assert_ne!(replayed, 0, "auth_rsa_private_ops corpus should not be empty");
}

#[test]
fn replay_auth_rsa_verify_corpus() {
    let replayed = replay_corpus_dir("auth_rsa_verify", corpus_dir("auth_rsa_verify"), auth_rsa_verify::run);
    assert_ne!(replayed, 0, "auth_rsa_verify corpus should not be empty");
}

#[test]
fn replay_auth_scrypt_corpus() {
    let replayed = replay_corpus_dir("auth_scrypt", corpus_dir("auth_scrypt"), auth_scrypt::run);
    assert_ne!(replayed, 0, "auth_scrypt corpus should not be empty");
}

#[test]
fn replay_auth_x25519_corpus() {
    let replayed = replay_corpus_dir("auth_x25519", corpus_dir("auth_x25519"), auth_x25519::run);
    assert_ne!(replayed, 0, "auth_x25519 corpus should not be empty");
}

#[test]
fn replay_checksum_crc_corpus() {
    let replayed = replay_corpus_dir("checksum_crc", corpus_dir("checksum_crc"), checksum_crc::run);
    assert_ne!(replayed, 0, "checksum_crc corpus should not be empty");
}

#[test]
fn replay_fast_rapidhash_corpus() {
    let replayed = replay_corpus_dir("fast_rapidhash", corpus_dir("fast_rapidhash"), fast_rapidhash::run);
    assert_ne!(replayed, 0, "fast_rapidhash corpus should not be empty");
}

#[test]
fn replay_fast_xxh3_corpus() {
    let replayed = replay_corpus_dir("fast_xxh3", corpus_dir("fast_xxh3"), fast_xxh3::run);
    assert_ne!(replayed, 0, "fast_xxh3 corpus should not be empty");
}

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

#[test]
fn replay_hash_blake2b_corpus() {
    let replayed = replay_corpus_dir("hash_blake2b", corpus_dir("hash_blake2b"), hash_blake2b::run);
    assert_ne!(replayed, 0, "hash_blake2b corpus should not be empty");
}

#[test]
fn replay_hash_blake2s_corpus() {
    let replayed = replay_corpus_dir("hash_blake2s", corpus_dir("hash_blake2s"), hash_blake2s::run);
    assert_ne!(replayed, 0, "hash_blake2s corpus should not be empty");
}

#[test]
fn replay_hash_blake3_corpus() {
    let replayed = replay_corpus_dir("hash_blake3", corpus_dir("hash_blake3"), hash_blake3::run);
    assert_ne!(replayed, 0, "hash_blake3 corpus should not be empty");
}

#[test]
fn replay_hash_blake3_derive_corpus() {
    let replayed = replay_corpus_dir("hash_blake3_derive", corpus_dir("hash_blake3_derive"), hash_blake3_derive::run);
    assert_ne!(replayed, 0, "hash_blake3_derive corpus should not be empty");
}

#[test]
fn replay_hash_blake3_keyed_corpus() {
    let replayed = replay_corpus_dir("hash_blake3_keyed", corpus_dir("hash_blake3_keyed"), hash_blake3_keyed::run);
    assert_ne!(replayed, 0, "hash_blake3_keyed corpus should not be empty");
}

#[test]
fn replay_hash_cshake256_corpus() {
    let replayed = replay_corpus_dir("hash_cshake256", corpus_dir("hash_cshake256"), hash_cshake256::run);
    assert_ne!(replayed, 0, "hash_cshake256 corpus should not be empty");
}

#[test]
fn replay_hash_sha2_corpus() {
    let replayed = replay_corpus_dir("hash_sha2", corpus_dir("hash_sha2"), hash_sha2::run);
    assert_ne!(replayed, 0, "hash_sha2 corpus should not be empty");
}

#[test]
fn replay_hash_sha3_corpus() {
    let replayed = replay_corpus_dir("hash_sha3", corpus_dir("hash_sha3"), hash_sha3::run);
    assert_ne!(replayed, 0, "hash_sha3 corpus should not be empty");
}

#[test]
fn replay_hex_parse_corpus() {
    let replayed = replay_corpus_dir("hex_parse", corpus_dir("hex_parse"), hex_parse::run);
    assert_ne!(replayed, 0, "hex_parse corpus should not be empty");
}

#[test]
fn replay_traits_io_corpus() {
    let replayed = replay_corpus_dir("traits_io", corpus_dir("traits_io"), traits_io::run);
    assert_ne!(replayed, 0, "traits_io corpus should not be empty");
}
