use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn repo_root() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn replay(target: &str, corpus_dir: PathBuf, run: fn(&[u8])) {
  let replayed = replay_corpus_dir(target, corpus_dir, run);
  assert_ne!(replayed, 0, "{target} corpus should not be empty");
}

#[path = "../fuzz/target_impls/aead_aegis256.rs"]
mod aead_aegis256;

#[path = "../fuzz/target_impls/aead_aes256gcm.rs"]
mod aead_aes256gcm;

#[path = "../fuzz/target_impls/aead_aes256gcmsiv.rs"]
mod aead_aes256gcmsiv;

#[path = "../fuzz/target_impls/aead_ascon128.rs"]
mod aead_ascon128;

#[path = "../fuzz/target_impls/aead_chacha20poly1305.rs"]
mod aead_chacha20poly1305;

#[path = "../fuzz/target_impls/aead_nonce_counter.rs"]
mod aead_nonce_counter;

#[path = "../fuzz/target_impls/aead_xchacha20poly1305.rs"]
mod aead_xchacha20poly1305;

#[path = "../fuzz/target_impls/auth_argon2d.rs"]
mod auth_argon2d;

#[path = "../fuzz/target_impls/auth_argon2i.rs"]
mod auth_argon2i;

#[path = "../fuzz/target_impls/auth_argon2id.rs"]
mod auth_argon2id;

#[path = "../fuzz/target_impls/auth_ed25519.rs"]
mod auth_ed25519;

#[path = "../fuzz/target_impls/auth_ed25519_verify.rs"]
mod auth_ed25519_verify;

#[path = "../fuzz/target_impls/auth_hkdf_sha256.rs"]
mod auth_hkdf_sha256;

#[path = "../fuzz/target_impls/auth_hkdf_sha384.rs"]
mod auth_hkdf_sha384;

#[path = "../fuzz/target_impls/auth_hmac_sha256.rs"]
mod auth_hmac_sha256;

#[path = "../fuzz/target_impls/auth_hmac_sha384.rs"]
mod auth_hmac_sha384;

#[path = "../fuzz/target_impls/auth_hmac_sha512.rs"]
mod auth_hmac_sha512;

#[path = "../fuzz/target_impls/auth_kmac256.rs"]
mod auth_kmac256;

#[path = "../fuzz/target_impls/auth_pbkdf2.rs"]
mod auth_pbkdf2;

#[path = "../fuzz/target_impls/auth_phc.rs"]
mod auth_phc;

#[path = "../fuzz/target_impls/auth_scrypt.rs"]
mod auth_scrypt;

#[path = "../fuzz/target_impls/auth_x25519.rs"]
mod auth_x25519;

#[path = "../fuzz/target_impls/checksum_crc.rs"]
mod checksum_crc;

#[path = "../fuzz/target_impls/checksum_crc16.rs"]
mod checksum_crc16;

#[path = "../fuzz/target_impls/checksum_crc24.rs"]
mod checksum_crc24;

#[path = "../fuzz/target_impls/checksum_crc32.rs"]
mod checksum_crc32;

#[path = "../fuzz/target_impls/checksum_crc64.rs"]
mod checksum_crc64;

#[path = "../fuzz/target_impls/fast_rapidhash.rs"]
mod fast_rapidhash;

#[path = "../fuzz/target_impls/fast_xxh3.rs"]
mod fast_xxh3;

#[path = "../fuzz/target_impls/hash_ascon.rs"]
mod hash_ascon;

#[path = "../fuzz/target_impls/hash_ascon_cxof.rs"]
mod hash_ascon_cxof;

#[path = "../fuzz/target_impls/hash_blake2b.rs"]
mod hash_blake2b;

#[path = "../fuzz/target_impls/hash_blake2s.rs"]
mod hash_blake2s;

#[path = "../fuzz/target_impls/hash_blake3.rs"]
mod hash_blake3;

#[path = "../fuzz/target_impls/hash_blake3_derive.rs"]
mod hash_blake3_derive;

#[path = "../fuzz/target_impls/hash_blake3_keyed.rs"]
mod hash_blake3_keyed;

#[path = "../fuzz/target_impls/hash_cshake256.rs"]
mod hash_cshake256;

#[path = "../fuzz/target_impls/hash_sha2.rs"]
mod hash_sha2;

#[path = "../fuzz/target_impls/hash_sha3.rs"]
mod hash_sha3;

#[path = "../fuzz/target_impls/hex_parse.rs"]
mod hex_parse;

#[path = "../fuzz/target_impls/traits_io.rs"]
mod traits_io;

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_aead_aegis256_corpus() {
  replay(
    "aead_aegis256",
    repo_root().join("fuzz/corpus/aead_aegis256"),
    aead_aegis256::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_aead_aes256gcm_corpus() {
  replay(
    "aead_aes256gcm",
    repo_root().join("fuzz/corpus/aead_aes256gcm"),
    aead_aes256gcm::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_aead_aes256gcmsiv_corpus() {
  replay(
    "aead_aes256gcmsiv",
    repo_root().join("fuzz/corpus/aead_aes256gcmsiv"),
    aead_aes256gcmsiv::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_aead_ascon128_corpus() {
  replay(
    "aead_ascon128",
    repo_root().join("fuzz/corpus/aead_ascon128"),
    aead_ascon128::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_aead_chacha20poly1305_corpus() {
  replay(
    "aead_chacha20poly1305",
    repo_root().join("fuzz/corpus/aead_chacha20poly1305"),
    aead_chacha20poly1305::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_aead_nonce_counter_corpus() {
  replay(
    "aead_nonce_counter",
    repo_root().join("fuzz/corpus/aead_nonce_counter"),
    aead_nonce_counter::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_aead_xchacha20poly1305_corpus() {
  replay(
    "aead_xchacha20poly1305",
    repo_root().join("fuzz/corpus/aead_xchacha20poly1305"),
    aead_xchacha20poly1305::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_auth_argon2d_corpus() {
  replay(
    "auth_argon2d",
    repo_root().join("fuzz/corpus/auth_argon2d"),
    auth_argon2d::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_auth_argon2i_corpus() {
  replay(
    "auth_argon2i",
    repo_root().join("fuzz/corpus/auth_argon2i"),
    auth_argon2i::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_auth_argon2id_corpus() {
  replay(
    "auth_argon2id",
    repo_root().join("fuzz/corpus/auth_argon2id"),
    auth_argon2id::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_auth_ed25519_corpus() {
  replay(
    "auth_ed25519",
    repo_root().join("fuzz/corpus/auth_ed25519"),
    auth_ed25519::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_auth_ed25519_verify_corpus() {
  replay(
    "auth_ed25519_verify",
    repo_root().join("fuzz/corpus/auth_ed25519_verify"),
    auth_ed25519_verify::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_auth_hkdf_sha256_corpus() {
  replay(
    "auth_hkdf_sha256",
    repo_root().join("fuzz/corpus/auth_hkdf_sha256"),
    auth_hkdf_sha256::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_auth_hkdf_sha384_corpus() {
  replay(
    "auth_hkdf_sha384",
    repo_root().join("fuzz/corpus/auth_hkdf_sha384"),
    auth_hkdf_sha384::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_auth_hmac_sha256_corpus() {
  replay(
    "auth_hmac_sha256",
    repo_root().join("fuzz/corpus/auth_hmac_sha256"),
    auth_hmac_sha256::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_auth_hmac_sha384_corpus() {
  replay(
    "auth_hmac_sha384",
    repo_root().join("fuzz/corpus/auth_hmac_sha384"),
    auth_hmac_sha384::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_auth_hmac_sha512_corpus() {
  replay(
    "auth_hmac_sha512",
    repo_root().join("fuzz/corpus/auth_hmac_sha512"),
    auth_hmac_sha512::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_auth_kmac256_corpus() {
  replay(
    "auth_kmac256",
    repo_root().join("fuzz/corpus/auth_kmac256"),
    auth_kmac256::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_auth_pbkdf2_corpus() {
  replay(
    "auth_pbkdf2",
    repo_root().join("fuzz/corpus/auth_pbkdf2"),
    auth_pbkdf2::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_auth_phc_corpus() {
  replay("auth_phc", repo_root().join("fuzz/corpus/auth_phc"), auth_phc::run);
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_auth_scrypt_corpus() {
  replay(
    "auth_scrypt",
    repo_root().join("fuzz/corpus/auth_scrypt"),
    auth_scrypt::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_auth_x25519_corpus() {
  replay(
    "auth_x25519",
    repo_root().join("fuzz/corpus/auth_x25519"),
    auth_x25519::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_checksum_crc_corpus() {
  replay(
    "checksum_crc",
    repo_root().join("fuzz/corpus/checksum_crc"),
    checksum_crc::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_fast_rapidhash_corpus() {
  replay(
    "fast_rapidhash",
    repo_root().join("fuzz/corpus/fast_rapidhash"),
    fast_rapidhash::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_fast_xxh3_corpus() {
  replay("fast_xxh3", repo_root().join("fuzz/corpus/fast_xxh3"), fast_xxh3::run);
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_hash_ascon_corpus() {
  replay(
    "hash_ascon",
    repo_root().join("fuzz/corpus/hash_ascon"),
    hash_ascon::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_hash_ascon_cxof_corpus() {
  replay(
    "hash_ascon_cxof",
    repo_root().join("fuzz/corpus/hash_ascon_cxof"),
    hash_ascon_cxof::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_hash_blake2b_corpus() {
  replay(
    "hash_blake2b",
    repo_root().join("fuzz/corpus/hash_blake2b"),
    hash_blake2b::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_hash_blake2s_corpus() {
  replay(
    "hash_blake2s",
    repo_root().join("fuzz/corpus/hash_blake2s"),
    hash_blake2s::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_hash_blake3_corpus() {
  replay(
    "hash_blake3",
    repo_root().join("fuzz/corpus/hash_blake3"),
    hash_blake3::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_hash_blake3_derive_corpus() {
  replay(
    "hash_blake3_derive",
    repo_root().join("fuzz/corpus/hash_blake3_derive"),
    hash_blake3_derive::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_hash_blake3_keyed_corpus() {
  replay(
    "hash_blake3_keyed",
    repo_root().join("fuzz/corpus/hash_blake3_keyed"),
    hash_blake3_keyed::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_hash_cshake256_corpus() {
  replay(
    "hash_cshake256",
    repo_root().join("fuzz/corpus/hash_cshake256"),
    hash_cshake256::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_hash_sha2_corpus() {
  replay("hash_sha2", repo_root().join("fuzz/corpus/hash_sha2"), hash_sha2::run);
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_hash_sha3_corpus() {
  replay("hash_sha3", repo_root().join("fuzz/corpus/hash_sha3"), hash_sha3::run);
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_hex_parse_corpus() {
  replay("hex_parse", repo_root().join("fuzz/corpus/hex_parse"), hex_parse::run);
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_full_traits_io_corpus() {
  replay("traits_io", repo_root().join("fuzz/corpus/traits_io"), traits_io::run);
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_aead_aegis256_aead_aegis256_corpus() {
  replay(
    "aead_aegis256",
    repo_root().join("fuzz-packages/aead-aegis256/corpus/aead_aegis256"),
    aead_aegis256::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_aead_aes256gcm_aead_aes256gcm_corpus() {
  replay(
    "aead_aes256gcm",
    repo_root().join("fuzz-packages/aead-aes256gcm/corpus/aead_aes256gcm"),
    aead_aes256gcm::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_aead_aes256gcmsiv_aead_aes256gcmsiv_corpus() {
  replay(
    "aead_aes256gcmsiv",
    repo_root().join("fuzz-packages/aead-aes256gcmsiv/corpus/aead_aes256gcmsiv"),
    aead_aes256gcmsiv::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_aead_ascon128_aead_ascon128_corpus() {
  replay(
    "aead_ascon128",
    repo_root().join("fuzz-packages/aead-ascon128/corpus/aead_ascon128"),
    aead_ascon128::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_aead_chacha20poly1305_aead_chacha20poly1305_corpus() {
  replay(
    "aead_chacha20poly1305",
    repo_root().join("fuzz-packages/aead-chacha20poly1305/corpus/aead_chacha20poly1305"),
    aead_chacha20poly1305::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_aead_nonce_counter_aead_nonce_counter_corpus() {
  replay(
    "aead_nonce_counter",
    repo_root().join("fuzz-packages/aead-nonce-counter/corpus/aead_nonce_counter"),
    aead_nonce_counter::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_aead_xchacha20poly1305_aead_xchacha20poly1305_corpus() {
  replay(
    "aead_xchacha20poly1305",
    repo_root().join("fuzz-packages/aead-xchacha20poly1305/corpus/aead_xchacha20poly1305"),
    aead_xchacha20poly1305::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_auth_argon2_auth_argon2d_corpus() {
  replay(
    "auth_argon2d",
    repo_root().join("fuzz-packages/auth-argon2/corpus/auth_argon2d"),
    auth_argon2d::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_auth_argon2_auth_argon2i_corpus() {
  replay(
    "auth_argon2i",
    repo_root().join("fuzz-packages/auth-argon2/corpus/auth_argon2i"),
    auth_argon2i::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_auth_argon2_auth_argon2id_corpus() {
  replay(
    "auth_argon2id",
    repo_root().join("fuzz-packages/auth-argon2/corpus/auth_argon2id"),
    auth_argon2id::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_auth_ed25519_auth_ed25519_corpus() {
  replay(
    "auth_ed25519",
    repo_root().join("fuzz-packages/auth-ed25519/corpus/auth_ed25519"),
    auth_ed25519::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_auth_ed25519_auth_ed25519_verify_corpus() {
  replay(
    "auth_ed25519_verify",
    repo_root().join("fuzz-packages/auth-ed25519/corpus/auth_ed25519_verify"),
    auth_ed25519_verify::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_auth_hkdf_auth_hkdf_sha256_corpus() {
  replay(
    "auth_hkdf_sha256",
    repo_root().join("fuzz-packages/auth-hkdf/corpus/auth_hkdf_sha256"),
    auth_hkdf_sha256::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_auth_hkdf_auth_hkdf_sha384_corpus() {
  replay(
    "auth_hkdf_sha384",
    repo_root().join("fuzz-packages/auth-hkdf/corpus/auth_hkdf_sha384"),
    auth_hkdf_sha384::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_auth_hmac_auth_hmac_sha256_corpus() {
  replay(
    "auth_hmac_sha256",
    repo_root().join("fuzz-packages/auth-hmac/corpus/auth_hmac_sha256"),
    auth_hmac_sha256::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_auth_hmac_auth_hmac_sha384_corpus() {
  replay(
    "auth_hmac_sha384",
    repo_root().join("fuzz-packages/auth-hmac/corpus/auth_hmac_sha384"),
    auth_hmac_sha384::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_auth_hmac_auth_hmac_sha512_corpus() {
  replay(
    "auth_hmac_sha512",
    repo_root().join("fuzz-packages/auth-hmac/corpus/auth_hmac_sha512"),
    auth_hmac_sha512::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_auth_kmac256_auth_kmac256_corpus() {
  replay(
    "auth_kmac256",
    repo_root().join("fuzz-packages/auth-kmac256/corpus/auth_kmac256"),
    auth_kmac256::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_auth_pbkdf2_auth_pbkdf2_corpus() {
  replay(
    "auth_pbkdf2",
    repo_root().join("fuzz-packages/auth-pbkdf2/corpus/auth_pbkdf2"),
    auth_pbkdf2::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_auth_phc_auth_phc_corpus() {
  replay(
    "auth_phc",
    repo_root().join("fuzz-packages/auth-phc/corpus/auth_phc"),
    auth_phc::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_auth_scrypt_auth_scrypt_corpus() {
  replay(
    "auth_scrypt",
    repo_root().join("fuzz-packages/auth-scrypt/corpus/auth_scrypt"),
    auth_scrypt::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_auth_x25519_auth_x25519_corpus() {
  replay(
    "auth_x25519",
    repo_root().join("fuzz-packages/auth-x25519/corpus/auth_x25519"),
    auth_x25519::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_checksums_crc16_checksum_crc16_corpus() {
  replay(
    "checksum_crc16",
    repo_root().join("fuzz-packages/checksums-crc16/corpus/checksum_crc16"),
    checksum_crc16::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_checksums_crc24_checksum_crc24_corpus() {
  replay(
    "checksum_crc24",
    repo_root().join("fuzz-packages/checksums-crc24/corpus/checksum_crc24"),
    checksum_crc24::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_checksums_crc32_checksum_crc32_corpus() {
  replay(
    "checksum_crc32",
    repo_root().join("fuzz-packages/checksums-crc32/corpus/checksum_crc32"),
    checksum_crc32::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_checksums_crc64_checksum_crc64_corpus() {
  replay(
    "checksum_crc64",
    repo_root().join("fuzz-packages/checksums-crc64/corpus/checksum_crc64"),
    checksum_crc64::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_fast_rapidhash_fast_rapidhash_corpus() {
  replay(
    "fast_rapidhash",
    repo_root().join("fuzz-packages/fast-rapidhash/corpus/fast_rapidhash"),
    fast_rapidhash::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_fast_xxh3_fast_xxh3_corpus() {
  replay(
    "fast_xxh3",
    repo_root().join("fuzz-packages/fast-xxh3/corpus/fast_xxh3"),
    fast_xxh3::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_hash_ascon_hash_ascon_corpus() {
  replay(
    "hash_ascon",
    repo_root().join("fuzz-packages/hash-ascon/corpus/hash_ascon"),
    hash_ascon::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_hash_ascon_hash_ascon_cxof_corpus() {
  replay(
    "hash_ascon_cxof",
    repo_root().join("fuzz-packages/hash-ascon/corpus/hash_ascon_cxof"),
    hash_ascon_cxof::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_hash_blake2_hash_blake2b_corpus() {
  replay(
    "hash_blake2b",
    repo_root().join("fuzz-packages/hash-blake2/corpus/hash_blake2b"),
    hash_blake2b::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_hash_blake2_hash_blake2s_corpus() {
  replay(
    "hash_blake2s",
    repo_root().join("fuzz-packages/hash-blake2/corpus/hash_blake2s"),
    hash_blake2s::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_hash_blake3_hash_blake3_corpus() {
  replay(
    "hash_blake3",
    repo_root().join("fuzz-packages/hash-blake3/corpus/hash_blake3"),
    hash_blake3::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_hash_blake3_hash_blake3_derive_corpus() {
  replay(
    "hash_blake3_derive",
    repo_root().join("fuzz-packages/hash-blake3/corpus/hash_blake3_derive"),
    hash_blake3_derive::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_hash_blake3_hash_blake3_keyed_corpus() {
  replay(
    "hash_blake3_keyed",
    repo_root().join("fuzz-packages/hash-blake3/corpus/hash_blake3_keyed"),
    hash_blake3_keyed::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_hash_sha2_hash_sha2_corpus() {
  replay(
    "hash_sha2",
    repo_root().join("fuzz-packages/hash-sha2/corpus/hash_sha2"),
    hash_sha2::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_hash_sha3_hash_cshake256_corpus() {
  replay(
    "hash_cshake256",
    repo_root().join("fuzz-packages/hash-sha3/corpus/hash_cshake256"),
    hash_cshake256::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_hash_sha3_hash_sha3_corpus() {
  replay(
    "hash_sha3",
    repo_root().join("fuzz-packages/hash-sha3/corpus/hash_sha3"),
    hash_sha3::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_surface_hex_parse_hex_parse_corpus() {
  replay(
    "hex_parse",
    repo_root().join("fuzz-packages/surface-hex-parse/corpus/hex_parse"),
    hex_parse::run,
  );
}

#[test]
#[ignore = "coverage-only fuzz corpus replay"]
fn replay_scoped_traits_io_traits_io_corpus() {
  replay(
    "traits_io",
    repo_root().join("fuzz-packages/traits-io/corpus/traits_io"),
    traits_io::run,
  );
}
