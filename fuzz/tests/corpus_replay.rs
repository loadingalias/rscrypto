use std::path::PathBuf;

use rscrypto_fuzz::replay_corpus_dir;

fn corpus_dir(target: &str) -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus").join(target)
}

macro_rules! replay_tests {
  ($($test:ident, $target:ident, $source:literal;)*) => {$(
    #[path = $source]
    mod $target;

    #[test]
    fn $test() {
      let name = stringify!($target);
      let replayed = replay_corpus_dir(name, corpus_dir(name), $target::run);
      assert_ne!(replayed, 0, "{name} corpus should not be empty");
    }
  )*};
}

replay_tests! {
  replay_aead_aegis256_corpus, aead_aegis256, "../target_impls/aead_aegis256.rs";
  replay_aead_aes128gcm_corpus, aead_aes128gcm, "../target_impls/aead_aes128gcm.rs";
  replay_aead_aes128gcmsiv_corpus, aead_aes128gcmsiv, "../target_impls/aead_aes128gcmsiv.rs";
  replay_aead_aes256gcm_corpus, aead_aes256gcm, "../target_impls/aead_aes256gcm.rs";
  replay_aead_aes256gcmsiv_corpus, aead_aes256gcmsiv, "../target_impls/aead_aes256gcmsiv.rs";
  replay_aead_ascon128_corpus, aead_ascon128, "../target_impls/aead_ascon128.rs";
  replay_aead_chacha20poly1305_corpus, aead_chacha20poly1305, "../target_impls/aead_chacha20poly1305.rs";
  replay_aead_header_protection_corpus, aead_header_protection, "../target_impls/aead_header_protection.rs";
  replay_aead_aes_siv_cmac256_corpus, aead_aes_siv_cmac256, "../target_impls/aead_aes_siv_cmac256.rs";
  replay_aead_nonce_counter_corpus, aead_nonce_counter, "../target_impls/aead_nonce_counter.rs";
  replay_aead_quic_lb_corpus, aead_quic_lb, "../target_impls/aead_quic_lb.rs";
  replay_aead_xchacha20poly1305_corpus, aead_xchacha20poly1305, "../target_impls/aead_xchacha20poly1305.rs";
  replay_auth_argon2d_corpus, auth_argon2d, "../target_impls/auth_argon2d.rs";
  replay_auth_argon2i_corpus, auth_argon2i, "../target_impls/auth_argon2i.rs";
  replay_auth_argon2id_corpus, auth_argon2id, "../target_impls/auth_argon2id.rs";
  replay_auth_ed25519_corpus, auth_ed25519, "../target_impls/auth_ed25519.rs";
  replay_auth_ed25519_verify_corpus, auth_ed25519_verify, "../target_impls/auth_ed25519_verify.rs";
  replay_auth_ecdsa_verify_corpus, auth_ecdsa_verify, "../target_impls/auth_ecdsa_verify.rs";
  replay_auth_ecdsa_sign_corpus, auth_ecdsa_sign, "../target_impls/auth_ecdsa_sign.rs";
  replay_auth_hkdf_sha256_corpus, auth_hkdf_sha256, "../target_impls/auth_hkdf_sha256.rs";
  replay_auth_hkdf_sha384_corpus, auth_hkdf_sha384, "../target_impls/auth_hkdf_sha384.rs";
  replay_auth_hmac_sha256_corpus, auth_hmac_sha256, "../target_impls/auth_hmac_sha256.rs";
  replay_auth_hmac_sha384_corpus, auth_hmac_sha384, "../target_impls/auth_hmac_sha384.rs";
  replay_auth_hmac_sha512_corpus, auth_hmac_sha512, "../target_impls/auth_hmac_sha512.rs";
  replay_auth_kmac256_corpus, auth_kmac256, "../target_impls/auth_kmac256.rs";
  replay_auth_mlkem512_corpus, auth_mlkem512, "../target_impls/auth_mlkem512.rs";
  replay_auth_mlkem768_corpus, auth_mlkem768, "../target_impls/auth_mlkem768.rs";
  replay_auth_mlkem1024_corpus, auth_mlkem1024, "../target_impls/auth_mlkem1024.rs";
  replay_auth_hpke_corpus, auth_hpke, "../target_impls/auth_hpke.rs";
  replay_auth_xwing_corpus, auth_xwing, "../target_impls/auth_xwing.rs";
  replay_auth_pbkdf2_corpus, auth_pbkdf2, "../target_impls/auth_pbkdf2.rs";
  replay_auth_phc_corpus, auth_phc, "../target_impls/auth_phc.rs";
  replay_auth_rsa_public_key_corpus, auth_rsa_public_key, "../target_impls/auth_rsa_public_key.rs";
  replay_auth_rsa_import_corpus, auth_rsa_import, "../target_impls/auth_rsa_import.rs";
  replay_auth_rsa_protocol_corpus, auth_rsa_protocol, "../target_impls/auth_rsa_protocol.rs";
  replay_auth_rsa_private_ops_corpus, auth_rsa_private_ops, "../target_impls/auth_rsa_private_ops.rs";
  replay_auth_rsa_verify_corpus, auth_rsa_verify, "../target_impls/auth_rsa_verify.rs";
  replay_auth_scrypt_corpus, auth_scrypt, "../target_impls/auth_scrypt.rs";
  replay_auth_siphash_corpus, auth_siphash, "../target_impls/auth_siphash.rs";
  replay_auth_x25519_corpus, auth_x25519, "../target_impls/auth_x25519.rs";
  replay_auth_p256_ecdh_corpus, auth_p256_ecdh, "../target_impls/auth_p256_ecdh.rs";
  replay_auth_p384_ecdh_corpus, auth_p384_ecdh, "../target_impls/auth_p384_ecdh.rs";
  replay_checksum_adler32_corpus, checksum_adler32, "../target_impls/checksum_adler32.rs";
  replay_checksum_crc_corpus, checksum_crc, "../target_impls/checksum_crc.rs";
  replay_fast_rapidhash_corpus, fast_rapidhash, "../target_impls/fast_rapidhash.rs";
  replay_fast_xxh3_corpus, fast_xxh3, "../target_impls/fast_xxh3.rs";
  replay_fast_xxh32_corpus, fast_xxh32, "../target_impls/fast_xxh32.rs";
  replay_fast_xxh64_corpus, fast_xxh64, "../target_impls/fast_xxh64.rs";
  replay_hash_ascon_corpus, hash_ascon, "../target_impls/hash_ascon.rs";
  replay_hash_ascon_cxof_corpus, hash_ascon_cxof, "../target_impls/hash_ascon_cxof.rs";
  replay_hash_blake2b_corpus, hash_blake2b, "../target_impls/hash_blake2b.rs";
  replay_hash_blake2s_corpus, hash_blake2s, "../target_impls/hash_blake2s.rs";
  replay_hash_blake3_corpus, hash_blake3, "../target_impls/hash_blake3.rs";
  replay_hash_blake3_derive_corpus, hash_blake3_derive, "../target_impls/hash_blake3_derive.rs";
  replay_hash_blake3_keyed_corpus, hash_blake3_keyed, "../target_impls/hash_blake3_keyed.rs";
  replay_hash_cshake256_corpus, hash_cshake256, "../target_impls/hash_cshake256.rs";
  replay_hash_sha2_corpus, hash_sha2, "../target_impls/hash_sha2.rs";
  replay_hash_sha3_corpus, hash_sha3, "../target_impls/hash_sha3.rs";
  replay_hash_websocket_sha1_corpus, hash_websocket_sha1, "../target_impls/hash_websocket_sha1.rs";
  replay_hex_parse_corpus, hex_parse, "../target_impls/hex_parse.rs";
  replay_traits_io_corpus, traits_io, "../target_impls/traits_io.rs";
  replay_auth_mldsa_corpus, auth_mldsa, "../target_impls/auth_mldsa.rs";
  replay_auth_composite_mldsa_corpus, auth_composite_mldsa, "../target_impls/auth_composite_mldsa.rs";
  replay_auth_slhdsa_corpus, auth_slhdsa, "../target_impls/auth_slhdsa.rs";
  replay_provider_quic_handshake_corpus, provider_quic_handshake, "../target_impls/provider_quic_handshake.rs";
  replay_provider_quic_header_corpus, provider_quic_header, "../target_impls/provider_quic_header.rs";
  replay_provider_quic_retry_corpus, provider_quic_retry, "../target_impls/provider_quic_retry.rs";
  replay_provider_quic_token_corpus, provider_quic_token, "../target_impls/provider_quic_token.rs";
  replay_provider_quic_lb_corpus, provider_quic_lb, "../target_impls/provider_quic_lb.rs";
  replay_auth_chacha20_rng_corpus, auth_chacha20_rng, "../target_impls/auth_chacha20_rng.rs";
  replay_auth_hkdf_sha512_corpus, auth_hkdf_sha512, "../target_impls/auth_hkdf_sha512.rs";
  replay_auth_hmac_sha3_corpus, auth_hmac_sha3, "../target_impls/auth_hmac_sha3.rs";
  replay_auth_kmac128_corpus, auth_kmac128, "../target_impls/auth_kmac128.rs";
  replay_auth_poly1305_corpus, auth_poly1305, "../target_impls/auth_poly1305.rs";
  replay_hash_cshake128_corpus, hash_cshake128, "../target_impls/hash_cshake128.rs";
  replay_serde_secrets_corpus, serde_secrets, "../target_impls/serde_secrets.rs";
}
