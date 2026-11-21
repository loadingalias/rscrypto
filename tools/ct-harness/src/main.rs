//! Final linked constant-time evidence binary.

fn main() {
  macro_rules! retain {
    ($($entry:path),+ $(,)?) => {
      $(let _ = core::hint::black_box($entry as *const ());)+
    };
  }

  retain!(
    rscrypto_ct_harness::zeroization::zeroize_entry_secret_bytes_32,
    rscrypto_ct_harness::ct_entry_owner_eq_16,
    rscrypto_ct_harness::ct_entry_owner_eq_28,
    rscrypto_ct_harness::ct_entry_owner_eq_32,
    rscrypto_ct_harness::ct_entry_owner_eq_48,
    rscrypto_ct_harness::ct_entry_owner_eq_64,
    rscrypto_ct_harness::ct_entry_owner_eq_1632,
    rscrypto_ct_harness::ct_entry_owner_eq_2400,
    rscrypto_ct_harness::ct_entry_owner_eq_3168,
    rscrypto_ct_harness::ct_entry_kmac256_verify,
    rscrypto_ct_harness::ct_entry_mlkem512_decapsulate,
    rscrypto_ct_harness::ct_entry_mlkem768_decapsulate,
    rscrypto_ct_harness::ct_entry_mlkem1024_decapsulate,
    rscrypto_ct_harness::ct_entry_argon2i_verify,
    rscrypto_ct_harness::ct_entry_argon2d_verify,
    rscrypto_ct_harness::ct_entry_argon2id_verify,
    rscrypto_ct_harness::ct_entry_scrypt_verify,
    rscrypto_ct_harness::ct_entry_rsa_pkcs1v15_sign_fixed_blinding,
    rscrypto_ct_harness::ct_entry_rsa_pss_sign_fixed_blinding,
    rscrypto_ct_harness::ct_entry_rsa_oaep_decrypt_fixed_blinding,
    rscrypto_ct_harness::ct_entry_rsa_pkcs1v15_decrypt_fixed_blinding,
  );

  retain!(rscrypto_ct_harness::ct_entry_rsa_private_window_select);

  retain!(rscrypto_ct_harness::ct_entry_poly1305_authenticate);

  retain!(rscrypto_ct_harness::ct_entry_siphash_hash);

  retain!(
    rscrypto_ct_harness::ct_entry_quic_lb_encrypt,
    rscrypto_ct_harness::ct_entry_quic_lb_decrypt,
  );

  retain!(rscrypto_ct_harness::ct_entry_chacha20_rng_fill);

  retain!(
    rscrypto_ct_harness::ct_entry_p256_ecdh_public_key,
    rscrypto_ct_harness::ct_entry_p256_ecdh_agree,
  );

  // RSA key preparation and the diagnostic PKCS#8 roundtrip stay outside the steady-state roots,
  // which borrow the prepared key.
  retain!(
    rscrypto_ct_harness::ct_entry_rsa_private_key_pkcs8_roundtrip,
    rscrypto_ct_harness::ct_entry_rsa_fixture_key_import,
    rscrypto_ct_harness::ct_entry_rsa_fixture_key_release,
    rscrypto_ct_harness::ct_entry_rsa_private_component_validation_32,
  );

  retain!(
    rscrypto_ct_harness::mldsa::ct_entry_mldsa_ntt,
    rscrypto_ct_harness::mldsa::ct_entry_mldsa_montgomery,
    rscrypto_ct_harness::mldsa::ct_entry_mldsa_inverse_ntt,
    rscrypto_ct_harness::mldsa::ct_entry_mldsa_inverse_ntt_portable,
    rscrypto_ct_harness::mldsa::ct_entry_mldsa_norm,
    rscrypto_ct_harness::mldsa::ct_entry_mldsa_rounding44,
    rscrypto_ct_harness::mldsa::ct_entry_mldsa_rounding65,
    rscrypto_ct_harness::mldsa::ct_entry_mldsa_product,
    rscrypto_ct_harness::mldsa::ct_entry_mldsa_accumulate,
    rscrypto_ct_harness::mldsa::ct_entry_mldsa_row_product,
    rscrypto_ct_harness::mldsa::ct_entry_mldsa_noise_eta2,
    rscrypto_ct_harness::mldsa::ct_entry_mldsa_noise_eta4,
    rscrypto_ct_harness::mldsa::ct_entry_mldsa_challenge44,
    rscrypto_ct_harness::mldsa::ct_entry_mldsa_challenge65,
    rscrypto_ct_harness::mldsa::ct_entry_mldsa_challenge87,
    rscrypto_ct_harness::mldsa::ct_entry_mldsa_mask17,
    rscrypto_ct_harness::mldsa::ct_entry_mldsa_mask19,
    rscrypto_ct_harness::mldsa::ct_entry_mldsa_prepare44,
    rscrypto_ct_harness::mldsa::ct_entry_mldsa_prepare65,
    rscrypto_ct_harness::mldsa::ct_entry_mldsa_prepare87,
  );

  // The asm scan walks every manifest root's closure in this binary, so each root must survive linking.
  retain!(
    rscrypto_ct_harness::ct_entry_aegis256_open,
    rscrypto_ct_harness::ct_entry_aes128gcm_open,
    rscrypto_ct_harness::ct_entry_aes128gcmsiv_open,
    rscrypto_ct_harness::ct_entry_aes256gcm_open,
    rscrypto_ct_harness::ct_entry_aes256gcmsiv_open,
    rscrypto_ct_harness::ct_entry_aes_siv_cmac256_open,
    rscrypto_ct_harness::ct_entry_aes_siv_cmac256_open_portable,
    rscrypto_ct_harness::ct_entry_argon2d_hash,
    rscrypto_ct_harness::ct_entry_argon2i_hash,
    rscrypto_ct_harness::ct_entry_argon2id_hash,
    rscrypto_ct_harness::ct_entry_ascon_aead128_open,
    rscrypto_ct_harness::ct_entry_blake2b256_keyed_digest,
    rscrypto_ct_harness::ct_entry_blake2b512_keyed_digest,
    rscrypto_ct_harness::ct_entry_blake2s128_keyed_digest,
    rscrypto_ct_harness::ct_entry_blake2s256_keyed_digest,
    rscrypto_ct_harness::ct_entry_blake3_keyed_digest,
    rscrypto_ct_harness::ct_entry_blake3_verify_keyed,
    rscrypto_ct_harness::ct_entry_chacha20poly1305_open,
    rscrypto_ct_harness::ct_entry_classification_best_effort,
    rscrypto_ct_harness::ct_entry_classification_public_only,
    rscrypto_ct_harness::ct_entry_ecdsa_p256_default_public_key,
    rscrypto_ct_harness::ct_entry_ecdsa_p256_default_sign,
    rscrypto_ct_harness::ct_entry_ecdsa_p256_public_key,
    rscrypto_ct_harness::ct_entry_ecdsa_p256_sign,
    rscrypto_ct_harness::ct_entry_ecdsa_p384_default_public_key,
    rscrypto_ct_harness::ct_entry_ecdsa_p384_default_sign,
    rscrypto_ct_harness::ct_entry_ecdsa_p384_public_key,
    rscrypto_ct_harness::ct_entry_ecdsa_p384_sign,
    rscrypto_ct_harness::ct_entry_ed25519_sign,
    rscrypto_ct_harness::ct_entry_ed25519_verify,
    rscrypto_ct_harness::ct_entry_hkdf_sha256_derive,
    rscrypto_ct_harness::ct_entry_hkdf_sha384_derive,
    rscrypto_ct_harness::ct_entry_hmac_sha256_verify,
    rscrypto_ct_harness::ct_entry_hmac_sha256_verify_truncated_64,
    rscrypto_ct_harness::ct_entry_hmac_sha384_verify,
    rscrypto_ct_harness::ct_entry_hmac_sha512_verify,
    rscrypto_ct_harness::ct_entry_hpke_x25519_derive_keypair,
    rscrypto_ct_harness::ct_entry_hpke_x25519_open,
    rscrypto_ct_harness::ct_entry_hpke_x25519_seal,
    rscrypto_ct_harness::ct_entry_hpke_xwing_derive_keypair,
    rscrypto_ct_harness::ct_entry_hpke_xwing_open,
    rscrypto_ct_harness::ct_entry_hpke_xwing_seal,
    rscrypto_ct_harness::ct_entry_mlkem1024_encapsulate,
    rscrypto_ct_harness::ct_entry_mlkem1024_keygen,
    rscrypto_ct_harness::ct_entry_mlkem512_encapsulate,
    rscrypto_ct_harness::ct_entry_mlkem512_keygen,
    rscrypto_ct_harness::ct_entry_mlkem768_encapsulate,
    rscrypto_ct_harness::ct_entry_mlkem768_keygen,
    rscrypto_ct_harness::ct_entry_mlkem_diag_compress_decompress,
    rscrypto_ct_harness::ct_entry_mlkem_diag_from_product_domain,
    rscrypto_ct_harness::ct_entry_mlkem_diag_inverse_ntt,
    rscrypto_ct_harness::ct_entry_mlkem_diag_multiply_ntts_accumulate_k3,
    rscrypto_ct_harness::ct_entry_mlkem_diag_multiply_ntts_accumulate_k4,
    rscrypto_ct_harness::ct_entry_mlkem_diag_multiply_ntts_add_assign,
    rscrypto_ct_harness::ct_entry_mlkem_diag_ntt,
    rscrypto_ct_harness::ct_entry_mlkem_diag_to_product_domain,
    rscrypto_ct_harness::ct_entry_p384_ecdh_agree,
    rscrypto_ct_harness::ct_entry_p384_ecdh_public_key,
    rscrypto_ct_harness::ct_entry_pbkdf2_sha256_derive,
    rscrypto_ct_harness::ct_entry_pbkdf2_sha256_verify,
    rscrypto_ct_harness::ct_entry_pbkdf2_sha256_verify_full,
    rscrypto_ct_harness::ct_entry_pbkdf2_sha512_derive,
    rscrypto_ct_harness::ct_entry_pbkdf2_sha512_verify,
    rscrypto_ct_harness::ct_entry_public_crc32_checksum,
    rscrypto_ct_harness::ct_entry_public_sha256_digest,
    rscrypto_ct_harness::ct_entry_secret_bytes32_debug_masked,
    rscrypto_ct_harness::ct_entry_x25519,
    rscrypto_ct_harness::ct_entry_x25519_public_key,
    rscrypto_ct_harness::ct_entry_xchacha20poly1305_open,
    rscrypto_ct_harness::ct_entry_xwing_decapsulate,
    rscrypto_ct_harness::ct_entry_xwing_encapsulate,
    rscrypto_ct_harness::ct_entry_xwing_keygen,
  );
}
