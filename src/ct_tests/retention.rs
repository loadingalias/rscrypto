//! Final linked constant-time evidence binary.

pub(super) fn retain_roots() {
  macro_rules! retain {
    ($($entry:path),+ $(,)?) => {
      $(let _ = core::hint::black_box($entry as *const ());)+
    };
  }

  retain!(
    super::operations::zeroization::zeroize_entry_secret_bytes_32,
    super::operations::ct_entry_owner_eq_16,
    super::operations::ct_entry_owner_eq_28,
    super::operations::ct_entry_owner_eq_32,
    super::operations::ct_entry_owner_eq_48,
    super::operations::ct_entry_owner_eq_64,
    super::operations::ct_entry_owner_eq_1632,
    super::operations::ct_entry_owner_eq_2400,
    super::operations::ct_entry_owner_eq_3168,
    super::operations::ct_entry_kmac256_verify,
    super::operations::ct_entry_mlkem512_decapsulate,
    super::operations::ct_entry_mlkem768_decapsulate,
    super::operations::ct_entry_mlkem1024_decapsulate,
    super::operations::ct_entry_argon2i_verify,
    super::operations::ct_entry_argon2d_verify,
    super::operations::ct_entry_argon2id_verify,
    super::operations::ct_entry_scrypt_verify,
    super::operations::ct_entry_rsa_pkcs1v15_sign_fixed_blinding,
    super::operations::ct_entry_rsa_pss_sign_fixed_blinding,
    super::operations::ct_entry_rsa_oaep_decrypt_fixed_blinding,
    super::operations::ct_entry_rsa_pkcs1v15_decrypt_fixed_blinding,
  );

  retain!(super::leaves::ct_entry_rsa_private_window_select);

  retain!(super::operations::ct_entry_poly1305_authenticate);

  retain!(super::operations::ct_entry_siphash_hash);

  retain!(
    super::operations::ct_entry_quic_lb_encrypt,
    super::operations::ct_entry_quic_lb_decrypt,
  );

  retain!(super::operations::ct_entry_chacha20_rng_fill);

  retain!(
    super::operations::ct_entry_p256_ecdh_public_key,
    super::operations::ct_entry_p256_ecdh_agree,
  );

  // RSA key preparation and the diagnostic PKCS#8 roundtrip stay outside the steady-state roots,
  // which borrow the prepared key.
  retain!(
    super::operations::ct_entry_rsa_private_key_pkcs8_roundtrip,
    super::operations::ct_entry_rsa_fixture_key_import,
    super::operations::ct_entry_rsa_fixture_key_release,
    super::leaves::ct_entry_rsa_private_component_validation_32,
  );

  retain!(
    super::mldsa::ct_entry_mldsa_ntt,
    super::mldsa::ct_entry_mldsa_montgomery,
    super::mldsa::ct_entry_mldsa_inverse_ntt,
    super::mldsa::ct_entry_mldsa_inverse_ntt_portable,
    super::mldsa::ct_entry_mldsa_norm,
    super::mldsa::ct_entry_mldsa_rounding44,
    super::mldsa::ct_entry_mldsa_rounding65,
    super::mldsa::ct_entry_mldsa_product,
    super::mldsa::ct_entry_mldsa_accumulate,
    super::mldsa::ct_entry_mldsa_row_product,
    super::mldsa::ct_entry_mldsa_noise_eta2,
    super::mldsa::ct_entry_mldsa_noise_eta4,
    super::mldsa::ct_entry_mldsa_challenge44,
    super::mldsa::ct_entry_mldsa_challenge65,
    super::mldsa::ct_entry_mldsa_challenge87,
    super::mldsa::ct_entry_mldsa_mask17,
    super::mldsa::ct_entry_mldsa_mask19,
    super::mldsa::ct_entry_mldsa_prepare44,
    super::mldsa::ct_entry_mldsa_prepare65,
    super::mldsa::ct_entry_mldsa_prepare87,
  );

  // The asm scan walks every manifest root's closure in this binary, so each root must survive linking.
  retain!(
    super::operations::ct_entry_aegis256_open,
    super::operations::ct_entry_aes128gcm_open,
    super::operations::ct_entry_aes128gcmsiv_open,
    super::operations::ct_entry_aes256gcm_open,
    super::operations::ct_entry_aes256gcmsiv_open,
    super::operations::ct_entry_aes_siv_cmac256_open,
    super::leaves::ct_entry_aes_siv_cmac256_open_portable,
    super::operations::ct_entry_argon2d_hash,
    super::operations::ct_entry_argon2i_hash,
    super::operations::ct_entry_argon2id_hash,
    super::operations::ct_entry_ascon_aead128_open,
    super::operations::ct_entry_blake2b256_keyed_digest,
    super::operations::ct_entry_blake2b512_keyed_digest,
    super::operations::ct_entry_blake2s128_keyed_digest,
    super::operations::ct_entry_blake2s256_keyed_digest,
    super::operations::ct_entry_blake3_keyed_digest,
    super::operations::ct_entry_blake3_verify_keyed,
    super::operations::ct_entry_chacha20poly1305_open,
    super::operations::ct_entry_classification_best_effort,
    super::operations::ct_entry_classification_public_only,
    super::operations::ct_entry_ecdsa_p256_default_public_key,
    super::operations::ct_entry_ecdsa_p256_default_sign,
    super::operations::ct_entry_ecdsa_p256_public_key,
    super::operations::ct_entry_ecdsa_p256_sign,
    super::operations::ct_entry_ecdsa_p384_default_public_key,
    super::operations::ct_entry_ecdsa_p384_default_sign,
    super::operations::ct_entry_ecdsa_p384_public_key,
    super::operations::ct_entry_ecdsa_p384_sign,
    super::operations::ct_entry_ed25519_sign,
    super::operations::ct_entry_ed25519_verify,
    super::operations::ct_entry_hkdf_sha256_derive,
    super::operations::ct_entry_hkdf_sha384_derive,
    super::operations::ct_entry_hmac_sha256_verify,
    super::operations::ct_entry_hmac_sha256_verify_truncated_64,
    super::operations::ct_entry_hmac_sha384_verify,
    super::operations::ct_entry_hmac_sha512_verify,
    super::operations::ct_entry_hpke_x25519_derive_keypair,
    super::operations::ct_entry_hpke_x25519_open,
    super::operations::ct_entry_hpke_x25519_seal,
    super::operations::ct_entry_hpke_xwing_derive_keypair,
    super::operations::ct_entry_hpke_xwing_open,
    super::operations::ct_entry_hpke_xwing_seal,
    super::operations::ct_entry_mlkem1024_encapsulate,
    super::operations::ct_entry_mlkem1024_keygen,
    super::operations::ct_entry_mlkem512_encapsulate,
    super::operations::ct_entry_mlkem512_keygen,
    super::operations::ct_entry_mlkem768_encapsulate,
    super::operations::ct_entry_mlkem768_keygen,
    super::leaves::ct_entry_mlkem_diag_compress_decompress,
    super::leaves::ct_entry_mlkem_diag_from_product_domain,
    super::leaves::ct_entry_mlkem_diag_inverse_ntt,
    super::leaves::ct_entry_mlkem_diag_multiply_ntts_accumulate_k3,
    super::leaves::ct_entry_mlkem_diag_multiply_ntts_accumulate_k4,
    super::leaves::ct_entry_mlkem_diag_multiply_ntts_add_assign,
    super::leaves::ct_entry_mlkem_diag_ntt,
    super::leaves::ct_entry_mlkem_diag_to_product_domain,
    super::operations::ct_entry_p384_ecdh_agree,
    super::operations::ct_entry_p384_ecdh_public_key,
    super::operations::ct_entry_pbkdf2_sha256_derive,
    super::operations::ct_entry_pbkdf2_sha256_verify,
    super::operations::ct_entry_pbkdf2_sha256_verify_full,
    super::operations::ct_entry_pbkdf2_sha512_derive,
    super::operations::ct_entry_pbkdf2_sha512_verify,
    super::operations::ct_entry_public_crc32_checksum,
    super::operations::ct_entry_public_sha256_digest,
    super::operations::ct_entry_secret_bytes32_debug_masked,
    super::operations::ct_entry_x25519,
    super::operations::ct_entry_x25519_public_key,
    super::operations::ct_entry_xchacha20poly1305_open,
    super::operations::ct_entry_xwing_decapsulate,
    super::operations::ct_entry_xwing_encapsulate,
    super::operations::ct_entry_xwing_keygen,
  );
}
