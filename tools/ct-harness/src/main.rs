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
    rscrypto_ct_harness::ct_entry_rsa_private_key_pkcs8_roundtrip,
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
}
