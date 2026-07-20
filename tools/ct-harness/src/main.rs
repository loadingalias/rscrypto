//! Final linked constant-time evidence binary.

fn main() {
  macro_rules! retain {
    ($($entry:path),+ $(,)?) => {
      $(let _ = std::hint::black_box($entry as *const ());)+
    };
  }

  retain!(
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
}
