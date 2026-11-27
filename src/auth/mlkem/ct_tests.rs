use super::*;

macro_rules! mlkem_diag_keygen_secret_noise {
  ($name:ident, $k:expr, $eta1_random_bytes:expr, $dk_pke_bytes:expr, $ek_bytes:expr, $doc_name:literal) => {
    #[doc = concat!("Diagnostic digest for ", $doc_name, " PKE key generation with fixed public matrix seed.")]
    /// Production key generation derives
    /// both seeds through the FIPS 203 `G(d || k)` expansion.
    #[inline]
    #[must_use]
    pub(crate) fn $name(rho: [u8; ML_KEM_SEED_SIZE], sigma: [u8; ML_KEM_SEED_SIZE]) -> [u8; ML_KEM_SHARED_SECRET_SIZE] {
      operations::ct_tests::diag_keygen_secret_noise_digest::<$k, $eta1_random_bytes, $dk_pke_bytes, $ek_bytes>(
        &rho, &sigma,
      )
    }
  };
}

/// Diagnostic digest for ML-KEM-768 prepared key generation with a fixed public matrix seed.
/// Production key generation derives `rho` and
/// `sigma` through the FIPS 203 `G(d || k)` expansion and takes `z` from its seed.
#[inline]
#[must_use]
pub(crate) fn diag_mlkem768_keygen_prepared_secret_noise_digest(
  rho: [u8; ML_KEM_SEED_SIZE],
  sigma: [u8; ML_KEM_SEED_SIZE],
  z: [u8; ML_KEM_SEED_SIZE],
) -> [u8; ML_KEM_SHARED_SECRET_SIZE] {
  operations::ct_tests::diag_keygen_prepared_secret_noise_digest::<3, 128, 1152, 1184, 2400>(&rho, &sigma, &z)
}

mlkem_diag_keygen_secret_noise!(diag_mlkem512_keygen_secret_noise_digest, 2, 192, 768, 800, "ML-KEM-512");

mlkem_diag_keygen_secret_noise!(
  diag_mlkem768_keygen_secret_noise_digest,
  3,
  128,
  1152,
  1184,
  "ML-KEM-768"
);

mlkem_diag_keygen_secret_noise!(
  diag_mlkem1024_keygen_secret_noise_digest,
  4,
  128,
  1536,
  1568,
  "ML-KEM-1024"
);

#[inline]
#[must_use]
pub(crate) fn diag_mlkem_ntt_input_digest(poly: [u16; 256]) -> u16 {
  operations::ct_tests::diag_ntt_input_digest(poly)
}

/// Diagnostic digest for the s390x z/Vector NTT kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
#[inline]
#[must_use]
pub(crate) unsafe fn diag_mlkem_s390x_ntt_input_digest(poly: [u16; 256]) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { operations::ct_tests::diag_s390x_ntt_input_digest(poly) }
}

#[inline]
#[must_use]
pub(crate) fn diag_mlkem_inverse_ntt_montgomery_product_input_digest(poly: [u16; 256]) -> u16 {
  operations::ct_tests::diag_inverse_ntt_montgomery_product_input_digest(poly)
}

/// Diagnostic digest for the s390x z/Vector inverse-NTT kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
#[inline]
#[must_use]
pub(crate) unsafe fn diag_mlkem_s390x_inverse_ntt_montgomery_product_input_digest(poly: [u16; 256]) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { operations::ct_tests::diag_s390x_inverse_ntt_montgomery_product_input_digest(poly) }
}

#[inline]
#[must_use]
pub(crate) fn diag_mlkem_multiply_ntts_add_assign_input_digest(a: [u16; 256], b: [u16; 256], acc: [u16; 256]) -> u16 {
  operations::ct_tests::diag_multiply_ntts_add_assign_input_digest(a, b, acc)
}

#[inline]
#[must_use]
pub(crate) fn diag_mlkem768_multiply_ntts_accumulate_input_digest(
  a: [[u16; 256]; 3],
  b: [[u16; 256]; 3],
  acc: [u16; 256],
) -> u16 {
  operations::ct_tests::diag_multiply_ntts_accumulate_k3_input_digest(a, b, acc)
}

#[inline]
#[must_use]
pub(crate) fn diag_mlkem1024_multiply_ntts_accumulate_input_digest(
  a: [[u16; 256]; 4],
  b: [[u16; 256]; 4],
  acc: [u16; 256],
) -> u16 {
  operations::ct_tests::diag_multiply_ntts_accumulate_k4_input_digest(a, b, acc)
}

#[inline]
#[must_use]
pub(crate) fn diag_mlkem_to_montgomery_product_domain_input_digest(poly: [u16; 256]) -> u16 {
  operations::ct_tests::diag_to_montgomery_product_domain_input_digest(poly)
}

#[inline]
#[must_use]
pub(crate) fn diag_mlkem_from_montgomery_product_domain_input_digest(poly: [u16; 256]) -> u16 {
  operations::ct_tests::diag_from_montgomery_product_domain_input_digest(poly)
}

/// Diagnostic digest for the s390x z/Vector product-domain conversion kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
#[inline]
#[must_use]
pub(crate) unsafe fn diag_mlkem_s390x_to_montgomery_product_domain_input_digest(poly: [u16; 256]) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { operations::ct_tests::diag_s390x_to_montgomery_product_domain_input_digest(poly) }
}

/// Diagnostic digest for the s390x z/Vector product-domain exit kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
#[inline]
#[must_use]
pub(crate) unsafe fn diag_mlkem_s390x_from_montgomery_product_domain_input_digest(poly: [u16; 256]) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { operations::ct_tests::diag_s390x_from_montgomery_product_domain_input_digest(poly) }
}

/// Diagnostic digest for the s390x z/Vector base-multiply accumulator kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
#[inline]
#[must_use]
pub(crate) unsafe fn diag_mlkem_s390x_multiply_ntts_add_assign_input_digest(
  a: [u16; 256],
  b: [u16; 256],
  acc: [u16; 256],
) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { operations::ct_tests::diag_s390x_multiply_ntts_add_assign_input_digest(a, b, acc) }
}

/// Diagnostic digest for the s390x z/Vector k=3 NTT dot-product kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
#[inline]
#[must_use]
pub(crate) unsafe fn diag_mlkem_s390x_multiply_ntts_accumulate_k3_input_digest(
  a: [[u16; 256]; 3],
  b: [[u16; 256]; 3],
  acc: [u16; 256],
) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { operations::ct_tests::diag_s390x_multiply_ntts_accumulate_k3_input_digest(a, b, acc) }
}

/// Diagnostic digest for the s390x z/Vector k=4 NTT dot-product kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
#[inline]
#[must_use]
pub(crate) unsafe fn diag_mlkem_s390x_multiply_ntts_accumulate_k4_input_digest(
  a: [[u16; 256]; 4],
  b: [[u16; 256]; 4],
  acc: [u16; 256],
) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { operations::ct_tests::diag_s390x_multiply_ntts_accumulate_k4_input_digest(a, b, acc) }
}

#[inline]
#[must_use]
pub(crate) fn diag_mlkem_compress_decompress_values_digest(values: [u16; 4]) -> u16 {
  operations::ct_tests::diag_compress_decompress_values_digest(values)
}

/// Diagnostic digest for the s390x z/Vector compress/decompress kernels.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
#[inline]
#[must_use]
pub(crate) unsafe fn diag_mlkem_s390x_compress_decompress_values_digest(values: [u16; 4]) -> u16 {
  // SAFETY: forwarded from this function's caller contract.
  unsafe { operations::ct_tests::diag_s390x_compress_decompress_values_digest(values) }
}
