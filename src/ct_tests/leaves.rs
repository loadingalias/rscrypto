use super::operations::{STATUS_ERR, read_array};
use crate::aead::ct_tests::diag_aes_siv_cmac256_open_portable;
use core::ptr;

/// Retain the production eight-limb selector with its normal capability dispatch.
#[unsafe(no_mangle)]
#[inline(never)]
pub(crate) fn ct_entry_rsa_private_window_select(table: &[u64; 128], window: u8) -> [u64; 8] {
  crate::auth::rsa::ct_tests::diag_rsa_private_select_window_power_8(table, window)
}

/// Read a fixed-size `u16` input array by value from a C pointer.
///
/// # Safety
///
/// `ptr` must be valid for reads of `N` initialized `u16` values and aligned
/// for `u16`.
unsafe fn read_u16_array<const N: usize>(ptr: *const u16) -> Option<[u16; N]> {
  if ptr.is_null() {
    return None;
  }

  let mut out = [0u16; N];
  // SAFETY: Copies a fixed-size `u16` input array from FFI memory because:
  // 1. The caller contract requires `ptr` to be valid for reads of `N` initialized `u16` values.
  // 2. The caller contract requires `ptr` to be aligned for `u16`.
  // 3. `out` is stack-owned storage valid for writes of `N` `u16` values.
  // 4. Source and destination cannot overlap because `out` is newly allocated stack storage.
  unsafe { ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), N) };
  Some(out)
}

/// Read a fixed-size ML-KEM polynomial vector by value from a C pointer.
///
/// # Safety
///
/// `ptr` must be valid for reads of `K * 256` initialized `u16` values and
/// aligned for `u16`.
unsafe fn read_mlkem_polyvec<const K: usize>(ptr: *const u16) -> Option<[[u16; 256]; K]> {
  if ptr.is_null() {
    return None;
  }

  let mut out = [[0u16; 256]; K];
  let coefficient_count = K.strict_mul(256);
  // SAFETY: Copies a fixed-size ML-KEM polyvec from FFI memory because:
  // 1. `coefficient_count` is exactly `K * 256`; strict multiplication rejects an impossible
  //    destination layout instead of wrapping the copy length.
  // 2. The caller contract requires `ptr` to be valid for reads of `coefficient_count` initialized
  //    `u16` values.
  // 3. The caller contract requires `ptr` to be aligned for `u16`.
  // 4. `out` is stack-owned storage valid for writes of `coefficient_count` `u16` values.
  // 5. Source and destination cannot overlap because `out` is newly allocated stack storage.
  unsafe { ptr::copy_nonoverlapping(ptr, out.as_mut_ptr().cast::<u16>(), coefficient_count) };
  Some(out)
}

/// Run the ML-KEM forward-NTT diagnostic through the constant-time C ABI harness.
///
/// # Safety
///
/// When non-null, `poly` must be aligned for `u16`, valid for reads of 256 initialized
/// coefficients, and immutable while copied. A null pointer is rejected.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn ct_entry_mlkem_diag_ntt(poly: *const u16) -> u16 {
  // SAFETY: The function contract requires a non-null `poly` to be aligned for `u16` and valid for
  // reads of exactly 256 initialized coefficients that remain immutable while copied.
  let Some(poly) = (unsafe { read_u16_array::<256>(poly) }) else {
    return 0;
  };

  #[cfg(target_arch = "s390x")]
  {
    // SAFETY: s390x CT artifact and IBM Z diagnostic runs are selected only for z/Vector-capable
    // runners. This root intentionally bypasses runtime dispatch so the artifact scans the
    // low-level ML-KEM z/Vector NTT kernel.
    unsafe { crate::auth::mlkem::ct_tests::diag_mlkem_s390x_ntt_input_digest(poly) }
  }

  #[cfg(not(target_arch = "s390x"))]
  crate::auth::mlkem::ct_tests::diag_mlkem_ntt_input_digest(poly)
}

/// Run the ML-KEM inverse-NTT diagnostic through the constant-time C ABI harness.
///
/// # Safety
///
/// When non-null, `poly` must be aligned for `u16`, valid for reads of 256 initialized
/// coefficients, and immutable while copied. A null pointer is rejected.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn ct_entry_mlkem_diag_inverse_ntt(poly: *const u16) -> u16 {
  // SAFETY: The function contract requires a non-null `poly` to be aligned for `u16` and valid for
  // reads of exactly 256 initialized coefficients that remain immutable while copied.
  let Some(poly) = (unsafe { read_u16_array::<256>(poly) }) else {
    return 0;
  };

  #[cfg(target_arch = "s390x")]
  {
    // SAFETY: s390x CT artifact and IBM Z diagnostic runs are selected only for z/Vector-capable
    // runners. This root intentionally bypasses runtime dispatch so the artifact scans the
    // low-level ML-KEM z/Vector inverse-NTT kernel.
    unsafe { crate::auth::mlkem::ct_tests::diag_mlkem_s390x_inverse_ntt_montgomery_product_input_digest(poly) }
  }

  #[cfg(not(target_arch = "s390x"))]
  crate::auth::mlkem::ct_tests::diag_mlkem_inverse_ntt_montgomery_product_input_digest(poly)
}

/// Run the ML-KEM to-product-domain diagnostic through the constant-time C ABI harness.
///
/// # Safety
///
/// When non-null, `poly` must be aligned for `u16`, valid for reads of 256 initialized
/// coefficients, and immutable while copied. A null pointer is rejected.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn ct_entry_mlkem_diag_to_product_domain(poly: *const u16) -> u16 {
  // SAFETY: The function contract requires an aligned, initialized, immutable 256-coefficient
  // source whenever poly is non-null; read_u16_array rejects null before copying.
  let Some(poly) = (unsafe { read_u16_array::<256>(poly) }) else {
    return 0;
  };

  #[cfg(target_arch = "s390x")]
  {
    // SAFETY: s390x CT artifact and IBM Z diagnostic runs are selected only for z/Vector-capable
    // runners. This root intentionally bypasses runtime dispatch so the artifact scans the
    // low-level ML-KEM z/Vector product-domain conversion kernel.
    unsafe { crate::auth::mlkem::ct_tests::diag_mlkem_s390x_to_montgomery_product_domain_input_digest(poly) }
  }

  #[cfg(not(target_arch = "s390x"))]
  crate::auth::mlkem::ct_tests::diag_mlkem_to_montgomery_product_domain_input_digest(poly)
}

/// Run the ML-KEM from-product-domain diagnostic through the constant-time C ABI harness.
///
/// # Safety
///
/// When non-null, `poly` must be aligned for `u16`, valid for reads of 256 initialized
/// coefficients, and immutable while copied. A null pointer is rejected.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn ct_entry_mlkem_diag_from_product_domain(poly: *const u16) -> u16 {
  // SAFETY: The function contract requires an aligned, initialized, immutable 256-coefficient
  // source whenever poly is non-null; read_u16_array rejects null before copying.
  let Some(poly) = (unsafe { read_u16_array::<256>(poly) }) else {
    return 0;
  };

  #[cfg(target_arch = "s390x")]
  {
    // SAFETY: s390x CT artifact and IBM Z diagnostic runs are selected only for z/Vector-capable
    // runners. This root intentionally bypasses runtime dispatch so the artifact scans the
    // low-level ML-KEM z/Vector product-domain exit kernel.
    unsafe { crate::auth::mlkem::ct_tests::diag_mlkem_s390x_from_montgomery_product_domain_input_digest(poly) }
  }

  #[cfg(not(target_arch = "s390x"))]
  crate::auth::mlkem::ct_tests::diag_mlkem_from_montgomery_product_domain_input_digest(poly)
}

/// Run the ML-KEM NTT multiply/add diagnostic through the constant-time C ABI harness.
///
/// # Safety
///
/// When non-null, each of `a`, `b`, and `acc` must be aligned for `u16`, valid for reads
/// of 256 initialized coefficients, and immutable while copied. Null pointers are rejected.
/// The three read-only ranges may overlap.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn ct_entry_mlkem_diag_multiply_ntts_add_assign(
  a: *const u16,
  b: *const u16,
  acc: *const u16,
) -> u16 {
  // SAFETY: The function contract requires an aligned, initialized, immutable 256-coefficient
  // source whenever a is non-null; read_u16_array rejects null before copying.
  let Some(a) = (unsafe { read_u16_array::<256>(a) }) else {
    return 0;
  };
  // SAFETY: The function contract provides the same guarantees for b. Its range may overlap a
  // because both operations only read and each result is independently owned.
  let Some(b) = (unsafe { read_u16_array::<256>(b) }) else {
    return 0;
  };
  // SAFETY: The function contract provides the same guarantees for acc. Its range may overlap a
  // or b because all three operations only read and each result is independently owned.
  let Some(acc) = (unsafe { read_u16_array::<256>(acc) }) else {
    return 0;
  };

  #[cfg(target_arch = "s390x")]
  {
    // SAFETY: s390x CT artifact and IBM Z diagnostic runs are selected only for z/Vector-capable
    // runners. This root intentionally bypasses runtime dispatch so the artifact scans the
    // low-level ML-KEM z/Vector base-multiply accumulator kernel.
    unsafe { crate::auth::mlkem::ct_tests::diag_mlkem_s390x_multiply_ntts_add_assign_input_digest(a, b, acc) }
  }

  #[cfg(not(target_arch = "s390x"))]
  crate::auth::mlkem::ct_tests::diag_mlkem_multiply_ntts_add_assign_input_digest(a, b, acc)
}

/// Run the ML-KEM K=3 NTT accumulation diagnostic through the constant-time C ABI harness.
///
/// # Safety
///
/// When non-null, `a` and `b` must each be aligned for `u16`, valid for reads of 768
/// initialized coefficients, and immutable while copied. `acc` has the same obligations for
/// 256 coefficients. Null pointers are rejected. The three read-only ranges may overlap.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn ct_entry_mlkem_diag_multiply_ntts_accumulate_k3(
  a: *const u16,
  b: *const u16,
  acc: *const u16,
) -> u16 {
  // SAFETY: The function contract requires an aligned, initialized, immutable 768-coefficient
  // source whenever a is non-null; read_mlkem_polyvec rejects null before copying.
  let Some(a) = (unsafe { read_mlkem_polyvec::<3>(a) }) else {
    return 0;
  };
  // SAFETY: The function contract provides the same guarantees for b. Its range may overlap a
  // because both operations only read and each result is independently owned.
  let Some(b) = (unsafe { read_mlkem_polyvec::<3>(b) }) else {
    return 0;
  };
  // SAFETY: The function contract requires an aligned, initialized, immutable 256-coefficient
  // acc source. Its range may overlap a or b because all three operations only read.
  let Some(acc) = (unsafe { read_u16_array::<256>(acc) }) else {
    return 0;
  };

  #[cfg(target_arch = "s390x")]
  {
    // SAFETY: s390x CT artifact and IBM Z diagnostic runs are selected only for z/Vector-capable
    // runners, and this root intentionally scans the direct k=3 dot-product kernel.
    unsafe { crate::auth::mlkem::ct_tests::diag_mlkem_s390x_multiply_ntts_accumulate_k3_input_digest(a, b, acc) }
  }

  #[cfg(not(target_arch = "s390x"))]
  crate::auth::mlkem::ct_tests::diag_mlkem768_multiply_ntts_accumulate_input_digest(a, b, acc)
}

/// Run the ML-KEM K=4 NTT accumulation diagnostic through the constant-time C ABI harness.
///
/// # Safety
///
/// When non-null, `a` and `b` must each be aligned for `u16`, valid for reads of 1,024
/// initialized coefficients, and immutable while copied. `acc` has the same obligations for
/// 256 coefficients. Null pointers are rejected. The three read-only ranges may overlap.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn ct_entry_mlkem_diag_multiply_ntts_accumulate_k4(
  a: *const u16,
  b: *const u16,
  acc: *const u16,
) -> u16 {
  // SAFETY: The function contract requires an aligned, initialized, immutable 1,024-coefficient
  // source whenever a is non-null; read_mlkem_polyvec rejects null before copying.
  let Some(a) = (unsafe { read_mlkem_polyvec::<4>(a) }) else {
    return 0;
  };
  // SAFETY: The function contract provides the same guarantees for b. Its range may overlap a
  // because both operations only read and each result is independently owned.
  let Some(b) = (unsafe { read_mlkem_polyvec::<4>(b) }) else {
    return 0;
  };
  // SAFETY: The function contract requires an aligned, initialized, immutable 256-coefficient
  // acc source. Its range may overlap a or b because all three operations only read.
  let Some(acc) = (unsafe { read_u16_array::<256>(acc) }) else {
    return 0;
  };

  #[cfg(target_arch = "s390x")]
  {
    // SAFETY: s390x CT artifact and IBM Z diagnostic runs are selected only for z/Vector-capable
    // runners, and this root intentionally scans the direct k=4 dot-product kernel.
    unsafe { crate::auth::mlkem::ct_tests::diag_mlkem_s390x_multiply_ntts_accumulate_k4_input_digest(a, b, acc) }
  }

  #[cfg(not(target_arch = "s390x"))]
  crate::auth::mlkem::ct_tests::diag_mlkem1024_multiply_ntts_accumulate_input_digest(a, b, acc)
}

/// Run the ML-KEM compression/decompression diagnostic through the constant-time C ABI harness.
///
/// # Safety
///
/// When non-null, `values` must be aligned for `u16`, valid for reads of four initialized
/// coefficients, and immutable while copied. A null pointer is rejected.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn ct_entry_mlkem_diag_compress_decompress(values: *const u16) -> u16 {
  // SAFETY: The function contract requires an aligned, initialized, immutable four-coefficient
  // source whenever values is non-null; read_u16_array rejects null before copying.
  let Some(values) = (unsafe { read_u16_array::<4>(values) }) else {
    return 0;
  };

  #[cfg(target_arch = "s390x")]
  {
    // SAFETY: s390x CT artifact and IBM Z diagnostic runs are selected only for z/Vector-capable
    // runners. This root intentionally bypasses runtime dispatch so the artifact scans the
    // low-level ML-KEM z/Vector compress/decompress kernels.
    unsafe { crate::auth::mlkem::ct_tests::diag_mlkem_s390x_compress_decompress_values_digest(values) }
  }

  #[cfg(not(target_arch = "s390x"))]
  crate::auth::mlkem::ct_tests::diag_mlkem_compress_decompress_values_digest(values)
}

/// Fixed-shape forced-portable AES-SIV-CMAC-256 complete-open harness.
///
/// # Safety
///
/// `key`, `nonce`, `aad`, `ciphertext`, and `tag` must reference 32, 16, 32, 48, and 16
/// initialized readable bytes respectively. Null pointers are rejected.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn ct_entry_aes_siv_cmac256_open_portable(
  key: *const u8,
  nonce: *const u8,
  aad: *const u8,
  ciphertext: *const u8,
  tag: *const u8,
) -> u8 {
  // SAFETY: Each read has the exact fixed-size validity requirement documented above.
  let Some(key) = (unsafe { read_array::<32>(key) }) else {
    return STATUS_ERR;
  };
  // SAFETY: Each read has the exact fixed-size validity requirement documented above.
  let Some(nonce) = (unsafe { read_array::<16>(nonce) }) else {
    return STATUS_ERR;
  };
  // SAFETY: Each read has the exact fixed-size validity requirement documented above.
  let Some(aad) = (unsafe { read_array::<32>(aad) }) else {
    return STATUS_ERR;
  };
  // SAFETY: Each read has the exact fixed-size validity requirement documented above.
  let Some(ciphertext) = (unsafe { read_array::<48>(ciphertext) }) else {
    return STATUS_ERR;
  };
  // SAFETY: Each read has the exact fixed-size validity requirement documented above.
  let Some(tag) = (unsafe { read_array::<16>(tag) }) else {
    return STATUS_ERR;
  };

  diag_aes_siv_cmac256_open_portable(&key, &nonce, &aad, &ciphertext, &tag)
}

/// Evaluate the bounded same-width RSA private-component validation leaf through the CT harness.
///
/// Returns 1 when `component` is canonical, odd, below `upper_bound`, and distinct from `other`,
/// and 0 otherwise. All three operands are 32-byte private values. A null pointer returns 0.
///
/// # Safety
///
/// - When non-null, `component`, `upper_bound`, and `other` must each be valid for reads of 32
///   initialized bytes and must not be mutated while copied. Null pointers are rejected.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn ct_entry_rsa_private_component_validation_32(
  component: *const u8,
  upper_bound: *const u8,
  other: *const u8,
) -> u8 {
  // SAFETY: The function contract requires non-null component to expose 32 initialized, readable
  // bytes that remain immutable while read_array copies them into owned storage.
  let Some(component) = (unsafe { read_array::<32>(component) }) else {
    return STATUS_ERR;
  };
  // SAFETY: The function contract requires non-null upper_bound to expose 32 initialized, readable
  // bytes that remain immutable while read_array copies them into owned storage.
  let Some(upper_bound) = (unsafe { read_array::<32>(upper_bound) }) else {
    return STATUS_ERR;
  };
  // SAFETY: The function contract requires non-null other to expose 32 initialized, readable bytes
  // that remain immutable while read_array copies them into owned storage.
  let Some(other) = (unsafe { read_array::<32>(other) }) else {
    return STATUS_ERR;
  };

  crate::auth::rsa::ct_tests::diag_rsa_private_component_validation_32(&component, &upper_bound, &other)
}
