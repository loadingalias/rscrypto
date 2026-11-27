use super::*;

pub(crate) fn diag_keygen_secret_noise_digest<
  const K: usize,
  const ETA1_RANDOM_BYTES: usize,
  const DK_PKE_BYTES: usize,
  const EK_BYTES: usize,
>(
  rho: &[u8; SEED_BYTES],
  sigma: &[u8; SEED_BYTES],
) -> [u8; HASH_BYTES] {
  let mut ek = [0u8; EK_BYTES];
  let mut dk_pke = [0u8; DK_PKE_BYTES];
  pke_keygen_expanded_into::<K, ETA1_RANDOM_BYTES, DK_PKE_BYTES, EK_BYTES>(rho, sigma, &mut ek, &mut dk_pke);
  let mut digest = [0u8; HASH_BYTES];

  for (i, byte) in ek.iter().chain(dk_pke.iter()).copied().enumerate() {
    digest[i & (HASH_BYTES - 1)] ^= byte;
  }

  ct::zeroize(&mut ek);
  ct::zeroize(&mut dk_pke);
  digest
}

/// Prepared key generation from a fixed public matrix seed, for timing evidence
/// that varies only the secret seeds.
pub(crate) fn diag_keygen_prepared_secret_noise_digest<
  const K: usize,
  const ETA1_RANDOM_BYTES: usize,
  const DK_PKE_BYTES: usize,
  const EK_BYTES: usize,
  const DK_BYTES: usize,
>(
  rho: &[u8; SEED_BYTES],
  sigma: &[u8; SEED_BYTES],
  z: &[u8; SEED_BYTES],
) -> [u8; HASH_BYTES] {
  let mut ek = [0u8; EK_BYTES];
  let mut dk = [0u8; DK_BYTES];
  let mut prepared = PreparedDecapsulationArithmetic::<K>::zeroed();
  keygen_prepared_expanded_into::<K, ETA1_RANDOM_BYTES, DK_PKE_BYTES, EK_BYTES, DK_BYTES>(
    rho,
    sigma,
    z,
    &mut ek,
    &mut dk,
    &mut prepared,
  );
  core::hint::black_box(&prepared);
  let mut digest = [0u8; HASH_BYTES];

  for (i, byte) in ek.iter().chain(dk.iter()).copied().enumerate() {
    digest[i & (HASH_BYTES - 1)] ^= byte;
  }

  ct::zeroize(&mut ek);
  ct::zeroize(&mut dk);
  digest
}

pub(crate) fn diag_ntt_input_digest(mut poly: Poly) -> u16 {
  ntt(&mut poly);
  let digest = diag_fold_poly(&poly);
  zeroize_poly(&mut poly);
  digest
}

/// Diagnostic digest for the s390x z/Vector NTT kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
pub(crate) unsafe fn diag_s390x_ntt_input_digest(mut poly: Poly) -> u16 {
  // SAFETY: Direct z/Vector diagnostic call because:
  // 1. The caller guarantees the s390x z/Vector facility is available.
  // 2. `poly` is a fixed 256-coefficient polynomial matching the kernel contract.
  // 3. This diagnostic root intentionally bypasses runtime dispatch so CT artifact generation scans
  //    the low-level s390x kernel itself.
  unsafe {
    s390x::ntt_vector(&mut poly);
  }
  let digest = diag_fold_poly(&poly);
  zeroize_poly(&mut poly);
  digest
}

pub(crate) fn diag_inverse_ntt_montgomery_product_input_digest(mut poly: Poly) -> u16 {
  inverse_ntt_montgomery_product(&mut poly);
  let digest = diag_fold_poly(&poly);
  zeroize_poly(&mut poly);
  digest
}

/// Diagnostic digest for the s390x z/Vector inverse-NTT kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
pub(crate) unsafe fn diag_s390x_inverse_ntt_montgomery_product_input_digest(mut poly: Poly) -> u16 {
  // SAFETY: Direct z/Vector diagnostic call because:
  // 1. The caller guarantees the s390x z/Vector facility is available.
  // 2. `poly` is a fixed 256-coefficient polynomial matching the kernel contract.
  // 3. `INV_NTT_PRODUCT_SCALE_MONT` is the public ML-KEM product-domain scale constant.
  // 4. This diagnostic root intentionally bypasses runtime dispatch so CT artifact generation scans
  //    the low-level s390x kernel itself.
  unsafe {
    s390x::inverse_ntt_vector(&mut poly, INV_NTT_PRODUCT_SCALE_MONT);
  }
  let digest = diag_fold_poly(&poly);
  zeroize_poly(&mut poly);
  digest
}

pub(crate) fn diag_multiply_ntts_add_assign_input_digest(a: Poly, b: Poly, mut acc: Poly) -> u16 {
  multiply_ntts_add_assign(&mut acc, &a, &b);
  let digest = diag_fold_poly(&acc);
  zeroize_poly(&mut acc);
  digest
}

pub(crate) fn diag_multiply_ntts_accumulate_k3_input_digest(
  mut a: PolyVec<3>,
  mut b: PolyVec<3>,
  mut acc: Poly,
) -> u16 {
  multiply_ntts_accumulate(&mut acc, &a, &b);
  let digest = diag_fold_poly(&acc);
  zeroize_polyvec(&mut a);
  zeroize_polyvec(&mut b);
  zeroize_poly(&mut acc);
  digest
}

pub(crate) fn diag_multiply_ntts_accumulate_k4_input_digest(
  mut a: PolyVec<4>,
  mut b: PolyVec<4>,
  mut acc: Poly,
) -> u16 {
  multiply_ntts_accumulate(&mut acc, &a, &b);
  let digest = diag_fold_poly(&acc);
  zeroize_polyvec(&mut a);
  zeroize_polyvec(&mut b);
  zeroize_poly(&mut acc);
  digest
}

pub(crate) fn diag_to_montgomery_product_domain_input_digest(mut poly: Poly) -> u16 {
  #[cfg(all(target_arch = "aarch64", not(miri), not(feature = "portable-only")))]
  super::tests::poly_to_montgomery_product_domain(&mut poly);
  #[cfg(not(all(target_arch = "aarch64", not(miri), not(feature = "portable-only"))))]
  poly_to_montgomery_product_domain(&mut poly);
  let digest = diag_fold_poly(&poly);
  zeroize_poly(&mut poly);
  digest
}

pub(crate) fn diag_from_montgomery_product_domain_input_digest(mut poly: Poly) -> u16 {
  poly_from_montgomery_product_domain(&mut poly);
  let digest = diag_fold_poly(&poly);
  zeroize_poly(&mut poly);
  digest
}

/// Diagnostic digest for the s390x z/Vector product-domain conversion kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
pub(crate) unsafe fn diag_s390x_to_montgomery_product_domain_input_digest(mut poly: Poly) -> u16 {
  // SAFETY: Direct z/Vector diagnostic call because:
  // 1. The caller guarantees the s390x z/Vector facility is available.
  // 2. `poly` is a fixed 256-coefficient polynomial matching the kernel contract.
  // 3. This diagnostic root intentionally bypasses runtime dispatch so CT artifact generation scans
  //    the low-level s390x product-domain conversion kernel itself.
  unsafe {
    s390x::to_montgomery_product_domain_vector(&mut poly);
  }
  let digest = diag_fold_poly(&poly);
  zeroize_poly(&mut poly);
  digest
}

/// Diagnostic digest for the s390x z/Vector product-domain exit kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
pub(crate) unsafe fn diag_s390x_from_montgomery_product_domain_input_digest(mut poly: Poly) -> u16 {
  // SAFETY: Direct z/Vector diagnostic call because:
  // 1. The caller guarantees the s390x z/Vector facility is available.
  // 2. `poly` is a fixed 256-coefficient polynomial matching the kernel contract.
  // 3. `MONT_R_SQUARED_MOD_Q` is the public ML-KEM conversion constant.
  // 4. This diagnostic root intentionally bypasses runtime dispatch so CT artifact generation scans
  //    the low-level s390x product-domain exit kernel itself.
  unsafe {
    s390x::from_montgomery_product_domain_vector(&mut poly);
  }
  let digest = diag_fold_poly(&poly);
  zeroize_poly(&mut poly);
  digest
}

/// Diagnostic digest for the s390x z/Vector base-multiply accumulator kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
pub(crate) unsafe fn diag_s390x_multiply_ntts_add_assign_input_digest(a: Poly, b: Poly, mut acc: Poly) -> u16 {
  // SAFETY: Direct z/Vector diagnostic call because:
  // 1. The caller guarantees the s390x z/Vector facility is available.
  // 2. `acc`, `a`, and `b` are fixed 256-coefficient polynomials matching the kernel contract.
  // 3. The borrowed inputs are stack-owned in this function and cannot alias `acc`.
  // 4. This diagnostic root intentionally bypasses runtime dispatch so CT artifact generation scans
  //    the low-level s390x kernel itself.
  unsafe {
    s390x::multiply_ntts_add_assign_vector(&mut acc, &a, &b);
  }
  let digest = diag_fold_poly(&acc);
  zeroize_poly(&mut acc);
  digest
}

/// Diagnostic digest for the s390x z/Vector k=3 NTT dot-product kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
pub(crate) unsafe fn diag_s390x_multiply_ntts_accumulate_k3_input_digest(
  mut a: PolyVec<3>,
  mut b: PolyVec<3>,
  mut acc: Poly,
) -> u16 {
  // SAFETY: Direct z/Vector diagnostic call because:
  // 1. The caller guarantees the s390x z/Vector facility is available.
  // 2. `acc`, `a`, and `b` contain fixed 256-coefficient polynomials matching the kernel contract.
  // 3. The borrowed inputs are stack-owned in this function and cannot alias `acc`.
  // 4. This diagnostic root intentionally bypasses runtime dispatch so CT artifact generation scans
  //    the low-level s390x kernel itself.
  unsafe {
    s390x::multiply_ntts_accumulate_k3_vector(&mut acc, [&a[0], &a[1], &a[2]], [&b[0], &b[1], &b[2]]);
  }
  let digest = diag_fold_poly(&acc);
  zeroize_polyvec(&mut a);
  zeroize_polyvec(&mut b);
  zeroize_poly(&mut acc);
  digest
}

/// Diagnostic digest for the s390x z/Vector k=4 NTT dot-product kernel.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
pub(crate) unsafe fn diag_s390x_multiply_ntts_accumulate_k4_input_digest(
  mut a: PolyVec<4>,
  mut b: PolyVec<4>,
  mut acc: Poly,
) -> u16 {
  // SAFETY: Direct z/Vector diagnostic call because:
  // 1. The caller guarantees the s390x z/Vector facility is available.
  // 2. `acc`, `a`, and `b` contain fixed 256-coefficient polynomials matching the kernel contract.
  // 3. The borrowed inputs are stack-owned in this function and cannot alias `acc`.
  // 4. This diagnostic root intentionally bypasses runtime dispatch so CT artifact generation scans
  //    the low-level s390x kernel itself.
  unsafe {
    s390x::multiply_ntts_accumulate_k4_vector(&mut acc, [&a[0], &a[1], &a[2], &a[3]], [&b[0], &b[1], &b[2], &b[3]]);
  }
  let digest = diag_fold_poly(&acc);
  zeroize_polyvec(&mut a);
  zeroize_polyvec(&mut b);
  zeroize_poly(&mut acc);
  digest
}

pub(crate) fn diag_compress_decompress_values_digest(values: [u16; 4]) -> u16 {
  let compressed = compress_values_4::<10>(values);
  let decompressed = decompress_values_4::<10>(compressed);
  let compressed_11 = compress_values_4::<11>(values);
  let decompressed_11 = decompress_values_4::<11>(compressed_11);
  let compressed_5 = compress_values_4::<5>(values);
  let decompressed_5 = decompress_values_4::<5>(compressed_5);

  let mut digest = 0u16;
  for value in compressed {
    digest = digest.rotate_left(3) ^ value;
  }
  for value in decompressed {
    digest = digest.rotate_left(5) ^ value;
  }
  for value in compressed_11 {
    digest = digest.rotate_left(7) ^ value;
  }
  for value in decompressed_11 {
    digest = digest.rotate_left(11) ^ value;
  }
  for value in compressed_5 {
    digest = digest.rotate_left(13) ^ value;
  }
  for value in decompressed_5 {
    digest = digest.rotate_left(2) ^ value;
  }
  digest
}

/// Diagnostic digest for the s390x z/Vector compress/decompress kernels.
///
/// # Safety
///
/// The caller must ensure the CPU supports the s390x z/Vector facility before
/// executing this function.
#[cfg(all(target_arch = "s390x", not(miri), not(feature = "portable-only")))]
pub(crate) unsafe fn diag_s390x_compress_decompress_values_digest(values: [u16; 4]) -> u16 {
  // SAFETY: Direct z/Vector diagnostic calls because:
  // 1. The caller guarantees the s390x z/Vector facility is available.
  // 2. `values` is exactly the four-coefficient shape required by the vector helpers.
  // 3. This diagnostic root intentionally bypasses runtime dispatch so CT artifact generation scans
  //    the low-level s390x compression kernels themselves.
  let compressed = unsafe { s390x::compress_values_4::<10>(values) };
  // SAFETY: Same z/Vector facility and fixed four-coefficient shape as above.
  let decompressed = unsafe { s390x::decompress_values_4::<10>(compressed) };
  // SAFETY: Same z/Vector facility and fixed four-coefficient shape as above.
  let compressed_11 = unsafe { s390x::compress_values_4::<11>(values) };
  // SAFETY: Same z/Vector facility and fixed four-coefficient shape as above.
  let decompressed_11 = unsafe { s390x::decompress_values_4::<11>(compressed_11) };
  // SAFETY: Same z/Vector facility and fixed four-coefficient shape as above.
  let compressed_5 = unsafe { s390x::compress_values_4::<5>(values) };
  // SAFETY: Same z/Vector facility and fixed four-coefficient shape as above.
  let decompressed_5 = unsafe { s390x::decompress_values_4::<5>(compressed_5) };

  let mut digest = 0u16;
  for value in compressed {
    digest = digest.rotate_left(3) ^ value;
  }
  for value in decompressed {
    digest = digest.rotate_left(5) ^ value;
  }
  for value in compressed_11 {
    digest = digest.rotate_left(7) ^ value;
  }
  for value in decompressed_11 {
    digest = digest.rotate_left(11) ^ value;
  }
  for value in compressed_5 {
    digest = digest.rotate_left(13) ^ value;
  }
  for value in decompressed_5 {
    digest = digest.rotate_left(2) ^ value;
  }
  digest
}

#[inline(never)]
fn diag_fold_poly(poly: &Poly) -> u16 {
  let mut acc = 0u16;
  for &coeff in poly {
    acc ^= coeff;
  }
  acc
}
