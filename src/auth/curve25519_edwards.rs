//! Shared Edwards25519 arithmetic used by both Ed25519 and X25519.
//!
//! X25519 public-key derivation is fixed-base scalar multiplication on the
//! same Curve25519 basepoint. Reuse the Ed25519 basepoint machinery instead of
//! maintaining a second precompute stack for the identical scalar-mul problem.

#[cfg(feature = "ed25519")]
#[path = "ed25519/constants.rs"]
pub(crate) mod constants;
#[path = "ed25519/field.rs"]
pub(crate) mod field;
#[cfg(target_arch = "x86_64")]
#[path = "ed25519/field_avx2.rs"]
pub(crate) mod field_avx2;
#[cfg(target_arch = "x86_64")]
#[path = "ed25519/field_ifma.rs"]
pub(crate) mod field_ifma;
#[path = "ed25519/point.rs"]
pub(crate) mod point;
#[cfg(target_arch = "x86_64")]
#[path = "ed25519/point_avx2.rs"]
pub(crate) mod point_avx2;
#[cfg(feature = "ed25519")]
#[path = "ed25519/scalar.rs"]
pub(crate) mod scalar;

#[cfg(all(feature = "diag", feature = "ed25519"))]
pub use point::diag_select_basepoint_cached_limb_digest as diag_ed25519_select_basepoint_cached_limb_digest;
#[cfg(all(feature = "diag", feature = "ed25519", target_arch = "x86_64"))]
pub use point_avx2::{
  diag_select_basepoint_cached_avx2_limb_digest as diag_ed25519_select_basepoint_cached_avx2_limb_digest,
  diag_select_basepoint_cached_ifma_limb_digest as diag_ed25519_select_basepoint_cached_ifma_limb_digest,
};

/// Dispatch `[s]B` (fixed-base scalar mul) to the fastest validated CT path.
#[cfg_attr(
  all(
    target_arch = "x86_64",
    target_os = "linux",
    not(any(test, miri, feature = "portable-only"))
  ),
  expect(
    dead_code,
    reason = "x86_64 Linux library builds use the assembly fixed-base entry points"
  )
)]
#[must_use]
pub(crate) fn basepoint_mul_dispatch(scalar_bytes: &[u8; 32]) -> point::ExtendedPoint {
  #[cfg(target_arch = "x86_64")]
  {
    let caps = crate::platform::caps();
    if caps.has(crate::platform::caps::x86::AVX512IFMA)
      && caps.has(crate::platform::caps::x86::AVX512VL)
      && caps.has(crate::platform::caps::x86::AVX2)
    {
      // SAFETY: AVX-512 IFMA + VL + AVX2 were confirmed by runtime detection.
      return unsafe { point_avx2::scalar_mul_basepoint_ifma(scalar_bytes) };
    }
    if caps.has(crate::platform::caps::x86::AVX2) {
      // SAFETY: AVX2 was confirmed by runtime detection.
      return unsafe { point_avx2::scalar_mul_basepoint_avx2(scalar_bytes) };
    }
  }

  point::ExtendedPoint::scalar_mul_basepoint(scalar_bytes)
}

/// Decompose a scalar encoding into signed radix-16 digits in `[-8, 8]`.
#[must_use]
fn scalar_radix_16(bytes: &[u8; 32]) -> [i8; 64] {
  debug_assert!(bytes[31] <= 127);

  let mut digits = [0i8; 64];
  for (index, byte) in bytes.iter().copied().enumerate() {
    let low = index.strict_mul(2);
    let high = low.strict_add(1);
    digits[low] = i8::from_ne_bytes([byte & 0x0F]);
    digits[high] = i8::from_ne_bytes([(byte >> 4) & 0x0F]);
  }

  for index in 0usize..63 {
    let next = index.strict_add(1);
    let carry = digits[index].strict_add(8) >> 4;
    digits[index] = digits[index].strict_sub(carry << 4);
    digits[next] = digits[next].strict_add(carry);
  }

  digits
}
