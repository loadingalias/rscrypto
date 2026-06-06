//! Shared Edwards25519 arithmetic used by both Ed25519 and X25519.
//!
//! X25519 public-key derivation is fixed-base scalar multiplication on the
//! same Curve25519 basepoint. Reuse the Ed25519 basepoint machinery instead of
//! maintaining a second precompute stack for the identical scalar-mul problem.

#[allow(dead_code)]
#[path = "ed25519/constants.rs"]
pub(crate) mod constants;
#[allow(dead_code)]
#[path = "ed25519/field.rs"]
pub(crate) mod field;
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
#[path = "ed25519/field_avx2.rs"]
pub(crate) mod field_avx2;
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
#[path = "ed25519/field_ifma.rs"]
pub(crate) mod field_ifma;
#[allow(dead_code)]
#[path = "ed25519/point.rs"]
pub(crate) mod point;
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
#[path = "ed25519/point_avx2.rs"]
pub(crate) mod point_avx2;
#[allow(dead_code)]
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
#[must_use]
#[allow(dead_code)]
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
