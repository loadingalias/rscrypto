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

/// Dispatch `[s]B` (fixed-base scalar mul) to the fastest available path.
#[must_use]
pub(crate) fn basepoint_mul_dispatch(scalar_bytes: &[u8; 32]) -> point::ExtendedPoint {
  #[cfg(target_arch = "x86_64")]
  {
    let caps = crate::platform::caps();
    if caps.has(crate::platform::caps::x86::AVX512IFMA)
      && caps.has(crate::platform::caps::x86::AVX512VL)
      && caps.has(crate::platform::caps::x86::AVX2)
    {
      // SAFETY: AVX-512 IFMA + VL + AVX2 confirmed by runtime detection.
      return unsafe { point_avx2::scalar_mul_basepoint_ifma(scalar_bytes) };
    }
    if caps.has(crate::platform::caps::x86::AVX2) {
      // SAFETY: AVX2 confirmed by runtime detection.
      return unsafe { point_avx2::scalar_mul_basepoint_avx2(scalar_bytes) };
    }
  }
  point::ExtendedPoint::scalar_mul_basepoint(scalar_bytes)
}
