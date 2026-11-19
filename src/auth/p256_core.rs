//! Private native P-256 dispatch boundary shared by ECDSA and ECDH.

#[cfg(feature = "p256-ecdh")]
use super::p256_portable::PublicPoint;

/// Parse and validate one canonical uncompressed SEC1 P-256 public point.
#[cfg(feature = "p256-ecdh")]
pub(super) fn public_point_from_sec1(bytes: &[u8]) -> Option<PublicPoint> {
  #[cfg(all(
    not(feature = "portable-only"),
    not(miri),
    any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    )
  ))]
  {
    let words = super::p256_portable::parse_sec1_words(bytes)?;
    super::p256_platform::p256_public_point(&words)
  }

  #[cfg(not(all(
    not(feature = "portable-only"),
    not(miri),
    any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    )
  )))]
  {
    PublicPoint::from_sec1_bytes(bytes)
  }
}

/// Multiply the P-256 generator by a validated nonzero scalar.
///
/// The result contains little-endian canonical affine `x || y` limbs. Native
/// fixed-base assembly is selected only on the ABIs where its complete proof
/// boundary is enabled. `portable-only` and Miri use the extracted Rust
/// authority on these same targets; other targets retain ECDSA's generic Rust
/// implementation.
#[cfg(all(
  any(
    feature = "ecdsa-p256",
    all(feature = "p256-ecdh", not(feature = "portable-only"), not(miri))
  ),
  any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  )
))]
pub(super) fn scalar_mul_generator_words(scalar: &[u64; 4]) -> [u64; 8] {
  #[cfg(all(
    not(feature = "portable-only"),
    not(miri),
    any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    )
  ))]
  {
    super::p256_platform::p256_scalarmulbase_generator(scalar)
  }

  #[cfg(not(all(
    not(feature = "portable-only"),
    not(miri),
    any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
    )
  )))]
  {
    super::p256_portable::scalar_mul_generator_words(scalar)
  }
}

/// Multiply a validated affine P-256 point by a validated nonzero scalar.
///
/// Inputs and output contain little-endian canonical affine limbs.
#[cfg(all(
  feature = "p256-ecdh",
  not(feature = "portable-only"),
  not(miri),
  any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  )
))]
pub(super) fn scalar_mul_words(scalar: &[u64; 4], point: &[u64; 8]) -> [u64; 8] {
  super::p256_platform::p256_scalarmul(scalar, point)
}
