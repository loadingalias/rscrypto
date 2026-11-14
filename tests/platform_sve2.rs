//! A CI row that must run the SVE2 kernels reports their capabilities.
//!
//! The SVE2 kernel tests return early on a host without SVE2, so a runner or
//! detection change could stop them in silence. The AArch64 Linux CI row sets
//! `RSCRYPTO_REQUIRE_SVE2`, and on it this test fails unless detection reports
//! the capabilities of those kernels and AES-GCM selects its SVE2 backend.

#![cfg(all(target_arch = "aarch64", feature = "std", not(feature = "portable-only"), not(miri)))]

use rscrypto::platform::{self, caps::aarch64};

#[test]
fn a_required_sve2_host_reports_and_selects_sve2() {
  if std::env::var_os("RSCRYPTO_REQUIRE_SVE2").is_none() {
    return;
  }
  let required = aarch64::SVE2
    .union(aarch64::SVE2_AES)
    .union(aarch64::SVE2_PMULL)
    .union(aarch64::SVE2_SHA3);
  let caps = platform::caps();
  assert!(
    caps.has(required),
    "RSCRYPTO_REQUIRE_SVE2 is set, but detection reports {:?}",
    caps.feature_names().collect::<Vec<_>>()
  );
  #[cfg(feature = "aes-gcm")]
  assert_eq!(
    rscrypto::aead::introspect::aes256gcm_backend(),
    "aarch64/sve2+aes+pmull",
    "AES-GCM selects its SVE2 backend"
  );
}
