//! CRC64 variants.
//!
//! CRC64 is commonly used in storage and compression. This module provides:
//! - [`Crc64`]: XZ / ECMA polynomial with reflected processing
//! - [`Crc64Nvme`]: NVMe storage CRC64

pub(crate) mod nvme;
pub(crate) mod xz;

pub use nvme::Crc64Nvme;
pub use xz::Crc64;

/// Returns the CRC64/XZ backend this build will use on the current machine.
#[doc(hidden)]
#[inline]
#[must_use]
pub fn selected_backend() -> &'static str {
  xz::selected_backend()
}

/// Returns the CRC64/NVME backend this build will use on the current machine.
#[doc(hidden)]
#[inline]
#[must_use]
pub fn selected_backend_nvme() -> &'static str {
  nvme::selected_backend()
}
