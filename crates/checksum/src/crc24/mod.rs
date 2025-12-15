//! CRC24 variants.
//!
//! This module currently provides CRC24/OpenPGP.

mod openpgp;

pub use openpgp::Crc24;

/// Returns the CRC24/OpenPGP backend this build will use on the current machine.
#[doc(hidden)]
#[inline]
#[must_use]
pub fn selected_backend() -> &'static str {
  openpgp::selected_backend()
}
