//! Shared CRC kernel names.

/// Reference (bitwise) kernel name - canonical implementation for verification.
pub(in crate::checksum) const REFERENCE: &str = "reference/bitwise";

/// Portable fallback kernel name (used by all CRC widths).
#[cfg(any(feature = "crc32", feature = "crc64"))]
pub(in crate::checksum) const PORTABLE_SLICE16: &str = "portable/slice16";

/// Portable slice-by-8 kernel name.
#[cfg(any(feature = "crc16", feature = "crc24"))]
pub(in crate::checksum) const PORTABLE_SLICE8: &str = "portable/slice8";
