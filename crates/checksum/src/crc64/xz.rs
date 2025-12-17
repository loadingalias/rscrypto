//! CRC64/XZ checksum (ECMA polynomial with reflected processing).
//!
//! Parameters (CRC Catalogue):
//! - width: 64
//! - poly: 0x42F0E1EBA9EA3693 (reflected: 0xC96C5795D7870F42)
//! - init: 0xFFFF_FFFF_FFFF_FFFF
//! - refin/refout: true
//! - xorout: 0xFFFF_FFFF_FFFF_FFFF
//!
//! # Usage
//!
//! ```
//! use checksum::Crc64;
//!
//! let crc = Crc64::checksum(b"123456789");
//! assert_eq!(crc, 0x995DC9BBDF1939FA);
//! ```

use traits::{Checksum, ChecksumCombine};

#[cfg(not(feature = "no-tables"))]
use crate::constants::crc64::TABLES;
#[cfg(not(feature = "no-tables"))]
macro_rules! table {
  ($idx:expr) => {
    TABLES.0[$idx]
  };
}

/// CRC64/XZ checksum.
///
/// This struct implements streaming CRC64 computation. CRC64/XZ uses the
/// ECMA-182 polynomial and is used by XZ compression and PostgreSQL.
#[derive(Clone, Debug)]
pub struct Crc64 {
  /// Current CRC state (inverted - XOR applied on finalize).
  state: u64,
  /// Initial value for reset.
  initial: u64,
}

impl Crc64 {
  /// Initial value for CRC64/XZ (all ones).
  const INIT: u64 = 0xFFFF_FFFF_FFFF_FFFF;
  const XOR_OUT: u64 = 0xFFFF_FFFF_FFFF_FFFF;

  #[inline]
  #[must_use]
  pub const fn new() -> Self {
    Self {
      state: Self::INIT,
      initial: Self::INIT,
    }
  }

  /// Create a new hasher that will resume from a previous CRC.
  #[inline]
  #[must_use]
  pub const fn resume(crc: u64) -> Self {
    Self {
      state: crc ^ Self::XOR_OUT,
      initial: crc ^ Self::XOR_OUT,
    }
  }

  /// Compute CRC64/XZ of `data` in one shot.
  #[inline]
  #[must_use]
  pub fn checksum(data: &[u8]) -> u64 {
    dispatch(Self::INIT, data) ^ Self::XOR_OUT
  }

  #[inline]
  pub fn update(&mut self, data: &[u8]) {
    self.state = dispatch(self.state, data);
  }

  #[inline]
  #[must_use]
  pub const fn finalize(&self) -> u64 {
    self.state ^ Self::XOR_OUT
  }

  #[inline]
  pub fn reset(&mut self) {
    self.state = self.initial;
  }

  #[inline]
  #[must_use]
  pub const fn state(&self) -> u64 {
    self.finalize()
  }

  /// Combine two CRC64/XZ values: `crc(A || B)` from `crc(A)`, `crc(B)`, `len(B)`.
  #[inline]
  #[must_use]
  pub fn combine(crc_a: u64, crc_b: u64, len_b: usize) -> u64 {
    crate::combine::crc64_combine(crc_a, crc_b, len_b)
  }
}

impl Default for Crc64 {
  #[inline]
  fn default() -> Self {
    Self::new()
  }
}

impl Checksum for Crc64 {
  const OUTPUT_SIZE: usize = 8;
  type Output = u64;

  #[inline]
  fn new() -> Self {
    Crc64::new()
  }

  #[inline]
  fn with_initial(initial: Self::Output) -> Self {
    Self {
      state: initial ^ Self::XOR_OUT,
      initial: initial ^ Self::XOR_OUT,
    }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    Crc64::update(self, data);
  }

  #[inline]
  fn finalize(&self) -> Self::Output {
    Crc64::finalize(self)
  }

  #[inline]
  fn reset(&mut self) {
    Crc64::reset(self);
  }

  #[inline]
  fn checksum(data: &[u8]) -> Self::Output {
    Crc64::checksum(data)
  }
}

impl ChecksumCombine for Crc64 {
  #[inline]
  fn combine(crc_a: Self::Output, crc_b: Self::Output, len_b: usize) -> Self::Output {
    Crc64::combine(crc_a, crc_b, len_b)
  }
}

#[inline]
#[must_use]
pub(super) fn selected_backend() -> &'static str {
  #[cfg(all(feature = "std", target_arch = "x86_64"))]
  {
    use std::sync::OnceLock;
    static BACKEND: OnceLock<&'static str> = OnceLock::new();
    BACKEND.get_or_init(|| {
      if std::arch::is_x86_feature_detected!("vpclmulqdq") && std::arch::is_x86_feature_detected!("avx512f") {
        "x86_64/vpclmulqdq"
      } else if std::arch::is_x86_feature_detected!("pclmulqdq") {
        "x86_64/pclmulqdq"
      } else {
        "portable/slicing-by-8"
      }
    })
  }

  #[cfg(all(feature = "std", target_arch = "aarch64"))]
  {
    use std::sync::OnceLock;
    static BACKEND: OnceLock<&'static str> = OnceLock::new();
    BACKEND.get_or_init(|| {
      if std::arch::is_aarch64_feature_detected!("aes") {
        if std::arch::is_aarch64_feature_detected!("sha3") {
          "aarch64/pmull+eor3"
        } else {
          "aarch64/pmull"
        }
      } else {
        "portable/slicing-by-8"
      }
    })
  }

  #[cfg(all(
    feature = "no-tables",
    not(all(feature = "std", target_arch = "x86_64")),
    not(all(feature = "std", target_arch = "aarch64"))
  ))]
  {
    "portable/bitwise"
  }

  #[cfg(all(
    not(feature = "no-tables"),
    not(all(feature = "std", target_arch = "x86_64")),
    not(all(feature = "std", target_arch = "aarch64"))
  ))]
  {
    "portable/slicing-by-8"
  }
}

#[cfg(feature = "std")]
impl std::io::Write for Crc64 {
  #[inline]
  fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
    self.update(buf);
    Ok(buf.len())
  }

  #[inline]
  fn flush(&mut self) -> std::io::Result<()> {
    Ok(())
  }
}

#[inline]
fn dispatch(crc: u64, data: &[u8]) -> u64 {
  // Miri cannot interpret SIMD intrinsics, so always use portable code.
  #[cfg(miri)]
  return compute_portable(crc, data);

  // Tier 1: Compile-time dispatch (when target features are known).
  #[allow(unreachable_code)]
  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "vpclmulqdq",
    target_feature = "avx512f",
    target_feature = "pclmulqdq"
  ))]
  return crate::simd::x86_64::vpclmul::compute_vpclmul_crc64_xz_enabled(crc, data);

  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "pclmulqdq",
    target_feature = "ssse3",
    not(target_feature = "vpclmulqdq")
  ))]
  return crate::simd::x86_64::pclmul::compute_pclmul_crc64_xz_enabled(crc, data);

  #[cfg(all(target_arch = "aarch64", target_feature = "aes", target_feature = "sha3"))]
  return crate::simd::aarch64::pmull::compute_pmull_crc64_xz_eor3_enabled(crc, data);

  #[cfg(all(target_arch = "aarch64", target_feature = "aes", not(target_feature = "sha3")))]
  return crate::simd::aarch64::pmull::compute_pmull_crc64_xz_enabled(crc, data);

  // Tier 2: Runtime dispatch (std only).
  // NOTE: Check for PCLMULQDQ (not just VPCLMULQDQ) to ensure runtime detection
  // kicks in when compile-time features aren't sufficient.
  #[cfg(all(
    feature = "std",
    target_arch = "x86_64",
    not(all(target_feature = "pclmulqdq", target_feature = "ssse3"))
  ))]
  {
    use std::sync::OnceLock;
    static DISPATCH: OnceLock<fn(u64, &[u8]) -> u64> = OnceLock::new();
    let f = DISPATCH.get_or_init(crate::simd::x86_64::detect_crc64_xz_best);
    return f(crc, data);
  }

  #[cfg(all(feature = "std", target_arch = "aarch64", not(target_feature = "aes")))]
  {
    use std::sync::OnceLock;
    static DISPATCH: OnceLock<fn(u64, &[u8]) -> u64> = OnceLock::new();
    let f = DISPATCH.get_or_init(crate::simd::aarch64::detect_crc64_xz_best);
    return f(crc, data);
  }

  // Tier 3: Portable fallback.
  #[allow(unreachable_code)]
  compute_portable(crc, data)
}

/// Compute CRC64/XZ over `data`, returning the updated *raw* CRC state.
#[inline]
pub(crate) fn compute_portable(crc: u64, data: &[u8]) -> u64 {
  #[cfg(feature = "no-tables")]
  {
    let mut crc = crc;
    for &b in data {
      crc = compute_byte(crc, b);
    }
    crc
  }

  #[cfg(not(feature = "no-tables"))]
  {
    let mut crc = crc;
    let mut chunks = data.chunks_exact(8);

    for chunk in chunks.by_ref() {
      let bytes: [u8; 8] = chunk.try_into().unwrap();
      let d = u64::from_le_bytes(bytes);

      let x = crc ^ d;

      crc = table!(7)[(x & 0xFF) as usize]
        ^ table!(6)[((x >> 8) & 0xFF) as usize]
        ^ table!(5)[((x >> 16) & 0xFF) as usize]
        ^ table!(4)[((x >> 24) & 0xFF) as usize]
        ^ table!(3)[((x >> 32) & 0xFF) as usize]
        ^ table!(2)[((x >> 40) & 0xFF) as usize]
        ^ table!(1)[((x >> 48) & 0xFF) as usize]
        ^ table!(0)[((x >> 56) & 0xFF) as usize];
    }

    for &byte in chunks.remainder() {
      crc = (crc >> 8) ^ table!(0)[((crc ^ u64::from(byte)) & 0xFF) as usize];
    }

    crc
  }
}

#[inline]
#[allow(dead_code)]
const fn compute_byte(crc: u64, byte: u8) -> u64 {
  #[cfg(feature = "no-tables")]
  {
    use crate::constants::crc64::POLYNOMIAL;
    let mut crc = crc ^ (byte as u64);
    let mut i = 0;
    while i < 8 {
      let mask = 0u64.wrapping_sub(crc & 1);
      crc = (crc >> 1) ^ (POLYNOMIAL & mask);
      i += 1;
    }
    crc
  }

  #[cfg(not(feature = "no-tables"))]
  {
    (crc >> 8) ^ table!(0)[((crc ^ (byte as u64)) & 0xFF) as usize]
  }
}

#[cfg(test)]
mod tests {
  extern crate std;

  use super::*;

  #[test]
  fn test_check_string() {
    assert_eq!(Crc64::checksum(b"123456789"), 0x995D_C9BB_DF19_39FA);
  }

  #[test]
  fn test_empty() {
    assert_eq!(Crc64::checksum(b""), 0x0000_0000_0000_0000);
  }

  #[test]
  fn test_incremental() {
    let mut h = Crc64::new();
    h.update(b"1234");
    h.update(b"56789");
    assert_eq!(h.finalize(), 0x995D_C9BB_DF19_39FA);
  }

  #[test]
  fn test_resume() {
    let data = b"hello world";
    let (a, b) = data.split_at(6);

    let crc_a = Crc64::checksum(a);
    let mut h = Crc64::resume(crc_a);
    h.update(b);
    assert_eq!(h.finalize(), Crc64::checksum(data));
  }
}
