//! Portable CRC32 (ISO-HDLC) implementation.
//!
//! - Default: slicing-by-8 using 8×256 lookup tables (~8KB).
//! - `no-tables`: bitwise fallback (no tables, slower).

#[cfg(not(feature = "no-tables"))]
use crate::constants::crc32::TABLES;
#[cfg(not(feature = "no-tables"))]
macro_rules! table {
  ($idx:expr) => {
    TABLES.0[$idx]
  };
}

/// Compute CRC32 (ISO-HDLC) over `data`, returning the updated *raw* CRC state.
///
/// `crc` is the internal register value (no final XOR applied).
#[inline]
#[allow(dead_code)] // Unused when compile-time hardware CRC is enabled.
pub fn compute(crc: u32, data: &[u8]) -> u32 {
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

      let lo = (crc as u64) ^ (d & 0xFFFF_FFFF);
      let hi = d >> 32;

      crc = table!(7)[(lo & 0xFF) as usize]
        ^ table!(6)[((lo >> 8) & 0xFF) as usize]
        ^ table!(5)[((lo >> 16) & 0xFF) as usize]
        ^ table!(4)[((lo >> 24) & 0xFF) as usize]
        ^ table!(3)[(hi & 0xFF) as usize]
        ^ table!(2)[((hi >> 8) & 0xFF) as usize]
        ^ table!(1)[((hi >> 16) & 0xFF) as usize]
        ^ table!(0)[((hi >> 24) & 0xFF) as usize];
    }

    for &byte in chunks.remainder() {
      crc = (crc >> 8) ^ table!(0)[((crc ^ u32::from(byte)) & 0xFF) as usize];
    }

    crc
  }
}

/// Compute CRC32 (ISO-HDLC) for a single byte.
#[inline]
#[allow(dead_code)] // Used by no-table fallback and potential SIMD cleanups.
pub const fn compute_byte(crc: u32, byte: u8) -> u32 {
  #[cfg(feature = "no-tables")]
  {
    use crate::constants::crc32::POLYNOMIAL;

    let mut crc = crc ^ byte as u32;
    let mut i = 0;
    while i < 8 {
      let mask = 0u32.wrapping_sub(crc & 1);
      crc = (crc >> 1) ^ (POLYNOMIAL & mask);
      i += 1;
    }
    crc
  }

  #[cfg(not(feature = "no-tables"))]
  {
    (crc >> 8) ^ table!(0)[((crc ^ byte as u32) & 0xFF) as usize]
  }
}

#[cfg(test)]
mod tests {
  extern crate std;

  use alloc::vec;

  use super::*;

  const CHECK_VALUE: u32 = 0xCBF4_3926;

  #[test]
  fn test_check_string() {
    let crc = compute(0xFFFF_FFFF, b"123456789") ^ 0xFFFF_FFFF;
    assert_eq!(crc, CHECK_VALUE);
  }

  #[test]
  fn test_empty() {
    let crc = compute(0xFFFF_FFFF, b"") ^ 0xFFFF_FFFF;
    assert_eq!(crc, 0x0000_0000);
  }

  #[test]
  fn test_various_lengths() {
    for len in 0..=128 {
      let data = vec![0xABu8; len];
      let _ = compute(0xFFFF_FFFF, &data);
    }
  }
}
