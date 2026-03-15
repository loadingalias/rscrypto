//! Helpers for CRC-24/OPENPGP acceleration via a reflected width32 domain.
//!
//! CRC-24/OPENPGP is specified as MSB-first, but the accelerated kernels reuse
//! the reflected "width32" folding/reduction structure by:
//! - Lifting CRC-24 into the top 24 bits of a 32-bit register (low 8 bits = 0),
//! - Converting to a reflected CRC32 domain by bit-reversing each input byte,
//! - Converting back after folding.

// SAFETY: All array indexing in this module uses bounded indices:
// - Table generation iterates i: 0..256 into a [u32; 256]
// - Table lookups use `& 0xFF` which produces indices 0..255
#![allow(clippy::indexing_slicing)]

use super::keys::CRC24_OPENPGP_POLY_REFLECTED;

const MASK24: u32 = 0x00FF_FFFF;

#[must_use]
#[inline]
pub(crate) const fn to_reflected_state(crc: u32) -> u32 {
  ((crc & MASK24).strict_shl(8)).reverse_bits()
}

#[must_use]
#[inline]
pub(crate) const fn from_reflected_state(state: u32) -> u32 {
  (state.reverse_bits().strict_shr(8)) & MASK24
}

/// Generate a reflected CRC-32 table entry (LSB-first) for one byte.
#[must_use]
const fn crc32_reflected_table_entry(poly: u32, index: u8) -> u32 {
  let mut crc = index as u32;
  let mut i: u32 = 0;
  while i < 8 {
    crc = if (crc & 1) != 0 { (crc >> 1) ^ poly } else { crc >> 1 };
    i = i.strict_add(1);
  }
  crc
}

#[must_use]
const fn generate_crc32_reflected_table(poly: u32) -> [u32; 256] {
  let mut table = [0u32; 256];
  let mut i: u16 = 0;
  while i < 256 {
    table[i as usize] = crc32_reflected_table_entry(poly, i as u8);
    i = i.strict_add(1);
  }
  table
}

const CRC32_REFLECTED_TABLE: [u32; 256] = generate_crc32_reflected_table(CRC24_OPENPGP_POLY_REFLECTED);

/// Update a reflected CRC32-domain state with `data`, bit-reversing each byte.
#[must_use]
#[inline]
pub(crate) fn crc24_reflected_update_bitrev_bytes(mut state: u32, data: &[u8]) -> u32 {
  for &byte in data {
    let b = byte.reverse_bits();
    state = CRC32_REFLECTED_TABLE[((state ^ (b as u32)) & 0xFF) as usize] ^ (state >> 8);
  }
  state
}
