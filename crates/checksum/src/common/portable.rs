//! Portable CRC implementations using lookup table algorithms.
//!
//! This module provides generic slice-by-N implementations for all CRC widths:
//! - CRC-16: slice-by-4, slice-by-8
//! - CRC-32: slice-by-8, slice-by-16
//! - CRC-64: slice-by-8, slice-by-16
//!
//! # Algorithm Overview
//!
//! Slice-by-N processes N bytes per iteration using N precomputed lookup tables.
//! Each table contains 256 entries representing the CRC contribution of a single
//! byte at a specific position in the input stream.
//!
//! The algorithm XORs the current CRC with N input bytes, then combines
//! N table lookups (one per byte position) using XOR. This achieves ~N×
//! throughput compared to byte-at-a-time processing.
//!
//! # Performance Characteristics
//!
//! | Width | Algorithm | Bytes/iter | Tables | Throughput |
//! |-------|-----------|------------|--------|------------|
//! | 16-bit | slice-by-4 | 4 | 4×256×u16 | ~1.5 GB/s |
//! | 16-bit | slice-by-8 | 8 | 8×256×u16 | ~2.5 GB/s |
//! | 32-bit | slice-by-8 | 8 | 8×256×u32 | ~4.0 GB/s |
//! | 32-bit | slice-by-16 | 16 | 16×256×u32 | ~5.0 GB/s |
//! | 64-bit | slice-by-8 | 8 | 8×256×u64 | ~2.0 GB/s |
//! | 64-bit | slice-by-16 | 16 | 16×256×u64 | ~3.0 GB/s |

// SAFETY: All array indexing in this module uses bounded indices:
// - chunks_exact guarantees chunk sizes
// - Table indices use `& 0xFF` (0..255) or explicit byte extraction
// Clippy cannot prove this in const fn contexts, but bounds are statically guaranteed.
#![allow(clippy::indexing_slicing)]
#![cfg_attr(all(target_arch = "wasm32", target_feature = "simd128"), allow(unsafe_code))]

// ─────────────────────────────────────────────────────────────────────────────
// Small Input Helpers
// ─────────────────────────────────────────────────────────────────────────────

macro_rules! tail_step {
  ($crc:ident, $table:ident, $byte:expr, $crc_ty:ty) => {
    $crc = $table[((($crc ^ ($byte as $crc_ty)) & 0xFF) as usize)] ^ ($crc >> 8);
  };
}

macro_rules! tail8_body {
  ($crc:ident, $data:ident, $table:ident, $crc_ty:ty) => {{
    // Unrolled processing for 0-7 bytes. Each arm is branchless after the match.
    // The `_ =>` arm handles ≥8 bytes by falling through to a loop; this case
    // never occurs when called from slice functions (as_chunks guarantees < 8),
    // but we keep it for safety and the compiler eliminates it.
    match $data.len() {
      0 => {}
      1 => {
        tail_step!($crc, $table, $data[0], $crc_ty);
      }
      2 => {
        tail_step!($crc, $table, $data[0], $crc_ty);
        tail_step!($crc, $table, $data[1], $crc_ty);
      }
      3 => {
        tail_step!($crc, $table, $data[0], $crc_ty);
        tail_step!($crc, $table, $data[1], $crc_ty);
        tail_step!($crc, $table, $data[2], $crc_ty);
      }
      4 => {
        tail_step!($crc, $table, $data[0], $crc_ty);
        tail_step!($crc, $table, $data[1], $crc_ty);
        tail_step!($crc, $table, $data[2], $crc_ty);
        tail_step!($crc, $table, $data[3], $crc_ty);
      }
      5 => {
        tail_step!($crc, $table, $data[0], $crc_ty);
        tail_step!($crc, $table, $data[1], $crc_ty);
        tail_step!($crc, $table, $data[2], $crc_ty);
        tail_step!($crc, $table, $data[3], $crc_ty);
        tail_step!($crc, $table, $data[4], $crc_ty);
      }
      6 => {
        tail_step!($crc, $table, $data[0], $crc_ty);
        tail_step!($crc, $table, $data[1], $crc_ty);
        tail_step!($crc, $table, $data[2], $crc_ty);
        tail_step!($crc, $table, $data[3], $crc_ty);
        tail_step!($crc, $table, $data[4], $crc_ty);
        tail_step!($crc, $table, $data[5], $crc_ty);
      }
      7 => {
        tail_step!($crc, $table, $data[0], $crc_ty);
        tail_step!($crc, $table, $data[1], $crc_ty);
        tail_step!($crc, $table, $data[2], $crc_ty);
        tail_step!($crc, $table, $data[3], $crc_ty);
        tail_step!($crc, $table, $data[4], $crc_ty);
        tail_step!($crc, $table, $data[5], $crc_ty);
        tail_step!($crc, $table, $data[6], $crc_ty);
      }
      // Fallback loop for ≥8 bytes (unreachable from slice functions, but safe)
      _ => {
        for &byte in $data {
          tail_step!($crc, $table, byte, $crc_ty);
        }
      }
    }
    $crc
  }};
}

macro_rules! tail4_body {
  ($crc:ident, $data:ident, $table:ident, $crc_ty:ty) => {{
    match $data.len() {
      0 => {}
      1 => {
        tail_step!($crc, $table, $data[0], $crc_ty);
      }
      2 => {
        tail_step!($crc, $table, $data[0], $crc_ty);
        tail_step!($crc, $table, $data[1], $crc_ty);
      }
      3 => {
        tail_step!($crc, $table, $data[0], $crc_ty);
        tail_step!($crc, $table, $data[1], $crc_ty);
        tail_step!($crc, $table, $data[2], $crc_ty);
      }
      // Fallback loop for ≥4 bytes (unreachable from slice functions, but safe)
      _ => {
        for &byte in $data {
          tail_step!($crc, $table, byte, $crc_ty);
        }
      }
    }
    $crc
  }};
}

/// Process a small tail (0-7 bytes) for 64-bit CRC with unrolled lookups.
///
/// This avoids loop overhead for the common case of small remainders.
/// The function is always inlined for zero call overhead.
#[inline(always)]
fn tail8_64(mut crc: u64, data: &[u8], table: &[u64; 256]) -> u64 {
  tail8_body!(crc, data, table, u64)
}

/// Process a small tail (0-7 bytes) for 32-bit CRC with unrolled lookups.
#[cfg(test)]
#[inline(always)]
fn tail8_32(mut crc: u32, data: &[u8], table: &[u32; 256]) -> u32 {
  tail8_body!(crc, data, table, u32)
}

/// Process a small tail (0-3 bytes) for 32-bit CRC with unrolled lookups.
/// Used by slice16_32 which processes 4-byte chunks.
#[inline(always)]
fn tail4_32(mut crc: u32, data: &[u8], table: &[u32; 256]) -> u32 {
  tail4_body!(crc, data, table, u32)
}

/// Process a small tail (0-3 bytes) for 16-bit CRC with unrolled lookups.
#[inline(always)]
fn tail4_16(mut crc: u16, data: &[u8], table: &[u16; 256]) -> u16 {
  tail4_body!(crc, data, table, u16)
}

/// Process a small tail (0-7 bytes) for 16-bit CRC with unrolled lookups.
#[inline(always)]
fn tail8_16(mut crc: u16, data: &[u8], table: &[u16; 256]) -> u16 {
  tail8_body!(crc, data, table, u16)
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-16 Portable Implementations
// ─────────────────────────────────────────────────────────────────────────────

/// Update CRC-16 state using slice-by-4 algorithm.
///
/// Processes 4 bytes per iteration (2× the CRC width in bytes).
///
/// # Arguments
///
/// * `crc` - Current CRC state (pre-inverted if applicable)
/// * `data` - Input data
/// * `tables` - 4 lookup tables (256 entries each)
#[inline]
pub(crate) fn slice4_16(mut crc: u16, data: &[u8], tables: &[[u16; 256]; 4]) -> u16 {
  let (chunks, remainder) = data.as_chunks::<4>();

  for chunk in chunks {
    // Read 4 bytes: first 2 XOR with CRC, next 2 are separate
    let a = u16::from_le_bytes([chunk[0], chunk[1]]) ^ crc;
    let b = u16::from_le_bytes([chunk[2], chunk[3]]);

    crc = tables[3][(a & 0xFF) as usize]
      ^ tables[2][((a >> 8) & 0xFF) as usize]
      ^ tables[1][(b & 0xFF) as usize]
      ^ tables[0][((b >> 8) & 0xFF) as usize];
  }

  // Process remaining bytes (0-3) with unrolled lookups
  tail4_16(crc, remainder, &tables[0])
}

/// Update CRC-16 state using slice-by-8 algorithm.
///
/// Processes 8 bytes per iteration (4× the CRC width in bytes).
///
/// # Arguments
///
/// * `crc` - Current CRC state (pre-inverted if applicable)
/// * `data` - Input data
/// * `tables` - 8 lookup tables (256 entries each)
#[inline]
pub(crate) fn slice8_16(mut crc: u16, data: &[u8], tables: &[[u16; 256]; 8]) -> u16 {
  let (chunks, remainder) = data.as_chunks::<8>();

  for chunk in chunks {
    // Read 8 bytes as 4 u16 values
    let a = u16::from_le_bytes([chunk[0], chunk[1]]) ^ crc;
    let b = u16::from_le_bytes([chunk[2], chunk[3]]);
    let c = u16::from_le_bytes([chunk[4], chunk[5]]);
    let d = u16::from_le_bytes([chunk[6], chunk[7]]);

    crc = tables[7][(a & 0xFF) as usize]
      ^ tables[6][((a >> 8) & 0xFF) as usize]
      ^ tables[5][(b & 0xFF) as usize]
      ^ tables[4][((b >> 8) & 0xFF) as usize]
      ^ tables[3][(c & 0xFF) as usize]
      ^ tables[2][((c >> 8) & 0xFF) as usize]
      ^ tables[1][(d & 0xFF) as usize]
      ^ tables[0][((d >> 8) & 0xFF) as usize];
  }

  // Process remaining bytes (0-7) with unrolled lookups
  tail8_16(crc, remainder, &tables[0])
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-24 Portable Implementations (MSB-first, OpenPGP)
// ─────────────────────────────────────────────────────────────────────────────

/// Update CRC-24 state using slice-by-4 algorithm (MSB-first).
///
/// CRC-24 is represented as a 24-bit value in the low bits of a `u32`.
/// Internally this expands to the top 24 bits of a `u32` so the slice-by-N
/// strategy matches the standard MSB-first CRC-32 table method.
///
/// # Arguments
///
/// * `crc` - Current CRC state (low 24 bits)
/// * `data` - Input data
/// * `tables` - 4 lookup tables (256 entries each), generated by `generate_crc24_tables_4`
#[inline]
pub(crate) fn slice4_24(mut crc: u32, data: &[u8], tables: &[[u32; 256]; 4]) -> u32 {
  const MASK24: u32 = 0x00FF_FFFF;

  let mut state = (crc & MASK24) << 8;
  let (chunks, remainder) = data.as_chunks::<4>();

  for chunk in chunks {
    let a = u32::from_be_bytes(*chunk) ^ state;
    state = tables[3][(a >> 24) as usize]
      ^ tables[2][((a >> 16) & 0xFF) as usize]
      ^ tables[1][((a >> 8) & 0xFF) as usize]
      ^ tables[0][(a & 0xFF) as usize];
  }

  for &byte in remainder {
    let index = (((state >> 24) as u8) ^ byte) as usize;
    state = tables[0][index] ^ (state << 8);
  }

  crc = (state >> 8) & MASK24;
  crc
}

/// Update CRC-24 state using slice-by-8 algorithm (MSB-first).
///
/// # Arguments
///
/// * `crc` - Current CRC state (low 24 bits)
/// * `data` - Input data
/// * `tables` - 8 lookup tables (256 entries each), generated by `generate_crc24_tables_8`
#[inline]
pub(crate) fn slice8_24(mut crc: u32, data: &[u8], tables: &[[u32; 256]; 8]) -> u32 {
  const MASK24: u32 = 0x00FF_FFFF;

  let mut state = (crc & MASK24) << 8;
  let (chunks, remainder) = data.as_chunks::<8>();

  for chunk in chunks {
    let a = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) ^ state;
    let b = u32::from_be_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);

    state = tables[7][(a >> 24) as usize]
      ^ tables[6][((a >> 16) & 0xFF) as usize]
      ^ tables[5][((a >> 8) & 0xFF) as usize]
      ^ tables[4][(a & 0xFF) as usize]
      ^ tables[3][(b >> 24) as usize]
      ^ tables[2][((b >> 16) & 0xFF) as usize]
      ^ tables[1][((b >> 8) & 0xFF) as usize]
      ^ tables[0][(b & 0xFF) as usize];
  }

  for &byte in remainder {
    let index = (((state >> 24) as u8) ^ byte) as usize;
    state = tables[0][index] ^ (state << 8);
  }

  crc = (state >> 8) & MASK24;
  crc
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-32 Portable Implementations
// ─────────────────────────────────────────────────────────────────────────────

/// Update CRC-32 state using slice-by-8 algorithm.
///
/// Processes 8 bytes per iteration (2× the CRC width in bytes).
///
/// # Arguments
///
/// * `crc` - Current CRC state (pre-inverted)
/// * `data` - Input data
/// * `tables` - 8 lookup tables (256 entries each)
#[cfg(test)]
#[inline]
pub fn slice8_32(mut crc: u32, data: &[u8], tables: &[[u32; 256]; 8]) -> u32 {
  let (chunks, remainder) = data.as_chunks::<8>();

  for chunk in chunks {
    let a = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) ^ crc;
    let b = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);

    crc = tables[7][(a & 0xFF) as usize]
      ^ tables[6][((a >> 8) & 0xFF) as usize]
      ^ tables[5][((a >> 16) & 0xFF) as usize]
      ^ tables[4][(a >> 24) as usize]
      ^ tables[3][(b & 0xFF) as usize]
      ^ tables[2][((b >> 8) & 0xFF) as usize]
      ^ tables[1][((b >> 16) & 0xFF) as usize]
      ^ tables[0][(b >> 24) as usize];
  }

  // Process remaining bytes (0-7) with unrolled lookups
  tail8_32(crc, remainder, &tables[0])
}

/// Update CRC-32 state using slice-by-16 algorithm.
///
/// Processes 16 bytes per iteration (4× the CRC width in bytes).
///
/// # Arguments
///
/// * `crc` - Current CRC state (pre-inverted)
/// * `data` - Input data
/// * `tables` - 16 lookup tables (256 entries each)
#[inline]
fn slice16_32_scalar(mut crc: u32, data: &[u8], tables: &[[u32; 256]; 16]) -> u32 {
  let (chunks4, remainder) = data.as_chunks::<4>();
  let mut quads = chunks4.chunks_exact(4);

  for quad in quads.by_ref() {
    let a = u32::from_le_bytes(quad[0]) ^ crc;
    let b = u32::from_le_bytes(quad[1]);
    let c = u32::from_le_bytes(quad[2]);
    let d = u32::from_le_bytes(quad[3]);

    crc = tables[15][(a & 0xFF) as usize]
      ^ tables[14][((a >> 8) & 0xFF) as usize]
      ^ tables[13][((a >> 16) & 0xFF) as usize]
      ^ tables[12][(a >> 24) as usize]
      ^ tables[11][(b & 0xFF) as usize]
      ^ tables[10][((b >> 8) & 0xFF) as usize]
      ^ tables[9][((b >> 16) & 0xFF) as usize]
      ^ tables[8][(b >> 24) as usize]
      ^ tables[7][(c & 0xFF) as usize]
      ^ tables[6][((c >> 8) & 0xFF) as usize]
      ^ tables[5][((c >> 16) & 0xFF) as usize]
      ^ tables[4][(c >> 24) as usize]
      ^ tables[3][(d & 0xFF) as usize]
      ^ tables[2][((d >> 8) & 0xFF) as usize]
      ^ tables[1][((d >> 16) & 0xFF) as usize]
      ^ tables[0][(d >> 24) as usize];
  }

  // Handle a 4-byte tail (one to three u32 chunks)
  for chunk in quads.remainder() {
    let val = u32::from_le_bytes(*chunk) ^ crc;
    crc = tables[3][(val & 0xFF) as usize]
      ^ tables[2][((val >> 8) & 0xFF) as usize]
      ^ tables[1][((val >> 16) & 0xFF) as usize]
      ^ tables[0][(val >> 24) as usize];
  }

  // Process remaining bytes (0-3) with unrolled lookups
  tail4_32(crc, remainder, &tables[0])
}

/// Update CRC-32 state using slice-by-16 algorithm.
///
/// Processes 16 bytes per iteration (4× the CRC width in bytes).
///
/// # Arguments
///
/// * `crc` - Current CRC state (pre-inverted)
/// * `data` - Input data
/// * `tables` - 16 lookup tables (256 entries each)
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline]
pub fn slice16_32(crc: u32, data: &[u8], tables: &[[u32; 256]; 16]) -> u32 {
  slice16_32_scalar(crc, data, tables)
}

/// Update CRC-32 state using slice-by-16 algorithm, using `wasm32/simd128` loads.
///
/// This is still a table-driven portable algorithm (no CLMUL/HWCRC), but uses
/// `v128` loads when `target_feature = "simd128"` is enabled.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
pub fn slice16_32(mut crc: u32, data: &[u8], tables: &[[u32; 256]; 16]) -> u32 {
  use core::arch::wasm32::v128;

  let mut ptr = data.as_ptr();
  let mut len = data.len();

  while len >= 16 {
    // SAFETY: `ptr` is within `data` and `len >= 16`.
    let v = unsafe { core::ptr::read_unaligned(ptr as *const v128) };
    // SAFETY: `v128` is a 16-byte value and `u32x4` lane layout matches memory order on wasm.
    let words: [u32; 4] = unsafe { core::mem::transmute(v) };

    let a = words[0] ^ crc;
    let b = words[1];
    let c = words[2];
    let d = words[3];

    crc = tables[15][(a & 0xFF) as usize]
      ^ tables[14][((a >> 8) & 0xFF) as usize]
      ^ tables[13][((a >> 16) & 0xFF) as usize]
      ^ tables[12][(a >> 24) as usize]
      ^ tables[11][(b & 0xFF) as usize]
      ^ tables[10][((b >> 8) & 0xFF) as usize]
      ^ tables[9][((b >> 16) & 0xFF) as usize]
      ^ tables[8][(b >> 24) as usize]
      ^ tables[7][(c & 0xFF) as usize]
      ^ tables[6][((c >> 8) & 0xFF) as usize]
      ^ tables[5][((c >> 16) & 0xFF) as usize]
      ^ tables[4][(c >> 24) as usize]
      ^ tables[3][(d & 0xFF) as usize]
      ^ tables[2][((d >> 8) & 0xFF) as usize]
      ^ tables[1][((d >> 16) & 0xFF) as usize]
      ^ tables[0][(d >> 24) as usize];

    // SAFETY: ptr stays within `data` due to the `len >= 16` loop guard.
    ptr = unsafe { ptr.add(16) };
    len = len.strict_sub(16);
  }

  // SAFETY: `ptr` points to the unprocessed tail of `data` with length `len`.
  let tail = unsafe { core::slice::from_raw_parts(ptr, len) };
  slice16_32_scalar(crc, tail, tables)
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-64 Portable Implementations
// ─────────────────────────────────────────────────────────────────────────────

/// Update CRC-64 state using slice-by-8 algorithm.
///
/// Processes 8 bytes per iteration (1× the CRC width in bytes).
///
/// # Arguments
///
/// * `crc` - Current CRC state (pre-inverted)
/// * `data` - Input data
/// * `tables` - 8 lookup tables (256 entries each)
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", test))]
#[inline]
pub fn slice8_64(mut crc: u64, data: &[u8], tables: &[[u64; 256]; 8]) -> u64 {
  let (chunks, remainder) = data.as_chunks::<8>();

  for chunk in chunks {
    let val = u64::from_le_bytes(*chunk) ^ crc;

    crc = tables[7][(val & 0xFF) as usize]
      ^ tables[6][((val >> 8) & 0xFF) as usize]
      ^ tables[5][((val >> 16) & 0xFF) as usize]
      ^ tables[4][((val >> 24) & 0xFF) as usize]
      ^ tables[3][((val >> 32) & 0xFF) as usize]
      ^ tables[2][((val >> 40) & 0xFF) as usize]
      ^ tables[1][((val >> 48) & 0xFF) as usize]
      ^ tables[0][(val >> 56) as usize];
  }

  // Process remaining bytes (0-7) with unrolled lookups
  tail8_64(crc, remainder, &tables[0])
}

/// Update CRC-64 state using slice-by-16 algorithm.
///
/// Processes 16 bytes per iteration (2× the CRC width in bytes).
/// Optimal for larger buffers where cache is warm.
///
/// # Arguments
///
/// * `crc` - Current CRC state (pre-inverted)
/// * `data` - Input data
/// * `tables` - 16 lookup tables (256 entries each)
#[inline]
fn slice16_64_scalar(mut crc: u64, data: &[u8], tables: &[[u64; 256]; 16]) -> u64 {
  let (chunks8, remainder) = data.as_chunks::<8>();
  let mut pairs = chunks8.chunks_exact(2);

  for pair in pairs.by_ref() {
    let a = u64::from_le_bytes(pair[0]) ^ crc;
    let b = u64::from_le_bytes(pair[1]);

    crc = tables[15][(a & 0xFF) as usize]
      ^ tables[14][((a >> 8) & 0xFF) as usize]
      ^ tables[13][((a >> 16) & 0xFF) as usize]
      ^ tables[12][((a >> 24) & 0xFF) as usize]
      ^ tables[11][((a >> 32) & 0xFF) as usize]
      ^ tables[10][((a >> 40) & 0xFF) as usize]
      ^ tables[9][((a >> 48) & 0xFF) as usize]
      ^ tables[8][(a >> 56) as usize]
      ^ tables[7][(b & 0xFF) as usize]
      ^ tables[6][((b >> 8) & 0xFF) as usize]
      ^ tables[5][((b >> 16) & 0xFF) as usize]
      ^ tables[4][((b >> 24) & 0xFF) as usize]
      ^ tables[3][((b >> 32) & 0xFF) as usize]
      ^ tables[2][((b >> 40) & 0xFF) as usize]
      ^ tables[1][((b >> 48) & 0xFF) as usize]
      ^ tables[0][(b >> 56) as usize];
  }

  // Handle an odd 8-byte tail
  if let [chunk] = pairs.remainder() {
    let val = u64::from_le_bytes(*chunk) ^ crc;
    crc = tables[7][(val & 0xFF) as usize]
      ^ tables[6][((val >> 8) & 0xFF) as usize]
      ^ tables[5][((val >> 16) & 0xFF) as usize]
      ^ tables[4][((val >> 24) & 0xFF) as usize]
      ^ tables[3][((val >> 32) & 0xFF) as usize]
      ^ tables[2][((val >> 40) & 0xFF) as usize]
      ^ tables[1][((val >> 48) & 0xFF) as usize]
      ^ tables[0][(val >> 56) as usize];
  }

  // Process remaining bytes (0-7) with unrolled lookups
  tail8_64(crc, remainder, &tables[0])
}

/// Update CRC-64 state using slice-by-16 algorithm.
///
/// Processes 16 bytes per iteration (2× the CRC width in bytes).
/// Optimal for larger buffers where cache is warm.
///
/// # Arguments
///
/// * `crc` - Current CRC state (pre-inverted)
/// * `data` - Input data
/// * `tables` - 16 lookup tables (256 entries each)
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline]
pub fn slice16_64(crc: u64, data: &[u8], tables: &[[u64; 256]; 16]) -> u64 {
  slice16_64_scalar(crc, data, tables)
}

/// Update CRC-64 state using slice-by-16 algorithm, using `wasm32/simd128` loads.
///
/// This is still a table-driven portable algorithm (no CLMUL), but uses `v128`
/// loads when `target_feature = "simd128"` is enabled.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
pub fn slice16_64(mut crc: u64, data: &[u8], tables: &[[u64; 256]; 16]) -> u64 {
  use core::arch::wasm32::v128;

  let mut ptr = data.as_ptr();
  let mut len = data.len();

  while len >= 16 {
    // SAFETY: `ptr` is within `data` and `len >= 16`.
    let v = unsafe { core::ptr::read_unaligned(ptr as *const v128) };
    // SAFETY: `v128` is a 16-byte value and `u64x2` lane layout matches memory order on wasm.
    let words: [u64; 2] = unsafe { core::mem::transmute(v) };

    let a = words[0] ^ crc;
    let b = words[1];

    crc = tables[15][(a & 0xFF) as usize]
      ^ tables[14][((a >> 8) & 0xFF) as usize]
      ^ tables[13][((a >> 16) & 0xFF) as usize]
      ^ tables[12][((a >> 24) & 0xFF) as usize]
      ^ tables[11][((a >> 32) & 0xFF) as usize]
      ^ tables[10][((a >> 40) & 0xFF) as usize]
      ^ tables[9][((a >> 48) & 0xFF) as usize]
      ^ tables[8][(a >> 56) as usize]
      ^ tables[7][(b & 0xFF) as usize]
      ^ tables[6][((b >> 8) & 0xFF) as usize]
      ^ tables[5][((b >> 16) & 0xFF) as usize]
      ^ tables[4][((b >> 24) & 0xFF) as usize]
      ^ tables[3][((b >> 32) & 0xFF) as usize]
      ^ tables[2][((b >> 40) & 0xFF) as usize]
      ^ tables[1][((b >> 48) & 0xFF) as usize]
      ^ tables[0][(b >> 56) as usize];

    // SAFETY: ptr stays within `data` due to the `len >= 16` loop guard.
    ptr = unsafe { ptr.add(16) };
    len = len.strict_sub(16);
  }

  // SAFETY: `ptr` points to the unprocessed tail of `data` with length `len`.
  let tail = unsafe { core::slice::from_raw_parts(ptr, len) };
  slice16_64_scalar(crc, tail, tables)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-16 Tests
  // ─────────────────────────────────────────────────────────────────────────

  /// Generate CRC-16 tables for testing (CCITT polynomial 0x8408 reflected).
  const fn crc16_table_entry(poly: u16, index: u8) -> u16 {
    let mut crc = index as u16;
    let mut i = 0;
    while i < 8 {
      if crc & 1 != 0 {
        crc = (crc >> 1) ^ poly;
      } else {
        crc >>= 1;
      }
      i += 1;
    }
    crc
  }

  const fn generate_crc16_tables_4(poly: u16) -> [[u16; 256]; 4] {
    let mut tables = [[0u16; 256]; 4];
    let mut i = 0u16;
    while i < 256 {
      tables[0][i as usize] = crc16_table_entry(poly, i as u8);
      i += 1;
    }
    let mut k = 1usize;
    while k < 4 {
      i = 0;
      while i < 256 {
        let prev = tables[k - 1][i as usize];
        tables[k][i as usize] = tables[0][(prev & 0xFF) as usize] ^ (prev >> 8);
        i += 1;
      }
      k += 1;
    }
    tables
  }

  const fn generate_crc16_tables_8(poly: u16) -> [[u16; 256]; 8] {
    let mut tables = [[0u16; 256]; 8];
    let mut i = 0u16;
    while i < 256 {
      tables[0][i as usize] = crc16_table_entry(poly, i as u8);
      i += 1;
    }
    let mut k = 1usize;
    while k < 8 {
      i = 0;
      while i < 256 {
        let prev = tables[k - 1][i as usize];
        tables[k][i as usize] = tables[0][(prev & 0xFF) as usize] ^ (prev >> 8);
        i += 1;
      }
      k += 1;
    }
    tables
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-32 Tests
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_slice8_32_empty() {
    let tables = crate::common::tables::generate_crc32_tables_8(crate::common::tables::CRC32_IEEE_POLY);
    let crc = slice8_32(!0, &[], &tables);
    assert_eq!(crc, !0);
  }

  #[test]
  fn test_slice16_32_empty() {
    let tables = crate::common::tables::generate_crc32_tables_16(crate::common::tables::CRC32_IEEE_POLY);
    let crc = slice16_32(!0, &[], &tables);
    assert_eq!(crc, !0);
  }

  #[test]
  fn test_slice16_32_matches_slice8_32() {
    let poly = crate::common::tables::CRC32_IEEE_POLY;
    let tables8 = crate::common::tables::generate_crc32_tables_8(poly);
    let tables16 = crate::common::tables::generate_crc32_tables_16(poly);
    let data = b"The quick brown fox jumps over the lazy dog";
    assert_eq!(slice8_32(!0, data, &tables8), slice16_32(!0, data, &tables16));
  }

  const CRC16_CCITT_POLY: u16 = 0x8408; // Reflected

  #[test]
  fn test_slice4_16_empty() {
    let tables = generate_crc16_tables_4(CRC16_CCITT_POLY);
    assert_eq!(slice4_16(!0, &[], &tables), !0);
  }

  #[test]
  fn test_slice8_16_empty() {
    let tables = generate_crc16_tables_8(CRC16_CCITT_POLY);
    assert_eq!(slice8_16(!0, &[], &tables), !0);
  }

  #[test]
  fn test_slice4_16_matches_slice8_16() {
    let tables4 = generate_crc16_tables_4(CRC16_CCITT_POLY);
    let tables8 = generate_crc16_tables_8(CRC16_CCITT_POLY);
    let data = b"The quick brown fox jumps over the lazy dog";

    let a = slice4_16(!0, data, &tables4);
    let b = slice8_16(!0, data, &tables8);
    assert_eq!(a, b);
  }

  #[test]
  fn test_slice4_16_incremental() {
    let tables = generate_crc16_tables_4(CRC16_CCITT_POLY);
    let data = b"hello world, this is a test";
    let full = slice4_16(!0, data, &tables);

    for split in [1, 3, 4, 5, 7, 8, 10, 15] {
      if split < data.len() {
        let crc1 = slice4_16(!0, &data[..split], &tables);
        let crc2 = slice4_16(crc1, &data[split..], &tables);
        assert_eq!(crc2, full, "Incremental failed at split {split}");
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-24 Tests
  // ─────────────────────────────────────────────────────────────────────────

  const CRC24_OPENPGP_INIT: u32 = 0x00B7_04CE;

  #[test]
  fn test_slice4_24_empty() {
    let tables = crate::common::tables::generate_crc24_tables_4(crate::common::tables::CRC24_OPENPGP_POLY);
    assert_eq!(slice4_24(CRC24_OPENPGP_INIT, &[], &tables), CRC24_OPENPGP_INIT);
  }

  #[test]
  fn test_slice8_24_empty() {
    let tables = crate::common::tables::generate_crc24_tables_8(crate::common::tables::CRC24_OPENPGP_POLY);
    assert_eq!(slice8_24(CRC24_OPENPGP_INIT, &[], &tables), CRC24_OPENPGP_INIT);
  }

  #[test]
  fn test_slice4_24_matches_slice8_24() {
    let poly = crate::common::tables::CRC24_OPENPGP_POLY;
    let tables4 = crate::common::tables::generate_crc24_tables_4(poly);
    let tables8 = crate::common::tables::generate_crc24_tables_8(poly);
    let data = b"The quick brown fox jumps over the lazy dog";

    let a = slice4_24(CRC24_OPENPGP_INIT, data, &tables4);
    let b = slice8_24(CRC24_OPENPGP_INIT, data, &tables8);
    assert_eq!(a, b);
  }

  #[test]
  fn test_slice4_24_incremental() {
    let tables = crate::common::tables::generate_crc24_tables_4(crate::common::tables::CRC24_OPENPGP_POLY);
    let data = b"hello world, this is a test";
    let full = slice4_24(CRC24_OPENPGP_INIT, data, &tables);

    for split in [1, 3, 4, 5, 7, 8, 10, 15] {
      if split < data.len() {
        let crc1 = slice4_24(CRC24_OPENPGP_INIT, &data[..split], &tables);
        let crc2 = slice4_24(crc1, &data[split..], &tables);
        assert_eq!(crc2, full, "Incremental failed at split {split}");
      }
    }
  }

  #[test]
  fn test_slice8_24_test_vector_openpgp() {
    let tables = crate::common::tables::generate_crc24_tables_8(crate::common::tables::CRC24_OPENPGP_POLY);
    let out = slice8_24(CRC24_OPENPGP_INIT, b"123456789", &tables);
    assert_eq!(out, 0x0021_CF02);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-64 Tests
  // ─────────────────────────────────────────────────────────────────────────

  /// Generate CRC-64 tables for testing.
  const fn crc64_table_entry(poly: u64, index: u8) -> u64 {
    let mut crc = index as u64;
    let mut i = 0;
    while i < 8 {
      if crc & 1 != 0 {
        crc = (crc >> 1) ^ poly;
      } else {
        crc >>= 1;
      }
      i += 1;
    }
    crc
  }

  const fn generate_crc64_tables_8(poly: u64) -> [[u64; 256]; 8] {
    let mut tables = [[0u64; 256]; 8];
    let mut i = 0u16;
    while i < 256 {
      tables[0][i as usize] = crc64_table_entry(poly, i as u8);
      i += 1;
    }
    let mut k = 1usize;
    while k < 8 {
      i = 0;
      while i < 256 {
        let prev = tables[k - 1][i as usize];
        tables[k][i as usize] = tables[0][(prev & 0xFF) as usize] ^ (prev >> 8);
        i += 1;
      }
      k += 1;
    }
    tables
  }

  const fn generate_crc64_tables_16(poly: u64) -> [[u64; 256]; 16] {
    let mut tables = [[0u64; 256]; 16];
    let mut i = 0u16;
    while i < 256 {
      tables[0][i as usize] = crc64_table_entry(poly, i as u8);
      i += 1;
    }
    let mut k = 1usize;
    while k < 16 {
      i = 0;
      while i < 256 {
        let prev = tables[k - 1][i as usize];
        tables[k][i as usize] = tables[0][(prev & 0xFF) as usize] ^ (prev >> 8);
        i += 1;
      }
      k += 1;
    }
    tables
  }

  const CRC64_XZ_POLY: u64 = 0xC96C_5795_D787_0F42; // Reflected

  #[test]
  fn test_slice8_64_empty() {
    let tables = generate_crc64_tables_8(CRC64_XZ_POLY);
    assert_eq!(slice8_64(!0, &[], &tables), !0);
  }

  #[test]
  fn test_slice16_64_empty() {
    let tables = generate_crc64_tables_16(CRC64_XZ_POLY);
    assert_eq!(slice16_64(!0, &[], &tables), !0);
  }

  #[test]
  fn test_slice8_64_matches_slice16_64() {
    let tables8 = generate_crc64_tables_8(CRC64_XZ_POLY);
    let tables16 = generate_crc64_tables_16(CRC64_XZ_POLY);
    let data = b"The quick brown fox jumps over the lazy dog";

    let a = slice8_64(!0, data, &tables8);
    let b = slice16_64(!0, data, &tables16);
    assert_eq!(a, b);
  }

  #[test]
  fn test_slice16_64_incremental() {
    let tables = generate_crc64_tables_16(CRC64_XZ_POLY);
    let data = b"hello world, this is a longer test string";
    let full = slice16_64(!0, data, &tables);

    for split in [1, 7, 8, 9, 15, 16, 17, 20] {
      if split < data.len() {
        let crc1 = slice16_64(!0, &data[..split], &tables);
        let crc2 = slice16_64(crc1, &data[split..], &tables);
        assert_eq!(crc2, full, "Incremental failed at split {split}");
      }
    }
  }

  #[test]
  fn test_crc64_xz_test_vector() {
    // "123456789" should produce 0x995DC9BBDF1939FA for CRC-64-XZ
    let tables = generate_crc64_tables_16(CRC64_XZ_POLY);
    let crc = slice16_64(!0, b"123456789", &tables) ^ !0;
    assert_eq!(crc, 0x995D_C9BB_DF19_39FA);
  }
}
