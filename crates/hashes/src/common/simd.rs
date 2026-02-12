//! Cross-architecture SIMD abstractions for hash primitives.
//!
//! This module provides architecture-agnostic traits for SIMD operations
//! commonly used in hash algorithm implementations. The goal is to enable
//! writing generic SIMD code that can be specialized for each platform.
//!
//! # Design Philosophy
//!
//! Rather than providing full SIMD wrappers (which would duplicate `std::arch`),
//! these traits capture the *patterns* used in hash algorithms:
//!
//! - **State vectors**: Hash state fits in 128-256 bits (4-8 u32 or 2-4 u64)
//! - **Rotations**: Hash algorithms heavily use bitwise rotations
//! - **XOR/ADD**: Mixing operations are XOR and modular addition
//! - **Shuffles**: Message schedule permutations
//!
//! # Usage
//!
//! Implementers create platform-specific types that implement these traits,
//! then write generic code against the traits. The compiler specializes
//! for each platform.
//!
//! ```text
//! fn blake3_g<V: Simd128>(state: &mut [V; 4], m: V) {
//!     state[0] = state[0].add32(state[1]).add32(m);
//!     state[3] = state[3].xor(state[0]).rotr32::<16>();
//!     // ... rest of G function
//! }
//! ```
//!
//! # Platform Support
//!
//! | Platform | 128-bit | 256-bit | 512-bit |
//! |----------|---------|---------|---------|
//! | x86_64 SSE | ✓ | - | - |
//! | x86_64 AVX2 | ✓ | ✓ | - |
//! | x86_64 AVX-512 | ✓ | ✓ | ✓ |
//! | aarch64 NEON | ✓ | - | - |
//! | aarch64 SVE2 | ✓ | ✓ | ✓* |
//!
//! *SVE2 vector length is implementation-defined; may not be exactly 512 bits.

// ─────────────────────────────────────────────────────────────────────────────
// 128-bit Vector Abstraction
// ─────────────────────────────────────────────────────────────────────────────

/// Abstraction over 128-bit vector operations.
///
/// Used by BLAKE3, BLAKE2, SHA-256 (non-SHA-NI paths), and other algorithms
/// that operate on 4×u32 or 2×u64 state vectors.
///
/// # Implementors
///
/// - `__m128i` (x86_64 SSE/AVX)
/// - `uint32x4_t` / `uint64x2_t` (aarch64 NEON)
/// - Scalar fallback for testing
pub trait Simd128: Copy + Clone + Sized {
  /// Load 16 bytes from memory (may be unaligned).
  fn load(src: &[u8; 16]) -> Self;

  /// Load 16 bytes from aligned memory.
  ///
  /// # Safety
  ///
  /// `src` must be 16-byte aligned.
  unsafe fn load_aligned(src: &[u8; 16]) -> Self;

  /// Store to 16 bytes of memory (may be unaligned).
  fn store(self, dst: &mut [u8; 16]);

  /// Store to 16 bytes of aligned memory.
  ///
  /// # Safety
  ///
  /// `dst` must be 16-byte aligned.
  unsafe fn store_aligned(self, dst: &mut [u8; 16]);

  /// Bitwise XOR.
  fn xor(self, other: Self) -> Self;

  /// Bitwise AND.
  fn and(self, other: Self) -> Self;

  /// Bitwise OR.
  fn or(self, other: Self) -> Self;

  /// Bitwise AND-NOT: `!self & other`.
  fn andnot(self, other: Self) -> Self;

  /// 4×u32 addition (wrapping).
  fn add32(self, other: Self) -> Self;

  /// 2×u64 addition (wrapping).
  fn add64(self, other: Self) -> Self;

  /// 4×u32 rotate right by `N` bits.
  ///
  /// N must be in range 1..=31.
  fn rotr32<const N: u32>(self) -> Self;

  /// 2×u64 rotate right by `N` bits.
  ///
  /// N must be in range 1..=63.
  fn rotr64<const N: u32>(self) -> Self;

  /// Shuffle 32-bit lanes according to immediate mask.
  ///
  /// The mask uses the same encoding as `_mm_shuffle_epi32`:
  /// each 2-bit field selects a source lane (0-3).
  fn shuffle32<const MASK: i32>(self) -> Self;

  /// Shuffle bytes according to control vector.
  ///
  /// Each byte in `ctrl` selects a byte from `self` (0-15),
  /// or zeros the result byte if the high bit is set.
  fn shuffle_bytes(self, ctrl: Self) -> Self;

  /// Create vector with all lanes set to zero.
  fn zero() -> Self;

  /// Create vector from 4×u32 values.
  fn from_u32x4(a: u32, b: u32, c: u32, d: u32) -> Self;

  /// Create vector from 2×u64 values.
  fn from_u64x2(a: u64, b: u64) -> Self;

  /// Extract 32-bit lane.
  fn extract32<const LANE: i32>(self) -> u32;

  /// Extract 64-bit lane.
  fn extract64<const LANE: i32>(self) -> u64;

  /// Interleave low 32-bit elements: `[a0, b0, a1, b1]`.
  fn unpack_lo32(self, other: Self) -> Self;

  /// Interleave high 32-bit elements: `[a2, b2, a3, b3]`.
  fn unpack_hi32(self, other: Self) -> Self;

  /// Interleave low 64-bit elements: `[a0, b0]`.
  fn unpack_lo64(self, other: Self) -> Self;

  /// Interleave high 64-bit elements: `[a1, b1]`.
  fn unpack_hi64(self, other: Self) -> Self;
}

// ─────────────────────────────────────────────────────────────────────────────
// 256-bit Vector Abstraction
// ─────────────────────────────────────────────────────────────────────────────

/// Abstraction over 256-bit vector operations.
///
/// Used by BLAKE3 2-way parallel, SHA-512 SIMD paths, and other algorithms
/// that benefit from processing multiple blocks in parallel.
///
/// # Implementors
///
/// - `__m256i` (x86_64 AVX2/AVX-512)
/// - SVE2 256-bit (aarch64, implementation-defined)
pub trait Simd256: Copy + Clone + Sized {
  /// The 128-bit half type for lane operations.
  type Half: Simd128;

  /// Load 32 bytes from memory (may be unaligned).
  fn load(src: &[u8; 32]) -> Self;

  /// Load 32 bytes from aligned memory.
  ///
  /// # Safety
  ///
  /// `src` must be 32-byte aligned.
  unsafe fn load_aligned(src: &[u8; 32]) -> Self;

  /// Store to 32 bytes of memory (may be unaligned).
  fn store(self, dst: &mut [u8; 32]);

  /// Store to 32 bytes of aligned memory.
  ///
  /// # Safety
  ///
  /// `dst` must be 32-byte aligned.
  unsafe fn store_aligned(self, dst: &mut [u8; 32]);

  /// Bitwise XOR.
  fn xor(self, other: Self) -> Self;

  /// Bitwise AND.
  fn and(self, other: Self) -> Self;

  /// Bitwise OR.
  fn or(self, other: Self) -> Self;

  /// 8×u32 addition (wrapping).
  fn add32(self, other: Self) -> Self;

  /// 4×u64 addition (wrapping).
  fn add64(self, other: Self) -> Self;

  /// 8×u32 rotate right by `N` bits.
  fn rotr32<const N: u32>(self) -> Self;

  /// 4×u64 rotate right by `N` bits.
  fn rotr64<const N: u32>(self) -> Self;

  /// Shuffle 32-bit lanes within 128-bit lanes.
  fn shuffle32<const MASK: i32>(self) -> Self;

  /// Shuffle bytes within 128-bit lanes.
  fn shuffle_bytes(self, ctrl: Self) -> Self;

  /// Create vector with all lanes set to zero.
  fn zero() -> Self;

  /// Extract low 128-bit lane.
  fn extract_lo(self) -> Self::Half;

  /// Extract high 128-bit lane.
  fn extract_hi(self) -> Self::Half;

  /// Combine two 128-bit halves into 256-bit vector.
  fn from_halves(lo: Self::Half, hi: Self::Half) -> Self;

  /// Permute 128-bit lanes across the vector.
  ///
  /// For AVX2: `vperm2i128` with immediate control.
  fn permute128<const MASK: i32>(self, other: Self) -> Self;

  /// Broadcast 128-bit value to both lanes.
  fn broadcast128(half: Self::Half) -> Self;
}

// ─────────────────────────────────────────────────────────────────────────────
// 512-bit Vector Abstraction
// ─────────────────────────────────────────────────────────────────────────────

/// Abstraction over 512-bit vector operations.
///
/// Used by BLAKE3 4-way parallel, SHA-512 wide paths, and algorithms
/// that can exploit maximum parallelism on AVX-512 hardware.
///
/// # Implementors
///
/// - `__m512i` (x86_64 AVX-512)
/// - SVE2 512-bit (aarch64, if hardware supports)
pub trait Simd512: Copy + Clone + Sized {
  /// The 256-bit half type for lane operations.
  type Half: Simd256;

  /// The 128-bit quarter type for lane operations.
  type Quarter: Simd128;

  /// Load 64 bytes from memory (may be unaligned).
  fn load(src: &[u8; 64]) -> Self;

  /// Load 64 bytes from aligned memory.
  ///
  /// # Safety
  ///
  /// `src` must be 64-byte aligned.
  unsafe fn load_aligned(src: &[u8; 64]) -> Self;

  /// Store to 64 bytes of memory (may be unaligned).
  fn store(self, dst: &mut [u8; 64]);

  /// Store to 64 bytes of aligned memory.
  ///
  /// # Safety
  ///
  /// `dst` must be 64-byte aligned.
  unsafe fn store_aligned(self, dst: &mut [u8; 64]);

  /// Bitwise XOR.
  fn xor(self, other: Self) -> Self;

  /// Bitwise AND.
  fn and(self, other: Self) -> Self;

  /// Bitwise OR.
  fn or(self, other: Self) -> Self;

  /// Three-way XOR: `a ^ b ^ c`.
  ///
  /// On AVX-512 with GFNI, this uses `vpternlogd` for single-instruction
  /// three-way XOR. On other platforms, falls back to two XORs.
  fn xor3(self, b: Self, c: Self) -> Self;

  /// 16×u32 addition (wrapping).
  fn add32(self, other: Self) -> Self;

  /// 8×u64 addition (wrapping).
  fn add64(self, other: Self) -> Self;

  /// 16×u32 rotate right by `N` bits.
  fn rotr32<const N: u32>(self) -> Self;

  /// 8×u64 rotate right by `N` bits.
  fn rotr64<const N: u32>(self) -> Self;

  /// Shuffle 32-bit lanes within 128-bit lanes.
  fn shuffle32<const MASK: i32>(self) -> Self;

  /// Shuffle bytes within 128-bit lanes.
  fn shuffle_bytes(self, ctrl: Self) -> Self;

  /// Create vector with all lanes set to zero.
  fn zero() -> Self;

  /// Extract 128-bit lane by index.
  fn extract128<const LANE: i32>(self) -> Self::Quarter;

  /// Extract 256-bit lane by index.
  fn extract256<const LANE: i32>(self) -> Self::Half;

  /// Broadcast 128-bit value to all lanes.
  fn broadcast128(quarter: Self::Quarter) -> Self;

  /// Broadcast 256-bit value to both lanes.
  fn broadcast256(half: Self::Half) -> Self;
}

// ─────────────────────────────────────────────────────────────────────────────
// Rotation Constants
// ─────────────────────────────────────────────────────────────────────────────

/// BLAKE3/BLAKE2s rotation constants (32-bit words).
pub mod blake_rot32 {
  /// First rotation: 16 bits.
  pub const R1: u32 = 16;
  /// Second rotation: 12 bits.
  pub const R2: u32 = 12;
  /// Third rotation: 8 bits.
  pub const R3: u32 = 8;
  /// Fourth rotation: 7 bits.
  pub const R4: u32 = 7;
}

/// BLAKE2b rotation constants (64-bit words).
pub mod blake_rot64 {
  /// First rotation: 32 bits.
  pub const R1: u32 = 32;
  /// Second rotation: 24 bits.
  pub const R2: u32 = 24;
  /// Third rotation: 16 bits.
  pub const R3: u32 = 16;
  /// Fourth rotation: 63 bits.
  pub const R4: u32 = 63;
}

// ─────────────────────────────────────────────────────────────────────────────
// Shuffle Masks
// ─────────────────────────────────────────────────────────────────────────────

/// Common shuffle masks for hash algorithms.
pub mod shuffle {
  /// Identity shuffle (no change): `[0, 1, 2, 3]`.
  pub const IDENTITY: i32 = 0b11_10_01_00;

  /// Rotate lanes left by 1: `[1, 2, 3, 0]`.
  pub const ROT_LEFT_1: i32 = 0b00_11_10_01;

  /// Rotate lanes left by 2: `[2, 3, 0, 1]`.
  pub const ROT_LEFT_2: i32 = 0b01_00_11_10;

  /// Rotate lanes left by 3: `[3, 0, 1, 2]`.
  pub const ROT_LEFT_3: i32 = 0b10_01_00_11;

  /// Reverse lanes: `[3, 2, 1, 0]`.
  pub const REVERSE: i32 = 0b00_01_10_11;

  /// BLAKE3 diagonalize shuffle for row 1.
  pub const BLAKE3_DIAG_R1: i32 = ROT_LEFT_1;

  /// BLAKE3 diagonalize shuffle for row 2.
  pub const BLAKE3_DIAG_R2: i32 = ROT_LEFT_2;

  /// BLAKE3 diagonalize shuffle for row 3.
  pub const BLAKE3_DIAG_R3: i32 = ROT_LEFT_3;

  /// BLAKE3 undiagonalize shuffle for row 1.
  pub const BLAKE3_UNDIAG_R1: i32 = ROT_LEFT_3;

  /// BLAKE3 undiagonalize shuffle for row 2.
  pub const BLAKE3_UNDIAG_R2: i32 = ROT_LEFT_2;

  /// BLAKE3 undiagonalize shuffle for row 3.
  pub const BLAKE3_UNDIAG_R3: i32 = ROT_LEFT_1;
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a slice to a fixed-size array reference.
///
/// # Panics
///
/// Panics if the slice length doesn't match N.
#[inline]
#[must_use]
pub fn slice_to_array<const N: usize>(slice: &[u8]) -> &[u8; N] {
  match slice.try_into() {
    Ok(arr) => arr,
    Err(_) => panic!("slice length mismatch"),
  }
}

/// Convert a mutable slice to a fixed-size mutable array reference.
///
/// # Panics
///
/// Panics if the slice length doesn't match N.
#[inline]
#[must_use]
pub fn slice_to_array_mut<const N: usize>(slice: &mut [u8]) -> &mut [u8; N] {
  match slice.try_into() {
    Ok(arr) => arr,
    Err(_) => panic!("slice length mismatch"),
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_shuffle_masks() {
    // Verify shuffle mask encoding: mask = (d << 6) | (c << 4) | (b << 2) | a
    // where result[i] = src[mask_field_i]
    assert_eq!(shuffle::IDENTITY, 0b11_10_01_00);
    assert_eq!(shuffle::ROT_LEFT_1, 0b00_11_10_01);
    assert_eq!(shuffle::REVERSE, 0b00_01_10_11);
  }

  #[test]
  fn test_rotation_constants() {
    // BLAKE3/BLAKE2s uses 32-bit words
    assert_eq!(
      blake_rot32::R1 + blake_rot32::R2 + blake_rot32::R3 + blake_rot32::R4,
      43
    );

    // BLAKE2b uses 64-bit words
    assert_eq!(
      blake_rot64::R1 + blake_rot64::R2 + blake_rot64::R3 + blake_rot64::R4,
      135
    );
  }

  #[test]
  fn test_slice_to_array() {
    let data = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let arr: &[u8; 16] = slice_to_array(&data);
    assert_eq!(arr, &data);
  }

  #[test]
  #[should_panic(expected = "slice length mismatch")]
  fn test_slice_to_array_wrong_size() {
    let data = [1u8, 2, 3, 4];
    let _: &[u8; 16] = slice_to_array(&data);
  }
}
