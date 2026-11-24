//! IBM Z PCC scalar multiplication for validated NIST ECDH inputs.
//!
//! The parameter layouts follow z/Architecture Principles of Operation, SA22-7832-14,
//! figures 7-372 and 7-373. Each coordinate and scalar is unsigned and big-endian.

use crate::{platform, traits::ct};

#[repr(align(8))]
struct ParameterBlock([u8; 4096]);

impl Drop for ParameterBlock {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

/// Multiply a validated affine point by a nonzero scalar below the curve order.
/// Write the requested coordinates. Return false when the hardware function is unavailable.
pub(super) fn multiply<const N: usize>(scalar: &[u8; N], coordinates: &[u8], result: &mut [u8]) -> bool {
  const { assert!(N == 32 || N == 48) };
  let point_bytes = N.strict_mul(2);
  assert_eq!(coordinates.len(), point_bytes);
  assert!(result.len() == N || result.len() == point_bytes);
  if !platform::caps().has(platform::caps::s390x::MSA9) {
    return false;
  }

  let mut functions = [0u64; 2];
  // SAFETY: MSA9 provides PCC query. GR1 points to its writable 16-byte result.
  // The query changes condition codes. It leaves both general registers unchanged.
  unsafe {
    core::arch::asm!(
      ".insn rre,0xb92c0000,0,0",
      in("r0") 0u64,
      in("r1") functions.as_mut_ptr(),
      options(nostack),
    );
  }
  // Query bits 64 and 65 are the high bits of the second big-endian word.
  let (function, mask) = if N == 32 { (64u64, 1 << 63) } else { (65, 1 << 62) };
  if functions[1] & mask == 0 {
    return false;
  }

  let mut block = ParameterBlock([0; 4096]);
  let (point, tail) = block.0[point_bytes..].split_at_mut(point_bytes);
  point.copy_from_slice(coordinates);
  tail[..N].copy_from_slice(scalar);
  let status: u64;
  // SAFETY: The query advertised this function. The full aligned block is writable and exclusive.
  // PCC consumes the scalar and point there. CC3 resumes with the same continuation state.
  // Shifting the IPM result right by 28 exposes the final two condition-code bits.
  unsafe {
    core::arch::asm!(
      "2:",
      ".insn rre,0xb92c0000,0,0",
      "jo 2b",
      "ipm {status}",
      status = lateout(reg) status,
      in("r0") function,
      in("r1") block.0.as_mut_ptr(),
      options(nostack),
    );
  }
  assert_eq!((status >> 28) & 3, 0, "validated finite point and nonzero scalar");
  result.copy_from_slice(&block.0[..result.len()]);
  true
}
